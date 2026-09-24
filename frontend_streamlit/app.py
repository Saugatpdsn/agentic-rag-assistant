"""Streamlit UI for the Agentic RAG Assistant.

Runs the LangGraph agent directly in this process — there is no separate
backend/API. Run it from the project root so relative paths in config.py
(.env, ./chroma_db, ./data/sample_docs) resolve correctly:

    uv sync
    streamlit run frontend_streamlit/app.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

# Make sure `rag_agent` (under src/) is importable even if the project
# wasn't installed with `uv sync` / `pip install -e .`.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage  # noqa: E402

from rag_agent.agent import build_agent  # noqa: E402
from rag_agent.config import settings  # noqa: E402
from rag_agent.ingest import ensure_seeded, ingest_paths  # noqa: E402
from rag_agent.llms import fast_model, heavy_model  # noqa: E402
from rag_agent.vectorstore import collection_count, list_sources  # noqa: E402

try:
    from langgraph.checkpoint.memory import InMemorySaver as _Saver
except ImportError:  # older langgraph versions
    from langgraph.checkpoint.memory import MemorySaver as _Saver

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🧠", layout="wide")

# Tool name -> (icon, human label), used for the little badges under each
# assistant reply and while a turn is streaming.
TOOL_LABELS = {
    "retrieve_documents": ("📄", "Document Search"),
    "web_search": ("🌐", "Web Search"),
}


def tool_badge(tool_name: str) -> str:
    icon, label = TOOL_LABELS.get(tool_name, ("🔧", tool_name))
    return f"{icon} {label}"


st.markdown(
    """
<style>
.source-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.source-card .src-title { font-weight: 600; font-size: 0.92rem; }
.source-card .src-snippet {
    font-size: 0.85rem; opacity: 0.8; margin-top: 4px;
    white-space: pre-wrap;
}
.tool-badges { opacity: 0.75; font-size: 0.82rem; margin-bottom: 2px; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Agent (built once per process, shared across sessions; conversation state
# is kept separate per browser session via a per-session thread_id)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Starting the agent...")
def get_agents():
    """Build a primary + fallback agent that share one checkpointer (so
    conversation history carries over if a turn has to fall back).

    503 "model overloaded" / 429 "quota exceeded" are per-model pools, not
    per-key — so on repeated failure, retrying the SAME model rarely helps,
    but switching to a DIFFERENT free-tier model usually does.
    """
    ensure_seeded()  # auto-ingest sample docs on first boot if empty
    checkpointer = _Saver()
    primary = build_agent(model=heavy_model(), checkpointer=checkpointer)
    fallback = build_agent(model=fast_model(), checkpointer=checkpointer)
    return primary, fallback


primary_agent, fallback_agent = get_agents()


# A single, persistent event loop shared by every chat turn in this process.
# asyncio.run() creates AND CLOSES a new loop each call; the Gemini client
# caches an async connection tied to whichever loop created it, so a second
# call on a fresh loop fails with "Event loop is closed". Reusing one loop
# across turns (via run_until_complete instead of asyncio.run) avoids that.
@st.cache_resource(show_spinner=False)
def get_event_loop() -> asyncio.AbstractEventLoop:
    return asyncio.new_event_loop()


def _init_state() -> None:
    st.session_state.setdefault("messages", [])  # [{role, content, sources, tools_used}]
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))


_init_state()


# --------------------------------------------------------------------------
# Agent driver
# --------------------------------------------------------------------------
def _result_preview(content, limit: int = 300) -> str:
    import json

    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return text[:limit]


async def _run_turn(agent, message: str, thread_id: str, on_event) -> None:
    """Drive one agent turn, calling on_event(kind, payload) for each update."""
    config = {"configurable": {"thread_id": thread_id}}
    seen_tool_starts: set[str] = set()

    async for mode, chunk in agent.astream(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        stream_mode=["messages", "updates", "custom"],
    ):
        if mode == "messages":
            msg, _meta = chunk
            if isinstance(msg, AIMessageChunk) and msg.text:
                on_event("token", {"delta": msg.text})

        elif mode == "updates":
            for _node_name, node_update in (chunk or {}).items():
                for m in (node_update or {}).get("messages", []) or []:
                    if isinstance(m, AIMessage) and m.tool_calls:
                        for tc in m.tool_calls:
                            tc_id = tc.get("id") or tc.get("name", "")
                            if tc_id in seen_tool_starts:
                                continue
                            seen_tool_starts.add(tc_id)
                            on_event("tool_start", {"tool": tc.get("name", "")})
                    if isinstance(m, ToolMessage):
                        on_event(
                            "tool_end",
                            {
                                "tool": m.name or "",
                                "ok": getattr(m, "status", "success") != "error",
                                "result_preview": _result_preview(m.content),
                            },
                        )

        elif mode == "custom":
            if isinstance(chunk, dict) and chunk.get("kind") == "sources":
                on_event("sources", {"tool": chunk.get("tool", ""), "sources": chunk.get("sources", [])})


_TRANSIENT_MARKERS = ("503", "overloaded", "unavailable", "high demand")
_RATE_LIMIT_MARKERS = ("429", "quota exceeded", "resource exhausted", "resource_exhausted")
_DAILY_QUOTA_MARKERS = ("perday", "requests per day", "generaterequestsperdayperprojectpermodel")


def _is_transient(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in _TRANSIENT_MARKERS)


def _is_rate_limited(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in _RATE_LIMIT_MARKERS)


def _is_daily_quota_exhausted(exc: Exception) -> bool:
    """A per-day free-tier quota won't recover within this session — retrying
    the same model is pointless until it resets (~midnight Pacific)."""
    return any(m in str(exc).lower() for m in _DAILY_QUOTA_MARKERS)


def run_turn_sync(
    message: str, thread_id: str, on_event, on_retry=None, on_fallback=None, max_attempts: int = 2
) -> None:
    """Run one turn against the primary model; on overload (503) or a
    per-minute rate limit (429), retry with backoff. On a per-day quota
    exhaustion, or repeated overload/rate-limiting, fall back immediately to
    a different free-tier model (separate quota + capacity pool) before
    giving up. Real errors (bad request, auth, etc.) are never retried.
    """
    import time

    loop = get_event_loop()
    asyncio.set_event_loop(loop)

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            loop.run_until_complete(_run_turn(primary_agent, message, thread_id, on_event))
            return
        except Exception as exc:
            last_exc = exc
            if _is_daily_quota_exhausted(exc):
                break  # no point retrying the same model today
            if not (_is_transient(exc) or _is_rate_limited(exc)):
                raise
            if attempt < max_attempts:
                if on_retry:
                    on_retry(attempt, max_attempts)
                time.sleep(min(2**attempt, 8))  # 2s, 4s, ...

    # Primary model is out of capacity or quota for now — try the fallback
    # model, which has its own separate capacity + daily quota pool, and
    # shares conversation history via the same checkpointer.
    if on_fallback:
        on_fallback()
    try:
        loop.run_until_complete(_run_turn(fallback_agent, message, thread_id, on_event))
    except Exception as exc:
        if _is_transient(exc) or _is_rate_limited(exc):
            raise last_exc from exc  # both pools exhausted/overloaded — surface the original message
        raise


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def render_tool_badges(tools_used: list[str]) -> None:
    if not tools_used:
        st.markdown('<div class="tool-badges">💭 Answered directly — no tools used</div>', unsafe_allow_html=True)
        return
    badges = " &nbsp;·&nbsp; ".join(tool_badge(t) for t in tools_used)
    st.markdown(f'<div class="tool-badges">{badges}</div>', unsafe_allow_html=True)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    docs = [s for s in sources if s.get("kind") == "document"]
    web = [s for s in sources if s.get("kind") == "web"]

    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        if docs:
            st.caption(f"From your documents ({len(docs)})")
            for s in docs:
                snippet = (s.get("snippet") or "").strip().replace("\n", " ")
                st.markdown(
                    f"""<div class="source-card">
<div class="src-title">📄 {s.get('title', 'document')}</div>
<div class="src-snippet">{snippet}</div>
</div>""",
                    unsafe_allow_html=True,
                )
        if web:
            if docs:
                st.divider()
            st.caption(f"From the web ({len(web)})")
            for s in web:
                title = s.get("title", "result")
                url = s.get("url")
                domain = urlparse(url).netloc if url else ""
                snippet = (s.get("snippet") or "").strip().replace("\n", " ")
                title_html = f'<a href="{url}" target="_blank">{title}</a>' if url else title
                st.markdown(
                    f"""<div class="source-card">
<div class="src-title">🌐 {title_html} {f'<span style="opacity:.6;font-weight:400;">({domain})</span>' if domain else ''}</div>
<div class="src-snippet">{snippet}</div>
</div>""",
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Agentic RAG")
    st.caption("An agent that decides, per question, whether to search your documents, search the web, or answer directly.")

    with st.expander("⚙️ Models & settings", expanded=False):
        st.markdown(
            f"""
- **Heavy model:** `{settings.model_heavy}`
- **Fallback model:** `{settings.model_fast}`
- **Embeddings:** `{settings.embedding_model}`
- **Web search:** `{settings.web_backend}`
"""
        )

    st.divider()
    st.subheader("📚 Knowledge base")
    sources_list = list_sources()
    total_chunks = collection_count()
    c1, c2 = st.columns(2)
    c1.metric("Documents", len(sources_list))
    c2.metric("Chunks", total_chunks)

    if sources_list:
        for name, count in sources_list:
            st.markdown(f"📄 **{name}** &nbsp;·&nbsp; {count} chunk{'s' if count != 1 else ''}", unsafe_allow_html=True)
    else:
        st.caption("No documents indexed yet.")

    with st.expander("➕ Add documents", expanded=not sources_list):
        uploaded = st.file_uploader(
            "PDF / TXT / MD files",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if st.button("Ingest", disabled=not uploaded, use_container_width=True):
            with st.spinner("Ingesting..."):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_paths = []
                    for f in uploaded:
                        p = os.path.join(tmp_dir, f.name)
                        with open(p, "wb") as out:
                            out.write(f.getvalue())
                        tmp_paths.append(p)
                    added = ingest_paths(tmp_paths)
            st.success(f"Added/updated {added} chunks from {len(uploaded)} file(s).")
            st.rerun()

    st.divider()
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.rerun()


# --------------------------------------------------------------------------
# Chat history
# --------------------------------------------------------------------------
st.title("Chat with your documents")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_tool_badges(msg.get("tools_used", []))
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))

# --------------------------------------------------------------------------
# Chat input
# --------------------------------------------------------------------------
prompt = st.chat_input("Ask a question about your documents or the web...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt, "sources": [], "tools_used": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        badge_placeholder = st.empty()
        text_placeholder = st.empty()
        tool_status = st.empty()
        state = {"accumulated": "", "sources": [], "tools_used": [], "error": None}

        def on_event(kind: str, data: dict) -> None:
            if kind == "tool_start":
                name = data.get("tool", "")
                if name not in state["tools_used"]:
                    state["tools_used"].append(name)
                tool_status.markdown(f"⏳ Using **{tool_badge(name)}**...")
            elif kind == "tool_end":
                ok = data.get("ok", True)
                tool_status.markdown(f"{'✅' if ok else '⚠️'} {tool_badge(data.get('tool', ''))} finished")
            elif kind == "token":
                state["accumulated"] += data.get("delta", "")
                text_placeholder.markdown(state["accumulated"] + "▌")
            elif kind == "sources":
                for s in data.get("sources", []):
                    if s not in state["sources"]:
                        state["sources"].append(s)

        def on_retry(attempt: int, max_attempts: int) -> None:
            # Google's servers are momentarily overloaded — reset any partial
            # output and try again rather than surfacing a scary error.
            state["accumulated"] = ""
            state["sources"] = []
            state["tools_used"] = []
            text_placeholder.empty()
            tool_status.warning(f"⏳ Gemini is busy, retrying... ({attempt}/{max_attempts - 1})")

        def on_fallback() -> None:
            state["accumulated"] = ""
            state["sources"] = []
            state["tools_used"] = []
            text_placeholder.empty()
            tool_status.warning(f"🔁 Switching to `{settings.model_fast}` (different capacity pool)...")

        try:
            run_turn_sync(prompt, st.session_state["thread_id"], on_event, on_retry=on_retry, on_fallback=on_fallback)
        except Exception as exc:  # keep the UI usable even if a turn fails
            msg_text = str(exc)
            low = msg_text.lower()
            if "perday" in low or "requests per day" in low:
                state["error"] = (
                    "You've hit today's free-tier daily request limit for both models. "
                    "It resets at midnight Pacific time — try again then, or add billing "
                    "to your Google AI Studio project for higher limits."
                )
            elif any(m in low for m in ("429", "quota exceeded", "resource exhausted")):
                state["error"] = "Gemini's free-tier rate limit was hit on both models. Please wait a minute and try again."
            elif any(m in low for m in ("503", "overloaded", "unavailable", "high demand")):
                state["error"] = "Gemini is overloaded on every available free-tier model right now. Please try again in a minute."
            else:
                state["error"] = msg_text

        tool_status.empty()
        badge_placeholder.markdown(
            '<div class="tool-badges">💭 Answered directly — no tools used</div>'
            if not state["tools_used"]
            else '<div class="tool-badges">'
            + " &nbsp;·&nbsp; ".join(tool_badge(t) for t in state["tools_used"])
            + "</div>",
            unsafe_allow_html=True,
        )
        text_placeholder.markdown(state["accumulated"] or "_(no response)_")
        if state["error"]:
            st.error(state["error"])
        render_sources(state["sources"])

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": state["accumulated"],
            "sources": state["sources"],
            "tools_used": state["tools_used"],
        }
    )