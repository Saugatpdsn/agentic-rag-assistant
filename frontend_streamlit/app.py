"""Streamlit UI for the Agentic RAG Assistant.

Runs the LangGraph agent directly in this process — there is no separate
backend/API. Run it from the project root so relative paths in config.py
(.env, ./chroma_db, ./data/sample_docs) resolve correctly:

    uv sync
    streamlit run frontend_streamlit/app.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

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
from rag_agent.vectorstore import collection_count  # noqa: E402

try:
    from langgraph.checkpoint.memory import InMemorySaver as _Saver
except ImportError:  # older langgraph versions
    from langgraph.checkpoint.memory import MemorySaver as _Saver

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🧠", layout="wide")


# --------------------------------------------------------------------------
# Agent (built once per process, shared across sessions; conversation state
# is kept separate per browser session via a per-session thread_id)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Starting the agent...")
def get_agents():
    """Build a primary + fallback agent that share one checkpointer (so
    conversation history carries over if a turn has to fall back).

    503 "model overloaded" is a per-model shared-capacity issue, not
    per-key — so on repeated overload, retrying the SAME model rarely
    helps, but switching to a DIFFERENT free-tier model usually does,
    since it draws from a separate serving pool.
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
    loop = asyncio.new_event_loop()
    return loop


def _init_state() -> None:
    st.session_state.setdefault("messages", [])  # [{role, content, sources}]
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))


_init_state()


# --------------------------------------------------------------------------
# Agent driver — mirrors the old SSE event stream, but yields locally
# --------------------------------------------------------------------------
def _result_preview(content, limit: int = 300) -> str:
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


def _is_transient(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in _TRANSIENT_MARKERS)


def run_turn_sync(
    message: str, thread_id: str, on_event, on_retry=None, on_fallback=None, max_attempts: int = 2
) -> None:
    """Run one turn against the primary model; on repeated overload (503),
    fall back to a different free-tier model (separate serving pool) before
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
            if not _is_transient(exc):
                raise
            if attempt < max_attempts:
                if on_retry:
                    on_retry(attempt, max_attempts)
                time.sleep(min(2**attempt, 8))  # 2s, 4s, ...

    # Primary model is consistently overloaded — try the fallback model,
    # which draws from a different capacity pool and shares conversation
    # history via the same checkpointer.
    if on_fallback:
        on_fallback()
    try:
        loop.run_until_complete(_run_turn(fallback_agent, message, thread_id, on_event))
    except Exception as exc:
        if _is_transient(exc):
            raise last_exc from exc  # both pools overloaded — surface the original message
        raise


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Agentic RAG")
    st.markdown(
        f"""
- **Fast model:** `{settings.model_fast}` (also used as overload fallback)
- **Heavy model:** `{settings.model_heavy}`
- **Embeddings:** `{settings.embedding_model}`
- **Web search:** `{settings.web_backend}`
- **Docs indexed:** `{collection_count()}`
"""
    )

    st.divider()
    st.subheader("Upload documents")
    uploaded = st.file_uploader(
        "PDF / TXT / MD files to add to the knowledge base",
        accept_multiple_files=True,
    )
    if st.button("Ingest", disabled=not uploaded, use_container_width=True):
        with st.spinner("Ingesting..."):
            tmp_paths = []
            with tempfile.TemporaryDirectory() as tmp_dir:
                for f in uploaded:
                    p = os.path.join(tmp_dir, f.name)
                    with open(p, "wb") as out:
                        out.write(f.getvalue())
                    tmp_paths.append(p)
                added = ingest_paths(tmp_paths)
        st.success(f"Added {added} chunks from {len(uploaded)} file(s).")
        st.rerun()

    st.divider()
    if st.button("New conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.rerun()


# --------------------------------------------------------------------------
# Chat history
# --------------------------------------------------------------------------
st.title("Chat with your documents")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# --------------------------------------------------------------------------
# Chat input
# --------------------------------------------------------------------------
prompt = st.chat_input("Ask a question about your documents or the web...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        tool_status = st.empty()
        state = {"accumulated": "", "sources": [], "error": None}

        def on_event(kind: str, data: dict) -> None:
            if kind == "tool_start":
                tool_status.info(f"🔧 Using tool: `{data.get('tool', '')}`")
            elif kind == "tool_end":
                ok = data.get("ok", True)
                tool_status.info(f"{'✅' if ok else '⚠️'} `{data.get('tool', '')}` finished")
            elif kind == "token":
                state["accumulated"] += data.get("delta", "")
                text_placeholder.markdown(state["accumulated"] + "▌")
            elif kind == "sources":
                for s in data.get("sources", []):
                    label = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
                    if label not in state["sources"]:
                        state["sources"].append(label)

        def on_retry(attempt: int, max_attempts: int) -> None:
            # Google's servers are momentarily overloaded — reset any partial
            # output and try again rather than surfacing a scary error.
            state["accumulated"] = ""
            state["sources"] = []
            text_placeholder.empty()
            tool_status.warning(f"⏳ Gemini is busy, retrying... ({attempt}/{max_attempts - 1})")

        def on_fallback() -> None:
            state["accumulated"] = ""
            state["sources"] = []
            text_placeholder.empty()
            tool_status.warning(f"🔁 Switching to `{settings.model_fast}` (different capacity pool)...")

        try:
            run_turn_sync(prompt, st.session_state["thread_id"], on_event, on_retry=on_retry, on_fallback=on_fallback)
        except Exception as exc:  # keep the UI usable even if a turn fails
            msg = str(exc)
            if any(m in msg.lower() for m in ("503", "overloaded", "unavailable", "high demand")):
                state["error"] = "Gemini is overloaded on every available free-tier model right now. Please try again in a minute."
            else:
                state["error"] = msg

        tool_status.empty()
        text_placeholder.markdown(state["accumulated"] or "_(no response)_")
        if state["error"]:
            st.error(state["error"])
        if state["sources"]:
            with st.expander(f"Sources ({len(state['sources'])})"):
                for src in state["sources"]:
                    st.markdown(f"- {src}")

    st.session_state["messages"].append(
        {"role": "assistant", "content": state["accumulated"], "sources": state["sources"]}
    )