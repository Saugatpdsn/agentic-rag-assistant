"""ChatGoogleGenerativeAI (Gemini) factories.

Thinking is explicitly turned off (thinking_budget=0). Gemini's thinking
models attach a thought_signature to function-call parts and require it to
be echoed back on every step of a tool-calling turn; langchain-google-genai
doesn't yet do this reliably through LangGraph's tool loop, which surfaces
as a 400 "missing thought_signature" error (see
https://github.com/langchain-ai/langchain-google/issues/1364). With
thinking disabled, no signature is ever generated, so there's nothing to
drop — this sidesteps the bug entirely rather than working around it.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from rag_agent.config import settings


def fast_model() -> ChatGoogleGenerativeAI:
    """Small, cheap model for routine steps."""
    return ChatGoogleGenerativeAI(
        model=settings.model_fast,
        google_api_key=settings.google_api_key,
        thinking_budget=0,
    )


def heavy_model() -> ChatGoogleGenerativeAI:
    """Capable reasoning model for planning + answer synthesis."""
    return ChatGoogleGenerativeAI(
        model=settings.model_heavy,
        google_api_key=settings.google_api_key,
        thinking_budget=0,
    )
