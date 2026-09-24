"""Chroma vector store, persisted to disk for zero-setup RAG."""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma

from rag_agent.config import settings
from rag_agent.embeddings import get_embeddings


@lru_cache
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


def collection_count() -> int:
    """Number of stored chunks (0 when empty / fresh)."""
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0


def list_sources() -> list[tuple[str, int]]:
    """Distinct ingested filenames with their chunk counts, sorted by name."""
    try:
        data = get_vectorstore()._collection.get(include=["metadatas"])
    except Exception:
        return []
    counts: dict[str, int] = {}
    for md in data.get("metadatas", []) or []:
        name = (md or {}).get("source", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return sorted(counts.items())