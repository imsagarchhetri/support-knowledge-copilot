"""Orchestrates ingestion: load+chunk -> dense (Qdrant) + sparse (BM25)."""
from __future__ import annotations

from .config import get_settings
from .ingest import load_and_chunk
from .sparse import SparseIndex
from .vectorstore import VectorStore


def build_index(strategy: str | None = None, rebuild: bool = True) -> int:
    s = get_settings()
    chunks = load_and_chunk(strategy=strategy or s.chunk_strategy)
    if not chunks:
        raise RuntimeError(f"no documents found in {s.docs_dir}")

    vs = VectorStore()
    if rebuild:
        vs.recreate()
    vs.index(chunks)

    sparse = SparseIndex()
    sparse.build(chunks)
    sparse.save()
    return len(chunks)
