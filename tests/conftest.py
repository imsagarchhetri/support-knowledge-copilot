"""Test fixtures: force ephemeral, keyless config and build the index once."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Configure BEFORE app modules read settings.
ROOT = Path(__file__).resolve().parents[1]
os.environ.update({
    "llm_provider": "test",
    "embedding_provider": "fastembed",
    "qdrant_in_memory": "true",
    "docs_dir": str(ROOT / "data" / "docs"),
    "sparse_index_path": str(ROOT / "data" / "bm25.test.pkl"),
})


@pytest.fixture(scope="session", autouse=True)
def built_index():
    from app.config import get_settings
    get_settings.cache_clear()
    from app.vectorstore import get_client, get_embedder
    get_client.cache_clear()
    get_embedder.cache_clear()
    from app.index_service import build_index
    n = build_index(rebuild=True)
    assert n > 0
    yield n
