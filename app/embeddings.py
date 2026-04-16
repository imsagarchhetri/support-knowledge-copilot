"""Embedding providers.

Default: fastembed with BAAI/bge-small-en-v1.5 (384-dim) — runs locally, no API
key, production-viable. Optional: OpenAI text-embedding-3-small.

Both implement the same `embed()` interface so the rest of the system is
provider-agnostic.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from .config import get_settings


class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name)
        self.dim = self._model.embedding_size if hasattr(self._model, "embedding_size") else 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]


class OpenAIEmbedder:
    def __init__(self, model_name: str, dim: int):
        from openai import OpenAI
        self._client = OpenAI()
        self._model = model_name
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_provider == "openai":
        return OpenAIEmbedder(s.openai_embedding_model, s.embedding_dim)
    return FastEmbedEmbedder(s.embedding_model)
