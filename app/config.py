"""Typed application settings (12-factor: configured via environment / .env).

All knobs that differ between local dev, CI, and production live here so nothing
is hard-coded in the pipeline.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # --- LLM ---
    llm_provider: Literal["openai", "anthropic", "ollama", "test"] = "test"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_host: str = "http://localhost:11434"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 600

    # --- Embeddings ---
    embedding_provider: Literal["fastembed", "openai"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"   # 384-dim, local, no key
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 384

    # --- Vector store (Qdrant) ---
    # Empty url => local on-disk Qdrant (great for dev/CI). Set to http://qdrant:6333
    # in docker-compose / production.
    qdrant_url: str | None = None
    qdrant_path: str = "./data/qdrant"     # used when qdrant_url is empty
    qdrant_in_memory: bool = False         # True => ephemeral (tests)
    collection: str = "support_docs"

    # --- Retrieval ---
    top_k: int = 5
    candidate_pool: int = 20
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    rrf_k: int = 60
    # refusal gate: answer only with a strong keyword OR semantic match
    bm25_floor: float = 4.5
    dense_floor: float = 0.65

    # --- Chunking ---
    chunk_strategy: Literal["heading", "fixed"] = "heading"
    chunk_size: int = 600
    chunk_overlap: int = 100

    docs_dir: str = "./data/docs"
    sparse_index_path: str = "./data/bm25.pkl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
