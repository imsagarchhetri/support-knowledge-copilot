"""Pydantic domain models — the assistant's public contract.

The contract is deliberately honest: every answer carries citations the system
actually verified, a confidence breakdown, and a list of things it could not
verify.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkMeta(BaseModel):
    chunk_id: str
    source: str
    heading: str
    doc_type: str = "doc"
    last_updated: str = "unknown"
    access_level: str = "internal"
    strategy: str = "heading"


class Chunk(ChunkMeta):
    text: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    heading: str
    text: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


class Citation(BaseModel):
    chunk_id: str
    source: str
    supported: bool
    support_score: float


class ConfidenceBreakdown(BaseModel):
    overall: float
    max_dense: float
    max_bm25: float
    citation_support_rate: float
    coverage: float


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list)
    confidence: ConfidenceBreakdown
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    refused: bool = False


class AskRequest(BaseModel):
    question: str
    hybrid: bool = True
    top_k: int | None = None


class IngestRequest(BaseModel):
    rebuild: bool = True
    strategy: str | None = None
