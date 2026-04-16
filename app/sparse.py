"""BM25 sparse index (rank_bm25), persisted to disk.

Kept over the SAME chunk IDs as the dense index so fusion is a clean merge.
BM25 catches exact tokens — error codes, SKUs, API names — that dense retrieval
often misses.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .config import get_settings
from .models import Chunk, RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class SparseIndex:
    def __init__(self):
        self.s = get_settings()
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def save(self):
        path = Path(self.s.sparse_index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"chunks": [c.model_dump() for c in self.chunks]}, f)

    def load(self):
        path = Path(self.s.sparse_index_path)
        with path.open("rb") as f:
            data = pickle.load(f)
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        return self

    def search(self, query: str, limit: int) -> list[tuple[RetrievedChunk, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:limit]
        out = []
        for idx, score in ranked:
            if score <= 0:
                continue
            c = self.chunks[idx]
            out.append((RetrievedChunk(chunk_id=c.chunk_id, source=c.source,
                                       heading=c.heading, text=c.text, score=score),
                        float(score)))
        return out
