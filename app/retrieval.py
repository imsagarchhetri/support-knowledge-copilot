"""Hybrid retrieval: Qdrant dense + BM25 sparse fused with Reciprocal Rank Fusion,
then a lightweight rerank. Exposes raw signals for the refusal gate.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import get_settings
from .models import RetrievedChunk
from .sparse import SparseIndex, tokenize
from .vectorstore import VectorStore


@dataclass
class Signals:
    max_dense: float
    max_bm25: float


class HybridRetriever:
    def __init__(self, vector: VectorStore | None = None,
                 sparse: SparseIndex | None = None):
        self.s = get_settings()
        self.vector = vector or VectorStore()
        self.sparse = sparse or SparseIndex().load()

    def retrieve(self, query: str, *, top_k: int | None = None,
                 hybrid: bool = True) -> tuple[list[RetrievedChunk], Signals]:
        top_k = top_k or self.s.top_k
        pool = self.s.candidate_pool

        dense = self.vector.search(query, pool)
        sparse = self.sparse.search(query, pool) if hybrid else []

        max_dense = dense[0][1] if dense else 0.0
        max_bm25 = sparse[0][1] if sparse else 0.0

        fused = self._rrf(dense, sparse)
        ranked = self._rerank(query, fused)[:top_k]
        return ranked, Signals(max_dense, max_bm25)

    def _rrf(self, dense, sparse) -> list[RetrievedChunk]:
        scores: dict[str, float] = {}
        objs: dict[str, RetrievedChunk] = {}
        meta: dict[str, dict] = {}
        for rank, (rc, _) in enumerate(dense):
            scores[rc.chunk_id] = scores.get(rc.chunk_id, 0.0) + \
                self.s.dense_weight / (self.s.rrf_k + rank + 1)
            objs[rc.chunk_id] = rc
            meta.setdefault(rc.chunk_id, {})["dense_rank"] = rank
        for rank, (rc, _) in enumerate(sparse):
            scores[rc.chunk_id] = scores.get(rc.chunk_id, 0.0) + \
                self.s.sparse_weight / (self.s.rrf_k + rank + 1)
            objs.setdefault(rc.chunk_id, rc)
            meta.setdefault(rc.chunk_id, {})["sparse_rank"] = rank
        out = []
        for cid, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            rc = objs[cid]
            rc.score = sc
            rc.dense_rank = meta[cid].get("dense_rank")
            rc.sparse_rank = meta[cid].get("sparse_rank")
            out.append(rc)
        return out

    def _rerank(self, query: str, hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
        q = set(tokenize(query))
        for h in hits:
            overlap = len(q & set(tokenize(h.text))) / (len(q) or 1)
            h.score = 0.7 * h.score + 0.3 * overlap
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits
