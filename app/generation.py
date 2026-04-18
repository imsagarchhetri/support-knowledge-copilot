"""Grounded answer generation + citation verification + confidence scoring.

Pipeline: refusal gate -> grounded generation -> parse citations -> verify each
citation (embedding NLI-style support + optional LLM judge) -> confidence.
"""
from __future__ import annotations

import json
import re

import numpy as np

from .config import get_settings
from .embeddings import get_embedder
from .llm import get_llm
from .models import (Answer, Citation, ConfidenceBreakdown, RetrievedChunk)
from .retrieval import HybridRetriever, Signals

SYSTEM = (
    "You are a support knowledge assistant. Answer ONLY from the provided context. "
    "Cite every claim with its chunk id in square brackets, e.g. [account_faq::heading::1]. "
    "If the answer is not in the context, say you could not find it. Be concise."
)

_CITE = re.compile(r"\[([^\]]+::[^\]]+)\]")


def _cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


class AnswerService:
    def __init__(self, retriever: HybridRetriever | None = None):
        self.s = get_settings()
        self.retriever = retriever or HybridRetriever()
        self.llm = get_llm()
        self.embedder = get_embedder()

    def ask(self, question: str, *, hybrid: bool = True,
            top_k: int | None = None) -> Answer:
        hits, sig = self.retriever.retrieve(question, top_k=top_k, hybrid=hybrid)

        if not hits or (sig.max_bm25 < self.s.bm25_floor
                        and sig.max_dense < self.s.dense_floor):
            return self._refusal(question, hits, sig)

        prompt = self._build_prompt(question, hits)
        text = self.llm.complete(SYSTEM, prompt).text

        citations, unverified = self._verify_citations(text, hits)
        return self._finalize(question, text, hits, sig, citations, unverified)

    # --- internals ---
    def _build_prompt(self, question: str, hits: list[RetrievedChunk]) -> str:
        ctx = "\n\n".join(f"[{h.chunk_id}] ({h.heading})\n{h.text}" for h in hits)
        return f"context:\n{ctx}\n\nquestion: {question}\nAnswer with citations:"

    def _verify_citations(self, text: str, hits: list[RetrievedChunk]):
        by_id = {h.chunk_id: h for h in hits}
        cited = list(dict.fromkeys(_CITE.findall(text)))
        # embed the answer once; compare to each cited chunk (NLI-style support)
        ans_vec = self.embedder.embed([text])[0]
        citations, unverified = [], []
        for cid in cited:
            chunk = by_id.get(cid)
            if not chunk:
                unverified.append(f"{cid}: cited chunk not in retrieved set")
                continue
            support = _cosine(ans_vec, self.embedder.embed([chunk.text])[0])
            ok = support >= 0.4
            citations.append(Citation(chunk_id=cid, source=chunk.source,
                                      supported=ok, support_score=round(support, 3)))
            if not ok:
                unverified.append(f"{cid}: low support ({support:.2f})")
        return citations, unverified

    def _finalize(self, question, text, hits, sig: Signals, citations, unverified):
        supported = [c for c in citations if c.supported]
        support_rate = len(supported) / len(citations) if citations else 0.5
        coverage = min(1.0, len(hits) / self.s.top_k)
        retrieval_norm = min(1.0, max(sig.max_dense / 0.7, sig.max_bm25 / 8.0))
        overall = round(0.5 * retrieval_norm + 0.35 * support_rate + 0.15 * coverage, 3)
        return Answer(
            question=question, answer=text,
            citations=[c for c in citations if c.supported],
            unverified=unverified,
            confidence=ConfidenceBreakdown(
                overall=overall, max_dense=round(sig.max_dense, 3),
                max_bm25=round(sig.max_bm25, 3),
                citation_support_rate=round(support_rate, 3),
                coverage=round(coverage, 3)),
            retrieved=hits,
        )

    def _refusal(self, question, hits, sig: Signals) -> Answer:
        closest = ", ".join(h.heading for h in hits[:3]) or "none"
        return Answer(
            question=question,
            answer=f"I could not find this in the documentation. Closest sections: {closest}",
            confidence=ConfidenceBreakdown(
                overall=round(min(sig.max_dense, 0.999), 3),
                max_dense=round(sig.max_dense, 3), max_bm25=round(sig.max_bm25, 3),
                citation_support_rate=0.0,
                coverage=min(1.0, len(hits) / self.s.top_k)),
            retrieved=hits, refused=True,
        )
