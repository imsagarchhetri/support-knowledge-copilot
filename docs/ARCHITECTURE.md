# Architecture

## System diagram

```
                         ┌──────────────────────── Ingestion ───────────────────────┐
 data/docs/*.{md,txt,    │ loaders ─► chunking (heading | fixed) ─► Chunk[+metadata] │
   html,pdf}             │                    │                         │            │
                         │           BGE embeddings              BM25 tokenization   │
                         │                    ▼                         ▼            │
                         │              Qdrant (dense)            rank_bm25 (sparse)  │
                         └────────────── same chunk IDs ───────────────┘            │
                                                  │
 question ─► FastAPI /ask ─► HybridRetriever ─────┤
                              dense (Qdrant cosine) ─┐
                              sparse (BM25)        ──┴─ RRF fuse ─► rerank ─► top-k
                                                  │            raw signals (max_dense, max_bm25)
                              AnswerService ───────┤
                                refusal gate (raw signals < floors? -> refuse)
                                grounded generation (LLM, context-only, cite chunk IDs)
                                citation verification (embedding NLI + optional judge)
                                confidence = f(retrieval, citation support, coverage)
                                                  ▼
                                          Answer (JSON contract)
                                                  │
                         Streamlit UI ◄───────────┘      eval/run_eval.py ◄── golden.jsonl
```

## Key decisions

### Dense + sparse over the same chunk IDs
Semantic search (BGE/Qdrant) finds paraphrases; BM25 nails exact tokens (`BIL-402`,
`API-429`, SKUs). Indexing both over identical chunk IDs makes **Reciprocal Rank
Fusion** a clean merge with no ID reconciliation.

### Refusal on raw signals, not the fused score
RRF produces rank-based scores that discard magnitude, so a fused score can't
distinguish "weakly relevant" from "strongly relevant." The refusal gate therefore
uses the **raw** max cosine and max BM25, with floors calibrated for BGE-small
(`dense_floor=0.65`, `bm25_floor=4.5`). Swap embeddings → recalibrate.

### Provider abstraction
`app/llm.py` and `app/embeddings.py` expose one interface each. Production can run
fully local (BGE + Ollama), fully hosted (OpenAI), or mixed. The `test` LLM
provider yields grounded answers and valid judge JSON so CI exercises the real
parsing/verification paths without keys.

### Citation verification
Citations are parsed from the answer, mapped back to retrieved chunks, and scored
for support via embedding cosine (a lightweight NLI proxy). Unsupported citations
are removed from the answer and surfaced under `unverified`. An LLM judge can be
layered on for nuanced claims (`json_mode` is already wired).

### Confidence
`overall = 0.5·retrieval_norm + 0.35·citation_support_rate + 0.15·coverage`,
returned with its components so consumers can set their own thresholds.

## Connection modes (Qdrant)
| Mode | Setting | Use |
|---|---|---|
| in-memory | `qdrant_in_memory=true` | tests (ephemeral) |
| local on-disk | default (`qdrant_path`) | single-process dev |
| server | `qdrant_url=http://qdrant:6333` | docker-compose / production |

## Scaling notes
- Move BM25 to a service (or use Qdrant's native sparse vectors) for large corpora.
- Add a cross-encoder reranker (e.g. `bge-reranker`) in `_rerank` for quality.
- Add metadata filters (access_level, doc_type) to the Qdrant query for
  multi-tenant / permissioned retrieval.
- Batch + cache embeddings; warm the model at startup.
```
