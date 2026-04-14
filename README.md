# Project 1 — Support Knowledge Copilot with Verified Citations (Production)

**Topic:** RAG · Hybrid Retrieval · Citation Verification · Retrieval Evaluation

A production-grade support knowledge assistant: it answers questions from internal
docs using **hybrid retrieval (Qdrant dense + BM25 sparse, fused with RRF)**,
generates grounded answers with citations, **verifies every citation**, scores
confidence, and **refuses honestly** when the answer isn't in the corpus.

> 📖 New here? Read [`docs/TUTORIAL.md`](docs/TUTORIAL.md) for a step-by-step
> build & run walkthrough, [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md)
> for every file/function with pseudocode, and
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design rationale.

## Real stack

| Concern | Tool |
|---|---|
| API | **FastAPI** (`app/api.py`) |
| Dense vector store | **Qdrant** (`qdrant-client`) — local on-disk, in-memory, or server |
| Sparse retrieval | **BM25** (`rank_bm25`) |
| Embeddings | **fastembed / BGE-small** (local, no key) or OpenAI `text-embedding-3-small` |
| LLM | OpenAI / Anthropic / Ollama (`test` provider for keyless CI) |
| Evals | RAGAS-style metrics (`eval/run_eval.py`), optional `ragas` |
| UI | **Streamlit** (`ui/streamlit_app.py`) |
| Packaging | **Docker Compose** (Qdrant + API + UI) |
| Tests | **pytest** (`tests/`) |

Everything runs **with zero API keys** out of the box: embeddings are local (BGE),
the vector store is local Qdrant, and `llm_provider=test` produces grounded
answers + valid citation-judge JSON so the whole pipeline is exercised. Point it
at OpenAI/Anthropic/Ollama and a Qdrant server for production by setting env vars.

## Quickstart (local, no keys)

```bash
cd 01-support-knowledge-copilot
python3.11 -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt

python -m app.cli ingest                                   # build Qdrant + BM25 indexes
python -m app.cli ask "How do I reset my password?"        # grounded answer + citations
python -m app.cli ask "What is your stock price?"          # honest refusal
python -m eval.run_eval --strategy hybrid                  # metrics report
pytest                                                      # 9 tests, real stack
```

### Run the service + UI

```bash
uvicorn app.api:app --reload          # http://localhost:8000/docs
streamlit run ui/streamlit_app.py     # http://localhost:8501
```

### Full Docker stack (real Qdrant server)

```bash
docker compose up --build
docker compose exec api python -m app.cli ingest
# UI: http://localhost:8501   API: http://localhost:8000/docs
```

### Use real models

```bash
export llm_provider=openai openai_api_key=sk-...      # hosted answers
export embedding_provider=openai                       # hosted embeddings
# or fully local:
export llm_provider=ollama llm_model=llama3.1
```

## Results (BGE-small, 14-question golden set)

| Metric | Hybrid | Dense-only |
|---|---|---|
| context_recall@k | **1.0** | 1.0 |
| answer_match | 1.0 | 1.0 |
| faithfulness (citation support) | 1.0 | 1.0 |
| refusal_accuracy | **1.0** | 1.0 |

On this small clean corpus BGE-small alone already reaches ceiling; the hybrid
gain shows up on **exact-token** queries (error codes, SKUs, API names — see
`tests/test_retrieval.py::test_bm25_catches_error_code`) and on larger, noisier
corpora. Both indexes share the same chunk IDs so RRF fusion is a clean merge.

## Configuration

All knobs are in `app/config.py` (env-overridable, `.env` supported). Key ones:
`llm_provider`, `embedding_provider`, `qdrant_url`/`qdrant_path`, `top_k`,
`bm25_floor`, `dense_floor` (refusal gate, calibrated for BGE-small).

## Layout

```
app/        config, models, embeddings, llm, ingest, vectorstore, sparse,
            retrieval, generation, index_service, api, cli
eval/       golden.jsonl + run_eval.py (RAGAS-style metrics)
ui/         streamlit_app.py
tests/      retrieval / generation / api  (pytest, real stack)
docs/       TUTORIAL.md, ARCHITECTURE.md
data/docs/  the support corpus
docker-compose.yml, Dockerfile, .env.example, requirements.txt, pyproject.toml
```

## Interview talking points

- Dense + sparse over the **same chunk IDs**; RRF fusion; why each catches
  different failures.
- Citation verification as a post-generation NLI-style check (+ optional LLM judge).
- Refusal gate on **raw** retrieval signals (RRF discards magnitude).
- Separating **retrieval** quality (recall) from **answer** quality in evals.
- Swappable providers behind one interface (local BGE/Ollama ↔ hosted OpenAI).
