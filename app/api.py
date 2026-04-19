"""FastAPI service.

    uvicorn app.api:app --reload

Endpoints:
    GET  /health
    POST /ingest   {rebuild, strategy}
    POST /ask      {question, hybrid, top_k}  -> Answer
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from . import __version__
from .config import get_settings
from .generation import AnswerService
from .index_service import build_index
from .models import AskRequest, IngestRequest, Answer

app = FastAPI(title="Support Knowledge Copilot", version=__version__)
_service: AnswerService | None = None


def service() -> AnswerService:
    global _service
    if _service is None:
        _service = AnswerService()
    return _service


@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "version": __version__,
            "llm_provider": s.llm_provider, "embedding_provider": s.embedding_provider}


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        n = build_index(strategy=req.strategy, rebuild=req.rebuild)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    global _service
    _service = None   # reset so the next /ask reloads the fresh index
    return {"indexed_chunks": n}


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest):
    try:
        return service().ask(req.question, hybrid=req.hybrid, top_k=req.top_k)
    except FileNotFoundError:
        raise HTTPException(409, "index not built; POST /ingest first")
