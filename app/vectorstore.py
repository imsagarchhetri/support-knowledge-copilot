"""Qdrant dense vector store wrapper.

Connection modes (chosen from settings):
  - qdrant_in_memory=True            -> ephemeral, for tests
  - qdrant_url set                   -> remote server (docker-compose / prod)
  - else                             -> local on-disk (qdrant_path), for dev
"""
from __future__ import annotations

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .config import get_settings
from .embeddings import get_embedder
from .models import Chunk, RetrievedChunk


@lru_cache
def get_client() -> QdrantClient:
    s = get_settings()
    if s.qdrant_in_memory:
        return QdrantClient(":memory:")
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url)
    return QdrantClient(path=s.qdrant_path)


def _point_id(chunk_id: str) -> str:
    # Qdrant needs UUID/int ids; derive a stable UUID from the chunk_id.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    def __init__(self):
        self.s = get_settings()
        self.client = get_client()
        self.embedder = get_embedder()

    def recreate(self):
        if self.client.collection_exists(self.s.collection):
            self.client.delete_collection(self.s.collection)
        self.client.create_collection(
            self.s.collection,
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )

    def index(self, chunks: list[Chunk]):
        vectors = self.embedder.embed([c.text for c in chunks])
        points = [
            PointStruct(id=_point_id(c.chunk_id), vector=v, payload=c.model_dump())
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(self.s.collection, points=points)

    def search(self, query: str, limit: int) -> list[tuple[RetrievedChunk, float]]:
        qv = self.embedder.embed([query])[0]
        hits = self.client.query_points(self.s.collection, query=qv, limit=limit,
                                        with_payload=True).points
        out = []
        for h in hits:
            p = h.payload
            out.append((RetrievedChunk(chunk_id=p["chunk_id"], source=p["source"],
                                       heading=p["heading"], text=p["text"],
                                       score=h.score), h.score))
        return out

    def count(self) -> int:
        return self.client.count(self.s.collection).count
