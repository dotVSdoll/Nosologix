from __future__ import annotations

from dataclasses import dataclass

from app.rag.embeddings import cosine_similarity
from app.schemas.chunk import DocumentChunk
from app.schemas.retrieval import RetrievalHit


@dataclass(frozen=True)
class VectorStoreItem:
    chunk: DocumentChunk
    vector: list[float]


class InMemoryVectorStore:
    """Small vector store for local development and deterministic tests."""

    def __init__(self) -> None:
        self._items: dict[str, VectorStoreItem] = {}

    def upsert(self, chunk: DocumentChunk, vector: list[float]) -> None:
        self._items[chunk.id] = VectorStoreItem(chunk=chunk, vector=vector)

    def upsert_many(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.upsert(chunk, vector)

    def search(self, query_vector: list[float], *, top_k: int = 5) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        scored_items = [
            (item.chunk, cosine_similarity(query_vector, item.vector))
            for item in self._items.values()
        ]
        scored_items.sort(key=lambda item: item[1], reverse=True)

        return [
            RetrievalHit(chunk=chunk, score=score, rank=index + 1)
            for index, (chunk, score) in enumerate(scored_items[:top_k])
        ]

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()
