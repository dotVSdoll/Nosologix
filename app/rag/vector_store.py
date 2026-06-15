from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.rag.embeddings import cosine_similarity
from app.schemas.chunk import DocumentChunk
from app.schemas.retrieval import RetrievalHit


@dataclass(frozen=True)
class VectorStoreItem:
    chunk: DocumentChunk
    vector: list[float]


class VectorStore(Protocol):
    def upsert_many(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError

    def search(self, query_vector: list[float], *, top_k: int = 5) -> list[RetrievalHit]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class VectorStoreProviderError(RuntimeError):
    pass


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


class ChromaVectorStore:
    """Persistent ChromaDB-backed vector store.

    ChromaDB is optional so CI and offline tests can keep using the in-memory store.
    """

    def __init__(
        self,
        *,
        persist_path: str | Path,
        collection_name: str = "med_rag_chunks",
    ) -> None:
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self._client = self._create_client(self.persist_path)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def upsert_many(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.content for chunk in chunks],
            metadatas=[_chunk_to_chroma_metadata(chunk) for chunk in chunks],
        )

    def search(self, query_vector: list[float], *, top_k: int = 5) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = _first_result_list(result.get("documents"))
        metadatas = _first_result_list(result.get("metadatas"))
        distances = _first_result_list(result.get("distances"))

        hits: list[RetrievalHit] = []
        for index, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances, strict=False),
            start=1,
        ):
            chunk = _chunk_from_chroma_metadata(
                content=str(document or ""),
                metadata=dict(metadata or {}),
            )
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=_distance_to_similarity(float(distance)),
                    rank=index,
                )
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    def clear(self) -> None:
        existing = self._collection.get()
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)

    @staticmethod
    def _create_client(persist_path: Path) -> Any:
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreProviderError(
                "ChromaDB vector store requires the optional rag dependencies. "
                "Install them with: pip install -e .[rag]"
            ) from exc

        persist_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(persist_path))


def create_vector_store(
    *,
    provider: str,
    persist_path: str | Path,
    collection_name: str = "med_rag_chunks",
) -> VectorStore:
    normalized_provider = provider.strip().lower()
    if normalized_provider in {"memory", "in-memory", "in_memory", "local"}:
        return InMemoryVectorStore()
    if normalized_provider in {"chroma", "chromadb"}:
        return ChromaVectorStore(
            persist_path=persist_path,
            collection_name=collection_name,
        )
    raise VectorStoreProviderError(f"Unsupported vector store provider: {provider}")


def _chunk_to_chroma_metadata(chunk: DocumentChunk) -> dict[str, str | int]:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "metadata_json": json.dumps(chunk.metadata, ensure_ascii=False),
    }


def _chunk_from_chroma_metadata(*, content: str, metadata: dict[str, Any]) -> DocumentChunk:
    raw_metadata = metadata.get("metadata_json", "{}")
    try:
        chunk_metadata = json.loads(str(raw_metadata))
    except json.JSONDecodeError:
        chunk_metadata = {}

    return DocumentChunk(
        id=str(metadata["chunk_id"]),
        document_id=str(metadata["document_id"]),
        content=content,
        chunk_index=int(metadata["chunk_index"]),
        start_char=int(metadata["start_char"]),
        end_char=int(metadata["end_char"]),
        metadata=chunk_metadata,
    )


def _first_result_list(value: Any) -> list[Any]:
    if not value:
        return []
    first = value[0]
    return list(first) if first else []


def _distance_to_similarity(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance))
