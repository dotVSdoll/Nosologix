import pytest

from app.rag.embeddings import HashEmbeddingModel, cosine_similarity
from app.rag.vector_store import InMemoryVectorStore
from app.schemas.chunk import DocumentChunk
from app.services.retrieval_service import RetrievalService


def _chunk(chunk_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="doc1",
        content=content,
        chunk_index=int(chunk_id[-1]),
        start_char=0,
        end_char=len(content),
        metadata={"source": "unit-test"},
    )


def test_hash_embedding_is_deterministic_and_normalized() -> None:
    model = HashEmbeddingModel(dimension=32)

    first = model.embed_text("hypertension blood pressure")
    second = model.embed_text("hypertension blood pressure")

    assert first == second
    assert len(first) == 32
    assert cosine_similarity(first, second) == pytest.approx(1.0)


def test_in_memory_vector_store_returns_most_relevant_chunk() -> None:
    model = HashEmbeddingModel(dimension=64)
    store = InMemoryVectorStore()
    chunks = [
        _chunk("chunk0", "blood pressure hypertension salt exercise"),
        _chunk("chunk1", "software deployment docker container"),
    ]
    store.upsert_many(chunks, model.embed_documents(chunk.content for chunk in chunks))

    hits = store.search(model.embed_text("hypertension pressure"), top_k=2)

    assert hits[0].chunk.id == "chunk0"
    assert hits[0].rank == 1
    assert hits[0].score >= hits[1].score
    assert store.count() == 2


def test_retrieval_service_ingests_indexes_and_searches(tmp_path) -> None:
    path = tmp_path / "health.md"
    path.write_text(
        "# Health notes\n\n"
        "Hypertension is related to blood pressure and salt intake.\n\n"
        "Docker images are used for software deployment.\n",
        encoding="utf-8",
    )
    service = RetrievalService(embedding_model=HashEmbeddingModel(dimension=64))

    ingested = service.ingest_and_index_document(path, chunk_size=80, chunk_overlap=10)
    hits = service.search("blood pressure hypertension", top_k=1)

    assert len(ingested.chunks) >= 2
    assert hits[0].chunk.document_id == ingested.document.id
    assert "Hypertension" in hits[0].chunk.content or "blood pressure" in hits[0].chunk.content


def test_vector_store_rejects_invalid_inputs() -> None:
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        store.search([0.1], top_k=0)
    with pytest.raises(ValueError):
        store.upsert_many([_chunk("chunk0", "content")], [])
