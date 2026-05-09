from pathlib import Path

from app.config import settings
from app.rag.embeddings import EmbeddingModel, create_embedding_model
from app.rag.vector_store import InMemoryVectorStore
from app.schemas.chunk import DocumentChunk
from app.schemas.ingestion import IngestedDocument
from app.schemas.retrieval import RetrievalHit
from app.services.ingestion_service import ingest_local_document


class RetrievalService:
    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel | None = None,
        vector_store: InMemoryVectorStore | None = None,
    ) -> None:
        self.embedding_model = embedding_model or create_embedding_model(
            provider=settings.embedding_provider,
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            device=settings.embedding_device,
            use_fp16=settings.embedding_use_fp16,
        )
        self.vector_store = vector_store or InMemoryVectorStore()

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        vectors = self.embedding_model.embed_documents(chunk.content for chunk in chunks)
        self.vector_store.upsert_many(chunks, vectors)
        return len(chunks)

    def ingest_and_index_document(
        self,
        path: str | Path,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> IngestedDocument:
        result = ingest_local_document(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.index_chunks(result.chunks)
        return result

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        query_vector = self.embedding_model.embed_text(query)
        return self.vector_store.search(query_vector, top_k=top_k)
