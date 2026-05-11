from pathlib import Path

from app.config import settings
from app.rag.embeddings import EmbeddingModel, create_embedding_model
from app.rag.rerankers import Reranker, create_reranker
from app.rag.vector_store import VectorStore, create_vector_store
from app.schemas.chunk import DocumentChunk
from app.schemas.ingestion import IngestedDocument
from app.schemas.retrieval import RetrievalHit
from app.services.ingestion_service import ingest_local_document

_DEFAULT_RERANKER = object()


class RetrievalService:
    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None | object = _DEFAULT_RERANKER,
    ) -> None:
        self.embedding_model = embedding_model or create_embedding_model(
            provider=settings.embedding_provider,
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            device=settings.embedding_device,
            use_fp16=settings.embedding_use_fp16,
        )
        self.vector_store = vector_store or create_vector_store(
            provider=settings.vector_store_provider,
            persist_path=settings.vector_store_path,
            collection_name=settings.vector_store_collection,
        )
        if reranker is _DEFAULT_RERANKER:
            self.reranker = create_reranker(
                provider=settings.reranker_provider,
                model_name=settings.reranker_model,
                use_fp16=settings.reranker_use_fp16,
            )
        else:
            self.reranker = reranker

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
        hits = self.vector_store.search(query_vector, top_k=top_k)
        if not self.reranker or not hits:
            return hits
        return rerank_hits(query=query, hits=hits, reranker=self.reranker)


def rerank_hits(*, query: str, hits: list[RetrievalHit], reranker: Reranker) -> list[RetrievalHit]:
    scores = reranker.score(query, [hit.chunk.content for hit in hits])
    if len(scores) != len(hits):
        raise ValueError("reranker scores and hits must have the same length")

    reranked_hits = [
        RetrievalHit(
            chunk=hit.chunk,
            score=hit.score,
            rank=hit.rank,
            retrieval_score=hit.retrieval_score if hit.retrieval_score is not None else hit.score,
            rerank_score=score,
        )
        for hit, score in zip(hits, scores, strict=True)
    ]
    reranked_hits.sort(key=lambda hit: hit.rerank_score or 0.0, reverse=True)
    return [
        hit.model_copy(update={"rank": index})
        for index, hit in enumerate(reranked_hits, start=1)
    ]
