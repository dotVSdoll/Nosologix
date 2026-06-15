from pydantic import BaseModel, Field

from app.schemas.chunk import DocumentChunk


class RetrievalHit(BaseModel):
    chunk: DocumentChunk
    score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    retrieval_score: float | None = Field(default=None, ge=0.0, le=1.0)
    rerank_score: float | None = Field(default=None, ge=0.0, le=1.0)
