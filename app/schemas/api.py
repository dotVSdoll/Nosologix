from pydantic import BaseModel, Field

from app.schemas.retrieval import RetrievalHit


class IngestLocalDocumentRequest(BaseModel):
    path: str = Field(min_length=1)
    chunk_size: int = Field(default=800, gt=0, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)


class IngestLocalDocumentResponse(BaseModel):
    document_id: str
    title: str
    source_path: str
    chunk_count: int
    status: str


class RetrievalSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0, le=20)


class RetrievalSearchResponse(BaseModel):
    query: str
    hits: list[RetrievalHit]
