from pydantic import BaseModel, Field

from app.schemas.retrieval import RetrievalHit
from app.schemas.safety import RiskLevel


class Citation(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str
    title: str | None = None
    source: str | None = None
    excerpt: str
    score: float


class GroundedAnswerRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0, le=10)
    min_score: float = Field(default=0.05, ge=0.0, le=1.0)
    min_citations: int = Field(default=1, ge=1, le=5)
    use_llm: bool = True


class GroundedAnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    confidence: str
    limitations: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    should_seek_doctor: bool = False
    safety_warnings: list[str] = Field(default_factory=list)
    evidence_status: str = "sufficient"
    evidence_warnings: list[str] = Field(default_factory=list)
    llm_error_type: str | None = None
    llm_error_status_code: int | None = None
    llm_error_code: str | None = None
    llm_error_retryable: bool | None = None
    used_model: str
    provider: str
    retrieval_hits: list[RetrievalHit] = Field(default_factory=list)
