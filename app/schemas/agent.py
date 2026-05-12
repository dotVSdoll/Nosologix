from pydantic import BaseModel, Field

from app.schemas.answer import GroundedAnswerRequest, GroundedAnswerResponse


class AgentStep(BaseModel):
    name: str
    status: str
    summary: str
    input_summary: str | None = None
    output_summary: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class AgenticRagRequest(GroundedAnswerRequest):
    include_trace: bool = True
    workflow_engine: str = "linear"


class AgenticRagResponse(BaseModel):
    question: str
    workflow_status: str
    workflow_engine: str = "linear"
    answer: GroundedAnswerResponse
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    steps: list[AgentStep] = Field(default_factory=list)


class AgentRunRecord(BaseModel):
    run_id: str
    created_at: str
    question: str
    workflow_status: str
    workflow_engine: str
    total_latency_ms: float = Field(ge=0.0)
    answer: dict
    steps: list[dict] = Field(default_factory=list)


class AgentRunsResponse(BaseModel):
    runs: list[AgentRunRecord]
    count: int
