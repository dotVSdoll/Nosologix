from pydantic import BaseModel, Field

from app.schemas.answer import GroundedAnswerRequest, GroundedAnswerResponse


class AgentStep(BaseModel):
    name: str
    status: str
    summary: str
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class AgenticRagRequest(GroundedAnswerRequest):
    include_trace: bool = True


class AgenticRagResponse(BaseModel):
    question: str
    workflow_status: str
    answer: GroundedAnswerResponse
    steps: list[AgentStep] = Field(default_factory=list)
