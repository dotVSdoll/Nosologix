from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str
    raw: dict = Field(default_factory=dict)
