from pydantic import BaseModel


class LLMConfigDiagnostics(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    has_api_key: bool
    timeout_seconds: float


class LLMCheckResponse(BaseModel):
    available: bool
    provider: str
    model: str
    message: str
    error_type: str | None = None
    status_code: int | None = None
    error_code: str | None = None
    retryable: bool | None = None
