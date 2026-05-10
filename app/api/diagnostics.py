from fastapi import APIRouter

from app.config import settings
from app.schemas.diagnostics import LLMCheckResponse, LLMConfigDiagnostics
from app.schemas.llm import ChatMessage
from app.services.llm_service import LLMServiceError, create_llm_client

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


@router.get("/llm-config", response_model=LLMConfigDiagnostics)
def llm_config() -> LLMConfigDiagnostics:
    return LLMConfigDiagnostics(
        provider=settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.qwen_base_url if settings.llm_provider.lower() == "qwen" else None,
        has_api_key=bool(settings.qwen_api_key.strip()),
        timeout_seconds=settings.llm_timeout_seconds,
    )


@router.post("/llm-check", response_model=LLMCheckResponse)
def llm_check() -> LLMCheckResponse:
    try:
        client = create_llm_client()
        response = client.chat(
            [ChatMessage(role="user", content="Reply with OK only.")],
            temperature=0,
        )
    except LLMServiceError as exc:
        return LLMCheckResponse(
            available=False,
            provider=exc.provider,
            model=settings.llm_model,
            message=exc.safe_message(),
            error_type=exc.error_type,
            status_code=exc.status_code,
            error_code=exc.error_code,
            retryable=exc.retryable,
        )

    return LLMCheckResponse(
        available=True,
        provider=response.provider,
        model=response.model,
        message=response.content[:120],
    )
