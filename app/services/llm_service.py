from __future__ import annotations

from typing import Any, Protocol

import httpx

from app.config import settings
from app.schemas.llm import ChatMessage, LLMResponse


class LLMServiceError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider: str = "unknown",
        error_type: str = "unknown",
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.error_type = error_type
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable

    def safe_message(self) -> str:
        status = f" HTTP {self.status_code}." if self.status_code else "."
        code = f" Provider code: {self.error_code}." if self.error_code else ""
        return f"{self.args[0]}{status}{code}"


class LLMClient(Protocol):
    provider: str
    model: str

    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> LLMResponse:
        raise NotImplementedError


class TemplateLLMClient:
    provider = "template"
    model = "template-local"

    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> LLMResponse:
        user_content = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        return LLMResponse(content=user_content, model=self.model, provider=self.provider)


class QwenLLMClient:
    provider = "qwen"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key.strip():
            raise LLMServiceError("QWEN_API_KEY is required when LLM_PROVIDER=qwen")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error_code, provider_message = _extract_provider_error(exc.response)
            raise LLMServiceError(
                provider_message or "Qwen chat request was rejected by the provider",
                provider=self.provider,
                error_type="http_status",
                status_code=exc.response.status_code,
                error_code=error_code,
                retryable=exc.response.status_code in {408, 429, 500, 502, 503, 504},
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMServiceError(
                "Qwen chat request timed out",
                provider=self.provider,
                error_type="timeout",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMServiceError(
                "Qwen chat request failed before receiving a valid response",
                provider=self.provider,
                error_type=exc.__class__.__name__,
                retryable=True,
            ) from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMServiceError(
                "Qwen chat response has unexpected format",
                provider=self.provider,
                error_type="invalid_response",
            ) from exc

        return LLMResponse(content=content, model=self.model, provider=self.provider, raw=data)


def create_llm_client() -> LLMClient:
    provider = settings.llm_provider.strip().lower()
    if provider in {"template", "local", "fake"}:
        return TemplateLLMClient()
    if provider in {"qwen", "dashscope"}:
        return QwenLLMClient(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    raise LLMServiceError(f"Unsupported LLM provider: {settings.llm_provider}")


def _extract_provider_error(response: httpx.Response) -> tuple[str | None, str | None]:
    try:
        data: Any = response.json()
    except ValueError:
        return None, None

    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return None, None

    code = error.get("code")
    message = error.get("message")
    return (
        str(code) if code else None,
        str(message) if message else None,
    )
