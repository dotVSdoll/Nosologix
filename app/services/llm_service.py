from __future__ import annotations

from typing import Protocol

import httpx

from app.config import settings
from app.schemas.llm import ChatMessage, LLMResponse


class LLMServiceError(RuntimeError):
    pass


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
        except httpx.HTTPError as exc:
            raise LLMServiceError(f"Qwen chat request failed: {exc}") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMServiceError("Qwen chat response has unexpected format") from exc

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
