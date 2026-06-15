import httpx
import pytest

from app.schemas.llm import ChatMessage
from app.services.llm_service import LLMServiceError, QwenLLMClient


def test_qwen_client_maps_http_status_to_safe_error(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        request = httpx.Request("POST", "https://example.test/chat/completions")
        response = httpx.Response(
            403,
            json={"error": {"code": "Forbidden", "message": "Model access denied"}},
            request=request,
        )
        return response

    monkeypatch.setattr(httpx, "post", fake_post)
    client = QwenLLMClient(api_key="test-key", base_url="https://example.test", model="qwen-plus")

    with pytest.raises(LLMServiceError) as exc_info:
        client.chat([ChatMessage(role="user", content="hello")])

    error = exc_info.value
    assert error.provider == "qwen"
    assert error.error_type == "http_status"
    assert error.status_code == 403
    assert error.error_code == "Forbidden"
    assert error.retryable is False
    assert "Model access denied" in error.safe_message()


def test_qwen_client_maps_timeout_to_retryable_error(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(httpx, "post", fake_post)
    client = QwenLLMClient(api_key="test-key", base_url="https://example.test", model="qwen-plus")

    with pytest.raises(LLMServiceError) as exc_info:
        client.chat([ChatMessage(role="user", content="hello")])

    error = exc_info.value
    assert error.error_type == "timeout"
    assert error.retryable is True
