from fastapi.testclient import TestClient

from app.main import app
from app.schemas.llm import ChatMessage
from app.services.llm_service import LLMServiceError

client = TestClient(app)


class HealthyClient:
    provider = "stub"
    model = "stub-model"

    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.2):
        from app.schemas.llm import LLMResponse

        return LLMResponse(content="OK", model=self.model, provider=self.provider)


class UnhealthyClient:
    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.2):
        raise LLMServiceError(
            "Provider denied access",
            provider="qwen",
            error_type="http_status",
            status_code=403,
            error_code="Forbidden",
            retryable=False,
        )


def test_llm_config_does_not_expose_api_key() -> None:
    response = client.get("/diagnostics/llm-config")

    assert response.status_code == 200
    payload = response.json()
    assert "has_api_key" in payload
    assert "api_key" not in payload
    assert "qwen_api_key" not in payload


def test_llm_check_returns_available_status(monkeypatch) -> None:
    monkeypatch.setattr("app.api.diagnostics.create_llm_client", lambda: HealthyClient())

    response = client.post("/diagnostics/llm-check")

    assert response.status_code == 200
    assert response.json()["available"] is True


def test_llm_check_returns_safe_error_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr("app.api.diagnostics.create_llm_client", lambda: UnhealthyClient())

    response = client.post("/diagnostics/llm-check")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["provider"] == "qwen"
    assert payload["status_code"] == 403
    assert payload["error_code"] == "Forbidden"
    assert "Provider denied access" in payload["message"]
