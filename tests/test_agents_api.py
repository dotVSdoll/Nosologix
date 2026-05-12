from fastapi.testclient import TestClient

from app.main import app
from app.services.app_state import retrieval_service

client = TestClient(app)


def setup_function() -> None:
    retrieval_service.vector_store.clear()


def test_agentic_rag_api_writes_trace(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "agent_runs.jsonl"
    monkeypatch.setattr("app.config.settings.agent_trace_enabled", True)
    monkeypatch.setattr("app.config.settings.agent_trace_path", str(trace_path))

    response = client.post(
        "/agents/rag",
        json={
            "question": "What is hypertension?",
            "top_k": 2,
            "use_llm": False,
            "include_trace": True,
            "workflow_engine": "linear",
        },
    )

    assert response.status_code == 200
    assert trace_path.exists()
    assert trace_path.read_text(encoding="utf-8").count("\n") == 1


def test_list_agent_runs_api_returns_recent_runs(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "agent_runs.jsonl"
    monkeypatch.setattr("app.config.settings.agent_trace_enabled", True)
    monkeypatch.setattr("app.config.settings.agent_trace_path", str(trace_path))
    for question in ("What is hypertension?", "Does insurance cover this?"):
        response = client.post(
            "/agents/rag",
            json={
                "question": question,
                "top_k": 2,
                "use_llm": False,
                "include_trace": True,
                "workflow_engine": "linear",
            },
        )
        assert response.status_code == 200

    response = client.get("/agents/runs?limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["runs"][0]["question"] == "Does insurance cover this?"


def test_list_agent_runs_api_rejects_invalid_limit() -> None:
    response = client.get("/agents/runs?limit=0")

    assert response.status_code == 400


def test_agentic_rag_api_rejects_unknown_workflow_engine() -> None:
    response = client.post(
        "/agents/rag",
        json={
            "question": "What is hypertension?",
            "workflow_engine": "unknown",
        },
    )

    assert response.status_code == 400
