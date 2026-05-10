from fastapi.testclient import TestClient

from app.main import app
from app.services.app_state import retrieval_service

client = TestClient(app)


def setup_function() -> None:
    retrieval_service.vector_store.clear()


def test_grounded_chat_api_returns_template_answer(tmp_path) -> None:
    path = tmp_path / "health.md"
    path.write_text(
        "Hypertension is related to blood pressure and salt intake.\n\n"
        "Docker images are used for software deployment.\n",
        encoding="utf-8",
    )
    ingest_response = client.post(
        "/documents/ingest-local",
        json={"path": str(path), "chunk_size": 80, "chunk_overlap": 10},
    )
    assert ingest_response.status_code == 200

    response = client.post(
        "/chat/grounded",
        json={"question": "What is hypertension?", "top_k": 2, "use_llm": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "template"
    assert payload["citations"]
    assert payload["retrieval_hits"]


def test_grounded_chat_api_rejects_blank_question() -> None:
    response = client.post("/chat/grounded", json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "question cannot be blank"
