from fastapi.testclient import TestClient

from app.main import app
from app.services.app_state import retrieval_service

client = TestClient(app)


def setup_function() -> None:
    retrieval_service.vector_store.clear()


def test_ingest_local_document_and_search(tmp_path) -> None:
    path = tmp_path / "health.md"
    path.write_text(
        "# Health notes\n\n"
        "Hypertension is related to blood pressure and salt intake.\n\n"
        "Docker images are used for software deployment.\n",
        encoding="utf-8",
    )

    ingest_response = client.post(
        "/documents/ingest-local",
        json={"path": str(path), "chunk_size": 80, "chunk_overlap": 10},
    )

    assert ingest_response.status_code == 200
    ingest_payload = ingest_response.json()
    assert ingest_payload["title"] == "health"
    assert ingest_payload["chunk_count"] >= 2
    assert ingest_payload["status"] == "chunked"

    search_response = client.post(
        "/retrieval/search",
        json={"query": "blood pressure hypertension", "top_k": 1},
    )

    assert search_response.status_code == 200
    search_payload = search_response.json()
    assert search_payload["query"] == "blood pressure hypertension"
    assert len(search_payload["hits"]) == 1
    assert search_payload["hits"][0]["rank"] == 1
    assert search_payload["hits"][0]["chunk"]["document_id"] == ingest_payload["document_id"]


def test_ingest_local_document_returns_404_for_missing_file() -> None:
    response = client.post(
        "/documents/ingest-local",
        json={"path": "E:/Agentpj/med-rag-agent/not-found.md"},
    )

    assert response.status_code == 404


def test_ingest_local_document_returns_400_for_unsupported_file(tmp_path) -> None:
    path = tmp_path / "image.png"
    path.write_text("not supported", encoding="utf-8")

    response = client.post("/documents/ingest-local", json={"path": str(path)})

    assert response.status_code == 400
    assert "Unsupported document type" in response.json()["detail"]


def test_search_rejects_blank_query() -> None:
    response = client.post("/retrieval/search", json={"query": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "query cannot be blank"
