from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_demo_page_loads() -> None:
    response = client.get("/demo")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Traceable Healthcare Agentic RAG" in response.text
    assert "/agents/rag" in response.text
    assert "/agents/runs?limit=8" in response.text
