from app.rag.embeddings import HashEmbeddingModel
from app.schemas.llm import ChatMessage, LLMResponse
from app.schemas.safety import RiskLevel
from app.services.answer_service import GroundedAnswerService
from app.services.retrieval_service import RetrievalService


class StubLLMClient:
    provider = "stub"
    model = "stub-model"

    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> LLMResponse:
        assert messages[0].role == "system"
        assert "[C1]" in messages[1].content
        return LLMResponse(
            content="Hypertension relates to blood pressure. [C1]",
            model=self.model,
            provider=self.provider,
        )


def test_grounded_answer_uses_llm_and_citations(tmp_path) -> None:
    path = tmp_path / "health.md"
    path.write_text(
        "Hypertension means blood pressure remains higher than recommended over time.\n\n"
        "Chest pain and difficulty breathing need urgent care.",
        encoding="utf-8",
    )
    retrieval = RetrievalService(embedding_model=HashEmbeddingModel(dimension=64))
    retrieval.ingest_and_index_document(path, chunk_size=90, chunk_overlap=10)
    service = GroundedAnswerService(retrieval_service=retrieval, llm_client=StubLLMClient())

    response = service.answer("What is hypertension?", top_k=2, min_score=0.0, use_llm=True)

    assert response.provider == "stub"
    assert response.used_model == "stub-model"
    assert response.citations
    assert "[C1]" in response.answer
    assert response.retrieval_hits
    assert response.risk_level == RiskLevel.MEDIUM
    assert response.should_seek_doctor is True


def test_grounded_answer_returns_low_confidence_without_evidence() -> None:
    retrieval = RetrievalService(embedding_model=HashEmbeddingModel(dimension=64))
    service = GroundedAnswerService(retrieval_service=retrieval, llm_client=StubLLMClient())

    response = service.answer("What is hypertension?", top_k=2)

    assert response.confidence == "low"
    assert response.citations == []
    assert response.used_model == "none"
    assert "not have enough" in response.answer


def test_grounded_answer_marks_emergency_even_without_evidence() -> None:
    retrieval = RetrievalService(embedding_model=HashEmbeddingModel(dimension=64))
    service = GroundedAnswerService(retrieval_service=retrieval, llm_client=StubLLMClient())

    response = service.answer("I have chest pain and difficulty breathing", top_k=2)

    assert response.confidence == "low"
    assert response.risk_level == RiskLevel.EMERGENCY
    assert response.should_seek_doctor is True
    assert response.safety_warnings
