from app.agents.langgraph_workflow import LangGraphAgenticRagWorkflow
from app.rag.embeddings import HashEmbeddingModel
from app.rag.vector_store import InMemoryVectorStore
from app.services.retrieval_service import RetrievalService


class CountingRetrievalService(RetrievalService):
    def __init__(self) -> None:
        super().__init__(
            embedding_model=HashEmbeddingModel(dimension=64),
            vector_store=InMemoryVectorStore(),
            reranker=None,
        )
        self.search_count = 0

    def search(self, query: str, *, top_k: int = 5):
        self.search_count += 1
        return super().search(query, top_k=top_k)


def test_langgraph_agentic_rag_workflow_returns_trace(tmp_path) -> None:
    path = tmp_path / "health.md"
    path.write_text(
        "Hypertension means high blood pressure.\n\n"
        "Lifestyle management includes reducing salt intake.",
        encoding="utf-8",
    )
    retrieval = CountingRetrievalService()
    retrieval.ingest_and_index_document(path, chunk_size=120, chunk_overlap=10)
    workflow = LangGraphAgenticRagWorkflow(retrieval_service=retrieval)

    response = workflow.run("What is hypertension?", min_score=0.0, use_llm=False)

    assert response.workflow_engine == "langgraph"
    assert response.workflow_status == "completed"
    assert retrieval.search_count == 1
    assert response.answer.citations
    assert [step.name for step in response.steps] == [
        "query_planner",
        "retriever",
        "evidence_critic",
        "safety_reviewer",
        "answer_composer",
    ]
    assert all(step.latency_ms >= 0.0 for step in response.steps)


def test_langgraph_agentic_rag_workflow_can_hide_trace() -> None:
    workflow = LangGraphAgenticRagWorkflow(retrieval_service=CountingRetrievalService())

    response = workflow.run("Does insurance cover this?", use_llm=False, include_trace=False)

    assert response.workflow_engine == "langgraph"
    assert response.steps == []
    assert response.workflow_status == "needs_more_evidence"


def test_langgraph_agentic_rag_workflow_routes_insufficient_evidence_to_abstain() -> None:
    workflow = LangGraphAgenticRagWorkflow(retrieval_service=CountingRetrievalService())

    response = workflow.run("Does insurance cover this?", use_llm=True, include_trace=True)

    assert response.workflow_status == "needs_more_evidence"
    assert response.answer.provider == "none"
    assert [step.name for step in response.steps] == [
        "query_planner",
        "retriever",
        "evidence_critic",
        "abstain_composer",
    ]
    assert "safety_reviewer" not in {step.name for step in response.steps}
    assert "answer_composer" not in {step.name for step in response.steps}
