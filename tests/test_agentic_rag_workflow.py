from app.agents.rag_workflow import AgenticRagWorkflow
from app.rag.embeddings import HashEmbeddingModel
from app.rag.vector_store import InMemoryVectorStore
from app.services.retrieval_service import RetrievalService


def _retrieval_service() -> RetrievalService:
    return RetrievalService(
        embedding_model=HashEmbeddingModel(dimension=64),
        vector_store=InMemoryVectorStore(),
        reranker=None,
    )


def test_agentic_rag_workflow_returns_trace_and_answer(tmp_path) -> None:
    path = tmp_path / "health.md"
    path.write_text(
        "Hypertension means blood pressure remains higher than recommended.\n\n"
        "Lifestyle management includes reducing salt intake and regular exercise.",
        encoding="utf-8",
    )
    retrieval = _retrieval_service()
    retrieval.ingest_and_index_document(path, chunk_size=120, chunk_overlap=10)
    workflow = AgenticRagWorkflow(retrieval_service=retrieval)

    response = workflow.run("What is hypertension?", top_k=2, min_score=0.0, use_llm=False)

    assert response.workflow_status == "completed"
    assert response.total_latency_ms >= 0.0
    assert response.answer.citations
    assert [step.name for step in response.steps] == [
        "query_planner",
        "retriever",
        "evidence_critic",
        "safety_reviewer",
        "answer_composer",
    ]
    assert response.steps[0].metadata["intent"] == "medical_qa"
    assert all(step.latency_ms >= 0.0 for step in response.steps)
    assert all(step.input_summary for step in response.steps)
    assert all(step.output_summary for step in response.steps)
    assert "top_chunk=" in response.steps[1].output_summary


def test_agentic_rag_workflow_can_hide_trace() -> None:
    workflow = AgenticRagWorkflow(retrieval_service=_retrieval_service())

    response = workflow.run("Does insurance cover this?", include_trace=False, use_llm=False)

    assert response.steps == []
    assert response.total_latency_ms >= 0.0
    assert response.workflow_status == "needs_more_evidence"
    assert response.answer.evidence_status == "insufficient"


def test_agentic_rag_workflow_marks_high_risk_status(tmp_path) -> None:
    path = tmp_path / "meds.md"
    path.write_text(
        "Medication dose decisions depend on age, weight, allergies, and diagnosis.",
        encoding="utf-8",
    )
    retrieval = _retrieval_service()
    retrieval.ingest_and_index_document(path, chunk_size=120, chunk_overlap=10)
    workflow = AgenticRagWorkflow(retrieval_service=retrieval)

    response = workflow.run(
        "How much aspirin should I take as a dose?",
        top_k=1,
        min_score=0.0,
        use_llm=False,
    )

    assert response.workflow_status == "completed_with_safety_warning"
    assert response.answer.risk_level == "high"
