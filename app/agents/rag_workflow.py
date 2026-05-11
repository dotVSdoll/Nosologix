from __future__ import annotations

from dataclasses import dataclass

from app.schemas.agent import AgenticRagResponse, AgentStep
from app.schemas.answer import GroundedAnswerResponse
from app.schemas.retrieval import RetrievalHit
from app.services.answer_service import GroundedAnswerService
from app.services.evidence_service import (
    build_evidence_warnings,
    evidence_status_from_warnings,
)
from app.services.llm_service import LLMClient
from app.services.retrieval_service import RetrievalService
from app.services.safety_service import assess_medical_safety


@dataclass(frozen=True)
class PlannedQuery:
    original_question: str
    retrieval_query: str
    intent: str


class AgenticRagWorkflow:
    """Small agentic RAG orchestrator that can later be swapped to LangGraph nodes."""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client

    def run(
        self,
        question: str,
        *,
        top_k: int = 5,
        min_score: float = 0.05,
        min_citations: int = 1,
        use_llm: bool = True,
        include_trace: bool = True,
    ) -> AgenticRagResponse:
        steps: list[AgentStep] = []
        planned_query = _plan_query(question)
        steps.append(
            AgentStep(
                name="query_planner",
                status="completed",
                summary=f"Planned retrieval query for intent: {planned_query.intent}.",
                metadata={
                    "intent": planned_query.intent,
                    "query_length": len(planned_query.retrieval_query),
                },
            )
        )

        hits = self.retrieval_service.search(planned_query.retrieval_query, top_k=top_k)
        steps.append(_retriever_step(hits))

        evidence_warnings = build_evidence_warnings(
            hits,
            min_score=min_score,
            min_citations=min_citations,
        )
        evidence_status = evidence_status_from_warnings(evidence_warnings)
        steps.append(
            AgentStep(
                name="evidence_critic",
                status=evidence_status,
                summary=_evidence_summary(evidence_status, evidence_warnings),
                metadata={
                    "min_score": min_score,
                    "min_citations": min_citations,
                    "warning_count": len(evidence_warnings),
                },
            )
        )

        safety = assess_medical_safety(planned_query.original_question)
        steps.append(
            AgentStep(
                name="safety_reviewer",
                status=safety.risk_level.value,
                summary="Reviewed medical safety risk for the user question.",
                metadata={
                    "should_seek_doctor": safety.should_seek_doctor,
                    "matched_rule_count": len(safety.matched_rules),
                },
            )
        )

        answer = self._answer(
            planned_query.original_question,
            top_k=top_k,
            min_score=min_score,
            min_citations=min_citations,
            use_llm=use_llm,
        )
        steps.append(
            AgentStep(
                name="answer_composer",
                status="completed",
                summary=f"Composed answer with provider={answer.provider}.",
                metadata={
                    "confidence": answer.confidence,
                    "citation_count": len(answer.citations),
                    "used_llm": answer.provider not in {"none", "template"},
                },
            )
        )

        return AgenticRagResponse(
            question=planned_query.original_question,
            workflow_status=_workflow_status(answer),
            answer=answer,
            steps=steps if include_trace else [],
        )

    def _answer(
        self,
        question: str,
        *,
        top_k: int,
        min_score: float,
        min_citations: int,
        use_llm: bool,
    ) -> GroundedAnswerResponse:
        service = GroundedAnswerService(
            retrieval_service=self.retrieval_service,
            llm_client=self.llm_client,
        )
        return service.answer(
            question,
            top_k=top_k,
            min_score=min_score,
            min_citations=min_citations,
            use_llm=use_llm,
        )


def _plan_query(question: str) -> PlannedQuery:
    normalized_question = " ".join(question.strip().split())
    intent = "medical_qa" if _looks_medical(normalized_question) else "general_qa"
    return PlannedQuery(
        original_question=normalized_question,
        retrieval_query=normalized_question,
        intent=intent,
    )


def _looks_medical(question: str) -> bool:
    text = question.lower()
    medical_terms = {
        "blood",
        "breathing",
        "chest pain",
        "diabetes",
        "dose",
        "health",
        "hypertension",
        "lab",
        "medication",
        "pregnant",
        "symptom",
    }
    return any(term in text for term in medical_terms)


def _retriever_step(hits: list[RetrievalHit]) -> AgentStep:
    top_score = hits[0].score if hits else 0.0
    return AgentStep(
        name="retriever",
        status="completed",
        summary=f"Retrieved {len(hits)} candidate chunk(s).",
        metadata={
            "hit_count": len(hits),
            "top_score": round(top_score, 4),
        },
    )


def _evidence_summary(evidence_status: str, evidence_warnings: list[str]) -> str:
    if evidence_status == "sufficient":
        return "Evidence passed quality thresholds."
    return "; ".join(evidence_warnings) or "Evidence did not pass quality thresholds."


def _workflow_status(answer: GroundedAnswerResponse) -> str:
    if answer.evidence_status == "insufficient":
        return "needs_more_evidence"
    if answer.risk_level.value in {"high", "emergency"}:
        return "completed_with_safety_warning"
    return "completed"
