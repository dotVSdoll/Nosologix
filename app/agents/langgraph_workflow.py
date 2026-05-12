from __future__ import annotations

from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.rag_workflow import (
    AgenticRagWorkflow,
    _elapsed_ms,
    _evidence_summary,
    _plan_query,
    _retriever_step,
    _summarize_text,
    _workflow_status,
)
from app.schemas.agent import AgenticRagResponse, AgentStep
from app.schemas.answer import GroundedAnswerResponse
from app.schemas.retrieval import RetrievalHit
from app.services.evidence_service import (
    build_evidence_warnings,
    evidence_status_from_warnings,
)
from app.services.llm_service import LLMClient
from app.services.retrieval_service import RetrievalService
from app.services.safety_service import assess_medical_safety


class AgenticRagGraphState(TypedDict, total=False):
    question: str
    top_k: int
    min_score: float
    min_citations: int
    use_llm: bool
    include_trace: bool
    workflow_start: float
    retrieval_query: str
    intent: str
    hits: list[RetrievalHit]
    evidence_status: str
    evidence_warnings: list[str]
    answer: GroundedAnswerResponse
    steps: list[AgentStep]
    response: AgenticRagResponse


class LangGraphAgenticRagWorkflow(AgenticRagWorkflow):
    """LangGraph-backed version of the dependency-light Agentic RAG workflow."""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        llm_client: LLMClient | None = None,
    ) -> None:
        super().__init__(retrieval_service=retrieval_service, llm_client=llm_client)
        self.graph = self._build_graph()

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
        result = self.graph.invoke(
            {
                "question": question,
                "top_k": top_k,
                "min_score": min_score,
                "min_citations": min_citations,
                "use_llm": use_llm,
                "include_trace": include_trace,
                "steps": [],
                "workflow_start": perf_counter(),
            }
        )
        return result["response"]

    def _build_graph(self) -> Any:
        graph = StateGraph(AgenticRagGraphState)
        graph.add_node("query_planner", self._query_planner_node)
        graph.add_node("retriever", self._retriever_node)
        graph.add_node("evidence_critic", self._evidence_critic_node)
        graph.add_node("safety_reviewer", self._safety_reviewer_node)
        graph.add_node("abstain_composer", self._abstain_composer_node)
        graph.add_node("answer_composer", self._answer_composer_node)
        graph.add_edge(START, "query_planner")
        graph.add_edge("query_planner", "retriever")
        graph.add_edge("retriever", "evidence_critic")
        graph.add_conditional_edges(
            "evidence_critic",
            _route_after_evidence,
            {
                "safety_reviewer": "safety_reviewer",
                "abstain_composer": "abstain_composer",
            },
        )
        graph.add_edge("abstain_composer", END)
        graph.add_edge("safety_reviewer", "answer_composer")
        graph.add_edge("answer_composer", END)
        return graph.compile()

    def _query_planner_node(self, state: AgenticRagGraphState) -> dict[str, Any]:
        step_start = perf_counter()
        planned_query = _plan_query(state["question"])
        step = AgentStep(
            name="query_planner",
            status="completed",
            summary=f"Planned retrieval query for intent: {planned_query.intent}.",
            input_summary=_summarize_text(state["question"]),
            output_summary=_summarize_text(planned_query.retrieval_query),
            latency_ms=_elapsed_ms(step_start),
            metadata={
                "intent": planned_query.intent,
                "query_length": len(planned_query.retrieval_query),
            },
        )
        return {
            "question": planned_query.original_question,
            "retrieval_query": planned_query.retrieval_query,
            "intent": planned_query.intent,
            "steps": [*state.get("steps", []), step],
        }

    def _retriever_node(self, state: AgenticRagGraphState) -> dict[str, Any]:
        step_start = perf_counter()
        hits = self.retrieval_service.search(state["retrieval_query"], top_k=state["top_k"])
        step = _retriever_step(
            hits=hits,
            query=state["retrieval_query"],
            latency_ms=_elapsed_ms(step_start),
        )
        return {
            "hits": hits,
            "steps": [*state.get("steps", []), step],
        }

    def _evidence_critic_node(self, state: AgenticRagGraphState) -> dict[str, Any]:
        step_start = perf_counter()
        evidence_warnings = build_evidence_warnings(
            state["hits"],
            min_score=state["min_score"],
            min_citations=state["min_citations"],
        )
        evidence_status = evidence_status_from_warnings(evidence_warnings)
        step = AgentStep(
            name="evidence_critic",
            status=evidence_status,
            summary=_evidence_summary(evidence_status, evidence_warnings),
            input_summary=f"{len(state['hits'])} hit(s), min_score={state['min_score']}",
            output_summary=evidence_status,
            latency_ms=_elapsed_ms(step_start),
            metadata={
                "min_score": state["min_score"],
                "min_citations": state["min_citations"],
                "warning_count": len(evidence_warnings),
            },
        )
        return {
            "evidence_status": evidence_status,
            "evidence_warnings": evidence_warnings,
            "steps": [*state.get("steps", []), step],
        }

    def _safety_reviewer_node(self, state: AgenticRagGraphState) -> dict[str, Any]:
        step_start = perf_counter()
        safety = assess_medical_safety(state["question"])
        step = AgentStep(
            name="safety_reviewer",
            status=safety.risk_level.value,
            summary="Reviewed medical safety risk for the user question.",
            input_summary=_summarize_text(state["question"]),
            output_summary=safety.risk_level.value,
            latency_ms=_elapsed_ms(step_start),
            metadata={
                "should_seek_doctor": safety.should_seek_doctor,
                "matched_rule_count": len(safety.matched_rules),
            },
        )
        return {"steps": [*state.get("steps", []), step]}

    def _abstain_composer_node(self, state: AgenticRagGraphState) -> dict[str, Any]:
        step_start = perf_counter()
        answer = self._answer(
            state["question"],
            hits=state["hits"],
            min_score=state["min_score"],
            min_citations=state["min_citations"],
            use_llm=False,
        )
        step = AgentStep(
            name="abstain_composer",
            status="completed",
            summary="Composed abstain answer because evidence was insufficient.",
            input_summary=f"{len(state['hits'])} hit(s), evidence_status=insufficient",
            output_summary=_summarize_text(answer.answer),
            latency_ms=_elapsed_ms(step_start),
            metadata={
                "confidence": answer.confidence,
                "citation_count": len(answer.citations),
                "used_llm": False,
            },
        )
        steps = [*state.get("steps", []), step]
        response = AgenticRagResponse(
            question=state["question"],
            workflow_status="needs_more_evidence",
            workflow_engine="langgraph",
            answer=answer,
            steps=steps if state["include_trace"] else [],
            total_latency_ms=_elapsed_ms(state["workflow_start"]),
        )
        return {
            "answer": answer,
            "response": response,
            "steps": steps,
        }

    def _answer_composer_node(self, state: AgenticRagGraphState) -> dict[str, Any]:
        step_start = perf_counter()
        answer = self._answer(
            state["question"],
            hits=state["hits"],
            min_score=state["min_score"],
            min_citations=state["min_citations"],
            use_llm=state["use_llm"],
        )
        step = AgentStep(
            name="answer_composer",
            status="completed",
            summary=f"Composed answer with provider={answer.provider}.",
            input_summary=f"{len(answer.citations)} citation(s), use_llm={state['use_llm']}",
            output_summary=_summarize_text(answer.answer),
            latency_ms=_elapsed_ms(step_start),
            metadata={
                "confidence": answer.confidence,
                "citation_count": len(answer.citations),
                "used_llm": answer.provider not in {"none", "template"},
            },
        )
        steps = [*state.get("steps", []), step]
        response = AgenticRagResponse(
            question=state["question"],
            workflow_status=_workflow_status(answer),
            workflow_engine="langgraph",
            answer=answer,
            steps=steps if state["include_trace"] else [],
            total_latency_ms=_elapsed_ms(state["workflow_start"]),
        )
        return {
            "answer": answer,
            "response": response,
            "steps": steps,
        }


def _route_after_evidence(state: AgenticRagGraphState) -> str:
    if state["evidence_status"] == "insufficient":
        return "abstain_composer"
    return "safety_reviewer"
