from __future__ import annotations

from app.schemas.answer import Citation, GroundedAnswerResponse
from app.schemas.llm import ChatMessage
from app.schemas.retrieval import RetrievalHit
from app.services.llm_service import (
    LLMClient,
    LLMServiceError,
    TemplateLLMClient,
    create_llm_client,
)
from app.services.retrieval_service import RetrievalService

SYSTEM_PROMPT = """You are a healthcare knowledge assistant for retrieval-augmented answers.
Follow these rules:
- Answer only from the provided evidence.
- Cite evidence with [C1], [C2] style markers.
- Do not diagnose, prescribe medication, or replace professional medical care.
- If evidence is insufficient, say so clearly.
- For urgent or severe symptoms, advise seeking emergency medical care.
"""


class GroundedAnswerService:
    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client or create_llm_client()

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        min_score: float = 0.05,
        use_llm: bool = True,
    ) -> GroundedAnswerResponse:
        normalized_question = question.strip()
        hits = self.retrieval_service.search(normalized_question, top_k=top_k)
        relevant_hits = [hit for hit in hits if hit.score >= min_score]
        citations = build_citations(relevant_hits)

        if not citations:
            return GroundedAnswerResponse(
                question=normalized_question,
                answer="I do not have enough retrieved evidence to answer this question reliably.",
                citations=[],
                confidence="low",
                limitations=["No retrieved chunk met the minimum evidence score."],
                used_model="none",
                provider="none",
                retrieval_hits=hits,
            )

        if use_llm:
            try:
                llm_response = self.llm_client.chat(
                    _build_grounded_messages(normalized_question, citations),
                    temperature=0.2,
                )
                answer_text = llm_response.content
                used_model = llm_response.model
                provider = llm_response.provider
            except LLMServiceError as exc:
                answer_text = _build_template_answer(normalized_question, citations)
                citations_note = "LLM generation failed; returned an extractive fallback."
                return GroundedAnswerResponse(
                    question=normalized_question,
                    answer=answer_text,
                    citations=citations,
                    confidence=_estimate_confidence(citations),
                    limitations=[citations_note, str(exc)],
                    used_model="template-fallback",
                    provider="template",
                    retrieval_hits=hits,
                )
        else:
            answer_text = _build_template_answer(normalized_question, citations)
            used_model = TemplateLLMClient.model
            provider = TemplateLLMClient.provider

        return GroundedAnswerResponse(
            question=normalized_question,
            answer=answer_text,
            citations=citations,
            confidence=_estimate_confidence(citations),
            limitations=[
                "This answer is for informational support and is not a medical diagnosis."
            ],
            used_model=used_model,
            provider=provider,
            retrieval_hits=hits,
        )


def build_citations(hits: list[RetrievalHit]) -> list[Citation]:
    citations: list[Citation] = []
    for index, hit in enumerate(hits, start=1):
        metadata = hit.chunk.metadata
        citations.append(
            Citation(
                citation_id=f"C{index}",
                chunk_id=hit.chunk.id,
                document_id=hit.chunk.document_id,
                title=str(metadata.get("title")) if metadata.get("title") else None,
                source=str(metadata.get("source")) if metadata.get("source") else None,
                excerpt=hit.chunk.content[:700],
                score=hit.score,
            )
        )
    return citations


def _build_grounded_messages(question: str, citations: list[Citation]) -> list[ChatMessage]:
    evidence = "\n\n".join(
        _format_citation_for_prompt(citation)
        for citation in citations
    )
    user_prompt = f"""Question:
{question}

Evidence:
{evidence}

Write a concise answer grounded in the evidence. Include citation markers like [C1]."""
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]


def _format_citation_for_prompt(citation: Citation) -> str:
    source = citation.source or citation.document_id
    return f"[{citation.citation_id}] Source: {source}\n{citation.excerpt}"


def _build_template_answer(question: str, citations: list[Citation]) -> str:
    evidence_lines = "\n".join(
        f"- [{citation.citation_id}] {citation.excerpt}" for citation in citations[:3]
    )
    return (
        f"Based on the retrieved evidence, here are the most relevant excerpts for: {question}\n"
        f"{evidence_lines}\n"
        "Please use this as informational support rather than a medical diagnosis."
    )


def _estimate_confidence(citations: list[Citation]) -> str:
    if not citations:
        return "low"
    best_score = max(citation.score for citation in citations)
    if best_score >= 0.55 and len(citations) >= 2:
        return "high"
    if best_score >= 0.2:
        return "medium"
    return "low"
