from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from app.rag.embeddings import create_embedding_model
from app.rag.vector_store import InMemoryVectorStore
from app.schemas.answer import GroundedAnswerResponse
from app.services.answer_service import GroundedAnswerService
from app.services.llm_service import TemplateLLMClient
from app.services.retrieval_service import RetrievalService


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_terms: list[str]
    expected_evidence_status: str
    expected_risk_level: str
    top_k: int = 3
    min_score: float = 0.05
    min_citations: int = 1


@dataclass(frozen=True)
class EvalDataset:
    documents: list[str]
    cases: list[EvalCase]


def load_eval_dataset(path: str | Path) -> EvalDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalDataset(
        documents=[str(document) for document in payload["documents"]],
        cases=[
            EvalCase(
                id=str(case["id"]),
                question=str(case["question"]),
                expected_terms=[str(term).lower() for term in case.get("expected_terms", [])],
                expected_evidence_status=str(case["expected_evidence_status"]),
                expected_risk_level=str(case["expected_risk_level"]),
                top_k=int(case.get("top_k", 3)),
                min_score=float(case.get("min_score", 0.05)),
                min_citations=int(case.get("min_citations", 1)),
            )
            for case in payload["cases"]
        ],
    )


def run_eval(
    dataset_path: str | Path,
    *,
    embedding_provider: str = "hash",
    embedding_model: str = "hash-local",
    embedding_dimension: int = 128,
    embedding_device: str = "cpu",
    embedding_use_fp16: bool = False,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path).resolve()
    dataset = load_eval_dataset(dataset_file)
    retrieval_service = RetrievalService(
        embedding_model=create_embedding_model(
            provider=embedding_provider,
            model_name=embedding_model,
            dimension=embedding_dimension,
            device=embedding_device,
            use_fp16=embedding_use_fp16,
        ),
        vector_store=InMemoryVectorStore(),
        reranker=None,
    )

    indexed_chunks = 0
    for document in dataset.documents:
        document_path = _resolve_dataset_path(dataset_file, document)
        indexed = retrieval_service.ingest_and_index_document(
            document_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        indexed_chunks += len(indexed.chunks)

    answer_service = GroundedAnswerService(
        retrieval_service=retrieval_service,
        llm_client=TemplateLLMClient(),
    )
    case_results = [
        _evaluate_case(answer_service=answer_service, case=case)
        for case in dataset.cases
    ]

    return {
        "dataset": str(dataset_file),
        "embedding_provider": embedding_provider,
        "indexed_documents": len(dataset.documents),
        "indexed_chunks": indexed_chunks,
        "case_count": len(case_results),
        "metrics": _summarize_metrics(case_results),
        "cases": case_results,
    }


def _evaluate_case(
    *,
    answer_service: GroundedAnswerService,
    case: EvalCase,
) -> dict[str, Any]:
    response = answer_service.answer(
        case.question,
        top_k=case.top_k,
        min_score=case.min_score,
        min_citations=case.min_citations,
        use_llm=False,
    )
    evidence_text = _join_retrieved_text(response).lower()
    expected_terms_found = [
        term for term in case.expected_terms if term.lower() in evidence_text
    ]
    retrieval_hit = not case.expected_terms or bool(expected_terms_found)
    evidence_status_match = response.evidence_status == case.expected_evidence_status
    safety_match = response.risk_level.value == case.expected_risk_level
    passed = retrieval_hit and evidence_status_match and safety_match

    return {
        "id": case.id,
        "question": case.question,
        "passed": passed,
        "retrieval_hit": retrieval_hit,
        "expected_terms_found": expected_terms_found,
        "expected_evidence_status": case.expected_evidence_status,
        "actual_evidence_status": response.evidence_status,
        "expected_risk_level": case.expected_risk_level,
        "actual_risk_level": response.risk_level.value,
        "citation_count": len(response.citations),
        "top_score": _top_score(response),
        "provider": response.provider,
        "used_model": response.used_model,
    }


def _summarize_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not case_results:
        return {
            "pass_rate": 0.0,
            "retrieval_hit_rate": 0.0,
            "evidence_status_accuracy": 0.0,
            "safety_accuracy": 0.0,
            "average_top_score": 0.0,
        }

    return {
        "pass_rate": _rate(case_results, "passed"),
        "retrieval_hit_rate": _rate(case_results, "retrieval_hit"),
        "evidence_status_accuracy": _match_rate(
            case_results,
            expected_key="expected_evidence_status",
            actual_key="actual_evidence_status",
        ),
        "safety_accuracy": _match_rate(
            case_results,
            expected_key="expected_risk_level",
            actual_key="actual_risk_level",
        ),
        "average_top_score": round(mean(result["top_score"] for result in case_results), 4),
    }


def _resolve_dataset_path(dataset_file: Path, document: str) -> Path:
    document_path = Path(document)
    if document_path.is_absolute():
        return document_path
    return (dataset_file.parent / document_path).resolve()


def _join_retrieved_text(response: GroundedAnswerResponse) -> str:
    citation_text = "\n".join(citation.excerpt for citation in response.citations)
    hit_text = "\n".join(hit.chunk.content for hit in response.retrieval_hits)
    return f"{citation_text}\n{hit_text}"


def _top_score(response: GroundedAnswerResponse) -> float:
    if not response.retrieval_hits:
        return 0.0
    return round(response.retrieval_hits[0].score, 4)


def _rate(results: list[dict[str, Any]], key: str) -> float:
    return round(sum(1 for result in results if result[key]) / len(results), 4)


def _match_rate(results: list[dict[str, Any]], *, expected_key: str, actual_key: str) -> float:
    return round(
        sum(1 for result in results if result[expected_key] == result[actual_key])
        / len(results),
        4,
    )
