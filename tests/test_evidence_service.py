from app.schemas.retrieval import RetrievalHit
from app.services.evidence_service import build_evidence_warnings, evidence_status_from_warnings
from tests.test_retrieval import _chunk


def test_evidence_warnings_when_no_hits() -> None:
    warnings = build_evidence_warnings([], min_score=0.1, min_citations=1)

    assert warnings == ["No evidence was retrieved."]
    assert evidence_status_from_warnings(warnings) == "insufficient"


def test_evidence_warnings_when_too_few_relevant_hits() -> None:
    hit = RetrievalHit(chunk=_chunk("chunk0", "content"), score=0.5, rank=1)

    warnings = build_evidence_warnings([hit], min_score=0.1, min_citations=2)

    assert warnings
    assert evidence_status_from_warnings(warnings) == "insufficient"


def test_evidence_status_sufficient_without_warnings() -> None:
    hits = [
        RetrievalHit(chunk=_chunk("chunk0", "content one"), score=0.5, rank=1),
        RetrievalHit(chunk=_chunk("chunk1", "content two"), score=0.4, rank=2),
    ]

    warnings = build_evidence_warnings(hits, min_score=0.1, min_citations=2)

    assert warnings == []
    assert evidence_status_from_warnings(warnings) == "sufficient"
