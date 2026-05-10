from app.schemas.retrieval import RetrievalHit


def build_evidence_warnings(
    hits: list[RetrievalHit],
    *,
    min_score: float,
    min_citations: int,
) -> list[str]:
    warnings: list[str] = []
    if not hits:
        return ["No evidence was retrieved."]

    relevant_hits = [hit for hit in hits if hit.score >= min_score]
    if len(relevant_hits) < min_citations:
        warnings.append(
            f"Only {len(relevant_hits)} retrieved chunk(s) met the minimum score threshold."
        )

    best_score = max(hit.score for hit in hits)
    if best_score < min_score:
        warnings.append(
            f"Best retrieval score {best_score:.3f} is below threshold {min_score:.3f}."
        )

    return warnings


def evidence_status_from_warnings(warnings: list[str]) -> str:
    return "insufficient" if warnings else "sufficient"
