import pytest

from app.rag.rerankers import (
    BgeReranker,
    KeywordOverlapReranker,
    RerankerProviderError,
    create_reranker,
)


def test_create_reranker_returns_none_when_disabled() -> None:
    assert create_reranker(provider="none", model_name="unused") is None


def test_keyword_overlap_reranker_scores_matching_document_higher() -> None:
    reranker = KeywordOverlapReranker()

    scores = reranker.score(
        "blood pressure hypertension",
        [
            "docker container deployment",
            "hypertension means high blood pressure",
        ],
    )

    assert scores[1] > scores[0]


def test_create_reranker_rejects_unknown_provider() -> None:
    with pytest.raises(RerankerProviderError):
        create_reranker(provider="unknown", model_name="unused")


def test_bge_reranker_reports_missing_optional_dependency(monkeypatch) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "FlagEmbedding":
            raise ImportError("missing FlagEmbedding")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RerankerProviderError, match="FlagEmbedding"):
        BgeReranker(model_name="BAAI/bge-reranker-v2-m3")
