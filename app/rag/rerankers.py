from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any, Protocol

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class RerankerProviderError(RuntimeError):
    pass


class Reranker(Protocol):
    provider_name: str

    def score(self, query: str, documents: Iterable[str]) -> list[float]:
        raise NotImplementedError


class NoopReranker:
    provider_name = "none"

    def score(self, query: str, documents: Iterable[str]) -> list[float]:
        return [0.0 for _ in documents]


class KeywordOverlapReranker:
    """Deterministic lexical reranker for tests and offline development."""

    provider_name = "keyword"

    def score(self, query: str, documents: Iterable[str]) -> list[float]:
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return [0.0 for _ in documents]

        scores: list[float] = []
        for document in documents:
            document_tokens = set(_tokenize(document))
            overlap = len(query_tokens & document_tokens)
            scores.append(overlap / len(query_tokens))
        return scores


class BgeReranker:
    """BGE reranker backed by FlagEmbedding, loaded only when configured."""

    provider_name = "bge-reranker"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        use_fp16: bool = False,
    ) -> None:
        self.model_name = model_name
        self._model = self._load_model(model_name=model_name, use_fp16=use_fp16)

    def score(self, query: str, documents: Iterable[str]) -> list[float]:
        pairs = [[query, document] for document in documents]
        if not pairs:
            return []

        raw_scores = self._model.compute_score(pairs)
        if isinstance(raw_scores, int | float):
            raw_scores = [raw_scores]
        return [_sigmoid(float(score)) for score in raw_scores]

    @staticmethod
    def _load_model(*, model_name: str, use_fp16: bool) -> Any:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RerankerProviderError(
                "BGE reranker requires the optional FlagEmbedding dependency. "
                "Install it with: pip install FlagEmbedding"
            ) from exc

        return FlagReranker(model_name, use_fp16=use_fp16)


def create_reranker(
    *,
    provider: str,
    model_name: str,
    use_fp16: bool = False,
) -> Reranker | None:
    normalized_provider = provider.strip().lower()
    if normalized_provider in {"", "none", "off", "disabled"}:
        return None
    if normalized_provider in {"keyword", "lexical", "overlap"}:
        return KeywordOverlapReranker()
    if normalized_provider in {"bge", "bge-reranker", "bge_reranker"}:
        return BgeReranker(
            model_name=model_name or "BAAI/bge-reranker-v2-m3",
            use_fp16=use_fp16,
        )
    raise RerankerProviderError(f"Unsupported reranker provider: {provider}")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
