from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from typing import Any, Protocol

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class EmbeddingProviderError(RuntimeError):
    pass


class EmbeddingModel(Protocol):
    dimension: int

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class HashEmbeddingModel:
    """Deterministic local embedding for tests and offline pipeline validation."""

    provider_name = "hash"

    def __init__(self, dimension: int = 128) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        return _normalize(vector)

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class BgeM3EmbeddingModel:
    """BGE-M3 embedding provider backed by FlagEmbedding, loaded lazily."""

    provider_name = "bge-m3"

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str = "cpu",
        use_fp16: bool = False,
        dimension: int = 1024,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model = self._load_model(model_name=model_name, device=device, use_fp16=use_fp16)

    def embed_text(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else [0.0] * self.dimension

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []

        encoded = self._model.encode(text_list)
        dense_vectors = encoded["dense_vecs"] if isinstance(encoded, dict) else encoded
        return [_normalize(_to_float_list(vector)) for vector in dense_vectors]

    @staticmethod
    def _load_model(*, model_name: str, device: str, use_fp16: bool) -> Any:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise EmbeddingProviderError(
                "BGE-M3 requires the optional FlagEmbedding dependency. "
                "Install it with: pip install FlagEmbedding"
            ) from exc

        try:
            return BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
        except TypeError:
            return BGEM3FlagModel(model_name, use_fp16=use_fp16)


def create_embedding_model(
    *,
    provider: str,
    model_name: str,
    dimension: int,
    device: str = "cpu",
    use_fp16: bool = False,
) -> EmbeddingModel:
    normalized_provider = provider.strip().lower()
    if normalized_provider in {"hash", "fake", "local"}:
        return HashEmbeddingModel(dimension=dimension)
    if normalized_provider in {"bge-m3", "bge_m3", "bge"}:
        return BgeM3EmbeddingModel(
            model_name=model_name or "BAAI/bge-m3",
            device=device,
            use_fp16=use_fp16,
            dimension=dimension or 1024,
        )
    raise EmbeddingProviderError(f"Unsupported embedding provider: {provider}")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _to_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        return [float(value) for value in vector.tolist()]
    return [float(value) for value in vector]
