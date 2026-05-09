import pytest

from app.rag.embeddings import (
    BgeM3EmbeddingModel,
    EmbeddingProviderError,
    HashEmbeddingModel,
    create_embedding_model,
)


def test_create_hash_embedding_model_from_provider_alias() -> None:
    model = create_embedding_model(provider="fake", model_name="hash-local", dimension=32)

    assert isinstance(model, HashEmbeddingModel)
    assert model.dimension == 32


def test_create_embedding_model_rejects_unknown_provider() -> None:
    with pytest.raises(EmbeddingProviderError):
        create_embedding_model(provider="unknown", model_name="demo", dimension=32)


def test_bge_m3_provider_reports_missing_optional_dependency(monkeypatch) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "FlagEmbedding":
            raise ImportError("missing FlagEmbedding")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(EmbeddingProviderError, match="FlagEmbedding"):
        BgeM3EmbeddingModel(model_name="BAAI/bge-m3")
