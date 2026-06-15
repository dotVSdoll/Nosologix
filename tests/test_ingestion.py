import pytest

from app.rag.loaders import UnsupportedDocumentTypeError, load_local_text_document
from app.rag.splitters import InvalidChunkConfigError, split_text_into_chunks
from app.schemas.document import DocumentSourceType, DocumentStatus
from app.services.ingestion_service import ingest_local_document


def test_load_markdown_document_normalizes_metadata(tmp_path) -> None:
    path = tmp_path / "hypertension.md"
    path.write_text("# Hypertension\n\nBlood pressure notes.\n", encoding="utf-8")

    document, content = load_local_text_document(path)

    assert document.title == "hypertension"
    assert document.source_type == DocumentSourceType.MARKDOWN
    assert document.character_count == len(content)
    assert len(document.content_hash) == 64
    assert content.startswith("# Hypertension")


def test_load_rejects_unsupported_file_type(tmp_path) -> None:
    path = tmp_path / "image.png"
    path.write_text("not a document", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError):
        load_local_text_document(path)


def test_split_text_into_chunks_preserves_traceable_metadata() -> None:
    text = "Alpha paragraph.\n\n" + "Beta content " * 40 + "\n\nFinal paragraph."

    chunks = split_text_into_chunks(
        document_id="doc123",
        text=text,
        source="sample.md",
        chunk_size=120,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert chunks[0].id == "doc123_0000"
    assert chunks[0].document_id == "doc123"
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["source"] == "sample.md"
    assert all(chunk.content for chunk in chunks)
    assert all(chunk.start_char < chunk.end_char for chunk in chunks)


def test_split_rejects_invalid_overlap() -> None:
    with pytest.raises(InvalidChunkConfigError):
        split_text_into_chunks(
            document_id="doc123",
            text="hello",
            source="sample.txt",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_ingest_local_document_returns_chunked_document(tmp_path) -> None:
    path = tmp_path / "guide.txt"
    path.write_text("Medical guide section.\n" * 40, encoding="utf-8")

    result = ingest_local_document(path, chunk_size=100, chunk_overlap=10)

    assert result.document.status == DocumentStatus.CHUNKED
    assert result.document.source_type == DocumentSourceType.TEXT
    assert len(result.chunks) >= 2
    assert {chunk.document_id for chunk in result.chunks} == {result.document.id}
