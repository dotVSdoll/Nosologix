from pathlib import Path

from app.rag.loaders import load_local_text_document
from app.rag.splitters import split_text_into_chunks
from app.schemas.document import DocumentStatus
from app.schemas.ingestion import IngestedDocument


def ingest_local_document(
    path: str | Path,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> IngestedDocument:
    document, content = load_local_text_document(path)
    chunks = split_text_into_chunks(
        document_id=document.id,
        text=content,
        source=document.source_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    document.status = DocumentStatus.CHUNKED
    return IngestedDocument(document=document, chunks=chunks)
