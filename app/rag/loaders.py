from __future__ import annotations

import hashlib
from pathlib import Path

from app.schemas.document import DocumentRecord, DocumentSourceType

_SUPPORTED_SUFFIXES = {
    ".txt": DocumentSourceType.TEXT,
    ".md": DocumentSourceType.MARKDOWN,
    ".markdown": DocumentSourceType.MARKDOWN,
}


class UnsupportedDocumentTypeError(ValueError):
    pass


def load_local_text_document(path: str | Path) -> tuple[DocumentRecord, str]:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"Document not found: {resolved_path}")

    source_type = _source_type_for_path(resolved_path)
    content = resolved_path.read_text(encoding="utf-8-sig")
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
    content_hash = hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()
    document = DocumentRecord.from_path(
        path=resolved_path,
        source_type=source_type,
        content_hash=content_hash,
        character_count=len(normalized_content),
    )
    return document, normalized_content


def _source_type_for_path(path: Path) -> DocumentSourceType:
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_SUFFIXES))
        raise UnsupportedDocumentTypeError(
            f"Unsupported document type '{suffix}'. Use one of: {supported}"
        )
    return _SUPPORTED_SUFFIXES[suffix]
