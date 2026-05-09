from __future__ import annotations

from app.schemas.chunk import DocumentChunk


class InvalidChunkConfigError(ValueError):
    pass


def split_text_into_chunks(
    *,
    document_id: str,
    text: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise InvalidChunkConfigError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise InvalidChunkConfigError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise InvalidChunkConfigError("chunk_overlap must be smaller than chunk_size")

    normalized_text = text.strip()
    if not normalized_text:
        return []

    spans = _build_chunk_spans(normalized_text, chunk_size, chunk_overlap)
    chunks: list[DocumentChunk] = []
    for index, (start, end) in enumerate(spans):
        content = normalized_text[start:end].strip()
        if not content:
            continue
        chunks.append(
            DocumentChunk(
                id=f"{document_id}_{index:04d}",
                document_id=document_id,
                content=content,
                chunk_index=index,
                start_char=start,
                end_char=end,
                metadata={
                    "source": source,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                },
            )
        )
    return chunks


def _build_chunk_spans(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        end = _find_natural_break(text, start, hard_end) if hard_end < text_length else hard_end
        if end <= start:
            end = hard_end
        spans.append((start, end))
        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)

    return spans


def _find_natural_break(text: str, start: int, hard_end: int) -> int:
    search_window = text[start:hard_end]
    for separator in ("\n\n", "\n", "?", ". ", " "):
        position = search_window.rfind(separator)
        if position >= int(len(search_window) * 0.5):
            return start + position + len(separator)
    return hard_end
