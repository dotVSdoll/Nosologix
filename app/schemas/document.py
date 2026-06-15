from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class DocumentSourceType(StrEnum):
    TEXT = "txt"
    MARKDOWN = "md"


class DocumentStatus(StrEnum):
    LOADED = "loaded"
    CHUNKED = "chunked"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    id: str
    title: str
    source_type: DocumentSourceType
    source_path: str
    content_hash: str
    character_count: int = Field(ge=0)
    status: DocumentStatus = DocumentStatus.LOADED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_path(
        cls,
        *,
        path: Path,
        source_type: DocumentSourceType,
        content_hash: str,
        character_count: int,
    ) -> "DocumentRecord":
        return cls(
            id=content_hash[:16],
            title=path.stem,
            source_type=source_type,
            source_path=str(path),
            content_hash=content_hash,
            character_count=character_count,
        )
