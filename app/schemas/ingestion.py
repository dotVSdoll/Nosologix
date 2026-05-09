from pydantic import BaseModel

from app.schemas.chunk import DocumentChunk
from app.schemas.document import DocumentRecord


class IngestedDocument(BaseModel):
    document: DocumentRecord
    chunks: list[DocumentChunk]
