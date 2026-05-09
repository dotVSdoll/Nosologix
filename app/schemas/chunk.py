from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    id: str
    document_id: str
    content: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    metadata: dict[str, str | int] = Field(default_factory=dict)
