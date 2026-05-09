from fastapi import APIRouter, HTTPException, status

from app.rag.loaders import UnsupportedDocumentTypeError
from app.rag.splitters import InvalidChunkConfigError
from app.schemas.api import IngestLocalDocumentRequest, IngestLocalDocumentResponse
from app.services.app_state import retrieval_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest-local", response_model=IngestLocalDocumentResponse)
def ingest_local_document(request: IngestLocalDocumentRequest) -> IngestLocalDocumentResponse:
    try:
        result = retrieval_service.ingest_and_index_document(
            request.path,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (UnsupportedDocumentTypeError, InvalidChunkConfigError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return IngestLocalDocumentResponse(
        document_id=result.document.id,
        title=result.document.title,
        source_path=result.document.source_path,
        chunk_count=len(result.chunks),
        status=result.document.status.value,
    )
