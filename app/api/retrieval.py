from fastapi import APIRouter, HTTPException, status

from app.schemas.api import RetrievalSearchRequest, RetrievalSearchResponse
from app.services.app_state import retrieval_service

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=RetrievalSearchResponse)
def search(request: RetrievalSearchRequest) -> RetrievalSearchResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query cannot be blank",
        )

    hits = retrieval_service.search(query, top_k=request.top_k)
    return RetrievalSearchResponse(query=query, hits=hits)
