from fastapi import APIRouter, HTTPException, status

from app.schemas.answer import GroundedAnswerRequest, GroundedAnswerResponse
from app.services.answer_service import GroundedAnswerService
from app.services.app_state import retrieval_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/grounded", response_model=GroundedAnswerResponse)
def grounded_answer(request: GroundedAnswerRequest) -> GroundedAnswerResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question cannot be blank",
        )

    service = GroundedAnswerService(retrieval_service=retrieval_service)
    return service.answer(
        question,
        top_k=request.top_k,
        min_score=request.min_score,
        min_citations=request.min_citations,
        use_llm=request.use_llm,
    )
