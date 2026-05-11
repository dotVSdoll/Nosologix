from fastapi import APIRouter, HTTPException, status

from app.agents.rag_workflow import AgenticRagWorkflow
from app.schemas.agent import AgenticRagRequest, AgenticRagResponse
from app.services.app_state import retrieval_service

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/rag", response_model=AgenticRagResponse)
def agentic_rag(request: AgenticRagRequest) -> AgenticRagResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question cannot be blank",
        )

    workflow = AgenticRagWorkflow(retrieval_service=retrieval_service)
    return workflow.run(
        question,
        top_k=request.top_k,
        min_score=request.min_score,
        min_citations=request.min_citations,
        use_llm=request.use_llm,
        include_trace=request.include_trace,
    )
