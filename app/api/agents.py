from fastapi import APIRouter, HTTPException, status

from app.agents.langgraph_workflow import LangGraphAgenticRagWorkflow
from app.agents.rag_workflow import AgenticRagWorkflow
from app.schemas.agent import AgenticRagRequest, AgenticRagResponse, AgentRunsResponse
from app.services.app_state import retrieval_service
from app.services.trace_service import create_agent_trace_reader, create_agent_trace_writer

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("/runs", response_model=AgentRunsResponse)
def list_agent_runs(limit: int = 20) -> AgentRunsResponse:
    if limit <= 0 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit must be between 1 and 100",
        )

    runs = create_agent_trace_reader().tail(limit=limit)
    return AgentRunsResponse(runs=runs, count=len(runs))


@router.post("/rag", response_model=AgenticRagResponse)
def agentic_rag(request: AgenticRagRequest) -> AgenticRagResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question cannot be blank",
        )

    workflow_engine = request.workflow_engine.strip().lower()
    if workflow_engine not in {"linear", "langgraph"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="workflow_engine must be one of: linear, langgraph",
        )

    workflow = (
        LangGraphAgenticRagWorkflow(retrieval_service=retrieval_service)
        if workflow_engine == "langgraph"
        else AgenticRagWorkflow(retrieval_service=retrieval_service)
    )
    response = workflow.run(
        question,
        top_k=request.top_k,
        min_score=request.min_score,
        min_citations=request.min_citations,
        use_llm=request.use_llm,
        include_trace=request.include_trace,
    )
    create_agent_trace_writer().write(response)
    return response
