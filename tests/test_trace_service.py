import json

from app.schemas.agent import AgenticRagResponse, AgentStep
from app.schemas.answer import GroundedAnswerResponse
from app.services.trace_service import AgentTraceWriter


def _response() -> AgenticRagResponse:
    return AgenticRagResponse(
        question="What is hypertension?",
        workflow_status="completed",
        workflow_engine="langgraph",
        total_latency_ms=12.3,
        answer=GroundedAnswerResponse(
            question="What is hypertension?",
            answer="Hypertension means high blood pressure. [C1]",
            citations=[],
            confidence="medium",
            evidence_status="sufficient",
            used_model="template-local",
            provider="template",
        ),
        steps=[
            AgentStep(
                name="query_planner",
                status="completed",
                summary="planned",
                input_summary="question",
                output_summary="query",
                latency_ms=1.0,
            )
        ],
    )


def test_agent_trace_writer_appends_jsonl_record(tmp_path) -> None:
    path = tmp_path / "agent_runs.jsonl"
    writer = AgentTraceWriter(enabled=True, path=path)

    run_id = writer.write(_response())

    assert run_id
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["run_id"] == run_id
    assert record["workflow_engine"] == "langgraph"
    assert record["answer"]["provider"] == "template"
    assert record["steps"][0]["name"] == "query_planner"


def test_agent_trace_writer_skips_when_disabled(tmp_path) -> None:
    path = tmp_path / "agent_runs.jsonl"
    writer = AgentTraceWriter(enabled=False, path=path)

    run_id = writer.write(_response())

    assert run_id is None
    assert not path.exists()
