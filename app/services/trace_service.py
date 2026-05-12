from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import settings
from app.schemas.agent import AgenticRagResponse


class AgentTraceWriter:
    def __init__(
        self,
        *,
        enabled: bool = True,
        path: str | Path = "./data/traces/agent_runs.jsonl",
    ) -> None:
        self.enabled = enabled
        self.path = Path(path)

    def write(self, response: AgenticRagResponse) -> str | None:
        if not self.enabled:
            return None

        run_id = str(uuid.uuid4())
        record = _build_trace_record(run_id=run_id, response=response)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return run_id


def create_agent_trace_writer() -> AgentTraceWriter:
    return AgentTraceWriter(
        enabled=settings.agent_trace_enabled,
        path=settings.agent_trace_path,
    )


def _build_trace_record(*, run_id: str, response: AgenticRagResponse) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "question": response.question,
        "workflow_status": response.workflow_status,
        "workflow_engine": response.workflow_engine,
        "total_latency_ms": response.total_latency_ms,
        "answer": {
            "confidence": response.answer.confidence,
            "provider": response.answer.provider,
            "used_model": response.answer.used_model,
            "evidence_status": response.answer.evidence_status,
            "risk_level": response.answer.risk_level.value,
            "citation_count": len(response.answer.citations),
            "retrieval_hit_count": len(response.answer.retrieval_hits),
            "llm_error_type": response.answer.llm_error_type,
            "llm_error_status_code": response.answer.llm_error_status_code,
            "llm_error_code": response.answer.llm_error_code,
        },
        "steps": [step.model_dump(mode="json") for step in response.steps],
    }
