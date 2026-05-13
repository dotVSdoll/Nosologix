from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return request_json(request)


def get_json(base_url: str, path: str) -> dict[str, Any]:
    request = Request(f"{base_url.rstrip('/')}{path}", method="GET")
    return request_json(request)


def request_json(request: Request) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot connect to service: {exc.reason}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local end-to-end demo against a running Med RAG Agent service."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="What is hypertension?")
    parser.add_argument("--workflow-engine", default="langgraph", choices=["linear", "langgraph"])
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.02)
    parser.add_argument(
        "--sample-path",
        default=str(Path("data/samples/health_sample.md").resolve()),
        help="Local document path visible to the API server.",
    )
    args = parser.parse_args()

    try:
        health = get_json(args.base_url, "/health")
        print(f"[1/4] health: {health['status']} ({health['app_name']})")

        ingestion = post_json(
            args.base_url,
            "/documents/ingest-local",
            {
                "path": args.sample_path,
                "chunk_size": 300,
                "chunk_overlap": 50,
            },
        )
        print(f"[2/4] ingested: {ingestion['chunk_count']} chunks from {ingestion['title']}")

        run = post_json(
            args.base_url,
            "/agents/rag",
            {
                "question": args.question,
                "top_k": args.top_k,
                "min_score": args.min_score,
                "min_citations": 1,
                "use_llm": args.use_llm,
                "include_trace": True,
                "workflow_engine": args.workflow_engine,
            },
        )
        answer = run["answer"]
        print(
            "[3/4] answer: "
            f"status={run['workflow_status']} "
            f"engine={run['workflow_engine']} "
            f"citations={len(answer['citations'])} "
            f"risk={answer['risk_level']}"
        )
        print(f"      preview: {answer['answer'][:220].replace(chr(10), ' ')}")

        runs = get_json(args.base_url, "/agents/runs?limit=1")
        latest = runs["runs"][0] if runs["runs"] else None
        if latest is None:
            print("[4/4] no run record found")
            return
        detail = get_json(args.base_url, f"/agents/runs/{latest['run_id']}")
        print(
            "[4/4] trace: "
            f"run_id={detail['run_id']} "
            f"steps={len(detail['steps'])} "
            f"latency_ms={round(detail['total_latency_ms'])}"
        )
    except RuntimeError as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        print("Make sure the API is running: .\\.venv\\Scripts\\python -m uvicorn app.main:app")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
