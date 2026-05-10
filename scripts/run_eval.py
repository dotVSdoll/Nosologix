from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.runner import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local RAG evaluation fixtures.")
    parser.add_argument(
        "--dataset",
        default="eval/fixtures/health_eval.json",
        help="Path to an eval dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        default="eval/reports/latest.json",
        help="Path to write the JSON eval report.",
    )
    parser.add_argument("--embedding-provider", default="hash")
    parser.add_argument("--embedding-model", default="hash-local")
    parser.add_argument("--embedding-dimension", type=int, default=128)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-use-fp16", action="store_true")
    args = parser.parse_args()

    report = run_eval(
        args.dataset,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        embedding_device=args.embedding_device,
        embedding_use_fp16=args.embedding_use_fp16,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
