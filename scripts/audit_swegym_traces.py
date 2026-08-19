#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.swegym_trace_audit import audit_grounded_traces  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit grounded SWE-Gym trajectory files.")
    parser.add_argument("--traces", required=True, help="Path to grounded_traces.jsonl.")
    parser.add_argument("--tasks", default=None, help="Optional SWE-Gym parquet/JSONL task file for gold patch file-hit.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    parser.add_argument("--max-misses", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_grounded_traces(args.traces, tasks_path=args.tasks)
    data = report.to_dict()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = dict(data)
    printable["misses"] = printable["misses"][: args.max_misses]
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
