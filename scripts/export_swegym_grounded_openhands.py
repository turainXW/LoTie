#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.swegym_openhands_export import export_grounded_to_openhands  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export grounded SWE-Gym traces to OpenHands-like JSONL formats.")
    parser.add_argument("--traces", required=True, help="Path to grounded_traces.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--include-planner-failures-in-sft", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_grounded_to_openhands(
        args.traces,
        args.output_dir,
        run_id=args.run_id,
        include_planner_failures_in_sft=args.include_planner_failures_in_sft,
    )
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
