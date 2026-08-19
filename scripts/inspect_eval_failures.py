#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_agent_baseline.eval_analysis import canonical_outcome, load_results, normalize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect failed Code Agent evaluation samples.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--category", help="Canonical failure category, such as tests_failed or no_effective_edit.")
    parser.add_argument("--dataset", help="MBPP+, HumanEval+, or SWE-smith.")
    parser.add_argument("--round", type=int, dest="round_index")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", help="Optional .jsonl output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_results(args.results)
    failures = []
    for row in rows:
        category = canonical_outcome(row)
        if category == "resolved":
            continue
        dataset = normalize_dataset(row.get("dataset"))
        round_index = int(row.get("sample_index") or 1)
        if args.category and category != args.category:
            continue
        if args.dataset and dataset.lower() != normalize_dataset(args.dataset).lower():
            continue
        if args.round_index and round_index != args.round_index:
            continue
        failures.append(
            {
                "instance_id": row.get("instance_id"),
                "dataset": dataset,
                "round": round_index,
                "failure_category": category,
                "agent_status": row.get("agent_status") or row.get("status"),
                "verifier_status": row.get("verifier_status"),
                "patch_present": bool(row.get("patch_present")),
                "patch_lines": int(row.get("patch_lines") or 0),
                "tool_steps": int(row.get("tool_steps") or 0),
                "trajectory_path": row.get("trajectory_path") or row.get("record_source"),
                "verifier_path": row.get("verifier_path") or row.get("verifier_source"),
            }
        )
    failures.sort(key=lambda row: (row["failure_category"], row["dataset"], str(row["instance_id"])))
    counts = Counter(row["failure_category"] for row in failures)
    selected = failures[: max(0, args.limit)]
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    print(json.dumps({"matched": len(failures), "shown": len(selected), "counts": dict(sorted(counts.items()))}, ensure_ascii=False))
    for row in selected:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

