#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.swebench_local import (  # noqa: E402
    create_local_swebench_suite,
    read_tasks_jsonl,
    verify_task,
    write_results_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and run a local SWE-bench-like harness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create local SWE-bench-like repos and tasks.jsonl.")
    init_parser.add_argument("--output-dir", default="data/swebench_local", help="Output directory.")

    verify_parser = subparsers.add_parser("verify", help="Run task validation commands.")
    verify_parser.add_argument("--tasks", default="data/swebench_local/tasks.jsonl", help="tasks.jsonl path.")
    verify_parser.add_argument("--output", default="data/swebench_local/results.jsonl", help="results.jsonl path.")
    verify_parser.add_argument("--apply-gold", action="store_true", help="Apply gold patches before verification.")
    verify_parser.add_argument("--timeout-sec", type=int, default=30, help="Validation timeout per task.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        output_dir = Path(args.output_dir)
        tasks = create_local_swebench_suite(output_dir)
        print(f"tasks={len(tasks)}")
        print(f"tasks_jsonl={(output_dir / 'tasks.jsonl').resolve()}")
        print(f"repos={(output_dir / 'repos').resolve()}")
        return

    if args.command == "verify":
        tasks = read_tasks_jsonl(args.tasks)
        results = [verify_task(task, apply_gold=args.apply_gold, timeout_sec=args.timeout_sec) for task in tasks]
        write_results_jsonl(args.output, results)
        passed = sum(1 for result in results if result.passed)
        print(f"results={Path(args.output).resolve()}")
        print(f"passed={passed}/{len(results)}")
        for result in results:
            print(json.dumps(result.to_dict(), ensure_ascii=False))
        if passed != len(results):
            raise SystemExit(1)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
