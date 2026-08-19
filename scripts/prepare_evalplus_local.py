#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from code_agent_baseline.evalplus_local import (  # noqa: E402
    build_runnable_tasks,
    materialize_function_workspaces,
    materialize_selected_tasks,
    run_gold_sanity,
    setup_shared_venv,
    verify_function_patch_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize and locally validate selected EvalPlus tasks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--selection", default="data/dataset_splits_v1/selection.json")
    materialize.add_argument("--mbpp-source", action="append", required=True)
    materialize.add_argument("--humaneval-source", action="append", required=True)
    materialize.add_argument("--output", default="data/evalplus_local/tasks.jsonl")

    setup = subparsers.add_parser("setup")
    setup.add_argument("--python", required=True)
    setup.add_argument("--venv", default=".codeagent/evalplus/venv")
    setup.add_argument("--timeout-sec", type=int, default=900)

    sanity = subparsers.add_parser("sanity")
    sanity.add_argument("--tasks", default="data/evalplus_local/tasks.jsonl")
    sanity.add_argument("--output", default="data/evalplus_local/gold_sanity.jsonl")
    sanity.add_argument("--python", default=".codeagent/evalplus/venv/bin/python")
    sanity.add_argument("--split", choices=["train", "validation", "evaluation"])
    sanity.add_argument("--dataset", choices=["mbppplus", "humanevalplus"])
    sanity.add_argument("--limit", type=int)
    sanity.add_argument("--timeout-sec", type=int, default=30)

    runnable = subparsers.add_parser("runnable")
    runnable.add_argument("--tasks", default="data/evalplus_local/tasks.jsonl")
    runnable.add_argument("--sanity-results", default="data/evalplus_local/gold_sanity.jsonl")
    runnable.add_argument("--output", default="data/evalplus_local/tasks_runnable.jsonl")

    workspace = subparsers.add_parser("workspace")
    workspace.add_argument("--tasks", default="data/evalplus_local/tasks_runnable.jsonl")
    workspace.add_argument("--output", default="data/evalplus_local/tasks_agent.jsonl")
    workspace.add_argument("--repo-root", default=".codeagent/evalplus/repos")

    verify = subparsers.add_parser("verify-records")
    verify.add_argument("--records", required=True)
    verify.add_argument("--tasks", default="data/evalplus_local/tasks_agent.jsonl")
    verify.add_argument("--output", required=True)
    verify.add_argument("--python", default=".codeagent/evalplus/venv/bin/python")
    verify.add_argument("--timeout-sec", type=int, default=30)
    verify.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "materialize":
        result = materialize_selected_tasks(
            args.selection,
            args.mbpp_source,
            args.humaneval_source,
            args.output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["missing"]:
            raise SystemExit(2)
        return
    if args.command == "setup":
        print(json.dumps(setup_shared_venv(args.python, args.venv, args.timeout_sec), ensure_ascii=False, indent=2))
        return
    if args.command == "runnable":
        print(json.dumps(build_runnable_tasks(args.tasks, args.sanity_results, args.output), ensure_ascii=False, indent=2))
        return
    if args.command == "workspace":
        print(json.dumps(materialize_function_workspaces(args.tasks, args.output, args.repo_root), ensure_ascii=False, indent=2))
        return
    if args.command == "verify-records":
        summary = verify_function_patch_records(
            args.records,
            args.tasks,
            args.output,
            args.python,
            timeout_sec=args.timeout_sec,
            limit=args.limit,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["blocked"]:
            raise SystemExit(2)
        return
    results = run_gold_sanity(
        args.tasks,
        args.output,
        args.python,
        split=args.split,
        dataset=args.dataset,
        limit=args.limit,
        timeout_sec=args.timeout_sec,
    )
    summary: dict[str, object] = {
        "tasks": len(results),
        "passed": sum(item.gold_sanity_passed for item in results),
        "failed_or_blocked": sum(not item.gold_sanity_passed for item in results),
        "status_counts": {},
        "output": str(Path(args.output).resolve()),
    }
    counts = summary["status_counts"]
    assert isinstance(counts, dict)
    for item in results:
        counts[item.status] = counts.get(item.status, 0) + 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed_or_blocked"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
