#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.swesmith_local import (  # noqa: E402
    build_runnable_tasks,
    materialize_selected_tasks,
    run_gold_sanity,
    setup_repositories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize and locally validate selected SWE-smith tasks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--selection", default="data/dataset_splits_v1/selection.json")
    materialize.add_argument("--source", action="append", required=True, help="Parquet or HF dataset-server JSON.")
    materialize.add_argument("--output", default="data/swesmith_local/tasks.jsonl")
    materialize.add_argument("--cache-root", default=".codeagent/swesmith")

    setup = subparsers.add_parser("setup")
    setup.add_argument("--tasks", default="data/swesmith_local/tasks.jsonl")
    setup.add_argument("--python", required=True, help="Python 3.10 executable used to create repository venvs.")
    setup.add_argument("--timeout-sec", type=int, default=1800)
    setup.add_argument("--output", default="data/swesmith_local/setup_results.jsonl")

    sanity = subparsers.add_parser("sanity")
    sanity.add_argument("--tasks", default="data/swesmith_local/tasks.jsonl")
    sanity.add_argument("--output", default="data/swesmith_local/gold_sanity.jsonl")
    sanity.add_argument("--split", choices=["train", "validation", "evaluation"])
    sanity.add_argument("--limit", type=int)
    sanity.add_argument("--p2p-limit", type=int, default=10)
    sanity.add_argument("--timeout-sec", type=int, default=600)

    runnable = subparsers.add_parser("runnable")
    runnable.add_argument("--tasks", default="data/swesmith_local/tasks.jsonl")
    runnable.add_argument("--sanity-results", default="data/swesmith_local/gold_sanity.jsonl")
    runnable.add_argument("--output", default="data/swesmith_local/tasks_runnable.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "materialize":
        manifest = materialize_selected_tasks(
            args.selection,
            args.source,
            args.output,
            cache_root=args.cache_root,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        if manifest["missing"]:
            raise SystemExit(2)
        return

    if args.command == "setup":
        results = setup_repositories(args.tasks, python_executable=args.python, timeout_sec=args.timeout_sec)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in results), encoding="utf-8")
        summary = {
            "repos": len(results),
            "ready": sum(item.status == "ready" for item in results),
            "blocked": sum(item.status != "ready" for item in results),
            "repo_bytes": sum(item.repo_bytes for item in results),
            "venv_bytes": sum(item.venv_bytes for item in results),
            "output": str(output.resolve()),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["blocked"]:
            raise SystemExit(2)
        return

    if args.command == "runnable":
        manifest = build_runnable_tasks(args.tasks, args.sanity_results, args.output)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    results = run_gold_sanity(
        args.tasks,
        args.output,
        split=args.split,
        limit=args.limit,
        p2p_limit=args.p2p_limit,
        timeout_sec=args.timeout_sec,
    )
    summary = {
        "tasks": len(results),
        "passed": sum(item.gold_sanity_passed for item in results),
        "failed_or_blocked": sum(not item.gold_sanity_passed for item in results),
        "status_counts": {},
        "output": str(Path(args.output).resolve()),
    }
    for item in results:
        summary["status_counts"][item.status] = summary["status_counts"].get(item.status, 0) + 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed_or_blocked"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
