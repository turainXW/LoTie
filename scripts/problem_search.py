#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.problem_search import (  # noqa: E402
    build_problem_index,
    problem_search,
    save_problem_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query a local problem/task search index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--root", action="append", default=["data"], help="Root directory or JSONL file to index.")
    build.add_argument("--output", default="data/problem_search/index.jsonl")

    search = subparsers.add_parser("search")
    search.add_argument("--query", required=True)
    search.add_argument("--index", default="data/problem_search/index.jsonl")
    search.add_argument("--root", action="append", default=None)
    search.add_argument("--top-k", type=int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        records = build_problem_index([Path(root) for root in args.root])
        save_problem_index(records, args.output)
        print(f"index={Path(args.output).resolve()}")
        print(f"records={len(records)}")
        return
    if args.command == "search":
        result = problem_search(args.query, index_path=args.index, roots=args.root, top_k=args.top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
