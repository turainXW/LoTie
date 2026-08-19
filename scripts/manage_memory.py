#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.memory import (  # noqa: E402
    MemoryRecord,
    add_memory,
    compact_memories,
    ensure_memory_store,
    learn_from_summary,
    load_memories,
    summarize_trajectory,
    write_session_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage CodeAgent-RL memory.")
    parser.add_argument("--memory-dir", default=".codeagent/memory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init")

    add = subparsers.add_parser("add")
    add.add_argument("--type", choices=["fact", "lesson", "preference", "failure_pattern"], required=True)
    add.add_argument("--content", required=True)
    add.add_argument("--tag", action="append", default=[])
    add.add_argument("--source", default="")

    subparsers.add_parser("list")

    compact = subparsers.add_parser("compact")
    compact.add_argument("--task", required=True)
    compact.add_argument("--max-records", type=int, default=8)
    compact.add_argument("--max-chars", type=int, default=3000)

    summarize = subparsers.add_parser("summarize-session")
    summarize.add_argument("--trajectory", required=True)
    summarize.add_argument("--learn", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    memory_dir = Path(args.memory_dir).resolve()
    if args.command == "init":
        ensure_memory_store(memory_dir)
        print(f"memory_dir={memory_dir}")
    elif args.command == "add":
        path = add_memory(
            memory_dir,
            MemoryRecord(type=args.type, content=args.content, tags=args.tag, source=args.source),
        )
        print(f"memory_file={path}")
    elif args.command == "list":
        for record in load_memories(memory_dir):
            tags = ",".join(record.tags)
            print(f"{record.type}\t{tags}\t{record.content}")
    elif args.command == "compact":
        print(compact_memories(memory_dir, args.task, max_records=args.max_records, max_chars=args.max_chars))
    elif args.command == "summarize-session":
        summary = summarize_trajectory(args.trajectory)
        summary_path, index_path = write_session_summary(memory_dir, summary)
        print(f"summary={summary_path}")
        print(f"session_index={index_path}")
        print(f"status={summary.status}")
        print(f"tools={','.join(summary.tools_used)}")
        if args.learn:
            for path in learn_from_summary(memory_dir, summary):
                print(f"learned={path}")


if __name__ == "__main__":
    main()

