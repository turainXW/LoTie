#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.repo_context import build_repo_map, compact_repo_context, save_repo_map, search_repo_context  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact repository map for CodeAgent-RL.")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default=None)
    parser.add_argument("--task", default="")
    parser.add_argument("--max-files", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=5000)
    parser.add_argument("--max-snippets", type=int, default=6)
    parser.add_argument("--json", action="store_true", help="Print structured ranked context instead of prompt text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve() if args.output else repo / ".codeagent" / "repo_map.json"
    repo_map = build_repo_map(repo)
    save_repo_map(repo_map, output)
    print(f"repo_map={output}")
    print(f"files={len(repo_map.files)}")
    print(f"python_files={repo_map.stats.get('python_file_count', 0)}")
    print(f"test_files={repo_map.stats.get('test_file_count', 0)}")
    if args.task:
        if args.json:
            import json

            result = search_repo_context(
                repo_map,
                args.task,
                max_files=args.max_files,
                max_snippets=args.max_snippets,
                max_chars=args.max_chars,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(compact_repo_context(repo_map, args.task, max_files=args.max_files, max_chars=args.max_chars))


if __name__ == "__main__":
    main()
