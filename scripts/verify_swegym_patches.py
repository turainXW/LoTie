#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.swegym_verifier import verify_swegym_patch_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify SWE-Gym patch trajectories with explicit Docker readiness checks.")
    parser.add_argument("--records", required=True, help="Patch trajectory JSONL, e.g. openhands_patch_rollout.jsonl.")
    parser.add_argument("--tasks", help="SWE-Gym tasks parquet/jsonl. Required for official Docker verification.")
    parser.add_argument("--output", required=True, help="Verification result JSONL.")
    parser.add_argument("--mode", choices=["dry-run", "docker", "local-venv"], default="dry-run")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-repo", help="Repository snapshot used by local-venv mode. Useful for one-repo batches.")
    parser.add_argument("--local-venv", help="Existing venv directory or Python executable. Reused across tasks.")
    parser.add_argument("--venv-cache-dir", help="Directory for automatically created local verifier venvs.")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install requirements.txt and the repository into the selected local venv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = verify_swegym_patch_records(
        args.records,
        args.output,
        tasks_path=args.tasks,
        mode=args.mode,
        timeout_sec=args.timeout_sec,
        limit=args.limit,
        local_repo_path=args.local_repo,
        local_venv_path=args.local_venv,
        venv_cache_dir=args.venv_cache_dir,
        install_deps=args.install_deps,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    if args.mode in {"docker", "local-venv"} and summary.blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
