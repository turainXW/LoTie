#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline import CodeAgent, RuleBasedSmokeModel  # noqa: E402
from code_agent_baseline.task_factory import (  # noqa: E402
    create_toy_addition_task,
    write_tasks_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/code_agent_smoke",
        help="Directory for toy repo, task file, and trajectories.",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to trajectories.jsonl instead of rewriting it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task = create_toy_addition_task(output_dir)
    write_tasks_jsonl(output_dir / "tasks.jsonl", [task])

    agent = CodeAgent(model=RuleBasedSmokeModel(), max_steps=args.max_steps)
    trajectory = agent.run(task)
    trajectory_path = output_dir / "trajectories.jsonl"
    if not args.append:
        trajectory_path.write_text("", encoding="utf-8")
    CodeAgent.append_jsonl(trajectory_path, trajectory)

    print(f"task_id={task.task_id}")
    print(f"repo_path={task.repo_path}")
    print(f"trajectory={trajectory_path}")
    print(f"final_status={trajectory.final_status}")


if __name__ == "__main__":
    main()
