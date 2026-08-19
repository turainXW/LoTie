#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export code agent trajectories into training-friendly JSONL.")
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        help="Trajectory JSONL file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/code_agent_training",
        help="Directory for exported training files.",
    )
    parser.add_argument(
        "--format",
        choices=["all", "sft", "reward", "episodes"],
        default="all",
        help="Training view to export.",
    )
    parser.add_argument("--stdout-tail", type=int, default=1200)
    parser.add_argument("--stderr-tail", type=int, default=1600)
    return parser.parse_args()


def load_trajectories(paths: list[str]) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    trajectories.append(json.loads(line))
        else:
            data = json.loads(text)
            if _is_codeagent_trajectory(data):
                trajectories.append(data)
            else:
                raise ValueError(f"Unsupported trajectory format for training export: {path}")
    return trajectories


def export_sft(trajectories: list[dict[str, Any]], output_path: Path, stdout_tail: int, stderr_tail: int) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for trajectory in trajectories:
            sanitized = sanitize_trajectory(trajectory, stdout_tail=stdout_tail, stderr_tail=stderr_tail)
            for index, step in enumerate(sanitized.get("steps", [])):
                history = sanitized.get("steps", [])[:index]
                record = {
                    "id": f"{sanitized['task_id']}::step_{step['step']:04d}",
                    "format": "sft_next_action_v1",
                    "task_id": sanitized["task_id"],
                    "agent": sanitized.get("agent", ""),
                    "model": sanitized.get("model", ""),
                    "final_status": sanitized.get("final_status", ""),
                    "reward": reward_for_status(sanitized.get("final_status", ""), len(sanitized.get("steps", []))),
                    "sample_weight": 1.0 if sanitized.get("final_status") == "pass" else 0.2,
                    "messages": [
                        *build_messages(sanitized, history),
                        {
                            "role": "assistant",
                            "content": json.dumps(step["action"], ensure_ascii=False),
                        },
                    ],
                    "target_action": step["action"],
                    "context": {
                        "initial_context": sanitized.get("initial_context", {}),
                        "history": history,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def export_rewards(trajectories: list[dict[str, Any]], output_path: Path, stdout_tail: int, stderr_tail: int) -> int:
    with output_path.open("w", encoding="utf-8") as f:
        for trajectory in trajectories:
            sanitized = sanitize_trajectory(trajectory, stdout_tail=stdout_tail, stderr_tail=stderr_tail)
            reward = reward_for_status(sanitized.get("final_status", ""), len(sanitized.get("steps", [])))
            record = {
                "id": sanitized["task_id"],
                "format": "reward_episode_v1",
                "task_id": sanitized["task_id"],
                "agent": sanitized.get("agent", ""),
                "model": sanitized.get("model", ""),
                "final_status": sanitized.get("final_status", ""),
                "reward": reward,
                "success": sanitized.get("final_status") == "pass",
                "steps": len(sanitized.get("steps", [])),
                "validation_command": sanitized.get("validation_command", ""),
                "final_stdout": sanitized.get("final_stdout", ""),
                "final_stderr": sanitized.get("final_stderr", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(trajectories)


def export_episodes(trajectories: list[dict[str, Any]], output_path: Path, stdout_tail: int, stderr_tail: int) -> int:
    with output_path.open("w", encoding="utf-8") as f:
        for trajectory in trajectories:
            sanitized = sanitize_trajectory(trajectory, stdout_tail=stdout_tail, stderr_tail=stderr_tail)
            sanitized["format"] = "episode_v1"
            sanitized["reward"] = reward_for_status(sanitized.get("final_status", ""), len(sanitized.get("steps", [])))
            f.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
    return len(trajectories)


def build_messages(trajectory: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, str]]:
    user_payload = {
        "task": {
            "instruction": trajectory.get("instruction", ""),
            "validation_command": trajectory.get("validation_command", ""),
        },
        "context": {
            "initial_context": trajectory.get("initial_context", {}),
            "history": history,
        },
        "request": "Return the next code-agent action as JSON with action_type, content, and rationale.",
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a code agent. Use repository context, command observations, and verifier feedback "
                "to choose one next action. Prefer running verification before and after edits. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
        },
    ]


def sanitize_trajectory(trajectory: dict[str, Any], *, stdout_tail: int, stderr_tail: int) -> dict[str, Any]:
    repo_path = trajectory.get("repo_path", "")
    sanitized = dict(trajectory)
    sanitized["repo_path"] = "<repo>"
    sanitized["validation_command"] = _sanitize_text(trajectory.get("validation_command", ""), repo_path)
    sanitized["final_stdout"] = _tail(_sanitize_text(trajectory.get("final_stdout", ""), repo_path), stdout_tail)
    sanitized["final_stderr"] = _tail(_sanitize_text(trajectory.get("final_stderr", ""), repo_path), stderr_tail)
    sanitized["initial_context"] = _sanitize_obj(trajectory.get("initial_context", {}), repo_path)
    steps = []
    for step in trajectory.get("steps", []):
        clean_step = dict(step)
        clean_step["stdout"] = _tail(_sanitize_text(step.get("stdout", ""), repo_path), stdout_tail)
        clean_step["stderr"] = _tail(_sanitize_text(step.get("stderr", ""), repo_path), stderr_tail)
        clean_step["action"] = _sanitize_obj(step.get("action", {}), repo_path)
        steps.append(clean_step)
    sanitized["steps"] = steps
    return sanitized


def reward_for_status(status: str, steps: int) -> float:
    if status == "pass":
        return round(max(0.1, 1.0 - 0.02 * max(0, steps - 1)), 4)
    if status == "timeout":
        return -0.5
    return 0.0


def _sanitize_obj(value: Any, repo_path: str) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_obj(item, repo_path) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_obj(item, repo_path) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value, repo_path)
    return value


def _sanitize_text(text: str, repo_path: str) -> str:
    if repo_path:
        text = text.replace(repo_path, "<repo>")
    text = re.sub(r"/Users/[^\\s:'\"]+", "<local_path>", text)
    return text


def _tail(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[-limit:]


def _is_codeagent_trajectory(data: dict[str, Any]) -> bool:
    return "task_id" in data and "steps" in data and "final_status" in data


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = load_trajectories(args.trajectory)

    counts: dict[str, int] = {}
    if args.format in {"all", "sft"}:
        counts["sft"] = export_sft(
            trajectories,
            output_dir / "sft_next_action.jsonl",
            stdout_tail=args.stdout_tail,
            stderr_tail=args.stderr_tail,
        )
    if args.format in {"all", "reward"}:
        counts["reward"] = export_rewards(
            trajectories,
            output_dir / "reward_episodes.jsonl",
            stdout_tail=args.stdout_tail,
            stderr_tail=args.stderr_tail,
        )
    if args.format in {"all", "episodes"}:
        counts["episodes"] = export_episodes(
            trajectories,
            output_dir / "episodes.jsonl",
            stdout_tail=args.stdout_tail,
            stderr_tail=args.stderr_tail,
        )

    manifest = {
        "source_trajectories": [str(Path(path).resolve()) for path in args.trajectory],
        "trajectory_count": len(trajectories),
        "outputs": {
            name: str((output_dir / filename).resolve())
            for name, filename in {
                "sft": "sft_next_action.jsonl",
                "reward": "reward_episodes.jsonl",
                "episodes": "episodes.jsonl",
            }.items()
            if name in counts
        },
        "counts": counts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output_dir={output_dir}")
    print(f"trajectory_count={len(trajectories)}")
    for name, count in counts.items():
        print(f"{name}_records={count}")


if __name__ == "__main__":
    main()
