#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.swe_trace import collect_swe_like_trace  # noqa: E402
from code_agent_baseline.swebench_local import create_local_swebench_suite, read_tasks_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SWE-bench-like CodeAct trajectories locally.")
    parser.add_argument("--suite-dir", default="data/swebench_like_rollout", help="Suite workspace directory.")
    parser.add_argument("--output-dir", default="data/swebench_like_rollout/trajectories", help="Output directory.")
    parser.add_argument("--timeout-sec", type=int, default=60, help="Per-command timeout.")
    parser.add_argument("--reuse-suite", action="store_true", help="Reuse existing tasks.jsonl instead of recreating repos.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the local suite N times, recreating repos each round.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    raw_dir = output_dir / "raw_traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = suite_dir / "tasks.jsonl"
    output_jsonl = output_dir / "output.jsonl"
    sft_jsonl = output_dir / "messages_sft.jsonl"
    verifier_jsonl = output_dir / "verifier_records.jsonl"
    output_jsonl.write_text("", encoding="utf-8")
    sft_jsonl.write_text("", encoding="utf-8")
    verifier_jsonl.write_text("", encoding="utf-8")

    traces = []
    task_count = 0
    for repeat_index in range(1, args.repeat + 1):
        if args.reuse_suite and tasks_path.exists() and args.repeat == 1:
            tasks = read_tasks_jsonl(tasks_path)
        else:
            tasks = create_local_swebench_suite(suite_dir)
        for task in tasks:
            task_count += 1
            trace = collect_swe_like_trace(task, timeout_sec=args.timeout_sec)
            traces.append(trace)
            episode_id = f"r{repeat_index:03d}_{task.instance_id}"
            raw_path = raw_dir / f"{episode_id}.json"
            raw_path.write_text(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            append_jsonl(output_jsonl, to_output_record(trace, raw_path, episode_id=episode_id))
            append_jsonl(sft_jsonl, to_sft_record(trace, episode_id=episode_id))
            append_jsonl(verifier_jsonl, to_verifier_record(trace, episode_id=episode_id))

    resolved = sum(1 for trace in traces if trace.resolved)
    manifest = {
        "format": "swe_like_rollout_manifest_v1",
        "suite_dir": str(suite_dir),
        "tasks": str(tasks_path),
        "output_jsonl": str(output_jsonl),
        "messages_sft_jsonl": str(sft_jsonl),
        "verifier_records_jsonl": str(verifier_jsonl),
        "raw_traces": str(raw_dir),
        "total": len(traces),
        "resolved": resolved,
        "repeat": args.repeat,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output_dir={output_dir}")
    print(f"tasks={len(traces)}")
    print(f"resolved={resolved}/{len(traces)}")
    print(f"output_jsonl={output_jsonl}")
    print(f"messages_sft_jsonl={sft_jsonl}")
    if resolved != len(traces):
        raise SystemExit(1)


def to_output_record(trace, raw_path: Path, *, episode_id: str | None = None) -> dict[str, object]:
    messages = sanitize_obj(trace.messages, trace.repo_path)
    report = sanitize_obj(trace.report, trace.repo_path)
    return {
        "format": "swe_like_openhands_output_v1",
        "episode_id": episode_id or trace.instance_id,
        "instance_id": trace.instance_id,
        "repo": trace.repo,
        "repo_path": "/workspace",
        "base_commit": trace.base_commit,
        "resolved": trace.resolved,
        "problem_statement": trace.problem_statement,
        "hints_text": trace.hints_text,
        "patch": trace.patch,
        "validation_command": trace.validation_command,
        "messages": messages,
        "test_result": {"report": report, "resolved": trace.resolved},
        "raw_trace_path": f"raw_traces/{raw_path.name}",
    }


def to_sft_record(trace, *, episode_id: str | None = None) -> dict[str, object]:
    return {
        "format": "openhands_sft_messages_v1",
        "episode_id": episode_id or trace.instance_id,
        "instance_id": trace.instance_id,
        "resolved": trace.resolved,
        "messages": sanitize_obj(trace.messages, trace.repo_path),
    }


def to_verifier_record(trace, *, episode_id: str | None = None) -> dict[str, object]:
    report = sanitize_obj(trace.report, trace.repo_path)
    messages = sanitize_obj(trace.messages, trace.repo_path)
    return {
        "format": "swe_like_verifier_v1",
        "episode_id": episode_id or trace.instance_id,
        "instance_id": trace.instance_id,
        "resolved": trace.resolved,
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "problem_statement": trace.problem_statement,
                        "trajectory": messages,
                        "patch": trace.patch,
                        "report": report,
                    },
                    ensure_ascii=False,
                ),
            },
            {"role": "assistant", "content": "True" if trace.resolved else "False"},
        ],
    }


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sanitize_obj(value, repo_path: str):
    if isinstance(value, dict):
        return {key: sanitize_obj(item, repo_path) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_obj(item, repo_path) for item in value]
    if isinstance(value, str):
        return value.replace(repo_path, "/workspace")
    return value


if __name__ == "__main__":
    main()
