#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.harness_version import harness_metadata  # noqa: E402
from code_agent_baseline.pass_at_k import summarize_pass_at_k  # noqa: E402


MODEL_FAILURE_STATUSES_WITHOUT_PATCH = {
    "compile_failed",
    "context_overflow",
    "max_steps",
    "no_edit",
    "parse_error",
}
INFRASTRUCTURE_STATUSES = {
    "infrastructure_blocked",
    "model_api_failure",
    "runner_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable, versioned multi-rollout Code Agent experiment.")
    parser.add_argument("--output-dir", default="outputs/deepseek_v4_flash_eval90_pass5_v1")
    parser.add_argument("--selection", default="data/eval90/selection.json")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "evaluation"],
        default="evaluation",
        help="Dataset split selected from --selection. The legacy eval90 selection only supports evaluation.",
    )
    parser.add_argument("--evalplus-tasks", default="data/evalplus_local/tasks_agent.jsonl")
    parser.add_argument("--swesmith-tasks", default="data/swesmith_local/tasks_runnable.jsonl")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--model-url", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--protocol-retries", type=int, default=2)
    parser.add_argument("--protocol-retry-temperature", type=float)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--thinking-mode", choices=["enabled", "disabled", "auto"], default="enabled")
    parser.add_argument("--reasoning-effort", choices=["low", "high", "xhigh", "max"], default="max")
    parser.add_argument("--function-max-steps", type=int, default=12)
    parser.add_argument("--repo-max-steps", type=int, default=40)
    parser.add_argument("--edit-checkpoint-step", type=int, default=10)
    parser.add_argument("--readonly-reminder-interval", type=int, default=3)
    parser.add_argument("--context-max-tokens", type=int, default=64_000)
    parser.add_argument("--model-timeout-sec", type=int, default=180)
    parser.add_argument("--function-verify-timeout-sec", type=int, default=30)
    parser.add_argument("--repo-verify-timeout-sec", type=int, default=600)
    parser.add_argument("--max-infra-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent task rollouts.")
    parser.add_argument("--sample-start", type=int, default=1)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument(
        "--dataset",
        choices=["function", "mbppplus", "humanevalplus", "swesmith_py"],
        help="Run both function datasets or one individual dataset.",
    )
    parser.add_argument("--max-tasks", type=int, help="Development-only task limit; recorded in the manifest.")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.sample_start < 1 or args.sample_start > args.samples:
        raise ValueError("--sample-start must be between 1 and --samples")
    sample_end = args.sample_end or args.samples
    if sample_end < args.sample_start or sample_end > args.samples:
        raise ValueError("--sample-end must be between --sample-start and --samples")
    if not args.plan_only and not os.environ.get(args.api_key_env):
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")

    selection_path = resolve_path(args.selection)
    evalplus_path = resolve_path(args.evalplus_tasks)
    swesmith_path = resolve_path(args.swesmith_tasks)
    output_dir = resolve_path(args.output_dir)
    tasks = load_selected_tasks(selection_path, evalplus_path, swesmith_path, split=args.split)
    if args.dataset:
        tasks = filter_tasks_by_dataset(tasks, args.dataset)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise ValueError("No tasks selected")

    config = experiment_config(args, tasks, selection_path, evalplus_path, swesmith_path)
    initialize_or_validate_experiment(output_dir, config)
    print(
        json.dumps(
            {
                "experiment_id": output_dir.name,
                "split": args.split,
                "tasks": len(tasks),
                "samples_per_task": args.samples,
                "target_rollouts": len(tasks) * args.samples,
                "sample_range_this_process": [args.sample_start, sample_end],
                "workers": args.workers,
                "output_dir": str(output_dir),
                "config_sha256": config["config_sha256"],
                "plan_only": args.plan_only,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.plan_only:
        return

    jobs = pending_jobs(args, output_dir, tasks, sample_end)
    if args.workers == 1:
        for job in jobs:
            run_sample(args, output_dir=output_dir, **job)
            refresh_summary(output_dir, args.samples)
    else:
        run_jobs_concurrently(args, output_dir, jobs)

    summary = refresh_summary(output_dir, args.samples)
    print("PASS_AT_K " + json.dumps(summary["aggregate"], ensure_ascii=False), flush=True)


def filter_tasks_by_dataset(tasks: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    selected = {"mbppplus", "humanevalplus"} if dataset == "function" else {dataset}
    return [task for task in tasks if task["experiment_dataset"] in selected]


def pending_jobs(
    args: argparse.Namespace,
    output_dir: Path,
    tasks: list[dict[str, Any]],
    sample_end: int,
) -> list[dict[str, Any]]:
    jobs = []
    for sample_index in range(args.sample_start, sample_end + 1):
        for task_index, task in enumerate(tasks, start=1):
            existing = load_sample_result(output_dir, task, sample_index)
            if existing and existing.get("sample_valid") is True:
                print_progress("SKIP", sample_index, args.samples, task_index, len(tasks), task, existing)
                continue
            jobs.append(
                {
                    "task": task,
                    "sample_index": sample_index,
                    "task_index": task_index,
                    "task_count": len(tasks),
                }
            )
    return jobs


def run_jobs_concurrently(
    args: argparse.Namespace,
    output_dir: Path,
    jobs: list[dict[str, Any]],
) -> None:
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="lottie-eval") as executor:
        futures: dict[Future[dict[str, Any]], dict[str, Any]] = {
            executor.submit(run_sample, args, output_dir=output_dir, **job): job for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:
                task = job["task"]
                print(
                    json.dumps(
                        {
                            "event": "WORKER_ERROR",
                            "sample": f"{job['sample_index']}/{args.samples}",
                            "task": f"{job['task_index']}/{job['task_count']}",
                            "instance_id": task["instance_id"],
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                raise
            refresh_summary(output_dir, args.samples)


def run_sample(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    task: dict[str, Any],
    sample_index: int,
    task_index: int,
    task_count: int,
) -> dict[str, Any]:
    sample_dir = sample_directory(output_dir, task, sample_index)
    sample_dir.mkdir(parents=True, exist_ok=True)
    previous_attempts = sorted(sample_dir.glob("attempt_*"))
    for attempt_index in range(len(previous_attempts) + 1, args.max_infra_attempts + 1):
        attempt_dir = sample_dir / f"attempt_{attempt_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        print_progress(
            "RUN",
            sample_index,
            args.samples,
            task_index,
            task_count,
            task,
            {"attempt_index": attempt_index},
        )
        started = time.monotonic()
        rollout_dir = attempt_dir / "rollout"
        run_log = attempt_dir / "runner.log"
        runner_returncode = tee_command(
            rollout_command(args, task, rollout_dir, attempt_dir / "workspace"),
            run_log,
            cwd=ROOT,
        )
        record = read_single_jsonl(rollout_dir / "openhands_patch_rollout.jsonl")
        if record is None:
            attempt = infrastructure_attempt(
                task=task,
                sample_index=sample_index,
                attempt_index=attempt_index,
                split=args.split,
                status="runner_missing_record",
                returncode=runner_returncode,
                started=started,
                attempt_dir=attempt_dir,
                harness=harness_metadata(
                    checkpoint_step=args.edit_checkpoint_step,
                    reminder_interval=args.readonly_reminder_interval,
                ),
            )
            write_json(attempt_dir / "attempt_result.json", attempt)
            continue

        verifier_path = attempt_dir / "verifier_result.jsonl"
        verify_log = attempt_dir / "verifier.log"
        verifier_returncode = tee_command(
            verifier_command(args, task, rollout_dir / "openhands_patch_rollout.jsonl", verifier_path),
            verify_log,
            cwd=ROOT,
        )
        verifier = read_single_jsonl(verifier_path)
        sample_result = classify_sample_result(
            task=task,
            record=record,
            verifier=verifier,
            sample_index=sample_index,
            attempt_index=attempt_index,
            split=args.split,
            runner_returncode=runner_returncode,
            verifier_returncode=verifier_returncode,
            elapsed_sec=round(time.monotonic() - started, 3),
            attempt_dir=attempt_dir,
            harness=harness_metadata(
                checkpoint_step=args.edit_checkpoint_step,
                reminder_interval=args.readonly_reminder_interval,
            ),
        )
        write_json(attempt_dir / "attempt_result.json", sample_result)
        if sample_result["sample_valid"]:
            write_json(sample_dir / "sample_result.json", sample_result)
            print_progress("DONE", sample_index, args.samples, task_index, task_count, task, sample_result)
            return sample_result
        print_progress("RETRY_INFRA", sample_index, args.samples, task_index, task_count, task, sample_result)

    failed = {
        "format": "lottie_pass_at_k_sample_v1",
        "instance_id": task["instance_id"],
        "dataset": task["experiment_dataset"],
        "sample_index": sample_index,
        "sample_valid": False,
        "benchmark_resolved": None,
        "status": "infrastructure_attempts_exhausted",
        "attempts": max(len(previous_attempts), args.max_infra_attempts),
        "harness": harness_metadata(
            checkpoint_step=args.edit_checkpoint_step,
            reminder_interval=args.readonly_reminder_interval,
        ),
        **split_policy(args.split),
    }
    write_json(sample_dir / "sample_result.json", failed)
    return failed


def rollout_command(
    args: argparse.Namespace,
    task: dict[str, Any],
    rollout_dir: Path,
    workspace_dir: Path,
) -> list[str]:
    is_repo = task["experiment_dataset"] == "swesmith_py"
    tasks_path = resolve_path(args.swesmith_tasks if is_repo else args.evalplus_tasks)
    max_steps = args.repo_max_steps if is_repo else args.function_max_steps
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "run_swegym_patch_rollout.py"),
        "--tasks",
        str(tasks_path),
        "--output-dir",
        str(rollout_dir),
        "--work-dir",
        str(workspace_dir),
        "--model",
        args.model,
        "--model-url",
        args.model_url,
        "--model-network",
        "direct",
        "--api-key-env",
        args.api_key_env,
        "--agent-mode",
        "bench",
        "--prompt-workspace-label",
        "repository",
        "--max-steps",
        str(max_steps),
        "--context-max-tokens",
        str(args.context_max_tokens),
        "--context-overflow-policy",
        "error",
        "--max-tokens",
        str(args.max_tokens),
        "--edit-checkpoint-step",
        str(args.edit_checkpoint_step),
        "--readonly-reminder-interval",
        str(args.readonly_reminder_interval),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--protocol-retries",
        str(args.protocol_retries),
        "--thinking-mode",
        args.thinking_mode,
        "--reasoning-effort",
        args.reasoning_effort,
        "--timeout-sec",
        str(args.model_timeout_sec),
        "--num",
        "1",
        "--instance-id",
        task["instance_id"],
    ]
    if args.protocol_retry_temperature is not None:
        command.extend(["--protocol-retry-temperature", str(args.protocol_retry_temperature)])
    if is_repo:
        command.extend(["--repo-cache", str(ROOT / ".codeagent" / "swesmith" / "repos")])
    else:
        command.extend(["--repo-cache", str(ROOT / ".codeagent" / "evalplus" / "repos")])
    return command


def verifier_command(
    args: argparse.Namespace,
    task: dict[str, Any],
    records_path: Path,
    output_path: Path,
) -> list[str]:
    python = str(ROOT / ".venv" / "bin" / "python")
    if task["experiment_dataset"] != "swesmith_py":
        return [
            python,
            str(ROOT / "scripts" / "prepare_evalplus_local.py"),
            "verify-records",
            "--records",
            str(records_path),
            "--tasks",
            str(resolve_path(args.evalplus_tasks)),
            "--output",
            str(output_path),
            "--python",
            str(ROOT / ".codeagent" / "evalplus" / "venv" / "bin" / "python"),
            "--timeout-sec",
            str(args.function_verify_timeout_sec),
        ]
    return [
        python,
        str(ROOT / "scripts" / "verify_swegym_patches.py"),
        "--records",
        str(records_path),
        "--tasks",
        str(resolve_path(args.swesmith_tasks)),
        "--output",
        str(output_path),
        "--mode",
        "local-venv",
        "--timeout-sec",
        str(args.repo_verify_timeout_sec),
        "--venv-cache-dir",
        str(ROOT / ".codeagent" / "swesmith" / "venvs"),
    ]


def classify_sample_result(
    *,
    task: dict[str, Any],
    record: dict[str, Any],
    verifier: dict[str, Any] | None,
    sample_index: int,
    attempt_index: int,
    split: str,
    runner_returncode: int,
    verifier_returncode: int,
    elapsed_sec: float,
    attempt_dir: Path,
    harness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    agent_status = str(record.get("status") or "unknown")
    verifier_status = str((verifier or {}).get("verifier_status") or "verifier_missing_result")
    resolved = (verifier or {}).get("benchmark_resolved")
    sample_valid = isinstance(resolved, bool)
    if (
        resolved is None
        and verifier_status == "blocked_missing_patch"
        and agent_status in MODEL_FAILURE_STATUSES_WITHOUT_PATCH
    ):
        resolved = False
        sample_valid = True
        verifier_status = f"model_failure_{agent_status}"
    if agent_status in INFRASTRUCTURE_STATUSES:
        resolved = None
        sample_valid = False
    steps = record.get("metadata", {}).get("tool_steps", [])
    usage = sum_usage(record.get("metadata", {}).get("model_response_events", []))
    return {
        "format": "lottie_pass_at_k_sample_v1",
        "instance_id": task["instance_id"],
        "dataset": task["experiment_dataset"],
        "repo": task.get("repo"),
        "sample_index": sample_index,
        "attempt_index": attempt_index,
        "sample_valid": sample_valid,
        "benchmark_resolved": resolved,
        "agent_status": agent_status,
        "verifier_status": verifier_status,
        "patch_present": bool(str(record.get("patch") or "").strip()),
        "patch_lines": len(str(record.get("patch") or "").splitlines()),
        "edited_files": record.get("edit_apply", {}).get("edited_files", []),
        "tool_steps": len(steps),
        "usage": usage,
        "elapsed_sec": elapsed_sec,
        "runner_returncode": runner_returncode,
        "verifier_returncode": verifier_returncode,
        "attempt_dir": str(attempt_dir),
        "trajectory_path": str(attempt_dir / "rollout" / "openhands_patch_rollout.jsonl"),
        "verifier_path": str(attempt_dir / "verifier_result.jsonl"),
        "harness": record.get("metadata", {}).get("harness")
        or harness
        or harness_metadata(checkpoint_step=10, reminder_interval=3),
        **split_policy(split),
    }


def infrastructure_attempt(
    task: dict[str, Any],
    sample_index: int,
    attempt_index: int,
    split: str,
    status: str,
    returncode: int,
    started: float,
    attempt_dir: Path,
    harness: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "lottie_pass_at_k_sample_v1",
        "instance_id": task["instance_id"],
        "dataset": task["experiment_dataset"],
        "sample_index": sample_index,
        "attempt_index": attempt_index,
        "sample_valid": False,
        "benchmark_resolved": None,
        "status": status,
        "runner_returncode": returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "attempt_dir": str(attempt_dir),
        "harness": harness,
        **split_policy(split),
    }


def refresh_summary(output_dir: Path, expected_samples: int) -> dict[str, Any]:
    rows = []
    for path in sorted(output_dir.glob("sample_*/tasks/*/*/sample_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("sample_valid") is True:
            rows.append(row)
    write_jsonl(output_dir / "sample_results.jsonl", rows)
    summary = summarize_pass_at_k(rows, expected_samples=expected_samples, ks=metric_ks(expected_samples))
    summary.update(
        {
            "format": "lottie_pass_at_k_summary_v1",
            "valid_rollouts": len(rows),
            "resolved_rollouts": sum(row.get("benchmark_resolved") is True for row in rows),
            "unresolved_rollouts": sum(row.get("benchmark_resolved") is False for row in rows),
            "blocked_samples_excluded": len(
                [
                    path
                    for path in output_dir.glob("sample_*/tasks/*/*/sample_result.json")
                    if json.loads(path.read_text(encoding="utf-8")).get("sample_valid") is not True
                ]
            ),
        }
    )
    write_json(output_dir / "pass_at_k_summary.json", summary)
    write_jsonl(output_dir / "task_pass_at_k.jsonl", summary["tasks"])
    return summary


def load_selected_tasks(
    selection_path: Path,
    evalplus_path: Path,
    swesmith_path: Path,
    *,
    split: str = "evaluation",
) -> list[dict[str, Any]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    evalplus = {row["instance_id"]: row for row in read_jsonl(evalplus_path)}
    swesmith = {row["instance_id"]: row for row in read_jsonl(swesmith_path)}
    ordered: list[dict[str, Any]] = []
    if "splits" in selection:
        selected = selection["splits"][split]
    elif "sets" in selection and split == "evaluation":
        selected = selection["sets"]
    else:
        raise ValueError(f"Selection {selection_path} does not provide split {split!r}")
    expected = {
        "mbppplus": [
            f"MBPP/{value}" for value in selected.get("mbppplus", {}).get("task_ids", [])
        ],
        "humanevalplus": list(selected.get("humanevalplus", {}).get("task_ids", [])),
        "swesmith_py": list(selected.get("swesmith_py", {}).get("instance_ids", [])),
    }
    for dataset in ("mbppplus", "humanevalplus", "swesmith_py"):
        source = swesmith if dataset == "swesmith_py" else evalplus
        for instance_id in expected[dataset]:
            if instance_id not in source:
                raise KeyError(f"Selected task missing from local task file: {instance_id}")
            task = dict(source[instance_id])
            task["experiment_dataset"] = dataset
            if task.get("split") != split:
                raise ValueError(f"Selected task has split {task.get('split')!r}, expected {split!r}: {instance_id}")
            ordered.append(task)
    expected_count = int(selected.get("count") or sum(len(values) for values in expected.values()))
    if len(ordered) != expected_count or len({task["instance_id"] for task in ordered}) != expected_count:
        raise ValueError(
            f"Selection split {split!r} must contain exactly {expected_count} unique tasks; found {len(ordered)}"
        )
    return ordered


def experiment_config(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    selection_path: Path,
    evalplus_path: Path,
    swesmith_path: Path,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "format": "lottie_pass_at_k_experiment_v1",
        "experiment_id": Path(args.output_dir).name,
        "samples_per_task": args.samples,
        "task_count": len(tasks),
        "task_ids": [task["instance_id"] for task in tasks],
        "task_hashes": {task["instance_id"]: hash_json(task) for task in tasks},
        "selection": {"path": str(selection_path), "sha256": hash_file(selection_path)},
        "task_sources": {
            "evalplus": {"path": str(evalplus_path), "sha256": hash_file(evalplus_path)},
            "swesmith": {"path": str(swesmith_path), "sha256": hash_file(swesmith_path)},
        },
        "model": {
            "name": args.model,
            "url": args.model_url,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "protocol_retry_temperature": args.protocol_retry_temperature,
            "max_tokens": args.max_tokens,
            "thinking_mode": args.thinking_mode,
            "reasoning_effort": args.reasoning_effort,
        },
        "harness": {
            **harness_metadata(
                checkpoint_step=args.edit_checkpoint_step,
                reminder_interval=args.readonly_reminder_interval,
            ),
            "agent_mode": "bench",
            "tool_profile": "official-core",
            "prompt_workspace_label": "repository",
            "function_max_steps": args.function_max_steps,
            "repo_max_steps": args.repo_max_steps,
            "context_max_tokens": args.context_max_tokens,
            "context_overflow_policy": "error",
            "protocol_retries": args.protocol_retries,
            "source_sha256": hash_python_tree([ROOT / "src", ROOT / "scripts"]),
        },
        "execution_policy": {
            "workers": args.workers,
            "model_timeout_sec": args.model_timeout_sec,
            "function_verify_timeout_sec": args.function_verify_timeout_sec,
            "repo_verify_timeout_sec": args.repo_verify_timeout_sec,
            "max_infrastructure_attempts_per_sample": args.max_infra_attempts,
            "api_retries_inside_rollout": 6,
            "api_retry_counts_as_new_sample": False,
        },
        "verifier": {
            "evalplus": "local_hidden_tests",
            "swesmith": "local_venv_fail_to_pass_plus_pass_to_pass",
            "official_comparable": False,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "selection_filter": {"dataset": args.dataset, "max_tasks": args.max_tasks},
        "split": args.split,
        **split_policy(args.split),
    }
    config["config_sha256"] = hash_json(config)
    return config


def initialize_or_validate_experiment(output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != config["config_sha256"]:
            if resume_compatible_hash(existing) != resume_compatible_hash(config):
                raise RuntimeError(
                    "Experiment configuration mismatch; choose a new --output-dir instead of mixing versions. "
                    f"existing={existing.get('config_sha256')} requested={config['config_sha256']}"
                )
            record_execution_policy(output_dir, config, event="resume_worker_change")
            return
        record_execution_policy(output_dir, config, event="resume")
        return
    write_json(manifest, config)
    record_execution_policy(output_dir, config, event="initialize")


def resume_compatible_hash(config: dict[str, Any]) -> str:
    """Hash semantic settings while allowing worker-only recovery changes."""
    compatible = json.loads(json.dumps(config))
    compatible.pop("config_sha256", None)
    execution_policy = compatible.get("execution_policy")
    if isinstance(execution_policy, dict):
        execution_policy.pop("workers", None)
    return hash_json(compatible)


def record_execution_policy(output_dir: Path, config: dict[str, Any], *, event: str) -> None:
    policy = config.get("execution_policy") or {}
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "pid": os.getpid(),
        "workers": policy.get("workers"),
        "model_timeout_sec": policy.get("model_timeout_sec"),
        "function_verify_timeout_sec": policy.get("function_verify_timeout_sec"),
        "repo_verify_timeout_sec": policy.get("repo_verify_timeout_sec"),
        "max_infrastructure_attempts_per_sample": policy.get("max_infrastructure_attempts_per_sample"),
        "config_sha256": config.get("config_sha256"),
        "resume_compatible_sha256": resume_compatible_hash(config),
        "harness_version": (config.get("harness") or {}).get("version"),
        "readonly_policy_version": ((config.get("harness") or {}).get("readonly_budget_policy") or {}).get(
            "version"
        ),
    }
    path = output_dir / "execution_policy_history.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_policy(split: str) -> dict[str, Any]:
    if split == "train":
        return {"data_role": "train", "exclude_from_training": False}
    return {"data_role": f"{split}_only", "exclude_from_training": True}


def metric_ks(expected_samples: int) -> tuple[int, ...]:
    return tuple(sorted({1, min(3, expected_samples), expected_samples}))


def sample_directory(output_dir: Path, task: dict[str, Any], sample_index: int) -> Path:
    return output_dir / f"sample_{sample_index:02d}" / "tasks" / task["experiment_dataset"] / safe_name(task["instance_id"])


def load_sample_result(output_dir: Path, task: dict[str, Any], sample_index: int) -> dict[str, Any] | None:
    path = sample_directory(output_dir, task, sample_index) / "sample_result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def print_progress(
    event: str,
    sample_index: int,
    samples: int,
    task_index: int,
    task_count: int,
    task: dict[str, Any],
    details: dict[str, Any],
) -> None:
    payload = {
        "event": event,
        "sample": f"{sample_index}/{samples}",
        "task": f"{task_index}/{task_count}",
        "instance_id": task["instance_id"],
    }
    for key in ("attempt_index", "sample_valid", "benchmark_resolved", "agent_status", "verifier_status"):
        if key in details:
            payload[key] = details[key]
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def tee_command(command: list[str], log_path: Path, *, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()


def sum_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    output = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    for event in events:
        usage = event.get("usage") or {}
        output["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        output["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        output["total_tokens"] += int(usage.get("total_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        output["cached_tokens"] += int(details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0)
    return output


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return f"{cleaned[:100]}_{hashlib.sha256(value.encode()).hexdigest()[:10]}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_single_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    rows = read_jsonl(path)
    return rows[0] if rows else None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def hash_python_tree(roots: list[Path]) -> str:
    digest = hashlib.sha256()
    files = sorted(path for root in roots for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
