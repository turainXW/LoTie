#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.swesmith_local import (  # noqa: E402
    cache_key,
    discovered_test_dependency_commands,
    read_jsonl,
    sanity_one_task,
    setup_repositories,
    venv_python,
)


DEFAULT_TASKS = ["data/swesmith_train_extension_v1/tasks_runnable.jsonl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild portable per-repository SWE-smith venvs without Docker."
    )
    parser.add_argument(
        "--tasks",
        action="append",
        help="Task JSONL. Repeat to merge multiple disjoint task sets.",
    )
    parser.add_argument("--cache-root", default=".codeagent/swesmith_portable")
    parser.add_argument("--output-dir", default="data/runtime/swesmith_portable")
    parser.add_argument("--python", help="Python 3.10 executable. Auto-detected when omitted.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent repository setups.")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument(
        "--smoke-per-repo",
        type=int,
        default=1,
        help="Gold-sanity tasks per repository after setup; 0 disables smoke checks.",
    )
    parser.add_argument("--sanity-timeout-sec", type=int, default=600)
    parser.add_argument("--p2p-limit", type=int, default=10)
    parser.add_argument(
        "--allow-python-version-mismatch",
        action="store_true",
        help="Allow a Python version other than the task-declared version.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_python(explicit: str | None) -> Path:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("PYTHON310"):
        candidates.append(os.environ["PYTHON310"])
    candidates.extend(["python3.10", "python3"])
    if sys.version_info[:2] == (3, 10):
        candidates.append(sys.executable)
    checked: set[str] = set()
    for candidate in candidates:
        resolved = shutil.which(candidate) or candidate
        if resolved in checked:
            continue
        checked.add(resolved)
        path = Path(resolved).expanduser().resolve()
        if not path.is_file():
            continue
        probe = subprocess.run(
            [str(path), "-c", "import json,sys; print(json.dumps(list(sys.version_info[:3])))"],
            text=True,
            capture_output=True,
            timeout=20,
        )
        if probe.returncode == 0:
            return path
    raise FileNotFoundError("No usable Python found. Install Python 3.10 or pass --python.")


def python_version(python: Path) -> tuple[int, int, int]:
    completed = subprocess.run(
        [str(python), "-c", "import json,sys; print(json.dumps(list(sys.version_info[:3])))"],
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    major, minor, patch = json.loads(completed.stdout)
    return int(major), int(minor), int(patch)


def materialize_runtime_tasks(
    task_paths: list[Path],
    output_path: Path,
    cache_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for path in task_paths:
        rows = read_jsonl(path)
        sources.append({"path": str(path), "sha256": sha256_file(path), "rows": len(rows)})
        for row in rows:
            instance_id = str(row["instance_id"])
            current = dict(row)
            repo = str(current["benchmark_repo"])
            key = cache_key(repo)
            current["local_repo_path"] = str(cache_root / "repos" / key)
            current["local_venv_path"] = str(cache_root / "venvs" / key)
            previous = by_id.get(instance_id)
            if previous is not None and canonical_task(previous) != canonical_task(current):
                raise ValueError(f"Conflicting duplicate task: {instance_id}")
            by_id[instance_id] = current
    tasks = list(by_id.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    return tasks, sources


def canonical_task(task: dict[str, Any]) -> str:
    portable = dict(task)
    portable.pop("local_repo_path", None)
    portable.pop("local_venv_path", None)
    return json.dumps(portable, sort_keys=True, ensure_ascii=False)


def setup_one(
    tasks_path: Path,
    repo: str,
    python: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    results = setup_repositories(
        tasks_path,
        python_executable=python,
        timeout_sec=timeout_sec,
        repo_filter={repo},
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one setup result for {repo}, found {len(results)}")
    return asdict(results[0])


def setup_all(
    tasks_path: Path,
    repos: list[str],
    python: Path,
    workers: int,
    timeout_sec: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="swesmith-setup") as executor:
        futures = {
            executor.submit(setup_one, tasks_path, repo, python, timeout_sec): repo for repo in repos
        }
        for future in as_completed(futures):
            repo = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "repo": repo,
                    "status": "blocked_setup_worker_error",
                    "output": repr(exc),
                }
            results.append(result)
            print(json.dumps({"event": "SETUP", **result}, ensure_ascii=False), flush=True)
    return sorted(results, key=lambda item: str(item["repo"]))


def select_smoke_tasks(tasks: list[dict[str, Any]], per_repo: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for task in tasks:
        repo = str(task["benchmark_repo"])
        if counts.get(repo, 0) >= per_repo:
            continue
        selected.append(task)
        counts[repo] = counts.get(repo, 0) + 1
    return selected


def run_smoke_checks(
    tasks: list[dict[str, Any]],
    workers: int,
    timeout_sec: int,
    p2p_limit: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="swesmith-sanity") as executor:
        futures = {
            executor.submit(
                sanity_one_task,
                task,
                p2p_limit=p2p_limit,
                timeout_sec=timeout_sec,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = asdict(future.result())
            except Exception as exc:
                result = {
                    "instance_id": task["instance_id"],
                    "repo": task["benchmark_repo"],
                    "status": "blocked_sanity_worker_error",
                    "gold_sanity_passed": False,
                    "output": {"exception": repr(exc)},
                }
            results.append(result)
            print(
                json.dumps(
                    {
                        "event": "SANITY",
                        "instance_id": result["instance_id"],
                        "repo": result["repo"],
                        "status": result["status"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return sorted(results, key=lambda item: str(item["instance_id"]))


def portable_freeze_lines(raw: str, repo_path: Path) -> list[str]:
    lines: list[str] = []
    repo_text = str(repo_path.resolve())
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("# Editable install") or line.startswith("-e "):
            continue
        if " @ file://" in line or repo_text in line:
            continue
        lines.append(line)
    return sorted(set(lines), key=str.casefold)


def write_environment_artifacts(
    tasks: list[dict[str, Any]],
    setup_results: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    by_repo: dict[str, dict[str, Any]] = {}
    for task in tasks:
        by_repo.setdefault(str(task["benchmark_repo"]), task)
    requirements_dir = output_dir / "requirements"
    requirements_dir.mkdir(parents=True, exist_ok=True)
    setup_by_repo = {str(item["repo"]): item for item in setup_results}
    recipes: list[dict[str, Any]] = []
    for repo in sorted(by_repo):
        task = by_repo[repo]
        repo_path = Path(task["local_repo_path"])
        venv_path = Path(task["local_venv_path"])
        key = cache_key(repo)
        requirement_path = requirements_dir / f"{key}.requirements.txt"
        freeze_status = "not_ready"
        lines: list[str] = []
        if setup_by_repo.get(repo, {}).get("status") == "ready":
            completed = subprocess.run(
                [str(venv_python(venv_path)), "-m", "pip", "freeze", "--all"],
                text=True,
                capture_output=True,
                timeout=120,
            )
            if completed.returncode == 0:
                lines = portable_freeze_lines(completed.stdout, repo_path)
                freeze_status = "written"
            else:
                freeze_status = "pip_freeze_failed"
        requirement_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        marker = venv_path / ".lottie-ready.json"
        marker_data = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
        recipes.append(
            {
                "repo": repo,
                "source_repo": task["source_repo"],
                "base_commit": task["base_commit"],
                "python_version": task.get("python_version", "3.10"),
                "install_commands": task.get("install_commands", []),
                "discovered_test_dependency_commands": discovered_test_dependency_commands(repo_path)
                if repo_path.is_dir()
                else [],
                "dependency_fingerprint": marker_data.get("fingerprint"),
                "requirements_snapshot": str(requirement_path.relative_to(output_dir)),
                "requirements_count": len(lines),
                "freeze_status": freeze_status,
                "snapshot_policy": "diagnostic_same-platform_snapshot; bootstrap uses repository recipes",
            }
        )
    write_jsonl(output_dir / "environment_recipes.jsonl", recipes)
    return recipes


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.smoke_per_repo < 0:
        raise ValueError("--smoke-per-repo must be non-negative")
    task_paths = [resolve_path(path) for path in (args.tasks or DEFAULT_TASKS)]
    for path in task_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    cache_root = resolve_path(args.cache_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_tasks = output_dir / "tasks_runtime.jsonl"
    python = detect_python(args.python)
    detected_version = python_version(python)

    tasks, sources = materialize_runtime_tasks(task_paths, runtime_tasks, cache_root)
    declared_versions = sorted({str(task.get("python_version") or "3.10") for task in tasks})
    if not args.allow_python_version_mismatch:
        expected = {(int(value.split(".")[0]), int(value.split(".")[1])) for value in declared_versions}
        if detected_version[:2] not in expected:
            raise RuntimeError(
                f"Detected Python {detected_version[0]}.{detected_version[1]}, "
                f"but task files declare {declared_versions}. Pass --python or explicitly allow mismatch."
            )

    repos = sorted({str(task["benchmark_repo"]) for task in tasks})
    setup_results = setup_all(runtime_tasks, repos, python, args.workers, args.timeout_sec)
    write_jsonl(output_dir / "setup_results.jsonl", setup_results)
    setup_ready = sum(item.get("status") == "ready" for item in setup_results)

    smoke_tasks = select_smoke_tasks(tasks, args.smoke_per_repo)
    sanity_results = (
        run_smoke_checks(
            smoke_tasks,
            args.workers,
            args.sanity_timeout_sec,
            args.p2p_limit,
        )
        if smoke_tasks and setup_ready == len(repos)
        else []
    )
    write_jsonl(output_dir / "smoke_sanity.jsonl", sanity_results)
    recipes = write_environment_artifacts(tasks, setup_results, output_dir)
    sanity_passed = sum(bool(item.get("gold_sanity_passed")) for item in sanity_results)
    manifest = {
        "format": "lottie_portable_swesmith_venvs_v1",
        "platform": platform.platform(),
        "python": {"path": str(python), "version": list(detected_version)},
        "task_sources": sources,
        "runtime_tasks": {"path": str(runtime_tasks), "sha256": sha256_file(runtime_tasks)},
        "cache_root": str(cache_root),
        "tasks": len(tasks),
        "repositories": len(repos),
        "workers": args.workers,
        "setup_ready": setup_ready,
        "setup_blocked": len(repos) - setup_ready,
        "smoke_tasks": len(sanity_results),
        "smoke_passed": sanity_passed,
        "smoke_failed": len(sanity_results) - sanity_passed,
        "environment_recipes": len(recipes),
        "docker_required": False,
        "official_comparable": False,
    }
    write_json(output_dir / "bootstrap_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if setup_ready != len(repos) or sanity_passed != len(sanity_results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
