from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .evalplus_local import (
    build_runnable_tasks as build_evalplus_runnable_tasks,
    materialize_function_workspaces,
    run_gold_sanity as run_evalplus_gold_sanity,
    setup_shared_venv,
)
from .swesmith_local import (
    build_runnable_tasks as build_swesmith_runnable_tasks,
    cache_key,
    run_gold_sanity as run_swesmith_gold_sanity,
    setup_repositories,
    venv_python,
)


EXPECTED_EVALUATION_COUNTS = {"mbppplus": 30, "humanevalplus": 30, "swesmith_py": 30}


@dataclass(frozen=True)
class Eval90Layout:
    project_root: Path
    state_root: Path
    function_tasks: Path
    function_agent_tasks: Path
    function_sanity: Path
    function_runnable: Path
    function_repo_root: Path
    function_venv: Path
    swesmith_tasks: Path
    swesmith_sanity: Path
    swesmith_runnable: Path
    swesmith_repo_root: Path
    swesmith_venv_root: Path
    reports: Path

    @classmethod
    def create(cls, project_root: str | Path, state_root: str | Path) -> "Eval90Layout":
        project = Path(project_root).expanduser().resolve()
        state = Path(state_root).expanduser().resolve()
        return cls(
            project_root=project,
            state_root=state,
            function_tasks=state / "tasks" / "evalplus_evaluation.jsonl",
            function_agent_tasks=state / "tasks" / "evalplus_agent_evaluation.jsonl",
            function_sanity=state / "results" / "evalplus_gold_sanity.jsonl",
            function_runnable=state / "tasks" / "evalplus_runnable_evaluation.jsonl",
            function_repo_root=state / "repos" / "evalplus",
            function_venv=state / "venvs" / "evalplus",
            swesmith_tasks=state / "tasks" / "swesmith_evaluation.jsonl",
            swesmith_sanity=state / "results" / "swesmith_gold_sanity.jsonl",
            swesmith_runnable=state / "tasks" / "swesmith_runnable_evaluation.jsonl",
            swesmith_repo_root=state / "repos" / "swesmith",
            swesmith_venv_root=state / "venvs" / "swesmith",
            reports=state / "reports",
        )

    def ensure_directories(self) -> None:
        for path in (
            self.function_tasks.parent,
            self.function_sanity.parent,
            self.function_repo_root,
            self.function_venv.parent,
            self.swesmith_repo_root,
            self.swesmith_venv_root,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)


def prepare_evaluation_tasks(
    layout: Eval90Layout,
    *,
    function_source: str | Path | None = None,
    swesmith_source: str | Path | None = None,
) -> dict[str, Any]:
    layout.ensure_directories()
    function_path = Path(function_source or layout.project_root / "data/evalplus_local/tasks.jsonl")
    swesmith_path = Path(swesmith_source or layout.project_root / "data/swesmith_local/tasks.jsonl")
    function_rows = [row for row in read_jsonl(function_path) if row.get("split") == "evaluation"]
    swesmith_rows = [row for row in read_jsonl(swesmith_path) if row.get("split") == "evaluation"]

    for row in swesmith_rows:
        repo_key = cache_key(str(row["benchmark_repo"]))
        row["local_repo_path"] = str(layout.swesmith_repo_root / repo_key)
        row["local_venv_path"] = str(layout.swesmith_venv_root / repo_key)

    write_jsonl(layout.function_tasks, function_rows)
    write_jsonl(layout.swesmith_tasks, swesmith_rows)
    manifest = {
        "format": "lottie_eval90_deployment_tasks_v1",
        "state_root": str(layout.state_root),
        "function_source": str(function_path.resolve()),
        "swesmith_source": str(swesmith_path.resolve()),
        "counts": {
            "mbppplus": count_rows(function_rows, "dataset", "mbppplus"),
            "humanevalplus": count_rows(function_rows, "dataset", "humanevalplus"),
            "swesmith_py": len(swesmith_rows),
        },
        "swesmith_repositories": sorted({str(row["benchmark_repo"]) for row in swesmith_rows}),
        "portable_paths_rewritten": True,
    }
    ensure_expected_counts(manifest["counts"])
    write_json(layout.reports / "task_manifest.json", manifest)
    return manifest


def setup_function_environment(
    layout: Eval90Layout,
    *,
    python_executable: str | Path,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    require_prepared(layout.function_tasks)
    venv = setup_shared_venv(python_executable, layout.function_venv, timeout_sec)
    workspace = materialize_function_workspaces(
        layout.function_tasks,
        layout.function_agent_tasks,
        layout.function_repo_root,
    )
    result = {"venv": venv, "workspaces": workspace}
    write_json(layout.reports / "function_setup.json", result)
    return result


def setup_swesmith_environment(
    layout: Eval90Layout,
    *,
    python_executable: str | Path,
    timeout_sec: int = 1800,
) -> dict[str, Any]:
    require_prepared(layout.swesmith_tasks)
    setup_results = setup_repositories(
        layout.swesmith_tasks,
        python_executable=python_executable,
        timeout_sec=timeout_sec,
    )
    rows = [asdict(item) for item in setup_results]
    write_jsonl(layout.reports / "swesmith_setup.jsonl", rows)
    summary = {
        "repositories": len(rows),
        "ready": sum(row["status"] == "ready" for row in rows),
        "blocked": sum(row["status"] != "ready" for row in rows),
        "repo_bytes": sum(int(row["repo_bytes"]) for row in rows),
        "venv_bytes": sum(int(row["venv_bytes"]) for row in rows),
        "results": rows,
    }
    write_json(layout.reports / "swesmith_setup_summary.json", summary)
    return summary


def sanity_function_tasks(
    layout: Eval90Layout,
    *,
    timeout_sec: int = 30,
) -> dict[str, Any]:
    python = layout.function_venv / "bin" / "python"
    results = run_evalplus_gold_sanity(
        layout.function_tasks,
        layout.function_sanity,
        python,
        split="evaluation",
        timeout_sec=timeout_sec,
    )
    runnable = build_evalplus_runnable_tasks(
        layout.function_tasks,
        layout.function_sanity,
        layout.function_runnable,
    )
    summary = {
        "tasks": len(results),
        "passed": sum(item.gold_sanity_passed for item in results),
        "status_counts": dict(Counter(item.status for item in results)),
        "runnable": runnable,
    }
    write_json(layout.reports / "function_sanity_summary.json", summary)
    return summary


def sanity_swesmith_tasks(
    layout: Eval90Layout,
    *,
    timeout_sec: int = 600,
    p2p_limit: int = 10,
) -> dict[str, Any]:
    results = run_swesmith_gold_sanity(
        layout.swesmith_tasks,
        layout.swesmith_sanity,
        split="evaluation",
        p2p_limit=p2p_limit,
        timeout_sec=timeout_sec,
    )
    runnable = build_swesmith_runnable_tasks(
        layout.swesmith_tasks,
        layout.swesmith_sanity,
        layout.swesmith_runnable,
    )
    summary = {
        "tasks": len(results),
        "passed": sum(item.gold_sanity_passed for item in results),
        "status_counts": dict(Counter(item.status for item in results)),
        "runnable": runnable,
    }
    write_json(layout.reports / "swesmith_sanity_summary.json", summary)
    return summary


def deployment_status(layout: Eval90Layout, *, python_executable: str | Path | None = None) -> dict[str, Any]:
    function_tasks = read_jsonl_if_exists(layout.function_tasks)
    function_results = read_jsonl_if_exists(layout.function_sanity)
    function_agent_tasks = read_jsonl_if_exists(layout.function_agent_tasks)
    swesmith_tasks = read_jsonl_if_exists(layout.swesmith_tasks)
    swesmith_results = read_jsonl_if_exists(layout.swesmith_sanity)
    setup_results = read_jsonl_if_exists(layout.reports / "swesmith_setup.jsonl")

    counts = {
        "mbppplus": count_rows(function_tasks, "dataset", "mbppplus"),
        "humanevalplus": count_rows(function_tasks, "dataset", "humanevalplus"),
        "swesmith_py": len(swesmith_tasks),
    }
    gold_passed = {
        "function": sum(row.get("gold_sanity_passed") is True for row in function_results),
        "swesmith": sum(row.get("gold_sanity_passed") is True for row in swesmith_results),
    }
    expected_repos = len({str(row.get("benchmark_repo")) for row in swesmith_tasks})
    ready_repos = sum(row.get("status") == "ready" for row in setup_results)
    python_info = inspect_python(python_executable) if python_executable else None
    checks = {
        "task_counts": counts == EXPECTED_EVALUATION_COUNTS,
        "function_venv": (layout.function_venv / "bin" / "python").is_file(),
        "function_workspaces": len(function_agent_tasks) == 60
        and all(Path(str(row.get("local_repo_path", ""))).is_dir() for row in function_agent_tasks),
        "function_gold": gold_passed["function"] == 60,
        "swesmith_repositories": expected_repos == 6 and ready_repos == 6,
        "swesmith_gold": gold_passed["swesmith"] == 30,
    }
    result = {
        "format": "lottie_eval90_deployment_status_v1",
        "state_root": str(layout.state_root),
        "ready": all(checks.values()),
        "checks": checks,
        "task_counts": counts,
        "gold_passed": gold_passed,
        "swesmith_repositories": {"expected": expected_repos, "ready": ready_repos},
        "python": python_info,
        "disk_bytes": directory_size(layout.state_root),
    }
    layout.reports.mkdir(parents=True, exist_ok=True)
    write_json(layout.reports / "deployment_status.json", result)
    return result


def doctor(layout: Eval90Layout, *, python_executable: str | Path) -> dict[str, Any]:
    python = inspect_python(python_executable)
    required_files = [
        layout.project_root / "data/dataset_splits_v1/selection.json",
        layout.project_root / "data/evalplus_local/tasks.jsonl",
        layout.project_root / "data/swesmith_local/tasks.jsonl",
    ]
    disk = shutil.disk_usage(layout.state_root.parent if layout.state_root.parent.exists() else layout.project_root)
    result = {
        "format": "lottie_eval90_doctor_v1",
        "python": python,
        "commands": {name: shutil.which(name) for name in ("git", "bash")},
        "required_files": {str(path): path.is_file() for path in required_files},
        "disk_free_bytes": disk.free,
    }
    result["ready"] = (
        python["available"]
        and python.get("major_minor") == "3.10"
        and all(result["commands"].values())
        and all(result["required_files"].values())
        and disk.free >= 5 * 1024**3
    )
    layout.reports.mkdir(parents=True, exist_ok=True)
    write_json(layout.reports / "doctor.json", result)
    return result


def inspect_python(executable: str | Path) -> dict[str, Any]:
    path = Path(executable).expanduser()
    if not path.is_file():
        return {"available": False, "executable": str(path)}
    completed = subprocess.run(
        [str(path), "-c", "import json,sys; print(json.dumps({'version':sys.version.split()[0],'prefix':sys.prefix}))"],
        text=True,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode:
        return {"available": False, "executable": str(path), "error": (completed.stderr or completed.stdout)[-2000:]}
    info = json.loads(completed.stdout)
    version = str(info["version"])
    return {
        "available": True,
        "executable": str(path.resolve()),
        "version": version,
        "major_minor": ".".join(version.split(".")[:2]),
        "prefix": info["prefix"],
    }


def ensure_expected_counts(counts: dict[str, int]) -> None:
    if counts != EXPECTED_EVALUATION_COUNTS:
        raise ValueError(f"Eval90 task count mismatch: expected {EXPECTED_EVALUATION_COUNTS}, got {counts}")


def require_prepared(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared task file not found: {path}. Run the prepare phase first.")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_jsonl_if_exists(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    return read_jsonl(target) if target.is_file() else []


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_rows(rows: Iterable[dict[str, Any]], key: str, value: str) -> int:
    return sum(str(row.get(key)) == value for row in rows)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
