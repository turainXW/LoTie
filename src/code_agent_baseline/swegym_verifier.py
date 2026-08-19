from __future__ import annotations

import json
import ast
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .runtime_env import task_runtime_env


IMAGE_FIELDS = (
    "image",
    "image_name",
    "docker_image",
    "docker_image_name",
    "eval_image",
    "instance_image",
)
COMMAND_FIELDS = (
    "test_command",
    "validation_command",
    "eval_command",
    "run_tests_command",
    "test_cmd",
)
DEFAULT_EVAL_IMAGE_REGISTRY = "docker.1ms.run/xingyaoww"


@dataclass
class SwegymVerifyResult:
    instance_id: str
    benchmark_resolved: bool | None
    verifier_status: str
    patch_present: bool
    image: str | None = None
    test_command: str | None = None
    returncode: int | None = None
    output: str = ""
    elapsed_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SwegymVerifySummary:
    total: int
    resolved: int
    unresolved: int
    blocked: int
    output_path: str
    status_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SwegymTaskPreflightResult:
    instance_id: str
    runnable: bool
    status: str
    image: str | None = None
    test_command: str | None = None
    returncode: int | None = None
    output: str = ""
    elapsed_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_swegym_patch_records(
    records_path: str | Path,
    output_path: str | Path,
    *,
    tasks_path: str | Path | None = None,
    mode: str = "dry-run",
    timeout_sec: int = 900,
    limit: int | None = None,
    local_repo_path: str | Path | None = None,
    local_venv_path: str | Path | None = None,
    venv_cache_dir: str | Path | None = None,
    install_deps: bool = False,
) -> SwegymVerifySummary:
    records = read_jsonl(records_path)
    if limit is not None:
        records = records[:limit]
    tasks = load_tasks_by_instance_id(tasks_path) if tasks_path else {}

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    cache_dir = Path(venv_cache_dir).resolve() if venv_cache_dir else output.parent / ".codeagent" / "venvs"

    results: list[SwegymVerifyResult] = []
    for record in records:
        task = tasks.get(str(record.get("instance_id")), {})
        result = verify_one_record(
            record,
            task,
            mode=mode,
            timeout_sec=timeout_sec,
            local_repo_path=local_repo_path,
            local_venv_path=local_venv_path,
            venv_cache_dir=cache_dir,
            install_deps=install_deps,
        )
        results.append(result)
        append_jsonl(output, result.to_dict())

    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.verifier_status] = status_counts.get(result.verifier_status, 0) + 1

    return SwegymVerifySummary(
        total=len(results),
        resolved=sum(result.benchmark_resolved is True for result in results),
        unresolved=sum(result.benchmark_resolved is False for result in results),
        blocked=sum(result.benchmark_resolved is None for result in results),
        output_path=str(output),
        status_counts=status_counts,
    )


def verify_one_record(
    record: dict[str, Any],
    task: dict[str, Any],
    *,
    mode: str,
    timeout_sec: int,
    local_repo_path: str | Path | None = None,
    local_venv_path: str | Path | None = None,
    venv_cache_dir: str | Path | None = None,
    install_deps: bool = False,
) -> SwegymVerifyResult:
    instance_id = str(record.get("instance_id", ""))
    patch = build_eval_patch(record, task)
    image = find_first_string(task, IMAGE_FIELDS) or infer_swegym_eval_image(instance_id)
    test_command = find_test_command(task)

    if not patch.strip():
        return SwegymVerifyResult(
            instance_id=instance_id,
            benchmark_resolved=None,
            verifier_status="blocked_missing_patch",
            patch_present=False,
            image=image,
            test_command=test_command,
        )

    if not test_command:
        return SwegymVerifyResult(
            instance_id=instance_id,
            benchmark_resolved=None,
            verifier_status="blocked_missing_test_command",
            patch_present=True,
            image=image,
            test_command=None,
            metadata={"available_task_fields": sorted(task.keys()) if task else []},
        )

    if mode == "local-venv":
        repo_path = resolve_local_repo_path(task, override=local_repo_path)
        if repo_path is None or not repo_path.is_dir():
            return SwegymVerifyResult(
                instance_id=instance_id,
                benchmark_resolved=None,
                verifier_status="blocked_missing_local_repo",
                patch_present=True,
                image=image,
                test_command=test_command,
                metadata={
                    "verifier_backend": "local_venv",
                    "local_repo_path": str(repo_path) if repo_path else None,
                },
            )
        return run_local_venv_verifier(
            instance_id=instance_id,
            patch=patch,
            bug_patch=str(task.get("bug_patch") or task.get("setup_patch") or ""),
            repo_path=repo_path,
            test_command=test_command,
            timeout_sec=timeout_sec,
            venv_path=resolve_local_venv_path(task, override=local_venv_path),
            venv_cache_dir=Path(venv_cache_dir).resolve() if venv_cache_dir else None,
            install_deps=install_deps,
            has_test_patch=bool(str(task.get("test_patch") or "").strip()),
        )

    if not image:
        return SwegymVerifyResult(
            instance_id=instance_id,
            benchmark_resolved=None,
            verifier_status="blocked_missing_official_image",
            patch_present=True,
            image=None,
            test_command=test_command,
            metadata={"available_task_fields": sorted(task.keys()) if task else []},
        )

    if mode == "dry-run":
        return SwegymVerifyResult(
            instance_id=instance_id,
            benchmark_resolved=None,
            verifier_status="ready_for_docker",
            patch_present=True,
            image=image,
            test_command=test_command,
        )

    if mode != "docker":
        raise ValueError(f"Unsupported verifier mode: {mode}")

    if shutil.which("docker") is None:
        return SwegymVerifyResult(
            instance_id=instance_id,
            benchmark_resolved=None,
            verifier_status="blocked_docker_unavailable",
            patch_present=True,
            image=image,
            test_command=test_command,
        )

    return run_docker_verifier(
        instance_id=instance_id,
        patch=patch,
        image=image,
        test_command=test_command,
        timeout_sec=timeout_sec,
        has_test_patch=bool(str(task.get("test_patch") or "").strip()),
    )


def run_local_venv_verifier(
    *,
    instance_id: str,
    patch: str,
    bug_patch: str,
    repo_path: Path,
    test_command: str,
    timeout_sec: int,
    venv_path: Path | None,
    venv_cache_dir: Path | None,
    install_deps: bool,
    has_test_patch: bool,
) -> SwegymVerifyResult:
    started = time.monotonic()
    selected_venv = venv_path or cached_venv_path(repo_path, venv_cache_dir)
    setup = ensure_local_venv(
        selected_venv,
        repo_path=repo_path,
        timeout_sec=timeout_sec,
        install_deps=install_deps,
        create=venv_path is None,
    )
    if setup["returncode"] != 0:
        return SwegymVerifyResult(
            instance_id=instance_id,
            benchmark_resolved=None,
            verifier_status=str(setup["status"]),
            patch_present=True,
            test_command=test_command,
            output=str(setup["output"])[-12000:],
            elapsed_sec=round(time.monotonic() - started, 3),
            metadata={
                "verifier_backend": "local_venv",
                "local_repo_path": str(repo_path),
                "venv_path": str(selected_venv),
            },
        )

    python = local_venv_python(selected_venv)
    with tempfile.TemporaryDirectory(prefix="codeagent_local_verify_") as tmp:
        temp_root = Path(tmp)
        workspace = temp_root / "repo"
        shutil.copytree(repo_path, workspace, ignore=shutil.ignore_patterns(*LOCAL_COPY_IGNORE_PATTERNS))
        patch_path = temp_root / "agent.patch"
        patch_path.write_text(patch, encoding="utf-8")
        if bug_patch.strip():
            bug_patch_path = temp_root / "bug.patch"
            bug_patch_path.write_text(bug_patch, encoding="utf-8")
            bug_setup = apply_patch_to_workspace(workspace, bug_patch_path, timeout_sec=timeout_sec)
            if bug_setup.returncode != 0:
                return SwegymVerifyResult(
                    instance_id=instance_id,
                    benchmark_resolved=None,
                    verifier_status="blocked_bug_setup_failed",
                    patch_present=True,
                    test_command=test_command,
                    returncode=bug_setup.returncode,
                    output=((bug_setup.stdout or "") + (bug_setup.stderr or ""))[-12000:],
                    elapsed_sec=round(time.monotonic() - started, 3),
                    metadata={
                        "verifier_backend": "local_venv",
                        "local_repo_path": str(repo_path),
                        "venv_path": str(selected_venv),
                        "bug_patch_present": True,
                    },
                )
        apply_result = apply_patch_to_workspace(workspace, patch_path, timeout_sec=timeout_sec)
        if apply_result.returncode != 0:
            return SwegymVerifyResult(
                instance_id=instance_id,
                benchmark_resolved=False,
                verifier_status="unresolved_patch_apply_failed",
                patch_present=True,
                test_command=test_command,
                returncode=apply_result.returncode,
                output=((apply_result.stdout or "") + (apply_result.stderr or ""))[-12000:],
                elapsed_sec=round(time.monotonic() - started, 3),
                metadata={
                    "verifier_backend": "local_venv",
                    "local_repo_path": str(repo_path),
                    "venv_path": str(selected_venv),
                    "has_test_patch": has_test_patch,
                    "bug_patch_present": bool(bug_patch.strip()),
                },
            )

        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(selected_venv)
        env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONNOUSERSITE"] = "1"
        python_paths = [str(workspace)]
        if (workspace / "src").is_dir():
            python_paths.insert(0, str(workspace / "src"))
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        try:
            completed = subprocess.run(
                test_command,
                cwd=workspace,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return SwegymVerifyResult(
                instance_id=instance_id,
                benchmark_resolved=False,
                verifier_status="timeout",
                patch_present=True,
                test_command=test_command,
                output=decode_timeout_output(exc.stdout) + decode_timeout_output(exc.stderr),
                elapsed_sec=round(time.monotonic() - started, 3),
                metadata={"verifier_backend": "local_venv", "venv_path": str(selected_venv)},
            )

    output = (completed.stdout or "") + (completed.stderr or "")
    return SwegymVerifyResult(
        instance_id=instance_id,
        benchmark_resolved=completed.returncode == 0,
        verifier_status="resolved" if completed.returncode == 0 else "unresolved",
        patch_present=True,
        test_command=test_command,
        returncode=completed.returncode,
        output=output[-12000:],
        elapsed_sec=round(time.monotonic() - started, 3),
        metadata={
            "verifier_backend": "local_venv",
            "local_repo_path": str(repo_path),
            "venv_path": str(selected_venv),
            "has_test_patch": has_test_patch,
            "bug_patch_present": bool(bug_patch.strip()),
            "official_comparable": False,
        },
    )


LOCAL_COPY_IGNORE_PATTERNS = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
)


def resolve_local_repo_path(task: dict[str, Any], *, override: str | Path | None = None) -> Path | None:
    if override:
        return Path(override).expanduser().resolve()
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    task_dir_value = task.get("_task_file_dir")
    task_dir = Path(str(task_dir_value)).resolve() if task_dir_value else None
    candidates = [task.get("local_repo_path"), metadata.get("local_repo_path")]
    if task_dir is not None:
        candidates.append(task_dir / "repo")
    candidates.append(metadata.get("source_repo"))
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate)).expanduser()
        if not path.is_absolute() and task_dir is not None:
            path = task_dir / path
        path = path.resolve()
        if path.is_dir():
            return path
    return None


def resolve_local_venv_path(task: dict[str, Any], *, override: str | Path | None = None) -> Path | None:
    if override:
        return Path(override).expanduser().resolve()
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    for candidate in (task.get("local_venv_path"), metadata.get("local_venv_path")):
        if not candidate:
            continue
        path = Path(str(candidate)).expanduser().resolve()
        if local_venv_python(path).is_file():
            return path
    return None


def cached_venv_path(repo_path: Path, cache_dir: Path | None) -> Path:
    root = cache_dir or repo_path.parent / ".codeagent" / "venvs"
    return root / dependency_fingerprint(repo_path)


def dependency_fingerprint(repo_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(repo_path.resolve()).encode())
    digest.update(sys.version.encode())
    for name in ("requirements.txt", "pyproject.toml", "setup.cfg", "setup.py"):
        path = repo_path / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def local_venv_python(venv_path: Path) -> Path:
    if venv_path.is_file():
        return venv_path
    posix_python = venv_path / "bin" / "python"
    if posix_python.is_file():
        return posix_python
    return venv_path / "Scripts" / "python.exe"


def ensure_local_venv(
    venv_path: Path,
    *,
    repo_path: Path,
    timeout_sec: int,
    install_deps: bool,
    create: bool,
) -> dict[str, Any]:
    python = local_venv_python(venv_path)
    if not python.is_file():
        if not create:
            return {
                "returncode": 1,
                "status": "blocked_local_venv_missing",
                "output": f"Python executable not found in local venv: {venv_path}",
            }
        venv_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "venv", str(venv_path)],
                text=True,
                capture_output=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": 1,
                "status": "blocked_local_venv_setup_timeout",
                "output": decode_timeout_output(exc.stdout) + decode_timeout_output(exc.stderr),
            }
        if completed.returncode != 0:
            return {
                "returncode": completed.returncode,
                "status": "blocked_local_venv_setup_failed",
                "output": (completed.stdout or "") + (completed.stderr or ""),
            }
        python = local_venv_python(venv_path)

    check_env = task_runtime_env(repo_path, venv_path=venv_path)
    check = subprocess.run(
        [str(python), "-c", "import pytest"],
        cwd=repo_path,
        text=True,
        capture_output=True,
        timeout=min(timeout_sec, 60),
        env=check_env,
    )
    if check.returncode != 0 and not create and not install_deps:
        return {
            "returncode": check.returncode,
            "status": "blocked_local_venv_missing_pytest",
            "output": check.stderr,
        }

    install_commands: list[list[str]] = []
    if check.returncode != 0:
        install_commands.append([str(python), "-m", "pip", "install", "pytest"])
    venv_root = venv_path if venv_path.is_dir() else python.parent.parent
    deps_marker = venv_root / f".codeagent-deps-{dependency_fingerprint(repo_path)}"
    if install_deps and not deps_marker.exists():
        if (repo_path / "requirements.txt").is_file():
            install_commands.append([str(python), "-m", "pip", "install", "-r", str(repo_path / "requirements.txt")])
        if any((repo_path / name).is_file() for name in ("pyproject.toml", "setup.py", "setup.cfg")):
            install_commands.append([str(python), "-m", "pip", "install", "-e", str(repo_path)])

    outputs = []
    for command in install_commands:
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": 1,
                "status": "blocked_dependency_install_timeout",
                "output": decode_timeout_output(exc.stdout) + decode_timeout_output(exc.stderr),
            }
        outputs.append((completed.stdout or "") + (completed.stderr or ""))
        if completed.returncode != 0:
            return {
                "returncode": completed.returncode,
                "status": "blocked_dependency_install_failed",
                "output": "\n".join(outputs),
            }
    if install_deps and not deps_marker.exists():
        deps_marker.write_text("ready\n", encoding="utf-8")
    return {"returncode": 0, "status": "ready", "output": "\n".join(outputs)}


def apply_patch_to_workspace(workspace: Path, patch_path: Path, *, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    git_result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch_path)],
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    if git_result.returncode == 0 or shutil.which("patch") is None:
        return git_result
    return subprocess.run(
        ["patch", "-p1", "-i", str(patch_path)],
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )


def preflight_swegym_task(
    task: dict[str, Any],
    *,
    mode: str = "dry-run",
    timeout_sec: int = 300,
    pull: bool = False,
) -> SwegymTaskPreflightResult:
    instance_id = str(task.get("instance_id", ""))
    image = find_first_string(task, IMAGE_FIELDS) or infer_swegym_eval_image(instance_id)
    test_command = find_test_command(task)

    if not image:
        return SwegymTaskPreflightResult(
            instance_id=instance_id,
            runnable=False,
            status="blocked_missing_official_image",
            image=None,
            test_command=test_command,
            metadata={"available_task_fields": sorted(task.keys()) if task else []},
        )
    if not test_command:
        return SwegymTaskPreflightResult(
            instance_id=instance_id,
            runnable=False,
            status="blocked_missing_test_command",
            image=image,
            test_command=None,
            metadata={"available_task_fields": sorted(task.keys()) if task else []},
        )
    if mode == "dry-run":
        return SwegymTaskPreflightResult(
            instance_id=instance_id,
            runnable=True,
            status="ready_for_docker",
            image=image,
            test_command=test_command,
        )
    if mode != "docker":
        raise ValueError(f"Unsupported preflight mode: {mode}")
    if shutil.which("docker") is None:
        return SwegymTaskPreflightResult(
            instance_id=instance_id,
            runnable=False,
            status="blocked_docker_unavailable",
            image=image,
            test_command=test_command,
        )
    return preflight_docker_image(
        instance_id=instance_id,
        image=image,
        test_command=test_command,
        timeout_sec=timeout_sec,
        pull=pull,
    )


def preflight_docker_image(
    *,
    instance_id: str,
    image: str,
    test_command: str,
    timeout_sec: int,
    pull: bool,
) -> SwegymTaskPreflightResult:
    started = time.monotonic()
    command = ["docker", "image", "inspect", image]
    if pull:
        command = ["docker", "pull", image]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        stdout = decode_timeout_output(exc.stdout)
        stderr = decode_timeout_output(exc.stderr)
        return SwegymTaskPreflightResult(
            instance_id=instance_id,
            runnable=False,
            status="blocked_image_preflight_timeout",
            image=image,
            test_command=test_command,
            output=(stdout + stderr)[-4000:],
            elapsed_sec=round(time.monotonic() - started, 3),
            metadata={"command": command},
        )

    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode == 0:
        return SwegymTaskPreflightResult(
            instance_id=instance_id,
            runnable=True,
            status="ready_for_docker",
            image=image,
            test_command=test_command,
            returncode=completed.returncode,
            output=output[-4000:],
            elapsed_sec=round(time.monotonic() - started, 3),
            metadata={"command": command, "pulled": pull},
        )

    status = classify_infra_failure(125, output) or "blocked_image_preflight_failed"
    return SwegymTaskPreflightResult(
        instance_id=instance_id,
        runnable=False,
        status=status,
        image=image,
        test_command=test_command,
        returncode=completed.returncode,
        output=output[-4000:],
        elapsed_sec=round(time.monotonic() - started, 3),
        metadata={"command": command, "pulled": pull},
    )


def build_eval_patch(record: dict[str, Any], task: dict[str, Any]) -> str:
    """Combine the model patch with the official test patch when available."""
    patch = str(record.get("patch") or "")
    test_patch = str(task.get("test_patch") or "")
    if not patch.strip() or not test_patch.strip():
        return patch
    return patch.rstrip() + "\n" + test_patch.lstrip()


def run_docker_verifier(
    *,
    instance_id: str,
    patch: str,
    image: str,
    test_command: str,
    timeout_sec: int,
    has_test_patch: bool = False,
) -> SwegymVerifyResult:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="swegym_verify_") as tmp:
        patch_path = Path(tmp) / "agent.patch"
        patch_path.write_text(patch, encoding="utf-8")
        container_cmd = (
            "set -e; "
            "if [ -d /testbed ]; then cd /testbed; "
            "elif [ -d /workspace ]; then cd /workspace; "
            "else cd /; fi; "
            "(git apply /tmp/agent.patch || patch -p1 < /tmp/agent.patch); "
            f"{test_command}"
        )
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{patch_path}:/tmp/agent.patch:ro",
            image,
            "/bin/bash",
            "-lc",
            container_cmd,
        ]
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = decode_timeout_output(exc.stdout)
            stderr = decode_timeout_output(exc.stderr)
            return SwegymVerifyResult(
                instance_id=instance_id,
                benchmark_resolved=False,
                verifier_status="timeout",
                patch_present=True,
                image=image,
                test_command=test_command,
                output=stdout + stderr,
                elapsed_sec=round(time.monotonic() - started, 3),
            )

    output = (completed.stdout or "") + (completed.stderr or "")
    infra_status = classify_infra_failure(completed.returncode, output)
    if infra_status:
        return SwegymVerifyResult(
            instance_id=instance_id,
            benchmark_resolved=None,
            verifier_status=infra_status,
            patch_present=True,
            image=image,
            test_command=test_command,
            returncode=completed.returncode,
            output=output[-12000:],
            elapsed_sec=round(time.monotonic() - started, 3),
            metadata={"has_test_patch": has_test_patch},
        )

    return SwegymVerifyResult(
        instance_id=instance_id,
        benchmark_resolved=completed.returncode == 0,
        verifier_status="resolved" if completed.returncode == 0 else "unresolved",
        patch_present=True,
        image=image,
        test_command=test_command,
        returncode=completed.returncode,
        output=output[-12000:],
        elapsed_sec=round(time.monotonic() - started, 3),
        metadata={"has_test_patch": has_test_patch},
    )


def classify_infra_failure(returncode: int, output: str) -> str | None:
    if returncode == 125:
        lowered = output.lower()
        image_pull_markers = (
            "unable to find image",
            "failed to resolve reference",
            "failed to copy",
            "could not fetch content descriptor",
            "manifest unknown",
            "pull access denied",
            "repository does not exist",
            "requested access to the resource is denied",
        )
        if any(marker in lowered for marker in image_pull_markers):
            return "blocked_image_pull_failed"
        return "blocked_docker_run_failed"
    return None


def load_tasks_by_instance_id(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    task_path = Path(path)
    if task_path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading parquet task files requires pandas and pyarrow.") from exc
        rows = pd.read_parquet(task_path).to_dict(orient="records")
    else:
        rows = read_jsonl(task_path)
    for row in rows:
        row.setdefault("_task_file_dir", str(task_path.resolve().parent))
    return {str(row.get("instance_id")): row for row in rows if row.get("instance_id") is not None}


def find_first_string(data: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def infer_swegym_eval_image(instance_id: str) -> str | None:
    """Infer the common SWE-Gym/SWE-Bench eval image name from an instance id."""
    if "__" not in instance_id:
        return None
    left, right = instance_id.split("__", 1)
    if not left or not right or "-" not in right:
        return None
    image_slug = f"{left}_s_{right}"
    return f"{DEFAULT_EVAL_IMAGE_REGISTRY}/sweb.eval.x86_64.{image_slug}:latest"


def find_test_command(task: dict[str, Any]) -> str | None:
    direct = find_first_string(task, COMMAND_FIELDS)
    if direct:
        return direct
    fail_to_pass = task.get("FAIL_TO_PASS") or task.get("fail_to_pass")
    if isinstance(fail_to_pass, str) and fail_to_pass.strip():
        parsed = parse_test_list_string(fail_to_pass)
        if parsed:
            return f"python -m pytest {' '.join(shlex.quote(item) for item in parsed)}"
        return f"python -m pytest {fail_to_pass}"
    fail_to_pass_items = as_list(fail_to_pass)
    if fail_to_pass_items:
        tests = " ".join(shlex.quote(str(item)) for item in fail_to_pass_items)
        return f"python -m pytest {tests}"
    return None


def parse_test_list_string(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        import re

        matches = re.findall(r"['\"]([^'\"]+)['\"]", text)
        if matches:
            return matches
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed if str(item).strip()]
    return []


def decode_timeout_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return []


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
