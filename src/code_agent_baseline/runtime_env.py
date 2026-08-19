from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_runtime_env(project_root: str | Path | None = None, *, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment that prefers the project-local virtualenv.

    This keeps agent subprocesses on the same Python/package set as the
    launcher even when the user starts `.venv/bin/codeagent` without first
    sourcing the activation script.
    """

    root = Path(project_root).resolve() if project_root else PROJECT_ROOT
    env = dict(base_env or os.environ)
    venv = root / ".venv"
    venv_bin = venv / "bin"

    if venv_bin.exists():
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"

    src = root / "src"
    leetcode = root / "benchmarks" / "leetcode_top10"
    pythonpath_parts = [str(src), str(leetcode)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    env.setdefault("MSWEA_GLOBAL_CONFIG_DIR", str(root / ".mswea"))
    env.setdefault("MSWEA_SILENT_STARTUP", "1")
    return env


def task_runtime_env(
    workspace: str | Path,
    *,
    venv_path: str | Path | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the isolated environment inherited by every task tool call."""

    root = Path(workspace).resolve()
    env = dict(base_env or os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    if venv_path:
        venv = Path(venv_path).expanduser().resolve()
        python = venv if venv.is_file() else venv / "bin" / "python"
        if not python.is_file():
            windows_python = venv / "Scripts" / "python.exe"
            python = windows_python if windows_python.is_file() else python
        if not python.is_file():
            raise FileNotFoundError(f"Python executable not found in task environment: {venv}")
        venv_root = python.parent.parent
        env["VIRTUAL_ENV"] = str(venv_root)
        env["PATH"] = f"{python.parent}{os.pathsep}{env.get('PATH', '')}"

    python_paths = [str(root)]
    if (root / "src").is_dir():
        python_paths.insert(0, str(root / "src"))
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def runtime_environment_info(env: dict[str, str], *, timeout_sec: int = 15) -> dict[str, Any]:
    """Return a small preflight record without exposing environment secrets."""

    try:
        completed = subprocess.run(
            ["python", "--version"],
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ready": False,
            "status": "infrastructure_blocked",
            "reason": "python_preflight_failed",
            "output": repr(exc),
            "venv_path": env.get("VIRTUAL_ENV"),
        }

    output = (completed.stdout or completed.stderr).strip()
    return {
        "ready": completed.returncode == 0,
        "status": "ready" if completed.returncode == 0 else "infrastructure_blocked",
        "reason": None if completed.returncode == 0 else "python_preflight_failed",
        "python_version": output,
        "python_executable": str(Path(env.get("VIRTUAL_ENV", "")) / "bin" / "python")
        if env.get("VIRTUAL_ENV")
        else "python",
        "venv_path": env.get("VIRTUAL_ENV"),
        "pythonpath": env.get("PYTHONPATH", ""),
    }
