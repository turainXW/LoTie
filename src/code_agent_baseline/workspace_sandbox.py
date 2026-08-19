from __future__ import annotations

import fnmatch
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_EXCLUDES = (
    ".git",
    ".mswea",
    ".codeagent/workspaces",
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "refs/open_source",
)


@dataclass
class WorkspaceSandbox:
    mode: str
    source_repo: str
    runtime_repo: str
    run_id: str
    manifest_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def prepare_workspace(
    repo: str | Path,
    *,
    sandbox: str = "copy",
    workspace_root: str | Path | None = None,
    run_id: str | None = None,
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
) -> WorkspaceSandbox:
    source = Path(repo).resolve()
    if sandbox == "direct":
        return WorkspaceSandbox(
            mode="direct",
            source_repo=str(source),
            runtime_repo=str(source),
            run_id=run_id or "direct",
        )
    if sandbox != "copy":
        raise ValueError(f"Unsupported sandbox mode: {sandbox}")

    run = run_id or time.strftime("run_%Y%m%d_%H%M%S")
    root = Path(workspace_root).resolve() if workspace_root else source / ".codeagent" / "workspaces"
    runtime = root / run / "repo"
    manifest = root / run / "sandbox_manifest.json"

    if runtime.exists():
        raise FileExistsError(f"Workspace already exists: {runtime}")
    runtime.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, runtime, ignore=_ignore_patterns(source, excludes), symlinks=False)

    workspace = WorkspaceSandbox(
        mode="copy",
        source_repo=str(source),
        runtime_repo=str(runtime),
        run_id=run,
        manifest_path=str(manifest),
    )
    manifest.write_text(json.dumps(workspace.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return workspace


def _ignore_patterns(source: Path, patterns: tuple[str, ...]):
    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set()
        dir_path = Path(directory)
        rel_dir = dir_path.relative_to(source).as_posix() if dir_path != source else ""
        for name in names:
            rel = f"{rel_dir}/{name}" if rel_dir else name
            for pattern in patterns:
                if rel == pattern or rel.startswith(f"{pattern}/") or fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
                    ignored.add(name)
                    break
        return ignored

    return ignore

