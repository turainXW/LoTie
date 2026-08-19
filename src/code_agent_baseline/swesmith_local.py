from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INSTALL_COMMANDS = ["python -m pip install -e ."]
PROFILE_INSTALL_COMMANDS = {
    "swesmith/agronholm__typeguard.b6a7e438": [
        "python -m pip install -e '.[test,doc]'",
        "python -m pip install 'mypy==1.13.0'",
    ],
    "swesmith/aio-libs__async-timeout.d0baa9f1": ["python -m pip install -r requirements.txt"],
    "swesmith/bottlepy__bottle.a8dfef30": ["python -m pip install -e .", "python -m pip install jinja2"],
    "swesmith/cknd__stackprinter.219fcc52": ["python -m pip install -e .", "python -m pip install numpy"],
    "swesmith/cloudpipe__cloudpickle.6220b0ce": [
        "python -m pip install 'flit_core<4'",
        "python -m pip install --no-build-isolation -e .",
    ],
    "swesmith/jd__tenacity.0d40e76f": ["python -m pip install -e '.[test]'"],
    "swesmith/joke2k__faker.8b401a7d": [
        "python -m pip install -e .",
        "python -m pip install freezegun ukpostcodeparser validators Pillow xmltodict",
    ],
    "swesmith/pudo__dataset.5c2dc8d3": ["python -m pip install -e ."],
    "swesmith/r1chardj0n3s__parse.30da9e4f": ["python -m pip install -r tests/requirements.txt"],
    "swesmith/seatgeek__thefuzz.8a05a3ee": ["python -m pip install -e .", "python -m pip install pycodestyle"],
    "swesmith/theskumar__python-dotenv.2b8635b7": ["python -m pip install -r requirements.txt", "python -m pip install -e ."],
    "swesmith/tobymao__sqlglot.036601ba": ["python -m pip install -e '.[dev]'"],
}


@dataclass
class RepoSetupResult:
    repo: str
    split: str
    source: str
    commit: str
    repo_path: str
    venv_path: str
    status: str
    elapsed_sec: float
    repo_bytes: int = 0
    venv_bytes: int = 0
    output: str = ""


@dataclass
class GoldSanityResult:
    instance_id: str
    split: str
    repo: str
    status: str
    gold_sanity_passed: bool
    elapsed_sec: float
    base_returncode: int | None = None
    base_p2p_returncode: int | None = None
    bug_returncode: int | None = None
    p2p_returncode: int | None = None
    gold_returncode: int | None = None
    gold_restored_clean_tree: bool = False
    fail_to_pass_count: int = 0
    pass_to_pass_count: int = 0
    pass_to_pass_checked: int = 0
    output: dict[str, str] = field(default_factory=dict)


def load_source_rows(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise RuntimeError("Reading SWE-smith parquet requires pandas and pyarrow.") from exc
            source_rows = pd.read_parquet(path).to_dict(orient="records")
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("rows"), list):
                source_rows = [item.get("row", item) for item in data["rows"]]
            elif isinstance(data, list):
                source_rows = data
            else:
                source_rows = [data]
        for row in source_rows:
            if isinstance(row, dict) and row.get("instance_id"):
                rows[str(row["instance_id"])] = json_safe(row)
    return rows


def materialize_selected_tasks(
    selection_path: str | Path,
    source_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    cache_root: str | Path,
) -> dict[str, Any]:
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    source_rows = load_source_rows(source_paths)
    cache = Path(cache_root).resolve()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    missing: list[str] = []
    for split in ("train", "validation", "evaluation"):
        config = selection["splits"][split]["swesmith_py"]
        profiles = config["profiles"]
        for instance_id in config["instance_ids"]:
            row = source_rows.get(str(instance_id))
            if row is None:
                missing.append(str(instance_id))
                continue
            benchmark_repo = str(row["repo"])
            profile = profiles.get(benchmark_repo)
            if profile is None:
                raise KeyError(f"Missing profile for {benchmark_repo} ({instance_id})")
            source_url = str(profile["source"])
            source_repo = github_repo_name(source_url)
            repo_key = cache_key(benchmark_repo)
            fail_to_pass = normalized_list(row.get("FAIL_TO_PASS"))
            pass_to_pass = normalized_list(row.get("PASS_TO_PASS"))
            test_nodes = fail_to_pass + pass_to_pass
            task = {
                **row,
                "repo": source_repo,
                "benchmark_repo": benchmark_repo,
                "source_repo": source_url,
                "base_commit": str(profile["commit"]),
                "bug_patch": str(row.get("patch") or ""),
                "patch": "",
                "split": split,
                "python_version": str(profile.get("python") or "3.10"),
                "install_commands": PROFILE_INSTALL_COMMANDS.get(benchmark_repo, DEFAULT_INSTALL_COMMANDS),
                "test_command": pytest_command(test_nodes),
                "local_repo_path": str(cache / "repos" / repo_key),
                "local_venv_path": str(cache / "venvs" / repo_key),
                "gold_policy": "reverse_bug_patch_and_test",
                "gold_visible_to_agent": False,
                "metadata": {
                    "dataset": "SWE-bench/SWE-smith",
                    "split": split,
                    "benchmark_repo": benchmark_repo,
                    "gold_patch_sha256": reverse_patch_sha256(str(row.get("patch") or "")),
                    "official_comparable": False,
                },
            }
            tasks.append(task)

    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    manifest = {
        "format": "lottie_swesmith_local_tasks_v1",
        "selection": str(Path(selection_path).resolve()),
        "sources": [str(Path(path).resolve()) for path in source_paths],
        "output": str(output.resolve()),
        "cache_root": str(cache),
        "tasks": len(tasks),
        "missing": missing,
        "split_counts": {split: sum(task["split"] == split for task in tasks) for split in ("train", "validation", "evaluation")},
        "repo_count": len({task["benchmark_repo"] for task in tasks}),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def setup_repositories(
    tasks_path: str | Path,
    *,
    python_executable: str | Path,
    timeout_sec: int = 1800,
    repo_filter: set[str] | None = None,
) -> list[RepoSetupResult]:
    tasks = read_jsonl(tasks_path)
    by_repo: dict[str, dict[str, Any]] = {}
    for task in tasks:
        by_repo.setdefault(str(task["benchmark_repo"]), task)

    results: list[RepoSetupResult] = []
    for benchmark_repo, task in by_repo.items():
        if repo_filter and benchmark_repo not in repo_filter:
            continue
        started = time.monotonic()
        repo_path = Path(task["local_repo_path"])
        venv_path = Path(task["local_venv_path"])
        outputs: list[str] = []
        status = "ready"
        try:
            ensure_repo_checkout(
                str(task["source_repo"]),
                str(task["base_commit"]),
                repo_path,
                timeout_sec=timeout_sec,
            )
            ensure_repo_venv(
                repo_path,
                venv_path,
                python_executable=Path(python_executable),
                install_commands=normalized_list(task.get("install_commands")),
                timeout_sec=timeout_sec,
                outputs=outputs,
            )
        except Exception as exc:
            status = "blocked_setup_failed"
            outputs.append(repr(exc))
        results.append(
            RepoSetupResult(
                repo=benchmark_repo,
                split=str(task["split"]),
                source=str(task["source_repo"]),
                commit=str(task["base_commit"]),
                repo_path=str(repo_path),
                venv_path=str(venv_path),
                status=status,
                elapsed_sec=round(time.monotonic() - started, 3),
                repo_bytes=directory_size(repo_path),
                venv_bytes=directory_size(venv_path),
                output="\n".join(outputs)[-12000:],
            )
        )
    return results


def run_gold_sanity(
    tasks_path: str | Path,
    output_path: str | Path,
    *,
    split: str | None = None,
    limit: int | None = None,
    p2p_limit: int = 10,
    timeout_sec: int = 600,
) -> list[GoldSanityResult]:
    tasks = [task for task in read_jsonl(tasks_path) if split is None or task.get("split") == split]
    if limit is not None:
        tasks = tasks[:limit]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    results: list[GoldSanityResult] = []
    for task in tasks:
        result = sanity_one_task(task, p2p_limit=p2p_limit, timeout_sec=timeout_sec)
        results.append(result)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    return results


def build_runnable_tasks(
    tasks_path: str | Path,
    sanity_results_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    tasks = read_jsonl(tasks_path)
    results = {str(item["instance_id"]): item for item in read_jsonl(sanity_results_path)}
    runnable = [task for task in tasks if bool(results.get(str(task["instance_id"]), {}).get("gold_sanity_passed"))]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in runnable:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    manifest = {
        "format": "lottie_swesmith_runnable_v1",
        "source_tasks": str(Path(tasks_path).resolve()),
        "gold_sanity_results": str(Path(sanity_results_path).resolve()),
        "output": str(output.resolve()),
        "tasks_in": len(tasks),
        "tasks_out": len(runnable),
        "excluded": len(tasks) - len(runnable),
        "split_counts": {
            split: sum(task.get("split") == split for task in runnable)
            for split in ("train", "validation", "evaluation")
        },
        "repo_count": len({task["benchmark_repo"] for task in runnable}),
        "verification_scope": {
            "fail_to_pass": "all listed tests",
            "pass_to_pass": "first 10 listed tests",
            "gold": "reverse bug patch, clean-tree equality, and all FAIL_TO_PASS tests",
            "official_comparable": False,
        },
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def sanity_one_task(task: dict[str, Any], *, p2p_limit: int, timeout_sec: int) -> GoldSanityResult:
    started = time.monotonic()
    instance_id = str(task["instance_id"])
    fail_to_pass = normalized_list(task.get("FAIL_TO_PASS"))
    pass_to_pass = normalized_list(task.get("PASS_TO_PASS"))
    checked_p2p = pass_to_pass[: max(0, p2p_limit)]
    repo_path = Path(task["local_repo_path"])
    venv_path = Path(task["local_venv_path"])
    common = {
        "instance_id": instance_id,
        "split": str(task["split"]),
        "repo": str(task["benchmark_repo"]),
        "elapsed_sec": 0.0,
        "fail_to_pass_count": len(fail_to_pass),
        "pass_to_pass_count": len(pass_to_pass),
        "pass_to_pass_checked": len(checked_p2p),
    }
    if not repo_path.is_dir() or not venv_python(venv_path).is_file():
        common["elapsed_sec"] = round(time.monotonic() - started, 3)
        return GoldSanityResult(
            **common,
            status="blocked_environment_missing",
            gold_sanity_passed=False,
        )

    with tempfile.TemporaryDirectory(prefix="lottie_gold_sanity_") as temp_dir:
        temp_root = Path(temp_dir)
        base_workspace = copy_sanity_workspace(repo_path, temp_root / "base")
        outputs: dict[str, str] = {}

        base = run_pytest(base_workspace, venv_path, fail_to_pass, timeout_sec)
        outputs["base"] = command_output(base)
        if base.returncode != 0:
            return finish_sanity(common, started, "blocked_base_tests_failed", False, base=base, outputs=outputs)

        base_p2p = run_pytest(base_workspace, venv_path, checked_p2p, timeout_sec) if checked_p2p else None
        if base_p2p is not None:
            outputs["base_p2p"] = command_output(base_p2p)
            if base_p2p.returncode != 0:
                return finish_sanity(
                    common,
                    started,
                    "blocked_base_pass_to_pass_failed",
                    False,
                    base=base,
                    base_p2p=base_p2p,
                    outputs=outputs,
                )

        bug_patch = temp_root / "bug.patch"
        bug_patch.write_text(str(task.get("bug_patch") or ""), encoding="utf-8")
        bug_workspace = copy_sanity_workspace(repo_path, temp_root / "bug")
        apply_bug = run_command(["git", "apply", "--whitespace=nowarn", str(bug_patch)], bug_workspace, timeout_sec)
        outputs["apply_bug"] = command_output(apply_bug)
        if apply_bug.returncode != 0:
            return finish_sanity(
                common,
                started,
                "blocked_bug_patch_apply_failed",
                False,
                base=base,
                base_p2p=base_p2p,
                outputs=outputs,
            )

        bug = run_pytest(bug_workspace, venv_path, fail_to_pass, timeout_sec)
        outputs["bug"] = command_output(bug)
        if bug.returncode != 1:
            status = "invalid_bug_did_not_fail" if bug.returncode == 0 else "blocked_bug_test_execution"
            return finish_sanity(
                common,
                started,
                status,
                False,
                base=base,
                base_p2p=base_p2p,
                bug=bug,
                outputs=outputs,
            )

        p2p = run_pytest(bug_workspace, venv_path, checked_p2p, timeout_sec) if checked_p2p else None
        if p2p is not None:
            outputs["p2p"] = command_output(p2p)
            if p2p.returncode != 0:
                return finish_sanity(
                    common,
                    started,
                    "invalid_pass_to_pass_regression",
                    False,
                    base=base,
                    base_p2p=base_p2p,
                    bug=bug,
                    p2p=p2p,
                    outputs=outputs,
                )

        gold_workspace = copy_sanity_workspace(repo_path, temp_root / "gold")
        gold_bug = run_command(["git", "apply", "--whitespace=nowarn", str(bug_patch)], gold_workspace, timeout_sec)
        outputs["apply_bug_for_gold"] = command_output(gold_bug)
        if gold_bug.returncode != 0:
            return finish_sanity(
                common,
                started,
                "blocked_gold_setup_failed",
                False,
                base=base,
                base_p2p=base_p2p,
                bug=bug,
                p2p=p2p,
                outputs=outputs,
            )
        reverse = run_command(
            ["git", "apply", "--reverse", "--whitespace=nowarn", str(bug_patch)],
            gold_workspace,
            timeout_sec,
        )
        outputs["apply_gold"] = command_output(reverse)
        if reverse.returncode != 0:
            return finish_sanity(
                common,
                started,
                "blocked_gold_patch_apply_failed",
                False,
                base=base,
                base_p2p=base_p2p,
                bug=bug,
                p2p=p2p,
                outputs=outputs,
            )

        diff = run_command(["git", "diff", "--exit-code", "--"], gold_workspace, timeout_sec)
        clean = diff.returncode == 0
        outputs["gold_diff"] = command_output(diff)
        if not clean:
            return finish_sanity(
                common,
                started,
                "invalid_gold_not_clean",
                False,
                base=base,
                base_p2p=base_p2p,
                bug=bug,
                p2p=p2p,
                clean=False,
                outputs=outputs,
            )

        gold = run_pytest(gold_workspace, venv_path, fail_to_pass, timeout_sec)
        outputs["gold"] = command_output(gold)
        passed = gold.returncode == 0
        return finish_sanity(
            common,
            started,
            "gold_sanity_passed" if passed else "invalid_gold_tests_failed",
            passed,
            base=base,
            base_p2p=base_p2p,
            bug=bug,
            p2p=p2p,
            gold=gold,
            clean=clean,
            outputs=outputs,
        )


def copy_sanity_workspace(repo_path: Path, dest: Path) -> Path:
    shutil.copytree(
        repo_path,
        dest,
        ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".coverage"),
    )
    restore = subprocess.run(
        ["git", "checkout", "--", "."],
        cwd=dest,
        text=True,
        capture_output=True,
        timeout=60,
    )
    if restore.returncode != 0:
        raise RuntimeError(restore.stderr.strip() or restore.stdout.strip() or f"Failed to restore {dest}")
    return dest


def finish_sanity(
    common: dict[str, Any],
    started: float,
    status: str,
    passed: bool,
    *,
    base: subprocess.CompletedProcess[str] | None = None,
    base_p2p: subprocess.CompletedProcess[str] | None = None,
    bug: subprocess.CompletedProcess[str] | None = None,
    p2p: subprocess.CompletedProcess[str] | None = None,
    gold: subprocess.CompletedProcess[str] | None = None,
    clean: bool = False,
    outputs: dict[str, str] | None = None,
) -> GoldSanityResult:
    values = dict(common)
    values["elapsed_sec"] = round(time.monotonic() - started, 3)
    return GoldSanityResult(
        **values,
        status=status,
        gold_sanity_passed=passed,
        base_returncode=base.returncode if base else None,
        base_p2p_returncode=base_p2p.returncode if base_p2p else None,
        bug_returncode=bug.returncode if bug else None,
        p2p_returncode=p2p.returncode if p2p else None,
        gold_returncode=gold.returncode if gold else None,
        gold_restored_clean_tree=clean,
        output=outputs or {},
    )


def ensure_repo_checkout(
    source: str,
    commit: str,
    dest: Path,
    *,
    timeout_sec: int,
    attempts: int = 3,
) -> None:
    marker = dest / ".lottie-checkout.json"
    if marker.is_file():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("commit") == commit:
            return
    errors: list[str] = []
    command_timeout = min(timeout_sec, 600)
    try:
        ensure_repo_tarball_checkout(source, commit, dest, timeout_sec=command_timeout)
        marker.write_text(
            json.dumps({"source": source, "commit": commit, "fetch": "github-tarball"}, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    except Exception as exc:
        errors.append(f"tarball attempt: {exc!r}")
    for attempt in range(1, max(1, attempts) + 1):
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        commands = [
            ["git", "init", "--quiet", str(dest)],
            ["git", "-C", str(dest), "remote", "add", "origin", source],
            [
                "git",
                "-c",
                "http.lowSpeedLimit=1024",
                "-c",
                "http.lowSpeedTime=120",
                "-C",
                str(dest),
                "fetch",
                "--no-tags",
                "--depth=1",
                "origin",
                commit,
            ],
            ["git", "-C", str(dest), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        ]
        failed = ""
        for command in commands:
            try:
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    timeout=command_timeout,
                )
            except subprocess.TimeoutExpired:
                failed = f"timeout after {command_timeout}s: {' '.join(command)}"
                break
            if completed.returncode != 0:
                failed = completed.stderr.strip() or completed.stdout.strip() or f"failed: {' '.join(command)}"
                break
        if not failed:
            marker.write_text(
                json.dumps({"source": source, "commit": commit, "fetch": "depth-1"}, indent=2) + "\n",
                encoding="utf-8",
            )
            return
        errors.append(f"attempt {attempt}/{attempts}: {failed}")
        if attempt < attempts:
            time.sleep(min(20, 2**attempt))
    raise RuntimeError("repository fetch failed:\n" + "\n".join(errors))


def ensure_repo_tarball_checkout(source: str, commit: str, dest: Path, *, timeout_sec: int) -> None:
    repo = github_repo_name(source)
    if "/" not in repo:
        raise ValueError(f"GitHub repository URL required for tarball fallback: {source}")
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lottie_repo_tarball_", dir=dest.parent) as temp_dir:
        temp_root = Path(temp_dir)
        archive = temp_root / "repo.tar.gz"
        download_file(url, archive, timeout_sec=timeout_sec)
        extracted = temp_root / "extracted"
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as handle:
            safe_extract_tar(handle, extracted)
        roots = [path for path in extracted.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"Expected one repository root in {url}, found {len(roots)}")
        shutil.move(str(roots[0]), str(dest))

    commands = [
        ["git", "init", "--quiet", str(dest)],
        ["git", "-C", str(dest), "add", "-A"],
        [
            "git",
            "-C",
            str(dest),
            "-c",
            "user.email=lottie@local",
            "-c",
            "user.name=Lottie Eval90",
            "commit",
            "--quiet",
            "-m",
            f"upstream snapshot {commit}",
        ],
    ]
    for command in commands:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout_sec)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"failed: {' '.join(command)}")


def download_file(url: str, output: Path, *, timeout_sec: int) -> None:
    curl = shutil.which("curl")
    if curl:
        completed = subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--retry",
                "3",
                "--retry-all-errors",
                "--connect-timeout",
                "20",
                "--max-time",
                str(timeout_sec),
                "--user-agent",
                "lottie-eval90-deployer",
                "--output",
                str(output),
                url,
            ],
            text=True,
            capture_output=True,
            timeout=timeout_sec + 30,
        )
        if completed.returncode == 0:
            return
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"curl failed for {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "lottie-eval90-deployer"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        output.write_bytes(response.read())


def safe_extract_tar(handle: tarfile.TarFile, dest: Path) -> None:
    root = dest.resolve()
    for member in handle.getmembers():
        target = (dest / member.name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Unsafe tar member path: {member.name}") from exc
    handle.extractall(dest)


def ensure_repo_venv(
    repo_path: Path,
    venv_path: Path,
    *,
    python_executable: Path,
    install_commands: list[str],
    timeout_sec: int,
    outputs: list[str],
) -> None:
    if not python_executable.is_file():
        raise FileNotFoundError(f"Python executable not found: {python_executable}")
    install_commands = [*install_commands, *discovered_test_dependency_commands(repo_path)]
    fingerprint = dependency_fingerprint(repo_path, python_executable, install_commands)
    marker = venv_path / ".lottie-ready.json"
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("fingerprint") == fingerprint:
        return
    if venv_path.exists():
        shutil.rmtree(venv_path)
    venv_path.parent.mkdir(parents=True, exist_ok=True)
    create = subprocess.run(
        [str(python_executable), "-m", "venv", str(venv_path)],
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    outputs.append(command_output(create))
    if create.returncode != 0:
        raise RuntimeError("venv creation failed")
    env = venv_environment(venv_path, repo_path)
    commands = ["python -m pip install --upgrade pip setuptools wheel pytest", *install_commands]
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=repo_path,
            shell=True,
            executable="/bin/bash",
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            env=env,
        )
        outputs.append(f"$ {command}\n{command_output(completed)}")
        if completed.returncode != 0:
            raise RuntimeError(f"dependency command failed: {command}")
    marker.write_text(json.dumps({"fingerprint": fingerprint}, indent=2) + "\n", encoding="utf-8")


def discovered_test_dependency_commands(repo_path: Path) -> list[str]:
    candidates = (
        "test_requirements.txt",
        "test-requirements.txt",
        "requirements-test.txt",
        "requirements/tests.txt",
        "requirements/test.txt",
        "requirements-dev.txt",
        "dev-requirements.txt",
    )
    return [f"python -m pip install -r {shlex.quote(name)}" for name in candidates if (repo_path / name).is_file()]


def run_pytest(repo: Path, venv_path: Path, nodes: list[str], timeout_sec: int) -> subprocess.CompletedProcess[str]:
    if not nodes:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="no tests selected\n", stderr="")
    command = [str(venv_python(venv_path)), "-m", "pytest", "-q", "--disable-warnings", "--tb=short", *nodes]
    return run_command(command, repo, timeout_sec, env=venv_environment(venv_path, repo))


def run_command(
    command: list[str],
    cwd: Path,
    timeout_sec: int,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout_sec, env=env)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
        return subprocess.CompletedProcess(args=command, returncode=124, stdout=stdout, stderr=stderr + "\nTIMEOUT")


def venv_environment(venv_path: Path, repo_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_path)
    env["PATH"] = str(venv_python(venv_path).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    python_paths = [str(repo_path)]
    if (repo_path / "src").is_dir():
        python_paths.insert(0, str(repo_path / "src"))
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def venv_python(venv_path: Path) -> Path:
    return venv_path / "bin" / "python"


def dependency_fingerprint(repo_path: Path, python_executable: Path, commands: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(str(python_executable).encode())
    digest.update(json.dumps(commands, sort_keys=True).encode())
    for name in ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "requirements-dev.txt"):
        path = repo_path / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def reverse_patch_sha256(patch: str) -> str:
    return hashlib.sha256(("reverse\0" + patch).encode()).hexdigest()


def pytest_command(nodes: list[str]) -> str:
    return "python -m pytest -q --disable-warnings --tb=short " + " ".join(shlex.quote(node) for node in nodes)


def normalized_list(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        value = decoded
    return [str(item) for item in value]


def github_repo_name(url: str) -> str:
    value = url.rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    marker = "github.com/"
    return value.split(marker, 1)[1] if marker in value else value


def cache_key(repo: str) -> str:
    return repo.replace("/", "__").replace(".", "_")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def command_output(result: subprocess.CompletedProcess[str]) -> str:
    return ((result.stdout or "") + (result.stderr or ""))[-12000:]


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return value
