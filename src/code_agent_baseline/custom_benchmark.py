from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import textwrap
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
}


@dataclass
class BenchmarkProjectResult:
    project_dir: Path
    task_path: Path
    tasks_jsonl_path: Path
    dockerfile_path: Path
    verifier_path: Path
    build_script_path: Path
    gold_record_path: Path | None
    generated_test_path: Path | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "project_dir": str(self.project_dir),
            "task_path": str(self.task_path),
            "tasks_jsonl_path": str(self.tasks_jsonl_path),
            "dockerfile_path": str(self.dockerfile_path),
            "verifier_path": str(self.verifier_path),
            "build_script_path": str(self.build_script_path),
            "gold_record_path": str(self.gold_record_path) if self.gold_record_path else None,
            "generated_test_path": str(self.generated_test_path) if self.generated_test_path else None,
        }


def create_benchmark_project(
    *,
    name: str,
    repo: str | Path,
    output_dir: str | Path,
    problem_statement: str,
    pytest_nodes: list[str],
    test_patch_path: str | Path | None = None,
    gold_patch_path: str | Path | None = None,
    image_name: str | None = None,
    python_version: str = "3.11",
    include_git: bool = False,
    generate_tests: bool = False,
    generated_test_file: str = "tests/test_generated_bug.py",
    model_url: str = "https://api.deepseek.com/chat/completions",
    model: str = "deepseek-chat",
    api_key_env: str = "DEEPSEEK_API_KEY",
    max_tokens: int = 3000,
) -> BenchmarkProjectResult:
    repo_path = Path(repo).resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        raise ValueError(f"Repository directory does not exist: {repo_path}")
    if not problem_statement.strip():
        raise ValueError("problem_statement is required")
    if not pytest_nodes and not generate_tests:
        raise ValueError("At least one --pytest-node is required unless --generate-tests is used.")

    slug = slugify(name)
    project_dir = Path(output_dir).resolve() / slug
    if project_dir.exists():
        raise FileExistsError(f"Project directory already exists: {project_dir}")

    image = image_name or f"local/codeagent.{slug}:latest"
    instance_id = f"custom__{slug}"
    project_dir.mkdir(parents=True)
    repo_out = project_dir / "repo"
    patches_dir = project_dir / "patches"
    scripts_dir = project_dir / "scripts"
    skills_dir = project_dir / "skills"
    patches_dir.mkdir()
    scripts_dir.mkdir()
    skills_dir.mkdir()

    copy_repo(repo_path, repo_out, include_git=include_git)

    generated_test_path: Path | None = None
    test_patch = ""
    if test_patch_path:
        test_patch = Path(test_patch_path).read_text(encoding="utf-8")
        (patches_dir / "test.patch").write_text(test_patch, encoding="utf-8")
    elif generate_tests:
        generated = generate_pytest_with_model(
            repo_path=repo_path,
            problem_statement=problem_statement,
            generated_test_file=generated_test_file,
            model_url=model_url,
            model=model,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
        )
        generated_test_path = patches_dir / generated_test_file
        generated_test_path.parent.mkdir(parents=True, exist_ok=True)
        generated_test_path.write_text(generated, encoding="utf-8")
        test_patch = build_new_file_patch(generated_test_file, generated)
        (patches_dir / "test.patch").write_text(test_patch, encoding="utf-8")
        if not pytest_nodes:
            pytest_nodes = [generated_test_file]
    else:
        (patches_dir / "test.patch").write_text("", encoding="utf-8")

    gold_patch = ""
    gold_record_path: Path | None = None
    if gold_patch_path:
        gold_patch = Path(gold_patch_path).read_text(encoding="utf-8")
        (patches_dir / "gold.patch").write_text(gold_patch, encoding="utf-8")
        gold_record_path = project_dir / "records.gold.jsonl"
        write_jsonl(
            gold_record_path,
            [
                {
                    "format": "custom_benchmark_patch_v1",
                    "instance_id": instance_id,
                    "patch": gold_patch,
                    "resolved": None,
                    "source": "gold_patch",
                }
            ],
        )
    else:
        (patches_dir / "gold.patch").write_text("", encoding="utf-8")

    fail_to_pass = pytest_nodes or [generated_test_file]
    base_commit = get_git_commit(repo_path)
    task = {
        "instance_id": instance_id,
        "repo": f"local/{slug}",
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "image": image,
        "docker_image": image,
        "FAIL_TO_PASS": fail_to_pass,
        "PASS_TO_PASS": [],
        "test_patch": test_patch,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": {
            "source_repo": str(repo_path),
            "local_repo_path": "repo",
            "project_format": "codeagent_custom_benchmark_v1",
            "generated_tests": generate_tests,
        },
    }
    task_path = project_dir / "task.json"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tasks_jsonl_path = project_dir / "tasks.jsonl"
    write_jsonl(tasks_jsonl_path, [task])

    dockerfile_path = project_dir / "Dockerfile"
    dockerfile_path.write_text(render_dockerfile(python_version=python_version), encoding="utf-8")
    verifier_path = project_dir / "verifier.sh"
    verifier_path.write_text(render_verifier_script(pytest_nodes=fail_to_pass), encoding="utf-8")
    verifier_path.chmod(0o755)
    build_script_path = project_dir / "build_image.sh"
    build_script_path.write_text(render_build_script(image=image), encoding="utf-8")
    build_script_path.chmod(0o755)

    (skills_dir / "pytest_generation_skill.md").write_text(render_pytest_generation_skill(), encoding="utf-8")
    (project_dir / "README.md").write_text(
        render_project_readme(slug=slug, image=image, pytest_nodes=fail_to_pass, has_gold=gold_record_path is not None),
        encoding="utf-8",
    )
    (project_dir / "generator_meta.json").write_text(
        json.dumps(
            {
                "format": "codeagent_custom_benchmark_project_v1",
                "name": name,
                "slug": slug,
                "source_repo": str(repo_path),
                "image": image,
                "generated_tests": generate_tests,
                "created_at": task["created_at"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return BenchmarkProjectResult(
        project_dir=project_dir,
        task_path=task_path,
        tasks_jsonl_path=tasks_jsonl_path,
        dockerfile_path=dockerfile_path,
        verifier_path=verifier_path,
        build_script_path=build_script_path,
        gold_record_path=gold_record_path,
        generated_test_path=generated_test_path,
    )


def copy_repo(source: Path, target: Path, *, include_git: bool) -> None:
    ignore_dirs = set(DEFAULT_IGNORE_DIRS)
    if include_git:
        ignore_dirs.discard(".git")

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = set()
        for name in names:
            if name in ignore_dirs:
                ignored.add(name)
            elif name.endswith((".pyc", ".pyo")):
                ignored.add(name)
        return ignored

    shutil.copytree(source, target, ignore=ignore)


def generate_pytest_with_model(
    *,
    repo_path: Path,
    problem_statement: str,
    generated_test_file: str,
    model_url: str,
    model: str,
    api_key_env: str,
    max_tokens: int,
) -> str:
    api_key = os.environ.get(api_key_env, "")
    if api_key_env and not api_key:
        raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
    prompt = render_test_generation_prompt(
        repo_path=repo_path,
        problem_statement=problem_statement,
        generated_test_file=generated_test_file,
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate minimal pytest tests for code-agent benchmarks. "
                    "Return only Python code, no markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(model_url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode())
    content = data["choices"][0]["message"].get("content") or ""
    return strip_code_fence(content).strip() + "\n"


def render_test_generation_prompt(*, repo_path: Path, problem_statement: str, generated_test_file: str) -> str:
    return textwrap.dedent(
        f"""
        Create a minimal pytest file for this bugfix benchmark.

        Target test file path: {generated_test_file}

        Problem statement:
        {problem_statement}

        Repository context:
        {repo_context_excerpt(repo_path)}

        Requirements:
        - Return only Python code.
        - The test should fail on the buggy implementation when possible.
        - The test should pass after a correct fix.
        - Keep dependencies limited to pytest and the repository itself.
        - Prefer one or two focused tests.
        """
    ).strip()


def repo_context_excerpt(repo_path: Path, *, max_files: int = 40, max_chars: int = 12000) -> str:
    files: list[str] = []
    for path in sorted(repo_path.rglob("*")):
        if len(files) >= max_files:
            break
        if not path.is_file() or any(part in DEFAULT_IGNORE_DIRS for part in path.parts):
            continue
        rel = path.relative_to(repo_path)
        if rel.suffix in {".py", ".toml", ".cfg", ".ini", ".md"}:
            files.append(str(rel))
    chunks = ["Files:", *[f"- {item}" for item in files]]
    for rel in files[:8]:
        path = repo_path / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:1200]
        except OSError:
            continue
        chunks.append(f"\n--- {rel} ---\n{text}")
    return "\n".join(chunks)[:max_chars]


def build_new_file_patch(path: str, content: str) -> str:
    normalized = content if content.endswith("\n") else content + "\n"
    lines = normalized.splitlines()
    patch = [
        f"diff --git a/{path} b/{path}",
        "new file mode 100644",
        "index 0000000..1111111",
        "--- /dev/null",
        f"+++ b/{path}",
        f"@@ -0,0 +1,{len(lines)} @@",
    ]
    patch.extend("+" + line for line in lines)
    return "\n".join(patch) + "\n"


def render_dockerfile(*, python_version: str) -> str:
    return textwrap.dedent(
        f"""
        FROM python:{python_version}-slim

        ENV PYTHONDONTWRITEBYTECODE=1
        ENV PYTHONUNBUFFERED=1

        RUN apt-get update && apt-get install -y --no-install-recommends \\
            git patch build-essential \\
            && rm -rf /var/lib/apt/lists/*

        WORKDIR /testbed
        COPY repo/ /testbed/
        COPY verifier.sh /verifier.sh
        RUN chmod +x /verifier.sh

        RUN python -m pip install --upgrade pip pytest \\
            && if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi \\
            && if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ]; then python -m pip install -e . || true; fi

        CMD ["/bin/bash"]
        """
    ).lstrip()


def render_verifier_script(*, pytest_nodes: list[str]) -> str:
    tests = " ".join(shell_quote(item) for item in pytest_nodes)
    return textwrap.dedent(
        f"""
        #!/usr/bin/env bash
        set -euo pipefail

        PATCH_PATH="${{1:-}}"
        if [ -z "$PATCH_PATH" ]; then
          echo "usage: ./verifier.sh /path/to/agent.patch" >&2
          exit 2
        fi

        cd /testbed
        git apply "$PATCH_PATH" || patch -p1 < "$PATCH_PATH"
        if [ -s /tmp/test.patch ]; then
          git apply /tmp/test.patch || patch -p1 < /tmp/test.patch
        fi
        python -m pytest {tests}
        """
    ).lstrip()


def render_build_script(*, image: str) -> str:
    return textwrap.dedent(
        f"""
        #!/usr/bin/env bash
        set -euo pipefail
        cd "$(dirname "$0")"
        docker build -t {shell_quote(image)} .
        """
    ).lstrip()


def render_project_readme(*, slug: str, image: str, pytest_nodes: list[str], has_gold: bool) -> str:
    gold_text = (
        "Gold patch record is available at `records.gold.jsonl`.\n"
        if has_gold
        else "No gold patch was provided; use agent-generated patch records for verification.\n"
    )
    tests = "\n".join(f"- `{node}`" for node in pytest_nodes)
    return textwrap.dedent(
        f"""
        # {slug}

        This is a CodeAgent custom benchmark project. It mirrors the SWE-Gym verifier shape:

        - `task.json` / `tasks.jsonl`: problem statement, image, test patch, FAIL_TO_PASS tests.
        - `repo/`: repository snapshot mounted into `/testbed` in Docker.
        - `patches/test.patch`: pytest changes applied during verification.
        - `Dockerfile`: reproducible test image.
        - `verifier.sh`: apply agent patch, apply test patch, run pytest.

        {gold_text}
        ## Build

        ```bash
        ./build_image.sh
        ```

        ## Verify a patch manually

        ```bash
        docker run --rm \\
          -v "$PWD/patches/test.patch:/tmp/test.patch:ro" \\
          -v "$PWD/agent.patch:/tmp/agent.patch:ro" \\
          {image} \\
          /bin/bash -lc '/verifier.sh /tmp/agent.patch'
        ```

        ## Verify with CodeAgent wrapper

        ```bash
        codeagent verify-swegym-patches \\
          --records records.gold.jsonl \\
          --tasks tasks.jsonl \\
          --output verify_results.jsonl \\
          --mode docker
        ```

        ## Verify locally without Docker

        Reuse an existing virtual environment for the fastest local loop:

        ```bash
        codeagent verify-swegym-patches \
          --records records.gold.jsonl \
          --tasks tasks.jsonl \
          --output verify_results.local.jsonl \
          --mode local-venv \
          --local-venv /path/to/.venv
        ```

        Omit `--local-venv` to create and cache a verifier environment automatically. Add
        `--install-deps` when this repository needs its requirements installed. Local venv
        verification is faster but is not an official SWE-Gym/SWE-smith Docker result.

        ## FAIL_TO_PASS

        {tests}
        """
    ).strip() + "\n"


def render_pytest_generation_skill() -> str:
    return textwrap.dedent(
        """
        # Pytest Generation Skill

        Use this skill to draft tests for a custom SWE-Gym-style benchmark.

        Rules:
        - Generate the smallest pytest test that captures the bug.
        - Prefer public APIs over internal state.
        - Avoid brittle timing, network, filesystem-global, and external-service dependencies.
        - The generated test must be treated as a draft until verifier checks prove:
          1. base repo + test patch fails, and
          2. gold/correct patch + test patch passes.
        - Do not use generated tests as training reward unless the verifier result is reproducible.
        """
    ).lstrip()


def get_git_commit(repo_path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:python)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("._-").lower()
    if not slug:
        raise ValueError("name must contain at least one alphanumeric character")
    return slug


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
