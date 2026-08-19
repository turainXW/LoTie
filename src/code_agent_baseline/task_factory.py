from __future__ import annotations

import json
from pathlib import Path

from .schemas import CodeTask


def create_toy_addition_task(output_dir: str | Path) -> CodeTask:
    root = Path(output_dir)
    repo = root / "toy_repo_0001"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "calculator.py").write_text(
        "def add(a, b):\n"
        "    return a - b\n",
        encoding="utf-8",
    )


def create_code_repair_benchmark_tasks(output_dir: str | Path) -> list[CodeTask]:
    root = Path(output_dir)
    return [
        _create_addition_repair_task(root),
        _create_slugify_repair_task(root),
        _create_mean_repair_task(root),
    ]


def _create_addition_repair_task(root: Path) -> CodeTask:
    repo = root / "code_repair_0001_addition"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "calculator.py").write_text(
        "def add(a, b):\n"
        "    return a - b\n",
        encoding="utf-8",
    )
    (repo / "test_task.py").write_text(
        "from calculator import add\n\n"
        "assert add(2, 3) == 5\n"
        "assert add(-2, -3) == -5\n"
        "assert add(-2, 3) == 1\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    return CodeTask(
        task_id="code_repair_0001_addition",
        repo_path=str(repo.resolve()),
        instruction="Fix calculator.add so all arithmetic tests pass.",
        validation_command="python3 -B test_task.py",
    )


def _create_slugify_repair_task(root: Path) -> CodeTask:
    repo = root / "code_repair_0002_slugify"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "text_utils.py").write_text(
        "def slugify(text):\n"
        "    return text.lower()\n",
        encoding="utf-8",
    )
    (repo / "test_task.py").write_text(
        "from text_utils import slugify\n\n"
        "assert slugify('Hello World') == 'hello-world'\n"
        "assert slugify('  Agentic Code Eval  ') == 'agentic-code-eval'\n"
        "assert slugify('LLM, Judge? No: tests!') == 'llm-judge-no-tests'\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    return CodeTask(
        task_id="code_repair_0002_slugify",
        repo_path=str(repo.resolve()),
        instruction="Implement text_utils.slugify to create lowercase URL slugs.",
        validation_command="python3 -B test_task.py",
    )


def _create_mean_repair_task(root: Path) -> CodeTask:
    repo = root / "code_repair_0003_mean"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "stats_utils.py").write_text(
        "def mean(values):\n"
        "    return sum(values) // len(values)\n",
        encoding="utf-8",
    )
    (repo / "test_task.py").write_text(
        "from stats_utils import mean\n\n"
        "assert mean([1, 2, 3]) == 2\n"
        "assert mean([1, 2]) == 1.5\n"
        "assert mean([-1, 1]) == 0\n"
        "try:\n"
        "    mean([])\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('mean([]) should raise ValueError')\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    return CodeTask(
        task_id="code_repair_0003_mean",
        repo_path=str(repo.resolve()),
        instruction="Fix stats_utils.mean to return a floating mean and reject empty input.",
        validation_command="python3 -B test_task.py",
    )
    (repo / "test_calculator.py").write_text(
        "from calculator import add\n\n"
        "def test_add_positive_numbers():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_add_negative_numbers():\n"
        "    assert add(-2, -3) == -5\n\n"
        "if __name__ == '__main__':\n"
        "    test_add_positive_numbers()\n"
        "    test_add_negative_numbers()\n"
        "    print('ok')\n",
        encoding="utf-8",
    )
    return CodeTask(
        task_id="code_smoke_0001",
        repo_path=str(repo.resolve()),
        instruction="Fix the calculator.add implementation so all tests pass.",
        validation_command="python3 -B test_calculator.py",
    )


def write_tasks_jsonl(path: str | Path, tasks: list[CodeTask]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
