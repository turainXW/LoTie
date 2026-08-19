from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LocalSweBenchTask:
    instance_id: str
    repo: str
    repo_path: str
    base_commit: str
    problem_statement: str
    hints_text: str
    patch: str
    test_patch: str
    validation_command: str
    FAIL_TO_PASS: list[str] = field(default_factory=list)
    PASS_TO_PASS: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LocalSweBenchResult:
    instance_id: str
    mode: str
    passed: bool
    exit_code: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def create_local_swebench_suite(output_dir: str | Path) -> list[LocalSweBenchTask]:
    root = Path(output_dir).resolve()
    repo_root = root / "repos"
    repo_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        _create_slug_parser_task(repo_root),
        _create_window_average_task(repo_root),
    ]
    write_tasks_jsonl(root / "tasks.jsonl", tasks)
    return tasks


def verify_task(task: LocalSweBenchTask, apply_gold: bool = False, timeout_sec: int = 30) -> LocalSweBenchResult:
    repo = Path(task.repo_path)
    if apply_gold:
        apply_unified_patch(repo, task.patch)
    completed = subprocess.run(
        task.validation_command,
        cwd=repo,
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    return LocalSweBenchResult(
        instance_id=task.instance_id,
        mode="gold_patch" if apply_gold else "base",
        passed=completed.returncode == 0,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def apply_unified_patch(repo: str | Path, patch_text: str) -> None:
    completed = subprocess.run(
        ["patch", "-p1"],
        cwd=repo,
        input=patch_text,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"patch failed\nstdout={completed.stdout}\nstderr={completed.stderr}")


def write_tasks_jsonl(path: str | Path, tasks: list[LocalSweBenchTask]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")


def read_tasks_jsonl(path: str | Path) -> list[LocalSweBenchTask]:
    tasks: list[LocalSweBenchTask] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(LocalSweBenchTask(**json.loads(line)))
    return tasks


def write_results_jsonl(path: str | Path, results: list[LocalSweBenchResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def _create_slug_parser_task(root: Path) -> LocalSweBenchTask:
    repo = root / "swe_local_0001_slug_parser"
    tests = repo / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (repo / "text_utils.py").write_text(
        "import re\n\n"
        "def slugify(text):\n"
        "    text = text.strip().lower()\n"
        "    return text.replace(' ', '-')\n",
        encoding="utf-8",
    )
    (tests / "__init__.py").write_text("", encoding="utf-8")
    (tests / "test_existing.py").write_text(
        "import unittest\n\n"
        "from text_utils import slugify\n\n\n"
        "class ExistingSlugTests(unittest.TestCase):\n"
        "    def test_simple_words(self):\n"
        "        self.assertEqual(slugify('Hello World'), 'hello-world')\n\n"
        "    def test_trims_outer_space(self):\n"
        "        self.assertEqual(slugify('  Agentic Code  '), 'agentic-code')\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    (tests / "test_issue.py").write_text(
        "import unittest\n\n"
        "from text_utils import slugify\n\n\n"
        "class IssueSlugTests(unittest.TestCase):\n"
        "    def test_punctuation_collapses_to_single_dash(self):\n"
        "        self.assertEqual(slugify('LLM, Judge? No: tests!'), 'llm-judge-no-tests')\n\n"
        "    def test_repeated_separators_are_collapsed(self):\n"
        "        self.assertEqual(slugify('A  B---C'), 'a-b-c')\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    patch = (
        "diff --git a/text_utils.py b/text_utils.py\n"
        "--- a/text_utils.py\n"
        "+++ b/text_utils.py\n"
        "@@ -1,5 +1,7 @@\n"
        " import re\n"
        " \n"
        " def slugify(text):\n"
        "-    text = text.strip().lower()\n"
        "-    return text.replace(' ', '-')\n"
        "+    text = text.strip().lower()\n"
        "+    text = re.sub(r'[^a-z0-9]+', '-', text)\n"
        "+    text = re.sub(r'-+', '-', text)\n"
        "+    return text.strip('-')\n"
    )
    test_patch = (
        "diff --git a/tests/test_issue.py b/tests/test_issue.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_issue.py\n"
    )
    return LocalSweBenchTask(
        instance_id="swe_local_0001_slug_parser",
        repo="local/text-utils",
        repo_path=str(repo),
        base_commit="local-base-0001",
        problem_statement=(
            "slugify keeps punctuation in generated slugs. Make it replace punctuation and repeated "
            "separators with a single dash while preserving existing behavior."
        ),
        hints_text="Use a regex-based normalization pass.",
        patch=patch,
        test_patch=test_patch,
        validation_command="python3 -B -m unittest discover -s tests",
        FAIL_TO_PASS=[
            "tests.test_issue.IssueSlugTests.test_punctuation_collapses_to_single_dash",
            "tests.test_issue.IssueSlugTests.test_repeated_separators_are_collapsed",
        ],
        PASS_TO_PASS=[
            "tests.test_existing.ExistingSlugTests.test_simple_words",
            "tests.test_existing.ExistingSlugTests.test_trims_outer_space",
        ],
    )


def _create_window_average_task(root: Path) -> LocalSweBenchTask:
    repo = root / "swe_local_0002_window_average"
    tests = repo / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (repo / "series.py").write_text(
        "def moving_average(values, window):\n"
        "    if window <= 0:\n"
        "        raise ValueError('window must be positive')\n"
        "    result = []\n"
        "    for index in range(len(values) - window):\n"
        "        result.append(sum(values[index:index + window]) / window)\n"
        "    return result\n",
        encoding="utf-8",
    )
    (tests / "__init__.py").write_text("", encoding="utf-8")
    (tests / "test_existing.py").write_text(
        "import unittest\n\n"
        "from series import moving_average\n\n\n"
        "class ExistingMovingAverageTests(unittest.TestCase):\n"
        "    def test_empty_values(self):\n"
        "        self.assertEqual(moving_average([], 1), [])\n\n"
        "    def test_bad_window(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            moving_average([1, 2, 3], 0)\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    (tests / "test_issue.py").write_text(
        "import unittest\n\n"
        "from series import moving_average\n\n\n"
        "class IssueMovingAverageTests(unittest.TestCase):\n"
        "    def test_includes_last_full_window(self):\n"
        "        self.assertEqual(moving_average([1, 3, 5], 2), [2.0, 4.0])\n\n"
        "    def test_window_equal_to_length(self):\n"
        "        self.assertEqual(moving_average([2, 4, 6], 3), [4.0])\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    patch = (
        "diff --git a/series.py b/series.py\n"
        "--- a/series.py\n"
        "+++ b/series.py\n"
        "@@ -2,6 +2,6 @@ def moving_average(values, window):\n"
        "     if window <= 0:\n"
        "         raise ValueError('window must be positive')\n"
        "     result = []\n"
        "-    for index in range(len(values) - window):\n"
        "+    for index in range(len(values) - window + 1):\n"
        "         result.append(sum(values[index:index + window]) / window)\n"
        "     return result\n"
    )
    test_patch = (
        "diff --git a/tests/test_issue.py b/tests/test_issue.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_issue.py\n"
    )
    return LocalSweBenchTask(
        instance_id="swe_local_0002_window_average",
        repo="local/series-utils",
        repo_path=str(repo),
        base_commit="local-base-0002",
        problem_statement=(
            "moving_average drops the final valid window. Include every full window, including the "
            "one that ends at the last input element."
        ),
        hints_text="Check the upper bound of the range.",
        patch=patch,
        test_patch=test_patch,
        validation_command="python3 -B -m unittest discover -s tests",
        FAIL_TO_PASS=[
            "tests.test_issue.IssueMovingAverageTests.test_includes_last_full_window",
            "tests.test_issue.IssueMovingAverageTests.test_window_equal_to_length",
        ],
        PASS_TO_PASS=[
            "tests.test_existing.ExistingMovingAverageTests.test_empty_values",
            "tests.test_existing.ExistingMovingAverageTests.test_bad_window",
        ],
    )
