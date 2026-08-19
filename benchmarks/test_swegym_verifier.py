import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from code_agent_baseline.swegym_verifier import ensure_local_venv, verify_swegym_patch_records


class SwegymVerifierTest(unittest.TestCase):
    def test_local_venv_preflight_includes_repository_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "src").mkdir()
            captured = {}

            def fake_run(command, **kwargs):
                captured.update(kwargs)
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("code_agent_baseline.swegym_verifier.subprocess.run", side_effect=fake_run):
                result = ensure_local_venv(
                    Path(sys.executable).parent.parent,
                    repo_path=repo,
                    timeout_sec=60,
                    install_deps=False,
                    create=False,
                )

            self.assertEqual(result["returncode"], 0)
            self.assertEqual(captured["cwd"], repo)
            first_pythonpath = Path(captured["env"]["PYTHONPATH"].split(":", 1)[0])
            self.assertEqual(first_pythonpath, (repo / "src").resolve())

    def test_local_venv_applies_patch_and_runs_pytest_in_repo_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            tests_dir = repo / "tests"
            tests_dir.mkdir(parents=True)
            (repo / "calculator.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            (tests_dir / "test_calculator.py").write_text(
                "from calculator import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
                encoding="utf-8",
            )
            patch_text = "\n".join(
                [
                    "diff --git a/calculator.py b/calculator.py",
                    "--- a/calculator.py",
                    "+++ b/calculator.py",
                    "@@ -1,2 +1,2 @@",
                    " def add(a, b):",
                    "-    return a - b",
                    "+    return a + b",
                ]
            ) + "\n"
            records = root / "records.jsonl"
            tasks = root / "tasks.jsonl"
            output = root / "verify.jsonl"
            records.write_text(
                json.dumps({"instance_id": "custom__add", "patch": patch_text}) + "\n",
                encoding="utf-8",
            )
            tasks.write_text(
                json.dumps(
                    {
                        "instance_id": "custom__add",
                        "FAIL_TO_PASS": ["tests/test_calculator.py::test_add"],
                        "metadata": {"local_repo_path": "repo"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = verify_swegym_patch_records(
                records,
                output,
                tasks_path=tasks,
                mode="local-venv",
                local_venv_path=Path(sys.executable).parent.parent,
            )
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(summary.resolved, 1)
            self.assertEqual(row["verifier_status"], "resolved")
            self.assertEqual(row["metadata"]["verifier_backend"], "local_venv")
            self.assertEqual(row["metadata"]["official_comparable"], False)
            self.assertIn("return a - b", (repo / "calculator.py").read_text(encoding="utf-8"))

    def test_local_venv_initializes_swesmith_bug_before_agent_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            tests_dir = repo / "tests"
            tests_dir.mkdir(parents=True)
            (repo / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            (tests_dir / "test_calculator.py").write_text(
                "from calculator import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
                encoding="utf-8",
            )
            bug_patch = "\n".join(
                [
                    "diff --git a/calculator.py b/calculator.py",
                    "--- a/calculator.py",
                    "+++ b/calculator.py",
                    "@@ -1,2 +1,2 @@",
                    " def add(a, b):",
                    "-    return a + b",
                    "+    return a - b",
                ]
            ) + "\n"
            fix_patch = bug_patch.replace("-    return a + b\n+    return a - b", "-    return a - b\n+    return a + b")
            records = root / "records.jsonl"
            tasks = root / "tasks.jsonl"
            output = root / "verify.jsonl"
            records.write_text(json.dumps({"instance_id": "smith__add", "patch": fix_patch}) + "\n", encoding="utf-8")
            tasks.write_text(
                json.dumps(
                    {
                        "instance_id": "smith__add",
                        "FAIL_TO_PASS": ["tests/test_calculator.py::test_add"],
                        "bug_patch": bug_patch,
                        "local_venv_path": str(Path(sys.executable).parent.parent),
                        "metadata": {"local_repo_path": "repo"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = verify_swegym_patch_records(
                records,
                output,
                tasks_path=tasks,
                mode="local-venv",
            )
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(summary.resolved, 1)
            self.assertEqual(row["metadata"]["bug_patch_present"], True)
            self.assertIn("return a + b", (repo / "calculator.py").read_text(encoding="utf-8"))

    def test_blocks_when_official_image_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            tasks = root / "tasks.jsonl"
            output = root / "verify.jsonl"
            records.write_text(
                json.dumps({"instance_id": "repo-task-1", "patch": "diff --git a/a.py b/a.py\n"}) + "\n",
                encoding="utf-8",
            )
            tasks.write_text(
                json.dumps({"instance_id": "repo-task-1", "FAIL_TO_PASS": ["tests/test_a.py::test_bug"]}) + "\n",
                encoding="utf-8",
            )

            summary = verify_swegym_patch_records(records, output, tasks_path=tasks)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary.blocked, 1)
            self.assertEqual(rows[0]["benchmark_resolved"], None)
            self.assertEqual(rows[0]["verifier_status"], "blocked_missing_official_image")

    def test_infers_swegym_eval_image_from_instance_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            tasks = root / "tasks.jsonl"
            output = root / "verify.jsonl"
            records.write_text(
                json.dumps({"instance_id": "bokeh__bokeh-13370", "patch": "diff --git a/a.py b/a.py\n"}) + "\n",
                encoding="utf-8",
            )
            tasks.write_text(
                json.dumps({"instance_id": "bokeh__bokeh-13370", "FAIL_TO_PASS": ["tests/test_a.py::test_bug"]})
                + "\n",
                encoding="utf-8",
            )

            verify_swegym_patch_records(records, output, tasks_path=tasks, mode="dry-run")
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(rows[0]["verifier_status"], "ready_for_docker")
            self.assertEqual(
                rows[0]["image"],
                "docker.1ms.run/xingyaoww/sweb.eval.x86_64.bokeh_s_bokeh-13370:latest",
            )

    def test_dry_run_reports_docker_ready_without_claiming_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            tasks = root / "tasks.jsonl"
            output = root / "verify.jsonl"
            records.write_text(
                json.dumps({"instance_id": "repo__task-1", "patch": "diff --git a/a.py b/a.py\n"}) + "\n",
                encoding="utf-8",
            )
            tasks.write_text(
                json.dumps(
                    {
                        "instance_id": "repo__task-1",
                        "docker_image": "swebench/example:latest",
                        "test_command": "python -m pytest tests/test_a.py::test_bug",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = verify_swegym_patch_records(records, output, tasks_path=tasks, mode="dry-run")
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary.blocked, 1)
            self.assertEqual(rows[0]["benchmark_resolved"], None)
            self.assertEqual(rows[0]["verifier_status"], "ready_for_docker")
            self.assertEqual(rows[0]["image"], "swebench/example:latest")

    def test_derives_pytest_command_from_fail_to_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            tasks = root / "tasks.jsonl"
            output = root / "verify.jsonl"
            records.write_text(
                json.dumps({"instance_id": "repo__task-1", "patch": "diff --git a/a.py b/a.py\n"}) + "\n",
                encoding="utf-8",
            )
            tasks.write_text(
                json.dumps(
                    {
                        "instance_id": "repo__task-1",
                        "image": "swebench/example:latest",
                        "FAIL_TO_PASS": ["tests/test_a.py::test_bug"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            verify_swegym_patch_records(records, output, tasks_path=tasks, mode="dry-run")
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(rows[0]["verifier_status"], "ready_for_docker")
            self.assertEqual(rows[0]["test_command"], "python -m pytest tests/test_a.py::test_bug")

    def test_docker_verifier_combines_agent_patch_with_official_test_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            tasks = root / "tasks.jsonl"
            output = root / "verify.jsonl"
            records.write_text(
                json.dumps({"instance_id": "repo__task-1", "patch": "diff --git a/a.py b/a.py\nagent\n"}) + "\n",
                encoding="utf-8",
            )
            tasks.write_text(
                json.dumps(
                    {
                        "instance_id": "repo__task-1",
                        "image": "swebench/example:latest",
                        "FAIL_TO_PASS": ["tests/test_a.py::test_bug"],
                        "test_patch": "diff --git a/tests/test_a.py b/tests/test_a.py\ntest\n",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            captured: dict[str, str] = {}

            def fake_run(command, text, capture_output, timeout):
                patch_arg = next(item for item in command if item.endswith(":/tmp/agent.patch:ro"))
                patch_path = Path(patch_arg.split(":", 1)[0])
                captured["patch"] = patch_path.read_text(encoding="utf-8")

                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return Completed()

            with patch("code_agent_baseline.swegym_verifier.shutil.which", return_value="/usr/bin/docker"):
                with patch("code_agent_baseline.swegym_verifier.subprocess.run", side_effect=fake_run):
                    summary = verify_swegym_patch_records(records, output, tasks_path=tasks, mode="docker")

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary.resolved, 1)
            self.assertIn("agent", captured["patch"])
            self.assertIn("test", captured["patch"])
            self.assertEqual(rows[0]["metadata"]["has_test_patch"], True)

    def test_docker_image_pull_failure_is_blocked_not_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            tasks = root / "tasks.jsonl"
            output = root / "verify.jsonl"
            records.write_text(
                json.dumps({"instance_id": "repo__task-1", "patch": "diff --git a/a.py b/a.py\n"}) + "\n",
                encoding="utf-8",
            )
            tasks.write_text(
                json.dumps(
                    {
                        "instance_id": "repo__task-1",
                        "image": "swebench/missing:latest",
                        "test_command": "python -m pytest tests/test_a.py::test_bug",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            class Completed:
                returncode = 125
                stdout = ""
                stderr = "Unable to find image 'swebench/missing:latest' locally\nfailed to resolve reference"

            with patch("code_agent_baseline.swegym_verifier.shutil.which", return_value="/usr/bin/docker"):
                with patch("code_agent_baseline.swegym_verifier.subprocess.run", return_value=Completed()):
                    summary = verify_swegym_patch_records(records, output, tasks_path=tasks, mode="docker")

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary.blocked, 1)
            self.assertEqual(summary.unresolved, 0)
            self.assertEqual(rows[0]["benchmark_resolved"], None)
            self.assertEqual(rows[0]["verifier_status"], "blocked_image_pull_failed")


if __name__ == "__main__":
    unittest.main()
