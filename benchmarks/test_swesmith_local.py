from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from code_agent_baseline.swesmith_local import (
    build_runnable_tasks,
    ensure_repo_checkout,
    materialize_selected_tasks,
    sanity_one_task,
)


class SwesmithLocalTests(unittest.TestCase):
    def test_checkout_fetches_only_the_fixed_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "repo"
            completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            with (
                patch(
                    "code_agent_baseline.swesmith_local.ensure_repo_tarball_checkout",
                    side_effect=RuntimeError("tarball unavailable"),
                ),
                patch("code_agent_baseline.swesmith_local.subprocess.run", return_value=completed) as run,
            ):
                ensure_repo_checkout("https://github.com/acme/demo", "abc123", dest, timeout_sec=900)

            commands = [call.args[0] for call in run.call_args_list]
            fetch = next(command for command in commands if "fetch" in command)
            checkout = next(command for command in commands if "checkout" in command)
            self.assertIn("--depth=1", fetch)
            self.assertEqual(fetch[-1], "abc123")
            self.assertEqual(checkout[-2:], ["--detach", "FETCH_HEAD"])
            marker = json.loads((dest / ".lottie-checkout.json").read_text(encoding="utf-8"))
            self.assertEqual(marker["fetch"], "depth-1")

    def test_checkout_prefers_fixed_commit_tarball(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "repo"

            def create_tarball_checkout(_source: str, _commit: str, target: Path, *, timeout_sec: int) -> None:
                self.assertEqual(timeout_sec, 600)
                target.mkdir(parents=True)

            with (
                patch(
                    "code_agent_baseline.swesmith_local.ensure_repo_tarball_checkout",
                    side_effect=create_tarball_checkout,
                ) as fallback,
                patch("code_agent_baseline.swesmith_local.subprocess.run") as run,
            ):
                ensure_repo_checkout(
                    "https://github.com/acme/demo",
                    "abc123",
                    dest,
                    timeout_sec=900,
                )

            fallback.assert_called_once_with(
                "https://github.com/acme/demo",
                "abc123",
                dest,
                timeout_sec=600,
            )
            run.assert_not_called()
            marker = json.loads((dest / ".lottie-checkout.json").read_text(encoding="utf-8"))
            self.assertEqual(marker["fetch"], "github-tarball")

    def test_runnable_export_only_keeps_gold_sanity_passed_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks.jsonl"
            results = root / "results.jsonl"
            output = root / "runnable.jsonl"
            tasks.write_text(
                json.dumps({"instance_id": "a", "split": "train", "benchmark_repo": "repo/a"}) + "\n"
                + json.dumps({"instance_id": "b", "split": "evaluation", "benchmark_repo": "repo/b"})
                + "\n",
                encoding="utf-8",
            )
            results.write_text(
                json.dumps({"instance_id": "a", "gold_sanity_passed": True}) + "\n"
                + json.dumps({"instance_id": "b", "gold_sanity_passed": False})
                + "\n",
                encoding="utf-8",
            )

            manifest = build_runnable_tasks(tasks, results, output)

            self.assertEqual(manifest["tasks_out"], 1)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["instance_id"], "a")

    def test_materializes_selected_task_with_hidden_bug_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = {
                "splits": {
                    name: {
                        "swesmith_py": {
                            "profiles": {
                                "swesmith/acme__demo.abc": {
                                    "source": "https://github.com/acme/demo",
                                    "commit": "abc123",
                                }
                            },
                            "instance_ids": ["acme__demo.abc.func_basic__one"] if name == "train" else [],
                        }
                    }
                    for name in ("train", "validation", "evaluation")
                }
            }
            rows = [
                {
                    "instance_id": "acme__demo.abc.func_basic__one",
                    "repo": "swesmith/acme__demo.abc",
                    "patch": "diff --git a/a.py b/a.py\n",
                    "FAIL_TO_PASS": ["tests/test_a.py::test_a"],
                    "PASS_TO_PASS": ["tests/test_a.py::test_b"],
                    "problem_statement": "Fix the behavior.",
                }
            ]
            selection_path = root / "selection.json"
            rows_path = root / "rows.json"
            output = root / "tasks.jsonl"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            rows_path.write_text(json.dumps(rows), encoding="utf-8")

            manifest = materialize_selected_tasks(selection_path, [rows_path], output, cache_root=root / "cache")
            task = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(manifest["tasks"], 1)
            self.assertEqual(task["repo"], "acme/demo")
            self.assertEqual(task["base_commit"], "abc123")
            self.assertEqual(task["bug_patch"], rows[0]["patch"])
            self.assertEqual(task["patch"], "")
            self.assertFalse(task["gold_visible_to_agent"])

    def test_gold_sanity_requires_base_pass_bug_fail_and_reverse_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            tests = repo / "tests"
            tests.mkdir()
            (tests / "test_calc.py").write_text(
                "from calc import add\n\n"
                "def test_add():\n    assert add(2, 3) == 5\n\n"
                "def test_identity():\n    assert add(4, 0) == 4\n",
                encoding="utf-8",
            )
            self._run(["git", "init"], repo)
            self._run(["git", "add", "."], repo)
            self._run(["git", "-c", "user.email=a@b.c", "-c", "user.name=test", "commit", "-m", "base"], repo)
            (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            bug_patch = self._run(["git", "diff"], repo).stdout
            self._run(["git", "checkout", "--", "calc.py"], repo)

            task = {
                "instance_id": "demo.func_basic__one",
                "split": "train",
                "benchmark_repo": "swesmith/demo",
                "local_repo_path": str(repo),
                "local_venv_path": str(Path(sys.executable).parent.parent),
                "bug_patch": bug_patch,
                "FAIL_TO_PASS": ["tests/test_calc.py::test_add"],
                "PASS_TO_PASS": ["tests/test_calc.py::test_identity"],
            }
            result = sanity_one_task(task, p2p_limit=1, timeout_sec=30)

            self.assertTrue(result.gold_sanity_passed, result)
            self.assertEqual(result.status, "gold_sanity_passed")
            self.assertEqual(result.base_returncode, 0)
            self.assertEqual(result.bug_returncode, 1)
            self.assertEqual(result.gold_returncode, 0)
            self.assertTrue(result.gold_restored_clean_tree)

    @staticmethod
    def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)


if __name__ == "__main__":
    unittest.main()
