from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from code_agent_baseline.evalplus_local import (
    _deserialize_mbpp_inputs,
    build_runnable_tasks,
    materialize_function_workspaces,
    materialize_selected_tasks,
    sanity_one_task,
    verify_one_function_record,
)


class EvalPlusLocalTests(unittest.TestCase):
    def test_official_mbpp_special_input_deserialization(self) -> None:
        self.assertEqual(_deserialize_mbpp_inputs("MBPP/124", [["1", "2j"]]), [[1.0, 2j]])
        self.assertEqual(_deserialize_mbpp_inputs("MBPP/252", [["(1+2j)"]]), [[1 + 2j]])
        self.assertEqual(
            _deserialize_mbpp_inputs("MBPP/580", [[[[1, 2], 4]]]),
            [(((1, 2), 4),)],
        )

    def test_materialize_and_validate_reference_with_negative_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = {
                "splits": {
                    "train": {
                        "mbppplus": {"task_ids": [1]},
                        "humanevalplus": {"task_ids": ["HumanEval/0"]},
                    },
                    "validation": {"mbppplus": {"task_ids": []}, "humanevalplus": {"task_ids": []}},
                    "evaluation": {"mbppplus": {"task_ids": []}, "humanevalplus": {"task_ids": []}},
                }
            }
            mbpp = {
                "rows": [
                    {
                        "row": {
                            "task_id": 1,
                            "prompt": "Add two numbers.",
                            "code": "def add(a, b):\n    return a + b\n",
                            "test_list": ["assert add(1, 2) == 3"],
                            "test_imports": [],
                            "test": "assert add(2, 3) == 5\n",
                        }
                    }
                ]
            }
            humaneval = {
                "rows": [
                    {
                        "row": {
                            "task_id": "HumanEval/0",
                            "prompt": "def double(x):\n    \"\"\"Return twice x.\"\"\"\n",
                            "canonical_solution": "    return x * 2\n",
                            "entry_point": "double",
                            "test": "def check(candidate):\n    assert candidate(3) == 6\n",
                        }
                    }
                ]
            }
            selection_path = root / "selection.json"
            mbpp_path = root / "mbpp.json"
            humaneval_path = root / "humaneval.json"
            tasks_path = root / "tasks.jsonl"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            mbpp_path.write_text(json.dumps(mbpp), encoding="utf-8")
            humaneval_path.write_text(json.dumps(humaneval), encoding="utf-8")

            manifest = materialize_selected_tasks(
                selection_path, [mbpp_path], [humaneval_path], tasks_path
            )
            tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(manifest["tasks"], 2)
            self.assertTrue(all(not task["gold_visible_to_agent"] for task in tasks))
            for task in tasks:
                result = sanity_one_task(task, sys.executable, timeout_sec=10)
                self.assertTrue(result.gold_sanity_passed, result)
                self.assertEqual(result.reference_returncode, 0)
                self.assertNotEqual(result.stub_returncode, 0)

    def test_runnable_only_keeps_passed_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks.jsonl"
            sanity = root / "sanity.jsonl"
            output = root / "runnable.jsonl"
            tasks.write_text(
                json.dumps({"instance_id": "a", "dataset": "mbppplus", "split": "train"}) + "\n"
                + json.dumps({"instance_id": "b", "dataset": "humanevalplus", "split": "evaluation"})
                + "\n",
                encoding="utf-8",
            )
            sanity.write_text(
                json.dumps({"instance_id": "a", "gold_sanity_passed": True}) + "\n"
                + json.dumps({"instance_id": "b", "gold_sanity_passed": False})
                + "\n",
                encoding="utf-8",
            )

            manifest = build_runnable_tasks(tasks, sanity, output)

            self.assertEqual(manifest["tasks_out"], 1)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["instance_id"], "a")

    def test_batch_sanity_accepts_relative_python_path(self) -> None:
        from code_agent_baseline.evalplus_local import run_gold_sanity

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            tasks = root / "tasks.jsonl"
            output = root / "results.jsonl"
            task = {
                "instance_id": "MBPP/1",
                "dataset": "mbppplus",
                "split": "train",
                "entry_point": "add",
                "reference_solution": "def add(a, b):\n    return a + b\n",
                "starter_code": "def add(a, b):\n    raise NotImplementedError()\n",
                "hidden_tests": "assert add(2, 3) == 5\n",
            }
            tasks.write_text(json.dumps(task) + "\n", encoding="utf-8")
            relative_python = Path(os.path.relpath(sys.executable, Path.cwd()))

            results = run_gold_sanity(tasks, output, relative_python, timeout_sec=10)

            self.assertTrue(results[0].gold_sanity_passed)

    def test_agent_workspace_hides_gold_and_verifier_scores_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_path = root / "tasks.jsonl"
            agent_tasks_path = root / "agent_tasks.jsonl"
            task = {
                "instance_id": "MBPP/1",
                "dataset": "mbppplus",
                "split": "evaluation",
                "problem_statement": "Implement add(a, b).",
                "entry_point": "add",
                "starter_code": "def add(a, b):\n    raise NotImplementedError()\n",
                "reference_solution": "def add(a, b):\n    return a + b\n",
                "base_inputs": [[1, 2]],
                "plus_inputs": [[-4, 7], [0, 0]],
                "atol": 0,
            }
            tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            materialize_function_workspaces(tasks_path, agent_tasks_path, root / "repos")
            agent_task = json.loads(agent_tasks_path.read_text(encoding="utf-8"))
            repo = Path(agent_task["local_repo_path"])

            self.assertNotIn("return a + b", (repo / "solution.py").read_text(encoding="utf-8"))
            self.assertFalse((repo / ".lottie_hidden_eval.py").exists())

            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.email=a@b.c", "-c", "user.name=test", "commit", "-m", "base"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            (repo / "solution.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            patch = subprocess.run(
                ["git", "diff"], cwd=repo, text=True, capture_output=True, check=True
            ).stdout
            subprocess.run(["git", "checkout", "--", "solution.py"], cwd=repo, check=True, capture_output=True)

            result = verify_one_function_record(
                {"instance_id": "MBPP/1", "patch": patch}, agent_task, sys.executable, timeout_sec=10
            )

            self.assertTrue(result.benchmark_resolved, result)
            self.assertEqual(result.verifier_status, "resolved")


if __name__ == "__main__":
    unittest.main()
