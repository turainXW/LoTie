import json
import tempfile
import unittest
from pathlib import Path

from code_agent_baseline.custom_benchmark import build_new_file_patch, create_benchmark_project


class CustomBenchmarkProjectTest(unittest.TestCase):
    def test_create_project_with_existing_test_and_gold_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo_src"
            repo.mkdir()
            (repo / "calculator.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            (repo / "requirements.txt").write_text("", encoding="utf-8")
            test_patch = root / "test.patch"
            test_patch.write_text(
                "\n".join(
                    [
                        "diff --git a/tests/test_calculator.py b/tests/test_calculator.py",
                        "new file mode 100644",
                        "--- /dev/null",
                        "+++ b/tests/test_calculator.py",
                        "@@ -0,0 +1,5 @@",
                        "+from calculator import add",
                        "+",
                        "+",
                        "+def test_add():",
                        "+    assert add(1, 2) == 3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gold_patch = root / "gold.patch"
            gold_patch.write_text(
                "\n".join(
                    [
                        "diff --git a/calculator.py b/calculator.py",
                        "--- a/calculator.py",
                        "+++ b/calculator.py",
                        "@@ -1,2 +1,2 @@",
                        " def add(a, b):",
                        "-    return a - b",
                        "+    return a + b",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = create_benchmark_project(
                name="Toy Add Bug",
                repo=repo,
                output_dir=root / "projects",
                problem_statement="add returns the wrong result",
                pytest_nodes=["tests/test_calculator.py::test_add"],
                test_patch_path=test_patch,
                gold_patch_path=gold_patch,
                image_name="local/toy-add:latest",
            )

            project = result.project_dir
            self.assertTrue((project / "Dockerfile").exists())
            self.assertTrue((project / "verifier.sh").exists())
            self.assertTrue((project / "build_image.sh").exists())
            self.assertTrue((project / "repo" / "calculator.py").exists())
            self.assertTrue((project / "skills" / "pytest_generation_skill.md").exists())
            self.assertTrue((project / "records.gold.jsonl").exists())

            task = json.loads((project / "task.json").read_text(encoding="utf-8"))
            self.assertEqual(task["instance_id"], "custom__toy_add_bug")
            self.assertEqual(task["image"], "local/toy-add:latest")
            self.assertEqual(task["FAIL_TO_PASS"], ["tests/test_calculator.py::test_add"])
            self.assertIn("test_add", task["test_patch"])

            task_lines = (project / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(task_lines), 1)
            record = json.loads((project / "records.gold.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["instance_id"], "custom__toy_add_bug")
            self.assertIn("return a + b", record["patch"])

    def test_build_new_file_patch(self) -> None:
        patch = build_new_file_patch("tests/test_bug.py", "def test_bug():\n    assert True\n")
        self.assertIn("diff --git a/tests/test_bug.py b/tests/test_bug.py", patch)
        self.assertIn("--- /dev/null", patch)
        self.assertIn("+++ b/tests/test_bug.py", patch)
        self.assertIn("+def test_bug():", patch)
        self.assertIn("+    assert True", patch)


if __name__ == "__main__":
    unittest.main()
