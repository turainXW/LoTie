from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from code_agent_baseline.eval90_deploy import Eval90Layout, deployment_status, prepare_evaluation_tasks


class Eval90DeployTests(unittest.TestCase):
    def test_prepare_filters_evaluation_and_rewrites_machine_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            (data / "evalplus_local").mkdir(parents=True)
            (data / "swesmith_local").mkdir(parents=True)
            function_rows = []
            for dataset in ("mbppplus", "humanevalplus"):
                function_rows.extend(
                    {"instance_id": f"{dataset}/{index}", "dataset": dataset, "split": "evaluation"}
                    for index in range(30)
                )
                function_rows.append({"instance_id": f"{dataset}/train", "dataset": dataset, "split": "train"})
            swe_rows = [
                {
                    "instance_id": f"repo-{index // 5}.task-{index}",
                    "benchmark_repo": f"swesmith/org__repo{index // 5}.abc",
                    "split": "evaluation",
                    "local_repo_path": "/old/machine/repo",
                    "local_venv_path": "/old/machine/venv",
                }
                for index in range(30)
            ]
            self._write_jsonl(data / "evalplus_local/tasks.jsonl", function_rows)
            self._write_jsonl(data / "swesmith_local/tasks.jsonl", swe_rows)
            layout = Eval90Layout.create(root, root / "state")

            manifest = prepare_evaluation_tasks(layout)
            prepared_swe = self._read_jsonl(layout.swesmith_tasks)

            self.assertEqual(manifest["counts"], {"mbppplus": 30, "humanevalplus": 30, "swesmith_py": 30})
            self.assertEqual(len(manifest["swesmith_repositories"]), 6)
            self.assertTrue(all(str(layout.state_root) in row["local_repo_path"] for row in prepared_swe))
            self.assertTrue(all("/old/machine" not in row["local_venv_path"] for row in prepared_swe))

    def test_status_is_not_ready_before_environments_and_sanity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = Eval90Layout.create(root, root / "state")
            layout.ensure_directories()

            status = deployment_status(layout)

            self.assertFalse(status["ready"])
            self.assertFalse(any(status["checks"].values()))
            self.assertTrue((layout.reports / "deployment_status.json").is_file())

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
