from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_swesmith_venvs.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_swesmith_venvs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class BootstrapSwesmithVenvsTest(unittest.TestCase):
    def test_materialize_runtime_tasks_rebases_machine_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "tasks.jsonl"
            output = root / "runtime" / "tasks_runtime.jsonl"
            cache = root / "cache"
            source.write_text(
                json.dumps(
                    {
                        "instance_id": "demo.task",
                        "benchmark_repo": "swesmith/acme__demo.abc123",
                        "local_repo_path": "/old/repo",
                        "local_venv_path": "/old/venv",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            tasks, sources = bootstrap.materialize_runtime_tasks([source], output, cache)

            self.assertEqual(len(tasks), 1)
            self.assertEqual(len(sources), 1)
            self.assertEqual(
                tasks[0]["local_repo_path"],
                str(cache / "repos" / "swesmith__acme__demo_abc123"),
            )
            self.assertEqual(
                tasks[0]["local_venv_path"],
                str(cache / "venvs" / "swesmith__acme__demo_abc123"),
            )

    def test_portable_freeze_removes_local_editable_entries(self) -> None:
        repo = Path("/tmp/demo-repo")
        raw = (
            "pytest==8.4.0\n"
            "# Editable install with no version control\n"
            "-e file:///tmp/demo-repo\n"
            "demo @ file:///tmp/demo-repo\n"
            "requests==2.32.0\n"
        )

        lines = bootstrap.portable_freeze_lines(raw, repo)

        self.assertEqual(lines, ["pytest==8.4.0", "requests==2.32.0"])

    def test_select_smoke_tasks_limits_each_repository(self) -> None:
        tasks = [
            {"instance_id": "a1", "benchmark_repo": "a"},
            {"instance_id": "a2", "benchmark_repo": "a"},
            {"instance_id": "b1", "benchmark_repo": "b"},
        ]

        selected = bootstrap.select_smoke_tasks(tasks, 1)

        self.assertEqual([task["instance_id"] for task in selected], ["a1", "b1"])


if __name__ == "__main__":
    unittest.main()
