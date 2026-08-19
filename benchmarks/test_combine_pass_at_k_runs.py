import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "combine_pass_at_k_runs",
    ROOT / "scripts" / "combine_pass_at_k_runs.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CombinePassAtKRunsTest(unittest.TestCase):
    def test_reports_current_samples_as_strict_same_configuration(self) -> None:
        rows = []
        for task in ("task-a", "task-b"):
            rows.append(
                {
                    "instance_id": task,
                    "dataset": "demo",
                    "sample_index": 1,
                    "cohort": "legacy",
                    "benchmark_resolved": True,
                }
            )
        rows.extend(
            [
                {
                    "instance_id": "task-a",
                    "dataset": "demo",
                    "sample_index": 2,
                    "cohort": "current",
                    "benchmark_resolved": True,
                },
                {
                    "instance_id": "task-a",
                    "dataset": "demo",
                    "sample_index": 3,
                    "cohort": "current",
                    "benchmark_resolved": False,
                },
                {
                    "instance_id": "task-b",
                    "dataset": "demo",
                    "sample_index": 2,
                    "cohort": "current",
                    "benchmark_resolved": False,
                },
                {
                    "instance_id": "task-b",
                    "dataset": "demo",
                    "sample_index": 3,
                    "cohort": "current",
                    "benchmark_resolved": False,
                },
            ]
        )

        result = MODULE.extra_summary(rows, expected_samples=3)

        current = result["current_same_harness"]
        self.assertTrue(current["same_harness_configuration"])
        self.assertFalse(current["strict_same_execution_policy"])
        self.assertEqual(current["complete_task_count"], 2)
        self.assertAlmostEqual(current["aggregate"]["pass@1"], 0.25)
        self.assertAlmostEqual(current["aggregate"]["pass@2"], 0.5)
        self.assertAlmostEqual(current["by_dataset"]["demo"]["pass@1"], 0.25)
        self.assertAlmostEqual(current["by_dataset"]["demo"]["pass@2"], 0.5)


if __name__ == "__main__":
    unittest.main()
