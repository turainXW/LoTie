import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_model_pass_at_k", ROOT / "scripts" / "analyze_model_pass_at_k.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ModelPassAtKAnalysisTest(unittest.TestCase):
    def test_merges_source_samples_into_one_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text(
                json.dumps({"instance_id": "a", "dataset": "mbppplus", "benchmark_resolved": True}) + "\n"
                + json.dumps({"instance_id": "b", "dataset": "mbppplus", "benchmark_resolved": False}) + "\n"
            )
            rows = []
            for sample_index in (1, 2):
                rows.extend(
                    [
                        {"instance_id": "a", "dataset": "mbppplus", "sample_index": sample_index, "benchmark_resolved": False},
                        {"instance_id": "b", "dataset": "mbppplus", "sample_index": sample_index, "benchmark_resolved": sample_index == 2},
                    ]
                )
            second.write_text("".join(json.dumps(row) + "\n" for row in rows))
            merged = MODULE.merge_model_rollouts("model", [first, second], 3)

        summary = MODULE.summarize_model("model", merged, 3)
        self.assertEqual([row["sample_index"] for row in merged], [1, 1, 2, 2, 3, 3])
        self.assertEqual(summary["complete_task_count"], 2)
        self.assertAlmostEqual(summary["pass_at_k"]["pass@1"], 1 / 3)
        self.assertAlmostEqual(summary["pass_at_k"]["pass@2"], 2 / 3)
        self.assertEqual(summary["pass_at_k"]["pass@3"], 1.0)

    def test_rejects_mismatched_task_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, task in enumerate(("a", "a", "b"), start=1):
                path = root / f"{index}.jsonl"
                path.write_text(json.dumps({"instance_id": task, "dataset": "demo", "benchmark_resolved": True}) + "\n")
                paths.append(path)
            with self.assertRaises(ValueError):
                MODULE.merge_model_rollouts("model", paths, 3)


if __name__ == "__main__":
    unittest.main()
