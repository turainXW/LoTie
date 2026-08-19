import unittest

from code_agent_baseline.pass_at_k import pass_at_k, summarize_pass_at_k


class PassAtKTest(unittest.TestCase):
    def test_standard_estimator(self) -> None:
        self.assertAlmostEqual(pass_at_k(5, 1, 1), 0.2)
        self.assertAlmostEqual(pass_at_k(5, 1, 3), 0.6)
        self.assertEqual(pass_at_k(5, 1, 5), 1.0)
        self.assertEqual(pass_at_k(5, 0, 5), 0.0)

    def test_summary_requires_valid_complete_samples_for_aggregate(self) -> None:
        rows = [
            {"instance_id": "a", "dataset": "demo", "sample_index": index, "benchmark_resolved": index == 1}
            for index in range(1, 6)
        ]
        rows.extend(
            [
                {"instance_id": "b", "dataset": "demo", "sample_index": 1, "benchmark_resolved": True},
                {"instance_id": "b", "dataset": "demo", "sample_index": 2, "benchmark_resolved": None},
            ]
        )

        summary = summarize_pass_at_k(rows, expected_samples=5)

        self.assertEqual(summary["complete_task_count"], 1)
        self.assertAlmostEqual(summary["aggregate"]["pass@1"], 0.2)
        self.assertTrue(summary["tasks"][0]["complete"])
        self.assertFalse(summary["tasks"][1]["complete"])

    def test_rejects_duplicate_sample_indexes(self) -> None:
        with self.assertRaises(ValueError):
            summarize_pass_at_k(
                [
                    {"instance_id": "a", "sample_index": 1, "benchmark_resolved": True},
                    {"instance_id": "a", "sample_index": 1, "benchmark_resolved": False},
                ],
                expected_samples=2,
            )


if __name__ == "__main__":
    unittest.main()
