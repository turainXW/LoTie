import json
from pathlib import Path
import unittest

from sao_value.critic_bucketing import (
    CriticLengthBucketPlanner,
    make_critic_bucket_profile,
)


ROOT = Path(__file__).resolve().parents[1]


class RecordedDatasetTest(unittest.TestCase):
    def test_training_split_plans_exactly_111_updates(self) -> None:
        samples_path = ROOT / "data" / "metadata" / "training" / "samples.jsonl"
        with samples_path.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        training_rows = [row for row in rows if row["split"] == "train"]

        self.assertEqual(len(rows), 988)
        self.assertEqual(len(training_rows), 888)
        profile = make_critic_bucket_profile(
            "critic_96gb_verified", target_trajectories_per_update=8
        )
        planner = CriticLengthBucketPlanner(
            [int(row["num_tokens"]) for row in training_rows],
            profile=profile,
            seed=20260823,
        )
        plans = planner.plan_epoch(1)

        self.assertEqual(len(plans), 111)
        self.assertTrue(all(update.trajectory_count == 8 for update in plans))

    def test_formal_holdout_has_20_rows_per_family(self) -> None:
        samples_path = ROOT / "data" / "metadata" / "holdout40" / "samples.jsonl"
        with samples_path.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        validation_rows = [row for row in rows if row["split"] == "validation"]

        self.assertEqual(len(rows), 398)
        self.assertEqual(len(validation_rows), 40)
        for family in ("function", "swesmith"):
            self.assertEqual(
                sum(row["dataset_family"] == family for row in validation_rows),
                20,
            )


if __name__ == "__main__":
    unittest.main()
