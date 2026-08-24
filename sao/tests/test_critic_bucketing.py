import unittest

from sao_value.critic_bucketing import (
    CriticLengthBucketPlanner,
    make_critic_bucket_profile,
)


class CriticBucketingTest(unittest.TestCase):
    def test_verified_profile_preserves_every_trajectory_once(self) -> None:
        lengths = [3000] * 9 + [7000] * 7 + [12000] * 5 + [60000] * 3
        profile = make_critic_bucket_profile(
            "critic_96gb_verified", target_trajectories_per_update=8
        )
        planner = CriticLengthBucketPlanner(
            lengths, profile=profile, seed=20260823
        )

        plans = planner.plan_epoch(1)
        observed = [
            row_index
            for update in plans
            for microbatch in update.microbatches
            for row_index in microbatch.row_indexes
        ]

        self.assertEqual(sorted(observed), list(range(len(lengths))))
        self.assertEqual(len(plans), 3)
        self.assertTrue(all(update.trajectory_count <= 8 for update in plans))

    def test_verified_profile_uses_expected_microbatch_sizes(self) -> None:
        profile = make_critic_bucket_profile("critic_96gb_verified")

        self.assertEqual(profile.microbatch_sizes, (8, 8, 4, 2, 2, 1))
        self.assertEqual(profile.bucket_index(4096), 0)
        self.assertEqual(profile.bucket_index(4097), 1)
        self.assertEqual(profile.bucket_index(65536), 5)


if __name__ == "__main__":
    unittest.main()
