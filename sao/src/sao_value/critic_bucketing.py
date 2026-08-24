from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
from typing import Any, Iterable, Sequence


CRITIC_BUCKET_BOUNDARIES = (4096, 8192, 16384, 32768, 49152, 65536)
CRITIC_BATCH_PROFILES = {
    "critic_96gb": (8, 4, 2, 1, 1, 1),
    "critic_96gb_fast": (8, 8, 4, 2, 2, 2),
    "critic_96gb_verified": (8, 8, 4, 2, 2, 1),
    "critic_safe": (4, 2, 1, 1, 1, 1),
}


@dataclass(frozen=True)
class CriticBucketProfile:
    name: str
    boundaries: tuple[int, ...]
    microbatch_sizes: tuple[int, ...]
    target_trajectories_per_update: int

    def __post_init__(self) -> None:
        if not self.boundaries:
            raise ValueError("at least one bucket boundary is required")
        if len(self.boundaries) != len(self.microbatch_sizes):
            raise ValueError("microbatch sizes must match bucket boundaries")
        if tuple(sorted(self.boundaries)) != self.boundaries:
            raise ValueError("bucket boundaries must be strictly increasing")
        if len(set(self.boundaries)) != len(self.boundaries):
            raise ValueError("bucket boundaries must be unique")
        if any(size <= 0 for size in self.microbatch_sizes):
            raise ValueError("microbatch sizes must be positive")
        if self.target_trajectories_per_update <= 0:
            raise ValueError("target trajectories per update must be positive")
        if any(
            self.target_trajectories_per_update % size
            for size in self.microbatch_sizes
        ):
            raise ValueError(
                "every microbatch size must divide target trajectories per update"
            )

    def bucket_index(self, length: int) -> int:
        if length <= 0:
            raise ValueError(f"sequence length must be positive, found {length}")
        for index, boundary in enumerate(self.boundaries):
            if length <= boundary:
                return index
        raise ValueError(
            f"sequence length {length} exceeds maximum boundary {self.boundaries[-1]}"
        )

    def bucket_label(self, bucket_index: int) -> str:
        lower = self.boundaries[bucket_index - 1] if bucket_index else 0
        upper = self.boundaries[bucket_index]
        return f"{lower + 1}-{upper}"


@dataclass(frozen=True)
class CriticMicrobatchPlan:
    bucket_index: int
    bucket_label: str
    row_indexes: tuple[int, ...]


@dataclass(frozen=True)
class CriticUpdatePlan:
    microbatches: tuple[CriticMicrobatchPlan, ...]

    @property
    def trajectory_count(self) -> int:
        return sum(len(batch.row_indexes) for batch in self.microbatches)

    @property
    def bucket_indexes(self) -> tuple[int, ...]:
        return tuple(sorted({batch.bucket_index for batch in self.microbatches}))

    @property
    def bucket_label(self) -> str:
        labels = tuple(dict.fromkeys(batch.bucket_label for batch in self.microbatches))
        return labels[0] if len(labels) == 1 else "mixed:" + ",".join(labels)


def make_critic_bucket_profile(
    name: str = "critic_96gb",
    *,
    target_trajectories_per_update: int = 8,
) -> CriticBucketProfile:
    try:
        microbatch_sizes = CRITIC_BATCH_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(CRITIC_BATCH_PROFILES))
        raise ValueError(f"unknown critic batch profile {name!r}; choose from {choices}") from exc
    return CriticBucketProfile(
        name=name,
        boundaries=CRITIC_BUCKET_BOUNDARIES,
        microbatch_sizes=microbatch_sizes,
        target_trajectories_per_update=target_trajectories_per_update,
    )


class CriticLengthBucketPlanner:
    """Plan length-homogeneous optimizer updates with trajectory-equal weight.

    Full optimizer updates contain exactly ``target_trajectories_per_update``
    trajectories. Every microbatch is length-homogeneous, while bucket tails
    may share an optimizer update through sequential forward/backward passes.
    Consequently, only the final global update may be partial.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        profile: CriticBucketProfile,
        seed: int,
    ) -> None:
        self.lengths = tuple(int(length) for length in lengths)
        self.profile = profile
        self.seed = int(seed)
        for length in self.lengths:
            profile.bucket_index(length)

    def plan_epoch(self, epoch: int) -> list[CriticUpdatePlan]:
        rng = random.Random(self.seed + int(epoch))
        buckets: dict[int, list[int]] = defaultdict(list)
        for row_index, length in enumerate(self.lengths):
            buckets[self.profile.bucket_index(length)].append(row_index)

        updates: list[CriticUpdatePlan] = []
        bucket_tails: list[tuple[int, list[int]]] = []
        target = self.profile.target_trajectories_per_update
        for bucket_index in range(len(self.profile.boundaries)):
            indexes = buckets[bucket_index]
            rng.shuffle(indexes)
            microbatch_size = self.profile.microbatch_sizes[bucket_index]
            full_row_count = len(indexes) - len(indexes) % target
            for update_offset in range(0, full_row_count, target):
                update_indexes = indexes[update_offset : update_offset + target]
                microbatches = tuple(
                    CriticMicrobatchPlan(
                        bucket_index=bucket_index,
                        bucket_label=self.profile.bucket_label(bucket_index),
                        row_indexes=tuple(
                            update_indexes[offset : offset + microbatch_size]
                        ),
                    )
                    for offset in range(0, len(update_indexes), microbatch_size)
                )
                updates.append(CriticUpdatePlan(microbatches=microbatches))
            tail = indexes[full_row_count:]
            if tail:
                bucket_tails.append((bucket_index, tail))

        rng.shuffle(bucket_tails)
        tail_microbatches: list[CriticMicrobatchPlan] = []
        tail_count = 0
        for bucket_index, indexes in bucket_tails:
            microbatch_size = self.profile.microbatch_sizes[bucket_index]
            cursor = 0
            while cursor < len(indexes):
                capacity = target - tail_count
                take = min(microbatch_size, capacity, len(indexes) - cursor)
                tail_microbatches.append(
                    CriticMicrobatchPlan(
                        bucket_index=bucket_index,
                        bucket_label=self.profile.bucket_label(bucket_index),
                        row_indexes=tuple(indexes[cursor : cursor + take]),
                    )
                )
                cursor += take
                tail_count += take
                if tail_count == target:
                    updates.append(CriticUpdatePlan(microbatches=tuple(tail_microbatches)))
                    tail_microbatches = []
                    tail_count = 0
        if tail_microbatches:
            updates.append(CriticUpdatePlan(microbatches=tuple(tail_microbatches)))
        rng.shuffle(updates)
        self._validate_plan(updates)
        return updates

    def _validate_plan(self, updates: Sequence[CriticUpdatePlan]) -> None:
        observed = [
            row_index
            for update in updates
            for microbatch in update.microbatches
            for row_index in microbatch.row_indexes
        ]
        if sorted(observed) != list(range(len(self.lengths))):
            raise AssertionError("bucket plan must contain every row exactly once")
        for update in updates:
            for microbatch in update.microbatches:
                expected_bucket = microbatch.bucket_index
                if len(microbatch.row_indexes) > self.profile.microbatch_sizes[expected_bucket]:
                    raise AssertionError("microbatch exceeds its configured size")
                if any(
                    self.profile.bucket_index(self.lengths[index]) != expected_bucket
                    for index in microbatch.row_indexes
                ):
                    raise AssertionError("microbatch mixes length buckets")
            if update.trajectory_count > self.profile.target_trajectories_per_update:
                raise AssertionError("optimizer update exceeds target trajectory count")

    def updates_per_epoch(self) -> int:
        target = self.profile.target_trajectories_per_update
        return math.ceil(len(self.lengths) / target)

    def summary(self, plans: Iterable[CriticUpdatePlan] | None = None) -> dict[str, Any]:
        counts = Counter(self.profile.bucket_index(length) for length in self.lengths)
        payload: dict[str, Any] = {
            "profile": self.profile.name,
            "target_trajectories_per_update": self.profile.target_trajectories_per_update,
            "rows": len(self.lengths),
            "updates_per_epoch": self.updates_per_epoch(),
            "buckets": [],
        }
        for bucket_index, boundary in enumerate(self.profile.boundaries):
            count = counts[bucket_index]
            microbatch_size = self.profile.microbatch_sizes[bucket_index]
            payload["buckets"].append(
                {
                    "bucket_index": bucket_index,
                    "label": self.profile.bucket_label(bucket_index),
                    "max_tokens": boundary,
                    "rows": count,
                    "microbatch_size": microbatch_size,
                    "microbatches_per_full_update": (
                        self.profile.target_trajectories_per_update // microbatch_size
                    ),
                    "target_sized_row_groups": math.ceil(
                        count / self.profile.target_trajectories_per_update
                    )
                    if count
                    else 0,
                }
            )
        if plans is not None:
            plan_list = list(plans)
            payload["planned_updates"] = len(plan_list)
            payload["partial_updates"] = sum(
                update.trajectory_count
                < self.profile.target_trajectories_per_update
                for update in plan_list
            )
        return payload
