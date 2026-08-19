from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def pass_at_k(n: int, c: int, k: int) -> float:
    """Return the standard unbiased pass@k estimator for n samples and c successes."""
    if n < 1:
        raise ValueError("n must be positive")
    if c < 0 or c > n:
        raise ValueError("c must be between 0 and n")
    if k < 1 or k > n:
        raise ValueError("k must be between 1 and n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize_pass_at_k(
    sample_rows: Iterable[dict[str, Any]],
    *,
    expected_samples: int,
    ks: Iterable[int] = (1, 3, 5),
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in sample_rows:
        instance_id = str(row["instance_id"])
        by_task.setdefault(instance_id, []).append(row)

    requested_ks = sorted({int(k) for k in ks if 1 <= int(k) <= expected_samples})
    tasks: list[dict[str, Any]] = []
    for instance_id in sorted(by_task):
        rows = sorted(by_task[instance_id], key=lambda item: int(item["sample_index"]))
        sample_indexes = [int(row["sample_index"]) for row in rows]
        if len(sample_indexes) != len(set(sample_indexes)):
            raise ValueError(f"Duplicate sample index for {instance_id}")
        valid = [row for row in rows if row.get("benchmark_resolved") is not None]
        n = len(valid)
        c = sum(row.get("benchmark_resolved") is True for row in valid)
        metrics = {
            f"pass@{k}": pass_at_k(n, c, k) if n >= k else None
            for k in requested_ks
        }
        tasks.append(
            {
                "instance_id": instance_id,
                "dataset": rows[0].get("dataset"),
                "samples_expected": expected_samples,
                "samples_present": len(rows),
                "samples_valid": n,
                "successes": c,
                "complete": len(rows) == expected_samples and n == expected_samples,
                **metrics,
            }
        )

    aggregate = {}
    complete = [task for task in tasks if task["complete"]]
    for k in requested_ks:
        values = [float(task[f"pass@{k}"]) for task in complete if task[f"pass@{k}"] is not None]
        aggregate[f"pass@{k}"] = sum(values) / len(values) if values else None
    return {
        "tasks": tasks,
        "aggregate": aggregate,
        "task_count": len(tasks),
        "complete_task_count": len(complete),
        "expected_samples_per_task": expected_samples,
    }
