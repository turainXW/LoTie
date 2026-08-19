from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DATASET_ALIASES = {
    "mbppplus": "MBPP+",
    "mbpp+": "MBPP+",
    "humanevalplus": "HumanEval+",
    "humaneval+": "HumanEval+",
    "swesmith_py": "SWE-smith",
    "swe-smith": "SWE-smith",
    "swesmith": "SWE-smith",
}
DATASET_ORDER = ("MBPP+", "HumanEval+", "SWE-smith")
INFRASTRUCTURE_STATUSES = {
    "infrastructure_blocked",
    "model_api_failure",
    "runner_error",
    "verifier_missing_result",
    "infrastructure_attempts_exhausted",
}


def load_results(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {source}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No result rows found in {source}")
    return rows


def normalize_dataset(value: Any) -> str:
    text = str(value or "unknown").strip()
    return DATASET_ALIASES.get(text.lower(), text)


def is_resolved(row: dict[str, Any]) -> bool:
    resolved = row.get("benchmark_resolved")
    if isinstance(resolved, bool):
        return resolved
    return str(row.get("verifier_status") or "").lower() == "resolved"


def canonical_outcome(row: dict[str, Any]) -> str:
    if is_resolved(row):
        return "resolved"

    agent = str(row.get("agent_status") or row.get("status") or "unknown").lower()
    verifier = str(row.get("verifier_status") or "unknown").lower()
    if row.get("sample_valid") is False or agent in INFRASTRUCTURE_STATUSES or verifier in INFRASTRUCTURE_STATUSES:
        return "infrastructure"
    if agent == "parse_error":
        return "protocol_parse_error"
    if agent == "context_overflow":
        return "context_overflow"
    if agent == "compile_failed":
        return "compile_failed"

    patch_present = row.get("patch_present")
    if patch_present is None:
        patch_present = bool(row.get("patch_lines") or row.get("edited_files"))
    if not patch_present:
        if agent == "max_steps" or verifier == "model_failure_max_steps":
            return "no_patch_max_steps"
        return "no_effective_edit"
    if verifier == "unresolved_tests_failed":
        return "tests_failed"
    if verifier in {"unresolved_no_edit", "model_failure_no_edit"}:
        return "no_effective_edit"
    if verifier in {"unresolved", "blocked_missing_patch"}:
        return "unresolved_patch"
    return verifier or "unknown"


def _rate(resolved: int, total: int) -> float:
    return round(100.0 * resolved / total, 2) if total else 0.0


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    resolved = sum(is_resolved(row) for row in materialized)
    datasets: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        subset = [row for row in materialized if normalize_dataset(row.get("dataset")) == dataset]
        if not subset:
            continue
        dataset_resolved = sum(is_resolved(row) for row in subset)
        datasets[dataset] = {
            "total": len(subset),
            "resolved": dataset_resolved,
            "pass_rate": _rate(dataset_resolved, len(subset)),
        }
    return {
        "total": len(materialized),
        "resolved": resolved,
        "pass_rate": _rate(resolved, len(materialized)),
        "by_dataset": datasets,
        "canonical_outcomes": dict(sorted(Counter(canonical_outcome(row) for row in materialized).items())),
        "agent_statuses": dict(
            sorted(
                Counter(str(row.get("agent_status") or row.get("status") or "unknown") for row in materialized).items()
            )
        ),
        "verifier_statuses": dict(
            sorted(Counter(str(row.get("verifier_status") or "unknown") for row in materialized).items())
        ),
        "patch_present": sum(bool(row.get("patch_present")) for row in materialized),
        "tool_steps": sum(int(row.get("tool_steps") or 0) for row in materialized),
        "mean_tool_steps": round(
            sum(int(row.get("tool_steps") or 0) for row in materialized) / len(materialized), 2
        )
        if materialized
        else 0.0,
    }


def summarize_run(label: str, source: str | Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        normalized["dataset"] = normalize_dataset(row.get("dataset"))
        normalized["sample_index"] = int(row.get("sample_index") or 1)
        normalized_rows.append(normalized)

    rounds = sorted({row["sample_index"] for row in normalized_rows})
    by_round = {
        str(round_index): summarize_rows(row for row in normalized_rows if row["sample_index"] == round_index)
        for round_index in rounds
    }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        by_task[str(row.get("instance_id") or "unknown")].append(row)
    success_distribution = Counter(sum(is_resolved(row) for row in task_rows) for task_rows in by_task.values())
    covered = sum(any(is_resolved(row) for row in task_rows) for task_rows in by_task.values())
    duplicate_pairs = len(normalized_rows) - len(
        {(str(row.get("instance_id")), row["sample_index"]) for row in normalized_rows}
    )
    return {
        "label": label,
        "source": str(Path(source).expanduser().resolve()),
        **summarize_rows(normalized_rows),
        "rounds": by_round,
        "task_count": len(by_task),
        "coverage": {
            "covered_tasks": covered,
            "total_tasks": len(by_task),
            "rate": _rate(covered, len(by_task)),
            "success_count_distribution": {
                str(count): task_count for count, task_count in sorted(success_distribution.items())
            },
        },
        "quality": {
            "duplicate_task_round_pairs": duplicate_pairs,
            "invalid_samples": sum(row.get("sample_valid") is False for row in normalized_rows),
            "excluded_from_training": sum(row.get("exclude_from_training") is True for row in normalized_rows),
        },
    }


def compare_runs(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    task_sets = []
    for summary in run_summaries:
        rows = load_results(summary["source"])
        task_sets.append({str(row.get("instance_id")) for row in rows})
    common = set.intersection(*task_sets) if task_sets else set()
    union = set.union(*task_sets) if task_sets else set()
    return {
        "run_count": len(run_summaries),
        "common_task_count": len(common),
        "union_task_count": len(union),
        "same_task_set": bool(task_sets) and all(task_set == task_sets[0] for task_set in task_sets[1:]),
    }


def analyze_runs(run_specs: Iterable[tuple[str, str | Path]]) -> dict[str, Any]:
    runs = [summarize_run(label, source, load_results(source)) for label, source in run_specs]
    return {
        "format": "lottie_eval_analysis_v1",
        "runs": runs,
        "comparison": compare_runs(runs),
    }
