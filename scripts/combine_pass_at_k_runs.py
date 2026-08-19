#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from code_agent_baseline.pass_at_k import summarize_pass_at_k  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine one legacy Pass@1 run with new independent samples.")
    parser.add_argument("--legacy-dir", default="outputs/deepseek_v4_flash_eval90")
    parser.add_argument("--current-dir", default="outputs/deepseek_v4_flash_eval90_pass5_v1")
    parser.add_argument("--current-sample", action="append", type=int, default=[])
    parser.add_argument("--output-dir", default="outputs/deepseek_v4_flash_eval90_compatibility_pass3_v1")
    parser.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy_dir = resolve(args.legacy_dir)
    current_dir = resolve(args.current_dir)
    output_dir = resolve(args.output_dir)
    current_samples = sorted(set(args.current_sample or [1, 2]))
    expected_samples = 1 + len(current_samples)

    legacy_results = keyed(read_jsonl(legacy_dir / "final_results.jsonl"))
    legacy_trajectories = keyed(read_jsonl(legacy_dir / "selected_trajectories.jsonl"))
    if set(legacy_results) != set(legacy_trajectories):
        raise ValueError("Legacy result and trajectory task sets differ")

    sample_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for instance_id in sorted(legacy_results):
        result = legacy_results[instance_id]
        sample_rows.append(
            sample_row(
                result,
                sample_index=1,
                source_sample_index=0,
                cohort="legacy",
                result_source=legacy_dir / "final_results.jsonl",
                trajectory_source=legacy_dir / "selected_trajectories.jsonl",
            )
        )
        trajectory_rows.append(
            trajectory_row(
                legacy_trajectories[instance_id],
                sample_index=1,
                source_sample_index=0,
                cohort="legacy",
            )
        )

    for combined_index, current_sample in enumerate(current_samples, start=2):
        results = current_sample_results(current_dir, current_sample)
        missing = set(legacy_results) - set(results)
        extra = set(results) - set(legacy_results)
        if extra:
            raise ValueError(f"Current sample {current_sample} has unexpected tasks: {sorted(extra)[:3]}")
        if args.require_complete and missing:
            raise RuntimeError(
                f"Current sample {current_sample} is incomplete: {len(results)}/90 valid tasks, "
                f"missing {len(missing)}"
            )
        for instance_id, result in sorted(results.items()):
            trajectory_path = Path(result["trajectory_path"])
            trajectory = read_jsonl(trajectory_path)
            if len(trajectory) != 1:
                raise ValueError(f"Expected one trajectory in {trajectory_path}")
            sample_rows.append(
                sample_row(
                    result,
                    sample_index=combined_index,
                    source_sample_index=current_sample,
                    cohort="current",
                    result_source=Path(result["attempt_dir"]) / "sample_result.json",
                    trajectory_source=trajectory_path,
                )
            )
            trajectory_rows.append(
                trajectory_row(
                    trajectory[0],
                    sample_index=combined_index,
                    source_sample_index=current_sample,
                    cohort="current",
                )
            )

    summary = summarize_pass_at_k(sample_rows, expected_samples=expected_samples, ks=range(1, expected_samples + 1))
    summary.update(extra_summary(sample_rows, expected_samples))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "sample_results.jsonl", sample_rows)
    write_jsonl(output_dir / "trajectories.jsonl", trajectory_rows)
    write_jsonl(output_dir / "task_pass_at_k.jsonl", summary["tasks"])
    write_json(output_dir / "pass_at_k_summary.json", summary)

    manifest = {
        "format": "lottie_compatibility_pass_at_k_v1",
        "metric_label": f"compatibility_pass@{expected_samples}",
        "strict_same_configuration": False,
        "compatibility_note": (
            "The legacy sample and current samples use the same task IDs, model family, tools, and verifier semantics, "
            "but differ in harness revision and stable workspace-label prompting. Report as compatibility Pass@k, not "
            "an official same-configuration Pass@k result."
        ),
        "expected_samples_per_task": expected_samples,
        "sample_mapping": [
            {"combined_sample_index": 1, "cohort": "legacy", "source_sample_index": 0},
            *[
                {"combined_sample_index": index, "cohort": "current", "source_sample_index": source}
                for index, source in enumerate(current_samples, start=2)
            ],
        ],
        "sources": {
            "legacy_summary": fingerprint(legacy_dir / "final_summary.json"),
            "legacy_results": fingerprint(legacy_dir / "final_results.jsonl"),
            "current_manifest": fingerprint(current_dir / "manifest.json"),
            **(
                {"current_recovery_audit": fingerprint(current_dir / "recovery_audit.json")}
                if (current_dir / "recovery_audit.json").is_file()
                else {}
            ),
        },
        "task_count_expected": len(legacy_results),
        "data_role": "evaluation_only",
        "exclude_from_training": True,
        "official_comparable": False,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "summary": summary}, ensure_ascii=False, indent=2))


def current_sample_results(root: Path, sample_index: int) -> dict[str, dict[str, Any]]:
    output = {}
    pattern = f"sample_{sample_index:02d}/tasks/*/*/sample_result.json"
    for path in root.glob(pattern):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("sample_valid") is not True:
            continue
        instance_id = str(row["instance_id"])
        if instance_id in output:
            raise ValueError(f"Duplicate task in current sample {sample_index}: {instance_id}")
        output[instance_id] = row
    return output


def sample_row(
    result: dict[str, Any],
    *,
    sample_index: int,
    source_sample_index: int,
    cohort: str,
    result_source: Path,
    trajectory_source: Path,
) -> dict[str, Any]:
    return {
        "instance_id": result["instance_id"],
        "dataset": normalize_dataset(result.get("dataset"), result["instance_id"]),
        "sample_index": sample_index,
        "source_sample_index": source_sample_index,
        "cohort": cohort,
        "benchmark_resolved": result.get("benchmark_resolved"),
        "agent_status": result.get("agent_status"),
        "verifier_status": result.get("verifier_status"),
        "result_source": str(result_source),
        "trajectory_source": str(trajectory_source),
        "data_role": "evaluation_only",
        "exclude_from_training": True,
    }


def trajectory_row(
    trajectory: dict[str, Any],
    *,
    sample_index: int,
    source_sample_index: int,
    cohort: str,
) -> dict[str, Any]:
    output = dict(trajectory)
    output["pass_at_k_sample"] = {
        "sample_index": sample_index,
        "source_sample_index": source_sample_index,
        "cohort": cohort,
        "data_role": "evaluation_only",
        "exclude_from_training": True,
    }
    return output


def extra_summary(rows: list[dict[str, Any]], expected_samples: int) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
        by_cohort[str(row["cohort"])].append(row)
    dataset_metrics = {}
    for dataset, items in sorted(by_dataset.items()):
        dataset_metrics[dataset] = summarize_pass_at_k(
            items,
            expected_samples=expected_samples,
            ks=range(1, expected_samples + 1),
        )["aggregate"]
    cohort_rollout_accuracy = {
        cohort: {
            "rollouts": len(items),
            "resolved": sum(item.get("benchmark_resolved") is True for item in items),
            "accuracy": (
                sum(item.get("benchmark_resolved") is True for item in items) / len(items) if items else None
            ),
        }
        for cohort, items in sorted(by_cohort.items())
    }
    current_items = by_cohort.get("current", [])
    current_sample_count = expected_samples - 1
    current_summary = summarize_pass_at_k(
        current_items,
        expected_samples=current_sample_count,
        ks=range(1, current_sample_count + 1),
    )
    current_by_dataset: dict[str, Any] = {}
    for dataset, items in sorted(by_dataset.items()):
        current_dataset_items = [item for item in items if item["cohort"] == "current"]
        current_by_dataset[dataset] = summarize_pass_at_k(
            current_dataset_items,
            expected_samples=current_sample_count,
            ks=range(1, current_sample_count + 1),
        )["aggregate"]
    return {
        "metric_label": f"compatibility_pass@{expected_samples}",
        "strict_same_configuration": False,
        "by_dataset": dataset_metrics,
        "cohort_rollout_accuracy": cohort_rollout_accuracy,
        "current_same_harness": {
            "same_harness_configuration": True,
            "strict_same_execution_policy": False,
            "execution_policy_note": (
                "Three current sample slots used a documented infrastructure recovery ceiling of six instead of "
                "three after a transient API outage; model, prompt, tools, and verifier settings were unchanged."
            ),
            "expected_samples_per_task": current_sample_count,
            "task_count": current_summary["task_count"],
            "complete_task_count": current_summary["complete_task_count"],
            "aggregate": current_summary["aggregate"],
            "by_dataset": current_by_dataset,
        },
        "valid_rollouts": len(rows),
        "resolved_rollouts": sum(row.get("benchmark_resolved") is True for row in rows),
    }


def normalize_dataset(value: Any, instance_id: str) -> str:
    if value in {"MBPP+", "mbppplus"} or instance_id.startswith("MBPP/"):
        return "MBPP+"
    if value in {"HumanEval+", "humanevalplus"} or instance_id.startswith("HumanEval/"):
        return "HumanEval+"
    return "SWE-smith"


def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        instance_id = str(row["instance_id"])
        if instance_id in output:
            raise ValueError(f"Duplicate task: {instance_id}")
        output[instance_id] = row
    return output


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fingerprint(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
