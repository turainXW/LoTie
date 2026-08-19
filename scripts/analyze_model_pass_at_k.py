#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_agent_baseline.eval_analysis import DATASET_ORDER, load_results, normalize_dataset
from code_agent_baseline.pass_at_k import summarize_pass_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple rollout files per model and report model-level Pass@1/2/3."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=PATH[,PATH...]",
        help="Model label and one or more JSONL files. Every source sample is treated as one rollout.",
    )
    parser.add_argument("--expected-rollouts", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comparability-note", default="")
    return parser.parse_args()


def parse_model(value: str) -> tuple[str, list[Path]]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH[,PATH...], got: {value}")
    label, raw_paths = value.rsplit("=", 1)
    paths = [Path(item.strip()) for item in raw_paths.split(",") if item.strip()]
    if not label.strip() or not paths:
        raise ValueError(f"Expected non-empty LABEL=PATH[,PATH...], got: {value}")
    return label.strip(), paths


def merge_model_rollouts(label: str, paths: list[Path], expected_rollouts: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    rollout_index = 0
    reference_tasks: set[str] | None = None
    for source in paths:
        rows = load_results(source)
        source_indexes = sorted({int(row.get("sample_index") or 1) for row in rows})
        for source_index in source_indexes:
            rollout_index += 1
            sample = [row for row in rows if int(row.get("sample_index") or 1) == source_index]
            task_ids = {str(row["instance_id"]) for row in sample}
            if len(task_ids) != len(sample):
                raise ValueError(f"Duplicate tasks in {source} sample {source_index}")
            if reference_tasks is None:
                reference_tasks = task_ids
            elif task_ids != reference_tasks:
                missing = sorted(reference_tasks - task_ids)
                extra = sorted(task_ids - reference_tasks)
                raise ValueError(
                    f"Task mismatch in {source} sample {source_index}: missing={missing[:3]}, extra={extra[:3]}"
                )
            for row in sample:
                normalized = dict(row)
                normalized.update(
                    {
                        "model_label": label,
                        "sample_index": rollout_index,
                        "source_sample_index": source_index,
                        "source_results": str(source.expanduser().resolve()),
                        "dataset": normalize_dataset(row.get("dataset")),
                    }
                )
                merged.append(normalized)
    if rollout_index != expected_rollouts:
        raise ValueError(f"{label} has {rollout_index} rollouts, expected {expected_rollouts}")
    return merged


def summarize_model(label: str, rows: list[dict[str, Any]], expected_rollouts: int) -> dict[str, Any]:
    overall = summarize_pass_at_k(rows, expected_samples=expected_rollouts, ks=range(1, expected_rollouts + 1))
    by_dataset = {}
    for dataset in DATASET_ORDER:
        subset = [row for row in rows if row["dataset"] == dataset]
        if not subset:
            continue
        summary = summarize_pass_at_k(
            subset,
            expected_samples=expected_rollouts,
            ks=range(1, expected_rollouts + 1),
        )
        by_dataset[dataset] = {
            "task_count": summary["task_count"],
            "complete_task_count": summary["complete_task_count"],
            "pass_at_k": summary["aggregate"],
            "success_distribution": dict(sorted(Counter(task["successes"] for task in summary["tasks"]).items())),
        }
    return {
        "label": label,
        "rollouts": expected_rollouts,
        "task_count": overall["task_count"],
        "complete_task_count": overall["complete_task_count"],
        "rollout_results": [
            {
                "sample_index": index,
                "resolved": sum(
                    row.get("benchmark_resolved") is True
                    for row in rows
                    if int(row["sample_index"]) == index
                ),
                "total": sum(1 for row in rows if int(row["sample_index"]) == index),
            }
            for index in range(1, expected_rollouts + 1)
        ],
        "pass_at_k": overall["aggregate"],
        "success_distribution": dict(sorted(Counter(task["successes"] for task in overall["tasks"]).items())),
        "by_dataset": by_dataset,
    }


def markdown_report(analysis: dict[str, Any]) -> str:
    ks = analysis["ks"]
    lines = ["# Model-level Pass@k Analysis", ""]
    if analysis.get("comparability_note"):
        lines.extend([f"> {analysis['comparability_note']}", ""])
    lines.extend(["## Overall", "", "| Model | " + " | ".join(f"Pass@{k}" for k in ks) + " |", "|---|" + "|".join("---:" for _ in ks) + "|"])
    for model in analysis["models"]:
        values = [f"{100 * model['pass_at_k'][f'pass@{k}']:.2f}%" for k in ks]
        lines.append(f"| {model['label']} | " + " | ".join(values) + " |")
    lines.extend(["", "## By Dataset", "", "| Model | Dataset | " + " | ".join(f"Pass@{k}" for k in ks) + " |", "|---|---|" + "|".join("---:" for _ in ks) + "|"])
    for model in analysis["models"]:
        for dataset in DATASET_ORDER:
            if dataset not in model["by_dataset"]:
                continue
            metrics = model["by_dataset"][dataset]["pass_at_k"]
            values = [f"{100 * metrics[f'pass@{k}']:.2f}%" for k in ks]
            lines.append(f"| {model['label']} | {dataset} | " + " | ".join(values) + " |")
    lines.extend(["", "## Rollouts", ""])
    for model in analysis["models"]:
        rounds = ", ".join(
            f"R{item['sample_index']}={item['resolved']}/{item['total']}" for item in model["rollout_results"]
        )
        lines.append(f"- {model['label']}: {rounds}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_rows = []
    models = []
    for spec in args.model:
        label, paths = parse_model(spec)
        rows = merge_model_rollouts(label, paths, args.expected_rollouts)
        merged_rows.extend(rows)
        models.append(summarize_model(label, rows, args.expected_rollouts))
    analysis = {
        "format": "lottie_model_pass_at_k_v1",
        "expected_rollouts": args.expected_rollouts,
        "ks": list(range(1, args.expected_rollouts + 1)),
        "comparability_note": args.comparability_note,
        "models": models,
    }
    (output_dir / "model_pass_at_k.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "model_pass_at_k.md").write_text(markdown_report(analysis), encoding="utf-8")
    (output_dir / "merged_samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged_rows), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "models": len(models), "samples": len(merged_rows)}))


if __name__ == "__main__":
    main()
