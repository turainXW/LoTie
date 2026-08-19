#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_agent_baseline.eval_analysis import DATASET_ORDER, analyze_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one or more Lottie sample_results.jsonl files.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Named result file; repeat for model or configuration comparisons.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--comparability-note",
        default="Runs may use different model, sampling, prompt, or harness configurations; compare as labeled.",
    )
    return parser.parse_args()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got: {value}")
    label, raw_path = value.rsplit("=", 1)
    if not label.strip() or not raw_path.strip():
        raise ValueError(f"Expected non-empty LABEL=PATH, got: {value}")
    return label.strip(), Path(raw_path.strip())


def markdown_report(analysis: dict, note: str) -> str:
    lines = ["# Code Agent Evaluation Analysis", "", f"> {note}", ""]
    lines.extend(["## Performance", "", "| Run | Samples | Resolved | Pass rate | Task coverage |", "|---|---:|---:|---:|---:|"])
    for run in analysis["runs"]:
        coverage = run["coverage"]
        lines.append(
            f"| {run['label']} | {run['total']} | {run['resolved']} | {run['pass_rate']:.2f}% | "
            f"{coverage['covered_tasks']}/{coverage['total_tasks']} ({coverage['rate']:.2f}%) |"
        )
    lines.extend(["", "## Dataset Breakdown", ""])
    lines.append("| Run | " + " | ".join(DATASET_ORDER) + " |")
    lines.append("|---|" + "|".join("---:" for _ in DATASET_ORDER) + "|")
    for run in analysis["runs"]:
        cells = []
        for dataset in DATASET_ORDER:
            row = run["by_dataset"].get(dataset)
            cells.append(f"{row['resolved']}/{row['total']} ({row['pass_rate']:.2f}%)" if row else "-")
        lines.append(f"| {run['label']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Failure Types", ""])
    failure_types = sorted(
        {
            outcome
            for run in analysis["runs"]
            for outcome in run["canonical_outcomes"]
            if outcome != "resolved"
        }
    )
    lines.append("| Run | " + " | ".join(failure_types) + " |")
    lines.append("|---|" + "|".join("---:" for _ in failure_types) + "|")
    for run in analysis["runs"]:
        cells = [str(run["canonical_outcomes"].get(outcome, 0)) for outcome in failure_types]
        lines.append(f"| {run['label']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Data Checks", ""])
    lines.append(f"- Same task set: `{analysis['comparison']['same_task_set']}`")
    lines.append(f"- Common tasks: `{analysis['comparison']['common_task_count']}`")
    for run in analysis["runs"]:
        quality = run["quality"]
        lines.append(
            f"- {run['label']}: duplicate task/round pairs={quality['duplicate_task_round_pairs']}, "
            f"invalid samples={quality['invalid_samples']}, excluded from training={quality['excluded_from_training']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_specs = [parse_run(value) for value in args.run]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = analyze_runs(run_specs)
    analysis["comparability_note"] = args.comparability_note
    json_path = output_dir / "evaluation_analysis.json"
    markdown_path = output_dir / "evaluation_analysis.md"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(analysis, args.comparability_note), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
