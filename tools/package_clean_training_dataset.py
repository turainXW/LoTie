#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.sft_masking import validate_json_tool_call  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a complete, audited training delivery from strict and salvage datasets."
    )
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--strict-dir", required=True)
    parser.add_argument("--salvage-dir", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_dir = Path(args.clean_dir).resolve()
    strict_dir = Path(args.strict_dir).resolve()
    salvage_dir = Path(args.salvage_dir).resolve()
    audit_dir = Path(args.audit_dir).resolve()
    report_path = Path(args.report).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    strict_rows = read_jsonl(strict_dir / "sft_strict.jsonl")
    salvage_rows = read_jsonl(salvage_dir / "sft_salvage.jsonl")
    audit_rows = read_jsonl(audit_dir / "trajectory_audit.jsonl")
    manual_rows = read_jsonl(salvage_dir / "manual_review.jsonl")
    rl_rows = read_jsonl(clean_dir / "rl_rollouts.jsonl")
    audit_by_id = {str(row["trajectory_id"]): row for row in audit_rows}

    validate_unique_ids(strict_rows, "strict")
    validate_unique_ids(salvage_rows, "salvage")
    strict_ids = {str(row["trajectory_id"]) for row in strict_rows}
    salvage_ids = {str(row["trajectory_id"]) for row in salvage_rows}
    if strict_ids & salvage_ids:
        raise ValueError("Strict and salvage trajectory IDs overlap")
    if salvage_ids != set(audit_by_id):
        raise ValueError("Salvage rows do not match independently audited rows")
    if any(row.get("integrity_passed") is not True for row in audit_rows):
        raise ValueError("At least one salvage trajectory failed independent audit")
    if len(rl_rows) != 1200:
        raise ValueError(f"Expected 1200 RL rows, found {len(rl_rows)}")

    output_dir.mkdir(parents=True)
    data_dir = output_dir / "data"
    audit_out = output_dir / "audit"
    metadata_dir = output_dir / "metadata"
    provenance_dir = output_dir / "provenance"
    report_dir = output_dir / "report"
    for directory in (data_dir, audit_out, metadata_dir, provenance_dir, report_dir):
        directory.mkdir()

    primary_rows: list[dict[str, Any]] = []
    optional_b_rows: list[dict[str, Any]] = []
    quarantine_c_rows: list[dict[str, Any]] = []
    for row in strict_rows:
        primary_rows.append(annotate_sft(row, cleaning_class="strict", quality_tier="S", sample_weight=1.0))
    for row in salvage_rows:
        trajectory_id = str(row["trajectory_id"])
        tier = str(audit_by_id[trajectory_id]["quality_tier"])
        annotated = annotate_sft(
            row,
            cleaning_class="salvage",
            quality_tier=tier,
            sample_weight=1.0 if tier == "A" else 0.5 if tier == "B" else 0.0,
        )
        if tier == "A":
            primary_rows.append(annotated)
        elif tier == "B":
            optional_b_rows.append(annotated)
        elif tier == "C":
            annotated["exclude_from_training"] = True
            annotated["training_recommendation"] = "quarantine_ablation_only"
            quarantine_c_rows.append(annotated)
        else:
            raise ValueError(f"Unknown salvage quality tier: {tier}")

    validate_training_rows(primary_rows, allow_excluded=False)
    validate_training_rows(optional_b_rows, allow_excluded=False)
    validate_training_rows(quarantine_c_rows, allow_excluded=True)
    validate_unique_ids(primary_rows, "primary")
    if len(primary_rows) != 1069 or len(optional_b_rows) != 13 or len(quarantine_c_rows) != 4:
        raise ValueError(
            "Unexpected SFT partition sizes: "
            f"primary={len(primary_rows)} optional_b={len(optional_b_rows)} quarantine_c={len(quarantine_c_rows)}"
        )

    write_jsonl(data_dir / "sft_primary.jsonl", primary_rows)
    write_jsonl(data_dir / "sft_optional_tier_b.jsonl", optional_b_rows)
    write_jsonl(data_dir / "sft_quarantine_tier_c.jsonl", quarantine_c_rows)
    shutil.copy2(clean_dir / "rl_rollouts.jsonl", data_dir / "rl_rollouts.jsonl")

    copies = {
        audit_out / "sft_manual_review.jsonl": salvage_dir / "manual_review.jsonl",
        audit_out / "sft_exclusions.jsonl": clean_dir / "sft_exclusions.jsonl",
        audit_out / "duplicates.jsonl": clean_dir / "duplicates.jsonl",
        audit_out / "repair_audit.jsonl": salvage_dir / "repair_audit.jsonl",
        audit_out / "trajectory_audit.jsonl": audit_dir / "trajectory_audit.jsonl",
        audit_out / "salvage_audit_summary.json": audit_dir / "summary.json",
        metadata_dir / "clean_manifest.json": clean_dir / "manifest.json",
        metadata_dir / "strict_manifest.json": strict_dir / "manifest.json",
        metadata_dir / "salvage_manifest.json": salvage_dir / "manifest.json",
        metadata_dir / "raw_run_manifest.json": Path(read_json(clean_dir / "manifest.json")["source_run"]) / "manifest.json",
        metadata_dir / "pass_at_k_summary.json": Path(read_json(clean_dir / "manifest.json")["source_run"])
        / "pass_at_k_summary.json",
        report_dir / "CLEANING_REPORT_ZH.md": report_path,
    }
    for destination, source in copies.items():
        shutil.copy2(source, destination)
    shutil.copytree(salvage_dir / "raw_backup", provenance_dir / "salvage_raw_backup")

    tier_counts = Counter(str(row["quality_tier"]) for row in audit_rows)
    manifest: dict[str, Any] = {
        "format": "lottie_clean_training_delivery_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": read_json(clean_dir / "manifest.json")["source_run"],
        "source_rollouts": len(rl_rows),
        "verifier_resolved": sum(row.get("reward") == 1.0 for row in rl_rows),
        "verifier_unresolved": sum(row.get("reward") == 0.0 for row in rl_rows),
        "sft_partition": {
            "strict": len(strict_rows),
            "salvage_tier_a": tier_counts["A"],
            "salvage_tier_b": tier_counts["B"],
            "salvage_tier_c": tier_counts["C"],
            "manual_review": len(manual_rows),
            "primary_rows": len(primary_rows),
            "optional_tier_b_rows": len(optional_b_rows),
            "quarantine_tier_c_rows": len(quarantine_c_rows),
        },
        "training_policy": {
            "default_sft": "data/sft_primary.jsonl (strict plus independently audited Tier A salvage)",
            "optional_sft": "Append data/sft_optional_tier_b.jsonl with sample_weight=0.5",
            "quarantine": "Do not train on Tier C or manual-review rows by default",
            "rl": "data/rl_rollouts.jsonl contains all 1200 valid rollouts with binary verifier reward",
            "tokenization": "Not tokenized. Use the exact target-model tokenizer and assistant-only loss masking.",
        },
        "integrity": {
            "strict_tool_calls_validated": True,
            "salvage_independent_audit_passed": len(audit_rows),
            "salvage_independent_audit_failed": 0,
            "raw_salvage_backup_included": True,
            "raw_workspaces_included": False,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    file_rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        file_rows.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    write_jsonl(metadata_dir / "file_manifest.jsonl", file_rows)

    checksum_paths = [path for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"]
    checksum_text = "".join(
        f"{file_sha256(path)}  {path.relative_to(output_dir)}\n" for path in checksum_paths
    )
    (output_dir / "SHA256SUMS").write_text(checksum_text, encoding="ascii")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def annotate_sft(
    row: dict[str, Any],
    *,
    cleaning_class: str,
    quality_tier: str,
    sample_weight: float,
) -> dict[str, Any]:
    annotated = dict(row)
    annotated["format"] = "lottie_clean_sft_delivery_v1"
    annotated["cleaning_class"] = cleaning_class
    annotated["quality_tier"] = quality_tier
    annotated["sample_weight"] = sample_weight
    annotated["training_recommendation"] = "primary" if sample_weight == 1.0 else "optional_lower_weight"
    return annotated


def validate_unique_ids(rows: list[dict[str, Any]], label: str) -> None:
    ids = [str(row.get("trajectory_id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Missing or duplicate trajectory_id in {label}")


def validate_training_rows(rows: list[dict[str, Any]], *, allow_excluded: bool) -> None:
    for row_index, row in enumerate(rows):
        if row.get("resolved") is not True:
            raise ValueError(f"Unresolved SFT row at index {row_index}")
        if not allow_excluded and row.get("exclude_from_training") is True:
            raise ValueError(f"Excluded row in trainable SFT at index {row_index}")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Missing messages at row {row_index}")
        for message_index, message in enumerate(messages):
            if message.get("role") == "assistant":
                validate_json_tool_call(str(message.get("content") or ""), message_index=message_index)
        last = messages[-1]
        if last.get("role") != "assistant" or validate_json_tool_call(str(last.get("content") or ""))["tool_name"] != "finish":
            raise ValueError(f"SFT row {row_index} does not end with finish")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
