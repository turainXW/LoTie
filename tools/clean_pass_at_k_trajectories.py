#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.sft_masking import (  # noqa: E402
    NON_TRAINABLE_STATUSES,
    normalize_training_messages,
)


SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"LTAI[A-Za-z0-9]{12,}"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build versioned SFT/RL datasets from a Pass@K trajectory run without modifying raw artifacts."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--append-finish-if-resolved",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("The cleaned output directory must be outside the raw run directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = sorted(run_dir.glob("sample_*/tasks/*/*/sample_result.json"))
    if not sample_paths:
        raise ValueError(f"No sample results found under {run_dir}")

    sft_path = output_dir / "sft_success.jsonl"
    rl_path = output_dir / "rl_rollouts.jsonl"
    exclusions_path = output_dir / "sft_exclusions.jsonl"
    duplicates_path = output_dir / "duplicates.jsonl"

    counts: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    verifier_statuses: Counter[str] = Counter()
    exclusion_reasons: Counter[str] = Counter()
    seen_sft_hashes: dict[str, str] = {}
    message_counts: list[int] = []
    message_chars: list[int] = []
    assistant_turns: list[int] = []
    secret_redactions = 0

    with (
        sft_path.open("w", encoding="utf-8") as sft_file,
        rl_path.open("w", encoding="utf-8") as rl_file,
        exclusions_path.open("w", encoding="utf-8") as exclusions_file,
        duplicates_path.open("w", encoding="utf-8") as duplicates_file,
    ):
        for sample_path in sample_paths:
            sample = read_json(sample_path)
            counts["sample_results"] += 1
            dataset = str(sample.get("dataset") or "unknown")
            datasets[dataset] += 1

            if sample.get("sample_valid") is not True or not isinstance(sample.get("benchmark_resolved"), bool):
                write_jsonl(
                    exclusions_file,
                    exclusion_record(sample, sample_path, run_dir, "invalid_or_infrastructure_sample"),
                )
                counts["invalid_samples"] += 1
                exclusion_reasons["invalid_or_infrastructure_sample"] += 1
                continue

            counts["valid_samples"] += 1
            agent_status = str(sample.get("agent_status") or "unknown")
            verifier_status = str(sample.get("verifier_status") or "unknown")
            statuses[agent_status] += 1
            verifier_statuses[verifier_status] += 1

            trace_path = resolve_trace_path(sample, sample_path)
            messages_path = trace_path.parent / "messages_sft.jsonl"
            try:
                trace = read_single_jsonl(messages_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                reason = "missing_or_invalid_messages"
                write_jsonl(
                    exclusions_file,
                    exclusion_record(sample, sample_path, run_dir, reason, error=repr(exc)),
                )
                counts["missing_messages"] += 1
                exclusion_reasons[reason] += 1
                continue

            raw_messages = trace.get("messages")
            if not structurally_valid_messages(raw_messages):
                reason = "invalid_message_schema"
                write_jsonl(exclusions_file, exclusion_record(sample, sample_path, run_dir, reason))
                counts["invalid_message_schema"] += 1
                exclusion_reasons[reason] += 1
                continue

            cleaned_messages, redactions = sanitize_messages(
                raw_messages,
                attempt_dir=Path(str(sample.get("attempt_dir") or sample_path.parent)),
            )
            secret_redactions += redactions
            resolved = bool(sample["benchmark_resolved"])
            protocol_valid = True
            protocol_error = None
            try:
                normalized_messages, finish_appended = normalize_training_messages(
                    cleaned_messages,
                    resolved=resolved,
                    append_finish_if_resolved=bool(args.append_finish_if_resolved),
                )
            except ValueError as exc:
                protocol_valid = False
                protocol_error = str(exc)
                normalized_messages = cleaned_messages
                finish_appended = False

            trajectory_id = trajectory_identifier(sample)
            relative_sample_path = relative_string(sample_path, run_dir)
            chars = sum(len(message["content"]) for message in normalized_messages)
            assistants = sum(message["role"] == "assistant" for message in normalized_messages)
            message_counts.append(len(normalized_messages))
            message_chars.append(chars)
            assistant_turns.append(assistants)

            rl_record = {
                "format": "lottie_clean_rl_rollout_v1",
                "trajectory_id": trajectory_id,
                "instance_id": sample.get("instance_id"),
                "dataset": dataset,
                "sample_index": sample.get("sample_index"),
                "reward": 1.0 if resolved else 0.0,
                "benchmark_resolved": resolved,
                "agent_status": agent_status,
                "verifier_status": verifier_status,
                "protocol_valid": protocol_valid,
                "protocol_error": protocol_error,
                "patch_present": bool(sample.get("patch_present")),
                "patch_lines": int(sample.get("patch_lines") or 0),
                "tool_steps": int(sample.get("tool_steps") or 0),
                "usage": sample.get("usage") or {},
                "message_count": len(normalized_messages),
                "message_chars": chars,
                "assistant_turns": assistants,
                "source_sample": relative_sample_path,
                "messages": normalized_messages,
            }
            write_jsonl(rl_file, rl_record)
            counts["rl_rows"] += 1
            counts["rl_positive"] += int(resolved)
            counts["rl_negative"] += int(not resolved)

            exclusion_reason = sft_exclusion_reason(
                sample=sample,
                resolved=resolved,
                protocol_valid=protocol_valid,
            )
            if exclusion_reason is not None:
                write_jsonl(
                    exclusions_file,
                    exclusion_record(
                        sample,
                        sample_path,
                        run_dir,
                        exclusion_reason,
                        error=protocol_error,
                    ),
                )
                exclusion_reasons[exclusion_reason] += 1
                continue

            conversation_hash = hash_json(normalized_messages)
            kept_trajectory = seen_sft_hashes.get(conversation_hash)
            if kept_trajectory is not None:
                write_jsonl(
                    duplicates_file,
                    {
                        "format": "lottie_clean_duplicate_v1",
                        "conversation_sha256": conversation_hash,
                        "kept_trajectory_id": kept_trajectory,
                        "duplicate_trajectory_id": trajectory_id,
                        "instance_id": sample.get("instance_id"),
                        "sample_index": sample.get("sample_index"),
                    },
                )
                counts["exact_duplicates"] += 1
                exclusion_reasons["exact_duplicate"] += 1
                continue
            seen_sft_hashes[conversation_hash] = trajectory_id

            write_jsonl(
                sft_file,
                {
                    "format": "lottie_clean_sft_messages_v1",
                    "trajectory_id": trajectory_id,
                    "instance_id": sample.get("instance_id"),
                    "dataset": dataset,
                    "sample_index": sample.get("sample_index"),
                    "resolved": True,
                    "status": agent_status,
                    "data_role": "train",
                    "exclude_from_training": False,
                    "finish_appended": finish_appended,
                    "conversation_sha256": conversation_hash,
                    "message_count": len(normalized_messages),
                    "message_chars": chars,
                    "assistant_turns": assistants,
                    "source_sample": relative_sample_path,
                    "messages": normalized_messages,
                },
            )
            counts["sft_rows"] += 1
            counts["finish_appended"] += int(finish_appended)

    manifest = {
        "format": "lottie_clean_trajectory_manifest_v1",
        "source_run": str(run_dir),
        "source_manifest_sha256": file_sha256(run_dir / "manifest.json"),
        "source_sample_result_count": len(sample_paths),
        "collection_in_progress": (run_dir / "collector.pid").is_file() and counts["valid_samples"] < 1200,
        "counts": dict(sorted(counts.items())),
        "datasets": dict(sorted(datasets.items())),
        "agent_statuses": dict(sorted(statuses.items())),
        "verifier_statuses": dict(sorted(verifier_statuses.items())),
        "sft_exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "secret_redactions": secret_redactions,
        "lengths": {
            "message_count": min_max(message_counts),
            "message_chars": min_max(message_chars),
            "assistant_turns": min_max(assistant_turns),
        },
        "policy": {
            "raw_run_is_read_only": True,
            "sft": "verified success, train split, canonical JSON tool calls, exact duplicates removed",
            "rl": "all verifier-valid train rollouts with binary reward",
            "non_trainable_statuses": sorted(NON_TRAINABLE_STATUSES),
            "absolute_paths_sanitized": True,
            "known_secret_shapes_redacted": True,
            "token_length_filter": "deferred until target tokenizer masking",
        },
        "outputs": {
            "sft_success": str(sft_path),
            "rl_rollouts": str(rl_path),
            "sft_exclusions": str(exclusions_path),
            "duplicates": str(duplicates_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def resolve_trace_path(sample: dict[str, Any], sample_path: Path) -> Path:
    configured = sample.get("trajectory_path")
    if configured:
        return Path(str(configured))
    attempt_index = int(sample.get("attempt_index") or 1)
    return sample_path.parent / f"attempt_{attempt_index:02d}" / "rollout" / "openhands_patch_rollout.jsonl"


def sft_exclusion_reason(*, sample: dict[str, Any], resolved: bool, protocol_valid: bool) -> str | None:
    if sample.get("exclude_from_training") or sample.get("data_role") == "evaluation_only":
        return "evaluation_excluded"
    if str(sample.get("agent_status") or "") in NON_TRAINABLE_STATUSES:
        return "non_trainable_status"
    if not resolved:
        return "verifier_unresolved"
    if not protocol_valid:
        return "invalid_tool_protocol"
    return None


def sanitize_messages(messages: list[dict[str, Any]], *, attempt_dir: Path) -> tuple[list[dict[str, str]], int]:
    replacements = (
        (str(attempt_dir / "workspace"), "<workspace>"),
        (str(ROOT), "<harness_root>"),
        (str(Path.home()), "<home>"),
    )
    cleaned: list[dict[str, str]] = []
    redactions = 0
    for message in messages:
        content = str(message["content"])
        for source, target in replacements:
            if source:
                content = content.replace(source, target)
        for pattern in SECRET_PATTERNS:
            content, count = pattern.subn("<redacted-secret>", content)
            redactions += count
        cleaned.append({"role": str(message["role"]), "content": content})
    return cleaned, redactions


def structurally_valid_messages(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(message, dict)
            and message.get("role") in {"system", "user", "assistant"}
            and isinstance(message.get("content"), str)
            for message in value
        )
    )


def exclusion_record(
    sample: dict[str, Any],
    sample_path: Path,
    run_dir: Path,
    reason: str,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "format": "lottie_clean_exclusion_v1",
        "instance_id": sample.get("instance_id"),
        "dataset": sample.get("dataset"),
        "sample_index": sample.get("sample_index"),
        "sample_valid": sample.get("sample_valid"),
        "benchmark_resolved": sample.get("benchmark_resolved"),
        "agent_status": sample.get("agent_status") or sample.get("status"),
        "verifier_status": sample.get("verifier_status"),
        "reason": reason,
        "error": error,
        "source_sample": relative_string(sample_path, run_dir),
    }


def trajectory_identifier(sample: dict[str, Any]) -> str:
    return f"{sample.get('dataset')}:{sample.get('instance_id')}:sample_{int(sample.get('sample_index') or 0):02d}"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_single_jsonl(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"Expected exactly one JSON object in {path}, found {len(rows)}")
    return rows[0]


def write_jsonl(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def relative_string(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def min_max(values: list[int]) -> dict[str, int] | None:
    return {"min": min(values), "max": max(values)} if values else None


if __name__ == "__main__":
    main()
