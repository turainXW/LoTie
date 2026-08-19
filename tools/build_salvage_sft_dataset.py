#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.sft_masking import (  # noqa: E402
    NON_TRAINABLE_STATUSES,
    normalize_training_messages,
    validate_json_tool_call,
)
from code_agent_baseline.tool_specs import TOOL_SPECS  # noqa: E402


SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"LTAI[A-Za-z0-9]{12,}"),
)
PARSER_RESULT_PREFIX = "EXECUTION RESULT of [parser]:"


@dataclass(frozen=True)
class Candidate:
    sample_path: Path
    sample: dict[str, Any]
    messages_path: Path
    trace_path: Path
    messages: list[dict[str, str]]
    invalid_indices: tuple[int, ...]

    @property
    def trajectory_id(self) -> str:
        return (
            f"{self.sample.get('dataset')}:{self.sample.get('instance_id')}:"
            f"sample_{int(self.sample.get('sample_index') or 0):02d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Back up and recover verifier-success trajectories containing malformed assistant tool calls."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--reuse-backup",
        help="Reuse and re-verify a raw_backup directory from an earlier immutable snapshot.",
    )
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
    validate_output_location(run_dir, output_dir)

    candidates = discover_candidates(run_dir)
    reused_backup = Path(args.reuse_backup).resolve() if args.reuse_backup else None
    if reused_backup is not None:
        frozen_ids = backup_candidate_ids(reused_backup)
        candidates = [candidate for candidate in candidates if candidate.trajectory_id in frozen_ids]
        if {candidate.trajectory_id for candidate in candidates} != frozen_ids:
            raise RuntimeError("Current source candidates do not match the frozen backup candidate index")
    if not candidates:
        raise ValueError("No verifier-success malformed-protocol trajectories were found")

    output_dir.mkdir(parents=True)
    snapshot_time = datetime.now(timezone.utc).isoformat()

    # Backups and their hashes are completed before any derived conversation is written.
    if reused_backup is None:
        backup_root = output_dir / "raw_backup"
        backup_root.mkdir()
        backup = back_up_sources(
            candidates,
            run_dir=run_dir,
            backup_root=backup_root,
            snapshot_time=snapshot_time,
        )
        (backup_root / "BACKUP_COMPLETE").write_text(
            json.dumps(
                {
                    "snapshot_id": backup["snapshot_id"],
                    "candidate_count": len(candidates),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        backup_root = reused_backup
        backup = verify_reused_backup(candidates, backup_root)

    salvage_path = output_dir / "sft_salvage.jsonl"
    audit_path = output_dir / "repair_audit.jsonl"
    manual_review_path = output_dir / "manual_review.jsonl"
    action_counts: Counter[str] = Counter()
    error_categories: Counter[str] = Counter()
    recovered = 0
    manual_review = 0
    finish_appended_count = 0
    removed_invalid_parser_pairs = 0
    removed_premature_finish_pairs = 0
    removed_protocol_rejection_pairs = 0
    secret_redactions = 0

    with (
        salvage_path.open("w", encoding="utf-8") as salvage_file,
        audit_path.open("w", encoding="utf-8") as audit_file,
        manual_review_path.open("w", encoding="utf-8") as review_file,
    ):
        for candidate in candidates:
            result = salvage_candidate(
                candidate,
                run_dir=run_dir,
                append_finish_if_resolved=bool(args.append_finish_if_resolved),
            )
            write_jsonl(audit_file, result["audit"])
            for event in result["audit"]["invalid_turns"]:
                action_counts[event["action"]] += 1
                error_categories[event["error_category"]] += 1
            for event in result["audit"]["premature_finish_turns"]:
                action_counts[event["action"]] += 1
            for event in result["audit"]["rejected_protocol_turns"]:
                action_counts[event["action"]] += 1

            if result["salvage"] is None:
                write_jsonl(review_file, result["manual_review"])
                manual_review += 1
                continue

            write_jsonl(salvage_file, result["salvage"])
            recovered += 1
            finish_appended_count += int(result["salvage"]["finish_appended"])
            removed_invalid_parser_pairs += int(result["salvage"]["removed_invalid_parser_pairs"])
            removed_premature_finish_pairs += int(result["salvage"]["removed_premature_finish_pairs"])
            removed_protocol_rejection_pairs += int(result["salvage"]["removed_protocol_rejection_pairs"])
            secret_redactions += int(result["salvage"]["secret_redactions"])

    manifest = {
        "format": "lottie_salvage_sft_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "snapshot_id": backup["snapshot_id"],
        "source_valid_rollouts_at_snapshot": backup["source_valid_rollouts"],
        "candidate_trajectories": len(candidates),
        "recovered_trajectories": recovered,
        "manual_review_trajectories": manual_review,
        "removed_invalid_parser_pairs": removed_invalid_parser_pairs,
        "removed_premature_finish_pairs": removed_premature_finish_pairs,
        "removed_protocol_rejection_pairs": removed_protocol_rejection_pairs,
        "finish_appended": finish_appended_count,
        "secret_redactions": secret_redactions,
        "action_counts": dict(sorted(action_counts.items())),
        "error_categories": dict(sorted(error_categories.items())),
        "isolation_policy": {
            "source_run_modified": False,
            "backup_completed_before_cleaning": True,
            "raw_backup_is_byte_exact": True,
            "raw_backup_reused": reused_backup is not None,
            "workspace_directories_in_backup": False,
            "source_metadata_preserved_in_raw_backup": True,
            "derived_metadata_stored_separately": True,
        },
        "repair_policy": {
            "automatic_rule": (
                "Drop only an invalid assistant message immediately followed by a parser rejection; "
                "or a premature finish immediately followed by completion-gate rejection. "
                "Also drop a schema-invalid/unknown tool call immediately followed by an explicit runtime protocol rejection. "
                "Both pairs have no environment side effect."
            ),
            "tool_calls_reconstructed_by_guessing": False,
            "unproven_cases": "manual_review",
            "all_remaining_assistant_messages": "canonical JSON tool calls",
            "resolved_without_finish": "append canonical finish after the final observation",
        },
        "outputs": {
            "salvage_sft": str(salvage_path),
            "repair_audit": str(audit_path),
            "manual_review": str(manual_review_path),
            "raw_backup": str(backup_root),
            "backup_file_manifest": str(backup_root / "file_manifest.jsonl"),
            "backup_candidate_index": str(backup_root / "candidate_index.jsonl"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def validate_output_location(run_dir: Path, output_dir: Path) -> None:
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("The salvage output must be outside the raw run directory")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {output_dir}")


def discover_candidates(run_dir: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for sample_path in sorted(run_dir.glob("sample_*/tasks/*/*/sample_result.json")):
        sample = read_json(sample_path)
        if sample.get("sample_valid") is not True or sample.get("benchmark_resolved") is not True:
            continue
        if sample.get("exclude_from_training") or sample.get("data_role") == "evaluation_only":
            continue
        if str(sample.get("agent_status") or "") in NON_TRAINABLE_STATUSES:
            continue

        trace_path = resolve_trace_path(sample, sample_path)
        messages_path = trace_path.parent / "messages_sft.jsonl"
        trace = read_single_jsonl(messages_path)
        messages = trace.get("messages")
        if not structurally_valid_messages(messages):
            continue
        typed_messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ]
        invalid_indices = tuple(
            index
            for index, message in enumerate(typed_messages)
            if message["role"] == "assistant" and not is_valid_tool_call(message["content"])
        )
        if invalid_indices:
            candidates.append(
                Candidate(
                    sample_path=sample_path,
                    sample=sample,
                    messages_path=messages_path,
                    trace_path=trace_path,
                    messages=typed_messages,
                    invalid_indices=invalid_indices,
                )
            )
    return candidates


def back_up_sources(
    candidates: list[Candidate],
    *,
    run_dir: Path,
    backup_root: Path,
    snapshot_time: str,
) -> dict[str, Any]:
    file_manifest_path = backup_root / "file_manifest.jsonl"
    candidate_index_path = backup_root / "candidate_index.jsonl"
    copied: dict[Path, dict[str, Any]] = {}

    top_level = [
        run_dir / "manifest.json",
        run_dir / "pass_at_k_summary.json",
        run_dir / "sample_results.jsonl",
        run_dir / "execution_policy_history.jsonl",
    ]
    with file_manifest_path.open("w", encoding="utf-8") as file_manifest:
        for path in top_level:
            if path.is_file():
                copied[path.resolve()] = copy_verified(path, run_dir, backup_root, file_manifest)

        with candidate_index_path.open("w", encoding="utf-8") as candidate_index:
            for candidate in candidates:
                artifacts = candidate_artifacts(candidate)
                artifact_records = []
                for path in artifacts:
                    resolved_path = path.resolve()
                    if resolved_path not in copied:
                        copied[resolved_path] = copy_verified(path, run_dir, backup_root, file_manifest)
                    artifact_records.append(copied[resolved_path])
                write_jsonl(
                    candidate_index,
                    {
                        "format": "lottie_salvage_raw_candidate_v1",
                        "trajectory_id": candidate.trajectory_id,
                        "instance_id": candidate.sample.get("instance_id"),
                        "dataset": candidate.sample.get("dataset"),
                        "sample_index": candidate.sample.get("sample_index"),
                        "invalid_assistant_indices": list(candidate.invalid_indices),
                        "source_sample_metadata_sha256": hash_json(candidate.sample),
                        "artifacts": artifact_records,
                    },
                )

    source_results = run_dir / "sample_results.jsonl"
    source_valid_rollouts = count_valid_rollouts(source_results)
    snapshot_id = hash_json(
        {
            "candidate_ids": [candidate.trajectory_id for candidate in candidates],
            "candidate_sample_hashes": [hash_json(candidate.sample) for candidate in candidates],
            "source_manifest_sha256": file_sha256(run_dir / "manifest.json"),
            "source_sample_results_backup_sha256": copied.get(source_results.resolve(), {}).get("sha256"),
        }
    )
    manifest = {
        "format": "lottie_salvage_raw_backup_manifest_v1",
        "snapshot_id": snapshot_id,
        "snapshot_time": snapshot_time,
        "source_run": str(run_dir),
        "candidate_count": len(candidates),
        "source_valid_rollouts": source_valid_rollouts,
        "file_count": len(copied),
        "total_bytes": sum(int(record["bytes"]) for record in copied.values()),
        "workspace_directories_excluded": True,
        "verification": "Every backup file SHA-256 matched its source at copy time.",
        "file_manifest": str(file_manifest_path),
        "candidate_index": str(candidate_index_path),
    }
    (backup_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def backup_candidate_ids(backup_root: Path) -> set[str]:
    index_path = backup_root / "candidate_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing backup candidate index: {index_path}")
    return {
        str(json.loads(line)["trajectory_id"])
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def verify_reused_backup(candidates: list[Candidate], backup_root: Path) -> dict[str, Any]:
    manifest = read_json(backup_root / "manifest.json")
    marker = read_json(backup_root / "BACKUP_COMPLETE")
    if marker.get("snapshot_id") != manifest.get("snapshot_id"):
        raise RuntimeError("Backup completion marker does not match its manifest")

    expected_metadata = {
        str(row["trajectory_id"]): str(row["source_sample_metadata_sha256"])
        for row in read_jsonl(backup_root / "candidate_index.jsonl")
    }
    current_metadata = {
        candidate.trajectory_id: hash_json(candidate.sample)
        for candidate in candidates
    }
    if current_metadata != expected_metadata:
        raise RuntimeError("Current source metadata does not match the frozen backup")

    file_rows = read_jsonl(backup_root / "file_manifest.jsonl")
    for row in file_rows:
        path = backup_root / str(row["backup_relative"])
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise RuntimeError(f"Backup artifact failed verification: {path}")
    if len(file_rows) != int(manifest.get("file_count") or -1):
        raise RuntimeError("Backup file count does not match its manifest")
    return manifest


def candidate_artifacts(candidate: Candidate) -> list[Path]:
    attempt_dir = Path(str(candidate.sample.get("attempt_dir") or candidate.trace_path.parents[1]))
    configured_verifier = candidate.sample.get("verifier_path")
    paths = [
        candidate.sample_path,
        candidate.messages_path,
        candidate.trace_path,
        attempt_dir / "runner.log",
        attempt_dir / "verifier_result.jsonl",
        attempt_dir / "patch.diff",
        candidate.trace_path.parent / "summary.json",
    ]
    if configured_verifier:
        paths.append(Path(str(configured_verifier)))
    raw_trace_dir = candidate.trace_path.parent / "raw_traces"
    if raw_trace_dir.is_dir():
        paths.extend(sorted(raw_trace_dir.glob("*.json")))
    return sorted({path.resolve() for path in paths if path.is_file()})


def copy_verified(source: Path, run_dir: Path, backup_root: Path, handle: Any) -> dict[str, Any]:
    source = source.resolve()
    try:
        relative = source.relative_to(run_dir)
    except ValueError:
        relative = Path("external") / f"{hashlib.sha256(str(source).encode()).hexdigest()[:12]}_{source.name}"
    destination = backup_root / "files" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(source)
    shutil.copy2(source, destination)
    destination_hash = file_sha256(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"Backup checksum mismatch for {source}")
    record = {
        "format": "lottie_salvage_backup_file_v1",
        "source_relative": relative_string(source, run_dir),
        "backup_relative": str(destination.relative_to(backup_root)),
        "bytes": source.stat().st_size,
        "sha256": source_hash,
    }
    write_jsonl(handle, record)
    return record


def salvage_candidate(
    candidate: Candidate,
    *,
    run_dir: Path,
    append_finish_if_resolved: bool,
) -> dict[str, Any]:
    invalid_turns: list[dict[str, Any]] = []
    remove_indices: set[int] = set()
    recoverable = True

    premature_finish_turns = find_premature_finish_turns(candidate.messages)
    for event in premature_finish_turns:
        remove_indices.update({event["assistant_index"], event["feedback_index"]})

    rejected_protocol_turns = find_rejected_protocol_turns(candidate.messages)
    for event in rejected_protocol_turns:
        remove_indices.update({event["assistant_index"], event["feedback_index"]})

    for index in candidate.invalid_indices:
        message = candidate.messages[index]
        error = validation_error(message["content"], index)
        parser_index = index + 1
        parser_feedback = (
            parser_index < len(candidate.messages)
            and candidate.messages[parser_index]["role"] == "user"
            and candidate.messages[parser_index]["content"].startswith(PARSER_RESULT_PREFIX)
        )
        recovery_index, recovery_tool = next_valid_assistant(candidate.messages, parser_index + 1)
        if parser_feedback:
            action = "drop_invalid_assistant_and_parser_feedback"
            rationale_zh = (
                f"assistant 消息 {index} 为{error_category_zh(classify_error(message['content'], error))}；"
                f"消息 {parser_index} 是 parser 拒绝反馈，未执行工具且未改变环境。"
                + (
                    f"删除这两条无副作用消息，保留消息 {recovery_index} 的 {recovery_tool} 重试。"
                    if recovery_index is not None
                    else "删除这两条无副作用消息，后续有效工具链保持不变。"
                )
            )
            remove_indices.update({index, parser_index})
        else:
            action = "manual_review_unpaired_invalid_assistant"
            rationale_zh = (
                f"assistant 消息 {index} 无法解析，且下一条不是明确的 parser 拒绝反馈；"
                "不能证明该消息没有环境副作用，因此不自动修改。"
            )
            recoverable = False

        invalid_turns.append(
            {
                "assistant_index": index,
                "error": error,
                "error_category": classify_error(message["content"], error),
                "content_chars": len(message["content"]),
                "content_sha256": text_sha256(message["content"]),
                "content_preview": message["content"][:500],
                "parser_feedback_index": parser_index if parser_feedback else None,
                "parser_feedback_sha256": (
                    text_sha256(candidate.messages[parser_index]["content"]) if parser_feedback else None
                ),
                "recovery_assistant_index": recovery_index,
                "recovery_tool_name": recovery_tool,
                "action": action,
                "environment_side_effect": False if parser_feedback else "unknown",
                "analysis_zh": rationale_zh,
            }
        )

    source_sample = relative_string(candidate.sample_path, run_dir)
    original_hash = hash_json(candidate.messages)
    base_audit = {
        "format": "lottie_salvage_repair_audit_v1",
        "trajectory_id": candidate.trajectory_id,
        "instance_id": candidate.sample.get("instance_id"),
        "dataset": candidate.sample.get("dataset"),
        "sample_index": candidate.sample.get("sample_index"),
        "benchmark_resolved": candidate.sample.get("benchmark_resolved"),
        "agent_status": candidate.sample.get("agent_status"),
        "verifier_status": candidate.sample.get("verifier_status"),
        "source_sample": source_sample,
        "source_sample_metadata_sha256": hash_json(candidate.sample),
        "original_conversation_sha256": original_hash,
        "original_message_count": len(candidate.messages),
        "invalid_assistant_turn_count": len(candidate.invalid_indices),
        "invalid_turns": invalid_turns,
        "premature_finish_turn_count": len(premature_finish_turns),
        "premature_finish_turns": premature_finish_turns,
        "rejected_protocol_turn_count": len(rejected_protocol_turns),
        "rejected_protocol_turns": rejected_protocol_turns,
        "automatic_recovery_safe": recoverable,
    }

    if not recoverable:
        audit = {
            **base_audit,
            "status": "manual_review",
            "cleaned_conversation_sha256": None,
            "cleaned_message_count": None,
        }
        return {
            "audit": audit,
            "salvage": None,
            "manual_review": {
                "format": "lottie_salvage_manual_review_v1",
                "trajectory_id": candidate.trajectory_id,
                "source_sample": source_sample,
                "reason": "unpaired_invalid_assistant",
                "audit": audit,
            },
        }

    filtered = [message for index, message in enumerate(candidate.messages) if index not in remove_indices]
    sanitized, redactions = sanitize_messages(
        filtered,
        attempt_dir=Path(str(candidate.sample.get("attempt_dir") or candidate.sample_path.parent)),
    )
    try:
        cleaned, finish_appended = normalize_training_messages(
            sanitized,
            resolved=True,
            append_finish_if_resolved=append_finish_if_resolved,
        )
        validate_observation_chain(cleaned)
    except ValueError as exc:
        audit = {
            **base_audit,
            "status": "manual_review",
            "post_clean_validation_error": str(exc),
            "cleaned_conversation_sha256": None,
            "cleaned_message_count": None,
        }
        return {
            "audit": audit,
            "salvage": None,
            "manual_review": {
                "format": "lottie_salvage_manual_review_v1",
                "trajectory_id": candidate.trajectory_id,
                "source_sample": source_sample,
                "reason": "post_clean_validation_failed",
                "audit": audit,
            },
        }

    cleaned_hash = hash_json(cleaned)
    audit = {
        **base_audit,
        "status": "recovered",
        "removed_message_indices": sorted(remove_indices),
        "removed_invalid_parser_pairs": sum(
            event["action"] == "drop_invalid_assistant_and_parser_feedback" for event in invalid_turns
        ),
        "removed_premature_finish_pairs": len(premature_finish_turns),
        "removed_protocol_rejection_pairs": len(rejected_protocol_turns),
        "removed_turn_pairs": len(remove_indices) // 2,
        "finish_appended": finish_appended,
        "secret_redactions": redactions,
        "cleaned_conversation_sha256": cleaned_hash,
        "cleaned_message_count": len(cleaned),
        "metadata_mutated": False,
    }
    salvage = {
        "format": "lottie_salvage_sft_messages_v1",
        "trajectory_id": candidate.trajectory_id,
        "instance_id": candidate.sample.get("instance_id"),
        "dataset": candidate.sample.get("dataset"),
        "sample_index": candidate.sample.get("sample_index"),
        "resolved": True,
        "status": candidate.sample.get("agent_status"),
        "data_role": "train",
        "exclude_from_training": False,
        "source_sample": source_sample,
        "source_sample_metadata_sha256": hash_json(candidate.sample),
        "original_conversation_sha256": original_hash,
        "conversation_sha256": cleaned_hash,
        "original_message_count": len(candidate.messages),
        "message_count": len(cleaned),
        "assistant_turns": sum(message["role"] == "assistant" for message in cleaned),
        "removed_turn_pairs": len(remove_indices) // 2,
        "removed_invalid_parser_pairs": sum(
            event["action"] == "drop_invalid_assistant_and_parser_feedback" for event in invalid_turns
        ),
        "removed_premature_finish_pairs": len(premature_finish_turns),
        "removed_protocol_rejection_pairs": len(rejected_protocol_turns),
        "finish_appended": finish_appended,
        "secret_redactions": redactions,
        "repair_audit_key": candidate.trajectory_id,
        "messages": cleaned,
    }
    return {"audit": audit, "salvage": salvage, "manual_review": None}


def find_premature_finish_turns(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, message in enumerate(messages[:-1]):
        if message["role"] != "assistant":
            continue
        try:
            call = validate_json_tool_call(message["content"], message_index=index)
        except ValueError:
            continue
        if call["tool_name"] != "finish":
            continue
        feedback_index = index + 1
        feedback = messages[feedback_index]
        rejected = (
            feedback["role"] == "user"
            and feedback["content"].startswith("EXECUTION RESULT of [finish]:")
            and "Completion gate rejected finish" in feedback["content"]
            and any(item["role"] == "assistant" for item in messages[feedback_index + 1 :])
        )
        if not rejected:
            continue
        recovery_index, recovery_tool = next_valid_assistant(messages, feedback_index + 1)
        events.append(
            {
                "assistant_index": index,
                "feedback_index": feedback_index,
                "feedback_sha256": text_sha256(feedback["content"]),
                "recovery_assistant_index": recovery_index,
                "recovery_tool_name": recovery_tool,
                "action": "drop_premature_finish_and_gate_rejection",
                "environment_side_effect": False,
                "analysis_zh": (
                    f"assistant 消息 {index} 提前调用 finish，被 completion gate 在消息 {feedback_index} 拒绝；"
                    f"该调用未修改环境。删除这一对消息，保留消息 {recovery_index} 的 {recovery_tool} 补充检查。"
                ),
            }
        )
    return events


def find_rejected_protocol_turns(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, message in enumerate(messages[:-1]):
        if message["role"] != "assistant":
            continue
        try:
            call = validate_json_tool_call(message["content"], message_index=index)
        except ValueError:
            continue
        protocol_errors = validate_tool_protocol(call)
        if not protocol_errors:
            continue
        feedback_index = index + 1
        feedback = messages[feedback_index]
        expected_prefix = f"EXECUTION RESULT of [{call['tool_name']}]:"
        if (
            feedback["role"] != "user"
            or not feedback["content"].startswith(expected_prefix)
            or not is_explicit_protocol_rejection(feedback["content"])
        ):
            continue
        recovery_index, recovery_tool = next_valid_assistant(messages, feedback_index + 1)
        events.append(
            {
                "assistant_index": index,
                "feedback_index": feedback_index,
                "tool_name": call["tool_name"],
                "protocol_errors": protocol_errors,
                "assistant_sha256": text_sha256(message["content"]),
                "feedback_sha256": text_sha256(feedback["content"]),
                "recovery_assistant_index": recovery_index,
                "recovery_tool_name": recovery_tool,
                "action": "drop_rejected_protocol_call_and_feedback",
                "environment_side_effect": False,
                "analysis_zh": (
                    f"assistant 消息 {index} 的 {call['tool_name']} 调用不符合工具协议；"
                    f"消息 {feedback_index} 是 runtime 明确拒绝，工具未执行且环境未改变。"
                    f"删除这一对消息，保留消息 {recovery_index} 的 {recovery_tool} 后续调用。"
                ),
            }
        )
    return events


def validate_tool_protocol(call: dict[str, Any]) -> list[str]:
    tool_name = str(call["tool_name"])
    if tool_name not in TOOL_SPECS:
        return ["unknown_tool"]
    return validate_schema(call["arguments"], TOOL_SPECS[tool_name].parameters)


def validate_schema(value: Any, schema: dict[str, Any], path: str = "arguments") -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return [f"{path}:not_object"]
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name}:required")
        if schema.get("additionalProperties") is False:
            errors.extend(f"{path}.{name}:unexpected" for name in value if name not in properties)
        for name, item in value.items():
            if name in properties:
                errors.extend(validate_schema(item, properties[name], f"{path}.{name}"))
    elif expected == "array":
        if not isinstance(value, list):
            return [f"{path}:not_array"]
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            errors.append(f"{path}:too_short")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            errors.append(f"{path}:too_long")
        for item in value:
            errors.extend(validate_schema(item, schema.get("items", {}), path))
    elif expected == "string":
        if not isinstance(value, str):
            return [f"{path}:not_string"]
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            errors.append(f"{path}:too_short")
    elif expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        errors.append(f"{path}:not_integer")
    elif expected == "boolean" and not isinstance(value, bool):
        errors.append(f"{path}:not_boolean")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}:not_in_enum")
    return errors


def is_explicit_protocol_rejection(content: str) -> bool:
    markers = (
        "Tool not allowed:",
        "command is required",
        "path is required",
        "query is required",
        "repo_url is required",
        "targets must be a JSON list",
        "Unsupported editor command:",
        "Protocol reminder:",
    )
    return any(marker in content for marker in markers)


def validate_observation_chain(messages: list[dict[str, str]]) -> None:
    if len(messages) < 3 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError("Conversation must begin with system then user")
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        call = validate_json_tool_call(message["content"], message_index=index)
        protocol_errors = validate_tool_protocol(call)
        if protocol_errors:
            raise ValueError(
                f"Assistant tool call at message {index} violates the declared tool schema: "
                + ", ".join(protocol_errors)
            )
        if call["tool_name"] == "finish":
            if index != len(messages) - 1:
                raise ValueError(f"finish must be terminal at message {index}")
            continue
        if index + 1 >= len(messages) or messages[index + 1]["role"] != "user":
            raise ValueError(f"Assistant tool call at message {index} has no following observation")


def next_valid_assistant(messages: list[dict[str, str]], start: int) -> tuple[int | None, str | None]:
    for index in range(start, len(messages)):
        message = messages[index]
        if message["role"] != "assistant":
            continue
        try:
            call = validate_json_tool_call(message["content"], message_index=index)
        except ValueError:
            continue
        return index, str(call["tool_name"])
    return None, None


def validation_error(content: str, index: int) -> str:
    try:
        validate_json_tool_call(content, message_index=index)
    except ValueError as exc:
        return str(exc)
    raise AssertionError("validation_error called for a valid assistant message")


def classify_error(content: str, error: str) -> str:
    stripped = content.lstrip()
    if not stripped.startswith("{"):
        return "non_json_text"
    lowered = error.lower()
    if "invalid \\escape" in lowered:
        return "invalid_escape"
    if "unterminated string" in lowered:
        return "unterminated_string"
    if "invalid control character" in lowered:
        return "invalid_control_character"
    if "extra data" in lowered:
        return "extra_data"
    if "delimiter" in lowered:
        return "missing_delimiter"
    if "exactly tool_name and arguments" in lowered or "must be" in lowered:
        return "invalid_tool_schema"
    return "invalid_json"


def error_category_zh(category: str) -> str:
    return {
        "non_json_text": "非 JSON 文本",
        "invalid_escape": "非法转义 JSON",
        "unterminated_string": "字符串截断 JSON",
        "invalid_control_character": "包含非法控制字符的 JSON",
        "extra_data": "包含额外文本的 JSON",
        "missing_delimiter": "缺少分隔符的 JSON",
        "invalid_tool_schema": "工具字段结构错误的 JSON",
        "invalid_json": "无效 JSON",
    }.get(category, "无效 JSON")


def is_valid_tool_call(content: str) -> bool:
    try:
        validate_json_tool_call(content)
    except ValueError:
        return False
    return True


def sanitize_messages(messages: list[dict[str, str]], *, attempt_dir: Path) -> tuple[list[dict[str, str]], int]:
    replacements = (
        (str(attempt_dir / "workspace"), "<workspace>"),
        (str(ROOT), "<harness_root>"),
        (str(Path.home()), "<home>"),
    )
    cleaned: list[dict[str, str]] = []
    redactions = 0
    for message in messages:
        content = message["content"]
        for source, target in replacements:
            if source:
                content = content.replace(source, target)
        for pattern in SECRET_PATTERNS:
            content, count = pattern.subn("<redacted-secret>", content)
            redactions += count
        cleaned.append({"role": message["role"], "content": content})
    return cleaned, redactions


def resolve_trace_path(sample: dict[str, Any], sample_path: Path) -> Path:
    configured = sample.get("trajectory_path")
    if configured:
        return Path(str(configured))
    attempt_index = int(sample.get("attempt_index") or 1)
    return sample_path.parent / f"attempt_{attempt_index:02d}" / "rollout" / "openhands_patch_rollout.jsonl"


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


def count_valid_rollouts(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count += int(row.get("sample_valid") is True)
    return count


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_single_jsonl(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"Expected one JSON object in {path}, found {len(rows)}")
    return rows[0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def relative_string(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
