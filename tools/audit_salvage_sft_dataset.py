#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from code_agent_baseline.tool_specs import TOOL_SPECS  # noqa: E402


PARSER_PREFIX = "EXECUTION RESULT of [parser]:"
OBSERVATION_RE = re.compile(r"^EXECUTION RESULT of \[([^\]]+)\]:")
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"LTAI[A-Za-z0-9]{12,}"),
)
ALLOWED_TOOLS = {
    "execute_bash",
    "str_replace_editor",
    "run_tests",
    "git_diff",
    "finish",
    "repo_context",
    "problem_search",
    "web_search",
    "download_repo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently audit a Lottie salvage SFT snapshot.")
    parser.add_argument("--salvage-dir", required=True)
    parser.add_argument("--backup-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    salvage_dir = Path(args.salvage_dir).resolve()
    backup_dir = Path(args.backup_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {output_dir}")
    output_dir.mkdir(parents=True)

    salvage_rows = keyed(read_jsonl(salvage_dir / "sft_salvage.jsonl"), "trajectory_id")
    all_repair_rows = keyed(read_jsonl(salvage_dir / "repair_audit.jsonl"), "trajectory_id")
    repair_rows = {
        trajectory_id: row
        for trajectory_id, row in all_repair_rows.items()
        if row.get("status") == "recovered"
    }
    candidate_rows = keyed(read_jsonl(backup_dir / "candidate_index.jsonl"), "trajectory_id")
    if set(salvage_rows) != set(repair_rows) or not set(salvage_rows).issubset(candidate_rows):
        raise ValueError("Salvage IDs do not match recovered repair audit IDs")

    backup_checks = verify_backup(backup_dir)
    reports: list[dict[str, Any]] = []
    tool_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    finding_counts: Counter[str] = Counter()

    for trajectory_id in sorted(salvage_rows):
        report = audit_trajectory(
            salvage_rows[trajectory_id],
            repair_rows[trajectory_id],
            candidate_rows[trajectory_id],
            backup_dir,
        )
        reports.append(report)
        tier_counts[report["quality_tier"]] += 1
        finding_counts.update(report["quality_flags"])
        tool_counts.update(report["tool_counts"])

    failed = [report for report in reports if not report["integrity_passed"]]
    summary = {
        "format": "lottie_salvage_independent_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "salvage_dir": str(salvage_dir),
        "backup_dir": str(backup_dir),
        "trajectory_count": len(reports),
        "candidate_count": len(candidate_rows),
        "manual_review_count": sum(row.get("status") == "manual_review" for row in all_repair_rows.values()),
        "integrity_passed": len(reports) - len(failed),
        "integrity_failed": len(failed),
        "backup": backup_checks,
        "quality_tiers": dict(sorted(tier_counts.items())),
        "quality_flags": dict(sorted(finding_counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "recommendation": {
            "tier_a": "Suitable for primary SFT after target-tokenizer length and loss-mask validation.",
            "tier_b": "Usable with lower sampling weight; contains a synthetic terminal finish or a rejected premature finish removal.",
            "tier_c": "Keep isolated for ablation/manual review because more than three failed parser turns were removed.",
        },
        "failed_trajectory_ids": [report["trajectory_id"] for report in failed],
    }

    write_jsonl(output_dir / "trajectory_audit.jsonl", reports)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def verify_backup(backup_dir: Path) -> dict[str, Any]:
    manifest = read_json(backup_dir / "manifest.json")
    marker = read_json(backup_dir / "BACKUP_COMPLETE")
    if marker.get("snapshot_id") != manifest.get("snapshot_id"):
        raise ValueError("Backup completion marker does not match manifest")
    rows = read_jsonl(backup_dir / "file_manifest.jsonl")
    failures: list[str] = []
    total_bytes = 0
    for row in rows:
        path = backup_dir / str(row["backup_relative"])
        total_bytes += int(row["bytes"])
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            failures.append(str(row["backup_relative"]))
    if len(rows) != int(manifest["file_count"]):
        failures.append("file_count_mismatch")
    if total_bytes != int(manifest["total_bytes"]):
        failures.append("total_bytes_mismatch")
    return {
        "snapshot_id": manifest["snapshot_id"],
        "files_checked": len(rows),
        "bytes_checked": total_bytes,
        "hash_failures": failures,
        "passed": not failures,
    }


def audit_trajectory(
    salvage: dict[str, Any],
    repair: dict[str, Any],
    candidate: dict[str, Any],
    backup_dir: Path,
) -> dict[str, Any]:
    artifacts = candidate["artifacts"]
    messages_path = artifact_path(artifacts, backup_dir, "/rollout/messages_sft.jsonl")
    sample_path = artifact_path(artifacts, backup_dir, "/sample_result.json")
    verifier_path = artifact_path(artifacts, backup_dir, "/verifier_result.jsonl")
    source_trace = read_single_jsonl(messages_path)
    original = source_trace.get("messages")
    if not isinstance(original, list):
        raise ValueError(f"Missing messages for {salvage['trajectory_id']}")
    sample = read_json(sample_path)
    cleaned = salvage["messages"]
    errors: list[str] = []

    check(hash_json(original) == repair["original_conversation_sha256"], "original_hash", errors)
    check(hash_json(sample) == repair["source_sample_metadata_sha256"], "sample_metadata_hash", errors)
    check(sample.get("sample_valid") is True, "sample_not_valid", errors)
    check(sample.get("benchmark_resolved") is True, "sample_not_resolved", errors)
    check(repair.get("status") == "recovered", "repair_not_recovered", errors)

    remove_indices = set(repair["removed_message_indices"])
    expected_indices: set[int] = set()
    for event in repair["invalid_turns"]:
        assistant_index = int(event["assistant_index"])
        parser_index = int(event["parser_feedback_index"])
        expected_indices.update({assistant_index, parser_index})
        check(original[assistant_index]["role"] == "assistant", "invalid_turn_not_assistant", errors)
        check(not valid_call(original[assistant_index]["content"]), "removed_turn_is_valid_json", errors)
        check(original[parser_index]["role"] == "user", "parser_feedback_not_user", errors)
        check(original[parser_index]["content"].startswith(PARSER_PREFIX), "missing_parser_rejection", errors)
        check(event.get("environment_side_effect") is False, "side_effect_not_false", errors)
    for event in repair["premature_finish_turns"]:
        assistant_index = int(event["assistant_index"])
        feedback_index = int(event["feedback_index"])
        expected_indices.update({assistant_index, feedback_index})
        call = parse_call(original[assistant_index]["content"], errors)
        check(call is not None and call["tool_name"] == "finish", "premature_turn_not_finish", errors)
        feedback = original[feedback_index]["content"]
        check("Completion gate rejected finish" in feedback, "finish_not_rejected", errors)
        check(event.get("environment_side_effect") is False, "finish_side_effect_not_false", errors)
    for event in repair.get("rejected_protocol_turns", []):
        assistant_index = int(event["assistant_index"])
        feedback_index = int(event["feedback_index"])
        expected_indices.update({assistant_index, feedback_index})
        check(original[assistant_index]["role"] == "assistant", "protocol_turn_not_assistant", errors)
        check(original[feedback_index]["role"] == "user", "protocol_feedback_not_user", errors)
        check(is_protocol_rejection(original[feedback_index]["content"]), "protocol_feedback_not_rejection", errors)
        check(event.get("environment_side_effect") is False, "protocol_side_effect_not_false", errors)
    check(remove_indices == expected_indices, "removed_indices_mismatch", errors)

    filtered = [message for index, message in enumerate(original) if index not in remove_indices]
    sanitized = sanitize(filtered, sample)
    original_finish = False
    terminal_finish_index = None
    for index in range(len(sanitized) - 1, -1, -1):
        if sanitized[index]["role"] != "assistant":
            continue
        call = parse_call(sanitized[index]["content"], errors)
        if call is not None and call["tool_name"] == "finish":
            original_finish = True
            terminal_finish_index = index
        break
    if terminal_finish_index is not None:
        expected_cleaned = sanitized[: terminal_finish_index + 1]
    else:
        expected_cleaned = list(sanitized)
        expected_cleaned.append(
            {"role": "assistant", "content": '{"tool_name":"finish","arguments":{}}'}
        )
    check(bool(repair["finish_appended"]) is (not original_finish), "finish_provenance_mismatch", errors)
    check(expected_cleaned == cleaned, "cleaned_messages_not_exact_transform", errors)
    check(hash_json(cleaned) == repair["cleaned_conversation_sha256"], "cleaned_audit_hash", errors)
    check(hash_json(cleaned) == salvage["conversation_sha256"], "cleaned_salvage_hash", errors)

    calls: list[dict[str, Any]] = []
    protocol_findings: list[dict[str, Any]] = []
    chain_errors: list[str] = []
    check(len(cleaned) >= 3 and cleaned[0]["role"] == "system" and cleaned[1]["role"] == "user", "invalid_prefix", chain_errors)
    for index, message in enumerate(cleaned):
        if message["role"] != "assistant":
            continue
        call = parse_call(message["content"], chain_errors)
        if call is None:
            continue
        calls.append(call)
        check(call["tool_name"] in ALLOWED_TOOLS, "unknown_tool", chain_errors)
        if call["tool_name"] in TOOL_SPECS:
            argument_errors = validate_schema(call["arguments"], TOOL_SPECS[call["tool_name"]].parameters)
            if argument_errors:
                chain_errors.append("invalid_tool_arguments")
                protocol_findings.append(
                    {
                        "message_index": index,
                        "tool_name": call["tool_name"],
                        "argument_errors": argument_errors,
                        "arguments": call["arguments"],
                        "observation_preview": (
                            cleaned[index + 1]["content"][:500]
                            if index + 1 < len(cleaned) and cleaned[index + 1]["role"] == "user"
                            else None
                        ),
                    }
                )
        if call["tool_name"] == "finish":
            check(index == len(cleaned) - 1, "finish_not_terminal", chain_errors)
            continue
        check(index + 1 < len(cleaned), "missing_observation", chain_errors)
        if index + 1 >= len(cleaned):
            continue
        observation = cleaned[index + 1]
        check(observation["role"] == "user", "observation_not_user", chain_errors)
        match = OBSERVATION_RE.match(observation["content"])
        check(match is not None, "observation_prefix_missing", chain_errors)
        if match is not None:
            check(match.group(1) == call["tool_name"], "observation_tool_mismatch", chain_errors)
        if is_protocol_rejection(observation["content"]):
            chain_errors.append("protocol_rejection_retained")
    errors.extend(chain_errors)

    verifier_rows = read_jsonl(verifier_path)
    verifier_resolved = any(
        row.get("benchmark_resolved") is True
        or row.get("resolved") is True
        or row.get("status") == "resolved"
        or row.get("verifier_status") == "resolved"
        for row in verifier_rows
    )
    check(verifier_resolved, "verifier_not_resolved", errors)
    check(not any(pattern.search(message["content"]) for pattern in SECRET_PATTERNS for message in cleaned), "secret_shape_found", errors)

    invalid_pairs = int(repair["removed_invalid_parser_pairs"])
    premature_pairs = int(repair["removed_premature_finish_pairs"])
    flags: list[str] = []
    if repair["finish_appended"]:
        flags.append("synthetic_finish")
    if premature_pairs:
        flags.append("premature_finish_removed")
    if invalid_pairs > 3:
        flags.append("high_repair_count")
    if invalid_pairs > 3:
        tier = "C"
    elif flags:
        tier = "B"
    else:
        tier = "A"
    counts = Counter(call["tool_name"] for call in calls)
    return {
        "format": "lottie_salvage_trajectory_audit_v1",
        "trajectory_id": salvage["trajectory_id"],
        "dataset": salvage["dataset"],
        "sample_index": salvage["sample_index"],
        "integrity_passed": not errors,
        "integrity_errors": sorted(set(errors)),
        "quality_tier": tier,
        "quality_flags": flags,
        "original_messages": len(original),
        "cleaned_messages": len(cleaned),
        "invalid_parser_pairs_removed": invalid_pairs,
        "premature_finish_pairs_removed": premature_pairs,
        "finish_provenance": "synthetic_from_verified_success" if repair["finish_appended"] else "original_model_output",
        "verifier_resolved": verifier_resolved,
        "protocol_findings": protocol_findings,
        "tool_counts": dict(sorted(counts.items())),
    }


def sanitize(messages: list[dict[str, str]], sample: dict[str, Any]) -> list[dict[str, str]]:
    attempt_dir = Path(str(sample.get("attempt_dir") or ""))
    replacements = (
        (str(attempt_dir / "workspace"), "<workspace>"),
        (str(ROOT), "<harness_root>"),
        (str(Path.home()), "<home>"),
    )
    cleaned: list[dict[str, str]] = []
    for message in messages:
        content = message["content"]
        for source, target in replacements:
            if source:
                content = content.replace(source, target)
        for pattern in SECRET_PATTERNS:
            content = pattern.sub("<redacted-secret>", content)
        cleaned.append({"role": message["role"], "content": content})
    return cleaned


def artifact_path(artifacts: list[dict[str, Any]], backup_dir: Path, suffix: str) -> Path:
    matches = [backup_dir / row["backup_relative"] for row in artifacts if row["source_relative"].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one artifact ending in {suffix}, found {len(matches)}")
    return matches[0]


def parse_call(content: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        errors.append("assistant_not_json")
        return None
    if not isinstance(value, dict) or set(value) != {"tool_name", "arguments"}:
        errors.append("assistant_schema_invalid")
        return None
    if not isinstance(value["tool_name"], str) or not isinstance(value["arguments"], dict):
        errors.append("assistant_schema_invalid")
        return None
    return value


def valid_call(content: str) -> bool:
    errors: list[str] = []
    return parse_call(content, errors) is not None


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
        for index, item in enumerate(value):
            errors.extend(validate_schema(item, schema.get("items", {}), f"{path}[{index}]"))
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


def is_protocol_rejection(content: str) -> bool:
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
    return content.startswith("EXECUTION RESULT of [") and any(marker in content for marker in markers)


def check(condition: bool, label: str, errors: list[str]) -> None:
    if not condition:
        errors.append(label)


def keyed(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result = {str(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key} values")
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_single_jsonl(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one JSONL row: {path}")
    return rows[0]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def hash_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
