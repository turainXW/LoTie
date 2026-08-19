#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer


CANONICAL_TOOLS = {"execute_bash", "str_replace_editor", "run_tests", "git_diff", "finish"}
EDITOR_COMMANDS = {"view", "str_replace", "line_replace", "insert", "create", "undo_edit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize Lottie conversations and build explicit assistant-only labels for Qwen SFT."
    )
    parser.add_argument("--input", required=True, help="Clean SFT JSONL containing messages.")
    parser.add_argument("--model-path", required=True, help="Local Qwen model/tokenizer directory.")
    parser.add_argument("--output-dir", required=True, help="Hugging Face Dataset.save_to_disk output.")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--overlap-units", type=int, default=2)
    parser.add_argument(
        "--keep-invalid-tool-calls",
        action="store_true",
        help="Train malformed/unknown tool calls instead of masking them. Not recommended.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("messages"), list) or not row["messages"]:
                raise ValueError(f"{path}:{line_number}: missing messages")
            rows.append(row)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def invalid_tool_call_reason(message: dict[str, Any]) -> str | None:
    if message.get("role") != "assistant":
        return None
    try:
        payload = json.loads(message.get("content", ""))
    except (json.JSONDecodeError, TypeError):
        return "invalid_json"
    if not isinstance(payload, dict) or set(payload) != {"tool_name", "arguments"}:
        return "invalid_envelope"
    tool_name = payload["tool_name"]
    arguments = payload["arguments"]
    if tool_name not in CANONICAL_TOOLS:
        return "unknown_tool"
    if not isinstance(arguments, dict):
        return "arguments_not_object"

    if tool_name == "execute_bash":
        if set(arguments) != {"command"}:
            return "execute_bash_schema"
        command = arguments.get("command")
        return None if isinstance(command, str) and command.strip() else "execute_bash_missing_command"

    if tool_name in {"git_diff", "finish"}:
        return None if not arguments else f"{tool_name}_unexpected_arguments"

    if tool_name == "run_tests":
        if not set(arguments) <= {"targets", "command"}:
            return "run_tests_schema"
        if "targets" in arguments and (
            not isinstance(arguments["targets"], list)
            or not all(isinstance(target, str) for target in arguments["targets"])
        ):
            return "run_tests_targets"
        if "command" in arguments and not isinstance(arguments["command"], str):
            return "run_tests_command"
        return None

    allowed_editor_keys = {
        "command",
        "path",
        "view_range",
        "old_str",
        "new_str",
        "file_text",
        "insert_line",
        "start_line",
        "end_line",
    }
    if not set(arguments) <= allowed_editor_keys:
        return "editor_unknown_arguments"
    command = arguments.get("command")
    path = arguments.get("path")
    if command not in EDITOR_COMMANDS or not isinstance(path, str) or not path.strip():
        return "editor_base_schema"
    required_by_command = {
        "view": {},
        "str_replace": {"old_str": str, "new_str": str},
        "line_replace": {"start_line": int, "end_line": int, "new_str": str},
        "insert": {"insert_line": int, "new_str": str},
        "create": {"file_text": str},
        "undo_edit": {},
    }
    for key, expected_type in required_by_command[command].items():
        if not isinstance(arguments.get(key), expected_type):
            return f"editor_{command}_{key}"
    if "view_range" in arguments and (
        not isinstance(arguments["view_range"], list)
        or len(arguments["view_range"]) != 2
        or not all(isinstance(value, int) for value in arguments["view_range"])
    ):
        return "editor_view_range"
    return None


def token_ids(tokenizer: Any, messages: list[dict[str, Any]], *, generation_prompt: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation_prompt,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    return [int(value) for value in encoded]


def encode_with_labels(
    tokenizer: Any,
    tagged_messages: list[dict[str, Any]],
    trainable_source_indexes: set[int],
) -> tuple[list[int], list[int]]:
    messages = [item["message"] for item in tagged_messages]
    full_ids = token_ids(tokenizer, messages, generation_prompt=False)
    labels = [-100] * len(full_ids)
    observed: set[int] = set()
    for local_index, tagged in enumerate(tagged_messages):
        message = tagged["message"]
        source_index = int(tagged["source_index"])
        if message.get("role") != "assistant" or source_index not in trainable_source_indexes:
            continue
        prompt_ids = token_ids(tokenizer, messages[:local_index], generation_prompt=True)
        completed_ids = token_ids(tokenizer, messages[: local_index + 1], generation_prompt=False)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Chat-template prompt is not a prefix at source message {source_index}")
        if full_ids[: len(completed_ids)] != completed_ids:
            raise ValueError(f"Chat-template completion is not a prefix at source message {source_index}")
        if len(completed_ids) <= len(prompt_ids):
            raise ValueError(f"Assistant span is empty at source message {source_index}")
        labels[len(prompt_ids) : len(completed_ids)] = full_ids[len(prompt_ids) : len(completed_ids)]
        observed.add(source_index)
    if observed != trainable_source_indexes:
        missing = sorted(trainable_source_indexes - observed)
        raise ValueError(f"Assistant messages were not labeled: {missing}")
    return full_ids, labels


def base_and_units(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    tagged = [{"source_index": index, "message": message} for index, message in enumerate(messages)]
    first_assistant = next(
        (index for index, message in enumerate(messages) if message.get("role") == "assistant"),
        None,
    )
    if first_assistant is None:
        raise ValueError("Conversation has no assistant message")
    base = tagged[:first_assistant]
    units: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in tagged[first_assistant:]:
        if item["message"].get("role") == "assistant" and current:
            units.append(current)
            current = []
        current.append(item)
    if current:
        units.append(current)
    if any(unit[0]["message"].get("role") != "assistant" for unit in units):
        raise ValueError("Every continuation unit must begin with an assistant message")
    return base, units


def flatten(units: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [message for unit in units for message in unit]


def build_windows(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    max_length: int,
    overlap_units: int,
    trainable_assistant_indexes: set[int],
) -> list[tuple[list[dict[str, Any]], set[int]]]:
    base, units = base_and_units(messages)
    all_assistant_indexes = {
        int(item["source_index"])
        for unit in units
        for item in unit
        if item["message"].get("role") == "assistant"
    }
    if not trainable_assistant_indexes <= all_assistant_indexes:
        raise ValueError("Trainable indexes contain non-assistant messages")
    full = base + flatten(units)
    if len(token_ids(tokenizer, [item["message"] for item in full], generation_prompt=False)) <= max_length:
        return [(full, trainable_assistant_indexes)]

    windows: list[tuple[list[dict[str, Any]], set[int]]] = []
    consumed = 0
    trained: set[int] = set()
    while consumed < len(units):
        overlap = min(overlap_units, consumed)
        while True:
            context_units = units[consumed - overlap : consumed]
            candidate_units: list[list[dict[str, Any]]] = []
            cursor = consumed
            while cursor < len(units):
                proposed = base + flatten(context_units + candidate_units + [units[cursor]])
                proposed_length = len(
                    token_ids(tokenizer, [item["message"] for item in proposed], generation_prompt=False)
                )
                if proposed_length > max_length:
                    break
                candidate_units.append(units[cursor])
                cursor += 1
            if candidate_units or overlap == 0:
                break
            overlap -= 1
        if not candidate_units:
            unit_index = consumed
            raise ValueError(f"One assistant/tool-result unit exceeds max_length at unit {unit_index}")
        tagged_window = base + flatten(context_units + candidate_units)
        trainable = {
            int(item["source_index"])
            for unit in candidate_units
            for item in unit
            if item["message"].get("role") == "assistant"
            and int(item["source_index"]) in trainable_assistant_indexes
        }
        if trained & trainable:
            raise ValueError("An assistant turn would be trained in more than one window")
        trained.update(trainable)
        windows.append((tagged_window, trainable))
        consumed += len(candidate_units)
    if trained != trainable_assistant_indexes:
        raise ValueError("Windowing did not cover every trainable assistant turn exactly once")
    return windows


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    source_rows = read_jsonl(input_path)
    prepared: list[dict[str, Any]] = []
    split_ids: list[str] = []
    source_assistant_turns = 0
    trainable_assistant_turns = 0
    labeled_assistant_turns = 0
    masked_invalid_turns: Counter[str] = Counter()

    for row in source_rows:
        messages = row["messages"]
        source_assistant_turns += sum(message.get("role") == "assistant" for message in messages)
        trainable_indexes: set[int] = set()
        for source_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            reason = None if args.keep_invalid_tool_calls else invalid_tool_call_reason(message)
            if reason is None:
                trainable_indexes.add(source_index)
            else:
                masked_invalid_turns[reason] += 1
        trainable_assistant_turns += len(trainable_indexes)
        windows = build_windows(
            tokenizer,
            messages,
            max_length=args.max_length,
            overlap_units=args.overlap_units,
            trainable_assistant_indexes=trainable_indexes,
        )
        if len(windows) > 1:
            split_ids.append(str(row["trajectory_id"]))
        for window_index, (tagged_messages, trainable_indexes) in enumerate(windows):
            labeled_assistant_turns += len(trainable_indexes)
            input_ids, labels = encode_with_labels(tokenizer, tagged_messages, trainable_indexes)
            if len(input_ids) > args.max_length:
                raise ValueError(f"Window exceeds max_length: {row['trajectory_id']}:{window_index}")
            supervised = sum(label != -100 for label in labels)
            if supervised == 0:
                raise ValueError(f"Window has no supervised tokens: {row['trajectory_id']}:{window_index}")
            if any(label != -100 and label != token for token, label in zip(input_ids, labels, strict=True)):
                raise ValueError(f"Label/input mismatch: {row['trajectory_id']}:{window_index}")
            prepared.append(
                {
                    "trajectory_id": str(row["trajectory_id"]),
                    "instance_id": str(row["instance_id"]),
                    "sample_index": int(row["sample_index"]),
                    "dataset": str(row["dataset"]),
                    "quality_tier": str(row.get("quality_tier") or "unknown"),
                    "sample_weight": float(row.get("sample_weight", 1.0)),
                    "window_index": window_index,
                    "window_count": len(windows),
                    "token_count": len(input_ids),
                    "supervised_token_count": supervised,
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

    if labeled_assistant_turns != trainable_assistant_turns:
        raise ValueError(
            "Assistant turn coverage mismatch: "
            f"trainable={trainable_assistant_turns} labeled={labeled_assistant_turns}"
        )

    output_dir.mkdir(parents=True)
    features = Features(
        {
            "trajectory_id": Value("string"),
            "instance_id": Value("string"),
            "sample_index": Value("int16"),
            "dataset": Value("string"),
            "quality_tier": Value("string"),
            "sample_weight": Value("float32"),
            "window_index": Value("int16"),
            "window_count": Value("int16"),
            "token_count": Value("int32"),
            "supervised_token_count": Value("int32"),
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "labels": Sequence(Value("int32")),
        }
    )
    dataset = Dataset.from_list(prepared, features=features)
    dataset.save_to_disk(str(output_dir / "dataset"))

    token_lengths = [row["token_count"] for row in prepared]
    supervised_lengths = [row["supervised_token_count"] for row in prepared]
    by_dataset = Counter(row["dataset"] for row in prepared)
    manifest = {
        "format": "lottie_qwen_assistant_masked_sft_v2",
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "model_path": str(Path(args.model_path).resolve()),
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "max_length": args.max_length,
        "overlap_units": args.overlap_units,
        "source_rows": len(source_rows),
        "output_windows": len(prepared),
        "source_assistant_turns": source_assistant_turns,
        "trainable_assistant_turns": trainable_assistant_turns,
        "labeled_assistant_turns": labeled_assistant_turns,
        "masked_invalid_assistant_turns": sum(masked_invalid_turns.values()),
        "masked_invalid_reasons": dict(sorted(masked_invalid_turns.items())),
        "mask_policy": {
            "assistant_content_and_turn_end": True,
            "assistant_header": False,
            "system_user_and_tool_feedback": False,
            "invalid_tool_calls": bool(not args.keep_invalid_tool_calls),
            "overlap_context_assistant_turns": False,
        },
        "split_trajectories": len(split_ids),
        "split_trajectory_ids": split_ids,
        "dataset_windows": dict(sorted(by_dataset.items())),
        "tokens": {
            "total": sum(token_lengths),
            "mean": round(sum(token_lengths) / len(token_lengths), 2),
            "p50": percentile(token_lengths, 0.50),
            "p90": percentile(token_lengths, 0.90),
            "p95": percentile(token_lengths, 0.95),
            "p99": percentile(token_lengths, 0.99),
            "max": max(token_lengths),
        },
        "supervised_tokens": {
            "total": sum(supervised_lengths),
            "mean": round(sum(supervised_lengths) / len(supervised_lengths), 2),
            "fraction": round(sum(supervised_lengths) / sum(token_lengths), 6),
            "min_per_window": min(supervised_lengths),
        },
        "integrity": {
            "assistant_only_labels": True,
            "all_windows_within_max_length": True,
            "all_windows_have_supervised_tokens": True,
            "labels_match_input_ids_or_ignore_index": True,
            "cross_trajectory_packing": False,
            "trainable_assistant_turns_covered_once": labeled_assistant_turns == trainable_assistant_turns,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
