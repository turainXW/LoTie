#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from transformers import AutoTokenizer


IGNORE_INDEX = -100
ROLES = ("system", "user", "assistant")
CANONICAL_TOOLS = {"execute_bash", "str_replace_editor", "run_tests", "git_diff", "finish"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit assistant-only Qwen ChatML labels in a tokenized Arrow dataset."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-length", type=int, default=32_768)
    parser.add_argument("--report")
    parser.add_argument("--max-errors", type=int, default=20)
    return parser.parse_args()


def encode(tokenizer: Any, text: str) -> list[int]:
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def require_single_special_id(tokenizer: Any, token: str) -> int:
    token_id = int(tokenizer.convert_tokens_to_ids(token))
    if token_id < 0 or tokenizer.convert_ids_to_tokens(token_id) != token:
        raise ValueError(f"Tokenizer does not expose {token!r} as one special token")
    return token_id


def contiguous_ranges(indexes: list[int]) -> list[tuple[int, int]]:
    if not indexes:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = indexes[0]
    for index in indexes[1:]:
        if index != previous + 1:
            ranges.append((start, previous + 1))
            start = index
        previous = index
    ranges.append((start, previous + 1))
    return ranges


def validate_tool_call(text: str) -> str:
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != {"tool_name", "arguments"}:
        raise ValueError("assistant JSON must contain exactly tool_name and arguments")
    if not isinstance(payload["tool_name"], str) or not payload["tool_name"]:
        raise ValueError("tool_name must be a non-empty string")
    if not isinstance(payload["arguments"], dict):
        raise ValueError("arguments must be an object")
    tool_name = payload["tool_name"]
    arguments = payload["arguments"]
    if tool_name not in CANONICAL_TOOLS:
        raise ValueError(f"unknown tool: {tool_name}")
    if tool_name == "execute_bash" and (
        set(arguments) != {"command"}
        or not isinstance(arguments.get("command"), str)
        or not arguments["command"].strip()
    ):
        raise ValueError("execute_bash requires exactly one non-empty command")
    if tool_name in {"git_diff", "finish"} and arguments:
        raise ValueError(f"{tool_name} does not accept arguments")
    if tool_name == "str_replace_editor":
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
            raise ValueError("str_replace_editor has unknown arguments")
        if arguments.get("command") not in {
            "view",
            "str_replace",
            "line_replace",
            "insert",
            "create",
            "undo_edit",
        } or not isinstance(arguments.get("path"), str) or not arguments["path"].strip():
            raise ValueError("str_replace_editor requires a valid command and non-empty path")
        editor_requirements = {
            "view": {},
            "str_replace": {"old_str": str, "new_str": str},
            "line_replace": {"start_line": int, "end_line": int, "new_str": str},
            "insert": {"insert_line": int, "new_str": str},
            "create": {"file_text": str},
            "undo_edit": {},
        }
        command = arguments["command"]
        for key, expected_type in editor_requirements[command].items():
            if not isinstance(arguments.get(key), expected_type):
                raise ValueError(f"str_replace_editor {command} requires {key}")
        if "view_range" in arguments and (
            not isinstance(arguments["view_range"], list)
            or len(arguments["view_range"]) != 2
            or not all(isinstance(value, int) for value in arguments["view_range"])
        ):
            raise ValueError("str_replace_editor view_range must be two integers")
    if tool_name == "run_tests":
        if not set(arguments) <= {"targets", "command"}:
            raise ValueError("run_tests has unknown arguments")
        if "targets" in arguments and (
            not isinstance(arguments["targets"], list)
            or not all(isinstance(target, str) for target in arguments["targets"])
        ):
            raise ValueError("run_tests targets must be a string array")
        if "command" in arguments and not isinstance(arguments["command"], str):
            raise ValueError("run_tests command must be a string")
    return tool_name


def inspect_row(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    im_start_id: int,
    im_end_id: int,
    role_headers: dict[str, list[int]],
    max_length: int,
) -> dict[str, Any]:
    input_ids = [int(token) for token in row["input_ids"]]
    labels = [int(token) for token in row["labels"]]
    attention_mask = [int(token) for token in row["attention_mask"]]
    if not input_ids or not (len(input_ids) == len(labels) == len(attention_mask)):
        raise ValueError("input_ids, labels, and attention_mask must have the same non-zero length")
    if len(input_ids) > max_length:
        raise ValueError(f"sequence length {len(input_ids)} exceeds {max_length}")
    if any(mask != 1 for mask in attention_mask):
        raise ValueError("saved unpadded records must have an all-one attention_mask")
    for index, (token, label) in enumerate(zip(input_ids, labels, strict=True)):
        if label not in (IGNORE_INDEX, token):
            raise ValueError(f"label/input mismatch at token {index}")

    starts = [index for index, token in enumerate(input_ids) if token == im_start_id]
    ends = [index for index, token in enumerate(input_ids) if token == im_end_id]
    if not starts or len(starts) != len(ends):
        raise ValueError(f"unbalanced ChatML markers: starts={len(starts)} ends={len(ends)}")
    if starts[0] != 0:
        raise ValueError(f"unexpected tokens before first ChatML message: {starts[0]}")

    expected_trainable: set[int] = set()
    role_counts: Counter[str] = Counter()
    trained_assistant_turns = 0
    context_assistant_turns = 0
    tool_names: Counter[str] = Counter()
    end_cursor = 0
    for message_index, start in enumerate(starts):
        next_start = starts[message_index + 1] if message_index + 1 < len(starts) else len(input_ids)
        matching_ends = [end for end in ends if start < end < next_start]
        if len(matching_ends) != 1:
            raise ValueError(f"message {message_index} has {len(matching_ends)} end markers")
        end = matching_ends[0]
        end_cursor = max(end_cursor, end + 1)

        role = None
        content_start = None
        for candidate in ROLES:
            header = role_headers[candidate]
            candidate_start = start + 1
            if input_ids[candidate_start : candidate_start + len(header)] == header:
                role = candidate
                content_start = candidate_start + len(header)
                break
        if role is None or content_start is None:
            decoded = tokenizer.decode(input_ids[start + 1 : min(end, start + 12)], skip_special_tokens=False)
            raise ValueError(f"unknown role header at message {message_index}: {decoded!r}")
        role_counts[role] += 1

        header_indexes = range(start, content_start)
        if any(labels[index] != IGNORE_INDEX for index in header_indexes):
            raise ValueError(f"{role} header participates in loss at message {message_index}")

        body_indexes = list(range(content_start, next_start))
        supervised = [index for index in body_indexes if labels[index] != IGNORE_INDEX]
        if role != "assistant":
            if supervised:
                raise ValueError(f"{role} message participates in loss at message {message_index}")
            continue

        if not supervised:
            context_assistant_turns += 1
            continue
        if supervised != body_indexes:
            ranges = contiguous_ranges(supervised)
            raise ValueError(f"assistant message is only partially supervised at message {message_index}: {ranges}")
        if labels[end] != im_end_id:
            raise ValueError(f"assistant end marker is not supervised at message {message_index}")
        content = tokenizer.decode(input_ids[content_start:end], skip_special_tokens=False)
        try:
            tool_name = validate_tool_call(content)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid supervised assistant JSON at message {message_index}: {exc}") from exc
        expected_trainable.update(body_indexes)
        trained_assistant_turns += 1
        tool_names[tool_name] += 1

    if end_cursor > len(input_ids):
        raise ValueError("message parser exceeded sequence length")
    observed_trainable = {index for index, label in enumerate(labels) if label != IGNORE_INDEX}
    if observed_trainable != expected_trainable:
        extra = sorted(observed_trainable - expected_trainable)[:10]
        missing = sorted(expected_trainable - observed_trainable)[:10]
        raise ValueError(f"mask boundary mismatch: extra={extra} missing={missing}")
    if int(row["supervised_token_count"]) != len(observed_trainable):
        raise ValueError(
            f"supervised_token_count mismatch: metadata={row['supervised_token_count']} actual={len(observed_trainable)}"
        )
    if trained_assistant_turns == 0:
        raise ValueError("window has no trained assistant turns")
    return {
        "tokens": len(input_ids),
        "supervised_tokens": len(observed_trainable),
        "messages": sum(role_counts.values()),
        "trained_assistant_turns": trained_assistant_turns,
        "context_assistant_turns": context_assistant_turns,
        "role_counts": role_counts,
        "tool_names": tool_names,
    }


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    dataset = load_from_disk(str(dataset_dir))
    im_start_id = require_single_special_id(tokenizer, "<|im_start|>")
    im_end_id = require_single_special_id(tokenizer, "<|im_end|>")
    role_headers = {role: encode(tokenizer, f"{role}\n") for role in ROLES}

    aggregate = Counter()
    role_counts: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    seen_windows: set[tuple[str, int]] = set()
    for row_index, row in enumerate(dataset):
        key = (str(row["trajectory_id"]), int(row["window_index"]))
        if key in seen_windows:
            errors.append({"row": row_index, "trajectory_id": key[0], "error": "duplicate window key"})
            continue
        seen_windows.add(key)
        try:
            stat = inspect_row(
                row,
                tokenizer=tokenizer,
                im_start_id=im_start_id,
                im_end_id=im_end_id,
                role_headers=role_headers,
                max_length=args.max_length,
            )
        except (KeyError, TypeError, ValueError) as exc:
            if len(errors) < args.max_errors:
                errors.append({"row": row_index, "trajectory_id": key[0], "error": str(exc)})
            continue
        for field in (
            "tokens",
            "supervised_tokens",
            "messages",
            "trained_assistant_turns",
            "context_assistant_turns",
        ):
            aggregate[field] += int(stat[field])
        role_counts.update(stat["role_counts"])
        tool_names.update(stat["tool_names"])

    report = {
        "format": "lottie_qwen_arrow_mask_audit_v1",
        "dataset_dir": str(dataset_dir),
        "model_path": str(Path(args.model_path).resolve()),
        "rows": len(dataset),
        "unique_windows": len(seen_windows),
        "tokens": aggregate["tokens"],
        "supervised_tokens": aggregate["supervised_tokens"],
        "supervised_fraction": round(aggregate["supervised_tokens"] / aggregate["tokens"], 6)
        if aggregate["tokens"]
        else 0.0,
        "messages": aggregate["messages"],
        "trained_assistant_turns": aggregate["trained_assistant_turns"],
        "masked_context_assistant_turns": aggregate["context_assistant_turns"],
        "role_counts": dict(sorted(role_counts.items())),
        "tool_names": dict(tool_names.most_common()),
        "checks": {
            "labels_match_input_ids_or_ignore_index": not errors,
            "non_assistant_tokens_masked": not errors,
            "assistant_headers_masked": not errors,
            "assistant_json_and_turn_end_supervised": not errors,
            "assistant_turn_masks_are_atomic": not errors,
            "metadata_supervised_counts_match": not errors,
        },
        "errors": errors,
        "passed": not errors and aggregate["trained_assistant_turns"] > 0,
    }
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
