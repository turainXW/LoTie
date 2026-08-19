from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


IGNORE_INDEX = -100
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
NON_TRAINABLE_STATUSES = {
    "context_overflow",
    "model_api_failure",
    "parse_error",
    "infrastructure_blocked",
    "runner_error",
}


class OffsetTokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> Any: ...


def training_row_decision(
    row: dict[str, Any],
    *,
    verifier_results: dict[str, dict[str, Any]] | None = None,
    include_unresolved: bool = False,
    task_split: str | None = None,
) -> tuple[bool, bool, str]:
    """Return whether a rollout may be tokenized, its resolved flag, and the decision reason."""

    declared_splits = {
        str(value)
        for value in (task_split, row.get("split"), row.get("data_role"))
        if value is not None
    }
    if row.get("exclude_from_training") or declared_splits & {"evaluation", "evaluation_only"}:
        return False, False, "evaluation_excluded"
    if row.get("status") in NON_TRAINABLE_STATUSES:
        return False, False, "non_trainable_status"
    verifier_row = (verifier_results or {}).get(str(row.get("instance_id")))
    resolved = (
        bool(verifier_row.get("benchmark_resolved"))
        if verifier_row is not None
        else bool(row.get("resolved"))
    )
    if not resolved and not include_unresolved:
        return False, False, "unresolved_excluded"
    return True, resolved, "included"


@dataclass(frozen=True)
class MessageMaskStat:
    index: int
    role: str
    token_start: int
    token_end: int
    token_count: int
    trainable_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "role": self.role,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_count": self.token_count,
            "trainable_tokens": self.trainable_tokens,
        }


@dataclass(frozen=True)
class MaskedConversation:
    text: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    message_stats: list[MessageMaskStat]

    @property
    def trainable_tokens(self) -> int:
        return sum(label != IGNORE_INDEX for label in self.labels)


@dataclass(frozen=True)
class MaskedDatasetReport:
    rows: int
    total_tokens: int
    trainable_tokens: int
    masked_tokens: int
    min_length: int
    max_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "total_tokens": self.total_tokens,
            "trainable_tokens": self.trainable_tokens,
            "masked_tokens": self.masked_tokens,
            "trainable_ratio": round(self.trainable_tokens / self.total_tokens, 6) if self.total_tokens else 0.0,
            "min_length": self.min_length,
            "max_length": self.max_length,
        }


def normalize_training_messages(
    messages: list[dict[str, Any]],
    *,
    resolved: bool,
    append_finish_if_resolved: bool = True,
) -> tuple[list[dict[str, str]], bool]:
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported role at message {index}: {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"Message {index} content must be a string")
        if role == "assistant":
            validate_json_tool_call(content, message_index=index)
        normalized.append({"role": role, "content": content})

    terminal_finish_index = next(
        (
            index
            for index in range(len(normalized) - 1, -1, -1)
            if normalized[index]["role"] == "assistant"
            and validate_json_tool_call(normalized[index]["content"])["tool_name"] == "finish"
        ),
        None,
    )
    if terminal_finish_index is not None:
        normalized = normalized[: terminal_finish_index + 1]

    finish_appended = False
    if resolved and append_finish_if_resolved and not has_terminal_finish(normalized):
        if normalized and normalized[-1]["role"] == "assistant":
            raise ValueError("Cannot append finish after a terminal assistant message without an observation")
        normalized.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"tool_name": "finish", "arguments": {}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
        finish_appended = True
    return normalized, finish_appended


def validate_json_tool_call(content: str, *, message_index: int | None = None) -> dict[str, Any]:
    location = f" at message {message_index}" if message_index is not None else ""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Assistant content{location} is not canonical JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"tool_name", "arguments"}:
        raise ValueError(f"Assistant content{location} must contain exactly tool_name and arguments")
    if not isinstance(payload["tool_name"], str) or not payload["tool_name"]:
        raise ValueError(f"tool_name{location} must be a non-empty string")
    if not isinstance(payload["arguments"], dict):
        raise ValueError(f"arguments{location} must be an object")
    return payload


def has_terminal_finish(messages: list[dict[str, str]]) -> bool:
    for message in reversed(messages):
        if message["role"] != "assistant":
            continue
        return validate_json_tool_call(message["content"])["tool_name"] == "finish"
    return False


def mask_qwen_chatml(
    messages: list[dict[str, str]],
    tokenizer: OffsetTokenizer,
) -> MaskedConversation:
    text_parts: list[str] = []
    message_char_spans: list[tuple[int, int, int, str]] = []
    trainable_char_spans: list[tuple[int, int]] = []
    cursor = 0

    for index, message in enumerate(messages):
        role = message["role"]
        content = message["content"]
        prefix = f"{IM_START}{role}\n"
        suffix = f"{IM_END}\n"
        rendered = prefix + content + suffix
        message_start = cursor
        content_start = cursor + len(prefix)
        cursor += len(rendered)
        message_end = cursor
        text_parts.append(rendered)
        message_char_spans.append((message_start, message_end, index, role))
        if role == "assistant":
            # Train the JSON response and its turn terminator, but not the assistant header.
            trainable_char_spans.append((content_start, message_end))

    text = "".join(text_parts)
    encoding = tokenizer.encode(text, add_special_tokens=False)
    input_ids = list(encoding.ids)
    offsets = list(encoding.offsets)
    labels = [IGNORE_INDEX] * len(input_ids)

    for token_index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if any(train_start <= start and end <= train_end for train_start, train_end in trainable_char_spans):
            labels[token_index] = input_ids[token_index]
        elif any(start < train_start < end or start < train_end < end for train_start, train_end in trainable_char_spans):
            raise ValueError(f"Tokenizer produced a token crossing a loss-mask boundary at token {token_index}")

    stats: list[MessageMaskStat] = []
    for message_start, message_end, index, role in message_char_spans:
        token_indices = [
            token_index
            for token_index, (start, end) in enumerate(offsets)
            if start < message_end and end > message_start
        ]
        token_start = token_indices[0] if token_indices else 0
        token_end = token_indices[-1] + 1 if token_indices else token_start
        stats.append(
            MessageMaskStat(
                index=index,
                role=role,
                token_start=token_start,
                token_end=token_end,
                token_count=len(token_indices),
                trainable_tokens=sum(labels[token_index] != IGNORE_INDEX for token_index in token_indices),
            )
        )

    return MaskedConversation(
        text=text,
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        message_stats=stats,
    )


def validate_masked_token_record(row: dict[str, Any], *, max_length: int | None = None) -> dict[str, int]:
    input_ids = row.get("input_ids")
    attention_mask = row.get("attention_mask")
    labels = row.get("labels")
    if not all(isinstance(value, list) for value in (input_ids, attention_mask, labels)):
        raise ValueError("input_ids, attention_mask, and labels must all be lists")
    if not input_ids:
        raise ValueError("input_ids must not be empty")
    if len(input_ids) != len(attention_mask) or len(input_ids) != len(labels):
        raise ValueError("input_ids, attention_mask, and labels must have identical lengths")
    if max_length is not None and len(input_ids) > max_length:
        raise ValueError(f"sequence length {len(input_ids)} exceeds max_length {max_length}")
    if any(not isinstance(token, int) or token < 0 for token in input_ids):
        raise ValueError("input_ids must contain non-negative integers")
    if any(mask not in {0, 1} for mask in attention_mask):
        raise ValueError("attention_mask must contain only 0 or 1")
    trainable_tokens = 0
    for index, (token, label, attention) in enumerate(zip(input_ids, labels, attention_mask)):
        if not isinstance(label, int):
            raise ValueError(f"label at token {index} is not an integer")
        if label != IGNORE_INDEX and label != token:
            raise ValueError(f"label at token {index} must equal input_id or {IGNORE_INDEX}")
        if attention == 0 and label != IGNORE_INDEX:
            raise ValueError(f"padding token {index} cannot participate in loss")
        trainable_tokens += int(label != IGNORE_INDEX)
    if trainable_tokens == 0:
        raise ValueError("record has no trainable assistant tokens")
    return {
        "token_count": len(input_ids),
        "trainable_tokens": trainable_tokens,
        "masked_tokens": len(input_ids) - trainable_tokens,
    }


def audit_masked_token_records(
    rows: list[dict[str, Any]],
    *,
    max_length: int | None = None,
) -> MaskedDatasetReport:
    if not rows:
        raise ValueError("masked token dataset is empty")
    stats = [validate_masked_token_record(row, max_length=max_length) for row in rows]
    lengths = [item["token_count"] for item in stats]
    trainable_tokens = sum(item["trainable_tokens"] for item in stats)
    total_tokens = sum(lengths)
    return MaskedDatasetReport(
        rows=len(rows),
        total_tokens=total_tokens,
        trainable_tokens=trainable_tokens,
        masked_tokens=total_tokens - trainable_tokens,
        min_length=min(lengths),
        max_length=max(lengths),
    )
