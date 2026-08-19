from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from code_agent_baseline.sft_masking import (
    IGNORE_INDEX,
    audit_masked_token_records,
    mask_qwen_chatml,
    normalize_training_messages,
    training_row_decision,
    validate_masked_token_record,
)


@dataclass
class CharacterEncoding:
    text: str

    @property
    def ids(self) -> list[int]:
        return [ord(char) for char in self.text]

    @property
    def offsets(self) -> list[tuple[int, int]]:
        return [(index, index + 1) for index in range(len(self.text))]


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> CharacterEncoding:
        assert not add_special_tokens
        return CharacterEncoding(text)


def test_evaluation_rows_are_never_trainable() -> None:
    include, resolved, reason = training_row_decision(
        {
            "instance_id": "eval-1",
            "resolved": True,
            "data_role": "evaluation_only",
            "exclude_from_training": True,
        },
        verifier_results={"eval-1": {"benchmark_resolved": True}},
        include_unresolved=True,
    )

    assert not include
    assert not resolved
    assert reason == "evaluation_excluded"


def test_task_catalog_split_blocks_legacy_evaluation_trace() -> None:
    include, resolved, reason = training_row_decision(
        {
            "instance_id": "legacy-eval-1",
            "resolved": True,
            "status": "resolved",
        },
        task_split="evaluation",
    )

    assert not include
    assert not resolved
    assert reason == "evaluation_excluded"


def test_verifier_result_is_authoritative_for_training() -> None:
    include, resolved, reason = training_row_decision(
        {"instance_id": "train-1", "resolved": True, "status": "resolved"},
        verifier_results={"train-1": {"benchmark_resolved": False}},
    )

    assert not include
    assert not resolved
    assert reason == "unresolved_excluded"


def test_masks_everything_except_assistant_json_and_turn_end() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": '{"tool_name":"execute_bash","arguments":{"command":"pwd"}}'},
        {"role": "user", "content": "EXECUTION RESULT of [execute_bash]: ok"},
    ]

    masked = mask_qwen_chatml(messages, CharacterTokenizer())

    assert len(masked.input_ids) == len(masked.labels) == len(masked.attention_mask)
    assert masked.message_stats[0].trainable_tokens == 0
    assert masked.message_stats[1].trainable_tokens == 0
    assert masked.message_stats[2].trainable_tokens > 0
    assert masked.message_stats[3].trainable_tokens == 0
    trainable_text = "".join(chr(token) for token, label in zip(masked.input_ids, masked.labels) if label != IGNORE_INDEX)
    assert trainable_text == messages[2]["content"] + "<|im_end|>\n"


def test_appends_canonical_finish_to_resolved_trace() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": '{"tool_name":"git_diff","arguments":{}}'},
        {"role": "user", "content": "EXECUTION RESULT of [git_diff]: patch"},
    ]

    normalized, appended = normalize_training_messages(messages, resolved=True)

    assert appended
    assert json.loads(normalized[-1]["content"]) == {"tool_name": "finish", "arguments": {}}


def test_sft_sequence_ends_at_model_generated_finish() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": '{"tool_name":"finish","arguments":{}}'},
        {"role": "user", "content": "EXECUTION RESULT of [finish]: finished"},
    ]

    normalized, appended = normalize_training_messages(messages, resolved=True)

    assert not appended
    assert normalized[-1]["role"] == "assistant"
    assert json.loads(normalized[-1]["content"])["tool_name"] == "finish"


def test_rejects_noncanonical_assistant_content() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "I will inspect the repository first."},
    ]

    with pytest.raises(ValueError, match="not canonical JSON"):
        normalize_training_messages(messages, resolved=False)


def test_audits_valid_masked_token_record() -> None:
    row = {
        "input_ids": [10, 11, 12, 13],
        "attention_mask": [1, 1, 1, 1],
        "labels": [-100, 11, 12, -100],
    }

    stats = validate_masked_token_record(row, max_length=4)
    report = audit_masked_token_records([row], max_length=4)

    assert stats == {"token_count": 4, "trainable_tokens": 2, "masked_tokens": 2}
    assert report.trainable_tokens == 2
    assert report.max_length == 4


def test_rejects_mismatched_label_token() -> None:
    row = {
        "input_ids": [10, 11],
        "attention_mask": [1, 1],
        "labels": [-100, 99],
    }

    with pytest.raises(ValueError, match="must equal input_id"):
        validate_masked_token_record(row)
