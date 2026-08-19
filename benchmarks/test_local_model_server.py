from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serve_local_transformers.py"
SPEC = importlib.util.spec_from_file_location("serve_local_transformers", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_validate_messages_keeps_supported_chat_fields() -> None:
    messages = MODULE.validate_messages(
        [
            {"role": "system", "content": "system", "ignored": True},
            {"role": "user", "content": "hello", "name": "caller"},
        ]
    )

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]


@pytest.mark.parametrize("messages", [None, [], [{"role": "tool", "content": "x"}], [{"role": "user"}]])
def test_validate_messages_rejects_invalid_payloads(messages: object) -> None:
    with pytest.raises(ValueError):
        MODULE.validate_messages(messages)


def test_completion_response_is_openai_compatible() -> None:
    response = MODULE.completion_response(
        model="qwen-local",
        content='{"tool_name":"finish","arguments":{}}',
        prompt_tokens=10,
        completion_tokens=5,
    )

    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["content"].startswith("{")
    assert response["usage"]["total_tokens"] == 15


def test_generation_token_ids_preserves_distinct_qwen_stop_and_padding_tokens() -> None:
    tokenizer = SimpleNamespace(eos_token_id=248046, pad_token_id=248044)
    config = SimpleNamespace(
        eos_token_id=None,
        text_config=SimpleNamespace(eos_token_id=248044),
    )

    assert MODULE.generation_token_ids(tokenizer, config) == {
        "eos_token_id": [248046, 248044],
        "pad_token_id": 248044,
    }


def test_generation_token_ids_falls_back_to_eos_for_padding() -> None:
    tokenizer = SimpleNamespace(eos_token_id=7, pad_token_id=None)

    assert MODULE.generation_token_ids(tokenizer) == {"eos_token_id": 7, "pad_token_id": 7}


def test_generation_token_ids_requires_eos() -> None:
    tokenizer = SimpleNamespace(eos_token_id=None, pad_token_id=None)

    with pytest.raises(ValueError, match="eos_token_id"):
        MODULE.generation_token_ids(tokenizer)


def test_select_tokenizer_path_prefers_adapter_tokenizer(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    assert MODULE.select_tokenizer_path(str(base), str(adapter)) == str(adapter)


def test_select_tokenizer_path_falls_back_to_base(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()

    assert MODULE.select_tokenizer_path(str(base), str(adapter)) == str(base)
    assert MODULE.select_tokenizer_path(str(base), None) == str(base)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"tool_name":"finish","arguments":{}}', True),
        ('thinking first\n{"tool_name":"finish","arguments":{}}', True),
        ('{"tool_name":"execute_bash","arguments":{"command":"pwd"}}\n{"tool_name":', True),
        ('prefix {"nested":true}', False),
        ('{"tool_name":"finish","arguments":{', False),
        ('{"tool_name":"bad "quote""}\n{"tool_name":"finish","arguments":{}}', True),
    ],
)
def test_contains_complete_top_level_json(text: str, expected: bool) -> None:
    assert MODULE.contains_complete_top_level_json(text) is expected
