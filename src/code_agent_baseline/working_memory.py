from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable


TokenCounter = Callable[[list[dict[str, Any]]], int]


class ContextLimitExceeded(RuntimeError):
    def __init__(self, snapshot: "ContextSnapshot") -> None:
        self.snapshot = snapshot
        super().__init__(
            f"Context input has {snapshot.full_history_tokens} tokens, exceeding "
            f"the configured limit of {snapshot.max_input_tokens}."
        )


@dataclass
class ContextSnapshot:
    full_message_count: int
    input_message_count: int
    full_history_chars: int
    input_chars: int
    compacted: bool
    summarized_message_count: int
    full_history_tokens: int = 0
    input_tokens: int = 0
    max_input_tokens: int | None = None
    token_counter: str = "estimated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model_input(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None = 26_000,
    max_chars: int | None = None,
    recent_turns: int = 6,
    summary_chars: int = 6_000,
    token_counter: TokenCounter | None = None,
    token_counter_name: str = "estimated",
    overflow_policy: str = "compact",
) -> tuple[list[dict[str, Any]], ContextSnapshot]:
    """Keep a full trajectory while presenting a bounded working memory to the model."""

    copied = [dict(message) for message in messages]
    counter = token_counter or estimate_message_tokens
    full_chars = message_chars(copied)
    full_tokens = counter(copied)
    over_limit = _over_limit(copied, max_tokens=max_tokens, max_chars=max_chars, token_counter=counter)
    full_snapshot = ContextSnapshot(
            full_message_count=len(copied),
            input_message_count=len(copied),
            full_history_chars=full_chars,
            input_chars=full_chars,
            compacted=False,
            summarized_message_count=0,
            full_history_tokens=full_tokens,
            input_tokens=full_tokens,
            max_input_tokens=max_tokens,
            token_counter=token_counter_name,
    )
    if not over_limit:
        return copied, full_snapshot
    if overflow_policy == "error":
        raise ContextLimitExceeded(full_snapshot)
    if overflow_policy != "compact":
        raise ValueError(f"Unsupported context overflow policy: {overflow_policy}")
    if len(copied) <= 2:
        raise ContextLimitExceeded(full_snapshot)

    prefix = copied[:2]
    history = copied[2:]
    keep_count = max(2, recent_turns * 2)
    dropped = history[:-keep_count] if len(history) > keep_count else []
    recent = history[-keep_count:]
    summary = summarize_messages(dropped, max_chars=summary_chars)
    memory_message = {
        "role": "user",
        "content": (
            "<WORKING_MEMORY>\n"
            "The following compact record summarizes earlier steps in this task. "
            "The recent raw tool calls and observations follow it and take precedence.\n"
            f"{summary}\n"
            "</WORKING_MEMORY>"
        ),
    }
    selected = [*prefix, memory_message, *recent]
    selected = fit_messages(
        selected,
        max_tokens=max_tokens,
        max_chars=max_chars,
        protected_prefix=3,
        token_counter=counter,
    )
    input_chars = message_chars(selected)
    input_tokens = counter(selected)
    return selected, ContextSnapshot(
        full_message_count=len(copied),
        input_message_count=len(selected),
        full_history_chars=full_chars,
        input_chars=input_chars,
        compacted=True,
        summarized_message_count=len(dropped),
        full_history_tokens=full_tokens,
        input_tokens=input_tokens,
        max_input_tokens=max_tokens,
        token_counter=token_counter_name,
    )


def summarize_messages(messages: list[dict[str, Any]], *, max_chars: int) -> str:
    lines: list[str] = []
    step = 0
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "assistant":
            step += 1
            tool_name, arguments = parse_tool_call(content)
            if tool_name:
                lines.append(f"step {step}: assistant called {tool_name}({compact_arguments(arguments)})")
            else:
                lines.append(f"step {step}: assistant output {clip(content, 240)}")
        elif role in {"user", "tool"}:
            first_line = content.splitlines()[0] if content else "empty observation"
            exit_code = extract_exit_code(content)
            suffix = f", exit_code={exit_code}" if exit_code is not None else ""
            lines.append(f"  observation: {clip(first_line, 220)}{suffix}")
    text = "\n".join(lines) or "No earlier steps."
    return clip(text, max_chars)


def fit_messages(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None,
    max_chars: int | None,
    protected_prefix: int,
    token_counter: TokenCounter,
) -> list[dict[str, Any]]:
    fitted = [dict(message) for message in messages]
    while (
        _over_limit(fitted, max_tokens=max_tokens, max_chars=max_chars, token_counter=token_counter)
        and len(fitted) > protected_prefix + 2
    ):
        fitted.pop(protected_prefix)
    if not _over_limit(fitted, max_tokens=max_tokens, max_chars=max_chars, token_counter=token_counter):
        return fitted

    for index in range(protected_prefix, len(fitted) - 1):
        content = str(fitted[index].get("content", ""))
        if len(content) > 4_000:
            fitted[index]["content"] = clip(content, 4_000)
    if not _over_limit(fitted, max_tokens=max_tokens, max_chars=max_chars, token_counter=token_counter):
        return fitted

    last = fitted[-1]
    last["content"] = _fit_last_message(
        fitted,
        max_tokens=max_tokens,
        max_chars=max_chars,
        token_counter=token_counter,
    )
    return fitted


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Conservative fallback for Qwen-style code/chat text when no tokenizer is configured."""

    rendered_chars = sum(len(str(item.get("role", ""))) + len(str(item.get("content", ""))) + 24 for item in messages)
    return max(1, math.ceil(rendered_chars / 3.5))


def load_token_counter(tokenizer_path: str | Path) -> tuple[TokenCounter, str]:
    """Load a Hugging Face tokenizer.json without requiring transformers."""

    from tokenizers import Tokenizer

    path = Path(tokenizer_path).expanduser().resolve()
    tokenizer = Tokenizer.from_file(str(path))

    def count(messages: list[dict[str, Any]]) -> int:
        rendered = "".join(
            f"<|im_start|>{item.get('role', '')}\n{item.get('content', '')}<|im_end|>\n" for item in messages
        )
        return len(tokenizer.encode(rendered).ids)

    return count, f"tokenizer:{path}"


def _over_limit(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None,
    max_chars: int | None,
    token_counter: TokenCounter,
) -> bool:
    if max_tokens is not None and token_counter(messages) > max_tokens:
        return True
    return max_chars is not None and message_chars(messages) > max_chars


def _fit_last_message(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None,
    max_chars: int | None,
    token_counter: TokenCounter,
) -> str:
    original = str(messages[-1].get("content", ""))
    low, high = 0, len(original)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = clip(original, middle)
        messages[-1]["content"] = candidate
        if _over_limit(messages, max_tokens=max_tokens, max_chars=max_chars, token_counter=token_counter):
            high = middle - 1
        else:
            best = candidate
            low = middle + 1
    return best or clip(original, 1_000)


def parse_tool_call(content: str) -> tuple[str, dict[str, Any]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return "", {}
    if not isinstance(payload, dict):
        return "", {}
    name = payload.get("tool_name")
    arguments = payload.get("arguments")
    return (str(name), arguments if isinstance(arguments, dict) else {}) if name else ("", {})


def compact_arguments(arguments: dict[str, Any]) -> str:
    selected = {}
    for key in ("command", "path", "query", "targets", "start_line", "end_line"):
        if key in arguments:
            selected[key] = arguments[key]
    return clip(json.dumps(selected, ensure_ascii=False, sort_keys=True), 320)


def extract_exit_code(content: str) -> str | None:
    marker = "[Command finished with exit code "
    if marker not in content:
        return None
    return content.split(marker, 1)[1].split("]", 1)[0]


def message_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages)


def clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n... compacted {omitted} chars ..."
