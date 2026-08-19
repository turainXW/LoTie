#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.sft_masking import (  # noqa: E402
    IGNORE_INDEX,
    mask_qwen_chatml,
    normalize_training_messages,
    training_row_decision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create assistant-only token masks for multi-turn Lottie trajectories.")
    parser.add_argument("--input", required=True, help="Rollout messages_sft.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-json", required=True, help="Target model tokenizer.json.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-unresolved", action="store_true")
    parser.add_argument(
        "--task-catalog",
        action="append",
        help=(
            "Task JSONL used as an authoritative instance_id-to-split map. Repeatable. "
            "Defaults to the bundled EvalPlus and SWE-smith catalogs when present."
        ),
    )
    parser.add_argument(
        "--verifier-results",
        help="Optional verifier JSONL; benchmark_resolved becomes authoritative when present.",
    )
    parser.add_argument(
        "--append-finish-if-resolved",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a canonical finish assistant turn to verified-success traces that did not terminate explicitly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Install the local-llm dependencies to use token-level masking") from exc

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    tokenizer_fingerprint = hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()
    source_rows = read_jsonl(input_path)
    task_catalogs = resolve_task_catalogs(args.task_catalog)
    task_splits = load_task_splits(task_catalogs)
    verifier_results = (
        {str(row.get("instance_id")): row for row in read_jsonl(Path(args.verifier_results))}
        if args.verifier_results
        else {}
    )
    if args.limit is not None:
        source_rows = source_rows[: args.limit]

    tokenized_path = output_dir / "masked_tokens.jsonl"
    readable_path = output_dir / "masked_messages.jsonl"
    tokenized_count = 0
    skipped_count = 0
    total_tokens = 0
    total_trainable_tokens = 0
    finish_appended_count = 0
    skipped_evaluation_count = 0
    skipped_unresolved_count = 0

    with tokenized_path.open("w", encoding="utf-8") as tokenized_file, readable_path.open(
        "w", encoding="utf-8"
    ) as readable_file:
        for row in source_rows:
            include_row, resolved, decision_reason = training_row_decision(
                row,
                verifier_results=verifier_results,
                include_unresolved=args.include_unresolved,
                task_split=task_splits.get(str(row.get("instance_id"))),
            )
            if not include_row:
                skipped_count += 1
                skipped_evaluation_count += int(decision_reason == "evaluation_excluded")
                skipped_unresolved_count += int(decision_reason == "unresolved_excluded")
                continue
            messages, finish_appended = normalize_training_messages(
                row.get("messages") or [],
                resolved=resolved,
                append_finish_if_resolved=args.append_finish_if_resolved,
            )
            masked = mask_qwen_chatml(messages, tokenizer)
            metadata = {
                "instance_id": row.get("instance_id"),
                "run_id": row.get("run_id"),
                "resolved": resolved,
                "status": row.get("status"),
                "token_count": len(masked.input_ids),
                "trainable_tokens": masked.trainable_tokens,
                "masked_tokens": len(masked.input_ids) - masked.trainable_tokens,
                "assistant_turns": sum(message["role"] == "assistant" for message in messages),
                "finish_appended": finish_appended,
                "chat_template": "qwen_chatml",
                "ignore_index": IGNORE_INDEX,
            }
            tokenized_file.write(
                json.dumps(
                    {
                        "format": "lottie_multiturn_masked_tokens_v1",
                        **metadata,
                        "tokenizer_sha256": tokenizer_fingerprint,
                        "input_ids": masked.input_ids,
                        "attention_mask": masked.attention_mask,
                        "labels": masked.labels,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            readable_file.write(
                json.dumps(
                    {
                        "format": "lottie_multiturn_message_mask_v1",
                        **metadata,
                        "messages": [
                            {
                                **message,
                                "loss": "train" if message["role"] == "assistant" else "masked",
                                "token_range": [
                                    masked.message_stats[index].token_start,
                                    masked.message_stats[index].token_end,
                                ],
                                "trainable_tokens": masked.message_stats[index].trainable_tokens,
                            }
                            for index, message in enumerate(messages)
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            tokenized_count += 1
            total_tokens += len(masked.input_ids)
            total_trainable_tokens += masked.trainable_tokens
            finish_appended_count += int(finish_appended)

    manifest: dict[str, Any] = {
        "format": "lottie_multiturn_mask_manifest_v1",
        "input": str(input_path.resolve()),
        "tokenizer_json": str(Path(args.tokenizer_json).resolve()),
        "tokenizer_sha256": tokenizer_fingerprint,
        "rows_seen": len(source_rows),
        "rows_tokenized": tokenized_count,
        "rows_skipped": skipped_count,
        "rows_skipped_evaluation": skipped_evaluation_count,
        "rows_skipped_unresolved": skipped_unresolved_count,
        "total_tokens": total_tokens,
        "total_trainable_tokens": total_trainable_tokens,
        "total_masked_tokens": total_tokens - total_trainable_tokens,
        "trainable_ratio": round(total_trainable_tokens / total_tokens, 6) if total_tokens else 0.0,
        "finish_appended_count": finish_appended_count,
        "verifier_results": str(Path(args.verifier_results).resolve()) if args.verifier_results else None,
        "task_catalogs": [str(path.resolve()) for path in task_catalogs],
        "task_split_matches": sum(str(row.get("instance_id")) in task_splits for row in source_rows),
        "success_source": "verifier.benchmark_resolved" if args.verifier_results else "rollout.resolved",
        "mask_policy": {
            "system": "-100",
            "user_and_tool_observation": "-100",
            "assistant_header": "-100",
            "assistant_json_and_im_end": "train",
        },
        "tokenized_jsonl": str(tokenized_path.resolve()),
        "readable_jsonl": str(readable_path.resolve()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_task_catalogs(configured: list[str] | None) -> list[Path]:
    if configured:
        paths = [Path(value) for value in configured]
    else:
        paths = [
            ROOT / "data/evalplus_local/tasks.jsonl",
            ROOT / "data/swesmith_local/tasks.jsonl",
        ]
    return [path for path in paths if path.is_file()]


def load_task_splits(paths: list[Path]) -> dict[str, str]:
    splits: dict[str, str] = {}
    for path in paths:
        for row in read_jsonl(path):
            instance_id = row.get("instance_id")
            split = row.get("split")
            if not instance_id or not split:
                continue
            key = str(instance_id)
            value = str(split)
            previous = splits.get(key)
            if previous is not None and previous != value:
                raise ValueError(f"Conflicting task splits for {key}: {previous} vs {value}")
            splits[key] = value
    return splits


if __name__ == "__main__":
    main()
