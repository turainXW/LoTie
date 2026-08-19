#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


NON_TRAINABLE_STATUSES = {
    "context_overflow",
    "model_api_failure",
    "parse_error",
    "infrastructure_blocked",
    "runner_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export rollout messages_sft.jsonl to an OpenHands SFT-compatible dataset.")
    parser.add_argument("--input", required=True, help="Path to messages_sft.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-unresolved", action="store_true", help="Keep unresolved records instead of success-only SFT.")
    parser.add_argument(
        "--verifier-results",
        help="Optional verifier JSONL. When provided, benchmark_resolved is authoritative for success filtering.",
    )
    parser.add_argument("--split-name", default="train.success.oss")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(input_path)
    verifier_results = {
        str(row.get("instance_id")): row
        for row in read_jsonl(Path(args.verifier_results))
    } if args.verifier_results else {}
    messages = []
    kept_instance_ids = []
    for row in source_rows:
        if row.get("exclude_from_training") or row.get("data_role") == "evaluation_only":
            continue
        verifier_row = verifier_results.get(str(row.get("instance_id")))
        resolved = bool(verifier_row.get("benchmark_resolved")) if verifier_row is not None else bool(row.get("resolved", True))
        if row.get("status") in NON_TRAINABLE_STATUSES:
            resolved = False
        if not args.include_unresolved and not resolved:
            continue
        row_messages = row.get("messages")
        if not is_valid_messages(row_messages):
            continue
        messages.append(row_messages)
        kept_instance_ids.append(row.get("instance_id"))

    parquet_path = output_dir / f"{args.split_name}-00000-of-00001.parquet"
    jsonl_path = output_dir / f"{args.split_name}.jsonl"
    write_parquet(parquet_path, messages)
    write_jsonl(jsonl_path, messages)

    manifest = {
        "format": "openhands_sft_messages_v1",
        "source": str(input_path),
        "rows_in": len(source_rows),
        "rows_out": len(messages),
        "include_unresolved": bool(args.include_unresolved),
        "verifier_results": str(Path(args.verifier_results).resolve()) if args.verifier_results else None,
        "success_source": "verifier.benchmark_resolved" if args.verifier_results else "rollout.resolved",
        "split_name": args.split_name,
        "schema": "messages: list<struct<content: string, role: string>>",
        "parquet": str(parquet_path),
        "jsonl": str(jsonl_path),
        "instance_ids": kept_instance_ids,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_valid_messages(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(item, dict) and isinstance(item.get("role"), str) and isinstance(item.get("content"), str) for item in value)


def write_parquet(path: Path, messages: list[list[dict[str, str]]]) -> None:
    message_type = pa.list_(pa.struct([pa.field("content", pa.string()), pa.field("role", pa.string())]))
    table = pa.table({"messages": pa.array(messages, type=message_type)})
    pq.write_table(table, path)


def write_jsonl(path: Path, messages: list[list[dict[str, str]]]) -> None:
    path.write_text("".join(json.dumps({"messages": item}, ensure_ascii=False) + "\n" for item in messages), encoding="utf-8")


if __name__ == "__main__":
    main()
