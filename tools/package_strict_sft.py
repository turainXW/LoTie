#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.sft_masking import validate_json_tool_call  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a validated strict-SFT-only delivery directory.")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_dir = Path(args.clean_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    clean_manifest_path = clean_dir / "manifest.json"
    clean_manifest = read_json(clean_manifest_path)
    source_sft = clean_dir / "sft_success.jsonl"
    rows = read_jsonl(source_sft)
    if not rows:
        raise ValueError("Strict SFT input is empty")

    seen_ids: set[str] = set()
    assistant_calls = 0
    finish_rows = 0
    message_counts: list[int] = []
    message_chars: list[int] = []
    for index, row in enumerate(rows):
        trajectory_id = str(row.get("trajectory_id") or "")
        if not trajectory_id or trajectory_id in seen_ids:
            raise ValueError(f"Missing or duplicate trajectory_id at row {index}: {trajectory_id!r}")
        seen_ids.add(trajectory_id)
        if row.get("resolved") is not True or row.get("exclude_from_training") is True:
            raise ValueError(f"Non-trainable row in strict SFT at index {index}")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Missing messages at row {index}")
        for message_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            validate_json_tool_call(str(message.get("content") or ""), message_index=message_index)
            assistant_calls += 1
        last = messages[-1]
        if last.get("role") != "assistant":
            raise ValueError(f"Strict trajectory does not end in assistant at row {index}")
        finish_rows += int(validate_json_tool_call(str(last["content"]))["tool_name"] == "finish")
        message_counts.append(len(messages))
        message_chars.append(sum(len(str(message.get("content") or "")) for message in messages))
    if finish_rows != len(rows):
        raise ValueError(f"Only {finish_rows}/{len(rows)} strict rows end with finish")

    output_dir.mkdir(parents=True)
    outputs = {
        "sft_strict.jsonl": source_sft,
        "sft_exclusions.jsonl": clean_dir / "sft_exclusions.jsonl",
        "duplicates.jsonl": clean_dir / "duplicates.jsonl",
    }
    for name, source in outputs.items():
        shutil.copy2(source, output_dir / name)

    data_files = {
        name: {
            "bytes": (output_dir / name).stat().st_size,
            "sha256": file_sha256(output_dir / name),
            "rows": count_lines(output_dir / name),
        }
        for name in outputs
    }
    manifest: dict[str, Any] = {
        "format": "lottie_strict_sft_package_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": clean_manifest.get("source_run"),
        "source_clean_manifest_sha256": file_sha256(clean_manifest_path),
        "source_manifest_sha256": clean_manifest.get("source_manifest_sha256"),
        "source_sample_result_count": clean_manifest.get("source_sample_result_count"),
        "source_valid_samples": (clean_manifest.get("counts") or {}).get("valid_samples"),
        "source_resolved_samples": (clean_manifest.get("counts") or {}).get("rl_positive"),
        "strict_sft_rows": len(rows),
        "unique_trajectory_ids": len(seen_ids),
        "assistant_json_calls_validated": assistant_calls,
        "terminal_finish_rows": finish_rows,
        "message_count": {"min": min(message_counts), "max": max(message_counts)},
        "message_chars": {"min": min(message_chars), "max": max(message_chars)},
        "training_state": "clean_messages_not_tokenized",
        "required_next_step": "tokenize with the exact target model tokenizer, then apply assistant-only token loss masks",
        "files": data_files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checksum_paths = [output_dir / name for name in outputs] + [manifest_path]
    checksum_text = "".join(f"{file_sha256(path)}  {path.name}\n" for path in checksum_paths)
    (output_dir / "SHA256SUMS").write_text(checksum_text, encoding="ascii")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


if __name__ == "__main__":
    main()
