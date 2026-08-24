#!/usr/bin/env python3
"""Build an auditable terminal-return dataset for SAO critic cold-start.

The builder keeps every trajectory intact. It tokenizes each message with the
model's native chat template, verifies that message-wise tokenization exactly
matches whole-conversation tokenization, and supervises only assistant action
tokens. System, task, and tool-observation tokens remain visible context but
receive a zero value-loss mask.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from transformers import AutoTokenizer


CHATML_ASSISTANT_HEADER = "<|im_start|>assistant\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument(
        "--additional-source",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help=(
            "Add a finalized sample_results.jsonl source without changing the "
            "primary rollout root. May be repeated."
        ),
    )
    parser.add_argument(
        "--fixed-validation-samples",
        type=Path,
        help=(
            "Reuse validation groups from an existing Critic samples.jsonl so "
            "new rollouts of held-out tasks remain held out."
        ),
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["function", "swesmith_py"])
    parser.add_argument("--max-seq-len", type=int, default=65536)
    parser.add_argument(
        "--overlength-policy",
        choices=("error", "exclude"),
        default="error",
        help=(
            "Fail on trajectories above --max-seq-len, or quarantine them in "
            "excluded_overlength.jsonl without truncation."
        ),
    )
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--split-seed", default="lottie-sao-critic-v1")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def read_single_jsonl(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one trajectory record in {path}, found {len(rows)}")
    return rows[0]


def parse_additional_sources(values: list[str]) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for value in values:
        dataset, separator, raw_path = value.partition("=")
        if not separator or not dataset or not raw_path:
            raise ValueError(
                f"Invalid --additional-source {value!r}; expected DATASET=PATH"
            )
        sources.append((dataset, Path(raw_path)))
    return sources


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_score(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return sorted(values)[index]


def group_key(row: dict[str, Any]) -> str:
    if row["dataset_family"] == "swesmith":
        return f"repo:{row['repo']}"
    return f"task:{row['instance_id']}"


def choose_validation_groups(
    rows: list[dict[str, Any]], ratio: float, seed: str, min_groups: int = 1
) -> set[str]:
    """Choose deterministic groups close to the requested row count.

    SWE-smith is split by repository; function tasks are split by task ID.
    This prevents closely related repository states from crossing the split.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    target = max(1, round(len(rows) * ratio))
    ordered = sorted(grouped, key=lambda key: stable_score(seed, key))
    selected: set[str] = set()
    count = 0

    # A repository holdout made from one large repository is leakage-safe but
    # too brittle to drive early stopping. Search deterministic repository
    # combinations for both row-count and class-balance fidelity.
    if min_groups > 1:
        target_positives = round(
            target * sum(int(row["reward"]) for row in rows) / len(rows)
        )
        candidates = []
        for combination in itertools.combinations(ordered, min_groups):
            combination_rows = [row for key in combination for row in grouped[key]]
            combination_count = len(combination_rows)
            combination_positives = sum(int(row["reward"]) for row in combination_rows)
            candidates.append(
                (
                    abs(combination_count - target),
                    abs(combination_positives - target_positives),
                    stable_score(seed, "|".join(sorted(combination))),
                    combination,
                )
            )
        if not candidates:
            raise ValueError(f"Cannot choose {min_groups} validation groups")
        selected.update(min(candidates)[-1])
        count = sum(len(grouped[key]) for key in selected)

    for key in ordered:
        if key in selected:
            continue
        size = len(grouped[key])
        if not selected or abs(target - (count + size)) <= abs(target - count):
            selected.add(key)
            count += size
        if count >= target:
            break

    # A calibration split must contain both terminal outcomes when available.
    all_labels = {int(row["reward"]) for row in rows}
    selected_labels = {
        int(row["reward"])
        for key in selected
        for row in grouped[key]
    }
    for missing in sorted(all_labels - selected_labels):
        candidates = [
            key
            for key in ordered
            if key not in selected
            and any(int(row["reward"]) == missing for row in grouped[key])
        ]
        if candidates:
            selected.add(candidates[0])
    return selected


def tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def template_ids(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return encoded["input_ids"] if hasattr(encoded, "keys") else encoded


def tokenize_with_action_mask(
    tokenizer: Any, messages: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, int]:
    if not messages or messages[0].get("role") != "system":
        raise ValueError("Trajectory must begin with a system message")

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    segment_texts = [
        tokenizer.apply_chat_template(
            [message],
            tokenize=False,
            add_generation_prompt=False,
        )
        for message in messages
    ]
    if full_text != "".join(segment_texts):
        raise ValueError("Message-wise chat rendering does not match full rendering")

    full_ids = template_ids(tokenizer, messages)
    concatenated_ids: list[int] = []
    action_mask: list[int] = []
    assistant_turns = 0

    for message, segment_text in zip(messages, segment_texts, strict=True):
        segment_ids = tokenizer_ids(tokenizer, segment_text)
        concatenated_ids.extend(segment_ids)
        segment_mask = [0] * len(segment_ids)
        if message.get("role") == "assistant":
            assistant_turns += 1
            if not segment_text.startswith(CHATML_ASSISTANT_HEADER):
                raise ValueError("Unexpected assistant rendering; refusing an approximate mask")
            header_ids = tokenizer_ids(tokenizer, CHATML_ASSISTANT_HEADER)
            if segment_ids[: len(header_ids)] != header_ids:
                raise ValueError("Assistant header tokens do not align with the rendered segment")
            for index in range(len(header_ids), len(segment_ids)):
                segment_mask[index] = 1
        action_mask.extend(segment_mask)

    if concatenated_ids != full_ids:
        raise ValueError("Message-wise token IDs do not match whole-conversation token IDs")
    if len(action_mask) != len(full_ids):
        raise AssertionError("Action mask length mismatch")
    if not any(action_mask):
        raise ValueError("Trajectory has no assistant action tokens")

    return (
        np.asarray(full_ids, dtype=np.int32),
        np.asarray(action_mask, dtype=np.uint8),
        assistant_turns,
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(row["num_tokens"]) for row in rows]
    actions = [int(row["num_action_tokens"]) for row in rows]
    return {
        "rows": len(rows),
        "rewards": dict(sorted(Counter(str(int(row["reward"])) for row in rows).items())),
        "datasets": dict(sorted(Counter(row["dataset_family"] for row in rows).items())),
        "tokens": {
            "total": sum(lengths),
            "min": min(lengths, default=0),
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths, default=0),
        },
        "action_tokens": {
            "total": sum(actions),
            "min": min(actions, default=0),
            "p50": percentile(actions, 0.50),
            "p90": percentile(actions, 0.90),
            "p95": percentile(actions, 0.95),
            "max": max(actions, default=0),
        },
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_ratio < 0.5:
        raise ValueError("--validation-ratio must be between 0 and 0.5")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    raw_rows: list[dict[str, Any]] = []
    source_files: list[Path] = []
    seen_slots: set[tuple[str, str, str, int]] = set()
    seen_trajectory_hashes: set[str] = set()
    source_specs = [
        (source_name, args.rollout_root / source_name / "sample_results.jsonl")
        for source_name in args.datasets
    ]
    source_specs.extend(parse_additional_sources(args.additional_source))
    for source_name, source_path in source_specs:
        if source_name not in args.datasets:
            raise ValueError(
                f"Additional source dataset {source_name!r} is not in --datasets"
            )
        source_files.append(source_path)
        for result in read_jsonl(source_path):
            if not bool(result.get("sample_valid")):
                raise ValueError(f"Invalid sample in finalized source: {result.get('instance_id')}")
            if result.get("data_role") != "train" or bool(result.get("exclude_from_training")):
                raise ValueError(f"Non-train sample reached critic source: {result.get('instance_id')}")
            slot = (
                str(source_path.resolve()),
                str(result["dataset"]),
                str(result["instance_id"]),
                int(result["sample_index"]),
            )
            if slot in seen_slots:
                raise ValueError(f"Duplicate rollout slot: {slot}")
            seen_slots.add(slot)

            trajectory_path = Path(result["trajectory_path"])
            trajectory = read_single_jsonl(trajectory_path)
            trajectory_sha256 = sha256_file(trajectory_path)
            if trajectory_sha256 in seen_trajectory_hashes:
                raise ValueError(
                    f"Duplicate trajectory content across finalized sources: {trajectory_path}"
                )
            seen_trajectory_hashes.add(trajectory_sha256)
            reward = float(bool(result.get("benchmark_resolved")))
            verifier_path = Path(result["verifier_path"])
            verifier = read_single_jsonl(verifier_path)
            if str(verifier.get("instance_id")) != str(result["instance_id"]):
                raise ValueError(f"Verifier instance mismatch for {slot}")
            if reward != float(bool(verifier.get("benchmark_resolved"))):
                raise ValueError(f"Summary/verifier reward mismatch for {slot}")
            family = "swesmith" if source_name == "swesmith_py" else "function"
            raw_rows.append(
                {
                    "dataset_family": family,
                    "dataset": result["dataset"],
                    "instance_id": result["instance_id"],
                    "sample_index": int(result["sample_index"]),
                    "repo": result.get("repo") or trajectory.get("repo") or "",
                    "reward": reward,
                    "agent_status": result.get("agent_status"),
                    "verifier_status": result.get("verifier_status"),
                    "source_results_path": str(source_path.resolve()),
                    "trajectory_path": str(trajectory_path),
                    "trajectory_sha256": trajectory_sha256,
                    "verifier_path": str(verifier_path),
                    "verifier_sha256": sha256_file(verifier_path),
                    "messages": trajectory["messages"],
                }
            )

    if args.fixed_validation_samples:
        fixed_rows = read_jsonl(args.fixed_validation_samples)
        fixed_validation_rows = [
            row for row in fixed_rows if row.get("split") == "validation"
        ]
        if not fixed_validation_rows:
            raise ValueError("Fixed validation samples contain no validation rows")
        validation_groups = {group_key(row) for row in fixed_validation_rows}
        for family in ("function", "swesmith"):
            if not any(
                row["dataset_family"] == family for row in fixed_validation_rows
            ):
                raise ValueError(
                    f"Fixed validation samples contain no {family} validation rows"
                )
    else:
        validation_groups = set()
        for family in ("function", "swesmith"):
            family_rows = [row for row in raw_rows if row["dataset_family"] == family]
            validation_groups.update(
                choose_validation_groups(
                    family_rows,
                    args.validation_ratio,
                    f"{args.split_seed}:{family}",
                    min_groups=4 if family == "swesmith" else 1,
                )
            )

    input_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    excluded_overlength: list[dict[str, Any]] = []
    offset = 0
    for row in raw_rows:
        input_ids, action_mask, assistant_turns = tokenize_with_action_mask(
            tokenizer,
            row.pop("messages"),
        )
        split = "validation" if group_key(row) in validation_groups else "train"
        if len(input_ids) > args.max_seq_len:
            if args.overlength_policy == "error":
                raise ValueError(
                    f"{row['instance_id']} has {len(input_ids)} tokens, above "
                    f"{args.max_seq_len}; refusing silent truncation"
                )
            excluded_overlength.append(
                {
                    **row,
                    "split": split,
                    "num_tokens": int(len(input_ids)),
                    "num_action_tokens": int(action_mask.sum()),
                    "assistant_turns": assistant_turns,
                    "exclude_reason": "over_max_seq_len",
                    "max_seq_len": args.max_seq_len,
                }
            )
            continue
        item = {
            **row,
            "record_index": len(metadata),
            "split": split,
            "offset": offset,
            "num_tokens": int(len(input_ids)),
            "num_action_tokens": int(action_mask.sum()),
            "assistant_turns": assistant_turns,
            "value_target": "terminal_verifier_return_gamma1",
        }
        metadata.append(item)
        input_parts.append(input_ids)
        mask_parts.append(action_mask)
        offset += len(input_ids)

    all_input_ids = np.concatenate(input_parts)
    all_action_mask = np.concatenate(mask_parts)
    if len(all_input_ids) != len(all_action_mask):
        raise AssertionError("Concatenated token/mask length mismatch")

    np.save(args.output_dir / "input_ids.npy", all_input_ids, allow_pickle=False)
    np.save(args.output_dir / "action_mask.npy", all_action_mask, allow_pickle=False)
    write_jsonl(args.output_dir / "samples.jsonl", metadata)
    write_jsonl(args.output_dir / "excluded_overlength.jsonl", excluded_overlength)

    train_rows = [row for row in metadata if row["split"] == "train"]
    validation_rows = [row for row in metadata if row["split"] == "validation"]
    split_group_counts = {
        split: {
            family: len(
                {
                    group_key(row)
                    for row in metadata
                    if row["split"] == split and row["dataset_family"] == family
                }
            )
            for family in ("function", "swesmith")
        }
        for split in ("train", "validation")
    }
    manifest = {
        "format": "lottie_sao_critic_terminal_return_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rollout_root": str(args.rollout_root.resolve()),
        "additional_sources": [
            {"dataset": dataset, "path": str(path.resolve())}
            for dataset, path in parse_additional_sources(args.additional_source)
        ],
        "model_path": str(args.model_path.resolve()),
        "tokenizer_class": tokenizer.__class__.__name__,
        "max_seq_len": args.max_seq_len,
        "overlength_policy": args.overlength_policy,
        "validation_ratio_requested": args.validation_ratio,
        "split_seed": args.split_seed,
        "split_policy": {
            "function": "instance_id",
            "swesmith": "repository_minimum_4_validation_repositories",
            "fixed_validation_samples": (
                str(args.fixed_validation_samples.resolve())
                if args.fixed_validation_samples
                else None
            ),
        },
        "split_group_counts": split_group_counts,
        "loss_mask": {
            "assistant_action_tokens": 1,
            "system_user_tool_observation_tokens": 0,
            "assistant_header_tokens": 0,
            "assistant_end_of_turn_tokens": 1,
        },
        "target": {
            "kind": "terminal_verifier_return",
            "gamma": 1.0,
            "positive": "benchmark_resolved=true",
            "negative": "benchmark_resolved=false",
        },
        "no_truncation": True,
        "no_eval_data": True,
        "source_sha256": {str(path): sha256_file(path) for path in source_files},
        "source_records": len(raw_rows),
        "excluded_overlength": summarize(excluded_overlength),
        "all": summarize(metadata),
        "train": summarize(train_rows),
        "validation": summarize(validation_rows),
        "artifacts": {
            "input_ids": "input_ids.npy",
            "action_mask": "action_mask.npy",
            "samples": "samples.jsonl",
            "excluded_overlength": "excluded_overlength.jsonl",
        },
    }
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    audit = {
        "records": len(metadata),
        "source_records": len(raw_rows),
        "excluded_overlength_count": len(excluded_overlength),
        "unique_slots": len(seen_slots),
        "unique_trajectory_hashes": len(seen_trajectory_hashes),
        "token_mask_length_equal": True,
        "whole_vs_message_tokenization_equal": True,
        "all_have_action_tokens": all(row["num_action_tokens"] > 0 for row in metadata),
        "max_observed_tokens": max(row["num_tokens"] for row in metadata),
        "max_source_tokens": max(
            row["num_tokens"] for row in metadata + excluded_overlength
        ),
        "max_allowed_tokens": args.max_seq_len,
        "trajectory_source_hashes_present": all(bool(row["trajectory_sha256"]) for row in metadata),
        "verifier_source_hashes_present": all(bool(row["verifier_sha256"]) for row in metadata),
        "train_validation_group_overlap": bool(
            {group_key(row) for row in train_rows}
            & {group_key(row) for row in validation_rows}
        ),
    }
    if audit["train_validation_group_overlap"]:
        raise AssertionError("Grouped split leaked across train and validation")
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"manifest": manifest, "audit": audit}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
