#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = (
    ("SWE-Gym/OpenHands-SFT-Trajectories", "train.success.oss", "sft_rows.jsonl"),
    ("SWE-Gym/OpenHands-Verifier-Trajectories", "train.mixture", "verifier_rows.jsonl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download small official SWE-Gym OpenHands trajectory samples.")
    parser.add_argument("--output-dir", default="data/official_openhands_samples")
    parser.add_argument("--limit", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for dataset, split, filename in DEFAULT_DATASETS:
        rows = load_rows(dataset, split, args.limit)
        output = output_dir / filename
        with output.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(normalize_row(row), ensure_ascii=False) + "\n")
        manifest.append({"dataset": dataset, "split": split, "rows": len(rows), "output": str(output)})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "manifest": manifest}, ensure_ascii=False, indent=2))


def load_rows(dataset: str, split: str, limit: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the SWE-Gym extra first: `pip install -e .[swegym]`.") from exc

    ds = load_dataset(dataset, split=split, streaming=True)
    rows = []
    for row in ds:
        rows.append(dict(row))
        if len(rows) >= limit:
            break
    return rows


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = json_safe(row)
    for key in ("messages", "trajectory", "tools"):
        value = normalized.get(key)
        if isinstance(value, str):
            parsed = parse_embedded_sequence(value)
            if parsed is not None:
                normalized[key] = json_safe(parsed)
    return normalized


def parse_embedded_sequence(value: str) -> Any | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None


if __name__ == "__main__":
    main()
