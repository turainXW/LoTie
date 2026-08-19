#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a JSONL subset of SWE-Gym tasks for selected patch records.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records_path = Path(args.records)
    task_path = Path(args.tasks)
    output_path = Path(args.output)
    ids = [
        json.loads(line)["instance_id"]
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    df = pd.read_parquet(task_path)
    subset = df[df["instance_id"].isin(ids)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in subset.to_dict(orient="records"):
            f.write(json.dumps(json_safe(row), ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"output": str(output_path), "records": len(ids), "tasks": len(subset)}, ensure_ascii=False))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return value


if __name__ == "__main__":
    main()
