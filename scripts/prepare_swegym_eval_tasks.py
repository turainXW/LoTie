#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SWE-Gym/SWE-Gym-Lite task JSONL with derived eval image names."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tasks", help="Local SWE-Gym task parquet/jsonl file.")
    source.add_argument("--dataset", help="Hugging Face dataset id, e.g. SWE-Gym/SWE-Gym-Lite.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--namespace", default="xingyaoww")
    parser.add_argument("--arch", default="x86_64")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--manifest-timeout-sec", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args)
    if args.limit is not None:
        rows = rows[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest_counts: dict[str, int] = {}
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            prepared = json_safe(row)
            image = derive_sweb_eval_image(
                str(prepared["instance_id"]),
                namespace=args.namespace,
                arch=args.arch,
                tag=args.tag,
            )
            prepared["image"] = image
            prepared["docker_image"] = image
            if args.check_manifest:
                status = check_manifest(image, timeout_sec=args.manifest_timeout_sec)
                prepared["image_manifest_status"] = status
                manifest_counts[status] = manifest_counts.get(status, 0) + 1
            f.write(json.dumps(prepared, ensure_ascii=False) + "\n")

    summary = {
        "output": str(output),
        "tasks": len(rows),
        "namespace": args.namespace,
        "arch": args.arch,
        "tag": args.tag,
        "manifest_counts": manifest_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.tasks:
        path = Path(args.tasks)
        if path.suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise RuntimeError("Reading parquet task files requires pandas and pyarrow.") from exc
            return pd.read_parquet(path).to_dict(orient="records")
        return read_jsonl(path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Loading Hugging Face datasets requires `pip install -e .[swegym]`.") from exc
    dataset = load_dataset(args.dataset, split=args.split)
    return [dict(row) for row in dataset]


def derive_sweb_eval_image(instance_id: str, *, namespace: str, arch: str, tag: str) -> str:
    if "__" not in instance_id:
        raise ValueError(f"Cannot derive SWE eval image from instance_id: {instance_id}")
    owner, issue_part = instance_id.split("__", 1)
    image_key = f"{owner}_s_{issue_part}".lower()
    return f"{namespace}/sweb.eval.{arch}.{image_key}:{tag}"


def check_manifest(image: str, *, timeout_sec: int) -> str:
    if subprocess.run(["which", "docker"], capture_output=True, text=True).returncode != 0:
        return "docker_unavailable"
    completed = subprocess.run(
        ["docker", "manifest", "inspect", image],
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    return "ok" if completed.returncode == 0 else "missing_or_inaccessible"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
