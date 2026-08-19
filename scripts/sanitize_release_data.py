#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


MACHINE_LOCAL_FIELDS = {"local_repo_path", "local_venv_path"}
PrefixReplacement = tuple[str, str]


def sanitize(value: Any, replacements: Iterable[PrefixReplacement] = ()) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize(item, replacements)
            for key, item in value.items()
            if key not in MACHINE_LOCAL_FIELDS
        }
    if isinstance(value, list):
        return [sanitize(item, replacements) for item in value]
    if isinstance(value, str):
        for source, target in replacements:
            value = value.replace(source, target)
    return value


def sanitize_jsonl(
    path: Path,
    replacements: Iterable[PrefixReplacement] = (),
) -> int:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(sanitize(json.loads(line), replacements))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def sanitize_json(
    path: Path,
    replacements: Iterable[PrefixReplacement] = (),
) -> int:
    value = sanitize(json.loads(path.read_text(encoding="utf-8")), replacements)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(value) if isinstance(value, list) else 1


def sanitize_path(
    path: Path,
    replacements: Iterable[PrefixReplacement] = (),
) -> int:
    if path.suffix == ".jsonl":
        return sanitize_jsonl(path, replacements)
    if path.suffix == ".json":
        return sanitize_json(path, replacements)
    raise ValueError(f"unsupported JSON path: {path}")


def parse_replacement(raw: str) -> PrefixReplacement:
    source, separator, target = raw.partition("=")
    if not separator or not source or not target:
        raise argparse.ArgumentTypeError("replacement must use SOURCE=TARGET")
    return source.rstrip("/"), target.rstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove machine-local fields and path prefixes from release JSON/JSONL data."
    )
    parser.add_argument(
        "--replace-prefix",
        action="append",
        default=[],
        type=parse_replacement,
        metavar="SOURCE=TARGET",
        help="Replace a machine-local path prefix in every string value; may be repeated.",
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    replacements = sorted(args.replace_prefix, key=lambda item: len(item[0]), reverse=True)
    for path in args.paths:
        print(f"sanitized={path} items={sanitize_path(path, replacements)}")


if __name__ == "__main__":
    main()
