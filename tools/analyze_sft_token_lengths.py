from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jinja2 import Environment
from tokenizers import Tokenizer


BUCKET_LIMITS = (4096, 8192, 16384, 32768)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure conversational SFT rows with an exact tokenizer and chat template.")
    parser.add_argument("--input", required=True, help="JSONL containing a messages field.")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json.")
    parser.add_argument("--tokenizer-config", required=True, help="Path to tokenizer_config.json with chat_template.")
    parser.add_argument("--output", help="Optional JSON report path.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("messages"), list):
                raise ValueError(f"{path}:{line_number}: messages must be a list")
            rows.append(row)
    return rows


def percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = max(0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * fraction) - 1))
    return sorted_values[index]


def bucket_name(length: int) -> str:
    previous = 0
    for limit in BUCKET_LIMITS:
        if length <= limit:
            return f"{previous + 1}-{limit}"
        previous = limit
    return f">{BUCKET_LIMITS[-1]}"


def summarize(lengths: list[int]) -> dict[str, Any]:
    ordered = sorted(lengths)
    return {
        "rows": len(ordered),
        "total_tokens": sum(ordered),
        "mean_tokens": round(sum(ordered) / len(ordered), 2) if ordered else 0,
        "min_tokens": ordered[0] if ordered else 0,
        "p50_tokens": percentile(ordered, 0.50),
        "p75_tokens": percentile(ordered, 0.75),
        "p90_tokens": percentile(ordered, 0.90),
        "p95_tokens": percentile(ordered, 0.95),
        "p99_tokens": percentile(ordered, 0.99),
        "max_tokens": ordered[-1] if ordered else 0,
        "buckets": dict(sorted(Counter(bucket_name(value) for value in ordered).items())),
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    tokenizer_config = json.loads(Path(args.tokenizer_config).read_text(encoding="utf-8"))
    template_source = tokenizer_config.get("chat_template")
    if not isinstance(template_source, str) or not template_source.strip():
        raise ValueError("tokenizer config has no chat_template")
    template = Environment(autoescape=False).from_string(template_source)

    measured: list[dict[str, Any]] = []
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        rendered = template.render(
            messages=row["messages"],
            tools=None,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        length = len(tokenizer.encode(rendered, add_special_tokens=False).ids)
        dataset = str(row.get("dataset") or "unknown")
        by_dataset[dataset].append(length)
        measured.append(
            {
                "trajectory_id": row.get("trajectory_id"),
                "instance_id": row.get("instance_id"),
                "sample_index": row.get("sample_index"),
                "dataset": dataset,
                "quality_tier": row.get("quality_tier"),
                "messages": len(row["messages"]),
                "tokens": length,
            }
        )

    report = {
        "format": "lottie_sft_token_length_report_v1",
        "input": str(Path(args.input).resolve()),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "tokenizer_class": tokenizer_config.get("tokenizer_class"),
        "model_max_length": tokenizer_config.get("model_max_length"),
        "chat_template_applied": True,
        "overall": summarize([row["tokens"] for row in measured]),
        "by_dataset": {name: summarize(lengths) for name, lengths in sorted(by_dataset.items())},
        "longest": sorted(measured, key=lambda row: row["tokens"], reverse=True)[:20],
    }
    rendered_report = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered_report, encoding="utf-8")
    print(rendered_report, end="")


if __name__ == "__main__":
    main()
