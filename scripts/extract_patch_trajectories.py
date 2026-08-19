#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract readable SWE-Gym patch trajectories from rollout JSONL.")
    parser.add_argument("--records", required=True, help="openhands_patch_rollout.jsonl")
    parser.add_argument("--tasks", help="Optional tasks_subset.jsonl for problem statements and FAIL_TO_PASS.")
    parser.add_argument("--verifier", help="Optional benchmark_verify_dryrun.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-message-chars", type=int, default=2200)
    parser.add_argument("--max-patch-lines", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.records)
    tasks = keyed(read_jsonl(args.tasks)) if args.tasks else {}
    verifier = keyed(read_jsonl(args.verifier)) if args.verifier else {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compact_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        instance_id = str(record.get("instance_id"))
        task = tasks.get(instance_id, {})
        verify = verifier.get(instance_id, {})
        compact = compact_record(record, task, verify)
        compact_rows.append(compact)
        index_rows.append(compact)
        safe_name = safe_filename(f"{idx:03d}_{instance_id}")
        (output_dir / f"{safe_name}.md").write_text(
            render_markdown(record, task, verify, args),
            encoding="utf-8",
        )

    write_jsonl(output_dir / "trajectory_summary.jsonl", compact_rows)
    (output_dir / "trajectory_index.md").write_text(render_index(index_rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "trajectories": len(records)}, ensure_ascii=False, indent=2))


def compact_record(record: dict[str, Any], task: dict[str, Any], verifier: dict[str, Any]) -> dict[str, Any]:
    verify = (((record.get("test_result") or {}).get("report") or {}).get("verify") or {})
    compile_results = verify.get("compile_results") or []
    return {
        "instance_id": record.get("instance_id"),
        "repo": record.get("repo"),
        "status": record.get("status"),
        "resolved": record.get("resolved"),
        "benchmark_resolved": verifier.get("benchmark_resolved"),
        "verifier_status": verifier.get("verifier_status"),
        "edited_files": (record.get("edit_apply") or {}).get("edited_files") or [],
        "gold_files": (record.get("metadata") or {}).get("gold_files") or [],
        "gold_file_hit": (record.get("metadata") or {}).get("gold_file_hit_after_edit"),
        "compile_attempted": bool(compile_results),
        "compile_failed": bool(verify.get("compile_failed")),
        "tests_attempted": bool(verify.get("tests_attempted")),
        "patch_lines": len(str(record.get("patch") or "").splitlines()),
        "message_count": len(record.get("messages") or []),
        "fail_to_pass": task.get("FAIL_TO_PASS") or [],
        "test_command": verifier.get("test_command"),
    }


def render_index(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SWE-Gym Patch Trajectory Index",
        "",
        "| # | instance | status | edited | gold hit | compile | verifier |",
        "|---:|---|---|---|---|---|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        edited = ", ".join(row["edited_files"]) or "-"
        compile_state = "failed" if row["compile_failed"] else "passed" if row["compile_attempted"] else "not-run"
        lines.append(
            "| {idx} | `{instance}` | `{status}` | {edited} | {gold_hit} | {compile_state} | `{verifier}` |".format(
                idx=idx,
                instance=row["instance_id"],
                status=row["status"],
                edited=edited,
                gold_hit=row["gold_file_hit"],
                compile_state=compile_state,
                verifier=row["verifier_status"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_markdown(record: dict[str, Any], task: dict[str, Any], verifier: dict[str, Any], args: argparse.Namespace) -> str:
    compact = compact_record(record, task, verifier)
    messages = record.get("messages") or []
    lines = [
        f"# {record.get('instance_id')}",
        "",
        "## Summary",
        "",
        fenced_json(compact),
        "",
        "## Problem",
        "",
        truncate(str(task.get("problem_statement") or first_user_problem(messages)), args.max_message_chars),
        "",
        "## Tool / Message Trace",
        "",
    ]
    for idx, message in enumerate(messages, start=1):
        role = message.get("role")
        name = message.get("name")
        title = f"{idx}. {role}" + (f": {name}" if name else "")
        lines.extend([f"### {title}", ""])
        content = str(message.get("content", ""))
        if looks_like_json(content):
            lines.append(fenced_json(parse_json_or_raw(content, args.max_message_chars)))
        else:
            lines.append("```text")
            lines.append(truncate(content, args.max_message_chars))
            lines.append("```")
        lines.append("")

    patch_lines = str(record.get("patch") or "").splitlines()
    lines.extend(["## Patch", "", "```diff"])
    lines.extend(patch_lines[: args.max_patch_lines])
    if len(patch_lines) > args.max_patch_lines:
        lines.append(f"... truncated {len(patch_lines) - args.max_patch_lines} lines ...")
    lines.extend(["```", ""])
    return "\n".join(lines)


def first_user_problem(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = str(message.get("content", ""))
            parsed = parse_json_or_raw(content, 4000)
            if isinstance(parsed, dict):
                return str(parsed.get("problem_statement") or content)
            return content
    return ""


def looks_like_json(content: str) -> bool:
    stripped = content.strip()
    return stripped.startswith("{") or stripped.startswith("[")


def parse_json_or_raw(content: str, max_chars: int) -> Any:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return truncate(content, max_chars)
    return shrink(value, max_chars)


def shrink(value: Any, max_chars: int) -> Any:
    if isinstance(value, dict):
        return {key: shrink(item, max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [shrink(item, max_chars) for item in value[:20]]
    if isinstance(value, str):
        return truncate(value, max_chars)
    return value


def truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n... truncated {len(value) - max_chars} chars ..."


def fenced_json(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("instance_id")): row for row in rows if row.get("instance_id") is not None}


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


if __name__ == "__main__":
    main()
