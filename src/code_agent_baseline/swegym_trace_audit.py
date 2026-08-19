from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceMiss:
    idx: int
    instance_id: str
    repo: str
    status: str
    parse_ok: bool
    pred_files: list[str]
    candidate_hit: bool
    gold_files: list[str]
    error: str | None = None


@dataclass
class TraceAuditReport:
    run_id: str
    total: int
    ok: int
    parse_ok: int
    missing_required: list[dict[str, Any]] = field(default_factory=list)
    parsed_key_coverage: dict[str, int] = field(default_factory=dict)
    candidate_hit: int = 0
    candidate_hit_rate: float = 0.0
    file_hit: int = 0
    file_hit_rate_all: float = 0.0
    file_hit_rate_ok: float = 0.0
    candidate_count_min_max: list[int] = field(default_factory=list)
    pred_file_count_min_max: list[int] = field(default_factory=list)
    tool_plan_steps_min_max: list[int] = field(default_factory=list)
    latency_avg: float = 0.0
    latency_p50: float = 0.0
    misses: list[TraceMiss] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["misses"] = [asdict(item) for item in self.misses]
        return data


def audit_grounded_traces(
    traces_path: str | Path,
    *,
    tasks_path: str | Path | None = None,
) -> TraceAuditReport:
    traces = Path(traces_path)
    rows = [json.loads(line) for line in traces.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = load_tasks_by_instance(tasks_path) if tasks_path else {}

    required = [
        "instance_id",
        "repo",
        "base_commit",
        "candidate_files",
        "prompt",
        "answer",
        "parsed",
        "parse_ok",
        "status",
    ]
    missing_required: list[dict[str, Any]] = []
    parsed_key_coverage: dict[str, int] = {}
    misses: list[TraceMiss] = []
    candidate_counts: list[int] = []
    pred_counts: list[int] = []
    tool_steps: list[int] = []
    latencies: list[float] = []
    ok = 0
    parse_ok = 0
    candidate_hit = 0
    file_hit = 0

    for idx, row in enumerate(rows, start=1):
        missing = [key for key in required if key not in row]
        if missing:
            missing_required.append({"instance_id": row.get("instance_id"), "missing": missing})

        status_ok = row.get("status") == "ok"
        parsed_ok = bool(row.get("parse_ok"))
        ok += int(status_ok)
        parse_ok += int(parsed_ok)

        parsed = normalized_parsed(row)
        for key in parsed:
            parsed_key_coverage[key] = parsed_key_coverage.get(key, 0) + 1

        gold_files = patch_files(str((tasks.get(row.get("instance_id")) or {}).get("patch") or ""))
        pred_files = predicted_files(row)
        cand_files = candidate_files(row)
        hit = bool(set(gold_files) & set(pred_files))
        c_hit = bool(set(gold_files) & set(cand_files))
        file_hit += int(hit)
        candidate_hit += int(c_hit)

        candidate_counts.append(len(cand_files))
        pred_counts.append(len(pred_files))
        tool_steps.append(len(parsed.get("tool_plan") or []))
        if row.get("latency_sec") is not None:
            latencies.append(float(row["latency_sec"]))

        if gold_files and not hit:
            misses.append(
                TraceMiss(
                    idx=idx,
                    instance_id=str(row.get("instance_id")),
                    repo=str(row.get("repo")),
                    status=str(row.get("status")),
                    parse_ok=parsed_ok,
                    pred_files=pred_files,
                    candidate_hit=c_hit,
                    gold_files=gold_files,
                    error=row.get("error"),
                )
            )

    total = len(rows)
    return TraceAuditReport(
        run_id=traces.parent.name,
        total=total,
        ok=ok,
        parse_ok=parse_ok,
        missing_required=missing_required,
        parsed_key_coverage=parsed_key_coverage,
        candidate_hit=candidate_hit,
        candidate_hit_rate=round(candidate_hit / total, 4) if total else 0.0,
        file_hit=file_hit,
        file_hit_rate_all=round(file_hit / total, 4) if total else 0.0,
        file_hit_rate_ok=round(file_hit / ok, 4) if ok else 0.0,
        candidate_count_min_max=_min_max(candidate_counts),
        pred_file_count_min_max=_min_max(pred_counts),
        tool_plan_steps_min_max=_min_max(tool_steps),
        latency_avg=round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        latency_p50=round(statistics.median(latencies), 3) if latencies else 0.0,
        misses=misses,
    )


def load_tasks_by_instance(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    task_path = Path(path)
    if not task_path.exists():
        return {}
    if task_path.suffix == ".parquet":
        import pandas as pd

        return {str(row.get("instance_id")): row for row in pd.read_parquet(task_path).to_dict(orient="records")}
    records: dict[str, dict[str, Any]] = {}
    for line in task_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            records[str(row.get("instance_id"))] = row
    return records


def patch_files(patch: str) -> list[str]:
    files: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
                files.append(path[2:] if path.startswith("b/") else path)
    return sorted(set(files))


def normalized_parsed(record: dict[str, Any]) -> dict[str, Any]:
    parsed = record.get("parsed") or record.get("planner_json") or record.get("json") or {}
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def predicted_files(record: dict[str, Any]) -> list[str]:
    parsed = normalized_parsed(record)
    files = parsed.get("files_to_inspect") or parsed.get("files") or parsed.get("target_files") or []
    return _extract_paths(files)


def candidate_files(record: dict[str, Any]) -> list[str]:
    return _extract_paths(record.get("candidate_files") or record.get("candidates") or [])


def _extract_paths(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    paths = []
    for item in items:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict) and item.get("path"):
            paths.append(str(item["path"]))
    return sorted(set(paths))


def _min_max(values: list[int]) -> list[int]:
    return [min(values), max(values)] if values else [0, 0]
