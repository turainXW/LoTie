from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_GLOBS = (
    "**/tasks.jsonl",
    "**/trajectories.jsonl",
    "**/episodes.jsonl",
)


@dataclass
class ProblemRecord:
    task_id: str
    instruction: str
    source_path: str
    kind: str
    validation_command: str = ""
    final_status: str = ""
    repo_path: str = ""
    files: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_problem_index(
    roots: list[str | Path],
    *,
    globs: tuple[str, ...] = DEFAULT_GLOBS,
    max_records_per_file: int = 1000,
) -> list[ProblemRecord]:
    records: list[ProblemRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for root_raw in roots:
        root = Path(root_raw)
        if not root.exists():
            continue
        candidate_files: list[Path] = []
        if root.is_file():
            candidate_files.append(root)
        else:
            for pattern in globs:
                candidate_files.extend(sorted(root.glob(pattern)))
        for path in candidate_files:
            for record in _records_from_jsonl(path, max_records=max_records_per_file):
                key = (record.task_id, record.instruction, record.source_path)
                if key in seen:
                    continue
                seen.add(key)
                records.append(record)
    return records


def save_problem_index(records: list[ProblemRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_problem_index(path: str | Path) -> list[ProblemRecord]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(ProblemRecord(**json.loads(line)))
    return records


def search_problems(records: list[ProblemRecord], query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
    top_k = max(1, min(int(top_k or 5), 50))
    ranked = []
    for record in records:
        score = _score(record, query)
        if score <= 0:
            continue
        result = record.to_dict()
        result["score"] = round(score, 4)
        ranked.append(result)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def problem_search(
    query: str,
    *,
    index_path: str | Path | None = None,
    roots: list[str | Path] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    if index_path and Path(index_path).exists():
        records = load_problem_index(index_path)
        source = str(Path(index_path).resolve())
    else:
        records = build_problem_index(roots or [Path("data")])
        source = "dynamic_roots"
    return {
        "ok": True,
        "query": query,
        "source": source,
        "num_records": len(records),
        "results": search_problems(records, query, top_k=top_k),
    }


def _records_from_jsonl(path: Path, *, max_records: int) -> list[ProblemRecord]:
    records: list[ProblemRecord] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines[:max_records]:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        record = _record_from_obj(obj, path)
        if record is not None:
            records.append(record)
    return records


def _record_from_obj(obj: dict[str, Any], path: Path) -> ProblemRecord | None:
    task_id = str(obj.get("task_id") or obj.get("id") or "")
    instruction = str(obj.get("instruction") or obj.get("problem_statement") or "")
    if not task_id or not instruction:
        return None
    files = _extract_files(obj)
    final_status = str(obj.get("final_status") or obj.get("status") or "")
    return ProblemRecord(
        task_id=task_id,
        instruction=instruction[:4000],
        source_path=str(path.resolve()),
        kind=_infer_kind(path, obj),
        validation_command=str(obj.get("validation_command") or ""),
        final_status=final_status,
        repo_path=str(obj.get("repo_path") or obj.get("repo") or ""),
        files=files[:50],
        tags=_infer_tags(obj, files),
    )


def _extract_files(obj: dict[str, Any]) -> list[str]:
    files: list[str] = []
    initial_context = obj.get("initial_context") or {}
    repo_context = initial_context.get("repo_context") or {}
    for file_obj in repo_context.get("files") or []:
        if isinstance(file_obj, dict) and file_obj.get("path"):
            files.append(str(file_obj["path"]))
    for step in obj.get("steps") or []:
        action = step.get("action") or {}
        content = str(action.get("content") or "")
        files.extend(re.findall(r"[A-Za-z0-9_./-]+\.(?:py|md|json|yaml|yml|toml|txt|sh)", content))
    return sorted(set(files))


def _infer_kind(path: Path, obj: dict[str, Any]) -> str:
    name = path.name
    if name == "tasks.jsonl":
        return "task"
    if name == "trajectories.jsonl":
        return "trajectory"
    if name == "episodes.jsonl":
        return "episode"
    if "messages" in obj:
        return "sft"
    return "unknown"


def _infer_tags(obj: dict[str, Any], files: list[str]) -> list[str]:
    tags = []
    text = " ".join([str(obj.get("instruction") or ""), " ".join(files)]).lower()
    for tag in ["repair", "test", "cli", "parser", "json", "memory", "context", "leetcode", "swegym"]:
        if tag in text:
            tags.append(tag)
    if obj.get("final_status") == "pass":
        tags.append("pass")
    return sorted(set(tags))


def _score(record: ProblemRecord, query: str) -> float:
    query_terms = set(_terms(query))
    if not query_terms:
        return 0.0
    haystack = " ".join(
        [
            record.task_id,
            record.instruction,
            record.validation_command,
            " ".join(record.files),
            " ".join(record.tags),
            record.final_status,
        ]
    )
    terms = _terms(haystack)
    term_set = set(terms)
    overlap = query_terms & term_set
    score = len(overlap) * 5.0
    for term in query_terms:
        score += min(terms.count(term), 3) * 0.5
    if record.final_status == "pass":
        score += 0.5
    if record.kind in {"trajectory", "episode"}:
        score += 0.25
    return score


def _terms(text: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text)]
