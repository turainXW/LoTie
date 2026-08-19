from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MEMORY_TYPES = {"fact", "lesson", "preference", "failure_pattern"}


@dataclass
class MemoryRecord:
    type: str
    content: str
    tags: list[str] = field(default_factory=list)
    source: str = ""
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data["created_at"]:
            data["created_at"] = _timestamp()
        return data


@dataclass
class SessionSummary:
    run_id: str
    task: str
    mode: str
    status: str
    submission: str
    tools_used: list[str]
    files_read: list[str]
    files_changed: list[str]
    blocked_actions: int
    trajectory_path: str
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data["created_at"]:
            data["created_at"] = _timestamp()
        return data


def ensure_memory_store(memory_dir: str | Path) -> Path:
    root = Path(memory_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in ["project_facts.jsonl", "lessons.jsonl", "preferences.jsonl", "failure_patterns.jsonl", "session_index.jsonl"]:
        path = root / name
        if not path.exists():
            path.write_text("", encoding="utf-8")
    return root


def add_memory(memory_dir: str | Path, record: MemoryRecord) -> Path:
    if record.type not in MEMORY_TYPES:
        raise ValueError(f"Unsupported memory type: {record.type}")
    root = ensure_memory_store(memory_dir)
    path = root / _file_for_type(record.type)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_memories(memory_dir: str | Path) -> list[MemoryRecord]:
    root = ensure_memory_store(memory_dir)
    records: list[MemoryRecord] = []
    for memory_type in sorted(MEMORY_TYPES):
        path = root / _file_for_type(memory_type)
        for item in _read_jsonl(path):
            records.append(MemoryRecord(**item))
    return records


def compact_memories(
    memory_dir: str | Path,
    task: str,
    *,
    max_records: int = 8,
    max_chars: int = 3000,
) -> str:
    records = load_memories(memory_dir)
    ranked = sorted(records, key=lambda record: _score_memory(record, task), reverse=True)[:max_records]
    lines = ["Relevant long-term memory:"]
    if not ranked:
        lines.append("- none yet")
    for record in ranked:
        tag_text = ", ".join(record.tags[:6]) if record.tags else "untagged"
        source = f"; source={record.source}" if record.source else ""
        lines.append(f"- [{record.type}] {record.content} (tags={tag_text}{source})")
    return "\n".join(lines)[:max_chars].rstrip()


def summarize_trajectory(trajectory_path: str | Path) -> SessionSummary:
    path = Path(trajectory_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    messages = data.get("messages", [])
    info = data.get("info", {})
    config = info.get("config", {})
    env_config = config.get("environment", {})
    agent_config = config.get("agent", {})

    user_task = ""
    for message in messages:
        if message.get("role") == "user" and "Task:" in str(message.get("content", "")):
            user_task = _extract_task(str(message.get("content", "")))
            break

    tools_used: set[str] = set()
    files_read: set[str] = set()
    files_changed: set[str] = set()
    blocked_actions = 0

    for message in messages:
        for action in message.get("extra", {}).get("actions", []):
            tool = action.get("tool", "bash")
            tools_used.add(tool)
            command = action.get("command", "")
            if command:
                _collect_file_hints(command, files_read, files_changed)
        extra = message.get("extra", {})
        if extra.get("blocked"):
            blocked_actions += 1
        if extra.get("tool"):
            tools_used.add(extra["tool"])

    run_id = path.stem
    output_path = agent_config.get("output_path")
    if output_path:
        run_id = Path(output_path).stem

    return SessionSummary(
        run_id=run_id,
        task=user_task,
        mode=env_config.get("mode", ""),
        status=info.get("exit_status", ""),
        submission=info.get("submission", ""),
        tools_used=sorted(tools_used),
        files_read=sorted(files_read),
        files_changed=sorted(files_changed),
        blocked_actions=blocked_actions,
        trajectory_path=str(path.resolve()),
    )


def write_session_summary(memory_dir: str | Path, summary: SessionSummary) -> tuple[Path, Path]:
    root = ensure_memory_store(memory_dir)
    session_dir = root / "sessions" / summary.run_id
    session_dir.mkdir(parents=True, exist_ok=True)
    summary_path = session_dir / "session_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    index_path = root / "session_index.jsonl"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary.to_dict(), ensure_ascii=False) + "\n")
    return summary_path, index_path


def learn_from_summary(memory_dir: str | Path, summary: SessionSummary) -> list[Path]:
    paths: list[Path] = []
    if summary.status:
        paths.append(
            add_memory(
                memory_dir,
                MemoryRecord(
                    type="fact",
                    content=f"Session {summary.run_id} ended with status {summary.status}.",
                    tags=["session", summary.mode, summary.status],
                    source=summary.run_id,
                ),
            )
        )
    if summary.blocked_actions:
        paths.append(
            add_memory(
                memory_dir,
                MemoryRecord(
                    type="lesson",
                    content=f"Session {summary.run_id} had {summary.blocked_actions} blocked action(s); keep mode permissions explicit.",
                    tags=["permission", "blocked", summary.mode],
                    source=summary.run_id,
                ),
            )
        )
    if "download_repo" in summary.tools_used:
        paths.append(
            add_memory(
                memory_dir,
                MemoryRecord(
                    type="lesson",
                    content="download_repo is a controlled tool for public GitHub repos and should run outside plan mode.",
                    tags=["download_repo", "github", "permission"],
                    source=summary.run_id,
                ),
            )
        )
    return paths


def _file_for_type(memory_type: str) -> str:
    return {
        "fact": "project_facts.jsonl",
        "lesson": "lessons.jsonl",
        "preference": "preferences.jsonl",
        "failure_pattern": "failure_patterns.jsonl",
    }[memory_type]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _score_memory(record: MemoryRecord, task: str) -> float:
    task_terms = set(_terms(task))
    memory_terms = set(_terms(" ".join([record.type, record.content, " ".join(record.tags)])))
    return len(task_terms & memory_terms) * 5.0 + len(record.tags)


def _terms(text: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text)]


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _extract_task(content: str) -> str:
    match = re.search(r"Task:\s*(.*?)\n\n", content, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return content[:500].strip()


def _collect_file_hints(command: str, files_read: set[str], files_changed: set[str]) -> None:
    tokens = re.findall(r"[\w./-]+\.[A-Za-z0-9_]+", command)
    read_markers = ("cat ", "sed -n", "nl ", "head ", "tail ", "rg ", "grep ")
    changed_markers = (">", ">>", "sed -i", "write_text", "touch ", "tee ")
    target = files_changed if any(marker in command for marker in changed_markers) else files_read
    if any(marker in command for marker in read_markers) or target is files_changed:
        for token in tokens:
            if not token.startswith("-") and "/" not in token[:1]:
                target.add(token)

