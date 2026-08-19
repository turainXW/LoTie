from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


OPENHANDS_SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer to solve tasks.\n"
    "<IMPORTANT>\n"
    "* If user provides a path, you should NOT assume it's relative to the current working directory. "
    "Instead, you should explore the file system to find the file before working on it.\n"
    "</IMPORTANT>\n\n"
    "You have access to bash, file editing, repository context, and a final answer tool."
)


@dataclass
class ExportManifest:
    format: str
    source: str
    output_dir: str
    sampled_path: str
    sft_path: str
    verifier_path: str
    count: int
    schema_notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def export_grounded_to_openhands(
    traces_path: str | Path,
    output_dir: str | Path,
    *,
    run_id: str | None = None,
    include_planner_failures_in_sft: bool = False,
) -> ExportManifest:
    traces = Path(traces_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in traces.read_text(encoding="utf-8").splitlines() if line.strip()]
    run_name = run_id or traces.parent.name

    sampled_path = out / "openhands_sampled.jsonl"
    sft_path = out / "openhands_sft_messages.jsonl"
    verifier_path = out / "openhands_verifier_records.jsonl"

    sampled_count = 0
    sft_count = 0
    verifier_count = 0
    with sampled_path.open("w", encoding="utf-8") as sampled, sft_path.open("w", encoding="utf-8") as sft, verifier_path.open(
        "w", encoding="utf-8"
    ) as verifier:
        for record in records:
            sampled_record = to_openhands_sampled_record(record, run_id=run_name)
            sampled.write(json.dumps(sampled_record, ensure_ascii=False) + "\n")
            sampled_count += 1

            sft_record = to_openhands_sft_record(record, run_id=run_name)
            if include_planner_failures_in_sft or sft_record["resolved"]:
                sft.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
                sft_count += 1

            verifier.write(json.dumps(to_openhands_verifier_record(record, run_id=run_name), ensure_ascii=False) + "\n")
            verifier_count += 1

    manifest = ExportManifest(
        format="codeagent_swegym_openhands_export_v1",
        source=str(traces),
        output_dir=str(out),
        sampled_path=str(sampled_path),
        sft_path=str(sft_path),
        verifier_path=str(verifier_path),
        count=len(records),
        schema_notes={
            "sampled": "Aligned to SWE-Gym/OpenHands-Sampled-Trajectories columns: instance_id, run_id, resolved, messages, tools, test_result.",
            "sft": "Aligned to SWE-Gym/OpenHands-SFT-Trajectories shape: messages.",
            "planner_only": True,
            "resolved_semantics": "False means no executable edit/test verifier was run; use file_hit/test_result.report fields for planner QA.",
            "sampled_count": sampled_count,
            "sft_count": sft_count,
            "verifier_count": verifier_count,
        },
    )
    (out / "manifest.json").write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def to_openhands_sampled_record(record: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    parsed = _parsed(record)
    messages = _messages(record)
    return {
        "format": "openhands_sampled_trajectory_v1",
        "instance_id": record.get("instance_id", ""),
        "run_id": run_id,
        "resolved": False,
        "messages": messages,
        "tools": _tools(),
        "test_result": {
            "apply_patch_output": None,
            "git_patch": "",
            "report": {
                "resolved": False,
                "planner_only": True,
                "status": record.get("status"),
                "parse_ok": bool(record.get("parse_ok")),
                "repo_status": record.get("repo_status"),
                "candidate_count": len(record.get("candidate_files") or []),
                "predicted_files": _predicted_files(parsed),
                "tests_to_run": parsed.get("tests_to_run", []),
                "root_cause_hypothesis": parsed.get("root_cause_hypothesis", ""),
                "minimal_fix_strategy": parsed.get("minimal_fix_strategy", ""),
                "confidence": parsed.get("confidence"),
                "latency_sec": record.get("latency_sec"),
                "error": record.get("error") or record.get("parse_error"),
            },
            "test_output": None,
        },
        "metadata": _metadata(record, parsed),
    }


def to_openhands_sft_record(record: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    return {
        "format": "openhands_sft_messages_v1",
        "instance_id": record.get("instance_id", ""),
        "run_id": run_id,
        "resolved": bool(record.get("status") == "ok" and record.get("parse_ok")),
        "messages": _messages(record),
    }


def to_openhands_verifier_record(record: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    sampled = to_openhands_sampled_record(record, run_id=run_id)
    return {
        "format": "openhands_verifier_messages_v1",
        "instance_id": sampled["instance_id"],
        "run_id": run_id,
        "resolved": sampled["resolved"],
        "messages": [
            {
                "role": "system",
                "content": "You are an expert judge evaluating whether an agent trajectory resolved a software engineering task.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "instance_id": sampled["instance_id"],
                        "trajectory": sampled["messages"],
                        "test_result": sampled["test_result"],
                    },
                    ensure_ascii=False,
                ),
            },
            {"role": "assistant", "content": "False"},
        ],
    }


def _messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    parsed = _parsed(record)
    user_payload = {
        "instance_id": record.get("instance_id"),
        "repo": record.get("repo"),
        "base_commit": record.get("base_commit"),
        "problem_statement": _extract_problem(record.get("prompt", "")),
        "candidate_files": record.get("candidate_files") or [],
        "request": "Inspect the repository context and propose the next files, tests, and repair plan.",
    }
    return [
        {"role": "system", "content": OPENHANDS_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "root_cause_hypothesis": parsed.get("root_cause_hypothesis", ""),
                    "files_to_inspect": parsed.get("files_to_inspect", []),
                    "tests_to_run": parsed.get("tests_to_run", []),
                    "tool_plan": parsed.get("tool_plan", []),
                    "minimal_fix_strategy": parsed.get("minimal_fix_strategy", ""),
                    "risks": parsed.get("risks", []),
                    "confidence": parsed.get("confidence", None),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "tool_calls": _tool_calls(parsed),
        },
    ]


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Execute a bash command in the terminal.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": "View or edit files in the workspace.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "path": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "repo_context",
                "description": "Retrieve ranked repository files and snippets for the current task.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        },
    ]


def _tool_calls(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for step in parsed.get("tool_plan") or []:
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool", "repo_context"))
        args = step.get("arguments") or step.get("args") or {}
        if not args and step.get("command_or_query"):
            args = {"query" if tool == "repo_context" else "command": step["command_or_query"]}
        calls.append({"type": "function", "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


def _metadata(record: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "repo": record.get("repo"),
        "repo_path": record.get("repo_path"),
        "repo_status": record.get("repo_status"),
        "base_commit": record.get("base_commit"),
        "candidate_files": record.get("candidate_files") or [],
        "usage": record.get("usage"),
        "raw_parse_error": record.get("parse_error"),
        "raw_answer": record.get("answer"),
        "planner_json": parsed,
    }


def _parsed(record: dict[str, Any]) -> dict[str, Any]:
    parsed = record.get("parsed") or {}
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _predicted_files(parsed: dict[str, Any]) -> list[str]:
    files = parsed.get("files_to_inspect") or []
    if not isinstance(files, list):
        return []
    return [str(item) for item in files if isinstance(item, str)]


def _extract_problem(prompt: str) -> str:
    marker = "Problem statement:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].split("\n\n", 1)[0].strip()
    return prompt[:4000]
