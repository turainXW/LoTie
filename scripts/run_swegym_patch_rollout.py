#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import math
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.harness_version import MODIFICATION_COMMANDS, harness_metadata  # noqa: E402
from code_agent_baseline.openhands_tools import OpenHandsToolParser, OpenHandsToolRuntime, ToolResult  # noqa: E402
from code_agent_baseline.model_transport import request_chat_completion  # noqa: E402
from code_agent_baseline.runtime_env import runtime_environment_info, task_runtime_env  # noqa: E402
from code_agent_baseline.tool_specs import (  # noqa: E402
    CORE_REPAIR_TOOLS,
    SWE_EXTENSION_TOOLS,
    render_tool_catalog,
    render_tool_few_shot,
    tool_schemas,
)
from code_agent_baseline.working_memory import ContextLimitExceeded, build_model_input, load_token_counter  # noqa: E402


OPENHANDS_SYSTEM_PROMPT = """You are a code repair agent. Your job is to inspect the repository, edit source files, and validate the change.

<RUNTIME>
- The shell and editor already run inside the repository workspace.
- The harness has already selected and activated the task Python environment for every command.
- Do not create, activate, or reinstall a virtual environment. Focus on inspecting and modifying the repository.
- Use repository-relative paths such as src/package/module.py.
- Do not assume /testbed, /repo, or /workspace exists. Run pwd if you need to confirm the current directory.
- Do not start commands with cd /testbed, cd /repo, or cd /workspace.
- Prefer commands like grep -n "needle" src/pkg/file.py or PYTHONPATH=src python -m pytest tests/test_file.py.
- If Python imports fail because the package is not installed, try PYTHONPATH=src before attempting a full install.
</RUNTIME>

<OUTPUT_PROTOCOL>
- Every assistant message MUST contain exactly one JSON object and nothing else.
- Do not write prose before or after the JSON.
- Do not use Markdown code fences.
- The JSON object MUST have this schema:
  {"tool_name": "FUNCTION_NAME", "arguments": {"PARAMETER_NAME": "VALUE"}}
- Use an empty object for tools without parameters, for example:
  {"tool_name": "finish", "arguments": {}}
- Do not use XML, <function=...>, <tool_call>, or <invoke> in assistant messages.
</OUTPUT_PROTOCOL>

<TERMINATION_POLICY>
- A successful task is not complete until you emit {"tool_name":"finish","arguments":{}}.
- After the latest source edit, first obtain a passing, non-skipped run_tests result and then inspect git_diff.
- Once those checks are complete and the patch is minimal, your next and only action MUST be finish.
- Do not send a prose final answer, repeat tests, or inspect more files after the completion gate is satisfied.
- If finish is rejected, perform only the missing check named by the tool result and call finish again.
</TERMINATION_POLICY>

<PATCH_POLICY>
- Make the minimal source/runtime code change that satisfies the issue.
- Do not modify tests, examples, docs, or benchmarks unless the issue explicitly asks for that file type.
- You may read tests/examples/docs and run targeted commands, but final edits should normally be in source files.
- If a test cannot run because the local package environment is incomplete, run py_compile or a small PYTHONPATH=src reproducer on edited source files.
</PATCH_POLICY>

<RECOVERY>
- If a tool says "command is required", retry with execute_bash and a non-empty command argument.
- If a tool says "path is required", retry str_replace_editor with a repository-relative path.
- If a tool says "old_str is not unique", view the relevant numbered lines, then use line_replace.
- If cd /testbed, cd /repo, or cd /workspace fails, retry without cd because the tool already runs in the repo workspace.
- If a test cannot run because optional project dependencies are missing, at least run py_compile on edited Python files.
</RECOVERY>
"""

@dataclass
class PatchStep:
    role: str
    content: str
    tool_call: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None


@dataclass
class PatchRecord:
    format: str
    instance_id: str
    run_id: str
    repo: str
    base_commit: str
    resolved: bool
    status: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    test_result: dict[str, Any]
    patch: str
    edit_apply: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeEnvironmentError(RuntimeError):
    def __init__(self, info: dict[str, Any]) -> None:
        self.info = info
        super().__init__(str(info.get("output") or info.get("reason") or "Task runtime environment is not ready"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real edit attempts for SWE-Gym tasks.")
    parser.add_argument("--tasks", default="/root/autodl-tmp/swegym_tasks/train-00000-of-00001.parquet")
    parser.add_argument("--grounded-traces", help="Optional planner traces used for candidate files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-cache", default="/root/autodl-tmp/swegym_repo_cache")
    parser.add_argument("--work-dir", default="/root/autodl-tmp/swegym_patch_workspaces")
    parser.add_argument("--auto-fetch-repos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-url", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument(
        "--model-network",
        choices=["direct", "system"],
        default="direct",
        help="Use a direct API connection by default; system honors OS/environment proxy settings.",
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY", help="Optional environment variable containing a bearer token.")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--protocol-retries",
        type=int,
        default=2,
        help="Maximum consecutive JSON protocol repair attempts before parse_error.",
    )
    parser.add_argument(
        "--protocol-retry-temperature",
        type=float,
        help="Optional lower temperature used only while repairing an invalid JSON tool call.",
    )
    parser.add_argument("--thinking-mode", choices=["enabled", "disabled", "auto"], default="enabled")
    parser.add_argument("--reasoning-effort", choices=["low", "high", "xhigh", "max"], default="max")
    parser.add_argument("--api-retries", type=int, default=6)
    parser.add_argument("--retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--retry-max-backoff-sec", type=float, default=30.0)
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--split", choices=["train", "validation", "evaluation"])
    parser.add_argument("--dataset", choices=["mbppplus", "humanevalplus", "swesmith_py"])
    parser.add_argument("--instance-id", action="append", help="Run only the selected instance ID; repeatable.")
    parser.add_argument("--max-file-chars", type=int, default=18000)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--repair-rounds", type=int, default=1, help="Extra edit rounds after compile failures.")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument(
        "--final-patch-scope",
        choices=["source", "all"],
        default="source",
        help="Which file changes to expose as the final patch. raw_patch is always kept in metadata.",
    )
    parser.add_argument(
        "--agent-mode",
        choices=["bench", "use"],
        default=None,
        help="bench uses the official SWE-Gym/OpenHands core tools; use adds richer local context tools.",
    )
    parser.add_argument("--tool-profile", choices=["official-core", "extended"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument(
        "--context-max-tokens",
        type=int,
        default=64_000,
        help="Maximum estimated/tokenized input budget before applying the configured overflow policy.",
    )
    parser.add_argument(
        "--context-tokenizer",
        help="Optional Hugging Face tokenizer.json used for exact context token counting.",
    )
    parser.add_argument(
        "--context-overflow-policy",
        choices=["error", "compact"],
        default="error",
        help="Bench default rejects over-limit traces; compact is available for interactive use.",
    )
    parser.add_argument(
        "--context-max-chars",
        type=int,
        default=None,
        help="Optional legacy character cap applied in addition to --context-max-tokens.",
    )
    parser.add_argument("--context-recent-turns", type=int, default=6)
    parser.add_argument(
        "--prompt-workspace-label",
        help="Stable logical workspace label for repeated-sampling experiments; tools still use the isolated real path.",
    )
    parser.add_argument(
        "--edit-checkpoint-step",
        type=int,
        default=10,
        help="Start tracking consecutive non-modification actions after this many steps.",
    )
    parser.add_argument(
        "--readonly-grace-steps",
        "--readonly-reminder-interval",
        dest="readonly_grace_steps",
        type=int,
        default=3,
        help="Append a convergence reminder after each run of this many non-modification actions.",
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Print each OpenHands tool call and result summary while the agent runs.",
    )
    parser.add_argument(
        "--step-output-chars",
        type=int,
        default=1200,
        help="Max characters of each tool result to show when --show-steps is enabled.",
    )
    return parser.parse_args()


def normalize_agent_mode(args: argparse.Namespace) -> None:
    if args.agent_mode is None:
        args.agent_mode = "use" if args.tool_profile == "extended" else "bench"
    args.tool_profile = "extended" if args.agent_mode == "use" else "official-core"


def openhands_system_prompt(tool_profile: str) -> str:
    names = list(CORE_REPAIR_TOOLS)
    if tool_profile == "extended":
        names.extend(SWE_EXTENSION_TOOLS)
    return "\n\n".join(
        [
            OPENHANDS_SYSTEM_PROMPT.strip(),
            render_tool_catalog(names),
            render_tool_few_shot(),
        ]
    )


def openhands_allowed_tools(tool_profile: str) -> set[str]:
    tools = set(CORE_REPAIR_TOOLS)
    if tool_profile == "extended":
        tools.update(SWE_EXTENSION_TOOLS)
    return tools


def main() -> None:
    args = parse_args()
    normalize_agent_mode(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_traces"
    raw_dir.mkdir(exist_ok=True)
    run_id = output_dir.name

    task_rows = load_task_rows(args.tasks)
    if args.split:
        task_rows = [row for row in task_rows if row.get("split") == args.split]
    if args.dataset:
        task_rows = [row for row in task_rows if row.get("dataset") == args.dataset]
    if args.instance_id:
        selected_ids = set(args.instance_id)
        task_rows = [row for row in task_rows if row.get("instance_id") in selected_ids]
    tasks = {row["instance_id"]: row for row in task_rows}
    if not tasks:
        raise ValueError("No tasks matched the requested split/dataset/instance filters")
    if args.grounded_traces:
        planners = [
            json.loads(line)
            for line in Path(args.grounded_traces).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        planners = [
            {"instance_id": task["instance_id"], "status": "ok", "parse_ok": True, "parsed": {}}
            for task in tasks.values()
        ]
    selected = [
        row
        for row in planners
        if row.get("status") == "ok" and row.get("parse_ok") and row.get("instance_id") in tasks
    ][: args.num]

    output_jsonl = output_dir / "openhands_patch_rollout.jsonl"
    sft_jsonl = output_dir / "messages_sft.jsonl"
    output_jsonl.write_text("", encoding="utf-8")
    sft_jsonl.write_text("", encoding="utf-8")

    records: list[PatchRecord] = []
    for idx, planner in enumerate(selected, start=1):
        task = tasks[planner["instance_id"]]
        if args.show_steps:
            print(
                f"\n=== TASK {idx}/{len(selected)} {task['instance_id']} repo={task['repo']} ===",
                flush=True,
            )
        try:
            record = run_one(task, planner, args=args, run_id=run_id, index=idx)
        except Exception as exc:
            record = error_record(task, planner, args=args, run_id=run_id, index=idx, exc=exc)
        records.append(record)
        (raw_dir / f"{idx:03d}_{safe_instance_name(record.instance_id)}.json").write_text(
            json.dumps(asdict(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        append_jsonl(output_jsonl, asdict(record))
        append_jsonl(
            sft_jsonl,
            {
                "format": "openhands_sft_messages_v1",
                "instance_id": record.instance_id,
                "run_id": run_id,
                "resolved": record.resolved,
                "status": record.status,
                "data_role": "evaluation_only" if task.get("split") == "evaluation" else task.get("split"),
                "exclude_from_training": bool(task.get("split") == "evaluation"),
                "messages": record.messages,
            },
        )
        print(
            json.dumps(
                {
                    "idx": idx,
                    "instance_id": record.instance_id,
                    "status": record.status,
                    "resolved": record.resolved,
                    "patch_lines": len(record.patch.splitlines()),
                    "edited_files": record.edit_apply.get("edited_files", []),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "run_id": run_id,
        "total": len(records),
        "edited": sum(1 for item in records if item.edit_apply.get("edited_files")),
        "resolved": sum(1 for item in records if item.resolved),
        "output_jsonl": str(output_jsonl),
        "sft_jsonl": str(sft_jsonl),
        "raw_traces": str(raw_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


def load_task_rows(path_like: str) -> list[dict[str, Any]]:
    path = Path(path_like)
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_one(task: dict[str, Any], planner: dict[str, Any], *, args: argparse.Namespace, run_id: str, index: int) -> PatchRecord:
    workspace = prepare_workspace(
        task,
        args.repo_cache,
        args.work_dir,
        run_id,
        index,
        auto_fetch_repos=args.auto_fetch_repos,
    )
    parsed = planner.get("parsed") or {}
    files = selected_files(parsed, planner)
    file_context = read_file_context(workspace, files, args.max_file_chars)
    tools = tool_schema(args.tool_profile)
    runtime_env, runtime_info = prepare_task_environment(task, workspace)
    if not runtime_info.get("ready"):
        raise RuntimeEnvironmentError(runtime_info)
    return run_one_openhands(
        task,
        planner,
        parsed=parsed,
        workspace=workspace,
        file_context=file_context,
        tools=tools,
        runtime_env=runtime_env,
        runtime_info=runtime_info,
        args=args,
        run_id=run_id,
    )


def context_input_budget(context_window_tokens: int | None, max_output_tokens: int) -> int | None:
    """Reserve generation space inside the model's total context window."""

    if context_window_tokens is None:
        return None
    return max(1, context_window_tokens - max(0, max_output_tokens))


def is_model_context_overflow(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "maximum context length" in message
        and "input tokens" in message
        and "output tokens" in message
    )


def run_one_openhands(
    task: dict[str, Any],
    planner: dict[str, Any],
    *,
    parsed: dict[str, Any],
    workspace: Path,
    file_context: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    runtime_env: dict[str, str],
    runtime_info: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
) -> PatchRecord:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": openhands_system_prompt(args.tool_profile)},
        {
            "role": "user",
            "content": openhands_user_prompt(
                task=task,
                workspace=workspace,
                parsed=parsed,
                file_context=file_context,
                tool_profile=args.tool_profile,
                runtime_info=runtime_info,
                workspace_label=args.prompt_workspace_label,
            ),
        },
    ]
    runtime = OpenHandsToolRuntime(
        workspace,
        timeout_sec=args.timeout_sec,
        max_view_chars=args.max_file_chars,
        max_command_output_chars=args.max_file_chars,
        allowed_tools=openhands_allowed_tools(args.tool_profile),
        runtime_env=runtime_env,
        test_command=str(task.get("test_command") or "") or None,
    )
    step_reports: list[dict[str, Any]] = []
    status = "max_steps"
    started = time.monotonic()
    consecutive_parse_failures = 0
    context_snapshots: list[dict[str, Any]] = []
    api_retry_events: list[dict[str, Any]] = []
    model_response_events: list[dict[str, Any]] = []
    args._api_retry_events = api_retry_events
    args._model_response_events = model_response_events
    if args.context_tokenizer:
        context_token_counter, context_counter_name = load_token_counter(args.context_tokenizer)
    else:
        context_token_counter, context_counter_name = None, "estimated_chars_per_3.5"

    for step in range(1, args.max_steps + 1):
        try:
            model_input, context_snapshot = build_model_input(
                messages,
                max_tokens=context_input_budget(args.context_max_tokens, args.max_tokens),
                max_chars=args.context_max_chars,
                recent_turns=args.context_recent_turns,
                token_counter=context_token_counter,
                token_counter_name=context_counter_name,
                overflow_policy=args.context_overflow_policy,
            )
        except ContextLimitExceeded as exc:
            context_snapshot = exc.snapshot
            context_snapshots.append({"step": step, **context_snapshot.to_dict(), "overflow": True})
            result_payload = {
                "tool_name": "context_window",
                "returncode": 1,
                "output": str(exc),
                "failure_type": "context_overflow",
            }
            messages.append(tool_observation_message("context_window", result_payload))
            step_reports.append({"step": step, "tool_call": None, "observation": result_payload})
            print_step_event(step, None, result_payload, args=args)
            status = "context_overflow"
            break
        context_snapshots.append({"step": step, **context_snapshot.to_dict()})
        try:
            repair_temperature = (
                args.protocol_retry_temperature if consecutive_parse_failures > 0 else None
            )
            answer = call_model(
                args.model_url,
                args.model,
                model_input,
                timeout_sec=args.timeout_sec,
                args=args,
                temperature=repair_temperature,
            )
        except Exception as exc:
            context_overflow = is_model_context_overflow(exc)
            result_payload = {
                "tool_name": "context_window" if context_overflow else "model_api",
                "returncode": 1,
                "output": repr(exc),
                "failure_type": "context_overflow" if context_overflow else "model_api_failure",
            }
            messages.append(tool_observation_message(result_payload["tool_name"], result_payload))
            step_reports.append({"step": step, "tool_call": None, "observation": result_payload})
            print_step_event(step, None, result_payload, args=args)
            status = result_payload["failure_type"]
            break
        messages.append({"role": "assistant", "content": answer})
        call = OpenHandsToolParser.parse_first(answer)
        if call is None:
            consecutive_parse_failures += 1
            result_payload = {
                "tool_name": "parser",
                "returncode": 1,
                "output": tool_call_retry_message(),
                "raw": answer,
                "protocol_retry": consecutive_parse_failures <= args.protocol_retries,
                "retry_temperature": args.protocol_retry_temperature,
            }
            messages.append(tool_observation_message("parser", result_payload))
            step_reports.append({"step": step, "tool_call": None, "observation": result_payload})
            print_step_event(step, None, result_payload, args=args)
            if consecutive_parse_failures <= args.protocol_retries:
                continue
            status = "parse_error"
            break
        consecutive_parse_failures = 0
        messages[-1]["content"] = json.dumps(
            {"tool_name": call.tool_name, "arguments": call.arguments},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        readonly_policy = readonly_budget_policy(
            step,
            call,
            step_reports,
            checkpoint_step=args.edit_checkpoint_step,
            grace_steps=args.readonly_grace_steps,
        )
        policy_result = completion_gate_result(call, workspace, step_reports)
        result = policy_result or runtime.execute(call)
        result_payload = result.to_dict()
        observation_message = tool_observation_message(call.tool_name, result_payload)
        if policy_result is None and readonly_policy == "append":
            result_payload["extra"] = {
                **(result_payload.get("extra") or {}),
                "action_policy": "edit_recommended",
                "readonly_budget_warning_shown": True,
                "blocked_readonly_action": False,
            }
            observation_message["content"] += (
                "\n<READONLY_BUDGET_CHECKPOINT>This action was executed and its actual result is shown above. "
                "The last three actions did not attempt a repository modification. Use the accumulated evidence "
                "to converge: make the smallest supported source edit, or, if the task is already fixed, validate "
                "the change and finish. Avoid rereading ranges already present in the conversation."
                "</READONLY_BUDGET_CHECKPOINT>"
            )
        prospective_step_reports = [
            *step_reports,
            {"step": step, "tool_call": call.to_dict(), "observation": result_payload},
        ]
        if call.tool_name == "git_diff" and result.returncode == 0 and completion_ready(workspace, prospective_step_reports):
            observation_message["content"] += (
                '\n<TERMINATION_CHECKPOINT>The patch exists, run_tests passed after the latest edit, and git_diff '
                'was reviewed. The completion gate is satisfied. Your next and only response must be '
                '{"tool_name":"finish","arguments":{}}.</TERMINATION_CHECKPOINT>'
            )
        messages.append(observation_message)
        step_reports.append({"step": step, "tool_call": call.to_dict(), "observation": result_payload})
        print_step_event(step, call, result_payload, args=args)
        if call.tool_name == "finish" and result.returncode == 0:
            status = "finished"
            break

    raw_patch = git_diff(workspace, scope="all")
    raw_edited_files = patch_files(raw_patch)
    patch = git_diff(workspace, scope=args.final_patch_scope)
    edited_files = patch_files(patch)
    verify = verify_workspace(workspace, edited_files, [], args)
    print_final_verify(verify, args=args)
    tests_run = [
        step["observation"]
        for step in step_reports
        if (step.get("tool_call") or {}).get("tool_name") == "run_tests"
    ]
    tests_passed = bool(tests_run) and all(
        item.get("returncode") == 0 and not (item.get("extra") or {}).get("all_skipped") for item in tests_run
    )
    terminal_failure = status in {
        "context_overflow",
        "model_api_failure",
        "parse_error",
        "infrastructure_blocked",
        "runner_error",
    }
    resolved = bool(patch.strip() and tests_passed and not terminal_failure)
    if not patch.strip() and status == "finished":
        status = "no_edit"
    elif verify.get("compile_failed"):
        status = "compile_failed"
    elif resolved:
        status = "resolved"
    elif patch.strip() and status == "finished":
        status = "edited"

    return PatchRecord(
        format="openhands_patch_rollout_v1",
        instance_id=task["instance_id"],
        run_id=run_id,
        repo=task["repo"],
        base_commit=task["base_commit"],
        resolved=resolved,
        status=status,
        messages=messages,
        tools=tools,
        test_result={
            "report": {
                "resolved": resolved,
                "status": status,
                "planner_only": False,
                "latency_sec": round(time.monotonic() - started, 3),
                "verify": verify,
                "tool_step_count": len(step_reports),
                "agent_mode": args.agent_mode,
            },
            "test_output": verify.get("test_output"),
        },
        patch=patch,
        edit_apply={"edited_files": edited_files, "applied": [], "errors": []},
        metadata={
            "workspace": str(workspace),
            "planner_files": selected_files(parsed, planner),
            "protocol": "openhands",
            "agent_mode": args.agent_mode,
            "tool_profile": args.tool_profile,
            "final_patch_scope": args.final_patch_scope,
            "raw_patch": raw_patch,
            "raw_edited_files": raw_edited_files,
            "filtered_out_files": sorted(set(raw_edited_files) - set(edited_files)),
            "tool_steps": step_reports,
            "gold_files": patch_files(task.get("patch", "")),
            "gold_file_hit_after_edit": bool(set(edited_files) & set(patch_files(task.get("patch", "")))),
            "raw_gold_file_hit_after_edit": bool(set(raw_edited_files) & set(patch_files(task.get("patch", "")))),
            "model": args.model,
            "model_url": args.model_url,
            "runtime_environment": runtime_info,
            "working_memory": {
                "max_tokens": context_input_budget(args.context_max_tokens, args.max_tokens),
                "model_context_window_tokens": args.context_max_tokens,
                "reserved_output_tokens": args.max_tokens,
                "max_chars": args.context_max_chars,
                "token_counter": context_counter_name,
                "overflow_policy": args.context_overflow_policy,
                "recent_turns": args.context_recent_turns,
                "snapshots": context_snapshots,
            },
            "api_retry_events": api_retry_events,
            "model_response_events": model_response_events,
            "model_sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "thinking_mode": args.thinking_mode,
                "reasoning_effort": args.reasoning_effort,
                "network": args.model_network,
            },
            "harness": harness_metadata(
                checkpoint_step=args.edit_checkpoint_step,
                reminder_interval=args.readonly_grace_steps,
            ),
            "task_split": task.get("split"),
            "data_role": "evaluation_only" if task.get("split") == "evaluation" else task.get("split"),
            "exclude_from_training": bool(task.get("split") == "evaluation"),
        },
    )


def openhands_user_prompt(
    *,
    task: dict[str, Any],
    workspace: Path,
    parsed: dict[str, Any],
    file_context: list[dict[str, Any]],
    tool_profile: str,
    runtime_info: dict[str, Any] | None = None,
    workspace_label: str | None = None,
) -> str:
    problem_statement = str(task.get("problem_statement", ""))
    hints_text = str(task.get("hints_text") or "").strip()
    hint_block = f"\n\n<hints>\n{hints_text}\n</hints>" if hints_text else ""
    public_runtime = public_runtime_info(runtime_info)
    prompt_workspace = workspace_label or str(workspace)
    prompt_directory = workspace_label or workspace.name
    if tool_profile == "official-core":
        environment_block = json.dumps(public_runtime, ensure_ascii=False)
        return (
            "<uploaded_files>\n"
            f"{prompt_workspace}\n"
            "</uploaded_files>\n"
            f"<runtime_environment>{environment_block}</runtime_environment>\n"
            f"I've uploaded a python code repository in the directory {prompt_directory}. "
            "Consider the following PR description:\n\n"
            "<pr_description>\n"
            f"{problem_statement}"
            f"{hint_block}\n"
            "</pr_description>\n\n"
            "Can you help me implement the necessary changes to the repository so that the requirements specified in the "
            "<pr_description> are met?\n"
            "I've already taken care of all changes to any of the test files described in the <pr_description>. "
            "This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Your task is to make the minimal changes to non-tests files in the repository to ensure the "
            "<pr_description> is satisfied.\n"
            "Follow these steps to resolve the issue:\n"
            "1. As a first step, explore the repo to familiarize yourself with its structure.\n"
            "2. Create or run a reproducer when useful to confirm the error.\n"
            "3. Edit the source code of the repo to resolve the issue.\n"
            "4. Rerun your reproducer or a targeted test and confirm that the error is fixed.\n"
            "5. Think about edge cases and make sure your fix handles them as well.\n"
            'After a passing run_tests result and git_diff review, end with exactly '
            '{"tool_name":"finish","arguments":{}} and no prose.\n'
            "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n"
        )

    user_payload = {
        "instance_id": task["instance_id"],
        "repo": task["repo"],
        "base_commit": task["base_commit"],
        "workspace": str(workspace),
        "runtime_environment": public_runtime,
        "runtime_instructions": [
            "The tool runtime already executes inside the repository workspace.",
            "The harness has already activated the task Python environment. Do not create or activate a venv.",
            "Do not use /testbed unless it exists; prefer pwd, relative paths, and the provided workspace path.",
            "For str_replace_editor path values, prefer repository-relative paths such as src/package/module.py.",
            "For str_replace_editor, always include command and path. Use command=line_replace with start_line/end_line when a string replacement is ambiguous.",
            "Never output bare Markdown code fences; wrap shell commands in execute_bash.",
            "Edit source/runtime files to fix behavior. Do not modify tests, examples, docs, or benchmarks unless the task explicitly asks for that file type.",
            "It is OK to create temporary repro scripts with execute_bash, but the final patch should be the minimal source change.",
        ],
        "problem_statement": problem_statement,
        "hints_text": hints_text,
        "planner": parsed,
        "candidate_file_context": [
            {"path": item.get("path"), "truncated": item.get("truncated", False), "numbered_content": item.get("numbered_content")}
            for item in file_context
        ],
        "instruction": (
            "Use the available tools to edit the repository. After the latest edit, obtain a passing run_tests result, "
            "review git_diff, and then emit exactly {\"tool_name\":\"finish\",\"arguments\":{}} with no prose."
        ),
    }
    return json.dumps(json_safe(user_payload), ensure_ascii=False, indent=2, allow_nan=False)


def public_runtime_info(runtime_info: dict[str, Any] | None) -> dict[str, Any]:
    info = runtime_info or {}
    keys = ("ready", "python_version", "python_executable", "venv_path", "environment_source")
    return {key: info.get(key) for key in keys if info.get(key) is not None}


def run_compile_repair_round(
    workspace: Path,
    messages: list[dict[str, Any]],
    edited_files: list[str],
    verify: dict[str, Any],
    *,
    args: argparse.Namespace,
    repair_round: int,
) -> dict[str, Any]:
    compile_errors = [
        {
            "command": item.get("command"),
            "returncode": item.get("returncode"),
            "output": item.get("output", "")[-4000:],
        }
        for item in verify.get("compile_results", [])
        if item.get("returncode") != 0
    ]
    repair_prompt = {
        "repair_round": repair_round,
        "instruction": "The previous edit produced compile errors. Return strict JSON edits that only fix the syntax/indentation/runtime import issue reported here. Keep the intended behavioral change if possible.",
        "compile_errors": compile_errors,
        "current_file_context": read_file_context(workspace, edited_files, args.max_file_chars),
    }
    messages.append({"role": "user", "content": json.dumps(repair_prompt, ensure_ascii=False, indent=2)})
    answer = call_model(args.model_url, args.model, messages, timeout_sec=args.timeout_sec, args=args)
    messages.append({"role": "assistant", "content": answer})
    repair = parse_json_answer(answer)
    apply_result = apply_edits(workspace, repair.get("edits") or [])
    messages.append(tool_observation_message("str_replace_editor", apply_result))
    return {"repair": repair, "apply_result": apply_result}


def tool_observation_message(tool_name: str, payload: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "user",
        "content": f"EXECUTION RESULT of [{tool_name}]:\n{format_tool_observation_payload(payload)}",
    }


def format_tool_observation_payload(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if isinstance(output, str):
        text = output
        returncode = payload.get("returncode")
        if returncode is not None and "[Command finished with exit code" not in text:
            text = text.rstrip() + f"\n[Command finished with exit code {returncode}]"
        reminder = protocol_reminder_for_tool_error(text)
        if reminder:
            text = text.rstrip() + "\n" + reminder
        return text
    return json.dumps(payload, ensure_ascii=False, indent=2)


def protocol_reminder_for_tool_error(text: str) -> str:
    if "No such file or directory" in text and any(path in text for path in ("/testbed", "/workspace", "/repo")):
        return "Runtime reminder: commands already run in the repo workspace. Retry without cd /testbed, cd /workspace, or cd /repo; use relative paths."
    if "command is required" in text:
        return 'Protocol reminder: retry with {"tool_name": "execute_bash", "arguments": {"command": "your command"}}.'
    if "path is required" in text:
        return 'Protocol reminder: retry str_replace_editor with {"tool_name": "str_replace_editor", "arguments": {"command": "view", "path": "repo/relative/path.py"}}.'
    if "Unsupported editor command" in text:
        return "Protocol reminder: str_replace_editor command must be one of view, create, str_replace, line_replace, insert, undo_edit."
    if "old_str is not unique" in text:
        return "Protocol reminder: view numbered lines, then use command=line_replace with start_line, end_line, and new_str."
    return ""


def print_step_event(step: int, call: Any | None, result: dict[str, Any], *, args: argparse.Namespace) -> None:
    if not args.show_steps:
        return
    if call is None:
        event = str(result.get("failure_type") or "parse_error").upper()
        print(f"[step {step:02d}] {event}", flush=True)
        print(_clip_text(str(result.get("output", "")), args.step_output_chars), flush=True)
        return
    print(f"[step {step:02d}] CALL {call.tool_name} {summarize_tool_call(call)}", flush=True)
    print(
        f"[step {step:02d}] RESULT returncode={result.get('returncode')} elapsed={result.get('elapsed_sec', 0)}s",
        flush=True,
    )
    output = str(result.get("output", "") or "")
    if output:
        print(_clip_text(output, args.step_output_chars), flush=True)


def print_final_verify(verify: dict[str, Any], *, args: argparse.Namespace) -> None:
    if not args.show_steps:
        return
    compile_results = verify.get("compile_results") or []
    test_results = verify.get("test_results") or []
    if not compile_results and not test_results:
        print("[final verify] no local compile/test commands were run", flush=True)
        return
    for idx, item in enumerate(compile_results, start=1):
        print(
            f"[final verify compile {idx}] command={item.get('command')} returncode={item.get('returncode')}",
            flush=True,
        )
        output = str(item.get("output", "") or "")
        if output:
            print(_clip_text(output, args.step_output_chars), flush=True)
    for idx, item in enumerate(test_results, start=1):
        print(
            f"[final verify test {idx}] command={item.get('command')} returncode={item.get('returncode')}",
            flush=True,
        )
        output = str(item.get("output", "") or "")
        if output:
            print(_clip_text(output, args.step_output_chars), flush=True)


def summarize_tool_call(call: Any) -> str:
    arguments = getattr(call, "arguments", {}) or {}
    if call.tool_name == "execute_bash":
        return f"command={json.dumps(str(arguments.get('command', '')), ensure_ascii=False)}"
    if call.tool_name == "str_replace_editor":
        command = str(arguments.get("command", ""))
        path = str(arguments.get("path", ""))
        parts = [f"command={command}", f"path={path}"]
        if arguments.get("view_range"):
            parts.append(f"view_range={arguments.get('view_range')}")
        if command in {"str_replace", "insert", "create"}:
            old_str = str(arguments.get("old_str", ""))
            new_str = str(arguments.get("new_str", arguments.get("file_text", "")))
            if old_str:
                parts.append(f"old_len={len(old_str)}")
            if new_str:
                parts.append(f"new_len={len(new_str)}")
        return " ".join(parts)
    if call.tool_name == "repo_context":
        return f"query={json.dumps(str(arguments.get('query', '')), ensure_ascii=False)}"
    if call.tool_name == "problem_search":
        query = json.dumps(str(arguments.get("query", "")), ensure_ascii=False)
        top_k = str(arguments.get("top_k", arguments.get("max_results", "")))
        return f"query={query} top_k={top_k}".strip()
    if call.tool_name == "web_search":
        query = json.dumps(str(arguments.get("query", "")), ensure_ascii=False)
        max_results = str(arguments.get("max_results", ""))
        return f"query={query} max_results={max_results}".strip()
    if call.tool_name == "download_repo":
        repo_url = str(arguments.get("repo_url", arguments.get("url", "")))
        return f"repo_url={json.dumps(repo_url, ensure_ascii=False)}"
    if call.tool_name == "finish":
        return ""
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True)


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... clipped {len(text) - max_chars} chars ..."


def build_openhands_training_messages(
    *,
    task: dict[str, Any],
    planner: dict[str, Any],
    file_context: list[dict[str, Any]],
    repair_steps: list[dict[str, Any]],
    final_verify: dict[str, Any],
    tool_profile: str = "official-core",
) -> list[dict[str, str]]:
    trace: list[dict[str, str]] = [
        {"role": "system", "content": openhands_system_prompt(tool_profile)},
        {
            "role": "user",
            "content": openhands_user_prompt(
                task=task,
                workspace=Path("/workspace") / str(task.get("repo", "repo")).replace("/", "__"),
                parsed=planner,
                file_context=file_context,
                tool_profile=tool_profile,
            ),
        },
    ]

    if tool_profile == "extended":
        query = " ".join(str(item) for item in planner.get("files_to_inspect") or []) or str(task.get("problem_statement", ""))[:300]
        trace.append({"role": "assistant", "content": format_tool_call("repo_context", {"query": query})})
        trace.append(
            tool_observation_message(
                "repo_context",
                {
                    "files": [
                        {"path": item.get("path"), "truncated": item.get("truncated", False)}
                        for item in file_context
                    ],
                    "planner": planner,
                },
            )
        )

    for ctx in file_context:
        path = str(ctx.get("path", ""))
        if not path:
            continue
        trace.append(
            {"role": "assistant", "content": format_tool_call("str_replace_editor", {"command": "view", "path": path})}
        )
        trace.append(
            tool_observation_message(
                "str_replace_editor",
                {
                    "command": "view",
                    "path": path,
                    "output": ctx.get("numbered_content") or ctx.get("content", ""),
                    "truncated": ctx.get("truncated", False),
                },
            )
        )

    for step in repair_steps:
        repair = step.get("repair") or {}
        edits = repair.get("edits") or []
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            trace.append(
                {
                    "role": "assistant",
                    "content": format_tool_call(
                        "str_replace_editor",
                        {
                            "command": "str_replace",
                            "path": edit.get("path", ""),
                            "old_str": edit.get("old_str", ""),
                            "new_str": edit.get("new_str", ""),
                        },
                    ),
                }
            )
        trace.append(tool_observation_message("str_replace_editor", step.get("apply_result") or {}))

        tests = repair.get("tests_to_run") or []
        if tests:
            command = str(tests[0])
        else:
            edited_files = (step.get("apply_result") or {}).get("edited_files") or []
            command = " && ".join(f"python -m py_compile {path}" for path in edited_files) or "python -m pytest"
        trace.append({"role": "assistant", "content": format_tool_call("execute_bash", {"command": command})})
        trace.append(tool_observation_message("execute_bash", step.get("verify") or final_verify))

    trace.append({"role": "assistant", "content": format_tool_call("finish", {})})
    return trace


def format_tool_call(function_name: str, parameters: dict[str, Any]) -> str:
    return json.dumps(
        {"tool_name": function_name, "arguments": {key: value for key, value in parameters.items() if value is not None and value != ""}},
        ensure_ascii=False,
    )


def clone_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False))


def merge_apply_result(base: dict[str, Any], extra: dict[str, Any]) -> None:
    base["edited_files"] = sorted(set(base.get("edited_files", [])) | set(extra.get("edited_files", [])))
    base.setdefault("applied", []).extend(extra.get("applied", []))
    base.setdefault("errors", []).extend(extra.get("errors", []))


def error_record(
    task: dict[str, Any],
    planner: dict[str, Any],
    *,
    args: argparse.Namespace,
    run_id: str,
    index: int,
    exc: Exception,
) -> PatchRecord:
    environment_info = exc.info if isinstance(exc, RuntimeEnvironmentError) else None
    status = "infrastructure_blocked" if environment_info else "runner_error"
    return PatchRecord(
        format="openhands_patch_rollout_v1",
        instance_id=task["instance_id"],
        run_id=run_id,
        repo=task["repo"],
        base_commit=task["base_commit"],
        resolved=False,
        status=status,
        messages=[
            {"role": "system", "content": openhands_system_prompt(args.tool_profile)},
            {"role": "user", "content": task.get("problem_statement", "")},
            tool_observation_message("runner", {"error": repr(exc)}),
        ],
        tools=tool_schema(args.tool_profile),
        test_result={"report": {"resolved": False, "status": status, "error": repr(exc)}, "test_output": None},
        patch="",
        edit_apply={"edited_files": [], "errors": [{"error": repr(exc)}]},
        metadata={
            "planner": planner,
            "index": index,
            "model": args.model,
            "model_url": args.model_url,
            "harness": harness_metadata(
                checkpoint_step=args.edit_checkpoint_step,
                reminder_interval=args.readonly_grace_steps,
            ),
            "runtime_environment": environment_info,
            "api_retry_events": list(getattr(args, "_api_retry_events", []) or []),
        },
    )


def prepare_workspace(
    task: dict[str, Any],
    repo_cache: str,
    work_dir: str,
    run_id: str,
    index: int,
    *,
    auto_fetch_repos: bool = True,
) -> Path:
    repo_cache_path = Path(repo_cache)
    local_repo = Path(str(task.get("local_repo_path") or "")).expanduser()
    if task.get("local_repo_path") and local_repo.is_dir():
        cache = local_repo.resolve()
    else:
        try:
            cache = find_cached_repo(task, repo_cache_path)
        except FileNotFoundError:
            if not auto_fetch_repos:
                raise
            cache = fetch_repo_to_cache(task, repo_cache_path)
    dest = Path(work_dir) / run_id / f"{index:03d}_{safe_instance_name(task['instance_id'])}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(cache, dest, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".mypy_cache"))
    subprocess.run(["git", "init"], cwd=dest, text=True, capture_output=True, timeout=30)
    bug_patch = str(task.get("bug_patch") or task.get("setup_patch") or "")
    if bug_patch.strip():
        apply_bug = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=dest,
            input=bug_patch,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if apply_bug.returncode != 0:
            raise RuntimeError(
                f"Failed to initialize buggy workspace for {task['instance_id']}: "
                f"{apply_bug.stderr.strip() or apply_bug.stdout.strip()}"
            )
    subprocess.run(["git", "add", "."], cwd=dest, text=True, capture_output=True, timeout=120)
    subprocess.run(
        ["git", "-c", "user.email=a@b.c", "-c", "user.name=agent", "commit", "-m", "base"],
        cwd=dest,
        text=True,
        capture_output=True,
        timeout=120,
    )
    return dest


def prepare_task_environment(task: dict[str, Any], workspace: Path) -> tuple[dict[str, str], dict[str, Any]]:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    configured = task.get("local_venv_path") or metadata.get("local_venv_path")
    source = "task"
    if not configured:
        project_venv = ROOT / ".venv"
        configured = project_venv if (project_venv / "bin" / "python").is_file() else None
        source = "project" if configured else "system"
    try:
        env = task_runtime_env(workspace, venv_path=configured)
    except FileNotFoundError as exc:
        info = {
            "ready": False,
            "status": "infrastructure_blocked",
            "reason": "task_venv_missing",
            "output": str(exc),
            "configured_venv_path": str(configured),
            "environment_source": source,
        }
        return {}, info
    info = runtime_environment_info(env)
    info["environment_source"] = source
    info["test_command"] = task.get("test_command")
    return env, info


def readonly_budget_policy(
    step: int,
    call: Any,
    step_reports: list[dict[str, Any]],
    *,
    checkpoint_step: int,
    grace_steps: int,
) -> str | None:
    if checkpoint_step < 0 or step <= checkpoint_step:
        return None
    if _is_modification_action(call) or str(getattr(call, "tool_name", "")) == "finish":
        return None
    interval = max(1, grace_steps)
    streak = 1
    for report in reversed(step_reports):
        if int(report.get("step", 0)) <= checkpoint_step:
            break
        tool_call = report.get("tool_call")
        if tool_call and _is_modification_action(tool_call):
            break
        streak += 1
    return "append" if streak % interval == 0 else None


def _is_modification_action(call: Any) -> bool:
    if isinstance(call, dict):
        tool_name = str(call.get("tool_name", ""))
        arguments = call.get("arguments") or {}
    else:
        tool_name = str(getattr(call, "tool_name", ""))
        arguments = getattr(call, "arguments", {}) or {}
    return tool_name == "str_replace_editor" and str(arguments.get("command", "")) in MODIFICATION_COMMANDS


def completion_gate_result(call: Any, workspace: Path, step_reports: list[dict[str, Any]]) -> ToolResult | None:
    if str(getattr(call, "tool_name", "")) != "finish" or not git_diff(workspace, scope="all").strip():
        return None
    last_edit_step = max(
        (
            int(step.get("step", 0))
            for step in step_reports
            if (step.get("tool_call") or {}).get("tool_name") == "str_replace_editor"
            and ((step.get("tool_call") or {}).get("arguments") or {}).get("command")
            in {"str_replace", "line_replace", "insert", "create", "undo_edit"}
        ),
        default=0,
    )
    valid_test = any(
        int(step.get("step", 0)) > last_edit_step
        and (step.get("tool_call") or {}).get("tool_name") == "run_tests"
        and (step.get("observation") or {}).get("returncode") == 0
        and not ((step.get("observation") or {}).get("extra") or {}).get("all_skipped")
        for step in step_reports
    )
    valid_diff = any(
        int(step.get("step", 0)) > last_edit_step
        and (step.get("tool_call") or {}).get("tool_name") == "git_diff"
        and (step.get("observation") or {}).get("returncode") == 0
        for step in step_reports
    )
    missing = []
    if not valid_test:
        missing.append("a passing, non-skipped run_tests after the latest edit")
    if not valid_diff:
        missing.append("git_diff after the latest edit")
    if not missing:
        return None
    return ToolResult(
        tool_name="finish",
        returncode=2,
        output=(
            "Completion gate rejected finish because a patch exists. Complete the missing checks first: "
            + "; ".join(missing)
            + ". Address failing or all-skipped tests before finishing."
        ),
        extra={"completion_gate": True, "missing_tools": missing},
    )


def completion_ready(workspace: Path, step_reports: list[dict[str, Any]]) -> bool:
    if not git_diff(workspace, scope="all").strip():
        return False
    finish = OpenHandsToolParser.parse_first('{"tool_name":"finish","arguments":{}}')
    return finish is not None and completion_gate_result(finish, workspace, step_reports) is None


def fetch_repo_to_cache(task: dict[str, Any], repo_cache: Path) -> Path:
    repo_cache.mkdir(parents=True, exist_ok=True)
    repo = str(task["repo"])
    commit = str(task["base_commit"])
    dest = repo_cache / cache_dir_name(repo, commit)
    if dest.exists():
        return dest

    tmp_dest = dest.with_name(dest.name + ".tmp")
    if tmp_dest.exists():
        shutil.rmtree(tmp_dest)

    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="swegym_repo_fetch_") as tmp:
        tmp_path = Path(tmp)
        try:
            clone_repo_at_commit(repo, commit, tmp_path / "repo")
            copy_repo_tree(tmp_path / "repo", tmp_dest)
        except Exception as exc:
            errors.append(f"git clone checkout failed: {exc!r}")
            if tmp_dest.exists():
                shutil.rmtree(tmp_dest)
            try:
                download_repo_tarball(repo, commit, tmp_path / "tarball_repo")
                copy_repo_tree(tmp_path / "tarball_repo", tmp_dest)
            except Exception as tar_exc:
                errors.append(f"tarball download failed: {tar_exc!r}")

    if not tmp_dest.exists():
        joined = "; ".join(errors)
        raise FileNotFoundError(
            f"Cached repo not found and auto-fetch failed for {task['instance_id']} "
            f"repo={repo} commit={commit}: {joined}"
        )
    tmp_dest.rename(dest)
    return dest


def cache_dir_name(repo: str, commit: str) -> str:
    return f"{repo.replace('/', '__')}__{commit[:12]}"


def safe_instance_name(value: Any) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in str(value))


def clone_repo_at_commit(repo: str, commit: str, dest: Path) -> None:
    clone_url = f"https://github.com/{repo}.git"
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", clone_url, str(dest)],
        text=True,
        capture_output=True,
        check=True,
        timeout=600,
    )
    checkout = subprocess.run(
        ["git", "-C", str(dest), "checkout", "--force", commit],
        text=True,
        capture_output=True,
        timeout=600,
    )
    if checkout.returncode == 0:
        return
    subprocess.run(
        ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit],
        text=True,
        capture_output=True,
        check=True,
        timeout=600,
    )
    subprocess.run(
        ["git", "-C", str(dest), "checkout", "--force", commit],
        text=True,
        capture_output=True,
        check=True,
        timeout=600,
    )


def download_repo_tarball(repo: str, commit: str, dest: Path) -> None:
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    tar_path = dest.with_suffix(".tar.gz")
    curl = shutil.which("curl")
    if curl:
        result = subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "--connect-timeout",
                "20",
                "--max-time",
                "180",
                "--user-agent",
                "codeagent-swegym-fetch",
                "--output",
                str(tar_path),
                url,
            ],
            text=True,
            capture_output=True,
            timeout=210,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"curl failed for {url}")
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "codeagent-swegym-fetch"})
        with urllib.request.urlopen(request, timeout=120) as response:
            tar_path.write_bytes(response.read())
    extract_root = dest.with_name(dest.name + "_extract")
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        safe_extract_tar(tar, extract_root)
    roots = [item for item in extract_root.iterdir() if item.is_dir()]
    if not roots:
        raise FileNotFoundError(f"No repository root found in tarball: {url}")
    roots[0].rename(dest)


def safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    resolved_dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        try:
            target.relative_to(resolved_dest)
        except ValueError as exc:
            raise ValueError(f"Unsafe tar member path: {member.name}")
    tar.extractall(dest)


def copy_repo_tree(src: Path, dest: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        "build",
        "dist",
    )
    shutil.copytree(src, dest, ignore=ignore)


def find_cached_repo(task: dict[str, Any], repo_cache: Path) -> Path:
    prefix = task["base_commit"][:12]
    repo_key = task["repo"].replace("/", "__").replace("-", "_")
    candidates = sorted(repo_cache.glob(f"{repo_key}__{prefix[:12]}*"))
    if candidates:
        return candidates[0]
    # Fallback for older cache names that preserve hyphens before slash replacement.
    alt_key = task["repo"].replace("/", "__")
    candidates = sorted(repo_cache.glob(f"{alt_key}__{prefix[:12]}*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Cached repo not found for {task['instance_id']} commit={task['base_commit']}")


def selected_files(parsed: dict[str, Any], planner: dict[str, Any]) -> list[str]:
    files = []
    for item in parsed.get("files_to_inspect") or []:
        if isinstance(item, str):
            files.append(item)
    for item in planner.get("candidate_files") or []:
        if isinstance(item, dict) and item.get("path"):
            files.append(str(item["path"]))
        elif isinstance(item, str):
            files.append(item)
    return list(dict.fromkeys(files))[:8]


def read_file_context(workspace: Path, files: list[str], max_chars: int) -> list[dict[str, Any]]:
    contexts = []
    for rel in files:
        path = (workspace / rel).resolve()
        if not str(path).startswith(str(workspace.resolve())) or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        clipped = text[:max_chars]
        numbered = "\n".join(f"{idx:04d}: {line}" for idx, line in enumerate(clipped.splitlines(), start=1))
        contexts.append(
            {
                "path": rel,
                "content": clipped,
                "numbered_content": numbered,
                "truncated": len(text) > max_chars,
            }
        )
    return contexts


def call_model(
    url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout_sec: int,
    args: argparse.Namespace,
    temperature: float | None = None,
) -> str:
    api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    if args.api_key_env and not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    return request_chat_completion(
        url=url,
        model=model,
        messages_for_attempt=lambda _attempt: [normalize_message_for_api(message) for message in messages],
        api_key=api_key,
        timeout_sec=timeout_sec,
        max_tokens=args.max_tokens,
        temperature=args.temperature if temperature is None else temperature,
        top_p=args.top_p,
        thinking=None if args.thinking_mode == "auto" else args.thinking_mode,
        reasoning_effort=args.reasoning_effort,
        retries=args.api_retries,
        backoff_sec=args.retry_backoff_sec,
        max_backoff_sec=args.retry_max_backoff_sec,
        retry_events=getattr(args, "_api_retry_events", None),
        response_events=getattr(args, "_model_response_events", None),
        use_system_proxy=args.model_network == "system",
    )


def shrink_messages_for_retry(messages: list[dict[str, Any]], attempt: int) -> list[dict[str, Any]]:
    max_chars = max(2000, 12000 // (2 ** max(0, attempt - 1)))
    shrunk = [shrink_message(message, max_chars) for message in messages]
    if attempt:
        shrunk.append(
            {
                "role": "user",
                "content": tool_call_retry_message(),
            }
        )
    return shrunk


def tool_call_retry_message() -> str:
    return (
        "No OpenHands JSON tool call found.\n"
        "Retry with exactly one JSON object and no prose. Example: "
        '{"tool_name": "execute_bash", "arguments": {"command": "pwd"}}'
    )


def is_normalized_json_tool_call(parsed_raw: str, original: str) -> bool:
    raw = parsed_raw.strip()
    return raw.startswith("{") and raw != original.strip()


def shrink_message(message: dict[str, Any], max_chars: int) -> dict[str, Any]:
    copied = normalize_message_for_api(message)
    content = copied.get("content")
    if isinstance(content, str) and len(content) > max_chars:
        parsed = try_parse_json(content)
        if isinstance(parsed, dict):
            copied["content"] = json.dumps(shrink_json_payload(parsed, max_chars), ensure_ascii=False, allow_nan=False)
        else:
            copied["content"] = content[:max_chars] + f"\n... truncated {len(content) - max_chars} chars for retry ..."
    return copied


def normalize_message_for_api(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    if role == "tool" and "tool_call_id" not in message:
        name = message.get("name", "tool")
        return {"role": "user", "content": f"Tool observation from {name}:\n{message.get('content', '')}"}
    allowed = {"role", "content", "name", "tool_call_id", "tool_calls"}
    return {key: value for key, value in message.items() if key in allowed and value is not None}


def shrink_json_payload(value: Any, max_chars: int) -> Any:
    if isinstance(value, dict):
        return {key: shrink_json_payload(item, max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [shrink_json_payload(item, max_chars) for item in value[:10]]
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + f"\n... truncated {len(value) - max_chars} chars for retry ..."
    return value


def try_parse_json(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return value


def parse_json_answer(answer: str) -> dict[str, Any]:
    text = answer.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
    return {"analysis": "failed to parse model JSON", "edits": [], "tests_to_run": [], "raw_answer": answer}


def apply_edits(workspace: Path, edits: list[dict[str, Any]]) -> dict[str, Any]:
    edited_files: list[str] = []
    applied: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for edit in edits:
        rel = str(edit.get("path", ""))
        old = str(edit.get("old_str", ""))
        new = str(edit.get("new_str", ""))
        path = (workspace / rel).resolve()
        if not rel or not str(path).startswith(str(workspace.resolve())):
            errors.append({"path": rel, "error": "invalid path"})
            continue
        if not path.exists():
            errors.append({"path": rel, "error": "path does not exist"})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        updated, method, detail = apply_single_edit(text, edit)
        if updated is None:
            errors.append({"path": rel, "error": "edit target not found", "detail": detail})
            continue
        path.write_text(updated, encoding="utf-8")
        edited_files.append(rel)
        applied.append({"path": rel, "method": method, "detail": detail})
    return {"edited_files": sorted(set(edited_files)), "applied": applied, "errors": errors}


def apply_single_edit(text: str, edit: dict[str, Any]) -> tuple[str | None, str, dict[str, Any]]:
    old = str(edit.get("old_str", ""))
    new = str(edit.get("new_str", ""))
    if old and old in text:
        return text.replace(old, new, 1), "exact_old_str", {"old_len": len(old), "new_len": len(new)}

    start_line = coerce_int(edit.get("start_line"))
    end_line = coerce_int(edit.get("end_line"))
    if start_line is not None and end_line is not None:
        replaced = replace_line_range(text, start_line, end_line, new)
        if replaced is not None:
            return replaced, "line_range", {"start_line": start_line, "end_line": end_line}

    if old:
        replaced, score, lines = fuzzy_replace(text, old, new)
        if replaced is not None:
            return replaced, "fuzzy_old_str", {"score": round(score, 4), "line_range": lines}

    return None, "not_applied", {"has_old_str": bool(old), "start_line": start_line, "end_line": end_line}


def coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def replace_line_range(text: str, start_line: int, end_line: int, replacement: str) -> str | None:
    lines = text.splitlines(keepends=True)
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        return None
    replacement_lines = replacement.splitlines(keepends=True)
    if replacement and not replacement.endswith(("\n", "\r")):
        replacement_lines[-1:] = [replacement_lines[-1] + ("\n" if lines[end_line - 1].endswith("\n") else "")]
    updated = lines[: start_line - 1] + replacement_lines + lines[end_line:]
    return "".join(updated)


def fuzzy_replace(text: str, old: str, new: str) -> tuple[str | None, float, list[int] | None]:
    lines = text.splitlines(keepends=True)
    old_lines = old.splitlines(keepends=True)
    if not lines or not old_lines:
        return None, 0.0, None

    best_score = 0.0
    best_range: tuple[int, int] | None = None
    target_len = len(old_lines)
    min_len = max(1, target_len - 3)
    max_len = min(len(lines), target_len + 3)
    normalized_old = normalize_for_match(old)

    for window_len in range(min_len, max_len + 1):
        for start in range(0, len(lines) - window_len + 1):
            candidate = "".join(lines[start : start + window_len])
            score = difflib.SequenceMatcher(None, normalized_old, normalize_for_match(candidate)).ratio()
            if score > best_score:
                best_score = score
                best_range = (start, start + window_len)

    if best_range is None or best_score < 0.86:
        return None, best_score, None

    start, end = best_range
    replacement_lines = new.splitlines(keepends=True)
    if new and not new.endswith(("\n", "\r")):
        replacement_lines[-1:] = [replacement_lines[-1] + ("\n" if lines[end - 1].endswith("\n") else "")]
    updated = lines[:start] + replacement_lines + lines[end:]
    return "".join(updated), best_score, [start + 1, end]


def normalize_for_match(value: str) -> str:
    normalized = "\n".join(line.rstrip() for line in value.replace("\r\n", "\n").split("\n")).strip()
    return "".join(normalized.split())


def verify_workspace(workspace: Path, edited_files: list[str], tests: list[str], args: argparse.Namespace) -> dict[str, Any]:
    compile_results = []
    compile_failed = False
    for rel in edited_files:
        if rel.endswith(".py"):
            result = run_cmd([sys.executable, "-m", "py_compile", rel], workspace, args.timeout_sec)
            compile_results.append(result)
            compile_failed = compile_failed or result["returncode"] != 0
    test_results = []
    if args.run_tests:
        for command in tests[:2]:
            test_results.append(run_cmd(command, workspace, args.timeout_sec))
    tests_attempted = bool(test_results)
    tests_passed = bool(test_results) and all(item["returncode"] == 0 for item in test_results)
    return {
        "compile_results": compile_results,
        "compile_failed": compile_failed,
        "tests_attempted": tests_attempted,
        "tests_passed": tests_passed,
        "test_results": test_results,
        "test_output": "\n".join(item.get("output", "")[-2000:] for item in test_results),
    }


def run_cmd(command: str | list[str], cwd: Path, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(command, cwd=cwd, shell=isinstance(command, str), text=True, capture_output=True, timeout=timeout)
    return {
        "command": command,
        "returncode": completed.returncode,
        "output": (completed.stdout or "") + (completed.stderr or ""),
        "elapsed_sec": round(time.monotonic() - started, 3),
    }


def git_diff(workspace: Path, *, scope: str = "all") -> str:
    command = ["git", "diff", "--", "."]
    if scope == "source":
        command.extend(source_patch_excludes())
    result = subprocess.run(command, cwd=workspace, text=True, capture_output=True, timeout=60)
    return result.stdout


def source_patch_excludes() -> list[str]:
    return [
        ":(exclude)tests/**",
        ":(exclude)**/tests/**",
        ":(exclude)test/**",
        ":(exclude)**/test/**",
        ":(exclude)examples/**",
        ":(exclude)**/examples/**",
        ":(exclude)docs/**",
        ":(exclude)**/docs/**",
        ":(exclude)doc/**",
        ":(exclude)**/doc/**",
        ":(exclude)benchmarks/**",
        ":(exclude)**/benchmarks/**",
        ":(exclude)*_test.py",
        ":(exclude)test_*.py",
        ":(exclude)**/*_test.py",
        ":(exclude)**/test_*.py",
    ]


def patch_files(patch: str) -> list[str]:
    files = []
    for line in str(patch).splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
                files.append(path[2:] if path.startswith("b/") else path)
    return sorted(set(files))


def tool_schema(tool_profile: str = "official-core") -> list[dict[str, Any]]:
    names = list(CORE_REPAIR_TOOLS)
    if tool_profile == "extended":
        names.extend(SWE_EXTENSION_TOOLS)
    return tool_schemas(names)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
