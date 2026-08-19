#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.harness_version import MODIFICATION_COMMANDS, harness_metadata  # noqa: E402
from code_agent_baseline.openhands_tools import OpenHandsToolParser, OpenHandsToolRuntime, ToolCall, ToolResult  # noqa: E402
from code_agent_baseline.model_transport import request_chat_completion  # noqa: E402
from code_agent_baseline.runtime_env import runtime_environment_info, task_runtime_env  # noqa: E402
from code_agent_baseline.tool_specs import (  # noqa: E402
    CORE_REPAIR_TOOLS,
    SWE_EXTENSION_TOOLS,
    render_tool_catalog,
    render_tool_few_shot,
)
from code_agent_baseline.working_memory import build_model_input  # noqa: E402
from code_agent_baseline.workspace_sandbox import prepare_workspace  # noqa: E402


SYSTEM_PROMPT = """Your name is miniCoder.
You are a practical code agent running in a terminal.
You solve coding tasks by inspecting files, editing source code, and running validation commands.

<RUNTIME>
- The shell and editor already run inside the repository workspace.
- The harness has already selected and activated the Python environment for every command.
- Do not create or activate a virtual environment unless the user explicitly asks you to change dependencies.
- Use repository-relative paths such as src/package/module.py.
- Do not assume /testbed, /repo, or /workspace exists. Run pwd if you need to confirm the current directory.
- Do not start commands with cd /testbed, cd /repo, or cd /workspace.
- Prefer commands like grep -n "needle" src/pkg/file.py or PYTHONPATH=src python -m pytest tests/test_file.py.
</RUNTIME>

<OUTPUT_PROTOCOL>
- If the user asks a general question that does not require repository inspection or edits, call:
  {"tool_name": "answer", "arguments": {"content": "your concise answer"}}
- Whenever you need to act on the repository, reply with exactly one JSON object and nothing else.
- Do not write prose before or after the JSON.
- Do not use Markdown code fences.
- The JSON object MUST have this schema:
  {"tool_name": "FUNCTION_NAME", "arguments": {"PARAMETER_NAME": "VALUE"}}
- Use an empty object for tools without parameters, for example:
  {"tool_name": "finish", "arguments": {}}
- Do not use XML, <function=...>, <tool_call>, or <invoke> in assistant messages.
</OUTPUT_PROTOCOL>

<BEHAVIOR>
- Prefer reading files before editing them.
- Keep changes minimal and task-focused.
- Run a targeted validation command when possible.
- Do not use emoji unless the user explicitly asks for them.
- Never ask for human help. If blocked, explain the blocker with answer(content) or finish.
</BEHAVIOR>
"""

@dataclass
class UseRunRecord:
    format: str
    run_id: str
    source_repo: str
    runtime_repo: str
    task: str
    agent_mode: str
    model: str
    status: str
    messages: list[dict[str, Any]]
    tool_steps: list[dict[str, Any]]
    patch: str
    harness: dict[str, Any] = field(default_factory=dict)
    final_message: str = ""
    summary: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-shot natural-language code task with visible tool steps.")
    parser.add_argument("task_positional", nargs="?", help="Natural-language task. Equivalent to --task.")
    parser.add_argument("--task", default=None, help="Natural-language task.")
    parser.add_argument("--repo", default=".", help="Repository/workspace path.")
    parser.add_argument("--venv", default=None, help="Optional task virtualenv path. Defaults to the source repo .venv.")
    parser.add_argument("--agent-mode", choices=["bench", "use"], default="use")
    parser.add_argument("--agent-name", default="miniCoder")
    parser.add_argument("--sandbox", choices=["copy", "direct"], default="copy")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default="outputs/codeagent_use")
    parser.add_argument("--model-url", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--model", default="deepseek-reasoner")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--api-retries", type=int, default=6)
    parser.add_argument("--retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--retry-max-backoff-sec", type=float, default=30.0)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-view-chars", type=int, default=20_000)
    parser.add_argument("--context-max-chars", type=int, default=48_000)
    parser.add_argument("--context-recent-turns", type=int, default=6)
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
    parser.add_argument("--show-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step-output-chars", type=int, default=1600)
    parser.add_argument("--show-result-output", action="store_true", help="Print tool result output while running.")
    parser.add_argument("--show-summary", action="store_true", help="Print the full JSON summary after the run.")
    parser.add_argument("--env-file", default=None, help="Optional .env file to load before reading API keys.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_files(args)
    task = args.task or args.task_positional
    if not task:
        raise SystemExit("Missing task. Use: codeagent use --task \"...\"")

    repo = Path(args.repo).resolve()
    run_id = args.run_id or time.strftime("use_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = prepare_workspace(
        repo,
        sandbox=args.sandbox,
        workspace_root=args.workspace_root,
        run_id=run_id,
    )
    runtime_repo = Path(workspace.runtime_repo).resolve()
    if args.sandbox == "copy":
        init_git_baseline(runtime_repo)

    if args.show_steps:
        print(f"user: {task}", flush=True)
        print(f"{args.agent_name}:", flush=True)

    record = run_agent(task, workspace, args=args, run_id=run_id)
    write_outputs(record, output_dir)
    if args.show_summary:
        print("SUMMARY " + json.dumps(record.summary, ensure_ascii=False), flush=True)
    print_completion_report(record, args)


def run_agent(
    task: str,
    workspace: Any,
    *,
    args: argparse.Namespace,
    run_id: str,
    session_context: str = "",
) -> UseRunRecord:
    runtime_repo = Path(workspace.runtime_repo).resolve()
    runtime_env, environment_info = prepare_use_environment(workspace, runtime_repo, args)
    if not environment_info.get("ready"):
        raise RuntimeError(f"Runtime environment preflight failed: {environment_info}")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(args.agent_mode)},
        {
            "role": "user",
            "content": build_user_prompt(
                task,
                runtime_repo,
                args.agent_mode,
                session_context=session_context,
                environment_info=environment_info,
            ),
        },
    ]
    runtime = OpenHandsToolRuntime(
        runtime_repo,
        timeout_sec=args.timeout_sec,
        max_view_chars=args.max_view_chars,
        allowed_tools=allowed_tools(args.agent_mode),
        runtime_env=runtime_env,
    )
    tool_steps: list[dict[str, Any]] = []
    status = "max_steps"
    started = time.monotonic()
    context_snapshots: list[dict[str, Any]] = []
    api_retry_events: list[dict[str, Any]] = []
    args._api_retry_events = api_retry_events

    for step in range(1, args.max_steps + 1):
        model_input, context_snapshot = build_model_input(
            messages,
            max_chars=args.context_max_chars,
            recent_turns=args.context_recent_turns,
        )
        context_snapshots.append({"step": step, **context_snapshot.to_dict()})
        try:
            answer = call_model(args.model_url, args.model, model_input, timeout_sec=args.timeout_sec, args=args)
        except Exception as exc:
            result = {
                "tool_name": "model_api",
                "returncode": 1,
                "output": repr(exc),
                "failure_type": "model_api_failure",
            }
            messages.append(tool_observation_message("model_api", result))
            tool_steps.append({"step": step, "tool_call": None, "observation": result})
            print_step(step, None, result, args)
            status = "model_api_failure"
            break
        messages.append({"role": "assistant", "content": answer})
        call = OpenHandsToolParser.parse_first(answer)
        if call is None:
            if args.agent_mode == "use" and answer.strip() and is_general_question(task):
                status = "answered"
                print_final_message(answer, args)
                break
            result = {"tool_name": "parser", "returncode": 1, "output": tool_call_retry_message(), "raw": answer}
            messages.append(tool_observation_message("parser", result))
            tool_steps.append({"step": step, "tool_call": None, "observation": result})
            print_step(step, None, result, args)
            status = "parse_error"
            break
        if is_normalized_json_tool_call(call.raw, answer):
            messages[-1]["content"] = call.raw
        if call.tool_name == "answer":
            content = str(call.arguments.get("content", "")).strip()
            result = {"tool_name": "answer", "returncode": 0, "output": content, "elapsed_sec": 0.0}
            messages.append(tool_observation_message("answer", result))
            tool_steps.append({"step": step, "tool_call": call.to_dict(), "observation": result})
            print_final_message(content, args)
            status = "answered"
            break
        readonly_policy = readonly_budget_policy(
            step,
            call,
            tool_steps,
            checkpoint_step=args.edit_checkpoint_step,
            grace_steps=args.readonly_grace_steps,
        )
        result_obj = completion_gate_result(call, runtime_repo, tool_steps)
        policy_result = result_obj
        result_obj = result_obj or runtime.execute(call)
        result = result_obj.to_dict()
        observation = tool_observation_message(call.tool_name, result)
        if policy_result is None and readonly_policy == "append":
            result["extra"] = {
                **(result.get("extra") or {}),
                "action_policy": "edit_recommended",
                "readonly_budget_warning_shown": True,
                "blocked_readonly_action": False,
            }
            observation["content"] += (
                "\n<READONLY_BUDGET_CHECKPOINT>This action was executed and its actual result is shown above. "
                "The last three actions did not attempt a repository modification. Use the accumulated evidence "
                "to converge: make the smallest supported source edit, or, if the task is already fixed, validate "
                "the change and finish. Avoid rereading ranges already present in the conversation."
                "</READONLY_BUDGET_CHECKPOINT>"
            )
        messages.append(observation)
        tool_steps.append({"step": step, "tool_call": call.to_dict(), "observation": result})
        print_step(step, call, result, args)
        if call.tool_name == "finish" and result.get("returncode") == 0:
            print_final_message(text_before_tool_call(answer), args)
            status = "finished"
            break

    patch = git_diff(runtime_repo)
    edited_files = patch_files(patch)
    if patch.strip() and status in {"finished", "answered"}:
        status = "edited"
    summary = {
        "run_id": run_id,
        "status": status,
        "agent_mode": args.agent_mode,
        "model": args.model,
        "source_repo": workspace.source_repo,
        "runtime_repo": workspace.runtime_repo,
        "steps": len(tool_steps),
        "patch_lines": len(patch.splitlines()),
        "edited_files": edited_files,
        "final_message": final_message_from_messages(messages),
        "session_context_chars": len(session_context),
        "runtime_environment": environment_info,
        "working_memory": {
            "max_chars": args.context_max_chars,
            "recent_turns": args.context_recent_turns,
            "snapshots": context_snapshots,
        },
        "api_retry_events": api_retry_events,
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    return UseRunRecord(
        format="codeagent_use_openhands_v1",
        run_id=run_id,
        source_repo=workspace.source_repo,
        runtime_repo=workspace.runtime_repo,
        task=task,
        agent_mode=args.agent_mode,
        model=args.model,
        status=status,
        messages=messages,
        tool_steps=tool_steps,
        patch=patch,
        harness=harness_metadata(
            checkpoint_step=args.edit_checkpoint_step,
            reminder_interval=args.readonly_grace_steps,
        ),
        final_message=str(summary.get("final_message", "")),
        summary=summary,
    )


def system_prompt(agent_mode: str) -> str:
    names = ["answer", *CORE_REPAIR_TOOLS]
    if agent_mode == "use":
        names.extend(SWE_EXTENSION_TOOLS)
    return "\n\n".join(
        [
            SYSTEM_PROMPT.strip(),
            render_tool_catalog(names),
            render_tool_few_shot(include_answer=True),
        ]
    )


def tool_call_retry_message() -> str:
    return (
        "No JSON tool call found.\n"
        "Retry with exactly one JSON object and no prose. Example: "
        '{"tool_name": "execute_bash", "arguments": {"command": "pwd"}}'
    )


def is_normalized_json_tool_call(parsed_raw: str, original: str) -> bool:
    raw = parsed_raw.strip()
    return raw.startswith("{") and raw != original.strip()


def allowed_tools(agent_mode: str) -> set[str]:
    tools = {"answer", *CORE_REPAIR_TOOLS}
    if agent_mode == "use":
        tools.update(SWE_EXTENSION_TOOLS)
    return tools


def build_user_prompt(
    task: str,
    runtime_repo: Path,
    agent_mode: str,
    *,
    session_context: str = "",
    environment_info: dict[str, Any] | None = None,
) -> str:
    mode_hint = (
        "You may use repo_context/problem_search before reading files when that helps locate relevant code. "
        "Use web_search for current public docs or open-source discovery, and download_repo only for public GitHub repositories."
        if agent_mode == "use"
        else "Use only the core OpenHands-style tools."
    )
    session_block = ""
    if session_context.strip():
        session_block = (
            "\nSession context from previous turns:\n"
            f"{session_context.strip()}\n\n"
            "Use this session context as compact memory only. Trust the current repository files and tool observations "
            "over stale summaries when they conflict.\n\n"
        )
    return (
        f"Repository workspace: {runtime_repo}\n"
        f"Runtime environment: {json.dumps(environment_info or {}, ensure_ascii=False)}\n"
        f"{session_block}"
        f"Task: {task}\n\n"
        f"{mode_hint}\n"
        "If this is a general question, call answer(content). "
        "If this is a repository task, work only inside the repository workspace and show progress through tool calls. "
        "When you believe the task is complete, provide a concise final note and call finish or answer(content)."
    )


def prepare_use_environment(workspace: Any, runtime_repo: Path, args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    source_repo = Path(workspace.source_repo).resolve()
    configured = Path(args.venv).expanduser() if getattr(args, "venv", None) else source_repo / ".venv"
    source = "explicit" if getattr(args, "venv", None) else "source_repo"
    if not (configured / "bin" / "python").is_file() and not configured.is_file():
        configured = ROOT / ".venv"
        source = "harness_project"
    env = task_runtime_env(runtime_repo, venv_path=configured if configured.exists() else None)
    info = runtime_environment_info(env)
    info["environment_source"] = source if configured.exists() else "system"
    return env, info


def readonly_budget_policy(
    step: int,
    call: Any,
    tool_steps: list[dict[str, Any]],
    *,
    checkpoint_step: int,
    grace_steps: int,
) -> str | None:
    if checkpoint_step < 0 or step <= checkpoint_step:
        return None
    tool_name = str(getattr(call, "tool_name", ""))
    if tool_name in {"finish", "answer"} or _is_modification_action(call):
        return None
    interval = max(1, grace_steps)
    streak = 1
    for report in reversed(tool_steps):
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


def completion_gate_result(call: Any, repo: Path, tool_steps: list[dict[str, Any]]) -> ToolResult | None:
    if str(getattr(call, "tool_name", "")) != "finish" or not git_diff(repo).strip():
        return None
    last_edit_step = max(
        (
            int(step.get("step", 0))
            for step in tool_steps
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
        for step in tool_steps
    )
    valid_diff = any(
        int(step.get("step", 0)) > last_edit_step
        and (step.get("tool_call") or {}).get("tool_name") == "git_diff"
        and (step.get("observation") or {}).get("returncode") == 0
        for step in tool_steps
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
        output="Completion gate rejected finish. Complete these checks first: " + "; ".join(missing),
        extra={"completion_gate": True, "missing_tools": missing},
    )


def call_model(url: str, model: str, messages: list[dict[str, Any]], *, timeout_sec: int, args: argparse.Namespace) -> str:
    api_key = os.environ.get(args.api_key_env, "")
    if args.api_key_env and not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    return request_chat_completion(
        url=url,
        model=model,
        messages_for_attempt=lambda attempt: shrink_messages(messages, attempt),
        api_key=api_key,
        timeout_sec=timeout_sec,
        max_tokens=args.max_tokens,
        retries=args.api_retries,
        backoff_sec=args.retry_backoff_sec,
        max_backoff_sec=args.retry_max_backoff_sec,
        retry_events=getattr(args, "_api_retry_events", None),
    )


def shrink_messages(messages: list[dict[str, Any]], attempt: int) -> list[dict[str, Any]]:
    max_chars = max(3000, 20000 // (2 ** max(0, attempt - 1)))
    result = []
    for message in messages:
        copied = {key: value for key, value in message.items() if key in {"role", "content", "name"}}
        content = copied.get("content")
        if isinstance(content, str) and len(content) > max_chars:
            copied["content"] = content[:max_chars] + f"\n... truncated {len(content) - max_chars} chars ..."
        result.append(copied)
    return result


def tool_observation_message(tool_name: str, payload: dict[str, Any]) -> dict[str, str]:
    output = payload.get("output")
    if isinstance(output, str):
        content = output.rstrip()
        if payload.get("returncode") is not None and "[Command finished with exit code" not in content:
            content += f"\n[Command finished with exit code {payload.get('returncode')}]"
        reminder = protocol_reminder_for_tool_error(content)
        if reminder:
            content += "\n" + reminder
    else:
        content = json.dumps(payload, ensure_ascii=False, indent=2)
    return {"role": "user", "content": f"EXECUTION RESULT of [{tool_name}]:\n{content}"}


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


def print_step(step: int, call: ToolCall | None, result: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.show_steps:
        return
    if call is None:
        print(f"[step {step:02d}] PARSE_ERROR", flush=True)
    else:
        print(f"[step {step:02d}] {summarize_call(call)}", flush=True)
    if not args.show_result_output:
        return
    print(f"[step {step:02d}] returncode={result.get('returncode')} elapsed={result.get('elapsed_sec', 0)}s", flush=True)
    output = str(result.get("output", "") or "")
    if output:
        print(clip(output, args.step_output_chars), flush=True)


def print_final_message(message: str, args: argparse.Namespace) -> None:
    text = message.strip()
    if not text or not args.show_steps:
        return
    print(clip(text, args.step_output_chars), flush=True)


def print_completion_report(record: UseRunRecord, args: argparse.Namespace) -> None:
    if not args.show_steps or record.status == "answered":
        return
    edited_files = record.summary.get("edited_files", [])
    if record.status in {"finished", "edited", "resolved"}:
        if edited_files:
            print(f"完成：修改了 {len(edited_files)} 个文件。", flush=True)
            for path in edited_files[:8]:
                print(f"- {path}", flush=True)
            if len(edited_files) > 8:
                print(f"- ... 还有 {len(edited_files) - 8} 个文件", flush=True)
        else:
            print("完成：未检测到文件修改。", flush=True)
        if record.summary.get("patch_lines", 0):
            print(f"patch 行数：{record.summary['patch_lines']}", flush=True)
        return
    print(f"未完成：{record.status}", flush=True)


def text_before_tool_call(text: str) -> str:
    if OpenHandsToolParser.parse_first(text):
        return ""
    marker = "<function="
    idx = text.find(marker)
    return text[:idx].strip() if idx >= 0 else text.strip()


def final_message_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", "") or "")
        call = OpenHandsToolParser.parse_first(content)
        if call:
            if call.tool_name == "answer":
                return str(call.arguments.get("content", "")).strip()
            continue
        text = text_before_tool_call(content)
        if text:
            return text
    return ""


def is_general_question(task: str) -> bool:
    normalized = task.strip().lower()
    markers = [
        "你好",
        "您好",
        "在吗",
        "你会做什么",
        "你能做什么",
        "能做什么",
        "怎么使用",
        "如何使用",
        "使用方法",
        "介绍一下",
        "help",
        "hello",
        "hi",
        "what can you do",
        "how do i use",
        "how to use",
    ]
    return any(marker in normalized for marker in markers)


def summarize_call(call: ToolCall) -> str:
    args = call.arguments
    if call.tool_name == "answer":
        return "answer"
    if call.tool_name == "execute_bash":
        return f"$ {str(args.get('command', ''))}"
    if call.tool_name == "str_replace_editor":
        command = str(args.get("command", ""))
        path = str(args.get("path", ""))
        detail = [f"edit:{command}", path]
        if command in {"str_replace", "insert", "create"}:
            old_str = str(args.get("old_str", ""))
            new_str = str(args.get("new_str", args.get("file_text", "")))
            if old_str:
                detail.append(f"old_len={len(old_str)}")
            if new_str:
                detail.append(f"new_len={len(new_str)}")
        return " ".join(detail)
    if call.tool_name in {"repo_context", "problem_search", "web_search"}:
        return f"{call.tool_name}: {str(args.get('query', ''))}"
    if call.tool_name == "download_repo":
        return f"download_repo: {str(args.get('repo_url', args.get('url', '')))}"
    if call.tool_name == "finish":
        return "finish"
    return call.tool_name


def write_outputs(record: UseRunRecord, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectory.json").write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "messages_sft.jsonl").write_text(
        json.dumps(
            {
                "format": "codeagent_use_messages_v1",
                "run_id": record.run_id,
                "status": record.status,
                "harness": record.harness,
                "messages": record.messages,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "patch.diff").write_text(record.patch, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(record.summary, ensure_ascii=False, indent=2), encoding="utf-8")


def init_git_baseline(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, text=True, capture_output=True, timeout=30)
    subprocess.run(["git", "add", "."], cwd=repo, text=True, capture_output=True, timeout=120)
    subprocess.run(
        ["git", "-c", "user.email=a@b.c", "-c", "user.name=agent", "commit", "-m", "base"],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=120,
    )


def git_diff(repo: Path) -> str:
    result = subprocess.run(["git", "diff", "--", "."], cwd=repo, text=True, capture_output=True, timeout=60)
    return result.stdout


def patch_files(patch: str) -> list[str]:
    files = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
                files.append(path[2:] if path.startswith("b/") else path)
    return sorted(set(files))


def load_env_files(args: argparse.Namespace) -> None:
    paths = []
    if args.env_file:
        paths.append(Path(args.env_file))
    paths.extend([ROOT / ".env", Path(args.repo).resolve() / ".env"])
    for path in paths:
        if path.exists():
            load_env_file(path)


def load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        if key not in os.environ or not os.environ[key]:
            os.environ[key] = value


def clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... clipped {len(text) - max_chars} chars ..."


if __name__ == "__main__":
    main()
