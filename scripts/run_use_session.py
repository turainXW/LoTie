#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.workspace_sandbox import prepare_workspace  # noqa: E402


def load_use_agent_module():
    path = ROOT / "scripts" / "run_use_agent.py"
    spec = importlib.util.spec_from_file_location("codeagent_run_use_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load use agent script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a task-by-task miniCoder session.")
    parser.add_argument("--repo", default=".", help="Repository/workspace path.")
    parser.add_argument("--venv", default=None, help="Optional task virtualenv path.")
    parser.add_argument("--agent-mode", choices=["bench", "use"], default="use")
    parser.add_argument("--agent-name", default="miniCoder")
    parser.add_argument("--sandbox", choices=["copy", "direct"], default="copy")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--output-dir", default="outputs/codeagent_session")
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
    parser.add_argument("--edit-checkpoint-step", type=int, default=10)
    parser.add_argument("--readonly-grace-steps", type=int, default=3)
    parser.add_argument("--show-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step-output-chars", type=int, default=1600)
    parser.add_argument("--show-result-output", action="store_true", help="Print tool result output while running.")
    parser.add_argument("--show-summary", action="store_true", help="Print the full JSON summary after each turn.")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--session-context-turns", type=int, default=6, help="How many previous turns to include in the next prompt.")
    parser.add_argument("--session-context-chars", type=int, default=6000, help="Maximum compact session context characters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_agent = load_use_agent_module()
    use_agent.load_env_files(args)

    repo = Path(args.repo).resolve()
    session_id = args.session_id or time.strftime("session_%Y%m%d_%H%M%S")
    session_output = Path(args.output_dir).resolve() / session_id
    session_output.mkdir(parents=True, exist_ok=True)

    workspace = prepare_workspace(
        repo,
        sandbox=args.sandbox,
        workspace_root=args.workspace_root,
        run_id=session_id,
    )
    runtime_repo = Path(workspace.runtime_repo).resolve()
    if args.sandbox == "copy":
        use_agent.init_git_baseline(runtime_repo)

    print(f"{args.agent_name}: session started. use /exit to quit, /status to show paths", flush=True)

    turn = 1
    session_history: list[dict[str, object]] = []
    while True:
        try:
            task = input("user: ").strip()
        except EOFError:
            print("", flush=True)
            break
        if not task:
            continue
        if task in {"/exit", "/quit", "exit", "quit"}:
            break
        if task == "/status":
            print(
                json.dumps(
                    {
                        "session_id": session_id,
                        "source_repo": str(repo),
                        "runtime_repo": str(runtime_repo),
                        "output_dir": str(session_output),
                        "turns_completed": turn - 1,
                        "session_context_turns": args.session_context_turns,
                        "session_context_chars": args.session_context_chars,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            continue

        turn_id = f"{session_id}_turn_{turn:03d}"
        turn_output = session_output / f"turn_{turn:03d}"
        try:
            print(f"{args.agent_name}:", flush=True)
            session_context = build_session_context(
                session_history,
                max_turns=args.session_context_turns,
                max_chars=args.session_context_chars,
            )
            record = use_agent.run_agent(task, workspace, args=args, run_id=turn_id, session_context=session_context)
            use_agent.write_outputs(record, turn_output)
            if args.show_summary:
                print("SUMMARY " + json.dumps(record.summary, ensure_ascii=False), flush=True)
            if args.sandbox == "copy":
                commit_turn(runtime_repo, turn)
            use_agent.print_completion_report(record, args)
            session_history.append(compact_turn_record(turn, task, record))
        except Exception as exc:
            error = {"turn": turn, "task": task, "error": repr(exc)}
            turn_output.mkdir(parents=True, exist_ok=True)
            (turn_output / "error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
            print("error " + json.dumps(error, ensure_ascii=False), flush=True)
            session_history.append({"turn": turn, "task": task, "status": "error", "error": repr(exc)})
        print("", flush=True)
        turn += 1

    print("session done", flush=True)


def commit_turn(repo: Path, turn: int) -> None:
    subprocess.run(["git", "add", "."], cwd=repo, text=True, capture_output=True, timeout=120)
    status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo, text=True, capture_output=True, timeout=60)
    if status.returncode == 0:
        return
    subprocess.run(
        ["git", "-c", "user.email=a@b.c", "-c", "user.name=agent", "commit", "-m", f"turn {turn}"],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=120,
    )


def build_session_context(history: list[dict[str, object]], *, max_turns: int, max_chars: int) -> str:
    if max_turns <= 0 or max_chars <= 0 or not history:
        return ""
    selected = history[-max_turns:]
    lines = [
        "The following is a compact summary of earlier turns in this same session.",
        "It records prior user requests, agent actions, edited files, and outcomes.",
    ]
    for item in selected:
        lines.append(
            "- turn {turn}: user={task}; status={status}; tools={tools}; edited_files={edited_files}; "
            "patch_lines={patch_lines}; final={final_message}".format(
                turn=item.get("turn", ""),
                task=_clip(str(item.get("task", "")), 400),
                status=item.get("status", ""),
                tools=", ".join(str(tool) for tool in item.get("tools", []) or []) or "none",
                edited_files=", ".join(str(path) for path in item.get("edited_files", []) or []) or "none",
                patch_lines=item.get("patch_lines", 0),
                final_message=_clip(str(item.get("final_message", item.get("error", ""))), 500),
            )
        )
        commands = item.get("commands", []) or []
        if commands:
            lines.append("  commands: " + " | ".join(_clip(str(command), 180) for command in commands[:6]))
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:].lstrip()


def compact_turn_record(turn: int, task: str, record: object) -> dict[str, object]:
    summary = getattr(record, "summary", {}) or {}
    tool_steps = getattr(record, "tool_steps", []) or []
    tools: list[str] = []
    commands: list[str] = []
    for step in tool_steps:
        call = (step.get("tool_call") or {}) if isinstance(step, dict) else {}
        tool_name = str(call.get("tool_name", ""))
        if tool_name:
            tools.append(tool_name)
        arguments = call.get("arguments") or {}
        if tool_name == "execute_bash" and isinstance(arguments, dict):
            command = str(arguments.get("command", "")).strip()
            if command:
                commands.append(command)
        elif tool_name == "str_replace_editor" and isinstance(arguments, dict):
            command = str(arguments.get("command", "")).strip()
            path = str(arguments.get("path", "")).strip()
            if command or path:
                commands.append(f"edit:{command} {path}".strip())
    return {
        "turn": turn,
        "task": task,
        "status": summary.get("status", ""),
        "final_message": summary.get("final_message", ""),
        "tools": sorted(set(tools)),
        "commands": commands,
        "edited_files": summary.get("edited_files", []),
        "patch_lines": summary.get("patch_lines", 0),
    }


def _clip(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


if __name__ == "__main__":
    main()
