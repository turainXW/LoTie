#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", str(ROOT / ".mswea"))
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
sys.path.insert(0, str(SRC))

from code_agent_baseline.mini_guarded import build_guarded_local_environment_class  # noqa: E402
from code_agent_baseline.mini_web_model import WebEnabledLitellmModel  # noqa: E402
from code_agent_baseline.context_manager import build_context_bundle  # noqa: E402
from code_agent_baseline.memory import learn_from_summary, summarize_trajectory, write_session_summary  # noqa: E402
from code_agent_baseline.runtime_env import project_runtime_env  # noqa: E402
from code_agent_baseline.workspace_sandbox import prepare_workspace  # noqa: E402


PLAN_TEMPLATE = """You are a coding agent running in PLAN mode.
You may inspect files, summarize repository context, and propose a plan.
You must not edit files, write files, delete files, commit, or push.
Use bash only for read-only inspection.
If web tools are enabled, you may call web_search to find open-source projects or docs.
Do not call download_repo in PLAN mode.
"""

BUILD_TEMPLATE = """You are a coding agent running in BUILD mode.
You may inspect files, edit code, run validation commands, and produce a final fix.
Keep changes minimal and verify them before submission.
If web tools are enabled, use web_search for discovery and download_repo for public GitHub repositories.
"""

DEBUG_TEMPLATE = """You are a coding agent running in DEBUG mode.
Focus on reading logs, reproducing failures, locating root causes, and proposing or applying minimal fixes.
Avoid destructive commands.
If web tools are enabled, use web_search for external references and download_repo for public GitHub repositories.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeAgent-RL on top of mini-swe-agent.")
    parser.add_argument("--repo", default=".", help="Repository/workspace path.")
    parser.add_argument("--task", required=True, help="Natural-language coding task.")
    parser.add_argument("--mode", choices=["plan", "build", "debug"], default="plan")
    parser.add_argument("--model", default="deepseek/deepseek-reasoner", help="LiteLLM model name for mini-swe-agent.")
    parser.add_argument("--output", default="data/code_agent_mini/trajectory.json")
    parser.add_argument("--step-limit", type=int, default=20)
    parser.add_argument("--cost-limit", type=float, default=3.0)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--require-approval-for-bash", action="store_true")
    parser.add_argument("--enable-web-tools", action="store_true")
    parser.add_argument(
        "--download-dir",
        default="refs/open_source",
        help="Directory, relative to --repo unless absolute, for controlled repository downloads.",
    )
    parser.add_argument("--disable-context", action="store_true")
    parser.add_argument("--refresh-repo-map", action="store_true")
    parser.add_argument("--repo-map", default=None)
    parser.add_argument("--skills-config", default=None)
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument("--context-max-files", type=int, default=12)
    parser.add_argument("--context-max-chars", type=int, default=5000)
    parser.add_argument("--skills-max-chars", type=int, default=2500)
    parser.add_argument("--memory-max-chars", type=int, default=3000)
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--summarize-session", action="store_true")
    parser.add_argument("--learn-from-session", action="store_true")
    parser.add_argument(
        "--sandbox",
        choices=["copy", "direct"],
        default="copy",
        help="copy runs the agent in an isolated workspace copy; direct runs in the source repo.",
    )
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from minisweagent.agents.default import DefaultAgent
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "mini-swe-agent is not installed. Install it first:\n"
            "  python3 -m pip install -r code_agent_quickstart/requirements-mini.txt"
        ) from exc

    repo = Path(args.repo).resolve()
    output_path = Path(args.output).resolve()
    workspace = prepare_workspace(
        repo,
        sandbox=args.sandbox,
        workspace_root=args.workspace_root,
        run_id=args.run_id,
    )
    runtime_repo = Path(workspace.runtime_repo).resolve()
    GuardedLocalEnvironment = build_guarded_local_environment_class()
    agent_env = project_runtime_env(ROOT)

    mode_template = {
        "plan": PLAN_TEMPLATE,
        "build": BUILD_TEMPLATE,
        "debug": DEBUG_TEMPLATE,
    }[args.mode]
    context_block = ""
    if not args.disable_context:
        bundle = build_context_bundle(
            repo,
            args.task,
            repo_map_path=args.repo_map,
            skills_config_path=args.skills_config,
            memory_dir=None if args.disable_memory else args.memory_dir,
            refresh_repo_map=args.refresh_repo_map,
            repo_max_files=args.context_max_files,
            repo_max_chars=args.context_max_chars,
            skills_max_chars=args.skills_max_chars,
            memory_max_chars=0 if args.disable_memory else args.memory_max_chars,
        )
        context_block = (
            f"\n\n<context_management>\n{bundle.prompt_block}\n"
            f"- repo_map_path: {bundle.repo_map_path}\n"
            f"- skills_config_path: {bundle.skills_config_path}\n"
            f"- memory_dir: {bundle.memory_dir}\n"
            "</context_management>\n"
        )

    agent = DefaultAgent(
        WebEnabledLitellmModel(model_name=args.model, enable_web_tools=args.enable_web_tools),
        GuardedLocalEnvironment(
            cwd=str(runtime_repo),
            timeout=args.timeout,
            mode=args.mode,
            require_approval_for_bash=args.require_approval_for_bash,
            enable_web_tools=args.enable_web_tools,
            download_dir=args.download_dir,
            env=agent_env,
        ),
        system_template=mode_template,
        instance_template=(
            "Task: {{task}}\n\n"
            f"Source repository: {repo}\n"
            f"Runtime sandbox repository: {runtime_repo}\n"
            "Work only in the runtime sandbox repository. Do not assume changes affect the source repository.\n"
            f"{context_block}\n"
            "When fully done, submit by running a command whose first output line is exactly:\n"
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"
            "Put your final answer on the following lines. Empty final answers are invalid.\n"
            "Example:\n"
            "python3 -c \"print('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'); print('Final answer here.')\"\n"
        ),
        step_limit=args.step_limit,
        cost_limit=args.cost_limit,
        output_path=output_path,
    )
    result = agent.run(args.task)
    print(f"mode={args.mode}")
    print(f"repo={repo}")
    print(f"sandbox={workspace.mode}")
    print(f"runtime_repo={runtime_repo}")
    print(f"output={output_path}")
    print(f"result={result}")
    if args.summarize_session or args.learn_from_session:
        memory_dir = Path(args.memory_dir).resolve() if args.memory_dir else repo / ".codeagent" / "memory"
        summary = summarize_trajectory(output_path)
        summary_path, index_path = write_session_summary(memory_dir, summary)
        print(f"session_summary={summary_path}")
        print(f"session_index={index_path}")
        if args.learn_from_session:
            for path in learn_from_summary(memory_dir, summary):
                print(f"learned_memory={path}")


if __name__ == "__main__":
    main()
