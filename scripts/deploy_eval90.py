#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from code_agent_baseline.eval90_deploy import (  # noqa: E402
    Eval90Layout,
    deployment_status,
    doctor,
    prepare_evaluation_tasks,
    sanity_function_tasks,
    sanity_swesmith_tasks,
    setup_function_environment,
    setup_swesmith_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the portable Lottie Eval90 local verifier environment.")
    parser.add_argument(
        "command",
        choices=["doctor", "prepare", "setup-functions", "setup-repos", "sanity-functions", "sanity-repos", "status", "all"],
    )
    parser.add_argument("--state-root", default=".codeagent/eval90", help="Persistent repos, venvs, tasks, and reports.")
    parser.add_argument("--python", default=None, help="Python 3.10 executable used to create verifier environments.")
    parser.add_argument("--setup-timeout-sec", type=int, default=1800)
    parser.add_argument("--function-timeout-sec", type=int, default=30)
    parser.add_argument("--repo-timeout-sec", type=int, default=600)
    parser.add_argument("--p2p-limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = Eval90Layout.create(ROOT, args.state_root)
    python = args.python
    if args.command not in {"prepare", "status"} and not python:
        raise SystemExit("--python must point to a Python 3.10 executable")

    if args.command == "doctor":
        emit(doctor(layout, python_executable=python))
        return
    if args.command == "prepare":
        emit(prepare_evaluation_tasks(layout))
        return
    if args.command == "setup-functions":
        emit(setup_function_environment(layout, python_executable=python, timeout_sec=args.setup_timeout_sec))
        return
    if args.command == "setup-repos":
        result = setup_swesmith_environment(layout, python_executable=python, timeout_sec=args.setup_timeout_sec)
        emit(result)
        if result["blocked"]:
            raise SystemExit(2)
        return
    if args.command == "sanity-functions":
        result = sanity_function_tasks(layout, timeout_sec=args.function_timeout_sec)
        emit(result)
        if result["passed"] != 60:
            raise SystemExit(2)
        return
    if args.command == "sanity-repos":
        result = sanity_swesmith_tasks(layout, timeout_sec=args.repo_timeout_sec, p2p_limit=args.p2p_limit)
        emit(result)
        if result["passed"] != 30:
            raise SystemExit(2)
        return
    if args.command == "status":
        result = deployment_status(layout, python_executable=python)
        emit(result)
        if not result["ready"]:
            raise SystemExit(2)
        return

    doctor_result = doctor(layout, python_executable=python)
    emit({"phase": "doctor", **doctor_result})
    if not doctor_result["ready"]:
        raise SystemExit(2)
    emit({"phase": "prepare", **prepare_evaluation_tasks(layout)})
    emit({"phase": "setup-functions", **setup_function_environment(layout, python_executable=python, timeout_sec=args.setup_timeout_sec)})
    repo_setup = setup_swesmith_environment(layout, python_executable=python, timeout_sec=args.setup_timeout_sec)
    emit({"phase": "setup-repos", **repo_setup})
    if repo_setup["blocked"]:
        raise SystemExit(2)
    function_sanity = sanity_function_tasks(layout, timeout_sec=args.function_timeout_sec)
    emit({"phase": "sanity-functions", **function_sanity})
    repo_sanity = sanity_swesmith_tasks(layout, timeout_sec=args.repo_timeout_sec, p2p_limit=args.p2p_limit)
    emit({"phase": "sanity-repos", **repo_sanity})
    result = deployment_status(layout, python_executable=python)
    emit({"phase": "status", **result})
    if not result["ready"]:
        raise SystemExit(2)


def emit(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
