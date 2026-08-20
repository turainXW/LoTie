from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    script_by_command = {
        "run": ROOT / "scripts" / "run_mini_agent.py",
        "use": ROOT / "scripts" / "run_use_agent.py",
        "session": ROOT / "scripts" / "run_use_session.py",
        "smoke": ROOT / "scripts" / "run_smoke.py",
        "repo-map": ROOT / "scripts" / "build_repo_map.py",
        "skills": ROOT / "scripts" / "manage_skills.py",
        "memory": ROOT / "scripts" / "manage_memory.py",
        "eval": ROOT / "scripts" / "evaluate.py",
        "export-training": ROOT / "scripts" / "export_training_data.py",
        "collect-swe-like": ROOT / "scripts" / "collect_swe_like_traces.py",
        "create-benchmark-project": ROOT / "scripts" / "create_benchmark_project.py",
        "download-openhands-samples": ROOT / "scripts" / "download_openhands_samples.py",
        "export-openhands-sft-dataset": ROOT / "scripts" / "export_openhands_sft_dataset.py",
        "problem-search": ROOT / "scripts" / "problem_search.py",
        "swebench-local": ROOT / "scripts" / "swebench_local.py",
        "audit-swegym-traces": ROOT / "scripts" / "audit_swegym_traces.py",
        "export-swegym-grounded": ROOT / "scripts" / "export_swegym_grounded_openhands.py",
        "prepare-swegym-eval-tasks": ROOT / "scripts" / "prepare_swegym_eval_tasks.py",
        "run-swegym-patch": ROOT / "scripts" / "run_swegym_patch_rollout.py",
        "verify-swegym-patches": ROOT / "scripts" / "verify_swegym_patches.py",
        "prepare-swesmith-local": ROOT / "scripts" / "prepare_swesmith_local.py",
        "prepare-evalplus-local": ROOT / "scripts" / "prepare_evalplus_local.py",
        "deploy-eval90": ROOT / "scripts" / "deploy_eval90.py",
        "pass-at-k": ROOT / "scripts" / "run_pass_at_k_experiment.py",
        "combine-pass-at-k": ROOT / "scripts" / "combine_pass_at_k_runs.py",
    }
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _print_help(script_by_command)
        return
    command = sys.argv[1]
    if command not in script_by_command:
        sys.argv = [str(script_by_command["use"]), *sys.argv[1:]]
        runpy.run_path(str(script_by_command["use"]), run_name="__main__")
        return
    sys.argv = [str(script_by_command[command]), *sys.argv[2:]]
    runpy.run_path(str(script_by_command[command]), run_name="__main__")


def _print_help(script_by_command: dict[str, Path]) -> None:
    parser = argparse.ArgumentParser(prog="lotie")
    parser.description = (
        "LoTie code-agent and trajectory pipeline CLI. "
        "Shortcut: `lotie \"your task\"` runs the default use-mode agent."
    )
    parser.add_argument("command", choices=sorted(script_by_command))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parser.print_help()


if __name__ == "__main__":
    main()
