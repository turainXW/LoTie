#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import unittest
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline import CodeAgent, RuleBasedRepairBenchmarkModel, RuleBasedSmokeModel  # noqa: E402
from code_agent_baseline.task_factory import (  # noqa: E402
    create_code_repair_benchmark_tasks,
    create_toy_addition_task,
    write_tasks_jsonl,
)


@dataclass
class SuiteReport:
    name: str
    status: str
    total: int
    passed: int
    failed: int
    elapsed_sec: float
    artifacts: dict[str, str] = field(default_factory=dict)
    details: dict[str, object] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeAgent-RL benchmark/evaluation suites.")
    parser.add_argument(
        "--suite",
        choices=["all", "codeagent", "smoke", "leetcode"],
        default="all",
        help="Evaluation suite to run.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/code_agent_eval",
        help="Directory for reports and suite artifacts.",
    )
    parser.add_argument("--max-steps", type=int, default=8, help="Max steps for the smoke agent.")
    return parser.parse_args()


def run_smoke_eval(output_dir: Path, max_steps: int) -> SuiteReport:
    started = time.perf_counter()
    suite_dir = output_dir / "smoke"
    suite_dir.mkdir(parents=True, exist_ok=True)

    task = create_toy_addition_task(suite_dir)
    write_tasks_jsonl(suite_dir / "tasks.jsonl", [task])

    agent = CodeAgent(model=RuleBasedSmokeModel(), max_steps=max_steps)
    trajectory = agent.run(task)
    trajectory_path = suite_dir / "trajectories.jsonl"
    trajectory_path.write_text(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")

    passed = int(trajectory.final_status == "pass")
    elapsed = round(time.perf_counter() - started, 4)
    return SuiteReport(
        name="smoke",
        status="pass" if passed else "fail",
        total=1,
        passed=passed,
        failed=1 - passed,
        elapsed_sec=elapsed,
        artifacts={
            "tasks": str((suite_dir / "tasks.jsonl").resolve()),
            "trajectory": str(trajectory_path.resolve()),
            "repo": str(Path(task.repo_path).resolve()),
        },
        details={
            "task_id": task.task_id,
            "final_status": trajectory.final_status,
            "steps": len(trajectory.steps),
            "validation_command": task.validation_command,
        },
    )


def run_leetcode_eval(output_dir: Path) -> SuiteReport:
    started = time.perf_counter()
    suite_dir = output_dir / "leetcode_top10"
    suite_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dir = ROOT / "benchmarks" / "leetcode_top10"
    sys.path.insert(0, str(benchmark_dir))
    loader = unittest.TestLoader()
    suite = loader.discover(str(benchmark_dir), pattern="test_*.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = stream.getvalue()
    output_path = suite_dir / "unittest_output.txt"
    output_path.write_text(output, encoding="utf-8")

    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    elapsed = round(time.perf_counter() - started, 4)
    return SuiteReport(
        name="leetcode_top10",
        status="pass" if result.wasSuccessful() else "fail",
        total=total,
        passed=passed,
        failed=failed,
        elapsed_sec=elapsed,
        artifacts={
            "unittest_output": str(output_path.resolve()),
            "solutions": str((benchmark_dir / "solutions.py").resolve()),
            "tests": str((benchmark_dir / "test_solutions.py").resolve()),
        },
        details={
            "failures": [_format_failure(test, err) for test, err in result.failures],
            "errors": [_format_failure(test, err) for test, err in result.errors],
            "skipped": len(result.skipped),
        },
    )


def run_codeagent_eval(output_dir: Path, max_steps: int) -> SuiteReport:
    started = time.perf_counter()
    suite_dir = output_dir / "codeagent"
    suite_dir.mkdir(parents=True, exist_ok=True)

    tasks = create_code_repair_benchmark_tasks(suite_dir / "repos")
    write_tasks_jsonl(suite_dir / "tasks.jsonl", tasks)

    agent = CodeAgent(model=RuleBasedRepairBenchmarkModel(), max_steps=max_steps)
    trajectory_path = suite_dir / "trajectories.jsonl"
    trajectory_path.write_text("", encoding="utf-8")

    task_results = []
    for task in tasks:
        trajectory = agent.run(task)
        CodeAgent.append_jsonl(trajectory_path, trajectory)
        task_results.append(
            {
                "task_id": task.task_id,
                "status": trajectory.final_status,
                "passed": trajectory.final_status == "pass",
                "steps": len(trajectory.steps),
                "repo": task.repo_path,
                "validation_command": task.validation_command,
            }
        )

    total = len(task_results)
    passed = sum(1 for result in task_results if result["passed"])
    failed = total - passed
    elapsed = round(time.perf_counter() - started, 4)
    return SuiteReport(
        name="codeagent",
        status="pass" if failed == 0 else "fail",
        total=total,
        passed=passed,
        failed=failed,
        elapsed_sec=elapsed,
        artifacts={
            "tasks": str((suite_dir / "tasks.jsonl").resolve()),
            "trajectories": str(trajectory_path.resolve()),
            "repos": str((suite_dir / "repos").resolve()),
        },
        details={
            "agent": "code_agent_baseline.CodeAgent",
            "model": RuleBasedRepairBenchmarkModel.name,
            "metric": "final_status == pass after running the verifier",
            "tasks": task_results,
        },
    )


def write_report(output_dir: Path, suites: list[SuiteReport]) -> Path:
    total = sum(suite.total for suite in suites)
    passed = sum(suite.passed for suite in suites)
    failed = sum(suite.failed for suite in suites)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "suite_count": len(suites),
            "passed_suites": sum(1 for suite in suites if suite.status == "pass"),
            "failed_suites": sum(1 for suite in suites if suite.status != "pass"),
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": failed,
            "pass_rate": passed / total if total else 0.0,
        },
        "suites": [asdict(suite) for suite in suites],
    }
    path = output_dir / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _format_failure(test: unittest.case.TestCase, error: str) -> dict[str, str]:
    return {"test": str(test), "error": error}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suites: list[SuiteReport] = []
    if args.suite in {"all", "codeagent"}:
        suites.append(run_codeagent_eval(output_dir, args.max_steps))
    if args.suite in {"smoke"}:
        suites.append(run_smoke_eval(output_dir, args.max_steps))
    if args.suite in {"all", "leetcode"}:
        suites.append(run_leetcode_eval(output_dir))

    report_path = write_report(output_dir, suites)
    total = sum(suite.total for suite in suites)
    passed = sum(suite.passed for suite in suites)
    failed = sum(suite.failed for suite in suites)
    print(f"report={report_path}")
    print(f"suites={len(suites)}")
    print(f"passed_tests={passed}")
    print(f"failed_tests={failed}")
    print(f"pass_rate={passed / total if total else 0.0:.4f}")
    for suite in suites:
        print(f"{suite.name}: status={suite.status} passed={suite.passed}/{suite.total} elapsed_sec={suite.elapsed_sec}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
