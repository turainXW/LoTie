#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.custom_benchmark import create_benchmark_project  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a custom SWE-Gym-style benchmark project.")
    parser.add_argument("--name", required=True, help="Benchmark name, used for directory and instance id.")
    parser.add_argument("--repo", required=True, help="Source repository directory to snapshot.")
    parser.add_argument("--output-dir", required=True, help="Parent directory for the generated benchmark project.")
    problem = parser.add_mutually_exclusive_group(required=True)
    problem.add_argument("--problem", help="Problem statement text.")
    problem.add_argument("--problem-file", help="File containing the problem statement.")
    parser.add_argument(
        "--pytest-node",
        action="append",
        default=[],
        help="FAIL_TO_PASS pytest node or file. Can be repeated.",
    )
    parser.add_argument("--test-patch", help="Existing pytest/test patch diff. Optional when --generate-tests is set.")
    parser.add_argument("--gold-patch", help="Optional reference fix patch; exported as records.gold.jsonl.")
    parser.add_argument("--image-name", help="Docker image tag. Default: local/codeagent.<name>:latest.")
    parser.add_argument("--python-version", default="3.11")
    parser.add_argument("--include-git", action="store_true", help="Keep .git in repo snapshot.")
    parser.add_argument("--generate-tests", action="store_true", help="Use an OpenAI-compatible model to draft pytest.")
    parser.add_argument("--generated-test-file", default="tests/test_generated_bug.py")
    parser.add_argument("--model-url", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem_statement = args.problem or Path(args.problem_file).read_text(encoding="utf-8")
    result = create_benchmark_project(
        name=args.name,
        repo=args.repo,
        output_dir=args.output_dir,
        problem_statement=problem_statement,
        pytest_nodes=list(args.pytest_node),
        test_patch_path=args.test_patch,
        gold_patch_path=args.gold_patch,
        image_name=args.image_name,
        python_version=args.python_version,
        include_git=args.include_git,
        generate_tests=args.generate_tests,
        generated_test_file=args.generated_test_file,
        model_url=args.model_url,
        model=args.model,
        api_key_env=args.api_key_env,
        max_tokens=args.max_tokens,
    )
    for key, value in result.to_dict().items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
