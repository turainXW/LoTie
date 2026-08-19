#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a SWE-smith training extension.")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--existing-selection", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--setup-results", required=True)
    parser.add_argument("--sanity-results", required=True)
    parser.add_argument("--runnable", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mutation_kind(instance_id: str) -> str:
    return instance_id.rsplit(".", 1)[-1].split("__", 1)[0]


def patch_changed_lines(patch: str) -> int:
    return sum(
        1
        for line in patch.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )


def normalized_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def distribution(values: list[float | int]) -> dict[str, float | int]:
    ordered = sorted(values)
    p90_index = max(0, min(len(ordered) - 1, int(0.9 * len(ordered)) - 1))
    return {
        "min": min(ordered),
        "median": statistics.median(ordered),
        "mean": round(statistics.fmean(ordered), 3),
        "p90": ordered[p90_index],
        "max": max(ordered),
    }


def existing_swe_ids(selection: dict[str, Any]) -> tuple[set[str], set[str]]:
    repos: set[str] = set()
    instances: set[str] = set()
    for split in ("train", "validation", "evaluation"):
        swe = selection["splits"][split]["swesmith_py"]
        repos.update(str(repo) for repo in swe["profiles"])
        instances.update(str(item) for item in swe["instance_ids"])
    return repos, instances


def main() -> None:
    args = parse_args()
    paths = {
        "selection": Path(args.selection).resolve(),
        "existing_selection": Path(args.existing_selection).resolve(),
        "tasks": Path(args.tasks).resolve(),
        "setup_results": Path(args.setup_results).resolve(),
        "sanity_results": Path(args.sanity_results).resolve(),
        "runnable": Path(args.runnable).resolve(),
    }
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    existing = json.loads(paths["existing_selection"].read_text(encoding="utf-8"))
    tasks = read_jsonl(paths["tasks"])
    setup = read_jsonl(paths["setup_results"])
    sanity = read_jsonl(paths["sanity_results"])
    runnable = read_jsonl(paths["runnable"])

    existing_repos, existing_instances = existing_swe_ids(existing)
    task_ids = [str(task["instance_id"]) for task in tasks]
    runnable_ids = [str(task["instance_id"]) for task in runnable]
    repos = [str(task["benchmark_repo"]) for task in tasks]
    per_repo = Counter(repos)
    mutations = Counter(mutation_kind(instance_id) for instance_id in task_ids)
    mutations_per_repo: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks:
        mutations_per_repo[str(task["benchmark_repo"])][mutation_kind(str(task["instance_id"]))] += 1

    checks = {
        "task_count_100": len(tasks) == 100,
        "runnable_count_100": len(runnable) == 100,
        "unique_instances": len(task_ids) == len(set(task_ids)),
        "runnable_matches_tasks": set(runnable_ids) == set(task_ids),
        "repo_count_20": len(per_repo) == 20,
        "five_tasks_per_repo": set(per_repo.values()) == {5},
        "repository_disjoint": not (set(repos) & existing_repos),
        "instance_disjoint": not (set(task_ids) & existing_instances),
        "setup_all_ready": len(setup) == 20 and all(item["status"] == "ready" for item in setup),
        "gold_all_passed": len(sanity) == 100 and all(item["gold_sanity_passed"] for item in sanity),
        "train_only": all(task["split"] == "train" for task in tasks),
    }
    report = {
        "format": "lottie_swesmith_train_extension_audit_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "counts": {
            "tasks": len(tasks),
            "runnable": len(runnable),
            "repositories": len(per_repo),
            "setup_ready": sum(item["status"] == "ready" for item in setup),
            "gold_sanity_passed": sum(bool(item["gold_sanity_passed"]) for item in sanity),
            "old_repository_overlap": len(set(repos) & existing_repos),
            "old_instance_overlap": len(set(task_ids) & existing_instances),
        },
        "distributions": {
            "mutation_kind": dict(sorted(mutations.items())),
            "changed_lines": distribution([patch_changed_lines(str(task["bug_patch"])) for task in tasks]),
            "fail_to_pass_tests": distribution([len(normalized_list(task["FAIL_TO_PASS"])) for task in tasks]),
            "pass_to_pass_tests": distribution([len(normalized_list(task["PASS_TO_PASS"])) for task in tasks]),
            "problem_statement_chars": distribution([len(str(task["problem_statement"])) for task in tasks]),
            "gold_sanity_elapsed_sec": distribution([float(item["elapsed_sec"]) for item in sanity]),
        },
        "per_repository": {
            repo: {
                "tasks": per_repo[repo],
                "mutation_kinds": dict(sorted(mutations_per_repo[repo].items())),
            }
            for repo in sorted(per_repo)
        },
        "dataset": selection["dataset"],
        "source_files": selection["source_files"],
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "status": report["status"], **report["counts"]}))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
