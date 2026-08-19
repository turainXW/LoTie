#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SELECTION_SEED = "lottie-swesmith-train-extension-v1"
DATASET_NAME = "SWE-bench/SWE-smith-py"
DATASET_REVISION = "77cab9055d42ab4a5c25c89a8f937096db13558e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a repository-disjoint SWE-smith training extension."
    )
    parser.add_argument("--swesmith-parquet", action="append", required=True)
    parser.add_argument(
        "--existing-selection",
        default="data/dataset_splits_v1/selection.json",
        help="Existing train/validation/evaluation split whose repositories and tasks must be excluded.",
    )
    parser.add_argument("--repo-count", type=int, default=20)
    parser.add_argument("--tasks-per-repo", type=int, default=5)
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Select this benchmark repository explicitly; repeat as needed.",
    )
    parser.add_argument(
        "--exclude-repo",
        action="append",
        default=[],
        help="Additional benchmark repository to exclude; repeat as needed.",
    )
    parser.add_argument(
        "--output",
        default="data/swesmith_train_extension_v1/selection.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stable_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{namespace}:{value}".encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_stats(patch: str) -> tuple[int, int]:
    files = patch.count("diff --git ")
    changed = sum(
        1
        for line in patch.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )
    return files, changed


def normalized_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def mutation_kind(instance_id: str) -> str:
    suffix = instance_id.rsplit(".", 1)[-1]
    return suffix.split("__", 1)[0]


def eligible_row(row: dict[str, Any]) -> bool:
    patch = str(row.get("patch") or "")
    files, changed = patch_stats(patch)
    fail_to_pass = normalized_list(row.get("FAIL_TO_PASS"))
    pass_to_pass = normalized_list(row.get("PASS_TO_PASS"))
    statement = str(row.get("problem_statement") or "").strip()
    return (
        files == 1
        and 2 <= changed <= 30
        and 1 <= len(fail_to_pass) <= 20
        and bool(pass_to_pass)
        and len(statement) >= 80
    )


def collect_existing(selection: dict[str, Any]) -> tuple[set[str], set[str]]:
    repos: set[str] = set()
    instances: set[str] = set()
    for split in ("train", "validation", "evaluation"):
        swe = selection["splits"][split]["swesmith_py"]
        repos.update(str(item) for item in swe["profiles"])
        instances.update(str(item) for item in swe["instance_ids"])
    return repos, instances


def repo_profile(benchmark_repo: str) -> dict[str, str]:
    if not benchmark_repo.startswith("swesmith/"):
        raise ValueError(f"Unexpected SWE-smith repository identifier: {benchmark_repo}")
    slug, commit = benchmark_repo.removeprefix("swesmith/").rsplit(".", 1)
    owner, repository = slug.split("__", 1)
    return {
        "source": f"https://github.com/{owner}/{repository}",
        "commit": commit,
        "commit_kind": "dataset_commit_prefix",
    }


def select_repo_tasks(rows: list[dict[str, Any]], repo: str, count: int) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[mutation_kind(str(row["instance_id"]))].append(row)
    for kind, values in grouped.items():
        values.sort(key=lambda row: stable_key(f"task:{repo}:{kind}", str(row["instance_id"])))
    kinds = sorted(grouped, key=lambda kind: stable_key(f"kind:{repo}", kind))
    selected: list[str] = []
    while len(selected) < count and kinds:
        remaining_kinds: list[str] = []
        for kind in kinds:
            if grouped[kind] and len(selected) < count:
                selected.append(str(grouped[kind].pop(0)["instance_id"]))
            if grouped[kind]:
                remaining_kinds.append(kind)
        kinds = remaining_kinds
    if len(selected) != count:
        raise ValueError(f"Not enough eligible tasks for {repo}: {len(selected)}/{count}")
    return selected


def main() -> None:
    args = parse_args()
    if args.repo_count < 1 or args.tasks_per_repo < 1:
        raise ValueError("repo-count and tasks-per-repo must both be positive")
    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing selection: {output}")

    source_paths = [Path(path).resolve() for path in args.swesmith_parquet]
    frames = [pd.read_parquet(path) for path in source_paths]
    frame = pd.concat(frames, ignore_index=True)
    existing_path = Path(args.existing_selection).resolve()
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    excluded_repos, excluded_instances = collect_existing(existing)
    excluded_repos.update(str(item) for item in args.exclude_repo)

    eligible_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame.to_dict(orient="records"):
        repo = str(row.get("repo") or "")
        instance_id = str(row.get("instance_id") or "")
        if repo in excluded_repos or instance_id in excluded_instances or not eligible_row(row):
            continue
        try:
            repo_profile(repo)
        except (ValueError, IndexError):
            continue
        eligible_by_repo[repo].append(row)

    candidate_repos = [
        repo for repo, rows in eligible_by_repo.items() if len(rows) >= args.tasks_per_repo
    ]
    candidate_repos.sort(key=lambda repo: stable_key("repo", repo))
    explicit_repos = [str(repo) for repo in args.repo]
    if explicit_repos and len(explicit_repos) != len(set(explicit_repos)):
        raise ValueError("Explicit --repo values must be unique")
    selected_repos = explicit_repos or candidate_repos[: args.repo_count]
    missing_explicit = [repo for repo in explicit_repos if repo not in candidate_repos]
    if missing_explicit:
        raise ValueError(
            "Explicit repositories are excluded or lack enough eligible tasks: "
            + ", ".join(missing_explicit)
        )
    if not explicit_repos and len(candidate_repos) < args.repo_count:
        raise ValueError(
            f"Not enough repository-disjoint candidates: need {args.repo_count}, "
            f"got {len(candidate_repos)}"
        )

    profiles = {repo: repo_profile(repo) for repo in selected_repos}
    instance_ids: list[str] = []
    per_repo_counts: dict[str, int] = {}
    for repo in selected_repos:
        selected = select_repo_tasks(eligible_by_repo[repo], repo, args.tasks_per_repo)
        instance_ids.extend(selected)
        per_repo_counts[repo] = len(selected)

    expected = len(selected_repos) * args.tasks_per_repo
    assert len(instance_ids) == expected
    assert len(instance_ids) == len(set(instance_ids))
    assert not (set(instance_ids) & excluded_instances)
    assert not (set(profiles) & excluded_repos)

    empty_swe = {"count": 0, "profiles": {}, "instance_ids": []}
    result = {
        "format": "lottie_swesmith_train_extension_v1",
        "selection_seed": SELECTION_SEED,
        "dataset": {"name": DATASET_NAME, "revision": DATASET_REVISION},
        "source_files": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": int(len(frame_part)),
            }
            for path, frame_part in zip(source_paths, frames, strict=True)
        ],
        "existing_selection": {
            "path": str(existing_path),
            "sha256": sha256_file(existing_path),
            "excluded_repo_count": len(excluded_repos),
            "excluded_instance_count": len(excluded_instances),
        },
        "gold_policy": {
            "visible_to_agent": False,
            "swesmith_py": "reverse hidden bug-injection patch; verify FAIL_TO_PASS and PASS_TO_PASS",
        },
        "filters": {
            "swesmith_py": "one file, 2-30 changed lines, 1-20 FAIL_TO_PASS, non-empty PASS_TO_PASS, statement >=80 chars",
            "repo_split": True,
            "repository_disjoint_from_existing_train_validation_evaluation": True,
            "task_selection": "mutation-kind stratified round robin with deterministic SHA-256 ranking",
        },
        "candidate_summary": {
            "dataset_rows": int(len(frame)),
            "eligible_repo_count": len(candidate_repos),
            "selected_repo_count": len(selected_repos),
            "tasks_per_repo": args.tasks_per_repo,
            "repository_selection": "explicit" if explicit_repos else "deterministic_sha256",
        },
        "splits": {
            "train": {
                "role": "trajectory_collection_and_training_extension",
                "count": expected,
                "swesmith_py": {
                    "count": expected,
                    "profiles": profiles,
                    "instance_ids": instance_ids,
                    "per_repo_counts": per_repo_counts,
                },
            },
            "validation": {"role": "unchanged_existing_split", "count": 0, "swesmith_py": empty_swe},
            "evaluation": {"role": "unchanged_existing_split", "count": 0, "swesmith_py": empty_swe},
        },
        "preflight": {
            "status": "selection_only",
            "rule": "Only setup-ready and gold-sanity-passed tasks enter the runnable denominator.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "tasks": expected,
                "repos": len(selected_repos),
                "candidate_repos": len(candidate_repos),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
