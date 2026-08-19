#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


TRAIN_REPOS = {
    "swesmith/Suor__funcy.207a7810": {
        "source": "https://github.com/Suor/funcy",
        "commit": "207a7810c216c7408596d463d3f429686e83b871",
    },
    "swesmith/agronholm__typeguard.b6a7e438": {
        "source": "https://github.com/agronholm/typeguard",
        "commit": "b6a7e4387c30a9f7d635712157c889eb073c1ea3",
    },
    "swesmith/cknd__stackprinter.219fcc52": {
        "source": "https://github.com/cknd/stackprinter",
        "commit": "219fcc522fa5fd6e440703358f6eb408f3ffc007",
    },
    "swesmith/jd__tenacity.0d40e76f": {
        "source": "https://github.com/jd/tenacity",
        "commit": "0d40e76f7d06d631fb127e1ec58c8bd776e70d49",
    },
    "swesmith/madzak__python-json-logger.5f85723f": {
        "source": "https://github.com/madzak/python-json-logger",
        "commit": "5f85723f4693c7289724fdcda84cfc0b62da74d4",
    },
}

VALIDATION_REPOS = {
    "swesmith/pytest-dev__iniconfig.16793ead": {
        "source": "https://github.com/pytest-dev/iniconfig",
        "commit": "16793eaddac67de0b8d621ae4e42e05b927e8d67",
    },
    "swesmith/seatgeek__thefuzz.8a05a3ee": {
        "source": "https://github.com/seatgeek/thefuzz",
        "commit": "8a05a3ee38cbd00a2d2f4bb31db34693b37a1fdd",
    },
    "swesmith/pudo__dataset.5c2dc8d3": {
        "source": "https://github.com/pudo/dataset",
        "commit": "5c2dc8d3af1e0af0290dcd7ae2cae92589f305a1",
    },
}

AMBIGUOUS_MBPP_IDS = {20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe Lottie train/validation/eval task splits.")
    parser.add_argument("--mbpp-rows", action="append", required=True, help="HF dataset-server rows JSON.")
    parser.add_argument("--humaneval-rows", action="append", required=True, help="HF dataset-server rows JSON.")
    parser.add_argument("--swesmith-parquet", action="append", required=True)
    parser.add_argument("--eval-selection", default="data/eval90/selection.json")
    parser.add_argument("--output", default="data/dataset_splits_v1/selection.json")
    return parser.parse_args()


def load_server_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.extend(item.get("row", item) for item in data["rows"])
    return rows


def stable_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"lottie-splits-v1:{namespace}:{value}".encode()).hexdigest()


def select_ids(candidates: list[Any], count: int, *, namespace: str) -> list[Any]:
    ranked = sorted(candidates, key=lambda item: stable_key(namespace, str(item)))
    if len(ranked) < count:
        raise ValueError(f"Not enough candidates for {namespace}: need {count}, got {len(ranked)}")
    return ranked[:count]


def patch_stats(patch: str) -> tuple[int, int]:
    files = patch.count("diff --git ")
    changed = sum(
        1
        for line in patch.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )
    return files, changed


def mutation_kind(instance_id: str) -> str:
    suffix = instance_id.rsplit(".", 1)[-1]
    return suffix.split("__", 1)[0]


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


def eligible_swesmith_rows(frame: pd.DataFrame, repo: str) -> list[dict[str, Any]]:
    rows = []
    for row in frame[frame["repo"] == repo].to_dict(orient="records"):
        patch = str(row.get("patch") or "")
        files, changed = patch_stats(patch)
        fail_to_pass = normalized_list(row.get("FAIL_TO_PASS"))
        pass_to_pass = normalized_list(row.get("PASS_TO_PASS"))
        statement = str(row.get("problem_statement") or "").strip()
        if files != 1 or not 2 <= changed <= 30:
            continue
        if not 1 <= len(fail_to_pass) <= 20 or not pass_to_pass:
            continue
        if len(statement) < 80:
            continue
        rows.append(row)
    return rows


def select_stratified_swesmith(frame: pd.DataFrame, repos: dict[str, Any], per_repo: int, role: str) -> list[str]:
    selected: list[str] = []
    for repo in repos:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eligible_swesmith_rows(frame, repo):
            grouped[mutation_kind(str(row["instance_id"]))].append(row)
        for kind, rows in grouped.items():
            rows.sort(key=lambda row: stable_key(f"{role}:{repo}:{kind}", str(row["instance_id"])))
        kinds = sorted(grouped, key=lambda kind: stable_key(f"{role}:{repo}:kind", kind))
        repo_selected: list[str] = []
        while len(repo_selected) < per_repo and kinds:
            next_kinds = []
            for kind in kinds:
                if grouped[kind] and len(repo_selected) < per_repo:
                    repo_selected.append(str(grouped[kind].pop(0)["instance_id"]))
                if grouped[kind]:
                    next_kinds.append(kind)
            kinds = next_kinds
        if len(repo_selected) != per_repo:
            raise ValueError(f"Not enough eligible SWE-smith tasks for {repo}: {len(repo_selected)}/{per_repo}")
        selected.extend(repo_selected)
    return selected


def main() -> None:
    args = parse_args()
    eval_selection = json.loads(Path(args.eval_selection).read_text(encoding="utf-8"))
    mbpp_rows = load_server_rows(args.mbpp_rows)
    humaneval_rows = load_server_rows(args.humaneval_rows)
    swe_frames = [pd.read_parquet(path) for path in args.swesmith_parquet]
    swe_frame = pd.concat(swe_frames, ignore_index=True)

    eval_mbpp = set(eval_selection["sets"]["mbppplus"]["task_ids"])
    eval_humaneval = set(eval_selection["sets"]["humanevalplus"]["task_ids"])
    eval_swesmith = set(eval_selection["sets"]["swesmith_py"]["instance_ids"])
    eval_repos = set(eval_selection["sets"]["swesmith_py"]["profiles"])

    mbpp_candidates = [
        row["task_id"]
        for row in mbpp_rows
        if row["task_id"] not in eval_mbpp
        and row["task_id"] not in AMBIGUOUS_MBPP_IDS
        and str(row.get("prompt") or "").strip()
        and str(row.get("code") or "").strip()
        and str(row.get("test") or "").strip()
    ]
    he_candidates = [
        row["task_id"]
        for row in humaneval_rows
        if row["task_id"] not in eval_humaneval
        and str(row.get("prompt") or "").strip()
        and str(row.get("canonical_solution") or "").strip()
        and str(row.get("test") or "").strip()
    ]

    val_mbpp = select_ids(mbpp_candidates, 15, namespace="validation:mbpp")
    train_mbpp = select_ids(
        [item for item in mbpp_candidates if item not in set(val_mbpp)],
        120,
        namespace="train:mbpp",
    )
    val_he = select_ids(he_candidates, 15, namespace="validation:humaneval")
    train_he = select_ids(
        [item for item in he_candidates if item not in set(val_he)],
        80,
        namespace="train:humaneval",
    )
    train_swe = select_stratified_swesmith(swe_frame, TRAIN_REPOS, 20, "train")
    val_swe = select_stratified_swesmith(swe_frame, VALIDATION_REPOS, 5, "validation")

    train_repos = set(TRAIN_REPOS)
    validation_repos = set(VALIDATION_REPOS)
    assert not (train_repos & validation_repos or train_repos & eval_repos or validation_repos & eval_repos)
    assert not (set(train_mbpp) & set(val_mbpp) or set(train_mbpp) & eval_mbpp or set(val_mbpp) & eval_mbpp)
    assert not (set(train_he) & set(val_he) or set(train_he) & eval_humaneval or set(val_he) & eval_humaneval)
    assert not (set(train_swe) & set(val_swe) or set(train_swe) & eval_swesmith or set(val_swe) & eval_swesmith)

    result = {
        "format": "lottie_dataset_splits_v1",
        "selection_seed": "lottie-splits-v1",
        "gold_policy": {
            "visible_to_agent": False,
            "mbppplus": "dataset code plus hidden EvalPlus tests",
            "humanevalplus": "canonical_solution plus hidden EvalPlus tests",
            "swesmith_py": "reverse of hidden bug-injection patch; scored by FAIL_TO_PASS and PASS_TO_PASS",
        },
        "filters": {
            "function_tasks": "non-empty prompt, reference, and tests; exclude evaluation IDs",
            "swesmith_py": "one file, 2-30 changed lines, 1-20 FAIL_TO_PASS, non-empty PASS_TO_PASS, statement >=80 chars",
            "repo_split": True,
        },
        "splits": {
            "train": {
                "role": "trajectory_collection_and_training",
                "count": 300,
                "mbppplus": {"count": 120, "task_ids": train_mbpp},
                "humanevalplus": {"count": 80, "task_ids": train_he},
                "swesmith_py": {"count": 100, "profiles": TRAIN_REPOS, "instance_ids": train_swe},
            },
            "validation": {
                "role": "prompt_tool_context_tuning_only",
                "count": 45,
                "mbppplus": {"count": 15, "task_ids": val_mbpp},
                "humanevalplus": {"count": 15, "task_ids": val_he},
                "swesmith_py": {"count": 15, "profiles": VALIDATION_REPOS, "instance_ids": val_swe},
            },
            "evaluation": {
                "role": "final_report_only",
                "count": 90,
                "selection_file": str(Path(args.eval_selection)),
                "mbppplus": {"count": 30, "task_ids": sorted(eval_mbpp)},
                "humanevalplus": {"count": 30, "task_ids": sorted(eval_humaneval)},
                "swesmith_py": {
                    "count": 30,
                    "profiles": eval_selection["sets"]["swesmith_py"]["profiles"],
                    "instance_ids": eval_selection["sets"]["swesmith_py"]["instance_ids"],
                },
            },
        },
        "rollout_plan": {
            "train_rollouts_per_task": 5,
            "expected_raw_training_trajectories": 1500,
            "validation_rollouts_per_task": 3,
            "evaluation_rollouts_per_task": 3,
        },
        "preflight": {
            "status": "selection_only",
            "rule": "Only gold-sanity-passed tasks enter the runnable denominator.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "train": 300, "validation": 45, "evaluation": 90}, ensure_ascii=False))


if __name__ == "__main__":
    main()
