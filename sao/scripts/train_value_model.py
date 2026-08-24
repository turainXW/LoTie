#!/usr/bin/env python3
"""Train and evaluate the terminal-return value model used by Lottie SAO.

The same entry point supports selection-only planning, interface smoke tests,
full-dataset training, periodic exact-resume checkpoints, and the final
family-specific calibration gate.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, get_constant_schedule_with_warmup

from sao_value.critic_bucketing import (
    CRITIC_BATCH_PROFILES,
    CRITIC_BUCKET_BOUNDARIES,
    CriticLengthBucketPlanner,
    CriticUpdatePlan,
    make_critic_bucket_profile,
)


FAMILIES = ("function", "swesmith")
MLP_LORA_MODULES = ("gate_proj", "up_proj", "down_proj")
ATTENTION_LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-data-dir",
        type=Path,
        help="Read validation rows/tokens from another leakage-safe Critic dataset.",
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--actor-adapter", type=Path, required=True)
    parser.add_argument(
        "--resume-critic-checkpoint",
        type=Path,
        help="Load critic_adapter and value_head.safetensors from a prior stage.",
    )
    parser.add_argument(
        "--resume-run-checkpoint",
        type=Path,
        help="Exactly resume model, optimizer, scheduler, RNG, and update position.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-cell", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-len", type=int, default=32768)
    parser.add_argument("--critic-lr", type=float, default=5e-6)
    parser.add_argument("--value-head-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="Write an exact-resume checkpoint every N optimizer updates; 0 disables it.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--train-selection",
        choices=("smoke", "all"),
        default="smoke",
        help="Use balanced smoke cells or every eligible training trajectory.",
    )
    parser.add_argument(
        "--batch-profile",
        choices=sorted(CRITIC_BATCH_PROFILES),
        default="critic_96gb",
    )
    parser.add_argument("--target-trajectories-per-update", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--interface-only", action="store_true")
    parser.add_argument(
        "--interface-bucket-max-tokens",
        type=int,
        choices=CRITIC_BUCKET_BOUNDARIES,
        help="For --interface-only, test the longest configured microbatch in this bucket.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(seed: int, row: dict[str, Any]) -> str:
    value = (
        f"{seed}|{row['dataset_family']}|{row['instance_id']}|"
        f"{row['sample_index']}|{row['num_tokens']}"
    )
    return hashlib.sha256(value.encode()).hexdigest()


def choose_even_lengths(
    rows: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["num_tokens"], stable_hash(seed, row)))
    if len(ordered) < count:
        raise ValueError(f"Need {count} rows but only found {len(ordered)}")
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indices]


def choose_smoke_rows(
    rows: list[dict[str, Any]], samples_per_cell: int, max_train_len: int, seed: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for family in FAMILIES:
        for reward in (0, 1):
            candidates = [
                row
                for row in rows
                if row["split"] == "train"
                and row["dataset_family"] == family
                and int(row["reward"]) == reward
                and int(row["num_tokens"]) <= max_train_len
            ]
            selected.extend(choose_even_lengths(candidates, samples_per_cell, seed))
    return selected


def choose_training_rows(
    rows: list[dict[str, Any]],
    *,
    selection: str,
    samples_per_cell: int,
    max_train_len: int,
    seed: int,
) -> list[dict[str, Any]]:
    if selection == "smoke":
        return choose_smoke_rows(rows, samples_per_cell, max_train_len, seed)
    selected = [
        row
        for row in rows
        if row["split"] == "train" and int(row["num_tokens"]) <= max_train_len
    ]
    if not selected:
        raise ValueError("No eligible training trajectories were found")
    return selected


def serialize_update_plans(
    plans: Iterable[CriticUpdatePlan], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    serialized = []
    for update_index, plan in enumerate(plans, 1):
        serialized.append(
            {
                "update": update_index,
                "bucket_indexes": plan.bucket_indexes,
                "bucket_label": plan.bucket_label,
                "trajectory_count": plan.trajectory_count,
                "microbatch_sizes": [len(batch.row_indexes) for batch in plan.microbatches],
                "microbatches": [
                    {
                        "bucket_index": microbatch.bucket_index,
                        "bucket_label": microbatch.bucket_label,
                        "rows": [
                            {
                                "row_index": row_index,
                                "instance_id": rows[row_index]["instance_id"],
                                "dataset_family": rows[row_index]["dataset_family"],
                                "reward": int(rows[row_index]["reward"]),
                                "num_tokens": int(rows[row_index]["num_tokens"]),
                            }
                            for row_index in microbatch.row_indexes
                        ],
                    }
                    for microbatch in plan.microbatches
                ],
            }
        )
    return serialized


def summarize_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for family in FAMILIES:
        for reward in (0, 1):
            cell = [
                row
                for row in rows
                if row["dataset_family"] == family and int(row["reward"]) == reward
            ]
            key = f"{family}:reward_{reward}"
            cells[key] = {
                "rows": len(cell),
                "min_tokens": min((row["num_tokens"] for row in cell), default=0),
                "max_tokens": max((row["num_tokens"] for row in cell), default=0),
                "instance_ids": [row["instance_id"] for row in cell],
            }
    return {"rows": len(rows), "cells": cells}


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    cursor = 0
    while cursor < len(scores):
        end = cursor + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[cursor]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(np.float64)
    predictions = predictions.astype(np.float64)
    clipped = np.clip(predictions, 0.0, 1.0)
    residual = labels - predictions
    variance = float(np.var(labels))
    return {
        "rows_or_tokens": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "prediction_mean": float(predictions.mean()),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "mse_raw": float(np.mean(residual**2)),
        "brier_clipped": float(np.mean((labels - clipped) ** 2)),
        "auc": rank_auc(labels, predictions),
        "explained_variance": (
            float(1.0 - np.var(residual) / variance) if variance > 0 else None
        ),
        "positive_prediction_mean": (
            float(predictions[labels == 1].mean()) if np.any(labels == 1) else None
        ),
        "negative_prediction_mean": (
            float(predictions[labels == 0].mean()) if np.any(labels == 0) else None
        ),
    }


def family_train_priors(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for family in FAMILIES:
        family_rows = [
            row for row in rows if row["split"] == "train" and row["dataset_family"] == family
        ]
        result[family] = float(np.mean([row["reward"] for row in family_rows]))
    return result


def constant_baseline_metrics(
    validation_rows: list[dict[str, Any]], priors: dict[str, float]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in FAMILIES:
        family_rows = [row for row in validation_rows if row["dataset_family"] == family]
        labels = np.asarray([row["reward"] for row in family_rows], dtype=np.float64)
        predictions = np.full_like(labels, priors[family])
        result[family] = binary_metrics(labels, predictions)
    return result


class TokenStore:
    def __init__(self, data_dir: Path):
        self.input_ids = np.load(data_dir / "input_ids.npy", mmap_mode="r")
        self.action_mask = np.load(data_dir / "action_mask.npy", mmap_mode="r")

    def get(self, row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(row["offset"])
        end = start + int(row["num_tokens"])
        input_ids = torch.from_numpy(np.asarray(self.input_ids[start:end], dtype=np.int64))
        action_mask = torch.from_numpy(
            np.asarray(self.action_mask[start:end], dtype=np.bool_)
        )
        return input_ids, action_mask

    def collate(
        self, rows: list[dict[str, Any]], *, pad_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not rows:
            raise ValueError("Cannot collate an empty critic microbatch")
        samples = [self.get(row) for row in rows]
        max_length = max(input_ids.numel() for input_ids, _ in samples)
        input_ids = torch.full(
            (len(rows), max_length), int(pad_token_id), dtype=torch.long
        )
        action_mask = torch.zeros((len(rows), max_length), dtype=torch.bool)
        attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
        for row_index, (row_input_ids, row_action_mask) in enumerate(samples):
            length = row_input_ids.numel()
            input_ids[row_index, :length] = row_input_ids
            action_mask[row_index, :length] = row_action_mask
            attention_mask[row_index, :length] = 1
        return input_ids, action_mask, attention_mask


class CriticModel(nn.Module):
    def __init__(self, peft_model: PeftModel, hidden_size: int, initial_bias: float):
        super().__init__()
        self.peft_model = peft_model
        self.backbone = peft_model.get_base_model().model
        self.value_head = nn.Linear(hidden_size, 1, dtype=torch.float32)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.value_head.bias, initial_bias)

    def forward_trajectory_values(
        self,
        input_ids: torch.Tensor,
        action_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        if input_ids.ndim != 2 or action_mask.shape != input_ids.shape:
            raise ValueError("Critic inputs and action masks must be [batch, sequence]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("Critic attention mask must match input IDs")
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        # Match verl's causal value alignment: the state immediately before an
        # action token predicts that token's value. This prevents the critic
        # from seeing the action it is meant to evaluate.
        result = []
        for row_index in range(input_ids.shape[0]):
            action_indices = torch.nonzero(
                action_mask[row_index], as_tuple=False
            ).squeeze(-1)
            value_indices = action_indices[action_indices > 0] - 1
            if value_indices.numel() == 0:
                raise RuntimeError("Trajectory has no causally aligned action tokens")
            hidden = output.last_hidden_state[row_index, value_indices]
            result.append(self.value_head(hidden.float()).squeeze(-1))
        return result

    def forward_values(
        self, input_ids: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        values = self.forward_trajectory_values(
            input_ids.unsqueeze(0),
            action_mask.unsqueeze(0),
            torch.ones_like(input_ids, dtype=torch.long).unsqueeze(0),
        )
        return values[0]


def configure_trainable_parameters(model: CriticModel) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.value_head.parameters():
        parameter.requires_grad = True

    trainable_mlp_lora = []
    frozen_attention_lora = []
    other_lora = []
    for name, parameter in model.peft_model.named_parameters():
        if "lora_" not in name:
            continue
        if any(f".{module}." in name for module in MLP_LORA_MODULES):
            parameter.requires_grad = True
            trainable_mlp_lora.append(name)
        elif any(f".{module}." in name for module in ATTENTION_LORA_MODULES):
            parameter.requires_grad = False
            frozen_attention_lora.append(name)
        else:
            other_lora.append(name)

    if not trainable_mlp_lora:
        raise RuntimeError("No trainable MLP LoRA parameters were found")
    if not frozen_attention_lora:
        raise RuntimeError("No frozen attention LoRA parameters were found")
    if any(parameter.requires_grad for name, parameter in model.peft_model.named_parameters() if any(f".{module}." in name for module in ATTENTION_LORA_MODULES)):
        raise RuntimeError("Attention LoRA is unexpectedly trainable")

    counts = {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "trainable_mlp_lora_tensors": len(trainable_mlp_lora),
        "frozen_attention_lora_tensors": len(frozen_attention_lora),
        "unclassified_lora_tensors": other_lora,
        "attention_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in model.peft_model.named_parameters()
            if parameter.requires_grad
            and any(f".{module}." in name for module in ATTENTION_LORA_MODULES)
        ),
    }
    return counts


def split_optimizer_parameters(model: CriticModel) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    critic_parameters = [
        parameter
        for name, parameter in model.peft_model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    value_parameters = [parameter for parameter in model.value_head.parameters() if parameter.requires_grad]
    return critic_parameters, value_parameters


@contextlib.contextmanager
def evaluation_mode(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def evaluate(
    model: CriticModel,
    store: TokenStore,
    rows: list[dict[str, Any]],
    device: torch.device,
    phase: str,
    progress_path: Path,
) -> dict[str, Any]:
    token_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    token_labels: dict[str, list[np.ndarray]] = defaultdict(list)
    trajectory_predictions: dict[str, list[float]] = defaultdict(list)
    final_predictions: dict[str, list[float]] = defaultdict(list)
    trajectory_labels: dict[str, list[float]] = defaultdict(list)
    per_row: list[dict[str, Any]] = []

    with evaluation_mode(model), torch.inference_mode():
        phase_started = time.time()
        for row_index, row in enumerate(rows, start=1):
            input_ids, action_mask = store.get(row)
            input_ids = input_ids.to(device, non_blocking=True)
            action_mask = action_mask.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                values = model.forward_values(input_ids, action_mask)
            values_np = values.float().cpu().numpy()
            label = float(row["reward"])
            family = row["dataset_family"]
            token_predictions[family].append(values_np)
            token_labels[family].append(np.full_like(values_np, label, dtype=np.float64))
            trajectory_predictions[family].append(float(values_np.mean()))
            final_predictions[family].append(float(values_np[-1]))
            trajectory_labels[family].append(label)
            per_row.append(
                {
                    "dataset_family": family,
                    "instance_id": row["instance_id"],
                    "reward": label,
                    "num_tokens": row["num_tokens"],
                    "num_action_tokens": row["num_action_tokens"],
                    "mean_value": float(values_np.mean()),
                    "final_value": float(values_np[-1]),
                }
            )
            progress = {
                "phase": phase,
                "row": row_index,
                "total_rows": len(rows),
                "dataset_family": family,
                "instance_id": row["instance_id"],
                "num_tokens": row["num_tokens"],
                "elapsed_sec": time.time() - phase_started,
                "gpu_memory_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                "gpu_memory_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
            }
            with progress_path.open("a") as handle:
                handle.write(json.dumps(progress, sort_keys=True) + "\n")
            if row_index == 1 or row_index % 5 == 0 or row_index == len(rows):
                print(
                    f"[{phase}] {row_index}/{len(rows)} "
                    f"{family} tokens={row['num_tokens']}",
                    flush=True,
                )
            del input_ids, action_mask, values

    metrics: dict[str, Any] = {}
    for family in FAMILIES:
        labels = np.asarray(trajectory_labels[family], dtype=np.float64)
        metrics[family] = {
            "trajectory_mean": binary_metrics(
                labels, np.asarray(trajectory_predictions[family], dtype=np.float64)
            ),
            "trajectory_final_token": binary_metrics(
                labels, np.asarray(final_predictions[family], dtype=np.float64)
            ),
            "token_weighted": binary_metrics(
                np.concatenate(token_labels[family]),
                np.concatenate(token_predictions[family]),
            ),
        }
    return {"metrics": metrics, "per_row": per_row}


def gate_result(
    after: dict[str, Any], constant_baseline: dict[str, Any]
) -> dict[str, Any]:
    family_results: dict[str, Any] = {}
    for family in FAMILIES:
        observed = after[family]["trajectory_mean"]
        baseline = constant_baseline[family]
        checks = {
            "brier_better_than_constant": observed["brier_clipped"] < baseline["brier_clipped"],
            "auc_above_random": observed["auc"] is not None and observed["auc"] > 0.5,
            "explained_variance_positive": observed["explained_variance"] is not None
            and observed["explained_variance"] > 0.0,
        }
        family_results[family] = {"passed": all(checks.values()), "checks": checks}
    return {
        "passed": all(result["passed"] for result in family_results.values()),
        "families": family_results,
        "policy": "all families must beat their train-prior constant baseline",
    }


def save_rng_state(path: Path) -> None:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }
    torch.save(state, path)


def load_rng_state(path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    checkpoint_dir: Path,
    *,
    peft_model: PeftModel,
    model: CriticModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    next_epoch: int,
    next_update_index: int,
    args: argparse.Namespace,
    source_checkpoint: Path,
    data_manifest_sha256: str,
) -> None:
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = checkpoint_dir.with_name(f".{checkpoint_dir.name}.tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir()
    peft_model.save_pretrained(
        temporary_dir / "critic_adapter", safe_serialization=True
    )
    save_file(
        {
            "value_head.weight": model.value_head.weight.detach().cpu(),
            "value_head.bias": model.value_head.bias.detach().cpu(),
        },
        str(temporary_dir / "value_head.safetensors"),
    )
    torch.save(optimizer.state_dict(), temporary_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), temporary_dir / "scheduler.pt")
    save_rng_state(temporary_dir / "rng_state.pt")
    trainer_state = {
        "format": "lottie_sao_critic_trainer_state_v1",
        "global_step": global_step,
        "next_epoch": next_epoch,
        "next_update_index": next_update_index,
        "total_epochs": args.epochs,
        "train_selection": args.train_selection,
        "batch_profile": args.batch_profile,
        "target_trajectories_per_update": args.target_trajectories_per_update,
        "data_manifest_sha256": data_manifest_sha256,
    }
    (temporary_dir / "trainer_state.json").write_text(
        json.dumps(trainer_state, indent=2, sort_keys=True) + "\n"
    )
    (temporary_dir / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "format": "lottie_sao_critic_checkpoint_v2",
                "global_steps": global_step,
                "epochs": args.epochs,
                "train_selection": args.train_selection,
                "batch_profile": args.batch_profile,
                "target_trajectories_per_update": args.target_trajectories_per_update,
                "source_checkpoint": str(source_checkpoint),
                "files": {
                    "critic_adapter": "critic_adapter",
                    "value_head": "value_head.safetensors",
                    "optimizer": "optimizer.pt",
                    "scheduler": "scheduler.pt",
                    "rng": "rng_state.pt",
                    "trainer_state": "trainer_state.json",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    os.replace(temporary_dir, checkpoint_dir)
    latest_path = checkpoint_dir.parent / "latest_checkpoint.json"
    latest_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_dir.resolve()),
                "global_step": global_step,
                "next_epoch": next_epoch,
                "next_update_index": next_update_index,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = read_jsonl(args.data_dir / "samples.jsonl")
    validation_data_dir = args.validation_data_dir or args.data_dir
    validation_source_rows = (
        rows
        if validation_data_dir.resolve() == args.data_dir.resolve()
        else read_jsonl(validation_data_dir / "samples.jsonl")
    )
    selected = choose_training_rows(
        rows,
        selection=args.train_selection,
        samples_per_cell=args.samples_per_cell,
        max_train_len=args.max_train_len,
        seed=args.seed,
    )
    validation_rows = [
        row for row in validation_source_rows if row["split"] == "validation"
    ]
    if not validation_rows:
        raise ValueError("No validation rows were found")
    priors = family_train_priors(rows)
    bucket_profile = make_critic_bucket_profile(
        args.batch_profile,
        target_trajectories_per_update=args.target_trajectories_per_update,
    )
    bucket_planner = CriticLengthBucketPlanner(
        [int(row["num_tokens"]) for row in selected],
        profile=bucket_profile,
        seed=args.seed,
    )
    first_epoch_plan = bucket_planner.plan_epoch(1)
    bucket_summary = bucket_planner.summary(first_epoch_plan)
    selection_manifest = {
        "format": "lottie_sao_critic_smoke_selection_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "data_manifest_sha256": sha256_file(args.data_dir / "dataset_manifest.json"),
        "validation_data_dir": str(validation_data_dir.resolve()),
        "validation_data_manifest_sha256": sha256_file(
            validation_data_dir / "dataset_manifest.json"
        ),
        "selection": summarize_selection(selected),
        "train_selection": args.train_selection,
        "bucket_plan": bucket_summary,
        "validation_rows": len(validation_rows),
        "family_train_priors": priors,
        "constant_baseline": constant_baseline_metrics(validation_rows, priors),
        "config": vars(args) | {
            "data_dir": str(args.data_dir),
            "base_model": str(args.base_model),
            "actor_adapter": str(args.actor_adapter),
            "resume_critic_checkpoint": (
                str(args.resume_critic_checkpoint)
                if args.resume_critic_checkpoint
                else None
            ),
            "resume_run_checkpoint": (
                str(args.resume_run_checkpoint) if args.resume_run_checkpoint else None
            ),
            "output_dir": str(args.output_dir),
            "validation_data_dir": str(validation_data_dir),
        },
    }
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n"
    )
    write_jsonl(args.output_dir / "selected_train_rows.jsonl", selected)
    write_jsonl(
        args.output_dir / "bucket_updates_epoch_001.jsonl",
        serialize_update_plans(first_epoch_plan, selected),
    )
    if args.selection_only:
        print(json.dumps(selection_manifest, indent=2, sort_keys=True))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the critic smoke test")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    if args.resume_run_checkpoint:
        adapter_path = args.resume_run_checkpoint / "critic_adapter"
    elif args.resume_critic_checkpoint:
        adapter_path = args.resume_critic_checkpoint / "critic_adapter"
    else:
        adapter_path = args.actor_adapter
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=True,
    )
    peft_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    peft_model.enable_input_require_grads()
    hidden_size = int(base_model.config.hidden_size)
    pad_token_id = base_model.config.pad_token_id
    if pad_token_id is None:
        pad_token_id = base_model.config.eos_token_id
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0]
    if pad_token_id is None:
        raise ValueError("Model config has neither pad_token_id nor eos_token_id")
    overall_train_prior = float(
        np.mean([row["reward"] for row in rows if row["split"] == "train"])
    )
    model = CriticModel(peft_model, hidden_size, overall_train_prior).to(device)
    value_checkpoint = args.resume_run_checkpoint or args.resume_critic_checkpoint
    if value_checkpoint:
        value_state = load_file(
            str(value_checkpoint / "value_head.safetensors")
        )
        model.value_head.load_state_dict(
            {
                "weight": value_state["value_head.weight"],
                "bias": value_state["value_head.bias"],
            }
        )
    parameter_audit = configure_trainable_parameters(model)
    (args.output_dir / "parameter_audit.json").write_text(
        json.dumps(parameter_audit, indent=2, sort_keys=True) + "\n"
    )
    print(f"[model] loaded; parameter audit: {parameter_audit}", flush=True)

    critic_parameters, value_parameters = split_optimizer_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": critic_parameters, "lr": args.critic_lr, "weight_decay": 0.0},
            {"params": value_parameters, "lr": args.value_head_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    total_steps = bucket_planner.updates_per_epoch() * args.epochs
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, max(1, total_steps - 1)),
    )
    resume_state: dict[str, Any] | None = None
    if args.resume_run_checkpoint:
        resume_state = json.loads(
            (args.resume_run_checkpoint / "trainer_state.json").read_text()
        )
        expected_manifest_sha256 = selection_manifest["data_manifest_sha256"]
        checks = {
            "total_epochs": args.epochs,
            "train_selection": args.train_selection,
            "batch_profile": args.batch_profile,
            "target_trajectories_per_update": args.target_trajectories_per_update,
            "data_manifest_sha256": expected_manifest_sha256,
        }
        for key, expected in checks.items():
            if resume_state.get(key) != expected:
                raise ValueError(
                    f"Resume checkpoint mismatch for {key}: "
                    f"{resume_state.get(key)!r} != {expected!r}"
                )
        optimizer.load_state_dict(
            torch.load(
                args.resume_run_checkpoint / "optimizer.pt",
                map_location="cpu",
                weights_only=False,
            )
        )
        scheduler.load_state_dict(
            torch.load(
                args.resume_run_checkpoint / "scheduler.pt",
                map_location="cpu",
                weights_only=False,
            )
        )
        load_rng_state(args.resume_run_checkpoint / "rng_state.pt")
    store = TokenStore(args.data_dir)
    validation_store = (
        store
        if validation_data_dir.resolve() == args.data_dir.resolve()
        else TokenStore(validation_data_dir)
    )

    if args.interface_only:
        if args.interface_bucket_max_tokens is not None:
            bucket_index = bucket_profile.boundaries.index(
                args.interface_bucket_max_tokens
            )
            candidate_indexes = [
                index
                for index, row in enumerate(selected)
                if bucket_profile.bucket_index(int(row["num_tokens"])) == bucket_index
            ]
            candidate_indexes.sort(
                key=lambda index: int(selected[index]["num_tokens"]), reverse=True
            )
            interface_indexes = candidate_indexes[
                : bucket_profile.microbatch_sizes[bucket_index]
            ]
            if not interface_indexes:
                raise ValueError(
                    f"No rows found for interface bucket {args.interface_bucket_max_tokens}"
                )
            interface_bucket_index = bucket_index
            interface_bucket_label = bucket_profile.bucket_label(bucket_index)
        else:
            microbatch_plan = max(
                (
                    microbatch
                    for update in first_epoch_plan
                    for microbatch in update.microbatches
                ),
                key=lambda microbatch: len(microbatch.row_indexes),
            )
            interface_indexes = list(microbatch_plan.row_indexes)
            interface_bucket_index = microbatch_plan.bucket_index
            interface_bucket_label = microbatch_plan.bucket_label
        interface_rows = [selected[index] for index in interface_indexes]
        input_ids, action_mask, attention_mask = store.collate(
            interface_rows, pad_token_id=int(pad_token_id)
        )
        real_tokens = int(attention_mask.sum().item())
        padded_tokens = int(attention_mask.numel())
        input_ids = input_ids.to(device, non_blocking=True)
        action_mask = action_mask.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        targets = torch.tensor(
            [float(row["reward"]) for row in interface_rows],
            device=device,
            dtype=torch.float32,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            trajectory_values = model.forward_trajectory_values(
                input_ids, action_mask, attention_mask
            )
            trajectory_losses = torch.stack(
                [
                    torch.mean((values - target) ** 2)
                    for values, target in zip(
                        trajectory_values, targets, strict=True
                    )
                ]
            )
            loss = trajectory_losses.mean()
        loss.backward()
        interface_result = {
            "completed": True,
            "bucket_index": interface_bucket_index,
            "bucket_label": interface_bucket_label,
            "microbatch_size": len(interface_rows),
            "instance_ids": [row["instance_id"] for row in interface_rows],
            "num_tokens": [int(row["num_tokens"]) for row in interface_rows],
            "num_aligned_action_values": [
                int(values.numel()) for values in trajectory_values
            ],
            "real_tokens": real_tokens,
            "padded_tokens": padded_tokens,
            "padding_fraction": 1.0 - real_tokens / padded_tokens,
            "loss": float(loss.detach().cpu()),
            "mlp_lora_nonzero_gradient": any(
                parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                for parameter in critic_parameters
            ),
            "value_head_nonzero_gradient": any(
                parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                for parameter in value_parameters
            ),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "optimizer_step_executed": False,
        }
        (args.output_dir / "interface_result.json").write_text(
            json.dumps(interface_result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(interface_result, indent=2, sort_keys=True), flush=True)
        return

    progress_path = args.output_dir / "eval_progress.jsonl"
    if args.resume_run_checkpoint and (args.output_dir / "before_metrics.json").is_file():
        before = {
            "metrics": json.loads((args.output_dir / "before_metrics.json").read_text()),
            "per_row": read_jsonl(args.output_dir / "before_predictions.jsonl"),
        }
    else:
        before = evaluate(
            model, validation_store, validation_rows, device, "before", progress_path
        )
        (args.output_dir / "before_metrics.json").write_text(
            json.dumps(before["metrics"], indent=2, sort_keys=True) + "\n"
        )
        write_jsonl(args.output_dir / "before_predictions.jsonl", before["per_row"])

    global_step = int(resume_state["global_step"]) if resume_state else 0
    resume_epoch = int(resume_state["next_epoch"]) if resume_state else 1
    resume_update_index = (
        int(resume_state["next_update_index"]) if resume_state else 0
    )
    nonzero_mlp_lora_gradients_seen = False
    nonzero_value_head_gradients_seen = False
    model.train()
    for epoch in range(1, args.epochs + 1):
        if epoch < resume_epoch:
            continue
        update_plans = first_epoch_plan if epoch == 1 else bucket_planner.plan_epoch(epoch)
        if len(update_plans) != bucket_planner.updates_per_epoch():
            raise AssertionError("Bucket plan update count changed across epochs")
        for update_index, update_plan in enumerate(update_plans):
            if epoch == resume_epoch and update_index < resume_update_index:
                continue
            optimizer.zero_grad(set_to_none=True)
            trajectory_loss_values: list[float] = []
            update_rows: list[dict[str, Any]] = []
            padded_tokens = 0
            real_tokens = 0
            action_tokens = 0
            for microbatch_plan in update_plan.microbatches:
                microbatch_rows = [
                    selected[index] for index in microbatch_plan.row_indexes
                ]
                update_rows.extend(microbatch_rows)
                input_ids, action_mask, attention_mask = store.collate(
                    microbatch_rows, pad_token_id=int(pad_token_id)
                )
                real_tokens += int(attention_mask.sum().item())
                padded_tokens += int(attention_mask.numel())
                action_tokens += int(action_mask.sum().item())
                input_ids = input_ids.to(device, non_blocking=True)
                action_mask = action_mask.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                targets = torch.tensor(
                    [float(row["reward"]) for row in microbatch_rows],
                    device=device,
                    dtype=torch.float32,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    trajectory_values = model.forward_trajectory_values(
                        input_ids, action_mask, attention_mask
                    )
                    trajectory_losses = torch.stack(
                        [
                            torch.mean((values - target) ** 2)
                            for values, target in zip(
                                trajectory_values, targets, strict=True
                            )
                        ]
                    )
                    scaled_loss = (
                        trajectory_losses.sum() / update_plan.trajectory_count
                    )
                if not torch.isfinite(scaled_loss):
                    raise FloatingPointError(
                        f"Non-finite loss at update {global_step + 1}"
                    )
                scaled_loss.backward()
                trajectory_loss_values.extend(
                    float(value) for value in trajectory_losses.detach().cpu()
                )
                del (
                    input_ids,
                    action_mask,
                    attention_mask,
                    targets,
                    trajectory_values,
                    trajectory_losses,
                    scaled_loss,
                )
            nonzero_mlp_lora_gradients_seen |= any(
                parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                for parameter in critic_parameters
            )
            nonzero_value_head_gradients_seen |= any(
                parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                for parameter in value_parameters
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            update_loss = float(np.mean(trajectory_loss_values))
            family_counts: dict[str, int] = defaultdict(int)
            for row in update_rows:
                family_counts[row["dataset_family"]] += 1
            row_log = {
                "epoch": epoch,
                "step": global_step,
                "bucket_indexes": update_plan.bucket_indexes,
                "bucket_label": update_plan.bucket_label,
                "microbatch_sizes": [
                    len(batch.row_indexes) for batch in update_plan.microbatches
                ],
                "trajectories": update_plan.trajectory_count,
                "dataset_families": dict(sorted(family_counts.items())),
                "rewards": {
                    str(reward): sum(int(row["reward"]) == reward for row in update_rows)
                    for reward in (0, 1)
                },
                "real_tokens": real_tokens,
                "padded_tokens": padded_tokens,
                "action_tokens": action_tokens,
                "padding_fraction": 1.0 - real_tokens / padded_tokens,
                "trajectory_mean_loss": update_loss,
                "grad_norm": float(grad_norm.detach().cpu()),
                "critic_lr": float(optimizer.param_groups[0]["lr"]),
                "value_head_lr": float(optimizer.param_groups[1]["lr"]),
                "gpu_memory_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                "gpu_memory_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
            }
            with (args.output_dir / "train_log.jsonl").open("a") as handle:
                handle.write(json.dumps(row_log, sort_keys=True) + "\n")
            if update_index + 1 == len(update_plans):
                next_epoch = epoch + 1
                next_update_index = 0
            else:
                next_epoch = epoch
                next_update_index = update_index + 1
            if args.save_every_steps and global_step % args.save_every_steps == 0:
                checkpoint_dir = (
                    args.output_dir
                    / "checkpoints"
                    / f"step_{global_step:06d}"
                )
                save_training_checkpoint(
                    checkpoint_dir,
                    peft_model=peft_model,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    next_epoch=next_epoch,
                    next_update_index=next_update_index,
                    args=args,
                    source_checkpoint=adapter_path,
                    data_manifest_sha256=selection_manifest["data_manifest_sha256"],
                )
                print(f"[checkpoint] saved {checkpoint_dir}", flush=True)
            if global_step == 1 or global_step % 4 == 0 or global_step == total_steps:
                print(
                    f"[train] {global_step}/{total_steps} epoch={epoch} "
                    f"bucket={update_plan.bucket_label} trajectories={update_plan.trajectory_count} "
                    f"loss={update_loss:.6f}",
                    flush=True,
                )

    after = evaluate(
        model, validation_store, validation_rows, device, "after", progress_path
    )
    (args.output_dir / "after_metrics.json").write_text(
        json.dumps(after["metrics"], indent=2, sort_keys=True) + "\n"
    )
    write_jsonl(args.output_dir / "after_predictions.jsonl", after["per_row"])
    gate = gate_result(after["metrics"], selection_manifest["constant_baseline"])
    gate["gradient_checks"] = {
        "mlp_lora_nonzero": nonzero_mlp_lora_gradients_seen,
        "value_head_nonzero": nonzero_value_head_gradients_seen,
    }
    gate["passed"] = bool(
        gate["passed"]
        and nonzero_mlp_lora_gradients_seen
        and nonzero_value_head_gradients_seen
    )

    checkpoint_dir = args.output_dir / "smoke_checkpoint"
    save_training_checkpoint(
        checkpoint_dir,
        peft_model=peft_model,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        next_epoch=args.epochs + 1,
        next_update_index=0,
        args=args,
        source_checkpoint=adapter_path,
        data_manifest_sha256=selection_manifest["data_manifest_sha256"],
    )

    result = {
        "format": "lottie_sao_critic_smoke_result_v1",
        "completed": True,
        "gate": gate,
        "global_steps": global_step,
        "epochs": args.epochs,
        "train_selection": args.train_selection,
        "bucket_plan": bucket_summary,
        "elapsed_sec": time.time() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "parameter_audit": parameter_audit,
        "nonzero_mlp_lora_gradients_seen": nonzero_mlp_lora_gradients_seen,
        "nonzero_value_head_gradients_seen": nonzero_value_head_gradients_seen,
        "constant_baseline": selection_manifest["constant_baseline"],
        "before": before["metrics"],
        "after": after["metrics"],
        "full_training_allowed": bool(gate["passed"]),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
