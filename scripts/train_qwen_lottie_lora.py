#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterator


IGNORE_INDEX = -100
BUCKET_BOUNDARIES = (4096, 8192, 16384, 24576, 32768)
BATCH_PROFILES = {
    "rtx_pro_6000_verified": (8, 2, 1, 1, 1),
    "safe_32k": (8, 4, 2, 1, 1),
    "throughput_16k": (4, 2, 1, 1, 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BF16 LoRA SFT for pre-masked Lottie Qwen Arrow datasets.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--resume-adapter-path",
        help="Load an existing PEFT adapter as the trainable starting point.",
    )
    parser.add_argument(
        "--resume-training-state-dir",
        help="Restore optimizer, scheduler, counters, and RNG state from a full checkpoint directory.",
    )
    parser.add_argument(
        "--mask-audit-report",
        help="Passed audit JSON. Defaults to mask_audit_v2.json next to the dataset directory.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--starting-epoch", type=int, default=1)
    parser.add_argument(
        "--scheduler-total-epochs",
        type=int,
        help="Cosine horizon for a resumable stage; may exceed --epochs for later continuation.",
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--batch-profile",
        choices=sorted(BATCH_PROFILES),
        default="rtx_pro_6000_verified",
    )
    parser.add_argument("--selection", choices=("all", "shortest", "longest"), default="all")
    parser.add_argument(
        "--smoke-bucket",
        type=int,
        choices=BUCKET_BOUNDARIES,
        help="Select the longest examples in one bucket, up to that bucket's configured batch size.",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save an adapter snapshot after every fully completed epoch.",
    )
    parser.add_argument(
        "--save-training-state",
        action="store_true",
        help="Save AdamW, scheduler, counters, and RNG state beside each adapter checkpoint.",
    )
    parser.add_argument("--no-save-final", action="store_true")
    return parser.parse_args()


@dataclass
class BatchStat:
    examples: int
    padded_tokens: int
    real_tokens: int
    supervised_tokens: int


class AssistantMaskCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(row["input_ids"]) for row in rows)
        input_ids = []
        attention_mask = []
        labels = []
        real_tokens = 0
        supervised_tokens = 0
        for row in rows:
            ids = [int(token) for token in row["input_ids"]]
            row_labels = [int(token) for token in row["labels"]]
            if len(ids) != len(row_labels):
                raise ValueError("input_ids and labels length mismatch")
            if any(label not in (IGNORE_INDEX, token) for token, label in zip(ids, row_labels, strict=True)):
                raise ValueError("labels must equal input_ids or -100")
            pad_length = max_length - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_length)
            attention_mask.append([1] * len(ids) + [0] * pad_length)
            labels.append(row_labels + [IGNORE_INDEX] * pad_length)
            real_tokens += len(ids)
            supervised_tokens += sum(label != IGNORE_INDEX for label in row_labels)
        if supervised_tokens == 0:
            raise ValueError("batch has no supervised assistant tokens")
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        batch["batch_stat"] = BatchStat(
            examples=len(rows),
            padded_tokens=len(rows) * max_length,
            real_tokens=real_tokens,
            supervised_tokens=supervised_tokens,
        )
        return batch


class LengthBucketBatchSampler:
    def __init__(self, lengths: list[int], *, batch_sizes: tuple[int, ...], seed: int) -> None:
        if len(batch_sizes) != len(BUCKET_BOUNDARIES):
            raise ValueError("batch_sizes must match bucket boundaries")
        self.lengths = lengths
        self.batch_sizes = batch_sizes
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _bucket_index(self, length: int) -> int:
        for index, boundary in enumerate(BUCKET_BOUNDARIES):
            if length <= boundary:
                return index
        raise ValueError(f"sequence length {length} exceeds {BUCKET_BOUNDARIES[-1]}")

    def batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, length in enumerate(self.lengths):
            buckets[self._bucket_index(length)].append(index)
        batches: list[list[int]] = []
        for bucket_index in range(len(BUCKET_BOUNDARIES)):
            indexes = buckets[bucket_index]
            rng.shuffle(indexes)
            indexes.sort(key=lambda index: self.lengths[index])
            batch_size = self.batch_sizes[bucket_index]
            batches.extend(indexes[offset : offset + batch_size] for offset in range(0, len(indexes), batch_size))
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches()

    def __len__(self) -> int:
        counts = [0] * len(BUCKET_BOUNDARIES)
        for length in self.lengths:
            counts[self._bucket_index(length)] += 1
        return sum(math.ceil(count / size) for count, size in zip(counts, self.batch_sizes, strict=True))


def select_dataset(
    dataset: Any,
    selection: str,
    max_train_samples: int | None,
    *,
    smoke_bucket: int | None,
    batch_sizes: tuple[int, ...],
) -> Any:
    if smoke_bucket is not None:
        bucket_index = BUCKET_BOUNDARIES.index(smoke_bucket)
        lower_bound = BUCKET_BOUNDARIES[bucket_index - 1] if bucket_index else 0
        candidates = [
            index
            for index, length in enumerate(dataset["token_count"])
            if lower_bound < int(length) <= smoke_bucket
        ]
        if not candidates:
            raise ValueError(f"no examples found in bucket ending at {smoke_bucket}")
        candidates.sort(key=lambda index: int(dataset[index]["token_count"]), reverse=True)
        dataset = dataset.select(candidates[: batch_sizes[bucket_index]])
    elif selection == "shortest":
        index = min(range(len(dataset)), key=lambda item: int(dataset[item]["token_count"]))
        dataset = dataset.select([index])
    elif selection == "longest":
        index = max(range(len(dataset)), key=lambda item: int(dataset[item]["token_count"]))
        dataset = dataset.select([index])
    if max_train_samples is not None:
        dataset = dataset.select(range(min(max_train_samples, len(dataset))))
    return dataset


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def atomic_torch_save(payload: Any, path: Path) -> None:
    import torch

    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training entry requires BF16 support")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.starting_epoch < 1:
        raise ValueError("--starting-epoch must be positive")
    scheduler_total_epochs = args.scheduler_total_epochs or args.epochs
    if scheduler_total_epochs < args.epochs:
        raise ValueError("--scheduler-total-epochs cannot be smaller than --epochs")
    if args.resume_training_state_dir and not args.resume_adapter_path:
        args.resume_adapter_path = args.resume_training_state_dir
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir).resolve()
    audit_path = (
        Path(args.mask_audit_report).resolve()
        if args.mask_audit_report
        else dataset_dir.parent / "mask_audit_v2.json"
    )
    if not audit_path.is_file():
        raise FileNotFoundError(f"A passed independent mask audit is required: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise ValueError(f"Mask audit did not pass: {audit_path}")
    full_dataset = load_from_disk(str(dataset_dir))
    if int(audit.get("rows", -1)) != len(full_dataset):
        raise ValueError("Mask audit row count does not match the Arrow dataset")
    if int(audit.get("supervised_tokens", -1)) != sum(int(value) for value in full_dataset["supervised_token_count"]):
        raise ValueError("Mask audit supervised-token count does not match the Arrow dataset")
    batch_sizes = BATCH_PROFILES[args.batch_profile]
    dataset = select_dataset(
        full_dataset,
        args.selection,
        args.max_train_samples,
        smoke_bucket=args.smoke_bucket,
        batch_sizes=batch_sizes,
    )
    if not len(dataset):
        raise ValueError("selected dataset is empty")
    lengths = [int(value) for value in dataset["token_count"]]
    if max(lengths) > BUCKET_BOUNDARIES[-1]:
        raise ValueError(f"dataset contains a sequence longer than {BUCKET_BOUNDARIES[-1]}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch_sampler = LengthBucketBatchSampler(lengths, batch_sizes=batch_sizes, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=AssistantMaskCollator(tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.resume_adapter_path:
        adapter_path = Path(args.resume_adapter_path).resolve()
        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"PEFT adapter checkpoint is incomplete: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        adapter_path = None
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
    model.enable_input_require_grads()
    model.train().cuda()

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    microbatches_per_epoch = len(dataloader)
    updates_per_epoch = math.ceil(microbatches_per_epoch / args.gradient_accumulation_steps)
    planned_updates_this_run = updates_per_epoch * args.epochs
    updates_this_run = (
        min(planned_updates_this_run, args.max_steps) if args.max_steps > 0 else planned_updates_this_run
    )
    scheduler_total_updates = updates_per_epoch * scheduler_total_epochs
    warmup_steps = int(scheduler_total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        scheduler_total_updates,
    )

    update_step = 0
    micro_step = 0
    processed_real_tokens = 0
    resume_state_dir = Path(args.resume_training_state_dir).resolve() if args.resume_training_state_dir else None
    if resume_state_dir:
        state_path = resume_state_dir / "trainer_state.json"
        required = [
            state_path,
            resume_state_dir / "optimizer.pt",
            resume_state_dir / "scheduler.pt",
            resume_state_dir / "rng_state.pt",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"full training checkpoint is incomplete: {missing}")
        resume_state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(resume_state["scheduler_total_updates"]) != scheduler_total_updates:
            raise ValueError("resumed scheduler horizon does not match --scheduler-total-epochs")
        optimizer.load_state_dict(
            torch.load(resume_state_dir / "optimizer.pt", map_location="cpu", weights_only=False)
        )
        scheduler.load_state_dict(
            torch.load(resume_state_dir / "scheduler.pt", map_location="cpu", weights_only=False)
        )
        rng_state = torch.load(resume_state_dir / "rng_state.pt", map_location="cpu", weights_only=False)
        random.setstate(rng_state["python"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state_all(rng_state["cuda"])
        update_step = int(resume_state["stage_update_step"])
        micro_step = int(resume_state["stage_micro_step"])
        processed_real_tokens = int(resume_state.get("processed_real_tokens", 0))
    initial_update_step = update_step
    target_update_step = initial_update_step + updates_this_run

    manifest = {
        "format": "lottie_qwen_lora_run_v2",
        "dataset_dir": str(dataset_dir),
        "mask_audit_report": str(audit_path),
        "model_path": str(Path(args.model_path).resolve()),
        "output_dir": str(output_dir),
        "dataset_rows": len(dataset),
        "tokens": sum(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "selection": args.selection,
        "smoke_bucket": args.smoke_bucket,
        "batch_profile": args.batch_profile,
        "bucket_boundaries": BUCKET_BOUNDARIES,
        "bucket_batch_sizes": batch_sizes,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "starting_epoch": args.starting_epoch,
        "scheduler_total_epochs": scheduler_total_epochs,
        "max_steps": args.max_steps,
        "microbatches_per_epoch": microbatches_per_epoch,
        "updates_per_epoch": updates_per_epoch,
        "planned_updates_this_run": updates_this_run,
        "target_stage_update_step": target_update_step,
        "scheduler_total_updates": scheduler_total_updates,
        "warmup_steps": warmup_steps,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "attn_implementation": args.attn_implementation,
        "intermediate_checkpoints": args.save_every_epoch,
        "training_state_checkpoints": args.save_training_state,
        "resume_adapter_path": str(adapter_path) if adapter_path else None,
        "resume_training_state_dir": str(resume_state_dir) if resume_state_dir else None,
        "resume_mode": "full_state" if resume_state_dir else "adapter_only" if adapter_path else "fresh",
        "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
        "device": torch.cuda.get_device_name(0),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "overfitting_monitor": {
            "train_metrics": [
                "loss",
                "supervised_token_weighted_loss",
                "gradient_norm",
                "learning_rate",
                "trainable_parameter_norm",
            ],
            "heldout_validation_required": True,
            "recommended_selection": "data/dataset_splits_v1/selection.json",
            "evaluation_split_excluded_from_tuning": True,
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    log_path = output_dir / "train_log.jsonl"
    epoch_metrics_path = output_dir / "epoch_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    accumulation_count = 0
    processed_real_tokens_this_run = 0
    run_micro_step = 0
    gradient_nonzero_seen = False
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    stop = False
    completed_epoch_numbers: list[int] = []
    epoch_checkpoints: list[str] = []

    def trainable_parameter_norm() -> float:
        squared_norm = 0.0
        with torch.no_grad():
            for parameter in trainable_parameters:
                squared_norm += float(parameter.detach().float().square().sum().item())
        return math.sqrt(squared_norm)

    def save_checkpoint(checkpoint_dir: Path, *, epoch_number: int, final: bool = False) -> dict[str, Any]:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        state_files: dict[str, str] = {}
        if args.save_training_state:
            atomic_torch_save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
            atomic_torch_save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
            atomic_torch_save(
                {
                    "python": random.getstate(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                },
                checkpoint_dir / "rng_state.pt",
            )
            trainer_state = {
                "format": "lottie_qwen_lora_trainer_state_v1",
                "completed_epoch": epoch_number,
                "stage_update_step": update_step,
                "stage_micro_step": micro_step,
                "processed_real_tokens": processed_real_tokens,
                "scheduler_total_updates": scheduler_total_updates,
                "warmup_steps": warmup_steps,
                "learning_rate": scheduler.get_last_lr()[0],
                "target_stage_update_step": target_update_step,
            }
            write_json(checkpoint_dir / "trainer_state.json", trainer_state)
            state_files = {
                "optimizer": "optimizer.pt",
                "scheduler": "scheduler.pt",
                "rng": "rng_state.pt",
                "trainer": "trainer_state.json",
            }
        checkpoint_report = {
            "format": "lottie_qwen_lora_epoch_checkpoint_v2",
            "epoch": epoch_number,
            "stage_updates": update_step,
            "stage_micro_steps": micro_step,
            "adapter_dir": str(checkpoint_dir),
            "final": final,
            "training_state_saved": args.save_training_state,
            "state_files": state_files,
            "resume_command_inputs": {
                "resume_adapter_path": str(checkpoint_dir),
                "resume_training_state_dir": str(checkpoint_dir) if args.save_training_state else None,
                "starting_epoch": epoch_number + 1,
                "scheduler_total_epochs": scheduler_total_epochs,
            },
        }
        write_json(checkpoint_dir / "checkpoint_manifest.json", checkpoint_report)
        print(json.dumps(checkpoint_report, ensure_ascii=False), flush=True)
        return checkpoint_report

    for local_epoch in range(args.epochs):
        epoch_number = args.starting_epoch + local_epoch
        batch_sampler.set_epoch(epoch_number - 1)
        epoch_microbatches = len(dataloader)
        epoch_losses: list[float] = []
        epoch_gradient_norms: list[float] = []
        epoch_weighted_loss = 0.0
        epoch_supervised_tokens = 0
        epoch_real_tokens = 0
        epoch_start_learning_rate = scheduler.get_last_lr()[0]
        for epoch_micro_step, batch in enumerate(dataloader, 1):
            micro_step += 1
            run_micro_step += 1
            batch_stat = batch.pop("batch_stat")
            processed_real_tokens += batch_stat.real_tokens
            processed_real_tokens_this_run += batch_stat.real_tokens
            epoch_real_tokens += batch_stat.real_tokens
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            observed_supervised = int((batch["labels"] != IGNORE_INDEX).sum().item())
            if observed_supervised != batch_stat.supervised_tokens:
                raise ValueError("collator changed the assistant-only mask")
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at micro step {micro_step}: {loss.item()}")
            loss_value = float(loss.detach().item())
            epoch_losses.append(loss_value)
            epoch_weighted_loss += loss_value * batch_stat.supervised_tokens
            epoch_supervised_tokens += batch_stat.supervised_tokens
            (loss / args.gradient_accumulation_steps).backward()
            accumulation_count += 1
            should_update = (
                accumulation_count == args.gradient_accumulation_steps or epoch_micro_step == epoch_microbatches
            )
            gradient_norm: float | None = None
            if should_update:
                if accumulation_count < args.gradient_accumulation_steps:
                    correction = args.gradient_accumulation_steps / accumulation_count
                    for parameter in trainable_parameters:
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                if not gradient_nonzero_seen:
                    gradient_nonzero_seen = any(
                        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
                        for parameter in trainable_parameters
                    )
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm).item()
                )
                epoch_gradient_norms.append(gradient_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                accumulation_count = 0
            if run_micro_step % args.logging_steps == 0 or should_update:
                elapsed = max(time.monotonic() - started, 1e-6)
                record = {
                    "epoch": epoch_number,
                    "local_epoch": local_epoch + 1,
                    "micro_step": micro_step,
                    "run_micro_step": run_micro_step,
                    "update_step": update_step,
                    "loss": round(loss_value, 6),
                    "gradient_norm": round(gradient_norm, 6) if gradient_norm is not None else None,
                    "examples": batch_stat.examples,
                    "real_tokens": batch_stat.real_tokens,
                    "padded_tokens": batch_stat.padded_tokens,
                    "supervised_tokens": batch_stat.supervised_tokens,
                    "tokens_per_sec_current_run": round(processed_real_tokens_this_run / elapsed, 2),
                    "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                    "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if update_step >= target_update_step:
                stop = True
                break
        epoch_completed = epoch_micro_step == epoch_microbatches
        if epoch_completed:
            completed_epoch_numbers.append(epoch_number)
            parameter_norm = trainable_parameter_norm()
            epoch_metrics = {
                "format": "lottie_qwen_lora_epoch_metrics_v1",
                "epoch": epoch_number,
                "microbatches": epoch_microbatches,
                "stage_update_step": update_step,
                "mean_microbatch_loss": statistics.mean(epoch_losses),
                "supervised_token_weighted_loss": epoch_weighted_loss / epoch_supervised_tokens,
                "gradient_norm_mean": statistics.mean(epoch_gradient_norms),
                "gradient_norm_max": max(epoch_gradient_norms),
                "learning_rate_start": epoch_start_learning_rate,
                "learning_rate_end": scheduler.get_last_lr()[0],
                "real_tokens": epoch_real_tokens,
                "supervised_tokens": epoch_supervised_tokens,
                "trainable_parameter_norm": parameter_norm,
            }
            with epoch_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
            print(json.dumps(epoch_metrics, ensure_ascii=False), flush=True)
            if args.save_every_epoch:
                checkpoint_dir = output_dir / f"epoch_{epoch_number:03d}_adapter"
                save_checkpoint(checkpoint_dir, epoch_number=epoch_number)
                epoch_checkpoints.append(str(checkpoint_dir))
        if stop:
            break

    final_epoch = completed_epoch_numbers[-1] if completed_epoch_numbers else args.starting_epoch - 1
    final_report = {
        "format": "lottie_qwen_lora_result_v2",
        "stage_updates": update_step,
        "stage_micro_steps": micro_step,
        "updates_this_run": update_step - initial_update_step,
        "micro_steps_this_run": run_micro_step,
        "completed_epoch_numbers": completed_epoch_numbers,
        "epoch_checkpoints": epoch_checkpoints,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "completed": True,
        "final_adapter_saved": not args.no_save_final,
        "training_state_saved": args.save_training_state,
        "nonzero_lora_gradients_seen": gradient_nonzero_seen,
    }
    if not args.no_save_final:
        final_dir = output_dir / "final_adapter"
        save_checkpoint(final_dir, epoch_number=final_epoch, final=True)
        final_report["final_adapter"] = str(final_dir)
    write_json(output_dir / "result.json", final_report)
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
