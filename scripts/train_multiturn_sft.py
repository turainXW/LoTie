#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.sft_masking import audit_masked_token_records  # noqa: E402


class MaskedTokenDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        return {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
        }


class MaskedTokenCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for assistant-masked multi-turn Lottie trajectories.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file")
    parser.add_argument("--model-path", help="Local model directory or Hugging Face model ID.")
    parser.add_argument("--adapter-path")
    parser.add_argument("--output-dir", default="outputs/lottie_multiturn_sft")
    parser.add_argument("--max-length", type=int, default=32_768)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Set 1-5 for a training smoke test.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(Path(args.train_file))
    if args.max_train_samples is not None:
        train_rows = train_rows[: args.max_train_samples]
    train_report = audit_masked_token_records(train_rows, max_length=args.max_length)
    dev_rows = read_jsonl(Path(args.dev_file)) if args.dev_file else []
    dev_report = audit_masked_token_records(dev_rows, max_length=args.max_length) if dev_rows else None
    audit = {
        "format": "lottie_multiturn_sft_preflight_v1",
        "train": train_report.to_dict(),
        "dev": dev_report.to_dict() if dev_report else None,
        "max_length": args.max_length,
        "validation": "passed",
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.validate_only:
        return
    if not args.model_path:
        raise ValueError("--model-path is required unless --validate-only is used")

    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Training requires torch, transformers, peft, and accelerate") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    expected_fingerprints = {str(row.get("tokenizer_sha256") or "") for row in train_rows + dev_rows}
    if "" in expected_fingerprints or len(expected_fingerprints) != 1:
        raise ValueError("Every masked row must contain the same non-empty tokenizer_sha256")
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is None:
        raise ValueError("The target model must provide a fast tokenizer for token fingerprint validation")
    actual_fingerprint = hashlib.sha256(backend_tokenizer.to_str().encode("utf-8")).hexdigest()
    expected_fingerprint = next(iter(expected_fingerprints))
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "Masked token IDs were generated by a different tokenizer: "
            f"dataset={expected_fingerprint}, model={actual_fingerprint}"
        )
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
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
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_preflight.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=bool(torch.cuda.is_available() and not bf16),
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=MaskedTokenDataset(train_rows),
        eval_dataset=MaskedTokenDataset(dev_rows) if dev_rows else None,
        data_collator=MaskedTokenCollator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = output_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final LoRA adapter to {final_dir}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
