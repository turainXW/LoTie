# Lottie SAO Value Model

This directory records the value-model cold start used before online SAO
training. It contains the runnable trainer, length-bucket planner, complete
tokenized datasets, audit metadata, and the artifacts from the Qwen3-4B run
completed on 2026-08-24.

The scope is intentionally narrow: this is **not** an online SAO or PPO
trainer. It trains a scalar value head and MLP LoRA parameters to predict the
terminal verifier return on every assistant action token.

## Layout

- `scripts/build_value_dataset.py`: builds token arrays and assistant-only masks.
- `scripts/train_value_model.py`: plans buckets, trains, evaluates, and checkpoints.
- `scripts/run_qwen3_4b_value_training.sh`: the reproducible one-epoch recipe.
- `src/sao_value/critic_bucketing.py`: length-homogeneous update planner.
- `data/`: complete compressed datasets plus expanded audit metadata.
- `artifacts/qwen3_4b_value_model_v1/`: logs, predictions, metrics, and manifests.
- `docs/value_model_training_zh.md`: detailed Chinese training notes and formulas.

## Quick Start

The recorded environment used Python 3.12.3, PyTorch 2.13.0+cu130,
Transformers 5.10.4, PEFT 0.20.0, and one NVIDIA RTX PRO 6000 Blackwell
Server Edition with 97,887 MiB VRAM.

```bash
cd sao
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
./scripts/unpack_data.sh

BASE_MODEL=/path/to/Qwen3-4B-Instruct-2507 \
ACTOR_ADAPTER=/path/to/epoch_003_adapter \
./scripts/run_qwen3_4b_value_training.sh
```

The run script refuses to overwrite an existing output directory. Periodic
checkpoints are disabled to reproduce the original one-shot run; the final
checkpoint still contains the Critic adapter, scalar value head, optimizer,
scheduler, RNG, trainer state, and checkpoint manifest.

## Recorded Recipe

| Item | Value |
| --- | --- |
| Base | Qwen3-4B-Instruct-2507 + Epoch3 Actor adapter |
| Train rows | 888 trajectories |
| Optimizer updates | 111 |
| Epochs | 1 |
| MLP LoRA LR | `5e-6` |
| Value head LR | `5e-5` |
| Warmup | 10 updates |
| Sequence limit | 65,536 tokens |
| Batch profile | `critic_96gb_verified` = `(8, 8, 4, 2, 2, 1)` |
| Target per update | 8 trajectories |
| Attention LoRA | frozen |
| Trainable LoRA | `gate_proj`, `up_proj`, `down_proj` |
| Peak allocated VRAM | 66.86 GiB |
| Wall time | 26,639.6 seconds (7.40 hours) |

## Recorded Result

The formal comparison uses the fixed 40-trajectory holdout. Metrics below are
trajectory-mean predictions.

| Family | Metric | Before | After | Constant baseline |
| --- | --- | ---: | ---: | ---: |
| Function | AUC | 0.4800 | 0.7867 | 0.5000 |
| Function | Brier | 0.2865 | 0.1996 | 0.1876 |
| Function | Explained variance | -0.0034 | 0.0858 | 0.0000 |
| SWE-smith | AUC | 0.3452 | 0.9524 | 0.5000 |
| SWE-smith | Brier | 0.2256 | 0.1772 | 0.2100 |
| SWE-smith | Explained variance | -0.0100 | 0.1563 | 0.0000 |

SWE-smith passed all three gates. Function improved strongly but did not beat
its constant-prior Brier score, so the historical all-family gate remained
false. Both MLP LoRA and value-head gradients were observed as nonzero.

## Checkpoint Note

The archived historical run wrote a v1 checkpoint manifest and did not emit
`trainer_state.json`. The published trainer includes the exact-resume work
added immediately afterward and writes a v2 manifest plus trainer state. The
metrics in `artifacts/` are unchanged historical outputs, not regenerated
claims.

## Data and Licensing

See `data/DATA_CARD.md` before redistributing or training. The archives contain
tokenized trajectories and benchmark-derived metadata. No new license grant is
made for upstream benchmark content or model artifacts, and model weights are
not included.
