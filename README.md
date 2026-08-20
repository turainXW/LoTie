# LoTie

LoTie (pronounced **LOH-tee**, approximately "洛蒂") is a compact code-agent
runtime and data pipeline for repository repair, trajectory collection,
assistant-only SFT preparation, and verifier-driven Pass@k evaluation.

[中文说明](README_zh.md) | [Full project report (Chinese)](reports/Lottie_Code_Agent_FULL_PROJECT_REPORT_ZH.md)

## What is included

- A JSON tool-calling agent loop for shell execution, structured editing,
  repository context, testing, patch generation, and resumable runs.
- Deterministic train/validation/evaluation splits over MBPP+, HumanEval+, and
  SWE-smith, with hidden gold data kept out of agent prompts.
- Concurrent and resumable rollout collection with infrastructure retry
  accounting and model/verifier failure separation.
- Auditable Strict, Salvage, and Quarantine trajectory partitions.
- Qwen chat-template conversion, assistant-only loss masks, and 4K-32K
  length-bucketed LoRA SFT utilities.
- EvalPlus and SWE-smith verification plus Pass@1, Pass@2, and Pass@3 reports.

## Pipeline

```text
task split
   -> isolated repository / function sandbox
   -> agent tool loop
   -> patch + tests + verifier result
   -> raw trajectory and rollout metadata
   -> Strict / Salvage / Quarantine audit
   -> target-tokenizer conversion and assistant-only mask
   -> length-bucketed SFT
   -> versioned Pass@k evaluation
```

The repository intentionally excludes model weights, raw task workspaces,
virtual environments, API credentials, and private SSH configuration.

## Quick start

Requirements: macOS or Linux, Python 3.10+, Git, and a model endpoint compatible
with the OpenAI chat-completions API.

```bash
git clone git@github.com:turainXW/LoTie.git
cd LoTie

# Minimal editable install. Use the default with no argument for dev + SWE tools.
bash scripts/install_dev.sh base
source scripts/dev_env.sh

lotie --help
python3 scripts/run_smoke.py --output-dir data/code_agent_smoke
```

The installer creates `.venv`, installs the package, initializes local agent
state under `.codeagent/`, and copies `.env.example` to `.env`. Secrets in
`.env` are ignored by Git.

### Run a one-shot repository task

```bash
lotie use \
  --repo /path/to/target-repository \
  --task "Fix the failing parser test and verify the smallest patch" \
  --sandbox copy \
  --model-url http://127.0.0.1:8000/v1 \
  --model local-model \
  --max-steps 40 \
  --show-steps
```

`--sandbox copy` is the safe default: the agent edits an isolated copy instead
of the source repository. Use `--sandbox direct` only when in-place changes are
intentional.

The preferred CLI is `lotie`. The historical `minicoder` and `codeagent`
entrypoints remain available for compatibility.

## Trajectory collection and evaluation

Plan a versioned multi-rollout experiment without executing it:

```bash
lotie pass-at-k \
  --selection data/dataset_splits_v1/selection.json \
  --split evaluation \
  --samples 3 \
  --workers 4 \
  --output-dir outputs/eval90_pass3 \
  --plan-only
```

Run `lotie pass-at-k --help` before a formal collection. The command records
the model endpoint, sampling parameters, task split, worker policy, verifier
timeouts, and valid sample slots so interrupted runs can resume without
overwriting completed results.

For local SWE-smith setup and macOS collection, see:

- [Portable macOS trajectory collection](docs/MACOS_PORTABLE_TRAJECTORY_COLLECTION_ZH.md)
- [Eval90 deployment](DEPLOY_EVAL90_zh.md)
- [General deployment notes](DEPLOY_zh.md)

## Included training data

The sanitized delivery under
`data/trajectories/lottie_train1200_cleaned_complete_v1/` contains:

| Artifact | Rows | Intended use |
| --- | ---: | --- |
| Raw valid rollouts | 1,200 | RL/reward analysis, including real model failures |
| Verifier-resolved rollouts | 1,090 | Successful behavior analysis |
| Strict SFT trajectories | 996 | Highest-confidence SFT source |
| Primary SFT trajectories | 1,069 | Strict + independently audited Tier-A salvage |
| Optional Tier-B trajectories | 13 | Lower-weight optional experiments |

Compressed data files are accompanied by manifests, SHA-256 provenance,
exclusion records, and per-trajectory repair audits. They are intentionally not
pre-tokenized: use the exact tokenizer and chat template of the target model.

```bash
python3 tools/prepare_qwen_sft_dataset.py --help
python3 tools/audit_qwen_sft_arrow.py --help
python3 scripts/train_qwen_lottie_lora.py --help
```

The training utilities supervise only valid assistant turns. System prompts,
user messages, tool observations, malformed protocol output, padding, and
overlapping history from long-window splits are masked from loss.

## Reference evaluation

The checked-in report uses one harness and sampling policy for 90 evaluation
tasks with three rollouts per task. The best Qwen3-4B LoRA checkpoint in that
run was Epoch 3.

| Model | Overall P@1 | Overall P@2 | Overall P@3 | SWE-smith P@1 | SWE-smith P@2 | SWE-smith P@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B base | 52.59% | 62.59% | 66.67% | 18.89% | 26.67% | 30.00% |
| LoTie Epoch 3 | 60.74% | 68.52% | 72.22% | 27.78% | 37.78% | 43.33% |

See [evaluation methodology and charts](reports/qwen_v3_eval_20260816/README.md).
SWE-smith local-venv results are marked `official_comparable=false`; use the
official containerized verifier for leaderboard-comparable claims.

## Repository layout

```text
LoTie/
├── src/code_agent_baseline/   # Agent loop, tools, context, model clients, CLI
├── scripts/                   # Collection, evaluation, deployment, and training
├── tools/                     # Cleaning, auditing, tokenization, and packaging
├── benchmarks/                # Unit and integration tests
├── configs/                   # Runtime defaults
├── data/
│   ├── dataset_splits_v1/     # Deterministic train/validation/eval selection
│   ├── evalplus_local/        # Sanitized function-task inputs
│   ├── swesmith_local/        # Sanitized repository-task inputs
│   └── trajectories/          # Compressed cleaned trajectory delivery
├── reports/                   # Training/evaluation metrics, plots, and analysis
├── docs/                      # Operational and portability documentation
├── requirements-*.txt         # Reproducible dependency profiles
└── pyproject.toml             # Package metadata and CLI entrypoints
```

Generated `outputs/`, `dist/`, `.venv/`, `.codeagent/`, logs, and local
workspaces are ignored.

## Verification

```bash
source scripts/dev_env.sh
PYTHONPATH=src:benchmarks/leetcode_top10 \
  python -m unittest discover -s benchmarks -p 'test_*.py'
```

Verifier outcomes are kept separate from agent execution status. Test failure,
`no_edit`, `max_steps`, and parse failure remain real model outcomes; only
explicit API, process, OOM, or verifier infrastructure failures are eligible
for infrastructure retry.

## Data and license notes

- Gold patches, canonical solutions, and hidden tests are not exposed in agent
  prompts.
- Dataset records retain upstream identifiers and should be used under their
  respective upstream terms.
- This repository currently ships without a general software license. Public
  visibility does not by itself grant redistribution or modification rights.

Before making a release archive, follow
[the open-source checklist](OPEN_SOURCE_CHECKLIST_zh.md).
