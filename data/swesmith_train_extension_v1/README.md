# SWE-smith Train Extension V1

This directory is isolated from the original 100-task SWE-smith training set.

Construction policy:

- Source dataset: `SWE-bench/SWE-smith-py` at revision
  `77cab9055d42ab4a5c25c89a8f937096db13558e`.
- Exclude every repository and instance already used by the original train,
  validation, or evaluation split.
- Keep one-file patches with 2-30 changed lines, 1-20 `FAIL_TO_PASS`
  tests, non-empty `PASS_TO_PASS`, and a problem statement of at least 80
  characters.
- Select 20 new repositories and 5 tasks per repository. Repository choices
  may be supplied explicitly after environment preflight; task choices always
  use deterministic SHA-256 ranking and mutation-kind stratification.
- Admit tasks to `tasks_runnable.jsonl` only after repository setup and the
  base/bug/gold sanity checks pass.

Generated files are intentionally kept separate from `data/swesmith_local`.
The selection builder refuses to overwrite an existing selection unless
`--force` is explicitly supplied.

Final preflight result:

- 100 selected tasks across 20 repositories, exactly 5 tasks per repository.
- 20/20 repository environments ready under
  `.codeagent/swesmith_train_extension_v1`.
- 100/100 tasks passed the base/bug/gold sanity sequence.
- Zero repository overlap and zero instance overlap with the existing
  train/validation/evaluation SWE-smith splits.

Artifacts:

- `selection.json`: immutable source revision, source hashes, filters, and IDs.
- `tasks.jsonl`: materialized task records with local environment metadata.
- `setup_results.jsonl`: repository setup outcomes.
- `gold_sanity.jsonl`: per-task base/bug/gold verification details.
- `tasks_runnable.jsonl`: the 100 tasks admitted for trajectory collection.
- `audit_report.json`: integrity checks, distributions, and artifact hashes.
