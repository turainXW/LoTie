# Recorded Run Artifacts

These files were copied from the completed remote run without recomputing the
metrics. They contain no model, optimizer, or RNG weights.

- `result.json`: consolidated before/after metrics, gate, memory, and timing.
- `train_log.jsonl`: one record per optimizer update.
- `selection_manifest.json`: selected data, priors, and bucket summary.
- `bucket_updates_epoch_001.jsonl`: exact trajectory grouping per update.
- `before_*` and `after_*`: formal fixed-holdout predictions and metrics.
- `eval_progress.jsonl`: evaluation progress telemetry.
- `parameter_audit.json`: trainable/frozen parameter audit.
- `checkpoint_manifest.json`: manifest from the historical final checkpoint.

Important: the historical manifest is format v1 and references checkpoint
weights that are not published here. The recorded v1 checkpoint did not have
`trainer_state.json`; the trainer now included in this directory writes a v2
exact-resume checkpoint with that file.

Checksums for the two primary claims:

```text
a1d350773f51683d07cdb3e5ba84c78cad9aaa3e1ef342b5162e5fee1e488ad3  result.json
6766a43775aa18ffebdebb39cc8b7c1dc3c54fef1102f446d48846ca3a629283  checkpoint_manifest.json
```
