# Harness V3 全量 Pass@3 评测图表

- 口径：5 个模型，每个模型 90 题 x 3 rollout = 270 条，共 1350 条有效评测样本。
- Harness：`lottie_code_agent_harness_v3`，策略 `append_nonmod3_action_reset_v1`。
- 采样：temperature=0.5，top_p=0.95，thinking disabled。
- 步数：函数题 20 步，SWE-smith 60 步。

| 模型 | Overall P@1 | P@2 | P@3 | SWE-smith P@1 | P@2 | P@3 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 4B | 66.67% | 77.04% | 80.00% | 41.11% | 50.00% | 53.33% |
| Qwen3 4B Base | 52.59% | 62.59% | 66.67% | 18.89% | 26.67% | 30.00% |
| Epoch 1 | 55.93% | 63.33% | 66.67% | 11.11% | 16.67% | 20.00% |
| Epoch 3 | 60.74% | 68.52% | 72.22% | 27.78% | 37.78% | 43.33% |
| Epoch 5 | 59.26% | 68.15% | 71.11% | 22.22% | 33.33% | 40.00% |

## 结论

- Epoch 3 是本轮最优微调 checkpoint：Overall P@1 相比 baseline 提升 8.15%，SWE-smith P@3 提升 13.33%。
- Epoch 5 相比 Epoch 3 略有回落：Overall P@1 变化 -1.48%，SWE-smith P@1 变化 -5.56%。
- Qwen3.5 4B 仍是最强参照，但 Epoch 3 已明显缩小 Qwen3 4B baseline 在仓库级任务上的差距。

## 图表口径

- `overall_pass_at_k.png` 使用收紧纵轴，便于观察 checkpoint 差异。
- `overall_pass_at_k_full_scale.png` 使用 0–100% 纵轴，避免视觉放大造成误判。
- 每张图同时提供 PNG 和 PDF；PDF 可直接用于论文或演示文稿。
