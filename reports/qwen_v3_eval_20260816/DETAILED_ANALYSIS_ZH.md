# Qwen3-4B Lottie SFT 数据、训练与 Harness V3 评测分析

## 1. 执行摘要

本报告联合分析了三类证据：

1. 训练数据包：1200 条采集轨迹，以及其中 1069 条 primary SFT 轨迹的 Qwen3 tokenizer 长度统计。
2. 完整训练日志：Epoch 1-5 共 2285 个 microbatch，约 3792.82 万 context token、354.78 万 supervised token。
3. Harness V3 正式评测：5 个模型版本，每个版本 90 题、每题 3 次，共 270 条有效 rollout，总计 1350 条。

核心结论如下：

- **Epoch 3 是当前最优 checkpoint。** 相比 Qwen3 4B Base，Overall Pass@1 提升 8.15 个百分点，SWE-smith Pass@3 提升 13.33 个百分点。
- **Epoch 5 已出现可观测的泛化回落。** 训练 loss 从 Epoch 3 的 0.1898 继续降到 0.1717，但 Overall Pass@1 从 60.74% 降到 59.26%，SWE-smith Pass@1 从 27.78% 降到 22.22%。这比“loss 还没收敛”更像重复数据上的继续拟合。
- **SFT 很快学会了函数题和“必须动手编辑”的行为，但仓库级能力形成较慢。** Epoch 1 的函数级 Pass@1 已接近 Qwen3.5 4B，但 SWE-smith 反而低于 Base；到 Epoch 3 才恢复为较均衡的能力组合。
- **当前瓶颈更像数据覆盖与训练分布，而不是 LoRA rank。** 仅 1071 条训练样本被重复 5 轮，Rank 32 已能显著改变行为；直接升 Rank 64 不太可能解决 SWE-smith 的任务盲区和 Epoch 5 回落。
- **下一轮应优先扩大独立、成功、仓库级轨迹的多样性，并按 token 而不是仅按样本数控制分布。** 不建议在现有数据上继续无条件训练 Epoch 6+。

最关键的诊断图：

- [训练 loss 与评测分叉](charts/detailed/loss_vs_evaluation_divergence.png)
- [各 checkpoint 相对 Base/Qwen3.5 的能力差值](charts/detailed/checkpoint_delta_heatmap.png)
- [函数级与仓库级能力权衡](charts/detailed/capability_tradeoff_map.png)
- [训练数据样本占比与 token 占比](charts/detailed/training_data_composition.png)
- [Verifier 失败模式热力图](charts/detailed/failure_mode_heatmap.png)

## 2. 数据口径与完整性

### 2.1 采集与清洗

原始采集共有 1200 条 train rollout：

| 项目 | 数量 |
|---|---:|
| Verifier resolved 正样本 | 1090 |
| Verifier unresolved 负样本 | 110 |
| 严格 SFT | 996 |
| Salvage Tier A | 73 |
| Primary SFT（严格 + Tier A） | 1069 |
| Optional Tier B | 13 |
| Quarantine Tier C | 4 |
| Manual review | 4 |
| 精确重复轨迹 | 0 |

Salvage 清洗没有猜测或重建工具调用；自动修复只删除了“无环境副作用的非法 assistant 输出 + 紧随其后的 parser/protocol rejection”成对消息，并保留独立审计记录。90 条 salvage 轨迹全部通过完整性复审，其中 73 条进入 primary SFT。

### 2.2 正式评测

每个模型都使用相同 Harness V3 与采样口径：

- Harness：`lottie_code_agent_harness_v3`
- 只读预算策略：`append_nonmod3_action_reset_v1`
- 连续 3 次非修改动作后提醒，发生修改后重置计数
- temperature=0.5，top_p=0.95，thinking disabled
- 函数题 max_steps=20；SWE-smith max_steps=60
- 60 道函数题 + 30 道 SWE-smith，每题 3 rollout

因此 checkpoint 间比较具备一致口径。评测样本只用于 evaluation，不在训练集中回流。

### 2.3 需要补齐的 provenance

当前存在两个需要在下一次训练前核对的问题：

1. Primary 包的 tokenizer 报告是 **1069 条 / 7,582,425 token**，正式训练配置记录为 **1071 条 / 7,585,639 token**，相差 2 条和 3214 token。现有本地元数据不能证明这两条具体来自哪里。
2. Primary tokenizer 报告中有 2 条超过 32768 token，最长 44802；训练配置的最大序列长度为 32768，训练日志最大实际长度为 32682。这两条必然经历了截断、排除或重新组装，具体策略需要从远端训练数据 manifest 再确认。

这两个问题不推翻当前评测结论，但会影响实验复现和后续增量训练，建议在数据版本中明确记录每一行的最终 token 数、截断方式与入选来源。

## 3. 训练数据分布

### 3.1 样本数与 token 质量

Primary SFT 共 1069 条、758.24 万 context token：

| 数据集 | 行数 | 行占比 | Context token | Token 占比 | 中位长度 | P95 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HumanEval+ | 305 | 28.53% | 1,164,447 | 15.36% | 3,579 | 4,670 | 22,902 |
| MBPP+ | 411 | 38.45% | 1,360,740 | 17.94% | 3,210 | 3,813 | 9,858 |
| SWE-smith | 353 | 33.02% | 5,057,238 | 66.69% | 13,499 | 25,293 | 44,802 |

图表：[训练数据构成](charts/detailed/training_data_composition.png)

最重要的分布事实是：**SWE-smith 只占 33.0% 的样本，却消耗 66.7% 的 context token。** 这意味着：

- 按“样本条数”看起来三类任务相对均衡，按“计算量和上下文占用”看则明显偏向仓库级任务。
- 但 assistant-only supervised token 只占全部 context 的约 9.35%，大量计算用于读取工具结果与历史上下文，这是 Agent SFT 的正常成本。
- SWE-smith 仍显著落后 Qwen3.5，说明问题不是简单的“仓库级 token 太少”，而更可能是项目/错误类型覆盖不足、成功路径单一、编辑与测试策略不够多样。

### 3.2 长度桶

| 长度 | 行数 | 占比 |
|---|---:|---:|
| <=4k | 653 | 61.09% |
| 4k-8k | 85 | 7.95% |
| 8k-16k | 235 | 21.98% |
| 16k-32k | 94 | 8.79% |
| >32k | 2 | 0.19% |

图表：[长度桶分布](charts/detailed/token_length_buckets.png)

整体中位数 3610 token、P90 15964、P95 19364、P99 27481。函数题基本集中在 4k 左右；8k 以上样本几乎由 SWE-smith 主导。因此动态 batch profile 是合理的，但后续训练最好同时记录：

- 每个长度桶的样本数、context token、supervised token；
- 每个 optimizer update 的数据集构成；
- 被截断轨迹是否保留了问题、关键编辑、测试结果和终止动作。

## 4. 训练过程

### 4.1 配置

| 参数 | 数值 |
|---|---|
| 基础模型 | Qwen3-4B-Instruct-2507 |
| 精度 / Attention | BF16 / SDPA |
| LoRA | r=32, alpha=64, dropout=0.05 |
| 目标层 | q/k/v/o + gate/up/down projections |
| Optimizer | AdamW |
| Epoch 1-2 LR | 5e-5，cosine + 13 update warmup |
| Epoch 3-5 LR | 1e-5 stage 起点，继续 cosine 衰减 |
| Gradient accumulation | 2 |
| Max grad norm | 1.0 |
| 每轮 | 457 microbatch，1071 examples，229 updates |
| 最大序列 | 32768 |

### 4.2 五轮统计

| Epoch | Token 加权 loss | Median loss | LR max | LR end | Grad norm median | Grad norm P95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2849 | 0.2611 | 5.00e-5 | 2.61e-5 | 未记录 | 未记录 |
| 2 | 0.2002 | 0.1906 | 2.61e-5 | 0 | 未记录 | 未记录 |
| 3 | 0.1898 | 0.1760 | 1.00e-5 | 7.58e-6 | 0.538 | 0.804 |
| 4 | 0.1785 | 0.1635 | 7.58e-6 | 2.54e-6 | 0.596 | 0.850 |
| 5 | 0.1717 | 0.1526 | 2.54e-6 | 0 | 0.610 | 0.892 |

图表：[完整训练动态](charts/detailed/training_dynamics_e1_e5.png)

观察：

- Epoch 1 的 loss 快速下降，说明 LoRA r32 容量足以吸收当前数据中的主要行为模式。
- Epoch 3-5 的 loss 仍缓慢下降，但收益越来越小；与此同时 gradient norm 的中位数和 P95 略有上升，表示剩余误差更集中在较难样本上，并非训练完全静止。
- 每轮都重复同样的 1071 条样本。五轮合计约 3792.82 万 context token、354.78 万 supervised token，但独立监督内容仍只有单轮约 70.96 万 supervised token。
- 训练显存峰值约 79.18 GiB allocated、最高约 94.15 GiB reserved，当前 96 GiB 卡配置接近合理上限。

### 4.3 过拟合信号

图表：[Loss 与评测分叉](charts/detailed/loss_vs_evaluation_divergence.png)

| 指标 | Epoch 1 | Epoch 3 | Epoch 5 | E5-E3 |
|---|---:|---:|---:|---:|
| Train loss | 0.2849 | 0.1898 | 0.1717 | -0.0180 |
| Overall P@1 | 55.93% | 60.74% | 59.26% | -1.48 pp |
| Overall P@3 | 66.67% | 72.22% | 71.11% | -1.11 pp |
| SWE-smith P@1 | 11.11% | 27.78% | 22.22% | -5.56 pp |
| SWE-smith P@3 | 20.00% | 43.33% | 40.00% | -3.33 pp |

Epoch 5 的 loss 更低但评测更差，已经满足一个实用意义上的 early-stopping 信号。由于只有 90 道评测题，1-2 个百分点的总体差异可能包含采样波动；但 SWE-smith P@1 的 5.56 个百分点回落相当于约 5 个任务级 Pass@1 贡献，方向与 max_steps 增加一致，应认真对待。

## 5. 正式评测结果

### 5.1 Overall

| 模型 | Pass@1 | Pass@2 | Pass@3 | Resolved rollout |
|---|---:|---:|---:|---:|
| Qwen3.5 4B | 66.67% | 77.04% | 80.00% | 180/270 |
| Qwen3 4B Base | 52.59% | 62.59% | 66.67% | 142/270 |
| Epoch 1 | 55.93% | 63.33% | 66.67% | 151/270 |
| **Epoch 3** | **60.74%** | **68.52%** | **72.22%** | **164/270** |
| Epoch 5 | 59.26% | 68.15% | 71.11% | 160/270 |

### 5.2 分数据集

| 模型 | HE P@1/P@3 | MBPP P@1/P@3 | SWE-smith P@1/P@3 |
|---|---:|---:|---:|
| Qwen3.5 4B | 78.89% / 93.33% | 80.00% / 93.33% | 41.11% / 53.33% |
| Qwen3 4B Base | 71.11% / 83.33% | 67.78% / 86.67% | 18.89% / 30.00% |
| Epoch 1 | 77.78% / 93.33% | 78.89% / 86.67% | 11.11% / 20.00% |
| **Epoch 3** | **76.67% / 90.00%** | **77.78% / 83.33%** | **27.78% / 43.33%** |
| Epoch 5 | 77.78% / 90.00% | 77.78% / 83.33% | 22.22% / 40.00% |

### 5.3 相对 Base 的增益

| 指标 | Epoch 1 | Epoch 3 | Epoch 5 |
|---|---:|---:|---:|
| Overall P@1 | +3.33 pp | **+8.15 pp** | +6.67 pp |
| Overall P@3 | +0.00 pp | **+5.56 pp** | +4.44 pp |
| HumanEval+ P@1 | +6.67 pp | +5.56 pp | +6.67 pp |
| HumanEval+ P@3 | **+10.00 pp** | +6.67 pp | +6.67 pp |
| MBPP+ P@1 | **+11.11 pp** | +10.00 pp | +10.00 pp |
| MBPP+ P@3 | +0.00 pp | -3.33 pp | -3.33 pp |
| SWE-smith P@1 | -7.78 pp | **+8.89 pp** | +3.33 pp |
| SWE-smith P@3 | -10.00 pp | **+13.33 pp** | +10.00 pp |

图表：[Checkpoint 差值热力图](charts/detailed/checkpoint_delta_heatmap.png)

这张表揭示了一个重要训练阶段变化：

- Epoch 1 先学会短函数题，尤其 Pass@1；但仓库级规划和步数管理变差。
- Epoch 3 才把函数级能力和仓库级能力重新平衡起来。
- Epoch 5 没有继续扩大能力边界，只是在既有分布上继续降低训练 loss。

### 5.4 与 Qwen3.5 4B 的差距

Epoch 3 是最接近 Qwen3.5 的版本：

| 指标 | Epoch 3 - Qwen3.5 |
|---|---:|
| Overall P@1 | -5.93 pp |
| Overall P@3 | -7.78 pp |
| HumanEval+ P@1 | -2.22 pp |
| HumanEval+ P@3 | -3.33 pp |
| MBPP+ P@1 | -2.22 pp |
| MBPP+ P@3 | -10.00 pp |
| SWE-smith P@1 | -13.33 pp |
| SWE-smith P@3 | -10.00 pp |

函数级 Pass@1 已经很接近 Qwen3.5；主要差距集中在：

1. SWE-smith 的单次成功率；
2. MBPP+ 的三次采样覆盖率；
3. 整体仍有更多“完全不会做”的任务。

## 6. 稳定性与 Pass@3 的价值

由 Pass@1/2/3 可以精确反推出 90 道题中分别有 0/1/2/3 次成功的任务数：

| 模型 | 0/3 | 1/3 | 2/3 | 3/3 | P@3-P@1 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5 4B | 18 | 8 | 20 | 44 | 13.33 pp |
| Qwen3 4B Base | 30 | 11 | 16 | 33 | 14.07 pp |
| Epoch 1 | 30 | 9 | 11 | 40 | 10.74 pp |
| Epoch 3 | 25 | 10 | 11 | 44 | 11.48 pp |
| Epoch 5 | 26 | 8 | 16 | 40 | 11.85 pp |

图表：[三次 rollout 一致性](charts/detailed/task_consistency_distribution.png)

解释：

- Epoch 3 有 44 道题三次全过，与 Qwen3.5 相同，说明它在“已经学会的题”上相当稳定。
- 但 Epoch 3 仍有 25 道题三次全失败，而 Qwen3.5 只有 18 道。当前差距主要是能力覆盖面，而不是已掌握题目的随机性。
- Pass@3 相比 Pass@1 为 Epoch 3 增加 11.48 个百分点，说明多采样仍有价值，但它不能解决这 25 道稳定失败任务。

## 7. 失败模式

| 模型 | Resolved | Tests failed | No edit | Max steps | Parse | Context | Other |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5 4B | 180 | 17 | 23 | 11 | 4 | 0 | 35 |
| Qwen3 4B Base | 142 | 46 | 46 | 1 | 2 | 1 | 32 |
| Epoch 1 | 151 | 35 | 4 | 46 | 3 | 6 | 25 |
| Epoch 3 | 164 | 41 | 1 | 21 | 4 | 3 | 36 |
| Epoch 5 | 160 | 38 | 2 | 31 | 3 | 1 | 35 |

图表：[失败模式热力图](charts/detailed/failure_mode_heatmap.png)

主要变化：

- Base 的核心问题是 no-edit：46/270。Epoch 1 将它降到 4，说明 SFT 对“调用编辑工具并继续工作”的行为监督非常有效。
- 但 Epoch 1 的 max-steps 从 1 激增到 46，失败没有完全消失，而是从“不动手”迁移成“会动手但无法在预算内闭环”。
- Epoch 3 将 max-steps 降到 21，同时 resolved 增至 164，是行为链条真正变完整的阶段。
- Epoch 5 的 max-steps 又回到 31，和 SWE-smith 回落一致。模型可能在重复训练后形成更长、更犹豫或更频繁检查的策略。
- `other` 是汇总中未进一步细分的 unresolved 状态，不能直接解释为同一种错误；需要原始 sample_result 才能继续拆分。

## 8. 统计不确定性

图表：[Overall 近似 95% 区间](charts/detailed/overall_uncertainty_intervals.png)

基于 90 个任务级 Pass@k 贡献的普通正态近似：

| 模型 | P@1 95% 区间 | P@3 95% 区间 |
|---|---:|---:|
| Qwen3.5 4B | 58.54%-74.80% | 71.69%-88.31% |
| Qwen3 4B Base | 43.71%-61.47% | 56.87%-76.46% |
| Epoch 1 | 46.70%-65.15% | 56.87%-76.46% |
| Epoch 3 | 51.77%-69.72% | 62.92%-81.53% |
| Epoch 5 | 50.39%-68.13% | 61.69%-80.53% |

这些区间说明：

- Epoch 3 与 Epoch 5 的小幅差异不能只凭单次总体比例断言具有统计显著性。
- 但 loss 持续下降、SWE-smith 和 max-steps 同时向坏方向变化，构成了更强的联合 early-stop 证据。
- 最理想的统计方式是使用同一任务的原始结果做 paired bootstrap，并报告每个 checkpoint 相对 Base 的 task-level win/tie/loss。当前本地只归档了汇总 JSON，无法完成配对分析。

## 9. 当前模型学到了什么

综合行为与结果，当前 SFT 至少学到了四类能力：

1. **工具协议与结构化输出。** Parse error 始终很低，表明严格清洗和 assistant-only mask 没有破坏工具调用格式。
2. **从观察进入修改。** No-edit 从 Base 的 46 大幅下降，说明编辑行为被明确强化。
3. **函数级实现模板。** HumanEval+/MBPP+ Pass@1 在第一轮就接近 Qwen3.5 4B。
4. **更长的仓库级闭环。** Epoch 3 开始把定位、编辑、测试和修复串起来，SWE-smith P@3 比 Base 高 13.33 个百分点。

尚未充分学到的部分：

- 对未见项目和错误形态的迁移；
- 在 60 步内及时停止探索并完成编辑/验证；
- 对同一仓库问题的多样化可行路径；
- 使当前 25 道“3/3 全失败”任务进入至少一次成功区域。

## 10. 下一阶段建议

### 10.1 Checkpoint 策略

- 将 **Epoch 3 固定为当前 champion**，保留 Epoch 5 作为过拟合对照。
- 不在现有 1071 条数据上直接继续 Epoch 6+。
- 后续每 0.5-1 个 epoch 做一次冻结评测；early stop 以 Overall P@1、SWE-smith P@1/P@3、max-steps 比例联合判断，而不是只看训练 loss。

### 10.2 数据扩展优先于 Rank 64

下一轮优先把“独立仓库级成功轨迹”扩大到当前的 2-4 倍，并重点增加：

- 新项目、新依赖体系、新测试框架；
- 当前评测中 3/3 全失败或 max-steps 高频的错误类型；
- 短而完整的定位 -> 编辑 -> 测试 -> 修复轨迹；
- 失败后恢复成功的路径，而不是只收集一次成功的直线路径；
- 多种合法编辑策略，降低单一风格记忆。

Rank 32 已足以让行为显著变化。Rank 64 可以作为严格控制变量的小规模 ablation，但优先级低于数据覆盖；若数据不变，Rank 64 更可能更快拟合训练集，而不是自动提高仓库级泛化。

### 10.3 重新设计采样分布

建议同时按“行数、context token、supervised token、长度桶”四个维度配平：

- 函数题保留为格式与基础代码能力锚点，但不再继续重复放大。
- SWE-smith 在行数上增加，同时控制单条长轨迹对 update 的占用。
- 对 16k-32k 轨迹做长度分层采样；超过 32k 的轨迹必须有显式截断审计。
- 可以构建两阶段 curriculum：先短闭环轨迹稳定工具行为，再混入长仓库级轨迹。

### 10.4 推荐的下一轮训练方案

一个保守、可归因的方案：

1. 以 Epoch 3 adapter 为起点。
2. 新增至少 1000-2000 条经过 verifier 的独立仓库级成功轨迹，旧数据只作为 replay。
3. 新旧数据按 supervised token 而非样本数混合，旧数据权重约 25%-40%。
4. 继续使用 LoRA r32；LR 从 5e-6 左右起步，短 warmup，先训练 1 个 epoch-equivalent。
5. 每 0.5 epoch 保存 adapter，并在固定 V3 子集上评测；一旦 SWE-smith 或 Overall 连续下降就停止。
6. 只有当新数据下 r32 的 train/eval 同时欠拟合，才测试 r64。

### 10.5 可进一步构造的图与所需数据

本次已生成 9 张新增图。若补齐原始 sample_results 和逐轨迹训练 manifest，还可以继续构造：

| 图表 | 作用 | 额外需要的数据 |
|---|---|---|
| Task-level win/tie/loss matrix | 直接看 E3 赢 Base 的具体题 | 每个任务每次 rollout 的成功标记 |
| Paired bootstrap delta forest plot | 给 checkpoint 增益置信区间 | 同上 |
| First-edit-step / total-step 分布 | 验证 max-steps 是否来自迟迟不编辑 | 每条轨迹工具步序列 |
| 工具调用 Sankey/转移矩阵 | 比较 Base/E1/E3 的行动策略 | 标准化 tool-call 序列 |
| 成功率 vs context length | 判断长上下文是否拖累模型 | 每条评测的 token 长度 |
| 成功率 vs repository/project | 找项目记忆与迁移盲区 | task 的 repo 标签 |
| 训练 token 来源随 update 变化 | 检查 batch 分布偏移 | 每个 microbatch 的 trajectory id |
| LoRA delta norm 按层热力图 | 判断 rank 是否成为瓶颈 | 各 checkpoint adapter 权重 |

## 11. 图表目录

新增图均同时输出 PNG 和 PDF：

1. `task_consistency_distribution`
2. `checkpoint_delta_heatmap`
3. `training_data_composition`
4. `token_length_buckets`
5. `training_dynamics_e1_e5`
6. `loss_vs_evaluation_divergence`
7. `failure_mode_heatmap`
8. `capability_tradeoff_map`
9. `overall_uncertainty_intervals`

可复现脚本：[build_detailed_analysis.py](build_detailed_analysis.py)  
结构化分析数据：[detailed_analysis_data.json](detailed_analysis_data.json)

## 12. 最终判断

这轮 SFT 是成功的：它把 Qwen3 4B Base 的 Overall Pass@1 从 52.59% 提升到 60.74%，并把 SWE-smith Pass@3 从 30.00% 提升到 43.33%。同时，函数级 Pass@1 已经接近 Qwen3.5 4B。

但当前结果还不等于“全面达到 Qwen3.5 4B”：Epoch 3 在 SWE-smith Pass@1 仍低 13.33 个百分点，并有 25/90 道题三次全失败。下一步最有价值的投入不是继续重复现有五轮，也不是先把 LoRA rank 翻倍，而是扩大高质量仓库级轨迹的覆盖面、补齐 task-level 评测数据，并用更频繁的 checkpoint 评测做 early stop。
