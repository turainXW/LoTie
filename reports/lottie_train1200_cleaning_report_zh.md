# Lottie Code Agent 训练轨迹清洗与交付报告

## 1. 数据范围

- 原始运行：`deepseek_v4_flash_train300_rollout4_v1`
- 数据角色：`train`
- 任务数：300
- 每题采样数：4
- 有效 rollout：1200
- Verifier 成功：1090
- Verifier 失败：110
- 数据集构成：HumanEval+ 320 条、MBPP+ 480 条、SWE-smith Python 400 条
- 评测指标：pass@1 90.83%、pass@2 93.11%、pass@3 93.67%

清洗过程不修改原始运行目录。所有输出写入新的版本化目录，协议恢复候选在处理前执行逐文件备份和 SHA-256 校验。

## 2. 清洗目标

本次清洗分别服务于两种训练方式：

1. **SFT**：只学习已通过 verifier、工具调用协议可靠、上下文与工具反馈链完整的 assistant 行为。
2. **RL/偏好分析**：保留全部 1200 条 verifier 可判定 rollout，并使用二值 reward 表示最终任务是否通过。

原始轨迹、SFT 样本和 RL rollout 不混为同一种准入标准。Verifier 失败轨迹对 RL 有价值，但默认不能作为正向 SFT 标签。

## 3. 清洗流水线

### 3.1 完整性检查

对 1200 个 `(instance_id, sample_index)` 槽位执行以下检查：

- 300 道任务是否均有 4 个 rollout；
- 是否存在重复槽位；
- `sample_result.json`、轨迹 JSONL、SFT messages 和 verifier 结果是否存在且非空；
- `sample_valid` 和 `benchmark_resolved` 是否为可判定布尔值；
- 轨迹消息是否满足 `system/user/assistant + content` 基础结构。

结果：1200 个槽位完整，无重复，最终无基础设施 blocked 样本。

### 3.2 隐私与路径规范化

- 将本机 workspace、项目根目录和用户主目录替换为稳定占位符；
- 扫描常见 API Key 形态并替换为 `<redacted-secret>`；
- 本批次命中密钥数为 0；
- 不把绝对路径作为模型要记忆的目标内容。

### 3.3 严格 SFT 准入

严格样本必须同时满足：

- `benchmark_resolved=true`；
- 不属于 `model_api_failure`、`parse_error`、`runner_error`、`context_overflow`、`infrastructure_blocked`；
- 每个 assistant 消息都能解析为规范 JSON Tool Calling；
- 工具名和参数结构符合 harness 声明；
- 非 `finish` 调用后存在对应环境反馈；
- 最后一条 assistant 消息为 `finish`；
- 对话 SHA-256 不与已有严格样本重复。

结果：996 条进入严格 SFT；共验证 11,147 个 assistant JSON 工具调用；精确重复为 0。

### 3.4 协议恢复（Salvage）

1090 条 verifier 成功轨迹中，有 94 条因中间出现非 JSON 文本、转义错误、JSON 截断或错误工具字段而未进入严格集。

自动恢复只允许删除已证明无环境副作用的成对消息：

- 非法 assistant 输出 + 紧随其后的 parser 拒绝反馈；
- 过早 `finish` + 紧随其后的 completion gate 拒绝反馈；
- schema 非法/未知工具调用 + 紧随其后的 runtime 协议拒绝反馈。

不根据上下文猜测工具名，不重写 arguments，不伪造工具执行结果。若无法证明错误调用未执行，则整条轨迹进入人工审查。

恢复结果：

- 候选：94 条
- 自动恢复：90 条
- 人工审查隔离：4 条
- 删除无副作用的非法 assistant/parser 对：129 对
- 删除过早 finish/gate 对：2 对
- 删除 runtime 明确拒绝的协议调用对：6 对
- 补充规范终止 `finish`：11 条

错误类型统计：

| 类型 | 次数 |
|---|---:|
| 非 JSON 文本 | 70 |
| 非法转义 | 26 |
| 缺少分隔符 | 19 |
| 字符串截断 | 13 |
| 非法控制字符 | 4 |
| 其他无效 JSON | 1 |

### 3.5 独立审计与分层

恢复后的 90 条轨迹由独立审计脚本重新检查，不直接信任恢复脚本的结论：

- 90/90 通过协议完整性审计；
- 662 个备份文件、43,635,498 字节全部通过 SHA-256 回验；
- 每个 assistant 调用重新按当前工具 schema 校验；
- 每个工具调用与后续环境反馈重新对齐；
- verifier 成功状态与源元数据重新核对。

质量分层：

| 等级 | 数量 | 默认策略 |
|---|---:|---|
| S（严格原生） | 996 | 主 SFT，权重 1.0 |
| A（低风险恢复） | 73 | 主 SFT，权重 1.0 |
| B（含合成 finish 或过早 finish 修复） | 13 | 可选加入，建议权重 0.5 |
| C（移除超过 3 个 parser 错误对） | 4 | 隔离，仅消融/人工复核 |
| Manual Review（恢复证据不足） | 4 | 隔离，不训练 |

因此默认主 SFT 为 1069 条（996 S + 73 A）；可选低权重扩展 13 条 B。C 和 Manual Review 不进入默认训练。

### 3.6 RL 数据

`rl_rollouts.jsonl` 保留全部 1200 条有效 rollout：

- reward=1：1090 条；
- reward=0：110 条；
- 保留 agent status、verifier status、patch、工具步数、token usage 和完整消息链；
- 模型失败与 verifier 失败不被重跑成正样本。

该文件适合后续做 outcome reward 分析、拒绝采样、DPO 配对候选生成或 verifier 驱动 RL，不应直接整体当作正向 SFT。

## 4. 交付文件

| 文件 | 用途 |
|---|---|
| `data/sft_primary.jsonl` | 默认 SFT：严格 S + 恢复 A，共 1069 条 |
| `data/sft_optional_tier_b.jsonl` | 可选低权重 SFT，共 13 条 |
| `data/sft_quarantine_tier_c.jsonl` | 隔离样本，共 4 条，默认禁止训练 |
| `data/rl_rollouts.jsonl` | 全部 1200 条带二值 reward 的 rollout |
| `audit/sft_manual_review.jsonl` | 4 条待人工审查轨迹 |
| `audit/repair_audit.jsonl` | 每条恢复操作及删除理由 |
| `audit/trajectory_audit.jsonl` | 独立审计结果和质量等级 |
| `provenance/salvage_raw_backup/` | 94 条恢复候选的字节级原始备份 |
| `metadata/*.json` | 原始运行、清洗、恢复和评测 manifest |
| `metadata/file_manifest.jsonl` | 交付包逐文件大小和 SHA-256 |
| `SHA256SUMS` | 上传后完整性校验 |

## 5. Tokenizer 与 Loss Mask

本地交付的是清洗后的 message 数据，不提前绑定 tokenizer。应在训练服务器上使用目标模型的**准确 tokenizer 和 chat template**执行：

1. 将完整多轮 messages 渲染并 tokenize；
2. system、user、环境/tool feedback、padding 的 label 设为 `-100`；
3. 仅 assistant 的 JSON Tool Calling 与终止 `finish` token 参与 SFT loss；
4. 校验 assistant span 边界，不能仅按字符长度切 mask；
5. 超过模型上下文上限的样本单独统计，不静默截断关键工具调用或反馈。

当前轨迹长度范围为 13-83 条消息、约 11,605-148,205 字符。最终 token 长度必须由 Qwen3 4B 实际 tokenizer 重新测量。

## 6. 训练建议

第一阶段建议只使用 `sft_primary.jsonl`，观察工具 JSON 合法率、任务成功率和长度分布。第二阶段可加入 Tier B，并使用 `sample_weight=0.5` 或降低采样概率。Tier C 与人工审查集暂不进入正式 SFT。

推荐训练前最后执行：

- Qwen3 4B tokenizer 长度统计；
- assistant-only mask 单元测试和随机样本可视化；
- 训练/评测 task ID 去重检查；
- 按 HumanEval+、MBPP+、SWE-smith 分层采样，避免函数题压过仓库工具能力；
- 保留 110 条失败 rollout，供后续偏好训练或 RL 使用。

## 7. 边界与限制

- Verifier 成功证明测试集合通过，不等价于所有潜在需求都被形式化覆盖；
- Salvage 只恢复协议噪声，不修复错误推理或错误代码；
- 11 条合成 finish 没有伪造代码操作，但会改变终止动作的来源，因此被降级到 B 或按审计规则标记；
- 当前尚未执行目标 tokenizer 的 token 上限过滤和 assistant-only token mask；
- 原始 1.5 GB workspace 不进入训练交付包，只保留恢复候选所需的可审计证据。
