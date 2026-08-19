# Qwen3-4B Code Agent SFT Mask V2

## 目标

在保留完整多轮执行上下文的前提下，只对可学习的 assistant 工具调用计算 loss。system、user、工具反馈、重叠窗口中的历史 assistant，以及 schema 不合法的 assistant 调用全部使用 `-100` 屏蔽。

## Mask 规则

- 输入始终保留完整的 `system -> user/task -> assistant/tool call -> tool feedback` 链路。
- 只监督合法 assistant JSON 的正文、`<|im_end|>` 和换行；assistant role header 不参与 loss。
- system、user 和工具反馈全部作为条件上下文，不参与 loss。
- 长轨迹按完整 assistant/feedback 单元切成 32K 窗口；重叠窗口中的历史 assistant 只作为上下文，避免重复训练。
- JSON 可解析但工具名或参数 schema 不合法的 assistant turn 整体 mask；对应错误反馈和后续恢复动作仍保留。
- `--keep-invalid-tool-calls` 只用于复现旧版本，不建议用于正式训练。

V2 与 V1 的 `input_ids`、`attention_mask` 完全一致。V2 仅在 24 个窗口中把 945 个非法调用 token 的 label 从 token id 改为 `-100`，没有新增任何监督 token。

## 数据审计结果

主训练集：

- 1,069 条源轨迹，切分为 1,071 个 32K 窗口。
- 7,585,639 个上下文 token，709,568 个监督 token。
- 12,604 个合法 assistant turn 参与训练。
- 24 个 schema 非法调用和 4 个重叠窗口历史调用被 mask。
- 独立审计通过：无 role 泄漏、无部分 assistant mask、无非法受监督 JSON、无长度越界。

可选 Tier-B 集：

- 13 条源轨迹，15 个窗口。
- 486 个 assistant turn，35,182 个监督 token。
- 独立审计通过；首轮训练不与主训练集混合。

## 重新构建与审计

```bash
python tools/prepare_qwen_sft_dataset.py \
  --input /path/to/strict_sft.jsonl \
  --model-path /path/to/Qwen3-4B-Instruct-2507 \
  --output-dir /path/to/qwen3_4b_instruct_2507_primary_32k_maskv2/dataset \
  --max-length 32768 \
  --overlap-units 2

python tools/audit_qwen_sft_arrow.py \
  --dataset-dir /path/to/qwen3_4b_instruct_2507_primary_32k_maskv2/dataset \
  --model-path /path/to/Qwen3-4B-Instruct-2507 \
  --report /path/to/qwen3_4b_instruct_2507_primary_32k_maskv2/mask_audit_v2.json
```

训练入口拒绝缺失审计报告、审计失败、行数不一致或监督 token 数不一致的数据。

## RTX PRO 6000 96GB 验证配置

已使用 BF16、SDPA、LoRA `r=32/alpha=64`、gradient checkpointing 完成各长度桶和混合长度测试。默认分桶为：

| 最大长度 | micro batch |
| --- | ---: |
| 4K | 8 |
| 8K | 2 |
| 16K | 1 |
| 24K | 1 |
| 32K | 1 |

10-update 混合测试结果：20 个 micro-step 全部成功，LoRA 梯度非零，约 3,118 token/s，峰值 allocated 78.97 GiB、reserved 85.40 GiB。标准 Transformers 全词表交叉熵下，更激进的 64K 总 token micro batch 会 OOM，因此不使用。

## 正式训练

```bash
screen -dmS lottie_sft_train bash -lc '
python scripts/train_qwen_lottie_lora.py \
  --dataset-dir /path/to/qwen3_4b_instruct_2507_primary_32k_maskv2/dataset \
  --model-path /path/to/Qwen3-4B-Instruct-2507 \
  --output-dir /path/to/runs/qwen3_4b_lottie_lora_maskv2_r32_e2 \
  --epochs 2 \
  --gradient-accumulation-steps 2 \
  --learning-rate 5e-5 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --save-every-epoch \
  --batch-profile rtx_pro_6000_verified \
  --attn-implementation sdpa \
  > /path/to/runs/qwen3_4b_lottie_lora_maskv2_r32_e2/train.log 2>&1'
```

训练每个 epoch 约 457 个 micro batch、229 次参数更新；两轮约 458 次更新。`--save-every-epoch` 会依次保存 `epoch_001_adapter` 和 `epoch_002_adapter`，脚本另外保存最终 adapter，并记录真实 token、padding token、监督 token、loss、吞吐和显存峰值。
