# SAO Value Model 训练流程

## 1. 目标与边界

这一步是在线 SAO 之前的 Critic 冷启动。输入是 Actor 生成的完整 agent
轨迹，监督信号是 verifier 在轨迹结束后给出的二元结果：成功为 `1`，失败为
`0`。它不执行 rollout、不更新 Actor，也不计算 GAE；中间步骤没有独立的
reward。

模型由三部分组成：

1. Qwen3-4B-Instruct-2507 基座。
2. Epoch3 Actor LoRA，其中 attention LoRA 全部冻结，只训练 MLP LoRA。
3. 隐状态到标量的 float32 线性 Value Head。

Value Head 权重以 `N(0, 1e-3)` 初始化，bias 使用训练集 terminal return
先验初始化。

## 2. 数据构建

`build_value_dataset.py` 对每条轨迹应用模型原生 chat template，并同时验证：

- message-wise tokenization 与整段 tokenization 完全一致；
- `input_ids` 与 `action_mask` 长度一致；
- system、user、tool observation 和 assistant header 的 mask 为 `0`；
- assistant action token 及其 end-of-turn token 的 mask 为 `1`；
- train/validation 按 task 或 repository 分组隔离；
- 超过 65,536 tokens 的轨迹不截断，单独隔离。

本次混合源共 989 条，保留 988 条，隔离 1 条 69,991-token 轨迹。混合数据
中的 888 条用于训练。正式门禁复用旧数据中的固定 40 条 holdout，其中
Function 20 条、SWE-smith 20 条。

## 3. Loss

对轨迹 `i`，记 assistant action token 集合为 `A_i`，terminal verifier return
为 `R_i in {0, 1}`，Value Model 在 token `t` 的输出为 `V_theta(s_it)`。单轨迹
loss 为：

```text
L_i(theta) = (1 / |A_i|) * sum_{t in A_i} (V_theta(s_it) - R_i)^2
```

一个 optimizer update 含 `N` 条轨迹，update loss 为：

```text
L_update(theta) = (1 / N) * sum_{i=1..N} L_i(theta)
```

因此每条轨迹权重相同，不会因为长轨迹 action token 更多而获得更大权重。
terminal return 会监督该轨迹的每个 action token，但这不等于拥有中间 reward；
它只是用最终结果训练每个前缀状态的成功概率估计。

## 4. 长度分桶与梯度累积

`critic_96gb_verified` 的边界和 microbatch size 为：

| Token 区间 | Microbatch |
| --- | ---: |
| 1-4,096 | 8 |
| 4,097-8,192 | 8 |
| 8,193-16,384 | 4 |
| 16,385-32,768 | 2 |
| 32,769-49,152 | 2 |
| 49,153-65,536 | 1 |

每个 microbatch 内只包含同一长度桶的轨迹。多个 microbatch 可以依次反向传播，
累计到 8 条轨迹后执行一次 optimizer step。888 条训练轨迹因此对应 111 个
optimizer updates。不同桶的单步 loss 分布不同，不应直接把相邻两步的变化解释
为训练趋势。

## 5. 优化配置

- Epoch: 1
- MLP LoRA LR: `5e-6`
- Value Head LR: `5e-5`
- Constant schedule with warmup: 10 updates
- Gradient clipping: global norm `1.0`
- Forward autocast: bfloat16
- Value Head parameters: float32
- Seed: `20260823`
- Periodic checkpoint: disabled in the recorded run

训练日志逐 update 记录 trajectory-mean loss、未裁剪 gradient norm、两个参数组
的 LR、桶、real/padded/action tokens、reward 组成和 GPU 显存。

## 6. 评估门禁

正式评估分别计算：

- trajectory mean：一条轨迹所有 action-token value 的均值；
- trajectory final token：最后一个 action token 的 value；
- token weighted：所有 action token 直接合并。

门禁使用 trajectory mean，并要求每个 family 同时满足：

1. AUC > 0.5；
2. Brier 低于该 family 的训练先验常数预测；
3. Explained Variance > 0；
4. MLP LoRA 与 Value Head 都出现非零梯度。

本次 SWE-smith 通过；Function 的 AUC 与 EV 通过，但 Brier `0.1996` 未优于
常数基线 `0.1876`，所以全 family 总门禁为 false。

## 7. 复现顺序

```bash
cd sao
pip install -e '.[test]'
./scripts/unpack_data.sh

BASE_MODEL=/path/to/Qwen3-4B-Instruct-2507 \
ACTOR_ADAPTER=/path/to/epoch_003_adapter \
./scripts/run_qwen3_4b_value_training.sh
```

脚本先运行 selection-only，确认分桶计划和 111 updates；再执行 before eval、
训练、after eval、门禁和最终 checkpoint。已有输出目录不会被覆盖。

## 8. 与在线 SAO 的衔接

最终 checkpoint 中需要加载：

- `critic_adapter/adapter_model.safetensors`
- `value_head.safetensors`

`optimizer.pt`、`scheduler.pt`、`rng_state.pt` 和 `trainer_state.json` 用于继续
Value Model 训练，不是在线 Actor 更新的替代品。真正的 SAO 还需要在线 rollout、
基于 Critic 的 step-level advantage/return 构造和 Actor objective；本目录没有把
普通 GRPO 冒充为 SAO。
