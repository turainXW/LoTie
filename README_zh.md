# LoTie：代码智能体轨迹采集、SFT 与评测框架

LoTie 读作 **“洛蒂”**（英语近似 **LOH-tee**）。它是一个面向真实代码修复任务的
轻量 Code Agent Runtime 与数据闭环：从任务切分、工具调用、代码修改、测试验证和
轨迹采集，一直覆盖到可审计清洗、assistant-only Loss Mask、长序列分桶 SFT 和 Pass@k
评测。

[English README](README.md) | [完整项目技术报告](reports/Lottie_Code_Agent_FULL_PROJECT_REPORT_ZH.md)

## 核心能力

- JSON Tool Calling 驱动的 Agent Loop，支持 Shell、结构化编辑、Repo Context、测试和 Patch。
- MBPP+、HumanEval+、SWE-smith 的确定性训练/验证/评测切分，Gold 不进入模型上下文。
- 并发、可断点恢复的多 rollout 采集，区分模型失败、Verifier 失败和基础设施异常。
- Strict、Salvage、Quarantine 三层可审计清洗，保留原始元数据和修复记录。
- Qwen 原生 Chat Template、assistant-only Mask，以及 4K-32K 长度分桶 LoRA SFT。
- EvalPlus 与 SWE-smith Verifier 闭环，生成 Pass@1、Pass@2、Pass@3 和失败归因报告。

```text
任务切分
  -> 隔离工作区 / 函数沙箱
  -> Agent 工具循环
  -> Patch + 测试 + Verifier
  -> 原始轨迹与 rollout 元数据
  -> Strict / Salvage / Quarantine 清洗
  -> 目标模型 Tokenizer + assistant-only Mask
  -> 长度分桶 SFT
  -> 统一口径 Pass@k 评测
```

仓库不包含模型权重、原始任务工作区、虚拟环境、API 密钥和 SSH 配置。

## 快速开始

需要 macOS 或 Linux、Python 3.10+、Git，以及兼容 OpenAI Chat Completions API 的
模型服务。

```bash
git clone git@github.com:turainXW/LoTie.git
cd LoTie

# 最小安装；不带参数默认安装 dev + SWE 相关依赖。
bash scripts/install_dev.sh base
source scripts/dev_env.sh

lotie --help
python3 scripts/run_smoke.py --output-dir data/code_agent_smoke
```

安装脚本会创建 `.venv`、以 editable 模式安装项目、初始化 `.codeagent/`，并从
`.env.example` 生成本地 `.env`。`.env` 已被 Git 忽略。

### 执行一次真实仓库任务

```bash
lotie use \
  --repo /path/to/target-repository \
  --task "修复失败的 parser 测试并验证最小 Patch" \
  --sandbox copy \
  --model-url http://127.0.0.1:8000/v1 \
  --model local-model \
  --max-steps 40 \
  --show-steps
```

默认推荐 `--sandbox copy`，Agent 只修改隔离副本；只有明确需要原地修改时才使用
`--sandbox direct`。主 CLI 是 `lotie`，同时保留 `minicoder` 和 `codeagent` 兼容入口。

## 轨迹采集与 Pass@k

先生成正式实验计划，不实际执行：

```bash
lotie pass-at-k \
  --selection data/dataset_splits_v1/selection.json \
  --split evaluation \
  --samples 3 \
  --workers 4 \
  --output-dir outputs/eval90_pass3 \
  --plan-only
```

正式运行前请执行 `lotie pass-at-k --help`。采集器会记录模型与采样参数、任务切分、
并发策略、Verifier 超时和有效槽位；中断后只跳过已完成的 `sample_valid=true` 槽位，
不会覆盖有效结果。

运行文档：

- [macOS 可移植 SWE-smith 环境与轨迹采集](docs/MACOS_PORTABLE_TRAJECTORY_COLLECTION_ZH.md)
- [Eval90 部署](DEPLOY_EVAL90_zh.md)
- [通用部署说明](DEPLOY_zh.md)

## 随仓库提供的数据

`data/trajectories/lottie_train1200_cleaned_complete_v1/` 是脱敏、压缩并带审计记录的
训练数据交付包：

| 数据 | 数量 | 默认用途 |
| --- | ---: | --- |
| 有效 rollout | 1,200 | RL/Reward 分析，保留真实模型失败 |
| Verifier 通过 rollout | 1,090 | 成功行为分析 |
| Strict SFT | 996 | 最高置信度 SFT 数据 |
| Primary SFT | 1,069 | Strict + 独立审计通过的 Tier-A Salvage |
| Optional Tier-B | 13 | 可选低权重实验 |

数据没有预先 Tokenize。应使用目标模型的精确 Tokenizer 与 Chat Template：

```bash
python3 tools/prepare_qwen_sft_dataset.py --help
python3 tools/audit_qwen_sft_arrow.py --help
python3 scripts/train_qwen_lottie_lora.py --help
```

训练时只监督合法 assistant turn；system、user、工具返回、非法协议文本、padding 和
长窗口重叠历史都不计算 Loss。

## 参考评测结果

仓库内报告采用统一 Harness 与采样口径，对 90 道评测题分别执行 3 次 rollout。本轮
Qwen3-4B LoRA 的最佳 checkpoint 是 Epoch 3：

| 模型 | Overall P@1 | P@2 | P@3 | SWE-smith P@1 | P@2 | P@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B Base | 52.59% | 62.59% | 66.67% | 18.89% | 26.67% | 30.00% |
| LoTie Epoch 3 | 60.74% | 68.52% | 72.22% | 27.78% | 37.78% | 43.33% |

完整口径和图表见[评测报告](reports/qwen_v3_eval_20260816/README.md)。本机 venv 方式的
SWE-smith 结果标记为 `official_comparable=false`；需要榜单可比结论时应使用官方容器
Verifier。

## 目录结构

```text
LoTie/
├── src/code_agent_baseline/   # Agent Loop、工具、上下文、模型客户端、CLI
├── scripts/                   # 采集、评测、部署和训练脚本
├── tools/                     # 清洗、审计、Tokenize 与数据打包
├── benchmarks/                # 单元测试和集成测试
├── configs/                   # Runtime 默认配置
├── data/
│   ├── dataset_splits_v1/     # 确定性 train/validation/eval 切分
│   ├── evalplus_local/        # 脱敏函数级任务
│   ├── swesmith_local/        # 脱敏仓库级任务
│   └── trajectories/          # 压缩后的清洗轨迹交付包
├── reports/                   # 训练/评测指标、图表和分析报告
├── docs/                      # 运行与迁移文档
├── requirements-*.txt         # 分层依赖环境
└── pyproject.toml             # 包信息与 CLI 入口
```

运行产生的 `outputs/`、`dist/`、`.venv/`、`.codeagent/`、日志和工作区不会提交。

## 验证

```bash
source scripts/dev_env.sh
PYTHONPATH=src:benchmarks/leetcode_top10 \
  python -m unittest discover -s benchmarks -p 'test_*.py'
```

模型行为状态与最终 Verifier 结果分开记录。测试失败、`no_edit`、`max_steps` 和解析失败
保留为真实模型结果；只有明确的 API、进程、OOM 或 Verifier 基础设施故障可以重试。

## 数据与授权说明

- Gold Patch、Canonical Solution 和隐藏测试不会进入 Agent Prompt。
- 数据保留上游任务标识，使用时需遵守各上游数据集条款。
- 当前仓库尚未附加通用软件许可证；公开可见不等于自动授予再分发或修改权。

发布归档前请核对[公开发布检查清单](OPEN_SOURCE_CHECKLIST_zh.md)。
