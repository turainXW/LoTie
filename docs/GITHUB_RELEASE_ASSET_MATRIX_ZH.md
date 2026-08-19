# GitHub 发布资产矩阵

本项目不能直接把开发目录整体提交到 GitHub。开发目录包含本机 venv、SWE-smith 仓库 checkout、运行 workspace、API 配置、原始轨迹和远程训练缓存。发布包应按资产用途和体积分层。

## 1. 直接进入 Git 仓库

| 模块 | 资产 | 说明 |
|---|---|---|
| Agent Runtime | `src/code_agent_baseline` | JSON Tool Calling、上下文、工具、Verifier、Pass@k |
| Harness | `scripts/run_swegym_patch_rollout.py`、`harness_versions.json` | 正式多轮执行与版本信息 |
| 任务构建 | `scripts/build_lottie_dataset_splits.py`、SWE-smith 构建脚本 | 确定性切分、扩展集构造与审计 |
| macOS 环境 | `scripts/bootstrap_swesmith_venvs.py` | 无 Docker 重建仓库和 Python 3.10 venv |
| 并发采集 | `scripts/collect_swesmith_rollouts_macos.sh`、Pass@k runner | workers、断点恢复、基础设施重试 |
| 清洗 | `tools/build_salvage_sft_dataset.py` 等 | Strict/Salvage/Quarantine 与交付包构建 |
| Token/Mask | `tools/prepare_qwen_sft_dataset.py`、审计脚本 | Qwen chat template、32K 切窗、assistant-only mask |
| SFT | `scripts/train_qwen_lottie_lora.py`、bucket smoke | LoRA 和动态 microbatch |
| 评测 | V3 汇总、指标 CSV、图表和分析脚本 | 不包含模型权重 |
| 文档 | 技术报告、macOS 恢复说明、清洗报告 | 当前实验口径 |

## 2. 以压缩文件进入仓库或 GitHub Release

| 资产 | 本地状态 | 发布策略 |
|---|---|---|
| `sft_primary.jsonl` | 1069 条，约 30.5 MB | 清理路径后 gzip，适合 GitHub Release；体积允许时可直接提交 |
| `sft_optional_tier_b.jsonl` | 13 条，约 1.3 MB | gzip 发布，默认不参与训练 |
| `rl_rollouts.jsonl` | 1200 条，约 35.8 MB | gzip 发布，保留 reward=0/1 |
| 清洗 audit/metadata | 本地完整 | 直接提交，保留 provenance 与质量口径 |
| V3 汇总和图表 | 本地完整 | 直接提交 |

原始 workspace、完整 attempt 日志和 1.5 GB 采集运行目录不进入普通 Git history。若必须公开，应单独做版本化 tarball，放 GitHub Release、对象存储或数据集平台，并提供 SHA-256。

## 3. 当前仅远程或本地未完整归档

本地审计未找到以下文件：

| 资产 | 当前风险 | 建议 |
|---|---|---|
| Qwen 最终 tokenized Arrow 数据 | 本地只有统计和部分日志 | 从训练服务器下载 dataset manifest、Arrow shards 和 Mask audit |
| Epoch 1/3/5 LoRA adapter | 本地没有 `adapter_model.safetensors` | 放 GitHub Release 或 Hugging Face，不进入普通 Git |
| Epoch 3/4/5 optimizer/scheduler/RNG state | 本地没有 `.pt` 状态 | 若需精确续训，必须从远程归档 |
| V3 全量 task-level `sample_results.jsonl` | 本地只有 combined summaries/tgz | 下载后单独压缩发布，便于 paired bootstrap |
| vLLM/model cache | 只用于远程运行 | 不发布；记录模型 ID、revision 和启动参数即可 |

2026-08-20 对最新训练服务器执行只读盘点时，SSH 在认证前后直接关闭，因而本次无法确认上述远程文件是否仍在线。发布说明必须写“待远程归档”，不能标记为已包含。

## 4. 永不提交

- `.env`、API key、GitHub token、SSH 密码与 expect 临时脚本；
- `.venv/`、`.codeagent/`、repo checkout 和仓库级 venv；
- 模型缓存和第三方 wheel cache；
- macOS `.DS_Store`、pytest/cache、PID 和实时日志；
- 含开发机绝对路径但未经过 sanitizer 的 task/trajectory 文件。

## 5. 发布前阻塞项

1. 当前根 Git 仓库没有配置 remote；本机也没有 `gh` CLI。
2. 项目尚未选择 `LICENSE`，公开发布前必须确认源码与数据许可证。
3. 需要决定 GitHub 仓库是 private 还是 public。private 可以先归档实验；public 还需核对 EvalPlus、SWE-smith、生成轨迹中代码片段的再分发条款。
4. 远程 LoRA、Arrow 和完整 V3 样本尚未拉回本地。

在以上问题解决前，可以生成 GitHub-ready staging 和校验包，但不应直接推到公开仓库。
