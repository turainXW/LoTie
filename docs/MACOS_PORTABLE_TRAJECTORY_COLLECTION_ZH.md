# macOS 无 Docker 环境恢复与并发轨迹采集

本流程用于在新的 macOS 主机上快速恢复 SWE-smith 仓库环境，并并发采集可断点续跑的 Agent 轨迹。它不上传或复制 venv 本体，而是根据固定仓库、commit、安装 recipe 和测试依赖重新构建。

## 为什么不提交 venv

Python venv 包含绝对路径、解释器路径、平台相关动态库和可编辑安装引用。即使两台机器都是 macOS，Python patch 版本、CPU 架构或 Homebrew 安装位置变化也可能导致复制后的 venv 失效。

GitHub 中应保存：

- 固定任务与仓库 commit；
- 每仓库安装命令；
- 环境构建脚本；
- requirements 快照和依赖 fingerprint；
- setup、Gold smoke 和运行 manifest。

本机忽略：

- `.venv/`：项目控制环境；
- `.codeagent/`：仓库 checkout 和每仓库 venv；
- `data/runtime/`：带本机绝对路径的 runtime task file；
- `outputs/`：采集结果、日志和 workspace。

## 1. 准备基础软件

SWE-smith 当前固定使用 Python 3.10：

```bash
brew install python@3.10
cd code_agent_quickstart
```

采集入口检测到 `.venv` 缺失时，会自动调用 `scripts/install_dev.sh` 创建项目 venv，并从 `pyproject.toml` 安装 `dev,swegym` 依赖；不需要手动复制旧 venv。确认解释器：

```bash
python3.10 --version
.venv/bin/python --version
```

项目 `.venv` 可以使用较新的 Python；每个待测仓库使用独立 Python 3.10 venv。

## 2. 自动构建仓库和 venv

第二批 100 道 SWE-smith：

```bash
.venv/bin/python scripts/bootstrap_swesmith_venvs.py \
  --tasks data/swesmith_train_extension_v1/tasks_runnable.jsonl \
  --cache-root .codeagent/swesmith_train_extension_v1_portable \
  --output-dir data/runtime/swesmith_train_extension_v1 \
  --python "$(command -v python3.10)" \
  --workers 4 \
  --smoke-per-repo 1
```

脚本会幂等执行：

1. 将任务文件中的旧机器路径重写到当前 `--cache-root`；
2. 按固定 commit 下载 20 个仓库；
3. 为每个仓库建立独立 Python 3.10 venv；
4. 执行 profile 安装命令并自动发现常见 test requirements；
5. 每仓库运行至少一道 base/bug/gold smoke；
6. 生成第三方 requirements 快照、安装 recipe 和 bootstrap manifest。

主要输出：

```text
data/runtime/swesmith_train_extension_v1/
  tasks_runtime.jsonl
  setup_results.jsonl
  smoke_sanity.jsonl
  environment_recipes.jsonl
  bootstrap_manifest.json
  requirements/
    <repo>.requirements.txt
```

`requirements/*.txt` 是同平台诊断快照，不是跨平台强制 lock。默认恢复仍以仓库自身依赖文件和项目维护的 profile 安装命令为准，避免陈旧 freeze 阻塞新机器。

## 3. 并发采集轨迹

在 `.env` 或当前 shell 中提供模型 API key：

```bash
export DEEPSEEK_API_KEY=...
```

默认执行 100 题、每题 4 rollout、4 workers：

```bash
WORKERS=4 SETUP_WORKERS=4 \
  zsh scripts/collect_swesmith_rollouts_macos.sh
```

这是完整的一键入口，首次运行会依次：

1. 创建项目 `.venv` 并安装 Harness requirements；
2. 下载固定 commit 的目标仓库；
3. 并发创建每仓库 Python 3.10 venv 并安装测试依赖；
4. 执行 Gold smoke；
5. 启动并发 rollout 采集。

如需使用指定解释器：

```bash
PROJECT_PYTHON="$(command -v python3.10)" \
PYTHON310="$(command -v python3.10)" \
WORKERS=4 SETUP_WORKERS=4 \
  zsh scripts/collect_swesmith_rollouts_macos.sh
```

该入口会先幂等检查环境，然后调用版本化 Pass@k runner。正式配置包括：

- `temperature=1.0`、`top_p=0.95`；
- 协议修复重试温度为 0；
- 单步最多 4000 token；
- SWE-smith 最多 40 个工具步；
- 64K 上下文，超限显式失败；
- 每槽位最多 3 次基础设施尝试；
- macOS 自动启用 `caffeinate`。

输出目录默认为：

```text
outputs/training_runs/deepseek_v4_flash_swesmith_extension100_rollout4_v1/
```

## 4. 并发与恢复语义

`WORKERS` 只改变同时运行的任务数，不改变模型、prompt、采样、Harness、任务或 Verifier。runner 对每个 `(instance_id, sample_index)` 建立独立目录：

- `sample_valid=true` 的槽位在恢复时自动跳过；
- 模型失败和 Verifier 失败作为有效负样本保留；
- 只有 API、runner、OOM 等基础设施失败允许原槽位重试；
- worker 数变化写入 `execution_policy_history.jsonl`；
- PID 文件阻止同一输出目录启动重复 collector。

建议先以 `WORKERS=2` 做少量 smoke，再提升到 4。macOS 本地仓库 verifier 同时涉及 CPU、磁盘和多个 pytest 进程；继续增大并发未必提高吞吐。

## 5. 使用原始 SWE-smith 任务集

可以通过环境变量切换任务源、selection、runtime 和输出目录：

```bash
SELECTION="$PWD/data/dataset_splits_v1/selection.json" \
SOURCE_TASKS="$PWD/data/swesmith_local/tasks_runnable.jsonl" \
RUNTIME_DIR="$PWD/data/runtime/swesmith_original" \
CACHE_ROOT="$PWD/.codeagent/swesmith_original_portable" \
RUN_DIR="$PWD/outputs/training_runs/swesmith_original_replay" \
WORKERS=4 \
  zsh scripts/collect_swesmith_rollouts_macos.sh
```

原始 selection 同时包含函数题，但采集脚本固定传入 `--dataset swesmith_py`，因此只运行对应 split 中的 SWE-smith 任务。

## 6. 适用边界

该流程适合本地轨迹生产和内部一致性评测，但仍标记 `official_comparable=false`。若要提交官方 SWE-bench/SWE-smith 排行或需要完全统一的 Linux ABI，应使用官方容器镜像和官方 verifier 环境。
