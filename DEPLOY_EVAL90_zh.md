# Lottie Eval90 Linux 部署

本文说明如何在一台 Linux 开发机上构建 Lottie Eval90 的本地 verifier 环境。部署过程不依赖
Docker，但 SWE-smith 结果必须标记为 `official_comparable=false`，不能与官方容器分数直接比较。

## 评测构成

| 类型 | 数量 | 验证方式 |
|---|---:|---|
| MBPP+ | 30 | 参考实现执行官方 base/plus inputs，空实现作为负对照 |
| HumanEval+ | 30 | 参考实现执行官方 base/plus inputs，空实现作为负对照 |
| SWE-smith-py | 30 | 6 个 Python 仓库；验证 base 通过、Bug 注入后失败、反向 Gold 恢复后通过 |

训练、验证和评测按 task ID 与仓库隔离。Eval90 只用于最终评测，不进入 SFT 轨迹、长期记忆或
`problem_search` 索引。

## 系统要求

- Linux x86_64 或 arm64。
- Git、Bash、至少 5 GB 可用磁盘空间。
- 一个用于运行 Harness 的 Python 3.10+。
- 一个独立 Python 3.10 解释器，用于创建 verifier venv。
- 可访问任务依赖对应的 Git 仓库与 Python 包索引。

Conda 示例：

```bash
conda create -y -p /data/lottie/eval90/python310 python=3.10 pip
export PYTHON_BIN=/data/lottie/eval90/python310/bin/python
```

## 一键部署

```bash
git clone <your-lottie-repository-url> lottie-agent
cd lottie-agent

export EVAL90_STATE_ROOT=/data/lottie/eval90/state
export PYTHON_BIN=/data/lottie/eval90/python310/bin/python

./scripts/deploy_eval90_linux.sh all
```

脚本会创建 Harness venv（如不存在），然后依次执行：

1. `doctor`：检查 Python 3.10、Git、数据文件和磁盘空间。
2. `prepare`：只抽取 Evaluation 任务，并按当前 `state-root` 重写绝对路径。
3. `setup-functions`：创建 EvalPlus 共享 venv 和 60 个隔离函数仓库。
4. `setup-repos`：克隆 6 个固定 commit，并为每个仓库创建独立 venv。
5. `sanity-functions`：运行 60 条函数题 Gold sanity。
6. `sanity-repos`：运行 30 条 SWE-smith Gold sanity。
7. `status`：输出最终可运行状态和磁盘占用。

## 分阶段与恢复

网络中断后无需删除目录。重新执行当前阶段即可；Git checkout 和 venv 都有 fingerprint marker。
GitHub 仓库优先下载固定 commit 的 codeload 归档并初始化本地干净 Git 基线；归档不可用时，
回退到同一 commit 的 depth-1 fetch，并对网络超时或低速断连自动重试 3 次。

```bash
./scripts/deploy_eval90_linux.sh doctor
./scripts/deploy_eval90_linux.sh prepare
./scripts/deploy_eval90_linux.sh setup-functions
./scripts/deploy_eval90_linux.sh setup-repos
./scripts/deploy_eval90_linux.sh sanity-functions
./scripts/deploy_eval90_linux.sh sanity-repos
./scripts/deploy_eval90_linux.sh status
```

也可以直接使用 Python CLI：

```bash
.venv/bin/codeagent deploy-eval90 status \
  --state-root /data/lottie/eval90/state \
  --python /data/lottie/eval90/python310/bin/python
```

## 生成目录

```text
$EVAL90_STATE_ROOT/
├── repos/
│   ├── evalplus/                 # 60 个函数题 workspace
│   └── swesmith/                 # 6 个固定 commit 仓库
├── venvs/
│   ├── evalplus/                 # 函数题共享环境
│   └── swesmith/                 # 每仓库独立环境
├── tasks/                        # 已重写为当前机器路径的任务清单
├── results/                      # 逐题 Gold sanity 结果
└── reports/
    ├── doctor.json
    ├── task_manifest.json
    ├── swesmith_setup.jsonl
    ├── function_sanity_summary.json
    ├── swesmith_sanity_summary.json
    └── deployment_status.json
```

完成标准是 `deployment_status.json` 中 `ready=true`，并同时满足：

- `task_counts = 30/30/30`。
- 60 个函数 workspace 与共享 venv 存在。
- 60 条函数题 Gold sanity 通过。
- 6 个 SWE-smith 仓库及 venv ready。
- 30 条 SWE-smith Gold sanity 通过。

## 可移植性与安全边界

- 不复制 macOS/Windows venv；目标机始终重新创建 Linux venv。
- 不把 API key、SSH 地址、模型权重或服务器密码写入仓库。
- 模型权重与 Eval90 verifier 解耦，评测环境可独立部署。
- Agent 看不到参考实现、Bug 注入 patch、Gold patch 和隐藏测试。
- EvalPlus/SWE-smith 数据及上游仓库继续遵循各自许可证；发布前应保留来源和版本信息。
- 本地 venv verifier 用于可复现开发与相对比较，不宣称官方 Docker 等价性。
