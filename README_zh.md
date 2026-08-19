# miniCoder：可训练代码智能体与轨迹采集框架

这是一个可以放在简历上的项目雏形：目标不是做一个 IDE 插件，而是基于
`mini-swe-agent` 扩展一个可观测、可训练、可评估的代码智能体。它能读取任务、
执行 shell 命令、修改代码、运行验证命令，并把完整 agent 轨迹保存下来，
后续可以用于 SFT/RL 数据构造。

## 简历项目描述

**miniCoder：面向代码任务执行、训练轨迹采集与评测的 Agentic Coding System**

- 基于 `mini-swe-agent` 的 bash-only agent loop，扩展 plan/build/debug 多模式权限
  控制，支持将自然语言任务转化为 shell action、文件修改、测试执行和最终状态判断。
- 实现可插拔模型后端，支持本地 rule-based smoke backend，并预留
  OpenAI-compatible HTTP 模型接口，便于切换到 GPT、Qwen、DeepSeek 或本地 vLLM。
- 设计统一 trajectory schema，记录 instruction、action、observation、exit code、
  validation status 等信息，为后续 SFT、reward modeling、GRPO/RL 训练提供数据。
- 构建 toy coding benchmark 和自动评估脚本，用于快速验证 agent 是否具备代码
  定位、编辑、运行测试、失败恢复等基本能力。
- 调研并对比 mini-swe-agent、SWE-agent、OpenHands、aider 等开源 coding agent，
  最终选择 mini-swe-agent 作为后续扩展和对照基线。

## 项目亮点

- **Code 能跑**：`scripts/run_smoke.py` 可以生成本地 toy repo，并跑完整零依赖
  smoke loop；安装 `mini-swe-agent` 后可用 `scripts/run_mini_agent.py` 跑 mini base。
- **Agentic 设计清晰**：每一步都遵循 action -> observation -> reflection -> next action。
- **Mini-first**：正式路径使用 `mini-swe-agent` 的 `DefaultAgent`，本项目只扩展 mode
  policy、guarded environment、repo context 和数据转换。
- **数据闭环**：每个 episode 输出 JSONL 轨迹，可直接接入训练/评估 pipeline。
- **低依赖**：第一版只用 Python 标准库，方便在新机器上快速复现。
- **可扩展**：后续可以把 rule-based backend 替换为 LLM，或者接入 `mini-swe-agent`。

## 快速运行

推荐先创建项目虚拟环境：

```bash
cd /path/to/code_agent_quickstart
bash scripts/install_dev.sh
source scripts/dev_env.sh
```

macOS 上无需 Docker 重建 SWE-smith 仓库 venv 并进行并发、断点续跑轨迹采集，参见
[`docs/MACOS_PORTABLE_TRAJECTORY_COLLECTION_ZH.md`](docs/MACOS_PORTABLE_TRAJECTORY_COLLECTION_ZH.md)。

安装后推荐使用 `minicoder` 命令；原有 `codeagent` 命令作为兼容别名继续可用。

默认安装 `dev,swegym` 依赖，包括 `mini-swe-agent`、`litellm`、`pytest`、`datasets`、
`pandas`、`pyarrow`、`huggingface-hub`。如果只想装最小运行环境：

```bash
bash scripts/install_dev.sh base
```

Agent 启动后会自动把项目 `.venv/bin` 注入内部命令的 `PATH`，所以 trajectory 里的
`python`/`python3` 默认会使用 `code_agent_quickstart/.venv` 中的包；不需要每个任务手动
`source .venv/bin/activate`。

```bash
cd /path/to/code_agent_quickstart
python3 scripts/run_smoke.py \
  --output-dir data/code_agent_smoke
```

运行后会生成：

```text
data/code_agent_smoke/
  tasks.jsonl
  trajectories.jsonl
  toy_repo_0001/
```

如果 `final_status=pass`，说明最小 agent loop 已经能完成一次代码修复任务。
第一版 smoke test 只依赖 Python 标准库，不需要安装 `pytest`。

## mini-swe-agent 入口

安装可选依赖：

```bash
cd /path/to/code_agent_quickstart
bash scripts/install_dev.sh base
source scripts/dev_env.sh
```

使用 mini 作为 base 框架运行：

```bash
python3 scripts/run_mini_agent.py \
  --repo /path/to/target-repo \
  --task "分析当前 code agent 项目结构并给出改进计划" \
  --mode plan \
  --sandbox copy \
  --enable-web-tools \
  --model gpt-4o \
  --output data/code_agent_mini/plan_trajectory.json
```

模式：

- `plan`：只读分析，阻断写文件、删除文件、提交、推送等命令。
- `build`：允许修改代码和运行验证，但仍阻断明显危险命令。
- `debug`：偏向日志、测试、错误定位，阻断破坏性命令。

Web 工具：

- `web_search`：搜索开源项目/文档。优先使用 `BRAVE_SEARCH_API_KEY` 或
  `TAVILY_API_KEY`；没有搜索 key 时 fallback 到 GitHub repo search。
- `download_repo`：只允许下载公开 GitHub 仓库，并且只能写入 `--download-dir` 指定目录。
  该工具在 `plan` 模式下会被阻断，需要切到 `build` 或 `debug`。

Sandbox：

- 默认 `--sandbox copy`：每次运行会把 `--repo` 复制到 `.codeagent/workspaces/<run_id>/repo`，
  agent 只在副本里读写，原仓库不会被直接修改。
- 可选 `--sandbox direct`：直接在原仓库运行，适合只读分析或明确需要原地修改的实验。
- 可选 Docker runtime sandbox：通过 `Dockerfile.runtime` / `docker-compose.runtime.yml`
  把 agent 的 bash、Python、pytest 和 patch 生成放进容器内执行。默认仍开启 workspace
  copy sandbox；详细用法见 `reports/lottie_docker_runtime_sandbox_zh.md`。

示例：让 agent 搜索/下载开源项目时打开 web 工具：

```bash
python3 code_agent_quickstart/scripts/run_mini_agent.py \
  --repo code_agent_quickstart \
  --task "下载 https://github.com/SWE-agent/mini-swe-agent 并总结目录结构" \
  --mode debug \
  --sandbox copy \
  --enable-web-tools \
  --download-dir refs/open_source \
  --model deepseek/deepseek-reasoner
```

## 目录结构

```text
code_agent_quickstart/
  README.md                    # 英文调研版
  README_zh.md                 # 中文项目版
  agentic_design_zh.md         # agentic 设计说明
  quickstart_plan.md           # 快速实现路线
  open_source_project_review_zh.md # 开源项目调研和设计取舍
  requirements-mini.txt        # mini-swe-agent 可选依赖
  scripts/
    run_smoke.py               # 一键 smoke test
    evaluate.py                # 统一测评入口，输出 report.json
    run_mini_agent.py          # mini-swe-agent base 入口
    build_repo_map.py          # 生成/检索 repo context/codebase map
    manage_skills.py           # 初始化、修改、禁用、compact skills
    manage_memory.py           # 长短期/session memory 管理
  src/code_agent_baseline/
    agent.py                   # agent loop
    mini_guarded.py            # mini GuardedLocalEnvironment / mode policy
    mini_web_model.py          # mini tool schema 扩展：web_search/download_repo
    web_tools.py               # 受控搜索和 GitHub 下载实现
    repo_context.py            # codebase map、import graph、任务相关检索
    skills.py                  # skill 配置管理与 compact
    context_manager.py         # 合并 repo map + skills 形成 prompt context
    memory.py                  # long-term memory/session summary
    workspace_sandbox.py       # copy/direct workspace sandbox
    model_clients.py           # 模型后端接口
    schemas.py                 # task/action/trajectory 数据结构
    task_factory.py            # toy coding task 生成
```

## 测评框架

当前有一个轻量统一测评入口：

```bash
PYTHONPATH=code_agent_quickstart/src python3 -m code_agent_baseline.cli eval \
  --suite all \
  --output-dir data/code_agent_eval
```

也可以安装后运行：

```bash
codeagent eval --suite all --output-dir data/code_agent_eval
```

第一版 suite：

- `codeagent`：真正的 code agent 测评。为每个任务生成一个带 bug 的小仓库，agent 自己执行
  action、修改代码、运行 verifier，最终按 `final_status=pass` 计分，并保存 trajectory。
- `leetcode`：解法 verifier suite。运行 LeetCode 前 10 题的 Python `unittest`，用于补充
  algorithm correctness 测试；它不是 agent 自己解题的轨迹测评。
- `smoke`：遗留的单题 loop smoke，只在 `--suite smoke` 时运行。

当前测评核心不是 LLM judge，而是 verifier-driven evaluation：

- 每个 codeagent 任务都有独立 repo、自然语言 instruction、`validation_command`。
- agent 轨迹记录每一步 action、stdout/stderr、exit code、耗时。
- 指标从最终 verifier 得到：`pass/fail`、`pass_rate`、`steps`、`elapsed_sec`。
- LLM judge 后续可作为二级指标，用来评估 plan 质量、patch 最小性、失败归因和安全策略遵守。

输出：

```text
data/code_agent_eval/
  report.json
  codeagent/
    tasks.jsonl
    trajectories.jsonl
    repos/
  leetcode_top10/
    unittest_output.txt
```

`report.json` 会记录 suite 状态、总测试数、通过数、失败数、pass rate、耗时和 artifact 路径。

## 训练数据导出

codeagent 测评轨迹可以导出成训练友好的 JSONL：

```bash
PYTHONPATH=code_agent_quickstart/src python3 -m code_agent_baseline.cli export-training \
  --trajectory data/code_agent_eval/codeagent/trajectories.jsonl \
  --output-dir data/code_agent_training
```

输出：

```text
data/code_agent_training/
  manifest.json
  sft_next_action.jsonl
  reward_episodes.jsonl
  episodes.jsonl
```

- `sft_next_action.jsonl`：把每个 episode 拆成 `history -> next action`，适合 SFT。
- `reward_episodes.jsonl`：每条轨迹一个 reward，当前由 verifier pass/fail 和步数得到。
- `episodes.jsonl`：脱敏后的完整 episode，适合复盘、过滤和二次转换。

训练样本会包含 `initial_context`，包括 compact repo map、文件摘要、运行目录、验证命令和 action space。
导出器会把本机绝对路径替换成 `<repo>` / `<local_path>`，避免训练数据绑定到当前机器。

## SWE-Gym / OpenHands 工具对齐

为了能直接复用 SWE-Gym 的 OpenHands SFT 轨迹，我们新增了 OpenHands-compatible
工具层：

```text
core tools:
  execute_bash(command)
  str_replace_editor(command, path, file_text?, old_str?, new_str?, insert_line?, view_range?)
  finish()

extension tools:
  web_search(query, max_results?)
  download_repo(repo_url, dest_name?)
  repo_context(query?)
  memory_search(query?)
  problem_search(query, top_k?, index_path?)
```

核心三工具对齐 OpenHands/SWE-Gym 轨迹格式；扩展工具保留我们自己的能力，如 web、
repo context、memory、题目搜索、受控下载和权限模式。

工具调用格式示例：

```text
<function=execute_bash>
<parameter=command>python3 -m pytest</parameter>
</function>
```

```text
<function=str_replace_editor>
<parameter=command>str_replace</parameter>
<parameter=path>src/foo.py</parameter>
<parameter=old_str>return a - b</parameter>
<parameter=new_str>return a + b</parameter>
</function>
```

本地实现位于 `src/code_agent_baseline/openhands_tools.py`，支持解析官方 OpenHands-style
`<function=...>` 文本，并执行本地 `execute_bash`、`str_replace_editor`、`finish`。
后续弱模型训练时可以学习 OpenHands 工具格式，推理时由我们的 runtime 执行，同时逐步加入更多工具选择。

## 题目搜索

`problem_search` 是本地题目/轨迹检索工具，用来让 agent 查询相似任务和历史经验：

```bash
PYTHONPATH=code_agent_quickstart/src python3 -m code_agent_baseline.cli problem-search build \
  --root data \
  --output data/problem_search/index.jsonl
```

```bash
PYTHONPATH=code_agent_quickstart/src python3 -m code_agent_baseline.cli problem-search search \
  --query "slugify url" \
  --index data/problem_search/index.jsonl \
  --top-k 3
```

工具调用格式：

```text
<function=problem_search>
<parameter=query>slugify url</parameter>
<parameter=top_k>3</parameter>
<parameter=index_path>data/problem_search/index.jsonl</parameter>
</function>
```

训练和评测时要避免数据泄漏：

- 可以检索历史 instruction、公开任务描述、失败类型、文件摘要和成功/失败状态。
- 不应把 gold patch、hidden tests、标准答案或当前 instance 的 oracle 信息放进可检索索引。
- 对 SWE-Gym/SWE-Bench，`problem_search` 应只用于跨任务经验检索，不能检索当前题目的参考 patch。

## 上下文管理

默认运行 `run_mini_agent.py` 时会自动生成：

```text
.codeagent/repo_map.json
.codeagent/skills.json
.codeagent/memory/
```

- `repo_map.json`：扫描仓库文件、语言、行数、Python imports/classes/functions、Markdown 标题、模块名、sha1、import graph 和 reverse import graph，并按任务检索相关文件。
- `skills.json`：保存可修改的 skill 配置，如 mode 权限、web 下载、trajectory 数据规范。
- `memory/`：保存长期 facts/lessons/preferences/failure_patterns，以及 session summaries。
- compact：运行时只把 top relevant repo map 和 skills 注入 prompt，避免把整个仓库塞给模型。

当前没有引入数据库。记忆、技能和 repo map 都是文件：

- 记忆：`.codeagent/memory/*.jsonl` 和 `.codeagent/memory/sessions/*/session_summary.json`
- 技能：`.codeagent/skills.json`
- 代码地图：`.codeagent/repo_map.json`

这种方式便于快速部署到开发机器，也便于把 agent 轨迹和记忆沉淀成学习数据。

手动生成 repo map：

```bash
python3 code_agent_quickstart/scripts/build_repo_map.py \
  --repo code_agent_quickstart \
  --task "download_repo 权限和 web 工具"
```

输出结构化 repo context：

```bash
python3 code_agent_quickstart/scripts/build_repo_map.py \
  --repo code_agent_quickstart \
  --task "repo_context import graph file locator" \
  --max-files 8 \
  --max-snippets 4 \
  --json
```

Agent 轨迹中可以直接调用：

```text
<function=repo_context>
<parameter=query>normalize slug punctuation failing test</parameter>
<parameter=max_files>8</parameter>
<parameter=max_snippets>4</parameter>
</function>
```

返回内容包括：

- ranked files：候选文件、分数、role、language、module、symbols。
- import graph hints：`imports` 和 `imported_by`，用于从测试文件反查源码。
- snippets：与 query 或 symbol 匹配的短片段，作为下一步 read/edit 的证据。

管理 skills：

```bash
python3 code_agent_quickstart/scripts/manage_skills.py --config code_agent_quickstart/.codeagent/skills.json init
python3 code_agent_quickstart/scripts/manage_skills.py --config code_agent_quickstart/.codeagent/skills.json list
python3 code_agent_quickstart/scripts/manage_skills.py --config code_agent_quickstart/.codeagent/skills.json compact --task "修复 plan 权限"
```

修改 skill 示例：

```bash
python3 code_agent_quickstart/scripts/manage_skills.py \
  --config code_agent_quickstart/.codeagent/skills.json \
  upsert \
  --name "custom_review" \
  --description "Review code changes before final submission." \
  --trigger review \
  --instruction "Before submitting, summarize changed files and validation results." \
  --priority 9
```

## Memory 管理

Memory 分三层：

- short-term：mini-swe-agent 当前 run 的 messages/tool observations。
- session memory：从 trajectory 生成 `session_summary.json` 和 `session_index.jsonl`。
- long-term memory：跨 run 复用的 facts、lessons、preferences、failure patterns。

初始化和添加长期记忆：

```bash
python3 code_agent_quickstart/scripts/manage_memory.py \
  --memory-dir code_agent_quickstart/.codeagent/memory init

python3 code_agent_quickstart/scripts/manage_memory.py \
  --memory-dir code_agent_quickstart/.codeagent/memory \
  add \
  --type lesson \
  --content "download_repo must run in debug/build mode, not plan mode." \
  --tag download_repo \
  --tag permission
```

按任务压缩相关记忆：

```bash
python3 code_agent_quickstart/scripts/manage_memory.py \
  --memory-dir code_agent_quickstart/.codeagent/memory \
  compact \
  --task "下载开源项目并分析权限"
```

从 trajectory 生成 session summary，并可写入长期记忆：

```bash
python3 code_agent_quickstart/scripts/manage_memory.py \
  --memory-dir code_agent_quickstart/.codeagent/memory \
  summarize-session \
  --trajectory data/code_agent_mini/deepseek_download_repo_trajectory.json \
  --learn
```

`run_mini_agent.py` 也支持自动总结：

```bash
--summarize-session
--learn-from-session
--memory-dir code_agent_quickstart/.codeagent/memory
--disable-memory
```

## SWE-like 轨迹采集

项目里新增了一个轻量 SWE-bench/SWE-Gym 风格的本地 harness，用来先把“任务、补丁、测试、轨迹、SFT 数据”链路跑通。它不依赖 Docker，适合在 Mac 或开发机上快速采集可学习的 code-agent 轨迹。

初始化本地 SWE-like task：

```bash
PYTHONPATH=code_agent_quickstart/src python3 -m code_agent_baseline.cli \
  swebench-local init \
  --output-dir data/swebench_local
```

验证 base/gold patch：

```bash
PYTHONPATH=code_agent_quickstart/src python3 -m code_agent_baseline.cli \
  swebench-local verify \
  --tasks data/swebench_local/tasks.jsonl \
  --output data/swebench_local/base_results.jsonl

PYTHONPATH=code_agent_quickstart/src python3 -m code_agent_baseline.cli \
  swebench-local verify \
  --tasks data/swebench_local/tasks.jsonl \
  --output data/swebench_local/gold_results.jsonl \
  --apply-gold
```

采集 OpenHands/SWE-Gym 相似格式轨迹：

```bash
PYTHONPATH=code_agent_quickstart/src python3 -m code_agent_baseline.cli \
  collect-swe-like \
  --suite-dir data/swebench_like_rollout \
  --output-dir data/swebench_like_rollout/trajectories
```

核心产物：

- `output.jsonl`：接近 OpenHands evaluation output 的完整轨迹，包含 `instance_id`、`problem_statement`、`patch`、`resolved`、`messages`、`FAIL_TO_PASS`/`PASS_TO_PASS` 测试结果。
- `messages_sft.jsonl`：用于 SFT 的 `messages` 列格式，assistant 使用严格 JSON Tool Calling，环境反馈作为下一条 observation 消息。
- `verifier_records.jsonl`：给 verifier/reward model 使用的 task、patch、test report、label。
- `raw_traces/*.json`：调试用完整轨迹，保留本地 workspace 细节。

当前这个 harness 的定位是“训练数据格式和 agent loop 对齐”。真正跑 SWE-Bench/SWE-Gym 时，还需要 Docker 隔离镜像、真实仓库 checkout、官方 test patch，以及按 instance 还原环境；但 agent 侧的轨迹格式已经先对齐到了 OpenHands 风格。

## 自定义 Benchmark Project

如果要自己构造一个完整的 SWE-Gym mini 项目，可以用：

```bash
codeagent create-benchmark-project \
  --name toy_add_bug \
  --repo /path/to/source_repo \
  --output-dir data/custom_benchmarks \
  --problem "add(a, b) returns subtraction instead of addition" \
  --test-patch /path/to/test.patch \
  --gold-patch /path/to/gold.patch \
  --pytest-node tests/test_calculator.py::test_add
```

生成目录包含：

- `repo/`：原始仓库快照，会复制到 Docker 的 `/testbed`。
- `task.json` / `tasks.jsonl`：SWE-Gym 风格任务记录，包含 `problem_statement`、`test_patch`、`FAIL_TO_PASS` 和 `image`。
- `patches/test.patch`：官方测试补丁，verifier 时应用。
- `records.gold.jsonl`：可选 gold patch 记录，可用来做闭环 sanity check。
- `Dockerfile` / `build_image.sh`：构建可执行测试环境。
- `verifier.sh`：在容器内应用 patch 和 test patch，然后运行 pytest。
- `skills/pytest_generation_skill.md`：用强模型生成 pytest 草稿时的约束说明。

构建镜像：

```bash
cd data/custom_benchmarks/toy_add_bug
./build_image.sh
```

用 gold patch 验证闭环：

```bash
codeagent verify-swegym-patches \
  --records records.gold.jsonl \
  --tasks tasks.jsonl \
  --output verify_results.gold.jsonl \
  --mode docker
```

没有可用 Docker 时，可以在 Mac 上复用已有虚拟环境执行同一份 Patch 和 pytest 闭环：

```bash
codeagent verify-swegym-patches \
  --records records.gold.jsonl \
  --tasks tasks.jsonl \
  --output verify_results.local.jsonl \
  --mode local-venv \
  --local-venv .venv
```

本地模式会把仓库复制到临时目录后再应用 agent patch 和 test patch，不修改原仓库。省略
`--local-venv` 时会在输出目录的 `.codeagent/venvs/` 中创建并缓存环境；仓库有额外依赖时
增加 `--install-deps`。该结果会标记 `verifier_backend=local_venv` 和
`official_comparable=false`，不能作为官方 SWE-Gym/SWE-smith Docker 成绩。
本地 venv 不是安全沙箱，测试命令拥有当前用户权限，只用于可信数据集和自己构造的仓库任务。

## MBPP+ / HumanEval+ 本地 Gold Sanity

两类函数任务共享一个 Python 3.10 环境，并使用 EvalPlus 官方 Release 的 `base_input`、
`plus_input` 和参考实现。Gold 对 Agent 隐藏，本地结果不作为官方 pass@1 分数。

```bash
codeagent prepare-evalplus-local setup \
  --python /opt/homebrew/bin/python3.10

codeagent prepare-evalplus-local sanity \
  --tasks data/evalplus_local/tasks.jsonl \
  --output data/evalplus_local/gold_sanity.jsonl

codeagent prepare-evalplus-local runnable \
  --tasks data/evalplus_local/tasks.jsonl \
  --sanity-results data/evalplus_local/gold_sanity.jsonl \
  --output data/evalplus_local/tasks_runnable.jsonl
```

sanity 要求参考实现可编译并执行全部官方输入，同时将目标函数替换为
`NotImplementedError` 后必须失败。MBPP 的复数、tuple 和 set 输入按照 EvalPlus 0.3.1
的反序列化规则恢复。

## SWE-smith 本地任务与 Gold Sanity

已筛选的 SWE-smith 任务可以物化为统一的本地任务格式。每个任务保存正确仓库 commit、隐藏的
`bug_patch`、`FAIL_TO_PASS`、`PASS_TO_PASS` 和本地环境位置。Agent 启动前由 harness 应用
`bug_patch`，gold 判定通过反向应用该 patch 完成，不向 Agent 暴露 gold。

构建仓库及其可复用 venv：

```bash
codeagent prepare-swesmith-local setup \
  --tasks data/swesmith_local/tasks.jsonl \
  --python "$(command -v python3.10)" \
  --output data/swesmith_local/setup_results.jsonl
```

执行 gold sanity 并生成可运行任务：

```bash
codeagent prepare-swesmith-local sanity \
  --tasks data/swesmith_local/tasks.jsonl \
  --output data/swesmith_local/gold_sanity.jsonl \
  --p2p-limit 10

codeagent prepare-swesmith-local runnable \
  --tasks data/swesmith_local/tasks.jsonl \
  --sanity-results data/swesmith_local/gold_sanity.jsonl \
  --output data/swesmith_local/tasks_runnable.jsonl
```

sanity 分别在三个干净 workspace 中验证 base、bug 和 reverse-gold。只有
`gold_sanity_passed=true` 的任务进入 runnable 清单；本地结果始终标记
`official_comparable=false`。

采集 JSON Tool Calling 轨迹：

```bash
codeagent run-swegym-patch \
  --tasks data/swesmith_local/tasks_runnable.jsonl \
  --agent-mode use \
  --output-dir outputs/swesmith_train_run_001 \
  --work-dir .codeagent/workspaces \
  --repo-cache .codeagent/repo_cache \
  --num 10 \
  --max-steps 32 \
  --run-tests \
  --show-steps
```

Agent patch 通过 verifier 后，按 verifier 的 `benchmark_resolved` 导出成功 SFT：

```bash
codeagent verify-swegym-patches \
  --records outputs/swesmith_train_run_001/openhands_patch_rollout.jsonl \
  --tasks data/swesmith_local/tasks_runnable.jsonl \
  --output outputs/swesmith_train_run_001/verifier_results.jsonl \
  --mode local-venv

codeagent export-openhands-sft-dataset \
  --input outputs/swesmith_train_run_001/messages_sft.jsonl \
  --verifier-results outputs/swesmith_train_run_001/verifier_results.jsonl \
  --output-dir data/lottie_sft_v1
```

如果没有现成 pytest，可以让 OpenAI-compatible 强模型生成测试草稿：

```bash
codeagent create-benchmark-project \
  --name toy_add_bug \
  --repo /path/to/source_repo \
  --output-dir data/custom_benchmarks \
  --problem-file problem.md \
  --generate-tests \
  --generated-test-file tests/test_generated_bug.py \
  --model deepseek-chat \
  --api-key-env DEEPSEEK_API_KEY
```

注意：模型生成的 pytest 只能算草稿。进入训练数据前必须验证：

- `base repo + test_patch` 能复现失败。
- `gold/correct patch + test_patch` 能通过。
- agent patch 的 reward 只来自 Docker verifier，不来自 LLM judge。

## Eval90 Linux 一键部署

项目提供可恢复的本地 verifier 部署入口，自动准备 30 MBPP+、30 HumanEval+ 和 30
SWE-smith-py 评测任务，并把机器相关路径重写到独立数据目录：

```bash
export EVAL90_STATE_ROOT=/data/lottie/eval90/state
export PYTHON_BIN=/data/lottie/eval90/python310/bin/python
./scripts/deploy_eval90_linux.sh all
```

部署过程、目录结构、Gold sanity 和开源边界见 [DEPLOY_EVAL90_zh.md](DEPLOY_EVAL90_zh.md)。

生成不含密钥、模型权重、venv 和运行输出的开源发布包：

```bash
./scripts/package_open_source.sh v0.1.0
```

发布前检查见 [OPEN_SOURCE_CHECKLIST_zh.md](OPEN_SOURCE_CHECKLIST_zh.md)。

## 后续路线

1. 用真实 LLM 替换 rule-based backend。
2. 增加更多 toy coding tasks：parser、CLI、unit test repair、data pipeline 修复。
3. 接入 `mini-swe-agent`，作为强 baseline 和 trajectory 对照组。
4. 把 trajectories 转成 SFT 数据：给定 instruction + history，预测下一步 action。
5. 设计 reward：测试通过、步骤数、无效命令惩罚、patch 最小化。
