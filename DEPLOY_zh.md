# 开发机器快速部署

目标：把本项目快速部署到新的开发机器，并保持 memory/skills/context 都用文件形式存储。
当前版本不需要数据库服务，适合直接拷贝目录、Git clone、或打包同步到开发机。

## 1. 准备环境

要求：

- Python 3.10+
- git
- 可选：Docker Desktop / Colima，用于 Docker runtime sandbox 和 SWE-Gym verifier

## 2. 安装

```bash
cd /path/to/agentrl/code_agent_quickstart
bash scripts/install_dev.sh
source scripts/dev_env.sh
```

默认会创建 `.venv`，安装 `dev,swegym` 依赖，并初始化 `.codeagent/` 文件状态。

可选安装层级：

```bash
bash scripts/install_dev.sh base       # 只安装 codeagent + mini-swe-agent
bash scripts/install_dev.sh dev        # 开发/测试
bash scripts/install_dev.sh swegym     # SWE-Gym 数据处理
bash scripts/install_dev.sh dev,swegym # 默认，本地开发推荐
bash scripts/install_dev.sh full       # dev + swegym + local-llm，不含 torch
```

也可以直接用 requirements：

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-swegym.txt
python -m pip install -r requirements-local-llm.txt
```

如果只是跑零依赖 smoke，也可以不安装包：

```bash
python3 scripts/run_smoke.py --output-dir ../data/code_agent_smoke
```

如果希望直接使用 Docker runtime sandbox：

```bash
docker build -f Dockerfile.runtime -t lottie-codeagent-runtime:latest .
./scripts/codeagent_docker.sh "分析当前项目的核心模块"
```

## 3. 配置 API Key

```bash
cp .env.example .env
```

然后填入：

```text
DEEPSEEK_API_KEY=...
```

可选：

```text
BRAVE_SEARCH_API_KEY=...
TAVILY_API_KEY=...
GITHUB_TOKEN=...
```

加载环境变量：

```bash
source scripts/dev_env.sh
```

`scripts/dev_env.sh` 会自动：

- 激活 `.venv`
- 加载 `.env`
- 设置 `PYTHONPATH=src:benchmarks/leetcode_top10`
- 设置 `MSWEA_GLOBAL_CONFIG_DIR=.mswea`，避免 mini-swe-agent 写到用户全局目录
- 设置 `MSWEA_SILENT_STARTUP=1`

即使直接运行 `.venv/bin/codeagent`，agent runtime 也会把项目 `.venv/bin` 注入内部命令的
`PATH`，保证执行轨迹中的 `python`/`python3` 使用同一套虚拟环境。

## 4. 初始化文件存储

本项目不依赖数据库。默认使用文件存储：

```text
.codeagent/
  repo_map.json
  skills.json
  memory/
    project_facts.jsonl
    lessons.jsonl
    preferences.jsonl
    failure_patterns.jsonl
    session_index.jsonl
    sessions/
```

这些文件就是当前的轻量状态层：

- `repo_map.json`：代码库结构索引，可随时重新生成。
- `skills.json`：agent skill/权限/行为配置，可手动编辑或用 CLI 修改。
- `memory/*.jsonl`：长期记忆，追加写入，方便 grep、diff、备份和迁移。
- `memory/sessions/`：每次轨迹总结，后续可转成学习样本。

初始化：

```bash
codeagent repo-map --repo . --task "初始化 code agent 项目"
codeagent skills --config .codeagent/skills.json init
codeagent memory --memory-dir .codeagent/memory init
```

## 5. 运行

只读分析：

```bash
codeagent run \
  --repo . \
  --task "只读分析项目结构并给出下一步计划" \
  --mode plan \
  --sandbox copy \
  --model deepseek/deepseek-reasoner \
  --output ../data/code_agent_mini/plan.json
```

允许受控下载开源项目：

```bash
codeagent run \
  --repo . \
  --task "下载 https://github.com/SWE-agent/mini-swe-agent 并总结目录结构" \
  --mode debug \
  --sandbox copy \
  --enable-web-tools \
  --download-dir refs/open_source \
  --model deepseek/deepseek-reasoner \
  --output ../data/code_agent_mini/download.json \
  --summarize-session \
  --learn-from-session
```

## 6. 迁移到另一台机器

需要带走：

```text
code_agent_quickstart/
  src/
  scripts/
  pyproject.toml
  configs/
  README_zh.md
  DEPLOY_zh.md
  .codeagent/skills.json
  .codeagent/memory/
```

通常不需要带走：

```text
.venv/
.mswea/
.codeagent/workspaces/
refs/open_source/
__pycache__/
```

轨迹数据如果用于学习，需要带走：

```text
data/code_agent_mini/*.json
data/code_agent_smoke/*.jsonl
```

推荐的最小迁移包：

```text
code_agent_quickstart/
  src/
  scripts/
  configs/
  pyproject.toml
  README_zh.md
  DEPLOY_zh.md
  .env.example
  .codeagent/skills.json
  .codeagent/memory/
```

迁移后在新机器执行：

```bash
bash scripts/install_dev.sh
cp .env.example .env
```

然后填入 `DEEPSEEK_API_KEY` 等 key，再运行 `codeagent smoke` 或 `codeagent run`。

## 7. 为什么先不用数据库

当前 memory 规模小，主要是 agent 运行事实、经验教训、偏好和 session summary。文件方案更适合第一版：

- 部署简单：不需要 Redis、Postgres、SQLite migration 或向量库服务。
- 可解释：每条 memory 都是 JSONL，能直接审计和作为训练数据来源。
- 可迁移：复制 `.codeagent/memory/` 就能迁移长期记忆。
- 可版本化：关键 skill/memory 可以纳入 Git 或单独备份。

后续如果需要语义检索，可以在不改上层接口的情况下增加 SQLite、DuckDB、LanceDB 或 Chroma backend；当前默认 backend 保持 file-only。
