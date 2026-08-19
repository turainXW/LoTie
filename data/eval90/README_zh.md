# Lottie Local Eval90

这是固定评测集候选清单，不得进入训练轨迹、problem_search 索引或长期记忆。

## 构成

- MBPP+：30 道清晰、低依赖的函数实现题。
- HumanEval+：30 道包含算法和边界条件的函数题。
- SWE-smith-py：30 道仓库修复题，来自 6 个纯 Python 仓库，每仓库 5 道。

完整 ID 和仓库 profile 位于 `selection.json`。

## 本地 SWE-smith 语义

SWE-smith-py 数据集里的 `patch` 是 Bug 注入 Patch，不是 Agent 应直接输出的答案。无 Docker
复现时按以下顺序执行：

1. 克隆 profile 中的原始仓库并 checkout 指定 commit。
2. 应用数据集 `patch`，得到有 Bug 的任务 workspace。
3. 确认 FAIL_TO_PASS 测试失败。
4. Agent 只看到问题描述和 Bug workspace，生成自己的修复 Patch。
5. local-venv verifier 应用 Agent Patch，再运行 FAIL_TO_PASS/PASS_TO_PASS。

本地结果必须标记 `official_comparable=false`。目前 Bottle 的
`lm_rewrite__lipuhwjq` 已完成 Mac venv preflight，其余 29 道仍需逐仓库验证 Python 3.12
兼容性和依赖安装情况。

## 搜索隔离

- MBPP+/HumanEval+ 禁用 WebSearch。
- 仓库题只允许查询公开 API、RFC 和项目文档。
- 禁止查询 instance_id、完整 problem statement、Gold/Bug Patch 或隐藏测试。
