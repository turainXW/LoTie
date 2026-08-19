# Lottie Dataset Splits v1

## 规模

| Split | MBPP+ | HumanEval+ | SWE-smith-py | 总计 |
|---|---:|---:|---:|---:|
| Train | 120 | 80 | 100 | 300 |
| Validation | 15 | 15 | 15 | 45 |
| Evaluation | 30 | 30 | 30 | 90 |

训练集用于采集轨迹和训练；验证集只用于调整 Prompt、Tool、Context 和超参数；评测集只用于
最终报告。验证集和评测集不得回流训练。

## 仓库隔离

SWE-smith-py 按仓库划分：

- Train：funcy、typeguard、stackprinter、tenacity、python-json-logger。
- Validation：iniconfig、thefuzz、dataset。
- Evaluation：exceptiongroup、bottle、oauthlib、soupsieve、sqlglot、trio。

三组仓库交集为空。函数题也保证 task ID 无交集。

## 轨迹计划

- Train：每题 5 次 rollout，预计 1,500 条原始训练轨迹。
- Validation：每题 3 次，仅用于选择配置。
- Evaluation：每题 3 次，仅用于报告 pass@1 和稳定性。

所有题在正式使用前必须通过 Gold sanity check。未通过环境安装、Bug 复现或 Gold 修复的题
标记为 `infrastructure_blocked`，不进入可运行任务的准确率分母。

完整 ID 位于 `selection.json`；划分生成逻辑位于
`scripts/build_lottie_dataset_splits.py`。
