# 开源发布检查清单

## 已自动处理

运行下面的命令生成精简发布包：

```bash
./scripts/package_open_source.sh v0.1.0
```

打包脚本会：

- 包含 Harness 源码、测试、部署工具、文档和 Eval90 必需任务文件。
- 排除 `.env`、venv、模型权重、运行轨迹、日志、PID、缓存和 Git 元数据。
- 清理任务清单中的本机仓库和虚拟环境绝对路径。
- 扫描常见 API key、AWS key、私钥、明文密码和开发机路径。
- 生成 `.tar.gz` 与对应 SHA256 文件。

## 发布前人工确认

1. 为项目选择许可证，并核对 EvalPlus、SWE-smith、mini-swe-agent 等上游许可证与署名要求。
2. 在全新 Linux 机器执行 `DEPLOY_EVAL90_zh.md` 中的 `doctor` 与 `prepare`。
3. 不提交本机生成的 `dist/`、`.codeagent/`、`outputs/`、模型权重和 SSH 配置。
4. README 中明确 SWE-smith local-venv verifier 的结果 `official_comparable=false`。
5. 发布 tag 后校验归档 SHA256，并记录 Python、任务数据和依赖版本。
