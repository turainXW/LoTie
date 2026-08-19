from __future__ import annotations

import re
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal

from .runtime_env import PROJECT_ROOT
from .web_tools import download_repo, web_search


os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", str(PROJECT_ROOT / ".mswea"))
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")


AgentMode = Literal["plan", "build", "debug"]


@dataclass(frozen=True)
class ModePolicy:
    mode: AgentMode
    require_approval_for_bash: bool = False
    forbidden_patterns: tuple[str, ...] = ()

    @staticmethod
    def for_mode(mode: AgentMode) -> "ModePolicy":
        write_patterns = (
            r"(^|[;&|]\s*)[^#\n]*\s(>|>>)\s*[A-Za-z0-9_./~-]",
            r"\btee\b",
            r"\bsed\s+-i\b",
            r"\bperl\s+-pi\b",
            r"\bmv\b",
            r"\bcp\b",
            r"\brm\b",
            r"\bgit\s+commit\b",
            r"\bgit\s+push\b",
            r"\bpython(?:3)?\b.*\bwrite_text\b",
            r"\bpython(?:3)?\b.*\bopen\(.*['\"]w",
        )
        destructive_patterns = (
            r"\bsudo\b",
            r"\bchmod\s+-R\b",
            r"\bchown\s+-R\b",
            r"\bmkfs\b",
            r"\bdd\s+",
        )
        if mode == "plan":
            return ModePolicy(
                mode=mode,
                require_approval_for_bash=False,
                forbidden_patterns=write_patterns + destructive_patterns,
            )
        if mode == "debug":
            return ModePolicy(
                mode=mode,
                require_approval_for_bash=False,
                forbidden_patterns=destructive_patterns,
            )
        return ModePolicy(
            mode=mode,
            require_approval_for_bash=False,
            forbidden_patterns=destructive_patterns,
        )

    def check_command(self, command: str) -> str | None:
        if self.require_approval_for_bash:
            return "Approval required before running bash in this mode."
        for pattern in self.forbidden_patterns:
            if re.search(pattern, command, re.IGNORECASE | re.DOTALL):
                return f"Action blocked in {self.mode} mode by pattern: {pattern}"
        return None


def build_guarded_local_environment_class():
    """Return a mini-swe-agent LocalEnvironment subclass with mode guards.

    Importing mini-swe-agent is intentionally delayed so the zero-dependency
    smoke baseline still works when mini is not installed.
    """

    try:
        from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mini-swe-agent is not installed. Install with: "
            "python3 -m pip install -r code_agent_quickstart/requirements-mini.txt"
        ) from exc

    class GuardedLocalEnvironmentConfig(LocalEnvironmentConfig):
        mode: AgentMode = "build"
        require_approval_for_bash: bool = False
        enable_web_tools: bool = False
        download_dir: str = "refs/open_source"

    class GuardedLocalEnvironment(LocalEnvironment):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, config_class=GuardedLocalEnvironmentConfig, **kwargs)
            policy = ModePolicy.for_mode(self.config.mode)
            self.policy = ModePolicy(
                mode=policy.mode,
                require_approval_for_bash=self.config.require_approval_for_bash,
                forbidden_patterns=policy.forbidden_patterns,
            )

        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
            tool = action.get("tool", "bash")
            if tool == "web_search":
                if not self.config.enable_web_tools:
                    return {
                        "output": "web_search is disabled. Re-run with --enable-web-tools.",
                        "returncode": 1,
                        "exception_info": "",
                        "extra": {"blocked": True, "mode": self.policy.mode, "tool": tool},
                    }
                result = web_search(action["query"], action.get("max_results", 5))
                return {
                    "output": json.dumps(result, ensure_ascii=False, indent=2),
                    "returncode": 0 if result.get("ok") else 1,
                    "exception_info": "",
                    "extra": {"tool": tool, "provider": result.get("provider")},
                }

            if tool == "download_repo":
                if not self.config.enable_web_tools:
                    return {
                        "output": "download_repo is disabled. Re-run with --enable-web-tools.",
                        "returncode": 1,
                        "exception_info": "",
                        "extra": {"blocked": True, "mode": self.policy.mode, "tool": tool},
                    }
                if self.policy.mode == "plan":
                    return {
                        "output": "download_repo is blocked in plan mode. Switch to build or debug mode.",
                        "returncode": 1,
                        "exception_info": "",
                        "extra": {"blocked": True, "mode": self.policy.mode, "tool": tool},
                    }
                cwd_path = Path(cwd or self.config.cwd or ".").resolve()
                download_dir = Path(self.config.download_dir)
                if not download_dir.is_absolute():
                    download_dir = cwd_path / download_dir
                result = download_repo(action["repo_url"], download_dir, action.get("dest_name"))
                return {
                    "output": json.dumps(result, ensure_ascii=False, indent=2),
                    "returncode": 0 if result.get("ok") else 1,
                    "exception_info": "",
                    "extra": {"tool": tool, "path": result.get("path")},
                }

            if tool != "bash":
                return {
                    "output": f"Unknown tool: {tool}",
                    "returncode": 1,
                    "exception_info": "",
                    "extra": {"blocked": True, "mode": self.policy.mode, "tool": tool},
                }

            command = action.get("command", "")
            block_reason = self.policy.check_command(command)
            if block_reason:
                return {
                    "output": block_reason,
                    "returncode": 1,
                    "exception_info": "",
                    "extra": {"blocked": True, "mode": self.policy.mode},
                }
            return super().execute(action, cwd, timeout=timeout)

        def serialize(self) -> dict:
            data = super().serialize()
            data["info"]["mode_policy"] = {
                "mode": self.policy.mode,
                "require_approval_for_bash": self.policy.require_approval_for_bash,
                "forbidden_patterns": list(self.policy.forbidden_patterns),
                "enable_web_tools": self.config.enable_web_tools,
                "download_dir": self.config.download_dir,
            }
            return data

    return GuardedLocalEnvironment
