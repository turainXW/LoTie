from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .model_clients import ModelBackend
from .repo_context import build_repo_map, compact_repo_context
from .runtime_env import project_runtime_env
from .schemas import AgentStep, CodeTask, Trajectory


class CodeAgent:
    def __init__(
        self,
        model: ModelBackend,
        max_steps: int = 8,
        command_timeout_sec: int = 30,
        env: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.command_timeout_sec = command_timeout_sec
        self.env = env or project_runtime_env()

    def run(self, task: CodeTask) -> Trajectory:
        initial_context = self._build_initial_context(task)
        trajectory = Trajectory(
            task=task,
            agent="code_agent_baseline",
            model=self.model.name,
            initial_context=initial_context,
        )
        repo = Path(task.repo_path)
        for step_num in range(1, self.max_steps + 1):
            action = self.model.next_action(task, trajectory.steps)
            if action.action_type == "finish":
                break
            started = time.monotonic()
            completed = subprocess.run(
                action.content,
                cwd=repo,
                shell=True,
                text=True,
                capture_output=True,
                timeout=self.command_timeout_sec,
                env=self.env,
            )
            elapsed = time.monotonic() - started
            trajectory.steps.append(
                AgentStep(
                    step=step_num,
                    action=action,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    exit_code=completed.returncode,
                    elapsed_sec=round(elapsed, 4),
                )
            )

            status, stdout, stderr = self.validate(task)
            trajectory.final_status = status
            trajectory.final_stdout = stdout
            trajectory.final_stderr = stderr
            if status == "pass":
                return trajectory

        status, stdout, stderr = self.validate(task)
        trajectory.final_status = status
        trajectory.final_stdout = stdout
        trajectory.final_stderr = stderr
        if status != "pass" and len(trajectory.steps) >= self.max_steps:
            trajectory.final_status = "timeout"
        return trajectory

    def validate(self, task: CodeTask) -> tuple[str, str, str]:
        completed = subprocess.run(
            task.validation_command,
            cwd=task.repo_path,
            shell=True,
            text=True,
            capture_output=True,
            timeout=self.command_timeout_sec,
            env=self.env,
        )
        status = "pass" if completed.returncode == 0 else "fail"
        return status, completed.stdout, completed.stderr

    @staticmethod
    def _build_initial_context(task: CodeTask) -> dict[str, object]:
        try:
            repo_map = build_repo_map(task.repo_path)
            compact_map = compact_repo_context(repo_map, task.instruction, max_files=8, max_chars=3000)
            files = [file_ctx.to_dict() for file_ctx in repo_map.files]
        except Exception as exc:  # pragma: no cover - context capture must not break task execution.
            return {
                "context_version": "codeagent_context_v1",
                "capture_error": str(exc),
            }
        return {
            "context_version": "codeagent_context_v1",
            "repo_context": {
                "compact_map": compact_map,
                "file_count": len(files),
                "files": files,
            },
            "runtime_context": {
                "cwd": task.repo_path,
                "validation_command": task.validation_command,
            },
            "policy_context": {
                "action_space": ["bash", "finish"],
                "success_signal": "validation_command exit_code == 0",
            },
        }

    @staticmethod
    def append_jsonl(path: str | Path, trajectory: Trajectory) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")
