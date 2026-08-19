from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


FinalStatus = Literal["pass", "fail", "timeout"]


@dataclass
class CodeTask:
    task_id: str
    repo_path: str
    instruction: str
    validation_command: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentAction:
    action_type: Literal["bash", "finish"]
    content: str
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentStep:
    step: int
    action: AgentAction
    stdout: str
    stderr: str
    exit_code: int
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.to_dict()
        return data


@dataclass
class Trajectory:
    task: CodeTask
    agent: str
    model: str
    initial_context: dict[str, Any] = field(default_factory=dict)
    steps: list[AgentStep] = field(default_factory=list)
    final_status: FinalStatus = "fail"
    final_stdout: str = ""
    final_stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "repo_path": self.task.repo_path,
            "instruction": self.task.instruction,
            "validation_command": self.task.validation_command,
            "agent": self.agent,
            "model": self.model,
            "initial_context": self.initial_context,
            "steps": [step.to_dict() for step in self.steps],
            "final_status": self.final_status,
            "final_stdout": self.final_stdout,
            "final_stderr": self.final_stderr,
        }
