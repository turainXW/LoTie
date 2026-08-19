from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod

from .schemas import AgentAction, AgentStep, CodeTask


class ModelBackend(ABC):
    name: str

    @abstractmethod
    def next_action(self, task: CodeTask, history: list[AgentStep]) -> AgentAction:
        raise NotImplementedError


class RuleBasedSmokeModel(ModelBackend):
    """A deterministic backend for validating the agent loop without an LLM."""

    name = "rule_based_smoke"

    def next_action(self, task: CodeTask, history: list[AgentStep]) -> AgentAction:
        if not history:
            return AgentAction(
                action_type="bash",
                content=task.validation_command,
                rationale="Run the failing validation command first.",
            )

        last = history[-1]
        if last.exit_code == 0:
            return AgentAction(
                action_type="finish",
                content="Validation passed.",
                rationale="The task is solved.",
            )

        if len(history) == 1:
            return AgentAction(
                action_type="bash",
                content=(
                    "python3 - <<'PY'\n"
                    "from pathlib import Path\n"
                    "import shutil\n"
                    "p = Path('calculator.py')\n"
                    "text = p.read_text()\n"
                    "text = text.replace('return a - b', 'return a + b')\n"
                    "p.write_text(text)\n"
                    "shutil.rmtree('__pycache__', ignore_errors=True)\n"
                    "PY"
                ),
                rationale="Patch the known toy bug in calculator.py.",
            )

        return AgentAction(
            action_type="bash",
            content=task.validation_command,
            rationale="Re-run validation after applying the patch.",
        )


class RuleBasedRepairBenchmarkModel(ModelBackend):
    """Deterministic benchmark backend for testing the CodeAgent loop itself.

    This is not an LLM replacement for real capability evaluation. It is a
    fixture backend that lets the evaluator verify task setup, command
    execution, patch application, validation, and trajectory recording.
    """

    name = "rule_based_repair_benchmark"

    def next_action(self, task: CodeTask, history: list[AgentStep]) -> AgentAction:
        if not history:
            return AgentAction(
                action_type="bash",
                content=task.validation_command,
                rationale="Run the verifier before editing.",
            )

        last = history[-1]
        if last.exit_code == 0:
            return AgentAction(
                action_type="finish",
                content="Validation passed.",
                rationale="The task is solved.",
            )

        if len(history) == 1:
            patch = _patch_command_for_task(task.task_id)
            return AgentAction(
                action_type="bash",
                content=patch,
                rationale=f"Apply the known minimal repair for {task.task_id}.",
            )

        return AgentAction(
            action_type="bash",
            content=task.validation_command,
            rationale="Re-run validation after applying the patch.",
        )


def _patch_command_for_task(task_id: str) -> str:
    patches = {
        "code_repair_0001_addition": (
            "python3 - <<'PY'\n"
            "from pathlib import Path\n"
            "p = Path('calculator.py')\n"
            "p.write_text('def add(a, b):\\n    return a + b\\n', encoding='utf-8')\n"
            "PY"
        ),
        "code_repair_0002_slugify": (
            "python3 - <<'PY'\n"
            "from pathlib import Path\n"
            "p = Path('text_utils.py')\n"
            "p.write_text('import re\\n\\n'\n"
            "             'def slugify(text):\\n'\n"
            "             '    text = text.strip().lower()\\n'\n"
            "             '    text = re.sub(r\\\"[^a-z0-9\\\\s-]\\\", \\\"\\\", text)\\n'\n"
            "             '    return re.sub(r\\\"[\\\\s-]+\\\", \\\"-\\\", text).strip(\\\"-\\\")\\n',\n"
            "             encoding='utf-8')\n"
            "PY"
        ),
        "code_repair_0003_mean": (
            "python3 - <<'PY'\n"
            "from pathlib import Path\n"
            "p = Path('stats_utils.py')\n"
            "p.write_text('def mean(values):\\n'\n"
            "             '    if not values:\\n'\n"
            "             '        raise ValueError(\\\"values must not be empty\\\")\\n'\n"
            "             '    return sum(values) / len(values)\\n',\n"
            "             encoding='utf-8')\n"
            "PY"
        ),
    }
    try:
        return patches[task_id]
    except KeyError as exc:
        raise ValueError(f"No benchmark patch fixture for task_id={task_id}") from exc


class OpenAICompatibleModel(ModelBackend):
    """Minimal OpenAI-compatible chat client using only the Python stdlib."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self.name = model
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")

    def next_action(self, task: CodeTask, history: list[AgentStep]) -> AgentAction:
        if not self.api_key:
            raise RuntimeError("Missing API key for OpenAI-compatible model backend.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a coding agent. Return strict JSON with keys: "
                    "action_type, content, rationale. action_type is bash or finish."
                ),
            },
            {
                "role": "user",
                "content": self._build_prompt(task, history),
            },
        ]
        payload = json.dumps({"model": self.model, "messages": messages}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = json.loads(response.read().decode())
        content = raw["choices"][0]["message"]["content"]
        data = json.loads(content)
        return AgentAction(
            action_type=data["action_type"],
            content=data["content"],
            rationale=data.get("rationale", ""),
        )

    @staticmethod
    def _build_prompt(task: CodeTask, history: list[AgentStep]) -> str:
        lines = [
            f"Task: {task.instruction}",
            f"Validation command: {task.validation_command}",
            "History:",
        ]
        for step in history:
            lines.append(f"Step {step.step}: {step.action.content}")
            lines.append(f"Exit code: {step.exit_code}")
            lines.append(f"Stdout:\n{step.stdout[-2000:]}")
            lines.append(f"Stderr:\n{step.stderr[-2000:]}")
        lines.append("Choose the next action.")
        return "\n\n".join(lines)
