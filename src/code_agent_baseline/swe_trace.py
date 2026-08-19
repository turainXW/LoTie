from __future__ import annotations

import difflib
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .openhands_tools import OpenHandsToolParser, OpenHandsToolRuntime, ToolCall, ToolResult
from .swebench_local import LocalSweBenchTask


OPENHANDS_CORE_SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer to solve tasks.
<IMPORTANT>
* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.
</IMPORTANT>

You have access to the following functions:

---- BEGIN FUNCTION #1: execute_bash ----
Description: Execute a bash command in the terminal.
Parameters:
  (1) command (string, required): The bash command to execute.
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: finish ----
Description: Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
No parameters are required for this function.
---- END FUNCTION #2 ----

---- BEGIN FUNCTION #3: str_replace_editor ----
Description: Custom editing tool for viewing, creating and editing files
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* The `undo_edit` command will revert the last edit made to the file at `path`

Parameters:
  (1) command (string, required): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
  (2) path (string, required): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
  (3) file_text (string, optional): Required parameter of `create` command, with the content of the file to be created.
  (4) old_str (string, optional): Required parameter of `str_replace` command containing the string in `path` to replace.
  (5) new_str (string, optional): Optional parameter of `str_replace` command containing the new string.
  (6) insert_line (integer, optional): Required parameter of `insert` command.
  (7) view_range (array, optional): Optional parameter of `view` command when `path` points to a file.
---- END FUNCTION #3 ----

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
</IMPORTANT>
"""


@dataclass
class TestCaseResult:
    name: str
    passed: bool
    exit_code: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SweTraceStep:
    step: int
    assistant_message: str
    tool_call: dict[str, object]
    observation: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SweTrace:
    instance_id: str
    repo: str
    repo_path: str
    base_commit: str
    problem_statement: str
    hints_text: str
    resolved: bool
    validation_command: str
    patch: str
    gold_patch: str
    steps: list[SweTraceStep] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    report: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "repo_path": self.repo_path,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "hints_text": self.hints_text,
            "resolved": self.resolved,
            "validation_command": self.validation_command,
            "patch": self.patch,
            "gold_patch": self.gold_patch,
            "steps": [step.to_dict() for step in self.steps],
            "messages": self.messages,
            "report": self.report,
        }


Policy = Callable[[LocalSweBenchTask], list[str]]


def collect_swe_like_trace(task: LocalSweBenchTask, *, policy: Policy | None = None, timeout_sec: int = 60) -> SweTrace:
    repo = Path(task.repo_path)
    before = snapshot_files(repo)
    runtime = OpenHandsToolRuntime(repo, timeout_sec=timeout_sec, allowed_tools={"execute_bash", "str_replace_editor", "finish"})
    messages = build_initial_messages(task)
    steps: list[SweTraceStep] = []

    for step_num, assistant_message in enumerate((policy or deterministic_policy)(task), start=1):
        messages.append({"role": "assistant", "content": assistant_message})
        call = OpenHandsToolParser.parse_first(assistant_message)
        if call is None:
            result = ToolResult("parse_error", 1, "No tool call found.")
        else:
            result = runtime.execute(call)
        observation = format_observation(call, result)
        messages.append({"role": "user", "content": observation})
        steps.append(
            SweTraceStep(
                step=step_num,
                assistant_message=assistant_message,
                tool_call=call.to_dict() if call else {},
                observation=result.to_dict(),
            )
        )
        if call and call.tool_name == "finish":
            break

    after = snapshot_files(repo)
    patch = unified_diff_snapshot(before, after)
    final = run_command(task.validation_command, repo, timeout_sec=timeout_sec)
    fail_to_pass = run_named_tests(task.FAIL_TO_PASS, repo, timeout_sec=timeout_sec)
    pass_to_pass = run_named_tests(task.PASS_TO_PASS, repo, timeout_sec=timeout_sec)
    resolved = final.exit_code == 0 and all(item.passed for item in fail_to_pass + pass_to_pass)
    report = {
        "final_verifier": final.to_dict(),
        "FAIL_TO_PASS": [item.to_dict() for item in fail_to_pass],
        "PASS_TO_PASS": [item.to_dict() for item in pass_to_pass],
        "step_count": len(steps),
        "patch_line_count": len(patch.splitlines()),
    }
    return SweTrace(
        instance_id=task.instance_id,
        repo=task.repo,
        repo_path=str(repo),
        base_commit=task.base_commit,
        problem_statement=task.problem_statement,
        hints_text=task.hints_text,
        resolved=resolved,
        validation_command=task.validation_command,
        patch=patch,
        gold_patch=task.patch,
        steps=steps,
        messages=messages,
        report=report,
    )


def build_initial_messages(task: LocalSweBenchTask) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": OPENHANDS_CORE_SYSTEM_PROMPT},
        {"role": "user", "content": official_user_prompt(task)},
    ]


def official_user_prompt(task: LocalSweBenchTask) -> str:
    hint_block = f"\n\n<hints>\n{task.hints_text}\n</hints>" if task.hints_text else ""
    repo_path = Path(task.repo_path)
    return (
        "<uploaded_files>\n"
        f"{repo_path}\n"
        "</uploaded_files>\n"
        f"I've uploaded a python code repository in the directory {repo_path.name}. "
        "Consider the following PR description:\n\n"
        "<pr_description>\n"
        f"{task.problem_statement}"
        f"{hint_block}\n"
        "</pr_description>\n\n"
        "Can you help me implement the necessary changes to the repository so that the requirements specified in the "
        "<pr_description> are met?\n"
        "I've already taken care of all changes to any of the test files described in the <pr_description>. "
        "This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
        "Your task is to make the minimal changes to non-tests files in the /repo directory to ensure the "
        "<pr_description> is satisfied.\n"
        "Follow these steps to resolve the issue:\n"
        "1. As a first step, explore the repo to familiarize yourself with its structure.\n"
        "2. Create a script to reproduce the error and execute it when useful.\n"
        "3. Edit the source code of the repo to resolve the issue.\n"
        "4. Rerun your reproduce script or targeted tests and confirm that the error is fixed.\n"
        "5. Think about edge cases and make sure your fix handles them as well.\n"
        "If you think you have solved the task, first send your answer to user through message and then finish the interaction.\n"
        "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n"
    )


def deterministic_policy(task: LocalSweBenchTask) -> list[str]:
    if task.instance_id == "swe_local_0001_slug_parser":
        return [
            _bash("python3 -B -m unittest discover -s tests"),
            _view("text_utils.py"),
            _replace(
                "text_utils.py",
                "    text = text.strip().lower()\n    return text.replace(' ', '-')",
                "    text = text.strip().lower()\n    text = re.sub(r'[^a-z0-9]+', '-', text)\n    text = re.sub(r'-+', '-', text)\n    return text.strip('-')",
            ),
            _bash("python3 -B -m unittest discover -s tests"),
            _finish(),
        ]
    if task.instance_id == "swe_local_0002_window_average":
        return [
            _bash("python3 -B -m unittest discover -s tests"),
            _view("series.py"),
            _replace("series.py", "    for index in range(len(values) - window):", "    for index in range(len(values) - window + 1):"),
            _bash("python3 -B -m unittest discover -s tests"),
            _finish(),
        ]
    raise ValueError(f"No deterministic policy for {task.instance_id}")


def run_command(command: str, cwd: Path, *, timeout_sec: int) -> TestCaseResult:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    elapsed = round(time.monotonic() - started, 4)
    return TestCaseResult(
        name=command,
        passed=completed.returncode == 0,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=f"{completed.stderr}\n<elapsed_sec>{elapsed}</elapsed_sec>",
    )


def run_named_tests(test_names: list[str], cwd: Path, *, timeout_sec: int) -> list[TestCaseResult]:
    results = []
    for name in test_names:
        results.append(run_command(f"python3 -B -m unittest {name}", cwd, timeout_sec=timeout_sec))
    return results


def snapshot_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        files[rel] = path.read_text(encoding="utf-8", errors="replace")
    return files


def unified_diff_snapshot(before: dict[str, str], after: dict[str, str]) -> str:
    chunks: list[str] = []
    for rel in sorted(set(before) | set(after)):
        old = before.get(rel, "")
        new = after.get(rel, "")
        if old == new:
            continue
        chunks.extend(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )
    return "".join(chunks)


def format_observation(call: ToolCall | None, result: ToolResult) -> str:
    tool_name = call.tool_name if call else result.tool_name
    output = result.output.rstrip()
    if result.returncode is not None:
        output = output + f"\n[Command finished with exit code {result.returncode}]"
    return f"EXECUTION RESULT of [{tool_name}]:\n{output}"


def _bash(command: str) -> str:
    return f"<function=execute_bash>\n<parameter=command>{command}</parameter>\n</function>"


def _view(path: str) -> str:
    return f"<function=str_replace_editor>\n<parameter=command>view</parameter>\n<parameter=path>{path}</parameter>\n</function>"


def _replace(path: str, old: str, new: str) -> str:
    return (
        "<function=str_replace_editor>\n"
        "<parameter=command>str_replace</parameter>\n"
        f"<parameter=path>{path}</parameter>\n"
        f"<parameter=old_str>{old}</parameter>\n"
        f"<parameter=new_str>{new}</parameter>\n"
        "</function>"
    )


def _finish() -> str:
    return "<function=finish>\n</function>"
