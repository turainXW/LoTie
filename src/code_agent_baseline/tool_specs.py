from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    use_when: str
    avoid_when: str
    parameters: dict[str, Any]
    output: str
    example: dict[str, Any]

    def api_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def object_schema(
    properties: dict[str, Any] | None = None,
    *,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }


TOOL_SPECS: dict[str, ToolSpec] = {
    "execute_bash": ToolSpec(
        name="execute_bash",
        description="Run one shell command inside the prepared repository workspace and task environment.",
        use_when="Search code, inspect metadata, run a reproducer, compile, or perform a command not covered by a structured tool.",
        avoid_when="Do not activate/install environments. Prefer run_tests for pytest/unittest and git_diff for patch inspection.",
        parameters=object_schema(
            {"command": {"type": "string", "minLength": 1, "description": "Non-empty shell command."}},
            required=["command"],
        ),
        output="returncode, stdout/stderr text, elapsed_sec, and the normalized command.",
        example={"tool_name": "execute_bash", "arguments": {"command": "rg -n \"normalize_name\" src tests"}},
    ),
    "str_replace_editor": ToolSpec(
        name="str_replace_editor",
        description="View or edit one repository-relative file with deterministic structured operations.",
        use_when="Read a focused line range or make the smallest supported source edit.",
        avoid_when="Do not use view repeatedly for ranges already observed. Do not edit tests unless the task explicitly requires it.",
        parameters=object_schema(
            {
                "command": {
                    "type": "string",
                    "enum": ["view", "str_replace", "line_replace", "insert", "create", "undo_edit"],
                    "description": "Editor operation.",
                },
                "path": {"type": "string", "minLength": 1, "description": "Repository-relative path."},
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Inclusive [start, end] lines for view; end=-1 means EOF.",
                },
                "old_str": {"type": "string", "description": "Unique source text required by str_replace."},
                "new_str": {"type": "string", "description": "Replacement/insertion text."},
                "file_text": {"type": "string", "description": "Full contents required by create."},
                "insert_line": {"type": "integer", "minimum": 0},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            required=["command", "path"],
        ),
        output="returncode plus numbered file contents for view, or a concise edit confirmation/error.",
        example={
            "tool_name": "str_replace_editor",
            "arguments": {
                "command": "str_replace",
                "path": "src/pkg/names.py",
                "old_str": "return value.lower()",
                "new_str": "return value.strip().lower()",
            },
        },
    ),
    "run_tests": ToolSpec(
        name="run_tests",
        description="Run tests in the harness-selected task environment and return structured counts.",
        use_when="Validate a focused test after editing; omit arguments to execute the task's configured verifier command.",
        avoid_when="Do not treat all-skipped tests as success. Do not use execute_bash for pytest when this tool is sufficient.",
        parameters=object_schema(
            {
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional pytest node IDs or test paths.",
                },
                "command": {"type": "string", "description": "Optional project-specific test command."},
            }
        ),
        output="returncode, test output, passed/failed/errors/skipped counts, all_skipped, elapsed_sec, and venv_path.",
        example={
            "tool_name": "run_tests",
            "arguments": {"targets": ["tests/test_names.py::test_normalize_name"]},
        },
    ),
    "git_diff": ToolSpec(
        name="git_diff",
        description="Show the current unstaged Git patch and list edited files.",
        use_when="Confirm the patch is minimal after editing and before finish.",
        avoid_when="Do not call repeatedly before any edit or use it as a substitute for tests.",
        parameters=object_schema(),
        output="returncode, unified diff text, edited_files, and has_changes.",
        example={"tool_name": "git_diff", "arguments": {}},
    ),
    "finish": ToolSpec(
        name="finish",
        description="Stop the rollout and submit the current workspace patch to the harness.",
        use_when="Immediately after the latest edit has a passing run_tests result and git_diff has been reviewed, or the task is genuinely impossible.",
        avoid_when="Do not finish immediately after reading, with untested edits, or before run_tests and git_diff after a patch.",
        parameters=object_schema(),
        output="A finished acknowledgement; the harness then extracts and verifies the Git patch.",
        example={"tool_name": "finish", "arguments": {}},
    ),
    "repo_context": ToolSpec(
        name="repo_context",
        description="Build/search the codebase map and return task-ranked files, symbols, imports, and snippets.",
        use_when="Start work in an unfamiliar or large repository and need likely files before raw reads.",
        avoid_when="Do not repeat the same query after the relevant source file has already been identified.",
        parameters=object_schema(
            {
                "query": {"type": "string", "description": "Task, symbol, error, or behavior to locate."},
                "max_files": {"type": "integer", "minimum": 1, "maximum": 30, "default": 12},
                "max_snippets": {"type": "integer", "minimum": 0, "maximum": 20, "default": 6},
                "refresh": {"type": "boolean", "default": False},
            },
            required=["query"],
        ),
        output="Ranked file candidates and compact evidence snippets plus repo-map metadata.",
        example={"tool_name": "repo_context", "arguments": {"query": "normalize_name whitespace behavior"}},
    ),
    "problem_search": ToolSpec(
        name="problem_search",
        description="Search local tasks, traces, and notes for similar debugging experience.",
        use_when="A prior local solution pattern may clarify an unfamiliar failure.",
        avoid_when="Do not copy a prior patch blindly or use it when repository evidence is already sufficient.",
        parameters=object_schema(
            {
                "query": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                "index_path": {"type": "string"},
            },
            required=["query"],
        ),
        output="Ranked local problem/trajectory matches with source metadata.",
        example={"tool_name": "problem_search", "arguments": {"query": "format exception parameter order", "top_k": 3}},
    ),
    "web_search": ToolSpec(
        name="web_search",
        description="Search public web sources through the configured search provider.",
        use_when="Current public documentation or an upstream API contract is necessary in use mode.",
        avoid_when="Disabled for fair benchmark runs; never search for benchmark answers or Gold patches.",
        parameters=object_schema(
            {
                "query": {"type": "string", "minLength": 1},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            required=["query"],
        ),
        output="Provider, query, success flag, and a list of titled URLs with descriptions.",
        example={"tool_name": "web_search", "arguments": {"query": "Python traceback format_exception_only signature", "max_results": 5}},
    ),
    "download_repo": ToolSpec(
        name="download_repo",
        description="Download a public GitHub repository into a workspace-controlled downloads directory.",
        use_when="The user explicitly needs an additional public repository in use mode.",
        avoid_when="Do not use for the current task repository; the harness already prepared it.",
        parameters=object_schema(
            {
                "repo_url": {"type": "string", "minLength": 1},
                "download_dir": {"type": "string"},
                "dest_name": {"type": "string"},
            },
            required=["repo_url"],
        ),
        output="Download status, destination path, source URL, and error details when blocked.",
        example={"tool_name": "download_repo", "arguments": {"repo_url": "https://github.com/org/project"}},
    ),
    "answer": ToolSpec(
        name="answer",
        description="Return a concise user-facing response when no repository action is required.",
        use_when="Answer a general question or report a genuine blocker in use mode.",
        avoid_when="Do not use instead of editing when the user requested a repository change.",
        parameters=object_schema(
            {"content": {"type": "string", "minLength": 1, "description": "Concise final response."}},
            required=["content"],
        ),
        output="The supplied content is displayed to the user and the turn ends.",
        example={"tool_name": "answer", "arguments": {"content": "The repository uses Python 3.10."}},
    ),
}


CORE_REPAIR_TOOLS = ("execute_bash", "str_replace_editor", "run_tests", "git_diff", "finish")
SWE_EXTENSION_TOOLS = ("repo_context", "problem_search", "web_search", "download_repo")


def tool_schemas(names: Iterable[str]) -> list[dict[str, Any]]:
    return [TOOL_SPECS[name].api_schema() for name in names]


def render_tool_catalog(names: Iterable[str]) -> str:
    selected_names = list(names)
    blocks = ["<TOOLS>"]
    for name in selected_names:
        spec = TOOL_SPECS[name]
        properties = spec.parameters.get("properties", {})
        required = set(spec.parameters.get("required", []))
        arguments = []
        for argument, schema in properties.items():
            kind = str(schema.get("type", "any"))
            requirement = "required" if argument in required else "optional"
            enum = f", one of {schema['enum']}" if schema.get("enum") else ""
            arguments.append(f"    - {argument}: {kind}, {requirement}{enum}. {schema.get('description', '')}".rstrip())
        if not arguments:
            arguments.append("    - no arguments")
        blocks.extend(
            [
                f"- {spec.name}: {spec.description}",
                f"  Use when: {spec.use_when}",
                f"  Avoid when: {spec.avoid_when}",
                "  Arguments:",
                *arguments,
                f"  Returns: {spec.output}",
                f"  Example: {json.dumps(spec.example, ensure_ascii=False, separators=(',', ':'))}",
            ]
        )
    if "finish" in selected_names:
        blocks.append(
            "Completion gate: when a patch exists, the harness rejects finish until run_tests and git_diff have "
            "both been called. A successful custom reproducer in execute_bash does not replace this structured gate."
        )
    blocks.append("</TOOLS>")
    return "\n".join(blocks)


def render_tool_few_shot(*, include_answer: bool = False) -> str:
    lines = [
        "<TOOL_FEW_SHOT>",
        "This is a generic workflow example, not repository evidence and not a solution to the current task.",
        "User: Fix normalize_name so surrounding whitespace is ignored and None is handled, then validate the focused test.",
        'Assistant: {"tool_name":"str_replace_editor","arguments":{"command":"view","path":"src/pkg/names.py","view_range":[1,80]}}',
        "Tool result: lines 1-8 show `return value.lower()`.",
        'Assistant: {"tool_name":"str_replace_editor","arguments":{"command":"str_replace","path":"src/pkg/names.py","old_str":"return value.lower()","new_str":"return value.strip().lower()"}}',
        "Tool result: replacement succeeded.",
        'Assistant: {"tool_name":"run_tests","arguments":{"targets":["tests/test_names.py::test_normalize_name"]}}',
        "Tool result: returncode=1, passed=1, failed=1; None still raises AttributeError.",
        'Assistant: {"tool_name":"str_replace_editor","arguments":{"command":"str_replace","path":"src/pkg/names.py","old_str":"return value.strip().lower()","new_str":"return \"\" if value is None else value.strip().lower()"}}',
        "Tool result: replacement succeeded.",
        'Assistant: {"tool_name":"run_tests","arguments":{"targets":["tests/test_names.py::test_normalize_name"]}}',
        "Tool result: returncode=0, passed=2, failed=0, skipped=0.",
        'Assistant: {"tool_name":"git_diff","arguments":{}}',
        "Tool result: one minimal source-file patch; has_changes=true. The completion gate is satisfied, so the next action must be finish.",
        'Assistant: {"tool_name":"finish","arguments":{}}',
    ]
    if include_answer:
        lines.extend(
            [
                "General-question example:",
                "User: Which Python environment is active?",
                'Assistant: {"tool_name":"answer","arguments":{"content":"The harness-selected task environment is active."}}',
            ]
        )
    lines.append("</TOOL_FEW_SHOT>")
    return "\n".join(lines)
