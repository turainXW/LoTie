from __future__ import annotations

import html
import json
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .runtime_env import project_runtime_env


TOOL_CALL_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
MALFORMED_PARAM_RE = re.compile(r"<parameter>([A-Za-z_][A-Za-z0-9_]*)>(.*?)</parameter>", re.DOTALL)
LENIENT_FUNCTION_RE = re.compile(
    r"<function=([^>]+)>(.*?)(?:</function>|</(?:figure|tool_call|observation)>)",
    re.DOTALL,
)
DIRECT_TOOL_RE = re.compile(
    r"<(execute_bash|str_replace_editor|run_tests|git_diff|finish|repo_context|problem_search|web_search|download_repo|answer)>(.*?)</\1>",
    re.DOTALL,
)
DIRECT_PARAM_RE = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>(.*?)</\1>", re.DOTALL)
INVOKE_TOOL_RE = re.compile(
    r"<invoke\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</invoke>",
    re.DOTALL,
)
DSML_INVOKE_RE = re.compile(
    r"<｜｜DSML｜｜invoke\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</｜｜DSML｜｜invoke>",
    re.DOTALL,
)
DSML_PARAM_RE = re.compile(
    r"<｜｜DSML｜｜parameter\s+name=[\"']([^\"']+)[\"'](?:\s+string=[\"'][^\"']+[\"'])?\s*>"
    r"(.*?)</｜｜DSML｜｜parameter>",
    re.DOTALL,
)
TAGGED_TOOL_NAME_RE = re.compile(
    r"<(?:tool_invoke|｜｜DSML｜｜)>\s*"
    r"<tool_name>([^<]+)</(?:tool_name|｜｜DSML｜｜)>"
    r"(.*?)(?:</invoke>|</tool_invoke>|</｜｜DSML｜｜>)",
    re.DOTALL,
)
QWEN_NAMED_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
FENCED_BASH_RE = re.compile(r"```(?:bash|sh|shell)\s*(.*?)```", re.DOTALL | re.IGNORECASE)
PARTIAL_FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*)", re.DOTALL)
SHELL_COMMAND_RE = re.compile(
    r"^(?:PYTHONPATH=\S+\s+|[A-Z_][A-Z0-9_]*=\S+\s+)*(?:python(?:\d+(?:\.\d+)?)?|pytest|pip|git|rg|grep|sed|cat|find|ls|pwd|head|tail|awk|perl|cp|mkdir|touch)\b"
)
FINISH_RE = re.compile(r"\b(done|finished|complete|completed|fixed|resolved|implemented|ready)\b", re.IGNORECASE)


CORE_TOOLS = {"execute_bash", "str_replace_editor", "run_tests", "git_diff", "finish"}
EXTENSION_TOOLS = {"web_search", "download_repo", "repo_context", "memory_search", "problem_search"}


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    tool_name: str
    returncode: int
    output: str
    elapsed_sec: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenHandsToolParser:
    @staticmethod
    def parse_first(text: str) -> ToolCall | None:
        match = TOOL_CALL_RE.search(text)
        if match:
            return OpenHandsToolParser._parse_function_match(match)
        lenient_match = LENIENT_FUNCTION_RE.search(text)
        if lenient_match:
            return OpenHandsToolParser._parse_function_match(lenient_match)
        direct_match = DIRECT_TOOL_RE.search(text)
        if direct_match:
            return OpenHandsToolParser._parse_direct_match(direct_match)
        dsml_match = DSML_INVOKE_RE.search(text)
        if dsml_match:
            return OpenHandsToolParser._parse_dsml_match(dsml_match)
        invoke_match = INVOKE_TOOL_RE.search(text)
        if invoke_match:
            return OpenHandsToolParser._parse_invoke_match(invoke_match)
        tagged_tool_match = TAGGED_TOOL_NAME_RE.search(text)
        if tagged_tool_match:
            return OpenHandsToolParser._parse_invoke_match(tagged_tool_match)
        qwen_match = QWEN_NAMED_TOOL_CALL_RE.search(text)
        if qwen_match:
            return OpenHandsToolParser._parse_qwen_named_match(qwen_match)
        fenced_match = FENCED_BASH_RE.search(text)
        if fenced_match:
            return ToolCall(
                tool_name="execute_bash",
                arguments={"command": html.unescape(fenced_match.group(1).strip())},
                raw=fenced_match.group(0),
            )
        json_call = OpenHandsToolParser._parse_json_tool_call(text)
        if json_call is not None:
            return json_call
        partial_match = PARTIAL_FUNCTION_RE.search(text)
        if partial_match:
            return OpenHandsToolParser._parse_function_match(partial_match)
        shell_call = OpenHandsToolParser._parse_shell_line(text)
        if shell_call is not None:
            return shell_call
        if FINISH_RE.search(text) and not any(marker in text for marker in ("Traceback", "Error:", "failed")):
            return ToolCall(tool_name="finish", arguments={}, raw=text)
        return None

    @staticmethod
    def parse_all(text: str) -> list[ToolCall]:
        calls = []
        for match in TOOL_CALL_RE.finditer(text):
            calls.append(OpenHandsToolParser._parse_function_match(match))
        for match in DIRECT_TOOL_RE.finditer(text):
            calls.append(OpenHandsToolParser._parse_direct_match(match))
        for match in DSML_INVOKE_RE.finditer(text):
            calls.append(OpenHandsToolParser._parse_dsml_match(match))
        for match in INVOKE_TOOL_RE.finditer(text):
            calls.append(OpenHandsToolParser._parse_invoke_match(match))
        for match in TAGGED_TOOL_NAME_RE.finditer(text):
            calls.append(OpenHandsToolParser._parse_invoke_match(match))
        return calls

    @staticmethod
    def _parse_function_match(match: re.Match[str]) -> ToolCall:
        tool_name = match.group(1).strip()
        body = match.group(2)
        arguments = {}
        for param_name, value in PARAM_RE.findall(body):
            arguments[param_name.strip()] = html.unescape(value.strip())
        for param_name, value in MALFORMED_PARAM_RE.findall(body):
            arguments.setdefault(param_name.strip(), html.unescape(value.strip()))
        return ToolCall(tool_name=tool_name, arguments=arguments, raw=match.group(0))

    @staticmethod
    def _parse_direct_match(match: re.Match[str]) -> ToolCall:
        tool_name = match.group(1).strip()
        body = match.group(2)
        arguments = {}
        for param_name, value in DIRECT_PARAM_RE.findall(body):
            if param_name == tool_name:
                continue
            arguments[param_name.strip()] = html.unescape(value.strip())
        return ToolCall(tool_name=tool_name, arguments=arguments, raw=match.group(0))

    @staticmethod
    def _parse_invoke_match(match: re.Match[str]) -> ToolCall:
        tool_name = match.group(1).strip()
        body = match.group(2)
        arguments = {}
        for param_name, value in DIRECT_PARAM_RE.findall(body):
            arguments[param_name.strip()] = html.unescape(value.strip())
        return ToolCall(tool_name=tool_name, arguments=arguments, raw=match.group(0))

    @staticmethod
    def _parse_dsml_match(match: re.Match[str]) -> ToolCall:
        tool_name = match.group(1).strip()
        arguments: dict[str, Any] = {}
        for param_name, value in DSML_PARAM_RE.findall(match.group(2)):
            name = param_name.strip()
            decoded = html.unescape(value.strip())
            if name == "view_range" and re.fullmatch(r"-?\d+\s*,\s*-?\d+", decoded):
                arguments[name] = [int(part.strip()) for part in decoded.split(",")]
            elif name in {"insert_line", "start_line", "end_line"} and re.fullmatch(r"-?\d+", decoded):
                arguments[name] = int(decoded)
            else:
                arguments[name] = decoded
        return ToolCall(tool_name=tool_name, arguments=arguments, raw=match.group(0))

    @staticmethod
    def _parse_qwen_named_match(match: re.Match[str]) -> ToolCall:
        tool_name = match.group(1).strip()
        try:
            payload = json.loads(match.group(2))
        except json.JSONDecodeError:
            payload = {}
        arguments = payload.get("arguments") if isinstance(payload, dict) else {}
        if not isinstance(arguments, dict):
            arguments = {}
        return ToolCall(
            tool_name=tool_name,
            arguments={str(key): value for key, value in arguments.items()},
            raw=match.group(0),
        )

    @staticmethod
    def _parse_json_tool_call(text: str) -> ToolCall | None:
        stripped = text.strip()
        candidates = [stripped]
        fenced_json = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced_json:
            candidates.insert(0, fenced_json.group(1).strip())
        candidates.extend(_json_object_candidates(stripped))
        candidates.extend(_json_close_repair_candidates(stripped))
        candidates.extend(_json_missing_key_quote_repair_candidates(stripped))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            name = payload.get("tool_name") or payload.get("name") or payload.get("function")
            if not isinstance(name, str):
                continue
            arguments = payload.get("arguments") or payload.get("parameters") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            return ToolCall(tool_name=name, arguments={str(k): v for k, v in arguments.items()}, raw=candidate)
        return None

    @staticmethod
    def _parse_shell_line(text: str) -> ToolCall | None:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if not lines:
            return None
        if len(lines) > 3:
            return None
        for line in lines:
            command = line[2:].strip() if line.startswith("$ ") else line
            if SHELL_COMMAND_RE.match(command):
                return ToolCall(tool_name="execute_bash", arguments={"command": html.unescape(command)}, raw=line)
        return None


def _json_close_repair_candidates(text: str) -> list[str]:
    if not text.startswith("{"):
        return []
    missing_braces = text.count("{") - text.count("}")
    missing_brackets = text.count("[") - text.count("]")
    if missing_braces < 0 or missing_brackets < 0:
        return []
    if missing_braces + missing_brackets == 0 or missing_braces + missing_brackets > 3:
        return []
    repaired = text + ("]" * missing_brackets) + ("}" * missing_braces)
    return [repaired]


def _json_missing_key_quote_repair_candidates(text: str) -> list[str]:
    if not text.startswith("{"):
        return []
    repaired = re.sub(r'"(arguments|parameters)\s*:', r'"\1":', text)
    return [repaired] if repaired != text else []


def _json_object_candidates(text: str) -> list[str]:
    decoder = json.JSONDecoder()
    candidates = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(text[index : index + end].strip())
        if len(candidates) >= 5:
            break
    return candidates


class OpenHandsToolRuntime:
    def __init__(
        self,
        workspace: str | Path,
        *,
        timeout_sec: int = 120,
        max_view_chars: int = 20_000,
        max_command_output_chars: int | None = None,
        allowed_tools: set[str] | None = None,
        runtime_env: dict[str, str] | None = None,
        test_command: str | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.timeout_sec = timeout_sec
        self.max_view_chars = max_view_chars
        self.max_command_output_chars = max_command_output_chars or max_view_chars
        self.allowed_tools = allowed_tools or (CORE_TOOLS | EXTENSION_TOOLS)
        self.env = dict(runtime_env) if runtime_env is not None else project_runtime_env(base_env=None)
        self.test_command = test_command
        self._undo_stack: dict[Path, list[str | None]] = {}
        self._output_artifact_index = 0

    def execute(self, call: ToolCall) -> ToolResult:
        call = coerce_common_tool_alias(call)
        if call.tool_name not in self.allowed_tools:
            return ToolResult(
                tool_name=call.tool_name,
                returncode=1,
                output=f"Tool not allowed: {call.tool_name}",
                extra={"blocked": True},
            )
        if call.tool_name == "execute_bash":
            return self._execute_bash(call)
        if call.tool_name == "str_replace_editor":
            return self._str_replace_editor(call)
        if call.tool_name == "run_tests":
            return self._run_tests(call)
        if call.tool_name == "git_diff":
            return self._git_diff(call)
        if call.tool_name == "finish":
            return ToolResult(tool_name=call.tool_name, returncode=0, output="Finished.", extra={"finished": True})
        if call.tool_name == "repo_context":
            return self._repo_context(call)
        if call.tool_name == "problem_search":
            return self._problem_search(call)
        if call.tool_name == "web_search":
            return self._web_search(call)
        if call.tool_name == "download_repo":
            return self._download_repo(call)
        return ToolResult(
            tool_name=call.tool_name,
            returncode=1,
            output=f"Extension tool is declared but not implemented in this runtime yet: {call.tool_name}",
            extra={"extension_tool": True, "implemented": False},
        )

    def _web_search(self, call: ToolCall) -> ToolResult:
        from .web_tools import web_search

        query = str(call.arguments.get("query", ""))
        if not query.strip():
            return ToolResult(call.tool_name, 1, "query is required", extra={"invalid_arguments": True})
        max_results = int(str(call.arguments.get("max_results", "5")))
        started = time.monotonic()
        result = web_search(query, max_results=max_results)
        return ToolResult(
            tool_name=call.tool_name,
            returncode=0 if result.get("ok") else 1,
            output=json.dumps(result, ensure_ascii=False, indent=2),
            elapsed_sec=round(time.monotonic() - started, 4),
            extra={"extension_tool": True, "implemented": True, "provider": result.get("provider")},
        )

    def _download_repo(self, call: ToolCall) -> ToolResult:
        from .web_tools import download_repo

        repo_url = str(call.arguments.get("repo_url", call.arguments.get("url", "")))
        if not repo_url.strip():
            return ToolResult(call.tool_name, 1, "repo_url is required", extra={"invalid_arguments": True})
        download_dir = str(call.arguments.get("download_dir", self.workspace / ".codeagent" / "downloads"))
        dest_name = call.arguments.get("dest_name")
        started = time.monotonic()
        result = download_repo(repo_url, download_dir=download_dir, dest_name=str(dest_name) if dest_name else None)
        return ToolResult(
            tool_name=call.tool_name,
            returncode=0 if result.get("ok") else 1,
            output=json.dumps(result, ensure_ascii=False, indent=2),
            elapsed_sec=round(time.monotonic() - started, 4),
            extra={"extension_tool": True, "implemented": True},
        )

    def _problem_search(self, call: ToolCall) -> ToolResult:
        from .problem_search import problem_search

        query = str(call.arguments.get("query", ""))
        top_k = int(str(call.arguments.get("top_k", call.arguments.get("max_results", "5"))))
        index_path = call.arguments.get("index_path")
        result = problem_search(
            query,
            index_path=str(index_path) if index_path else None,
            roots=[self.workspace],
            top_k=top_k,
        )
        return ToolResult(
            tool_name=call.tool_name,
            returncode=0,
            output=json.dumps(result, ensure_ascii=False, indent=2),
            extra={"extension_tool": True, "implemented": True},
        )

    def _repo_context(self, call: ToolCall) -> ToolResult:
        from .repo_context import build_repo_map, load_repo_map, save_repo_map, search_repo_context

        query = str(call.arguments.get("query", ""))
        max_files = int(str(call.arguments.get("max_files", "12")))
        max_snippets = int(str(call.arguments.get("max_snippets", "6")))
        refresh = str(call.arguments.get("refresh", "false")).lower() in {"1", "true", "yes"}
        map_path = self.workspace / ".codeagent" / "repo_map.json"
        if refresh or not map_path.exists():
            repo_map = build_repo_map(self.workspace)
            save_repo_map(repo_map, map_path)
        else:
            repo_map = load_repo_map(map_path)
        result = search_repo_context(
            repo_map,
            query,
            max_files=max_files,
            max_snippets=max_snippets,
            max_chars=self.max_view_chars,
        )
        return ToolResult(
            tool_name=call.tool_name,
            returncode=0,
            output=json.dumps(result, ensure_ascii=False, indent=2),
            extra={
                "extension_tool": True,
                "implemented": True,
                "repo_map_path": str(map_path),
                "file_count": repo_map.stats.get("file_count", len(repo_map.files)),
            },
        )

    def _execute_bash(self, call: ToolCall) -> ToolResult:
        command = sanitize_workspace_cd(str(call.arguments.get("command", "")))
        if not command.strip():
            return ToolResult(call.tool_name, 1, "command is required", extra={"invalid_arguments": True})
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace,
                shell=True,
                text=True,
                capture_output=True,
                timeout=self.timeout_sec,
                env=self.env,
            )
            elapsed = round(time.monotonic() - started, 4)
            full_output = _format_command_output(completed.stdout, completed.stderr)
            output, artifact = self._bounded_command_output(full_output, "execute_bash")
            return ToolResult(
                tool_name=call.tool_name,
                returncode=completed.returncode,
                output=output,
                elapsed_sec=elapsed,
                extra={
                    "command": command,
                    "full_output_path": artifact,
                    "full_output_chars": len(full_output),
                    "output_clipped": artifact is not None,
                },
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = round(time.monotonic() - started, 4)
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            full_output = _format_command_output(stdout, stderr) + "\nCommand timed out."
            output, artifact = self._bounded_command_output(full_output, "execute_bash_timeout")
            return ToolResult(
                tool_name=call.tool_name,
                returncode=124,
                output=output,
                elapsed_sec=elapsed,
                extra={
                    "command": command,
                    "timeout": True,
                    "full_output_path": artifact,
                    "full_output_chars": len(full_output),
                    "output_clipped": artifact is not None,
                },
            )

    def _bounded_command_output(self, output: str, stem: str) -> tuple[str, str | None]:
        if len(output) <= self.max_command_output_chars:
            return output, None
        self._output_artifact_index += 1
        artifact_dir = self.workspace / ".codeagent" / "tool_outputs"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact = artifact_dir / f"{self._output_artifact_index:03d}_{stem}.txt"
        artifact.write_text(output, encoding="utf-8")
        return clip_head_tail(output, self.max_command_output_chars), str(artifact)

    def _run_tests(self, call: ToolCall) -> ToolResult:
        command = str(call.arguments.get("command", "")).strip()
        targets = call.arguments.get("targets")
        if not command and targets:
            if isinstance(targets, str):
                try:
                    parsed = json.loads(targets)
                    targets = parsed if isinstance(parsed, list) else [targets]
                except json.JSONDecodeError:
                    targets = [targets]
            if not isinstance(targets, list):
                return ToolResult(call.tool_name, 1, "targets must be a JSON list", extra={"invalid_arguments": True})
            command = "python -m pytest -q --disable-warnings --tb=short " + " ".join(
                shlex.quote(str(target)) for target in targets
            )
        if not command:
            command = str(self.test_command or "python -m pytest -q --disable-warnings --tb=short")

        result = self._execute_bash(ToolCall("execute_bash", {"command": command}, raw=call.raw))
        counts = parse_test_counts(result.output)
        all_skipped = bool(counts.get("skipped")) and not any(counts.get(key) for key in ("passed", "failed", "errors"))
        output = result.output
        if all_skipped:
            output = output.rstrip() + (
                "\n<test-warning>All selected tests were skipped. This is not evidence that the fix works; "
                "check the task runtime or use a targeted reproducer.</test-warning>"
            )
        return ToolResult(
            tool_name=call.tool_name,
            returncode=result.returncode,
            output=output,
            elapsed_sec=result.elapsed_sec,
            extra={
                "command": command,
                "counts": counts,
                "all_skipped": all_skipped,
                "venv_path": self.env.get("VIRTUAL_ENV"),
            },
        )

    def _git_diff(self, call: ToolCall) -> ToolResult:
        started = time.monotonic()
        completed = subprocess.run(
            ["git", "diff", "--", "."],
            cwd=self.workspace,
            text=True,
            capture_output=True,
            timeout=self.timeout_sec,
            env=self.env,
        )
        output = completed.stdout
        if completed.stderr:
            output += f"\n<stderr>\n{completed.stderr}</stderr>"
        if len(output) > self.max_view_chars:
            output = output[: self.max_view_chars] + "\n<response clipped>"
        changed = subprocess.run(
            ["git", "diff", "--name-only", "--", "."],
            cwd=self.workspace,
            text=True,
            capture_output=True,
            timeout=self.timeout_sec,
            env=self.env,
        )
        files = [line for line in changed.stdout.splitlines() if line.strip()]
        return ToolResult(
            tool_name=call.tool_name,
            returncode=completed.returncode,
            output=output or "No changes.",
            elapsed_sec=round(time.monotonic() - started, 4),
            extra={"edited_files": files, "has_changes": bool(files)},
        )

    def _str_replace_editor(self, call: ToolCall) -> ToolResult:
        command = str(call.arguments.get("command", ""))
        try:
            path = self._resolve_path(str(call.arguments.get("path", "")))
        except ValueError as exc:
            return ToolResult(call.tool_name, 1, str(exc), extra={"invalid_arguments": True})
        if command == "view":
            return self._editor_view(path, call)
        if command == "create":
            return self._editor_create(path, str(call.arguments.get("file_text", "")))
        if command == "str_replace":
            return self._editor_str_replace(
                path,
                old_str=str(call.arguments.get("old_str", "")),
                new_str=str(call.arguments.get("new_str", "")),
            )
        if command == "insert":
            return self._editor_insert(
                path,
                insert_line=int(str(call.arguments.get("insert_line", "0"))),
                new_str=str(call.arguments.get("new_str", "")),
            )
        if command == "line_replace":
            return self._editor_line_replace(
                path,
                start_line=int(str(call.arguments.get("start_line", "0"))),
                end_line=int(str(call.arguments.get("end_line", "0"))),
                new_str=str(call.arguments.get("new_str", "")),
            )
        if command == "undo_edit":
            return self._editor_undo(path)
        return ToolResult(call.tool_name, 1, f"Unsupported editor command: {command}")

    def _editor_view(self, path: Path, call: ToolCall) -> ToolResult:
        if path.is_dir():
            entries = []
            for child in sorted(path.rglob("*")):
                if child == path or child.name.startswith("."):
                    continue
                rel = child.relative_to(path)
                if len(rel.parts) > 2:
                    continue
                suffix = "/" if child.is_dir() else ""
                entries.append(f"{rel.as_posix()}{suffix}")
            return ToolResult("str_replace_editor", 0, "\n".join(entries)[: self.max_view_chars])
        if not path.exists():
            return ToolResult("str_replace_editor", 1, f"Path does not exist: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        view_range_raw = call.arguments.get("view_range")
        if view_range_raw:
            try:
                view_range = json.loads(str(view_range_raw))
                start = max(1, int(view_range[0]))
                end = len(lines) if int(view_range[1]) == -1 else min(len(lines), int(view_range[1]))
                lines = lines[start - 1 : end]
                base_line = start
            except Exception:
                base_line = 1
        else:
            base_line = 1
        rendered = "\n".join(f"{idx:6}\t{line}" for idx, line in enumerate(lines, start=base_line))
        if len(rendered) > self.max_view_chars:
            rendered = rendered[: self.max_view_chars] + "\n<response clipped>"
        return ToolResult("str_replace_editor", 0, rendered)

    def _editor_create(self, path: Path, file_text: str) -> ToolResult:
        if path.exists():
            return ToolResult("str_replace_editor", 1, f"Cannot create; path already exists: {path}")
        self._save_undo(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file_text, encoding="utf-8")
        return ToolResult("str_replace_editor", 0, f"Created file: {path}")

    def _editor_str_replace(self, path: Path, *, old_str: str, new_str: str) -> ToolResult:
        if not path.is_file():
            return ToolResult("str_replace_editor", 1, f"Path is not a file: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old_str)
        escaped_newline_fallback = False
        if count == 0 and "\\n" in old_str and "\n" not in old_str:
            decoded_old_str = old_str.replace("\\r\\n", "\n").replace("\\n", "\n")
            decoded_count = text.count(decoded_old_str)
            if decoded_count:
                old_str = decoded_old_str
                if "\n" not in new_str:
                    new_str = new_str.replace("\\r\\n", "\n").replace("\\n", "\n")
                count = decoded_count
                escaped_newline_fallback = True
        if count == 0:
            return ToolResult("str_replace_editor", 1, "old_str was not found.")
        if count > 1:
            return ToolResult(
                "str_replace_editor",
                1,
                "old_str is not unique. Use command=line_replace with start_line/end_line, or include more surrounding context in old_str.",
            )
        self._save_undo(path)
        path.write_text(text.replace(old_str, new_str), encoding="utf-8")
        return ToolResult(
            "str_replace_editor",
            0,
            f"Replaced text in: {path}",
            extra={"escaped_newline_fallback": escaped_newline_fallback},
        )

    def _editor_insert(self, path: Path, *, insert_line: int, new_str: str) -> ToolResult:
        if not path.is_file():
            return ToolResult("str_replace_editor", 1, f"Path is not a file: {path}")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if insert_line < 0 or insert_line > len(lines):
            return ToolResult("str_replace_editor", 1, f"insert_line out of range: {insert_line}")
        self._save_undo(path)
        insert_lines = new_str.splitlines()
        updated = lines[:insert_line] + insert_lines + lines[insert_line:]
        path.write_text("\n".join(updated) + "\n", encoding="utf-8")
        return ToolResult("str_replace_editor", 0, f"Inserted text in: {path}")

    def _editor_line_replace(self, path: Path, *, start_line: int, end_line: int, new_str: str) -> ToolResult:
        if not path.is_file():
            return ToolResult("str_replace_editor", 1, f"Path is not a file: {path}")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if start_line < 1 or end_line < start_line or end_line > len(lines):
            return ToolResult(
                "str_replace_editor",
                1,
                f"line range out of bounds: start_line={start_line} end_line={end_line} total_lines={len(lines)}",
            )
        self._save_undo(path)
        replacement = new_str.splitlines()
        updated = lines[: start_line - 1] + replacement + lines[end_line:]
        path.write_text("\n".join(updated) + "\n", encoding="utf-8")
        return ToolResult("str_replace_editor", 0, f"Replaced lines {start_line}-{end_line} in: {path}")

    def _editor_undo(self, path: Path) -> ToolResult:
        stack = self._undo_stack.get(path)
        if not stack:
            return ToolResult("str_replace_editor", 1, f"No edit to undo for: {path}")
        previous = stack.pop()
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(previous, encoding="utf-8")
        return ToolResult("str_replace_editor", 0, f"Undid last edit for: {path}")

    def _save_undo(self, path: Path) -> None:
        previous = path.read_text(encoding="utf-8", errors="replace") if path.exists() else None
        self._undo_stack.setdefault(path, []).append(previous)

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path:
            raise ValueError("path is required")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace: {raw_path}") from exc
        return resolved


def openhands_tool_spec(include_extensions: bool = True) -> str:
    tools = [
        "execute_bash(command)",
        "str_replace_editor(command, path, file_text?, old_str?, new_str?, insert_line?, start_line?, end_line?, view_range?)",
        "run_tests(command?, targets?)",
        "git_diff()",
        "finish()",
    ]
    if include_extensions:
        tools.extend(
            [
                "web_search(query, max_results?)",
                "download_repo(repo_url, dest_name?)",
                "repo_context(query?)",
                "memory_search(query?)",
                "problem_search(query, top_k?, index_path?)",
            ]
        )
    return "\n".join(f"- {tool}" for tool in tools)


def _format_command_output(stdout: str, stderr: str) -> str:
    parts = []
    if stdout:
        parts.append(f"<stdout>\n{stdout}</stdout>")
    if stderr:
        parts.append(f"<stderr>\n{stderr}</stderr>")
    if not parts:
        return ""
    return "\n".join(parts)


def clip_head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker_template = "\n<response clipped; {omitted} chars omitted; full output saved as artifact>\n"
    marker = marker_template.format(omitted=0)
    available = max(2, max_chars - len(marker))
    head_chars = max(1, available * 2 // 3)
    tail_chars = max(1, available - head_chars)
    omitted = len(text) - head_chars - tail_chars
    marker = marker_template.format(omitted=omitted)
    available = max(2, max_chars - len(marker))
    head_chars = max(1, available * 2 // 3)
    tail_chars = max(1, available - head_chars)
    return text[:head_chars] + marker + text[-tail_chars:]


def parse_test_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    patterns = {
        "passed": r"(\d+)\s+passed",
        "failed": r"(\d+)\s+failed",
        "errors": r"(\d+)\s+errors?",
        "skipped": r"(\d+)\s+skipped",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        if matches:
            counts[key] = int(matches[-1])
    return counts


def sanitize_workspace_cd(command: str) -> str:
    stripped = command.strip()
    bad_roots = r"(?:testbed|workspace|repo|home/user)"
    shell_join = re.match(rf"^cd\s+/{bad_roots}(?:\s+2>/dev/null)?\s*(?:&&|;)\s*(.+)$", stripped, re.DOTALL)
    if shell_join:
        return shell_join.group(1).strip()
    fallback_join = re.match(rf"^cd\s+/{bad_roots}(?:\s+2>/dev/null)?\s*\|\|\s*pwd\s*&&\s*(.+)$", stripped, re.DOTALL)
    if fallback_join:
        return fallback_join.group(1).strip()
    return command


def coerce_common_tool_alias(call: ToolCall) -> ToolCall:
    if call.tool_name != "grep":
        return call
    pattern = call.arguments.get("pattern") or call.arguments.get("query") or call.arguments.get("string") or ""
    path = call.arguments.get("path") or "."
    if not str(pattern).strip():
        return call
    command = f"grep -rn {shlex.quote(str(pattern))} {shlex.quote(str(path))} --include='*.py' | head -50"
    return ToolCall(tool_name="execute_bash", arguments={"command": command}, raw=call.raw)
