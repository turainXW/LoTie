import json
from pathlib import Path
from unittest import mock
import subprocess
import sys
import tempfile
import unittest

from code_agent_baseline.openhands_tools import OpenHandsToolParser, OpenHandsToolRuntime
from code_agent_baseline.runtime_env import task_runtime_env
from code_agent_baseline.web_tools import _extract_bocha_results, web_search


class OpenHandsToolsTest(unittest.TestCase):
    def test_parse_execute_bash(self) -> None:
        text = (
            "Let's inspect.\n"
            "<function=execute_bash>\n"
            "<parameter=command>python3 -V</parameter>\n"
            "</function>"
        )
        call = OpenHandsToolParser.parse_first(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "execute_bash")
        self.assertEqual(call.arguments["command"], "python3 -V")

    def test_parse_direct_tool_tag_with_prefix_text(self) -> None:
        text = (
            "I'll inspect first."
            "<execute_bash>\n"
            "<command>pwd && ls</command>\n"
            "</execute_bash>"
        )
        call = OpenHandsToolParser.parse_first(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "execute_bash")
        self.assertEqual(call.arguments["command"], "pwd && ls")

    def test_parse_invoke_tool_tag(self) -> None:
        text = (
            "<tool_call>\n"
            "<invoke name=\"execute_bash\">\n"
            "<command>pwd</command>\n"
            "</invoke>\n"
            "</tool_call>"
        )
        call = OpenHandsToolParser.parse_first(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "execute_bash")
        self.assertEqual(call.arguments["command"], "pwd")

    def test_parse_qwen_named_tool_call_wrapper(self) -> None:
        text = '<tool_call>execute_bash{"arguments": {"command":"pwd"}}\n</tool_call>'

        call = OpenHandsToolParser.parse_first(text)

        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "execute_bash")
        self.assertEqual(call.arguments, {"command": "pwd"})

    def test_parse_deepseek_dsml_tool_call(self) -> None:
        text = (
            '<｜｜DSML｜｜tool_calls>\n'
            '<｜｜DSML｜｜invoke name="str_replace_editor">\n'
            '<｜｜DSML｜｜parameter name="command" string="true">view</｜｜DSML｜｜parameter>\n'
            '<｜｜DSML｜｜parameter name="path" string="true">bottle.py</｜｜DSML｜｜parameter>\n'
            '<｜｜DSML｜｜parameter name="view_range" string="false">2300,2365</｜｜DSML｜｜parameter>\n'
            '</｜｜DSML｜｜invoke>\n'
            '</｜｜DSML｜｜tool_calls>'
        )

        call = OpenHandsToolParser.parse_first(text)

        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "str_replace_editor")
        self.assertEqual(call.arguments["command"], "view")
        self.assertEqual(call.arguments["path"], "bottle.py")
        self.assertEqual(call.arguments["view_range"], [2300, 2365])

    def test_parse_tool_invoke_with_nested_tool_name(self) -> None:
        text = (
            "<tool_invoke>\n"
            "<tool_name>execute_bash</tool_name>\n"
            '<command>grep -n "DATE_PART" sqlglot/dialects/dialect.py</command>\n'
            "</invoke>"
        )

        call = OpenHandsToolParser.parse_first(text)

        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "execute_bash")
        self.assertEqual(call.arguments["command"], 'grep -n "DATE_PART" sqlglot/dialects/dialect.py')

    def test_parse_tool_invoke_with_mixed_dsml_closing_tag(self) -> None:
        text = (
            "<tool_calls>\n<tool_invoke>\n"
            "<tool_name>execute_bash</｜｜DSML｜｜>\n"
            '<command>grep -rn "EPOCH" sqlglot tests</command>\n'
            "</invoke>\n</tool_calls>"
        )

        call = OpenHandsToolParser.parse_first(text)

        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "execute_bash")
        self.assertEqual(call.arguments["command"], 'grep -rn "EPOCH" sqlglot tests')

    def test_parse_dsml_wrapper_with_nested_tool_name(self) -> None:
        text = (
            "<｜｜DSML｜｜>\n"
            "<tool_name>execute_bash</｜｜DSML｜｜>\n"
            '<command>grep -rn "DATE_PART" tests</command>\n'
            "</｜｜DSML｜｜>"
        )

        call = OpenHandsToolParser.parse_first(text)

        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "execute_bash")
        self.assertEqual(call.arguments["command"], 'grep -rn "DATE_PART" tests')

    def test_parse_malformed_parameter_tag(self) -> None:
        text = "<function=execute_bash><parameter>command>pwd</parameter></function>"
        call = OpenHandsToolParser.parse_first(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "execute_bash")
        self.assertEqual(call.arguments["command"], "pwd")

    def test_parse_wrong_closing_tag(self) -> None:
        text = (
            "<function=execute_bash>"
            "<parameter=command>grep -n size file.py</parameter>"
            "</figure>"
        )
        call = OpenHandsToolParser.parse_first(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "execute_bash")
        self.assertEqual(call.arguments["command"], "grep -n size file.py")

    def test_parse_fenced_bash_as_execute_bash(self) -> None:
        text = "```bash\nsed -n '1,5p' demo.py\n```</observation>"
        call = OpenHandsToolParser.parse_first(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "execute_bash")
        self.assertEqual(call.arguments["command"], "sed -n '1,5p' demo.py")

    def test_parse_json_tool_call(self) -> None:
        text = '{"tool_name": "execute_bash", "arguments": {"command": "pwd"}}'
        call = OpenHandsToolParser.parse_first(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "execute_bash")
        self.assertEqual(call.arguments["command"], "pwd")

    def test_parse_json_tool_call_repairs_missing_final_brace(self) -> None:
        text = '{"tool_name": "answer", "arguments": {"content": "done"}'
        call = OpenHandsToolParser.parse_first(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "answer")
        self.assertEqual(call.arguments["content"], "done")
        self.assertEqual(call.raw, '{"tool_name": "answer", "arguments": {"content": "done"}}')

    def test_parse_json_tool_call_repairs_missing_arguments_key_quote(self) -> None:
        call = OpenHandsToolParser.parse_first('{"tool_name":"finish","arguments:{}}')

        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "finish")
        self.assertEqual(call.arguments, {})

    def test_parse_first_json_tool_call_inside_prose(self) -> None:
        text = (
            "Let me inspect first.\n\n"
            '{"tool_name": "str_replace_editor", "arguments": {"command": "view", "path": "solution.py"}}\n\n'
            '{"tool_name": "str_replace_editor", "arguments": {"command": "view", "path": "tests/test_solution.py"}}'
        )
        call = OpenHandsToolParser.parse_first(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "str_replace_editor")
        self.assertEqual(call.arguments["path"], "solution.py")

    def test_parse_partial_function_tag(self) -> None:
        text = "<function=execute_bash><parameter=command>grep -n foo src/app.py</parameter>"
        call = OpenHandsToolParser.parse_first(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "execute_bash")
        self.assertEqual(call.arguments["command"], "grep -n foo src/app.py")

    def test_parse_plain_shell_line(self) -> None:
        call = OpenHandsToolParser.parse_first("PYTHONPATH=src python -m pytest tests/test_app.py")
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "execute_bash")
        self.assertEqual(call.arguments["command"], "PYTHONPATH=src python -m pytest tests/test_app.py")

    def test_parse_finish_intent(self) -> None:
        call = OpenHandsToolParser.parse_first("The fix is implemented and validation is complete.")
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.tool_name, "finish")

    def test_editor_view_replace_insert_undo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            path = workspace / "demo.py"
            path.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            runtime = OpenHandsToolRuntime(workspace)

            view = runtime.execute(
                OpenHandsToolParser.parse_first(
                    "<function=str_replace_editor><parameter=command>view</parameter>"
                    "<parameter=path>demo.py</parameter></function>"
                )
            )
            self.assertEqual(view.returncode, 0)
            self.assertIn("return a - b", view.output)

            replace = runtime.execute(
                OpenHandsToolParser.parse_first(
                    "<function=str_replace_editor><parameter=command>str_replace</parameter>"
                    "<parameter=path>demo.py</parameter>"
                    "<parameter=old_str>return a - b</parameter>"
                    "<parameter=new_str>return a + b</parameter></function>"
                )
            )
            self.assertEqual(replace.returncode, 0)
            self.assertIn("return a + b", path.read_text(encoding="utf-8"))

            insert = runtime.execute(
                OpenHandsToolParser.parse_first(
                    "<function=str_replace_editor><parameter=command>insert</parameter>"
                    "<parameter=path>demo.py</parameter>"
                    "<parameter=insert_line>0</parameter>"
                    "<parameter=new_str># generated</parameter></function>"
                )
            )
            self.assertEqual(insert.returncode, 0)
            self.assertTrue(path.read_text(encoding="utf-8").startswith("# generated"))

            undo = runtime.execute(
                OpenHandsToolParser.parse_first(
                    "<function=str_replace_editor><parameter=command>undo_edit</parameter>"
                    "<parameter=path>demo.py</parameter></function>"
                )
            )
            self.assertEqual(undo.returncode, 0)
            self.assertFalse(path.read_text(encoding="utf-8").startswith("# generated"))

    def test_editor_line_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            path = workspace / "demo.py"
            path.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
            runtime = OpenHandsToolRuntime(workspace)
            call = OpenHandsToolParser.parse_first(
                "<function=str_replace_editor>"
                "<parameter=command>line_replace</parameter>"
                "<parameter=path>demo.py</parameter>"
                "<parameter=start_line>2</parameter>"
                "<parameter=end_line>2</parameter>"
                "<parameter=new_str>b = 20</parameter>"
                "</function>"
            )
            result = runtime.execute(call)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(path.read_text(encoding="utf-8"), "a = 1\nb = 20\nc = 3\n")

    def test_editor_str_replace_recovers_double_escaped_newlines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            path = workspace / "solution.py"
            path.write_text("def answer():\n    raise NotImplementedError()\n", encoding="utf-8")
            runtime = OpenHandsToolRuntime(workspace)
            payload = {
                "tool_name": "str_replace_editor",
                "arguments": {
                    "command": "str_replace",
                    "path": "solution.py",
                    "old_str": r"def answer():\n    raise NotImplementedError()",
                    "new_str": r"def answer():\n    return 42",
                },
            }

            result = runtime.execute(OpenHandsToolParser.parse_first(json.dumps(payload)))

            self.assertEqual(result.returncode, 0)
            self.assertTrue(result.extra["escaped_newline_fallback"])
            self.assertEqual(path.read_text(encoding="utf-8"), "def answer():\n    return 42\n")

    def test_bash_executes_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = OpenHandsToolRuntime(tmp)
            call = OpenHandsToolParser.parse_first(
                "<function=execute_bash><parameter=command>pwd</parameter></function>"
            )
            result = runtime.execute(call)
            self.assertEqual(result.returncode, 0)
            self.assertIn(tmp, result.output)

    def test_bash_strips_common_wrong_workspace_cd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = OpenHandsToolRuntime(tmp)
            call = OpenHandsToolParser.parse_first(
                "<function=execute_bash>"
                "<parameter=command>cd /testbed 2>/dev/null || pwd && pwd</parameter>"
                "</function>"
            )
            result = runtime.execute(call)
            self.assertEqual(result.returncode, 0)
            self.assertIn(tmp, result.output)
            self.assertNotIn("testbed", result.output)

    def test_bash_missing_command_returns_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = OpenHandsToolRuntime(tmp)
            call = OpenHandsToolParser.parse_first("<function=execute_bash></function>")
            result = runtime.execute(call)
            self.assertEqual(result.returncode, 1)
            self.assertIn("command is required", result.output)

    def test_bash_clips_large_output_and_preserves_full_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = OpenHandsToolRuntime(tmp, max_command_output_chars=400)
            call = OpenHandsToolParser.parse_first(
                '{"tool_name":"execute_bash","arguments":{"command":"python -c \'print(\\\"A\\\" * 2000)\'"}}'
            )
            result = runtime.execute(call)

            self.assertEqual(result.returncode, 0)
            self.assertLessEqual(len(result.output), 400)
            self.assertIn("response clipped", result.output)
            self.assertTrue(result.extra["output_clipped"])
            artifact = Path(result.extra["full_output_path"])
            self.assertTrue(artifact.is_file())
            self.assertGreater(len(artifact.read_text(encoding="utf-8")), 2_000)

    def test_run_tests_uses_bound_environment_and_returns_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "test_demo.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
            env = task_runtime_env(workspace, venv_path=Path(sys.executable).parent.parent)
            runtime = OpenHandsToolRuntime(workspace, runtime_env=env, allowed_tools={"run_tests"})
            result = runtime.execute(
                OpenHandsToolParser.parse_first(
                    '{"tool_name":"run_tests","arguments":{"targets":["test_demo.py::test_ok"]}}'
                )
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.extra["counts"]["passed"], 1)
            self.assertFalse(result.extra["all_skipped"])
            self.assertEqual(result.extra["venv_path"], str(Path(sys.executable).parent.parent.resolve()))

    def test_run_tests_warns_when_every_test_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "test_demo.py").write_text(
                "import pytest\n@pytest.mark.skip(reason='wrong runtime')\ndef test_skip():\n    pass\n",
                encoding="utf-8",
            )
            env = task_runtime_env(workspace, venv_path=Path(sys.executable).parent.parent)
            runtime = OpenHandsToolRuntime(workspace, runtime_env=env, allowed_tools={"run_tests"})
            result = runtime.execute(
                OpenHandsToolParser.parse_first(
                    '{"tool_name":"run_tests","arguments":{"targets":["test_demo.py::test_skip"]}}'
                )
            )
            self.assertEqual(result.returncode, 0)
            self.assertTrue(result.extra["all_skipped"])
            self.assertIn("not evidence that the fix works", result.output)

    def test_git_diff_returns_patch_and_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            path = workspace / "demo.py"
            path.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
            subprocess.run(["git", "add", "demo.py"], cwd=workspace, check=True, capture_output=True)
            path.write_text("VALUE = 2\n", encoding="utf-8")
            runtime = OpenHandsToolRuntime(workspace, allowed_tools={"git_diff"})
            result = runtime.execute(OpenHandsToolParser.parse_first('{"tool_name":"git_diff","arguments":{}}'))
            self.assertEqual(result.returncode, 0)
            self.assertIn("+VALUE = 2", result.output)
            self.assertEqual(result.extra["edited_files"], ["demo.py"])

    def test_editor_missing_path_returns_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = OpenHandsToolRuntime(tmp)
            call = OpenHandsToolParser.parse_first(
                "<function=str_replace_editor><parameter=command>view</parameter></function>"
            )
            result = runtime.execute(call)
            self.assertEqual(result.returncode, 1)
            self.assertIn("path is required", result.output)

    def test_repo_context_tool_returns_ranked_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pkg").mkdir()
            (workspace / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (workspace / "pkg" / "slug.py").write_text(
                "import re\n\n"
                "def normalize_slug(value):\n"
                "    return re.sub(r'[^a-z0-9]+', '-', value.lower())\n",
                encoding="utf-8",
            )
            (workspace / "tests").mkdir()
            (workspace / "tests" / "test_slug.py").write_text(
                "from pkg.slug import normalize_slug\n\n"
                "def test_slug():\n"
                "    assert normalize_slug('Hello World') == 'hello-world'\n",
                encoding="utf-8",
            )
            runtime = OpenHandsToolRuntime(workspace)
            call = OpenHandsToolParser.parse_first(
                "<function=repo_context>"
                "<parameter=query>normalize slug punctuation</parameter>"
                "<parameter=max_files>4</parameter>"
                "<parameter=max_snippets>2</parameter>"
                "</function>"
            )
            result = runtime.execute(call)
            self.assertEqual(result.returncode, 0)
            self.assertIn("pkg/slug.py", result.output)
            self.assertIn("normalize_slug", result.output)
            self.assertTrue((workspace / ".codeagent" / "repo_map.json").exists())

    def test_common_grep_alias_maps_to_bash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "app.py").write_text("needle = 1\n", encoding="utf-8")
            runtime = OpenHandsToolRuntime(workspace)
            result = runtime.execute(
                OpenHandsToolParser.parse_first(
                    '{"tool_name": "grep", "arguments": {"string": "needle", "path": "."}}'
                )
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.tool_name, "execute_bash")
            self.assertIn("app.py", result.output)

    def test_web_search_tool_uses_json_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = OpenHandsToolRuntime(tmp, allowed_tools={"web_search"})
            with mock.patch(
                "code_agent_baseline.web_tools.web_search",
                return_value={
                    "provider": "mock",
                    "ok": True,
                    "query": "code agent benchmark",
                    "results": [{"title": "demo", "url": "https://example.com"}],
                },
            ):
                result = runtime.execute(
                    OpenHandsToolParser.parse_first(
                        '{"tool_name": "web_search", "arguments": {"query": "code agent benchmark", "max_results": 2}}'
                    )
                )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.tool_name, "web_search")
            self.assertIn('"provider": "mock"', result.output)
            self.assertEqual(result.extra["provider"], "mock")

    def test_bocha_result_extraction(self) -> None:
        payload = {
            "code": 200,
            "data": {
                "webPages": {
                    "value": [
                        {
                            "name": "Demo title",
                            "url": "https://example.com/demo",
                            "summary": "Demo summary",
                            "siteName": "Example",
                        }
                    ]
                }
            },
        }
        results = _extract_bocha_results(payload, 3)
        self.assertEqual(results[0]["title"], "Demo title")
        self.assertEqual(results[0]["url"], "https://example.com/demo")
        self.assertEqual(results[0]["description"], "Demo summary")

    def test_bocha_provider_has_first_priority(self) -> None:
        with mock.patch.dict("os.environ", {"BOCHA_API_KEY": "test", "BRAVE_SEARCH_API_KEY": "test"}, clear=True):
            with mock.patch(
                "code_agent_baseline.web_tools._bocha_search",
                return_value={"provider": "bocha", "ok": True, "query": "demo", "results": []},
            ) as bocha:
                result = web_search("demo")
        self.assertEqual(result["provider"], "bocha")
        bocha.assert_called_once()


if __name__ == "__main__":
    unittest.main()
