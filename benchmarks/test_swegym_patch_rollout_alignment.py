import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
import unittest.mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_swegym_patch_rollout", ROOT / "scripts" / "run_swegym_patch_rollout.py")
assert SPEC is not None and SPEC.loader is not None
rollout = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rollout
SPEC.loader.exec_module(rollout)


class SwegymPatchRolloutAlignmentTest(unittest.TestCase):
    def test_context_budget_reserves_requested_generation_tokens(self) -> None:
        self.assertEqual(rollout.context_input_budget(64_000, 1_024), 62_976)
        self.assertEqual(rollout.context_input_budget(64_000, 0), 64_000)
        self.assertIsNone(rollout.context_input_budget(None, 1_024))

    def test_vllm_context_limit_error_is_not_an_infrastructure_failure(self) -> None:
        error = RuntimeError(
            "This model's maximum context length is 64000 tokens. However, you requested "
            "1024 output tokens and your prompt contains at least 62977 input tokens."
        )
        self.assertTrue(rollout.is_model_context_overflow(error))
        self.assertFalse(rollout.is_model_context_overflow(RuntimeError("HTTP 500 from model API")))

    def test_harness_metadata_identifies_current_reminder_policy(self) -> None:
        metadata = rollout.harness_metadata(checkpoint_step=10, reminder_interval=3)

        self.assertEqual(metadata["version"], "lottie_code_agent_harness_v3")
        policy = metadata["readonly_budget_policy"]
        self.assertEqual(policy["version"], "append_nonmod3_action_reset_v1")
        self.assertEqual(policy["checkpoint_step"], 10)
        self.assertEqual(policy["reminder_interval"], 3)
        self.assertTrue(policy["reset_on_modification_action"])
        self.assertFalse(policy["blocks_actions"])
        self.assertTrue(policy["preserves_tool_result"])

    def test_bench_prompt_can_use_stable_workspace_label(self) -> None:
        prompt = rollout.openhands_user_prompt(
            task={"instance_id": "demo", "repo": "a/b", "base_commit": "abc", "problem_statement": "Fix it."},
            workspace=Path("/tmp/random-sample-path"),
            parsed={},
            file_context=[],
            tool_profile="official-core",
            runtime_info={"ready": True, "python_version": "Python 3.10"},
            workspace_label="repository",
        )

        self.assertIn("<uploaded_files>\nrepository\n</uploaded_files>", prompt)
        self.assertIn("directory repository", prompt)
        self.assertNotIn("random-sample-path", prompt)

    def test_instance_ids_are_safe_for_artifact_paths(self) -> None:
        self.assertEqual(rollout.safe_instance_name("HumanEval/32"), "HumanEval_32")

    def test_model_retry_replays_identical_messages(self) -> None:
        args = rollout.parse_args
        namespace = type(
            "Args",
            (),
            {
                "api_key_env": "",
                "max_tokens": 100,
                "temperature": 1.0,
                "top_p": 0.95,
                "thinking_mode": "enabled",
                "reasoning_effort": "max",
                "api_retries": 2,
                "retry_backoff_sec": 0.0,
                "retry_max_backoff_sec": 0.0,
                "model_network": "direct",
                "_api_retry_events": [],
                "_model_response_events": [],
            },
        )()
        messages = [{"role": "user", "content": "keep this exact context"}]

        with unittest.mock.patch.object(rollout, "request_chat_completion", return_value="ok") as request:
            self.assertEqual(
                rollout.call_model("https://example.test", "demo", messages, timeout_sec=5, args=namespace),
                "ok",
            )

        messages_for_attempt = request.call_args.kwargs["messages_for_attempt"]
        self.assertEqual(messages_for_attempt(0), messages_for_attempt(2))
        self.assertFalse(request.call_args.kwargs["use_system_proxy"])

    def test_protocol_repair_can_override_sampling_temperature(self) -> None:
        namespace = type(
            "Args",
            (),
            {
                "api_key_env": "",
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.95,
                "thinking_mode": "disabled",
                "reasoning_effort": "low",
                "api_retries": 2,
                "retry_backoff_sec": 0.0,
                "retry_max_backoff_sec": 0.0,
                "model_network": "direct",
                "_api_retry_events": [],
                "_model_response_events": [],
            },
        )()

        with unittest.mock.patch.object(rollout, "request_chat_completion", return_value="ok") as request:
            rollout.call_model(
                "https://example.test",
                "demo",
                [{"role": "user", "content": "repair JSON"}],
                timeout_sec=5,
                args=namespace,
                temperature=0.0,
            )

        self.assertEqual(request.call_args.kwargs["temperature"], 0.0)

    def test_system_prompt_contains_strict_tool_protocol(self) -> None:
        prompt = rollout.openhands_system_prompt("extended")
        self.assertIn("Every assistant message MUST contain exactly one JSON object", prompt)
        self.assertIn('"tool_name": "FUNCTION_NAME"', prompt)
        self.assertIn('"arguments": {"PARAMETER_NAME": "VALUE"}', prompt)
        self.assertIn("line_replace", prompt)
        self.assertIn("Do not assume /testbed", prompt)
        self.assertIn("Do not start commands with cd /testbed", prompt)
        self.assertIn("minimal source/runtime code change", prompt)
        self.assertIn("Do not modify tests, examples, docs, or benchmarks", prompt)
        self.assertIn("already selected and activated", prompt)
        self.assertIn("run_tests", prompt)
        self.assertIn("git_diff", prompt)
        self.assertIn("A successful task is not complete until you emit", prompt)
        self.assertIn("your next and only action MUST be finish", prompt)
        self.assertIn("Do not send a prose final answer", prompt)
        self.assertEqual(prompt.count("<TOOLS>"), 1)
        self.assertEqual(prompt.count("<TOOL_FEW_SHOT>"), 1)
        self.assertIn("Use when:", prompt)
        self.assertIn("Avoid when:", prompt)
        self.assertIn("Returns:", prompt)
        self.assertNotIn("XML function call", prompt)
        self.assertNotIn("<parameter=PARAMETER_NAME>", prompt)

    def test_user_prompt_has_one_unambiguous_json_terminal_action(self) -> None:
        prompt = rollout.openhands_user_prompt(
            task={
                "instance_id": "demo",
                "repo": "demo/repo",
                "base_commit": "abc",
                "problem_statement": "Fix the bug.",
                "hints_text": "",
            },
            workspace=Path("/tmp/demo"),
            parsed={},
            file_context=[],
            tool_profile="official-core",
            runtime_info={"ready": True},
        )

        self.assertIn('{"tool_name":"finish","arguments":{}}', prompt)
        self.assertNotIn("send a final answer and then call finish", prompt)

    def test_tool_error_protocol_reminders(self) -> None:
        self.assertIn("execute_bash", rollout.protocol_reminder_for_tool_error("command is required"))
        path_reminder = rollout.protocol_reminder_for_tool_error("path is required")
        self.assertIn('"tool_name": "str_replace_editor"', path_reminder)
        self.assertIn('"path": "repo/relative/path.py"', path_reminder)
        self.assertIn("line_replace", rollout.protocol_reminder_for_tool_error("old_str is not unique"))
        runtime_reminder = rollout.protocol_reminder_for_tool_error("/bin/sh: cd: /testbed: No such file or directory")
        self.assertIn("already run in the repo workspace", runtime_reminder)

    def test_official_core_messages_match_openhands_sft_surface(self) -> None:
        messages = rollout.build_openhands_training_messages(
            task={
                "instance_id": "demo__repo-1",
                "repo": "demo/repo",
                "base_commit": "abc123",
                "problem_statement": "Fix parser bug.",
                "hints_text": "",
            },
            planner={"files_to_inspect": ["pkg/parser.py"]},
            file_context=[
                {
                    "path": "pkg/parser.py",
                    "numbered_content": "0001: def parse(value):\n0002:     return value",
                    "truncated": False,
                }
            ],
            repair_steps=[
                {
                    "repair": {
                        "edits": [{"path": "pkg/parser.py", "old_str": "return value", "new_str": "return value.strip()"}],
                        "tests_to_run": ["pytest tests/test_parser.py"],
                    },
                    "apply_result": {"edited_files": ["pkg/parser.py"]},
                    "verify": {"output": "1 passed", "returncode": 0},
                }
            ],
            final_verify={},
            tool_profile="official-core",
        )

        self.assertEqual([item["role"] for item in messages[:4]], ["system", "user", "assistant", "user"])
        self.assertFalse(any("OBSERVATION:" in item["content"] for item in messages))
        self.assertFalse(any("<function=repo_context>" in item["content"] for item in messages))
        self.assertFalse(any("problem_search" in item["content"] for item in messages))

        assistant_tools = assistant_tool_names(messages)
        self.assertEqual(set(assistant_tools), {"str_replace_editor", "execute_bash", "finish"})
        self.assertFalse(any("<function=" in item["content"] for item in messages if item["role"] == "assistant"))
        self.assertEqual(
            rollout.openhands_allowed_tools("official-core"),
            {"str_replace_editor", "execute_bash", "run_tests", "git_diff", "finish"},
        )

    def test_use_mode_messages_can_use_richer_context_tools(self) -> None:
        messages = rollout.build_openhands_training_messages(
            task={"instance_id": "demo", "repo": "demo/repo", "base_commit": "abc", "problem_statement": "Fix bug."},
            planner={"files_to_inspect": ["pkg/mod.py"]},
            file_context=[],
            repair_steps=[],
            final_verify={},
            tool_profile="extended",
        )

        self.assertIn("repo_context", messages[0]["content"])
        self.assertIn("problem_search", messages[0]["content"])
        self.assertIn("web_search", messages[0]["content"])
        self.assertIn("download_repo", messages[0]["content"])
        self.assertIn("repo_context", assistant_tool_names(messages))
        self.assertEqual(
            rollout.openhands_allowed_tools("extended"),
            {
                "str_replace_editor",
                "execute_bash",
                "run_tests",
                "git_diff",
                "finish",
                "repo_context",
                "problem_search",
                "web_search",
                "download_repo",
            },
        )

    def test_saved_tool_schemas_include_descriptions_and_parameters(self) -> None:
        schemas = rollout.tool_schema("official-core")
        self.assertEqual(
            [item["function"]["name"] for item in schemas],
            ["execute_bash", "str_replace_editor", "run_tests", "git_diff", "finish"],
        )
        for item in schemas:
            self.assertTrue(item["function"]["description"])
            parameters = item["function"]["parameters"]
            self.assertEqual(parameters["type"], "object")
            self.assertIn("properties", parameters)
            self.assertIn("required", parameters)

    def test_task_environment_uses_configured_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            task = {"local_venv_path": str(Path(sys.executable).parent.parent), "test_command": "python -m pytest -q"}
            env, info = rollout.prepare_task_environment(task, workspace)

            self.assertTrue(info["ready"])
            self.assertEqual(info["environment_source"], "task")
            self.assertEqual(env["VIRTUAL_ENV"], str(Path(sys.executable).parent.parent.resolve()))
            self.assertIn(str(workspace), env["PYTHONPATH"])

    def test_missing_task_environment_is_infrastructure_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, info = rollout.prepare_task_environment(
                {"local_venv_path": str(Path(tmp) / "missing")},
                Path(tmp),
            )
            self.assertFalse(info["ready"])
            self.assertEqual(info["status"], "infrastructure_blocked")
            self.assertEqual(info["reason"], "task_venv_missing")

    def test_public_runtime_info_does_not_inject_full_test_command(self) -> None:
        info = rollout.public_runtime_info(
            {
                "ready": True,
                "python_version": "Python 3.10.20",
                "venv_path": "/tmp/venv",
                "test_command": "python -m pytest " + ("test.py " * 100),
            }
        )
        self.assertNotIn("test_command", info)
        self.assertEqual(info["python_version"], "Python 3.10.20")

    def test_readonly_budget_reminds_every_three_non_modification_actions(self) -> None:
        bash = rollout.OpenHandsToolParser.parse_first(
            '{"tool_name":"execute_bash","arguments":{"command":"pwd"}}'
        )
        view = rollout.OpenHandsToolParser.parse_first(
            '{"tool_name":"str_replace_editor","arguments":{"command":"view","path":"mod.py"}}'
        )
        edit = rollout.OpenHandsToolParser.parse_first(
            '{"tool_name":"str_replace_editor","arguments":{"command":"str_replace","path":"mod.py","old_str":"1","new_str":"2"}}'
        )
        self.assertIsNone(rollout.readonly_budget_policy(11, bash, [], checkpoint_step=10, grace_steps=3))
        steps = [
            {"step": 11, "tool_call": bash.to_dict()},
            {"step": 12, "tool_call": view.to_dict()},
        ]
        self.assertEqual(
            rollout.readonly_budget_policy(13, bash, steps, checkpoint_step=10, grace_steps=3),
            "append",
        )
        steps.append({"step": 13, "tool_call": bash.to_dict()})
        self.assertIsNone(rollout.readonly_budget_policy(14, bash, steps, checkpoint_step=10, grace_steps=3))

        steps.append({"step": 14, "tool_call": edit.to_dict()})
        self.assertIsNone(rollout.readonly_budget_policy(15, bash, steps, checkpoint_step=10, grace_steps=3))
        steps.extend(
            [
                {"step": 15, "tool_call": bash.to_dict()},
                {"step": 16, "tool_call": view.to_dict()},
            ]
        )
        self.assertEqual(
            rollout.readonly_budget_policy(17, bash, steps, checkpoint_step=10, grace_steps=3),
            "append",
        )
        self.assertIsNone(rollout.readonly_budget_policy(17, edit, steps, checkpoint_step=10, grace_steps=3))

    def test_completion_gate_requires_passing_test_and_diff_after_latest_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            path = workspace / "mod.py"
            path.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
            subprocess.run(["git", "add", "mod.py"], cwd=workspace, check=True, capture_output=True)
            path.write_text("VALUE = 2\n", encoding="utf-8")
            finish = rollout.OpenHandsToolParser.parse_first('{"tool_name":"finish","arguments":{}}')

            blocked = rollout.completion_gate_result(finish, workspace, [])
            self.assertIsNotNone(blocked)
            assert blocked is not None
            self.assertEqual(len(blocked.extra["missing_tools"]), 2)

            steps = [
                {
                    "step": 1,
                    "tool_call": {
                        "tool_name": "str_replace_editor",
                        "arguments": {"command": "str_replace", "path": "mod.py"},
                    },
                    "observation": {"returncode": 0},
                },
                {
                    "step": 2,
                    "tool_call": {"tool_name": "run_tests", "arguments": {}},
                    "observation": {"returncode": 0, "extra": {"all_skipped": False}},
                },
                {
                    "step": 3,
                    "tool_call": {"tool_name": "git_diff", "arguments": {}},
                    "observation": {"returncode": 0},
                },
            ]
            self.assertIsNone(rollout.completion_gate_result(finish, workspace, steps))
            self.assertTrue(rollout.completion_ready(workspace, steps))

            steps[1]["observation"] = {"returncode": 1, "extra": {"all_skipped": False}}
            failed_test = rollout.completion_gate_result(finish, workspace, steps)
            self.assertIsNotNone(failed_test)
            assert failed_test is not None
            self.assertIn("passing", failed_test.output)
            self.assertFalse(rollout.completion_ready(workspace, steps))

    def test_prepare_workspace_auto_fetches_missing_repo_cache(self) -> None:
        task = {
            "instance_id": "demo__repo-1",
            "repo": "demo/repo",
            "base_commit": "abcdef1234567890",
        }
        original_fetch = rollout.fetch_repo_to_cache

        def fake_fetch_repo_to_cache(task, repo_cache):
            dest = Path(repo_cache) / rollout.cache_dir_name(task["repo"], task["base_commit"])
            dest.mkdir(parents=True)
            (dest / "pkg").mkdir()
            (dest / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
            return dest

        with tempfile.TemporaryDirectory() as tmp:
            try:
                rollout.fetch_repo_to_cache = fake_fetch_repo_to_cache
                workspace = rollout.prepare_workspace(
                    task,
                    str(Path(tmp) / "cache"),
                    str(Path(tmp) / "work"),
                    "run1",
                    1,
                    auto_fetch_repos=True,
                )
            finally:
                rollout.fetch_repo_to_cache = original_fetch

            self.assertTrue((workspace / "pkg" / "mod.py").exists())
            self.assertTrue((Path(tmp) / "cache" / "demo__repo__abcdef123456").exists())

    def test_source_patch_scope_filters_tests_and_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "src" / "pkg").mkdir(parents=True)
            (workspace / "tests").mkdir()
            (workspace / "examples").mkdir()
            (workspace / "src" / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
            (workspace / "tests" / "test_mod.py").write_text("def test_value():\n    assert True\n", encoding="utf-8")
            (workspace / "examples" / "demo.py").write_text("print('demo')\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)

            (workspace / "src" / "pkg" / "mod.py").write_text("VALUE = 2\n", encoding="utf-8")
            (workspace / "tests" / "test_mod.py").write_text("def test_value():\n    assert False\n", encoding="utf-8")
            (workspace / "examples" / "demo.py").write_text("print('changed')\n", encoding="utf-8")

            raw_patch = rollout.git_diff(workspace, scope="all")
            source_patch = rollout.git_diff(workspace, scope="source")

            self.assertIn("src/pkg/mod.py", raw_patch)
            self.assertIn("tests/test_mod.py", raw_patch)
            self.assertIn("examples/demo.py", raw_patch)
            self.assertIn("src/pkg/mod.py", source_patch)
            self.assertNotIn("tests/test_mod.py", source_patch)
            self.assertNotIn("examples/demo.py", source_patch)


def assistant_tool_names(messages: list[dict[str, str]]) -> list[str]:
    tools = []
    for item in messages:
        if item["role"] != "assistant":
            continue
        payload = json.loads(item["content"])
        tools.append(payload["tool_name"])
    return tools


if __name__ == "__main__":
    unittest.main()
