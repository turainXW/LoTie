from __future__ import annotations

from pathlib import Path

from tools.build_salvage_sft_dataset import Candidate, salvage_candidate


def candidate(messages: list[dict[str, str]], invalid_indices: tuple[int, ...]) -> Candidate:
    return Candidate(
        sample_path=Path("/raw/sample_01/tasks/demo/task/sample_result.json"),
        sample={
            "instance_id": "demo-task",
            "dataset": "swesmith_py",
            "sample_index": 1,
            "benchmark_resolved": True,
            "agent_status": "resolved",
            "verifier_status": "resolved",
            "attempt_dir": "/raw/sample_01/tasks/demo/task/attempt_01",
        },
        messages_path=Path("/raw/messages_sft.jsonl"),
        trace_path=Path("/raw/openhands_patch_rollout.jsonl"),
        messages=messages,
        invalid_indices=invalid_indices,
    )


def test_drops_only_invalid_assistant_parser_pair() -> None:
    item = candidate(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "I will inspect first"},
            {"role": "user", "content": "EXECUTION RESULT of [parser]:\nNo tool call"},
            {
                "role": "assistant",
                "content": '{"tool_name":"execute_bash","arguments":{"command":"pwd"}}',
            },
            {"role": "user", "content": "EXECUTION RESULT of [execute_bash]:\nok"},
        ],
        (2,),
    )

    result = salvage_candidate(item, run_dir=Path("/raw"), append_finish_if_resolved=True)

    assert result["salvage"] is not None
    assert result["audit"]["removed_message_indices"] == [2, 3]
    assert result["salvage"]["messages"][-1]["content"] == '{"tool_name":"finish","arguments":{}}'
    assert all("I will inspect" not in message["content"] for message in result["salvage"]["messages"])


def test_unpaired_invalid_assistant_requires_manual_review() -> None:
    item = candidate(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "not json"},
            {"role": "user", "content": "EXECUTION RESULT of [execute_bash]:\nok"},
        ],
        (2,),
    )

    result = salvage_candidate(item, run_dir=Path("/raw"), append_finish_if_resolved=True)

    assert result["salvage"] is None
    assert result["audit"]["status"] == "manual_review"
    assert result["audit"]["invalid_turns"][0]["environment_side_effect"] == "unknown"


def test_drops_rejected_premature_finish_pair() -> None:
    item = candidate(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '{"tool_name":"finish","arguments":{}}'},
            {
                "role": "user",
                "content": (
                    "EXECUTION RESULT of [finish]:\n"
                    "Completion gate rejected finish because git_diff is missing."
                ),
            },
            {"role": "assistant", "content": '{"tool_name":"git_diff","arguments":{}}'},
            {"role": "user", "content": "EXECUTION RESULT of [git_diff]:\npatch"},
            {"role": "assistant", "content": "not json"},
            {"role": "user", "content": "EXECUTION RESULT of [parser]:\nNo tool call"},
            {"role": "assistant", "content": '{"tool_name":"finish","arguments":{}}'},
            {"role": "user", "content": "EXECUTION RESULT of [finish]:\nFinished."},
        ],
        (6,),
    )

    result = salvage_candidate(item, run_dir=Path("/raw"), append_finish_if_resolved=True)

    assert result["salvage"] is not None
    assert result["audit"]["removed_message_indices"] == [2, 3, 6, 7]
    assert result["salvage"]["removed_premature_finish_pairs"] == 1
    assert result["salvage"]["messages"][-1]["content"] == '{"tool_name":"finish","arguments":{}}'


def test_drops_schema_invalid_tool_call_after_runtime_rejection() -> None:
    item = candidate(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '{"tool_name":"execute_bash","arguments":{}}'},
            {
                "role": "user",
                "content": (
                    "EXECUTION RESULT of [execute_bash]:\ncommand is required\n"
                    "[Command finished with exit code 1]\nProtocol reminder: retry with a command."
                ),
            },
            {
                "role": "assistant",
                "content": '{"tool_name":"execute_bash","arguments":{"command":"pwd"}}',
            },
            {"role": "user", "content": "EXECUTION RESULT of [execute_bash]:\nok"},
        ],
        (),
    )

    result = salvage_candidate(item, run_dir=Path("/raw"), append_finish_if_resolved=True)

    assert result["salvage"] is not None
    assert result["audit"]["removed_message_indices"] == [2, 3]
    assert result["salvage"]["removed_protocol_rejection_pairs"] == 1
    assert all("command is required" not in message["content"] for message in result["salvage"]["messages"])


def test_drops_unknown_tool_only_when_runtime_explicitly_rejects_it() -> None:
    item = candidate(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '{"tool_name":"str_replace__editor","arguments":{}}'},
            {
                "role": "user",
                "content": (
                    "EXECUTION RESULT of [str_replace__editor]:\n"
                    "Tool not allowed: str_replace__editor\n[Command finished with exit code 1]"
                ),
            },
            {"role": "assistant", "content": '{"tool_name":"git_diff","arguments":{}}'},
            {"role": "user", "content": "EXECUTION RESULT of [git_diff]:\npatch"},
        ],
        (),
    )

    result = salvage_candidate(item, run_dir=Path("/raw"), append_finish_if_resolved=True)

    assert result["salvage"] is not None
    assert result["audit"]["removed_message_indices"] == [2, 3]
    assert result["audit"]["rejected_protocol_turns"][0]["protocol_errors"] == ["unknown_tool"]


def test_quarantines_executed_schema_drift_instead_of_rewriting_it() -> None:
    item = candidate(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": '{"tool_name":"execute_bash","arguments":{"command":"pwd","note":"read only"}}',
            },
            {"role": "user", "content": "EXECUTION RESULT of [execute_bash]:\n/raw"},
        ],
        (),
    )

    result = salvage_candidate(item, run_dir=Path("/raw"), append_finish_if_resolved=True)

    assert result["salvage"] is None
    assert result["audit"]["status"] == "manual_review"
    assert "violates the declared tool schema" in result["audit"]["post_clean_validation_error"]
