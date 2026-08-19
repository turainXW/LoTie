from __future__ import annotations

from typing import Any


HARNESS_VERSION = "lottie_code_agent_harness_v3"
READONLY_POLICY_VERSION = "append_nonmod3_action_reset_v1"
MODIFICATION_COMMANDS = frozenset({"str_replace", "line_replace", "insert", "create", "undo_edit"})


def harness_metadata(*, checkpoint_step: int, reminder_interval: int) -> dict[str, Any]:
    return {
        "version": HARNESS_VERSION,
        "readonly_budget_policy": {
            "version": READONLY_POLICY_VERSION,
            "checkpoint_step": checkpoint_step,
            "reminder_interval": max(1, reminder_interval),
            "reset_on_modification_action": True,
            "modification_commands": sorted(MODIFICATION_COMMANDS),
            "checks_repository_diff": False,
            "blocks_actions": False,
            "preserves_tool_result": True,
        },
    }
