import unittest

from code_agent_baseline.working_memory import ContextLimitExceeded, build_model_input


class WorkingMemoryTest(unittest.TestCase):
    def test_short_history_is_preserved_exactly(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '{"tool_name":"execute_bash","arguments":{"command":"pwd"}}'},
            {"role": "user", "content": "EXECUTION RESULT: ok"},
        ]
        model_input, snapshot = build_model_input(messages, max_tokens=None, max_chars=10_000)
        self.assertEqual(model_input, messages)
        self.assertFalse(snapshot.compacted)

    def test_long_history_keeps_recent_observation_and_summarizes_older_steps(self) -> None:
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        for index in range(10):
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        '{"tool_name":"execute_bash","arguments":{"command":"sed -n '
                        f"'{index},{index + 20}p' src/mod.py\"}}"
                    ),
                }
            )
            messages.append({"role": "user", "content": f"EXECUTION RESULT {index}: " + ("x" * 1200)})
        latest = messages[-1]["content"]

        model_input, snapshot = build_model_input(
            messages,
            max_tokens=None,
            max_chars=6_000,
            recent_turns=2,
            summary_chars=1_500,
        )

        self.assertTrue(snapshot.compacted)
        self.assertLessEqual(snapshot.input_chars, 6_000)
        self.assertEqual(model_input[-1]["content"], latest)
        self.assertIn("<WORKING_MEMORY>", model_input[2]["content"])
        self.assertGreater(snapshot.summarized_message_count, 0)

    def test_token_budget_does_not_compact_a_large_but_in_budget_trace(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "x" * 72_000},
        ]

        model_input, snapshot = build_model_input(
            messages,
            max_tokens=26_000,
            token_counter=lambda _: 19_600,
            token_counter_name="test-tokenizer",
        )

        self.assertEqual(model_input, messages)
        self.assertFalse(snapshot.compacted)
        self.assertEqual(snapshot.full_history_tokens, 19_600)
        self.assertEqual(snapshot.token_counter, "test-tokenizer")

    def test_error_policy_rejects_over_limit_history_without_summary(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]

        with self.assertRaises(ContextLimitExceeded) as raised:
            build_model_input(
                messages,
                max_tokens=100,
                token_counter=lambda _: 101,
                overflow_policy="error",
            )

        self.assertEqual(raised.exception.snapshot.full_history_tokens, 101)
        self.assertFalse(raised.exception.snapshot.compacted)


if __name__ == "__main__":
    unittest.main()
