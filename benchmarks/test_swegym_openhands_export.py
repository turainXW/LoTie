import json
from pathlib import Path
import tempfile
import unittest

from code_agent_baseline.swegym_openhands_export import export_grounded_to_openhands


class SwegymOpenHandsExportTest(unittest.TestCase):
    def test_exports_grounded_trace_to_openhands_like_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traces = root / "grounded_traces.jsonl"
            traces.write_text(
                json.dumps(
                    {
                        "instance_id": "repo__proj-1",
                        "repo": "repo/proj",
                        "base_commit": "abc123",
                        "candidate_files": [{"path": "pkg/mod.py"}],
                        "prompt": "Problem statement:\nFix the parser.\n\nCandidate files:",
                        "answer": "{}",
                        "parsed": {
                            "root_cause_hypothesis": "Parser drops a case.",
                            "files_to_inspect": ["pkg/mod.py"],
                            "tests_to_run": ["pytest tests/test_mod.py"],
                            "tool_plan": [{"tool": "repo_context", "command_or_query": "parser"}],
                            "minimal_fix_strategy": "Patch parser branch.",
                            "confidence": 0.7,
                        },
                        "parse_ok": True,
                        "status": "ok",
                        "repo_status": "cached",
                        "latency_sec": 1.2,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = export_grounded_to_openhands(traces, root / "out")

            sampled = [json.loads(line) for line in Path(manifest.sampled_path).read_text(encoding="utf-8").splitlines()]
            sft = [json.loads(line) for line in Path(manifest.sft_path).read_text(encoding="utf-8").splitlines()]
            verifier = [json.loads(line) for line in Path(manifest.verifier_path).read_text(encoding="utf-8").splitlines()]

            self.assertEqual(len(sampled), 1)
            self.assertEqual(len(sft), 1)
            self.assertEqual(len(verifier), 1)
            self.assertEqual(sampled[0]["instance_id"], "repo__proj-1")
            self.assertIn("messages", sampled[0])
            self.assertIn("tools", sampled[0])
            self.assertIn("test_result", sampled[0])
            self.assertTrue(sft[0]["resolved"])
            self.assertFalse(sampled[0]["resolved"])
            self.assertTrue(sampled[0]["test_result"]["report"]["planner_only"])


if __name__ == "__main__":
    unittest.main()
