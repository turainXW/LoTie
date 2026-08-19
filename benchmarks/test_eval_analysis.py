import tempfile
import unittest
import importlib.util
from pathlib import Path

from code_agent_baseline.eval_analysis import analyze_runs, canonical_outcome, normalize_dataset


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_eval_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_eval_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyze_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze_script)


class EvalAnalysisTest(unittest.TestCase):
    def test_run_spec_allows_equals_in_label(self) -> None:
        label, path = analyze_script.parse_run("Qwen3-4B t=0=results.jsonl")

        self.assertEqual(label, "Qwen3-4B t=0")
        self.assertEqual(path, Path("results.jsonl"))

    def test_canonical_outcome_is_mutually_exclusive(self) -> None:
        self.assertEqual(canonical_outcome({"benchmark_resolved": True, "agent_status": "max_steps"}), "resolved")
        self.assertEqual(
            canonical_outcome(
                {
                    "benchmark_resolved": False,
                    "agent_status": "resolved",
                    "verifier_status": "unresolved_tests_failed",
                    "patch_present": True,
                }
            ),
            "tests_failed",
        )
        self.assertEqual(
            canonical_outcome(
                {
                    "benchmark_resolved": False,
                    "agent_status": "parse_error",
                    "verifier_status": "unresolved_no_edit",
                }
            ),
            "protocol_parse_error",
        )

    def test_dataset_aliases(self) -> None:
        self.assertEqual(normalize_dataset("mbppplus"), "MBPP+")
        self.assertEqual(normalize_dataset("HumanEval+"), "HumanEval+")
        self.assertEqual(normalize_dataset("swesmith_py"), "SWE-smith")

    def test_analyze_runs_computes_rounds_and_coverage(self) -> None:
        rows = [
            '{"instance_id":"a","dataset":"mbppplus","sample_index":1,"benchmark_resolved":true,"verifier_status":"resolved"}',
            '{"instance_id":"a","dataset":"mbppplus","sample_index":2,"benchmark_resolved":false,"verifier_status":"unresolved_tests_failed","patch_present":true}',
            '{"instance_id":"b","dataset":"humanevalplus","sample_index":1,"benchmark_resolved":false,"verifier_status":"unresolved_no_edit","patch_present":false}',
            '{"instance_id":"b","dataset":"humanevalplus","sample_index":2,"benchmark_resolved":true,"verifier_status":"resolved"}',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            analysis = analyze_runs([("demo", path)])

        run = analysis["runs"][0]
        self.assertEqual(run["resolved"], 2)
        self.assertEqual(run["coverage"]["covered_tasks"], 2)
        self.assertEqual(run["coverage"]["rate"], 100.0)
        self.assertEqual(run["rounds"]["1"]["resolved"], 1)
        self.assertEqual(run["rounds"]["2"]["resolved"], 1)
        self.assertEqual(run["canonical_outcomes"]["tests_failed"], 1)
        self.assertEqual(run["canonical_outcomes"]["no_effective_edit"], 1)


if __name__ == "__main__":
    unittest.main()
