from pathlib import Path
import json
import tempfile
import unittest

from code_agent_baseline.openhands_tools import OpenHandsToolParser, OpenHandsToolRuntime
from code_agent_baseline.problem_search import build_problem_index, problem_search, save_problem_index


class ProblemSearchTest(unittest.TestCase):
    def test_build_and_search_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks.jsonl"
            tasks.write_text(
                json.dumps(
                    {
                        "task_id": "task_slugify",
                        "instruction": "Implement slugify for URL slugs.",
                        "validation_command": "python3 -B test_task.py",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "task_id": "task_mean",
                        "instruction": "Fix mean to return floating point average.",
                        "validation_command": "python3 -B test_task.py",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = build_problem_index([root])
            self.assertEqual(len(records), 2)
            results = problem_search("slug url", roots=[root], top_k=1)
            self.assertEqual(results["results"][0]["task_id"], "task_slugify")

            index = root / "index.jsonl"
            save_problem_index(records, index)
            indexed = problem_search("floating average", index_path=index, top_k=1)
            self.assertEqual(indexed["results"][0]["task_id"], "task_mean")

    def test_openhands_problem_search_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tasks.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": "task_parser",
                        "instruction": "Fix parser edge case for empty JSON lines.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            runtime = OpenHandsToolRuntime(root)
            call = OpenHandsToolParser.parse_first(
                "<function=problem_search>"
                "<parameter=query>json parser</parameter>"
                "<parameter=top_k>1</parameter>"
                "</function>"
            )
            result = runtime.execute(call)
            self.assertEqual(result.returncode, 0)
            payload = json.loads(result.output)
            self.assertEqual(payload["results"][0]["task_id"], "task_parser")


if __name__ == "__main__":
    unittest.main()
