from pathlib import Path
import tempfile
import unittest

from code_agent_baseline.repo_context import build_repo_map, compact_repo_context, search_repo_context


class RepoContextTest(unittest.TestCase):
    def test_builds_module_graph_and_snippets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "pkg").mkdir()
            (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (repo / "pkg" / "parser.py").write_text(
                "from pkg.text import clean_text\n\n"
                "class Parser:\n"
                "    def parse_title(self, value):\n"
                "        return clean_text(value)\n",
                encoding="utf-8",
            )
            (repo / "pkg" / "text.py").write_text(
                "def clean_text(value):\n"
                "    return value.strip().lower()\n",
                encoding="utf-8",
            )

            repo_map = build_repo_map(repo)
            result = search_repo_context(repo_map, "parse title clean text", max_files=3, max_snippets=2)
            paths = [item["path"] for item in result["files"]]

            self.assertIn("pkg/parser.py", paths)
            self.assertIn("pkg.text", repo_map.import_graph["pkg.parser"])
            self.assertIn("pkg.parser", repo_map.reverse_import_graph["pkg.text"])
            self.assertTrue(result["snippets"])

    def test_compact_context_keeps_backward_compatible_prompt_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "demo.py").write_text("def run_demo():\n    return 1\n", encoding="utf-8")
            text = compact_repo_context(build_repo_map(repo), "run demo", max_files=2, max_chars=2000)
            self.assertIn("Repository context map", text)
            self.assertIn("demo.py", text)
            self.assertIn("symbols=run_demo", text)


if __name__ == "__main__":
    unittest.main()
