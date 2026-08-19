import unittest

from code_agent_baseline.tool_specs import (
    CORE_REPAIR_TOOLS,
    SWE_EXTENSION_TOOLS,
    TOOL_SPECS,
    render_tool_catalog,
    render_tool_few_shot,
    tool_schemas,
)


class ToolSpecsTest(unittest.TestCase):
    def test_every_exposed_tool_has_complete_schema_and_guidance(self) -> None:
        names = [*CORE_REPAIR_TOOLS, *SWE_EXTENSION_TOOLS, "answer"]
        schemas = tool_schemas(names)
        self.assertEqual([item["function"]["name"] for item in schemas], names)
        for name, schema in zip(names, schemas):
            spec = TOOL_SPECS[name]
            function = schema["function"]
            self.assertTrue(function["description"])
            self.assertEqual(function["parameters"]["type"], "object")
            self.assertIn("properties", function["parameters"])
            self.assertIn("required", function["parameters"])
            self.assertFalse(function["parameters"]["additionalProperties"])
            self.assertTrue(spec.use_when)
            self.assertTrue(spec.avoid_when)
            self.assertTrue(spec.output)
            self.assertEqual(spec.example["tool_name"], name)

    def test_catalog_is_generated_once_with_decision_and_output_guidance(self) -> None:
        catalog = render_tool_catalog(CORE_REPAIR_TOOLS)
        self.assertEqual(catalog.count("<TOOLS>"), 1)
        self.assertEqual(catalog.count("</TOOLS>"), 1)
        for name in CORE_REPAIR_TOOLS:
            self.assertEqual(catalog.count(f"- {name}:"), 1)
        self.assertIn("Use when:", catalog)
        self.assertIn("Avoid when:", catalog)
        self.assertIn("Returns:", catalog)
        self.assertIn("Example:", catalog)
        self.assertNotIn("repo_context", catalog)

    def test_few_shot_covers_complete_edit_validation_workflow(self) -> None:
        few_shot = render_tool_few_shot()
        positions = [
            few_shot.index('"command":"view"'),
            few_shot.index('"command":"str_replace"'),
            few_shot.index('"tool_name":"run_tests"'),
            few_shot.index('"tool_name":"git_diff"'),
            few_shot.index('"tool_name":"finish"'),
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(few_shot.count('"tool_name":"run_tests"'), 2)
        self.assertEqual(few_shot.count('"command":"str_replace"'), 2)
        self.assertIn("returncode=1", few_shot)
        self.assertIn("returncode=0", few_shot)
        self.assertIn("generic workflow example", few_shot)
        self.assertNotIn('"tool_name":"answer"', few_shot)
        self.assertIn('"tool_name":"answer"', render_tool_few_shot(include_answer=True))


if __name__ == "__main__":
    unittest.main()
