import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "data" / "dataset_splits_v1" / "selection.json"


class DatasetSplitsTest(unittest.TestCase):
    def test_split_counts_and_no_task_leakage(self) -> None:
        data = json.loads(SELECTION.read_text(encoding="utf-8"))
        expected = {
            "train": (300, 120, 80, 100),
            "validation": (45, 15, 15, 15),
            "evaluation": (90, 30, 30, 30),
        }
        for role, (total, mbpp, humaneval, swesmith) in expected.items():
            split = data["splits"][role]
            self.assertEqual(split["count"], total)
            self.assertEqual(len(split["mbppplus"]["task_ids"]), mbpp)
            self.assertEqual(len(split["humanevalplus"]["task_ids"]), humaneval)
            self.assertEqual(len(split["swesmith_py"]["instance_ids"]), swesmith)

        for dataset, id_field in (
            ("mbppplus", "task_ids"),
            ("humanevalplus", "task_ids"),
            ("swesmith_py", "instance_ids"),
        ):
            train = set(data["splits"]["train"][dataset][id_field])
            validation = set(data["splits"]["validation"][dataset][id_field])
            evaluation = set(data["splits"]["evaluation"][dataset][id_field])
            self.assertFalse(train & validation)
            self.assertFalse(train & evaluation)
            self.assertFalse(validation & evaluation)

    def test_swesmith_repositories_are_split_by_repo(self) -> None:
        data = json.loads(SELECTION.read_text(encoding="utf-8"))
        repos = {
            role: set(data["splits"][role]["swesmith_py"]["profiles"])
            for role in ("train", "validation", "evaluation")
        }
        self.assertFalse(repos["train"] & repos["validation"])
        self.assertFalse(repos["train"] & repos["evaluation"])
        self.assertFalse(repos["validation"] & repos["evaluation"])


if __name__ == "__main__":
    unittest.main()
