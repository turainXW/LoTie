from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sanitize_release_data.py"
SPEC = importlib.util.spec_from_file_location("sanitize_release_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_sanitize_jsonl_removes_machine_paths_recursively(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "instance_id": "task-1",
                "local_repo_path": "/private/repo",
                "metadata": {"local_venv_path": "/private/venv", "split": "evaluation"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert MODULE.sanitize_jsonl(path) == 1
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row == {"instance_id": "task-1", "metadata": {"split": "evaluation"}}


def test_sanitize_json_replaces_paths_inside_nested_previews(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "source": "/Users/example/project/data/input.jsonl",
                "preview": "tool failed in /private/tmp/repo-1/test.py",
                "nested": ["/root/autodl-tmp/run/checkpoint"],
            }
        ),
        encoding="utf-8",
    )

    replacements = [
        ("/Users/example/project", "<project_root>"),
        ("/private/tmp", "<tmp>"),
        ("/root/autodl-tmp", "<remote_root>"),
    ]
    assert MODULE.sanitize_json(path, replacements) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "source": "<project_root>/data/input.jsonl",
        "preview": "tool failed in <tmp>/repo-1/test.py",
        "nested": ["<remote_root>/run/checkpoint"],
    }


def test_parse_replacement_sorts_are_applied_safely() -> None:
    replacements = sorted(
        [("/Users/example", "<home>"), ("/Users/example/project", "<project_root>")],
        key=lambda item: len(item[0]),
        reverse=True,
    )
    assert MODULE.sanitize("/Users/example/project/data", replacements) == (
        "<project_root>/data"
    )
