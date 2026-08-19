from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .memory import compact_memories, ensure_memory_store
from .repo_context import build_repo_map, compact_repo_context, load_repo_map, save_repo_map
from .skills import compact_skills, ensure_skills_config, load_skills


@dataclass
class ContextBundle:
    repo_map_path: Path
    skills_config_path: Path
    memory_dir: Path
    prompt_block: str


def build_context_bundle(
    repo: str | Path,
    task: str,
    *,
    repo_map_path: str | Path | None = None,
    skills_config_path: str | Path | None = None,
    memory_dir: str | Path | None = None,
    refresh_repo_map: bool = False,
    repo_max_files: int = 12,
    repo_max_chars: int = 5000,
    skills_max_chars: int = 2500,
    memory_max_chars: int = 3000,
) -> ContextBundle:
    repo_path = Path(repo).resolve()
    state_dir = repo_path / ".codeagent"
    map_path = Path(repo_map_path) if repo_map_path else state_dir / "repo_map.json"
    skills_path = Path(skills_config_path) if skills_config_path else state_dir / "skills.json"
    memory_path = Path(memory_dir) if memory_dir else state_dir / "memory"

    if refresh_repo_map or not map_path.exists():
        repo_map = build_repo_map(repo_path)
        save_repo_map(repo_map, map_path)
    else:
        repo_map = load_repo_map(map_path)

    ensure_skills_config(skills_path)
    skills = load_skills(skills_path)
    ensure_memory_store(memory_path)

    repo_block = compact_repo_context(
        repo_map,
        task,
        max_files=repo_max_files,
        max_chars=repo_max_chars,
    )
    skills_block = compact_skills(skills, task, max_chars=skills_max_chars)
    memory_block = compact_memories(memory_path, task, max_chars=memory_max_chars)

    prompt_block = "\n\n".join(
        [
            "Context management policy:",
            "- Treat this context as a compact map, not as the full source of truth.",
            "- Prefer targeted file reads based on the repo map.",
            "- Keep observations compact; summarize long outputs before continuing.",
            repo_block,
            skills_block,
            memory_block,
        ]
    )
    return ContextBundle(
        repo_map_path=map_path,
        skills_config_path=skills_path,
        memory_dir=memory_path,
        prompt_block=prompt_block,
    )
