from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillConfig:
    name: str
    description: str
    triggers: list[str]
    instructions: list[str]
    enabled: bool = True
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SKILLS = [
    SkillConfig(
        name="mini_base_loop",
        description="Use mini-swe-agent style loop: inspect, act, observe, iterate, submit.",
        triggers=["mini", "agent", "loop", "trajectory", "轨迹"],
        instructions=[
            "Prefer small, reversible steps.",
            "Always inspect before editing.",
            "Submit only after a validation or clear final analysis.",
        ],
        priority=5,
    ),
    SkillConfig(
        name="mode_permissions",
        description="Mode-aware permission policy for plan/build/debug.",
        triggers=["mode", "permission", "权限", "plan", "build", "debug"],
        instructions=[
            "In plan mode, do not edit, delete, commit, push, or download repositories.",
            "In build mode, keep code changes minimal and run validation.",
            "In debug mode, prioritize logs, reproduction commands, and root-cause localization.",
        ],
        priority=10,
    ),
    SkillConfig(
        name="repo_context",
        description="Use repo map before reading files in large repositories.",
        triggers=["repo", "context", "codebase", "map", "上下文", "仓库"],
        instructions=[
            "Use the repository context map to pick relevant files first.",
            "Read full files only after selecting likely relevant paths.",
            "Prefer targeted rg/sed reads over broad cat of many files.",
        ],
        priority=8,
    ),
    SkillConfig(
        name="web_download",
        description="Controlled web search and GitHub repository download.",
        triggers=["web", "search", "download", "github", "开源", "下载"],
        instructions=[
            "Use web_search for discovery when enabled.",
            "Use download_repo only for public GitHub repositories and only outside plan mode.",
            "Summarize downloaded repository path and license/readme evidence.",
        ],
        priority=7,
    ),
    SkillConfig(
        name="trajectory_data",
        description="Keep trajectories useful for SFT/RL conversion.",
        triggers=["sft", "rl", "grpo", "trajectory", "数据", "训练"],
        instructions=[
            "Keep action rationales and observations concise but informative.",
            "Preserve pass/fail/blocked/timeout state in final summaries.",
            "Avoid hiding important validation failures.",
        ],
        priority=6,
    ),
]


def default_skill_registry() -> dict[str, Any]:
    return {"version": 1, "skills": [skill.to_dict() for skill in DEFAULT_SKILLS]}


def ensure_skills_config(path: str | Path) -> None:
    config_path = Path(path)
    if config_path.exists():
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(default_skill_registry(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_skills(path: str | Path) -> list[SkillConfig]:
    ensure_skills_config(path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SkillConfig(**item) for item in data.get("skills", [])]


def save_skills(path: str | Path, skills: list[SkillConfig]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"version": 1, "skills": [skill.to_dict() for skill in skills]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upsert_skill(path: str | Path, skill: SkillConfig) -> None:
    skills = load_skills(path)
    by_name = {existing.name: existing for existing in skills}
    by_name[skill.name] = skill
    save_skills(path, sorted(by_name.values(), key=lambda item: item.name))


def set_skill_enabled(path: str | Path, name: str, enabled: bool) -> None:
    skills = load_skills(path)
    found = False
    for skill in skills:
        if skill.name == name:
            skill.enabled = enabled
            found = True
    if not found:
        raise ValueError(f"Unknown skill: {name}")
    save_skills(path, skills)


def compact_skills(skills: list[SkillConfig], task: str, *, max_skills: int = 5, max_chars: int = 2500) -> str:
    selected = sorted(
        [skill for skill in skills if skill.enabled],
        key=lambda skill: _score_skill(skill, task),
        reverse=True,
    )[:max_skills]
    lines = ["Relevant skill configuration:"]
    for skill in selected:
        lines.append(f"- {skill.name}: {skill.description}")
        for instruction in skill.instructions[:5]:
            lines.append(f"  - {instruction}")
    return "\n".join(lines)[:max_chars].rstrip()


def _score_skill(skill: SkillConfig, task: str) -> float:
    task_terms = set(_terms(task))
    trigger_terms = set(_terms(" ".join(skill.triggers + [skill.name, skill.description])))
    return len(task_terms & trigger_terms) * 5.0 + skill.priority


def _terms(text: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text)]

