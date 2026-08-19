#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_agent_baseline.skills import (  # noqa: E402
    SkillConfig,
    compact_skills,
    ensure_skills_config,
    load_skills,
    set_skill_enabled,
    upsert_skill,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage CodeAgent-RL skill configuration.")
    parser.add_argument("--config", default=".codeagent/skills.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init")
    subparsers.add_parser("list")

    compact = subparsers.add_parser("compact")
    compact.add_argument("--task", required=True)
    compact.add_argument("--max-chars", type=int, default=2500)

    upsert = subparsers.add_parser("upsert")
    upsert.add_argument("--name", required=True)
    upsert.add_argument("--description", required=True)
    upsert.add_argument("--trigger", action="append", default=[])
    upsert.add_argument("--instruction", action="append", required=True)
    upsert.add_argument("--priority", type=int, default=0)

    enable = subparsers.add_parser("enable")
    enable.add_argument("name")

    disable = subparsers.add_parser("disable")
    disable.add_argument("name")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Path(args.config).resolve()
    if args.command == "init":
        ensure_skills_config(config)
        print(f"skills_config={config}")
    elif args.command == "list":
        for skill in load_skills(config):
            state = "enabled" if skill.enabled else "disabled"
            print(f"{skill.name}\t{state}\tpriority={skill.priority}\t{skill.description}")
    elif args.command == "compact":
        print(compact_skills(load_skills(config), args.task, max_chars=args.max_chars))
    elif args.command == "upsert":
        upsert_skill(
            config,
            SkillConfig(
                name=args.name,
                description=args.description,
                triggers=args.trigger,
                instructions=args.instruction,
                priority=args.priority,
            ),
        )
        print(f"upserted={args.name}")
    elif args.command == "enable":
        set_skill_enabled(config, args.name, True)
        print(f"enabled={args.name}")
    elif args.command == "disable":
        set_skill_enabled(config, args.name, False)
        print(f"disabled={args.name}")


if __name__ == "__main__":
    main()

