# Open-source code agent quickstart survey

Date: 2026-06-14

This note surveys open-source coding agents that can be adapted quickly as a
local baseline for `agentrl`. The main target is not an IDE product; it is a
small, inspectable agent loop that can produce trajectories for later SFT/RL.

## Short answer

Use `mini-swe-agent` first.

Why:

- It is intentionally minimal: the README describes the core agent class as
  roughly 100 lines of Python.
- Its only real action interface is bash, so the environment and trajectory are
  easy to instrument.
- It has a linear message/action history, which is convenient for debugging,
  converting to JSONL, and later fine-tuning.
- It supports local environments and sandbox backends such as Docker/Podman.
- It is MIT licensed and installable from PyPI.

Repository:
https://github.com/SWE-agent/mini-swe-agent

Docs:
https://mini-swe-agent.com

## Candidate comparison

| Candidate | Best fit | License | Setup friction | Fit for this repo |
| --- | --- | --- | --- | --- |
| `mini-swe-agent` | Minimal local code agent and trajectory baseline | MIT | Low | Best first choice |
| `SWE-agent` | SWE-bench style issue repair with configurable tools/history | MIT | Medium | Good second choice if custom tool interfaces matter |
| `OpenHands` | Full software-agent platform with SDK, CLI, GUI, sandboxing | MIT core, enterprise folder separate | Medium-high | Useful reference, heavier than needed for first baseline |
| `aider` | Terminal pair-programming assistant | Apache-2.0 | Low | Good user-facing tool, less ideal for autonomous RL trajectory collection |
| `AutoCodeRover` | Structure-aware GitHub issue repair/SWE-bench workflows | Source-available, not a clean OSS baseline | Medium-high, Docker oriented | Useful reference, but avoid as first reusable baseline |
| `opencode`, `Cline`, `Roo Code` | CLI/editor coding products | MIT or Apache-2.0 depending on project | Low-medium | Useful UX references, not the cleanest research baseline |

## Recommendation

Start with `mini-swe-agent` as a separate baseline module:

1. Clone or vendor a minimal subset under `refs/mini-swe-agent` or keep it as an
   external dependency.
2. Run a 3-task smoke test on tiny local coding tasks.
3. Add a thin trajectory logger that writes one JSONL object per episode.
4. Convert successful and failed trajectories into the same style as existing
   agent/tool datasets in this repo.
5. Only move to `SWE-agent` or `OpenHands` if we need richer tool interfaces,
   browser/GUI interaction, or official SWE-bench harness support.

## Why not start with a full platform

OpenHands is strong and actively maintained, but it brings a full platform
surface: SDK, CLI, GUI, REST API, Docker images, and enterprise/cloud pieces.
That is useful later, but for a first local experiment it increases the number
of moving parts. The repo here already has custom data pipelines and evaluation
scripts, so a small agent loop is easier to instrument.

## Sources checked

- `mini-swe-agent`: https://github.com/SWE-agent/mini-swe-agent
- `SWE-agent`: https://github.com/SWE-agent/SWE-agent
- `OpenHands`: https://github.com/OpenHands/OpenHands
- `aider`: https://github.com/Aider-AI/aider
- `AutoCodeRover`: https://github.com/AutoCodeRoverSG/auto-code-rover
- `opencode`: https://github.com/anomalyco/opencode
- `Cline`: https://github.com/cline/cline
- `Roo Code`: https://github.com/RooCodeInc/Roo-Code
