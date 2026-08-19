#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


IMAGE_FIELDS = (
    "image",
    "image_name",
    "docker_image",
    "docker_image_name",
    "eval_image",
    "instance_image",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight SWE-Gym task Docker images before running agent rollouts.")
    parser.add_argument("--tasks", required=True, help="SWE-Gym task JSONL/parquet.")
    parser.add_argument("--output", required=True, help="Per-task image preflight JSONL.")
    parser.add_argument("--runnable-output", required=True, help="Task JSONL containing only runnable image tasks.")
    parser.add_argument("--blocked-output", required=True, help="Task JSONL containing image-blocked tasks.")
    parser.add_argument("--namespace", default="xingyaoww")
    parser.add_argument("--mirror-prefix", default="docker.1ms.run")
    parser.add_argument("--arch", default="x86_64")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--pull", action="store_true", help="Run docker pull when image is not already present.")
    parser.add_argument("--check-manifest", action="store_true", help="Run docker manifest inspect before docker pull.")
    parser.add_argument("--timeout-sec", type=int, default=240, help="Timeout per docker pull/inspect candidate.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stop-after-runnable", type=int, help="Stop once this many runnable tasks have been found.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.tasks))
    if args.limit is not None:
        rows = rows[: args.limit]

    output = Path(args.output)
    runnable_output = Path(args.runnable_output)
    blocked_output = Path(args.blocked_output)
    for path in (output, runnable_output, blocked_output):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    counts: dict[str, int] = {}
    runnable_count = 0
    blocked_count = 0
    processed_count = 0
    for row in rows:
        if args.stop_after_runnable is not None and runnable_count >= args.stop_after_runnable:
            break
        processed_count += 1
        result = preflight_task_image(
            row,
            namespace=args.namespace,
            mirror_prefix=args.mirror_prefix,
            arch=args.arch,
            tag=args.tag,
            pull=args.pull,
            check_manifest=args.check_manifest,
            timeout_sec=args.timeout_sec,
        )
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        append_jsonl(output, result)

        enriched = dict(row)
        if result.get("selected_image"):
            enriched["image"] = result["selected_image"]
            enriched["docker_image"] = result["selected_image"]
        enriched["image_preflight_status"] = result["status"]
        enriched["image_preflight"] = {
            "selected_image": result.get("selected_image"),
            "attempts": result.get("attempts", []),
        }
        if result["runnable"]:
            runnable_count += 1
            append_jsonl(runnable_output, json_safe(enriched))
        else:
            blocked_count += 1
            append_jsonl(blocked_output, json_safe(enriched))

    summary = {
        "tasks": len(rows),
        "processed": processed_count,
        "runnable": runnable_count,
        "blocked": blocked_count,
        "status_counts": counts,
        "output": str(output),
        "runnable_output": str(runnable_output),
        "blocked_output": str(blocked_output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def preflight_task_image(
    row: dict[str, Any],
    *,
    namespace: str,
    mirror_prefix: str,
    arch: str,
    tag: str,
    pull: bool,
    check_manifest: bool,
    timeout_sec: int,
) -> dict[str, Any]:
    instance_id = str(row.get("instance_id") or "")
    candidates = image_candidates(
        row,
        namespace=namespace,
        mirror_prefix=mirror_prefix,
        arch=arch,
        tag=tag,
    )
    attempts = []
    for image in candidates:
        inspect = run_docker(["docker", "image", "inspect", image], timeout_sec=timeout_sec)
        attempts.append({"image": image, "action": "inspect", **inspect})
        if inspect["ok"]:
            return {
                "instance_id": instance_id,
                "runnable": True,
                "status": "local_image_exists",
                "selected_image": image,
                "attempts": attempts,
            }
        if not pull:
            continue
        if check_manifest:
            manifest_result = run_docker(
                ["docker", "manifest", "inspect", image],
                timeout_sec=timeout_sec,
                use_system_timeout=True,
            )
            attempts.append({"image": image, "action": "manifest", **manifest_result})
            if not manifest_result["ok"]:
                continue
        pull_result = run_docker(["docker", "pull", image], timeout_sec=timeout_sec)
        attempts.append({"image": image, "action": "pull", **pull_result})
        if pull_result["ok"]:
            return {
                "instance_id": instance_id,
                "runnable": True,
                "status": "pulled",
                "selected_image": image,
                "attempts": attempts,
            }

    return {
        "instance_id": instance_id,
        "runnable": False,
        "status": classify_blocked_status(attempts),
        "selected_image": None,
        "attempts": attempts,
    }


def image_candidates(
    row: dict[str, Any],
    *,
    namespace: str,
    mirror_prefix: str,
    arch: str,
    tag: str,
) -> list[str]:
    candidates: list[str] = []
    for field in IMAGE_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    inferred = infer_image(str(row.get("instance_id") or ""), namespace=namespace, arch=arch, tag=tag)
    if inferred:
        candidates.append(f"{mirror_prefix}/{inferred}")
        candidates.append(inferred)

    deduped = []
    seen = set()
    for image in candidates:
        if image not in seen:
            seen.add(image)
            deduped.append(image)
    return deduped


def infer_image(instance_id: str, *, namespace: str, arch: str, tag: str) -> str | None:
    if "__" not in instance_id:
        return None
    owner, issue_part = instance_id.split("__", 1)
    if not owner or not issue_part:
        return None
    image_key = f"{owner}_s_{issue_part}".lower()
    return f"{namespace}/sweb.eval.{arch}.{image_key}:{tag}"


def run_docker(command: list[str], *, timeout_sec: int, use_system_timeout: bool = True) -> dict[str, Any]:
    return run_docker_command(command, timeout_sec=timeout_sec, use_system_timeout=use_system_timeout)


def run_docker_command(command: list[str], *, timeout_sec: int, use_system_timeout: bool) -> dict[str, Any]:
    started = time.monotonic()
    run_command = command
    timeout_bin = shutil.which("timeout")
    if use_system_timeout and timeout_bin:
        run_command = [timeout_bin, "--kill-after=10s", str(timeout_sec), *command]
    try:
        completed = subprocess.run(run_command, capture_output=True, text=True, timeout=timeout_sec + 15)
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "status": "timeout",
            "elapsed_sec": round(time.monotonic() - started, 3),
            "output": (decode(exc.stdout) + decode(exc.stderr))[-4000:],
        }
    output = ((completed.stdout or "") + (completed.stderr or ""))[-4000:]
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "status": "ok" if completed.returncode == 0 else "failed",
        "elapsed_sec": round(time.monotonic() - started, 3),
        "output": output,
    }


def classify_blocked_status(attempts: list[dict[str, Any]]) -> str:
    outputs = "\n".join(str(attempt.get("output") or "").lower() for attempt in attempts)
    statuses = {str(attempt.get("status")) for attempt in attempts}
    if "timeout" in statuses:
        return "blocked_image_pull_timeout"
    if "not found" in outputs or "manifest unknown" in outputs:
        return "blocked_image_not_found"
    if "content descriptor" in outputs or "failed to copy" in outputs or "eof" in outputs:
        return "blocked_image_layer_fetch_failed"
    if "unauthorized" in outputs or "denied" in outputs or "pull access denied" in outputs:
        return "blocked_image_access_denied"
    return "blocked_image_pull_failed"


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading parquet task files requires pandas and pyarrow.") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    return read_jsonl(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return value


def decode(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


if __name__ == "__main__":
    main()
