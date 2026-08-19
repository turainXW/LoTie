#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a live dashboard for a resumable Pass@k run.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-rollouts", required=True, type=int)
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--runner-log")
    parser.add_argument("--label", default="pass-at-k")
    parser.add_argument("--interval-sec", type=float, default=2.0)
    parser.add_argument("--stop-when-complete", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "live_status.json"
    metrics_path = output_dir / "live_metrics.jsonl"
    peaks = load_peaks(status_path)
    while True:
        snapshot = build_snapshot(args, output_dir, peaks)
        peaks = snapshot["gpu"]["peaks"]
        write_json(status_path, snapshot)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        print(compact_line(snapshot), flush=True)
        if args.once or (args.stop_when_complete and snapshot["progress"]["valid"] >= args.target_rollouts):
            return
        time.sleep(args.interval_sec)


def build_snapshot(args: argparse.Namespace, output_dir: Path, peaks: dict[str, float]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    rows = read_jsonl(output_dir / "sample_results.jsonl")
    summary = read_json(output_dir / "pass_at_k_summary.json")
    gpu = gpu_snapshot()
    updated_peaks = {
        "memory_mib": max(float(peaks.get("memory_mib", 0)), float(gpu.get("memory_mib", 0))),
        "utilization_pct": max(float(peaks.get("utilization_pct", 0)), float(gpu.get("utilization_pct", 0))),
        "temperature_c": max(float(peaks.get("temperature_c", 0)), float(gpu.get("temperature_c", 0))),
        "power_w": max(float(peaks.get("power_w", 0)), float(gpu.get("power_w", 0))),
    }
    latest_mtime = latest_result_mtime(output_dir)
    usage = Counter()
    for row in rows:
        usage.update({key: int(value or 0) for key, value in (row.get("usage") or {}).items()})
    valid = len(rows)
    target = args.target_rollouts
    progress = {
        "valid": valid,
        "target": target,
        "remaining": max(0, target - valid),
        "percent": round(100 * valid / target, 2) if target else 100.0,
        "resolved": sum(row.get("benchmark_resolved") is True for row in rows),
        "unresolved": sum(row.get("benchmark_resolved") is False for row in rows),
        "by_dataset": dict(Counter(str(row.get("dataset") or "unknown") for row in rows)),
        "agent_status": dict(Counter(str(row.get("agent_status") or "unknown") for row in rows)),
        "verifier_status": dict(Counter(str(row.get("verifier_status") or "unknown") for row in rows)),
    }
    return {
        "format": "lottie_pass_at_k_live_status_v1",
        "timestamp": now.isoformat(),
        "label": args.label,
        "output_dir": str(output_dir),
        "progress": progress,
        "pass_at_k": summary.get("aggregate", {}),
        "usage": dict(usage),
        "gpu": {**gpu, "peaks": updated_peaks},
        "vllm": vllm_metrics(args.metrics_url),
        "runner": {
            "processes": process_count("run_pass_at_k_experiment.py"),
            "log": args.runner_log,
            "log_mtime": file_mtime(args.runner_log),
            "last_result_at": datetime.fromtimestamp(latest_mtime, timezone.utc).isoformat() if latest_mtime else None,
            "stale_sec": round(max(0.0, now.timestamp() - latest_mtime), 1) if latest_mtime else None,
        },
    }


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True, timeout=10).splitlines()[0]
        memory, total, utilization, temperature, power = [part.strip() for part in line.split(",")]
        return {
            "memory_mib": int(memory),
            "memory_total_mib": int(total),
            "utilization_pct": int(utilization),
            "temperature_c": int(temperature),
            "power_w": float(power),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def vllm_metrics(url: str) -> dict[str, float | None]:
    wanted = {
        "vllm:num_requests_running": "requests_running",
        "vllm:num_requests_waiting": "requests_waiting",
        "vllm:kv_cache_usage_perc": "kv_cache_usage",
    }
    output: dict[str, float | None] = {value: None for value in wanted.values()}
    try:
        with urlopen(url, timeout=5) as response:
            text = response.read().decode("utf-8", errors="replace")
    except OSError:
        return output
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric = line.split("{", 1)[0].split(" ", 1)[0]
        if metric not in wanted:
            continue
        try:
            output[wanted[metric]] = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
    return output


def compact_line(snapshot: dict[str, Any]) -> str:
    progress = snapshot["progress"]
    gpu = snapshot["gpu"]
    vllm = snapshot["vllm"]
    return (
        f"{snapshot['timestamp']} {snapshot['label']} "
        f"progress={progress['valid']}/{progress['target']} resolved={progress['resolved']} "
        f"gpu={gpu.get('memory_mib', 0)}/{gpu.get('memory_total_mib', 0)}MiB "
        f"util={gpu.get('utilization_pct', 0)}% running={vllm.get('requests_running')} "
        f"waiting={vllm.get('requests_waiting')} kv={vllm.get('kv_cache_usage')}"
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_peaks(status_path: Path) -> dict[str, float]:
    return (read_json(status_path).get("gpu") or {}).get("peaks") or {}


def latest_result_mtime(output_dir: Path) -> float:
    mtimes = [path.stat().st_mtime for path in output_dir.glob("sample_*/tasks/*/*/sample_result.json")]
    return max(mtimes, default=0.0)


def file_mtime(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromtimestamp(Path(value).stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def process_count(pattern: str) -> int:
    try:
        output = subprocess.check_output(["pgrep", "-af", pattern], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in output.splitlines() if "monitor_pass_at_k_experiment.py" not in line)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
