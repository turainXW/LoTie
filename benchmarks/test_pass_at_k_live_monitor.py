from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "monitor_pass_at_k_experiment", ROOT / "scripts" / "monitor_pass_at_k_experiment.py"
)
assert SPEC and SPEC.loader
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def test_vllm_metrics_parses_prometheus_gauges(monkeypatch) -> None:
    payload = b'''\
vllm:num_requests_running{engine="0"} 3.0
vllm:num_requests_waiting{engine="0"} 1.0
vllm:kv_cache_usage_perc{engine="0"} 0.125
'''

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return payload

    monkeypatch.setattr(MONITOR, "urlopen", lambda *_args, **_kwargs: Response())
    assert MONITOR.vllm_metrics("http://example.test/metrics") == {
        "requests_running": 3.0,
        "requests_waiting": 1.0,
        "kv_cache_usage": 0.125,
    }


def test_compact_line_contains_progress_and_runtime_metrics() -> None:
    line = MONITOR.compact_line(
        {
            "timestamp": "now",
            "label": "epoch1",
            "progress": {"valid": 4, "target": 12, "resolved": 3},
            "gpu": {"memory_mib": 10, "memory_total_mib": 20, "utilization_pct": 75},
            "vllm": {"requests_running": 4.0, "requests_waiting": 0.0, "kv_cache_usage": 0.2},
        }
    )
    assert "progress=4/12" in line
    assert "running=4.0" in line
