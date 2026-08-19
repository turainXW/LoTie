from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "round_robin_model_proxy.py"
SPEC = importlib.util.spec_from_file_location("round_robin_model_proxy", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_backend_pool_rotates_starting_backend() -> None:
    pool = MODULE.BackendPool(["http://one/", "http://two", "http://three/"])

    assert pool.next_order() == ["http://one", "http://two", "http://three"]
    assert pool.next_order() == ["http://two", "http://three", "http://one"]
    assert pool.next_order() == ["http://three", "http://one", "http://two"]


def test_backend_pool_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="at least one backend"):
        MODULE.BackendPool([])


def test_normalize_backend_removes_trailing_slashes() -> None:
    assert MODULE.normalize_backend("http://127.0.0.1:8000///") == "http://127.0.0.1:8000"
