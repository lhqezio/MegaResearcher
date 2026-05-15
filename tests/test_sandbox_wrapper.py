"""Tests for the Vercel Sandbox wrapper with FakeSandboxBackend.

Run from plugin root:
    python3 tests/test_sandbox_wrapper.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.sandbox import (
    set_backend,
    spin_up,
    execute,
    tear_down,
    cost_so_far,
    BudgetBreach,
    FakeSandboxBackend,
    ExecutionResult,
)


def test_spin_up_returns_sandbox_id():
    set_backend(FakeSandboxBackend(canned_id="sb_abc"))
    sid = spin_up(image="python:3.11", timeout_seconds=60, budget_usd=5.0)
    assert sid == "sb_abc"


def test_execute_returns_result():
    set_backend(FakeSandboxBackend(canned_id="sb_x", canned_stdout="hello\n", canned_exit=0))
    sid = spin_up(image="python:3.11", timeout_seconds=60, budget_usd=5.0)
    result = execute(sid, "echo hello")
    assert isinstance(result, ExecutionResult)
    assert result.stdout == "hello\n"
    assert result.exit_code == 0


def test_tear_down_marks_torn():
    backend = FakeSandboxBackend(canned_id="sb_y")
    set_backend(backend)
    sid = spin_up(image="x", timeout_seconds=60, budget_usd=5.0)
    tear_down(sid)
    assert backend.torn_down == ["sb_y"]


def test_cost_so_far_reads_backend():
    backend = FakeSandboxBackend(canned_id="sb_z", canned_cost=1.23)
    set_backend(backend)
    sid = spin_up(image="x", timeout_seconds=60, budget_usd=5.0)
    assert cost_so_far(sid) == 1.23


def test_budget_breach_during_execute():
    backend = FakeSandboxBackend(canned_id="sb_q", canned_cost=10.0)
    set_backend(backend)
    sid = spin_up(image="x", timeout_seconds=60, budget_usd=5.0)
    try:
        execute(sid, "expensive")
        assert False, "expected BudgetBreach"
    except BudgetBreach as e:
        assert "5.0" in str(e)


def test_timeout_yields_exit_124():
    backend = FakeSandboxBackend(canned_id="sb_t", canned_exit=124)
    set_backend(backend)
    sid = spin_up(image="x", timeout_seconds=60, budget_usd=5.0)
    result = execute(sid, "sleep 999")
    assert result.exit_code == 124


if __name__ == "__main__":
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures.append((name, str(e)))
                print(f"FAIL {name}: {e}")
    if failures:
        sys.exit(1)
    print("All sandbox wrapper tests pass.")
