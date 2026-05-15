"""Tests for Runner base class + result schema.

Run from plugin root:
    python3 tests/test_runner_base.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.runners._base import (
    Runner,
    validate_result,
    make_failed_result,
    REQUIRED_FIELDS,
)


def test_required_fields_present():
    expected = {
        "hypothesis_id", "substrate", "metric_name",
        "baseline_value", "treatment_value",
        "p_value", "ci_low", "ci_high", "n", "seed",
        "runtime_seconds", "cost_usd",
        "status", "failure_code", "failure_message",
    }
    assert set(REQUIRED_FIELDS) == expected


def test_validate_complete_result():
    result = {f: None for f in REQUIRED_FIELDS}
    result["status"] = "completed"
    ok, errs = validate_result(result)
    assert ok, errs


def test_validate_missing_field():
    result = {f: None for f in REQUIRED_FIELDS}
    del result["substrate"]
    result["status"] = "completed"
    ok, errs = validate_result(result)
    assert not ok
    assert any("substrate" in e for e in errs)


def test_validate_bad_status():
    result = {f: None for f in REQUIRED_FIELDS}
    result["status"] = "maybe"
    ok, errs = validate_result(result)
    assert not ok


def test_make_failed_result_satisfies_schema():
    result = make_failed_result(
        hypothesis_id="S1",
        substrate="SPECS-Review-Benchmark",
        failure_code="runner_not_implemented",
        failure_message="skeleton",
    )
    ok, errs = validate_result(result)
    assert ok, errs
    assert result["status"] == "failed"
    assert result["failure_code"] == "runner_not_implemented"


def test_runner_subclass_must_define_run():
    class IncompleteRunner(Runner):
        substrate = "x"
    try:
        IncompleteRunner().run({"hypothesis_id": "S1"})
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


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
    print("All runner base tests pass.")
