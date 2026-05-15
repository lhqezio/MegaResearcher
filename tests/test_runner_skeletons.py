"""Tests confirming each skeleton runner returns a schema-valid failed result.

Run from plugin root:
    python3 tests/test_runner_skeletons.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.runners._base import validate_result
from lib.runners.specs.runner import run as specs_run
from lib.runners.abgen.runner import run as abgen_run
from lib.runners.citeme.runner import run as citeme_run
from lib.runners.limitgen.runner import run as limitgen_run
from lib.runners.paperwrite_bench.runner import run as paperwrite_bench_run


def _check_skeleton(run_fn, expected_substrate: str):
    result = run_fn({"hypothesis_id": "test"})
    ok, errs = validate_result(result)
    assert ok, errs
    assert result["status"] == "failed"
    assert result["failure_code"] == "runner_not_implemented"
    assert result["substrate"] == expected_substrate


def test_specs_skeleton():
    _check_skeleton(specs_run, "SPECS-Review-Benchmark")


def test_abgen_skeleton():
    _check_skeleton(abgen_run, "AbGen")


def test_citeme_skeleton():
    _check_skeleton(citeme_run, "CiteME")


def test_limitgen_skeleton():
    _check_skeleton(limitgen_run, "LimitGen")


def test_paperwrite_bench_skeleton():
    _check_skeleton(paperwrite_bench_run, "PaperWrite-Bench")


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
    print("All skeleton runner tests pass.")
