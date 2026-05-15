"""Tests for experiment.py orchestration.

Run from plugin root:
    python3 tests/test_experiment_orchestrator.py
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.sandbox import set_backend, FakeSandboxBackend
from lib.paper_chain.experiment import (
    select_runner,
    dispatch_experiment,
)


def _make_protocol(substrate: str) -> Path:
    text = f"""# Eval design — test

## Pre-registered settings

- Substrate: {substrate}
- Sample size: 5
- Seed: 42
- Baselines: baseline-A
- Metric: detection-rate
- Decision rule: lift >= 0.05
"""
    fh = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    fh.write(text)
    fh.close()
    return Path(fh.name)


def test_select_runner_known_substrate():
    runner = select_runner("SPECS-Review-Benchmark")
    assert runner is not None
    assert runner.SUBSTRATE == "SPECS-Review-Benchmark"


def test_select_runner_unknown_substrate():
    runner = select_runner("UnknownBench")
    assert runner is None


def test_dispatch_experiment_writes_three_artifacts():
    set_backend(FakeSandboxBackend(canned_id="sb_x", canned_stdout="ignored"))
    output_dir = Path(tempfile.mkdtemp(prefix="exp-"))
    protocol = _make_protocol("SPECS-Review-Benchmark")
    results_path = dispatch_experiment(
        hypothesis_id="S1",
        protocol_path=protocol,
        output_dir=output_dir,
        sandbox_budget_usd=5.0,
        api_budget_usd=5.0,
    )
    assert results_path == output_dir / "results.json"
    assert results_path.exists()
    assert (output_dir / "repro.yaml").exists()
    assert (output_dir / "runner-output.log").exists()


def test_dispatch_records_failed_for_skeleton():
    """Skeleton runner returns failed_runner_not_implemented — orchestrator
    should write that result faithfully without retry."""
    set_backend(FakeSandboxBackend(canned_id="sb_x"))
    output_dir = Path(tempfile.mkdtemp(prefix="exp-"))
    protocol = _make_protocol("SPECS-Review-Benchmark")
    results_path = dispatch_experiment(
        hypothesis_id="S1",
        protocol_path=protocol,
        output_dir=output_dir,
        sandbox_budget_usd=5.0,
        api_budget_usd=5.0,
    )
    result = json.loads(results_path.read_text())
    assert result["status"] == "failed"
    assert result["failure_code"] == "runner_not_implemented"


def test_dispatch_unsupported_substrate_writes_failed():
    set_backend(FakeSandboxBackend(canned_id="sb_x"))
    output_dir = Path(tempfile.mkdtemp(prefix="exp-"))
    protocol = _make_protocol("UnknownBench")
    results_path = dispatch_experiment(
        hypothesis_id="S1",
        protocol_path=protocol,
        output_dir=output_dir,
        sandbox_budget_usd=5.0,
        api_budget_usd=5.0,
    )
    result = json.loads(results_path.read_text())
    assert result["status"] == "failed"
    assert result["failure_code"] == "unsupported_substrate"


def test_dispatch_malformed_protocol_writes_failed():
    set_backend(FakeSandboxBackend(canned_id="sb_x"))
    output_dir = Path(tempfile.mkdtemp(prefix="exp-"))
    bad_protocol = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    bad_protocol.write("# Random text with no structure\n")
    bad_protocol.close()
    results_path = dispatch_experiment(
        hypothesis_id="S1",
        protocol_path=Path(bad_protocol.name),
        output_dir=output_dir,
        sandbox_budget_usd=5.0,
        api_budget_usd=5.0,
    )
    result = json.loads(results_path.read_text())
    assert result["status"] == "failed"
    assert result["failure_code"] == "failed_parse"


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
    print("All experiment orchestrator tests pass.")
