"""Tests for paper-chain pre-flight checks.

Run from plugin root:
    python3 tests/test_preflight.py
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.preflight import preflight_check


def _make_run(novelty_target: str | None, with_output: bool, with_eval_designers: int) -> Path:
    """Create a temporary run dir matching the swarm-state.yaml shape."""
    run = Path(tempfile.mkdtemp(prefix="run-"))
    if with_output:
        (run / "output.md").write_text("# Research direction\n")
    if novelty_target is not None:
        (run / "swarm-state.yaml").write_text(f"novelty_target: {novelty_target}\n")
    for i in range(with_eval_designers):
        d = run / f"eval-designer-S{i+1}"
        d.mkdir()
        (d / "output.md").write_text(f"# Eval design {i+1}\n")
    return run


def test_happy_path():
    run = _make_run("hypothesis", with_output=True, with_eval_designers=3)
    ok, reason = preflight_check(run)
    assert ok, f"Expected OK, got refusal: {reason}"


def test_missing_output_md():
    run = _make_run("hypothesis", with_output=False, with_eval_designers=3)
    ok, reason = preflight_check(run)
    assert not ok
    assert "output.md" in reason, f"Expected reason to name output.md; got: {reason}"


def test_missing_swarm_state():
    run = _make_run(novelty_target=None, with_output=True, with_eval_designers=3)
    ok, reason = preflight_check(run)
    assert not ok
    assert "swarm-state" in reason


def test_wrong_novelty_target_gap_finding():
    run = _make_run("gap-finding", with_output=True, with_eval_designers=0)
    ok, reason = preflight_check(run)
    assert not ok
    assert "hypothesis" in reason and "gap-finding" in reason


def test_no_eval_designer_outputs():
    run = _make_run("hypothesis", with_output=True, with_eval_designers=0)
    ok, reason = preflight_check(run)
    assert not ok
    assert "eval-designer" in reason


def test_preflight_warns_about_vercel_token_when_paper():
    """When --paper is set and VERCEL_TOKEN absent, preflight returns ok=True
    with a non-empty warnings list. The presence of a warning does not block;
    it just informs the user that Phase 6.5 will fail if it tries to use the
    Vercel backend."""
    import os
    run = _make_run("hypothesis", with_output=True, with_eval_designers=3)
    saved = os.environ.pop("VERCEL_TOKEN", None)
    try:
        from lib.paper_chain.preflight import preflight_check_with_paper
        ok, reason, warnings = preflight_check_with_paper(run, paper_mode=True)
        assert ok
        assert any("VERCEL_TOKEN" in w for w in warnings)
    finally:
        if saved is not None:
            os.environ["VERCEL_TOKEN"] = saved


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
    print("All preflight tests pass.")
