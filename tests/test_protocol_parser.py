"""Tests for eval-designer protocol parsing.

Run from plugin root:
    python3 tests/test_protocol_parser.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.protocol_parser import parse_protocol

FIXTURES = PLUGIN_ROOT / "tests" / "fixtures" / "protocols"


def test_parse_specs_protocol():
    result = parse_protocol(FIXTURES / "specs_protocol.md")
    assert result["substrate"] == "SPECS-Review-Benchmark", f"got {result['substrate']!r}"
    assert result["sample_size"] == 22
    assert result["seed"] == 42


def test_parse_baselines_list():
    result = parse_protocol(FIXTURES / "specs_protocol.md")
    assert "stage-matched same-family 2-stage" in result["baselines"][0]


def test_parse_metric():
    result = parse_protocol(FIXTURES / "specs_protocol.md")
    assert any("flaw-detection" in m for m in result["metrics"]), result["metrics"]


def test_parse_decision_rule():
    result = parse_protocol(FIXTURES / "specs_protocol.md")
    assert len(result["decision_rules"]) >= 1
    assert "0.05" in str(result["decision_rules"][0])


def test_parse_malformed_returns_empty():
    result = parse_protocol(FIXTURES / "malformed.md")
    assert result == {}, f"expected empty dict, got {result!r}"


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
    print("All protocol parser tests pass.")
