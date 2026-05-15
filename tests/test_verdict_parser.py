"""Tests for verdict-line parsing from review files.

Run from plugin root:
    python3 tests/test_verdict_parser.py
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.verdict import parse_verdict


def _write(text: str) -> Path:
    fh = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    fh.write(text)
    fh.close()
    return Path(fh.name)


def test_approve():
    p = _write("# Review\n\nSummary...\n\nVERDICT: APPROVE\n")
    assert parse_verdict(p) == "APPROVE", "Expected APPROVE"


def test_revise():
    p = _write("# Review\nVERDICT: REVISE\n")
    assert parse_verdict(p) == "REVISE", "Expected REVISE"


def test_kill():
    p = _write("# Review\nVERDICT: KILL\n")
    assert parse_verdict(p) == "KILL", "Expected KILL"


def test_verdict_must_be_last_nonblank_line():
    # Verdict line not at end → still parsed (we scan, not strict-last)
    p = _write("# Review\nVERDICT: APPROVE\n\nSome trailing notes.\n")
    assert parse_verdict(p) == "APPROVE"


def test_no_verdict():
    p = _write("# Review\n\nNo verdict here.\n")
    assert parse_verdict(p) is None


def test_malformed_verdict():
    p = _write("# Review\nVERDICT: MAYBE\n")
    assert parse_verdict(p) is None, "Unknown verdict word should return None"


def test_case_sensitivity():
    p = _write("# Review\nverdict: approve\n")
    assert parse_verdict(p) is None, "Lowercase 'verdict:' should not match"


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
    print("All verdict tests pass.")
