"""Tests for runaway-revision regression detection.

Run from plugin root:
    python3 tests/test_regression.py
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.regression import (
    extract_weaknesses,
    detect_regression,
)


REVIEW_V1 = """# Review v1
## Strengths
- Good idea.

## Weaknesses
- W1: Insufficient ablation coverage.
- W2: Citation for claim X does not resolve.
- W3: Method section unclear about step 3.

## Suggested Revisions
...

VERDICT: REVISE
"""

REVIEW_V2_ALL_CLOSED_NEW_PROBLEMS = """# Review v2
## Strengths
- Improved.

## Weaknesses
- W4: New citation Y is also unresolved.
- W5: Ablation table has off-by-one error.
- W6: Discussion contradicts results.
- W7: New related-work section misattributes finding.

VERDICT: REVISE
"""

REVIEW_V2_PARTIAL_CLOSE = """# Review v2
## Weaknesses
- W2: Citation still unresolved (carried over).
- W4: New small typo in abstract.

VERDICT: REVISE
"""


def _write(text: str) -> Path:
    fh = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    fh.write(text)
    fh.close()
    return Path(fh.name)


def test_extract_weaknesses_basic():
    p = _write(REVIEW_V1)
    ws = extract_weaknesses(p)
    assert len(ws) == 3, f"Expected 3 weaknesses, got {len(ws)}: {ws}"
    assert ws[0].startswith("W1:"), ws[0]


def test_extract_weaknesses_empty_section():
    p = _write("## Weaknesses\n\nVERDICT: APPROVE\n")
    ws = extract_weaknesses(p)
    assert ws == []


def test_regression_fires_when_new_outnumber_closed():
    v1 = _write(REVIEW_V1)
    v2 = _write(REVIEW_V2_ALL_CLOSED_NEW_PROBLEMS)
    flagged, closed_count, new_count = detect_regression(v1, v2)
    # v1 had 3, v2 has 4 NEW; closed = 3 (all v1 weaknesses gone), new = 4 — regression
    assert flagged, f"Expected regression flag: closed={closed_count}, new={new_count}"
    assert new_count == 4
    assert closed_count == 3


def test_regression_does_not_fire_on_partial_close():
    v1 = _write(REVIEW_V1)
    v2 = _write(REVIEW_V2_PARTIAL_CLOSE)
    # v1 had 3; v2 has 2 (W2 carried, W4 new). closed=2 (W1, W3), new=1 (W4). No regression.
    flagged, closed_count, new_count = detect_regression(v1, v2)
    assert not flagged
    assert closed_count == 2
    assert new_count == 1


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
    print("All regression tests pass.")
