"""Tests for Phase 9 finalize logic.

Run from plugin root:
    python3 tests/test_finalize.py
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.finalize import finalize_paper


def _setup_paper_dir(latest_draft: str) -> Path:
    """Build a paper/ dir with draft-v1.md, draft-v2.md (if v2), review-v1.md, log."""
    run = Path(tempfile.mkdtemp(prefix="run-"))
    paper = run / "paper"
    paper.mkdir()
    (paper / "draft-v1.md").write_text("# Draft v1\n\nContent.\n")
    if latest_draft == "v2":
        (paper / "draft-v2.md").write_text("# Draft v2\n\nRevised.\n")
    (paper / "review-v1.md").write_text("# Review v1\n\nVERDICT: REVISE\n")
    (paper / "revision-log.jsonl").write_text(
        '{"round":1,"review_point_index":0,"addressed":true,'
        '"change_summary":"fixed W1","line_range_modified":[10,15]}\n'
    )
    return paper


def test_finalize_with_v1_only():
    paper = _setup_paper_dir(latest_draft="v1")
    out = finalize_paper(paper, final_verdict="APPROVE")
    assert out == paper / "paper.md"
    assert out.exists()
    assert "Draft v1" in out.read_text()
    history = paper / "paper-history.md"
    assert history.exists()
    assert "Review v1" in history.read_text()


def test_finalize_with_v2():
    paper = _setup_paper_dir(latest_draft="v2")
    out = finalize_paper(paper, final_verdict="APPROVE")
    assert "Draft v2" in out.read_text(), "paper.md must point at latest draft"


def test_finalize_includes_revision_log_in_history():
    paper = _setup_paper_dir(latest_draft="v2")
    finalize_paper(paper, final_verdict="APPROVE")
    history = (paper / "paper-history.md").read_text()
    assert "fixed W1" in history, "revision-log entries must appear in history"


def test_finalize_records_final_verdict():
    paper = _setup_paper_dir(latest_draft="v1")
    finalize_paper(paper, final_verdict="APPROVE")
    history = (paper / "paper-history.md").read_text()
    assert "Final verdict: APPROVE" in history


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
    print("All finalize tests pass.")
