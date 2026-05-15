"""Tests for the paper/ scaffold helper.

Run from plugin root:
    python3 tests/test_scaffold.py
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.scaffold import scaffold_paper_dir


def test_creates_paper_subdir():
    run = Path(tempfile.mkdtemp(prefix="run-"))
    paper = scaffold_paper_dir(run)
    assert paper == run / "paper"
    assert paper.is_dir()


def test_creates_revision_log_jsonl():
    run = Path(tempfile.mkdtemp(prefix="run-"))
    paper = scaffold_paper_dir(run)
    log = paper / "revision-log.jsonl"
    assert log.exists() and log.read_text() == "", "revision-log.jsonl should be empty file"


def test_idempotent():
    run = Path(tempfile.mkdtemp(prefix="run-"))
    p1 = scaffold_paper_dir(run)
    (p1 / "draft-v1.md").write_text("# draft")
    p2 = scaffold_paper_dir(run)  # safe to re-run
    assert p1 == p2
    assert (p1 / "draft-v1.md").exists(), "Idempotent scaffold must not destroy existing content"


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
    print("All scaffold tests pass.")
