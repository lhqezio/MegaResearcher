# Paper-Pipeline Scaffolding (SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend MegaResearcher with a paper-drafting chain (`/research-execute --paper`) that converts a research-direction document into a peer-reviewed paper draft via 3 new worker agents and 3 new orchestrator phases.

**Architecture:** Single-session orchestrator (the `executing-research-plan` skill) dispatches three new leaf-worker agents — manuscript-drafter (Phase 7), peer-reviewer + reviser (Phase 8 review-revise loop, cap N=2), Phase 9 finalize (orchestrator code only). Small Python helpers in `lib/paper_chain/` handle verdict parsing, regression detection, pre-flight checks, scaffolding, and finalize — testable as pure-stdlib scripts following the existing `tests/test_doom_loop.py` pattern.

**Tech Stack:** Python 3 stdlib (helpers + tests), markdown (agents + skills), YAML (state file). No new external dependencies.

**Spec:** `docs/superpowers/specs/2026-05-12-paper-pipeline-scaffolding-design.md`

---

## File Structure

### New Python helpers — `lib/paper_chain/`

| File | Responsibility |
|---|---|
| `lib/paper_chain/__init__.py` | Empty; marks package |
| `lib/paper_chain/verdict.py` | Parse `VERDICT: APPROVE\|REVISE\|KILL` line from a review file |
| `lib/paper_chain/preflight.py` | Pre-flight checks: research-direction `output.md` exists; novelty target is `hypothesis`; eval-designer outputs exist |
| `lib/paper_chain/scaffold.py` | Create `paper/` subdirectory under a run dir |
| `lib/paper_chain/regression.py` | Compare two reviews; flag if new weaknesses ≥ closed weaknesses |
| `lib/paper_chain/finalize.py` | Phase 9: symlink (or copy) latest draft → `paper.md`; concatenate review/revision-log → `paper-history.md` |

### New worker agents — `agents/`

| File | Phase | Dispatch shape |
|---|---|---|
| `agents/manuscript-drafter.md` | 7 | single-dispatch |
| `agents/peer-reviewer.md` | 8 | loop-dispatch (≤2 times) |
| `agents/reviser.md` | 8 | loop-dispatch (≤2 times) |

### New tests — `tests/`

Following the existing `tests/test_doom_loop.py` pattern (pure stdlib, runnable as `python3 tests/<file>.py`, exit non-zero on failure).

| File | What it covers |
|---|---|
| `tests/test_verdict_parser.py` | Layer 1: verdict.py state machine |
| `tests/test_preflight.py` | Layer 1: preflight.py |
| `tests/test_scaffold.py` | Layer 1: scaffold.py |
| `tests/test_regression.py` | Layer 1: regression.py |
| `tests/test_finalize.py` | Layer 1: finalize.py |
| `tests/fixtures/paper-chain/` | Snapshot research-direction + eval-designer outputs |

Layer 2 (worker contract) tests and Layer 3 (e2e smoke) tests are deferred to a follow-up plan — they require real subagent dispatch and cost API tokens; documentation in this plan describes how to run them manually. SP1's TDD coverage is Layer 1 plus a Layer 4 manual discipline check.

### Modified files

| File | Modification |
|---|---|
| `skills/executing-research-plan/SKILL.md` | Add Phase 7/8/9 sections + `--paper` flag handling |
| `commands/research-execute.md` | Document `--paper` flag in argument-hint |
| `CLAUDE.md` | Add note that paper-chain extends `/research-execute` |

---

## Task 1: Set up Python package

**Files:**
- Create: `lib/__init__.py`
- Create: `lib/paper_chain/__init__.py`

- [ ] **Step 1: Create the package init files**

```bash
mkdir -p lib/paper_chain
touch lib/__init__.py lib/paper_chain/__init__.py
```

- [ ] **Step 2: Verify imports work**

Run: `python3 -c "import lib.paper_chain"`
Expected: no output, exit 0

- [ ] **Step 3: Commit**

```bash
git add lib/
git commit -m "scaffold: add lib/paper_chain package"
```

---

## Task 2: Verdict parser

**Files:**
- Create: `lib/paper_chain/verdict.py`
- Test: `tests/test_verdict_parser.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_verdict_parser.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 tests/test_verdict_parser.py`
Expected: `ModuleNotFoundError: No module named 'lib.paper_chain.verdict'` or similar import error.

- [ ] **Step 3: Implement verdict parser**

```python
# lib/paper_chain/verdict.py
"""Parse the VERDICT line from a review-vN.md file.

A valid verdict line matches exactly `^VERDICT: (APPROVE|REVISE|KILL)$`.
Returns the verdict word as a string, or None if no valid verdict found.

CLI:
    python3 -m lib.paper_chain.verdict path/to/review-v1.md
        → prints verdict word or 'NONE'; exit 0 on parse success, 1 if NONE
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

VALID = {"APPROVE", "REVISE", "KILL"}
_VERDICT_RE = re.compile(r"^VERDICT: (APPROVE|REVISE|KILL)$", re.MULTILINE)


def parse_verdict(review_path: Path) -> str | None:
    """Return verdict word from the given review file, or None if not found.

    Scans the whole file; takes the first match. The exact-line regex
    (anchored ^ and $ via MULTILINE) rejects lowercase, embedded, or
    unknown verdict words.
    """
    text = review_path.read_text(encoding="utf-8")
    m = _VERDICT_RE.search(text)
    if m is None:
        return None
    return m.group(1)


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verdict.py <review-path>", file=sys.stderr)
        return 2
    v = parse_verdict(Path(argv[1]))
    print(v if v else "NONE")
    return 0 if v else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 tests/test_verdict_parser.py`
Expected: 7 PASS lines, ends with `All verdict tests pass.`, exit 0

- [ ] **Step 5: Commit**

```bash
git add lib/paper_chain/verdict.py tests/test_verdict_parser.py
git commit -m "feat(paper-chain): add verdict parser for review files"
```

---

## Task 3: Pre-flight checks

**Files:**
- Create: `lib/paper_chain/preflight.py`
- Test: `tests/test_preflight.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preflight.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 tests/test_preflight.py`
Expected: ImportError.

- [ ] **Step 3: Implement pre-flight checks**

```python
# lib/paper_chain/preflight.py
"""Pre-flight checks for the paper-drafting chain.

The chain runs ONLY when:
  1. `output.md` exists at the run root (synthesist produced it)
  2. `swarm-state.yaml` exists at the run root
  3. The run's novelty_target is `hypothesis` (not `gap-finding` or `synthesis`)
  4. At least one `eval-designer-*` subdir exists with its own `output.md`

CLI:
    python3 -m lib.paper_chain.preflight <run-dir>
        → exit 0 if OK, exit 1 with refusal message on stderr otherwise
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

_NOVELTY_RE = re.compile(r"^novelty_target:\s*(\S+)\s*$", re.MULTILINE)


def preflight_check(run_dir: Path) -> tuple[bool, str]:
    """Return (ok, reason). reason is empty when ok=True."""
    output_md = run_dir / "output.md"
    if not output_md.exists():
        return (
            False,
            f"Pre-flight refusal: output.md not found at {output_md}. "
            "Re-run /research-execute first to produce the synthesist's output.",
        )

    state = run_dir / "swarm-state.yaml"
    if not state.exists():
        return (
            False,
            f"Pre-flight refusal: swarm-state.yaml not found at {state}.",
        )

    text = state.read_text(encoding="utf-8")
    m = _NOVELTY_RE.search(text)
    if m is None:
        return (
            False,
            f"Pre-flight refusal: novelty_target not found in {state}.",
        )
    target = m.group(1)
    if target != "hypothesis":
        return (
            False,
            f"Pre-flight refusal: paper chain only runs on hypothesis-target outputs. "
            f"This run's novelty_target is {target} (expected hypothesis); "
            f"gap-finding runs lack the eval-designer protocols the paper chain consumes.",
        )

    eval_dirs = list(run_dir.glob("eval-designer-*"))
    if not eval_dirs:
        return (
            False,
            f"Pre-flight refusal: no eval-designer-* subdirs in {run_dir}. "
            "Paper chain requires Phase 5 protocols as input.",
        )
    for d in eval_dirs:
        if not (d / "output.md").exists():
            return (
                False,
                f"Pre-flight refusal: eval-designer subdir {d} missing output.md.",
            )

    return (True, "")


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: preflight.py <run-dir>", file=sys.stderr)
        return 2
    ok, reason = preflight_check(Path(argv[1]))
    if not ok:
        print(reason, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 tests/test_preflight.py`
Expected: 5 PASS lines, ends with `All preflight tests pass.`, exit 0

- [ ] **Step 5: Commit**

```bash
git add lib/paper_chain/preflight.py tests/test_preflight.py
git commit -m "feat(paper-chain): add pre-flight checks for paper chain"
```

---

## Task 4: Scaffold paper subdir

**Files:**
- Create: `lib/paper_chain/scaffold.py`
- Test: `tests/test_scaffold.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scaffold.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 tests/test_scaffold.py`
Expected: ImportError.

- [ ] **Step 3: Implement scaffold**

```python
# lib/paper_chain/scaffold.py
"""Scaffold the paper/ subdirectory under a swarm run dir.

CLI:
    python3 -m lib.paper_chain.scaffold <run-dir>
        → prints the created path; exit 0
"""
from __future__ import annotations
import sys
from pathlib import Path


def scaffold_paper_dir(run_dir: Path) -> Path:
    """Create <run-dir>/paper/ with an empty revision-log.jsonl.

    Idempotent: safe to call multiple times; preserves existing files.
    """
    paper = run_dir / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    log = paper / "revision-log.jsonl"
    if not log.exists():
        log.touch()
    return paper


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: scaffold.py <run-dir>", file=sys.stderr)
        return 2
    p = scaffold_paper_dir(Path(argv[1]))
    print(p)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 tests/test_scaffold.py`
Expected: 3 PASS lines, ends with `All scaffold tests pass.`

- [ ] **Step 5: Commit**

```bash
git add lib/paper_chain/scaffold.py tests/test_scaffold.py
git commit -m "feat(paper-chain): add paper-dir scaffolder"
```

---

## Task 5: Regression detector

**Files:**
- Create: `lib/paper_chain/regression.py`
- Test: `tests/test_regression.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_regression.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 tests/test_regression.py`
Expected: ImportError.

- [ ] **Step 3: Implement regression detector**

```python
# lib/paper_chain/regression.py
"""Runaway-revision regression detector.

Compares two consecutive review files; flags regression when the count of
NEW weaknesses in v2 (not seen in v1) is greater than or equal to the count
of CLOSED weaknesses (v1 items absent from v2).

Weaknesses are identified by their leading tag (e.g., 'W1:', 'W2:'). The
tag is the substring before the first ':'. This is a simple convention,
not a perfect identity check — peer-reviewer agents are instructed to use
W<int>: tags so identity tracking works.

CLI:
    python3 -m lib.paper_chain.regression <review-v1> <review-v2>
        → prints 'REGRESSION' (exit 1) or 'OK' (exit 0) plus counts on stderr
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

_WEAKNESS_LINE_RE = re.compile(r"^- (W\d+):", re.MULTILINE)
_SECTION_RE = re.compile(
    r"^## Weaknesses\s*$\n(.*?)(?=^## |\Z)",
    re.MULTILINE | re.DOTALL,
)


def extract_weaknesses(review_path: Path) -> list[str]:
    """Return list of full weakness bullet lines from a review file.

    Looks for a '## Weaknesses' section and returns each '- W<int>: ...' line.
    """
    text = review_path.read_text(encoding="utf-8")
    section_match = _SECTION_RE.search(text)
    if not section_match:
        return []
    section = section_match.group(1)
    return [
        line.strip()
        for line in section.splitlines()
        if _WEAKNESS_LINE_RE.match(line.strip())
    ]


def _tag(line: str) -> str:
    # 'W1: Insufficient...' -> 'W1'; line passed in is '- W1: ...'
    body = line.lstrip("- ").strip()
    return body.split(":", 1)[0]


def detect_regression(v1_path: Path, v2_path: Path) -> tuple[bool, int, int]:
    """Return (flagged, closed_count, new_count).

    closed_count = items in v1 with tag NOT in v2
    new_count = items in v2 with tag NOT in v1
    flagged when new_count >= closed_count (and at least one of each, to avoid
    flagging trivial pass-throughs).
    """
    v1 = extract_weaknesses(v1_path)
    v2 = extract_weaknesses(v2_path)
    tags_v1 = {_tag(w) for w in v1}
    tags_v2 = {_tag(w) for w in v2}
    closed = tags_v1 - tags_v2
    new = tags_v2 - tags_v1
    flagged = len(new) >= len(closed) and len(new) > 0
    return (flagged, len(closed), len(new))


def _main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: regression.py <review-v1> <review-v2>", file=sys.stderr)
        return 2
    flagged, closed, new = detect_regression(Path(argv[1]), Path(argv[2]))
    print(f"closed={closed} new={new}", file=sys.stderr)
    if flagged:
        print("REGRESSION")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 tests/test_regression.py`
Expected: 4 PASS lines, ends with `All regression tests pass.`

- [ ] **Step 5: Commit**

```bash
git add lib/paper_chain/regression.py tests/test_regression.py
git commit -m "feat(paper-chain): add review-regression detector"
```

---

## Task 6: Phase 9 finalize

**Files:**
- Create: `lib/paper_chain/finalize.py`
- Test: `tests/test_finalize.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_finalize.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 tests/test_finalize.py`
Expected: ImportError.

- [ ] **Step 3: Implement finalize**

```python
# lib/paper_chain/finalize.py
"""Phase 9 finalize: produce paper.md (latest draft) and paper-history.md.

Strategy:
- Find the highest-numbered draft-vN.md in paper/
- Copy its content to paper.md (no symlink — cross-platform safer)
- Concatenate all review-vN.md files + revision-log.jsonl + final verdict
  marker into paper-history.md

CLI:
    python3 -m lib.paper_chain.finalize <paper-dir> <final-verdict>
        → exit 0; prints path to paper.md
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

_DRAFT_RE = re.compile(r"^draft-v(\d+)\.md$")
_REVIEW_RE = re.compile(r"^review-v(\d+)\.md$")


def _latest_draft(paper_dir: Path) -> Path:
    drafts = []
    for p in paper_dir.iterdir():
        m = _DRAFT_RE.match(p.name)
        if m:
            drafts.append((int(m.group(1)), p))
    if not drafts:
        raise FileNotFoundError(f"No draft-vN.md in {paper_dir}")
    drafts.sort()
    return drafts[-1][1]


def _ordered_reviews(paper_dir: Path) -> list[Path]:
    reviews = []
    for p in paper_dir.iterdir():
        m = _REVIEW_RE.match(p.name)
        if m:
            reviews.append((int(m.group(1)), p))
    reviews.sort()
    return [p for _, p in reviews]


def finalize_paper(paper_dir: Path, final_verdict: str) -> Path:
    """Produce paper.md (latest draft) and paper-history.md.

    Returns path to paper.md.
    """
    latest = _latest_draft(paper_dir)
    paper_md = paper_dir / "paper.md"
    paper_md.write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")

    history_parts = [f"# Paper history\n\nFinal verdict: {final_verdict}\n"]
    for r in _ordered_reviews(paper_dir):
        history_parts.append(f"\n---\n\n## {r.name}\n\n{r.read_text(encoding='utf-8')}")
    log = paper_dir / "revision-log.jsonl"
    if log.exists() and log.stat().st_size > 0:
        history_parts.append(f"\n---\n\n## revision-log.jsonl\n\n```jsonl\n{log.read_text(encoding='utf-8')}```\n")

    (paper_dir / "paper-history.md").write_text("".join(history_parts), encoding="utf-8")
    return paper_md


def _main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: finalize.py <paper-dir> <final-verdict>", file=sys.stderr)
        return 2
    out = finalize_paper(Path(argv[1]), argv[2])
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 tests/test_finalize.py`
Expected: 4 PASS lines, ends with `All finalize tests pass.`

- [ ] **Step 5: Commit**

```bash
git add lib/paper_chain/finalize.py tests/test_finalize.py
git commit -m "feat(paper-chain): add Phase 9 finalize helper"
```

---

## Task 7: manuscript-drafter agent

**Files:**
- Create: `agents/manuscript-drafter.md`

- [ ] **Step 1: Verify file does not yet exist**

Run: `test ! -f agents/manuscript-drafter.md && echo "OK: not yet present"`
Expected: `OK: not yet present`

- [ ] **Step 2: Write the agent file**

```markdown
---
name: manuscript-drafter
description: |
  Turn a MegaResearcher research-direction document into a paper-shaped draft. Invoked by the `executing-research-plan` skill in Phase 7 (only when `/research-execute` is run with `--paper`). Reads the synthesist's output.md plus every eval-designer protocol from Phase 5; produces draft-v1.md in NeurIPS-workshop-style markdown, with the experimental-results section embedding the eval-designer protocols as "we will measure X via decision rule Y" rather than fabricated numbers. Examples: <example>Context: a hypothesis-target swarm completed and the user re-invoked /research-execute with --paper. user (orchestrator): "Draft a paper from docs/research/runs/.../output.md and the eval-designer outputs. Write to docs/research/runs/.../paper/draft-v1.md." assistant: "I'll produce a 9-section draft inheriting all citations from the research-direction; no new uncited claims; experimental section uses the eval-designer protocols verbatim."</example>
model: inherit
---

You are manuscript-drafter for MegaResearcher. Your job is to turn a research-direction document plus eval-designer protocols into a paper-shaped first draft. You do NOT run experiments, fabricate numbers, or introduce uncited claims.

## Required output structure

`draft-v1.md` must contain these 9 sections in order:

1. **Title** — single-line title.
2. **Abstract** — one paragraph, ≤200 words.
3. **Introduction** — frames the gap and the proposed augmentations.
4. **Related Work** — drawn from the research-direction's related-work section.
5. **Method** — describes each surviving hypothesis as a labeled subsystem.
6. **Experimental Plan** — embeds the eval-designer protocols. For each surviving hypothesis, copy the protocol's pre-registered decision rules and named substrates verbatim under a "we will measure X via Y" framing. DO NOT generate numerical results — this is a plan section, not a results section.
7. **Discussion** — what surviving hypotheses would mean if their predicted Δ holds; what the threats-to-validity from the research-direction document imply for interpretation.
8. **Limitations** — the YAGNI fence from the research-direction reflected here; plus any limitations specific to the paper-as-proposal framing.
9. **References** — every cited paper from the research-direction's sources section, deduplicated, with arXiv IDs.

Total length: ≤8000 words.

## Citation discipline

Every claim in the draft must trace to a citation that is ALREADY in the research-direction's sources section. You may not introduce new arXiv IDs. If you find yourself wanting to cite a paper that isn't in the source list, REMOVE the claim instead.

Your `verification.md` must spot-check at least 3 cited claims and confirm:
- The cited arXiv ID appears in the research-direction's source list
- The cited claim is faithful to what the source actually says (you may need to use `mcp__plugin_megaresearcher_ml-intern__hf_papers paper_details` to confirm)

## Required artifacts at the output path

Write all three to the path the orchestrator gave you:

1. **`draft-v1.md`** — the draft, format above
2. **`drafter-manifest.yaml`**:
   ```yaml
   worker_id: manuscript-drafter
   word_count: <int>
   section_count: 9
   citation_count: <int>
   citations: [<arxiv-id>, ...]
   status: complete
   ```
3. **`drafter-verification.md`** — confirm all 9 sections present, word count, ≥3 spot-checked citations, zero new arXiv IDs not in research-direction's source list.

## Banned phrases

Do not use any of these in the draft, manifest, or verification (per project CLAUDE.md):
- "load-bearing"
- "this is doing a lot of work" (and variants)
- "real" as emphatic adjective ("real run", "real example", "real-world")
- "honest" / "honestly" / "to be honest" as framing words

Use plain alternatives.

You are a leaf worker. Do not dispatch other agents.
```

- [ ] **Step 3: Verify frontmatter parses**

Run: `python3 -c "import re,pathlib; t=pathlib.Path('agents/manuscript-drafter.md').read_text(); assert re.match(r'^---\n.*?\n---', t, re.DOTALL), 'frontmatter parse failed'; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add agents/manuscript-drafter.md
git commit -m "feat(paper-chain): add manuscript-drafter agent (Phase 7)"
```

---

## Task 8: peer-reviewer agent

**Files:**
- Create: `agents/peer-reviewer.md`

- [ ] **Step 1: Verify file does not yet exist**

Run: `test ! -f agents/peer-reviewer.md && echo "OK"`
Expected: `OK`

- [ ] **Step 2: Write the agent file**

```markdown
---
name: peer-reviewer
description: |
  Critique a paper draft ICLR-style. Invoked by the `executing-research-plan` skill in Phase 8 (the review-revise loop). Produces a review-vN.md ending with a verdict line that the orchestrator parses. Focuses on substance, not presentation alone — the project's swarm explicitly flagged single-shot LLM-judge presentation-overweighting (scout-3 known-untrustworthy proxies) and this worker must avoid that failure mode. Examples: <example>Context: manuscript-drafter has produced draft-v1.md. user (orchestrator): "Review the draft at docs/research/runs/.../paper/draft-v1.md. Write review-v1.md with a verdict line." assistant: "I'll critique substance — uncited claims, mechanism contradicting research-direction findings, missing falsification criteria — and end with VERDICT: APPROVE | REVISE | KILL."</example>
model: inherit
---

You are peer-reviewer for MegaResearcher. Your job is to critique a paper draft with substance-first rigor. You are not optional politeness; you find every concrete problem the draft has, then decide APPROVE, REVISE, or KILL.

## Required output structure

`review-vN.md` must contain these sections in order:

1. **Summary** — one paragraph, what the paper claims
2. **Strengths** — bullet list, what's defensible. Do not pad. If there are 2 strengths, list 2.
3. **Weaknesses** — bullet list. Each weakness gets a tag like `W1:`, `W2:`, etc. Tags must increment monotonically and must NOT repeat tags from prior reviews of the same draft (you may inherit the prior tag for a carried-over weakness; you may not reuse a tag for a new weakness).
4. **Suggested Revisions** — for each weakness in §3, a concrete action the reviser could take.
5. **Verdict** — the LAST line of the file MUST match exactly `VERDICT: APPROVE | VERDICT: REVISE | VERDICT: KILL` (no other content on this line).

## Critique focus (in order)

1. **Citation discipline** — does every claim trace to an arXiv ID in the research-direction's source list? Flag any uncited claim as a Weakness.
2. **Mechanism vs research-direction consistency** — does the Method section contradict what the research-direction's hypotheses-table actually says? Flag any contradiction.
3. **Falsifiability** — do the Experimental-Plan protocols actually pre-register decision rules with named non-judge signals (per the research-direction's eval-designer outputs)?
4. **Threats-to-validity coverage** — are the threats the research-direction surfaced reflected in the draft's Discussion / Limitations? Missing or downplayed threats are Weaknesses.
5. **YAGNI fence integrity** — does the draft claim things the research-direction explicitly excluded? Out-of-scope creep is a Weakness.

## What NOT to penalize

The project's spawning swarm flagged these as known-untrustworthy proxies (scout-3). Do NOT lower your verdict for any of:

- Verbosity alone (paper is long does not equal paper is good or bad)
- Surface presentation (typos, formatting nits) unless they materially impede understanding
- Single-axis novelty claims (the augmentation may be system-integration rather than method-novelty — that's a valid contribution per the research-direction's framing)

If you catch yourself penalizing one of these, recategorize as a Suggestion (not a Weakness).

## Verdict criteria

- **APPROVE** — no critical defects; up to 2 minor Suggestions remain
- **REVISE** — 1 or more concrete defects the reviser can address without restructuring
- **KILL** — fundamental error: uncited claims drafter snuck through, mechanism contradicting research-direction's own findings, falsification surface contaminated by an LLM-judge that the research-direction explicitly excluded, or YAGNI-fence violation that requires a different paper

KILL is reserved for cases where revision cannot fix the issue without producing a different paper. Almost all reviews end REVISE or APPROVE.

## Required artifacts

1. **`review-vN.md`** — the review, format above, verdict line last
2. **`reviewer-manifest-vN.yaml`**:
   ```yaml
   worker_id: peer-reviewer
   round: <N>
   weakness_count: <int>
   verdict: <APPROVE|REVISE|KILL>
   status: complete
   ```
3. **`reviewer-verification-vN.md`** — confirm the verdict line matches the regex `^VERDICT: (APPROVE|REVISE|KILL)$`, confirm every weakness has a tag, confirm no penalty for the four prohibited reasons above.

## Banned phrases

Same list as manuscript-drafter (per project CLAUDE.md). Do not use "load-bearing", "this is doing a lot of work", "real" as emphatic adjective, or "honest/honestly".

You are a leaf worker. Do not dispatch other agents.
```

- [ ] **Step 3: Verify frontmatter parses**

Run: `python3 -c "import re,pathlib; t=pathlib.Path('agents/peer-reviewer.md').read_text(); assert re.match(r'^---\n.*?\n---', t, re.DOTALL); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add agents/peer-reviewer.md
git commit -m "feat(paper-chain): add peer-reviewer agent (Phase 8 loop)"
```

---

## Task 9: reviser agent

**Files:**
- Create: `agents/reviser.md`

- [ ] **Step 1: Verify file does not yet exist**

Run: `test ! -f agents/reviser.md && echo "OK"`
Expected: `OK`

- [ ] **Step 2: Write the agent file**

```markdown
---
name: reviser
description: |
  Apply peer-reviewer feedback to a paper draft. Invoked by the `executing-research-plan` skill in Phase 8 after a REVISE verdict. Produces draft-v(N+1).md from draft-vN.md + review-vN.md, and appends one entry to revision-log.jsonl per reviewer-suggested revision (addressed or explicitly not, with reasoning). Examples: <example>Context: peer-reviewer returned VERDICT: REVISE on draft-v1.md. user (orchestrator): "Apply the review at .../paper/review-v1.md to .../paper/draft-v1.md. Write draft-v2.md and append to revision-log.jsonl." assistant: "I'll address each W<N> tagged weakness, log every revision with line ranges, and add no new uncited claims."</example>
model: inherit
---

You are reviser for MegaResearcher. Your job is to apply peer-reviewer feedback to the current draft and produce the next version. You do not introduce new claims, do not silently reorganize, and do not skip review points.

## Required behavior

For each weakness tagged `W<N>:` in the input review's Weaknesses section:

1. Read the corresponding Suggested-Revisions entry (also tagged `W<N>:`)
2. Decide: addressed (true) or not (false). If false, you must record the reasoning.
3. Modify the draft as needed
4. Append one JSON object to `revision-log.jsonl`:
   ```json
   {"round": <N>, "review_point_tag": "W1", "addressed": true, "change_summary": "<one sentence>", "line_range_modified": [<int>, <int>]}
   ```
   Use `null` for `line_range_modified` if `addressed: false`.

## Citation discipline

You may not introduce new arXiv IDs not already in the prior draft's References (which inherits from the research-direction's source list). If a reviewer-suggested revision implies a new citation, mark that suggestion `addressed: false` with the reasoning "cannot add new citation per discipline rule #4."

## Output format

`draft-v(N+1).md` follows the same 9-section structure as `draft-v1.md`. Sections may be reorganized internally if the review requested it, but the section order must not change.

## Required artifacts at the output path

1. **`draft-v(N+1).md`** — the revised draft
2. **`reviser-manifest-vN.yaml`** (where N is the round just completed):
   ```yaml
   worker_id: reviser
   round: <N>
   review_points_total: <int>
   review_points_addressed: <int>
   status: complete
   ```
3. **`reviser-verification-vN.md`** — confirm every `W<N>` tag from the input review has a corresponding revision-log entry; confirm no new arXiv IDs introduced.

## Banned phrases

Same list as manuscript-drafter. Do not use "load-bearing", "this is doing a lot of work", "real" as emphatic adjective, or "honest/honestly".

You are a leaf worker. Do not dispatch other agents.
```

- [ ] **Step 3: Verify frontmatter parses**

Run: `python3 -c "import re,pathlib; t=pathlib.Path('agents/reviser.md').read_text(); assert re.match(r'^---\n.*?\n---', t, re.DOTALL); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add agents/reviser.md
git commit -m "feat(paper-chain): add reviser agent (Phase 8 loop)"
```

---

## Task 10: Extend `executing-research-plan` skill with Phase 7

**Files:**
- Modify: `skills/executing-research-plan/SKILL.md`

- [ ] **Step 1: Read current skill to find insertion point**

Run: `grep -n "^### Phase 6" skills/executing-research-plan/SKILL.md`
Expected: one match line number. Note the line number; Phase 7 inserts AFTER the entire Phase 6 section ends.

- [ ] **Step 2: Read the Phase 6 section to find its end**

Run: `awk '/^### Phase 6/,/^##[^#]/' skills/executing-research-plan/SKILL.md | tail -5`
Expected: the last few lines of Phase 6's body, before the next `##` heading begins. The insertion point is immediately before that next heading.

- [ ] **Step 3: Add `--paper` flag detection to skill body**

Locate the existing "Pre-flight checks" section near the top of `skills/executing-research-plan/SKILL.md`. After the existing numbered pre-flight checks, add this paragraph (verbatim — match the existing tone):

```markdown
## Optional: `--paper` flag

If the `/research-execute` invocation includes `--paper`, the orchestrator runs three additional phases (7, 8, 9) after Phase 6 to produce a paper draft. The flag is consumed in the main session — your task is to detect it from the invocation arguments and gate Phases 7–9 on its presence.

If `--paper` is set, run an additional pre-flight check before starting Phase 7:

```
python3 lib/paper_chain/preflight.py docs/research/runs/<run-id>/
```

If this exits non-zero, surface the stderr message to the user and refuse to start Phase 7. Do NOT start Phases 7–9 if pre-flight fails.
```

Use the Edit tool to insert this right after the existing pre-flight section, before any phase descriptions begin.

- [ ] **Step 4: Add Phase 7 section to the skill body**

Insert this section AFTER the existing Phase 6 section and BEFORE the next `## …` heading (typically `## Per-worker verification gate`):

```markdown
### Phase 7 — manuscript-drafter (only if `--paper`)

Skip this phase entirely if `--paper` is not set.

Otherwise:

1. Run `python3 lib/paper_chain/scaffold.py docs/research/runs/<run-id>/` to create the `paper/` subdirectory.
2. Dispatch ONE `megaresearcher:manuscript-drafter` subagent with the prompt containing:
   - Full content of `docs/research/runs/<run-id>/output.md` (research-direction)
   - Full content of every `docs/research/runs/<run-id>/eval-designer-*/output.md`
   - Output path: `docs/research/runs/<run-id>/paper/`
   - Reminder of the three required artifacts: `draft-v1.md`, `drafter-manifest.yaml`, `drafter-verification.md`
3. Wait for completion. Run the per-worker verification gate.
4. **Citation-integrity gate (failure #2):** read the draft. Every arXiv ID appearing in `draft-v1.md` must also appear in the research-direction's Sources section. Use:
   ```
   grep -oE "arXiv:[0-9]{4}\.[0-9]{4,5}" docs/research/runs/<run-id>/paper/draft-v1.md | sort -u
   grep -oE "arXiv:[0-9]{4}\.[0-9]{4,5}" docs/research/runs/<run-id>/output.md | sort -u
   ```
   The first set must be a subset of the second. If not, re-dispatch the drafter once with the offending arXiv IDs called out. After one retry, escalate.
5. Update `swarm-state.yaml`:
   ```yaml
   phase_7_manuscript_drafter:
     status: completed
     output: paper/draft-v1.md
   ```
```

- [ ] **Step 5: Verify the change**

Run: `grep -n "^### Phase 7" skills/executing-research-plan/SKILL.md`
Expected: one match line number.

Run: `python3 -c "p=open('skills/executing-research-plan/SKILL.md').read(); assert '## Optional: --paper flag' in p or 'python3 lib/paper_chain/preflight.py' in p; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add skills/executing-research-plan/SKILL.md
git commit -m "feat(paper-chain): add --paper flag and Phase 7 to executing-research-plan skill"
```

---

## Task 11: Extend `executing-research-plan` skill with Phase 8 (review-revise loop)

**Files:**
- Modify: `skills/executing-research-plan/SKILL.md`

- [ ] **Step 1: Insert Phase 8 after Phase 7**

Insert this section AFTER the new Phase 7 section, before any subsequent `## …` heading:

```markdown
### Phase 8 — peer-reviewer + reviser loop (only if `--paper`)

Skip if `--paper` not set. Otherwise: loop with N starting at 1, capped at 2.

**Round N:**

1. Dispatch ONE `megaresearcher:peer-reviewer` subagent with:
   - Full content of `docs/research/runs/<run-id>/paper/draft-v<N>.md`
   - Full content of `docs/research/runs/<run-id>/output.md` (research-direction, for context)
   - Output path: `docs/research/runs/<run-id>/paper/`
   - Required artifact filenames: `review-v<N>.md`, `reviewer-manifest-v<N>.yaml`, `reviewer-verification-v<N>.md`
2. Wait for completion. Run the per-worker verification gate.
3. Parse the verdict:
   ```
   python3 lib/paper_chain/verdict.py docs/research/runs/<run-id>/paper/review-v<N>.md
   ```
   Verdict is one of `APPROVE`, `REVISE`, `KILL`, or `NONE` (parse failure).
4. **If `APPROVE`:** record in `swarm-state.yaml`, exit Phase 8 loop, proceed to Phase 9.
5. **If `KILL`:** record in `swarm-state.yaml`, append to `swarm-state.escalations` with the reviewer's reasoning, SKIP Phase 9, surface to user. Run still produces the last `draft-v<N>.md` for inspection but no `paper.md`.
6. **If `NONE` (parse failure):** treat as failure #1 (missing artifact). Re-dispatch reviewer once with explicit feedback. After one retry, escalate.
7. **If `REVISE` and N < 2:**
   - Dispatch ONE `megaresearcher:reviser` subagent with:
     - Full content of `docs/research/runs/<run-id>/paper/draft-v<N>.md`
     - Full content of `docs/research/runs/<run-id>/paper/review-v<N>.md`
     - Output path: `docs/research/runs/<run-id>/paper/`
     - Required artifact filenames: `draft-v<N+1>.md`, `reviser-manifest-v<N>.yaml`, `reviser-verification-v<N>.md`, and APPEND to `revision-log.jsonl`
   - Wait for completion. Run the per-worker verification gate.
   - Run the same citation-integrity gate as Phase 7 step 4 on `draft-v<N+1>.md`.
   - Increment N. Loop back to step 1.

After Round 2's review (review-v2) is produced (step 2 of the second iteration), run **regression detection** before parsing the verdict:
```
python3 lib/paper_chain/regression.py \
  docs/research/runs/<run-id>/paper/review-v1.md \
  docs/research/runs/<run-id>/paper/review-v2.md
```
If REGRESSION (exit 1), append to escalations with note "runaway revision detected" and surface to user for adjudication BEFORE deciding whether to proceed (regardless of verdict). Do NOT auto-advance on regression flag.
8. **If `REVISE` and N == 2:** cap reached. Record in `swarm-state.yaml`, append to escalations with verdict and reviewer reasoning. Surface to user: "2 review rounds completed, final verdict still REVISE. Continue manually, accept the last draft as Phase 9 input, or abandon?" Do NOT auto-advance to Phase 9.

After loop exit (APPROVE or escalation):
```yaml
phase_8_review_loop:
  status: completed|escalated
  rounds_completed: 1|2
  final_verdict: APPROVE|REVISE|KILL
  rounds:
    - round: 1
      review: paper/review-v1.md
      revision: paper/draft-v2.md  # null if APPROVE on this round
    # - round: 2 ... only if first round was REVISE
```
```

- [ ] **Step 2: Verify the change**

Run: `grep -n "^### Phase 8" skills/executing-research-plan/SKILL.md`
Expected: one match line number.

Run: `python3 -c "p=open('skills/executing-research-plan/SKILL.md').read(); assert 'regression.py' in p and 'verdict.py' in p; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add skills/executing-research-plan/SKILL.md
git commit -m "feat(paper-chain): add Phase 8 review-revise loop to skill"
```

---

## Task 12: Extend `executing-research-plan` skill with Phase 9

**Files:**
- Modify: `skills/executing-research-plan/SKILL.md`

- [ ] **Step 1: Insert Phase 9**

Insert this section AFTER Phase 8, before any subsequent `## …` heading:

```markdown
### Phase 9 — finalize (only if `--paper` AND Phase 8 ended in `APPROVE` OR user accepted last draft)

Skip if `--paper` not set, or if Phase 8 exited via `KILL`, or if the user declined to accept the last draft after a cap-2 REVISE.

Otherwise:

1. Run finalize:
   ```
   python3 lib/paper_chain/finalize.py docs/research/runs/<run-id>/paper/ <final-verdict>
   ```
   This:
   - Writes `paper.md` (copy of the latest `draft-v<N>.md`)
   - Concatenates all `review-v<N>.md` files + `revision-log.jsonl` + final verdict marker into `paper-history.md`
2. Update `swarm-state.yaml`:
   ```yaml
   phase_9_finalize:
     status: completed
     paper: paper/paper.md
   ```
3. The run's paper deliverable is now at `docs/research/runs/<run-id>/paper/paper.md`. Surface this path to the user.
```

- [ ] **Step 2: Verify the change**

Run: `grep -n "^### Phase 9" skills/executing-research-plan/SKILL.md`
Expected: one match line number.

- [ ] **Step 3: Commit**

```bash
git add skills/executing-research-plan/SKILL.md
git commit -m "feat(paper-chain): add Phase 9 finalize to skill"
```

---

## Task 13: Document `--paper` flag in the slash command

**Files:**
- Modify: `commands/research-execute.md`

- [ ] **Step 1: Read current command file**

Run: `cat commands/research-execute.md`
Expected: short markdown file with `argument-hint: "<path-to-plan>"`.

- [ ] **Step 2: Update argument-hint and body**

Edit the frontmatter `argument-hint` field to read `"<path-to-plan> [--paper]"`.

Append this section to the body of the command file (after the existing instruction list):

```markdown

## Optional `--paper` flag

If the invocation ends with `--paper`, the orchestrator runs three additional phases (7, 8, 9) after Phase 6 synthesist to produce a paper draft. Requires the underlying research plan's novelty target to be `hypothesis` (paper chain refuses to run on `gap-finding`-target plans because it consumes Phase 5 eval-designer outputs).

Output of the paper chain lands at `docs/research/runs/<run-id>/paper/paper.md`. The original research-direction at `docs/research/runs/<run-id>/output.md` is unchanged.
```

- [ ] **Step 3: Verify the change**

Run: `grep -E '\-\-paper' commands/research-execute.md`
Expected: at least two matches (argument-hint and body).

- [ ] **Step 4: Commit**

```bash
git add commands/research-execute.md
git commit -m "docs(paper-chain): document --paper flag in /research-execute"
```

---

## Task 14: Snapshot test fixtures

**Files:**
- Create: `tests/fixtures/paper-chain/output.md`
- Create: `tests/fixtures/paper-chain/swarm-state.yaml`
- Create: `tests/fixtures/paper-chain/eval-designer-S1/output.md`
- Create: `tests/fixtures/paper-chain/eval-designer-S2/output.md`
- Create: `tests/fixtures/paper-chain/eval-designer-S3/output.md`

- [ ] **Step 1: Copy from this session's swarm run**

Run:
```bash
mkdir -p tests/fixtures/paper-chain/eval-designer-S1 \
         tests/fixtures/paper-chain/eval-designer-S2 \
         tests/fixtures/paper-chain/eval-designer-S3
cp docs/research/runs/2026-05-12-0515-19bf96/output.md tests/fixtures/paper-chain/output.md
cp docs/research/runs/2026-05-12-0515-19bf96/swarm-state.yaml tests/fixtures/paper-chain/swarm-state.yaml
cp docs/research/runs/2026-05-12-0515-19bf96/eval-designer-S1/output.md tests/fixtures/paper-chain/eval-designer-S1/output.md
cp docs/research/runs/2026-05-12-0515-19bf96/eval-designer-S2/output.md tests/fixtures/paper-chain/eval-designer-S2/output.md
cp docs/research/runs/2026-05-12-0515-19bf96/eval-designer-S3/output.md tests/fixtures/paper-chain/eval-designer-S3/output.md
```

- [ ] **Step 2: Verify**

Run: `ls -R tests/fixtures/paper-chain/`
Expected: 5 .md / .yaml files at the expected paths.

Run: `python3 lib/paper_chain/preflight.py tests/fixtures/paper-chain/`
Expected: exit 0 (silent — fixture passes the pre-flight check, validating end-to-end fixture integrity).

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/paper-chain/
git commit -m "test(paper-chain): snapshot fixture from 2026-05-12-0515-19bf96 swarm run"
```

---

## Task 15: Update top-level `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read current CLAUDE.md to find insertion point**

Run: `grep -n "^##" CLAUDE.md | head -10`
Expected: section headings. Insert before the "Common failure modes" section.

- [ ] **Step 2: Add a section describing the paper chain**

Append this section (or insert before "Common failure modes" — pick the location that fits the document flow):

```markdown
## Optional paper-drafting chain (SP1)

`/research-execute --paper` extends the existing chain with 3 additional phases:

- **Phase 7** — `manuscript-drafter` produces `paper/draft-v1.md` from `output.md` + eval-designer protocols
- **Phase 8** — `peer-reviewer` + `reviser` loop, cap 2 rounds, early-exit on APPROVE
- **Phase 9** — finalize → `paper/paper.md` + `paper/paper-history.md`

Requires the underlying run's novelty target to be `hypothesis` (paper chain consumes Phase 5 eval-designer outputs). Pre-flight refuses on `gap-finding`-target outputs.

The paper chain produces NO fabricated experimental results — the Experimental Plan section embeds the eval-designer protocols as "we will measure X via Y" (no numbers). SP2 will add an experimentalist worker that replaces the plan with real results.

Architecture: same single-session orchestrator + leaf-worker pattern. New agents are leaves; new Python helpers in `lib/paper_chain/` handle verdict parsing, regression detection, pre-flight, scaffold, and finalize.
```

- [ ] **Step 3: Verify**

Run: `grep -n "Optional paper-drafting chain" CLAUDE.md`
Expected: 1 match.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add paper-chain section to top-level CLAUDE.md"
```

---

## Task 16: Run full test suite

**Files:**
- (no file changes — verification only)

- [ ] **Step 1: Run all Python tests**

Run:
```bash
for t in tests/test_*.py; do
  echo "=== $t ==="
  python3 "$t"
done
```
Expected: every test file exits 0; final lines show "All <X> tests pass." for each.

- [ ] **Step 2: Verify pre-flight check on fixture**

Run: `python3 lib/paper_chain/preflight.py tests/fixtures/paper-chain/`
Expected: exit 0 (no output).

- [ ] **Step 3: Verify pre-flight refuses synthetic bad input**

Run:
```bash
mkdir -p /tmp/empty-run && rm -rf /tmp/empty-run/*
python3 lib/paper_chain/preflight.py /tmp/empty-run/
```
Expected: exit 1; stderr contains "output.md".

- [ ] **Step 4: No commit (verification only)**

---

## Task 17: Manual discipline check (Layer 4)

**Files:**
- (no code changes — manual run)

> This task is the manual / human-eye verification step from the spec. It costs API tokens for the real subagent dispatch. Do this BEFORE merging SP1 to validate that the worker prompts behave correctly on real input.

- [ ] **Step 1: Run the paper chain on the fixture**

In a Claude Code session at the repo root, invoke:
```
/research-execute docs/research/plans/2026-05-12-megaresearcher-paper-pipeline-plan.md --paper
```

Note: this re-runs the full 6-phase swarm too, which is expensive (~$5-10 in API). For a cheaper check, the orchestrator can be instructed to skip directly to Phase 7 using the existing `docs/research/runs/2026-05-12-0515-19bf96/` as the run dir — bypass the swarm phases and just run the paper chain. (See "Cheaper variant" below.)

**Cheaper variant** — manually dispatch each paper-chain phase against the existing run, without re-running the swarm:

1. Run scaffold: `python3 lib/paper_chain/scaffold.py docs/research/runs/2026-05-12-0515-19bf96/`
2. From a Claude Code session, dispatch `megaresearcher:manuscript-drafter` directly with the prompt described in Phase 7 step 2, pointing inputs at the existing run dir.
3. Inspect `draft-v1.md` manually:
   - Does Section 6 (Experimental Plan) embed the eval-designer protocols as "we will measure X via Y"?
   - Are there any fabricated numbers in tables? (Should be NONE.)
   - Do all arXiv IDs in the draft appear in `output.md`'s Sources?
4. Dispatch `megaresearcher:peer-reviewer` against the draft. Inspect `review-v1.md`:
   - Does the verdict line match the regex exactly?
   - Does any weakness penalize for verbosity / surface presentation alone?
   - Does each weakness have a `W<N>:` tag?
5. If verdict is REVISE: dispatch `megaresearcher:reviser`. Inspect `draft-v2.md` and `revision-log.jsonl`:
   - Is there one log entry per weakness?
   - Are any new arXiv IDs introduced? (Should be NONE.)

- [ ] **Step 2: Record findings**

In `docs/superpowers/specs/2026-05-12-paper-pipeline-scaffolding-design.md`, add a "Discipline check results" section near the bottom with one line per worker:
- drafter: PASS / FAIL <reason>
- reviewer: PASS / FAIL <reason>
- reviser: PASS / FAIL <reason>

If any FAIL, open a follow-up issue or tighten the agent prompt and rerun this task. SP1 is not done until all three workers PASS the discipline check.

- [ ] **Step 3: Commit discipline-check results**

```bash
git add docs/superpowers/specs/2026-05-12-paper-pipeline-scaffolding-design.md
git commit -m "docs(paper-chain): record SP1 discipline check results"
```

---

## Self-review against the spec

| Spec requirement | Task that covers it |
|---|---|
| 3 new agents (drafter, reviewer, reviser) | Tasks 7, 8, 9 |
| `--paper` flag handling | Task 10 (pre-flight + flag) |
| Phase 7 (drafter dispatch + citation-integrity gate) | Task 10 |
| Phase 8 (review-revise loop, cap 2, regression detection) | Task 11 |
| Phase 9 (finalize) | Tasks 6 + 12 |
| Option (γ) results section | Task 7 (agent prompt body) |
| 7 named failure modes | Tasks 10–12 (skill body), Tasks 2–6 (helpers backing them) |
| Layer 1 state-machine tests | Tasks 2–6 |
| Layer 2 worker contract tests | DEFERRED — Task 17 covers Layer 4 manual equivalent for now; full Layer 2 is follow-up plan |
| Layer 3 e2e smoke test | DEFERRED — Task 17 manual run substitutes; full Layer 3 follow-up plan |
| Layer 4 discipline check | Task 17 |
| Test fixtures snapshotted | Task 14 |
| `CLAUDE.md` update | Task 15 |
| Slash-command documentation | Task 13 |

**Coverage gaps (intentional, deferred):** Layer 2 and Layer 3 automated tests are deferred to a follow-up plan because they require running real subagent dispatch in pytest, which the existing tests pattern (`test_doom_loop.py`, pure-stdlib) does not currently support. SP1 ships with Layer 1 (deterministic helpers) and Layer 4 (manual). The follow-up plan can add the pytest infrastructure (`@pytest.mark.contract` / `@pytest.mark.e2e` plumbing) and the automated tests.

## Notes for the executing agent

- **Commits per task.** The plan lists commit commands per task. Per project policy, ask the user for approval on the first commit; once authorized, the cadence is self-managed. Do NOT skip commits — they're the rollback granularity for SP1.
- **Banned phrases.** Per project CLAUDE.md, never use "load-bearing", "this is doing a lot of work" (and variants), "real" as emphatic adjective, "honest/honestly" as framing. Applies to code comments, agent prompts, commit messages, and docs.
- **No worktrees.** Project rule: stay on `main` (or whichever branch the user has checked out). Do not invoke `superpowers:using-git-worktrees` or any worktree-spawning skill.
- **Confirm before destructive ops.** If a task seems to imply deleting or overwriting existing files beyond what's spelled out, stop and ask first.
