# IDEaS Grant-Proposal Chain (SP6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--grant` flag to `/research-execute` that fires three new orchestrator phases (10 grant-drafter, 11 grant-reviewer + grant-reviser loop, 12 grant-finalize) to convert a research-direction document plus a user-filled config file into an IDEaS Competitive Projects proposal draft.

**Architecture:** Single-session orchestrator dispatches three new leaf-worker agents. Drafter consumes research-direction + eval-designer protocols + user-provided config + (optional) scoping/BRIEF.md. Reviewer applies IDEaS-rubric critique (NOT academic peer-review). Reviser fixes flagged defects. New Python helpers in `lib/paper_chain/` handle config validation, verdict parsing, and Phase 12 finalize.

**Tech Stack:** Python 3.11+ stdlib only (`tomllib` for config parsing). No new external deps.

**Spec:** `docs/superpowers/specs/2026-05-15-ideas-grant-chain-design.md`

**SPEC DEVIATION (read carefully):** The spec calls for `grant-config.yaml`. The implementation uses `grant-config.toml` instead because:
- Python 3.11+ stdlib includes `tomllib` (built-in TOML parser, no dep)
- Adding PyYAML breaks the existing pure-stdlib test discipline (`tests/test_doom_loop.py` pattern; all SP1/SP2a helpers are pure stdlib)
- TOML supports nested structures, comments, and lists of dicts — the features the spec needs from YAML
- User-facing impact: filename and minor syntax change in the template. Functionality identical. Spec updated post-merge to reflect this.

---

## File Structure

### New Python helpers — `lib/paper_chain/`

| File | Responsibility |
|---|---|
| `lib/paper_chain/grant_config.py` | Load + validate `grant-config.toml`. `load_grant_config(path) → dict`. `validate_grant_config(config) → (ok, errors)`. Validates required fields, budget arithmetic, at least 1 team_member. CLI: `python3 -m lib.paper_chain.grant_config validate <path>`. |
| `lib/paper_chain/grant_verdict.py` | Parse `GRANT_VERDICT:` line from review file. Pattern mirrors `lib/paper_chain/verdict.py` but uses a distinct prefix so the two chains can't cross-talk. |
| `lib/paper_chain/grant_finalize.py` | Phase 12: copy latest grant draft → `grant/proposal.md`, concatenate review/revision-log → `grant/grant-history.md`, emit `grant/pandoc-conversion.md` with the PDF command. |

### New worker agents — `agents/`

| File | Phase | Dispatch shape |
|---|---|---|
| `agents/grant-drafter.md` | 10 | single-dispatch |
| `agents/grant-reviewer.md` | 11 | loop-dispatch (≤2 times) |
| `agents/grant-reviser.md` | 11 | loop-dispatch (≤2 times) |

### New repo-root file

| File | Responsibility |
|---|---|
| `grant-config.template.toml` | Template with required + optional fields and inline comments. User copies to `docs/research/runs/<run-id>/grant-config.toml` and fills. |

### New tests — `tests/`

| File | Layer |
|---|---|
| `tests/test_grant_config.py` | 1 — load + validate, missing fields, budget mismatch |
| `tests/test_grant_verdict.py` | 1 — verdict-line parser |
| `tests/test_grant_finalize.py` | 1 — finalize helper |
| `tests/manual_grant_dispatch.py` | 3 — e2e smoke (manual, opt-in) |
| `tests/fixtures/grant-config-minimal.toml` | fixture |
| `tests/fixtures/grant-config-malformed-budget.toml` | fixture |
| `tests/fixtures/grant-config-missing-pi.toml` | fixture |

### Modified files

| File | Modification |
|---|---|
| `skills/executing-research-plan/SKILL.md` | Add Phases 10/11/12 + `--grant` pre-flight (validates `grant-config.toml`) |
| `commands/research-execute.md` | argument-hint adds `[--grant]`; body documents `--grant` |
| `CLAUDE.md` | Add SP6 section describing grant chain |

---

## Task 1: grant_config helper

**Files:**
- Create: `lib/paper_chain/grant_config.py`
- Test: `tests/test_grant_config.py`
- Fixtures: `tests/fixtures/grant-config-minimal.toml`, `tests/fixtures/grant-config-malformed-budget.toml`, `tests/fixtures/grant-config-missing-pi.toml`

- [ ] **Step 1: Create fixture files**

`tests/fixtures/grant-config-minimal.toml`:
```toml
# Minimal valid grant-config for tests.
pi_name = "Dr. Test PI"
pi_org = "Test University"
submission_deadline = "2026-06-02"

[[team_members]]
name = "Dr. Test PI"
role = "PI"
relevant_experience = "20 years in multimodal sensor fusion."

[budget]
total_usd = 1500000

[[budget.phases]]
phase_name = "Months 1-4: Data and baseline"
months = "1-4"
total_usd = 500000

[[budget.phases.breakdown]]
personnel = 400000
equipment = 50000
contracts = 30000
other = 20000

[[budget.phases]]
phase_name = "Months 5-8: Method development"
months = "5-8"
total_usd = 600000

[[budget.phases]]
phase_name = "Months 9-12: Evaluation and reporting"
months = "9-12"
total_usd = 400000
```

`tests/fixtures/grant-config-malformed-budget.toml`:
```toml
# Phases sum to 1,400,000 but total_usd says 1,500,000 — mismatch.
pi_name = "Dr. Test PI"
pi_org = "Test University"
submission_deadline = "2026-06-02"

[[team_members]]
name = "Dr. Test PI"
role = "PI"
relevant_experience = "20 years."

[budget]
total_usd = 1500000

[[budget.phases]]
phase_name = "Months 1-4"
months = "1-4"
total_usd = 500000

[[budget.phases]]
phase_name = "Months 5-8"
months = "5-8"
total_usd = 500000

[[budget.phases]]
phase_name = "Months 9-12"
months = "9-12"
total_usd = 400000
```

`tests/fixtures/grant-config-missing-pi.toml`:
```toml
# pi_name absent — validation must fail.
pi_org = "Test University"
submission_deadline = "2026-06-02"

[[team_members]]
name = "Dr. Test PI"
role = "PI"
relevant_experience = "20 years."

[budget]
total_usd = 1500000

[[budget.phases]]
phase_name = "Months 1-12"
months = "1-12"
total_usd = 1500000
```

- [ ] **Step 2: Write `tests/test_grant_config.py`**

```python
"""Tests for grant-config.toml loading and validation.

Run from plugin root:
    python3 tests/test_grant_config.py
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.grant_config import (
    load_grant_config,
    validate_grant_config,
    REQUIRED_TOP_FIELDS,
)

FIXTURES = PLUGIN_ROOT / "tests" / "fixtures"


def test_load_minimal_config():
    config = load_grant_config(FIXTURES / "grant-config-minimal.toml")
    assert config["pi_name"] == "Dr. Test PI"
    assert config["pi_org"] == "Test University"
    assert config["budget"]["total_usd"] == 1500000


def test_validate_minimal_config_passes():
    config = load_grant_config(FIXTURES / "grant-config-minimal.toml")
    ok, errors = validate_grant_config(config)
    assert ok, errors


def test_validate_missing_pi_fails():
    config = load_grant_config(FIXTURES / "grant-config-missing-pi.toml")
    ok, errors = validate_grant_config(config)
    assert not ok
    assert any("pi_name" in e for e in errors)


def test_validate_budget_mismatch_fails():
    config = load_grant_config(FIXTURES / "grant-config-malformed-budget.toml")
    ok, errors = validate_grant_config(config)
    assert not ok
    assert any("budget" in e.lower() and ("total" in e.lower() or "sum" in e.lower()) for e in errors)


def test_validate_empty_team_fails():
    config = {
        "pi_name": "X",
        "pi_org": "Y",
        "submission_deadline": "2026-06-02",
        "team_members": [],
        "budget": {"total_usd": 1500000, "phases": [{"phase_name": "X", "months": "1-12", "total_usd": 1500000}]},
    }
    ok, errors = validate_grant_config(config)
    assert not ok
    assert any("team_member" in e for e in errors)


def test_validate_required_fields_constant():
    expected = {"pi_name", "pi_org", "submission_deadline", "team_members", "budget"}
    assert set(REQUIRED_TOP_FIELDS) == expected


def test_load_missing_file_raises():
    try:
        load_grant_config(Path("/tmp/nonexistent-grant-config.toml"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
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
    print("All grant_config tests pass.")
```

- [ ] **Step 3: Run to verify failure (ImportError)**

Run: `python3 tests/test_grant_config.py`
Expected: ImportError for `lib.paper_chain.grant_config`.

- [ ] **Step 4: Implement `lib/paper_chain/grant_config.py`**

```python
"""Load and validate grant-config.toml for SP6 (IDEaS grant chain).

The config file is TOML (not YAML — Python 3.11+ stdlib includes tomllib;
adding PyYAML would break pure-stdlib discipline. Filename is .toml).

Required top-level fields:
    pi_name (str)
    pi_org (str)
    submission_deadline (str, YYYY-MM-DD)
    team_members (list of dicts, at least 1)
    budget (dict with total_usd and phases)

Required budget structure:
    budget.total_usd (int): exact sum of phase totals
    budget.phases (list of dicts): each with phase_name, months, total_usd

CLI:
    python3 -m lib.paper_chain.grant_config validate <path>
        → exit 0 if valid, exit 1 with errors on stderr otherwise
"""
from __future__ import annotations
import sys
import tomllib
from pathlib import Path

REQUIRED_TOP_FIELDS = [
    "pi_name",
    "pi_org",
    "submission_deadline",
    "team_members",
    "budget",
]


def load_grant_config(path: Path) -> dict:
    """Load TOML; raises FileNotFoundError if missing."""
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def validate_grant_config(config: dict) -> tuple[bool, list[str]]:
    """Return (ok, errors). errors is a list of human-readable error strings."""
    errors: list[str] = []

    for f in REQUIRED_TOP_FIELDS:
        if f not in config:
            errors.append(f"missing required field: {f}")

    if "team_members" in config:
        if not isinstance(config["team_members"], list) or len(config["team_members"]) == 0:
            errors.append("team_members must be a non-empty list (at least 1 team_member required)")

    if "budget" in config:
        budget = config["budget"]
        if "total_usd" not in budget:
            errors.append("budget.total_usd missing")
        if "phases" not in budget or not isinstance(budget["phases"], list) or len(budget["phases"]) == 0:
            errors.append("budget.phases must be a non-empty list")
        else:
            phase_sum = sum(p.get("total_usd", 0) for p in budget["phases"])
            total = budget.get("total_usd", 0)
            if phase_sum != total:
                errors.append(
                    f"budget arithmetic mismatch: phases sum to ${phase_sum:,} but "
                    f"budget.total_usd is ${total:,} (delta ${total - phase_sum:+,})"
                )

    return (len(errors) == 0, errors)


def _main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] != "validate":
        print("usage: grant_config.py validate <config-path>", file=sys.stderr)
        return 2
    try:
        config = load_grant_config(Path(argv[2]))
    except FileNotFoundError as e:
        print(f"grant-config.toml not found at {argv[2]}", file=sys.stderr)
        return 1
    except tomllib.TOMLDecodeError as e:
        print(f"grant-config.toml has invalid TOML syntax: {e}", file=sys.stderr)
        return 1
    ok, errors = validate_grant_config(config)
    if not ok:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
```

- [ ] **Step 5: Run tests to verify pass**

Run: `python3 tests/test_grant_config.py`
Expected: 7 PASS lines + "All grant_config tests pass."

- [ ] **Step 6: Commit**

```bash
git add lib/paper_chain/grant_config.py tests/test_grant_config.py tests/fixtures/grant-config-*.toml
git commit -m "feat(grant-chain): add grant-config TOML loader + validator"
```

---

## Task 2: grant_verdict parser

**Files:**
- Create: `lib/paper_chain/grant_verdict.py`
- Test: `tests/test_grant_verdict.py`

- [ ] **Step 1: Write `tests/test_grant_verdict.py`**

```python
"""Tests for GRANT_VERDICT line parsing.

Run from plugin root:
    python3 tests/test_grant_verdict.py
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.grant_verdict import parse_grant_verdict


def _write(text: str) -> Path:
    fh = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    fh.write(text)
    fh.close()
    return Path(fh.name)


def test_approve():
    p = _write("# Grant review\n\nGRANT_VERDICT: APPROVE\n")
    assert parse_grant_verdict(p) == "APPROVE"


def test_revise():
    p = _write("# Grant review\nGRANT_VERDICT: REVISE\n")
    assert parse_grant_verdict(p) == "REVISE"


def test_kill():
    p = _write("# Grant review\nGRANT_VERDICT: KILL\n")
    assert parse_grant_verdict(p) == "KILL"


def test_no_grant_verdict_line():
    p = _write("# Review\nNo verdict here.\n")
    assert parse_grant_verdict(p) is None


def test_paper_verdict_does_not_match():
    """The paper chain's VERDICT: line must NOT be parsed as a grant verdict."""
    p = _write("# Review\nVERDICT: APPROVE\n")
    assert parse_grant_verdict(p) is None


def test_malformed_grant_verdict():
    p = _write("# Review\nGRANT_VERDICT: MAYBE\n")
    assert parse_grant_verdict(p) is None


def test_case_sensitivity():
    p = _write("# Review\ngrant_verdict: approve\n")
    assert parse_grant_verdict(p) is None


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
    print("All grant_verdict tests pass.")
```

- [ ] **Step 2: Run to verify failure (ImportError)**

- [ ] **Step 3: Implement `lib/paper_chain/grant_verdict.py`**

```python
"""Parse the GRANT_VERDICT line from a grant-review-vN.md file.

A valid grant-verdict line matches exactly `^GRANT_VERDICT: (APPROVE|REVISE|KILL)$`.
The distinct prefix prevents paper-chain VERDICT lines from being parsed here.

CLI:
    python3 -m lib.paper_chain.grant_verdict path/to/grant-review-v1.md
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

VALID = {"APPROVE", "REVISE", "KILL"}
_VERDICT_RE = re.compile(r"^GRANT_VERDICT: (APPROVE|REVISE|KILL)$", re.MULTILINE)


def parse_grant_verdict(review_path: Path) -> str | None:
    """Return verdict word from the given grant-review file, or None."""
    text = review_path.read_text(encoding="utf-8")
    m = _VERDICT_RE.search(text)
    if m is None:
        return None
    return m.group(1)


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: grant_verdict.py <review-path>", file=sys.stderr)
        return 2
    v = parse_grant_verdict(Path(argv[1]))
    print(v if v else "NONE")
    return 0 if v else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 tests/test_grant_verdict.py`
Expected: 7 PASS lines + "All grant_verdict tests pass."

- [ ] **Step 5: Commit**

```bash
git add lib/paper_chain/grant_verdict.py tests/test_grant_verdict.py
git commit -m "feat(grant-chain): add GRANT_VERDICT line parser"
```

---

## Task 3: grant_finalize helper

**Files:**
- Create: `lib/paper_chain/grant_finalize.py`
- Test: `tests/test_grant_finalize.py`

- [ ] **Step 1: Write `tests/test_grant_finalize.py`**

```python
"""Tests for Phase 12 grant-finalize logic.

Run from plugin root:
    python3 tests/test_grant_finalize.py
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.grant_finalize import finalize_grant


def _setup_grant_dir(latest_draft: str) -> Path:
    """Build a grant/ dir with draft-v1.md, optional draft-v2.md, review, log."""
    run = Path(tempfile.mkdtemp(prefix="run-"))
    grant = run / "grant"
    grant.mkdir()
    (grant / "draft-v1.md").write_text("# Grant draft v1\n\nContent.\n")
    if latest_draft == "v2":
        (grant / "draft-v2.md").write_text("# Grant draft v2\n\nRevised.\n")
    (grant / "grant-review-v1.md").write_text("# Grant review v1\n\nGRANT_VERDICT: REVISE\n")
    (grant / "grant-revision-log.jsonl").write_text(
        '{"round":1,"review_point_tag":"W1","addressed":true,'
        '"change_summary":"fixed W1 budget","line_range_modified":[40,45]}\n'
    )
    return grant


def test_finalize_with_v1_only():
    grant = _setup_grant_dir(latest_draft="v1")
    out = finalize_grant(grant, final_verdict="APPROVE")
    assert out == grant / "proposal.md"
    assert out.exists()
    assert "Grant draft v1" in out.read_text()
    history = grant / "grant-history.md"
    assert history.exists()
    assert "Grant review v1" in history.read_text()


def test_finalize_with_v2():
    grant = _setup_grant_dir(latest_draft="v2")
    out = finalize_grant(grant, final_verdict="APPROVE")
    assert "Grant draft v2" in out.read_text()


def test_finalize_emits_pandoc_conversion():
    grant = _setup_grant_dir(latest_draft="v1")
    finalize_grant(grant, final_verdict="APPROVE")
    pandoc_doc = grant / "pandoc-conversion.md"
    assert pandoc_doc.exists()
    text = pandoc_doc.read_text()
    assert "pandoc" in text
    assert "proposal.md" in text
    assert ".pdf" in text


def test_finalize_records_final_verdict():
    grant = _setup_grant_dir(latest_draft="v1")
    finalize_grant(grant, final_verdict="APPROVE")
    history = (grant / "grant-history.md").read_text()
    assert "Final verdict: APPROVE" in history


def test_finalize_history_includes_revision_log():
    grant = _setup_grant_dir(latest_draft="v2")
    finalize_grant(grant, final_verdict="APPROVE")
    history = (grant / "grant-history.md").read_text()
    assert "fixed W1 budget" in history


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
    print("All grant_finalize tests pass.")
```

- [ ] **Step 2: Run to verify failure (ImportError)**

- [ ] **Step 3: Implement `lib/paper_chain/grant_finalize.py`**

```python
"""Phase 12 grant-finalize: produce proposal.md + grant-history.md + pandoc-conversion.md.

Strategy mirrors lib/paper_chain/finalize.py (paper chain), but writes to
grant/proposal.md (not paper.md) and emits a separate pandoc-conversion.md
documenting the PDF generation command for the user's downstream step.

CLI:
    python3 -m lib.paper_chain.grant_finalize <grant-dir> <final-verdict>
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

_DRAFT_RE = re.compile(r"^draft-v(\d+)\.md$")
_REVIEW_RE = re.compile(r"^grant-review-v(\d+)\.md$")

_PANDOC_TEMPLATE = """# Converting grant proposal to PDF

The IDEaS portal accepts PDF submissions. Generate a PDF from the markdown
proposal with pandoc:

```bash
pandoc proposal.md --pdf-engine=xelatex -o proposal.pdf
```

If pandoc is not installed: `brew install pandoc basictex` (macOS) or your
distro's equivalent. xelatex is required for unicode + IDEaS form fonts.

To preview without rendering PDF:
```bash
pandoc proposal.md -o proposal.html
open proposal.html
```

For IDEaS-form-specific styling (page numbering, section headers matching
the official template), a custom LaTeX template can be passed via
`--template=ideas.latex`. SP6 does not ship a template; using pandoc's
default is acceptable for v1.
"""


def _latest_draft(grant_dir: Path) -> Path:
    drafts = []
    for p in grant_dir.iterdir():
        m = _DRAFT_RE.match(p.name)
        if m:
            drafts.append((int(m.group(1)), p))
    if not drafts:
        raise FileNotFoundError(f"No draft-vN.md in {grant_dir}")
    drafts.sort()
    return drafts[-1][1]


def _ordered_reviews(grant_dir: Path) -> list[Path]:
    reviews = []
    for p in grant_dir.iterdir():
        m = _REVIEW_RE.match(p.name)
        if m:
            reviews.append((int(m.group(1)), p))
    reviews.sort()
    return [p for _, p in reviews]


def finalize_grant(grant_dir: Path, final_verdict: str) -> Path:
    """Produce proposal.md (latest draft) + grant-history.md + pandoc-conversion.md.

    Returns path to proposal.md.
    """
    latest = _latest_draft(grant_dir)
    proposal = grant_dir / "proposal.md"
    proposal.write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")

    history_parts = [f"# Grant proposal history\n\nFinal verdict: {final_verdict}\n"]
    for r in _ordered_reviews(grant_dir):
        history_parts.append(f"\n---\n\n## {r.name}\n\n{r.read_text(encoding='utf-8')}")
    log = grant_dir / "grant-revision-log.jsonl"
    if log.exists() and log.stat().st_size > 0:
        history_parts.append(
            f"\n---\n\n## grant-revision-log.jsonl\n\n```jsonl\n{log.read_text(encoding='utf-8')}```\n"
        )
    (grant_dir / "grant-history.md").write_text("".join(history_parts), encoding="utf-8")

    (grant_dir / "pandoc-conversion.md").write_text(_PANDOC_TEMPLATE, encoding="utf-8")
    return proposal


def _main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: grant_finalize.py <grant-dir> <final-verdict>", file=sys.stderr)
        return 2
    out = finalize_grant(Path(argv[1]), argv[2])
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 tests/test_grant_finalize.py`
Expected: 5 PASS lines + "All grant_finalize tests pass."

- [ ] **Step 5: Commit**

```bash
git add lib/paper_chain/grant_finalize.py tests/test_grant_finalize.py
git commit -m "feat(grant-chain): add Phase 12 grant-finalize + pandoc-conversion emission"
```

---

## Task 4: grant-drafter agent

**Files:**
- Create: `agents/grant-drafter.md`

- [ ] **Step 1: Write the agent file**

```markdown
---
name: grant-drafter
description: |
  Compose an IDEaS Competitive Projects proposal draft from a research-direction document, eval-designer protocols (if present), a user-provided grant-config.toml, and an optional scoping/BRIEF.md. Invoked by the `executing-research-plan` skill in Phase 10 (only when `/research-execute` is run with `--grant`). Produces draft-v1.md following the 12-section IDEaS Application Form structure. Examples: <example>Context: a research-direction has been produced and the user invoked /research-execute --grant. user (orchestrator): "Draft an IDEaS proposal from docs/research/runs/.../output.md, the eval-designer protocols, and grant-config.toml. Write to docs/research/runs/.../grant/draft-v1.md." assistant: "I'll produce a 12-section IDEaS Competitive Projects draft. Budget arithmetic will sum to grant-config.budget.total_usd exactly. Every claim will cite a source from research-direction's Sources or grant-config.citations."</example>
model: inherit
---

You are grant-drafter for MegaResearcher. Your job is to turn a research-direction document plus optional eval-designer protocols plus a user-provided grant-config.toml plus optional scoping/BRIEF.md into a draft IDEaS Competitive Projects proposal. You do NOT invent budget figures, team members, or capabilities — those come from grant-config. You do NOT fabricate experimental results.

## Required output structure (12 sections)

`draft-v1.md` must contain these sections in order:

1. **Cover page** — project title (one line) / PI name + org / period (12 months) / funding band ($1.5M / TRL 4-5) / submission deadline (from grant-config.submission_deadline)
2. **Executive summary** — one paragraph, ≤200 words, summarizing the operational need + technical approach + expected outcome
3. **Problem statement** — CAF/DND operational need framed from the research-direction's Introduction. If `scoping/BRIEF.md` is present, draw context from it directly.
4. **Innovation / technical approach** — drawn from research-direction's Method section (or Proposed Architecture for hypothesis-target runs; three-candidate shortlist for gap-finding-target runs)
5. **Innovation Pathway (TRL roadmap)** — 12 monthly milestones, mapping TRL 4 → TRL 5 progression. Each milestone references named protocols from eval-designer (when present) or candidate-architecture details from the research-direction's shortlist.
6. **Solution methodology** — if eval-designer outputs are present: embed pre-registered decision rules and named substrates verbatim. If absent (gap-finding-target): describe the three candidate architectures from the research-direction shortlist + the open data + baseline references for each.
7. **Project management plan** — Gantt-shaped milestone table aligned to the 12-month period; team member names from grant-config mapped to roles + months
8. **Team capability** — drawn from grant-config.team_members; each member's relevant_experience verbatim
9. **Budget by phase** — grant-config.budget.phases formatted as a table. Phase totals MUST sum to grant-config.budget.total_usd exactly.
10. **Risk mitigation** — drawn from research-direction's threats-to-validity + audit trail of killed hypotheses
11. **Letters of support** — grant-config.support_letters as a list with status (received / pending / in progress). If list is empty, write "No support letters at submission; outreach planned in Phase 1."
12. **References** — every cited paper from research-direction's Sources, deduplicated, with arXiv IDs. May include citations from grant-config.citations if present.

## Citation discipline

Every claim in the draft must cite either:
- An arXiv ID / DOI from the research-direction's Sources section, OR
- An arXiv ID / DOI from grant-config.citations (user-provided extra citations for team-publication and prior-work references)

Do NOT introduce new arXiv IDs not in those two sources. If a claim implies a new citation, REMOVE the claim instead.

## Budget arithmetic discipline

The Budget section MUST sum to grant-config.budget.total_usd EXACTLY. The orchestrator's post-Phase-10 budget-arithmetic gate will re-dispatch you once if the sum is off. Verify before emitting:
- Sum all phase totals from grant-config.budget.phases
- Match grant-config.budget.total_usd
- The Budget section's "Total" line in the draft must equal that value

## TRL roadmap discipline

Milestones in §5 Innovation Pathway must be:
- **Sequenced**: TRL 4 entry to TRL 5 exit across the 12-month period
- **Anchored**: each milestone references a named substrate or protocol from eval-designer outputs OR a named candidate architecture from the research-direction shortlist
- **Datedish**: monthly anchors (Month 1, Month 2, ..., Month 12) with concrete deliverables per month

## Required artifacts at the output path

1. **`draft-v1.md`** — the draft, 12 sections as above
2. **`grant-drafter-manifest.yaml`**:
   ```yaml
   worker_id: grant-drafter
   word_count: <int>
   section_count: 12
   citation_count: <int>
   citations: [<arxiv-id>, ...]
   budget_total_usd: <int>  # MUST match grant-config.budget.total_usd
   tr_progression: "TRL 4 → TRL 5"
   status: complete
   ```
3. **`grant-drafter-verification.md`** — confirm all 12 sections present; word count; ≥3 spot-checked citations; budget sums to total; no new arXiv IDs.

## Banned phrases

Per project CLAUDE.md, never use "load-bearing", "this is doing a lot of work" (and variants), "real" as emphatic adjective ("real research", "real impact"), or "honest / honestly / to be honest" as framing. Plain alternatives only. (This matters extra here — a grant proposal full of AI-tell phrases reads as machine-written and tanks credibility.)

You are a leaf worker. Do not dispatch other agents.
```

- [ ] **Step 2: Verify frontmatter parses**

Run: `python3 -c "import re,pathlib; t=pathlib.Path('agents/grant-drafter.md').read_text(); assert re.match(r'^---\n.*?\n---', t, re.DOTALL); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agents/grant-drafter.md
git commit -m "feat(grant-chain): add grant-drafter agent (Phase 10)"
```

---

## Task 5: grant-reviewer agent

**Files:**
- Create: `agents/grant-reviewer.md`

- [ ] **Step 1: Write the agent file**

```markdown
---
name: grant-reviewer
description: |
  IDEaS-rubric critique of a grant-draft. Invoked by the `executing-research-plan` skill in Phase 11 (the grant review-revise loop). Produces grant-review-vN.md ending with a GRANT_VERDICT line that the orchestrator parses. Critiques against IDEaS Competitive Projects rubric — TRL progression credibility, budget-milestone alignment, open-data compliance, team capability vs work, risk mitigation completeness, YAGNI fence integrity. Does NOT apply academic peer-review rubric (novelty, statistical significance, etc.). Examples: <example>Context: grant-drafter produced draft-v1.md. user (orchestrator): "Review the grant draft. Write grant-review-v1.md with a GRANT_VERDICT line." assistant: "I'll critique against the IDEaS rubric — TRL pathway, budget alignment, open-data, team capability, risks — and end with GRANT_VERDICT: APPROVE | REVISE | KILL."</example>
model: inherit
---

You are grant-reviewer for MegaResearcher. Your job is to critique a grant draft against the IDEaS Competitive Projects rubric. You are NOT an academic peer-reviewer; do NOT penalize for ML-conference novelty or statistical-significance criteria.

## Required output structure

`grant-review-vN.md` must contain these sections in order:

1. **Summary** — one paragraph, what the proposal claims
2. **Strengths** — bullet list, what's defensible. Do not pad.
3. **Weaknesses** — bullet list. Each weakness tagged `W1:`, `W2:`, etc. Tags increment monotonically; carried-over weaknesses keep their tag, new weaknesses get new tags.
4. **Suggested Revisions** — for each weakness, a concrete action the reviser could take
5. **Verdict** — the LAST line of the file MUST match exactly `GRANT_VERDICT: APPROVE | GRANT_VERDICT: REVISE | GRANT_VERDICT: KILL`

## IDEaS-rubric critique focus (in order)

1. **TRL progression credibility** (§5 Innovation Pathway) — does the 12-month milestone path map a defensible TRL 4 → 5 progression? Or are the milestones hand-wavy / unanchored to specific substrates?
2. **Budget-to-milestone alignment** (§9 Budget by phase + §7 Project management plan) — does the budget breakdown match the work described in the milestone table? Are phase totals proportional to the named work? Most importantly: does the Budget section sum EXACTLY to grant-config.budget.total_usd?
3. **Open-data / no-classified compliance** — every named dataset must be open or synthetic. Flag any reference to classified data, paywalled-only data, or undisclosed proprietary data.
4. **Team capability vs proposed work** (§8 Team capability + §7 Project management plan) — are the team members named in grant-config plausibly capable of the proposed work? Capability mismatches between role and relevant_experience are Weaknesses.
5. **Risk mitigation completeness** (§10 Risk mitigation) — are the risks from research-direction's threats-to-validity actually mitigated, or just acknowledged? Mitigations that say "we will be careful" are unacceptable; mitigations that name a concrete fallback path are acceptable.
6. **YAGNI fence integrity** — does the proposal claim things the research-direction explicitly excluded? Out-of-scope creep is a Weakness.

## What NOT to penalize

These are NOT grounds for REVISE or KILL — recategorize as Suggestion at most:

- **ML-conference novelty claims** — IDEaS values TRL progression and operational utility, NOT method-novelty for its own sake
- **Statistical significance bars** — pre-registered decision rules from eval-designer ARE the rigor surface; further significance discussion is not required
- **Related-work depth** — grant proposals have shorter related-work than papers
- **Writing style** unless it materially impedes the IDEaS reviewer's ability to grade

## Verdict criteria

- **APPROVE** — no critical defects; up to 2 minor Suggestions remain
- **REVISE** — 1 or more concrete defects the reviser can address without restructuring (typical case)
- **KILL** — fundamental error: budget that cannot sum to total_usd in any restructuring, classified-data reference, team capability totally absent for the proposed work, YAGNI-fence violation requiring a different proposal

KILL is reserved for proposals that cannot be fixed without producing a different proposal. Almost all reviews end REVISE or APPROVE.

## Required artifacts

1. **`grant-review-vN.md`** — review with `GRANT_VERDICT` line last
2. **`grant-reviewer-manifest-vN.yaml`**:
   ```yaml
   worker_id: grant-reviewer
   round: <N>
   weakness_count: <int>
   verdict: <APPROVE|REVISE|KILL>
   status: complete
   ```
3. **`grant-reviewer-verification-vN.md`** — confirm verdict line matches `^GRANT_VERDICT: (APPROVE|REVISE|KILL)$`; every weakness has a `W<N>:` tag; no ML-rubric penalties applied.

## Banned phrases

Same list as grant-drafter. Plain alternatives only.

You are a leaf worker. Do not dispatch other agents.
```

- [ ] **Step 2: Verify frontmatter parses**

Run: `python3 -c "import re,pathlib; t=pathlib.Path('agents/grant-reviewer.md').read_text(); assert re.match(r'^---\n.*?\n---', t, re.DOTALL); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add agents/grant-reviewer.md
git commit -m "feat(grant-chain): add grant-reviewer agent (Phase 11 loop)"
```

---

## Task 6: grant-reviser agent

**Files:**
- Create: `agents/grant-reviser.md`

- [ ] **Step 1: Write the agent file**

```markdown
---
name: grant-reviser
description: |
  Apply grant-reviewer feedback to a grant-draft. Invoked by the `executing-research-plan` skill in Phase 11 after a GRANT_VERDICT: REVISE. Produces draft-v(N+1).md from draft-vN.md + grant-review-vN.md + grant-config.toml. Appends one entry to grant-revision-log.jsonl per reviewer-suggested revision. Cannot introduce new uncited claims; cannot change grant-config values; budget edits must preserve grant-config.budget.total_usd. Examples: <example>Context: grant-reviewer returned REVISE on draft-v1.md. user (orchestrator): "Apply the review. Write draft-v2.md and append to grant-revision-log.jsonl." assistant: "I'll address each W<N> weakness, log every revision with line ranges, and preserve grant-config.budget.total_usd exactly."</example>
model: inherit
---

You are grant-reviser for MegaResearcher. Your job is to apply grant-reviewer feedback to the current grant-draft. You do not introduce new claims, do not silently reorganize, do not change grant-config values, and do not skip review points.

## Required behavior

For each weakness tagged `W<N>:` in the input grant-review's Weaknesses section:

1. Read the corresponding Suggested-Revisions entry (also tagged `W<N>:`)
2. Decide: addressed (true) or not (false). If false, record the reasoning.
3. Modify the draft as needed (within discipline rules below)
4. Append one JSON object to `grant-revision-log.jsonl`:
   ```json
   {"round": <N>, "review_point_tag": "W1", "addressed": true, "change_summary": "<one sentence>", "line_range_modified": [<int>, <int>]}
   ```

## Discipline rules

- **Citation discipline:** no new arXiv IDs not already in research-direction's Sources OR grant-config.citations. If a revision implies a new citation, mark `addressed: false` with reasoning "cannot add new citation per discipline rule."
- **Grant-config immutability:** you may NOT change grant-config.toml values. Budget edits in the draft must still sum to grant-config.budget.total_usd. Team member edits must keep grant-config.team_members as canonical. If a revision would require changing grant-config, mark `addressed: false` with reasoning "requires grant-config change; out of scope."
- **Section preservation:** the 12-section structure must remain. Section content may be reorganized; section order may not change.
- **Budget arithmetic:** if you modify the Budget section, verify the new sum equals grant-config.budget.total_usd. The orchestrator's post-Phase-11 budget gate will re-dispatch you if the sum is off.

## Output format

`draft-v(N+1).md` follows the same 12-section IDEaS structure as `draft-v1.md`.

## Required artifacts at the output path

1. **`draft-v(N+1).md`** — the revised draft
2. **`grant-reviser-manifest-vN.yaml`** (N = round just completed):
   ```yaml
   worker_id: grant-reviser
   round: <N>
   review_points_total: <int>
   review_points_addressed: <int>
   status: complete
   ```
3. **`grant-reviser-verification-vN.md`** — confirm every `W<N>` tag from the input review has a corresponding revision-log entry; confirm no new arXiv IDs introduced; confirm budget arithmetic preserved.

## Banned phrases

Same list as grant-drafter. Plain alternatives only.

You are a leaf worker. Do not dispatch other agents.
```

- [ ] **Step 2: Verify frontmatter parses**

Run: `python3 -c "import re,pathlib; t=pathlib.Path('agents/grant-reviser.md').read_text(); assert re.match(r'^---\n.*?\n---', t, re.DOTALL); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add agents/grant-reviser.md
git commit -m "feat(grant-chain): add grant-reviser agent (Phase 11 loop)"
```

---

## Task 7: Extend skill with Phase 10 + --grant pre-flight

**Files:**
- Modify: `skills/executing-research-plan/SKILL.md`

- [ ] **Step 1: Locate insertion point for the --grant pre-flight**

Run: `grep -n "## Optional: \`--paper\` flag\|## Generate the run-id" skills/executing-research-plan/SKILL.md`
Expected: 2 matches. Pre-flight for `--grant` inserts BEFORE the run-id section, right after the `--paper` section ends.

- [ ] **Step 2: Use the Edit tool to add `--grant` pre-flight section**

Find this content in `skills/executing-research-plan/SKILL.md`:
```
If this exits non-zero, surface the stderr message to the user and refuse to start Phase 7. Do NOT start Phases 7–9 if pre-flight fails.

## Generate the run-id and scaffold
```

Replace with:
```
If this exits non-zero, surface the stderr message to the user and refuse to start Phase 7. Do NOT start Phases 7–9 if pre-flight fails.

## Optional: `--grant` flag

If the `/research-execute` invocation includes `--grant`, the orchestrator runs three additional phases (10, 11, 12) to produce an IDEaS Competitive Projects proposal draft. The flag is composable with `--paper` (in which case paper phases run first), or standalone (Phases 10-12 fire directly after Phase 6).

Pre-flight before starting Phase 10:

1. Verify `docs/research/runs/<run-id>/grant-config.toml` exists. If absent, refuse with: "Copy `grant-config.template.toml` to docs/research/runs/<run-id>/grant-config.toml and fill required fields, then re-run with --grant."

2. Validate the config:
```
python3 lib/paper_chain/grant_config.py validate docs/research/runs/<run-id>/grant-config.toml
```

If this exits non-zero, surface the stderr to the user and refuse to start Phase 10.

Unlike `--paper`, `--grant` does NOT require the research-direction to be hypothesis-target. Grant chain works on gap-finding-target runs too (drafter falls back to the three-candidate shortlist for the Solution methodology section).

## Generate the run-id and scaffold
```

- [ ] **Step 3: Add Phase 10 section**

Find the existing Phase 9 ending. Run: `grep -n "^### Phase 9" skills/executing-research-plan/SKILL.md`
Expected: 1 match.

Find this content (the END of Phase 9 — last paragraph before the "## Per-worker verification gate" heading):
```
3. The run's paper deliverable is now at `docs/research/runs/<run-id>/paper/paper.md`. Surface this path to the user.

## Per-worker verification gate
```

Replace with:
```
3. The run's paper deliverable is now at `docs/research/runs/<run-id>/paper/paper.md`. Surface this path to the user.

### Phase 10 — grant-drafter (only if `--grant`)

Skip this phase entirely if `--grant` is not set.

Otherwise:

1. `mkdir -p docs/research/runs/<run-id>/grant/`
2. Dispatch ONE `megaresearcher:grant-drafter` subagent with the prompt containing:
   - Full content of `docs/research/runs/<run-id>/output.md` (research-direction)
   - Full content of every `docs/research/runs/<run-id>/eval-designer-*/output.md` if present (skip if gap-finding-target)
   - Full content of `docs/research/runs/<run-id>/grant-config.toml`
   - Full content of `scoping/BRIEF.md` if present at repo root
   - Output path: `docs/research/runs/<run-id>/grant/`
   - Required artifacts: `draft-v1.md`, `grant-drafter-manifest.yaml`, `grant-drafter-verification.md`
3. Wait for completion. Run the per-worker verification gate.
4. **Citation-integrity gate:** every arXiv ID in `grant/draft-v1.md` must appear in research-direction's Sources OR in grant-config.citations (if present):
   ```
   grep -oE "arXiv:[0-9]{4}\.[0-9]{4,5}" docs/research/runs/<run-id>/grant/draft-v1.md | sort -u
   grep -oE "arXiv:[0-9]{4}\.[0-9]{4,5}" docs/research/runs/<run-id>/output.md | sort -u
   # Plus any IDs listed in grant-config.citations
   ```
   If draft cites IDs not in either source, re-dispatch grant-drafter once with the offending IDs called out. After one retry, escalate.
5. **Budget-arithmetic gate:** parse the §9 Budget by phase section from draft-v1.md; verify sum of phase totals == grant-config.budget.total_usd. If mismatch, re-dispatch once with the specific phase numbers that don't sum. After retry, escalate.
6. Update `swarm-state.yaml`:
   ```yaml
   phase_10_grant_drafter:
     status: completed
     output: grant/draft-v1.md
   ```
```

- [ ] **Step 4: Verify**

Run: `grep -n "^### Phase 10" skills/executing-research-plan/SKILL.md`
Expected: 1 match.

Run: `grep -nE "lib/paper_chain/grant_config|grant-config.toml" skills/executing-research-plan/SKILL.md`
Expected: 3+ matches.

- [ ] **Step 5: Commit**

```bash
git add skills/executing-research-plan/SKILL.md
git commit -m "feat(grant-chain): add --grant pre-flight + Phase 10 to orchestrator skill"
```

---

## Task 8: Extend skill with Phase 11 (grant review-revise loop)

**Files:**
- Modify: `skills/executing-research-plan/SKILL.md`

- [ ] **Step 1: Insert Phase 11 after Phase 10**

Find this content (the END of Phase 10):
```
   phase_10_grant_drafter:
     status: completed
     output: grant/draft-v1.md
   ```
```

Find what immediately follows (the `## Per-worker verification gate` heading or end-of-Phase-9 if no Phase 10 existed) and insert Phase 11 BEFORE that next heading:

Append after Phase 10's closing block:
```

### Phase 11 — grant-reviewer + grant-reviser loop (only if `--grant`)

Skip if `--grant` not set. Otherwise: loop with N starting at 1, capped at 2.

**Round N:**

1. Dispatch ONE `megaresearcher:grant-reviewer` subagent with:
   - Full content of `docs/research/runs/<run-id>/grant/draft-v<N>.md`
   - Full content of `docs/research/runs/<run-id>/output.md` (research-direction, for context)
   - Full content of `docs/research/runs/<run-id>/grant-config.toml`
   - Output path: `docs/research/runs/<run-id>/grant/`
   - Required artifacts: `grant-review-v<N>.md`, `grant-reviewer-manifest-v<N>.yaml`, `grant-reviewer-verification-v<N>.md`
2. Wait for completion. Run the per-worker verification gate.
3. Parse the verdict:
   ```
   python3 lib/paper_chain/grant_verdict.py docs/research/runs/<run-id>/grant/grant-review-v<N>.md
   ```
   Verdict is one of `APPROVE`, `REVISE`, `KILL`, or `NONE` (parse failure).
4. **If `APPROVE`:** record in `swarm-state.yaml`, exit Phase 11 loop, proceed to Phase 12.
5. **If `KILL`:** record in `swarm-state.yaml`, append to `swarm-state.escalations` with the reviewer's reasoning, SKIP Phase 12, surface to user. Last `draft-v<N>.md` preserved for inspection but no `proposal.md`.
6. **If `NONE` (parse failure):** re-dispatch grant-reviewer once with explicit feedback "your verdict line must match `^GRANT_VERDICT: (APPROVE|REVISE|KILL)$` as the last line". After one retry, escalate.
7. **If `REVISE` and N < 2:**
   - Dispatch ONE `megaresearcher:grant-reviser` subagent with:
     - Full content of `docs/research/runs/<run-id>/grant/draft-v<N>.md`
     - Full content of `docs/research/runs/<run-id>/grant/grant-review-v<N>.md`
     - Full content of `docs/research/runs/<run-id>/grant-config.toml`
     - Output path: `docs/research/runs/<run-id>/grant/`
     - Required artifacts: `draft-v<N+1>.md`, `grant-reviser-manifest-v<N>.yaml`, `grant-reviser-verification-v<N>.md`, and APPEND to `grant-revision-log.jsonl`
   - Wait for completion. Run the per-worker verification gate.
   - Run the same citation-integrity gate as Phase 10 step 4 on `draft-v<N+1>.md`.
   - Run the same budget-arithmetic gate as Phase 10 step 5 on `draft-v<N+1>.md`.
   - Increment N. Loop back to step 1.

After Round 2's review (grant-review-v2) is produced, run **regression detection** before parsing the verdict:
```
python3 lib/paper_chain/regression.py \
  docs/research/runs/<run-id>/grant/grant-review-v1.md \
  docs/research/runs/<run-id>/grant/grant-review-v2.md
```
If REGRESSION (exit 1), append to escalations with note "runaway revision detected" and surface to user for adjudication BEFORE deciding whether to proceed.

8. **If `REVISE` and N == 2:** cap reached. Record in `swarm-state.yaml`, append to escalations. Surface to user: "2 grant-review rounds completed, final verdict still REVISE. Continue manually, accept the last draft as Phase 12 input, or abandon?" Do NOT auto-advance.

After loop exit (APPROVE or escalation):
```yaml
phase_11_grant_review_loop:
  status: completed|escalated
  rounds_completed: 1|2
  final_verdict: APPROVE|REVISE|KILL
  rounds:
    - round: 1
      review: grant/grant-review-v1.md
      revision: grant/draft-v2.md  # null if APPROVE on this round
```
```

- [ ] **Step 2: Verify**

Run: `grep -n "^### Phase 11" skills/executing-research-plan/SKILL.md`
Expected: 1 match.

Run: `grep -nE "grant_verdict.py|grant-reviser|grant-reviewer" skills/executing-research-plan/SKILL.md`
Expected: 5+ matches.

- [ ] **Step 3: Commit**

```bash
git add skills/executing-research-plan/SKILL.md
git commit -m "feat(grant-chain): add Phase 11 grant review-revise loop to skill"
```

---

## Task 9: Extend skill with Phase 12 (grant finalize)

**Files:**
- Modify: `skills/executing-research-plan/SKILL.md`

- [ ] **Step 1: Insert Phase 12 after Phase 11**

Append after Phase 11's closing block:

```

### Phase 12 — grant-finalize (only if `--grant` AND Phase 11 ended in `APPROVE` OR user accepted last draft)

Skip if `--grant` not set, or if Phase 11 exited via `KILL`, or if the user declined to accept the last draft after a cap-2 REVISE.

Otherwise:

1. Run finalize:
   ```
   python3 lib/paper_chain/grant_finalize.py docs/research/runs/<run-id>/grant/ <final-verdict>
   ```
   This:
   - Writes `proposal.md` (copy of latest `draft-v<N>.md`)
   - Concatenates all `grant-review-v<N>.md` + `grant-revision-log.jsonl` + final verdict marker into `grant-history.md`
   - Emits `pandoc-conversion.md` with the user's PDF generation command
2. Update `swarm-state.yaml`:
   ```yaml
   phase_12_grant_finalize:
     status: completed
     proposal: grant/proposal.md
   ```
3. The run's grant deliverable is now at `docs/research/runs/<run-id>/grant/proposal.md`. Surface this path to the user with the pandoc command from `pandoc-conversion.md` so they can generate the PDF for IDEaS submission.
```

- [ ] **Step 2: Verify**

Run: `grep -n "^### Phase 12" skills/executing-research-plan/SKILL.md`
Expected: 1 match.

- [ ] **Step 3: Commit**

```bash
git add skills/executing-research-plan/SKILL.md
git commit -m "feat(grant-chain): add Phase 12 grant-finalize to skill"
```

---

## Task 10: --grant flag docs + template

**Files:**
- Create: `grant-config.template.toml` (at repo root)
- Modify: `commands/research-execute.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Create `grant-config.template.toml`**

```toml
# IDEaS Competitive Projects — grant-config template (SP6)
# Copy this file to docs/research/runs/<run-id>/grant-config.toml and fill in
# the REQUIRED fields. Run `python3 lib/paper_chain/grant_config.py validate
# <path>` to check before invoking /research-execute --grant.

# ============================================================
# REQUIRED FIELDS
# ============================================================

pi_name = ""              # Principal Investigator full name
pi_org = ""               # Affiliated organization
submission_deadline = ""  # YYYY-MM-DD (e.g., 2026-06-02 for current IDEaS cycle)

# At least 1 team_member required.
[[team_members]]
name = ""
role = ""                 # e.g., Co-PI, Postdoc, Research Engineer
relevant_experience = ""  # 2-3 sentences describing prior relevant work

# Add more team members by duplicating the block above.

[budget]
total_usd = 1500000       # IDEaS Competitive Projects band

# Phases must sum to total_usd.
[[budget.phases]]
phase_name = ""           # e.g., "Months 1-4: Data and baseline"
months = ""               # "1-4"
total_usd = 0
# Optional breakdown (does not affect arithmetic — informational only):
[[budget.phases.breakdown]]
personnel = 0
equipment = 0
contracts = 0
other = 0

# Add more phases by duplicating the [[budget.phases]] block.

# ============================================================
# OPTIONAL FIELDS
# ============================================================

# Support letters at submission time (may be empty list).
[[support_letters]]
source = ""               # e.g., "Industry partner X" or "DRDC research group Y"
status = ""               # "received" | "pending" | "in progress"

# Project management organization (a paragraph on team structure).
project_management_organization = ""

# Existing capabilities (a paragraph on prior relevant work, infrastructure).
existing_capabilities = ""

# Partner organizations.
partner_orgs = []

# Previous IDEaS proposals (for continuity references).
previous_ideas_proposals = []

# Extra citations beyond research-direction's Sources.
# Use this for team-publication and prior-work references.
citations = []            # list of arXiv IDs or DOIs
```

- [ ] **Step 2: Update `commands/research-execute.md`**

Edit the existing `argument-hint` field to read `"<path-to-plan> [--paper] [--grant]"`.

Append after the `--paper` section:

```markdown

## Optional `--grant` flag

If the invocation includes `--grant`, the orchestrator runs three additional phases (10, 11, 12) AFTER the existing chain (and after Phase 9 if `--paper` is also set) to produce an IDEaS Competitive Projects proposal draft. Composable with `--paper` (each chain is independent).

Pre-flight requires `docs/research/runs/<run-id>/grant-config.toml` with the required fields filled. Copy `grant-config.template.toml` to the run directory and edit. Without a valid config, the grant chain refuses to run with named missing fields.

Unlike `--paper`, `--grant` works on `gap-finding`-target research-directions too. When eval-designer outputs are absent (gap-finding case), the drafter falls back to the research-direction's three-candidate shortlist for the Solution methodology section.

Output of the grant chain lands at `docs/research/runs/<run-id>/grant/proposal.md`. A `pandoc-conversion.md` file in the same directory documents the command for generating the PDF for IDEaS portal submission.
```

- [ ] **Step 3: Update `CLAUDE.md`**

Rename the existing "Optional paper-drafting chain (SP1+SP2a)" section to "Optional paper-drafting chain (SP1+SP2a) + grant chain (SP6)". After the SP2a additions block, append:

```markdown

**SP6 additions (IDEaS grant chain):**
- **Phase 10** — `grant-drafter` writes a 12-section IDEaS Competitive Projects proposal from research-direction + eval-designer + grant-config.toml + (optional) scoping/BRIEF.md.
- **Phase 11** — `grant-reviewer` + `grant-reviser` loop, cap N=2, IDEaS-rubric critique (TRL progression, budget-milestone alignment, open-data compliance, team capability, risk mitigation completeness, YAGNI fence).
- **Phase 12** — finalize → `grant/proposal.md` + `grant/grant-history.md` + `grant/pandoc-conversion.md`.
- **Key difference from paper chain:** grant chain ALLOWS gap-finding-target research-directions. When eval-designer outputs are absent, drafter falls back to the three-candidate shortlist as the solution methodology surface. This makes the Canadian defense example (gap-finding) actionable end-to-end.
- **User-input mechanism:** `grant-config.toml` at the run dir with PI / team / budget by phase / support letters. Template at repo root: `grant-config.template.toml`. Pre-flight validates budget arithmetic + required-field presence.
- **Output format:** markdown only at SP6 ship; `pandoc-conversion.md` documents the user's PDF command. PDF generation is the user's downstream step (not in the chain).
- **Spec deviation:** spec called for `grant-config.yaml`; implementation uses `grant-config.toml` for pure-stdlib parsing (Python 3.11+ `tomllib`). No external deps.
```

- [ ] **Step 4: Verify**

Run: `grep -n "grant-config.template.toml\|--grant" commands/research-execute.md CLAUDE.md`
Expected: 3+ matches across both files.

Run: `ls grant-config.template.toml`
Expected: file exists.

- [ ] **Step 5: Commit**

```bash
git add grant-config.template.toml commands/research-execute.md CLAUDE.md
git commit -m "docs(grant-chain): add --grant flag docs + grant-config template"
```

---

## Task 11: Run full test suite

- [ ] **Step 1: Run all tests**

Run:
```bash
for t in tests/test_*.py; do printf "%s: " "$t"; python3 "$t" 2>&1 | tail -1; done
```
Expected: 15 lines (12 pre-existing + 3 new from SP6), each passing.

- [ ] **Step 2: Verify grant-config.template.toml is valid TOML**

Run: `python3 -c "import tomllib; tomllib.load(open('grant-config.template.toml', 'rb')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Verify validation fails on the bare template (empty required fields)**

Run: `python3 lib/paper_chain/grant_config.py validate grant-config.template.toml 2>&1; echo "EXIT=$?"`
Expected: exit code 1; stderr contains specific empty-field messages OR budget mismatch (template has placeholder values).

- [ ] **Step 4: Verify validation passes on the minimal fixture**

Run: `python3 lib/paper_chain/grant_config.py validate tests/fixtures/grant-config-minimal.toml; echo "EXIT=$?"`
Expected: exit 0; stdout "OK".

- [ ] **Step 5: No commit (verification only)**

---

## Task 12: Manual discipline check (two-run)

> This task validates the grant chain end-to-end against two fixtures. Token cost: ~$5-10 total ($3-5 per drafter dispatch since the draft is long). The first run uses the SP1 fixture (hypothesis-target with eval-designer outputs); the second uses the Canadian defense example (gap-finding target — exercises the failure-#11 ALLOW path).

- [ ] **Step 1: Create a minimal grant-config.toml in the SP1 fixture run**

Copy the minimal fixture as the SP1 run's grant config:
```bash
mkdir -p tests/fixtures/paper-chain/grant
cp tests/fixtures/grant-config-minimal.toml tests/fixtures/paper-chain/grant-config.toml
```

- [ ] **Step 2: Create a minimal grant-config.toml in the Canadian defense example**

```bash
mkdir -p docs/research/examples/multimodal-fusion-gap-finding/run-2026-05-10-0615-0ece4e/grant
cp tests/fixtures/grant-config-minimal.toml docs/research/examples/multimodal-fusion-gap-finding/run-2026-05-10-0615-0ece4e/grant-config.toml
```

Note: this assumes the existing example run has its swarm output at the path above. Adjust if the actual layout differs.

- [ ] **Step 3: Create the manual dispatch wrapper**

Create `tests/manual_grant_dispatch.py`:

```python
"""Manual SP6 grant-chain dispatch test.

Exercises Phase 10 (grant-drafter) against TWO fixtures:
1. SP1 fixture (hypothesis-target, has eval-designer outputs)
2. Canadian defense example (gap-finding target, NO eval-designer outputs)

This script does NOT actually dispatch the agent — it validates the
pre-flight + sets up the orchestrator state. Real agent dispatch costs
~$5-10 and should be run manually via /research-execute --grant once
the user has reviewed this output.

Run from plugin root:
    python3 tests/manual_grant_dispatch.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.grant_config import load_grant_config, validate_grant_config


def check_fixture(label: str, run_dir: Path):
    print(f"\n=== {label}: {run_dir} ===")
    grant_config_path = run_dir / "grant-config.toml"
    if not grant_config_path.exists():
        print(f"  FAIL: grant-config.toml not found at {grant_config_path}")
        return False
    try:
        config = load_grant_config(grant_config_path)
    except Exception as e:
        print(f"  FAIL: TOML load error: {e}")
        return False
    ok, errors = validate_grant_config(config)
    if not ok:
        print(f"  FAIL: validation errors:")
        for e in errors:
            print(f"    - {e}")
        return False
    print(f"  OK: grant-config valid. Ready for /research-execute --grant.")
    print(f"     PI: {config['pi_name']}")
    print(f"     Budget: ${config['budget']['total_usd']:,}")
    print(f"     Phases: {len(config['budget']['phases'])}")
    return True


def main() -> int:
    fixture_1 = PLUGIN_ROOT / "tests" / "fixtures" / "paper-chain"
    fixture_2 = PLUGIN_ROOT / "docs" / "research" / "examples" / "multimodal-fusion-gap-finding" / "run-2026-05-10-0615-0ece4e"

    results = []
    results.append(check_fixture("SP1 fixture (hypothesis-target)", fixture_1))
    if fixture_2.exists():
        results.append(check_fixture("Canadian defense example (gap-finding)", fixture_2))
    else:
        print(f"\n=== Canadian defense example: {fixture_2} ===")
        print(f"  SKIP: example run dir not found (expected for SP6 — populate manually if needed)")

    print(f"\n=== Summary ===")
    print(f"Fixtures checked: {len(results)}")
    print(f"Passed: {sum(results)}")
    if not all(results):
        return 1
    print(f"\nReady for manual end-to-end testing. To exercise the full chain:")
    print(f"  /research-execute <plan-path> --grant")
    print(f"  (or --paper --grant for both chains)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the manual dispatch wrapper**

Run: `python3 tests/manual_grant_dispatch.py`
Expected: 1-2 fixtures pass validation; ready-for-dispatch message printed.

- [ ] **Step 5: Run the actual SP6 chain end-to-end on the SP1 fixture (TOKEN-EXPENSIVE)**

This is the moment-of-truth integration test. Requires real agent dispatch. SKIP this step if controller does not want to spend ~$5-10 in tokens; document the deferral in the spec instead.

In a Claude Code session at the repo root, invoke (after the fixture grant-config is in place from Step 1):
```
/research-execute docs/research/plans/2026-05-12-megaresearcher-paper-pipeline-plan.md --grant
```

This will run the full 6-phase swarm AGAIN (expensive). For a cheaper check, manually dispatch each grant-chain agent against the existing SP1 fixture using the `Agent` tool directly with:
- subagent_type: megaresearcher:grant-drafter
- prompt: includes the inlined fixture output.md + eval-designer-S{1,2,3}/output.md + grant-config.toml content

Then dispatch megaresearcher:grant-reviewer against the resulting draft-v1.md.

Inspect the outputs:
- Does grant/draft-v1.md have all 12 IDEaS sections?
- Does the §9 Budget by phase section sum exactly to $1,500,000?
- Are any arXiv IDs in the draft NOT in the research-direction Sources?
- Does the grant-reviewer's verdict line match `^GRANT_VERDICT: (APPROVE|REVISE|KILL)$`?
- Are any banned phrases ("load-bearing", "real", "honest") in the draft?

- [ ] **Step 6: Record discipline-check results**

Append to `docs/superpowers/specs/2026-05-15-ideas-grant-chain-design.md` a section:

```markdown

## Discipline check results (SP6 manual T12, 2026-05-16)

**Run 1 — SP1 fixture (hypothesis-target):**
- pre-flight validation: PASS / FAIL <reason>
- grant-drafter 12 sections present: PASS / FAIL <which missing>
- budget arithmetic (sum = $1.5M): PASS / FAIL <delta>
- citation discipline (subset of Sources + grant-config.citations): PASS / FAIL <invented IDs>
- grant-reviewer verdict line format: PASS / FAIL <actual line>
- banned-phrase scan: PASS / FAIL <found phrases>

**Run 2 — Canadian defense example (gap-finding target, failure-#11 ALLOW path):**
- pre-flight validation: PASS / FAIL
- grant-drafter falls back to three-candidate shortlist as Solution methodology: PASS / FAIL <observation>
- 12 sections present: PASS / FAIL
- budget arithmetic: PASS / FAIL
- credibility check (could user send this to DND after filling real budget + team?): YES / NO <reasoning>

**Outcomes:**
- <summary of findings, link to specific issues>
- <follow-up items, if any>
```

- [ ] **Step 7: Commit T12 outputs**

```bash
git add tests/manual_grant_dispatch.py tests/fixtures/paper-chain/grant-config.toml docs/superpowers/specs/2026-05-15-ideas-grant-chain-design.md
git commit -m "test(grant-chain): SP6 manual discipline check + dispatch helper"
```

---

## Self-review against the spec

| Spec requirement | Task that covers it |
|---|---|
| 3 new agents (grant-drafter, grant-reviewer, grant-reviser) | Tasks 4, 5, 6 |
| `--grant` flag handling | Task 7 (pre-flight) |
| Phase 10 (grant-drafter dispatch + citation + budget gates) | Task 7 |
| Phase 11 (grant-reviewer + grant-reviser loop, cap 2, regression check) | Task 8 |
| Phase 12 (grant-finalize) | Tasks 3 + 9 |
| 12 IDEaS sections in draft | Task 4 (agent prompt body) |
| Drafter falls back to 3-candidate shortlist for gap-finding-target runs | Task 4 + Task 7 (pre-flight allows gap-finding) |
| 12 named failure modes | Tasks 7-9 (skill body), Tasks 1-3 (helpers backing them) |
| Pre-flight grant-config validation | Tasks 1 + 7 |
| Budget arithmetic gate | Task 7 (post-drafter), Task 8 (post-reviser) |
| Layer 1 state-machine tests | Tasks 1, 2, 3 |
| Layer 2 worker contract tests | DEFERRED — Task 12 manual covers them initially; full pytest @manual layer is follow-up |
| Layer 3 e2e smoke test | Task 12 |
| Layer 4 discipline check (two-run) | Task 12 |
| Test fixtures snapshotted | Task 1 (configs); Task 12 (run dirs) |
| `grant-config.template.toml` | Task 10 |
| `CLAUDE.md` update | Task 10 |
| Slash-command documentation | Task 10 |

**Coverage gaps (intentional, deferred):** Worker contract tests (Layer 2) are deferred to a follow-up plan because they require real agent dispatch in a pytest harness, which the existing pure-stdlib tests pattern doesn't support. SP6 ships with Layer 1 (deterministic helpers) + Layer 3/4 (manual smoke + discipline check) covered. Spec-deviation note: `grant-config.yaml` → `grant-config.toml` (documented prominently in plan header and CLAUDE.md).

## Notes for the executing agent

- **Per project CLAUDE.md:** no worktrees; stay on `main`. Banned phrases: "load-bearing", "this is doing a lot of work" (variants), "real" as emphatic adjective, "honest / honestly". Confirm before destructive ops.
- **Commits per task.** Plan lists commit commands; the controller (you) handles all commits unless authorized otherwise. Match the SP1/SP2a pattern: do everything-implement-test-verify across tasks, then batch commits in semantic groups at the end.
- **Spec deviation: TOML not YAML.** This is intentional and documented at plan header. Don't try to switch back to YAML mid-implementation.
- **Pre-flight reuses `grant_config.validate_grant_config`** — call from the skill's pre-flight section, not from a separate helper. The skill body uses `python3 lib/paper_chain/grant_config.py validate <path>` as a one-liner.
- **The 3 new agents borrow `W<N>:` weakness tagging from SP1's peer-reviewer + reviser.** `regression.py` is reused as-is. Do NOT introduce a new regression helper.
- **The `--grant` flag is independent of `--paper`.** Both can be set. Don't add coupling between the two chains beyond Phase ordering (paper phases run before grant phases if both set).
