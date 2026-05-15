# Paper-Pipeline Scaffolding (Sub-Project 1) — Design

**Status:** draft
**Created:** 2026-05-12
**Parent project:** Wide-port augmentation of MegaResearcher into an end-to-end paper pipeline (Level 3 — full, decomposed into 5 sub-projects)
**This sub-project:** SP1 — paper-pipeline scaffolding (no experimentalist, no S1/S2/S3 wiring yet)

## Goal

Extend MegaResearcher so that, after a successful research-direction swarm run, the user can opt in to a paper-drafting chain (`/research-execute --paper`) that produces a peer-reviewed, paper-shaped draft from the existing research-direction document. No sandbox infrastructure, no real experimental results, no hypothesis-augmentations wired in. SP1 validates the orchestrator-can-drive-paper-pipeline pattern; SP2–SP5 build on it.

## Provenance

The wide-port direction follows from the swarm run `docs/research/runs/2026-05-12-0515-19bf96/` (see `docs/research/runs/2026-05-12-0515-19bf96/output.md`). That run surfaced three surviving hypotheses (S1 cross-family routing, S2 length-debias wrapper, S3 majority-vote on structured decisions) — all of which require paper-generation workers (drafter, peer-reviewer, reviser) that MegaResearcher does not currently have. SP1 builds those workers; SP3/4/5 layer the hypotheses on top.

## Scope (in)

- New worker agents: `manuscript-drafter`, `peer-reviewer`, `reviser`
- New orchestrator phases: 7 (drafter), 8 (review-revise loop, cap 2 rounds), 9 (finalize)
- New CLI surface: `/research-execute --paper` flag
- Artifact format: position-paper-style markdown, 7 required sections
- Results section: option (γ) — embed Phase 5 eval-designer protocols as "we will measure X via Y", no fabricated numbers
- Error handling: 7 named failure modes with deterministic responses
- Tests: 4 layers (state-machine / worker contract / e2e smoke / discipline check)

## Scope (out — YAGNI fence)

- Experimentalist worker + sandbox integration (deferred to SP2)
- Cross-family / cross-tier model routing (deferred to SP3 — S1 implementation)
- Length-debias wrapper on reviewer scores (deferred to SP4 — S2 implementation)
- Majority-vote on structured decisions (deferred to SP5 — S3 implementation)
- LaTeX output / camera-ready formatting
- Multi-paper-per-run output (one paper per run only)
- Non-`hypothesis`-target research-direction inputs (paper chain refuses to run on `gap-finding`-target outputs because Phase 5 protocols are required as input)

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Existing chain (unchanged):                                       │
│   /research-init → /research-execute                              │
│   Produces: docs/research/runs/<run-id>/output.md                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ if --paper flag set
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ NEW: Paper-draft phases 7-9                                       │
│                                                                    │
│   Phase 7 — manuscript-drafter (single dispatch)                  │
│     Inputs:  output.md + eval-designer-*/output.md                │
│     Output:  paper/draft-v1.md                                    │
│                                                                    │
│   Phase 8 — peer-reviewer + reviser (loop, cap N=2)               │
│     reviewer reads draft-vN → writes review-vN with verdict       │
│     APPROVE → exit, go to Phase 9                                 │
│     KILL    → exit, escalate, no Phase 9                          │
│     REVISE  → reviser writes draft-v(N+1), loop if N<2            │
│                                                                    │
│   Phase 9 — finalize (single dispatch)                            │
│     symlink draft-v(final) → paper/paper.md                       │
│     concatenate review/revision-log → paper-history.md            │
└──────────────────────────────────────────────────────────────────┘
```

Same single-session-orchestrator + leaf-worker pattern as the existing chain. No nested dispatch. File-based artifact passing.

## Components

### `agents/manuscript-drafter.md` (NEW)

- **Remit:** turn research-direction + eval-designer protocols into a paper draft
- **Inputs:** inlined into prompt — full content of `output.md` + every `eval-designer-*/output.md` from Phase 5
- **Outputs:** `paper/draft-v1.md`, `paper/drafter-manifest.yaml`, `paper/drafter-verification.md`
- **Format requirements for `draft-v1.md`:** 9 sections in order — Title, Abstract, Introduction, Related Work, Method, Experimental Plan (option γ — embeds eval-designer protocols as "we will measure X via decision rule Y"), Discussion, Limitations, References. ≤8000 words (rough NeurIPS-workshop length budget; pages-as-rendered varies by formatter).
- **Discipline:** every claim cites a source already in the research-direction's source list; no new uncited claims. Drafter's own `verification.md` must spot-check ≥3 cited claims resolve via `hf_papers paper_details`.

### `agents/peer-reviewer.md` (NEW)

- **Remit:** ICLR-style critique of the current draft
- **Inputs:** current `paper/draft-vN.md` + research-direction's `output.md` (for context only — reviewer reads the paper itself for the review)
- **Outputs:** `paper/review-vN.md`, `paper/reviewer-manifest-vN.yaml`, `paper/reviewer-verification-vN.md`
- **Format requirements for `review-vN.md`:** Summary / Strengths / Weaknesses / Suggested Revisions / Verdict line. Verdict line is the LAST line and must match `^VERDICT: (APPROVE|REVISE|KILL)$`.
- **Discipline:** critique substance, not presentation alone. Reviewer's prompt explicitly forbids penalizing for verbosity alone (cites scout-3's known-untrustworthy proxies from the spawning swarm). KILL is reserved for fundamental errors (uncited claims drafter snuck through, mechanism contradicting the research-direction's own findings, etc.) — not "needs work."

### `agents/reviser.md` (NEW)

- **Remit:** apply reviewer feedback
- **Inputs:** `paper/draft-vN.md` + `paper/review-vN.md`
- **Outputs:** `paper/draft-v(N+1).md`, `paper/reviser-manifest-vN.yaml`, `paper/reviser-verification-vN.md`, append entry to `paper/revision-log.jsonl`
- **Format for `revision-log.jsonl`:** one JSON object per appended line: `{round: N, review_point_index: int, addressed: bool, change_summary: str, line_range_modified: [int, int]}`
- **Discipline:** cannot introduce new uncited claims. Revision log is the audit trail — every reviewer-suggested-revision item gets a log entry (even if `addressed: false`, with reasoning).

### Orchestrator extension: `skills/executing-research-plan/SKILL.md` and the orchestrator-as-skill code

- Parse `--paper` flag from `/research-execute` invocation string
- Phase 6 synthesist completes → if `--paper`: scaffold `paper/` subdir, fire Phase 7
- Phase 8 loop driver: read reviewer verdict, branch (APPROVE → 9 | KILL → escalate | REVISE → reviser → loop if N<2 | REVISE at N=2 → escalate)
- Phase 9 finalize: write `paper/paper.md` (symlink to latest draft), concatenate `review-v*.md` + `revision-log.jsonl` → `paper-history.md`
- All state changes recorded in `swarm-state.yaml`

## Data flow

(See Section 3 of the brainstorming dialogue — diagrammatic flow lives there; reproduced concretely in the implementation plan.)

Key state additions to `swarm-state.yaml`:

```yaml
phases:
  phase_7_manuscript_drafter:
    status: pending|completed|escalated
    output: paper/draft-v1.md
  phase_8_review_loop:
    status: pending|completed|escalated
    rounds_completed: 0|1|2
    final_verdict: APPROVE|REVISE|KILL
    rounds:
      - round: 1
        review: paper/review-v1.md
        revision: paper/draft-v2.md  # null if APPROVE on this round
      # - round: 2 ... only if first round was REVISE
  phase_9_finalize:
    status: pending|completed
    paper: paper/paper.md
```

## Error handling

| # | Failure | Detection | Response |
|---|---|---|---|
| 1 | manuscript-drafter missing artifacts | Per-worker verification gate (existing pattern) | Re-dispatch once with explicit feedback. After one retry: escalate. |
| 2 | Drafter introduces uncited claims | Orchestrator scans draft's arXiv IDs against research-direction's source list | Re-dispatch drafter once with offending paragraphs called out. After one retry: escalate. |
| 3 | peer-reviewer returns `KILL` | Verdict-line parse | Exit Phase 8. Skip Phase 9. Escalate. Last `draft-vN.md` preserved for user inspection. |
| 4 | Cap-2 reached without APPROVE | Loop counter | Exit Phase 8. Escalate: "2 review rounds completed, final verdict REVISE. Accept last draft / continue manually / abandon?" |
| 5 | Runaway revision | After Round 2: diff review-v1 weaknesses vs review-v2 weaknesses; if new ≥ closed, flag regression | Same escalation as #4 with regression note. |
| 6 | `--paper` set but no research-direction `output.md` | Pre-flight before Phase 7 | Refuse: "Phase 6 synthesist did not produce output.md. Re-run /research-execute first." |
| 7 | `--paper` set but Phase 5 eval-designer outputs missing | Pre-flight | Refuse: "Paper chain requires Phase 5 eval-designer protocols. Research-direction was produced with novelty target gap-finding; paper chain only runs on hypothesis-target outputs." |

All escalations append to `swarm-state.escalations` with `{worker, failure_code, retry_count, recommendation}`. Never silent-advance.

## Testing

### Layer 1 — Orchestrator state-machine tests (deterministic)

Pure-Python tests in `tests/test_paper_orchestrator.py`. Mock subagent dispatch. Cover:
- `--paper` flag parsing
- Pre-flight refusals (#6, #7)
- Verdict-line parsing including malformed inputs
- Loop-counter logic (N=1 REVISE → round 2; N=2 REVISE → escalation)
- Regression-detection logic (#5)
- `swarm-state.yaml` phase transitions

### Layer 2 — Worker contract tests (real subagent dispatch, expensive)

For each of `manuscript-drafter`, `peer-reviewer`, `reviser`: one fixture test in `tests/contract/test_<worker>.py`.
- Fixture: research-direction from `docs/research/runs/2026-05-12-0515-19bf96/` (this session's own swarm run)
- Run the real worker once
- Assert: three artifacts present, format requirements satisfied
- For drafter: every arXiv ID in `draft-v1.md` appears in research-direction's source list
- For reviewer: verdict line matches the regex
- For reviser: revision-log has one entry per review-point in input review

Marked `@pytest.mark.contract`; skipped on default `pytest`, run via `pytest -m contract` before merging worker changes.

### Layer 3 — End-to-end smoke test (slow, opt-in)

`tests/e2e/test_paper_chain.py`. Runs full `/research-execute --paper` against the fixture run. Asserts:
- `paper/paper.md` exists at end
- `paper-history.md` exists with ≥1 round logged
- All three new phases in swarm-state are `completed` or `escalated`
- Any worker hitting the cap has an escalation entry

Marked `@pytest.mark.e2e`; run before tagging a release.

### Layer 4 — Discipline check (manual, run-once before merge)

Before merging SP1, run the chain on `docs/research/runs/2026-05-12-0515-19bf96/output.md` and inspect:
- Drafter uses option (γ) — protocols as "we will measure X via Y", zero fabricated numbers
- Reviewer critiques substance per scout-3's untrustworthy-proxy list (no length-bias penalty)
- Revision log honestly records each review point

Human-eye check. Failure → tighten worker prompts before SP1 ships.

### Test fixtures

`tests/fixtures/paper-chain/` contains a snapshot copy of:
- `output.md` (research-direction)
- `eval-designer-S1/output.md`
- `eval-designer-S2/output.md`
- `eval-designer-S3/output.md`

Snapshotted to avoid coupling tests to the live `docs/research/runs/` tree.

## Implementation notes (handed to writing-plans)

- Existing skill is `skills/executing-research-plan/SKILL.md`. The orchestrator state-machine lives in the skill's text (it's a skill-as-orchestrator, per project CLAUDE.md). Extension is additive to the skill body — new sections for Phase 7/8/9 + the `--paper` flag handling.
- Existing agents live in `agents/<name>.md`. New agents follow the same frontmatter + body format. Reference: any existing agent file, e.g., `agents/red-team.md` for the verdict-line + revision-loop pattern.
- The orchestrator dispatches subagents via `Agent` tool calls with `subagent_type: megaresearcher:<agent-name>`. New agents will need plugin manifest entries.
- The Python test layer interacts with the orchestrator skill how? — the skill executes in a Claude Code session, not in pytest. Layer 1 tests do not invoke the skill; they unit-test extracted Python helpers (e.g., verdict parsing, regression detection) that the orchestrator skill calls out to. Plan must factor those helpers out as importable code.

## Open questions for writing-plans (none blocking — flagging for the plan author)

- Where does the `--paper` flag actually get parsed? The `/research-execute` slash command spec lives in `commands/`. Plan must locate the parse-site.
- Symlink behavior on Windows (Claude Code runs cross-platform) — `paper.md → draft-vN.md` symlink may need to be a file-copy on Windows. Plan should fall back to copy if symlink fails.

## Decisions locked in

- 2026-05-12 · SP1 only this round · Level 3 decomposed into SP1–SP5; SP1 is the prerequisite gate for SP2–SP5.
- 2026-05-12 · Entry via `--paper` flag (option B) · Single command, two outputs (research-direction + paper draft). Couples chains intentionally for now; can refactor later if needed.
- 2026-05-12 · Drafter → reviewer → reviser, cap 2 rounds, early-exit on APPROVE (option iii) · Mirrors existing red-team revision-loop pattern.
- 2026-05-12 · Experimental-results section uses eval-designer protocols (option γ) · No fabricated numbers; honest by construction; sets up SP2 to drop in real results.
- 2026-05-12 · `paper/` subdir under existing `docs/research/runs/<run-id>/` · Co-located with research run; one paper per run.
- 2026-05-12 · Cap-3 escalation pattern reused from existing hypothesis-revision loop · No new escalation primitives.
