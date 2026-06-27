# IDEaS Grant-Proposal Chain (SP6) — Design

**Status:** draft
**Created:** 2026-05-15
**Parent project:** Wide-port augmentation of MegaResearcher into an end-to-end research-to-grant pipeline.
**This sub-project:** SP6 — `/research-execute --grant` produces an IDEaS Competitive Projects proposal draft from a swarm-produced research-direction plus a user-provided `grant-config.yaml`. Composable with `--paper`. Independent from the paper chain.

## Goal

Add three new orchestrator phases (10 — grant-drafter; 11 — grant-reviewer + grant-reviser loop; 12 — grant-finalize) that turn a research-direction document plus a user-filled `grant-config.yaml` into an IDEaS Competitive Projects proposal draft. The chain integrates the research-direction's intellectual content (problem statement, technical approach, eval-designer protocols as proposed experiments, threats-to-validity as risk mitigations) with user-provided proposal-only material (PI/team CVs, budget by phase, support letters) and emits a markdown draft following the official IDEaS form structure. Pandoc-to-PDF is a documented downstream step; not in scope for the chain itself.

## Provenance

The Canadian defense example at `docs/research/examples/multimodal-fusion-gap-finding/` explicitly states its purpose is to feed an IDEaS Competitive Projects proposal in the TRL 4-5 / $1.5M / 12-month band. SP1's paper chain produces ML-conference papers — wrong format for IDEaS. SP6 adds the format. The chain works on gap-finding-target research-directions (unlike the paper chain) because grants propose future research, not report completed experiments.

## Scope (in)

- 3 new worker agents: `grant-drafter`, `grant-reviewer`, `grant-reviser`
- 3 new orchestrator phases: 10, 11, 12
- New CLI surface: `--grant` flag on `/research-execute`. Composable with `--paper`.
- New artifact format: 12-section IDEaS proposal markdown
- New user-input mechanism: `grant-config.yaml` with required fields (PI, team, budget by phase, support letters)
- New Python helpers: `lib/paper_chain/grant_config.py`, `lib/paper_chain/grant_verdict.py`, `lib/paper_chain/grant_finalize.py`
- Citation-integrity gate + budget-arithmetic gate after grant-drafter
- Failure handling: 12 named modes
- Tests: 4 layers
- Grant chain ALLOWS gap-finding-target research-directions (the paper chain doesn't)

## Scope (out — YAGNI fence)

- Other Canadian grant programs (NSERC, NRC IRAP, etc.) — out of scope for SP6 v1; a future SP could generalize
- US programs (NSF, NIH, DARPA) — out of scope
- LaTeX template generation / direct PDF emission — user runs pandoc separately with the command we emit
- Auto-filling team CVs from public profiles (LinkedIn, ORCID) — manual fill via grant-config.yaml
- Auto-fetching IDEaS rubric updates from canada.ca — IDEaS section list is hardcoded in this design
- IP / data plan boilerplate generation — agent uses grant-config-provided fields; doesn't synthesize legal language
- Multi-proposal generation per run — one proposal per run; future-work if users want IDEaS + NSERC variants from one swarm
- Re-using rejected proposals from past IDEaS cycles — no integration with prior submissions

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Existing chain (unchanged):                                       │
│   Phase 1-6 (research swarm)                                      │
│   if --paper:                                                      │
│     Phase 6.5 (experimentalist) → 7 (drafter) → 8 (revise) → 9 (final)│
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ if --grant
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ NEW: Grant-drafting phases 10-12 (--grant-gated)                  │
│                                                                    │
│   Pre-flight (additional): grant-config.yaml valid                │
│   - All required fields populated                                 │
│   - budget.phases sum to budget.total_usd                         │
│   - At least 1 team_member                                        │
│   Refuse if invalid                                                │
│                                                                    │
│   Phase 10 — grant-drafter (single dispatch)                      │
│     Inputs: output.md + eval-designer-*/output.md (if present) +  │
│             grant-config.yaml + scoping/BRIEF.md (optional)       │
│     Output: grant/draft-v1.md (12 IDEaS sections)                 │
│                                                                    │
│   Phase 11 — grant-reviewer + grant-reviser loop (cap N=2)        │
│     reviewer reads draft-vN; writes grant-review-vN.md with       │
│     GRANT_VERDICT: APPROVE | REVISE | KILL                        │
│     APPROVE → exit, Phase 12                                       │
│     KILL    → exit, escalate, no Phase 12                         │
│     REVISE  → grant-reviser writes draft-v(N+1); loop if N<2      │
│                                                                    │
│   Phase 12 — grant-finalize                                       │
│     copy latest draft → grant/proposal.md                         │
│     concatenate review + revision-log → grant/grant-history.md    │
│     emit grant/pandoc-conversion.md with the PDF command           │
└──────────────────────────────────────────────────────────────────┘
```

**Composability:**
- `--paper` alone: Phases 7-9 (existing SP1 chain)
- `--grant` alone: Phases 10-12 directly after Phase 6
- `--paper --grant`: 7-9 then 10-12
- Each chain is independent — grant doesn't read paper outputs (it reads research-direction + eval-designer outputs + grant-config)
- Phase 6.5 (experimentalist) does NOT trigger from `--grant` alone — grants propose experiments, they don't include results

Same single-session orchestrator + leaf-worker pattern. No nested dispatch. File-based artifact passing.

## Components

### `agents/grant-drafter.md` (NEW)

- **Remit:** turn research-direction + eval-designer protocols + grant-config + (optional) BRIEF into a 12-section IDEaS Competitive Projects proposal draft
- **Inputs (inlined into prompt):** `output.md` content; all `eval-designer-*/output.md` content (if present); parsed `grant-config.yaml`; `scoping/BRIEF.md` if present
- **Outputs:** `grant/draft-v1.md`, `grant/grant-drafter-manifest.yaml`, `grant/grant-drafter-verification.md`
- **Format requirements for `draft-v1.md` — 12 sections in order:**
  1. Cover page — project title, PI name + org, period (12 months), funding band ($1.5M / TRL 4-5), submission date
  2. Executive summary — one paragraph, ≤200 words
  3. Problem statement — CAF/DND operational need; drawn from research-direction Introduction
  4. Innovation / technical approach — drawn from Method
  5. Innovation Pathway (TRL roadmap) — 12 monthly milestones, TRL 4 → TRL 5 progression
  6. Solution methodology — experimental protocols verbatim from eval-designer outputs (if present); fallback to research-direction's three-candidate shortlist for gap-finding-target runs
  7. Project management plan — Gantt-shaped milestone table, team roles, decision points
  8. Team capability — from grant-config team_members
  9. Budget by phase — from grant-config budget.phases, summed to $1.5M (or whatever total_usd grant-config specifies)
  10. Risk mitigation — drawn from research-direction's threats-to-validity + audit trail of killed hypotheses
  11. Letters of support — from grant-config support_letters list with status (received / pending / in progress)
  12. References — every cited paper from research-direction's sources, deduplicated
- **Discipline:** every claim cites either a source already in research-direction's Sources OR a citation provided in grant-config. No new uncited claims. Budget arithmetic must total exactly to grant-config.total_usd. TRL roadmap milestones reference protocol-named substrates / experiments (or shortlist-named candidates if gap-finding).

### `agents/grant-reviewer.md` (NEW, loop-dispatch)

- **Remit:** IDEaS-rubric critique (NOT academic peer-review)
- **Inputs:** `grant/draft-vN.md` + research-direction `output.md` + grant-config.yaml
- **Outputs:** `grant/grant-review-vN.md`, manifest-vN, verification-vN. Verdict line LAST, matches `^GRANT_VERDICT: (APPROVE|REVISE|KILL)$`.
- **Critique focus (in order, distinct from ML peer-reviewer):**
  1. TRL progression credibility — Innovation Pathway must map defensible TRL 4 → 5 progression
  2. Budget-to-milestone alignment — phases sum to total; phase costs match phase work
  3. Open-data / no-classified compliance — every dataset must be open or synthetic
  4. Team capability vs proposed work — named members plausibly capable
  5. Risk mitigation completeness — research-direction threats actually mitigated, not just acknowledged
  6. YAGNI fence integrity — proposal doesn't claim what research-direction excluded
- **Explicit NON-criteria:** ML-conference rubric items (novelty bar, statistical significance, related-work depth) are NOT grounds for REVISE/KILL

### `agents/grant-reviser.md` (NEW, loop-dispatch)

- **Remit:** apply grant-reviewer feedback
- **Inputs:** `grant/draft-vN.md` + `grant/grant-review-vN.md` + grant-config.yaml
- **Outputs:** `grant/draft-v(N+1).md` + manifest + verification + append entry to `grant/grant-revision-log.jsonl`
- **Discipline:** cannot introduce new uncited claims; cannot change grant-config values; budget edits must preserve config total_usd

### Python helpers — `lib/paper_chain/`

| File | Responsibility |
|---|---|
| `lib/paper_chain/grant_config.py` | Load + validate `grant-config.yaml`. `load_grant_config(path) → dict`. `validate_grant_config(config) → (ok, errors)`. Validates required fields present, budget arithmetic correct, at least 1 team_member. CLI: `python3 -m lib.paper_chain.grant_config validate <path>`. |
| `lib/paper_chain/grant_verdict.py` | Parse `GRANT_VERDICT:` line from `grant-review-vN.md`. `parse_grant_verdict(path) → str \| None`. Separate from SP1's `verdict.py` so the two chains can't cross-talk. |
| `lib/paper_chain/grant_finalize.py` | Phase 12 logic. `finalize_grant(grant_dir, final_verdict) → Path` writes `proposal.md` (copy of latest draft) + `grant-history.md` (review/revision-log concat) + `pandoc-conversion.md` (the user's PDF command). |

`lib/paper_chain/regression.py` is reused as-is — `W<N>` tag tracking is format-agnostic.

### Orchestrator extension — `skills/executing-research-plan/SKILL.md`

- Parse `--grant` flag
- Pre-flight: when `--grant` set, require `grant-config.yaml` exists at `docs/research/runs/<run-id>/grant-config.yaml`; validate via `grant_config.validate_grant_config`
- Add Phase 10 / 11 / 12 sections (after Phase 9 if `--paper` also set; immediately after Phase 6 otherwise)

### New repo-root file — `grant-config.template.yaml`

Template with all required fields, blank values, inline comments. User copies to their run dir + fills.

```yaml
# IDEaS Competitive Projects — grant-config template
# Copy to docs/research/runs/<run-id>/grant-config.yaml and fill required fields.

# REQUIRED
pi_name: ""              # Principal Investigator full name
pi_org: ""               # Affiliated organization
submission_deadline: ""  # YYYY-MM-DD (e.g., 2026-06-02)

team_members:            # at least 1 required
  - name: ""
    role: ""             # e.g., Co-PI, Postdoc, Research Engineer
    relevant_experience: ""  # 2-3 sentence summary

budget:
  total_usd: 1500000     # IDEaS Competitive Projects band
  phases:                # must sum to total_usd
    - phase_name: ""     # e.g., "Months 1-4: Data and baseline"
      months: ""         # "1-4"
      total_usd: 0
      breakdown:
        personnel: 0
        equipment: 0
        contracts: 0
        other: 0

support_letters:         # may be empty
  - source: ""           # e.g., "Industry partner X" or "DRDC research group Y"
    status: ""           # "received" | "pending" | "in progress"

# OPTIONAL
project_management_organization: ""  # how the team is organized
existing_capabilities: ""            # paragraph on prior relevant work
partner_orgs: []                     # list of collaborator orgs
previous_ideas_proposals: []         # list of prior submissions if any
citations:                           # extra citations beyond research-direction's
  - ""                               # arXiv IDs or DOIs for grant-only claims
```

## Data flow

(Diagrammed in brainstorming Section 3; reproduced in implementation plan.)

Key state additions to `swarm-state.yaml`:

```yaml
phases:
  phase_10_grant_drafter:
    status: pending|completed|escalated
    output: grant/draft-v1.md
  phase_11_grant_review_loop:
    status: pending|completed|escalated
    rounds_completed: 1|2
    final_verdict: APPROVE|REVISE|KILL
    rounds:
      - round: 1
        review: grant/grant-review-v1.md
        revision: grant/draft-v2.md  # null if APPROVE this round
  phase_12_grant_finalize:
    status: pending|completed
    proposal: grant/proposal.md
```

## Error handling

| # | Failure | Detection | Response |
|---|---|---|---|
| 1 | `grant-config.yaml` missing | Pre-flight | Refuse: "Copy `grant-config.template.yaml` to `<run-id>/grant-config.yaml` and fill." No retry. |
| 2 | Required fields missing in config | `grant_config.validate` | Refuse with named missing fields. No retry. |
| 3 | Budget arithmetic mismatch in config | `grant_config.validate` | Refuse with delta. No retry. |
| 4 | grant-drafter missing artifacts | Per-worker gate | Re-dispatch once with feedback. After retry: escalate. |
| 5 | Drafter introduces uncited claims | Citation-integrity gate after Phase 10 | Re-dispatch once with offending IDs. After retry: escalate. |
| 6 | Drafter budget arithmetic mismatch in draft | Budget-arithmetic gate after Phase 10 | Re-dispatch once with specific phase numbers. After retry: escalate. |
| 7 | grant-reviewer returns `GRANT_VERDICT: KILL` | Verdict-line parse | Exit Phase 11. Skip Phase 12. Surface reasoning. Last draft preserved. |
| 8 | Cap-2 reached without APPROVE | Loop counter | Surface adjudication: accept last / continue manually / abandon. |
| 9 | Runaway revision (review-v2 new ≥ closed) | Reused `regression.py` | Append regression note. Surface to user pre-Phase 12. Don't auto-advance. |
| 10 | `--grant` set, no research-direction output.md | Pre-flight | Refuse: "Run /research-execute first." |
| 11 | `--grant` set, research-direction is gap-finding target | Pre-flight | **ALLOW** — drafter uses three-candidate shortlist as solution methodology fallback. |
| 12 | Reviser changes grant-config values | grant-reviser-verification + second budget gate | Re-dispatch reviser once with "budget edits must preserve config total." After retry: escalate. |

All escalations append `{worker, failure_code, retry_count, recommendation}` to `swarm-state.escalations`.

## Testing

### Layer 1 — Helpers, deterministic

- `tests/test_grant_config.py` — load + validate; missing fields per-row; budget mismatch; empty team_members; YAML parse errors
- `tests/test_grant_verdict.py` — APPROVE / REVISE / KILL parse; malformed; case-sensitivity
- `tests/test_grant_finalize.py` — latest-draft selection; history concat with revision-log; pandoc-conversion.md emitted

### Layer 2 — Worker contract tests (manual, expensive)

- `tests/contract/test_grant_drafter.py` — fixture: research-direction + eval-designer + minimal grant-config. Assert 3 artifacts, 12 IDEaS sections, citations subset, budget sums to total
- `tests/contract/test_grant_reviewer.py` — fixture: known-flawed grant-draft with 3 deliberate defects. Assert verdict regex; all 3 defects flagged; no ML-rubric penalties
- `tests/contract/test_grant_reviser.py` — fixture: grant-draft + grant-review with 3 weaknesses. Assert draft-v2 exists; revision-log has 3 entries; no new arXiv IDs; budget unchanged

### Layer 3 — End-to-end smoke test (manual, expensive)

`tests/manual_grant_dispatch.py` — full Phase 10-12 chain on SP1 fixture + hand-crafted grant-config. Asserts proposal.md exists, grant-history.md non-empty, no missing artifacts.

### Layer 4 — Manual discipline check (two-run)

**Run 1** — `--grant` on SP1 fixture (hypothesis-target, has eval-designer outputs). Inspect: 12 sections, budget math, TRL roadmap credibility, banned-phrase scan.

**Run 2** — `--grant` on Canadian defense example (gap-finding target, NO eval-designer outputs — the failure-#11 ALLOW case). Inspect: drafter falls back to three-candidate shortlist correctly; output is credible IDEaS draft; user could fill real budget + team and submit.

Results recorded in this spec's "Discipline check results" section.

### Test fixtures

- `tests/fixtures/grant-config-minimal.yaml` — valid minimal config
- `tests/fixtures/grant-config-malformed-budget.yaml` — phases sum ≠ total_usd
- `tests/fixtures/grant-config-missing-pi.yaml` — pi_name absent
- `tests/fixtures/known-flawed-grant-draft.md` — Canadian-defense-shaped draft with 3 deliberate defects
- `tests/fixtures/paper-chain/` — reused from SP1

## Implementation notes (handed to writing-plans)

- IDEaS section list in this design is hardcoded based on the official IDEaS Competitive Projects Application Form structure (as of 2025-2026 program cycle). If IDEaS updates the form, a maintenance task updates the drafter agent prompt — not in scope for SP6.
- Pandoc-to-PDF is OUT of scope; we emit the command in `pandoc-conversion.md` and the user runs it. PDF generation requires a LaTeX install (xelatex) which the chain doesn't validate.
- Phase 10's grant-drafter MAY consume `scoping/BRIEF.md` if present (the existing Canadian-defense example uses one). Drafter prompt reads it conditionally — absent → no error.
- Citation discipline: the drafter MAY cite from grant-config.citations (a user-provided list of extra arXiv IDs / DOIs) IN ADDITION to research-direction Sources. This lets users add team-publication citations not in the swarm output.
- The reused `regression.py` works because `W<N>` weakness tags are format-agnostic. grant-reviewer must use the same `W<N>:` convention as peer-reviewer.

## Open questions for writing-plans (none blocking)

- Should `grant-config.yaml` live in the run dir or repo root? Spec says run dir (consistent with run-scoped artifacts); plan author may add a fallback to repo-root path.
- Should the chain optionally re-validate budget against ACTUAL pandoc-rendered PDF page count (IDEaS form has page limits)? Out of scope for SP6 v1; flag as future-work.

## Decisions locked in

- 2026-05-15 · IDEaS Competitive Projects ONLY for SP6 · Generic Canadian grant generality deferred; lessons from SP6 inform any future generalization.
- 2026-05-15 · `--grant` flag on `/research-execute` (option i) · Composable with `--paper`. Independent chain.
- 2026-05-15 · Drafter + reviewer + reviser pattern (option c) · Mirrors SP1's loop discipline. TRL milestone generation inside drafter.
- 2026-05-15 · Pre-flight `grant-config.yaml` (option β) · Required-fields validated; missing-fields refuses with named errors.
- 2026-05-15 · Phases 10/11/12 after SP1's 7/8/9 if both set; immediately after Phase 6 if `--grant` alone.
- 2026-05-15 · Markdown only at ship; pandoc-conversion.md emitted but PDF generation OUT of scope.
- 2026-05-15 · Revision loop cap N=2, early-exit on APPROVE, same as SP1.
- 2026-05-15 · Grant chain ALLOWS gap-finding-target research-directions (the paper chain doesn't). Solution-methodology section falls back to research-direction's three-candidate shortlist. This is the key SP6 differentiator from SP1.
- 2026-05-15 · `regression.py` reused as-is (W-tag format-agnostic).
- 2026-05-15 · No banned phrases ("load-bearing", "this is doing a lot of work" + variants, "real" as emphatic adjective, "honest/honestly" as framing) in any agent output or doc.
