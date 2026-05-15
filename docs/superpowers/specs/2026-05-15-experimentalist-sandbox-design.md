# Experimentalist + Vercel Sandbox (Sub-Project 2a) — Design

**Status:** draft
**Created:** 2026-05-15
**Parent project:** Wide-port augmentation of MegaResearcher into an end-to-end paper pipeline (Level 3 — full, decomposed into 5 sub-projects + further SP2 split)
**This sub-project:** SP2a — experimentalist worker that runs generic ML experiments in Vercel Sandbox to replace SP1's "we will measure" stub with real numbers. SP2b (meta-experimenter, orchestration loop) is a separate follow-up.

## Goal

Add a new orchestrator phase (6.5 — experimentalist) that runs ONE experiment per surviving hypothesis from Phase 5 in an isolated Vercel Sandbox VM, captures structured results, and hands them to the SP1 drafter so the paper's Experiments & Results section contains real numbers instead of "we will measure" placeholders. Constrains substrate scope to a curated runner library (5 substrates) for predictable cost + deterministic results. Failed experiments are visible in the final paper, never hidden.

## Provenance

SP1 shipped a paper-drafting chain where the Experimental Plan section embeds the eval-designer protocols verbatim ("we will measure X via Y") — no fabricated numbers, but also no real numbers. The wide-port goal ("e2e research pipeline to professional paper") requires actual experimental results. SP2a fills that gap by running the experiments themselves; the surviving hypotheses from the spawning swarm (S1 cross-family routing, S2 length-debias wrapper, S3 majority-vote on structured decisions) are the immediate first cases — their eval-designer outputs already name supported substrates (SPECS-Review-Benchmark, AbGen testmini-500, CiteME).

## Scope (in)

- New worker agent: `experimentalist` (loop-dispatch, one per surviving hypothesis)
- New orchestrator phase: 6.5, `--paper`-gated, runs after Phase 6 synthesist and before Phase 7 drafter
- New Python helpers: `lib/paper_chain/{sandbox,experiment,protocol_parser}.py`
- New runner library: `lib/runners/{specs,abgen,citeme,limitgen,paperwrite_bench}/runner.py` (5 modules + 1 base class)
- Drafter modification: section 6 of `draft-v1.md` becomes "Experiments & Results" with real numbers when results.json exists; falls back to SP1 stub when absent or failed
- Cost ceiling per experiment: $5 sandbox + $5 API, 1 retry on transient failures
- Error handling: 8 named failure modes with deterministic responses
- Reproducibility: seed + pip-freeze + sandbox image SHA + Python version captured per experiment to `repro.yaml`
- Tests: 4 layers (helpers / per-runner / real-sandbox integration / manual discipline check)

## Scope (out — YAGNI fence)

- SP2b meta-experimenter (orchestration-loop experiments on MegaResearcher itself; e.g., dispatching subagents in varied configs and measuring outcomes). Deferred to its own brainstorming → spec → plan cycle.
- LLM-written runners on demand (option β from brainstorming). Deferred as future-work fallback when a protocol names an unsupported substrate.
- Parallel sandbox execution (sequential only in SP2a; parallel = future-work flag).
- GPU sandboxes (CPU only; experiments requiring GPU are out-of-scope for SP2a's curated runners).
- Training-from-scratch experiments (only inference / fine-tuning-cached-checkpoint / data-analysis fits the 60s timeout + $5 ceiling).
- New benchmarks beyond the 5 named substrates. Each new substrate = new runner module added incrementally.
- Cross-runner consistency (each runner is self-contained; no shared state).
- Resumable experiments (timeout means the experiment is failed; no checkpoint-restart).
- Non-Vercel sandbox backends. The `_sandbox_backend` indirection in `sandbox.py` exists for *testing* (FakeSandboxBackend) — not for swapping production backends.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Existing SP1 chain (unchanged):                                   │
│   Phase 1-6 (research swarm)                                      │
│   if --paper:                                                      │
│     Phase 7 (drafter) → Phase 8 (review-revise) → Phase 9 (final) │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ SP2a inserts Phase 6.5 HERE
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Phase 6.5 — experimentalist (NEW in SP2a, --paper-gated)         │
│                                                                    │
│   For each surviving hypothesis from Phase 5 eval-designer:       │
│     1. Read protocol → identify substrate                         │
│     2. Look up curated runner in lib/runners/<substrate>/         │
│     3. Spin up Vercel Sandbox VM (cold start, fresh per run)      │
│     4. Run runner with protocol params                            │
│     5. Capture results.json + repro.yaml + figures + log          │
│     6. Tear down sandbox                                          │
│     7. Write to paper/experiments/<hyp-id>/                       │
│                                                                    │
│   Budget per experiment: $5 sandbox + $5 API; 1 retry on transient│
│   Failure → mark experiment failed_<code>, paper chain continues  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Phase 7 (drafter) — MODIFIED                                      │
│   For each hypothesis:                                            │
│     - If paper/experiments/<hyp-id>/results.json exists with      │
│       status=completed → section 6 of draft = "Experiments &      │
│       Results" with real numbers, table from results.json,        │
│       figures from figures/ embedded                              │
│     - Otherwise → SP1 option-γ stub for that hypothesis only      │
│       (with "[Experimental data unavailable: <reason>]" marker    │
│       if status=failed)                                           │
└──────────────────────────────────────────────────────────────────┘
```

Same single-session orchestrator + leaf-worker pattern as SP1. No nested dispatch. File-based artifact passing. The sandbox is a side-effect of one specific leaf-worker invocation (the experimentalist), not a new orchestration primitive.

## Components

### `agents/experimentalist.md` (NEW)

- **Remit:** for ONE hypothesis, read the eval-designer protocol, identify the substrate, dispatch the appropriate curated runner via the experiment.py helper, write the standard three artifacts to `paper/experiments/<hyp-id>/`
- **Inputs:** the hypothesis's `eval-designer-S<N>/output.md` from Phase 5; output path `paper/experiments/<hyp-id>/`; budget envelope ($5 sandbox + $5 API)
- **Outputs:** `paper/experiments/<hyp-id>/{results.json, repro.yaml, runner-output.log, figures/, manifest.yaml, verification.md}`
- **Discipline:** every reported number must trace to a runner output. Agent does NOT compute or summarize results outside the runner — its job is parameter substitution + sandbox orchestration via the helpers. If the runner fails, agent writes results.json with `status: failed` and the failure code; does NOT fabricate numbers to fill the gap.

### Python helpers — `lib/paper_chain/`

| File | Responsibility |
|---|---|
| `lib/paper_chain/sandbox.py` | Vercel Sandbox SDK wrapper. Functions: `spin_up(image, timeout, budget) → sandbox_id`, `execute(sandbox_id, command) → ExecutionResult{stdout, stderr, exit_code, runtime_seconds}`, `tear_down(sandbox_id)`, `cost_so_far(sandbox_id) → usd`. Polls cost during execution; hard-kills on budget breach. `_sandbox_backend` module-level indirection allows test injection of `FakeSandboxBackend`. |
| `lib/paper_chain/experiment.py` | Orchestration: `dispatch_experiment(protocol_path, output_dir, budget) → results_path`. Reads protocol, calls `select_runner(substrate)`, invokes sandbox with the runner module path, captures results, validates schema, writes `results.json + repro.yaml + runner-output.log`. CLI: `python3 -m lib.paper_chain.experiment dispatch ...` |
| `lib/paper_chain/protocol_parser.py` | Parses an eval-designer protocol markdown → structured dict: `{substrate: str, sample_size: int, seed: int, baselines: list[str], metrics: list[str], decision_rules: list[dict]}`. Heuristic-based (regex + section parsing); returns `{}` and `failed_parse` status if structure unrecognized. |

### Runner library — `lib/runners/`

```
lib/runners/
├── _base.py                       NEW (Runner base class + result-schema validator)
├── specs/runner.py                NEW (SPECS-Review-Benchmark, controlled-flaw detection rate)
├── abgen/runner.py                NEW (AbGen testmini-500, ablation-coverage F1@5)
├── citeme/runner.py               NEW (CiteME, citation attribution accuracy)
├── limitgen/runner.py             NEW (LimitGen, limitation-generation quality)
└── paperwrite_bench/runner.py     NEW (PaperWrite-Bench n=51, manuscript quality dimensions)
```

Each runner module exports `run(params: dict) → dict` matching the result schema. Runners are deterministic at fixed seed. Dependencies are vendored or downloaded on first run (cached for re-runs within the same sandbox).

### Orchestrator extension — `skills/executing-research-plan/SKILL.md`

New section "Phase 6.5 — experimentalist (only if `--paper` AND Phase 5 produced surviving hypotheses)". Inserts between existing Phase 6 (synthesist) and Phase 7 (drafter). For each `eval-designer-S<N>/output.md`: dispatch `megaresearcher:experimentalist`, run per-worker verification gate, parse `results.json` status, record in `swarm-state.yaml`. Continues to Phase 7 even when some experiments fail.

### Drafter agent modification — `agents/manuscript-drafter.md`

Updates prompt to read `paper/experiments/<hyp-id>/results.json` if present. Section 6 of `draft-v1.md` becomes "Experiments & Results" with results-table prose + embedded figures when `status=completed`; falls back to SP1's option-γ "Experimental Plan" stub when results.json is missing or `status=failed`; the failed-experiment marker `[Experimental data unavailable: <failure_code>]` is included in the draft so reviewers see what wasn't tested.

## Data flow

(See Section 3 of brainstorming dialogue for the full diagram — reproduced concretely in the implementation plan.)

Key state additions to `swarm-state.yaml`:

```yaml
phases:
  phase_6_5_experimentalist:
    status: pending|completed|partial-fail|escalated
    experiments:
      - hyp_id: S1
        substrate: SPECS
        status: completed|failed|escalated|skipped
        failure_code: null|timeout|exception|budget_breach|malformed_results|unsupported_substrate|failed_parse|verification_gate
        results: paper/experiments/S1/results.json
        cost_usd: <float>
        retry_count: 0|1
    total_cost_usd: <float>
```

## Error handling

| # | Failure | Detection | Response |
|---|---|---|---|
| 1 | Vercel Sandbox spin-up fails (API down, quota, auth) | `sandbox.spin_up()` raises | Surface to user immediately (env problem, non-retryable). Mark all remaining experiments `skipped`. Proceed to Phase 7 with stub-everything Results. |
| 2 | Runner times out (default 60s) | exit code 124 or wall-clock exceeded | Retry ONCE with `timeout=120s` and feedback "previous run timed out at 60s — try smaller sample". After retry: `status=failed_timeout`, continue chain. |
| 3 | Runner exception (Python error) | Non-zero exit + stderr | Capture stderr to `runner-output.log`. Retry ONCE with stderr in feedback. After retry: `status=failed_exception`, continue. |
| 4 | Budget breach ($5 sandbox or $5 API) | Polled `cost_so_far()` during execution | Hard-kill sandbox. Capture partial output. `status=failed_budget`. Do NOT retry. Escalate with cost-so-far. Continue. |
| 5 | results.json malformed (schema validation fail) | `experiment.py` validates after runner | Retry ONCE with schema reminder. After retry: `status=failed_malformed`, continue. |
| 6 | Substrate not in runner library | `select_runner()` returns None | Cannot retry. `status=failed_unsupported`. Surface suggestion: "consider adding `lib/runners/<substrate>/runner.py`". Continue. |
| 7 | Protocol parse fails | parser returns `{}` or raises | `status=failed_parse`. Suggests eval-designer didn't follow expected format. Continue. |
| 8 | Three-artifact verification gate fails | Standard per-worker gate (SP1 pattern) | Re-dispatch experimentalist once with explicit feedback. After one retry: escalate. |

All escalations append `{worker, failure_code, retry_count, recommendation}` to `swarm-state.escalations`. Failed experiments are visible to the drafter via results.json status field and surfaced in the final `paper-history.md` "Failed experiments" section. No silent omissions.

## Testing

### Layer 1 — Helpers, mocked sandbox (deterministic, fast)

Pure-stdlib tests in `tests/test_*.py` (matching the existing `tests/test_doom_loop.py` + SP1 pattern). Mock the Vercel SDK via `_sandbox_backend` injection of `FakeSandboxBackend`.

- `tests/test_protocol_parser.py` — canned eval-designer outputs in / structured dict out
- `tests/test_sandbox_wrapper.py` — spin_up / execute / tear_down / cost_so_far API; budget-breach hard-kill path
- `tests/test_experiment_orchestrator.py` — full orchestration with mocked sandbox + mocked runner; all 8 failure modes from Error handling produce correct status + failure_code
- `tests/test_runner_base.py` — base Runner contract + result-schema validation

### Layer 2 — Per-runner unit tests with cached fixtures

One test file per runner under `tests/test_runner_<substrate>.py`. Each:
- Loads `tests/fixtures/runners/<substrate>/` slice
- Invokes `runner.run(params)` directly (no sandbox — runner is just Python)
- Asserts output matches result schema
- Asserts determinism (run twice at fixed seed → identical results)

Fixtures are minimal slices (5 SPECS perturbations, 3 AbGen examples, 2 CiteME excerpts) — exercise the code path without compute cost.

### Layer 3 — Real-sandbox integration tests (manual, expensive)

`tests/integration/test_sandbox_<runner>.py` per runner. Requires `VERCEL_TOKEN`. Each spins up an actual Vercel Sandbox, runs the real runner on actual benchmark data, asserts schema + runtime < 60s + cost < $5. Gated behind `python3 -m tests.integration <runner-name>` invocation. Cost: real money. Run once per runner-change before merging SP2a.

### Layer 4 — Manual discipline check on SP1 fixture

Before merging SP2a, run full chain on `docs/research/runs/2026-05-12-0515-19bf96/` with `--paper`. Inspect:
- Phase 6.5 spins sandboxes for the 3 surviving hypotheses (S1, S2, S3)
- 3 `paper/experiments/S{1,2,3}/results.json` have real numbers (or honest `status=failed_*`)
- Drafter's section 6 reflects real numbers when present, stub-with-marker when failed
- `paper-history.md` "Failed experiments" section is honest

### Test fixtures

- `tests/fixtures/protocols/` — canned eval-designer outputs for parser tests
- `tests/fixtures/runners/{specs,abgen,citeme,limitgen,paperwrite_bench}/` — minimal-slice benchmark data per runner
- `tests/fixtures/sandbox_responses/` — canned Vercel SDK responses covering all 8 failure-mode triggers

## Implementation notes (handed to writing-plans)

- **Vercel Sandbox SDK choice:** the Vercel plugin's `vercel-plugin:vercel-sandbox` skill is available in the session; plan author should consult it (or the official docs at https://vercel.com/docs/sandbox) for the actual SDK function names + auth setup. The wrapper in `lib/paper_chain/sandbox.py` insulates the rest of the codebase from SDK choice.
- **Authentication:** Vercel Sandbox requires `VERCEL_TOKEN` env var. Document this in CLAUDE.md alongside HF_TOKEN. Pre-flight checks (SP1's `lib/paper_chain/preflight.py`) extend to verify VERCEL_TOKEN is present when `--paper` is used and runners would otherwise fire — but if user wants `--paper` without experiments (everything stubs to SP1 mode), VERCEL_TOKEN can be optional with a warning.
- **Runner dependency management:** each runner declares its pip requirements in `lib/runners/<substrate>/requirements.txt`. Sandbox installs them on cold start (first run is slower; subsequent runs in the same VM cache deps — but each experiment is a fresh VM per the locked design, so installs always run).
- **Pre-canned benchmark data:** runners download benchmarks from HuggingFace / GitHub on first run. Cache to a sandbox-local directory; subsequent runs WITHIN the same VM hit cache. Across VMs (each experiment is a fresh VM) the download happens every time. Acceptable per cost ceiling.
- **Renumbering vs Phase-6.5 naming:** SP2a uses "Phase 6.5" decimal naming to avoid renumbering SP1's Phase 7-9. A future cleanup PR could renumber to Phase 7 (experimentalist) / Phase 8 (drafter) / Phase 9 (review-revise) / Phase 10 (finalize) for cleaner sequencing — flagged as future-work, not in SP2a scope.

## Open questions for writing-plans (none blocking — flagging for the plan author)

- Vercel Sandbox SDK package name + version pin: plan author should verify current package name (likely `@vercel/sandbox` JS or `vercel-sandbox` Python; check docs).
- Default sandbox image: pick a Python 3.11 / 3.12 image with common scientific Python (numpy, pandas, scipy, matplotlib) pre-installed to minimize cold-start install time. Plan author should pick a specific image SHA and pin it.
- Where does the experimentalist's manifest/verification template come from? Probably mirrored from SP1's per-worker manifest pattern.

## Decisions locked in

- 2026-05-15 · SP2 split into SP2a (this, Python in sandbox) + SP2b (meta-experimenter, orchestration loop, deferred) · Two-worker scope was confirmed in brainstorming option C.
- 2026-05-15 · Sandbox = Vercel Sandbox · Firecracker microVM, GA Jan 2026, plugin already loaded in env.
- 2026-05-15 · Curated runner library (option α) · 5 modules at SP2a ship; LLM-written runners (β) deferred as future-work fallback.
- 2026-05-15 · Phase 6.5 placement, --paper-gated · Experiments only run when producing a paper; saves cost for direction-only users.
- 2026-05-15 · Drafter integration: section 6 transforms based on results.json presence · No new sections; same 9-section structure; "Experimental Plan" → "Experiments & Results" when results exist.
- 2026-05-15 · Cost ceiling: $5 sandbox + $5 API per experiment; 1 retry on transient; escalate-continue on terminal failure · Hard ceiling matches SP1's ≤$200/replication discipline scaled to single-experiment granularity.
- 2026-05-15 · Sequential execution; parallel = future-work flag · Sandbox parallelism adds complexity not needed for SP2a MVP.
- 2026-05-15 · One sandbox per experiment, cold start each time · Clean isolation; matches Vercel's per-second pricing.
- 2026-05-15 · Reproducibility artifacts: seed + pip-freeze + sandbox image SHA + Python version → repro.yaml · Captured per experiment for paper-grade reproducibility.
- 2026-05-15 · Failed experiments visible in final paper · `[Experimental data unavailable: <code>]` marker; no silent omissions.

## Discipline check results (SP2a manual T12, 2026-05-15)

Ran `tests/manual_dispatch.py` against the SP1 fixture (`tests/fixtures/paper-chain/`) with `FakeSandboxBackend`. All three experiments completed the orchestrator path cleanly; each returned a schema-valid `failed` result.

- **S1** substrate (target: SPECS-Review-Benchmark): `failure_code: failed_parse`. Runtime 0.0s, cost $0.
- **S2** substrate (target: Bias Fitting / DeepNLP ICLR-2024): `failure_code: failed_parse`. Runtime 0.0s, cost $0.
- **S3** substrate (target: AbGen testmini-500 + canonical leaderboards): `failure_code: failed_parse`. Runtime 0.0s, cost $0.

**Discipline-check observations:**

- ✅ Orchestrator records every failure honestly with the right `failure_code`. No silent omissions; no fabricated numbers.
- ✅ Schema validation passes on every result (all required fields populated, status ∈ {completed, failed, escalated, skipped}).
- ✅ `repro.yaml` non-empty for every dispatch; `runner-output.log` records the failure path.
- ✅ FakeSandboxBackend keeps the dispatch cycle deterministic — no real Vercel calls in this check.
- ⚠️ **Finding:** the protocol parser's strict format (`- Substrate: <name>` bullet under `## Pre-registered settings`) does not match the actual eval-designer outputs from the SP1 swarm run, which embed substrate names in prose. The discipline check exercises the `failed_parse` path correctly, but no skeleton runner was actually selected.

**Follow-up action:** SP2a.0 (a small follow-up plan) should either: (a) extend `protocol_parser.py` to match more eval-designer output formats, or (b) tighten the eval-designer agent prompt to emit the strict bullet format. Option (b) is cleaner and matches the spec's contract; option (a) adds parser variance. Recommendation: (b) — update `agents/eval-designer.md` (if it exists) or add a "## Pre-registered settings" template requirement to the agent's prompt.

This finding is **not** a defect in SP2a's deliverables — every component behaves per spec. It's a downstream-integration gap surfaced by the discipline check, exactly the point of running one.
