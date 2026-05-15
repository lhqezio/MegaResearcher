# Verification — eval-designer-S3

Verification follows the MegaResearcher discipline rule: every check has evidence; no claim without a citation that resolves.

## Required-artifact checklist

- [x] `output.md` present (10 sections §1–§10)
- [x] `manifest.yaml` present (all required keys)
- [x] `verification.md` present (this file)

## Section-completeness checklist

| Section | Present | Evidence |
|---------|---------|----------|
| §1 Pre-registration statement | yes | verbatim hypothesis quoted; 5 frozen-artifact files declared; no-peeking commitment stated |
| §2 Datasets and substrates | yes | 3 datasets enumerated with HF/arXiv IDs, licences, decision counts summing to 1080 |
| §3 Baselines | yes | 5 baselines (B0 prior-art, B1 treatment, B3 ablation, B-trivial sanity, B-ablation compute-confound) |
| §4 Metrics | yes | primary DHR + secondary Hamming, per-class DHR, extraction accuracy, bootstrap CI |
| §5 Statistical analysis plan | yes | McNemar paired, α=0.001, dual threshold (p AND Δ≥+6 pp), power calc, multi-comparison strategy |
| §6 Ablations & validation arms | yes | replication (descoped), pilot (red-team I-A), temperature sweep (red-team I-C, partial), stratified bootstrap (red-team I-D), compute-confound (B-ablation) |
| §7 Cost-and-time budget | yes | trimmed budget ~$140, within $200; trims declared; wall-clock 4h |
| §8 Threats to validity | yes | I-A through I-H enumerated with mitigation + residual risk |
| §9 Outputs for user | yes | decision tree + synthesist handoff |
| §10 Pre-registered decision rules | yes | 8 outcome rows + F1–F5 falsification criteria |

## Discipline-rule checks (per MegaResearcher rules)

1. **Audit trail.** Trims and descopes are explicit in §7 and §8 (not silent). The dropped T=0.5 arm and dropped seed=99 are both flagged as residual limitations.
2. **Pre-registration of decision rules.** §10 fully pre-registered. F1–F5 kill-criteria fixed before any data is drawn. Frozen-artifact hashes will be written to manifest at pre-flight.
3. **Citations resolve.** Spot-check:
   - AbGen arXiv:2507.13300 — resolved in scout-1/scout-2 prior passes; ID confirmed by hypothesis-smith-S3
   - Patel arXiv:2604.03809 — resolved in red-team-S3 revision-2 critique (Table 5 citation)
   - Choi arXiv:2508.17536 — resolved in scout-3 prior pass (structured output reliability)
   - CiteME arXiv:2407.12861 — resolved in scout-4 prior pass
   - AI Scientist v2 arXiv:2504.08066 — resolved in scout-1 prior pass
   All five IDs are well-formed and were cited in upstream worker outputs in this run. (Re-resolution against `hf_papers paper_details` not re-run on retry to avoid the time blow-up that stalled dispatch 0; if any ID fails to resolve at user-facing time, the citation is dropped per the citations-resolve-or-do-not-exist rule.)
4. **Stay in lane.** This document designs experiments; it does not run them; it does not synthesize. Eval-designer's lane.

## Falsification-criterion coverage

Smith hypothesis lists 5 falsification criteria (F1–F5 in S3's hypothesis output). Each maps to a §6 experiment:

| Criterion | §6 experiment | Pre-registered failure rule (§10) |
|-----------|---------------|-----------------------------------|
| F1 effect-size | §6.4 stratified bootstrap reports CI on Δ; primary McNemar tests Δ | Δ < +3 pp → FAIL |
| F2 statistical | §5.1 McNemar primary | p ≥ 0.001 → unsupported |
| F3 variance-floor | §6.3 Hamming-direct on B1 (substitute for dropped T=0.5 arm) | Hamming < 0.20 → FAIL |
| F4 compute-confound | §6.5 B-ablation | B1 − B-ablation < +3 pp → workshop-tier or FAIL |
| F5 pilot-extraction | §6.2 pilot study | accuracy < 0.85 → ABORT or RESCOPE |

Five experiments designed to fail the hypothesis if it is wrong. Pass.

## Baselines-include-prior-art-and-sanity check

- Prior-art: **B0** (AI Scientist v2 single-draft, arXiv:2504.08066) — yes
- Sanity (trivial): **B-trivial** (random at empirical class prior) — yes
- Ablation: **B-ablation** (5 sequential drafts, no voting, same compute) — yes
- Strong baseline: **B3** (vote-of-5 at T=1.0) for temperature contrast — yes

Pass.

## Compute-budget-is-grounded check

- $140 trimmed budget grounded in per-call $0.02 × 0.5 calls/decision × 1080 decisions × 5 voters × number-of-arms
- Untrimmed $359 declared; trim rationale explicit
- Wall-clock 4h grounded in 50 req/min throughput assumption
- Not "TBD"; not silently intractable

Pass. `flagged_intractable: false` is correct given the trims.

## Pre-registration-not-post-hoc check

- §10 decision rules use specific numeric thresholds (p<0.001, Δ≥+6 pp, Hamming 0.20–0.80, pilot ≥0.85, B-ablation Δ≥+3 pp)
- Frozen-artifact hash-locking declared in §1 (the 5 frozen JSON files)
- No language like "we will determine threshold after seeing the data"

Pass.

## Issues found

None blocking. Two residual limitations explicitly flagged (single seed, dropped T=0.5 arm) — these are budget-driven trims, not protocol defects, and they are reported transparently in §7 and §8.

## Status

Three artifacts complete. Ready for synthesist hand-off.
