# Verification (revision-1)

Verification of `output.md` and `manifest.yaml` for hypothesis-smith-6 revision-1.

## Hypothesis structure checks

### If/then form
- ✅ §2 hypothesis statement begins "If [setup] … then [predicted observable outcome]"
- ✅ The "then" clause specifies: (a) compression ratio r = ΔA_TTT(K=4) / max(ΔA_other(K=4)) ≤ 0.5, and (b) |A_TTT(K=4) − A_η=0(K=4)| ≤ 1.5 abs pts
- ✅ Yes/no answers possible from finite experiment

### At least 3 falsification criteria, each genuinely falsifiable
- ✅ §5 lists F1–F5 (5 criteria, exceeds minimum of 3)
- ✅ F1 (compression ratio > 0.5): genuinely falsifiable, has powered statistical test (§5a)
- ✅ F2 (TTT–η=0 convergence at K=4 within 1.5 pts): genuinely falsifiable
- ✅ F3 (CoT releases redundancy by ≥0.2 in ratio difference): genuinely falsifiable
- ✅ F4 (depth-axis: r_d=1 < 0.7 OR r_d≥3 ≥ r_d=1): genuinely falsifiable, two-sided
- ✅ F5 (LaCT replicates compression to within +2.0 of TTT-Linear): genuinely falsifiable
- ✅ §5a includes pre-registered sample-size analysis with explicit noise-floor estimate (binomial+seed SEM ≈ 0.54 pts on ΔA, bootstrap CI on r ≈ ±0.3 well below 0.5/0.33 separation)

### Every mechanism claim has a citation
- ✅ M1 ("TTT is a learnable mapping that induces a linear-attention operator"): cites arXiv:2602.21204 §5.1 Theorem 5.1, §5.2
- ✅ M2 ("TRM's outer loop is iterative refinement of the same backbone-induced operator"): cites arXiv:2510.04871 §4.1, §4.5 Table 1
- ✅ M3 ("composition is sub-additive because both axes refine the same effective object"): cites arXiv:2407.04620 §2.1 (TTT trains different W_t per token), arXiv:2510.04871 §4.1 (TRM reruns backbone)
- ✅ M4 ("frozen-η=0 TTT control isolates the inner-loop-iteration property"): cites arXiv:2602.21204 §5.1 Theorem 5.1 corollary (η=0 ≡ static linear attention)
- ✅ M5 ("CRUXEval-X is the right test surface"): cites arXiv:2401.03065, arXiv:2408.13001, arXiv:2401.12947
- ✅ M6 ("non-additivity formalization"): self-deriving from M1+M3, no external claim needing citation
- ✅ Speculative-but-flagged content from revision-0 has been removed; the new §3 has no unflagged speculation

### All cited arxiv IDs verified by reading
- ✅ arXiv:2602.21204: read paper_details + §4 + §5 directly (load-bearing — every cited subsection verified)
- ✅ arXiv:2407.04620: read paper_details + §2.1 directly (Figure 4 citation tightened — see I2 fix)
- ✅ arXiv:2510.04871: read paper_details + §4 directly (Table 1, §4.1, §4.4 "less is more", §4.5 attention-on ablation)
- ✅ arXiv:2502.05171, arXiv:2510.25741: read paper_details (used for I1 scale-precedent grounding)
- ✅ All other cited arxiv IDs were verified in revision-0; carry-forward citations have not been re-fetched in revision-1 but are unchanged from revision-0 verification.

### Risks section non-empty (forces honest self-critique)
- ✅ §7 contains R1–R6:
  - R1: scale boundary (effect may only appear at ≥0.5B)
  - R2: η=0 may not be a clean linear-attention surrogate
  - R3: additive composition holds (the steelman from red-team §6)
  - R4: Mamba may not be a clean control (red-team I4)
  - R5: bimodality in seed distributions (red-team S3)
  - R6: TRM's "less is more" property may break entirely at 125M (red-team I1)

## Revision-specific checks (red-team round 1 objections)

| Objection | Severity | Addressed? | Where |
|---|---|---|---|
| C1 — 2602.21204 contradicts M3 mechanism | Critical | ✅ Pivoted to redundancy-substitution mechanism grounded in 2602.21204 §5.1-5.2's positive findings | "Changes from revision-0" §1 + §3 M1-M3 |
| C2 — F1-F5 statistically near-vacuous | Critical | ✅ Demoted variance metrics; new F1-F5 are first-moment with explicit power analysis (§5a). CRUXEval-X (15K items × 19 languages) replaces CRUXEval-O as primary | §2 hypothesis statement, §4 magnitudes, §5 criteria, §5a sample-size analysis |
| C3 — Architecture unspecified (reset vs persist) | Critical | ✅ PRIMARY = reset (TRM-faithful), ABLATION = persist (single-seed × 4 backbones at K=4) | §3 M2-M3, §6 "TRM operator wrapping" |
| I1 — Scale mismatch (TRM 5M, hypothesis 125M) | Important | ✅ Engaged via Huginn (3.5B), Ouro (2.6B), MoR (0.5B–7B) precedents at depth-recursion-at-scale; flagged as R6 risk; pre-registered kill-switch (TRM K=4 dense must beat non-recursion baseline) | "Changes from revision-0" I1, §7 R6, §6 sanity check |
| I2 — Figure 4 citation slip | Important | ✅ Tightened: Figure 4 = "loss reduces but cannot reach zero". Used 2407.04620 §2.1 end ("trains a different sequence of weights W_1, …, W_T for every input sequence") for W_t non-stationarity | "Changes from revision-0" I2, §3 M3 |
| I3 — Cohort size at d=3 unestimated | Important | ✅ Estimated and pre-registered: CRUXEval-X d=3 cohort ≥3K items (vs CRUXEval-O ≈100-150 items). Power analysis in §5a | "Changes from revision-0" I3, §5a |
| I4 — Mamba is not a clean control | Important | ✅ η=0 frozen-TTT control promoted from ablation to primary control alongside Mamba and dense; Longhorn 2407.14207 framing acknowledged in §3, R4 | §4 backbones table, §6 backbones, §7 R4 |
| I5 — Destructive framing chosen for non-additivity | Important | ✅ Pivoted to redundancy/substitution framing derived from 2602.21204 §5.2 mechanistic explanation. Non-additivity preserved (sub-additive saturation) but mechanism-driven, not shape-driven | "Changes from revision-0" I5, §3 M3, §3 M6 |
| S1 — §6a tests test-time-only, not training-time gradient flow | Suggestion | ✅ Revised §6a to fine-tune (2B tokens) rather than test-time wrap | §6a |
| S2 — Within-seed bootstrap CI for std estimates | Suggestion | ✅ §5a includes bootstrap CI methodology; mean-based criteria avoid std-of-std issue | §5 first paragraph, §5a |
| S3 — Bimodality test (Hartigan dip) | Suggestion | ✅ Pre-registered in R5 risk | §7 R5 |

All Critical and Important objections addressed. All Suggestions addressed.

## Discipline rules check

- ✅ **Single hypothesis** — H6-SUB is a single, coherent prediction
- ✅ **Architectural recursion** — yes, joint TRM × TTT
- ✅ **YAGNI** — pruned variance-amplification framing entirely; primary surface is one benchmark (CRUXEval-X) with one cross-check (CRUXEval-O); four backbones at one parameter scale
- ✅ **Citation** — every mechanism claim cited (per §3 M1-M5)
- ✅ **Coherence** — hypothesis statement, mechanism, predictions, falsification criteria all reference the same compression-ratio metric
- ✅ **Non-additivity** — preserved via sub-additive saturation prediction (compression ratio ≤0.5)
- ✅ **Falsifiability** — F1-F5 are all falsifiable in finite experiment; sample-size analysis in §5a shows the experiment is statistically powered

## Manifest checks

- ✅ `revision: 1` (incremented from 0)
- ✅ `falsifiable: true`
- ✅ `mechanism_grounded: true`
- ✅ `targeting_gap` matches red-team's gap reference (gap-finder-1 Gap 6)
- ✅ `addressed_red_team_round: 1` and `red_team_critique_path` populated
- ✅ `revision_notes` field added explaining the pivot
- ✅ `architecture_specification` field added (addresses C3)
- ✅ Hypothesis title updated from "H6-VAR" to "H6-SUB"

## Final assessment

Verification status: **PASS**

The hypothesis is defensible, falsifiable in a finite experiment, mechanism-grounded, and addresses every red-team objection. The pivot from destructive-interference to sub-additive-redundancy was forced by C1's correctness — the original mechanism could not survive 2602.21204 §4 — and the new mechanism is grounded in 2602.21204's own positive findings (§5.1-5.2). The metric pivot from second-moment (variance) to first-moment (mean ratios) addresses C2's statistical-vacuity objection at a 15K-item benchmark with pre-registered power analysis.

Submit to red-team round 2.
