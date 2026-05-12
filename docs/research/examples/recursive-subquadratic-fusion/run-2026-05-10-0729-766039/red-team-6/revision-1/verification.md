# Verification — Red-team round 2 for H6-SUB

## Verification-before-completion checks

### 1. Independent literature queries (≥3 required)

Ran four queries via `mcp__plugin_megaresearcher_ml-intern__hf_papers` `search`, distinct from gap-finder-1's and round-1's:

| # | Query | Hits | Composes TRM × TTT? |
|---|---|---|---|
| 1 | `TTT Tiny Recursive Model joint composition sequence depth recursion` | 10 | No |
| 2 | `linear attention looped transformer iterative refinement code reasoning depth ablation` | 10 | No |
| 3 | `stacking inner loop outer loop test-time training redundancy ablation` | 10 | No (closest: 2604.21106 — recurrence-equivalence φ=0.46, no TTT) |
| 4 | `recursive depth iteration TTT learnable mapping backbone substitute saturation` | 10 | No |

**Verdict:** Gap claim survives. The closest adjacent paper (2604.21106) is a pretraining-loss recurrence-vs-parameters scaling law — does not vary backbone class, does not use TTT. Recommend smith add 2604.21106 to bibliography as adjacent reference (S1).

### 2. Citation spot-checks (≥3 required)

| # | Citation | Verified via | Match |
|---|---|---|---|
| 1 | 2602.21204 §5.1 Theorem 5.1 (TTT ≡ learned linear attention) | `read_paper` § 5 | ✓ Verbatim match |
| 2 | 2602.21204 §5.2 (inner loop is operator-inducing mapping, not storage) | `read_paper` § 5 | ✓ Verbatim match — supports smith's "wrapping at training time avoids train-test mismatch" framing |
| 3 | 2602.21204 §5.3 (LaCT also fits linear-attention reinterpretation) | `read_paper` § 5.3 | ✓ Verbatim match — F5 prediction is grounded |
| 4 | 2510.04871 §4.5 (TRM K-loop substitutes for self-attention) | `read_paper` § 4.5 | ⚠ OVERGENERALIZED — §4.5 says attention removal helps only on Sudoku-Extreme L≤D; "suboptimal for tasks with large context length, such as Maze-Hard and ARC-AGI." Important issue I3. |
| 5 | 2510.04871 §2.4 / §4.1 (deep supervision / full recursion process) | `read_paper` § 2.4, § 4.1 | ✓ Match for `n` and `T` definitions; ⚠ smith's K_arch axis ambiguous re. T vs n vs N_sup gradient flow (Important issue I4). |
| 6 | 2401.03065 §3 (CRUXEval = 800 functions from str/dict/list) | `read_paper` § 3 | ✓ Match — but see below for X-expansion factual issue. |
| 7 | 2408.13001 (CRUXEval-X 19K tests × 19 languages) | `paper_details` + `read_paper` § 3 | ✓ Total = 19K = ~800 base × 19. ⚠ Smith treats as 19× independent items (Important issue I1). |

### 3. Verdict-vs-severity consistency check

Verdict: APPROVE.
Critical objections: 0.
Important objections: 5 (I1–I5).
Suggestions: 3 (S1–S3).

APPROVE with 0 Critical and 5 Important is consistent: the round-1 Critical objections (C1, C2, C3) are all resolved by the mechanism pivot and the architecture commitment. The new Important objections are eval-designer-tractable and do not require another revision round. They concern (a) cohort-size accounting, (b) η=0 control specification, (c) TRM scope re-framing, (d) gradient-flow specification, (e) AST depth definition — all of which the eval-designer can specify in Phase 5 without changing the hypothesis text.

I would defend this hypothesis publicly: it has a defensible gap, a mechanism grounded in cited theorems, and a falsifiable F2 prediction (TTT-Linear K=4 ≈ TTT-η=0 K=4) that distinguishes from the strongest steelman ("TTT just has less headroom") via the predicted quantitative coincidence between the trained-inner-loop value at K=1 and the TRM-outer-iteration value at K=4 on η=0.

### 4. Round-1 Critical objection re-check

| Round-1 Critical | Status | Evidence |
|---|---|---|
| C1 — 2602.21204 contradicts memorization mechanism | ✓ RESOLVED via mechanism pivot from destructive interference to sub-additive redundancy. New mechanism cites §5.1 Theorem 5.1 directly. The smith's framing — "training-time wrapping avoids the train-test mismatch failure mode of 2602.21204 §4.1's inference-time sweep" — is internally consistent with §5.2. |
| C2 — F1-F5 statistically vacuous | ✓ Substantively addressed via mean-ratio metric on CRUXEval-X. F1 and F3 well-powered; F2 and F4-d=1 marginally powered; smith pre-registers escalation to 10 seeds in ambiguous zone. Residual issue (I1, language-correlation) is eval-designer-tractable. |
| C3 — Architecture unspecified | ✓ RESOLVED. Reset-between-outer-iterations is committed as primary; persist as ablation. |

### 5. Confidence level

High confidence in APPROVE. The mechanism pivot is real and the cited theorems support the new framing. The remaining issues are operational refinements that don't threaten the hypothesis's defensibility. If F2's tight prediction (|A_TTT-Linear K=4 − A_TTT-η=0 K=4| ≤ 1.5) holds in the experiment, the redundancy mechanism is positively supported. If it fails, the hypothesis cleanly falsifies — exactly what a falsification criterion should do.
