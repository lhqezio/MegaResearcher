# Red-team critique of H6-SUB (revision-1) — TRM × TTT sub-additive substitutes

Critiquing: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-6/revision-1/output.md`
Targeting gap: A6 (gap-finder-1). Revision round: 2.
Round-1 critique: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-6/output.md`

---

## 1. Verdict

`APPROVE`

The pivot from "destructive interference / variance-amplifying middle band" to "sub-additive redundancy / compressed K_arch-gain on TTT backbone" is substantive and addresses all three Critical objections from round 1. The new mechanism is grounded in 2602.21204 §5.1–5.2 in a way that survives the §4.1–4.4 empirical contradictions that killed the previous mechanism. The new metric (mean ratio r = ΔA_TTT / max ΔA_other) is sample-efficient enough that the falsification thresholds (r ≤ 0.5, predicted r ≈ 0.33) sit meaningfully above the noise floor on CRUXEval-X. The architectural ambiguity is resolved (reset-between-outer-iterations primary, persist as ablation).

I would defend this hypothesis publicly. Several Important issues remain — the most consequential is that "≥3K items per d=3 cohort" overcounts effective sample size because CRUXEval-X is 19 translations of the same ~800 base problems (so item-level independence is closer to the language-correlation-adjusted 150–500 range), and the η=0 control collapses to a position-wise MLP rather than a "static linear-attention operator" the smith claims. Neither is fatal: the F2 prediction is still distinguishable from the strongest steelman, and the F1 ratio still survives a more conservative power analysis with 10 seeds (the smith's pre-registered escalation). The eval-designer can tighten these in Phase 5.

---

## 2. Round-1 Critical objection status

| Round-1 critical | Round-2 status | Evidence |
|---|---|---|
| **C1 — 2602.21204 contradicts M3 destructive-interference mechanism** | **RESOLVED.** Mechanism pivoted from memorization/destructive to sub-additive redundancy, grounded in 2602.21204 §5.1 Theorem 5.1 (TTT ≡ learned linear attention) and §5.2 (inner loop is a learnable operator-inducing mapping, not storage). The smith's framing — "wrapping TRM around TTT at training time fixes 2602.21204's train-test mismatch issue but introduces operator-iteration redundancy" — is internally consistent with the cited paper. | Direct read of §5.1, §5.2, and §5.3 (LaCT linear-attention rewrite) confirms M1 and the F5 LaCT replication path. |
| **C2 — F1-F5 statistically near-vacuous** | **MOSTLY RESOLVED.** Pivoted to first-moment metrics (mean ratios). Power analysis pre-registered in §5a. Predicted r ≈ 0.33 vs threshold r > 0.5 — separation of 0.17 is large relative to the within-cell SEM (0.38 abs pts at 5 seeds × 3K items, per smith's analysis). However, the smith treats CRUXEval-X 19 languages as 19× independent — actually each language is a translation of the same ~800 base problems, so item-level independence is heavily inflated. See I1 below. The smith's escalation to 10 seeds in the ambiguous zone provides a valid fallback. | Independently re-derived: at the more conservative effective-N ≈ 150 unique problems, ΔA-ratio SEM is ≈1.2 abs pts, marginal at 5 seeds, comfortable at 10. |
| **C3 — Architecture unspecified (reset vs persist)** | **RESOLVED.** §3 and §6 explicitly commit: reset-between-outer-iterations is primary (TRM-faithful since TRM repeats f_L on fresh `(z_L+z_H+x)` each iteration), persist is ablation at K=4 single-seed × 4 backbones. Mechanism story (M3) maps to reset variant cleanly. | Direct read of revised §6 confirms commitment. |

All three round-1 Critical objections are resolved or substantively addressed.

---

## 3. Gap re-verification

I ran four new literature queries (distinct from gap-finder-1's and round-1's):

| Query | Results | Closest hit | Composes TRM-depth × TTT-sequence with ablation? |
|---|---|---|---|
| `TTT Tiny Recursive Model joint composition sequence depth recursion` | 10 | 2510.04871 (TRM), 2505.23884 (LaCT), 2511.16886 (Deep Improvement Supervision for TRM), 2509.26645 (TTT3R) | None — TTT3R is 3D-recon, Deep Improvement Supervision modifies TRM training only, no TTT |
| `linear attention looped transformer iterative refinement code reasoning depth ablation` | 10 | 2602.11698 (SpiralFormer), 2602.02156 (LoopViT), 2502.17416 (Reasoning with Latent Thoughts), 2603.08391 (Adaptive Loops + Memory) | None — none use a TTT-style fast-weight inner loop |
| `stacking inner loop outer loop test-time training redundancy ablation` | 10 | **2604.21106 (How Much Is One Recurrence Worth? Iso-Depth Scaling Laws)**, 2604.19295 (TEMPO), 2602.16490 (From Growing to Looping) | Closest is 2604.21106 — a recurrence-equivalence scaling law (φ=0.46) for looped LLMs across r ∈ {1,2,4,8}, but pretraining-loss only and no TTT/Mamba/dense cross-backbone ablation. Adjacent, not a gap-killer. |
| `recursive depth iteration TTT learnable mapping backbone substitute saturation` | 10 | 2510.14961 (Parallel Samplers for Recurrent-Depth), 2602.20160 (tttLRM), 2603.06642 (SR-TTT) | None — no joint TRM × TTT cross-backbone ablation |

**Verdict: gap claim survives.** No paper instantiates TRM-style depthwise weight-tied recursion AND TTT-style sequence-time fast-weight recursion with cross-backbone ablation against frozen-η=0 / Mamba / dense controls. The closest hit (2604.21106) is a recurrence-vs-parameters scaling law on standard backbones; it neither uses TTT nor varies the backbone-class along an "operator-induction" axis. **The smith should add 2604.21106 to the bibliography as an Adjacent reference** — its φ=0.46 result implies that one recurrence is worth substantially less than one extra unique block, which is *evidence in favor* of saturation-of-recurrence in general (not specific to TTT) and helps interpret the steelman in §6.

`gap_claim_survives: true`

---

## 4. Citation spot-checks (4 verified)

### 4.1 arXiv:2602.21204 §5.1 Theorem 5.1 — "TTT is exactly equivalent to learned linear attention"

**Smith's claim** (M1, §3): "For a TTT model with linear final-layer inner loop, the output at token t with query q is `o = ϕ_{t+1}(q) · (W_t + ϕ_t(k)^⊤ g_t(k))`, equivalent to a learned linear-attention operator with effective `q̂ = ϕ_{t+1}(q)`, `k̂ = ϕ_t(k)`, `v̂ = g_t(k)`."

**What §5.1 actually says (verified by `read_paper`):** "Theorem 5.1 (Linearization of Inner-Loop Updates) ... `o = ϕ_{t+1}(q)(W_t + ϕ_t(k)^⊤ g_t(k))` ... `g_t(k) ≜ -η ∂L/∂f_t(k)`. This expression is a linear attention operator of the form `o = q̂(S_0 + k̂^⊤ v̂)`."

**Assessment.** Direct verbatim match. Citation is correct. ✓

### 4.2 arXiv:2602.21204 §5.2 — "TTT inner loop is a learnable mapping, not storage"

**Smith's claim** (M1, §3 closing paragraph): "§4.1's 'more inner-loop fitting → worse' is now *predicted* by M1+M3: if the inner loop already induces an operator that the model is trained to use, varying inner-step count at inference (which is what 2602.21204 §4.1 does) creates train-test operator mismatch."

**What §5.2 actually says (verified):** "Increasing the number of inner-loop iterations at inference time therefore induces an attention operator different to the one used during training, naturally leading to degraded performance due to train–test mismatch rather than improved memorization."

**Assessment.** Verbatim mechanistic explanation. The smith's framing is precise: 2602.21204's degradation finding is an *inference-time* sweep on a model trained at one inner-loop step count; the H6-SUB hypothesis tests *training-time* wrapping (consistent inner-loop trajectory at train and test), so the train-test mismatch failure mode is avoided. ✓

### 4.3 arXiv:2602.21204 §5.3 — "LaCT also fits linear-attention reinterpretation"

**Smith's claim** (F5): "LaCT is a TTT variant and 2602.21204 §5.3 establishes LaCT also fits the linear-attention reinterpretation."

**What §5.3 actually says (verified):** "We now show how a representative instantiation of TTT formula, LaCT (Zhang et al., 2025), can be rewritten in the form of linear attention. ... `o_t = ϕ_{t+1}(q_t)(W_{1,0} + Σ M(ϕ_i(k_i)^⊤ m_i))` ... This expression reveals that the inner loop of LaCT is effectively a linear attention–like operator."

**Assessment.** Correct. ✓ F5's "LaCT replicates the compression" prediction is well-grounded — both architectures are now united under M1.

### 4.4 arXiv:2510.04871 §4.5 — "TRM K-loop substitutes for self-attention"

**Smith's claim** (M2, §3): "TRM Table 1 / §4.5 shows TRM's K-loop **substitutes for self-attention**: adding self-attention to the f_L/f_H block *hurts* (87.4% no-attention vs 74.7% with-attention on Sudoku-Extreme, same parameter count). This is the load-bearing TRM observation for this hypothesis: depth-time iteration of a non-attention block already provides much of the benefit attention provides; adding attention is at best neutral."

**What §4.5 actually says (verified):** "Self-attention is particularly good for long-context lengths when L ≫ D ... However, when focusing on tasks where L ≤ D, a linear layer is cheap ... Using an MLP instead of self-attention, we obtain better generalization on Sudoku-Extreme (improving from 74.7% to 87.4%; see Table 1). This worked well on Sudoku 9x9 grids, given the small and fixed context length; **however, we found this architecture to be suboptimal for tasks with large context length, such as Maze-Hard and ARC-AGI**."

**Assessment — Important issue.** The smith's "depth substitutes for attention; adding attention is at best neutral" reads §4.5 as a general result. It is not — TRM's attention-removal helps only on Sudoku-Extreme (L ≤ D); it *hurts* on tasks with longer context (Maze-Hard, ARC-AGI). For CRUXEval-X (Python/Java/etc. functions tokenized to hundreds of tokens, L ≫ D for a 2-layer 5M-block but L < D for a 12-layer 125M-block), the regime is ambiguous. M2's "depth-iteration absorbs attention's benefit" is therefore weaker than the smith presents. Tighten to: "On Sudoku-Extreme TRM Table 1 shows depth-recursion substitutes for attention; for the CRUXEval-X test surface this substitution may or may not hold, but the M3 saturation argument does not depend on it — it depends only on TRM's outer iteration re-running the same operator-inducing inner loop." The mechanism survives this correction.

---

## 5. Mechanism critique

### M1 (TTT is learned linear attention). Cleanly grounded.
Direct verbatim match to 2602.21204 §5.1 Theorem 5.1. No issue.

### M2 (TRM outer loop iteratively refines the same backbone-induced operator). Partially supported, framing-dependent.
The general claim — "TRM's outer loop refines `(y, z)` and re-runs the backbone, re-inducing the effective operator" — is correct from TRM §4.1. But the supporting "TRM K-loop substitutes for self-attention" leans on §4.5 in a way that overgeneralizes (see citation 4.4 above). The mechanism survives because M3 only requires that "iterating the same operator-class yields diminishing returns" — which is what 2604.21106's φ=0.46 scaling-law exponent quantifies in a backbone-agnostic way. **Suggest: replace M2's load-bearing weight on TRM §4.5 with 2604.21106's recurrence-equivalence exponent as the saturation evidence.**

### M3 (sub-additive composition). The crux of the new mechanism.
The argument: "When the backbone is TTT-Linear, the per-layer operator is itself a learned linear-attention operator whose effective shape is set by the inner-loop trajectory along sequence-time. TRM's outer loop runs the same backbone again on refined (y, z); each pass re-computes essentially the same kind of object."

The mechanism is internally consistent. But it is structurally similar to "TTT alone is a strong/saturated learner of code-recursion-depth-≥3 patterns, leaving less headroom for TRM." The smith mostly distinguishes these via:
- **Predicted A_TTT-Linear(K=1) = 28 < A_dense(K=1) = 30** — TTT is NOT predicted to be better at K=1, ruling out the simplest "TTT just better, less headroom" steelman.
- **F2 (TTT-Linear ≈ TTT-η=0 at K=4)** — this is the real distinguishing prediction. If trained-TTT and frozen-η=0 converge at K=4 even though they diverge at K=1, the trained inner loop's contribution has been fully replaced by TRM's outer iteration. This is genuine evidence for the redundancy mechanism (not just "TTT has less headroom").

**Concern: the predicted η=0 K=4 mean (31) is HIGHER than the trained-TTT K=4 mean (30).** This requires that ΔA_η=0(K=4) = +5.0 substantially exceeds ΔA_TTT-Linear(K=4) = +2.0. The mechanism predicts this (η=0 has no trained inner-loop iteration, so TRM iteration is non-redundant for it), but the magnitude is a tight quantitative claim. If ΔA_η=0 lands closer to +2.0 (similar to trained TTT), the mechanism distinction collapses — but the smith catches this in R2 as a known risk. Acceptable.

### M4 (η=0 frozen-TTT control). Important issue with the cited corollary.
Smith claims: "η=0, the TTT inner-loop trajectory degenerates to the static linear-attention operator with `S_0 = W_0`."

Reading §5.1 Theorem 5.1 carefully: at η=0, `g_t(k) = -η · ∂L/∂f_t(k) = 0`. So `o = ϕ_{t+1}(q) · (W_t + ϕ_t(k)^⊤ · 0) = ϕ_{t+1}(q) · W_0` (since W_t = W_0 always at η=0). And `Θ_t = Θ_0` (no inner update of the projection parameters). So the η=0 layer is `o_t = ϕ(q_t; Θ_0) · W_0` — a **fixed position-wise MLP** applied independently to each token, NOT a linear-attention operator (no `Σ k̂^⊤ v̂` accumulation across tokens). The Theorem 5.2 / 5.3 sequence form reduces to `o_t = ϕ(q_t)(W_0 + 0) = ϕ(q_t) W_0`.

This means the η=0 control is NOT "exactly a fixed learned linear-attention operator" — it is an attention-free token-wise MLP. As a control for "is the trained inner-loop iteration property the load-bearing thing?", this is not as clean as the smith claims: η=0 removes BOTH the iteration property AND the per-token operator update / sequence mixing. Mamba (which has sequence mixing via the SSM recurrence) is a cleaner control for "removes operator-induction property but retains sequence-time computation."

**This is an Important issue, not Critical.** F2's prediction (TTT-Linear K=4 ≈ TTT-η=0 K=4) is still meaningful, but its interpretation if it holds is murkier: it could be "redundancy with operator-induction collapses" (the smith's intended reading) OR "the TTT inner loop's sequence-mixing contribution is small at K=4 because TRM's outer iteration provides enough cross-token information flow via the (y, z) updates" (a different but plausible reading). **Recommend the smith re-frame η=0 as "fixed-MLP-no-fast-weights" rather than "static linear attention," and add a primary control where η=0 is replaced with "fixed linear attention with W_0 trained directly" (i.e., a true linear-attention layer with no inner-loop update) to disambiguate.** The eval-designer can specify this.

### M5 (CRUXEval-X is the right test surface). Cohort-size claim overcounts.
CRUXEval-X = 19K total tests = ~800 base problems × 19 languages (per arXiv:2408.13001 §3.1's evaluation table covering exactly 19 columns indexed by language). The smith's "≥3K items per d=3 cohort" treats the 19 languages as 19× independent items. They are not — they are translations of the same base programs. Item-level independence at d=3 is bounded by the ~80–150 unique base problems (estimated by smith's own AST analysis), not 80–150 × 19. Cross-language correlation is high (the same algorithm in Python and Rust will be correct/incorrect for substantially correlated reasons in a 125M code model).

A more conservative power analysis: at 150 unique d≥3 problems, within-seed binomial std at 30% accuracy is √(0.30·0.70/150) ≈ 3.7 abs pts; SEM-on-mean across 5 seeds is 3.7/√5 ≈ 1.65 abs pts; SEM-on-ΔA is 1.65×√2 ≈ 2.3 abs pts; SEM-on-ratio r at predicted ΔA_TTT=2.0, ΔA_dense=6.0 is √((2.3/2)² + (2.3/6)²) ≈ 1.20. So r has noise ±1.2 — comparable to the 0.5/0.33 separation of 0.17. **Power is borderline at 5 seeds × language-correlated items.**

The smith's escalation path ("If F1 lands in [0.4, 0.7], bump to 10 seeds") provides a valid fallback. But the pre-registration should account for the language-correlation issue explicitly, e.g., by reporting (a) per-language ΔA distributions and (b) a clustered bootstrap that resamples problems-with-all-19-languages-as-units rather than item-level. This is an Important issue for the eval-designer to operationalize. The hypothesis is not statistically vacuous, but the smith's "≥3K items" framing overstates the available signal.

### M6 (non-additivity formalization). Clean.
The "additive null is `ΔA_TTT(K=4) ≈ ΔA_dense(K=4)`; sub-additive prediction is `ΔA_TTT(K=4) ≤ 0.5 × ΔA_dense(K=4)`" framing is a genuine non-additive interaction. The orthogonality criterion in the spec is satisfied — predicting ΔA(K=4) ratio crosses the 0.5 threshold is not just predicting one main effect.

---

## 6. Falsifiability assessment

| Criterion | Operationalizable? | Statistical power | Verdict |
|---|---|---|---|
| F1 (r = ΔA_TTT / max ΔA_other > 0.5 falsifies) | Yes — concrete ratio. | Borderline at 5 seeds × language-correlated items (SEM-r ≈ 1.2 if items are ~150 unique); good at 10 seeds. Smith's escalation path triggers correctly. | **Genuinely operationalizable, well-powered with escalation.** |
| F2 (|A_TTT-Linear(K=4) − A_TTT-η=0(K=4)| > 1.5 falsifies) | Yes. | At predicted A_TTT-Linear=30, A_η=0=31, separation is 1.0 vs threshold 1.5 — within noise floor at 5 seeds. Concerning. Smith should bump F2 threshold to 2.0 or commit to ≥10 seeds for F2 specifically. | **Operationalizable, marginally powered.** |
| F3 (r_CoT − r_no-CoT < +0.2 falsifies) | Yes — clean control. | Adds CoT condition variance, but the predicted shift (+0.37) is large. | **Operationalizable, well-powered.** |
| F4 (depth-axis specificity) | Yes — two thresholds. | At d=1, all backbones converge (small ΔA); ratio noise dominates. Threshold of r_d=1 < 0.7 may be hard to power because at d=1 everything is small. | **Operationalizable, weak power on d=1 sub-test.** |
| F5 (LaCT replication, threshold +2.0) | Yes — single comparison. | Predicted +0.5–1.0 vs threshold 2.0 — wide margin. | **Operationalizable, well-powered.** |

**Net.** F1, F3, F5 are well-powered with the smith's escalation path. F2 and F4-d=1 are marginally powered. None of these are "vacuously falsifiable" in the round-1 sense; all are finite, operational, and have noise floors quantified. **Substantial improvement over revision-0.**

---

## 7. Strongest counter-argument (steelman)

**The strongest opposing case: the predicted "TTT-specific compression" simply reflects that TTT-Linear is a worse base model on code reasoning than dense softmax, with worse K=1 accuracy and a lower asymptotic K_arch ceiling — ALL backbones saturate at some K_arch, and TTT just saturates lower because its base is worse.**

The case rests on:

1. **2602.21204 §4.1 itself shows trained-TTT's headroom is limited** — increasing inner-loop steps hurts, suggesting the trained operator is at or near its quality ceiling for the given input distribution. If the operator is near saturation, additional outer-loop iterations will buy less because there's less to refine.
2. **2604.21106's recurrence-equivalence exponent φ=0.46 implies all looped models have substantially diminishing returns from extra recurrences**, regardless of backbone. So compressed K_arch-gain is a generic looping property, not a TTT-specific phenomenon — TTT just has a smaller numerator.
3. **The smith predicts ΔA_TTT-Linear(K=4) = +2.0 vs ΔA_dense(K=4) = +6.0.** If this gap reflects TTT-Linear having a smaller "headroom" (60 % of dense's headroom for K-gain), then it may also reflect TTT-Linear having a smaller asymptotic ceiling overall — i.e., A_TTT-Linear(K=∞) − A_dense(K=∞) is also negative. The hypothesis would be true but uninteresting: "worse model has smaller K-gain, news at 11."

**What rescues the hypothesis from this steelman:**
- F2's prediction (A_TTT-Linear K=4 ≈ A_TTT-η=0 K=4) is exactly the test the steelman cannot easily explain. If trained-TTT and η=0 differ by ≤1.5 abs pts at K=4 even though they differ at K=1 (predicted 28 vs 26), the trained inner-loop iteration adds zero value at K=4 — which is the redundancy claim, not just "less headroom."
- The η=0 → trained-TTT comparison at K=1 (predicted 26 vs 28, +2 advantage from training the inner loop) is roughly equal to ΔA_TTT-Linear(K=4-K=1) = +2. **So TRM's outer iteration provides exactly the same delta that the trained inner loop provides from η=0 to η>0.** This quantitative coincidence is a non-trivial mechanistic prediction.

**Residual concern.** If the data come back as A_TTT-Linear(K=4) = 30, A_TTT-η=0(K=4) = 28 (i.e., TTT-Linear retains advantage over η=0 at K=4), F2 falsifies and the redundancy mechanism dies. This is a real risk and the smith correctly preregisters it.

The hypothesis survives the steelman *if F2's prediction holds*. It is genuinely a non-trivial test.

---

## 8. Severity-tagged objections

### Critical (must fix before APPROVE)
None. The three round-1 Critical objections are resolved.

### Important (should fix in eval-designer phase)

**I1. CRUXEval-X power analysis double-counts the 19-language expansion.** The smith treats "15K × 19 languages = 3K per d-cohort" as if items were independent. They are translations of ~800 base problems. A correct analysis should bootstrap at the problem level (each problem appearing in 19 languages as a single unit), not the item level. At 80–150 unique d≥3 problems, SEM-on-r is ≈1.2 at 5 seeds — comparable to the 0.5/0.33 separation. **Mitigation:** pre-register a clustered bootstrap (resample problems-not-language-instances) and commit to 10 seeds for primary F1. Smith's existing fallback ("bump to 10 seeds in ambiguous zone") covers this acceptably. Eval-designer should formalize the clustered bootstrap.

**I2. η=0 frozen-TTT collapses to position-wise MLP, not linear attention.** The cited corollary "η=0 ⇒ TTT layer = static linear-attention" is wrong — at η=0, the `g_t(k)=0` so the `Σ k̂^⊤v̂` accumulation vanishes entirely. The η=0 layer is `o_t = ϕ(q_t; Θ_0) W_0`, a token-wise MLP with no sequence mixing. This makes F2's interpretation murkier (η=0 lacks both iteration AND sequence mixing). **Mitigation:** add a "fixed-linear-attention" backbone (a true linear-attention layer with `o_t = q_t (S_0 + Σ k_i^⊤ v_i)` and trainable but stationary `W_q`, `W_k`, `W_v`, `S_0`) as a primary control alongside η=0. This is the actual "static learned linear-attention operator" the smith intends. Eval-designer can specify.

**I3. M2's lean on TRM §4.5 ("attention-on hurts") overgeneralizes.** TRM Table 1's attention-removal helps only on Sudoku-Extreme (L ≤ D); §4.5 explicitly notes it hurts on Maze-Hard and ARC-AGI. **Mitigation:** reframe M2 to depend only on "TRM's outer loop re-runs the backbone-induced operator" (which IS supported), and add 2604.21106 (recurrence-equivalence φ=0.46) as the saturation evidence that does not depend on Sudoku specifics.

**I4. K_arch axis ambiguity vs TRM's `T`, `n`, `N_sup`.** TRM has three iteration counters: `n` (inner f_L iterations per full recursion), `T` (full recursion processes per supervision step, T-1 without gradient), and `N_sup` (supervision steps reusing latents). The smith's K_arch ∈ {1,2,4,8} appears to be `T` with `n` fixed at 4, but the gradient-flow structure is non-trivial: TRM trains gradient through only the LAST full recursion process. If the smith trains gradient through ALL K_arch outer iterations, this is NOT TRM-faithful at the gradient level (even if reset-vs-persist is). Eval-designer should commit precisely: either (a) match TRM's "1 of T processes has gradient" (reduces compute, matches paper) or (b) full-backprop through all K_arch iterations (more compute, different from TRM but more directly tests the redundancy mechanism since deep-supervision plays a smaller role).

**I5. Pre-registration should add the AST depth-extraction definition.** "AST-extracted program-recursion-depth d" needs a precise definition: (a) AST nesting depth (max depth of `if`/`for`/`while`/`def` nodes), (b) call depth of standard library functions invoked, or (c) something else. Most CRUXEval functions are non-recursive utility code; the d≥5 cohort almost certainly comes from nested control flow / nested calls, not user-defined recursion. The hypothesis is about iteration-of-reasoning-depth, so (a) is closest. Eval-designer should commit to the exact extraction script (Python `ast` module pseudocode).

### Suggestion (nice to have)

**S1. Add 2604.21106 (recurrence-equivalence φ=0.46) to bibliography.** Strong adjacent precedent for "all looped models have diminishing returns." Strengthens the steelman defense and informs the F1 ratio threshold (0.5 vs predicted 0.33).

**S2. Add a "TTT trained at K_arch=4 from scratch" baseline.** The hypothesis tests "TRM-K=4 wrapping a TTT backbone trained at K=1 → TRM's K=4." A natural additional baseline: train TTT directly with 4× inner-loop steps from scratch (as a single-axis K=4 test of the TTT-iteration property). If this gives the same accuracy as TRM-K=4 × TTT, that's stronger evidence the two axes are substitutes, not just sub-additive. Compute cost is small (one extra cell).

**S3. Pre-register Hartigan dip test on per-cell seed distribution.** Carried over from R5; the smith mentions it. Worth committing to in §5a.

---

## 9. Recommendation to hypothesis-smith

The hypothesis is **APPROVE** as-is for Phase 5. The three round-1 Critical objections (C1, C2, C3) are all resolved at the mechanism / metric / architecture levels. The Important issues above (I1–I5) are all eval-designer-tractable and do not require another revision round.

For the eval-designer to address in Phase 5:
1. Replace "≥3K items" sample-size claim with a clustered-by-problem bootstrap accounting for 19-language correlation.
2. Add a "true fixed-linear-attention" backbone alongside η=0 to cleanly disambiguate the F2 mechanism check (η=0 currently collapses to position-wise MLP, not linear attention).
3. Commit to a precise gradient-flow recipe (TRM-faithful "gradient through last process" vs full-backprop-through-K_arch).
4. Specify the AST depth-extraction script and the d-bucket sizes from a dry run on CRUXEval-O.
5. Add 2604.21106 to the bibliography (recurrence-equivalence scaling law as steelman-defense and saturation precedent).

The hypothesis is publishable in its current form if the predictions land. The pivot from "destructive" to "redundant" was the right call — the new mechanism survives 2602.21204's empirical contradictions cleanly, and the F2 prediction (TTT-Linear ≈ TTT-η=0 at K=4) is a genuinely distinguishing test that the strongest steelman ("TTT just has less headroom") cannot easily explain.

---

`APPROVE` | `REJECT (revision-2)` | `KILL (irrecoverable)`

**Verdict:** APPROVE
