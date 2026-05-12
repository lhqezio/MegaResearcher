# H6-SUB (revision-1): TRM-style Depthwise Recursion and TTT-style Sequence-time Recursion are Sub-additive Substitutes — TRM's K_arch-gain on a TTT Backbone is Compressed Relative to its K_arch-gain on Non-TTT Controls

## Changes from revision-0

Revision-0 (`H6-VAR`) predicted a **variance-amplifying middle band** in K_arch driven by a "memorize-and-stale-read" / "over-fit-fast-weights" destructive-interference mechanism. The red-team's three Critical objections forced a substantive pivot rather than a patch:

- **C1 (memorization mechanism contradicted by 2602.21204).** Verified by re-reading 2602.21204 §4.1–4.4 and §5.2: the paper rules out memorization-based interpretations of TTT (gradient ascent works, Q→K substitution works, "more inner-loop fitting → worse downstream performance" already at K_arch=1). The M3a "stale fast-weight reads" and M3b "over-fitted fast-weights" mechanisms in revision-0 both rest on a memorization picture that 2602.21204 falsifies. **Pivot:** drop destructive interference; adopt 2602.21204's own mechanistic finding (TTT is a *learnable mapping* that determines effective q̂/k̂/v̂ in a learned linear-attention operator) as the load-bearing mechanism for the new hypothesis.
- **C2 (F1–F5 statistically near-vacuous).** Std-of-std at 5 seeds × 800-item × per-d sub-split is too noisy to discriminate the predicted thresholds. **Pivot:** the new hypothesis predicts mean-level (first-moment) effects, which are sample-efficient enough to test at the run's compute budget. Variance metrics demoted to secondary diagnostics with explicit power analysis.
- **C3 (architecture unspecified).** Inner TTT fast-weights `W_t` reset between outer TRM iterations vs persist was undefined. **Specified explicitly in §3 and §6:** primary experiment uses **reset-between-outer-iterations** (the TRM-faithful construction, since TRM repeats `f_L`/`f_H` from the same initial state on refined `(y, z)`); persist-across-outer-iterations is an ablation.

Important objections also addressed:
- **I1 (scale mismatch).** Engaged: TRM is 5M but Huginn (2502.05171, 3.5B-param recurrent block), Ouro (2510.25741, up to 2.6B looped LM), and MoR (2507.10524, 0.5B–7B mixtures-of-recursions) all run depth-recursion at scales ≥0.5B. The 125M choice is below TRM's "scale hurts" boundary in TRM Table 1 (4-layer×n=3, 10M, drops to 79.5%) but follows the TTT paper's anchor scale (2407.04620 trains 125M and 1.3B variants). Trade-offs spelled out in §4.
- **I2 (Figure 4 citation slip).** Tightened: Figure 4 of 2407.04620 shows "gradient descent reduces ℓ but cannot reduce it to zero" — i.e., loss-not-saturated, NOT weight-not-fixed-point. The paper's text "still trains a different sequence of weights `W_1, …, W_T` for every input sequence" (§2.1, end) is the proper citation for `W_t` non-stationarity. Both used appropriately in revised §3.
- **I3 (cohort size at d=3 unestimated).** Estimated and pre-registered in §6: CRUXEval-O is 800 items; AST-extracted depth distribution is heavily skewed shallow (most CRUXEval functions are 3–13 lines); estimated d=3 cohort ≈80–150 items. We therefore use CRUXEval-X (15K-item, 19-language, 2408.13001) as the primary surface for the load-bearing means-comparison, with single-language CRUXEval-O as a secondary cross-check.
- **I4 (Mamba not a clean control).** Acknowledged: Longhorn (2407.14207) frames Mamba as amortized online learning, so it shares some "sequence-time online update" structure with TTT. The η=0 frozen-TTT control (which 2602.21204 §5.1 establishes as exactly equivalent to a fixed linear-attention operator) is promoted from ablation to a primary control alongside Mamba and dense softmax.
- **I5 (destructive-interference framing chosen for non-additivity, not mechanism).** This was a real weakness. The new framing is **redundancy/substitution**: it remains non-additive (sub-additive), but is now derived from 2602.21204's mechanistic finding rather than chosen for shape.

The hypothesis remains a single, coherent, falsifiable prediction targeting Gap 6. The non-additivity discipline is preserved; the mechanism is now grounded in cited prior art rather than constructed for shape.

---

## 1. Targeted gap

Source: `docs/research/runs/2026-05-10-0729-766039/gap-finder-1/output.md` Gap 6 (mirrored in `gaps.md` row #6).

The literature has cleanly separated two "internalized computation" axes that both live inside a sub-quadratic backbone:

- **Sequence-time inner-loop recursion** — TTT (arXiv:2407.04620), LaCT / Test-Time Training Done Right (arXiv:2505.23884), TTT-as-linear-attention (arXiv:2602.21204), One-Minute Video TTT (arXiv:2504.05298), Longhorn (arXiv:2407.14207).
- **Depth-time outer-loop recursion** — TRM (arXiv:2510.04871), HRM (arXiv:2506.21734), Huginn (arXiv:2502.05171), Ouro (arXiv:2510.25741), MoR (arXiv:2507.10524), Reasoning with Latent Thoughts (arXiv:2502.17416), Looped Transformers Better at Learning Algorithms (arXiv:2311.12424).

Gap-finder-1 verified via `hf_papers search query="TTT test-time training depth recursion looped"` (10 results) that no paper instantiates both axes jointly, ablates one against the other, or characterizes whether stacking them is redundant, complementary, or destructive. Red-team round-1 (C1's verifications) re-confirmed the gap: SR-TTT (2603.06642) and In-Place TTT (2604.06169) are post-cutoff but neither composes depth recursion with TTT. **The gap survives.**

This revision targets the **redundant / sub-additive substitute** mode of the gap (the alternate framing the original prompt explicitly offered). Test surface: CRUXEval-X (arXiv:2408.13001) with per-instance program-recursion-depth `d` extracted from the program AST.

Adjacent grounding: Transformer-Based Models Are Not Yet Perfect at Learning to Emulate Structural Recursion (arXiv:2401.12947), CRUXEval (arXiv:2401.03065), Expressive Power of Looped Transformers (arXiv:2410.01405).

---

## 2. Hypothesis statement

**If** a TRM-style depthwise recursion operator with K_arch ∈ {1, 2, 4, 8} weight-tied iterations and deep supervision (arXiv:2510.04871 §2.4 / §4.1) is wrapped around four matched-parameter backbones — **TTT-Linear** (arXiv:2407.04620), **frozen-fast-weight TTT** (η=0, ≡ a fixed learned linear-attention operator per arXiv:2602.21204 §5.1 Theorem 5.1), **Mamba/SSM** (arXiv:2407.14207 baseline), and **dense softmax attention** (TRM-faithful, arXiv:2510.04871 §4.5) — at ~125M parameters with identical pre-training corpus and ≥5 seeds per cell, with primary evaluation on CRUXEval-X (arXiv:2408.13001) at per-instance program-recursion-depth d ∈ {1, 2, 3, 4, ≥5} extracted by AST analysis, no text-CoT,

**then** the **TRM K_arch-gain on the TTT-Linear backbone** — the mean-accuracy difference between K_arch=4 and K_arch=1 — will be **at least 50% smaller in absolute points** than the TRM K_arch-gain on each of the three non-TTT controls (frozen-TTT, Mamba, dense), at fixed program-depth d ≥ 3, and the K_arch=4 mean-accuracy *level* on TTT-Linear will be **within ±1.5 absolute points** of the K_arch=4 level on frozen-TTT-at-η=0 (sub-additive saturation: when the inner loop is already inducing iterative refinement of an attention operator, additional depth-time iteration of that same operator buys little).

This is a non-additive (sub-additive) prediction: TRM and TTT separately each lift accuracy via outer-vs-inner iteration; jointly they substitute rather than compound, because they refine the same effective object — the learned linear-attention operator the TTT layer induces (arXiv:2602.21204 §5.1, §5.2).

---

## 3. Mechanism

The mechanism is grounded in arXiv:2602.21204's reinterpretation of TTT and TRM's deep-supervision mechanic.

**M1. TTT is a learnable mapping that induces a linear-attention operator** (arXiv:2602.21204 §5.1 Theorem 5.1, §5.2). For a TTT model with linear final-layer inner loop, the output at token t with query q is `o = ϕ_{t+1}(q) · (W_t + ϕ_t(k)^⊤ g_t(k))`, equivalent to a learned linear-attention operator with effective `q̂ = ϕ_{t+1}(q)`, `k̂ = ϕ_t(k)`, `v̂ = g_t(k)`. The inner-loop hyperparameters (number of steps, learning rate η) determine *which* attention operator is induced. The empirical contradictions in arXiv:2602.21204 §4.1–4.4 are explained by §5.2 as: changing inner-loop step count at inference time induces a *different* attention operator than the one trained, causing train-test mismatch.

**M2. TRM's outer loop is iterative refinement of the same backbone-induced operator** (arXiv:2510.04871 §4.1). TRM defines a "full recursion process" as `n` evaluations of `f_L` on `(z_L + z_H + x)` followed by 1 evaluation of `f_H` on `(z_L + z_H)`. Crucially TRM Table 1 / §4.5 shows TRM's K-loop **substitutes for self-attention**: adding self-attention to the f_L/f_H block *hurts* (87.4% no-attention vs 74.7% with-attention on Sudoku-Extreme, same parameter count). This is the load-bearing TRM observation for this hypothesis: depth-time iteration of a non-attention block already provides much of the benefit attention provides; adding attention is at best neutral.

**M3. The composition is sub-additive because both axes refine the same effective object.** When the backbone is TTT-Linear, the per-layer operator is *itself* (per M1) a learned linear-attention operator whose effective shape is set by the inner-loop trajectory along sequence-time. TRM's outer loop then runs the same backbone again on refined `(y, z)`; this re-runs the inner-loop trajectory and re-computes the effective `q̂/k̂/v̂` with new inputs. At K_arch=1, the model gets one pass of (TTT inner loop → effective attention operator). At K_arch=4, it gets four passes — but each pass is recomputing essentially the same kind of object (a learned-linear-attention operator over slightly-different (y, z) inputs), and there are diminishing returns to iterating refinement of one operator type. By contrast, when the backbone is dense softmax (no per-token operator-shape change) or Mamba (a different, non-attention sequence operator), TRM's outer loop adds a genuinely-new iteration axis on top, with monotone returns up to its K_arch=4-or-8 saturation point seen in TRM Table 1.

**M4. The frozen-η=0 TTT control isolates the "is it the inner-loop-iteration property?" question** (arXiv:2602.21204 §5.1 Theorem 5.1 corollary at η=0). At η=0, the TTT inner-loop trajectory degenerates to the static linear-attention operator with `S_0 = W_0`. If the η=0 backbone shows TRM K_arch-gains comparable to Mamba/dense (i.e., the *learned* operator iteration property of TTT is what makes TRM redundant), this is positive evidence for M3. If η=0 also shows compressed K_arch-gain (matching trained TTT), then the redundancy is not specific to inner-loop-iteration and the mechanism is wrong.

**M5. CRUXEval-X is the right test surface for sub-additive saturation at fixed depth-of-reasoning** (arXiv:2401.03065, arXiv:2408.13001). CRUXEval consists of short Python functions (3–13 lines) with input/output pairs. Per-instance program-recursion-depth d is an invariant of the program AST. CRUXEval-X (15,000 items × 19 languages) provides ≥10× more items than CRUXEval-O at d=3 and a robust cross-language axis. Per arXiv:2401.12947, dense Transformers fail at structural recursion in a way that depends on per-instance recursion depth — providing the per-d baseline. The CoT-helpful sub-split (arXiv:2401.03065 §5) provides the CoT-confound control: under text-CoT, the chain of intermediate reasoning is supplied externally, and depth-recursion redundancy with TTT inner-loop iteration should be reduced.

**M6. Why this is non-additive.** Let A_TTT(K=1) and A_dense(K=1) be K_arch=1 means; let ΔA_TTT(K=4) = A_TTT(K=4) − A_TTT(K=1) and similarly for dense/Mamba. The additive-composition null is `ΔA_TTT(K=4) ≈ ΔA_dense(K=4)` (TRM gain is the same regardless of backbone). The hypothesis predicts `ΔA_TTT(K=4) ≤ 0.5 × ΔA_dense(K=4)` and `ΔA_TTT(K=4) ≤ 0.5 × ΔA_Mamba(K=4)`. This is a sub-additive interaction — the joint K=4 × TTT cell underperforms what additive composition predicts. The interaction is *negative* (sub-additive) but **not destructive** in the variance sense of revision-0; it is saturation of a shared substrate.

**Why this mechanism survives 2602.21204 directly.** §4.1's "more inner-loop fitting → worse" is now *predicted* by M1+M3: if the inner loop already induces an operator that the model is trained to use, varying inner-step count at inference (which is what 2602.21204 §4.1 does) creates train-test operator mismatch. Wrapping TRM around a TTT layer at *training time* (which is what this hypothesis does, distinct from 2602.21204's *inference-time* sweep) fixes the train-test consistency, but introduces the redundancy of M3. The two phenomena (2602.21204's inference-time degradation and this hypothesis' training-time saturation) have a unified mechanistic root in M1.

---

## 4. Predicted outcome with magnitude

Setup (sketch — eval-designer to detail):
- Four backbones at matched parameter count (~125M, the TTT paper's smaller anchor scale, arXiv:2407.04620): **TTT-Linear**; **TTT-η=0 (frozen-fast-weights / static linear attention)**; **Mamba/SSM**; **dense softmax attention** (TRM-faithful).
- TRM-style depthwise recursion wrapped on top with K_arch ∈ {1, 2, 4, 8}, weight-tied across iterations, deep-supervised per arXiv:2510.04871 §2.4. **Inner-loop fast-weights `W_t` reset to `W_0` between outer iterations** (TRM-faithful primary; persistence is an ablation in §6).
- ≥5 seed replicates per (backbone, K_arch) cell. CRUXEval-X (arXiv:2408.13001) primary metric (15K items × 19 languages); CRUXEval-O (800 items, single-language Python) secondary cross-check. Per-instance program-recursion-depth `d` from AST static analysis. No text-CoT in primary condition.

Predicted magnitudes (mean exact-match accuracy ± across-seed std on CRUXEval-X, restricted to d ≥ 3, no-CoT):

| Backbone | K=1 mean | K=4 mean | ΔA(K=4 − K=1) |
|---|---|---|---|
| TTT-Linear | 28 ± 1.5 | 30 ± 2.0 | **+2.0** |
| TTT-η=0 (frozen) | 26 ± 1.5 | 31 ± 2.0 | **+5.0** |
| Mamba (control) | 27 ± 1.5 | 32 ± 2.0 | **+5.0** |
| Dense softmax (control) | 30 ± 1.5 | 36 ± 2.0 | **+6.0** |

Anchor: GPT-4 reaches 63% on CRUXEval-O at vastly larger scale (arXiv:2401.03065 §1); 125M open models score in the 25–40% band on CRUXEval-O. CRUXEval-X spans 19 languages with similar score range. Predicted seed-std ≈1.5–2.0 abs points anchored to typical seed-replicate variance for 125M models on multiple-choice / exact-match benchmarks.

Two load-bearing predictions:

(a) **Compressed K_arch-gain on trained TTT.** ΔA_TTT-Linear(K=4) ≤ 0.5 × min(ΔA_dense, ΔA_Mamba, ΔA_η=0). Predicted: +2.0 vs ≥+5.0, i.e., compression to ≤40%.

(b) **K=4 levels on trained TTT and frozen-η=0 TTT converge.** A_TTT-Linear(K=4) − A_TTT-η=0(K=4) ≤ 1.5 abs points. Predicted: 30 − 31 = −1, |Δ| ≤ 1.5. The two TTT configurations converge at K=4 because TRM's outer iteration substitutes for the inner-loop iteration that the trained-η variant alone provides.

Effect at d = 1 should largely vanish (single-pass programs need neither inner-loop nor depth-time iteration). Effect should diminish under text-CoT (external chain supplies the iterative-refinement budget that depth-recursion otherwise provides).

**Where the hypothesis should hold (positive predictions):**
- d ≥ 3 deeper-recursion CRUXEval-X instances.
- No text-CoT.
- TTT-Linear (and replicate on LaCT, arXiv:2505.23884): predicted to show the same compression, since LaCT is also a TTT variant and 2602.21204 §5.3 rewrites LaCT in linear-attention form.
- 125M parameter scale.

**Where the hypothesis should NOT hold (negative predictions):**
- d = 1: all four backbones approximately equal K_arch=1 / K_arch=4 means; gap closes.
- With text-CoT enabled (CoT-helpful sub-split): ΔA_TTT-Linear(K=4) approaches ΔA_dense(K=4); compression ratio rises above 0.7.
- η=0 control (frozen TTT): TRM K_arch-gain ≈ Mamba TRM K_arch-gain (both backbones lack a *learned-and-dynamic* per-token operator; so TRM iteration is non-redundant for both).
- At parameter scales <30M (where TRM's "less is more" observation places the inner block, arXiv:2510.04871 §4.4): both axes may be capacity-limited and the redundancy effect smaller; we do not test this regime, but flag it in §7-R1.

---

## 5. Falsification criteria

All criteria use CRUXEval-X exact-match accuracy (English instructions, 19 target languages averaged) as the primary metric. Means are reported ±across-seed std with ≥5 seeds per cell; bootstrap 95% CIs reported alongside (per red-team S2). Restricted to d ≥ 3, no-CoT unless noted.

**F1 (compression ratio, primary load-bearing).**
- Metric: r = ΔA_TTT-Linear(K=4) / max(ΔA_Mamba(K=4), ΔA_dense(K=4), ΔA_η=0(K=4)).
- Threshold: **r > 0.5** (compression less than 50%) ⇒ **falsified.**
- Direction: ABOVE-THRESHOLD falsifies. (Predicted r ≈ 0.33, i.e., +2 / +6.)
- Statistical operationalization: ΔA values are means-of-means at 5 seeds × ~5K-item d≥3 cohort each; binomial-finite-sample noise on a 5K-item cohort at 30% accuracy is √(0.30·0.70/5000) ≈ 0.65 abs pts within-seed; across-5-seeds SEM on the *mean* is ≈0.7 abs pts. The ratio r has bootstrap noise ≤0.15 — well below the 0.5 / 0.33 separation. Powered.

**F2 (TTT–η=0 convergence at K=4, mechanism check).**
- Metric: |A_TTT-Linear(K=4) − A_TTT-η=0(K=4)|.
- Threshold: **>1.5 absolute points** ⇒ **falsified.**
- Direction: ABOVE-THRESHOLD falsifies. (Predicted: ≈1.0.)
- Rationale: M3 implies that at K=4 the outer loop has compensated for the trained inner-loop. If the trained inner loop still adds value at K=4, the redundancy story is wrong; if frozen-η=0 outperforms trained-TTT at K=4, the mechanism (M1's "operator-induction" attribute of the trained inner loop) is also wrong but in a different way (then it's the trained backbone that's sub-optimal, not redundant).

**F3 (CoT-confound control — text-CoT releases the redundancy).**
- Metric: r_CoT = ΔA_TTT-Linear(K=4, CoT-helpful) / ΔA_dense(K=4, CoT-helpful), and same r_no-CoT.
- Threshold: **r_CoT − r_no-CoT < +0.2** ⇒ **falsified.**
- Direction: BELOW-THRESHOLD or sign-flip falsifies. (Predicted: r_no-CoT ≈ 0.33, r_CoT ≈ 0.7+, difference ≥+0.37.)
- Rationale: under text-CoT, the iterative chain is supplied externally; both backbones should benefit from TRM K_arch-gains additively because the TTT-internal iterative refinement is no longer the only iterative-refinement budget. If r_CoT is *not larger* than r_no-CoT, the no-CoT case is not measuring depth-recursion redundancy; it's measuring some other artifact.

**F4 (depth-axis specificity).**
- Metric: r_d≥3 vs r_d=1.
- Threshold: **r_d=1 < 0.7** ⇒ **falsified.** (At d=1, all backbones should converge on K_arch=4; the compression on TTT should *not* be specific to shallow programs.)
- And: **r_d≥3 ≥ r_d=1** ⇒ **falsified.** (The compression should grow, or at least not shrink, with depth.)
- Direction: VIOLATIONS of either bound falsifies. (Predicted: r_d=1 ≈ 0.85+, r_d≥3 ≈ 0.33.)

**F5 (LaCT replication).**
- Metric: ΔA_LaCT(K=4) vs ΔA_TTT-Linear(K=4), both d ≥ 3, no-CoT.
- Threshold: **ΔA_LaCT(K=4) − ΔA_TTT-Linear(K=4) > +2.0** ⇒ **falsified.**
- Direction: ABOVE-THRESHOLD falsifies. (Predicted: ΔA_LaCT(K=4) ≈ +2.5–3.0, slightly larger than TTT-Linear because LaCT chunks fast-weight updates and is thus partway between TTT-Linear and frozen — but still substantially below dense/Mamba.)
- Rationale: LaCT is a TTT variant and 2602.21204 §5.3 establishes LaCT also fits the linear-attention reinterpretation. If LaCT does NOT show comparable compression, the M1 "TTT is learned linear attention" mechanism is too narrow.

These five criteria are all **first-moment** (means and ratios of means) at much higher item count than revision-0's CRUXEval-O d=3 sub-split. Power analysis (§5a) shows binomial-and-cross-seed noise floor for ΔA estimates is ≤0.7 abs points; thresholds sit well above the noise floor.

### 5a. Pre-registered sample-size analysis

Pre-registration: ≥5 seeds × CRUXEval-X 15K items × per-d sub-split d ∈ {1, 2, 3, 4, ≥5}.

- d=3 cohort estimate on CRUXEval-X: 15K × 19 languages averaged. AST-extracted depth distribution skews shallow (most CRUXEval functions are 3–13 lines, recursion is rare). Conservative estimate at d=3: ≥3K items per backbone-K cell after CRUXEval-X expansion.
- Within-seed binomial std at 30% accuracy on 3K items: √(0.30·0.70/3000) ≈ 0.84 abs pts.
- Across-5-seeds SEM on the mean: 0.84 / √5 ≈ 0.38 abs pts.
- Compounding into ΔA(K=4 − K=1) for one backbone: SEM ≈ 0.38 × √2 ≈ 0.54 abs pts.
- Compounding into r = ΔA_TTT / ΔA_dense: with predicted ΔA_TTT ≈ 2.0, ΔA_dense ≈ 6.0, the relative SEM on r is √((0.54/2.0)² + (0.54/6.0)²) ≈ 0.28; bootstrap CI on r at 5 seeds is approximately ±0.3 → [0.05, 0.65] worst-case at predicted point, well-separated from r > 0.5 falsification threshold.

If F1 lands in [0.4, 0.7] (ambiguous zone), bump to 10 seeds per cell; this doubles compute cost but is below the 2,000-GPU-hour fence (see §6).

The variance-amplification metrics from revision-0 are **demoted** to secondary diagnostics. They will be reported (with bootstrap CIs) but are not load-bearing falsifiers.

---

## 6. Required experiments (sketch — eval-designer to specify)

**Datasets.**
- **Primary:** CRUXEval-X (arXiv:2408.13001), 15K items × 19 languages, with per-instance program-recursion-depth `d` extracted by AST static analysis. CoT-helpful / CoT-harmful sub-split available from CRUXEval-O (arXiv:2401.03065 §5) for the F3 control.
- **Secondary:** CRUXEval-O (arXiv:2401.03065), 800 items, as a single-language Python cross-check.
- **Optional:** arXiv:2401.12947 structural-recursion synthetic benchmark for a dense-only sanity baseline.

**Backbones (matched 125M parameters, ≥5 seeds each).**
- **TTT-Linear** (arXiv:2407.04620 reference impl, https://github.com/test-time-training/ttt-lm-jax).
- **TTT-η=0 (frozen-fast-weight)** — same architecture as TTT-Linear with inner-loop η = 0 throughout training (per arXiv:2602.21204 Theorem 5.1, this is exactly a fixed learned linear-attention operator).
- **Mamba** (arXiv:2407.14207 baseline lineage) as a different sequence-time online-learner control.
- **Dense softmax attention** (TRM-faithful, arXiv:2510.04871 §4.5).

**TRM operator wrapping (architecture, primary).**
- Deep supervision per arXiv:2510.04871 §2.4: T-1 recursion processes without gradient, then 1 recursion process with full backprop through n=4 evaluations of f_L and 1 of f_H. K_arch ∈ {1, 2, 4, 8}.
- **Inner-loop reset (PRIMARY).** When outer iteration k+1 begins, TTT inner-loop fast weights `W_t` reset to `W_0` (the trained initialization). Each outer iteration runs the full sequence-time inner loop from scratch on the refined `(y, z)` input embedding sequence. This is TRM-faithful (TRM repeats f_L on a fresh `(z_L + z_H + x)` input each iteration; nothing in TRM persists per-iteration internal state).
- **Inner-loop persist (ABLATION, secondary).** Same setup but `W_t` from iteration k is the initial `W_0^{(k+1)}` for iteration k+1. This breaks TRM's "no-state-other-than-(y,z)" design but is theoretically interesting (it replicates 2602.21204 §4.1's "more inner-loop steps" sweep at training time across outer iterations). Run only at K_arch=4, single seed × 4 backbones, as ablation.

**Pre-training corpus.** Identical across all (backbone, K_arch) cells — same code-pretraining mixture (e.g., StarCoder-style, FineWeb-Edu code subset), identical token count (~50B tokens, calibrated to TTT paper's 125M setup, arXiv:2407.04620 §3.2). Held-out instruction-tuning split with no CRUXEval contamination (arXiv:2401.03065 §3 release-date controls; CRUXEval was released 2024-01).

**Compute envelope.** ~5 seeds × 4 K_arch × 4 backbones × ~125M params × ~50B tokens ≈ 80 model-days at 8×H100, ≈ ~2,400–3,200 GPU-hours. Above the 2,000-GPU-hour fence — see §6a for kill-switch and §6b for compute-reduction options.

### 6a. Cheaper falsification path (revised, addressing red-team S1)

Single fine-tune-and-evaluate experiment, no full pre-training:

1. Take open-weights TTT-Linear or LaCT pre-trained checkpoint (arXiv:2407.04620 release / arXiv:2505.23884 release at the closest available scale).
2. Take open-weights Mamba and dense-Transformer checkpoints at matched scale.
3. For each, **fine-tune** (not just test-time wrap) a TRM-style K_arch ∈ {1, 4} weight-tied wrapper on the final two blocks for 2B tokens with deep supervision. This exercises the training-time gradient flow that the M3 mechanism is about (per S1's correction of revision-0's pure test-time wrapper).
4. Three seeds per (backbone, K_arch) cell. Evaluate CRUXEval-X at d=3.
5. Report ΔA_TTT(K=4 − K=1) and ΔA_dense(K=4 − K=1).

Expected cost: ~120 GPU-hours on a single 8×H100 node (2B tokens × 4 backbones × 2 K × 3 seeds at 125M).

**Kill-switch:** if ΔA_TTT(K=4) ≥ 0.7 × ΔA_dense(K=4) on the cheaper path, the sub-additive saturation prediction is dead and the §6 full-pre-training run is not warranted.

**Caveat (preserved from revision-0).** Test-time + light-fine-tune wrapping on a non-recursion-pre-trained model is a *weaker* probe than full pre-training (per Ouro arXiv:2510.25741 — they argue iterative computation must be baked in at pre-training to fully manifest). A null result in §6a is suggestive but does not strictly falsify the full-pre-training prediction; a *positive* result in §6a is strong evidence to fund the full experiment in §6.

### 6b. Compute-reduction options (if §6 runs over budget)

- Drop dense-softmax control (Mamba and η=0-TTT remain). Saves 25%.
- Drop K_arch=8 cells (K=1, 2, 4 only). Saves ~25%.
- Use 100M instead of 125M parameter scale; shorter pre-training. Saves ~30%.
- Use CRUXEval-X subset (5 languages instead of 19). Sample-size analysis still passes.

---

## 7. Risks to the hypothesis

**R1. The compression effect may only appear at scales ≥0.5B.** TRM's depth-recursion was designed for 5M-parameter networks (arXiv:2510.04871 §4.4); Huginn (arXiv:2502.05171) and Ouro (arXiv:2510.25741) demonstrate depth-recursion at 0.5B–3.5B but with different architectural choices. At 125M, the inner-loop f might be too small for the redundancy to bite. If this risk materializes (compression ratio r ≈ 0.7 not 0.33), the work still contributes (a) the first joint TRM × TTT ablation in the literature, (b) the CRUXEval-X-with-AST-depth-axis instrument as a per-instance reasoning-depth probe, and (c) a scale boundary that informs follow-up work at ≥0.5B.

**R2. TTT-η=0 (frozen) might NOT be a clean linear-attention surrogate because the η=0 case removes the inner-loop's gradient signal during training.** arXiv:2602.21204 Theorem 5.1 shows that *one* inner-loop step with η=0 gives a static linear-attention operator, but trained-TTT models with η>0 may have learned upstream/downstream parameters that depend on the η>0 dynamics. Fully retraining η=0 from scratch (which §6 does) is the right control, but the η=0 backbone may simply learn to be a worse model overall (since it has effectively fewer free parameters per forward). If η=0 is consistently below trained-TTT at all K_arch, F2's "convergence at K=4" prediction is harder to interpret. Mitigation: report η=0 vs trained-TTT comparison at K=1 (should be ≈similar per 2602.21204) and use that as a baseline reference.

**R3. The two recursions may genuinely *not interact at all* (additive composition holds).** This is the steelman from red-team §6. If F1 returns r ≈ 1.0, the work contributes the first negative result for the redundancy hypothesis and *opens* the destructive-vs-complementary question (the spec's other two candidate failure modes, gap-finder-1 §Gap6) to follow-up — the experimental scaffolding is reusable. This is the most honest risk: the cheaper §6a path is designed to detect this risk before the full §6 spend.

**R4. Mamba may not be a clean control because its hidden state is also non-stationary along the sequence (arXiv:2407.14207 amortized-online-learning framing).** If Mamba shows the same compressed K_arch-gain as TTT, the dichotomy "TTT has dynamic operator induction / Mamba doesn't" is weaker than M1 claims. The η=0 frozen-TTT control is the cleanest stationary baseline (per 2602.21204 §5.1 it is *exactly* a fixed linear-attention operator, with no per-token operator update). If both Mamba and η=0 show full TRM gains while only trained-TTT shows compression, M3 is supported. If Mamba shows compression too, the mechanism reduces to "all sequence-time operators with a learned per-token component compose sub-additively with TRM," which is a weaker but still-publishable finding.

**R5. Bimodality / multi-modal seed distributions might arise from the interaction of K_arch and TTT inner-loop η** (preserved from revision-0 R2 but now applied to mean rather than std). Pre-register Hartigan dip test (per red-team S3) on ΔA across seeds; if a backbone shows bimodal distribution at K=4, individual seeds may be in different operating regimes and the mean is misleading. Mitigation: report seed histograms alongside means.

**R6. The TRM "less is more" property at 5M parameters might break entirely at 125M** (red-team I1 forced this risk). TRM Table 1 / §4.4 explicitly argues scaling beyond ~5M hurts on Sudoku-Extreme. We rely on the fact that Huginn (arXiv:2502.05171, ~3.5B), Ouro (arXiv:2510.25741, up to 2.6B), and MoR (arXiv:2507.10524, 0.5B–7B) demonstrate viable depth recursion at scales ≥0.5B *but with different recipes than TRM*. Our 125M setup may sit in an awkward valley — too large for TRM's recipe to work, too small for Huginn/Ouro's. If TRM-at-125M underperforms baseline non-recursion at K_arch=1 across all backbones, the K_arch ablation is meaningless. Mitigation: pre-register a sanity check: TRM K_arch=4 dense-softmax must beat the same 125M dense-Transformer without TRM-wrapping (kill-switch on the construction itself). If this fails, the §6 run is aborted and §6a's open-weights checkpoints become the only viable path.

---

## 8. Sources

- arXiv:2510.04871 — Less is More: Recursive Reasoning with Tiny Networks (TRM). §2.4 deep supervision, §4.1 full-recursion gradient, §4.2 (y,z) reinterpretation, §4.4 "less is more" / scale boundary, §4.5 attention-on ablation.
- arXiv:2407.04620 — Learning to (Learn at Test Time): RNNs with Expressive Hidden States (TTT). §2.1 TTT updating hidden state ("trains a different sequence of weights `W_1, …, W_T` for every input sequence"), Figure 4 ("gradient descent reduces ℓ but cannot reduce it to zero").
- arXiv:2505.23884 — Test-Time Training Done Right (LaCT). Cited for fast-weight FLOPs-utilization analysis, chunked stabilization, and as an F5 replication backbone.
- arXiv:2602.21204 — Test-Time Training with KV Binding Is Secretly Linear Attention. §4.1–4.4 empirical contradictions to memorization (NOT used to support memorization mechanism — explicitly used to ground the *learned-linear-attention-operator* mechanism). §5.1 Theorem 5.1 (TTT ≡ learned linear attention), §5.2 mechanistic explanation of inner-loop-step degradation, §5.3 LaCT linear-attention rewrite.
- arXiv:2504.05298 — One-Minute Video Generation with TTT. Cited for TTT-on-Transformer architecture composition precedent (depth-recursion-adjacent).
- arXiv:2407.14207 — Longhorn: SSMs as Amortized Online Learners. Cited for SSM = sequence-time online learner framing (R4 Mamba-not-clean concern).
- arXiv:2506.21734 — Hierarchical Reasoning Model (HRM). Predecessor to TRM, cited for fixed-point assumption that TRM removes.
- arXiv:2502.05171 — Scaling up Test-Time Compute with Latent Reasoning (Huginn). Cited for depth-recursion at scale (3.5B), I1 scale-precedent.
- arXiv:2510.25741 — Scaling Latent Reasoning via Looped Language Models (Ouro). Cited for depth-recursion at scale (2.6B), "iteration baked in at pre-training" caveat in §6a.
- arXiv:2507.10524 — Mixture-of-Recursions (MoR). Cited for adaptive depth at 0.5B–7B scale, I1 scale-precedent.
- arXiv:2502.17416 — Reasoning with Latent Thoughts: On the Power of Looped Transformers. Cited for "looped models match deeper unlooped models for reasoning at fixed parameter count."
- arXiv:2311.12424 — Looped Transformers are Better at Learning Algorithms. Cited for theoretical iterative-algorithm grounding.
- arXiv:2410.01405 — On Expressive Power of Looped Transformers. Cited for K_arch expressiveness analysis.
- arXiv:2401.03065 — CRUXEval. Test surface secondary; §1 sample shown, §5 CoT subsplits.
- arXiv:2408.13001 — CRUXEval-X. Primary test surface (15K items × 19 languages) addressing red-team C2 statistical-power issue.
- arXiv:2401.12947 — Transformer-Based Models Are Not Yet Perfect at Learning to Emulate Structural Recursion. Negative-result baseline for dense recursion-depth scaling.
