# H6-VAR: Stacking TRM-style Depthwise Recursion on TTT-style Sequence-time Recursion Causes a Variance-Amplifying Phase, Not an Additive Lift

## 1. Targeted gap

Source: `docs/research/runs/2026-05-10-0729-766039/gap-finder-1/output.md` Gap 6 (mirrored in `gaps.md` row #6).

The literature has cleanly separated two "internalized computation" axes that both live inside a sub-quadratic backbone:

- **Sequence-time inner-loop recursion** — TTT (arXiv:2407.04620), LaCT / Test-Time Training Done Right (arXiv:2505.23884), TTT-as-linear-attention (arXiv:2602.21204), One-Minute Video TTT (arXiv:2504.05298), Longhorn (arXiv:2407.14207). The hidden state IS a small ML model whose fast weights `W_t` are updated by a self-supervised gradient step *per token along the sequence*.
- **Depth-time outer-loop recursion** — TRM (arXiv:2510.04871), HRM (arXiv:2506.21734), Huginn (arXiv:2502.05171), Ouro (arXiv:2510.25741), MoR (arXiv:2507.10524), Reasoning with Latent Thoughts (arXiv:2502.17416), Looped Transformers Better at Learning Algorithms (arXiv:2311.12424). A weight-tied operator is iterated K times *along depth, every token in parallel*.

Gap-finder-1 verified via `hf_papers search query="TTT test-time training depth recursion looped"` (10 results) that no paper instantiates both axes jointly, ablates one against the other, or characterizes whether stacking them is redundant, complementary, or destructive. The gap-finder identified three candidate failure modes; this hypothesis targets the **destructive-interference** mode, scored at the test-surface (CRUXEval — arXiv:2401.03065) where program-recursion-depth is per-instance available.

Adjacent grounding citations: Transformer-Based Models Are Not Yet Perfect at Learning to Emulate Structural Recursion (arXiv:2401.12947), CRUXEval-X (arXiv:2408.13001), Expressive Power of Looped Transformers (arXiv:2410.01405).

## 2. Hypothesis statement

**If** a TRM-style depthwise recursion operator with K_arch ∈ {1, 2, 4, 8, 16} weight-tied iterations is wrapped around a TTT-layer backbone (TTT-Linear or LaCT-style; arXiv:2407.04620 / arXiv:2505.23884) and the same operator is also wrapped around two non-TTT control backbones (a Mamba / SSM backbone with no inner-loop fast weights, arXiv:2407.14207 baseline; and a dense softmax attention backbone, TRM-style attention-on, arXiv:2510.04871), and all three are matched on parameters, training tokens, and seed-replicate count (≥5 seeds), and evaluated on CRUXEval-O input/output prediction split by per-instance program-recursion-depth d ∈ {1, 2, 3, 4, ≥5} extracted from the program AST, **then**:

(a) the **TTT × TRM-recursion** stack will exhibit a non-monotone K_arch curve characterized by a **variance-amplifying middle band** at K_arch ∈ {2, 4} — across-seed standard deviation of CRUXEval-O accuracy at least **2.0×** the across-seed standard deviation of the K_arch=1 TTT baseline (computed at fixed program-depth d=3, equation in §5);

(b) the **non-TTT controls** (Mamba, dense softmax) under the same TRM operator will show **monotone non-decreasing** mean accuracy in K_arch and **flat or shrinking** across-seed variance — replicating the depthwise-recursion findings of TRM (arXiv:2510.04871 Table 1) and Huginn (arXiv:2502.05171);

(c) the **interaction term** (TTT-backbone variance lift) − (Mamba-backbone variance lift) measured at K_arch=4 will be at least **1.5×** the larger of the two marginals, an interaction not predicted by any additive composition of the two recursions' separately reported behaviors.

This is a non-additive prediction: the same TRM operator and the same TTT layer, when each is run alone, are stable; the interaction has the destructive signature.

## 3. Mechanism

The mechanism rests on a clock-mismatch between two nested learning loops.

**M1. TTT's inner loop produces a non-stationary hidden state along the sequence.** TTT layers (arXiv:2407.04620 §2.1) define the hidden state as the weights `W_t` of a small model `f`, with output `z_t = f(x_t; W_t)` and update `W_t = W_{t-1} − η ∇ℓ(W_{t-1}; x_t)`. The authors explicitly show (arXiv:2407.04620 Figure 4) that for a 125M-parameter TTT-Linear network the per-token TTT loss `ℓ(W_t; x_t)` is **monotonically decreasing but does not reach a fixed point within sequence length T=2048** — `W_t` is a moving target throughout the forward pass. LaCT (arXiv:2505.23884 abstract) further reports that small minibatch sizes yield "extremely low FLOPs utilization" because the inner gradient steps are small and many; the entire LaCT contribution is to deliberately defer fast-weight stabilization until a chunk boundary. Longhorn (arXiv:2407.14207) recovers this picture from the SSM side: the SSM update IS an amortized online learning step, so its hidden state is non-stationary by construction along the sequence axis.

**M2. TRM's outer loop reads the hidden state K times within the SAME forward pass.** TRM (arXiv:2510.04871 §4.1) defines a full recursion as `n` evaluations of `f_L` on `(z_L + z_H + x)` followed by one evaluation of `f_H` on `(z_L + z_H)`, repeated `T` times outside of gradients. Crucially, every iteration of the TRM outer loop **re-runs the same backbone forward pass on the same token positions** to produce a refined latent `z`. If the backbone contains TTT layers, then iteration k of TRM reads the inner fast-weight trajectory `W_1, …, W_T` produced by iteration k−1's input embedding sequence. But iteration k provides a *different* input embedding sequence (because TRM updates `(y, z)` between iterations, arXiv:2510.04871 §4.2) — so `W_t^(k)` ≠ `W_t^(k−1)`.

**M3. Two distinct destructive regimes follow.**

- **M3a (stale fast-weight reads).** If TRM iteration k+1 is computed before the TTT inner loop has stabilized `W_T^(k+1)` ≈ `W_T^(k)`, then iteration k+1's outer-loop update to `(y, z)` is conditioned on a fast-weight trajectory that has not yet absorbed the iteration-k change. The outer loop's effective gradient signal during deep supervision (arXiv:2510.04871 §2.4 / §4.1: TRM back-propagates through the full final recursion) flows through a `W_t` that depends on inputs the supervision target does not match. Bai et al. 2019 fixed-point gradient theory (which TRM cites for HRM and replaces with full back-prop through n evaluations) explicitly requires a fixed-point assumption that TTT (arXiv:2407.04620 Figure 4) does not satisfy.

- **M3b (over-fitted fast-weights).** Conversely, if at large η or many inner steps the TTT inner loop converges *too* fast on iteration k's inputs, then iteration k+1 sees a `W_T^(k)` that has memorized iteration k's now-stale latent — the outer loop has no effective state to refine. This is the analogue of the "over-fitting on a single test sequence" failure noted as a TTT instability mode in arXiv:2505.23884 (LaCT motivation).

The tension between M3a and M3b means there exists an intermediate K_arch where the fast-weight state is *partially* stabilized — neither stationary enough for outer-loop reads to be consistent across iterations nor unstable enough for the outer loop to dominate. In this band, different seeds will converge to different `(W, y, z)` trajectories. **This is the variance-amplifying middle band** the hypothesis predicts. At K_arch=1, no outer-loop reads happen, no interference exists. At K_arch=16, the outer loop has dominated and the TTT layer is being implicitly used as a near-static linear-attention layer (arXiv:2602.21204 establishes the TTT-Linear / linear-attention equivalence at fixed `W`).

**M4. CRUXEval is the right test surface for the destructive interaction.** CRUXEval (arXiv:2401.03065 §1) consists of 800 short Python functions (3-13 lines) with input/output pairs; per-instance loop and recursion depth are an *invariant of the program AST*, not a property of the model. arXiv:2401.12947 establishes that dense Transformers fail at structural recursion in a way that depends on per-instance recursion depth — providing the per-instance depth-axis baseline. Reasoning with Latent Thoughts (arXiv:2502.17416 abstract) argues looped models match deeper unlooped models *for reasoning at fixed parameter count*. CRUXEval's CoT-helpful and CoT-harmful subsplits (arXiv:2401.03065 §5 cited in gap-finder) let us factor out text-CoT confounds: the destructive prediction must hold under the no-CoT condition, where program-depth d is the only chain-length variable. CRUXEval-X (arXiv:2408.13001) provides 19-language extension as a robustness control.

**M5. Why this is non-additive.** The K_arch=1 TTT baseline is stable across seeds (arXiv:2407.04620 Figure 4 shows tight loss curves). The K_arch=4 TRM × Mamba and TRM × dense backbones are stable across seeds (arXiv:2510.04871 Table 1, Sudoku-Extreme TRM ablations show consistent ≥80% accuracy, no high variance). Yet the predicted TRM × TTT variance-lift at K_arch=4 cannot arise as a sum of these — it is a multiplicative interaction term. The two loops do not share an additive reasoning budget; they share a *substrate*, the fast-weight state, that one writes to and the other reads from at incompatible rates.

**Speculative-but-flagged.** Whether the variance lift manifests as bimodal seed distributions (some seeds finding M3a regime, some finding M3b), or as continuous broadening, is not pinned down by prior work — predicted as bimodal but listed in §7 as a risk.

## 4. Predicted outcome with magnitude

Setup (sketch — eval-designer to detail):
- Three backbones at matched parameter count (∼125M, the TTT paper's anchor scale, arXiv:2407.04620): **TTT-Linear** (the simplest TTT variant); **Mamba/SSM** (no inner-loop fast weights, Longhorn-style); **dense softmax attention** (TRM-faithful, attention-on per arXiv:2510.04871 §4.5 ablation showing 74.7% with self-attention on Sudoku).
- TRM-style depthwise recursion wrapped on top with K_arch ∈ {1, 2, 4, 8, 16}, weight-tied across iterations, deep-supervised per arXiv:2510.04871 §2.4.
- ≥5 seed replicates per (backbone, K_arch) cell. CRUXEval-O (output prediction) primary metric; per-instance program recursion depth d extracted via AST static analysis. No text-CoT in primary condition.

Predicted magnitudes (mean accuracy ± across-seed std, fixed d=3):

| Backbone | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| TTT-Linear | 32 ± 1.5 | 33 ± 4.0 | 30 ± **6-9** | 35 ± 4.0 | 38 ± 2.0 |
| Mamba (control) | 30 ± 1.5 | 33 ± 1.5 | 36 ± 1.5 | 38 ± 1.5 | 39 ± 1.5 |
| Dense softmax (control) | 35 ± 1.5 | 39 ± 1.5 | 42 ± 1.5 | 44 ± 1.5 | 45 ± 1.5 |

Anchor: GPT-4 reaches 63% on CRUXEval-O at vastly larger scale (arXiv:2401.03065 §1); 125M open models score in the 25-40% band. Predicted std at K=1 anchored to typical seed-replicate variance for 125M models on multiple-choice / exact-match benchmarks (≈1-2 absolute points).

The signature of the hypothesis is the **single TTT × K∈{2,4} cell with std ≥3.0 absolute points**, not the mean curve shape. Mean accuracy at K=4 may even be slightly *below* K=1 for TTT (a sign of M3a/M3b dominance), but the load-bearing prediction is the variance lift, not a sign-flipped mean.

**Where the hypothesis should hold (positive predictions):**
- d ∈ {3, 4, ≥5} (deeper-recursion CRUXEval instances, where TRM's outer loop is most consequential).
- No text-CoT (the primary condition).
- TTT-Linear and LaCT separately (LaCT predicted to *reduce* but not eliminate the variance lift, because LaCT chunks fast-weight updates and partly stabilizes them — predicted std at K=4 LaCT ≈ 4.0, intermediate between TTT-Linear ≈ 6-9 and Mamba ≈ 1.5).
- Replicates on CRUXEval-X (arXiv:2408.13001) Java and C++ subsets.

**Where the hypothesis should NOT hold (negative predictions):**
- d = 1 (single-pass programs): all backbones near-flat in K_arch.
- With text-CoT enabled: CoT provides an external chain-length axis that swamps the interaction term — predicted std for TTT × K=4 with CoT ≤2.5 (CoT-helpful sub-split).
- K_arch=1 (no outer loop): TTT and controls within ±1 std of each other.
- At parameter scales <30M: TTT inner-loop effects too small to interfere; predicted std lift ≤1.0.

## 5. Falsification criteria

All criteria use CRUXEval-O exact-match accuracy as the primary metric. Across-seed std uses ≥5 seeds per cell, std computed across seed-mean accuracies (not within-seed micro-batch variance). All conditions: no-CoT, fixed d=3 unless noted.

**F1 (variance lift, primary).**
- Metric: σ(TTT, K_arch=4) where σ is across-seed std of CRUXEval-O accuracy.
- Threshold: σ(TTT, K=4) < 2.0 × σ(TTT, K=1).
- Direction: BELOW-THRESHOLD falsifies. (Predicted: σ(TTT, K=4) ≥ 4× σ(TTT, K=1), i.e., ≥6.0 abs points if K=1 is ≈1.5.)

**F2 (interaction term, load-bearing non-additivity).**
- Metric: (σ(TTT, K=4) − σ(TTT, K=1)) − (σ(Mamba, K=4) − σ(Mamba, K=1)).
- Threshold: < +1.5 absolute points.
- Direction: BELOW-THRESHOLD falsifies. The destructive interaction must dominate the marginals; if the variance lift is matched in the Mamba control, it is not a TTT-specific phenomenon and the mechanism (M3a/M3b clock-mismatch) is wrong. (Predicted: ≥+3.0.)

**F3 (non-monotone K_arch curve).**
- Metric: σ(TTT, K=4) compared to interpolation between σ(TTT, K=1) and σ(TTT, K=16).
- Threshold: σ(TTT, K=4) − ½ · (σ(TTT, K=1) + σ(TTT, K=16)) < +1.0.
- Direction: BELOW-THRESHOLD falsifies. The hypothesis predicts a *peaked* variance curve in K_arch with maximum at K∈{2,4}; if the curve is monotone in K (TTT just gets noisier with more outer loops), the M3a/M3b *intermediate-K* mechanism is wrong even if marginals look right. (Predicted: variance peak ≥+3.0 above the K=1/K=16 midpoint.)

**F4 (depth-axis specificity, mechanism check).**
- Metric: σ(TTT, K=4, d=1) compared to σ(TTT, K=4, d≥4).
- Threshold: σ(TTT, K=4, d≥4) − σ(TTT, K=4, d=1) < +1.5.
- Direction: BELOW-THRESHOLD falsifies. The destructive interaction should grow with program-recursion-depth (since TRM's outer loop is most needed for deeper d). If shallow-d shows the same variance lift, the variance is a generic training-instability artifact, not a depth-of-reasoning interaction. (Predicted: ≥+2.0.)

**F5 (CoT-confound control, recursion-vs-CoT distinction).**
- Metric: σ(TTT, K=4) under no-CoT minus σ(TTT, K=4) under CoT-helpful sub-split.
- Threshold: < 0 (i.e., CoT condition has equal-or-larger variance).
- Direction: BELOW-THRESHOLD or negative falsifies. If the variance lift is invariant to whether external text-CoT is provided, the phenomenon may be a generic training-noise issue, not a *latent* depth-of-reasoning interaction. (Predicted: no-CoT std ≥ +2.0 above CoT std.)

## 6. Required experiments (sketch — eval-designer to specify)

**Datasets.**
- CRUXEval-O (arXiv:2401.03065), 800 functions, with per-instance program-recursion-depth `d` extracted by AST static analysis (count call-depth + loop-nesting); CoT-helpful / CoT-harmful sub-split provided in the original release.
- CRUXEval-X Java + C++ subsets (arXiv:2408.13001) for cross-language robustness.
- Optional: arXiv:2401.12947 structural-recursion synthetic benchmark for a dense-only sanity baseline.

**Backbones (matched 125M parameters, ≥5 seeds each).**
- TTT-Linear (arXiv:2407.04620 reference impl, https://github.com/test-time-training/ttt-lm-jax).
- LaCT (arXiv:2505.23884) as a TTT variant.
- Mamba (arXiv:2407.14207 baseline lineage) as no-fast-weight SSM control.
- Dense softmax attention (TRM faithful, arXiv:2510.04871 §4.5).

**TRM operator wrapping.** Weight-tied K_arch outer iterations of the full backbone, deep supervision per arXiv:2510.04871 §2.4. K_arch ∈ {1, 2, 4, 8, 16}.

**Ablations.**
- TTT learning rate η ∈ {η_default, η_default/10}: tests whether reducing fast-weight non-stationarity moves the variance peak.
- LaCT chunk size: tests whether explicit fast-weight stabilization eliminates the peak.
- η = 0 (frozen TTT layer ≈ linear attention, arXiv:2602.21204): the variance lift must vanish; failure to vanish falsifies the mechanism (extra falsifier F6).

**Pre-training corpus.** Identical across all (backbone, K_arch) cells — same code-pretraining mixture, identical token count. Held-out instruction-tuning split with no CRUXEval contamination (arXiv:2401.03065 §3 release-date controls).

**Compute envelope.** ~5 seeds × 5 K_arch × 4 backbones × ~125M params × ~50B tokens ≈ 100 model-days at 8×H100, ≈ ~3,000-4,000 GPU-hours; above the 2,000-GPU-hour fence — see §6a for kill-switch.

### 6a. Cheaper falsification path

Single test-time wrapping experiment, no pre-training:

1. Take an open-weights TTT-Linear or LaCT checkpoint (arXiv:2407.04620 release / arXiv:2505.23884 release).
2. Take an open-weights Mamba checkpoint at matched scale.
3. For each, apply TRM-style K_arch ∈ {1, 2, 4, 8} weight-tied recursion as a *test-time* wrapper on the final block (Huginn-style inference recursion, arXiv:2502.05171 §5 methodology).
4. Evaluate CRUXEval-O at d=3, 5 seeds varying *only* the random projection used to initialize the weight-tied wrapper (since the underlying checkpoint is frozen, this is the only variance source — but the prediction is that even this small variance source amplifies through the TTT × K=4 cell).
5. Report σ(TTT, K=4, init-seed) vs σ(Mamba, K=4, init-seed).

Expected cost: ~50 GPU-hours on a single 8×H100 node. **Kill-switch:** if σ(TTT, K=4) − σ(TTT, K=1) < +1.0 and σ(Mamba, K=4) − σ(Mamba, K=1) is similar, then the destructive-interaction prediction is dead before any pre-training run.

**Caveat (important).** Test-time-only TRM recursion on a non-recursion-pre-trained model is a *weaker* probe than full pre-training (per Ouro arXiv:2510.25741 — they argue iterative computation must be baked in at pre-training to fully manifest). A null result in §6a is suggestive but does not strictly falsify the full-pre-training prediction; a *positive* result in §6a is strong evidence to fund the full experiment in §6.

## 7. Risks to the hypothesis

**R1. The variance lift may only appear at scales >>125M.** TTT inner-loop expressivity grows with the fast-weight model size; at 125M, the inner `f` is small enough that its non-stationarity may be absorbed by deep-supervision regularization. If this risk materializes, the work still contributes (a) the first joint TRM × TTT ablation in the literature, (b) the CRUXEval-with-program-depth-axis instrument as a per-instance reasoning-depth probe, and (c) a negative result on the destructive-interaction prediction with a specific scale boundary that informs follow-up.

**R2. The variance lift may be bimodal across seeds (some seeds fall in M3a regime, some in M3b) rather than continuous broadening.** Bimodality would still satisfy F1-F3 (std is increased by either pattern) but would point to a different remediation (mode-aware training rather than fast-weight stabilization). The hypothesis is robust to this — std lift is the load-bearing observable, not the shape of the seed distribution. Still, eval-designer should report the seed histogram, not just std.

**R3. CRUXEval may saturate the variance signal because its 800 items are too few.** Per-cell std on a 800-item benchmark with per-d sub-splits has noisy estimation. If standard-error bars on σ are wide enough that F1's 2.0× threshold is statistically inconclusive, the work still contributes the methodology and a stronger benchmark recommendation (CRUXEval-X 19-language at ~15K items, arXiv:2408.13001). The hypothesis is then *unfinished* rather than falsified — eval-designer should pre-register a sample-size analysis.

**R4. The two recursions may genuinely *not interact at all*.** The simplest red-team objection: TTT and TRM operate on orthogonal substrates (sequence-time vs depth-time), and the fast-weight non-stationarity simply doesn't propagate to the outer-loop deep-supervision gradients. If this is so, F1-F3 all return near-zero. The work then contributes the first negative result for the destructive-interference hypothesis and *opens* the redundant-vs-complementary question (the spec's other two candidate failure modes, gap-finder-1 §Gap6) to follow-up — the experimental scaffolding is reusable.

**R5. Mamba may not be a clean control because its hidden state is also non-stationary along the sequence (arXiv:2407.14207 amortized-online-learning framing).** If Mamba shows a smaller-but-nonzero variance lift, the dichotomy "TTT has fast weights / Mamba doesn't" is weaker than M1 claims. Mitigation: include both Mamba and a *truly stationary* control — a fixed linear-attention layer (the η=0 TTT, arXiv:2602.21204 equivalence) — as a third control. The η=0 TTT control is also F6 above.

## 8. Sources

- arXiv:2510.04871 — Less is More: Recursive Reasoning with Tiny Networks (TRM). Sections referenced: §2.4 deep supervision, §4.1 full-recursion gradient, §4.2 (y,z) reinterpretation, §4.5 attention-on ablation.
- arXiv:2407.04620 — Learning to (Learn at Test Time): RNNs with Expressive Hidden States (TTT). Sections referenced: §2.1 TTT updating hidden state, §2.2 inner/outer loop, Figure 4 inner-loop non-convergence at T=2048.
- arXiv:2505.23884 — Test-Time Training Done Right (LaCT). Cited for fast-weight FLOPs-utilization analysis and chunked stabilization.
- arXiv:2602.21204 — TTT with KV Binding Is Secretly Linear Attention. Cited for the η=0 / static-fast-weight equivalence used in F6 control.
- arXiv:2504.05298 — One-Minute Video Generation with TTT. Cited for TTT-on-Transformer architecture composition precedent.
- arXiv:2407.14207 — Longhorn: SSMs as Amortized Online Learners. Cited for SSM = sequence-time online learner framing (R5 control concern).
- arXiv:2506.21734 — Hierarchical Reasoning Model (HRM). Predecessor to TRM, cited for fixed-point assumption that TRM removes.
- arXiv:2502.05171 — Scaling up Test-Time Compute with Latent Reasoning (Huginn). Cited for inference-time recursion methodology in §6a.
- arXiv:2510.25741 — Scaling Latent Reasoning via Looped Language Models (Ouro). Cited for "iteration must be baked in at pre-training" caveat in §6a.
- arXiv:2507.10524 — Mixture-of-Recursions (MoR). Cited for adaptive depth precedent.
- arXiv:2502.17416 — Reasoning with Latent Thoughts: On the Power of Looped Transformers. Cited for "looped models match deeper unlooped models for reasoning at fixed parameter count" theoretical baseline.
- arXiv:2311.12424 — Looped Transformers are Better at Learning Algorithms. Cited for theoretical iterative-algorithm grounding.
- arXiv:2410.01405 — On Expressive Power of Looped Transformers. Cited for K_arch expressiveness analysis.
- arXiv:2401.03065 — CRUXEval. Primary test surface; §1 sample shown, §5 CoT subsplits.
- arXiv:2408.13001 — CRUXEval-X. Robustness control across 19 languages.
- arXiv:2401.12947 — Transformer-Based Models Are Not Yet Perfect at Learning to Emulate Structural Recursion. Negative-result baseline for dense recursion-depth scaling.
