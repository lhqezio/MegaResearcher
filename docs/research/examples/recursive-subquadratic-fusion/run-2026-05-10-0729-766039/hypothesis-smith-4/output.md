# Hypothesis H4 — The halting logit is ill-defined under content-dependent block sparsity, and conditioning halting on the NSA selection-coverage signal produces a measurable, non-additive improvement on multi-hop reasoning that independent training cannot replicate

## 1. Targeted gap

Source: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/gap-finder-1/output.md` Gap 4 (lines 55–66).

Restated: Per-token *depth* halting (ACT 1603.08983, PonderNet 2107.05407, Universal Transformer 1807.03819, MoR 2507.10524, LoopFormer 2602.11451) and per-token *attention* sparse routing (NSA 2502.11089, MoBA 2502.13189, DSA / DeepSeek-V3.2 2512.02556) both produce per-token decisions on the *same* hidden state. Sparse Universal Transformer 2310.07096 and MoEUT 2405.16039 only address sparse-over-experts, not sparse-over-token-positions. No published architecture trains both jointly. The structural worry the gap-finder isolates: when NSA-style top-n block selection drops a key block on iteration t, the halting logit for tokens that needed cross-attention to that block is computed on a corrupted representation. The two routers are jointly underspecified.

## 2. Hypothesis statement

If we train a TRM-style recursive forward pass (K outer loops, weight-tied, per-token PonderNet halting head) on top of an NSA backbone, comparing four coupling regimes — (R0) no halting, fixed K=K\_max, NSA dense-during-training; (R1) independent: PonderNet halting head and NSA selection trained jointly but with no information passed between them; (R2) coverage-conditioned halting: the halting logit at loop k for token t is augmented with a scalar **coverage signal** c_{t,k} = (sum over selected blocks b in NSA top-n at loop k of attention mass on b) / (sum over all blocks of estimated importance at loop k), so halting is suppressed when NSA dropped high-importance blocks; (R3) shared-router: a single router emits both halting and block-selection logits — then on a multi-hop long-context reasoning benchmark (MuSiQue / FRAMES / RULER multi-hop variant) at matched FLOPs budget, R2 outperforms R1 by ≥ 3 points exact-match, R0 by ≥ 1.5 points exact-match, and R3 underperforms R2 by ≥ 1 point exact-match. The R2-vs-R0 gap must hold *only* when the fraction of multi-hop questions whose evidence chain crosses ≥ 3 NSA blocks exceeds ≈ 40%; on single-hop NIAH it should be within ±0.5 points of R0. This is non-additive: R2's improvement is not "halting helps + sparsity helps" but specifically "halting decisions made aware of which blocks were dropped."

## 3. Mechanism

**M1 — NSA selection is content-dependent and discrete, so the post-attention representation at iteration t is a *random function of which blocks were selected*, conditioned on q_t.** NSA's block-selection picks top-n blocks by an importance score derived from compressed-token attention (NSA paper §3, 2502.11089). The token's residual stream after the attention layer therefore omits information from non-selected blocks. This is documented in the NSA paper (the "selection branch" depends on q_t and on per-block importance) and is structurally identical to the Quest / InfLLM importance-score selection that NSA contrasts with (2502.11089 §6.1).

**M2 — PonderNet's halting head is a learned function of the post-attention hidden state.** PonderNet (2107.05407) defines λ_k = sigmoid(W·h_k + b) where h_k is the iterate at loop k; ACT (1603.08983) is structurally similar; the loop is halted with cumulative probability ∏(1−λ_j)·λ_k. By M1, h_k is a function of the selected-block subset, so λ_k inherits a sparsity-induced variance: for tokens whose true halting decision depends on information in non-selected blocks, λ_k is computed on a representation that is by construction missing the discriminative signal.

**M3 — Coverage signals are recoverable from NSA without extra forward passes.** NSA already computes per-block importance scores for the top-n selection (2502.11089 §3); summing the importance mass of the selected blocks over the importance mass of all blocks yields a scalar in [0,1] that measures "how much of the relevant context was actually attended this loop." Conditioning λ_k on this scalar is a one-parameter additive head (no extra forward pass), so R2 is a coupling that costs ~zero extra FLOPs. This is mechanically analogous to MoR's KV-reuse routing (2507.10524) which also reuses already-computed routing scores.

**M4 — Why R2 should beat R3 (shared-router).** A shared router emitting both halting and selection logits forces the two decisions through the same low-rank bottleneck; this is structurally what MoEUT (2405.16039) does for expert-choice + halting and what Sparse Universal Transformer (2310.07096) does for expert sparsity + halting. The MoEUT/SUT papers report that *sparse-over-experts with halting* works at parameter parity with non-shared models — but does not improve over sparse-over-experts alone except via the parameter-efficiency channel. By analogy, a shared halting+selection router cannot encode "halt later precisely because *this loop's* selection was lossy" — it has to commit to one decision before observing the other.

**M5 — Why the effect should depend on multi-hop chain length.** Retrieval Head (2404.15574) shows long-context factuality is mediated by a small set of attention heads that need precise positional access. SCBench 2412.10319 (cited via the gap-finder) shows sparse-attention KV compression is more robust on simple retrieval than on multi-hop. Single-hop tasks rarely need cross-block retrieval; multi-hop tasks chain ≥ 2–3 retrievals across blocks. So coverage-conditioning should matter only when the evidence chain spans multiple blocks.

**M6 — Tunnel Vision compatibility.** ParaThinker / Tunnel Vision (2509.04475) warns that naive depth scaling compounds errors. Coverage-conditioned halting is a *negative-feedback* mechanism (drops in coverage → suppress halting → keep iterating) and should counteract Tunnel Vision specifically when tunnel-vision is caused by premature halting on a sparsity-corrupted state. This is a prediction, not a guarantee.

Speculative element flagged: the precise quantitative form of c_{t,k} (raw mass, log-mass, or KL between selected-block-importance and all-block-importance) is not predicted by prior art. Three concrete forms must be ablated; the hypothesis predicts at least one of them (we suggest log-mass) achieves the stated effect; if all three fail, M3 is falsified.

## 4. Predicted outcome with magnitude

Setup: A ~1B-parameter recursive transformer, K_max = 4 outer loops, weight-tied per loop. Backbone is NSA (2502.11089) with the published block size and top-n schedule. Trained on a long-context mixture (RedPajama subset + multi-hop reasoning data) at 32K context, ≈ 30B tokens. Each regime trained from the same initialization and with FLOPs matched (so R0 with K=4 is the FLOPs upper bound; halting-active regimes get to use the saved compute on more training tokens or more iterations on hard tokens).

| Regime | Training | Predicted MuSiQue EM | Predicted FRAMES EM | Predicted NIAH-multi-hop |
|---|---|---|---|---|
| R0 (no halting, K=4 fixed) | NSA + dense recursion | baseline | baseline | baseline |
| R1 (independent) | NSA + PonderNet, no signal | R0 −1.0 to +0.5 | R0 −1.0 to +0.5 | R0 ±0.3 |
| R2 (coverage-conditioned) | R1 + c_{t,k} feature | **R0 + 1.5 to +3.5** | **R0 + 1.5 to +3.0** | **R0 + 2.0 to +5.0** |
| R3 (shared router) | one head, two outputs | R2 −1.0 to −2.5 | R2 −1.0 to −2.5 | R2 −1.5 to −3.0 |

Magnitude reasoning: SCBench 2412.10319 and Sparse Frontier 2504.17768 report 2–6 point swings between sparse patterns on multi-hop benchmarks at this scale; MoR 2507.10524 reports 1–3 point gains from token-level adaptive recursion at similar scale. R2's predicted magnitude (1.5–3.5 EM on MuSiQue) is at the low end of the SCBench range because we are claiming an *interaction* effect, not a pattern-design effect.

Conditions where R2 should NOT hold:
- Single-hop NIAH (R2 ≈ R0 ± 0.5 EM): the coverage signal carries no useful information when one block suffices.
- Short context (< 4K tokens): NSA selects almost all blocks, c_{t,k} ≈ 1 for all t,k, and R2 collapses to R1.
- Very dense top-n schedules (top-n covers > 80% of blocks): same collapse.
- When K_max = 1: there is no halting decision to condition.

## 5. Falsification criteria

Each criterion is a result that, if observed, falsifies the hypothesis.

**F1 (non-additivity).** On MuSiQue at 32K context, FLOPs-matched, R2 EM ≤ R1 EM + 0.5. Direction: less. Threshold: 0.5 EM. If R2 does not beat R1 by more than 0.5 EM, then conditioning on coverage adds no information beyond what independent training already extracts, and the gap-finder's "halting logit is ill-defined under sparsity" claim is empirically null.

**F2 (sparsity-dependence).** The R2 minus R0 gap on a pure single-hop NIAH benchmark (Needle-in-a-Haystack with one needle, exact match) is ≥ 1.0 EM. Direction: greater. Threshold: 1.0 EM. If R2 wins on tasks where the mechanism predicts no win, the effect is generic recursion-tuning, not coverage-conditioning, and the mechanism (M5) is wrong.

**F3 (coverage-signal-decoding).** The fitted scalar weight on c_{t,k} in R2's halting head, after training, is statistically indistinguishable from zero (|w_c| / std(w_c) < 1.0 across 3 seeds). Direction: less in magnitude. Threshold: 1.0 standardized units. If the model does not actually use the coverage signal yet still outperforms R1, then whatever mechanism is operating is not the one we claimed.

**F4 (collapse under dense selection).** When NSA top-n is set so that ≥ 80% of blocks are always selected (degenerate sparsity), R2 EM minus R1 EM is ≥ 1.0. Direction: greater. Threshold: 1.0 EM. If the gap survives the limit where there is *nothing to drop*, the mechanism is not "halting depends on what was dropped."

**F5 (R3 ordering).** R3 (shared router) EM ≥ R2 EM on MuSiQue. Direction: greater-or-equal. Threshold: 0.0 EM. If a shared router matches or beats coverage-conditioning, then the asymmetric decoupling story (M4) is wrong; the right architectural decision is just "share parameters."

## 6. Required experiments (sketch — eval-designer fills in)

- **Datasets.** Multi-hop: MuSiQue, FRAMES, HotpotQA-distractor at 32K context; RULER multi-hop subset; for the no-effect arm: NIAH-single, NIAH-multi at matched length.
- **Backbones.** A from-scratch NSA model (one of two scales: 350M and 1.3B) with K_max ∈ {1, 2, 4} outer loops, weight-tied. K=1 controls for "is the effect just from PonderNet?"
- **Baselines (mandatory).** (a) Dense-attention TRM at matched FLOPs (does sparsity even help?); (b) NSA without halting at K=K_max; (c) NSA with halting but no recursion (K=1, halting reduces to layer-wise early exit and should be a null result); (d) MoR-style token-level recursion on dense attention (does the effect require sparsity at all?).
- **Ablations.** (i) Form of c_{t,k}: raw mass, log-mass, KL. (ii) Coverage computed from selection branch only vs from compressed branch only vs combined. (iii) Lock NSA selection to a non-content-dependent (random) pattern at the same top-n: R2-with-random-selection should collapse to R0 ± noise — this *is* F4-like and is the cleanest direct test of M1.
- **Metrics.** Exact match on multi-hop QA, F1, average compute budget (E[K_t]·E[k_t/n_blocks]), and the standardized weight |w_c| / std(w_c) on the coverage feature for F3.

## 7. Risks to the hypothesis

**Risk A — The halting head learns to ignore c_{t,k} because the residual stream already encodes "I am missing information."** A sufficiently expressive halting MLP may extract the coverage signal directly from h_k without needing it added explicitly. If so, R1 ≈ R2 (F1 fires), but the gap-finder's underlying claim (sparsity corrupts halting) may still be true; we would have shown only that *explicit coverage conditioning is unnecessary*, not that the joint problem is benign. Contribution under this risk: a clean negative result on whether explicit coupling is needed.

**Risk B — NSA's selection at training time is already coverage-aware (the auxiliary-loss path described in NSA §6.1).** Even though NSA chose the no-auxiliary path, the model may inductively pick blocks such that the residual stream is rarely "missing" critical info; in which case the variance the hypothesis depends on is small, and all regimes converge. Contribution: bounds on how much joint coupling matters in practice for production sparse architectures.

**Risk C — The interaction is dominated by HRM-style fixed-point fragility (2601.10679) before sparsity-induced halting corruption matters.** Recursive operators may fail to converge for reasons unrelated to halting (HRM critique). Contribution: a clean separation of "recursive convergence failure" from "sparsity-induced halting failure" by ablating against dense recursion at matched K.

**Risk D — At achievable training scales (≤ 1.3B, 30B tokens), no regime matches dense-attention baselines, and the *between-regime* differences are within noise of the *NSA-vs-dense* gap.** Sparse Frontier 2504.17768 documents that sparse-attention performance on reasoning is scale-sensitive. Contribution: a clear scaling-frontier signal of when the joint problem becomes architecturally load-bearing.

**Risk E — The chosen coverage statistic c_{t,k} is the wrong functional form, but a different one would work.** F3 fires for our chosen form but not for a different form. Contribution: a forced hyperparameter search (the eval-designer must include three forms) ensures we either confirm or reject the family of forms together.

## 8. Cheaper-falsification path

A single experiment could falsify the hypothesis at < 1/10th the full cost.

**Mini-test:** Take a *publicly released NSA-trained model* (or DSA / DeepSeek-V3.2 2512.02556, since DSA is the closest production-scale instance) and freeze it. Append a 2-layer recursive head (weight-tied, K=4) with a PonderNet halting MLP, and fine-tune only the recursive head + halting head on a small multi-hop QA dataset (~10k examples). Train R1 and R2 variants only. If R2 does not exceed R1 by ≥ 1.0 EM on MuSiQue under this lightweight setup, the hypothesis is at minimum significantly weakened — the pre-trained backbone has presumably internalized the coverage info already (Risk A), so the marginal benefit of explicit conditioning should be a *floor* on the from-scratch advantage. Cost: ≈ 1 GPU-week vs ≈ 8–16 GPU-weeks for the full from-scratch ladder.

Cheaper still (smoke test, < 1 GPU-day): on a fixed NSA-trained backbone, run inference-time only. For each multi-hop test example, compute c_{t,k} per token per loop, and correlate (across examples) the c_{t,k} of the answer-position token at the final loop with answer correctness. If this correlation is < 0.05 in absolute value, M1 is empirically null — sparsity does not correlate with halting-relevant errors — and the hypothesis is dead before the from-scratch run.

## 9. Sources

| arxiv ID | Title | Used for |
|---|---|---|
| 2510.04871 | Tiny Recursive Model (TRM) | Recursion substrate |
| 1603.08983 | Adaptive Computation Time | Halting M2 |
| 2107.05407 | PonderNet | Halting M2, λ_k formulation |
| 1807.03819 | Universal Transformer | Per-position halting |
| 2507.10524 | Mixture-of-Recursions (MoR) | Token-level recursion, KV-reuse analogy M3 |
| 2602.11451 | LoopFormer | Adaptive-depth halting precedent |
| 2502.11089 | NSA | Sparse routing M1, selection branch §3, §6.1 |
| 2502.13189 | MoBA | Sparse routing alternative |
| 2512.02556 | DeepSeek-V3.2 / DSA | Sparse routing alternative, cheap-test backbone |
| 2310.07096 | Sparse Universal Transformer | Sparse-over-experts halting (M4) |
| 2405.16039 | MoEUT | Sparse-over-experts halting (M4) |
| 2404.15574 | Retrieval Head | M5 multi-hop dependence |
| 2412.10319 | SCBench | Sparse-attention multi-turn empirics |
| 2504.17768 | Sparse Frontier | Sparse-pattern reasoning vs retrieval, Risk D |
| 2509.04475 | ParaThinker / Tunnel Vision | M6 |
| 2601.10679 | HRM mechanistic critique | Risk C |
