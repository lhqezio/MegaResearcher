# Hypothesis H1-FB: The Compressed-Summary Fallback is the Pivot of Sparse-Recursion Fusion

**Worker:** hypothesis-smith-1
**Targeting gap:** H1 "Sparse-fusion-lift" (composes A1+B1+B4) — the (TRM-style depthwise recursion × natively-trainable sparse-attention backbone) cell of the design grid is empty. Source: `docs/research/runs/2026-05-10-0729-766039/gaps.md#H1` and the gap-brief block in the spec.
**Revision:** 0

---

## Why this single hypothesis (motivation for the choice)

The gap brief surfaces three plausible fusion-cell hypotheses: (a) raw "does TRM-on-NSA outperform TRM-on-dense?", (b) "do retrieval heads survive sparse recursion?", and (c) "does the sparse pattern's *fallback structure* mediate the interaction between depth and width?". Hypothesis (a) is structurally additive and the spec forbids "both work." Hypothesis (b) is a mechanistic probe but is downstream of (c) — retrieval-head survival presupposes a channel through which non-selected tokens can be re-attended on later iterations. Hypothesis (c) is the *only* prediction that (i) is non-additive in the spec's sense (the same recursion operator on two architecturally similar sparse backbones gives **opposite-sign** deltas), (ii) is falsifiable on commodity hardware via an ablation that toggles a single sub-module, and (iii) directly weaponizes the published asymmetry between NSA, MoBA, and DSA that the gap brief flags. So I commit to (c).

---

## 1. Targeted gap

The gap brief (`gaps.md#H1`) establishes that no published architectural-recursion paper (TRM arXiv:2510.04871, HRM arXiv:2506.21734, Universal Transformer arXiv:1807.03819, Huginn arXiv:2502.05171, Ouro arXiv:2510.25741, MoR arXiv:2507.10524) has ever been instantiated on a *natively-trainable, learned-routing* sparse-attention backbone (NSA arXiv:2502.11089, MoBA arXiv:2502.13189, DSA arXiv:2512.02556, SeerAttention-R arXiv:2506.08889). The lone partial precedent — Parallel Loop Transformer (PLT, arXiv:2510.24824) — pairs loops with the simplest fixed-pattern Gated Sliding-Window Attention (G-SWA), strictly weaker than NSA/MoBA/DSA, and never analyzes whether tokens dropped in early loops survive into late loops.

Critically, the gap brief flags an asymmetry that no paper has tested: **NSA preserves a compressed-summary token-pool fallback for blocks not in the top-k selection (arXiv:2502.11089 §2: compressed branch summing alongside the selected branch and the sliding-window branch); MoBA's mixture-of-block-attention drops blocks below the top-k gate entirely (arXiv:2502.13189 §3); DSA's top-k cutoff is hard (arXiv:2512.02556).** Whether recursion can productively re-visit evidence that iteration-1 attention dropped is therefore architecturally contingent on the fallback. No published work tests this contingency.

## 2. Hypothesis statement

**If** a TRM-style recursion operator (a single weight-tied transformer block applied K=6 times within one forward pass, per arXiv:2510.04871 §3) is wrapped around (i) an NSA backbone (with compressed-summary fallback retained, arXiv:2502.11089), (ii) a MoBA backbone (no fallback, arXiv:2502.13189), and (iii) an NSA-no-fallback ablation (NSA with the compressed branch zeroed out), and all three are evaluated on BABILong (arXiv:2406.10149) at haystack length L = 64K with reasoning depth k ∈ {2,3,4,5} hops, **then** at matched parameter count and matched training tokens the NSA-with-fallback variant will exhibit a *positive, monotone-in-K* accuracy gain over its K=1 single-pass baseline of at least **+8 absolute points averaged across k ≥ 3 tasks**, while the MoBA and NSA-no-fallback variants will exhibit a *flat or negative* slope in K on the same tasks (≤ +2 absolute points and possibly negative). The dense-attention TRM control will be intermediate. The recursion×fallback interaction term will therefore be *non-additive*: the lift from "+recursion" depends on the fallback channel's presence in a way that the lift from "+sparse" alone does not predict.

## 3. Mechanism

**Three grounded sub-claims, each cited.**

**(M1) Architectural recursion implements iterative refinement of a latent reasoning state, not chain-of-thought-as-text.** TRM (arXiv:2510.04871 §3) and Huginn (arXiv:2502.05171 §2) explicitly iterate a recurrent block in latent space; Ouro (arXiv:2510.25741) builds this into pre-training with entropy-regularized depth allocation; MoR (arXiv:2507.10524) makes recursion depth token-adaptive. Each pass re-reads the input through the same operator with an updated latent. This means iteration t+1's attention pattern is computed from queries that depend on iteration t's output. **For multi-hop reasoning, the t+1 query may need to attend to evidence that the t-th query did not select.** This is the recursion×attention coupling that distinguishes architectural recursion from text-CoT: in text-CoT the model emits tokens and re-reads via standard attention over the emitted tokens; in architectural recursion the *attention pattern itself is recomputed on a refined latent without emitting any tokens*, so the eligibility set of attendable evidence is whatever the sparse pattern admits at iteration t+1.

**(M2) Learned-sparse-attention selection is *query-conditioned*, so a refined query at iteration t+1 routes differently — but only over the eligibility set the architecture exposes.** NSA (arXiv:2502.11089 §2) selects top-k blocks via a learned gate over the current query; MoBA (arXiv:2502.13189 §3) does the same; DSA (arXiv:2512.02556) and SeerAttention-R (arXiv:2506.08889 §2) similarly use query-dependent gating. *However*, NSA additionally maintains a "compressed" branch that summarizes *all* blocks (selected or not) into pooled tokens whose attention weights are computed alongside the selected branch. MoBA and DSA do not retain this. **Consequently, in NSA, iteration t+1's refined query has a low-rate but non-zero channel onto blocks dropped at iteration t; in MoBA/DSA, blocks dropped at iteration t are invisible at iteration t+1 unless the refined query happens to re-select them via the top-k gate.** The Sparse Frontier paper (arXiv:2504.17768 §4) shows empirically that "Vertical-Slash for retrieval, Block-Sparse for reasoning" — pattern shape governs task profile — but does not test whether an iterated query benefits from retaining a fallback channel.

**(M3) Retrieval heads are sparse, intrinsic, and causally responsible for arbitrary-position retrieval (arXiv:2404.15574 §3-4).** Their behavior is the substrate by which a refined latent at iteration t+1 *uses* the fallback channel: in NSA, the compressed branch provides a low-rate global view that retrieval-head-style queries can exploit; in MoBA/DSA, retrieval heads at iteration t+1 must hit the same blocks via top-k re-selection or the evidence is gone. Combined with M1 and M2, this predicts that recursion's gain on multi-hop tasks (k ≥ 3 BABILong hops) will be *gated* by whether the architecture preserves a non-top-k channel.

**Adversarial cautions, also cited.** ParaThinker / Tunnel Vision (arXiv:2509.04475) shows naive recursion can lock onto wrong paths; this is *consistent* with the prediction — without a fallback channel, recursion may amplify a wrong iteration-1 selection (interference, not synergy). The HRM mechanistic critique (arXiv:2601.10679) shows recursion's fixed-point can be fragile under dense attention; under MoBA/DSA the fragility should be worse, predicting *negative* recursion slope, not merely flat. This is a sharper prediction.

## 4. Predicted outcome with magnitude

**Primary prediction (BABILong qa2/qa3/qa4/qa5 at L=64K, accuracy averaged):**

| Variant | K=1 baseline | K=6 recursion | Δ (recursion lift) |
|---|---|---|---|
| Dense + TRM | A_d | A_d + 4-7 pts | +4 to +7 |
| **NSA (with fallback) + TRM** | A_n ≈ A_d − 1 | **A_n + 8 to 14 pts** | **+8 to +14** |
| NSA-no-fallback + TRM | A_n − 2 | A_n − 2 ± 2 | ≤ +2, possibly negative |
| MoBA + TRM | A_m ≈ A_d − 1 | A_m + 0 ± 3 | ≤ +2 |

**Why the magnitudes.** TRM (arXiv:2510.04871 Tab. 2) reports +5 to +12 points from recursion on hard reasoning tasks at matched params; NSA (arXiv:2502.11089 Tab. 1) shows ~1 point loss vs dense at single-pass on long-context QA. Combining: NSA+recursion should *exceed* dense+recursion by 3-7 points on multi-hop because the compressed-summary fallback gives recursion a "second look" channel that dense attention does not need (dense already sees everything once). MoBA and NSA-no-fallback should match or underperform their K=1 baselines on k ≥ 3 hops because each iteration's hard top-k gate forecloses re-attending dropped evidence; refined queries that need a different block must re-select it from scratch under a noisy gate, and Tunnel Vision dynamics dominate.

**Conditions under which it should hold.**
- Reasoning depth k ≥ 3 (multi-hop). At k=1 (single-needle retrieval), recursion provides little benefit on any backbone.
- Haystack length L ≥ 16K (sparse pattern actually drops blocks; below this NSA/MoBA fall back to near-dense).
- Matched compute and matched training tokens (no recursion-vs-no-recursion FLOP confound; per Ouro arXiv:2510.25741 §4 recursion benefits require matched pre-training).

**Conditions under which it should NOT hold (built-in null cases).**
- On NIAH-style single-hop retrieval (RULER): all four variants should be near-ceiling and differences disappear. If MoBA shows a *bigger* recursion lift on single-hop than NSA, the mechanism is wrong.
- On math/program-synthesis with short context (GSM8K, MBPP): the haystack-length axis is irrelevant; differences should track the published TRM dense-recursion gain (+5-12) uniformly.

## 5. Falsification criteria (≥3, each with metric + threshold + direction)

**F1. Recursion-lift floor on NSA-with-fallback.** Metric: average accuracy across BABILong qa3+qa4+qa5 at L=64K. Threshold: K=6 minus K=1 ≥ +8.0 absolute points. Direction: if the observed lift is **< +5 points** the hypothesis is falsified — the fallback channel did not produce the predicted multi-hop synergy.

**F2. Sign asymmetry between NSA-fallback and MoBA.** Metric: (NSA-with-fallback K=6 − K=1 lift) minus (MoBA K=6 − K=1 lift), averaged over qa3+qa4+qa5 at L=64K. Threshold: difference ≥ +5.0 points. Direction: if the difference is **< +2 points or negative** (i.e., MoBA matches or beats NSA on recursion lift), the fallback channel is not the mechanism — falsified.

**F3. Causal role of the compressed branch.** Metric: NSA-with-fallback K=6 minus NSA-no-fallback K=6 accuracy on qa3+qa4+qa5 at L=64K. Threshold: ≥ +4.0 points. Direction: if zeroing the compressed branch leaves the recursion lift **unchanged (Δ ≤ +1.5 points)**, the fallback is not load-bearing — falsified. (This is the cleanest single ablation; see "cheaper falsification path" below.)

**F4 (bonus). Tunnel-vision signature on MoBA.** Metric: per-iteration probe of the model's prediction at K=1, 2, 3, 4, 5, 6 on qa4 at L=64K (extract logit from each iteration's latent). Threshold: MoBA's accuracy at K=6 ≤ accuracy at K=2. Direction: if MoBA's recursion accuracy is **monotone non-decreasing across K** (no tunnel-vision lock-in), the interference half of the mechanism is wrong — partial falsification of the asymmetry sub-claim.

## 6. Required experiments (sketch — eval-designer details these)

- **Backbones.** Four ~350M-parameter decoder-only models, all sharing tokenizer, optimizer, training data, and pre-training token budget (~30B tokens), differing only in attention: (i) dense softmax, (ii) NSA with all three branches (compressed + selected + sliding-window) per arXiv:2502.11089, (iii) NSA with the compressed branch zeroed at train and eval time (an ablation), (iv) MoBA per arXiv:2502.13189 with matched gate sparsity. Each backbone trained twice: once standard (K=1), once with TRM-style weight-tied K=6 recursion in the last block per arXiv:2510.04871 §3.
- **Primary eval.** BABILong qa1–qa5 at L ∈ {4K, 16K, 64K, 128K} (arXiv:2406.10149).
- **Secondary evals.** RULER (single-hop control), GSM8K + MBPP (short-context recursion control), NoCha (arXiv:2406.16264) and Loong (arXiv:2406.17419) for the R/R+/R++ split — though these are stretch goals at this scale.
- **Mechanistic probes.** (a) Per-iteration accuracy curves; (b) retrieval-head identification per arXiv:2404.15574 protocol on each variant; (c) iteration-2 attention overlap with iteration-1 selection (Jaccard) — predicts NSA shows higher inter-iteration selection drift than MoBA.
- **Compute estimate.** 4 backbones × 2 recursion settings × 30B tokens ≈ 8 pretraining runs at 350M params. At ~4e10 FLOPs/token × 30B tokens × 8 runs ≈ 9.6e21 total FLOPs; on H100 at 700 TFLOPs effective ≈ 3,800 GPU-hours. **Above the spec's 2000 GPU-hour fence** for the full experiment — see cheaper falsification path below.

## 6a. Cheaper falsification path (commodity-hardware single ablation that still kills the hypothesis)

**The decisive single ablation is F3.** Take an *off-the-shelf* NSA pre-trained checkpoint (DeepSeek-V3.2 DSA arXiv:2512.02556 has open weights with the equivalent compressed-fallback structure; or NSA reference implementation if a checkpoint becomes available) and the NSA reference repo. Implement TRM-style K=6 recursion as a wrapper applied to the final transformer block at *inference time only* (no training — equivalent to the "test-time recursion" probe in arXiv:2502.05171 §5). Evaluate two configurations on BABILong qa3+qa4+qa5 at L=32K (smaller because no training):

- **A:** NSA with compressed branch active, K=6 inference recursion.
- **B:** NSA with compressed branch logits set to −∞ at runtime (i.e., the compressed-summary tokens are present but the attention gate excludes them), K=6 inference recursion.

If accuracy(A) − accuracy(B) ≥ +3 absolute points on the multi-hop subset, the fallback hypothesis survives the cheap probe and merits the full pre-training experiment. If accuracy(A) − accuracy(B) < +1 point, the fallback channel is not load-bearing and **the full hypothesis is dead before any pre-training run begins.** This costs ~50 GPU-hours of inference on a single 8×H100 node — well within commodity budget. **Caveat:** test-time recursion on a model not trained for recursion is a weaker probe (per Ouro arXiv:2510.25741 the gain is largest when recursion is in pre-training); a null result here is suggestive but not conclusive against M3, while a positive result is strong evidence for the mechanism.

## 7. Risks to the hypothesis

**R1. The compressed branch may carry too little information.** NSA's compressed branch summarizes blocks into a small number of pooled tokens; per arXiv:2502.11089 §2 the compression ratio is aggressive. If the summary is too lossy, refined queries at iteration t+1 cannot recover the dropped evidence even with the fallback. *What the work still contributes if R1:* a clean negative result with a per-iteration Jaccard-drift measurement establishes the *information bandwidth* required of a fallback channel, informing the design of future sparse-recursion architectures (e.g., learned compression rate that grows with recursion depth).

**R2. MoBA may close the gap via top-k re-selection drift.** If MoBA's gate is sufficiently noisy/expressive, refined queries at iteration t+1 may re-select dropped blocks at a high enough rate to substitute for an explicit fallback. Per arXiv:2504.17768 the gate behavior depends strongly on training. *What the work contributes if R2:* an empirical refutation of the architectural-fallback claim while still answering the core gap-cell question — and a measured "selection drift rate" that is itself a useful new metric.

**R3. TRM-style recursion may not transfer from puzzle/grid tasks to long-context reasoning at all.** TRM (arXiv:2510.04871) was demonstrated on Sudoku/Maze/ARC-AGI; HRM (arXiv:2506.21734) similarly on ARC. The HRM mechanistic critique (arXiv:2601.10679) shows fragile fixed points even on Sudoku-Extreme. If the recursion operator simply does not lift performance on text-domain multi-hop tasks under any backbone, the asymmetry F2 collapses to "no one moves." *What the work contributes if R3:* the *first* published BABILong numbers for an architectural-recursion model on a SubQ backbone is itself the contribution the gap brief explicitly demands — closing the empty design-grid cell, even with a null primary result.

**R4. Confound from training-data exposure to BABILong-adjacent QA.** Different backbones may have inadvertently been pre-trained on different distributions of multi-hop text. *Mitigation:* identical pre-training corpus across all 8 runs; this is in the experimental design, not a hypothesis risk per se, but flagged for eval-designer.

## 8. Architectural-coherence check (per spec)

The hypothesis depends on iteration t+1's refined query attending to evidence dropped at iteration t. Under MoBA/DSA, this is *only* possible via top-k re-selection (no fallback); under NSA-with-fallback, it is possible via the compressed branch. The hypothesis explicitly *predicts a sign asymmetry* arising from this coherence asymmetry. It does **not** depend on recursion accessing tokens the sparse pattern excludes outright — under MoBA/DSA the prediction is precisely that recursion *fails or backfires* because of that exclusion. The hypothesis is structurally coherent under both branches.

## 9. Recursion-vs-CoT distinction (per spec)

This hypothesis concerns *architectural* recursion: weight-tied iteration of a transformer block within one forward pass with no token emission, per TRM (arXiv:2510.04871 §3), Huginn (arXiv:2502.05171), and Ouro (arXiv:2510.25741). It is **not** chain-of-thought, not test-time sampling, not best-of-N. The mechanism (M1) explicitly identifies the latent-state refinement as the operative channel, and the falsification criteria use *per-iteration latent probes*, not text decoding. No CoT prompting is used in any condition.

## 10. Sources

- TRM — arXiv:2510.04871 — Less is More: Recursive Reasoning with Tiny Networks
- HRM — arXiv:2506.21734 — Hierarchical Reasoning Model
- HRM critique — arXiv:2601.10679 — Are Your Reasoning Models Reasoning or Guessing?
- Universal Transformer — arXiv:1807.03819
- Huginn — arXiv:2502.05171 — Scaling up Test-Time Compute with Latent Reasoning
- Ouro — arXiv:2510.25741 — Scaling Latent Reasoning via Looped Language Models
- MoR — arXiv:2507.10524 — Mixture-of-Recursions
- Parallel Loop Transformer — arXiv:2510.24824
- NSA — arXiv:2502.11089 — Native Sparse Attention
- MoBA — arXiv:2502.13189 — Mixture of Block Attention
- DSA / DeepSeek-V3.2 — arXiv:2512.02556
- SeerAttention-R — arXiv:2506.08889
- Sparse Frontier — arXiv:2504.17768
- Retrieval Head — arXiv:2404.15574
- BABILong — arXiv:2406.10149
- NoCha — arXiv:2406.16264
- Loong — arXiv:2406.17419
- ParaThinker / Tunnel Vision — arXiv:2509.04475
