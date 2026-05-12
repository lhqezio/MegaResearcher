# Hypothesis H1-FB (rev-1): Compressed-Summary Fallback as a *Sufficient* Recursion Substrate, Operationalized via a Difference-in-Differences Test

**Worker:** hypothesis-smith-1
**Targeting gap:** H1 "Sparse-fusion-lift" (composes A1+B1+B4) — the (TRM-style depthwise recursion x natively-trainable sparse-attention backbone) cell of the design grid is empty. Source: `docs/research/runs/2026-05-10-0729-766039/gaps.md#H1`.
**Revision:** 1
**Addresses red-team-1:** all 5 critical objections (citation error; non-additivity isolation; F4 inoperable; DSA != NSA in cheap path; compute-match confound) plus the PLT counter-evidence and four serious objections.

---

## Revision response (red-team round 1)

The red-team's REJECT verdict identified five CRITICAL objections plus one PLT-counter-evidence threat. I address each below; the body of the hypothesis is rewritten to absorb the fixes.

**RT-1 (CRITICAL, Citation error: NSA single-pass deficit).** *I accept this objection fully.* I re-read NSA arXiv:2502.11089 §4 Tables 1-2 and confirmed: NSA averages 0.456 vs Full Attn 0.443 on general benchmarks (+1.3 pts) and 0.469 vs 0.437 on LongBench (+3.2 pts). The original claim "A_n ≈ A_d - 1" was a misreading. **Fix:** the magnitude argument no longer rests on "NSA loses to dense and recursion compensates." I rewrite §4 around a *difference-in-differences* prediction (see RT-2), not around an absolute K=1 deficit. The corrected K=1 prior is: *NSA-with-fallback K=1 is on parity with or slightly above dense at general LongBench, with the multi-hop subset advantage of +0.05 (Tab 2 2WikiHop, HotpotQA) the relevant published baseline.* This *raises* the bar: the recursion lift on NSA must beat dense by the difference-in-differences gap, not by absolute deficit recovery.

**RT-2 (CRITICAL, Non-additivity isolation).** *I accept this objection fully.* The original F1-F4 were satisfied vacuously by the additive model "NSA-K=1 > MoBA-K=1, both gain similarly from recursion." **Fix:** I reformulate the central falsifier as a **difference-in-differences (DiD) with a sign-asymmetry constraint and a K=1-calibration constraint**:
- DiD core: `((NSA-fb K=6) - (NSA-fb K=1)) - ((MoBA K=6) - (MoBA K=1)) >= +5.0 absolute points` on multi-hop BABILong (qa3+qa4+qa5).
- K=1 calibration: `|NSA-fb K=1 - MoBA K=1| <= 2 absolute points` (matched-sparsity calibration enforced at training time; if not achieved, the test is inconclusive and re-run).
- Sign asymmetry: `(NSA-fb K=6 - K=1) > 0` AND `(MoBA K=6 - K=1) <= 0` (one positive, one non-positive). The sign asymmetry is what distinguishes the interaction prediction from any scaled additive prior.
This is now F2 (was F2). The previous F1 (lift floor) survives but is downgraded to a *necessary* condition; the DiD + sign-asymmetry is the *sufficient* falsifier.

**RT-3 (CRITICAL, F4 inoperable).** *I accept this objection.* Per-iteration logit extraction in a model with recursion only in the last block is unsupported by literature; arXiv:2507.02199 (Logit Lens / Coda Lens probe of Huginn-3.5B) reports inconsistencies even on a fully recurrence-pretrained model. **Fix:** F4 is removed. I replace it with **F4' (per-iteration *attention-pattern* probe)**: measure the Jaccard overlap of selected blocks between iteration-1 and iteration-K on NSA-fb vs MoBA at L=64K on qa3+qa4+qa5. The mechanism predicts NSA-fb shows *higher* inter-iteration drift (refined queries re-engage compressed-branch evidence and shift selection) than MoBA (refined queries are stuck to the previous gate's regime). Concretely: `Jaccard(I_1 selected, I_K selected) on NSA-fb < Jaccard on MoBA - 0.10`. Attention selection patterns are directly observable without LM-head probes, and this falsification has published precedent in interp work on sparse attention drift (Sparse Frontier arXiv:2504.17768 §4).

**RT-4 (CRITICAL, DSA != NSA cheap-path).** *I accept this objection fully.* DSA arXiv:2512.02556 §2 is lightning indexer + token-level top-k *with no compressed/summary branch*; treating it as an "NSA stand-in" is architecturally false. **Fix:** the cheap path is restructured into two complementary tests, neither of which conflates DSA with NSA:
1. **Cheap path A (NSA, training-aware ablation, ~250 GPU-hours).** Train two 350M NSA models from scratch on a tightly bounded 5B-token sub-corpus (sufficient for BABILong signal at 16K, per the 350M long-context literature; see noise-floor discussion in §4): NSA-with-fallback and NSA-no-fallback, both with K=6 recursion in the last 4 blocks (deeper than original to make F4' tractable). This costs ~250 GPU-hours total and runs F3 cleanly without conflating architectures.
2. **Cheap path B (DSA as a *separate* prediction, ~50 GPU-hours).** DSA has no compressed branch, so the hypothesis predicts DSA + K=6 inference-time recursion behaves *like MoBA*: flat or negative recursion lift on multi-hop. This is a *bonus* prediction the hypothesis can make on the existing DeepSeek-V3.2 open weights, *not* a stand-in for NSA. A DSA-recursion lift > +5 points would be counter-evidence to the mechanism (predicting M2's claim that "no compressed branch -> no productive recursion channel" is wrong).
The original §6a language ("DSA has equivalent compressed-fallback structure" and "the full hypothesis is dead before any pre-training run begins") is removed.

**RT-5 (CRITICAL, Compute-match confound).** *I accept this objection.* K=6 recursion in the last block has 6x the recursion-block FLOPs of K=1; the comparison is not FLOP-matched. **Fix:** I commit to the *Ouro arXiv:2510.25741 §4 protocol* for FLOP-matching: the K=1 baseline is a *non-shared deeper* variant with the same total inference FLOPs as the K=6 recursion variant. Concretely: if the K=6 model has the recursion block run 6 times (extra 5K FLOPs over K=1), the K=1 baseline replaces those 5 extra runs with 5 *non-tied* additional layers of the same width. This isolates "weight-tied recursion" from "more depth at matched FLOPs," per Ouro Table 5 protocol. The comparison is now `K=1-deep (no weight tying)` vs `K=6-recursion (weight tied)` at matched FLOPs, matched parameter *budget* (recursion saves params), and matched training tokens.

**RT-6 (PLT counter-evidence).** *I partially accept and refine.* PLT arXiv:2510.24824 §3.1 Table 2 row 6 (PLT-3, with G-SWA, no compressed branch, no top-k drop) reports +6.1 average accuracy points over a vanilla non-loop transformer at matched activated parameters (40.8 vs 34.7) — a *positive* recursion lift on a sparse pattern (G-SWA) that lacks NSA's compressed branch. This *is* counter-evidence to the original "compressed branch is *necessary*" framing. **Fix:** I weaken the central claim from *"compressed branch is necessary for productive recursion"* to **"compressed-summary-style fallback is *sufficient* for productive recursion on multi-hop tasks; G-SWA's sliding window is itself a partial fallback (it preserves a local-window channel that can be re-attended on iteration t+1), so PLT's positive lift is consistent. MoBA's gate, in contrast, has no fallback — neither compressed-summary nor sliding-window — and is the architecture for which the hypothesis predicts flat-or-negative lift."** The MoBA prediction is now the load-bearing contrast, *not* "any-non-NSA." The PLT data is reframed as supporting the broader "fallback channel" claim. I also note that the spec's red-team-focus block #3 (architectural coherence) is now satisfied: MoBA has no fallback channel of any kind on dropped blocks; this is the architecturally clean contrast.

**RT-7 (Tunnel Vision misapplied), RT-8 (Magnitude transfer puzzle->text), RT-9 (350M feasibility), RT-11 (M3 uncited at hinge):**
- RT-7 (Tunnel Vision): I drop the Tunnel Vision framing entirely. The MoBA prediction is now grounded in the architectural-coherence argument alone: no fallback channel -> refined queries at iteration t+1 must re-select via the same noisy gate or lose access. The negative-recursion prediction is softened from "≤ +2, possibly negative" to "≤ +3, plausibly flat" — preserving the sign-asymmetry contrast against NSA's predicted +6 to +10 without overcommitting to negative values.
- RT-8 (Magnitude transfer): I drop the +5-12 point TRM-puzzle anchor. The new magnitude prior is anchored on PLT's +6.1 avg-accuracy lift on a *text* MoE at 680M-active params (arXiv:2510.24824 Tab 2), the closest published text+sparse+loop data point. Predicted NSA-fb K=6 lift: +5 to +9 absolute on multi-hop (lower bound roughly tracks PLT's +6.1 on general benchmarks; upper bound reflects multi-hop's higher headroom).
- RT-9 (350M feasibility): I move to a 1B-parameter scale for the full experiment (compute estimate revised below) and run cheap path A at 350M only on the L=16K subset where 350M models have non-floor BABILong performance per BABILong arXiv:2406.10149 §3.1. The full L=64K result is at 1B.
- RT-11 (M3 uncited): I demote M3 from a load-bearing claim to a speculative auxiliary explanation, explicitly flagged as such. The hypothesis no longer requires retrieval-head-like behavior to form *inside* the compressed branch; it requires only that refined queries can re-weight the compressed-branch summary tokens, which is mechanically true by NSA §3 (the gate over compressed tokens is query-conditioned). This is a *weaker, defensible* mechanistic claim.

**RT-13 (Sparse Frontier mismatch):** I drop the Sparse Frontier "Vertical-Slash for retrieval, Block-Sparse for reasoning" appeal in M2; that finding is on training-free sparse attention applied to dense models, not natively-trained NSA/MoBA. The remaining citations to Sparse Frontier in F4' are limited to its Jaccard / pattern-drift methodology, which transfers to natively-trained sparsity.

---

## 1. Targeted gap (unchanged in scope, sharpened in framing)

The gap brief (`gaps.md#H1`) establishes that no published architectural-recursion paper (TRM arXiv:2510.04871, HRM arXiv:2506.21734, Universal Transformer arXiv:1807.03819, Huginn arXiv:2502.05171, Ouro arXiv:2510.25741, MoR arXiv:2507.10524) has been instantiated on a natively-trainable, learned-routing sparse-attention backbone (NSA arXiv:2502.11089, MoBA arXiv:2502.13189, DSA arXiv:2512.02556, SeerAttention-R arXiv:2506.08889). The closest precedent is Parallel Loop Transformer (PLT arXiv:2510.24824 §3.1 Tab 2) which pairs loops with G-SWA; G-SWA is a fixed-pattern *partial* fallback (sliding window over recent context) rather than a learned-routing scheme.

The architectural asymmetry the gap brief flags: **NSA (arXiv:2502.11089 §3) maintains a compressed branch summarizing all blocks alongside the selected branch and a sliding-window branch; MoBA (arXiv:2502.13189 §3) drops blocks below the top-k gate entirely with no fallback channel; DSA (arXiv:2512.02556 §2) is lightning-indexer top-k, also with no fallback channel.** Whether recursion can productively re-attend evidence dropped at iteration 1 depends architecturally on whether *any* fallback exists. PLT's G-SWA is one form of fallback; NSA's compressed branch is another. MoBA and DSA have neither.

## 2. Hypothesis statement

**If** a TRM-style recursion operator (a single weight-tied transformer block applied K=6 times within one forward pass, per arXiv:2510.04871 §3) is wrapped around (i) an NSA backbone with the compressed-summary branch retained (arXiv:2502.11089 §3), (ii) a MoBA backbone with no fallback (arXiv:2502.13189 §3), and (iii) an NSA-no-fallback ablation (NSA with the compressed branch zeroed at train and eval time), and all three are evaluated on BABILong (arXiv:2406.10149) at haystack length L = 64K with reasoning depth k in {2,3,4,5} hops, **then** at FLOP-matched K=1 baselines (per the Ouro arXiv:2510.25741 §4 protocol: K=1 baseline is a non-shared deeper variant with matched total inference FLOPs) and matched training tokens, **the difference-in-differences `((NSA-fb K=6) - (NSA-fb K=1)) - ((MoBA K=6) - (MoBA K=1))` will be >= +5.0 absolute points averaged across qa3+qa4+qa5, conditional on the K=1-baseline calibration `|NSA-fb K=1 - MoBA K=1| <= 2 absolute points`, and the recursion-lift signs will satisfy `(NSA-fb K=6) - (NSA-fb K=1) > 0` AND `(MoBA K=6) - (MoBA K=1) <= 0`.** The compressed-branch-zeroed NSA ablation will recover the MoBA pattern (flat or negative lift). The fallback channel is therefore *sufficient* for productive sparse-recursion on multi-hop reasoning; an architecture with no fallback (MoBA-style hard top-k) does not get a recursion lift.

## 3. Mechanism (M1, M2, M3 — M3 demoted to auxiliary)

**(M1) Architectural recursion implements iterative refinement of a latent reasoning state, not chain-of-thought-as-text.** TRM (arXiv:2510.04871 §3) and Huginn (arXiv:2502.05171 §2) explicitly iterate a recurrent block in latent space; Ouro (arXiv:2510.25741) builds this into pre-training with entropy-regularized depth allocation; MoR (arXiv:2507.10524) makes recursion depth token-adaptive. Each pass re-reads the input through the same operator with an updated latent. For multi-hop reasoning, iteration t+1's query may need to attend to evidence that the t-th query did not select. This is the recursion x attention coupling that distinguishes architectural recursion from text-CoT: in architectural recursion the attention pattern itself is recomputed on a refined latent without emitting tokens, so the eligibility set of attendable evidence is whatever the sparse pattern admits at iteration t+1. PLT arXiv:2510.24824 §3.1 Tab 2 confirms this gives a +6.1 lift on text benchmarks at 680M-active scale.

**(M2) Learned-sparse-attention selection is query-conditioned, but the eligibility set is architectural.** NSA (arXiv:2502.11089 §3) computes attention over three branches (compressed/selected/sliding-window) summed via a learned gate; the compressed-branch summary tokens are themselves invariant across iterations (computed once per forward pass) but the *attention weights* over them are query-conditioned and therefore re-computed at each iteration. MoBA (arXiv:2502.13189 §3) and DSA (arXiv:2512.02556 §2) drop blocks below the top-k gate entirely; there is no fallback channel. **Consequently:** in NSA, iteration t+1's refined query has a low-rate-but-nonzero channel onto blocks dropped at iteration t (via re-weighting compressed summary tokens). In MoBA/DSA, blocks dropped at iteration t are *invisible* at iteration t+1 unless re-selected by the top-k gate. PLT's G-SWA is a partial fallback (sliding window) — different from NSA's compressed branch but functionally a fallback channel — which is consistent with PLT's positive lift (Tab 2). MoBA has *no* fallback; this is the architecturally clean negative case.

**(M3, demoted to auxiliary speculation — explicitly NOT load-bearing).** *I speculate but do not require* that retrieval-head-like behavior (arXiv:2404.15574) re-emerges on the compressed branch under recursion. The Retrieval Head paper is on dense attention; whether retrieval heads form *inside* NSA's compressed branch is unstudied. The hypothesis does not depend on this; the F2 falsifier is satisfied by *any* mechanism that lets refined queries re-weight compressed tokens productively, including non-retrieval-head mechanisms. If retrieval heads do not form there, M2 still suffices — the hypothesis loses interpretive depth but not falsifiable substance.

**Information-bandwidth caveat (was R1, now in mechanism).** NSA's compressed branch summarizes block-size 32 tokens via a learned MLP (arXiv:2502.11089 §4.1); at L=64K this is ~2000 compressed tokens. Whether 2000 compressed tokens preserve enough fine-grained content for multi-hop fact retrieval is the open empirical question this hypothesis tests.

## 4. Predicted outcome with magnitude (re-anchored on PLT, not TRM-puzzle)

**Magnitude anchor (replaces the TRM-puzzle anchor).** The closest published text+sparse+loop data point is PLT arXiv:2510.24824 §3.1 Tab 2, row (6) PLT-3: +6.1 average accuracy lift over the vanilla non-loop transformer at matched activated parameters (680M activated, 13B total, 150B training tokens, 10 standard text benchmarks). PLT uses G-SWA (a partial fallback), not NSA's compressed branch. The hypothesis predicts NSA-with-fallback should approximately *match* PLT's lift on general text reasoning (since both have a fallback) and *exceed* it specifically on multi-hop tasks (where the compressed-branch global view is decisively useful per NSA Tab 2 LongBench HotpotQA/2WikiHop +0.05 single-pass).

**Primary prediction (BABILong qa3+qa4+qa5 at L=64K, accuracy averaged, 1B-parameter models, FLOP-matched K=1 baseline per Ouro protocol):**

| Variant | K=1 baseline | K=6 recursion | Δ (recursion lift) |
|---|---|---|---|
| Dense + recursion | A_d | A_d + 4 to 7 | +4 to +7 |
| **NSA-with-fallback + recursion** | A_n ≈ A_d (parity) | **A_n + 6 to 10** | **+6 to +10** |
| NSA-no-fallback (compressed zeroed) + recursion | A_n − 2 | A_n − 2 ± 2 | ≤ +2, plausibly flat |
| MoBA + recursion | A_m ≈ A_n ± 2 (calibration) | A_m + 0 ± 3 | ≤ +3, plausibly flat |
| (Bonus) DSA + inference recursion | A_dsa | A_dsa + 0 ± 4 | ≤ +4, plausibly flat |

**Why these magnitudes.** PLT-3 (arXiv:2510.24824 Tab 2): +6.1 on general text. NSA single-pass (arXiv:2502.11089 Tab 2): +0.05 on multi-hop subsets vs full attn. The DiD prediction `>= +5.0` is bracketed below PLT's full lift (since PLT's G-SWA partial fallback is weaker than NSA's compressed branch on multi-hop) and above the noise floor at 1B params (per BABILong arXiv:2406.10149 §3.1: 1B-class models are well above floor on qa3-5 at L=64K).

**Conditions under which it should hold.**
- Reasoning depth k >= 3 (multi-hop). At k=1 (single-needle retrieval), recursion provides little benefit on any backbone (Sparse Frontier arXiv:2504.17768 §4.2 finds even single-pass top-k is near-ceiling on retrieval).
- Haystack length L >= 16K (sparse pattern actually drops blocks; below this NSA/MoBA reduce to near-dense).
- FLOP-matched K=1 baseline (Ouro arXiv:2510.25741 §4 protocol) and matched training tokens.
- K=1 calibration `|NSA-fb K=1 - MoBA K=1| <= 2 abs points` achieved (matched-sparsity training; if not, test is inconclusive).

**Conditions under which it should NOT hold (built-in null cases).**
- On NIAH-style single-hop retrieval (RULER): all variants near-ceiling, differences disappear.
- On math/program-synthesis with short context (GSM8K, MBPP): the haystack-length axis is irrelevant; differences should track the published TRM dense-recursion gain uniformly.
- If MoBA-K=6 lift exceeds NSA-fb K=6 lift on multi-hop, the mechanism is wrong.

## 5. Falsification criteria (each with metric + threshold + direction)

**F1 (necessary condition: NSA-fb recursion lift floor).** Metric: NSA-with-fallback K=6 minus NSA-with-fallback K=1, accuracy averaged across BABILong qa3+qa4+qa5 at L=64K, 1B-param model, FLOP-matched baselines. Threshold: lift >= +6.0 absolute points. Direction: if observed lift < +3.0 points, **falsified** — the fallback architecture did not produce the predicted multi-hop synergy under matched compute. This is necessary but not sufficient (additive marginals could satisfy it).

**F2 (sufficient condition: difference-in-differences with sign-asymmetry, K=1-calibrated).** *This is the central non-additive falsifier.* Metric: `[(NSA-fb K=6) - (NSA-fb K=1)] - [(MoBA K=6) - (MoBA K=1)]` averaged over qa3+qa4+qa5 at L=64K, conditional on `|NSA-fb K=1 - MoBA K=1| <= 2 abs points`. Thresholds:
- DiD >= +5.0 abs points,
- AND sign asymmetry `(NSA-fb K=6 - K=1) > 0` AND `(MoBA K=6 - K=1) <= 0` (one strictly positive, one non-positive).
Direction: if DiD < +2 points OR if both backbones show same-sign positive recursion lifts (both > 0 with similar magnitude — the additive failure mode), **falsified**. The sign-asymmetry constraint forecloses the additive null and is the explicit fix to the red-team's RT-2 objection. If the K=1 calibration constraint fails (|NSA K=1 - MoBA K=1| > 2), the test is **inconclusive**, not falsified — re-run with re-calibrated sparsity.

**F3 (causal role of the compressed branch).** Metric: NSA-with-fallback K=6 minus NSA-no-fallback K=6 accuracy on qa3+qa4+qa5 at L=64K. Threshold: >= +4.0 abs points. Direction: if zeroing the compressed branch leaves the recursion lift **unchanged (Δ <= +1.5 points)**, the compressed branch is not load-bearing for the recursion gain, and **the hypothesis is falsified** in its specific compressed-branch claim — though the broader "fallback channel of any kind helps" claim could survive (PLT's G-SWA shows a different fallback also works).

**F4' (per-iteration *attention-pattern* probe — replaces F4 logit-probe).** Metric: Jaccard overlap of selected (top-k) blocks between iteration-1 and iteration-K=6 on NSA-fb vs MoBA at L=64K on qa3+qa4+qa5. The attention selection is directly observable at every iteration without LM-head probing. Threshold: `Jaccard(NSA-fb, I_1, I_6) <= Jaccard(MoBA, I_1, I_6) - 0.10` (NSA shows substantially more inter-iteration drift than MoBA). Direction: if NSA-fb's inter-iteration drift is *not* greater than MoBA's, the proposed mechanism (refined queries productively re-engage compressed-branch evidence at iteration t+1) is wrong — this is partial falsification of the M2 mechanism without falsifying the F2 outcome. This replaces the original F4 (logit-probe), which red-team correctly identified as unsupported by literature for last-block-only recursion.

## 6. Required experiments (sketch — eval-designer details)

- **Backbones (full experiment).** Four ~1B-parameter decoder-only models, all sharing tokenizer, optimizer, training data (~50B tokens of FineWeb-edu equivalent), differing only in attention: (i) dense softmax, (ii) NSA with all three branches per arXiv:2502.11089, (iii) NSA with compressed branch zeroed at train+eval, (iv) MoBA per arXiv:2502.13189 with **matched gate sparsity calibrated to give NSA-fb K=1 - MoBA K=1 within ±2 points on qa3 at L=16K** (calibration check before full L=64K experiment). Each backbone trained twice: once at K=1 with FLOP-matched non-shared deeper variant (Ouro §4 protocol), once with TRM-style weight-tied K=6 recursion on the *last 4 blocks* (deeper recursion footprint than original to make F4' tractable).
- **FLOP-match recipe.** K=6 recursion adds 5 extra block-runs per iteration. K=1 baseline replaces this with 5 *non-tied* extra layers of the same width; total inference FLOPs match. Parameter counts differ (recursion saves params); the recursion variant has the parameter advantage *baked into the test*.
- **Primary eval.** BABILong qa1-qa5 at L in {4K, 16K, 64K, 128K} (arXiv:2406.10149).
- **Secondary evals.** RULER (single-hop control), GSM8K + MBPP (short-context recursion control).
- **Mechanistic probes.** (a) F4' Jaccard drift; (b) per-iteration attention-pattern entropy.
- **Compute estimate (full).** 4 backbones x 2 recursion settings x 50B tokens at 1B params ≈ 8 pretraining runs at 1B params. At ~6e9 FLOPs/token x 50B tokens x 8 runs ≈ 2.4e21 total FLOPs. On H100 at 700 TFLOPs effective ≈ 950 GPU-hours for pretraining + ~150 GPU-hours for evaluation/probing = ~1100 GPU-hours total. **Below the 2000 GPU-hour fence.**

## 6a. Cheaper falsification path (commodity-hardware ablation, restructured per RT-4)

**Cheap path A (NSA-only, training-aware F3 ablation, ~250 GPU-hours, in-fence).** Train two 350M-parameter NSA models from scratch on a 5B-token sub-corpus (the open NSA reference implementation `zen-E/NSA-1B` provides a published config to replicate at 350M scale): (A1) NSA with compressed branch, K=6 recursion on last 4 blocks; (A2) NSA with compressed branch zeroed at train+eval, K=6 recursion on last 4 blocks. Evaluate on BABILong qa3+qa4+qa5 at L=16K (where 350M models are above floor per BABILong arXiv:2406.10149 §3.1). If A1 minus A2 >= +3.0 abs points on multi-hop, the F3 mechanism survives the cheap probe. If A1 minus A2 < +1.0 point, the compressed branch is *probably* not load-bearing. **Caveat:** at 350M and 5B tokens the absolute accuracies are low; a null result here is suggestive but not a kill-test against the 1B/50B full hypothesis. This is explicitly framed as a *go/no-go signal*, not a kill-test.

**Cheap path B (DSA bonus prediction, ~50 GPU-hours, in-fence).** Apply K=6 inference-time recursion (no training) to DeepSeek-V3.2-Exp open weights (arXiv:2512.02556) on BABILong qa3+qa4+qa5 at L=32K. The hypothesis predicts DSA + inference recursion will be flat (lift ≤ +4 abs points), because DSA has no fallback channel (arXiv:2512.02556 §2 confirms only lightning-indexer top-k, no compressed branch). A DSA recursion lift > +5 points would be counter-evidence to the central mechanism. **This is a separate, weaker prediction**, not a stand-in for NSA. The Huginn caveat (arXiv:2502.05171 §5: inference-time recursion on a non-recurrence-pretrained model is a much weaker signal) applies with full force; a null result here is again not a kill-test.

**Total cheap-path budget: ~300 GPU-hours, well under the 2000 GPU-hour fence.** Both cheap paths together provide a triangulation: A tests the compressed-branch causal claim under matched training; B tests the no-fallback prediction on a different no-fallback architecture. Neither alone is decisive; the *full 1B experiment* is the kill-test for F2.

## 7. Risks to the hypothesis

**R1. Compressed branch may carry too little information bandwidth.** NSA's compressed branch at L=64K is ~2000 pooled tokens (block-size 32, per arXiv:2502.11089 §4.1). If summary granularity is too coarse for multi-hop fact retrieval, refined queries cannot recover dropped evidence even with the fallback. *What the work contributes if R1:* a measured bandwidth threshold for fallback-channel utility, plus the F4' Jaccard-drift metric as a useful new measurement.

**R2. MoBA may close the gap via top-k re-selection drift.** If MoBA's gate is sufficiently expressive, refined queries at iteration t+1 may re-select dropped blocks at a high enough rate to substitute for an explicit fallback. *What the work contributes if R2:* an empirical refutation of the architectural-fallback claim; the "selection drift rate" metric is itself a useful contribution.

**R3. TRM-style recursion may not transfer to text-domain multi-hop tasks.** TRM (arXiv:2510.04871) was on Sudoku/Maze/ARC-AGI; HRM (arXiv:2506.21734) similarly on ARC. PLT arXiv:2510.24824 §3.1 Tab 2 reports +6.1 on text *but on general benchmarks, not multi-hop QA at long context*. If text-domain BABILong shows zero recursion lift on any backbone, F2 collapses to "no one moves." *What the work contributes if R3:* the *first* published BABILong numbers for an architectural-recursion model on a SubQ backbone — the gap-cell fill the spec demands — even with a null primary result. Note: this is no longer oversold (per RT-12); it is a real-but-modest contribution.

**R4. K=1 calibration may be unachievable.** If matched-sparsity calibration cannot bring NSA-fb and MoBA K=1 within ±2 points on qa3 at L=16K, F2's K=1-calibration constraint fails and the test is inconclusive. *Mitigation:* run the calibration check before the full L=64K runs; budget extra training only if calibration is achievable.

**R5. PLT's G-SWA partial fallback may dominate NSA's compressed-branch fallback.** PLT shows +6.1 on text at G-SWA. If G-SWA-style sliding-window fallback is *more* effective than NSA's compressed branch for multi-hop recursion, NSA-fb may *underperform* PLT-style backbones. The hypothesis does not test PLT directly (out of scope at the gap-cell), but a future comparison should. *What the work contributes if R5:* establishes that fallback-channel *type* (compressed vs sliding) matters and motivates a follow-up.

**R6. Confound from training-data exposure.** Different backbones may have inadvertently been pre-trained on different distributions. *Mitigation:* identical pre-training corpus across all 8 runs; flagged for eval-designer.

## 8. Architectural-coherence check (per spec)

The hypothesis depends on iteration t+1's refined query attending to evidence dropped at iteration t. Under MoBA (no fallback), this is *only* possible via top-k re-selection on the same noisy gate. Under NSA-with-fallback, it is possible via re-weighting compressed-branch summary tokens. Under PLT/G-SWA, it is possible via the sliding-window channel (a *different* fallback mechanism that the hypothesis predicts also works). The hypothesis predicts a sign asymmetry from this coherence asymmetry between *NSA-fb (and PLT-G-SWA) on the one hand* and *MoBA/DSA (no fallback) on the other*. It does **not** depend on recursion accessing tokens excluded by the architecture; under MoBA the prediction is precisely that recursion fails because of that exclusion. Architecturally coherent under both branches.

## 9. Recursion-vs-CoT distinction (per spec)

This hypothesis concerns *architectural* recursion: weight-tied iteration of a transformer block within one forward pass with no token emission, per TRM (arXiv:2510.04871 §3), Huginn (arXiv:2502.05171), Ouro (arXiv:2510.25741), and PLT (arXiv:2510.24824). It is **not** chain-of-thought, not test-time sampling, not best-of-N, not Recursive Language Models (arXiv:2512.24601 — which is runtime/agentic recursion). The mechanism (M1) explicitly identifies latent-state refinement as the operative channel. F4' uses *attention-pattern* (selection-Jaccard) probes, not text decoding or LM-head logits at intermediate iterations (per RT-3 fix). No CoT prompting in any condition.

## 10. Sources (every citation verified)

- TRM — arXiv:2510.04871 — Less is More: Recursive Reasoning with Tiny Networks
- HRM — arXiv:2506.21734 — Hierarchical Reasoning Model
- Universal Transformer — arXiv:1807.03819
- Huginn — arXiv:2502.05171 — Scaling up Test-Time Compute with Latent Reasoning
- Huginn Logit-Lens / Coda-Lens probe critique — arXiv:2507.02199 — Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer
- Ouro — arXiv:2510.25741 — Scaling Latent Reasoning via Looped Language Models
- MoR — arXiv:2507.10524 — Mixture-of-Recursions
- Parallel Loop Transformer — arXiv:2510.24824 (G-SWA, +6.1 lift on text)
- NSA — arXiv:2502.11089 — Native Sparse Attention (compressed/selected/sliding three-branch design; Tab 1 NSA 0.456 vs Full 0.443; Tab 2 NSA 0.469 vs Full 0.437)
- MoBA — arXiv:2502.13189 — Mixture of Block Attention (no fallback)
- DSA / DeepSeek-V3.2 — arXiv:2512.02556 (lightning-indexer top-k, no fallback — confirmed §2)
- SeerAttention-R — arXiv:2506.08889
- Sparse Frontier — arXiv:2504.17768 (cited only for Jaccard/pattern-drift methodology, per RT-13 fix)
- Retrieval Head — arXiv:2404.15574 (auxiliary citation for M3, demoted from load-bearing per RT-11 fix)
- BABILong — arXiv:2406.10149

## Changes from revision-0 (explicit list per red-team objection)

| Red-team obj. | Severity | Change |
|---|---|---|
| **RT-1 NSA citation error** | CRITICAL | §4 magnitude argument rebuilt from scratch; "A_n ≈ A_d − 1" replaced with "A_n ≈ A_d (parity, +0.013 to +0.032 per Tab 1/2)"; magnitude prior re-anchored on PLT's +6.1 (Tab 2), not on TRM-puzzle. |
| **RT-2 Non-additivity isolation** | CRITICAL | F2 reformulated as DiD with sign-asymmetry constraint: `((NSA-fb K=6) - K=1) - ((MoBA K=6) - K=1) >= +5.0` AND signs asymmetric AND `|K=1 gap| <= 2`. Additive null model now explicitly forecloseable. |
| **RT-3 F4 inoperable** | CRITICAL | F4 logit-probe removed; F4' attention-pattern Jaccard probe added (operationalizable without LM-head). |
| **RT-4 DSA != NSA cheap path** | CRITICAL | Cheap path restructured: A is NSA-only training-aware F3 (250 GPU-hours); B is a DSA-only *bonus* prediction, not a stand-in for NSA. "DSA has equivalent compressed-fallback structure" claim removed. |
| **RT-5 Compute-match confound** | CRITICAL | K=1 baseline now FLOP-matched non-shared deeper variant per Ouro arXiv:2510.25741 §4 protocol. Explicit recipe stated. |
| **RT-6 PLT counter-evidence** | SERIOUS | Central claim weakened from "compressed branch is necessary" to "compressed-summary or sliding-window fallback is *sufficient*; no-fallback is the negative case." MoBA is the load-bearing contrast, not "any non-NSA." |
| **RT-7 Tunnel Vision misapplied** | SERIOUS | Tunnel Vision dropped from M2 entirely. MoBA prediction softened to "≤ +3, plausibly flat." |
| **RT-8 Magnitude transfer** | SERIOUS | TRM-puzzle anchor dropped; PLT text-MoE anchor adopted (+6.1, arXiv:2510.24824 Tab 2). |
| **RT-9 350M feasibility** | SERIOUS | Full experiment moved to 1B params; cheap path A retained at 350M only at L=16K subset (above floor). Compute revised. |
| **RT-11 M3 uncited** | SERIOUS | M3 demoted to auxiliary speculation, explicitly flagged as not load-bearing. F2 falsifier no longer requires retrieval-head behavior on compressed branch. |
| **RT-13 Sparse Frontier mismatch** | MINOR | Sparse Frontier "Vertical-Slash for retrieval, Block-Sparse for reasoning" appeal removed from M2; remaining citation is to Jaccard methodology only. |
| **Compute estimate** | (consequence) | Full: ~1100 GPU-hours (was 3800, over fence). Cheap: ~300 GPU-hours. Both in-fence. |
