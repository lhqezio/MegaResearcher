# Red-team-1 critique of H1-FB ("The Compressed-Summary Fallback is the Pivot of Sparse-Recursion Fusion")

**Worker:** red-team-1
**Critiquing:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-1/output.md`
**Targeting gap (per hypothesis):** A1 + B1 + B4 — (TRM-style depthwise recursion × natively-trainable sparse-attention backbone), the empty design-grid cell.
**Revision round:** 1

---

## Gap re-verification

**Independent queries I ran (this session):**

- `hf_papers search "recursive transformer NSA native sparse attention BABILong"` (limit 10) — returned NSA, MoBA-related, Sparse Frontier, NOSA, Flash Sparse Attention, Combiner, SSA, SeerAttention, Nexus. None pair architectural recursion with NSA/MoBA/DSA on BABILong. Closest: Nexus (2512.03377) which uses "recursive self-attention" but as a *single-pass* refinement of K/Q before the final attention — NOT depthwise weight-tied recursion in the TRM sense. Does not affect the gap.
- `hf_papers search "looped depthwise iteration sparse attention compressed fallback"` (limit 10) — no architectural-recursion + learned-sparse fusion on BABILong. SeerAttention, NOSA, SparseD, Lag-Relative Sparse Attention, ShadowKV, HiP all return; none recurse depthwise.
- `hf_papers search "TRM tiny recursive model long context multi-hop reasoning"` (limit 10) — TRM itself, Deep Improvement Supervision (2511.16886, TRM training scheme), Recursive Models for Long-Horizon Reasoning (2603.02112), Recursive Language Models (2603.15653 / 2512.24601), When Thoughts Meet Facts (2510.07499). The "recursive language models" cluster is *runtime/agentic* recursion — not architectural — and the spec's YAGNI fence explicitly distinguishes these. Does not collapse the gap.
- `hf_papers search "tiny recursive model latent reasoning text long context language"` (limit 8) — Recursive Language Models (2512.24601, runtime), Michelangelo (2409.12640, eval not architecture), survey papers. No counterexamples.

**Verdict on the gap claim:** The gap survives. No published architectural recursion has been instantiated on a natively-trainable sparse-attention backbone with BABILong as the eval. The hypothesis is genuinely targeting an empty cell.

---

## Citation spot-checks

**(SC1) NSA "compressed branch" claim (M2 of the hypothesis).** I read NSA §3 (`hf_papers read_paper 2502.11089 methodology`) and §4 (experiments). NSA does have three branches — compressed, selected, sliding-window — and they are summed via learned gating. The mechanism description is accurate at this granularity. **However**, M2 says "in NSA, iteration t+1's refined query has a low-rate but non-zero channel onto blocks dropped at iteration t." NSA's compressed branch, per §3, summarizes contiguous blocks via a learned MLP; the summarization is computed *once per layer at training time*, not per query. So the "low-rate channel" exists but is NOT query-conditioned — it's a fixed pooled summary that the *attention weights* over it are query-conditioned. The mechanistic story still works (refined queries can re-weight summary tokens), but the hypothesis's framing slightly overstates the channel's expressive power.

**(SC2) NSA reported single-pass deficit vs dense (Table 1, "A_n ≈ A_d − 1").** Reading NSA §4 Tables 1 and 2 (`read_paper 2502.11089 experiments`): Table 1 (general benchmarks): NSA average 0.456 vs Full Attn 0.443 — NSA is +1.3 points *better* on average. Table 2 (LongBench): NSA average 0.469 vs Full Attn 0.437 — NSA is +3.2 points *better*. **The hypothesis's claim "NSA shows ~1 point loss vs dense at single-pass on long-context QA" is factually wrong** — NSA's published numbers show *gains* over full attention at 27B/270B-tokens scale, not a 1-point loss. The hypothesis cites "arXiv:2502.11089 Tab. 1" to justify this — the citation does not support the claim. This is a CRITICAL citation discipline failure.

**(SC3) PLT "loop-recursion provides a benefit at sliding-window sparsity" (cited in critique focus #5 and as an architectural-coherence challenge).** I read PLT §3.1 (`read_paper 2510.24824 "3.1 Accuracy Comparisons under the Same Parameters"`). PLT-3 (with G-SWA, no compressed branch, no top-k block selection) reaches +6.1 average accuracy points over the vanilla non-loop transformer at matched parameters. **This is direct counter-evidence to M2/M3.** The hypothesis predicts that without a "compressed-summary fallback channel," recursion will be *flat or negative*. PLT shows recursion is *strongly positive* on a sliding-window backbone that lacks the compressed-summary channel. The hypothesis acknowledges PLT exists but treats it as "the simplest fixed-pattern sparsity, strictly weaker than NSA" — yet that "weaker" sparsity *does* support productive recursion in the only published data point. The hypothesis's prediction that MoBA's recursion will be "≤ +2 absolute points and possibly negative" is now actively contradicted by the closest published analog.

**(SC4) TRM's "+5 to +12 points from recursion on hard reasoning tasks at matched params."** I read TRM §4.5 and §5 (`read_paper 2510.04871 "5 Results"`). TRM is evaluated at 5M–19M parameters on Sudoku-Extreme (9×9 grid), Maze-Hard (30×30), and ARC-AGI (30×30). Section 4.5 explicitly states TRM uses "an MLP applied on the sequence length" instead of self-attention for fixed-context puzzle tasks. The +5–12 lift cited by the hypothesis is from puzzle/grid tasks where the input is a finite small grid — not text. The hypothesis transfers this magnitude to BABILong at L=64K text on 350M-parameter models. **This is an unjustified domain transfer of magnitude**: small-puzzle attention-free recursion gain ≠ large-text learned-sparse-attention recursion gain. The hypothesis does not justify the transfer with any intermediate evidence.

**(SC5) ParaThinker / Tunnel Vision (2509.04475) cited for "naive recursion can lock onto wrong paths."** I read paper details. ParaThinker frames Tunnel Vision as a *sequential-CoT*-text-emission failure mode, not architectural-recursion. The hypothesis adapts the term to predict architectural-recursion failure on MoBA (M2). This is a non-trivial conceptual leap — the ParaThinker mechanism is about token-emission paths, not weight-tied latent iteration. The hypothesis's own §10 lists 2509.04475 as "ParaThinker / Tunnel Vision," but the borrowed prediction is uncalibrated. Notably, the spec's red-team-focus block #5 explicitly flags this: "is the Tunnel Vision mechanism actually known to apply to architectural latent recursion? H5 is exploring exactly that question; H1 is borrowing the prediction without owning the assumption." This is a SERIOUS citation/mechanism issue.

**(SC6) DSA "has open weights with the equivalent compressed-fallback structure" (cheap path F3).** I read DeepSeek-V3.2 paper details (`paper_details 2512.02556`) and `find_all_resources 2512.02556`. DSA's design is a "lightning indexer" + token-level selection — there is no "compressed branch" in the NSA sense. The DSA architecture is structurally different from NSA (token-level top-k vs NSA's three-branch compressed/selected/sliding). The hypothesis states they have "equivalent compressed-fallback structure" — this is **architecturally false**. F3's cheap-path falsification depends on this equivalence. Open-weights NSA is limited to a single community 1B checkpoint (`zen-E/NSA-1B`, 104 downloads) — not the 27B reference NSA from the paper. The cheap path is *not* readily executable as described.

---

## Mechanism critique

**(M1)** Architectural recursion as latent-state refinement, distinguished from text-CoT — accurately cited (TRM, Huginn, Ouro, MoR). This sub-claim is grounded.

**(M2)** Query-conditioned learned-sparse-attention selection + NSA-only compressed channel onto dropped blocks. Architecturally true at first order, but the M2 framing implies the compressed channel preserves *information* about non-selected blocks in a way that refined queries can productively re-engage. Two unstated assumptions:
- The compressed branch's pooled tokens preserve enough *fine-grained content* to enable retrieval-head-style behavior at iteration t+1. NSA's compression block size is 32; at L=64K this is 2000 compressed tokens, each summarizing 32 raw tokens — not obviously sufficient for multi-hop fact retrieval. The hypothesis acknowledges this in R1 but doesn't quantify the bandwidth.
- Refined queries at t+1 will *learn* to use the compressed branch differently from t=1 queries. There is no published evidence this learning occurs at K=6 recursion in a 350M model trained on 30B tokens. (Ouro 2510.25741 demonstrates entropy-regularized depth allocation works in pre-training, but at much larger scale and with adaptive halting, not fixed K=6.)

**(M3)** Retrieval heads (2404.15574) provide the substrate for compressed-branch usage. The Retrieval Head paper is on dense attention; its applicability to NSA-style compressed branches is unstudied. M3's framing presupposes that retrieval-head-like behavior emerges in the compressed branch of an NSA model — this is the **structurally load-bearing assumption** of the hypothesis and it is **uncited**. If retrieval heads do *not* form in the compressed branch (a plausible alternative — the compressed branch may functionally specialize as a coarse position prior, not as a retrieval substrate), the entire mechanism fails.

**Implicit architectural-coherence issue (Critique focus #3).** M2 claims iteration t+1's refined query has a "channel onto blocks dropped at iteration t" via the compressed branch. But the compressed branch is computed from *all* blocks in the sequence — it is *not* recomputed per iteration when only the recursion is the last block (per §6 of the hypothesis: "TRM-style weight-tied K=6 recursion in the last block"). So the compressed-branch tokens are *fixed* across all 6 iterations; only the attention weights over them change. The hypothesis's framing of "second look channel" presumes the compressed branch carries fact-specific information; if it instead carries position/style information, the second look retrieves nothing new. The hypothesis is one-step away from incoherent on this point — it is structurally coherent only if the compressed branch's information content is rich enough to be the bottleneck. This is not established.

---

## Falsifiability assessment

**F1 (NSA-with-fallback K=6 minus K=1 ≥ +8 abs points on qa3+qa4+qa5).** Operationalizable if we can train both K=1 and K=6 NSA-with-recursion models. **However**: BABILong qa3-qa5 at L=64K is at floor for many open 7-14B models (per BABILong §3.1: best models below 80% on QA3 even at 4K context, much lower at 64K). At 350M parameters, K=1 baseline accuracy is plausibly near random (e.g., 25-35% for 4-option QA), and a +8 absolute lift could be inside the noise floor. The hypothesis does not specify the noise floor or randomness from data ordering / seed. **Falsifiability conditional on noise floor being well below 8 points; this is unstated and dubious at 350M scale.** SERIOUS issue.

**F2 (NSA-fallback minus MoBA recursion lift ≥ +5 points).** Same issue — depends on K=1 baselines being similar across NSA and MoBA at 350M. Sparse Frontier (2504.17768) does not report multi-hop comparisons of NSA vs MoBA; the prior on "what should the MoBA K=1 baseline be?" is missing. The threshold of 5 points is plausible but uncalibrated.

**F3 (NSA-with-fallback K=6 minus NSA-no-fallback K=6 ≥ +4 abs points; or, in cheap path, ≥ +3 inference-only).** This is the most cleanly falsifiable criterion if executable. **But the cheap path has two structural problems:**
(i) NSA is not trained for inference-time post-hoc recursion. The cheap path applies K=6 wrapper to "the final transformer block at inference time only" on a model not trained for recursion. The hypothesis itself (R3 + §6a caveat) acknowledges Ouro shows recursion gains require pre-training. Per Huginn data (Table 4 in `read_paper 2502.05171 5`): the same architecture at K=1 on a recurrence-pretrained model is *catastrophically* worse than at K=32 (e.g., GSM8K-CoT 0.00 vs 42.08); a K=1-pretrained model attempted at K=6 inference is not the same hypothesis.
(ii) Zeroing the compressed branch at inference on a model trained *with* the compressed branch is a hard distribution shift. The variant B (compressed-branch logits set to −∞) is a destructive ablation that breaks training-time invariants. A B − A gap reflects "model can't function without its trained branches" rather than "compressed branch is load-bearing for recursion specifically." **Confounded test**, per critique focus #5.
(iii) No off-the-shelf NSA checkpoint of paper-quality exists. Only one 1B community checkpoint (`zen-E/NSA-1B`) is published. Whether the cheap path runs at all on the actually-available NSA model is uncertain.

The cheap path tests something *different* from the full hypothesis, and the hypothesis acknowledges this only in passing.

**F4 (per-iteration logit probe; tunnel-vision signature on MoBA).** F4 requires extracting logits from each iteration's latent. The proposed architecture has recursion only in the last block (§6: "TRM-style weight-tied K=6 recursion in the last block"). The standard transformer pipeline applies the LM head only after the *entire* stack is run; the logit at K=t<6 requires applying the LM head to a non-final-iteration latent that the LM head was *not* trained to interpret. Per Huginn 2507.02199 ("Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer") — a Huginn-3.5B critique paper — extracting interpretable per-iteration logits is *itself* an open research problem and prior probing reports inconsistencies. **F4 is not straightforwardly operationalizable as a falsification criterion at the proposed scale.** SERIOUS issue.

---

## Strongest counter-argument (steelman)

**The opposing position:** *"The lift NSA+recursion shows over MoBA+recursion is fully explained by NSA's superior single-pass long-context retention, not by any recursion-specific mechanism."*

Construction:
1. NSA-paper Table 2 (LongBench, multi-hop subsets like 2WikiHop and HotpotQA) shows NSA *outperforms* dense full attention at K=1: 0.356 vs 0.305 on 2WikiHop, 0.437 vs 0.350 on HotpotQA. NSA is already better at multi-hop than dense at K=1.
2. MoBA does not publish equivalent multi-hop numbers under matched conditions; if MoBA's K=1 multi-hop is degraded vs NSA, then any recursion lift may simply reflect MoBA's K=1 is worse and its K=6 hits the same ceiling NSA's K=6 does.
3. Sparse Frontier (2504.17768 §4.2) reports that Block-Sparse (NSA-like) helps multi-hop, while top-k page methods like Quest are weaker on multi-hop. This is a *single-pass* effect, not a recursion effect.
4. PLT shows recursion-without-compressed-branch *works* (G-SWA, +6.1 points). So the recursion lift is largely backbone-agnostic.

**Implication:** the F2 sign asymmetry — if observed — could be entirely explained by:
- NSA-K=1 already has higher multi-hop ceiling than MoBA-K=1 (Sparse Frontier-style)
- Both backbones gain similarly from recursion (PLT-style, +5/+6 average)
- The K=6 ceiling on NSA is higher than the K=6 ceiling on MoBA simply because the K=1 starting point was higher
- This explains the "both work" pattern but is *additive* — NOT non-additive. The hypothesis's central claim (non-additivity from compressed-branch fallback) is then unsupported.

**This steelman is a concrete CRITICAL threat.** Critique focus #6 explicitly flagged it: "could the predicted asymmetry be explained by separate marginals (NSA preserves more info than MoBA — this alone could explain both K=1 and K=6 gaps without any architectural-recursion specifically)?" The hypothesis does not foreclose this. Falsification criteria F1–F4 do not isolate the *interaction* term; they test joint outcomes that the additive model also predicts.

To make the hypothesis non-additively falsifiable, F2 would need to be reformulated as a *difference-in-differences*: (NSA K=6 − NSA K=1) − (MoBA K=6 − MoBA K=1) ≥ +5 points, controlled for the K=1 gap (NSA K=1 − MoBA K=1). The current formulation does not control for the K=1 gap, so a "MoBA is just worse" world satisfies the hypothesis vacuously.

---

## Recursion-sense discipline (Critique focus #4)

The hypothesis explicitly distinguishes architectural recursion from text-CoT in §9. **However**, the predicted gain mechanism reduces to "more compute under sparse attention helps long-context tasks":
- Each of K=6 iterations re-applies the same sparse-attention block.
- Each iteration computes new attention weights over the same compressed/selected/sliding-window branches.
- Net effect: more attention compute on the same evidence pool.

This is operationally indistinguishable from "give the model 6× more compute via a deeper non-shared backbone." The hypothesis does not articulate a test that separates "recursion specifically" from "more depth." The Ouro paper does this by comparing fixed-depth at matched FLOPs to recurrence-pretrained at matched FLOPs; the hypothesis hand-waves this with "Conditions: matched compute and matched training tokens" but does not specify how matched-FLOPs is achieved (a 6×-recursion model has 6× FLOPs of K=1 unless the FFN is also reduced).

In §6 the hypothesis says "differing only in attention" but doesn't address whether K=6 vs K=1 share parameter count and FLOPs — by construction K=6 has 6× the inference FLOPs of K=1 (the recursion block runs 6 times). So the comparison K=1 baseline vs K=6 is **NOT compute-matched.** This is a confound that the hypothesis must clean up. SERIOUS issue.

---

## Severity-tagged objections

**Critical (must fix):**

1. **Citation misrepresentation of NSA's single-pass performance vs dense.** [§4 of hypothesis: "A_n ≈ A_d − 1" cites NSA Tab. 1.] The cited table shows NSA *outperforms* dense, not loses 1 point. This reverses the central argument that "NSA+recursion should *exceed* dense+recursion by 3-7 points on multi-hop because the compressed-summary fallback gives recursion a 'second look' channel that dense attention does not need (dense already sees everything once)." If NSA is already better at K=1, the explanation for K=6 lift is muddled.
2. **Additive-vs-non-additive identification failure.** F1–F4 test joint outcomes that are predicted by the additive model "NSA K=1 > MoBA K=1, both gain similarly from recursion." The hypothesis does not isolate the interaction term. Per critique focus #6 and the steelman: this is the single biggest threat to the novelty claim. Reformulation needed: F2 should be a difference-in-differences that controls for K=1 gap, AND the prediction should specify that the K=1 gap should be smaller than the recursion-lift differential (e.g., MoBA K=1 ≈ NSA K=1 within ±2 points by design — perhaps via matched-sparsity calibration).
3. **F4 logit-probing is not operationalizable in the proposed architecture.** Per-iteration logit extraction from a 350M model with recursion in the last block requires the LM head to be applicable to non-final-iteration latents. This is unsupported and prior depth-recurrent probing literature (2507.02199 Huginn critique) reports inconsistencies. Either F4 must be reformulated (e.g., probe attention patterns, not logits), or the architecture must place recursion deeper in the stack with auxiliary supervision per Huginn/TRM.
4. **DSA ≠ NSA structurally; cheap path F3 conflates them.** §6a claims "DeepSeek-V3.2 DSA arXiv:2512.02556 has open weights with the equivalent compressed-fallback structure." DSA does not have a compressed branch; it has a lightning indexer. The cheap-path falsification is not executable on DSA; it requires an actual NSA checkpoint, which is not officially released (only `zen-E/NSA-1B` community checkpoint exists). The cheap path is structurally underspecified.
5. **Compute-matching confound.** K=1 vs K=6 are not FLOP-matched in the proposed design; K=6 has 6× the recursion-block FLOPs of K=1 (the block runs 6 times). The "recursion lift" therefore conflates "recursion" with "more compute." This is the exact confound that the spec's recursion-vs-CoT distinction is meant to prevent. Need explicit FLOP-matched K=1 baseline (e.g., a non-shared deeper variant, or matched-FLOP shallower-FFN configuration).

**Serious (should fix):**

6. **Tunnel Vision misapplied.** ParaThinker 2509.04475's Tunnel Vision is a sequential-CoT failure mode; the hypothesis borrows it for architectural recursion without justification. The "interference, not synergy" prediction for MoBA is uncalibrated and may not occur (PLT data shows it doesn't).
7. **PLT counter-evidence not addressed.** PLT-3 with G-SWA (no compressed branch, no top-k drop) gets +6.1 average accuracy lift from recursion. This *directly contradicts* the prediction "recursion without compressed branch will be flat or negative." The hypothesis dismisses PLT as "strictly weaker than NSA/MoBA/DSA" but the *positive* recursion lift on PLT undermines the asymmetry claim.
8. **Magnitude transfer from puzzle to text.** TRM's +5–12 lift is on 5M-parameter, attention-free, fixed-grid puzzle tasks. Generalizing to 350M-parameter learned-sparse text BABILong is a major domain transfer. No published evidence supports magnitudes at the proposed scale.
9. **350M-parameter BABILong feasibility.** BABILong qa3+qa4+qa5 at L=64K is hard for 7-14B models (best <80% per BABILong §3.1). At 350M the K=1 baseline may be near floor (~25-35%), making +8 absolute points either inside noise or impossible to distinguish from chance. No noise-floor estimate is given.
10. **Scale plausibility — 3,800 GPU-hours overshoots fence.** Hypothesis acknowledges 3,800 > 2,000 GPU-hour fence and offers cheap path F3. But the cheap path tests a different hypothesis (test-time recursion, not pre-training recursion) and has architecture-availability issues (CRITICAL #4). The full experiment is over-fence; the cheap path is over-promised.
11. **Mechanism (M3) is uncited at the load-bearing point.** Retrieval heads forming *inside the compressed branch* is the hinge of M3 but is unsupported by literature. Retrieval Head (2404.15574) is dense-attention-only.

**Minor (suggestion):**

12. **R3 fallback ("first published BABILong on architectural-recursion + sparse backbone") is a real but small contribution.** If the primary prediction fails, the residual contribution is descriptive. The hypothesis's framing in §1 ("the *only* prediction that is non-additive in the spec's sense") oversells. A null R3 outcome would still be publishable but should not be presented as if it satisfies the spec's novelty target.
13. **Sparse Frontier (2504.17768) cited at §3 M2 is mismatched.** Sparse Frontier studies *training-free* sparse attention applied to dense-trained models (Vertical-Slash, Block-Sparse, Quest). It does not include NSA or MoBA. The "Vertical-Slash for retrieval, Block-Sparse for reasoning" finding does not transfer one-to-one to NSA/MoBA's natively-trained sparsity.
14. **HRM critique 2601.10679 is a January 2026 paper; consistency with the spec's date is fine but its central finding is "guessing not reasoning" — a sharper statement than the hypothesis's borrowed claim "fragile fixed points."** The hypothesis softens the critique paper's finding. Ought to engage more directly.

---

## Recommendation to hypothesis-smith (revision guidance)

**The hypothesis is REJECT-able but salvageable.** Concrete revision items:

**(R-1) Fix the citation discipline.** Re-read NSA Tables 1/2 and MoBA Tables. Update §4 magnitudes. NSA is not 1-point-below-dense; it's parity-or-better. The K=1 gap between NSA and MoBA is the actual unknown; the hypothesis must commit to a calibrated K=1 prediction or acknowledge the K=1 gap is itself the threat to non-additivity.

**(R-2) Reformulate F2 as difference-in-differences.** Specify that the recursion-lift differential must exceed the K=1 gap. Concretely: *"((NSA-fallback K=6) − (NSA-fallback K=1)) − ((MoBA K=6) − (MoBA K=1)) ≥ +5 points"* AND *"|NSA-fallback K=1 − MoBA K=1| ≤ 2 points (calibration constraint via matched sparsity)."* Without the second clause, additive marginals satisfy F2 vacuously.

**(R-3) Fix the F3 cheap-path scope.** Either (i) drop F3 entirely if no NSA reference checkpoint can be obtained at sufficient scale (1B `zen-E/NSA-1B` may suffice for a *signal* but not for a kill-test); or (ii) reframe F3 explicitly as a *test-time-only suggestive probe* with clearly stated limitations, and remove the language "the full hypothesis is dead before any pre-training run begins." A negative test-time result on a model not trained for recursion does not kill a pre-training-with-recursion hypothesis (the hypothesis itself acknowledges this in §6a's caveat).

**(R-4) Replace F4 with a falsifiable mechanistic probe.** Per-iteration logit extraction is not viable. Replace with: per-iteration *attention-pattern* probe — measure the Jaccard overlap between iteration-1 and iteration-K selected blocks on NSA vs MoBA. The prediction "NSA shows *higher* inter-iteration drift than MoBA" is operationalizable without requiring logits at intermediate iterations.

**(R-5) Address the compute-match confound.** Specify the K=1 baseline configuration that has FLOP-matched compute to K=6. Either (i) a non-shared 6-layer-deeper variant, or (ii) a wider K=1 model with matched FLOPs. State explicitly which baseline is the K=1 anchor.

**(R-6) Engage with PLT counter-evidence.** Update §3 to either explain why PLT's recursion benefit on G-SWA does not transfer to the MoBA prediction, or weaken the MoBA prediction from "flat or negative" to "smaller than NSA's." The PLT data argues for the latter.

**(R-7) Address noise floor and feasibility at 350M.** Cite a 350M-class baseline on BABILong (or commit to running one). If the K=1 baseline is at floor, the +8 prediction is unmeasurable. Either move to 1B parameter scale (raising compute), accept R3 as the primary deliverable (descriptive, not predictive), or revise the prediction.

**(R-8) Cite or test the M3 load-bearing assumption.** Either (i) cite literature establishing retrieval-head-like behavior in compressed-summary branches of NSA-class models, or (ii) add a falsification criterion that *directly* tests whether the compressed branch carries fact-specific (vs position-prior) information. Without this, M3 floats.

**If the smith addresses (R-2), (R-4), (R-5), and at least one of (R-1, R-3, R-6, R-7), the hypothesis would survive a second-round critique. The current version has too many citation/operationalization gaps to be approved.**

---

## Verdict

REJECT (revision-1)
