# Red-Team Critique of H4 — Revision 1

Critiquing: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-4/revision-1/output.md`
Targeting gap: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/gap-finder-1/output.md` Gap 4
Revision round: 2

---

## 1. Verdict

`APPROVE`

The smith addressed all three Critical objections (C1, C2, C3) substantively, not cosmetically. SSA's Proposition 4.1, Theorem 1, and Appendix F (the load-bearing new citations) are verified verbatim from the paper. The c-A operationalization escapes the self-justification trap by being supervised on full-attention dropped mass (independent of NSA's own selection). MoR's expert-choice routing is now correctly cited and is the actually-published recipe. Falsification thresholds are above the noise floor with the committed seed budget. F-Calib is a real cheap pre-test that can kill the hypothesis before the from-scratch run.

There are residual Important and Suggestion-level concerns (architectural composition stability of double-discrete-routing, c-A feature blindness inheritance, Prism citation framing) but none are critical to whether the hypothesis is well-formed and falsifiable. They are eval-designer territory.

The orchestrator's reframe concern is real but mitigated: the revision narrows the hypothesis from "halting × sparse routing coupling" to "halting × sparse-routing coupling *via* miscalibration-correction signal." This is a strictly sharper, more falsifiable version of the same claim — it still occupies Gap 4's empty cell, just with a precise mechanism. The smith makes this transparent and pre-commits to falsification (F-Calib, F-StopGrad, F-Self-vs-A) that distinguishes the strong from the weak version.

---

## 2. Round-1 critical objections — status

### C1 (self-justification of c_{t,k}) — FIXED

Round-1 issue: c_{t,k} = sum-of-selected-p_t^slc / sum-of-all-p_t^slc approaches 1 by construction.

Revision-1 fix: c_{t,k} reformulated as a proxy for δ(t,k) = 1 − AttnSparsity(t,k) per SSA Theorem 1 / Definition 1, which references *full-attention* mass, not NSA's own estimate. Two operationalizations:

- **c-A**: trained regressor ĉ_φ(features) → δ̂(t,k), supervised by δ_true measured on 5% of training steps where full attention runs in parallel with NSA. Features include NSA gate weights, compression-branch entropy, top-n block index spread, RoPE-band positional features.
- **c-B**: JS-divergence between selection-branch and compression-branch attention distributions.

I verified SSA Proposition 4.1 verbatim from arXiv:2511.20102 §4.2: *"Training exclusively with sparse attention induces: 1. Learning Deficiency: For any token j in a dropped block, its gradient is zero, preventing the model from learning to attend to tokens outside the selected blocks."* I verified Theorem 1 verbatim: *"‖h_full(t) − h_sparse(t)‖ ≤ δ(t) (max ‖v(j)‖ + ‖h_sparse(t)‖)"*. I verified Appendix F: *"replacing all three with full attention (fa+fa+fa) yields extremely high perplexity (191.3)"* and the conclusion *"only the selection module extrapolates well to full attention, and the compression and sliding window modules both fail."*

The miscalibration claim is now load-bearing on cited theorems and empirical results, not on hand-waving. The escape from self-justification is real: ĉ_φ is supervised by a *target* (δ_true from full attention) that is independent of NSA's selection branch. The F-Self-vs-A test (R2-A vs R2-self by ≥1.0 EM) is the cleanest test of whether c-A actually adds information beyond NSA's self-coverage form. **Status: Critical → Resolved.**

Residual concern (Important, not Critical): the c-A regressor's *features* (NSA gate weights, top-n block index spread) inherit NSA's blindness about non-selected blocks. Compression-branch entropy and RoPE-spectral features partially escape this, but the smith should explicitly characterize which features are blindness-immune. F-StopGrad partially detects feature-vs-interaction confusion. Note for eval-designer.

### C2 (falsification thresholds at noise floor) — FIXED

Round-1 issue: F1 threshold 0.5 EM at 5 seeds with MuSiQue@32K dev set ~2.4k items; per-cell std ≥0.5 EM in published lit; F1 structurally indistinguishable from zero.

Revision-1 fix:
- ≥10 seeds per cell, paired bootstrap n=10000, α=0.05 two-sided.
- Per-cell σ budget ≤0.6 EM, pre-measured on R0 with 10 seeds; if exceeded, escalate to ≥15 seeds.
- F1 threshold raised from 0.5 to 1.5 EM. F2 from 1.0 to 1.5. F3 from 1.0 to 1.5. F4 from 1.0 to 1.5.
- Primary benchmark switched from MuSiQue@32K (~2.4k items, contrived padding) to RULER multi-hop variable-tracking (more items, native long-context).
- MuSiQue demoted to secondary cell.
- F2 calibrated to 60–90% baseline accuracy band, not at saturation ceiling.

Power check: with σ_per_cell ≤ 0.6 EM, paired-difference SE = 0.6/√10 ≈ 0.19. At α=0.05 two-sided, detectable effect ≈ 2 × 0.19 = 0.38 EM. So a true effect of 1.5 EM is detectable at well above 80% power. **Status: Critical → Resolved.**

### C3 (TRM/PonderNet recipe conflation) — FIXED

Round-1 issue: revision-0 labeled "TRM-style + PonderNet halting" — a chimera. TRM uses ACT Q-learning, not λ_n.

Revision-1 fix: committed to **MoR's expert-choice routing** (arXiv:2507.10524 §2.2.1) as the halting recipe, with explicit citation of the sigmoid gate equation `g_t^r = G(θ_r^T H_t^r)`. All "TRM-style" framing removed. TRM listed in §9 as "Acknowledged but NOT used for halting recipe (C3 fix)."

I verified MoR §2.2.1 directly: Eq. 2.1 specifies `g_t^r = G(θ_r^T H_t^r)` with G being a sigmoid or tanh activation, top-k selection via β-percentile threshold P_β, hierarchical filtering (only tokens selected at step r can be re-evaluated at r+1). This is the actual published mechanism. The revision faithfully cites it. **Status: Critical → Resolved.**

---

## 3. Reframe assessment (orchestrator's flag)

The orchestrator asked whether the revision morphs from "halting × sparse routing coupling is ill-defined" to "can a learned δ-proxy regressor improve halting?"

**My finding: the revision narrows the same gap, it does not move to a different one.**

Original Gap 4: per-token depth halting × per-token sparse routing, jointly trained, evaluated on multi-hop. **No published paper occupies this cell** (verified independently below).

Revision-1 H4: **same cell**, with a precise mechanism — the coupling matters specifically because NSA's selection branch exhibits Gradient Update Deficiency on non-selected blocks, so MoR's gate computed on h_k inherits selection-branch blindness on multi-hop tokens whose evidence chain crosses dropped blocks. The δ-proxy regressor is the *signal* the coupling needs, not the gap itself.

This is a sharpening, not a pivot. The smith makes the strong/weak version distinction transparent (M4 vs M-Steelman) and pre-commits F-StopGrad to detect "feature-not-interaction" outcomes. F-Self-vs-A is the test of whether the C1-fix actually buys anything over revision-0's framing.

If F-Calib fails: hypothesis dies pre-experiment, novel measurement of NSA's miscalibration on RULER-multi-hop is the contribution.
If F-StopGrad fires: M4-strong is wrong, M4-weak (feature, not interaction) holds, hypothesis downgrades but still contributes.
If F-Self-vs-A fires: c-A doesn't beat c-self, the C1 fix was empty, the miscalibration claim is null.

These are three distinct, operationalizable falsifiers that bracket the claim from three sides. This is *good* hypothesis structure.

---

## 4. Gap re-verification

I ran three independent literature queries beyond the gap-finder's two and round-1's three.

**Query 4**: `MoR mixture of recursions sparse attention NSA combination`
**Result count**: 10 papers. Returns include MoR itself (2507.10524), MoSA (2505.00315 — content-based learnable sparse attention via expert-choice routing, but no recursion or halting), MoA (2406.14909 — sparse attention configurations per head, no recursion), and unrelated MoE papers. **None combine MoR's expert-choice recursive routing with NSA-style block selection.**

**Query 5**: `dropped attention mass conditioning halting depth gate full attention proxy supervision`
**Result count**: 10 papers. None. Closest hits are gated attention (2505.06708 — head-specific sigmoid gate after softmax, no halting), attention dropout (2310.18738 — regularization, not halting). **No paper uses dropped attention mass as a feature for depth-routing.**

**Query 6**: `recursive transformer adaptive computation sparse attention long context reasoning halting`
**Result count**: 8 papers. Returns ReSSFormer (2510.01585, fixed-K), Sparse Frontier (2504.17768, sparse attention scaling, no recursion), Adaptive Loops (2603.08391, dense attention + halting). **No paper does adaptive depth halting + sparse attention jointly.**

**Verdict**: Gap 4 survives all six queries (3 from gap-finder, 3 from round-1, 3 from round-2). **gap_claim_survives: true.**

---

## 5. Citation spot-checks (revision-1 added/load-bearing)

### (a) SSA arXiv:2511.20102 §4.2 — Proposition 4.1 (Gradient Update Deficiency).
**Verified verbatim.** Quote: *"Training exclusively with sparse attention induces: 1. Learning Deficiency: For any token j in a dropped block, its gradient is zero, preventing the model from learning to attend to tokens outside the selected blocks. 2. Attention Suppression Deficiency: By excluding dropped tokens from the softmax denominator, SA models bypass the competitive pressure that forces FA models to learn globally contrastive attention distributions."* The smith's claim that "tokens in dropped blocks receive zero gradient, so the importance estimator never learns counterfactual relevance" is **a direct paraphrase of Proposition 4.1 part 1**. Severity: OK.

### (b) SSA arXiv:2511.20102 §4.2 — Theorem 1 (Sparse Attention Error Bound).
**Verified verbatim.** Quote: *"‖h_full(t) − h_sparse(t)‖ ≤ δ(t) (max_{j∈S^c(t)} ‖v(j)‖ + ‖h_sparse(t)‖)... This bound scales linearly with δ(t), motivating the learning of inherently sparser attention distributions to minimize approximation error under inference."* The smith's claim "approximation error scales linearly with dropped full-attention mass" is exactly correct. Severity: OK.

### (c) SSA arXiv:2511.20102 Appendix F — Compression branch fails to extrapolate.
**Verified verbatim.** Quote: *"replacing all three with full attention (fa+fa+fa) yields extremely high perplexity (191.3)... only the selection module extrapolates well to full attention, and the compression and sliding window modules both fail."* The smith's claim "compression branch fails to extrapolate to full attention (perplexity 191.3 when replacing all three branches with full attention; isolating the failure to compression and sliding-window)" is a **precise paraphrase** of Appendix F. This is the load-bearing rebuttal to the steelman ("compression carries non-selected-block info"). Severity: OK.

### (d) MoR arXiv:2507.10524 §2.2.1 — Expert-choice routing.
**Verified.** §2.2.1 specifies expert-choice routing with `g_t^r = G(θ_r^T H_t^r)` (G being sigmoid or tanh), top-k selection via β-percentile threshold, and hierarchical filtering. The smith's recipe matches the paper exactly. The smith should specify which KV caching strategy (recursion-wise vs recursive sharing) — MoR experiments show recursion-wise outperforms recursive sharing slightly but this matters for NSA composition. Severity: Suggestion (eval-designer should specify).

### (e) Prism arXiv:2602.08426.
**Citation framing is stretched but defensible.** Prism is *about* improving block-sparse attention's importance estimator via spectral-aware coarse-grained attention; it identifies destructive interference between RoPE and mean-pooling as the failure mode. The smith uses Prism for "RoPE-spectral signal" as a feature in the c-A regressor, which is a downstream application of Prism's core finding. Not technically wrong, but the smith should note that Prism is being used as a *feature engineering source*, not as evidence for the dropped-mass claim. Severity: Suggestion.

### (f) ReSSFormer arXiv:2510.01585 §3.
**Verified.** R2MU iterates K times (fixed, not adaptive). ASAM uses sparsemax + top-k routing + MoE expert sparsity. No per-token depth halting. The smith's claim that ReSSFormer "instantiates roughly our R0 baseline (no halting, fixed K, sparse attention)" is **substantively correct**. Severity: OK.

### (g) Adaptive Loops arXiv:2603.08391.
**Verified via paper_details.** Adaptive per-layer looping + halting + dense attention + gated memory banks. The smith correctly positions this as the "dense-attention version of our recipe" and includes it as mandatory baseline (a). Severity: OK.

---

## 6. Mechanism critique (revision-1)

### M1 (three-branch architecture with subset-conditional variance) — Better.
Revision-1 acknowledges NSA's three branches, but argues the variance argument applies *subset-conditionally* on tokens whose evidence spans non-selected, non-windowed blocks. This is grounded in SSA Appendix F: compression branch fails to extrapolate, so it carries summary but not discriminative information. The "variance dominates *for this subset*" framing is more honest than revision-0's blanket claim. Severity: OK.

### M3 (c_{t,k} as δ(t,k) proxy) — Resolved (see C1 above).

### M4 (R2 beats R1 iff NSA's selection is miscalibrated) — Strong commitment.
This is the load-bearing claim. It is now contingent on F-Calib passing. M4 commits to the disjunction: *either* NSA is miscalibrated on multi-hop and R2-A beats R1, *or* NSA is well-calibrated and R2-A ≈ R1 (steelman wins). Both branches produce a contribution. Severity: OK.

### M-Steelman — Engaged directly.
The smith engages SSA Appendix F empirically as the rebuttal to "compression carries enough info." This is the right move: the steelman's "redundant coverage" claim is empirically falsified for full-attention extrapolation in NSA's published architecture. The hypothesis still requires the more specific claim that this empirical failure transfers from "extrapolation perplexity" to "multi-hop QA performance," which is asserted but not directly validated by SSA's experiments. Severity: Important. Eval-designer should note that "compression branch fails to preserve discriminative signal for multi-hop" is an *implicit* extrapolation from SSA's perplexity finding, not a direct measurement.

### M5 (chain-length dependence) — Carried over, supported.
Severity: OK.

---

## 7. Falsifiability assessment (revision-1)

| Criterion | Operationalizable? | Threshold reasonable? | Power adequate? |
|---|---|---|---|
| F-Calib (median rank > top-n on ≥30%) | Yes — finite measurement on RULER dev | Yes (gating pre-test) | <24 GPU-hours, no power issue |
| F1 (R2-A > R1 by ≥1.5 EM) | Yes | Yes (>2σ_se with 10 seeds) | OK |
| F2 (single-hop NIAH 60–90% band, R2-A − R0 ≥1.5) | Yes | Yes (away from saturation per I6) | OK |
| F3 (\|w_c\|/std < 1.5 across 10 seeds) | Yes | Yes | OK |
| F4 (degenerate top-n ≥80%, R2-A − R1 ≥1.5) | Yes | Yes | OK |
| F-Self vs A (R2-A > R2-self by ≥1.0 EM) | Yes | Yes — the cleanest C1 test | OK |
| F-StopGrad (\|R2-A − R2-A_stopgrad\| < 0.7) | Yes | Yes — partial-falsification pre-committed | OK |

All six post-Calib criteria are genuinely operationalizable into a finite experiment. F-Calib alone can falsify the entire claim before from-scratch training. F-StopGrad distinguishes "interaction" from "useful feature." F-Self-vs-A directly tests the C1 fix. Severity: OK across the board.

The smith correctly removed F5 R3-ordering as a falsification (per round-1 I3) and demoted it to exploratory contrast. The revision-0 smoke test ("inference-time correlation < 0.05") is correctly removed and replaced by F-Calib.

---

## 8. Strongest counter-argument (revision-1 steelman)

The remaining strongest counter-argument: **NSA's training, even without explicit miscalibration correction, induces importance estimates that correlate sufficiently with downstream multi-hop relevance in practice.** This is what F-Calib tests. If F-Calib fails (median gold-evidence-block rank ≤ top-n on multi-hop), the entire hypothesis dies pre-experiment.

A new, sharper steelman against the revised hypothesis: **the c-A regressor's features inherit NSA's selection-branch blindness.** If NSA's selection misses gold-evidence blocks, NSA's gate weights and top-n index spread will not "know" they were missed (the blindness is upstream of the features). Compression-branch entropy and RoPE-band features may partially compensate, but if the regressor trained on δ_true relies most heavily on the blind features, c-A's predictive power on miscalibrated tokens will be weak.

Mitigation: the smith's F-Self-vs-A test is *exactly* the test of whether c-A adds information that c-self lacks. If the blindness inheritance is severe, c-A and c-self will perform similarly, and F-Self-vs-A will fire. This is a self-correcting design.

---

## 9. Severity-tagged objections

### Critical (must fix in revision)
**None.** All three round-1 Critical objections (C1, C2, C3) are resolved. No new Critical issues.

### Important (should fix, but does not block APPROVE)

**I1. c-A regressor feature blindness inheritance.** Some features (NSA gate weights, top-n block index spread) are downstream of NSA's selection branch and inherit its blindness about non-selected blocks. The smith should explicitly note in §3 (M3) which features are blindness-immune (compression-branch entropy, RoPE-band) vs blindness-inheriting (NSA gate weights, top-n spread). F-Self-vs-A tests the consequence, but eval-designer should plan ablation (i) to include "blindness-immune features only" vs "all features." Note for eval-designer.

**I2. Double-discrete-routing training stability.** MoR's expert-choice gate has β-percentile thresholding (discrete top-k via auxiliary loss); NSA's selection branch has top-n block selection (also discrete, with importance score). The composition has *two* simultaneous discrete decisions per token per layer. Training stability is not characterized — neither MoR nor NSA papers report instability with their respective single-discrete gating, but the composition may not. Eval-designer should monitor gradient norms and gate-collapse during early training, and consider a Gumbel-softmax warmup for one or both routers if collapse occurs.

**I3. KV caching strategy unspecified.** MoR §2.2.1 evaluates two strategies (recursion-wise caching vs recursive KV sharing). NSA's three-branch attention has its own KV layout per branch. The hypothesis does not specify which combination is used. This affects FLOPs-matching, cache size, and whether NSA's compression branch is recomputed at each recursion step or shared. Eval-designer must specify.

**I4. Compression-branch-fails-on-multi-hop is an implicit extrapolation.** SSA Appendix F shows the compression branch fails to extrapolate to full attention in *perplexity*. The hypothesis extends this to "compression branch fails to preserve discriminative signal for multi-hop QA." This is plausible but not directly measured by SSA. F-Calib partially tests this (if gold-evidence block rank > top-n, the compression branch is not compensating). Eval-designer should include a "compression-branch-only" attention probe as a diagnostic.

**I5. F-Calib threshold (≥30% of multi-hop questions) is a guess.** The smith picks 30% as the threshold for "miscalibration claim is non-null." There is no prior art establishing this number. The smith should pre-register the threshold based on a small pilot (e.g., measure miscalibration on HotpotQA-distractor first) before running on RULER-multi-hop. Or commit to "any positive rate of miscalibration > random-baseline-rate."

### Suggestion (nice to have)

**S1. Prism citation framing.** Prism (arXiv:2602.08426) is about improving block-sparse attention's *own* importance estimator via spectral-aware coarse attention; the smith uses it as a feature source for the c-A regressor. The connection is real (RoPE spectral structure matters for block selection) but the citation should be reframed as "feature engineering inspiration" rather than as evidence for the dropped-mass claim itself.

**S2. Expected magnitude of R0 baseline.** The smith does not state expected RULER-multi-hop variable-tracking EM at 1B-scale NSA. SSA Table 3 shows NSA at 32k with continual training scores ~93–99% NIAH and ~19% LongBench. RULER multi-hop variable-tracking is harder than NIAH; magnitude expectations should be calibrated. Eval-designer.

**S3. R2-A magnitude prediction (R0 + 2.0 to +4.0 EM) is at the upper end of MoR's reported 1–3 point gains and SSA's 1.5–4 point swings.** The combination assumption (1–2 from recursion + 1–2 from miscalibration-correction) is reasonable but not strictly additive at the architectural level. Eval-designer should treat the lower end (+2.0) as the threshold; the upper end (+4.0) is aspirational.

**S4. MuSiQue@32K demoted but still listed.** Now demoted to secondary; this is correct. Could be removed entirely, but keeping it as a robustness check is defensible.

---

## 10. Recommendation to hypothesis-smith

The hypothesis is approveable and ready for eval-designer. **No further revision is required.**

Forward to eval-designer with these notes (carry-overs to Phase 5):

1. **Specify KV caching strategy** for the MoR-NSA composition (I3). Default recommendation: recursion-wise caching, recompute NSA branches at each recursion step. This is more expensive but cleanest for the mechanism.
2. **Plan ablation that isolates blindness-immune features** in the c-A regressor (I1). Add "compression-entropy + RoPE-band only" as a c-A variant.
3. **Monitor training stability** for double-discrete-routing collapse (I2). Pre-commit to a Gumbel-softmax warmup as a fallback.
4. **Add a compression-branch-only attention probe** as a diagnostic for I4.
5. **Pre-register F-Calib threshold based on a HotpotQA-distractor pilot** (I5). The 30% number is a guess; it should be calibrated against a known-easy multi-hop benchmark first.
6. **Treat R2-A predicted magnitude as a band** (S3): if R2-A − R0 < 2.0 EM, F1 fires; the +4.0 EM upper end is aspirational, not a threshold.
7. **Reframe Prism citation** (S1) as feature-engineering inspiration, not evidence.

The smith made a substantive, technically-rigorous revision. The miscalibration claim is now grounded in a real, peer-reviewed published theorem (SSA Proposition 4.1 / Theorem 1) with corroborating empirical evidence (Appendix F). The c-A operationalization escapes the self-justification trap. The recipe conflation is fixed. The thresholds are above the noise floor with the committed seed budget. The falsification structure brackets the claim from three sides (F-Calib, F-StopGrad, F-Self-vs-A).

I would defend this hypothesis publicly. **APPROVE.**

---

`APPROVE`
