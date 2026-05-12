# Hypothesis H4 (revision 1) — On an NSA backbone with MoR-style per-token recursion, conditioning the routing-depth gate on a *full-attention dropped-mass* proxy (not NSA's self-reported importance) produces a non-additive multi-hop reasoning gain over independent training, because NSA's selection branch suffers a documented gradient update deficiency that makes its self-reported coverage structurally optimistic

## Changes from revision-0

This revision rewrites the hypothesis around three structural fixes demanded by red-team-4:

1. **C1 fix — coverage signal redefined.** Revision-0 computed c_{t,k} from NSA's own importance scores p_t^slc, which is self-justifying (the ratio "selected mass / total estimated mass" approaches 1 by construction). Revision-1 commits to BOTH halves of red-team's option (a)+(b): an explicit, citable claim that NSA's importance estimate is *structurally miscalibrated on multi-hop QA* (grounded in SSA, arXiv:2511.20102, Proposition 4.1 "Gradient Update Deficiency" and Theorem 1's dropped-attention-mass bound), AND a reframed c_{t,k} that is now a **proxy for δ(t)** = dropped full-attention mass — measured, not self-reported. Section 3 (Mechanism) is rewritten end-to-end. Sections 4–7 are updated for consequences.
2. **C2 fix — falsification thresholds recalibrated and primary benchmark changed.** Revision-0 used MuSiQue@32K with thresholds 0.5–1.0 EM at 5 seeds — at the noise floor. Revision-1 commits to (i) **≥10 seeds per cell**, (ii) **primary benchmark = RULER-multi-hop variable-tracking subset** (more items than MuSiQue@32K, native long-context), (iii) thresholds raised to ≥1.5 EM with explicit per-cell std budgets and a paired bootstrap test (n=10 seeds, α=0.05). MuSiQue is demoted to a secondary cell.
3. **C3 fix — recipe conflation eliminated.** Revision-0 labeled a chimera as "TRM-style recursive + PonderNet halting." TRM (arXiv:2510.04871 §3.2) actually uses ACT Q-learning, and PonderNet (arXiv:2107.05407) is per-sequence. Revision-1 commits to **MoR's expert-choice routing** (arXiv:2507.10524 §2.2.1) — already-published, sparse-routing-friendly, per-token, with a sigmoid gate `g_t^r = σ(θ_r^T H_t^r)` operating on hidden state. This is what red-team recommended as option (c).

Additional changes:
- **Adjacent prior art cited**: ReSSFormer (arXiv:2510.01585) — recursive sparse + multi-hop QA at fixed K, no per-token depth halting. Adaptive Loops (arXiv:2603.08391) — adaptive looping + halting on dense, no sparse routing. Section 1 explains why neither subsumes Gap 4.
- **Steelman engaged directly** in §3 (M-Steelman) and a new gradient-stopped-c_{t,k} ablation (replacing the I1 "smoke test").
- **M6 dropped** (Tunnel Vision was a strained analogy per red-team Suggestion S1).
- **M1 restated** to acknowledge NSA's three-branch architecture (compression + selection + sliding-window) per I2, and the variance argument re-derived under the gated combination.
- **R3 falsification (F5) downgraded** to exploratory contrast per I3 — MoEUT/SUT analogies showed parity, not underperformance.
- **F2 (single-hop NIAH)** moved off the saturation ceiling per I6: use NIAH at a length where baseline lies in 60–90%, not 95–99%.

---

## 1. Targeted gap

Source: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/gap-finder-1/output.md` Gap 4 (lines 55–66).

Restated: Per-token *depth* routing (ACT 1603.08983, PonderNet 2107.05407, Universal Transformer 1807.03819, MoR 2507.10524, LoopFormer 2602.11451) and per-token *attention* sparse routing (NSA 2502.11089, MoBA 2502.13189, DSA / DeepSeek-V3.2 2512.02556) both produce per-token decisions on the *same* hidden state. The structural worry the gap-finder isolates: when NSA-style top-n block selection drops a block on iteration k, the halting/routing logit for tokens that needed cross-attention to that block is computed on a corrupted representation, and the two routers are jointly underspecified.

**Adjacent prior art (per red-team round 1):**
- **ReSSFormer (arXiv:2510.01585)** combines recursive R2MU (fixed K) + ASAM (sparse top-k routing + sparsemax) and evaluates on multi-hop QA. R2MU's K is uniform across tokens (no per-token depth halting). It thus instantiates roughly our **R0 baseline (no halting, fixed K, sparse attention)**. Gap 4 — joint training of *per-token* depth halting with *per-token* sparse routing — is unoccupied: ReSSFormer does not have a per-token halting head. Our R1/R2/R3 cells therefore extend ReSSFormer in a direction it explicitly does not address.
- **Adaptive Loops in Transformers (arXiv:2603.08391)** does adaptive per-layer looping with learned halting on **dense** attention. This instantiates the dense-attention version of our recipe — useful as a baseline (mandatory baseline (d) in §6) and a reference for "is sparsity needed at all?" but does not engage the sparsity-corrupted-halting question.

So the cell {per-token sparse routing × per-token depth halting × jointly trained × evaluated on multi-hop} remains unoccupied. The hypothesis-smith team verified independently across three queries.

---

## 2. Hypothesis statement

If we train an NSA backbone (arXiv:2502.11089) with per-token recursive depth via MoR's expert-choice routing (arXiv:2507.10524 §2.2.1, sigmoid gate on hidden state, K_max=4 weight-tied recursion blocks), and we compare four coupling regimes —

- **R0**: no depth routing, fixed K=K_max, NSA backbone (≈ ReSSFormer-with-NSA);
- **R1**: independent — MoR's expert-choice routing trained jointly with NSA, no information shared between the two routers;
- **R2**: **dropped-mass-conditioned** — the per-recursion-step gate score is augmented with a scalar c_{t,k} ∈ [0,1] that is a *trained proxy for the dropped full-attention mass δ(t,k)* at recursion step k for token t (defined in §3, **not** computed from NSA's own p_t^slc);
- **R3**: shared-router — a single linear head emits both depth-gate and block-selection logits;

— then on a multi-hop long-context reasoning benchmark (primary: RULER multi-hop variable-tracking at 32K; secondary: FRAMES, MuSiQue@32K) at matched-FLOPs budget, **R2 outperforms R1 by ≥1.5 EM** (paired bootstrap, n=10 seeds, α=0.05), **R0 by ≥2.0 EM**, and on single-hop NIAH calibrated to the 60–90% baseline band, R2 is within ±1.0 EM of R0. The R2-vs-R0 gap must hold *only* when the fraction of multi-hop questions whose evidence chain crosses ≥3 NSA blocks exceeds ≈40%; below this threshold and on single-hop, R2 ≈ R0. **The improvement is non-additive: R2's gain is not "halting helps + sparsity helps" but specifically "depth decisions made aware of how much full-attention mass NSA dropped this step."**

The hypothesis commits to a **strong, falsifiable miscalibration claim**: NSA's selection-branch importance estimate p_t^slc is, by Proposition 4.1 of SSA (2511.20102), subject to a **Gradient Update Deficiency** — tokens in dropped blocks receive zero gradient, so the estimator never learns to identify when full-attention would have attended to them. Therefore the ratio (selected mass / NSA-estimated total) is structurally optimistic on multi-hop QA, where gold evidence chains often span blocks NSA's importance head has not learned to surface. We commit to a **calibration pre-test (S3 from red-team round 1, now mandatory in §6)** that measures this miscalibration directly before the main experiment.

---

## 3. Mechanism

### M1 — NSA's three-branch architecture and the variance argument (revised)

NSA combines three branches via a learned gate (NSA §3.2):
- **Compression**: each preceding block summarized into a single token via a learned MLP, full coverage but lossy.
- **Selection**: top-n blocks chosen by p_t^slc, full token resolution on selected, **zero coverage on non-selected** (these tokens receive no gradient through this branch — SSA Proposition 4.1).
- **Sliding window**: most recent w tokens, full resolution on local context.

The gated combination means the post-attention representation h_k at recursion step k is **not purely a function of selected blocks** (red-team I2 acknowledged). However, the three branches are *not* equally informative for cross-block multi-hop retrieval:
- Compression is lossy by construction, and SSA Appendix F documents that NSA's compression branch *fails to extrapolate to full attention* (perplexity 191.3 when replacing all three branches with full attention; isolating the failure to compression and sliding-window). This means compression carries *summary* information from non-selected blocks but does not preserve the *fine-grained positional/entity signals* that multi-hop retrieval requires.
- Sliding window only covers the most recent w tokens, so it cannot supply non-selected-block information at long range.

Therefore, when a multi-hop question's evidence chain spans a block outside the top-n and outside the sliding window, **the only path for that evidence into h_k is the lossy compression branch**, which does not preserve the discriminative signal. The variance attributable to selection-branch dropout dominates *for this subset of tokens*, even if it is a small component of total halting-head variance averaged across all tokens. M2's variance argument therefore predicts a **subset-conditional** effect (M5).

### M2 — MoR's gate inherits sparsity-induced subset-variance

MoR's expert-choice gate (2507.10524 Eq. 2.1) computes `g_t^r = σ(θ_r^T H_t^r)` from the hidden state at recursion step r. By M1, H_t^r is a function of which blocks NSA selected at step r. For the subset of tokens whose true depth-gate decision depends on information in non-selected, non-windowed blocks, g_t^r is computed on a representation that is provably missing that signal (compression branch is information-lossy per SSA Appendix F). The gate score therefore inherits a *subset-conditional bias*, not just variance, in the direction of premature halting (because the compression branch's bias is toward summary/global features that look "easy").

### M3 — A *non-self-justifying* coverage signal: c_{t,k} as a δ(t,k) proxy (revision-1 core fix)

Revision-0's c_{t,k} = (sum over selected blocks of p_t^slc) / (sum over all blocks of p_t^slc) is **self-justifying**: by definition, selecting top-n blocks ranked by p_t^slc maximizes the numerator. Red-team C1 was correct.

The fix is to compute c_{t,k} as a **proxy for δ(t,k) = 1 − AttnSparsity(t,k)**, where AttnSparsity is defined with respect to *full attention*, not NSA's own estimate (SSA Theorem 1, Definition 1). Two operationalizations are mandatory ablations:

**(c-A) Direct measurement during training (SSA-style mixed regime).** Following SSA (2511.20102 §5) — but with a much smaller mixing ratio than SSA proposes — on a fraction p_full ≈ 5% of training steps, run full attention in parallel with NSA on the same batch. Compute the *true* dropped attention mass δ_true(t,k) = 1 − Σ_{j ∈ S(t)} a_full(t,j) for each query token. On the remaining 95% of training steps, train a tiny **calibration regressor** ĉ_φ(features) → δ̂(t,k) where features include: NSA gate weights, compression-branch attention entropy, top-n block index spread, and a learned RoPE-band positional feature (Prism's spectral signal, arXiv:2602.08426). At inference, c_{t,k} = ĉ_φ(features), no full attention required. This is **structurally non-tautological**: ĉ_φ is supervised by full-attention dropped mass, which is independent of NSA's own selection. The 5% full-attention mixing cost is bounded and tunable.

**(c-B) Indirect measurement via gate-disagreement.** Compute c_{t,k} = JS-divergence between NSA's selection-branch attention distribution p_t^slc and a *secondary* attention distribution computed on the compressed branch only. Intuition: when these disagree sharply, the selection branch is committing to blocks the compression-branch summary thinks are uninformative — a soft signal of mis-selection. This avoids any full-attention computation but is a weaker proxy.

**(c-C) Form ablations** (carry-over from revision-0): for the chosen operationalization, ablate the functional form (raw scalar vs log vs sigmoid-warped). Pre-committed in §6.

The hypothesis predicts **c-A > c-B > raw-NSA-self-coverage** in EM gain; if c-A does not beat raw-NSA-self-coverage, the miscalibration claim is null and F-Calib (§5) fires.

### M4 — Why R2 should beat R1 *if and only if* NSA's selection is miscalibrated on multi-hop

This is the critical commitment. The mechanism is:

> "MoR's depth gate, conditioned on h_k alone, cannot distinguish 'I have all the evidence I need' from 'I am missing evidence in a block NSA dropped because its importance estimator does not know that block is relevant for this multi-hop question.' An external signal of dropped-full-attention mass disambiguates these."

This claim **requires** NSA's importance estimator to be miscalibrated — i.e., p_t^slc must under-rank some blocks that full attention would have weighted heavily. SSA Proposition 4.1 provides the mechanism for this miscalibration: tokens in dropped blocks receive zero gradient through the selection branch during training, so the estimator never learns counterfactual relevance. SSA Theorem 1 quantifies the consequence: ‖h_full − h_sparse‖ ≤ δ(t)·(...) — the approximation error scales *linearly* with dropped full-attention mass, and SSA's experiments (§7.4) show the bound is non-vacuous on long-context reasoning.

**If NSA is well-calibrated on multi-hop** (the steelman, see M-Steelman below), then δ(t,k) is small everywhere on the support of multi-hop tokens, c_{t,k} is near-constant, and R2 ≈ R1. **Then F-Calib falsifies the hypothesis with a cheap pre-test.**

### M5 — Effect should depend on multi-hop chain length

(Carried over from revision-0.) Retrieval Head (2404.15574) shows long-context factuality is mediated by sparse attention heads needing precise positional access; SCBench (2412.10319) shows sparse-attention KV compression is more robust on simple retrieval than multi-hop. Single-hop tasks rarely need cross-block retrieval; multi-hop tasks chain ≥2–3 retrievals across blocks. Coverage-conditioning therefore matters only when the evidence chain spans multiple blocks. This restricts the effect to multi-hop benchmarks with ≥3-block evidence chains (RULER multi-hop, FRAMES) and predicts no effect on single-hop NIAH.

### M-Steelman — direct engagement (new in revision-1)

The red-team's strongest counter-argument:

> *"NSA's three-branch architecture already provides redundant coverage. The variance is small; the residual stream encodes 'I am missing information' implicitly. R2 ≈ R1 ± noise."*

We engage this directly. SSA's Appendix F is the load-bearing rebuttal: empirical evidence that NSA's compression branch (the alleged source of redundant coverage for non-selected blocks) **cannot extrapolate to full attention** (perplexity 191.3 when isolated). The compression branch carries *summary* information, not the discriminative *positional/entity* information multi-hop QA needs. Therefore the redundancy the steelman claims is real for *mean* loss but not for the *conditional* loss on multi-hop tokens with ≥3-block evidence chains.

If the steelman is right anyway (R2 ≈ R1), F1 (§5) fires. The hypothesis is then falsified, *but* the investigation produces a positive result: SSA's dropped-mass theory predicts a regime where R2 ≈ R1, and confirmation of that regime informs sparse-attention training. Contribution-under-failure noted in Risk A (§7).

**A new dedicated falsification (F-StopGrad, replacing revision-0's smoke test):** train R2 with c_{t,k} *gradient-stopped* on the depth-gate (so c_{t,k} is a frozen feature, not a learning signal). If R2_stopgrad ≈ R2_unfrozen, then c_{t,k} is just a redundant feature the rest of the network already encodes — the steelman wins on a technicality, and the "interaction" interpretation is wrong. We pre-commit to declaring this outcome a partial win (the feature still helps) but a falsification of the strong interaction claim.

---

## 4. Predicted outcome with magnitude

Setup: A ~1B-parameter recursive transformer with MoR's expert-choice routing (2507.10524), K_max=4 weight-tied recursion blocks, NSA backbone (2502.11089) with the published block size and top-n schedule. Trained on a long-context mixture (~30B tokens, 32K context). Each regime trained from the same initialization at FLOPs-matched budget.

**Statistical contract: ≥10 seeds per cell.** Per-cell std budget: ≤0.6 EM at this scale (will pre-measure on R0 with 10 seeds before announcing thresholds final; if budget exceeded, escalate to ≥15 seeds). Test: paired bootstrap on the seed-pair vector, n_resamples=10000, α=0.05 two-sided.

| Regime | Description | RULER-multi-hop EM (primary) | FRAMES EM | MuSiQue@32K EM |
|---|---|---|---|---|
| **R0** | No halting, K=4 fixed (≈ ReSSFormer-with-NSA) | baseline | baseline | baseline |
| **R1** | Independent: MoR routing + NSA, no signal | R0 ±0.5 EM | R0 ±0.5 EM | R0 ±0.5 EM |
| **R2-A** | c_{t,k} = ĉ_φ(features), supervised by δ_true | **R0 + 2.0 to +4.0 EM** | **R0 + 1.5 to +3.5 EM** | **R0 + 1.0 to +3.0 EM** |
| **R2-B** | c_{t,k} = JSD(p_slc, p_cmp) | R0 + 1.0 to +2.5 EM | R0 + 0.5 to +2.0 EM | R0 + 0.5 to +2.0 EM |
| **R2-self** | c_{t,k} = NSA's own ratio (revision-0 form) | R0 + 0.0 to +1.0 EM (predicted null) | R0 + 0.0 to +1.0 EM | R0 + 0.0 to +1.0 EM |
| **R3** | Shared router (exploratory, no falsification) | report only | report only | report only |

Magnitude reasoning:
- SSA (2511.20102 Tab.4–6) reports 1.5–4 point swings between sparse-attention training schemes on long-context benchmarks at ~1B scale.
- MoR (2507.10524 §3.2) reports 1–3 point gains from token-level adaptive recursion at similar scale.
- Combining a 1–2 point recursion gain (Risk: may be smaller for NSA than dense) with a 1–2 point miscalibration-correction gain (interaction) yields the R2-A range.
- We deliberately predict R2-A > R2-self by ≥1.0 EM as the cleanest test of the C1 fix: if NSA's self-reported coverage worked, R2-self would match R2-A.

Conditions under which R2 should NOT hold:
- Single-hop NIAH at any length (R2 within ±1.0 EM of R0): coverage signal carries no useful information when one block suffices.
- Short context (<4K tokens): NSA selects almost all blocks, δ ≈ 0.
- Very dense top-n (>80% blocks selected): δ ≈ 0; M3 collapses.
- K_max=1: no depth decision to condition.
- If NSA is well-calibrated on the target multi-hop benchmark (F-Calib pre-test).

---

## 5. Falsification criteria

≥10 seeds per cell, paired bootstrap n_resamples=10000, α=0.05 two-sided.

**F-Calib (NEW, pre-experiment, mandatory).** *On a fixed pre-trained NSA model (or NSA-mini we train from scratch as part of R0), measure the miscalibration of p_t^slc on RULER-multi-hop dev: for each gold answer-position token, compute the rank of the gold-evidence block under p_t^slc (averaged over depth/heads). If the median rank is ≤ NSA's top-n value (i.e., NSA already picks the gold block), then **the miscalibration claim is null** and the hypothesis is dead before the joint-training run. Predicted: median gold-evidence-block rank > top-n on ≥30% of multi-hop questions.* This is a cheap pre-test (<24 GPU-hours) — the mandatory gate before the from-scratch experiment.

**F1 (non-additivity of R2-A).** R2-A EM ≤ R1 EM + 1.5 on RULER-multi-hop primary cell. Direction: less. Threshold: 1.5 EM raised from 0.5. With 10 seeds and pre-measured per-cell std ≤ 0.6 EM, a true gap of ≥1.5 EM is detectable at p<0.05. If R2-A does not beat R1 by >1.5 EM, the dropped-mass conditioning adds no information beyond what independent training already extracts.

**F2 (sparsity-dependence).** R2-A − R0 EM gap on single-hop NIAH (calibrated to 60–90% baseline band — *not at saturation per red-team I6*) ≥ 1.5 EM. Direction: greater. Threshold: 1.5 EM raised from 1.0. If R2-A wins on tasks where the mechanism predicts no win, the effect is generic recursion-tuning, not coverage-conditioning.

**F3 (signal-decoding).** Fitted scalar weight on c_{t,k} in R2-A's depth-gate, after training, has |w_c|/std(w_c) < 1.5 across 10 seeds. Direction: less in magnitude. Threshold: 1.5 raised from 1.0. If the model does not actually use c_{t,k} yet still outperforms R1, whatever mechanism is operating is not the one we claimed.

**F4 (collapse under dense selection).** When NSA top-n is set so ≥80% of blocks are always selected (degenerate sparsity, δ ≈ 0), R2-A − R1 EM ≥ 1.5. Direction: greater. Threshold: 1.5 EM raised from 1.0. If the gap survives the limit where there's nothing to drop, the mechanism is not "depth gate depends on what was dropped."

**F-Self vs A (NEW, replaces F5 R3-ordering).** R2-A EM ≤ R2-self EM + 1.0 on RULER-multi-hop. Direction: less. Threshold: 1.0 EM. If the δ-proxy form (c-A) does not beat NSA's self-reported coverage form by >1.0 EM, the C1 fix is empty — NSA's self-reported coverage was already sufficient, and the miscalibration claim was null. This is the *cleanest test* of the C1 critique.

**F-StopGrad (NEW, replaces revision-0's smoke test).** R2-A trained with c_{t,k} gradient-stopped equals R2-A unfrozen within 0.7 EM. Direction: less in magnitude of |R2-A − R2-A_stopgrad|. If c_{t,k} as a *learning signal* (gradient flowing) and c_{t,k} as a *frozen feature* perform identically, the strong "interaction" framing of M4 is wrong — c_{t,k} is just a useful redundant feature, and the contribution downgrades from "non-additive interaction" to "feature-engineered scalar." Pre-committed as a partial-falsification: it falsifies M4-strong but the hypothesis still contributes by establishing the feature.

(Removed F5 R3-ordering as a falsification per red-team I3 — MoEUT/SUT analogies show parity, not underperformance. R3 is reported as exploratory contrast only.)

---

## 6. Required experiments (sketch — eval-designer fills in)

**Pre-experiment gate (mandatory, <24 GPU-hours):**
- **F-Calib pre-test:** Train (or use a publicly released) ~350M NSA model, freeze it, measure median rank of gold-evidence block under p_t^slc on RULER-multi-hop dev. Gate the from-scratch run on this passing.

**Datasets (revised primary).**
- **Primary**: RULER multi-hop variable-tracking subset at 32K context (more items than MuSiQue@32K, native long-context).
- **Secondary**: FRAMES (2409.12941); MuSiQue@32K demoted to secondary.
- **Single-hop control**: NIAH-single calibrated to 60–90% baseline band per red-team I6.
- **Saturation control**: HotpotQA-distractor at 16K (medium context, multi-hop).

**Backbones.** From-scratch NSA at 350M and 1.3B. K_max ∈ {1,2,4} weight-tied. K=1 controls "is the effect from MoR alone?"

**Baselines (mandatory).**
- (a) **Dense-attention MoR-recursion baseline (Adaptive Loops 2603.08391-style)** at matched FLOPs — does sparsity even help?
- (b) NSA with no recursion at K=K_max — does recursion even help?
- (c) **ReSSFormer-with-NSA** (R0) — fair comparison to the closest published architecture.
- (d) NSA + MoR with K=1 — halting reduces to none, expected null.

**Ablations.**
- **(i) c_{t,k} operationalization**: c-A (δ-proxy regressor), c-B (JS-divergence), c-self (revision-0 self-coverage), c-form variants (raw / log / sigmoid-warped).
- **(ii) Coverage source**: selection branch only / compression branch only / combined (NSA gate-weighted).
- **(iii) Lock NSA selection to non-content-dependent (random) at same top-n**: R2-with-random-selection should collapse to R0 ± noise (clean test of M1).
- **(iv) Calibration pre-test split**: F-Calib measured on independent split from training; gate decision pre-registered.
- **(v) Mixing ratio for c-A regressor**: full-attention computed at p ∈ {1%, 5%, 10%} of steps.

**Metrics.** EM, F1, average compute budget (E[K_t]), |w_c|/std(w_c), c-A's RMSE against δ_true on a held-out full-attention sample, and predicted-vs-actual gold-block rank on F-Calib.

---

## 7. Risks to the hypothesis

**Risk A (steelman wins) — NSA is well-calibrated on multi-hop in practice.** F-Calib pre-test reveals median gold-evidence-block rank ≤ top-n. Consequence: hypothesis dies pre-experiment. Contribution: a clean, citable measurement of NSA's effective miscalibration on a specific multi-hop benchmark; informs whether NSA's training induces good importance estimates *de facto*.

**Risk B (compression-branch covers it) — NSA's compression branch carries enough non-selected-block info that c_{t,k} adds nothing.** Contradicts SSA Appendix F empirically, but the production deployment may differ. Consequence: F1 fires. Contribution: bound on how much joint coupling matters when compression is well-trained.

**Risk C (M4-strong is wrong, M4-weak is right) — c_{t,k} helps only as a feature, not as an interaction.** F-StopGrad fires. Consequence: framing downgrades from "non-additive interaction" to "useful feature." Contribution: still useful, with a clear architectural recommendation that does not require any "interaction" interpretation.

**Risk D (HRM-style fixed-point fragility) — recursive operators fail to converge for reasons unrelated to halting (HRM critique 2601.10679).** Consequence: all R-cells fail similarly, the *between-regime* differences are within HRM-style noise of the *NSA-vs-dense* gap. Contribution: clean separation of recursive-convergence failure from sparsity-induced gating failure via baseline (a) vs (b).

**Risk E (scale-frontier) — at ≤1.3B / 30B tokens, no regime matches dense-attention baselines, and within-regime gaps are dominated by NSA-vs-dense (2504.17768).** Contribution: scaling-frontier signal of when the joint problem becomes architecturally load-bearing.

**Risk F (c_{t,k} form-family wrong) — F3 fires for our chosen forms but not for an unsearched form.** Mitigated by ablation (i) which forces c-A / c-B / c-self / form variants. Contribution: forced family-level rejection or confirmation.

---

## 8. Cheaper-falsification path (revised)

Single-experiment falsification at ~1/10th of full cost:

**Path 1 (now mandatory pre-experiment): F-Calib gate.** <24 GPU-hours. Described in §6. Falsifies the entire miscalibration claim if median gold-evidence-block rank ≤ top-n.

**Path 2 (mid-cost, ~1 GPU-week): Frozen-backbone fine-tune.** Take a publicly released NSA-trained model (or DSA / DeepSeek-V3.2 2512.02556). Append a 2-layer recursive head with MoR's expert-choice routing, K=4 weight-tied. Train R1, R2-A, R2-self variants on a small multi-hop QA set (~10k examples). If R2-A − R1 < 1.0 EM on RULER-multi-hop subset under this lightweight setup, the hypothesis is at minimum significantly weakened. **Caveat (red-team I4)**: with a strong pre-trained backbone, R1 ≈ R2 may be the *expected* outcome (the pre-trained model has internalized coverage info already — Risk A inherits). We pre-commit to interpreting Path 2's R1 ≈ R2 as "weak evidence against from-scratch hypothesis, escalate to full from-scratch only if Path 2 R2-A − R1 ≥ 0.5 EM."

**Tier gate:** F-Calib passes → Path 2 → if Path 2 R2-A − R1 ≥ 0.5 EM → escalate to from-scratch full ladder. Otherwise terminate.

(Removed revision-0's <1 GPU-day "smoke test" — red-team I1 was correct that an inference-time correlation of c_{t,k} with correctness on a pre-trained backbone does not falsify training-time conditioning. F-Calib replaces it as a real, falsifying pre-test.)

---

## 9. Sources

| arxiv ID | Title | Used for |
|---|---|---|
| 1603.08983 | Adaptive Computation Time | M2 background, ACT family |
| 1807.03819 | Universal Transformer | Per-position halting (background) |
| 2107.05407 | PonderNet | M2 background (no longer load-bearing in revision-1) |
| 2402.16837 | Latent multi-hop reasoning in LLMs | M5 multi-hop motivation |
| 2404.15574 | Retrieval Head | M5 multi-hop retrieval |
| 2409.12941 | FRAMES | Secondary benchmark |
| 2502.11089 | NSA | Backbone, M1, three-branch architecture |
| 2502.13189 | MoBA | Sparse routing alternative (background) |
| 2507.10524 | MoR | **Halting recipe (replaces TRM-style)**, M3 KV reuse |
| 2510.01585 | ReSSFormer | Adjacent prior art (R0 ≈ ReSSFormer-with-NSA) |
| 2510.04871 | TRM | Acknowledged but NOT used for halting recipe (C3 fix) |
| 2511.20102 | SSA | **C1 fix core**: Proposition 4.1 (Gradient Update Deficiency), Theorem 1 (dropped-mass bound), Appendix F (compression branch fails to extrapolate) |
| 2512.02556 | DeepSeek-V3.2 / DSA | Cheap-test backbone option |
| 2412.10319 | SCBench | M5 multi-hop sparse-attention robustness |
| 2504.17768 | Sparse Frontier | Risk E (scale frontier) |
| 2602.08426 | Prism | M3 c-A features (RoPE spectral signal); M1 mean-pooling-as-low-pass |
| 2603.08391 | Adaptive Loops | Adjacent prior art (dense-attention version of recipe), baseline (a) |
| 2602.07150 | On Randomness in Agentic Evals | C2 fix: noise-floor evidence, 10-seed minimum |
| 2601.10679 | HRM mechanistic critique | Risk D |
| 2310.07096 | Sparse Universal Transformer | Background (no longer load-bearing) |
| 2405.16039 | MoEUT | Background (no longer load-bearing) |

**Removed from revision-0:**
- 2509.04475 (ParaThinker / Tunnel Vision) — M6 dropped per red-team S1.
- 2602.11451 (LoopFormer) — listed in revision-0 but not load-bearing; retained mention in §1 as part of family.

**Added in revision-1:**
- 2511.20102 (SSA) — **load-bearing for C1 fix.** Proposition 4.1 + Theorem 1 + Appendix F.
- 2510.01585 (ReSSFormer) — adjacent prior art per red-team S2.
- 2603.08391 (Adaptive Loops) — adjacent prior art per red-team S2; mandatory baseline (a).
- 2602.08426 (Prism) — c-A regressor features (RoPE spectral band as cheap calibrator).
- 2602.07150 (Randomness in Agentic Evals) — noise-floor citation for C2 fix.
- 2409.12941 (FRAMES) — secondary benchmark cite.
- 2402.16837 (Latent multi-hop) — M5 motivation for multi-hop subset condition.
