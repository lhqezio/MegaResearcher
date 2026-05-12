# Red-Team Critique of H4 — coverage-conditioned halting on NSA

Critiquing: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-4/output.md`
Targeting gap: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/gap-finder-1/output.md` Gap 4
Revision round: 1 (initial review)

---

## 1. Verdict

`REJECT (revision-1)`

The hypothesis has a sound mechanistic core (coverage signal recoverable from NSA softmax — that part *is* technically grounded by §3.3.2 of the NSA paper) and a real, surviving novelty gap (Gap 4 holds even after independent verification). But three load-bearing structural problems prevent approval:

1. **A self-justification loop in the coverage signal**: c_{t,k} is constructed from NSA's own importance scores, which by definition are maximized over selected blocks. The signal is near-tautological unless NSA's importance estimate is *miscalibrated* — and the hypothesis does not commit to that weaker, more interesting claim.
2. **A recipe conflation**: "TRM-style recursive forward pass with PonderNet halting head" is not faithful to TRM (TRM uses Q-learning ACT, not λ_n = sigmoid PonderNet halting). The hypothesis is a *new* recipe being mislabeled as established.
3. **Falsification thresholds at or below the MuSiQue 1B-scale noise floor** (F1 threshold 0.5 EM, F2 1.0 EM, F4 1.0 EM). With 5 seeds and a ~2.4k-example dev set, per-cell std typically exceeds these gaps; the hypothesis is structurally unfalsifiable at the proposed scale.

These are fixable in revision; the gap and mechanism are not dead. KILL would be premature.

---

## 2. Gap re-verification

I ran three independent literature queries beyond the gap-finder's two.

**Query 1**: `PonderNet ACT halting NSA sparse attention selection joint training`
**Result count**: 10 papers. None instantiate joint per-token depth halting + per-token NSA-style block selection. Closest hits are NSA itself (2502.11089), variants of attention sparsification (Twilight 2502.02770, BLASST 2512.12087, NOSA 2510.13602) — none have a depth-halting axis.

**Query 2**: `looped recursive transformer NSA block sparse multi-hop reasoning`
**Result count**: 10 papers. Returned **ReSSFormer (2510.01585)** — a recursive sparse structured transformer with R2MU (recurrent inference, fixed K) + ASAM (sparse top-k routing + sparsemax + expert MoE) + SOES, evaluated on multi-hop QA. Read §3.1–3.4: ReSSFormer iterates K times (fixed, not adaptive) and uses sparse attention, but does **not** have per-token depth halting. The "Block recurrently applied K times" loop in §3.1 is uniform across tokens. Also returned **Adaptive Loops and Memory in Transformers (2603.08391)** — adaptive per-layer looping with learned halting on dense attention, no sparse routing. **PLT 2510.24824** — looped + sliding-window (already in the gap-finder corpus).

**Query 3**: `joint depth halting attention sparsity per-token reasoning early exit`
**Result count**: 10 papers. None match. Returned token-level early exits (SkipDecode 2307.02628, River-LLM 2604.18396, LYNX 2512.05325) — these are decoding-time exits in dense models, not joint depth+sparsity training. SpecExit 2509.24248 and PALBERT 2204.03276 are likewise dense-attention.

**Verdict on gap**: The gap claim **survives**. No paper jointly trains per-token depth halting with per-token sparse attention routing. ReSSFormer is the closest near-neighbor and has fixed-K recursion, so the smith's R0 (no halting, fixed K) baseline approximates ReSSFormer-with-NSA, but the R1/R2/R3 cells are unoccupied. **Suggestion**: smith should cite ReSSFormer 2510.01585 and Adaptive Loops 2603.08391 as adjacent prior art and explain why R0 ≠ ReSSFormer-with-NSA (smaller K, weight-tying details).

---

## 3. Citation spot-checks

I verified six citations directly via `read_paper` / `paper_details`. Summary:

**(a) NSA 2502.11089 §3.3.2 — Importance Score Computation.** Verified. The paper says: `p_t^cmp = Softmax(q_t^T · K̃_t^cmp)` and `p_t^slc = p_t^cmp` (when blocking schemes match), with Equation (9) giving the spatial-relationship version when they differ. The smith's claim that c_{t,k} = "(sum over selected blocks of importance mass) / (sum over all blocks of importance mass)" is **technically computable from p_t^slc with no extra forward pass**. M3 is grounded. **However**, the smith elides the structural problem: p_t^slc is *exactly* what NSA used to pick the top-n blocks. Coverage in this construction is bounded above by 1 and approaches 1 *by construction*: if you select the top-n blocks ranked by p_t^slc, the ratio (selected mass) / (total mass) is monotonically increasing in n and approaches 1 as the top-n captures the bulk of the softmax mass. This is the **self-justification problem** flagged in the attack vector list. Severity: **Critical**.

**(b) PonderNet 2107.05407 §2.2.** Verified. PonderNet's λ_n is computed from h_n via the step function s. **The smith claims "PonderNet's halting head is a learned function of the post-attention hidden state."** This is roughly correct but glosses over an important detail: PonderNet as published is **per-sequence**, not per-token (the step function takes x → y, halting once for the whole sample). Universal Transformer (1807.03819) and ACT (1603.08983) are per-position. The smith uses "PonderNet" as a shorthand for "per-token PonderNet-style," which is not quite what 2107.05407 specifies. Severity: **Important** (terminology, not load-bearing).

**(c) TRM 2510.04871 §3.2 / Figure 3.** Verified — and this surfaces a problem. TRM's pseudocode (`def deep_recursion`) shows ACT-style **Q-learning** halting: `loss += binary_cross_entropy(q_hat, (y_hat == y_true))`, with `q_hat = Q_head(y)` and `if q_hat > 0: break`. TRM does NOT use PonderNet's λ_n cumulative-product formulation. The smith's "TRM-style recursive forward pass (K outer loops, weight-tied) with per-token PonderNet halting head" is **a new combination, not a faithful instantiation of TRM**. Section 3.2 of the TRM paper explicitly contrasts ACT (which TRM uses) with the alternatives. The smith is mixing two incompatible recipes and labeling the chimera as "TRM-style." Severity: **Important**. The fix is to either (i) replace "TRM-style" with "Universal Transformer-style with PonderNet halting" or (ii) keep TRM's Q-learning ACT and reframe as "ACT halting + NSA coverage signal."

**(d) ParaThinker / Tunnel Vision 2509.04475.** Verified — and the citation is **stretched**. Tunnel Vision in ParaThinker refers to *sequential CoT generation hitting a ceiling where parallel thought paths help*. It is about token-level autoregressive CoT, NOT about depthwise recursion or premature halting. The smith's M6 ("Coverage-conditioned halting is a *negative-feedback* mechanism (drops in coverage → suppress halting → keep iterating) and should counteract Tunnel Vision specifically when tunnel-vision is caused by premature halting on a sparsity-corrupted state") imports a concept from a different setting and asserts a connection without justification. The smith does flag this as "a prediction, not a guarantee," which mitigates but does not eliminate the issue. Severity: **Suggestion** — drop M6 or reframe as a much weaker analogy.

**(e) MoR 2507.10524 §3.** Verified. MoR uses expert-choice / token-choice routing for adaptive recursion, with KV reuse. It does **not** use PonderNet halting. The smith's M3 ("mechanically analogous to MoR's KV-reuse routing") is fair as analogy. The smith's claim that MoR is in the "per-token *depth* routing" family is correct. Severity: **OK**.

**(f) SCBench 2412.10319 / Sparse Frontier 2504.17768.** SCBench abstract emphasizes KV-cache-centric eval and "highlights robustness of sparse attention methods." The smith's framing "sparse-attention KV compression is more robust on simple retrieval than on multi-hop" is consistent with the abstract but I did not read §empirical to verify the precise multi-hop claim. The **2–6 point swing** number cited as magnitude reasoning would need direct §verification before being load-bearing. Severity: **Suggestion** — eval-designer should confirm the magnitude number.

---

## 4. Mechanism critique

**M1 (NSA selection is content-dependent and discrete → post-attention representation is a random function of selected blocks).** Substantively correct, but the smith conflates two things: NSA's selection is content-dependent, but the post-attention representation also includes outputs from the **compression branch** and **sliding-window branch** (NSA has three branches, gated). Equation (5)-style gated combination in NSA §3.2 means the representation is **not** purely a function of the selected blocks — it's a weighted combination of compressed, selected, and window outputs. The "random function of selected blocks" claim is too strong: information from non-selected blocks survives via the compressed branch (lossy but present). This **weakens** the variance argument in M2.

**M2 (PonderNet halting head inherits sparsity-induced variance).** Conditional on M1, this argument is structurally fine — but its load-bearing-ness is reduced when M1 is correctly stated. NSA's compression branch already delivers *lossy global* information; the variance attributed to selection-branch dropout is one component of total halting-head variance, not the dominant component. The mechanism predicts a smaller effect than the smith claims.

**M3 (Coverage signals are recoverable from NSA without extra forward passes).** Correct mechanically (per NSA §3.3.2). **But the self-justification problem is severe**: the importance score p_t^slc is what NSA used to select blocks. The ratio (selected mass) / (total mass) approaches 1 by construction unless NSA's importance estimate is miscalibrated. In the limit where NSA selects the top-n blocks ranked by p_t^slc, c_{t,k} ≥ (sum of top-n p_t^slc values) / 1 ≈ a high number, depending only on top-n and the sharpness of p_t^slc — not on whether information was actually preserved. The variance the hypothesis depends on is in **the residual mass** (sum over non-selected blocks of p_t^slc), which is bounded above by (1 − sum over selected). This is precisely the situation where the *importance estimate itself* must be wrong for the mechanism to bite. The hypothesis implicitly bets on NSA being miscalibrated, which is a **weaker, less interesting claim** than the framing suggests, and contradicts NSA's own §6.1 robustness arguments. Severity: **Critical**.

**M4 (R3 underperforms R2 by ≥1 EM via shared low-rank bottleneck).** The smith cites MoEUT (2405.16039) and Sparse Universal Transformer (2310.07096) as analogues. **The analogy is reversed**: MoEUT shows shared expert-choice + halting works *at parameter parity* with non-shared, not that shared underperforms separate routers. SUT's setup is similar. The smith is using a parity result to justify a *negative* prediction about shared routers — this is a non-sequitur. The R3 < R2 prediction is essentially quantitative-thumbsuck (the "≥ 1 EM" threshold is unjustified). Severity: **Important**. Either re-cite (find a paper that actually shows shared routers underperform separate ones for related decisions) or downgrade R3 from a falsification threshold to an exploratory contrast.

**M5 (effect depends on multi-hop chain length).** Reasonable, supported by Retrieval Head 2404.15574 (verified) and SCBench 2412.10319 (citation roughly correct). Severity: **OK**.

**M6 (Tunnel Vision compatibility).** Cited paper does not address this scenario; smith already flags as speculative. Drop or weaken. Severity: **Suggestion**.

**Speculative element on c_{t,k} form (raw mass / log-mass / KL).** Pre-committing to ablate three forms is good practice; if all three fail, M3 falsified. Severity: **OK**.

---

## 5. Falsifiability assessment

I evaluate each F-criterion against (i) operationalizability and (ii) noise-floor calibration.

**F1 (R2 EM ≤ R1 EM + 0.5 on MuSiQue at 32K).** Operationalizable. **Threshold is at noise floor**: MuSiQue at 1B scale with 5 seeds typically shows per-cell std ≥ 0.5 EM (the dev set is ~2.4k examples). 0.5 EM is not statistically distinguishable from zero with 3-5 seeds. The smith should commit to a specific test (paired bootstrap, Welch's t with explicit n_seeds and α) and a per-cell std budget — if the budget is exceeded, more seeds are required to keep F1 falsifiable. **Severity: Critical**. As stated, F1 is structurally non-distinguishable from "no effect."

**F2 (R2 minus R0 on single-hop NIAH ≥ 1.0 EM falsifies).** Direction: greater. Single-hop NIAH ceilings are typically ≥ 95% on small models if any retrieval works at all; a 1.0 EM gap at the ceiling is plausible but could easily be saturation noise. Operationalizable but the threshold is a guess. Severity: **Important**.

**F3 (|w_c|/std(w_c) < 1.0 across 3 seeds falsifies).** This is well-formed and operationalizable: fit the halting head, extract the c_{t,k} weight, take z-score across seeds. **Cleanest of the five criteria.** Severity: **OK**.

**F4 (R2 − R1 ≥ 1.0 with degenerate top-n covering ≥ 80% of blocks falsifies).** Direction: greater. Operationalizable (just set top-n high). 1.0 EM threshold again at noise floor. The 80% threshold is also somewhat arbitrary — at 80% selection NSA is already nearly dense; the smith should pick the threshold based on a measured calibration of c_{t,k}-variance vs top-n. Severity: **Important**.

**F5 (R3 EM ≥ R2 EM falsifies M4).** Operationalizable but tied directly to the suspect M4 claim (see §4). If the smith downgrades R3 from a prediction to an exploratory contrast, F5 is no longer load-bearing. Severity: **Important**.

**Cheaper falsification path.** The "correlation < 0.05" smoke test is **not a falsification path**. Even if c_{t,k} − correctness correlation is 0 at inference time on a fixed pretrained backbone, explicit conditioning at training time could still help via regularization. The smoke test is at most a sanity check. Severity: **Important** — relabel as "preliminary sanity check, not a falsification."

The mid-cost falsification (frozen NSA + recursive head + halting head, ~1 GPU-week) is more defensible but inherits Risk A: with a strong pretrained backbone, R1 ≈ R2 is the *expected* outcome (the pretrained model already encodes coverage info). The smith pre-positions this as "a clean negative result," which is honest, but means **F1 is structurally likely to fire even if the from-scratch hypothesis is correct**. This is a design pathology — the cheap-path falsification cannot reject the from-scratch hypothesis, it can only weaken it.

---

## 6. Strongest counter-argument (steelman)

The strongest opposing case is:

> *NSA's three-branch architecture (compressed + selected + sliding-window) already provides redundant coverage. Coverage-conditioning the halting head is solving a problem that doesn't exist — the variance is small, and the residual stream encodes "I am missing information" implicitly through the gated combination of branches. Independent training (R1) lets the halting head learn this implicitly via the standard backprop signal; explicit conditioning (R2) may reduce variance slightly but at the cost of a feature-engineered scalar that competes with the much higher-dimensional implicit signal already in h_k. R2 ≈ R1 ± noise. The R2 > R1 by ≥ 3 EM prediction is implausible at 1B scale.*
>
> *Furthermore, the joint problem is benign in production sparse architectures (Risk B): NSA's training objective shapes the importance estimator to **be** coverage-aware; the auxiliary-loss path described in §6.1 is one mechanism, but the no-auxiliary path NSA chose still ends up with importance scores that correlate with what's needed for downstream tasks. Otherwise NSA wouldn't work. So at training-time, c_{t,k} is approximately constant (≈ 1) across selections, and conditioning on it is conditioning on a near-constant.*

This is a serious counter-argument and the smith's Risk A/B already partly acknowledge it. **The hypothesis as stated does not adequately rebut this steelman**: the smith says "may show only that explicit coverage conditioning is unnecessary" but does not commit to an experimental comparison that would distinguish the two cases (smith's mechanism vs steelman: implicit-already-encodes). The right rebuttal would be the F3 criterion (|w_c|/std), but F3 alone doesn't distinguish "the model uses c_{t,k} because it's a useful redundant signal" from "the model uses c_{t,k} because it carries information not in h_k." A targeted ablation: train R2 with c_{t,k} *frozen at training time* (gradient-stopped on c_{t,k} for the halting head); if R2_frozen ≈ R2_unfrozen, then c_{t,k} is just a feature, not an interaction signal.

---

## 7. Severity-tagged objections

### Critical (must fix in revision)

**C1. Self-justification of c_{t,k}.** The coverage signal is computed from NSA's own importance scores, which are maximized over selected blocks by definition. The mechanism implicitly requires NSA's importance estimate to be miscalibrated (otherwise c_{t,k} ≈ constant). The hypothesis must commit to a precise statement of *what kind of miscalibration* it predicts and add an experiment that measures it directly (e.g., correlation between p_t^slc and oracle-block-importance derived from gold answer-positions). If NSA is well-calibrated, the mechanism is null.

**C2. F1 / F2 / F4 thresholds at MuSiQue 1B noise floor.** Per-cell std at 5 seeds on MuSiQue at 1B scale is ≥ 0.5 EM in published literature (see Sparse Frontier 2504.17768 and similar). F1's 0.5 threshold is structurally non-distinguishable from zero. The hypothesis must either (i) commit to ≥ 10 seeds, (ii) raise thresholds to ≥ 1.5 EM (which then collides with the predicted-magnitude band), or (iii) add an explicit per-cell std budget and tighten the experimental contract.

**C3. TRM-PonderNet recipe conflation.** TRM uses ACT-style Q-learning halting, not PonderNet λ_n. The smith's "TRM-style recursive forward pass with per-token PonderNet halting head" is a new chimera being labeled as established. Either (i) reframe as "Universal Transformer / per-token ACT-style halting" or (ii) keep TRM's Q-learning ACT and adapt M2/M3 to that formulation. The current framing is technically misleading.

### Important (should fix)

**I1. Cheap-path falsification (correlation smoke test) is not a falsification.** Inference-time correlation < 0.05 doesn't kill training-time conditioning. Relabel and add a real cheap-path falsification: train R2 with c_{t,k} *gradient-stopped* and compare to R2 with gradient flowing.

**I2. M1 overstates the variance argument.** NSA's compression branch and sliding-window branch deliver information from non-selected blocks. The post-attention representation is not purely a function of selected blocks. Restate M1 to acknowledge the three-branch architecture and re-derive M2's variance argument under the gated combination.

**I3. M4 (R3 < R2 by ≥ 1 EM) misuses MoEUT/SUT analogy.** Those papers show parity, not underperformance. Either find supporting prior art or downgrade R3 from a predicted-magnitude claim to an exploratory contrast (no falsification threshold).

**I4. Scale/cost.** 350M-1.3B from-scratch NSA training, 30B tokens, 8+ runs is well above the spec's 2000-GPU-hour fence (estimating ~60-80 GPU-weeks for the full ladder). The mid-cost path (frozen NSA + recursive head, ~1 GPU-week) is reasonable but inherits Risk A. The hypothesis needs an explicit experimental tier ("if the cheap path returns R2 > R1 by ≥ X, escalate to from-scratch; else terminate") rather than asserting both as equally valid.

**I5. PonderNet is per-sequence, not per-token in 2107.05407.** Restate as "per-token PonderNet-style" or cite Universal Transformer 1807.03819 / ACT 1603.08983 for the per-token formulation.

**I6. F2 threshold (1.0 EM on single-hop NIAH).** NIAH ceilings on small models are noisy near saturation; a 1.0 EM gap at 95-99% accuracy is essentially noise. Pick a metric (e.g., NIAH at a length where baseline accuracy is in the 60-90% band) and recalibrate.

**I7. MuSiQue at 32K context is contrived.** MuSiQue's native context is < 2K. Wrapping it in 30K of distractor padding tests retrieval over filler more than multi-hop reasoning. Use FRAMES or RULER-multi-hop as the primary benchmark; relegate MuSiQue@32K to a secondary cell.

### Suggestion (nice to have)

**S1. Drop or downgrade M6 (Tunnel Vision).** The cited paper 2509.04475 addresses sequential CoT, not depthwise recursion. The connection is speculative.

**S2. Cite ReSSFormer 2510.01585 and Adaptive Loops 2603.08391** as adjacent prior art and explain why they don't subsume Gap 4 (fixed-K, dense-attention respectively).

**S3. Add a calibration experiment for p_t^slc.** Before the main experiment, measure how peaked NSA's importance estimate is on multi-hop QA — if it's near-deterministic (top-n captures > 99% of mass), c_{t,k} is near-constant and the mechanism is dead before any joint-training run.

---

## 8. Recommendation to hypothesis-smith

The hypothesis is salvageable. The gap is real, the c_{t,k} signal is technically computable, and the architecture is genuinely novel. But three things must change in revision:

1. **Tighten the mechanism**: Commit to either "NSA importance scores are miscalibrated on multi-hop" (a strong, falsifiable claim that needs its own preliminary experiment) or "the c_{t,k} feature is informative as regularization, not as an interaction signal." Both are interesting; the current framing waves at both and commits to neither.

2. **Recalibrate falsification thresholds**: Either raise to ≥ 1.5 EM or commit to ≥ 10 seeds with an explicit per-cell std budget. F1 at 0.5 EM with 5 seeds is structurally unfalsifiable.

3. **Fix the TRM-PonderNet conflation**: Pick one halting recipe (ACT Q-learning *or* PonderNet λ_n) and stay with it. The current "TRM-style + PonderNet" is a misnomer that will draw immediate fire.

Bonus: drop M6, add ReSSFormer/Adaptive Loops as citations, add the gradient-stopped-c_{t,k} ablation as a real falsification of "interaction vs feature."

If the smith's revision addresses C1 (commits to a precise miscalibration claim or reframes as feature/regularization) and C2 (recalibrates thresholds) and C3 (cleans up recipe), the hypothesis is approveable. If C1 cannot be defended with a preliminary experiment showing NSA's importance estimate is *actually* miscalibrated on the target benchmarks, the hypothesis becomes a KILL — the steelman will hold.

---

`REJECT (revision-1)`
