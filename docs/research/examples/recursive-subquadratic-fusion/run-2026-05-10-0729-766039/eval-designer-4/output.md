# Eval Design — H4: dropped-mass-conditioned MoR halting on NSA

Designing for: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-4/revision-1/output.md`
Red-team round-2 verdict: APPROVE (revision-1) — `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-4/revision-1/output.md`
Carry-over residual concerns from red-team round 2: I1 (regressor feature blindness), I2 (double-discrete routing stability), I3 (KV caching), I4 (compression-branch-fails-on-multi-hop is implicit extrapolation), I5 (F-Calib threshold unanchored).

---

## 1. Hypothesis being tested (restated for self-containment)

On an NSA backbone (arXiv:2502.11089) trained from scratch with per-token recursive depth via MoR's expert-choice routing (arXiv:2507.10524 §2.2.1, sigmoid gate `g_t^r = G(θ_r^T H_t^r)`, weight-tied K_max=4 recursion blocks), four coupling regimes are compared at matched FLOPs:

- **R0**: no halting, fixed K=K_max, NSA backbone (≈ ReSSFormer-with-NSA, 2510.01585).
- **R1**: independent — MoR routing trained jointly with NSA, no information shared between routers.
- **R2-A**: dropped-mass-conditioned (c-A) — gate score augmented with c_{t,k} ∈ [0,1] = a *trained regressor* ĉ_φ(features) → δ̂(t,k), supervised on δ_true measured during a 5%-mixture full-attention training regime (SSA arXiv:2511.20102 §5).
- **R2-B**: gate-disagreement variant (c-B) — c_{t,k} = JS-divergence(p_t^slc, p_t^cmp).
- **R2-self**: revision-0 form — c_{t,k} = NSA's self-reported coverage ratio (predicted null).
- **R3**: shared-router (exploratory contrast, no falsification threshold).

**Predicted contract** (≥10 seeds per cell, paired bootstrap n=10000 α=0.05 two-sided):
- R2-A − R1 ≥ 1.5 EM on RULER-multi-hop variable-tracking @32K.
- R2-A − R0 ≥ 2.0 EM on RULER-multi-hop variable-tracking @32K.
- R2-A − R2-self ≥ 1.0 EM (cleanest test of the C1 fix).
- |R2-A − R2-A_stopgrad| < 0.7 EM falsifies the strong-interaction framing.
- R2-A − R0 within ±1.0 EM on single-hop NIAH calibrated to 60–90% baseline band.

**Pre-experiment gate (mandatory)**:
- F-Calib: ≥30% of multi-hop gold-evidence-block tokens have median gold-block rank under p_t^slc strictly greater than NSA's top-n value, on a frozen NSA-mini measured on a HotpotQA-distractor pilot. The 30% threshold is calibrated against the pilot per red-team I5 (see §5).

**Falsification criteria** (full set in §6).

---

## 2. Datasets

### 2.1 Primary multi-hop benchmark — RULER variable-tracking @32K

| Field | Value |
|---|---|
| Name | RULER (Real-context evaluation of Long-context models), variable-tracking subtask |
| HF dataset ID | `simonjegou/ruler` configs `4096`, `8192`, `16384` (filter `task == "variable_tracking"`) for shorter cells; **32K must be generated locally** via NVIDIA's official generator at `https://github.com/NVIDIA/RULER` (Apache-2.0). |
| Licence | Apache-2.0 (NVIDIA original); `simonjegou/ruler` mirror inherits Apache-2.0. **Verified via hf_inspect_dataset 2026-05-10**: dataset is valid, schema = (context, question, answer_prefix, answer, task, max_new_tokens), 16384 split is 180 MB. simonjegou's mirror only ships up to 16K; for 32K we must regenerate. |
| Why appropriate | RULER variable-tracking explicitly threads ≥3 mentions of a variable through a long haystack — every gold answer requires retrieving multiple distinct positions, by construction satisfying the M5 ≥3-block-evidence-chain condition. NIAH single-key items in the same dataset serve as the F2 single-hop control. |
| Splits | RULER ships only `test`. We pre-register a 50/50 random split of the 32K test items into `eval-A` (used for hyperparameter calibration & pilot measurement) and `eval-B` (held out for the formal F1/F2/F4/F-Self/F-StopGrad tests). Seed for the split is fixed at 1 and pre-registered. |
| Sample sizes | RULER variable-tracking @32K with `num_samples=500` per split is the published default. We commit to `num_samples=2000` per split (eval-A=1000, eval-B=1000) to lift power, generated from non-overlapping random seeds. |
| Power | With σ_per_cell budget = 0.6 EM, paired-difference SE = 0.6/√10 ≈ 0.19 EM; α=0.05 two-sided minimum-detectable effect ≈ 0.38 EM. A true 1.5 EM gap at this SE has > 99% power. The 2000 items per split is a contingency for the σ budget being violated; if observed σ > 0.6 EM on R0 pre-measurement, we double seeds per cell to 20 (already pre-committed). |

### 2.2 Secondary — FRAMES + MuSiQue@32K

| Field | FRAMES | MuSiQue@32K |
|---|---|---|
| HF dataset ID | `google/frames-benchmark` | `dgslibisey/MuSiQue` |
| Licence | Apache-2.0 (verified via hf_inspect_dataset 2026-05-10) | CC-BY-4.0 (verified) |
| Why | Multi-document multi-hop with reasoning-type tags (numerical, tabular, multiple-constraints) — covers a different multi-hop distribution than RULER variable-tracking. | Standard 2-/3-/4-hop bridging questions; we wrap `paragraphs` field in 32K of distractor-padding from the same dataset's other items per the original MuSiQue@32K protocol (revision-1 §6 demoted MuSiQue from primary, retained as robustness check). |
| Sample sizes | 824 test items (single split, full set used). | 2417 validation items, full set used. |
| Splits | `test` only (no train; we use it as held-out eval, retrieval prompt = question only, no oracle docs). | `validation` split as held-out eval. |

### 2.3 Single-hop saturation control — RULER NIAH-single, calibrated band

| Field | Value |
|---|---|
| Name | RULER `niah_single_1` task |
| HF dataset ID | `simonjegou/ruler` (filter `task == "niah_single_1"`) |
| Licence | Apache-2.0 |
| Why | Per red-team I6, F2 must use a single-hop benchmark in a non-saturated band (60–90% baseline accuracy). NIAH-single-1 is the canonical NSA NIAH stress; we tune context length so R0 lies between 60 and 90% accuracy. The expected band given NSA's published 32K results (≈99% NIAH on Pile @32K, SSA Tab.3) suggests we need to push to 64K context — but per the red-team's explicit warning that any threshold near saturation is noise, we instead reduce model capacity (350M not 1.3B) so that R0 NIAH-single @32K lies in the band. |
| Splits | Same eval-A/eval-B 50/50 split as RULER variable-tracking, joint sampling. |
| Sample sizes | 500 per split. NIAH is high-variance per-item; at σ_per_cell ≤ 0.7 we still detect the F2 1.5 EM threshold at >0.8 power. |

### 2.4 Saturation cross-check — HotpotQA-distractor @16K

| Field | Value |
|---|---|
| Name | HotpotQA distractor split, repacked at 16K context |
| HF dataset ID | `hotpotqa/hotpot_qa` config `distractor` (verified via hf_inspect_dataset 2026-05-10; train + validation splits, distractor schema includes supporting_facts + 10-doc context) |
| Licence | CC-BY-SA-4.0 |
| Why | Used **only** as the F-Calib calibration pilot (§5, I5 fix). Picking the F-Calib threshold needs a known-easy multi-hop benchmark to anchor. HotpotQA at 16K is a moderate-context multi-hop benchmark that NSA-mini will score competently on; we measure miscalibration on it first to set the F-Calib threshold for RULER. |
| Sample sizes | 1000-item dev subsample. |

### 2.5 Pre-training corpus

- **HuggingFaceFW/fineweb-edu** (default config; ~4.3 TB; ODC-By 1.0). Subsample to ~30B tokens. Verified via hf_inspect_dataset 2026-05-10.
- For long-context up-training (8K → 32K), follow MoR §B.1's recipe: use `togethercomputer/RedPajama-Data-1T` `arxiv` and `wikipedia` slices for documents with native long context. Verified.

---

## 3. Backbones and matched-FLOPs design

### 3.1 Model sizes

Two scales, pre-registered:
- **Tier-1 (mandatory)**: 350M parameters, 16-layer, d_model=1024, 16 heads, RoPE base 10K, NSA block size B=64, top-n=16 blocks (≈25% selection rate at 32K), sliding window w=512. 30B-token training, 8K→32K curriculum.
- **Tier-2 (contingent on Tier-1)**: 1.3B parameters, 24-layer, d_model=1536, 24 heads, NSA B=64, top-n=16, w=512. 60B tokens. Only run if Tier-1 R2-A − R1 ≥ 0.5 EM (the gate threshold for from-scratch escalation per revision-1 §8 Path 2).

### 3.2 Matched FLOPs

Following MoR §B.3, "matched FLOPs" means matching forward+backward token-FLOPs at training, *not* parameter count. R0 fixes K=4 (every token gets 4 recursion passes). MoR-cells (R1/R2-*/R3) compute average E[K_t] under expert-choice routing with β chosen so 50% of tokens stop at K=2 — this matches R0's per-token FLOPs at training. K_max=4. We **lock the routing top-k threshold β** identically across R1/R2-*/R3 (β=0.5, the MoR default), so any EM difference between cells is not attributable to differing compute budgets.

We additionally include R0-K2 (fixed K=2, half of R0's compute) and R0-K6 (fixed K=6, 1.5× compute) as compute-axis controls per Risk D — this isolates "is the effect coming from spending more compute on hard tokens, or from the coupling signal?"

### 3.3 KV caching strategy (red-team I3, must specify)

Two MoR caching strategies are compared in MoR §3:
- **Recursion-wise caching**: each recursion step has its own KV cache, recomputed per step.
- **Recursive sharing**: KV is shared across recursion steps.

We pre-register **recursion-wise caching for the primary cells** (R0/R1/R2-A) per red-team's recommendation: it is more expensive but (i) cleanest for the M3 mechanism — the per-step δ(t,k) signal is well-defined when each recursion step has its own NSA branches, (ii) MoR §3 shows recursion-wise outperforms recursive-sharing slightly, so we are not penalising the proposed regime by picking it. Recursive-sharing is reported as Ablation A6 (§7).

NSA's three branches share KV across recursion steps within a single recursion step's attention computation (the standard NSA layout); but the compression branch's mean-pool MLP is recomputed per recursion step (the per-step h^r changes the pool inputs).

### 3.4 Double-discrete-routing stability protocol (red-team I2)

Two simultaneous discrete decisions per token per layer (MoR top-k via β-percentile on the routing logit; NSA top-n on block scores). We pre-register the following stability protocol:

1. **First 5% of training**: Gumbel-softmax warmup on both routers (τ=1.0 → 0.1 cosine decay).
2. **Monitor at every 1000 steps**: gradient-norm of the routing head; gate-collapse rate (fraction of tokens routed to a single bucket); gate entropy.
3. **Trigger condition**: if gate-collapse rate exceeds 0.85 for 3 consecutive checks, reset to Gumbel and add an entropy-regularizer with weight 0.01 (pre-registered fallback).
4. Report all of the above as part of the experiment record so that "training was unstable, R2-A failed" cannot be used as a post-hoc explanation.

### 3.5 Mandatory baselines (see §4)

---

## 4. Baselines

Five baselines, three categories:

### 4.1 Strongest prior-art baseline — Adaptive Loops (arXiv:2603.08391)

Dense-attention adaptive looping with per-layer halting. Trained at matched FLOPs and parameters. Does sparsity even help? If R0 ≥ Adaptive Loops on multi-hop EM at the same FLOPs, the entire experiment may be redirected (sparsity adds nothing; the R2 gain is doing recursion-tuning under a sparsity constraint that does not pay off). Reference implementation: paper §3, no public code as of 2026-05-10 (we re-implement from the paper). **Report card includes Adaptive Loops vs R0 explicitly.**

### 4.2 Architectural ablations of the proposed technique

- **R2-A (full proposal)** — c-A regressor with all features (NSA gate weights, compression-branch entropy, top-n block index spread, RoPE-band positional features per Prism arXiv:2602.08426).
- **R2-A-blind** — c-A regressor restricted to **blindness-immune features only** (compression-branch entropy + RoPE-band positional features). Per red-team I1: features downstream of NSA's selection branch (NSA gate weights, top-n index spread) inherit the selection-branch's blindness about non-selected blocks. R2-A-blind isolates whether the blindness-immune features alone supply enough signal. **Pre-registered prediction**: R2-A-blind − R1 ≥ 0.5 EM (weaker than R2-A's ≥1.5 EM); if R2-A-blind ≈ R1, the c-A regressor's gain is coming from the blindness-inheriting features and is suspect.
- **R2-B** — c-B (JS-divergence) operationalization.
- **R2-self** — revision-0 self-coverage form (predicted null per F-Self vs A).
- **R2-A_stopgrad** — c-A trained with c_{t,k} gradient-stopped on the depth-gate, isolating "useful feature" from "interaction signal" (F-StopGrad).
- **R2-A-random-NSA** — c-A trained but NSA selection is *replaced* with random top-n (M1 control). Should collapse to R0 ± noise; if R2-A-random-NSA still beats R0, the gain is not from coupling at all.

### 4.3 Sanity / trivial baselines

- **Trivial-1**: copy the question's first noun phrase as the answer (string-match heuristic). RULER variable-tracking floor; expected 0% EM.
- **Trivial-2**: nearest-neighbor retrieval over the context (BM25 top-1 sentence). NIAH ceiling for retrieval-only; expected ≥ 90% on NIAH-single, ≤ 5% on RULER VT.
- **R0-K1** — NSA backbone with K=1 (no recursion). Tests "is recursion even useful?" Expected null on multi-hop EM (no depth budget to work with).

---

## 5. Metrics

### 5.1 Primary

| Metric | Definition | Tied to |
|---|---|---|
| **EM (exact match)** | RULER's official evaluator (substring match for variable-tracking and NIAH; HotpotQA's normalize_answer for HotpotQA; FRAMES's official LLM-as-judge for FRAMES per the FRAMES paper §3.3) | F1, F2, F4, F-Self, F-StopGrad |
| **F1 token-overlap** | RULER + MuSiQue standard | Robustness; report alongside EM |

### 5.2 Halting / routing diagnostics

| Metric | Definition | Tied to |
|---|---|---|
| **E[K_t]** (average recursion depth per token) | Mean over eval set of MoR's routed depth | Compute matching; F4 (degenerate top-n) |
| **|w_c|/std(w_c)** | Fitted scalar weight on c_{t,k} in R2-A's depth-gate, z-scored across 10 seeds | F3 (signal-decoding) |
| **c-A regressor RMSE** | RMSE of ĉ_φ vs δ_true on a held-out 5% full-attention training-batch sample | Diagnostic for whether c-A actually learns δ |
| **Gate-collapse rate** | Fraction of tokens routed to the same bucket | Stability monitor (red-team I2) |

### 5.3 F-Calib metrics (per I4, I5)

| Metric | Definition | Tied to |
|---|---|---|
| **Gold-evidence-block rank under p_t^slc** | For each gold-answer-position token in a multi-hop question, rank of the gold-evidence block among all blocks under the selection-branch importance score. Median across heads/layers. | F-Calib gate |
| **Compression-branch attention probe** | Mean attention mass on gold-evidence positions delivered by the compression branch alone (red-team I4 — directly tests whether compression "covers" non-selected blocks for multi-hop) | Diagnostic for I4; reported alongside F-Calib |

### 5.4 Failure-mode metrics (red-team I2 + Risk D)

| Metric | Diagnoses |
|---|---|
| Training-loss curves and per-layer gradient norms | Risk D HRM-style fixed-point fragility |
| Routing entropy over training | Gate collapse |
| NSA selection sparsity (effective fraction of blocks selected) | Whether "top-n" is doing its job |

---

## 6. Statistical analysis plan (pre-registered)

### 6.1 Per-cell variability budget

- **Pre-experiment R0 σ measurement**: 10 seeds of R0 trained, σ_per_cell measured on RULER VT @32K eval-B. **Decision rule**: if σ_per_cell > 0.6 EM, escalate every cell to 15 seeds (per revision-1 §4). If σ > 1.0 EM, halt and re-architect (this would imply baseline noise dominates the 1.5 EM threshold even at 15 seeds).
- **Power**: with σ ≤ 0.6 EM and 10 seeds, paired-difference SE = 0.6/√10 ≈ 0.19 EM; minimum-detectable effect at α=0.05 two-sided ≈ 0.38 EM. F1 (1.5 EM) detected at >99% power; F-Self (1.0 EM) at >99% power; F-StopGrad equivalence band of ±0.7 EM detectable at >90% power (a formal two-one-sided-tests TOST procedure is used for F-StopGrad equivalence — see §6.3).

### 6.2 Tests

- **F1, F2, F4, F-Self vs A (one-sided gain)**: paired bootstrap on the seed-pair vector, n_resamples=10000, α=0.05 one-sided (we predict R2-A > R1; the H_a is directional). Decision: reject H_0: R2-A − R1 ≤ 1.5 EM when bootstrap CI lower bound > 1.5.
- **F-Calib (gate)**: binomial test on the fraction of multi-hop questions whose median gold-block rank > top-n. H_0: fraction = baseline (set by HotpotQA pilot; see §6.4). Reject H_0 if observed fraction > pilot-derived threshold + 1.96·SE.
- **F3 (signal-decoding)**: |w_c|/std(w_c) computed across 10 seeds. Decision: F3 fires (hypothesis falsified on signal-use) if z < 1.5.
- **F-StopGrad (equivalence)**: TOST (two one-sided tests) at α=0.05 for the equivalence band ±0.7 EM. Decision: F-StopGrad fires if both TOST bounds reject. This is a *partial* falsification per revision-1 §5.

### 6.3 Multiple-comparison correction

Six confirmatory tests are pre-registered: F-Calib (gate, run first; rest only run if F-Calib passes), F1, F2, F3, F4, F-Self vs A. F-StopGrad is a partial-falsification test, treated as a separate family.

We pre-register **Holm-Bonferroni correction at family-wise α=0.05 across {F1, F2, F3, F4, F-Self vs A}** — five tests. Per-test α floor: 0.05/5 = 0.01 for the most-significant test (Holm sequence). F-Calib is a gate (uncorrected, unique decision), F-StopGrad is in its own family (uncorrected).

### 6.4 F-Calib threshold calibration (red-team I5 fix)

The 30% threshold in revision-1 was a guess. We replace it with:

1. Run F-Calib pilot on **HotpotQA-distractor @16K** (a known-easy multi-hop benchmark — supporting facts identify gold paragraphs unambiguously). Measure the fraction of multi-hop questions whose median gold-block rank > top-n on a frozen NSA-mini.
2. Pre-register: **F-Calib for RULER VT @32K passes iff observed-fraction > HotpotQA-pilot-fraction + 5pp** (at least 5 percentage points more miscalibrated than the easier benchmark — anchored, not arbitrary).
3. If HotpotQA pilot itself shows ≥ 30% miscalibration, the 30% revision-1 number is supported, and we additionally apply the +5pp anchor on top.

This makes F-Calib's threshold a *measurement-derived* anchor rather than a chosen number.

### 6.5 Pre-registration record

A signed, timestamped pre-registration document will be committed to `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/eval-designer-4/preregistration.md` BEFORE training the first cell. It will contain:
- The eval-A/eval-B random split seed (1).
- σ_per_cell measurement protocol.
- All threshold values listed in §6.1–6.4.
- The Holm-Bonferroni corrected α schedule.
- The HotpotQA pilot's measured fraction (filled in once measured; gating decision deterministic from there).

Post-hoc threshold changes are forbidden.

---

## 7. Falsification experiments (one per criterion)

### 7.1 F-Calib (pre-experiment gate, mandatory)

**Predicted-fail scenario**: NSA-mini's selection branch correctly identifies gold-evidence blocks on RULER VT.
**Procedure**:
1. Train NSA-mini (350M, 8K context, 5B tokens). ~4 GPU-days.
2. Evaluate on (a) HotpotQA-distractor @16K to fix the calibration anchor, (b) RULER VT @32K to test the gating condition.
3. For each test item, identify gold-evidence-block tokens (from RULER's `answer_prefix` matching and HotpotQA's `supporting_facts`). Compute median rank of gold-evidence block under p_t^slc averaged over heads × layers.
4. Apply the calibrated threshold (§6.4).

**What constitutes falsification**: F-Calib fails iff (RULER fraction-with-rank > top-n) ≤ (HotpotQA pilot fraction + 5pp). If F-Calib fails, the entire H4 is null pre-experiment — terminate, write up F-Calib as the contribution.

### 7.2 F1 (non-additivity of R2-A on multi-hop)

**Predicted-fail scenario**: R2-A − R1 ≤ 1.5 EM on RULER VT @32K eval-B.
**Procedure**: Train R0, R1, R2-A at Tier-1 with 10 seeds each (or 15 if σ-budget exceeded). Run paired bootstrap.
**Falsification**: One-sided 95% bootstrap CI lower bound for (R2-A − R1) ≤ 1.5 EM. After Holm correction at α=0.01 if needed.

### 7.3 F2 (sparsity-dependence on single-hop)

**Predicted-fail scenario**: R2-A − R0 ≥ 1.5 EM on RULER NIAH-single calibrated to 60–90% baseline band.
**Procedure**: Train R0, R2-A at Tier-1, 10 seeds each. Tune NIAH context length so that R0 achieves 60–90% on NIAH-single (likely 32K at 350M; if R0 saturates above 90%, increase to 64K).
**Falsification**: Bootstrap CI for (R2-A − R0) on NIAH lower bound ≥ 1.5 EM. Direction: if R2-A wins on tasks where it should not, the effect is generic recursion-tuning.

### 7.4 F3 (signal-decoding)

**Predicted-fail scenario**: model does not learn to use c_{t,k}, yet R2-A still beats R1.
**Procedure**: After R2-A training, extract the scalar weight w_c on c_{t,k} from each layer's MoR gate. Compute |w_c|/std(w_c) across 10 seeds.
**Falsification**: If z < 1.5, the model is not weighting c_{t,k} statistically reliably, so whatever drives the gain is not the coverage feature.

### 7.5 F4 (collapse under dense selection)

**Predicted-fail scenario**: Setting NSA top-n so ≥80% of blocks are selected; R2-A still beats R1 by ≥1.5 EM.
**Procedure**: Train R1, R2-A at Tier-1 with top-n configured so the average selection rate is ≥80% (block size B=64, set top-n ≈ ceil(0.8·n_blocks) at 32K → top-n ≈ 410 of ~512). 10 seeds each.
**Falsification**: Bootstrap CI lower bound ≥ 1.5 EM. If the gap survives the limit where there's nothing to drop, the mechanism is not "depth gate uses dropped mass."

### 7.6 F-Self vs A (cleanest C1-fix test)

**Predicted-fail scenario**: NSA's self-reported coverage is sufficient — R2-A does not beat R2-self by >1.0 EM.
**Procedure**: Train R2-self alongside R2-A, 10 seeds. Compare on RULER VT @32K eval-B.
**Falsification**: Bootstrap CI for (R2-A − R2-self) lower bound ≤ 1.0 EM. If R2-A doesn't beat R2-self meaningfully, the C1 fix was empty.

### 7.7 F-StopGrad (interaction vs feature; partial falsification)

**Predicted-fail scenario**: c_{t,k} as a frozen feature performs identically to c_{t,k} as a learning signal.
**Procedure**: Train R2-A_stopgrad (gradient stopped on c_{t,k} when fed to the gate; the gate sees c_{t,k} as a constant input but does not push gradient back into the regressor through this path). 10 seeds.
**Falsification**: TOST at ±0.7 EM equivalence band. If both TOSTs reject, R2-A and R2-A_stopgrad are equivalent — the strong-interaction framing is wrong, downgrade to "useful feature" framing per revision-1 Risk C.

### 7.8 Diagnostic — compression-branch attention probe (red-team I4)

**Goal**: directly test the I4 concern that "compression branch fails to preserve discriminative signal for multi-hop QA" (an implicit extrapolation from SSA Appendix F's perplexity finding).
**Procedure**: On the trained R0 model, for each gold-evidence-block token in RULER VT @32K, measure the fraction of attention mass the compression branch places on the (compressed representation of the) gold-evidence block. Compare to the fraction the selection branch would have placed under full attention.
**What it tells us**: if the compression branch *does* correctly attend to gold-evidence positions (high mass), then the steelman wins on a different axis — compression recovers what selection drops, and R2-A ≈ R1 is the *expected* outcome. Diagnostic, not a falsification.

---

## 8. Ablations (component isolation)

| ID | Ablation | What it isolates |
|---|---|---|
| A1 | c-A feature subsets: full / blindness-immune-only (R2-A-blind) / blindness-inheriting-only | I1 — does the gain come from blindness-immune features? |
| A2 | c-A regressor functional form: raw scalar / log-mass / sigmoid-warped | C-form variants (revision-1 §6 ablation iii) |
| A3 | Coverage source: selection-only / compression-only / NSA-gate-weighted combined | M1 three-branch isolation |
| A4 | NSA selection lock to random (R2-A-random-NSA) | M1 control: should collapse to R0 |
| A5 | Mixing ratio for c-A regressor full-attention probe: p_full ∈ {1%, 5%, 10%} | Cost-vs-quality of the calibration regressor |
| A6 | KV caching: recursion-wise (default) vs recursive sharing | I3, MoR's two strategies |
| A7 | K_max ∈ {1, 2, 4} | Recursion-depth headroom |
| A8 | Gumbel-softmax warmup duration: 0%, 5%, 10% of training | I2 stability |
| A9 | NSA top-n ∈ {8, 16, 32, 64, ≥80%-of-blocks} | F4 coverage axis |
| A10 | c-A trained on a held-out 5% sample only (no within-batch full-attn pass) — rules out batch-level distribution leak | Soundness of c-A supervision |

A1 is the **primary I1 ablation** — pre-registered as a fork of the F1/F-Self test set, with R2-A-blind reported alongside R2-A.

---

## 9. Compute budget (honest estimate)

### 9.1 Per-cell training cost (Tier-1, 350M, 30B tokens)

Token-FLOPs at 32K context: ~2 × 350M × 30B ≈ 2.1e19 forward+backward FLOPs. On 8 × H100-80GB at 50% MFU and bf16, that's ~2.1e19 / (8 × 990e12 × 0.5) ≈ 5.3 hours/H100 × 8 = 42 hours wall, or ~340 H100-hours per cell at Tier-1.

For NSA + recursion, the recursion overhead is ~K_max × forward (weight-tied), but the MoR routing offsets this by averaging E[K_t] ≈ 2 — so effective compute is matched at ~340 H100-hours/seed/cell.

### 9.2 Total Tier-1 cost

Cells: R0 + R0-K1 + R0-K2 + R0-K6 + R1 + R2-A + R2-A-blind + R2-B + R2-self + R2-A_stopgrad + R2-A-random-NSA + R3 + Adaptive-Loops baseline = **13 cells** × 10 seeds = 130 cell-runs. At 340 H100-hours each: **44,200 H100-hours**.

This is **substantially over any reasonable academic budget**. Per the spec's instruction to flag intractable compute: **`flagged_intractable: true`**.

### 9.3 Reduced ladder — actually executable

The 130-cell ladder is a flagged maximalist plan. The minimum executable ladder, pre-registered:

| Stage | Cells × seeds | H100-hours |
|---|---|---|
| **Stage 0**: NSA-mini for F-Calib (350M, 8K, 5B tokens, 1 seed) | 1 × 1 | 100 |
| **Stage 1 (gate)**: R0 σ-measurement (R0 only, 10 seeds, Tier-1) | 1 × 10 | 3,400 |
| **Stage 2 (Path 2 from revision-1 §8)**: frozen-backbone fine-tune of R1, R2-A, R2-self, R2-A_stopgrad on a publicly released NSA model; 5 seeds each on RULER VT @32K subsample | 4 × 5 | 800 |
| **Stage 3 (escalation, gated on Stage 2 R2-A − R1 ≥ 0.5 EM)**: from-scratch core falsification cells (R0, R1, R2-A, R2-self, R2-A_stopgrad), 10 seeds each, Tier-1 | 5 × 10 | 17,000 |
| **Stage 4 (gated on Stage 3 surviving F1/F-Self)**: full ladder remaining cells (R2-A-blind, R2-B, R2-A-random-NSA, R3, F4 high-top-n cell, F2 NIAH-single cell, Adaptive-Loops baseline) | 7 × 10 | 23,800 |

**Stage 0 alone (~100 H100-hours, ~12 hours wall on 8 H100s)** can falsify the entire hypothesis via F-Calib. **Stage 0 + 2 ≈ 900 H100-hours** completes the cheapest falsification path and supplies the gate decision for from-scratch escalation.

Tier-2 (1.3B, 60B tokens) is **pre-registered as out-of-scope** unless Tier-1 produces a publishable positive result; at that point Tier-2 cost (~5× Tier-1 per cell) is justified to confirm scaling.

### 9.4 Cost summary

| Plan | H100-hours | $ at $3/hr | Verdict |
|---|---|---|---|
| Maximalist (full 130-cell ladder) | 44,200 | $132,600 | **flagged intractable** |
| Stage 0+2 (cheapest falsification) | 900 | $2,700 | Tractable |
| Stage 0+1+3 (core from-scratch falsification, gated) | 20,500 | $61,500 | Tractable for an industry lab |
| Stages 0+1+2+3+4 (ladder up to F4 + ablations) | 45,100 | $135,300 | At industry-lab scale; still flagged |

**Honest recommendation**: the staged design with Stage-2 as the gate is the intended path. Anything beyond Stage 3 is contingent on Stage 3's outcome, not pre-committed.

---

## 10. Risks to the experiment (data leakage, baseline-tuning asymmetry, evaluation-suite drift)

| Risk | What it would do to the result | Mitigation |
|---|---|---|
| **R-leak-1**: HotpotQA pilot for F-Calib threshold uses the same `supporting_facts` annotation layout as RULER's variable-tracking gold positions, but RULER's questions have *different* gold-block extents (variable position ≠ supporting paragraph). | F-Calib threshold could be miscalibrated — too lax on RULER. | Pre-register the threshold conservatively (HotpotQA fraction + 5pp) and report sensitivity analysis at +0pp, +5pp, +10pp. |
| **R-leak-2**: c-A regressor trained on full-attention pairs on the *training* corpus could overfit if any RULER-VT-style synthetic patterns leak into training (e.g., explicit "magic number"–style phrasing in fineweb-edu). | R2-A − R1 inflated by leak. | Run a string-match leak test: scan pretraining corpus for RULER's templating patterns ("hidden number", "remember the number"), exclude documents that match. Pre-register the exclusion. |
| **R-tuning-1**: R2-A might benefit from more hyperparameter sweeps than R1 because of the extra c_{t,k} machinery. | Asymmetric tuning effort masks like a real effect. | Pre-register **fixed hyperparameters across R1, R2-A, R2-self, R2-A_stopgrad** (lr, weight decay, batch size, warmup, β routing threshold). The c-A regressor's own hyperparameters (regressor depth, mixing rate p_full) are tuned only on Stage 0 NSA-mini measurements, frozen for Stage 3+. |
| **R-tuning-2**: NSA's own recipe (block size, top-n, sliding window) was tuned for dense-attention parity, not for joint training with MoR. We may be running NSA in a regime it was not designed for. | All R-cells underperform a hypothetical "NSA with co-tuned NSA-MoR config." | Pre-register the published NSA config (B=64, top-n=16, w=512) as the universal config. Report sensitivity to each at A9 — but don't tune for R2-A specifically. |
| **R-eval-drift-1**: RULER VT @32K is regenerated locally; randomness in haystack content could create test items easier or harder than the published distribution. | Comparison to other published RULER scores not directly possible; absolute EM numbers may differ. | Report the regeneration seed (set to 1 by convention). The 50/50 eval-A/eval-B split is internal; cross-cell comparisons within-experiment are unaffected. |
| **R-eval-drift-2**: FRAMES uses LLM-as-judge (GPT-4-class) for grading. The judge model evolves; re-running in 6 months produces different absolute EM. | Numbers not directly reproducible. | Pin the judge model and prompt template to the FRAMES paper §3.3 exactly. Record the judge-model version in the run record. Also report exact-string-match as a judge-free secondary metric. |
| **R-power-1**: σ_per_cell on RULER VT @32K at 350M may exceed 0.6 EM if the model is far from saturated, in which case 10 seeds is underpowered for F1. | False-negative on F1; we falsely conclude R2-A ≈ R1. | Pre-committed escalation: σ > 0.6 → 15 seeds; σ > 1.0 → halt and re-architect. |
| **R-stability**: I2 double-discrete routing collapses → all R-cells fail similarly, baselines look better. | Misleading negative result on the recipe entirely. | Stability protocol §3.4. |
| **R-implicit-extrapolation (red-team I4)**: SSA Appendix F shows compression-branch perplexity-extrapolation failure, not multi-hop EM failure. We're extending the claim. | If compression branch *does* cover non-selected info on multi-hop, R2-A ≈ R1 expected. | The §7.8 compression-branch attention probe directly tests this; reported as a diagnostic alongside F1. If the probe shows compression branch covers gold-evidence positions, F1 firing is the *expected* outcome, not a hypothesis weakness. |

---

## 11. Cheaper-falsification path (gate before from-scratch)

Per spec instruction to specify a cheap path:

1. **Stage 0 (~100 H100-hours)**: F-Calib on NSA-mini. If fails, terminate, contribute the F-Calib measurement.
2. **Stage 0.5 (~150 H100-hours)**: c-A regressor *correlation smoke test* — train ĉ_φ on a frozen-NSA's training data (no joint training), measure RMSE of ĉ_φ vs δ_true on a held-out full-attention sample. **Pre-registered gate**: if RMSE > 0.4 (i.e., the regressor cannot recover δ from features), R2-A's premise fails. Cost: tiny.
3. **Stage 2 (Path 2 from revision-1 §8, ~800 H100-hours)**: frozen-backbone fine-tune of R1, R2-A, R2-self, R2-A_stopgrad. Gate: if R2-A − R1 ≥ 0.5 EM, escalate to from-scratch.

Stages 0, 0.5, 2 together cost ~1,050 H100-hours = ~$3,150 — falsifies or escalates the full hypothesis at <3% of the maximalist budget.

---

## 12. Sources

| arxiv ID / dataset / repo | What | Used for |
|---|---|---|
| arXiv:2502.11089 | NSA | Backbone, 3-branch architecture, importance-score computation |
| arXiv:2507.10524 | MoR | Halting recipe (expert-choice routing, §2.2.1), KV caching strategies (§3) |
| arXiv:2511.20102 | SSA | Proposition 4.1 (Gradient Update Deficiency), Theorem 1, Appendix F (compression-branch perplexity extrapolation), 5% full-attention mixing recipe |
| arXiv:2510.01585 | ReSSFormer | R0 baseline calibration |
| arXiv:2603.08391 | Adaptive Loops | Mandatory dense-attention prior-art baseline (§4.1) |
| arXiv:2602.08426 | Prism | RoPE spectral features for c-A regressor |
| arXiv:2404.06654 | RULER | Primary benchmark; variable-tracking subtask |
| arXiv:2409.12941 | FRAMES | Secondary benchmark, judge protocol |
| arXiv:2108.00573 | MuSiQue | Secondary benchmark |
| arXiv:1809.09600 | HotpotQA | F-Calib pilot benchmark |
| `simonjegou/ruler` | HF dataset | RULER 4K/8K/16K mirror; 32K regenerated locally |
| `google/frames-benchmark` | HF dataset | FRAMES |
| `dgslibisey/MuSiQue` | HF dataset | MuSiQue |
| `hotpotqa/hotpot_qa` | HF dataset | HotpotQA distractor for F-Calib pilot |
| `HuggingFaceFW/fineweb-edu` | HF dataset | Pretraining corpus |
| `togethercomputer/RedPajama-Data-1T` | HF dataset | Long-context up-training (arxiv + wikipedia slices) |
| `https://github.com/NVIDIA/RULER` | GitHub repo (Apache-2.0) | RULER 32K generator |
| `https://github.com/fla-org/native-sparse-attention` | GitHub repo | NSA reference Triton implementation |
| arXiv:2602.07150 | Randomness in Agentic Evals | Noise-floor citation supporting 10-seed minimum |
| arXiv:2412.10319 | SCBench | M5 multi-hop sparse-attention robustness |
| arXiv:2504.17768 | Sparse Frontier | Risk E (scale frontier) |
