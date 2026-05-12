# Red-team critique of H6-VAR (Stacking TRM-style depthwise recursion on TTT-style sequence-time recursion causes a variance-amplifying phase)

Critiquing: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-6/output.md`
Targeting gap: A6 (gap-finder-1) + B8 reference. Revision round: 1 (initial).

---

## 1. Verdict

`REJECT (revision-1)`

The hypothesis sits on a defensible gap (TRM × TTT joint instantiation has not been published and the verification queries confirm this), and it has the right "non-additive prediction" shape. But the critique below identifies (a) a load-bearing citation that *contradicts* rather than supports the M3 mechanism, (b) two F-criteria that are statistically vacuous at the proposed sample size, (c) an unspecified architectural construction that makes the whole experiment ambiguous, and (d) a primary metric (across-seed std) whose noise envelope at 5 seeds × 800-item benchmark × per-d sub-split swamps the predicted effect. None of these is fatal — they are all addressable in revision — but in their current form the hypothesis cannot be defended publicly.

---

## 2. Gap re-verification

I ran four independent literature queries with phrasing distinct from the gap-finder's:

| Query | Result count | Closest hit | Composes TRM-depth × TTT-sequence? |
|---|---|---|---|
| `TRM TTT depthwise recursion fast weights composition` | 10 | 2510.04871 (TRM), 2505.23884 (LaCT), 2603.06642 (SR-TTT), 2604.06169 (In-Place TTT), 2511.16886 (Deep Improvement Supervision for TRM) | None |
| `test-time training looped transformer outer loop nested recursion` | 10 | 2310.13807 (Learning to Learn at Test Time — actually a *classification-domain* nested-loop predecessor), 2602.11698 (SpiralFormer multi-resolution loops), 2502.17416 (Reasoning with Latent Thoughts) | None |
| `CRUXEval program recursion depth code reasoning small model` | 10 | 2510.04871, 2401.03065, 2511.08653 (CGAR), 2511.16886 | None |
| `TTT TRM HRM nested recursion test-time training fast weights destructive interference` | 10 | 2604.19295 (TEMPO — *training-time* TTT for reasoning models, RL flavor), 2602.21204 (TTT-as-Linear-Attention) | None |

**Verdict on the gap:** The gap claim survives. None of the verification queries surfaced a paper that simultaneously instantiates TRM-style depthwise weight-tied recursion AND TTT-style sequence-time fast-weight recursion with cross-axis ablation. Two new-to-the-bibliography candidates appeared (SR-TTT 2603.06642 and In-Place TTT 2604.06169, both from after 2025-09 cutoff but they are NOT depth-recursion papers — they are about TTT-internal sparsification and inference-time MLP adaptation respectively). The gap is genuine.

`gap_claim_survives: true`

---

## 3. Citation spot-checks

### 3.1 TTT (arXiv:2407.04620) §2.1, Figure 4 — claim: "W_t doesn't reach a fixed point within T=2048"

**What the smith claims.** "The authors explicitly show (arXiv:2407.04620 Figure 4) that for a 125M-parameter TTT-Linear network the per-token TTT loss `ℓ(W_t; x_t)` is **monotonically decreasing but does not reach a fixed point within sequence length T=2048** — `W_t` is a moving target throughout the forward pass."

**What §2.1 / Figure 4 actually says** (verified via `read_paper`): "As shown in Figure 4, gradient descent is able to reduce ℓ, but cannot reduce it to zero." The figure shows the inner-loop loss `ℓ(W_t; x_t)` averaged over 2048-length test sequences for the first three TTT layers in a 125M model.

**Assessment.** The smith is *correct in spirit* but the conflation between "loss does not reach zero" and "weights do not reach a fixed point" is sloppy. A fixed point of the SGD update `W_t = W_{t-1} − η∇ℓ` requires `∇ℓ = 0` — which would imply zero loss only if the loss is non-negative and minimized at zero (the squared-reconstruction loss is). So the conclusion is right, but the smith is treating Figure 4 as direct evidence of weight non-stationarity when it is direct evidence of loss-decrease-without-saturation. **This is an Important issue, not Critical.** Tighten the citation in revision.

### 3.2 TTT-as-Linear-Attention (arXiv:2602.21204) — claim used in F6 control: "η=0 ⇒ TTT layer = static linear attention"

**What the smith claims.** §2.6 of the H6 output: "the η=0 / static-fast-weight equivalence used in F6 control."

**What 2602.21204 actually says** (verified by reading §3 and §4.1-4.4): The paper argues TTT-KVB *is* learned linear attention **regardless of η** — not specifically at η=0. More damaging:
- **§4.1 finding:** "despite improved inner-loop fitting, downstream performance degrades consistently as the number of inner-loop steps increases. This inverse relationship holds across both LLMs and novel view synthesis (NVS) tasks. Such behavior directly contradicts the memorization-based interpretation."
- **§4.2 finding:** TTT works *comparably with gradient ascent* in the inner loop — i.e., flipping the sign of the inner-loop update does not destroy task performance.

**Assessment — CRITICAL.** This is the most serious finding in this critique:
1. The smith's M3b mechanism — "if at large η or many inner steps the TTT inner loop converges *too* fast on iteration k's inputs, then iteration k+1 sees a `W_T^(k)` that has memorized iteration k's now-stale latent" — is *already at work in published TTT* without any TRM wrapping. 2602.21204 §4.1 directly shows that more inner-loop fitting hurts downstream performance, at K_arch=1. This means the K_arch=1 baseline std cited by the smith (1.5 abs pts) may be a substantial *under*estimate, AND the M3b regime is not specifically a TRM × TTT interaction — it is a property of TTT alone.
2. The smith's M3a mechanism rests on the inner loop "memorizing" the iteration-k input embedding sequence so that outer-iteration k+1's reads are stale. But 2602.21204 §4.2-4.4 argues TTT does NOT operate by memorization at all — gradient ascent and Q-replaced-with-K both leave performance comparable. The "memorize-and-stale-read" mental model is exactly what 2602.21204 falsifies. The M3a mechanism is therefore not grounded; if TTT is learned linear attention, then the iterations refining `W_t` may be doing something other than storing key-value associations to be read by an outer loop.

The smith cites 2602.21204 only for the η=0 → linear-attention equivalence (a trivial corollary), but does not engage with the paper's main claim — which directly attacks both M3 sub-mechanisms.

### 3.3 TRM (arXiv:2510.04871) §2.4, §4.1, §4.2 — claim: TRM back-propagates through the full final recursion

**What the smith claims.** "TRM back-propagates through the full final recursion." Smith cites §4.1 for the "n evaluations of f_L on (z_L+z_H+x) followed by one evaluation of f_H on (z_L+z_H), repeated T times outside of gradients" pattern.

**What §4.1 actually says** (verified): TRM runs `T-1` recursion processes WITHOUT gradient, then ONE recursion process WITH backpropagation through n evaluations of f_L and 1 of f_H. Section 4.4 emphasizes "Less is more" — TRM is a 2-layer, 5M-parameter model on Sudoku-Extreme/Maze/ARC.

**Assessment.** The smith's description is technically correct but elides a crucial scale issue: **TRM was specifically designed for tiny networks on small puzzle tasks.** TRM Table 1 shows their best result is 5M parameters with `T=3, n=6`. Section 4.4 ("Less is more") explicitly argues that *adding* more parameters HURT TRM's generalization on Sudoku-Extreme. The smith proposes wrapping K_arch ∈ {1,2,4,8,16} TRM-style outer iterations around a 125M-parameter TTT/Mamba/dense backbone — this is **not** what TRM did, and TRM's own paper is some evidence that scaling the inner block to 125M may break the deep-supervision regime entirely. **This is an Important issue.** The proposed architecture is not TRM-faithful at the parameter scale chosen.

---

## 4. Mechanism critique

### M1 (TTT non-stationarity along sequence). Partially supported.
Figure 4 of TTT shows monotonically decreasing loss not reaching zero — sufficient to argue `W_t` is not stationary. But "non-stationary" ≠ "non-stationary in a way that interferes with outer-loop reads." The smith does not argue *why* the specific non-stationarity should be destructive (vs e.g. simply slow). Citation 2602.21204 §4.2 (gradient ascent works) actively suggests the inner-loop update is doing something *other* than what the smith's interference story assumes.

### M2 (TRM outer loop reads inner state K times). Mostly correct on the operator side.
But ignores that TRM's design specifically uses small networks; the construction's behavior at 125M is uncharted. The smith's "iteration k+1 provides a *different* input embedding sequence (because TRM updates `(y, z)` between iterations)" is right, but the construction in TRM is on a 4-layer transformer with deep supervision — not on a TTT layer that already maintains internal fast weights along the sequence axis. **The compute graph is not specified.** When iteration k+1 starts, do the TTT inner-loop fast weights `W_t` reset to the initial `W_0` (in which case the inner loop replays from scratch with new `(y,z)` inputs)? Or do they persist from iteration k (in which case the *inner-loop trajectory* is being twice-conditioned on iteration-k inputs and once on iteration-k+1 inputs, in an unusual hybrid)? The mechanism story (M3a vs M3b) presumes one of these but does not say which. Critical for revision.

### M3a (stale fast-weight reads). Plausible but unfounded by direct empirical precedent.
The "Bai et al. 2019 fixed-point gradient theory" reference is invoked correctly — TRM (§2.3, §4.1) explicitly *replaces* this approximation with full backprop precisely because the fixed-point assumption is unsafe. So at the construction level, TRM has already abandoned the assumption that the smith identifies as violated by TTT. **But** TRM did so for the OUTER-loop fixed-point of `(z_L, z_H)`, not for any inner-loop fixed-point of TTT's `W_t`. The smith conflates these. The mechanism therefore needs to be re-stated as: "TRM does deep supervision through the full inner trajectory, but because TTT's `W_t` is itself an SGD trajectory along the sequence, the deep-supervision gradients flow through `∇W_t` for every t — making the outer-loop gradient noise compound with the inner-loop noise." This is a different mechanism than "stale reads" and the smith should pick one.

### M3b (over-fitted fast weights). Contradicted by 2602.21204 §4.1.
2602.21204 §4.1 directly says "more inner-loop fitting hurts downstream performance" — at K_arch=1 baseline TTT. So the M3b regime is *the regular operating regime of TTT alone*, not a TRM-induced destructive regime. The hypothesis predicts M3b emerges at *large K_arch*, but 2602.21204 shows it emerges already at K_arch=1 with more inner steps. This makes the predicted variance peak shape (low at K=1, peak at K=2-4, low at K=16) inconsistent with the mechanism: if M3b is already active at K=1, why would it be *less* active there?

### M4 (CRUXEval is the right test surface). Concerns about per-d cohort size.
CRUXEval = 800 items. AST-extracted program-recursion-depth `d` partitions the 800 items into 5 buckets {1, 2, 3, 4, ≥5}. CRUXEval's actual depth distribution is heavily skewed shallow (3-13 line functions; deep recursion is rare). Per-d cohort size at d=3 may be ~100-200 items — fine for mean accuracy, but *across-seed std* of accuracy on a 100-item subset at 25-40% accuracy band has substantial variance; a back-of-envelope binomial std on 150 items at 35% is √(0.35·0.65/150) ≈ 3.9 percentage points *within-seed*. The across-seed std can easily be 1-3 abs points just from sub-sampling noise. The smith's K=1 anchor of σ ≈ 1.5 is plausible but the threshold "σ at K=4 ≥ 6.0 abs points" sits in a region where binomial-finite-sample noise dominates the signal. This is a Critical statistical-design issue.

### M5 (non-additivity). The argument is genuine but the smith should engage 2602.21204 §4.1 directly.
The K=1 TTT std anchor depends on TTT alone being "stable across seeds." But 2602.21204 §4.1 shows the relationship is non-monotone in *inner-loop steps* alone. Without controlling for inner-loop steps, the K=1 std could already be much larger than 1.5 abs points.

---

## 5. Falsifiability assessment

| Criterion | Operationalizable? | Statistical power at 5 seeds × 800-item × d=3 split? | Verdict |
|---|---|---|---|
| F1 (σ(TTT,K=4) < 2.0× σ(TTT,K=1)) | Yes — concrete | **Insufficient.** Std-of-std at 5 seeds has CoV ≈ 1/√(2(N-1)) ≈ 0.35; ratio of two stds at this CoV cannot reliably distinguish 2.0× from 1.0× | **Falsifiable in principle, statistically vacuous in practice** |
| F2 (interaction term ≥ +1.5 abs pts) | Yes — concrete | Worse than F1 — interaction = (σ−σ) − (σ−σ) is sum of 4 noisy estimates; std-noise floor at this sample size is ≈1.5-2.0 abs pts. Threshold is at the noise floor. | **Statistically near-vacuous** |
| F3 (non-monotone curve, σ(K=4) − ½(σ(K=1) + σ(K=16)) < +1.0) | Yes | Same issue as F1; std-of-std at 5 seeds is too noisy | **Statistically vacuous at proposed N** |
| F4 (depth-axis specificity, σ(d≥4) − σ(d=1) < +1.5) | Yes | d=1 cohort and d≥4 cohort are even smaller sub-splits; per-cell std estimates are noisier; threshold near noise floor | **Weak power** |
| F5 (CoT-confound control) | Yes — clean conceptual control | Same finite-sample issues, plus CoT condition itself adds variance in non-uniform ways across difficulty | **Operationalizable, weakly powered** |
| F6 (η=0 frozen TTT control, in §6 ablations) | Yes | At η=0, TTT degenerates to a known operator (linear attention per 2602.21204) — clean test | **Strong, recommend promoting from ablation to F-criterion** |

**Net.** F1-F5 all fail my "statistically operationalizable into a finite experiment" bar at the proposed N=5 seeds × 800-item × d=3 split. The hypothesis is *falsifiable in principle* but *not falsifiable in practice* with the proposed budget. **Critical.**

A pre-registered sample-size analysis is mentioned in R3 — but the smith treats this as a "hypothesis is unfinished" outcome, not as a Phase-5 design constraint. It needs to be a hard pre-registration: target std-of-std power ≥ 0.8 to detect a 2.0× ratio, which probably requires ≥15 seeds per cell, or use of full CRUXEval-X (15K items) instead of CRUXEval-O (800 items).

---

## 6. Strongest counter-argument (steelman)

The strongest opposing case: **TRM-depth and TTT-sequence are simply orthogonal axes that compose ADDITIVELY** with no destructive interaction.

The case rests on three observations:

1. **TTT is learned linear attention (2602.21204).** Under the linear-attention reinterpretation, TTT's "fast weights" `W_t` are just a particular parameterization of an attention operator with a structured update rule. Linear attention is a stationary operator at the layer level; its sequence-time updates do not create a "moving target" any more than dense softmax attention's `KV-cache` does. Wrapping this in TRM-style outer iterations is *exactly* what TRM does on dense softmax — just with a different attention parameterization. The non-additive interaction the smith predicts collapses into the standard "depthwise-loop-wrapping-attention" phenomenology, which TRM Table 1 shows is *positive* (74.7% with self-attention vs 87.4% with TRM's 2-layer-no-attention baseline).

2. **TRM's ablation table (Table 1) shows that adding self-attention to TRM HURTS** (87.4% no-attention vs 74.7% with-attention at the same parameter count). The TRM authors interpret this as "less is more" — depth-of-recursion substitutes for attention expressiveness. If the smith's mechanism were right, adding self-attention should produce variance, not just lower mean. TRM's own ablations show *lower mean, stable variance* — which is the additive-not-destructive picture.

3. **MoR (2507.10524) and Huginn (2502.05171) successfully wrap depth-recursion around large dense backbones** without reporting destructive variance. If destructive interference were a generic depth-recursion-on-non-trivial-attention phenomenon, MoR and Huginn would have reported it. They report the opposite — clean per-token depth halting and adaptive recursion at scale.

The steelman concludes: the variance peak the smith predicts may simply not exist, and the K_arch curve for TRM × TTT may look like the K_arch curve for TRM × Mamba (monotone non-decreasing mean, flat variance), with the only difference being that TTT's mean is somewhat lower (per 2602.21204 §4.1's "more inner-loop fitting hurts" finding). In that case, F1-F5 all return null and the hypothesis returns a clean negative result — but at 3000-4000 GPU-hours, a null result is a poor return on the swarm's compute budget.

The smith partially acknowledges this in R4 ("the two recursions may genuinely *not interact at all*"), but does not engage 2602.21204's specific empirical contradictions. Critically, R4's mitigation — "the experimental scaffolding is reusable" — is not a strong enough payoff if the prediction is wrong.

---

## 7. Severity-tagged objections

### Critical (must fix before APPROVE)

**C1. Citation 2602.21204 directly contradicts the M3 mechanism, not just supports F6.** §4.1 of 2602.21204 shows "more inner-loop fitting hurts downstream performance" at K_arch=1 — meaning M3b is already active in K=1 baseline TTT, undermining the predicted shape of the variance curve. §4.2 shows gradient ascent works comparably — undermining the "memorize-and-stale-read" picture in M3a. Smith must engage these findings, not just cite the η=0 corollary.

**C2. F1-F5 are statistically near-vacuous at 5 seeds × 800-item × per-d sub-split.** Std-of-std CoV ≈ 0.35 at 5 seeds; binomial-finite-sample-noise at d=3 cohort (~150 items, 35% accuracy) ≈ 3.9 abs pts within-seed; F1's 2.0× ratio threshold sits below this noise floor. Pre-registered sample-size analysis must yield ≥15 seeds per cell, or move to CRUXEval-X (15K items), or both.

**C3. The "wrap TRM around full TTT backbone" architecture is unspecified at a load-bearing junction.** When outer-iteration k+1 begins, do the inner TTT fast weights `W_t` reset to `W_0` or persist from iteration k? Mechanism M3a vs M3b depends on this choice. Smith must commit to one variant and explain why the other isn't tested (or test both as an ablation).

### Important (should fix)

**I1. The 125M scale × 5M-parameter-TRM mismatch.** TRM is explicitly a 5M-parameter, 2-layer construction; TRM Table 1 / §4.4 ("Less is more") argues that scaling beyond this *hurts* generalization. Wrapping TRM-style K outer iterations around a 125M backbone is not what TRM did. Either justify the scale extrapolation or rescope to ≤30M parameters (where R1 already concedes the prediction may not hold).

**I2. Citation slip on Figure 4 of 2407.04620.** The cited figure shows loss-not-reaching-zero, not weight-not-reaching-fixed-point. They are related but not identical. Tighten the citation.

**I3. Cohort size at per-d sub-split.** CRUXEval's d=3 cohort size is not estimated; smith should provide AST-extracted depth-distribution before pre-registration. If d=3 has <100 items, primary metric becomes unreliable.

**I4. Mamba is not a clean control — and the smith already knows.** R5 acknowledges this. The η=0 frozen-TTT control listed in §6 ablations is a stronger baseline; promote it from "ablation" to a primary-control alongside Mamba.

**I5. The "destructive-interference" framing was chosen because it produces a non-additive prediction, not because mechanism reasoning forces it.** Smith openly states this. This is honest but means the hypothesis is more in the form "let's see if there's a non-additive phenomenon" than "mechanism X predicts non-additivity Y." Strengthen by making the *positive* mechanism story (M3a or M3b, picked) more concrete and load-bearing.

### Suggestion (nice to have)

**S1. The cheaper falsification path (§6a) tests something different from the full pre-training prediction.** Test-time wrapping of frozen checkpoints with random projection variation only varies the wrapper init seed; it does not exercise the training-time deep-supervision gradient flow that the M3 mechanism is about. The §6a result is at best weakly correlated with the §6 prediction. Consider re-scoping §6a to: "fine-tune K_arch=4 TRM wrapper on a frozen TTT backbone for 1B tokens × 3 seeds vs same on Mamba" — closer to the training-time story.

**S2. Add a within-seed bootstrap CI for std estimates.** At N=5 seeds, std point estimates are unstable; report bootstrap 95% CI in addition to point estimate, and pre-register acceptance criterion in CI terms.

**S3. R2's "bimodal seed distribution" caveat is interesting and underexplored.** If bimodality is the failure mode, std is not the right metric — multimodality test (Hartigan dip test) is. Worth pre-registering.

---

## 8. Recommendation to hypothesis-smith

This hypothesis can be saved in a single revision round. The gap is real, the non-additivity framing is publishable, and the test surface (CRUXEval × per-d) is reasonable. But three things must change:

1. **Engage 2602.21204 §4.1-4.4 directly.** The "TTT is secretly linear attention" paper has empirical findings that *attack* the M3 mechanism. Either (a) revise M3 to be compatible — e.g., reformulate as "noise-amplifying" rather than "memorization-stale-read" — or (b) preregister a clean K_arch=1 TTT baseline experiment (which 2602.21204 §4.1 implies will already show high variance/non-monotonicity in inner steps) and use that as the proper "additive" baseline against which non-additivity is measured. Currently, the smith's K=1 std anchor of 1.5 abs pts is uncalibrated to 2602.21204's findings.

2. **Make the F-criteria statistically powered.** Either ≥15 seeds per cell (probably ≥6,000 GPU-hours, well above the 2,000-hour fence and would require eval-designer to scope this down hard), OR run on CRUXEval-X (15K items) so per-d cohort is large enough for std-of-std to converge, OR pre-register that the work targets *the existence of a non-monotone shape* (testable by 5-point regression-on-K) rather than per-cell variance ratios. The latter is statistically much more efficient.

3. **Specify the architecture.** Pick: "inner TTT fast weights reset between outer iterations" or "persist." Both are valid TRM-on-TTT constructions; pick one for the primary experiment, mention the other in ablations. The mechanism story needs to commit.

If these revisions land, I'd APPROVE in round 2. If after revision the smith does not engage 2602.21204's findings or the architecture remains ambiguous, KILL becomes the right call — the gap is real but the hypothesis-as-stated cannot be tested.

---

`APPROVE` | `REJECT (revision-1)` | `KILL (irrecoverable)`

**Verdict:** REJECT (revision-1)
