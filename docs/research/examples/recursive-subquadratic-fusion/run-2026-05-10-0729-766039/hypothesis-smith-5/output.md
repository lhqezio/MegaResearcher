# H5 — Latent Tunnel Vision: Architectural Recursion Inherits CoT Lock-In Under Sparse Attention via Spurious Fixed-Point Attractors

**Role:** hypothesis-smith-5
**Targeting gap:** `gap-finder-2/output.md` Gap #7 (B7) — "the (architectural recursion × Tunnel-Vision under SubQ) intersection is empty"
**Revision:** 0
**Polarity:** Architectural recursion INHERITS Tunnel Vision under sparse attention (strong-falsification axis).

---

## 1. Targeted gap

The fusion thesis pins architectural recursion (TRM-style, K passes within a single forward pass; arXiv:2510.04871) onto subquadratic-attention backbones for long-context reasoning. Gap-finder-2 #7 establishes that *no* prior work tests whether the **Tunnel-Vision** failure mode of Chain-of-Thought (CoT) — where early flawed reasoning irreversibly locks the model into a suboptimal trajectory (arXiv:2509.04475 §2.2) — has an architectural-recursion analogue, and whether that analogue compounds with sparse attention's known position-bias amplification (arXiv:2502.01951 Theorems 4.1–4.2; arXiv:2307.03172) and multi-matching super-linearity (arXiv:2410.04422). All published parallel-thinking remediations (arXiv:2509.07980; arXiv:2412.19707; arXiv:2504.07052) operate at the prompt/agent layer; none touches architectural recursion under subquadratic attention.

The conventional response — "Tunnel Vision is a discrete-token-commitment phenomenon, latent recursion has no committed token, therefore it cannot lock in" — is the comforting null. This hypothesis attacks that null. If correct, the fusion thesis must be reformulated; if wrong, the falsification yields a positive separation result that strengthens the rest of the program.

---

## 2. Hypothesis statement (if/then form)

**If** a TRM-style architecturally recursive operator (K passes of latent refinement on a fixed deep-supervision target; arXiv:2510.04871 §4.1) is composed with a subquadratic attention backbone (sparsity ratio s ∈ {0.1, 0.25, 0.5}; e.g., NSA arXiv:2502.11089 / Sparse-Frontier arXiv:2504.17768 patterns) and run on a position-controlled long-context reasoning stimulus that requires re-attending to mid-context evidence (license-clean NoLiMa-style construction, distractor-loaded multi-matching tasks), **then** for K ≥ K\* (a stimulus-and-sparsity-dependent threshold we predict ≈ 4–8), the recursive operator will lock into a **spurious latent fixed point** (arXiv:2601.10679 §4.4) that *increases* with K rather than decreases — operationally manifest as (a) a per-instance lock-in rate that rises super-linearly in K under fixed s; (b) a Lost-in-the-Middle U-curve (arXiv:2307.03172) whose mid-context trough deepens with K rather than flattening; (c) a multi-matching super-linearity exponent (arXiv:2410.04422) that grows with K rather than shrinking, with the growth rate scaling at least linearly with the sparsity ratio s.

The non-additive prediction: a multiplicative interaction term **s × K** with a positive coefficient that exceeds the sum of the marginal s-only and K-only effects. A purely additive failure-mode model is insufficient; the prediction is that sparsity *amplifies* recursive lock-in beyond what either factor alone would produce.

---

## 3. Mechanism

The mechanism rests on three cited claims that compose:

**M1 — Spurious-fixed-point lock-in is a documented latent-space failure mode of architectural recursion.** The HRM mechanistic critique (arXiv:2601.10679 §3.1, §4.4, §5) shows that the hierarchical recursive operator can converge to "false fixed points" — attractors in latent space corresponding to *wrong* outputs that the model lingers near "for many segments before finally leaping out" (§4.4) or "converges to a false fixed point (non-trivial failure mode)." TRM (arXiv:2510.04871 §4.1) explicitly drops the fixed-point assumption and uses deep supervision to push trajectories toward the correct solution, but does not eliminate the existence of attractors — it merely removes the *theoretical justification* for their use. The empirical question is whether deep-supervised TRM still *exhibits* spurious attractors at inference time when the input distribution shifts to long-context retrieval-conditioned reasoning, away from the fixed-context puzzle regime where TRM was demonstrated.

**M2 — Sparse attention exponentially amplifies position bias toward center-node tokens with depth, and this amplification has been *proven* to extend to sparse masks.** Position-Bias Emergence (arXiv:2502.01951 Theorem 4.1 for causal mask, Theorem 4.2 for sliding-window mask of width w) provides the load-bearing graph-theoretic argument: as iterative attention depth t → ∞, P^(t)(z_i = 1 | X^(0)) → 1 — every token's effective context collapses toward the first token, with sliding-window masks merely slowing the rate to ϵ^⌈(N−1)/(w−1)⌉ rather than eliminating the convergence. Architectural recursion *adds* iterative depth (each of K passes re-applies the attention operator), so under a sparse mask the per-token effective context contracts faster as a function of (s × K) than under either alone. This is the formal handle that yields the multiplicative interaction prediction.

**M3 — Score dilution and multi-matching super-linearity provide the long-context stress under which lock-in becomes observable.** Score Dilution at Test Time (arXiv:2512.13898) establishes that long-context attention scores dilute across the context, making single-pass retrieval lossy. Hyper-multi-step (arXiv:2410.04422) shows that multi-matching long-context retrieval is super-linearly hard. Long Context, Less Focus (arXiv:2602.15028) establishes attention dilution in fixed-capacity transformers. Under these conditions, a recursive operator with deep supervision trained on (or transferred from) a different distribution would, by M1, settle on whichever latent attractor is closest in the training distribution — and by M2, that attractor would be increasingly determined by sparsity-amplified center-node tokens (typically prompt prefix and early context) rather than mid-context evidence the task requires.

**Why this is qualitatively distinct from CoT Tunnel Vision (the spec requires this engagement).** CoT Tunnel Vision (arXiv:2509.04475 §2.2) operates in **discrete output space**: the model emits tokens; each token is sampled from a softmax; the next forward pass conditions on the committed token. The lock-in mechanism is "irreversible commit." Architectural recursion (arXiv:2510.04871 §4.1) operates in **continuous latent space**: each pass updates z without committing to discrete tokens. *On its face* this should escape Tunnel Vision — there is no commit-and-reinforce step. The hypothesis claims the escape is illusory because (i) the deep-supervision objective itself acts as an implicit commitment device — the operator is trained to make z_H closer to a solution at every supervision step, so an early wrong direction is reinforced over iterations; (ii) per M1, the operator can possess spurious attractors; (iii) per M2, sparse attention's center-node bias is the kind of architectural prior that biases *which* attractor is closest at initialization. The continuous-vs-discrete distinction does not protect against attractor-dynamics lock-in; it only protects against *one specific mechanism* (token-level sampling commitment) of lock-in.

If M1–M3 are correct and compose as stated, the predicted effect is mechanistically grounded. If empirical results contradict the prediction (e.g., K does not interact multiplicatively with s), then *which* of M1, M2, or M3 fails is itself diagnostic — and that diagnostic value justifies the experiment regardless of polarity.

---

## 4. Predicted outcome with magnitude

Let:
- **L(s, K)** = lock-in rate: fraction of instances on which the post-recursion answer differs from a corrected single-pass baseline trajectory (operationalization in §6).
- **U(s, K)** = U-curve depth: difference in accuracy between best end-position and worst mid-position on a position-controlled needle stimulus (Lost-in-the-Middle metric, arXiv:2307.03172).
- **α(s, K)** = super-linearity exponent of the multi-matching cost curve (arXiv:2410.04422), fit as accuracy ∝ N^{−α} where N is the number of needles.

**Predictions, each with magnitude and direction:**

1. **L(s, K) is super-linear in K under fixed s ≥ 0.5 (heavy sparsity).** Specifically: L(0.5, K=8) − L(0.5, K=1) ≥ +15 percentage points, with the K=8 lock-in rate exceeding 35% on the position-controlled stimulus. Under dense attention (s = 1.0, full attention), L(1.0, K=8) − L(1.0, K=1) ≤ +5 pp. The interaction term ∂²L/∂s∂K is positive and non-trivial.

2. **U(s, K) deepens with K under sparse attention, flattens or improves under dense.** Specifically: U(0.5, K=8) − U(0.5, K=1) ≥ +8 pp (mid-context trough deepens), while U(1.0, K=8) − U(1.0, K=1) ≤ +2 pp or is negative (recursion helps under dense, hurts under sparse). The fusion-thesis-favorable direction would be U(0.5, K=8) − U(0.5, K=1) < 0; this hypothesis predicts the opposite sign.

3. **α(s, K) grows with K under sparse attention.** Specifically: α(0.5, K=8) − α(0.5, K=1) ≥ +0.10 (super-linearity worsens by at least 0.10 in the exponent), while under dense α changes by less than half this magnitude.

4. **The K\* threshold scales with sparsity.** K\* (smallest K at which lock-in rate exceeds dense baseline by 10 pp) is predicted to be ≈ 4 at s = 0.25 and ≈ 2 at s = 0.5. At s = 0.1 (very heavy sparsity), K\* ≤ 2 — i.e., even shallow recursion is enough to lock in.

**Conditions under which the hypothesis should hold:**
- The stimulus requires re-attending to mid-context evidence (positions roughly between 30%–70% of context length) that is not literally string-matchable to the query — i.e., NoLiMa-style fact-chaining (arXiv:2502.05167) under a license-clean reconstruction.
- Context length ≥ 8K tokens (long enough for sparse-attention center-node bias to dominate, per arXiv:2502.01951 Theorem 4.2 rate dependence on N/w).
- Architectural recursion uses deep supervision (per TRM arXiv:2510.04871 §4.1); models without deep supervision are out of scope.

**Conditions under which the hypothesis should NOT hold (and would be falsified):**
- Short-context fixed-vocabulary puzzle tasks (Sudoku-Extreme, ARC-AGI) — TRM's demonstrated regime (arXiv:2510.04871 §5). The prediction is specifically about long-context retrieval-conditioned reasoning; if the predicted lock-in fails to manifest there, the hypothesis is wrong.
- Dense attention (s = 1.0) — the multiplicative interaction is the load-bearing claim.
- Stimuli where the correct answer is dominated by content in the first 10% or last 10% of the context (where center-node bias accidentally aligns with truth).

---

## 5. Falsification criteria (each with metric, threshold, direction)

A hypothesis is unfalsifiable unless specific results would refute it. The following four criteria each, individually, falsify the hypothesis if observed:

**F1 — No multiplicative interaction.** Metric: ∂²L/∂s∂K estimated by 2x2 factorial regression over {s ∈ {0.5, 1.0}} × {K ∈ {1, 8}}. Threshold: if the interaction coefficient has 95% confidence interval crossing zero, OR is positive but smaller than +5 pp, the multiplicative-interaction prediction is falsified. Direction: prediction is positive interaction ≥ +10 pp; threshold for falsification is < +5 pp.

**F2 — Recursion monotonically improves U-curve under sparse attention.** Metric: U(0.5, K=8) − U(0.5, K=1). Threshold: if the difference is ≤ 0 (recursion *flattens* the mid-context trough rather than deepening it) on the position-controlled stimulus, the U-curve prediction is falsified. Direction: prediction is ≥ +8 pp deepening; falsification is any non-positive value.

**F3 — Super-linearity exponent does not grow with K under sparse attention.** Metric: α(0.5, K=8) − α(0.5, K=1). Threshold: if Δα ≤ +0.02 (essentially no growth), the multi-matching prediction is falsified. Direction: prediction is ≥ +0.10; falsification is ≤ +0.02.

**F4 — Lock-in does not manifest at any sparsity at K = 8.** Metric: max over s ∈ {0.1, 0.25, 0.5} of L(s, 8) − L(s, 1). Threshold: if this max is ≤ +5 pp, lock-in is not architecturally amplified by recursion under any tested sparsity, and the core mechanism claim (M1 + M2 composition) is falsified. Direction: prediction is at least one s for which the increment is ≥ +15 pp.

Any single criterion (F1–F4) failing individually constrains the hypothesis. F1 + F4 both failing falsifies it cleanly. The hypothesis is in trouble only if at least F1 *and* F4 hold (i.e., positive interaction *and* lock-in manifests at some sparsity).

**Anti-falsifiability hedge prevention.** I commit in advance: if all four criteria fail, the hypothesis is wrong — and that is the spec's strong-falsification axis paying off, because the result "architectural recursion does NOT inherit Tunnel Vision under sparse attention" is a positive contribution to the fusion thesis. The hypothesis is asymmetrically valuable: confirmation reshapes the fusion thesis; refutation strengthens it.

---

## 6. Required experiments (sketch — eval-designer details)

**Stimulus construction.**
- License-clean reconstruction of a NoLiMa-style position-controlled needle stimulus (NoLiMa itself is non-commercial; arXiv:2502.05167 is for grounding only, not for redistribution). Construction: synthetic biographies with attribute chains where the query requires a 2-hop fact link (e.g., "Who lives in the city where Alice's mother was born?") and the surface forms of the needle do not lexically match the query — eliminates the literal-matching shortcut.
- Position grid: needle placed at 8 positions ∈ {5%, 15%, 30%, 45%, 55%, 70%, 85%, 95%} of context length.
- Multi-needle variant: N ∈ {1, 2, 4, 8, 16} simultaneous needles for the α-exponent measurement (Hyper-multi-step protocol, arXiv:2410.04422).
- Context lengths: {8K, 32K, 128K} tokens.

**Models & ablations (factorial design).**
- **Backbone × Sparsity:** dense baseline (s = 1.0); NSA-trained sparse (arXiv:2502.11089) at s ∈ {0.5, 0.25, 0.1}; Sparse-Frontier patterns Vertical-Slash / Block-Sparse (arXiv:2504.17768) for cross-pattern check. At least one independently-pretrained sparse-attention checkpoint per s-level — *not* post-hoc sparsified — to avoid distribution-shift confound.
- **Recursion × Depth:** TRM-style architectural recursion at K ∈ {1, 2, 4, 8, 16}, with deep supervision (per arXiv:2510.04871 §4.1, §2.4).
- **Critical control: CoT-only baseline at matched FLOPs.** For each K, the matched-FLOPs CoT control runs sequential CoT generation under the *same* backbone with the same total compute — this lets us separate "lock-in caused by recursion" from "lock-in caused by spending more compute under sparse attention." Without this control, F1 cannot be cleanly attributed.
- **Critical control: dense backbone, K-varied recursion.** Tests whether recursion *alone* (without sparse attention) produces lock-in — calibrates whether the multiplicative interaction is real or whether recursion is the only driver.

**Operationalization of L(s, K) (lock-in rate).** For each instance and each (s, K), record the K=1 latent-decoded answer, then run K passes; lock-in is observed when the K-pass answer differs from the K=1 *correct* answer, or persistently differs across K ∈ {2, 4, 8} (i.e., the recursion is stuck in a wrong attractor). Triangulate against (a) trajectory-divergence: KL between layer-wise hidden-state distributions at K=1 vs K=8 on wrong-answered instances; (b) attractor-basin probe: re-initialize z with random perturbation at K=4 and check whether K=8 returns to the same wrong answer (true attractor behavior, per arXiv:2601.10679 §4.4 protocol).

**Diagnostic decomposition.** If F1 falsifies (no multiplicative interaction), but F2/F3 confirm, the failure may be due to deep supervision *not* propagating wrongness — diagnostic for whether TRM's training-regime escape extends to long-context. If F4 fails at small s (s = 0.1) but holds at s = 0.5, the threshold K\* is sparsity-dependent, partially confirming. The experimental design supports these decompositions, not just the binary verdict.

---

## 7. Risks to the hypothesis

**Risk 1 — Deep supervision dominates.** TRM's deep-supervision objective (arXiv:2510.04871 §2.4, §4.1) is specifically designed to push z_H toward correct y at every supervision step. If the long-context training distribution is rich enough, the supervised gradient could overwhelm any spurious attractor formed by sparse attention's center-node bias. *Even if this risk materializes*, the hypothesis still contributes: the experiment characterizes the *training-regime conditions* under which deep supervision suffices, which is itself uncharted (arXiv:2601.10679 only tests fixed-context puzzles).

**Risk 2 — TRM-style recursion at long context simply does not work at all.** TRM was demonstrated on Sudoku, Maze, ARC-AGI — all fixed-context. Section 4.5 of arXiv:2510.04871 specifically uses an attention-free architecture for fixed small contexts. A recursive operator transferred to long-context may simply degrade to noise at K > 1, in which case the lock-in measurement is meaningless because the model never produces stable answers. *Even if this risk materializes*, the result "TRM-style recursion fails to transfer to long context" is itself a load-bearing finding for the fusion thesis, falsifying the broader hypothesis-1 thesis as well.

**Risk 3 — The interaction is positive but the wrong direction is K-helping under sparsity (recursion compensates for sparsity loss).** It is plausible that recursion *amortizes* sparse attention's information loss by re-reading — gap-finder #7 itself notes this competing direction. If U(0.5, K=8) − U(0.5, K=1) is meaningfully *negative* (mid-context retrieval improves with recursion under sparse), the hypothesis is falsified, but the resulting evidence directly strengthens the fusion thesis. *Even if this risk materializes*, the experiment delivers a positive separation result that is precisely the spec's success condition.

**Risk 4 — Operationalization confound: lock-in vs convergence.** A K-pass that produces the same wrong answer across K ∈ {2, 4, 8} could be either spurious attractor lock-in (the hypothesis) or the model correctly predicting that the input is unanswerable. Without ground-truth answerability annotation, lock-in could be over-counted. *Mitigation:* the attractor-basin probe (perturb-and-rerun) operationalizes the *attractor* claim distinctly from the *wrong-answer* claim, so this confound is testable.

**Risk 5 — Sparsity-pattern dependence.** Position-Bias Emergence (arXiv:2502.01951) Theorems 4.2, 4.3 show different masks (sliding-window, prefix) yield different center-node structures. NSA, MoBA, Quest, Vertical-Slash, Block-Sparse all have different patterns. The hypothesis predicts the multiplicative interaction holds for sparsity *in general*, but it might hold for some patterns and not others. *Even if this risk materializes*, identifying which sparse-attention patterns are recursion-safe is itself an architectural-coherence finding that constrains the fusion thesis design space.

---

## 8. Sources

| Citation | arxiv ID | Role in this hypothesis |
|---|---|---|
| Less is More: Recursive Reasoning with Tiny Networks (TRM) | 2510.04871 | Defines architectural recursion with deep supervision (§2.4, §4.1); the operator under test |
| Are Your Reasoning Models Reasoning or Guessing? Mechanistic Analysis of HRM | 2601.10679 | Documents spurious latent fixed points (§4.4, §5) — load-bearing for M1 |
| ParaThinker / Tunnel Vision | 2509.04475 | Defines Tunnel Vision in CoT (§2.2); the failure mode whose analogue we test |
| Lost in the Middle | 2307.03172 | U-curve metric (M3, F2) |
| Hyper-multi-step | 2410.04422 | Super-linearity exponent metric (M3, F3) |
| Score Dilution at Test Time | 2512.13898 | Long-context attention dilution (M3) |
| Long Context, Less Focus | 2602.15028 | Attention dilution mechanism (M3) |
| Position-Bias Emergence | 2502.01951 | Graph-theoretic position-bias proofs for sparse masks (M2; Theorems 4.1–4.3) |
| NoLiMa | 2502.05167 | Position-controlled stimulus design (license-flagged: reconstruct, do not redistribute) |
| Native Sparse Attention (NSA) | 2502.11089 | Subquadratic-attention backbone candidate |
| The Sparse Frontier | 2504.17768 | Sparsity patterns × tasks taxonomy; Vertical-Slash, Block-Sparse |
| Parallel-R1 | 2509.07980 | Confirms parallel-thinking remediations operate at prompt layer, not architectural |
| Thought Rollback | 2412.19707 | Same — parallel/backtracking at prompt layer only |
| To Backtrack or Not to Backtrack | 2504.07052 | Same — limits of sequential search at prompt layer |
