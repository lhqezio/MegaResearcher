# Recursive Reasoning on Subquadratic-Attention Backbones — Synthesist Output

**Run id:** `2026-05-10-0729-766039`
**Spec:** `docs/research/specs/2026-05-10-recursive-subquadratic-fusion-spec.md`
**Plan:** `docs/research/plans/2026-05-10-recursive-subquadratic-fusion-plan.md`
**Novelty target:** `hypothesis`
**Surviving hypotheses:** 5 (H1, H3, H4, H5, H6)
**Killed hypotheses:** 1 (H2 — preserved in section 4)
**Total worker invocations:** 5 scouts + 2 gap-finders + 6 hypothesis-smiths × 2 rounds + 6 red-team × 2 rounds + 5 eval-designers + 1 synthesist = 38

---

## 1. Problem framing and the (depth × context) plane

The fusion thesis is whether **architectural recursion** — a small operator applied recursively to its own output for K steps within a single forward pass, in the style of TRM (arXiv:2510.04871) — can be productively layered on **subquadratic-attention backbones** in the style of NSA (arXiv:2502.11089), MoBA (arXiv:2502.13189), DSA / DeepSeek-V3.2 (arXiv:2512.02556), and SubQ (Subquadratic, 2026, **industrial blog, no peer-reviewed paper** — flag preserved throughout the run), and whether the formal regime of Gupta–Huang–Saha–Xu–Ye (arXiv:2505.14840) constrains where this fusion can deliver depth-of-reasoning AND sub-quadratic context scaling jointly.

Five literature-scout subagents surveyed 158 entries. Mapping them onto a (reasoning-depth × context-length) plane:

- **Depth-only axis (recursion lineage).** TRM (arXiv:2510.04871), HRM (arXiv:2506.21734), Universal Transformer (arXiv:1807.03819), Huginn / depth-recurrent transformer (arXiv:2502.05171; 3.5B), Ouro / LoopLM (arXiv:2510.25741; up to 2.6B), Mixture-of-Recursions / MoR (arXiv:2507.10524; 0.5–7B), Parallel Loop Transformer / PLT (arXiv:2510.24824), Reasoning with Latent Thoughts (arXiv:2502.17416), Looped Transformers Better at Algorithms (arXiv:2311.12424), Iso-Depth Scaling Laws (arXiv:2604.21106), Relaxed Recursive Transformers (arXiv:2410.20672), Retrofitted Recurrence (arXiv:2511.07384), Think-at-Hard (arXiv:2511.08577), Adaptive Loops in Transformers (arXiv:2603.08391), HRM mechanistic critique (arXiv:2601.10679), Parcae spectral-radius scaling laws (arXiv:2604.12946), Huginn Coda-Lens probe (arXiv:2507.02199). All of these except PLT use **dense attention**.
- **Context-only axis (subquadratic-attention lineage).** NSA (compressed + selected + sliding three-branch design; arXiv:2502.11089), MoBA (top-k block gate, no fallback; arXiv:2502.13189), DSA / DeepSeek-V3.2 (lightning-indexer top-k, no compressed branch; arXiv:2512.02556), SeerAttention-R (arXiv:2506.08889), Sparse Frontier (arXiv:2504.17768), Quest (arXiv:2406.10774), Longformer / BigBird / Reformer / StreamingLLM (lineage), the Gupta et al. formal subquadratic regime (arXiv:2505.14840), SSA (arXiv:2511.20102 — Gradient Update Deficiency Proposition 4.1, dropped-attention-mass Theorem 1, compression-branch-fails-to-extrapolate Appendix F). SubQ (2026 industrial blog) is treated as one architectural data point and **not** a product comparison.
- **State-space / linear-attention adjacents.** Mamba / Mamba-2 / Longhorn (arXiv:2407.14207), RWKV-7, RetNet, Hyena, Based, GLA, TTT (arXiv:2407.04620), LaCT (arXiv:2505.23884), TTT-as-linear-attention (arXiv:2602.21204 §5.1 Theorem 5.1), Fixed-Point RNNs (arXiv:2503.10799), Score Dilution at Test Time (arXiv:2512.13898), Long Context Less Focus (arXiv:2602.15028).
- **Joint-axis prior art (the fusion's near-neighbours).** PLT (arXiv:2510.24824) is the *only* scouted paper coupling loop recursion with a subquadratic pattern (gated sliding-window, G-SWA; +6.1 average accuracy lift over a 680M-active dense-MoE non-loop baseline, Table 2). Fixed-Point RNNs (arXiv:2503.10799) iterates a Mamba transition mixer to a Banach fixed point — already a form of recursion on a linear-RNN substrate, and the closest theoretical neighbour. Superlinear Multi-Step Attention (arXiv:2601.18401) builds an O(L^(1+1/N)) multi-step subquadratic architecture but does not test out-of-distribution long-context. ReSSFormer (arXiv:2510.01585) combines fixed-K recursion with sparse top-k routing on multi-hop QA but lacks per-token depth halting.

**The (depth × context) plane is therefore almost empty.** The only published cell occupants are PLT (loops × G-SWA, dense-MoE backbone, general benchmarks), FPR (fixed-point solver-iteration on Mamba mixer, A_5/S_5 state tracking), and ReSSFormer (fixed-K outer recursion with sparse top-k). No paper instantiates **TRM-style architectural recursion** on **NSA / MoBA / DSA / SeerAttention-R**; no paper measures retrieval-head retention as a function of (sparse pattern × K_arch); no paper jointly trains per-token depth halting with per-token sparse routing; no paper composes TRM-style depth recursion with TTT-style sequence-time recursion.

**Architectural-recursion-vs-agent-recursion distinction (load-bearing, reasserted).** Throughout this run, "recursion" means a learned operator applied to its own latent output for K steps within one forward pass — no token emission between iterations, no external scaffold. This excludes Recursive Language Models / RLM (arXiv:2512.24601, programmatic recursion via tool-use), SRLM (arXiv:2603.15653), chain-of-thought as recursion, best-of-N, and tool-using agents that call themselves. Hypothesis H3's mechanism (M3) explicitly maps TRM's `(x, y, z)` recurrence onto a long-context transformer as **output-conditioned re-feed without token commit**; H1, H4, H5, H6 use weight-tied iteration of a backbone block at deep supervision. The synthesist preserves this distinction because the spec's YAGNI fence is uncompromising about it.

The two gap-finders condensed the 158-paper landscape into **19 architecture-side and architecture × evaluation gaps**, of which six were composed into hypotheses (H1–H6) and 13 were retained for the audit trail (section 4).

---

## 2. Surviving hypotheses with falsification criteria

Five hypotheses survived the round-2 red-team approval. Each is summarised with its claimed gap, mechanism in one paragraph, predicted outcome with magnitude, and falsification criteria stated **verbatim** from the approved revision-1 documents (metric + threshold + direction).

### H1 — Compressed-Summary Fallback as a Sufficient Recursion Substrate

**Claimed gap.** Composed gaps A1+B1+B4 (`gaps.md#H1`). No published paper instantiates TRM-style depthwise recursion on a *natively-trainable, learned-routing* sparse-attention backbone (NSA, MoBA, DSA, SeerAttention-R). PLT (arXiv:2510.24824) is the closest precedent and uses G-SWA (a fixed-pattern sliding-window partial fallback), not learned routing. The hypothesis-smith verified the empty cell across `hf_papers` queries on NSA + recursion + BABILong and MoBA + recursion + multi-hop.

**Mechanism (one paragraph).** NSA maintains three branches gated together (compressed/selected/sliding-window; arXiv:2502.11089 §3); MoBA drops below-top-k blocks entirely with no fallback channel (arXiv:2502.13189 §3); DSA's lightning-indexer top-k has no compressed branch (arXiv:2512.02556 §2). Under TRM-style weight-tied recursion (arXiv:2510.04871 §3), iteration *t+1*'s refined query may need to attend to evidence dropped by iteration *t*'s top-k. NSA permits this via re-weighting compressed-branch summary tokens (a low-rate-but-nonzero channel for un-selected blocks); MoBA does not. PLT's positive +6.1 lift on G-SWA (arXiv:2510.24824 Table 2, row PLT-3) confirms that *some* fallback channel is sufficient for productive sparse-recursion. The compression branch summarises blocks of size 32 (~2000 compressed tokens at L=64K); whether this granularity preserves enough fine-grained content for multi-hop fact retrieval is the empirical question. Comparison is FLOP-matched per the iso-depth recipe (arXiv:2410.20672 / arXiv:2604.21106; eval-designer-1 corrected the rev-1 mis-citation of Ouro §4).

**Predicted outcome with magnitude.** At 1B parameters, BABILong qa3+qa4+qa5 averaged at L=64K, FLOP-matched K=1 vs K=6 baselines:

| Variant | K=1 | K=6 | Δ (recursion lift) |
|---|---|---|---|
| Dense + recursion | A_d | A_d + 4 to 7 | +4 to +7 |
| **NSA-with-fallback + recursion** | A_n ≈ A_d | **A_n + 6 to 10** | **+6 to +10** |
| NSA-no-fallback (compressed zeroed) | A_n − 2 | A_n − 2 ± 2 | ≤ +2 |
| MoBA + recursion | A_m ≈ A_n ± 2 | A_m + 0 ± 3 | ≤ +3 |

**Falsification criteria (verbatim).**
- **F1 (necessary).** `(NSA-fb K=6) − (NSA-fb K=1) ≥ +6.0` absolute points on BABILong qa3+qa4+qa5 at L=64K, 1B params, FLOP-matched. Direction: if observed lift `< +3.0` points, falsified.
- **F2 (central, sufficient — non-additive DiD with sign asymmetry).** `[(NSA-fb K=6) − (NSA-fb K=1)] − [(MoBA K=6) − (MoBA K=1)] ≥ +5.0` AND `(NSA-fb K=6 − K=1) > 0` AND `(MoBA K=6 − K=1) ≤ 0` AND K=1 calibration `|NSA-fb K=1 − MoBA K=1| ≤ 2`. Falsified if DiD `< +2` OR same-sign positive both sides. If K=1 calibration fails (gap > 2), test is **inconclusive**, not falsified.
- **F3 (causal role of the compressed branch).** `(NSA-fb K=6) − (NSA-no-fb K=6) ≥ +4.0`. Falsified if Δ ≤ +1.5.
- **F4' (mechanistic Jaccard probe).** `Jaccard_NSA-fb(I_1, I_6) ≤ Jaccard_MoBA(I_1, I_6) − 0.10` on selected blocks at L=64K on qa3+qa4+qa5 (replaces the original logit-probe F4 that red-team-1 round-1 correctly identified as inoperable in last-block-only recursion).

**Red-team approval rationale (round 2).** All 5 critical and 6 serious round-1 objections substantively addressed. The non-additive DiD with sign-asymmetry constraint is the central falsifier and forecloses the additive-marginal failure mode. PLT mechanism heterogeneity (G-SWA per-loop-KV vs NSA cross-iteration compressed summary) is acknowledged but not load-bearing for F2. One Important issue (Ouro §4 citation mis-specification for the FLOP-match protocol) was identified and corrected in eval-designer-1.

### H3 — Architectural Recursion Plus an All-Tokens Fallback Channel Produces Non-Additive Recovery of Long-Context Retrieval

**Claimed gap.** A8 + B9. Retrieval Head (arXiv:2404.15574) shows long-context factuality is mediated by a small intrinsic set of attention heads. No paper has measured retrieval-head retention as a function of (sparse-attention pattern × architectural-recursion depth K).

**Mechanism (one paragraph).** Mechanism decomposed into M2a (compression-signal channel: NSA's compressed branch passes a coarse signal — block-size 32, per-needle SNR ≈ 1/32 — from every block, so refined-query iterations can re-weight it) + M2b (query-refinement channel: output-conditioned re-feed appends pass-k draft answer to the input, so pass-(k+1) layer queries are computed on a different residual stream, re-running the selection step regardless of whether a compression channel exists). NSA gets M2a + M2b; Quest (arXiv:2406.10774) and DSA (arXiv:2512.02556) get M2b only with finer (Quest) or coarser granularity than MoBA's block-mean-pool gate. M3 maps TRM's `(x, y, z)` (arXiv:2510.04871 §3) onto a long-context transformer: `x` = original prompt, `y` = pass-k draft answer (32–128 tokens), `z` = final-layer hidden states at the answer-span positions, forward computation = `full_forward(concat(x, "\nAnswer-so-far: ", y_prev))`. R3 (smearing under recursion, citing Huginn arXiv:2502.05171 and Fixed-Point RNNs arXiv:2503.10799) is engaged in mechanism — the differential prediction (F2) survives symmetric smearing; only asymmetric smearing falsifies.

**Predicted outcome with magnitude.** 1.3B parameter scale, NoLiMa at 32K context, K ∈ {1, 2, 4, 8} via output-conditioned re-feed:

| Architecture | K=1 retention | K=4 retention | Recovery |
|---|---|---|---|
| Dense (control) | 1.0 | 0.95–1.0 | ≈ 0 |
| **NSA (M2a + M2b)** | 1.0 | **0.92–1.0** | **+0.0 to +0.08** |
| Quest (M2b only) | 1.0 | 0.85–0.95 | −0.05 to +0.05 |
| DSA (M2b only) | 1.0 | 0.83–0.95 | −0.07 to +0.05 |
| MoBA (weakest M2b) | 1.0 | 0.80–0.90 | −0.10 to +0.0 |

**Falsification criteria (verbatim).**
- **F2 (PRIMARY differential, non-additive).** `[retention_NSA(K=4) − retention_NSA(K=1)] − [retention_MoBA(K=4) − retention_MoBA(K=1)] ≥ +0.05` absolute, within-architecture (each architecture's own H_A re-identified per arXiv:2404.15574 §2). Direction: less than +0.05 falsifies.
- **F5 (ordering).** Rank order at K=4 across {NSA, Quest, DSA, MoBA} matches `NSA > Quest ≥ DSA > MoBA` with at most one swap.
- **F3 (downstream task transfer).** On NoLiMa at 32K, `(NSA Δacc K=4 vs K=1) − (MoBA Δacc K=4 vs K=1) ≥ +3` percentage points; on RULER NIAH-13 ≥ +2 pp.
- **F1 (consistency check, demoted from primary).** `retention_NSA(K=4) − retention_NSA(K=1) ≥ −0.10` (recursion must not catastrophically destroy NSA's retrieval heads).
- **F6 (mechanism check).** With NSA's compression branch zeroed (`g_cmp = 0`), the differential should collapse to MoBA-class within ±0.03; failing this falsifies M2a as load-bearing.

**Red-team approval rationale.** All 4 critical objections (cheap kill-test inconsistent with mechanism, head-index alignment across architectures, Quest-prediction–mechanism mismatch, TRM puzzle-shape mapping unspecified) addressed. Differential-as-primary survives R3 smearing. Output-conditioned re-feed as the cheap-test recursion is mechanism-faithful (the cheap test now exercises M3) at the price of being a weaker form of recursion than published TRM (no separate latent z-stream). Honest acknowledgement that R5 — "output-conditioned re-feed too weak" — is a real risk that would convert the result to a YAGNI-bound negative.

### H4 — Joint Halting × Sparse Routing: Dropped-Mass-Conditioned MoR on NSA

**Claimed gap.** A4 (gap-finder-1 #4). Per-token *depth* routing (ACT, PonderNet, Universal Transformer, MoR arXiv:2507.10524, LoopFormer) and per-token *attention* sparse routing (NSA, MoBA, DSA) both produce per-token decisions on the same hidden state, but no paper jointly trains them. ReSSFormer (arXiv:2510.01585) has fixed-K + sparse top-k but no per-token depth halting; Adaptive Loops (arXiv:2603.08391) has per-token halting but on dense.

**Mechanism (one paragraph).** SSA (arXiv:2511.20102) Proposition 4.1 establishes a "Gradient Update Deficiency": tokens in dropped blocks receive zero gradient through NSA's selection branch during training, so the importance estimator `p_t^slc` never learns counterfactual relevance and is therefore *structurally* miscalibrated on multi-hop QA. Theorem 1 quantifies the consequence: `‖h_full − h_sparse‖ ≤ δ(t)·(...)` where δ(t) = dropped full-attention mass. NSA's compression branch, per SSA Appendix F, "fails to extrapolate to full attention" (perplexity 191.3 when isolated) — it carries summary information, not the discriminative positional/entity signal multi-hop QA needs. MoR's expert-choice gate (arXiv:2507.10524 §2.2.1, `g_t^r = σ(θ_r^T H_t^r)`) inherits this miscalibration. The fix: condition the depth gate on a non-self-justifying proxy `c_{t,k}` for δ(t,k), trained as a regressor `ĉ_φ(features) → δ̂(t,k)` supervised by full-attention dropped mass on a 5%-mixture training regime (revision-1 §3 M3 c-A operationalisation).

**Predicted outcome with magnitude.** 1B-parameter NSA + MoR with K_max = 4, RULER multi-hop variable-tracking primary at 32K, ≥10 seeds, paired bootstrap n=10000 α=0.05 two-sided:

| Regime | RULER multi-hop ΔEM vs R0 |
|---|---|
| R0 (no halting, K=4 fixed; ≈ ReSSFormer-with-NSA) | baseline |
| R1 (independent) | ±0.5 EM |
| **R2-A (c-A, δ-proxy regressor)** | **+2.0 to +4.0 EM** |
| R2-B (c-B, JSD gate-disagreement) | +1.0 to +2.5 EM |
| R2-self (revision-0 self-coverage form, predicted null) | +0.0 to +1.0 EM |

**Falsification criteria (verbatim).**
- **F-Calib (NEW pre-experiment gate, mandatory, < 24 GPU-hours).** Median rank of gold-evidence block under `p_t^slc` on RULER-multi-hop dev > NSA's top-n on ≥30% of multi-hop questions. If this fails, miscalibration claim is null and hypothesis is dead pre-experiment.
- **F1 (non-additivity).** R2-A − R1 EM > 1.5 on RULER-multi-hop primary cell (paired bootstrap, ≥10 seeds, std ≤ 0.6 EM). Falsified if R2-A does not beat R1 by >1.5 EM.
- **F2 (sparsity-dependence).** R2-A − R0 EM gap on single-hop NIAH (calibrated to 60–90% baseline) ≥ 1.5 EM (note: prediction is direction "*not greater than* 1.5"; the spec is that R2-A wins on multi-hop but not on single-hop — a violation in direction "greater" falsifies). Falsified if R2-A wins on tasks where the mechanism predicts no win.
- **F3 (signal-decoding).** `|w_c|/std(w_c) ≥ 1.5` for the fitted scalar weight on `c_{t,k}` in R2-A's depth-gate; falsified below.
- **F4 (collapse under dense selection).** When NSA top-n is set so ≥80% of blocks are always selected (δ ≈ 0), R2-A − R1 EM ≥ 1.5 falsifies (the gap should *not* survive δ → 0).
- **F-Self vs A.** R2-A − R2-self EM > 1.0; falsified below.
- **F-StopGrad.** `|R2-A − R2-A_stopgrad| > 0.7 EM`; if equal, the strong-interaction framing (M4) is falsified but the feature-engineered scalar still helps (partial-falsification of M4-strong, hypothesis still contributes).

**Red-team approval rationale.** All 3 critical objections addressed: (C1) self-justifying coverage signal replaced with δ-proxy regressor supervised by full-attention dropped mass; (C2) thresholds raised to ≥1.5 EM with ≥10 seeds and primary benchmark moved to RULER multi-hop variable-tracking; (C3) recipe conflation eliminated — TRM-style + PonderNet chimera replaced with MoR's published expert-choice routing. F-Calib is a mandatory pre-experiment gate that can kill the hypothesis cheaply.

### H5 — Latent Plateau, Not Lock-In: Architectural Recursion Escapes Tunnel Vision Under Sparse Attention but Fails to Scale on OOD Long Context

**POLARITY FLIPPED from revision-0** (lock-in → plateau) per round-1 red-team C2/C3 (deep-supervision-as-commitment-device misread; HRM-attractor mechanism does not transfer to TRM).

**Claimed gap.** B7. Tunnel Vision (arXiv:2509.04475 §2.2) is a discrete-token-commitment failure inapplicable to architectural recursion (no token commit between passes). HRM mechanistic critique (arXiv:2601.10679 §4.4) documents spurious-fixed-point attractors specific to HRM's fixed-point assumption + 1-step IFT gradient — TRM (arXiv:2510.04871 §4.1) explicitly removes both. The empty cell: how does a deep-supervision-trained TRM-style operator behave when transferred from training-distribution puzzles to long-context retrieval-conditioned reasoning under a subquadratic-attention backbone?

**Mechanism (one paragraph).** M1: deep-supervision-trained recursive operators contract toward a fixed point under bounded spectral radius (Parcae arXiv:2604.12946 §3, ρ(A̅) < 1); empirical evidence from Huginn (arXiv:2507.02199 §3.4: scaling 4 → 32 recurrent steps yields 3.11 → 4.93 GSM8K, plateau). M2: under weight-tied recursion + sparse mask, A^t converges by Perron-Frobenius to its dominant eigenvector — a content-blind first-token-sink direction (arXiv:2502.01951 Theorems 4.1–4.2 graph-theoretic position-bias under causal/sliding-window masks). M3: long-context out-of-distribution stimulus is the regime where M1+M2 manifest as plateau — iterates have no training-distribution-aligned target to converge toward, so they contract toward the content-blind direction, producing a uniform signal that the LM head cannot use to disambiguate mid-context evidence. The hypothesis explicitly **escapes** Tunnel Vision (no token commit → no irreversible poisoning) but suffers a milder, distinct failure (plateau).

**Predicted outcome with magnitude.** 200 instances per (sparsity × K × position) cell. Δ(s, K) = accuracy(s, K=8) − accuracy(s, K=1) on long-context OOD stimulus; Δ_train(K=8) on Sudoku-Extreme.

- Δ(s=0.5, K=8) ∈ (−3, +3) pp (statistically null).
- Δ_train(K=8) ≥ +20 pp (positive control).
- Δ(s=0.5, 8) − Δ(s=1.0, 8) ≤ −5 pp (sparse-vs-dense interaction).

**Falsification criteria (verbatim).**
- **F1 (recursion HELPS).** Δ(s=0.5, K=8) ≥ +10 pp falsifies plateau in the recursion-helps direction.
- **F2 (recursion HURTS).** Δ(s=0.5, K=8) ≤ −10 pp falsifies in the lock-in direction (revision-0 framing was right).
- **F3 (no sparse-vs-dense interaction).** `|Δ(s=0.5, 8) − Δ(s=1.0, 8)| ≤ 5 pp` falsifies the load-bearing M2 amplification claim (eval-designer-5 widened to ±10 pp per red-team I-B power calibration).
- **F4 (positive-control gate).** Δ_train(K=8) ≤ +10 pp on Sudoku-Extreme — operator is broken; experiment is uninterpretable; not a falsification of H5 but a kill of the experimental setup.
- **TOST equivalence commitment.** |Δ(s=0.5, K=8)| ≤ 3 pp with TOST p < 0.05 against ±5 pp equivalence margin (the plateau prediction is a positive statistical claim that can also fail).

**Red-team approval rationale.** Polarity flip is a substantive concession to round-1 critique (revision-0's commitment-device framing misread arXiv:2510.04871 §4.1; HRM-attractor mechanism does not transfer to TRM). Mechanism is now positively grounded in Huginn's plateau evidence and Parcae's contraction argument rather than chosen for shape. F1 and F2 are symmetric falsifications in opposite directions — there is no rhetorical escape direction in which "the result is informative either way."

### H6 — TRM and TTT are Sub-additive Substitutes (TRM's K_arch-gain is Compressed on a TTT Backbone)

**MECHANISM PIVOTED from revision-0** (destructive interference / memorize-and-stale-read → sub-additive redundancy) per round-1 red-team C1 (memorization mechanism contradicted by arXiv:2602.21204 §4.1–4.4).

**Claimed gap.** A6 + B8. TTT (arXiv:2407.04620), LaCT (arXiv:2505.23884), TTT-as-linear-attention (arXiv:2602.21204), Longhorn (arXiv:2407.14207) and depth-time recursion (TRM, HRM, Huginn, Ouro, MoR) have never been instantiated jointly, ablated against each other, or characterised for redundancy / complementarity / destructive interference. SR-TTT (arXiv:2603.06642) and In-Place TTT (arXiv:2604.06169) do not compose depth recursion with TTT.

**Mechanism (one paragraph).** TTT-as-linear-attention (arXiv:2602.21204 §5.1 Theorem 5.1) reinterprets TTT as a learnable mapping that induces a linear-attention operator with effective `q̂ = ϕ_{t+1}(q)`, `k̂ = ϕ_t(k)`, `v̂ = g_t(k)`; the inner-loop hyperparameters determine *which* attention operator is induced. TRM's outer loop (arXiv:2510.04871 §4.1) is iterative refinement of the same backbone-induced operator. When the backbone is TTT-Linear, both axes refine the same effective object — a learned-linear-attention operator over slightly-different `(y, z)` inputs — and the composition is sub-additive (saturation of a shared substrate). The frozen-η=0 TTT control (per arXiv:2602.21204 Theorem 5.1 corollary) isolates the "learnable inner-loop iteration" property: at η=0 the inner loop degenerates to a static operator, and TRM's K_arch-gain should match Mamba/dense gains. Eval-designer-6 carried forward a red-team Important issue (η=0 actually collapses to a position-wise MLP, not static linear attention) and added a true Fixed Linear Attention (FLA-stationary) backbone alongside η=0 so F2 has the correct distinguisher.

**Predicted outcome with magnitude.** 125M parameters (TTT paper anchor scale, arXiv:2407.04620), CRUXEval-X (arXiv:2408.13001) at AST-extracted program-recursion-depth d ≥ 3, no-CoT, ≥5 seeds:

| Backbone | K=1 mean | K=4 mean | ΔA(K=4 − K=1) |
|---|---|---|---|
| **TTT-Linear** | 28 ± 1.5 | 30 ± 2.0 | **+2.0** |
| TTT-η=0 (frozen) | 26 ± 1.5 | 31 ± 2.0 | +5.0 |
| Mamba | 27 ± 1.5 | 32 ± 2.0 | +5.0 |
| Dense softmax | 30 ± 1.5 | 36 ± 2.0 | +6.0 |

Predicted compression ratio r = ΔA_TTT / max(non-TTT) ≈ 0.33.

**Falsification criteria (verbatim).**
- **F1 (primary, compression ratio).** r > 0.8 OR r ≤ 0.0 with mean direction (TTT actively hurt) falsifies. Predicted band r ∈ (0.0, 0.5]; r ∈ (0.5, 0.8) is ambiguous → 10-seed escalation.
- **F2 (mechanism distinguisher).** `|A_TTT-Linear(K=4) − A_TTT-η=0(K=4)| ≤ 1.5` absolute points; falsified if > 1.5. (Predicted ≈ 1.0.)
- **F3 (CoT confound).** `r_CoT − r_no-CoT ≥ +0.2`; falsified if `< +0.2` or sign-flip.
- **F4 (depth-axis specificity).** Compression must grow with d: `r_{d≥3} ≤ r_{d=1}` AND `r_{d=1} ≥ 0.7`.
- **F5 (LaCT replication).** `ΔA_LaCT(K=4) − ΔA_TTT-Linear(K=4) ≤ +2.0`; falsified above.

**Red-team approval rationale.** All 3 critical objections substantively addressed. The mechanism is now grounded in arXiv:2602.21204's published reinterpretation rather than constructed for shape. Variance-amplification metrics from revision-0 demoted to secondary diagnostics with explicit power analysis. CRUXEval-X (15K items × 19 languages) is the right surface for the load-bearing first-moment test; CRUXEval-O at 800 items single-language is secondary.

---

## 3. Per-hypothesis eval designs

| Hypothesis | Primary dataset | OOD dataset | Baselines | Primary metrics | Key ablation | Stats plan | Full GPU-hr | Cheaper falsification path |
|---|---|---|---|---|---|---|---|---|
| **H1** | BABILong qa3+qa4+qa5 @64K (`RMT-team/babilong-1k-samples`, Apache-2.0) | RULER NIAH (`simonjegou/ruler`, Apache-2.0); GSM8K + MBPP short-context recursion control | NSA-no-fb K=6 (training-aware); MoBA K=6; Dense K=1 FLOP-matched non-shared deeper; PLT-G-SWA arm; DSA inference-time recursion; majority-class floor | DiD (NSA-fb, MoBA recursion lifts); F4' Jaccard inter-iteration drift on selected blocks | Compressed-branch zeroed at train+eval (F3) | 5 seeds × 8 backbone-runs × 50B tokens; per-cell std calibrated to ≤0.6 EM; paired bootstrap | **1395 GPU-hr full** | **300 GPU-hr** (NSA-only training-aware F3 ablation at 350M + DSA bonus prediction on open weights). NOT flagged intractable. |
| **H3** | NoLiMa @32K (`amodaresi/NoLiMa`, **CC-BY-NC-4.0 flagged**) | RULER NIAH-13 (Apache-2.0); Needle Threading (`jonathan-roberts1/needle-threading`); EverMemBench-S if released | Dense; NSA-no-cmp; G&A distillation (competing baseline); Quest post-hoc | Within-arch retrieval-head retention (per-arch H_A re-identified); NoLiMa Δacc | Compression branch zeroed (F6); output-replay vs re-feed (mechanism check b) | 5 seeds; differential SE 0.024; +0.05 threshold ~2.1σ; 95% CI on differential | **16,000 GPU-hr full (intractable)** | **1200 GPU-hr** post-hoc swap on dense backbone (training-free output-conditioned re-feed at 1.3B). **flagged_intractable: true; cheap path is the recommended primary.** |
| **H4** | RULER multi-hop variable-tracking @32K (`simonjegou/ruler` regenerated to 32K via NVIDIA generator, Apache-2.0) | FRAMES (`google/frames-benchmark`, Apache-2.0); MuSiQue@32K (`dgslibisey/MuSiQue`, CC-BY-4.0); HotpotQA-distractor @16K | Dense+MoR (Adaptive Loops, arXiv:2603.08391); NSA no-recursion (K=K_max); ReSSFormer-with-NSA (R0); NSA+MoR K=1 (degenerate); R2-self (revision-0 form); R2-B (JSD); shared-router R3 | EM with paired bootstrap; F-Calib gold-evidence-block rank | c-A vs c-B vs c-self vs gradient-stopped (F-StopGrad); random-selection sanity; coverage-source ablation (selection / compression / combined) | ≥10 seeds; std budget ≤0.6 EM; paired bootstrap n=10000 α=0.05 two-sided; F-Calib pre-experiment gate <24 GPU-hr | **44,200 GPU-hr maximalist 130-cell ladder (intractable)** | **1050 GPU-hr** staged 350M Tier-1 ladder; F-Calib gate (24 GPU-hr) → frozen-backbone fine-tune (~1 GPU-week) → Tier-1 from-scratch only if R2-A − R1 ≥ 0.5 EM. **flagged_intractable: true.** |
| **H5** | Constructed Biographical 2-Hop (CB2H, in-house **CC-BY-4.0**, license-clean replacement for NoLiMa) | BABILong qa3/qa5/qa15/qa18 @ {8K, 32K, 128K} (Apache-2.0); Sudoku-Extreme positive control (`sapientinc/sudoku-extreme`) | NSA at s=0.5; MoBA at s=0.5; Sparse-Frontier Vertical-Slash; Dense at s=1.0; matched-FLOPs CoT control at K=8 | Δ(s, K=8) accuracy delta; TOST equivalence (Lakens 2017; Schuirmann 1987); U-curve depth | K-grid {1,2,4,8}; matched-FLOPs CoT vs architectural; attractor-basin probe (perturb latent z) | 200 instances/cell; per-cell SE 3.5 pp; difference SE 5 pp; TOST p<0.05 against ±5 pp; one-sided F1/F2 | **9920 GPU-hr full** | **2500 GPU-hr** (drop 128K tier; reduce backbone factorial). NOT flagged intractable. |
| **H6** | CRUXEval-X 19-lang @ d≥3 cohort (`xhwl/cruxeval-x`, MIT) | CRUXEval-O 800-item Python (`cruxeval-org/cruxeval`, MIT) | TTT-η=0 (frozen); FLA-stationary (true linear attention, post red-team I2); Mamba; Dense softmax; LaCT for F5 replication | Compression ratio r = ΔA_TTT / max(non-TTT); A_TTT-Linear(K=4) − A_TTT-η=0(K=4) | Inner-loop reset vs persist between outer iterations; full-backprop-through-K_arch vs gradient-through-last-process-only; CoT-helpful sub-split | ≥5 seeds; hierarchical (problem-clustered) bootstrap (effective N at d≥3 ≈ 80–150 base problems); 10-seed escalation in r ∈ (0.5, 0.8) ambiguity zone | **8336 GPU-hr full Phase A+B+C (intractable)** | **150 GPU-hr** in-fence cheap-path-A (single fine-tune-and-evaluate; 4 backbones × 2 K-values × 3 seeds × 2B tokens at 125M); **insufficient** for the predicted r=0.33 vs threshold 0.5 distinction. **flagged_intractable: true.** |

**Pre-registration discipline.** All eval designs commit thresholds, sample sizes, and analysis plans before data collection. H1 and H4 explicitly state inconclusive zones (calibration failure → re-run, not falsify). H5 commits to TOST equivalence so the null prediction is positively claimable. H6 pre-registers a 10-seed escalation if r lands in the (0.5, 0.8) ambiguity band.

**Total swarm compute envelope.** Maximalist sum across all five surviving hypotheses ≈ 79.8K GPU-hours (driven by H4's 44.2K-cell ladder and H3's 16K from-scratch substrate). Cheap-path sum ≈ 5.2K GPU-hours; H3, H4, and H6 explicitly recommend executing the cheap path as the primary route.

---

## 4. Killed-hypothesis audit trail

Every hypothesis killed or dropped during this run is preserved here per the spec's audit-trail requirement.

### H2 — TC⁰ Escape via TRM-Recursion on SSMs (KILLED — escalation per cap-3 rule)

- **Title.** Iteration-scope and parameterisation differences between TRM-style and Fixed-Point-RNN-style depth-recurrence on a Mamba backbone, evaluated on long-tactic-chain Lean proof search, gated by a synthetic permutation-composition probe.
- **Claimed gap.** A2 + A7 + B5. No published work compares block-level depth-recurrence (TRM-style; arXiv:2510.04871) and solver-iteration on the transition mixer (FPR-style; arXiv:2503.10799) on the same Mamba substrate under the same training recipe, evaluated on formal proof search (PutnamBench arXiv:2407.11214, miniF2F arXiv:2109.00110, CoqGym arXiv:1905.09381).
- **Mechanism attempted (revision-0 → revision-1).** Revision-0: contraction-bound argument that FPR cannot represent permutation composition / TC⁰-escape via CoT-Solves-Serial extrapolation to constant-K depth iteration. Revision-1: parameterisation-and-iteration-scope comparison under matched compute, gated by a synthetic permutation-composition probe at L ∈ {5, 10, 50, 200}; Lean prediction conditional on probe.
- **Round-1 red-team objections.** C1 — FPR misrepresented (the contraction is on the depth-iteration *solver*, not on the function class; FPR-Mamba demonstrably solves S_5 per arXiv:2503.10799 §4 Fig 4). C2 — CoT-Solves-Serial does not generalise to constant-K depth iteration (K passes of a TC⁰ block at fixed K stays in TC⁰). C3 — Frozen Mamba + TRM wrapper is unprecedented training recipe. C4 — Magnitude calibration extrapolated from regular-language to Lean.
- **Round-2 red-team objections.** Round-1 C1 and C2 were *honestly retracted* (rare and admirable). However, fixing C3 introduced **a new critical defect**: Retrofitted Recurrence (arXiv:2511.07384) was cited as the "published recipe" for the unprecedented training protocol, but that paper is the Geiping/Huginn-style Prelude→RecurrentBlock→Coda single-state iteration applied to *Transformer* substrates (TinyLlama, OLMo, Llama-3.2). It does not study TRM-style two-state z_L/z_H with deep supervision, does not study FPR-style transition-mixer-only Banach-fixed-point iteration, and does not study Mamba substrates. Applying Retrofitted Recurrence's "broad recipe" to either novel recursion modality on Mamba is **doubly novel** (new modality + new substrate), and the smith's claim "this is no longer a novel training recipe — it is the published recipe re-applied to a Mamba substrate" was misleading. Plus three Important issues: I1 — the L=200 magnitude threshold of ≥10 absolute points is uncalibrated against any prior; FPR §D.2 only publishes train-length-16 numbers up to L=50 (not 200); I2 — chain-length × variant ratio prediction is operationally fragile under base-rate saturation/floor effects; I3 — F6 substrate-control fallback ("if intractable") makes the criterion partially unfalsifiable.
- **Verdict.** REJECT (revision-2). The hypothesis was not killed for being scientifically wrong — the gap survives, the synthetic-probe gate is the right shape — but the smith committed the *same shape of citation/precedent error* in two consecutive rounds: round 1 retracted FPR contraction-bound and CoT-Solves-Serial claims (papers that did not establish what was claimed); round 2 misrepresented Retrofitted Recurrence as a TRM-style precedent. Continuing into round 3 with high probability of similar shape of error.
- **One-sentence kill reason.** Repeated citation/precedent errors across two rounds: revision-0's TRM-vs-FPR architectural distinction and revision-1's training-recipe precedent both rely on papers that do not establish what is claimed; killed for audit trail per cap-3 rule with the spec's ≥3-surviving-hypotheses minimum already exceeded by the other five.
- **Lesson contributed.** When a hypothesis depends on a *novel architectural distinction* whose mechanism is supported only by adjacent prior art, the citation discipline of the smith on the load-bearing claims is a leading indicator of viability. Two strikes of the same shape is a kill signal. The gap (architectural recursion on diagonal-state SSMs evaluated on formal proof search) is real and survives — a future run with a smith who can ground every load-bearing claim verbatim should retry it, possibly using the cheaper synthetic-probe gate as the primary deliverable.

### Other gaps considered but not pursued in this run

The following 13 valid gaps were de-prioritised in Phase 3 to keep the hypothesis-smith fan-out tractable. They are documented for transparency.

**Architecture-side gaps not selected (gap-finder-1):** A3 — DEQ-style implicit-fixed-point depth never combined with subquadratic operator inside *f*; A5 — Looped-Transformers-as-Computer construction's sparsity-ratio break point unmeasured; A9 — DSA's lightning indexer is itself O(L²) and recursive amortisation of routing across loops is untried (HISA paper); A10 — Four-way distinction (recursion refines: SSM state / latent answer / memory tokens / input embeddings) never posed for sub-quadratic substrates.

**Architecture × evaluation gaps not selected (gap-finder-2):** B2 — Michelangelo / LSQ open replication × recursion-depth K under SubQ; B3 — Sparse Frontier method-by-task taxonomy × chain-length-scaling regime; B6 — SWE-bench-Verified / SWE-bench-Pro × architectural recursion at edit-LOC quartiles (subsumed partially by H1's R++ tests but not deep on long-horizon edits).

**Discarded by gap-finders (4 + 3 = 7):** Gap-finder-1 discarded 4 gaps as already addressed in published literature or not load-bearing for the fusion thesis; gap-finder-2 discarded 3 as evaluation-only without a candidate architectural mechanism. See gap-finder-{1,2}/output.md for the full discard list.

---

## 5. YAGNI fence reflection

Each spec out-of-scope item is confirmed addressed in this run.

- **No model training in this run.** Confirmed. The swarm produced research direction with eval protocols; no model was trained, fine-tuned, or evaluated empirically. Eval-designers were authorised to *design* multi-GPU runs (eval-designer-1 at 1395 GPU-hr; eval-designer-3 at 16K maximalist / 1200 cheap; eval-designer-4 at 44.2K maximalist / 1050 cheap; eval-designer-5 at 9920; eval-designer-6 at 8336 maximalist / 150 cheap) but the swarm did not execute them.
- **No hardware-level / CUDA / kernel optimisations.** Confirmed. NSA, MoBA, DSA, TTT, Mamba are treated as primitives — abstract behaviour (sparsity pattern, compute scaling, what they can/cannot represent) is reasoned about, not their kernel implementations. References to FlashSparseAttention, MTraining, DHSA, AsyncTLS surfaced in scout-2's literature search but were not invoked in any hypothesis.
- **No quantisation / distillation / MoE / speculative-decoding survey.** Confirmed. G&A distillation (arXiv:2602.11374) appears in H3 only as a *competing baseline* (not a survey item); MoE appears only as PLT's Seed-MoE substrate (arXiv:2510.24824); no quantisation, distillation as a topic, or speculative-decoding work was surveyed.
- **No general post-Transformer architecture survey.** Confirmed. Mamba / RWKV / RetNet / Hyena / Based / GLA cited only where they materially inform a hypothesis (H6 uses Mamba and Longhorn arXiv:2407.14207 as controls; H2 used Mamba as substrate; the rest of the family is bibliographic context only).
- **No AGI / consciousness / general-intelligence claims.** Confirmed. TRM's headline ARC-AGI numbers are treated as a benchmark result for a small architecture; no commentary on AGI timelines or "reasoning emergence." The recursion-vs-CoT and recursion-vs-agent distinctions are reasserted in section 1.
- **No SubQ as commercial product evaluation.** Confirmed. SubQ (Subquadratic 2026 industrial blog) is cited as one architectural data point in scout-2's lineage. The **industrial-blog flag** is preserved across all worker outputs that cite it (scout-2, gap-finder-1, hypothesis-smith-1, eval-designer-1) — every reference includes "industrial blog, no peer-reviewed paper" or equivalent. No comparison of SubQ to other vendors as products.
- **No agent-scaffolded "recursion".** Confirmed and reasserted. "Recursion" in this run means a learned operator applied to its own latent output for K steps within one forward pass with no token emission. RLM (arXiv:2512.24601) and SRLM (arXiv:2603.15653) are cited only in H3's §1 to clarify non-overlap. H3's TRM-mapping (M3) explicitly uses output-conditioned re-feed *without token commit between iterations* — this is the closest the run gets to the agent-recursion line, and it stays on the architectural side because the unmodified backbone is rerun on a longer prompt within one forward pass; no external agent loop.

---

## 6. Recommended next actions

A reader of this document should know which surviving hypothesis to invest in first and why.

**Recommended first hypothesis: H1.** It is the only surviving hypothesis whose **full eval design is in-fence at 1395 GPU-hours** (eval-designer-1 NOT flagged intractable). Its central falsifier (the non-additive DiD with sign-asymmetry constraint, F2) is the cleanest non-additive interaction prediction in the swarm: NSA-fb shows positive recursion lift, MoBA shows non-positive recursion lift, DiD ≥ +5.0 absolute points on BABILong qa3+qa4+qa5 at L=64K, conditional on K=1 calibration. The cheap-path A (NSA-only training-aware F3 ablation, ~250 GPU-hr at 350M params) provides a go/no-go signal before the full 1B run. The smallest meaningful experiment is therefore: **train two 350M NSA models from scratch on 5B tokens of FineWeb-edu — one with the compressed branch retained, one with it zeroed at train+eval — both with K=6 weight-tied recursion in the last 4 blocks; measure F3 differential on BABILong qa3+qa4+qa5 at L=16K**. A positive differential funds the 1B full run; a flat differential kills the compressed-branch causal claim cheaply.

**Second priority: H4 with the F-Calib pre-experiment gate (24 GPU-hr).** Before any from-scratch NSA+MoR run, the cheap pre-test measures whether NSA's `p_t^slc` is actually miscalibrated on RULER-multi-hop dev. If the median rank of gold-evidence blocks is ≤ NSA's top-n on most multi-hop questions, the entire hypothesis is dead before significant compute is spent. This is the cleanest pre-experiment kill in the run.

**Third priority: H6 cheap-path-A (150 GPU-hr) with the explicit acknowledgement that it is insufficient.** Eval-designer-6 flagged the cheap path as in-fence-but-insufficient for the predicted r=0.33 vs threshold 0.5 distinction. If the H6 fine-tune-and-evaluate run shows ΔA_TTT(K=4) ≥ 0.7 × ΔA_dense(K=4), the sub-additive saturation prediction is dead and the maximalist 8336 GPU-hr run is not warranted. If ΔA_TTT(K=4) < 0.5 × ΔA_dense(K=4) emerges even at the underpowered 150 GPU-hr scale, the directional signal motivates funding the full Phase A.

**H3 and H5 are deferred.** H3's full design is intractable at 16K GPU-hr; the 1200 GPU-hr post-hoc-swap cheap path is mechanism-faithful but explicitly weaker (output-conditioned re-feed without native sparse training), so a null result there does not strictly falsify the from-scratch prediction. H5 at 9920 GPU-hr is in-fence-feasible but its TOST-equivalence machinery is unusual and the CB2H stimulus construction is in-house — the cost of the stimulus pipeline plus the operator-not-transferring risk (Risk 1: TRM-style operator may not transfer to long context at all) makes it a riskier first investment than H1.

**One thing the swarm flagged but did not pursue.** Several gap-finder items would benefit from a follow-up run, especially A9 (DSA's lightning indexer is itself O(L²); recursive amortisation of routing across loops untried) and A10 (the four-way distinction of what recursion refines: SSM state / latent answer / memory tokens / input embeddings). Both are architecturally well-defined and would cleanly extend H1 / H3.

**No null result was papered over.** H2 was killed cleanly; the four other hypotheses survived genuine red-team attack with substantive revisions including a polarity flip (H5) and a mechanism pivot (H6).

---

## 7. Run metadata

- **Run id:** `2026-05-10-0729-766039`
- **Spec authored:** 2026-05-10
- **Run executed:** 2026-05-10
- **Novelty target:** `hypothesis`
- **Max parallelism:** 4
- **Worker invocations:** 5 literature-scouts; 2 gap-finders; 6 hypothesis-smiths × 2 rounds = 12; 6 red-team workers × 2 rounds = 12; 5 eval-designers; 1 synthesist. Total = 37 worker outputs.
- **Phase outcomes.** Phase 1 (literature, 158 entries surveyed, 1 industrial-blog flag preserved): PASS. Phase 2 (gap-finding, 19 candidate gaps, 6 selected, 13 retained for audit): PASS. Phase 3 (hypothesis-smith, 6 originals + 6 revisions): PASS. Phase 4 (red-team, 6 round-1 REJECTs, 5 round-2 APPROVEs, 1 round-2 REJECT escalated to kill per cap-3 rule): PASS. Phase 5 (eval-designer, 5 designs, 3 flagged intractable at maximalist scope, all 5 with feasible cheaper-falsification paths): PASS. Phase 6 (synthesist): in progress.
- **Spec success criteria status.** ≥3 hypotheses surviving red-team: **5 surviving** (well above minimum). Eval design depth: every surviving hypothesis has primary + OOD dataset, ≥2 baselines (in fact 4–7), primary + secondary metrics, ≥1 ablation (in fact 3–7), pre-registered statistical plan, budget. Audit trail: H2 preserved with verdict + reason; 13 unselected gaps listed. Synthesist document: this file, target 6–10 pages. Citation discipline: every claim cites arXiv ID / HF ID / repo SHA / industrial blog (flagged); the SubQ industrial-blog flag preserved across all references.

---

## 8. References

This list deduplicates citations across all worker outputs. Industrial-blog citations are flagged. arXiv IDs are verified against scout phase 1 outputs.

**Architectural recursion lineage.**
- arXiv:2510.04871 — Tiny Recursive Model (TRM)
- arXiv:2506.21734 — HRM (Hierarchical Reasoning Model)
- arXiv:2601.10679 — HRM mechanistic critique
- arXiv:1807.03819 — Universal Transformer
- arXiv:2502.05171 — Huginn / Scaling Test-Time Compute with Latent Reasoning
- arXiv:2507.02199 — Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer (Coda Lens probe)
- arXiv:2510.25741 — Ouro / Scaling Latent Reasoning via Looped LMs
- arXiv:2507.10524 — Mixture-of-Recursions (MoR)
- arXiv:2510.24824 — Parallel Loop Transformer (PLT, G-SWA)
- arXiv:2502.17416 — Reasoning with Latent Thoughts
- arXiv:2311.12424 — Looped Transformers Better at Learning Algorithms
- arXiv:2410.01405 — On Expressive Power of Looped Transformers
- arXiv:2604.21106 — Iso-Depth Scaling Laws for Looped LMs
- arXiv:2410.20672 — Relaxed Recursive Transformers
- arXiv:2511.07384 — Retrofitted Recurrence (McLeish et al.)
- arXiv:2511.08577 — Think-at-Hard
- arXiv:2603.08391 — Adaptive Loops in Transformers
- arXiv:2604.12946 — Parcae: Scaling Laws for Stable Looped LMs
- arXiv:2510.01585 — ReSSFormer

**Subquadratic-attention lineage.**
- arXiv:2502.11089 — Native Sparse Attention (NSA, three-branch design)
- arXiv:2502.13189 — MoBA (Mixture of Block Attention, no fallback)
- arXiv:2512.02556 — DeepSeek-V3.2 / DSA (lightning-indexer top-k)
- arXiv:2506.08889 — SeerAttention-R
- arXiv:2504.17768 — Sparse Frontier
- arXiv:2406.10774 — Quest (query-aware top-k page selection)
- arXiv:2505.14840 — Gupta–Huang–Saha–Xu–Ye, formal subquadratic-attention regime
- arXiv:2511.20102 — SSA (Gradient Update Deficiency Prop. 4.1; Theorem 1; Appendix F)
- arXiv:2511.00819 — NSA + Latent Attention variant
- arXiv:2407.15891 — RazorAttention (retrieval-head exploitation for KV compression)
- **SubQ — Subquadratic, 2026 (industrial blog, no peer-reviewed paper).** Flagged across all citing workers.

**SSM and linear-attention adjacents.**
- arXiv:2407.14207 — Longhorn (SSMs as amortised online learners)
- arXiv:2503.10799 — Fixed-Point RNNs
- arXiv:2407.04620 — TTT (RNNs with Expressive Hidden States)
- arXiv:2505.23884 — LaCT (Test-Time Training Done Right)
- arXiv:2602.21204 — TTT with KV Binding Is Secretly Linear Attention
- arXiv:2504.05298 — One-Minute Video Generation with TTT
- arXiv:2603.06642 — SR-TTT (post-cutoff, mentioned)
- arXiv:2604.06169 — In-Place TTT (post-cutoff, mentioned)
- arXiv:2404.08819 — Illusion of State (regular-language state tracking)
- arXiv:2412.06148 / 2411.12537 / 2412.19350 / 2410.03810 — TC⁰ limits of diagonal SSMs
- arXiv:2602.08426 — Prism (RoPE spectral signal)

**Long-context reasoning benchmarks and failure modes.**
- arXiv:2406.10149 — BABILong (HF: `RMT-team/babilong`, `RMT-team/babilong-1k-samples`; Apache-2.0)
- arXiv:2404.06654 — RULER (HF: `simonjegou/ruler`; Apache-2.0)
- arXiv:2502.05167 — NoLiMa (HF: `amodaresi/NoLiMa`; **CC-BY-NC-4.0 flagged**)
- arXiv:2410.04422 — Hyper-multi-step
- arXiv:2407.11963 — NeedleBench (HF: `opencompass/NeedleBench`)
- arXiv:2411.05000 — Needle Threading (HF: `jonathan-roberts1/needle-threading`)
- arXiv:2601.20276 — Beyond the Needle's Illusion / EverMemBench-S (HF status unverified at run time)
- arXiv:2404.15574 — Retrieval Head Mechanistically Explains Long-Context Factuality
- arXiv:2602.11374 — G&A heads / Retrieval-Aware Distillation for Transformer-SSM Hybrids
- arXiv:2307.03172 — Lost in the Middle
- arXiv:2502.01951 — Position-Bias Emergence under causal/sliding-window masks (Theorems 4.1–4.2)
- arXiv:2412.10319 — SCBench (sparse-attention KV compression robustness)
- arXiv:2512.13898 — Score Dilution at Test Time
- arXiv:2602.15028 — Long Context, Less Focus
- arXiv:2509.04475 — ParaThinker / Tunnel Vision in CoT
- arXiv:2509.07980 — Parallel-R1
- arXiv:2412.19707 — Thought Rollback
- arXiv:2504.07052 — To Backtrack or Not to Backtrack
- arXiv:2510.05862 — Context Denoising Training
- arXiv:2602.07150 — On Randomness in Agentic Evals (10-seed minimum)
- arXiv:2402.16837 — Latent multi-hop reasoning in LLMs
- arXiv:2409.12941 — FRAMES (HF: `google/frames-benchmark`; Apache-2.0)
- HF: `dgslibisey/MuSiQue` (CC-BY-4.0)
- HF: `hotpotqa/hotpot_qa` (distractor split, CC-BY-SA-4.0)
- HF: `HuggingFaceFW/fineweb-edu` (ODC-BY-1.0)

**Math / proof / program-synthesis benchmarks.**
- arXiv:2407.11214 — PutnamBench
- arXiv:2109.00110 — miniF2F
- arXiv:1905.09381 — CoqGym
- arXiv:2306.15626 — LeanDojo
- arXiv:2401.03065 — CRUXEval (HF: `cruxeval-org/cruxeval`; MIT)
- arXiv:2408.13001 — CRUXEval-X (HF: `xhwl/cruxeval-x`; MIT)
- arXiv:2401.12947 — Transformers Are Not Yet Perfect at Structural Recursion
- arXiv:2503.06692 — InftyThink
- arXiv:2402.12875 — CoT-Solves-Serial
- arXiv:2601.18401 — Superlinear Multi-Step Attention
- HF: `sapientinc/sudoku-extreme`
- HF: `RMT-team/babilong` (qa1–qa20)

**Agent-recursion / programmatic recursion (cited only to clarify non-overlap).**
- arXiv:2512.24601 — Recursive Language Models
- arXiv:2603.15653 — SRLM

**Repo / open-implementation references.**
- `fla-org/native-sparse-attention` (Apache-2.0); `tilde-research/nsa-impl` (MIT) — NSA reference impls
- `MoonshotAI/MoBA` (Apache-2.0)
- `SamsungSAILMontreal/TinyRecursiveModels` (Apache-2.0)
- `test-time-training/ttt-lm-jax` (TTT reference impl)
- `github.com/NVIDIA/RULER` (Apache-2.0)
- `github.com/adobe-research/NoLiMa` (CC-BY-NC-4.0 flagged)

**Industrial-blog flagged citation.** SubQ — Subquadratic, 2026 — *industrial blog, no peer-reviewed paper.* Flag preserved at scout-2/output.md, gap-finder-1/output.md, hypothesis-smith-1/output.md, eval-designer-1/output.md.

---

*End of synthesist output. This document is the run's primary deliverable. Run-root copy at `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/output.md`; synthesist-subdir copy at `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/synthesist/output.md`.*
