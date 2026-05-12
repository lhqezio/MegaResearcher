# Hypothesis H2 — TC⁰ escape via external TRM-style recursion on diagonal SSMs, distinguished from Fixed-Point-RNN internal iteration, on long-tactic-chain formal proof search

## 1. Targeted gap

This hypothesis addresses **gap H2 — TC⁰ escape via external recursion on SSMs (composes A2 + A7 + B5)**, as identified by the gap-finders:

- gap-finder-1, Gap 2: "TRM-style depthwise recursion has never been instantiated on any SSM / linear-RNN backbone … and the conceptual relationship to Fixed-Point RNNs (arXiv:2503.10799) is unspecified" (`docs/research/runs/2026-05-10-0729-766039/gap-finder-1/output.md` line 27).
- gap-finder-1, Gap 7: "Diagonal-state SSMs … live in TC⁰ … whether *external* depthwise recursion lifts a diagonal-state SSM out of TC⁰ has never been tested … with two competing internal fixes" (line 97).
- gap-finder-2, Gap 5: "no SSM/linear-attention paper reports PutnamBench/miniF2F/CoqGym at all … proof-success-rate vs architectural recursion depth K_arch for {dense, Mamba, Mamba+recursion, Fixed-Point-RNN} with chain-length stratified by tactic count" (line 54).

The composed gap: external TRM-style architectural recursion (arXiv:2510.04871) layered on a diagonal-state SSM (Mamba arXiv:2312.00752, Mamba-2 arXiv:2405.21060, RWKV pre-7 arXiv:2305.13048, GLA arXiv:2312.06635, DeltaNet arXiv:2406.06484) has never been tested on a state-tracking-bound benchmark. Three published TC⁰-escape mechanisms exist internal to the backbone — negative-eigenvalue transitions (arXiv:2411.12537), dense / non-diagonal selective transitions (SD-SSM, arXiv:2412.19350), and RWKV-7's vector-valued ICL-rate construction (arXiv:2503.14456). One *adjacent* mechanism iterates the SSM's own state-transition operator K times toward a contraction-map fixed point that recovers state tracking (Fixed-Point RNNs, arXiv:2503.10799). External TRM-style recursion is a *fourth* candidate fix, and is conceptually distinct from Fixed-Point-RNN iteration in three ways (see §3).

## 2. Hypothesis statement

**If** an *external* TRM-style recursion (a learned block of the form `state ← f(state, input)` re-applied K_arch times within a single forward pass with deep supervision over the recursion trajectory, per arXiv:2510.04871) is wrapped around a frozen diagonal-state SSM backbone (Mamba, arXiv:2312.00752) and evaluated on long-tactic-chain Lean proof search at fixed Mathlib retrieval cutoff (LeanDojo, arXiv:2306.15626), **then** at chain-length stratum L_long (≥ 50 tactic steps; PutnamBench arXiv:2407.11214 + miniF2F arXiv:2109.00110 hard subset) the proof-success-rate gap between Mamba+TRM-recursion(K_arch ≥ 6) and Mamba+TRM-recursion(K_arch = 1) will be *strictly larger* than the gap between Fixed-Point-RNN-Mamba(K_iter ≥ 6) and Fixed-Point-RNN-Mamba(K_iter = 1), with the difference exceeding 5 absolute percentage points; and **simultaneously** at chain-length stratum L_short (≤ 10 tactic steps) the two K-vs-1 gaps will be statistically indistinguishable (within 2 absolute points), producing a **non-additive interaction** between (mechanism × chain length).

This is a comparative, falsifiable claim about *where* and *how strongly* each TC⁰-escape mechanism delivers expressivity gains. The two recursion mechanisms are predicted to behave the same on short chains (where TC⁰ is sufficient) and to diverge on long chains (where iterated computation is required), with TRM-style external recursion delivering more long-chain rescue per iteration than Fixed-Point-RNN internal iteration — for a mechanistic reason that is itself testable (§3).

## 3. Mechanism

### Why diagonal-state SSMs fail at long tactic chains

Diagonal selective SSMs (Mamba arXiv:2312.00752, RWKV pre-7 arXiv:2305.13048, GLA arXiv:2312.06635, mLSTM, DeltaNet arXiv:2406.06484) compute a per-channel scalar recurrence; their state-transition matrices are diagonal, so the per-channel hidden state evolves independently. Illusion of State (arXiv:2404.08819) and Computational Limits via Circuit Complexity (arXiv:2412.06148) jointly prove these models live in DLOGTIME-uniform TC⁰ at constant depth and poly(n) precision, and therefore cannot solve permutation composition, parity, or arithmetic-formula evaluation in one forward pass. Mamba's COPY bound (arXiv:2410.03810) gives a concrete lower bound on the failure mode. Long tactic chains in Lean / Coq are a state-tracking task: each tactic application updates the proof state (a non-commutative product of substitutions), and verifying chain validity requires composing them. By the TC⁰ argument, a one-pass diagonal-state SSM cannot solve this for chain length L exceeding a constant.

### Why external TRM-style recursion plausibly escapes TC⁰

CoT Solves Inherently Serial Problems (arXiv:2402.12875) proves that constant-depth Transformers with O(T) iterations of *sequential generation* leave TC⁰ and reach P/poly. The proof generalizes to any constant-depth substrate: K iterations of a learned block compose to depth K × d_block. TRM (arXiv:2510.04871) iterates a learned block (attention + FFN + residual + a small latent answer state) K_arch times within the forward pass under deep supervision, with no contraction-map constraint — the iterated map is free to be expansive, periodic, or attractor-seeking. Applying TRM-style recursion around a Mamba block lifts the effective circuit depth from d_Mamba to K_arch × d_Mamba; for sufficient K_arch this is no longer constant in the input, and the TC⁰ bound no longer applies (per the depth-buys-expressivity result of arXiv:2402.12875).

### Why Fixed-Point-RNN internal iteration is *different* (and predicted to be weaker on long chains)

Fixed-Point RNNs (arXiv:2503.10799) iterate the *internal state-transition matrix* of a single diagonal linear RNN K_iter times; the iteration is constrained to be a contraction map so it converges to a unique fixed point that parameterizes a *dense* linear RNN. The escape mechanism is: a dense linear RNN (non-diagonal transition) is strictly more expressive than a diagonal one and can solve regular-language state-tracking tasks (compatible with SD-SSM, arXiv:2412.19350). Three structural differences from TRM-style recursion:

1. **Object refined.** Fixed-Point RNNs refine the *transition operator* (a matrix in the SSM cell). TRM refines the *latent state and inputs jointly* (per arXiv:2510.04871 §2). The TRM-iterated object carries information about the proof tree, not just the recurrence dynamics.
2. **Contraction constraint.** Fixed-Point RNNs *require* a contraction map to converge (arXiv:2503.10799 abstract / formal claim). TRM does not — the iterated map can implement non-contractive compositions, including the non-commutative permutation products that long tactic chains demand. Permutation composition is *not* a contraction map; it has eigenvalues on the unit circle. A contraction-constrained iteration cannot represent permutation composition exactly; an unconstrained iteration can. This is the load-bearing claim of the hypothesis.
3. **Iteration scope.** Fixed-Point RNN iteration is *over one layer's transition matrix*; TRM iteration is *over the entire block including residual stream*. The latter accumulates depth; the former does not (it's a re-parameterization of a single dense linear RNN).

### Why the interaction is non-additive

On short chains (≤ ~10 tactics), a one-pass diagonal SSM is already within TC⁰ capacity for the requisite state-tracking depth (Illusion of State arXiv:2404.08819 §4 establishes the one-pass capacity scales as ~O(log L) on parity-like tasks for fixed-precision constant-depth networks). Iterating either mechanism adds little because the substrate already suffices. On long chains, the TC⁰ wall is hit; *both* iteration mechanisms help, but they help unequally because Fixed-Point-RNN iteration is contraction-bounded (cannot represent non-contractive permutation composition) while TRM-style external iteration is not. So the K-vs-1 effect scales with chain length, AND the per-K effect is larger for TRM than for Fixed-Point-RNN — neither factor is independent of the other. This is a non-additive interaction in the formal ANOVA sense: the (mechanism × chain length) interaction term is non-zero with sign predicted in advance.

### Distinction from CoT and from agent scaffolding

Both mechanisms are *architectural* — the iteration happens inside one forward pass, not by emitting tokens, not by external proof-search scaffolding. The output token sequence is a flat tactic chain; only the per-token forward-pass compute differs. M1 (arXiv:2504.10449), Scaling Reasoning without Attention (arXiv:2505.22425), and Thinking Slow Fast (arXiv:2502.20339) achieve reasoning with SSM/linear-attention substrates via *output CoT*, not depth-time recursion in the forward pass; this hypothesis is orthogonal to all three. Test-time-sampling (best-of-N, MCTS over tactics) is held constant across arms.

## 4. Predicted outcome with magnitude

**Primary metric:** pass-rate on PutnamBench (arXiv:2407.11214) + miniF2F-test (arXiv:2109.00110, Lean 4 port) under the LeanDojo (arXiv:2306.15626) execution harness with retrieval cutoff K=100 premises.

**Stratification:** chain-length-stratified by tactic count of the *minimal* known proof, into:
- L_short: ≤ 10 tactics (most miniF2F-IMO low-tier; PutnamBench warmups)
- L_med: 11–49 tactics
- L_long: ≥ 50 tactics (PutnamBench hard tier; Coq long-tactic chains via CoqGym arXiv:1905.09381)

**Arms (K_arch / K_iter held at the same value where both apply):**
1. Mamba (off-the-shelf checkpoint, K_arch=1)
2. Mamba + TRM-style external recursion, K_arch ∈ {1, 2, 4, 6, 8}
3. Fixed-Point-RNN-Mamba (arXiv:2503.10799 instantiation), K_iter ∈ {1, 2, 4, 6, 8}
4. RWKV-7 Goose (arXiv:2503.14456), K_arch=1 (its own ICL-rate fix is "internal")
5. Dense Transformer of matched parameter count, K_arch=1 (control for raw capacity)

**Predicted effect sizes (point estimates with reasoning):**

| Stratum | Mamba(K=1) | Mamba+TRM(K=6) | FPR-Mamba(K=6) | RWKV-7(K=1) | Dense(K=1) |
|---|---|---|---|---|---|
| L_short | 25% | 27% (±2) | 27% (±2) | 28% | 30% |
| L_med | 12% | 19% | 16% | 18% | 22% |
| L_long | 2% | 11% (±3) | 5% (±3) | 6% | 14% |

The numerical anchor for L_short (25%) is calibrated to ReProver (arXiv:2306.15626) baseline behavior under retrieval K=100 on miniF2F low-tier; the L_long Mamba(K=1) ≈ 2% anchor is calibrated to the well-documented catastrophic drop of one-pass diagonal SSMs on regular-language tasks beyond constant length (SD-SSM ablation, arXiv:2412.19350 §5; Negative Eigenvalues parity curves, arXiv:2411.12537 Fig 2).

**Headline predicted gaps (the falsifiable core):**

- L_long: [Mamba+TRM(K=6) − Mamba+TRM(K=1)] − [FPR-Mamba(K=6) − FPR-Mamba(K=1)] ≥ +5 absolute points (TRM gap > FPR gap on long chains).
- L_short: |[Mamba+TRM(K=6) − Mamba+TRM(K=1)] − [FPR-Mamba(K=6) − FPR-Mamba(K=1)]| ≤ 2 absolute points (no mechanism advantage on short chains).
- Mechanism × chain-length interaction term in a 2×3 ANOVA over {TRM, FPR} × {short, med, long}: F-statistic significant at p < 0.05 with sign as predicted (TRM advantage grows with L).

**Conditions under which the hypothesis should hold:**
- Same Mamba pretrained checkpoint for arms 1–3.
- Recursion depth K_arch held compute-matched against K_iter so the per-token FLOPs are within ±10%.
- Retrieval cutoff K=100 fixed across arms.
- Chain-length stratum bins large enough (≥ 30 problems each) for ±3-point CIs.

**Conditions under which the hypothesis should NOT hold (i.e., the prediction does not apply):**
- If retrieval K is so low that proof-state context is truncated, all arms collapse to noise; chain-length stratification becomes meaningless.
- If the chosen K_arch budget is pre-trained-into-the-checkpoint (e.g., the Mamba checkpoint was already trained with TRM scaffolding), the comparison degenerates.
- On problems decomposed into many short sub-proofs by a *retrieval-driven* premise lookup such that no single forward pass needs > 10 tactic steps of state-tracking, the gap collapses by construction.

## 5. Falsification criteria

Each criterion specifies metric + threshold + direction, in line with the discipline rules.

**F1. (Equal-mechanism falsification.)** If on L_long, [Mamba+TRM(K=6) − Mamba+TRM(K=1)] − [FPR-Mamba(K=6) − FPR-Mamba(K=1)] ≤ +1 absolute point in pass-rate, with 95% CI excluding +5, the hypothesis is falsified — the two recursion mechanisms deliver equivalent expressivity gains, and the contraction-vs-no-contraction structural argument (§3) is empirically wrong.

**F2. (Short-chain leakage falsification.)** If on L_short, |[Mamba+TRM(K=6) − Mamba+TRM(K=1)]| ≥ 5 absolute points OR [Mamba+TRM(K=6) − Mamba+TRM(K=1)] vs [FPR-Mamba(K=6) − FPR-Mamba(K=1)] differ by ≥ 5 points, the non-additivity prediction is falsified — the mechanism gain is *not* concentrated where TC⁰ is the binding constraint, suggesting the gain (if any) comes from something other than TC⁰ escape (e.g., raw parameter expansion, optimization stability), undercutting the mechanism story.

**F3. (Internal-fix dominance falsification.)** If RWKV-7(K=1) on L_long matches or exceeds Mamba+TRM(K=6) within 2 absolute points, then RWKV-7's internal vector-valued ICL-rate construction (arXiv:2503.14456) already provides the TC⁰ escape "for free," and external TRM-style recursion is structurally redundant. The thesis that TRM-style recursion adds value beyond the best published internal fix is falsified. (This is the comparative claim against gap-finder-1 Gap 7's third candidate fix.)

**F4. (Coherence falsification.)** If at K_arch ∈ {2, 4, 6, 8} the long-chain pass-rate is *non-monotone* (e.g., K=4 is best and K=8 is worse by ≥ 3 points), and the same non-monotonicity does NOT appear in Fixed-Point-RNN-Mamba, the depth-buys-expressivity mechanism (arXiv:2402.12875) does not transfer to TRM-style recursion on diagonal SSMs, and the structural justification in §3 is wrong even if the K=6 vs K=1 single-point comparison passes by accident.

**F5. (Retrieval-dependence falsification, sanity check.)** If the hypothesis result holds at retrieval K=100 but reverses at K=200 (i.e., Mamba+TRM advantage on L_long disappears or inverts when more premises are available), then the apparent recursion gain is actually a retrieval-compensation artifact — the SSM was failing because it couldn't read enough context, not because of TC⁰. The mechanism story is then falsified at the level of attribution.

## 6. Required experiments (sketch — eval-designer fills in)

- **Primary dataset.** PutnamBench Lean 4 (arXiv:2407.11214, Apache-2.0/MIT) + miniF2F-test Lean 4 (arXiv:2109.00110) — together ~640 + 244 problems with hand-stratifiable known-proof tactic counts. Optionally CoqGym (arXiv:1905.09381) for L_long density.
- **Execution harness.** LeanDojo (arXiv:2306.15626, MIT) — provides the retrieval-K knob and a deterministic tactic-application API. Use ReProver as the retrieval baseline.
- **Backbones.**
  - Mamba (arXiv:2312.00752) — off-the-shelf 2.8B / 1.4B pretrained checkpoint.
  - Fixed-Point-RNN-Mamba (arXiv:2503.10799) — author-released Mamba variant or our reconstruction with K_iter ∈ {1, 2, 4, 6, 8}.
  - RWKV-7 Goose (arXiv:2503.14456) — published 1.5B / 3B checkpoints.
  - Dense Transformer at matched parameter count — Llama-3.2 1B/3B, control for raw capacity.
- **TRM wrapper.** Re-implement TRM (arXiv:2510.04871) as an outer learned block iterating around a frozen Mamba block; deep supervision over recursion trajectory; YAGNI: no new training of the Mamba backbone, only the TRM wrapper params.
- **Baselines / ablations.**
  - K_arch = 1 (recursion off) vs K_arch ∈ {2, 4, 6, 8}.
  - K_iter = 1 vs K_iter ∈ {2, 4, 6, 8} for Fixed-Point-RNN-Mamba.
  - Dense Transformer with same K_arch wrapper (does TRM-style recursion help dense too? — predicted: yes on L_long but with smaller gap because dense already has more depth).
  - Test-time-sampling held constant: greedy + beam=4 across all arms.
- **Stratification.** Bin problems by minimal-known-proof tactic count; report per-bin pass@1 and pass@8.
- **Compute envelope.** Per-arm forward FLOPs matched within ±10% by adjusting K so that K × backbone-FLOPs + wrapper-FLOPs is constant; this is the cleanest controlled comparison.

## 7. Cheaper falsification path

Smaller-scale ablation that still kills the hypothesis if it fails: **synthetic permutation-composition probe at variable composition depth**, on the same backbones, with the same K_arch / K_iter sweep, *without* training new checkpoints.

- Task: S_5 permutation composition (per Illusion of State arXiv:2404.08819 setup), with composition lengths L ∈ {5, 10, 50, 200}.
- Probe: linear classifier on the final hidden state predicting the composed permutation. No fine-tuning of the backbone.
- Predict: at L = 5–10, all (TRM-K=6, FPR-K=6, Mamba-K=1) achieve > 90% probe accuracy. At L = 200, Mamba-K=1 → ~3% (chance 1/120), FPR-K=6 → ~30% (contraction-bounded), TRM-K=6 → ~70%; gap [TRM-K=6 − FPR-K=6] ≥ +20 points at L=200, ≤ +3 points at L=5.

This kills the hypothesis at < 1% of full-eval cost: if TRM and FPR perform identically on synthetic permutation composition across L, the contraction-vs-no-contraction structural argument is wrong before any Lean compute is spent. Conversely, if the synthetic gap appears as predicted but the Lean gap doesn't, that itself is interesting (it means the Lean failure mode is *not* permutation-composition-bound), and the synthesist can re-target the hypothesis to which proof tasks load on permutation composition.

## 8. Risks to the hypothesis

Three independent ways this could be wrong, with what the hypothesis still contributes if each materializes:

**R1. The Lean failure mode is not state-tracking-bound at all.** It might be retrieval-bound or premise-selection-bound, in which case neither recursion mechanism helps at L_long because the SSM is failing for an unrelated reason. *Contribution if so:* the experiment cleanly factors out the TC⁰ axis from the retrieval axis (via F5), which is itself a useful negative result for the field — many papers conflate them.

**R2. Compute-matching dominates the comparison.** TRM at K_arch=6 may simply have more wrapper parameters than Fixed-Point-RNN at K_iter=6, and the gap may reflect parameter count not iteration mechanism. *Contribution if so:* the experiment forces the field to design recursion mechanisms with parameter-count control, exposing a previously implicit confound; we report the (FLOP-matched, param-matched, K-matched) triple grid as a deliverable.

**R3. RWKV-7's internal fix dominates on L_long.** If RWKV-7(K=1) ≥ Mamba+TRM(K=6) on long chains (F3), TRM-style external recursion is redundant on subquadratic backbones. *Contribution if so:* a head-to-head between three distinct TC⁰-escape mechanisms (negative eigenvalues, dense transitions via FPR, RWKV-7 ICL-rate construction, external TRM recursion) on the same benchmark is *itself* the contribution; the field currently has zero such comparisons.

A fourth risk worth flagging: TRM's deep supervision objective may not be transfer-applicable to Lean tactic prediction without joint fine-tuning of the backbone, in which case the wrapper is poorly optimized and the K_arch sweep is uninformative. Mitigation: a small held-out fine-tuning protocol on LeanDojo Mathlib data, with the wrapper-only fine-tuning variant reported alongside the frozen-backbone variant.

## 9. Sources

All cited arxiv IDs were resolved via `hf_papers paper_details` during preparation:

- TRM (Tiny Recursive Model) — arXiv:2510.04871
- Fixed-Point RNNs — arXiv:2503.10799
- HRM (Hierarchical Reasoning Model) ancestor — arXiv:2506.21734 (cited via 2510.04871)
- Huginn / latent recurrent depth — arXiv:2502.05171
- Mamba — arXiv:2312.00752
- Mamba-2 — arXiv:2405.21060
- RWKV — arXiv:2305.13048
- RWKV-7 Goose — arXiv:2503.14456
- RetNet — arXiv:2307.08621
- GLA — arXiv:2312.06635
- DeltaNet — arXiv:2406.06484
- Hyena — arXiv:2302.10866
- Illusion of State in SSMs — arXiv:2404.08819
- Computational Limits via Circuit Complexity — arXiv:2412.06148
- Negative Eigenvalues for State Tracking — arXiv:2411.12537
- SD-SSM (Selective SSMs on Regular Languages) — arXiv:2412.19350
- Mamba COPY bound — arXiv:2410.03810
- CoT Solves Inherently Serial Problems — arXiv:2402.12875
- PutnamBench — arXiv:2407.11214
- miniF2F — arXiv:2109.00110
- CoqGym — arXiv:1905.09381
- LeanDojo — arXiv:2306.15626
- M1 — arXiv:2504.10449
- Scaling Reasoning without Attention — arXiv:2505.22425
- Thinking Slow Fast — arXiv:2502.20339

Internal references:
- Gap-finder-1 output: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/gap-finder-1/output.md` (Gaps 2 and 7)
- Gap-finder-2 output: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/gap-finder-2/output.md` (Gap 5)
