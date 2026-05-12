# Hypothesis H2 (revision 1) — Iteration-scope and parameterization differences between TRM-style and Fixed-Point-RNN-style depth-recurrence on a Mamba backbone, evaluated on long-tactic-chain Lean proof search, gated by a synthetic permutation-composition probe

## Changes from revision-0 (response to red-team round 1)

The red-team identified four critical and several important objections against revision-0. Each is addressed below; the rest of the document is rewritten consistent with the responses.

**C1 — FPR misrepresented (CRITICAL).** *Acknowledged. Revision-0 was wrong.* I re-read FPR §3 (Fixed-Points as an RNN Layer) and §4 (Fixed-Point Mamba) including Figure 4. The contraction-map property in FPR is on the *depth-iteration solver* (Banach-Caccioppoli convergence guarantee, Theorem 3.1), and the converged fixed point parameterizes a *dense* linear RNN with non-diagonal mixer 𝐐_t, which is strictly more expressive than diagonal Mamba and which the paper's headline experiment (Figure 4) shows solving A_5 and S_5 — i.e., permutation composition. The revision-0 claim "FPR cannot represent permutation composition exactly" is **directly falsified by the cited paper**. I have **dropped** the contraction-vs-no-contraction structural argument entirely. The new mechanism story (§3) is parameterization- and iteration-scope-based and contains *no* expressivity-class claims that distinguish TRM from FPR.

**C2 — CoT-Solves-Serial over-extrapolated (CRITICAL).** *Acknowledged. Revision-0 was wrong.* The CoT-Solves-Serial result (arXiv:2402.12875) requires T autoregressively-emitted tokens fed back as inputs and embedding-dimension log(n); it does *not* generalize to constant-K depth iteration over a fixed-size hidden state. K passes of a TC⁰ Mamba block at constant K is closed under TC⁰ composition. I have **dropped the TC⁰-escape framing**. The hypothesis is now an empirical *parameterization comparison* under matched compute, with no claim that any mechanism leaves TC⁰ at constant K. (R3 in revision-0's risks section is also dropped: F3's "RWKV-7 dominates" outcome is now an *equally interesting* empirical result rather than a "comparative claim falsification.")

**C3 — Frozen Mamba unprecedented (CRITICAL).** *Acknowledged.* Revision-0's "frozen Mamba + TRM wrapper" combined two things that have never been shown to work together. I have **adopted Retrofitted Recurrence's continued-pretraining recipe** (arXiv:2511.07384, McLeish et al., Nov 2025): initialize from a pretrained Mamba checkpoint, surgery the architecture to insert depth-recurrence (TRM-style block-level recursion or FPR-style transition-mixer iteration), train *all* parameters under a curriculum-of-recurrences continued-pretraining schedule on a math+code corpus, then fine-tune on Mathlib. Concretely: (a) all backbone params receive gradients during continued pretraining; (b) curriculum: linear schedule of mean recurrence count from 1 → 8 over the first 75% of continued pretraining, constant 8 thereafter (per Retrofitted Recurrence §4.2); (c) Muon optimizer (per §4.3.1). This is no longer a novel training recipe — it is the published recipe re-applied to a Mamba substrate.

**C4 — Magnitude calibration speculative (CRITICAL).** *Acknowledged.* Revision-0 anchored L_long ≈ 2% to synthetic regular-language SD-SSM and Negative-Eigenvalues curves, which is multiple distribution shifts away from Lean tactic prediction. I have **switched the headline magnitude prediction to a *relative* (ratio) form** robust to base rate, and added a **mandatory Lean-Mamba pilot** (~80 H100 hours) before the main eval to calibrate the actual base rate. The pilot's job is to lock in the absolute magnitudes before the main run; pre-registered thresholds are stated as ratios of pilot-measured rates. See §4.

**I1 — Mamba has no Lean pretraining.** *Acknowledged and folded into C3 / C4.* Continued pretraining on a math+code corpus (Nemotron-CC-Math-v1, per Retrofitted Recurrence §4.3.2) plus Mathlib fine-tuning is now part of every arm; the comparison is between {TRM-style, FPR-style, no-recurrence} all initialized from the same Mamba checkpoint and trained under the same continued-pretraining FLOP budget.

**I2 — F1 statistical power.** *Acknowledged.* Bin sizes increased to ≥ 100 problems on L_long (CoqGym arXiv:1905.09381 supplements PutnamBench-hard for sample size), and the threshold is restated as a power-calculated ratio rather than absolute points. The cheaper falsification path (synthetic permutation-composition probe at variable depth, §7) is now a **gating prerequisite** before any Lean compute is spent — if TRM and FPR are indistinguishable on the synthetic probe across composition lengths L ∈ {5, 10, 50, 200}, the parameterization-difference hypothesis is dead before Lean.

**I3 — F3 fallback narrative.** *Acknowledged.* The §8 R3 retreat ("head-to-head is itself the contribution") is a familiar dodge. I have **removed the fallback** and restated F3 as a clean falsification: if RWKV-7(K=1) on L_long matches or exceeds the better of {TRM-Mamba(K=8), FPR-Mamba(K=8)} within statistical noise, the depth-recurrence-on-Mamba framing is dead.

**I4 — TRM is also attractor-seeking.** *Acknowledged.* TRM §4.1 (verbatim): "by the design of the deep supervision goal, running a few full recursion processes (even without gradients) is expected to bring us closer to the solution." Both TRM and FPR are *trained to be attractor-seeking* — TRM through the deep supervision objective, FPR through the contraction constraint on the solver. The revised mechanism story (§3) makes no claim that TRM is unconstrained relative to FPR.

**I5 — Param-count confound.** *Acknowledged.* A new falsification criterion F6 is added: a parameter-matched dense Transformer with TRM-style recursion (i.e., Retrofitted Recurrence's published recipe applied to a Llama-3.2 baseline) is included as a control arm. If the predicted ordering on L_long is preserved when the Mamba substrate is replaced with a Transformer, the SSM-specific story is falsified.

**S1 / Adjacent prior art.** Retrofitted Recurrence (arXiv:2511.07384) and Think-at-Hard (arXiv:2511.08577) are now first-class baselines and architectural precedents in §3 and §6.

**Net effect on the hypothesis.** The original "TRM beats FPR on L_long because TC⁰ escape with no contraction" mechanism is dead. The revised hypothesis is sharply weaker and more honest: it is a **parameterization-and-iteration-scope** comparison (block-level depth recursion vs solver-iteration on the transition mixer) on a single substrate (Mamba) under a single published training recipe (continued pretraining per Retrofitted Recurrence), gated by a cheap synthetic probe, with the directional prediction supported only by *empirical* differences in iteration-scope rather than expressivity-class arguments. The headline magnitude prediction is now a *ratio* tied to a pilot-measured base rate. **A null result on the synthetic probe collapses the entire hypothesis**, which is the right shape for a falsifiable claim.

---

## 1. Targeted gap

This hypothesis addresses **gap H2 — depthwise recursion on diagonal-state SSMs evaluated on formal proof search (composes A2 + A7 + B5)**, as identified by:

- gap-finder-1, Gap 2: "TRM-style depthwise recursion has never been instantiated on any SSM / linear-RNN backbone … and the conceptual relationship to Fixed-Point RNNs (arXiv:2503.10799) is unspecified" (`docs/research/runs/2026-05-10-0729-766039/gap-finder-1/output.md` line 27).
- gap-finder-1, Gap 7: "Diagonal-state SSMs … live in TC⁰ … whether *external* depthwise recursion lifts a diagonal-state SSM out of TC⁰ has never been tested … with two competing internal fixes" (line 97).
- gap-finder-2, Gap 5: "no SSM/linear-attention paper reports PutnamBench/miniF2F/CoqGym at all … proof-success-rate vs architectural recursion depth K_arch for {dense, Mamba, Mamba+recursion, Fixed-Point-RNN} with chain-length stratified by tactic count" (line 54).

The narrow, surviving gap (per red-team §2): no published work compares block-level depth-recurrence (TRM-style; arXiv:2510.04871) and solver-iteration-on-transition-mixer (FPR-style; arXiv:2503.10799) on the *same* Mamba substrate under a *same* continued-pretraining recipe (Retrofitted Recurrence; arXiv:2511.07384), evaluated on formal proof search (PutnamBench arXiv:2407.11214, miniF2F arXiv:2109.00110, CoqGym arXiv:1905.09381). This is a head-to-head architectural comparison the field has not run.

The novelty is reduced (depth-recurrent retrofitting on pretrained Transformers is now established by Retrofitted Recurrence and Think-at-Hard arXiv:2511.08577); however, neither paper does this on a Mamba substrate, and neither evaluates on formal proof search.

## 2. Hypothesis statement

**If** two depth-recurrence variants — (a) TRM-style block-level recursion (arXiv:2510.04871; an outer learned block iterating around a Mamba block under deep supervision) and (b) FPR-style solver-iteration on the Mamba transition mixer (arXiv:2503.10799; iterating the channel-mixing matrix 𝐐_t to a Banach fixed point parameterizing a dense linear RNN) — are each retrofitted onto the *same* pretrained Mamba checkpoint via the *same* continued-pretraining recipe (curriculum of recurrences per Retrofitted Recurrence arXiv:2511.07384, §4.2), and evaluated on long-tactic-chain Lean / Coq formal proof search at fixed retrieval cutoff (LeanDojo arXiv:2306.15626, retrieval K=100), **then** on a synthetic permutation-composition probe at composition length L ∈ {5, 10, 50, 200} (Illusion of State setup, arXiv:2404.08819) the two variants will exhibit **distinguishable accuracy curves at L ∈ {50, 200}** — specifically, the TRM-style variant's accuracy at L=200 will exceed the FPR-style variant's accuracy at L=200 by at least 10 absolute points, **OR** the FPR-style variant's accuracy at L=200 will exceed the TRM-style variant's accuracy at L=200 by at least 10 absolute points (i.e., the variants are not interchangeable; the iteration-scope difference shows up empirically with a sign that the experiment will reveal).

**Conditional on the synthetic probe showing a directional gap** (≥10 points in either direction at L=200), the *Lean prediction* is then: on the long-tactic-chain stratum (L_long: ≥ 50 minimal-proof tactics, PutnamBench-hard + CoqGym), the variant that won the synthetic probe will outperform the loser by a ratio of at least 1.5× the loser's *pilot-measured* L_long pass-rate, with the K=8 vs K=1 gap on L_long larger than the K=8 vs K=1 gap on L_short by a factor of at least 2×.

**This is two predictions chained:** (P1) the synthetic probe will show a ≥10-point directional gap between iteration scopes at L=200 (a Yes/No outcome with a definite sign); (P2) *if* P1 holds, the same direction transfers to Lean L_long with the stated ratio. P2 is conditional on P1.

The honest weakening from revision-0: I do **not** predict in advance which variant wins. The mechanism arguments in revision-0 that made TRM the favorite were factually wrong against FPR's cited paper. What I do predict, on the parameterization-scope grounds in §3, is that the variants are *empirically distinguishable* in a chain-length-dependent way — one of them is meaningfully better at long-chain composition, and the directional gap at L_long is at least 1.5× stronger than at L_short. The experiment is designed to *find out* which scope wins.

## 3. Mechanism

### What the two variants do, exactly

**FPR-style variant** (arXiv:2503.10799 §4.2, Eq. 12-14): the Mamba selective transition is augmented with a depth-iteration index ℓ ∈ {1, ..., K_iter}. At each iteration, the input x_t is updated to x̃_t^ℓ = 𝐐_t^ℓ(x_t − y_t^{ℓ−1}) + y_t^{ℓ−1}, and the matrix state 𝐇_t^ℓ is updated by the standard Mamba scan with this adjusted input. The iteration is *over the channel-mixing component only*; the converged fixed point parameterizes a dense linear RNN (non-diagonal Q^{-1}Λ transition; FPR §3).

**TRM-style variant** (arXiv:2510.04871 §4): a learned block g iterating around a (possibly small) Mamba inner block f_L, with two latent states (z_L, z_H) per token. At each of K_arch outer iterations: z_L ← f_L(z_L + z_H + x), repeated n times, then z_H ← f_H(z_L + z_H), where f_L and f_H share weights (TRM §4.3 "Single network"). Backpropagation is through the full recursion; deep supervision objective per TRM §2.4. The iteration is *over the entire residual stream including z_H*, which carries an evolving "current best answer" beyond the Mamba state.

### What the structural difference is (and is *not*)

The two variants differ in **what gets iterated**, not in **which expressivity class they reach**:

1. **Object iterated.** FPR iterates the channel mixer 𝐐_t (a parameterization of the SSM transition dynamics). TRM iterates a block-level latent (z_L, z_H) that explicitly carries an "answer estimate" through depth. *This is a real architectural difference, conceded by the red-team in §4.3.* However, expressivity-class claims about which can or cannot represent permutation composition are NOT made: FPR-Mamba demonstrably solves S_5 (FPR §4.4 Fig 4), so the contraction-on-the-solver is not an expressivity bottleneck.

2. **Recurrence-depth scaling.** FPR-Mamba's K_iter depth iterations recover the *same dense linear RNN* function — they are an iterative solver for a fixed parameterization (FPR §3). Increasing K_iter improves *solver quality at inference*, not the function class. TRM-style K_arch iterations stack a learned block, accumulating depth in the conventional sense. Whether this difference matters empirically on tasks where iterative refinement of a partial answer is useful (long proof search) is exactly what the experiment measures.

3. **Information channel between iterations.** In FPR, the inter-iteration information is the contracted output y_t^{ℓ−1} fed into the next iteration's mixer (Eq. 12). In TRM, the inter-iteration information is the full latent state including z_H (the answer estimate). *Empirically*, on tasks that are search-like (Lean proof search; iterative answer refinement on Sudoku/Maze where TRM was published) rather than scan-like (regular-language state tracking where FPR was published), TRM's z_H channel may carry more useful signal per iteration. This is a directional hypothesis (if anything, TRM > FPR on Lean), but I do **not** predict the direction with confidence — see §2 (the prediction is "they differ by ≥ 10 points at L=200, sign TBD").

### Why the synthetic probe is the right gate

If the two iteration scopes were equivalent at all composition depths on the synthetic permutation-composition task — the cleanest test of "iterated computation" — then any gap on Lean would be attributable to confounds (parameter count, optimization noise, retrieval interaction) rather than the iteration-scope difference. The synthetic probe at L ∈ {5, 10, 50, 200} (Illusion of State setup) is the right gate because (a) FPR-Mamba's published Figure 4 uses exactly this setup; (b) TRM was originally evaluated on iterative-refinement puzzles (Sudoku, ARC-AGI) which load on depth-iteration — but never on regular-language state tracking; (c) if TRM can be retrofitted onto Mamba but cannot match FPR-Mamba on permutation composition (where FPR-Mamba's Fig 4 reports near-100%), that is the cleanest possible negative signal.

### What I am explicitly NOT claiming (relative to revision-0)

- **No TC⁰-escape claim.** K passes of a TC⁰ block at fixed K stays in TC⁰. Both variants gain expressivity at constant K only via the *new parameterization* (FPR's dense Q⁻¹Λ; TRM's iterated z_H), not via depth-buys-expressivity arguments à la CoT-Solves-Serial.
- **No claim that TRM is "unconstrained."** TRM's deep supervision objective trains the iteration to be attractor-seeking (TRM §4.1). Both variants are trained-to-be-attractor-seeking.
- **No claim that FPR cannot represent permutation composition.** FPR §4 Fig 4 directly shows FP-Mamba solving A_5 and S_5.
- **No claim that frozen Mamba + a wrapper trained alone works.** All params receive gradients under continued pretraining (Retrofitted Recurrence §4).

### Distinction from CoT and from agent scaffolding (unchanged from revision-0)

Both variants are *architectural* — the iteration happens inside one forward pass, not by emitting tokens, not by external proof-search scaffolding. M1 (arXiv:2504.10449), Scaling Reasoning without Attention (arXiv:2505.22425), and Thinking Slow Fast (arXiv:2502.20339) achieve reasoning with SSM/linear-attention substrates via *output CoT*, not depth-time recursion in the forward pass; this hypothesis is orthogonal to all three. Test-time-sampling is held constant across arms.

## 4. Predicted outcome with magnitude

### Two-stage prediction (gated)

**Stage 1 — Synthetic permutation-composition probe (gating).**

Backbones: same Mamba checkpoint, retrofitted via continued pretraining (per Retrofitted Recurrence §4.2 curriculum) to two variants — TRM-Mamba and FPR-Mamba — both at K = 8 mean recurrences. Probe: S_5 composition at L ∈ {5, 10, 50, 200}, train length = 16 (per FPR Fig 4 setup).

| L | TRM-Mamba(K=8) | FPR-Mamba(K=8) | Mamba-K=1 (control) |
|---|---|---|---|
| 5 | ≥ 90% | ≥ 90% | ≥ 90% (all in-distribution) |
| 10 | ≥ 80% | ≥ 80% | ~50% |
| 50 | TBD | ≥ 70% (per FPR Fig 4) | ~5% |
| 200 | TBD | TBD | ~1% (chance 1/120) |

**Headline gating prediction (Stage 1):** at L=200, |Acc(TRM-Mamba) − Acc(FPR-Mamba)| ≥ 10 absolute points. The sign is left to the experiment. Both variants are expected to substantially beat Mamba-K=1 on L ≥ 50 (TC⁰ argument and FPR Fig 4 already establish this for FPR; TRM is the open question).

**Stage 2 — Lean / Coq evaluation, conditional on Stage 1 directional gap.**

If Stage 1 returns a directional gap ≥ 10 points at L=200 (call the winner V_win, loser V_loss):

**Lean ratio prediction:** on L_long stratum (≥ 50 tactics; PutnamBench-hard + CoqGym), pass@1(V_win) / pass@1(V_loss) ≥ 1.5, where the absolute pass-rate is calibrated against the **pilot-measured** L_long Mamba-K=1 baseline (no a priori 2% claim).

**Lean × chain-length interaction prediction:** on L_short (≤ 10 tactics), pass@1(V_win) / pass@1(V_loss) ≤ 1.15. I.e., the ratio gap on long chains is at least ~1.3× the ratio gap on short chains. (Tested via two one-sided ratio tests at α=0.05 each.)

### Pre-registered Lean pilot

Before the main eval, a Lean pilot of ~80 H100-hours: continued-pretrain a 1.4B Mamba checkpoint at K_iter ∈ {1, 4} for FPR-style and K_arch ∈ {1, 4} for TRM-style on a 5B-token math+code subset, fine-tune on Mathlib, evaluate on miniF2F-test (244 problems). The pilot's job:

1. Measure the actual L_long base rate for Mamba-K=1 (replaces the speculative 2% from revision-0).
2. Confirm both retrofit recipes produce non-degenerate accuracy (sanity check on the continued-pretraining recipe transferring from Transformer to Mamba).
3. Estimate per-bin variance to lock in bin sizes for the main eval.

If the pilot returns < 1% L_long pass-rate for any non-control arm, the main eval is canceled (the substrate is too weak for the comparison to be meaningful at the target compute budget).

### Conditions under which the hypothesis should hold
- Same base Mamba-2 checkpoint (arXiv:2405.21060) for both variants.
- Same continued-pretraining tokens, same curriculum, same Muon optimizer (per Retrofitted Recurrence §4.3.1).
- Compute matched within ±10% per arm (FLOPs/token at K=8 recurrence).
- Retrieval cutoff K=100 fixed across arms.
- L_long stratum bin ≥ 100 problems (PutnamBench-hard ~ 60 + CoqGym long-tactic-chain ~ 100 = ≥ 160; powered for ratio detection at 1.5× per the §5 power calc).

### Conditions under which the hypothesis should NOT apply
- If the retrofit recipe (Retrofitted Recurrence §4) does not transfer from Transformer to Mamba — i.e., neither variant's pilot exceeds Mamba-K=1 by a meaningful margin — the comparison is not interpretable (mitigation: pilot gates the main eval).
- If retrieval K saturates the relevant context, both variants collapse to a retrieval problem, and chain-length stratification becomes meaningless (mitigation: F5 retrieval ablation).
- If the Mathlib fine-tuning data has insufficient L_long examples to train either variant to use depth-recurrence on long chains, the prediction degenerates (mitigation: stratified data sampling during fine-tuning).

## 5. Falsification criteria

Each criterion specifies metric + threshold + direction. Power calculations are noted where relevant.

**F1. (Synthetic probe equivalence — primary gate.)** If on the S_5 composition probe at L=200, |Acc(TRM-Mamba(K=8)) − Acc(FPR-Mamba(K=8))| ≤ 5 absolute points (95% CI excluding the 10-point threshold; n=2000 per L bin gives SE ≈ 1 point per arm, so SE on the difference ≈ 1.4 points — well-powered), the iteration-scope-difference hypothesis is falsified at the cleanest level of evidence, and the Lean experiments are *not run*. The two iteration scopes are functionally interchangeable on iterated computation.

**F2. (Lean ratio prediction.)** If Stage 1 returns a directional gap ≥ 10 points at L=200, but Stage 2 returns pass@1(V_win)/pass@1(V_loss) ≤ 1.15 on L_long with 95% CI excluding 1.5, then the iteration-scope advantage seen on synthetic does not transfer to formal proof search. The hypothesis's attribution claim is falsified. (Power: with ≥ 100 L_long problems per arm and pass-rates ≥ 5% per the pilot, a 1.5× ratio with 1.15 lower-bound floor is detectable at α=0.05 per a one-sided ratio test.)

**F3. (RWKV-7 dominance.)** If RWKV-7-Goose(K=1) (arXiv:2503.14456, off-the-shelf 1.5B/3B) on L_long matches or exceeds the better of {TRM-Mamba(K=8), FPR-Mamba(K=8)} within 2 absolute points (95% CI), then RWKV-7's internal vector-valued ICL-rate construction provides equivalent state-tracking capability "for free" without any depth-recurrence retrofit, and the entire premise (depth-recurrence retrofit on Mamba is the right approach) is falsified. **No fallback narrative.** This is a clean F.

**F4. (Coherence — non-monotonicity in K.)** If at K_arch ∈ {2, 4, 6, 8} the L_long pass-rate is non-monotone for the winning variant (the K=4 result is best by ≥ 3 points, and K=8 is worse by ≥ 3 points), the depth-recurrence framing is suspect at the regime tested — additional K_arch iterations should not hurt under a continued-pretraining curriculum that explicitly trained for K up to 8 (per Retrofitted Recurrence §4.2). Failure mode: optimization issue or curriculum mis-specification.

**F5. (Retrieval-dependence sanity.)** If results hold at retrieval K=100 but reverse at retrieval K=200 (V_win at K=100 ≠ V_win at K=200, or the ratio collapses), the iteration-scope advantage is a retrieval-compensation artifact, not a state-tracking advantage. Falsifies the attribution.

**F6. (Substrate-specific story — NEW from I5.)** If a parameter- and FLOP-matched Llama-3.2-1B + TRM-style retrofit (per Retrofitted Recurrence §4.3.2) shows the *same* directional gap on L_long with a similar ratio, the SSM-specific framing (the comparison being interesting because Mamba's diagonal mixer creates the relevant gap structure) is falsified — the result is a generic "block-level recursion vs solver-iteration on the transition mixer" effect, not a Mamba-specific effect. (Note: this requires also implementing FPR-style on a Transformer, which is non-trivial; if intractable, F6 is reduced to "TRM-style retrofit on Llama-3.2 produces similar L_long pass-rate to TRM-Mamba" as a weaker check on substrate-specificity.)

## 6. Required experiments (sketch — eval-designer fills in)

- **Synthetic gate (Stage 1, mandatory).** S_5 composition probe at L ∈ {5, 10, 50, 200}, train length = 16 (FPR Fig 4 setup). Backbones: same continued-pretrained Mamba checkpoint with TRM-style and FPR-style retrofits, plus Mamba-K=1 control. Linear probe or full-classifier head per FPR §4.4. Expected runtime: ~40 H100-hours (small models, synthetic data).
- **Lean pilot (mandatory if Stage 1 passes).** ~80 H100-hours per the §4 spec; goal is to lock magnitudes and bin sizes.
- **Main Lean eval (only if pilot passes sanity).**
  - **Backbones.** Mamba-2 1.4B (arXiv:2405.21060) checkpoint, retrofitted via continued pretraining at curriculum [1→8] mean recurrences over 50B tokens of math+code (Nemotron-CC-Math-v1 per Retrofitted Recurrence §4.3.2), then Mathlib fine-tuned. Three retrofit arms: TRM-style (block-level), FPR-style (transition-mixer-iterating), and a no-recurrence control.
  - **Comparators.** RWKV-7-Goose (1.5B / 3B); Llama-3.2-1B + TRM-style retrofit (per Retrofitted Recurrence published recipe; tests F6); ReProver baseline (arXiv:2306.15626) for retrieval-only sanity floor.
  - **Datasets.** PutnamBench Lean 4 (~640 problems), miniF2F-test Lean 4 (244 problems), CoqGym (arXiv:1905.09381, ~70K Coq proofs; sample ~500 long-chain proofs for L_long).
  - **Stratification.** Bin by minimal-known-proof tactic count: L_short ≤ 10, L_med 11-49, L_long ≥ 50, with ≥ 100 problems per arm per bin on L_long (CoqGym backfills if needed).
  - **Test-time-sampling held constant.** Greedy + beam=4 across all arms.
  - **Compute-matching.** FLOPs/token at K=8 matched within ±10% across TRM-Mamba and FPR-Mamba.

- **Comparison axes (the deliverables).**
  - K-sweep (K_arch / K_iter ∈ {1, 2, 4, 8}) per arm per bin.
  - Pass@1 and Pass@8 per arm per bin.
  - Ratio test V_win/V_loss per bin (with 95% CIs).
  - F6 substrate-control: TRM-Llama-3.2 vs TRM-Mamba on L_long.

## 7. Cheaper falsification path (now also the primary gate)

The synthetic permutation-composition probe (S_5 at L ∈ {5, 10, 50, 200}, FPR Fig 4 setup) is now both the cheapest falsification *and* the gating prerequisite for any Lean compute. If F1 fires (the variants differ by ≤ 5 absolute points at L=200), the entire hypothesis dies for ~40 H100-hours of compute — no Mathlib fine-tuning, no main eval. This is a 50× compute reduction over the full eval.

The probe is the cleanest possible test of the iteration-scope distinction because:
- Permutation composition is the canonical state-tracking task on which both FPR and the SSM-TC⁰ literature agree the diagonal substrate fails.
- FPR's Figure 4 has already done the FPR-Mamba arm; we only need to run the TRM-Mamba arm and the matched control.
- No formal-proof confounds (retrieval, premise selection, Lean compiler errors, etc.) — it isolates iterated-computation capacity.
- Train length = 16 with eval at L ∈ {50, 200} stresses length generalization, which is exactly what depth-recurrent iteration is supposed to deliver.

## 8. Risks to the hypothesis

**R1. Both retrofits work equivalently well on Lean despite differing on synthetic.** The synthetic probe might separate the variants because permutation composition specifically matches FPR's design (dense linear RNN), while Lean tactic prediction is dominated by retrieval/premise selection where the iteration-scope difference is invisible. *Contribution if so:* the experiment cleanly factors retrieval from state-tracking on Lean, which the field currently conflates.

**R2. Neither retrofit transfers to Mamba.** Retrofitted Recurrence (arXiv:2511.07384) was published only on Transformers (TinyLlama, OLMo, Llama-3.2). The continued-pretraining recipe may not transfer cleanly to a Mamba substrate (different per-token compute structure, different scan parallelism). *Contribution if so:* a documented negative result on retrofit transfer from Transformer to Mamba is itself useful — it tells the field that the looped-Transformer literature does not generalize to SSM substrates without modification. Mitigation: pilot gates main eval.

**R3. Parameter-count confound dominates.** TRM-style and FPR-style differ in trainable parameter count even at matched FLOPs. *Contribution if so:* F6 (substrate control via Llama-3.2 + TRM-style) and the FLOP+param double-match grid expose this, and the experiment becomes a methodology contribution: "depth-recurrence retrofit comparisons must control for both compute and parameter count or they're confounded." Mitigation: F6 + reported triple-grid.

**R4. RWKV-7 dominates and the whole framing dies (F3 fires).** *Contribution if so:* a clean negative result on Mamba-retrofit-vs-RWKV-7-internal-fix on formal proof, which has never been published. Honest: this is a real falsification, not a fallback.

**R5. The synthetic-Lean correspondence breaks (F2 fires).** Stage 1 separates the variants but Stage 2 doesn't. *Contribution:* documents that synthetic state-tracking probes do not predict formal-proof-search performance — this is itself important methodological evidence for the field's reliance on regular-language probes.

## 9. Sources

All cited arxiv IDs were resolved via `hf_papers paper_details` during preparation:

- TRM (Tiny Recursive Model) — arXiv:2510.04871 ✓ (re-read §4.1, §4.3 for revision)
- Fixed-Point RNNs — arXiv:2503.10799 ✓ (re-read §4 / §4.2 / Fig 4 for revision; revision-0 misrepresented this)
- HRM — arXiv:2506.21734
- **Retrofitted Recurrence — arXiv:2511.07384 ✓** (NEW citation; Nov 2025 Transformer continued-pretraining recipe; basis for the new training protocol)
- **Think-at-Hard — arXiv:2511.08577 ✓** (NEW citation; Nov 2025 looped-Transformer with selective LoRA iterations; architectural precedent)
- Huginn / latent recurrent depth — arXiv:2502.05171
- Mamba — arXiv:2312.00752
- Mamba-2 — arXiv:2405.21060
- RWKV-7 Goose — arXiv:2503.14456
- Illusion of State in SSMs — arXiv:2404.08819
- Computational Limits via Circuit Complexity — arXiv:2412.06148
- Negative Eigenvalues for State Tracking — arXiv:2411.12537
- SD-SSM — arXiv:2412.19350
- Mamba COPY bound — arXiv:2410.03810
- CoT Solves Inherently Serial Problems — arXiv:2402.12875 (now cited only for context, NOT as TC⁰-escape mechanism)
- PutnamBench — arXiv:2407.11214
- miniF2F — arXiv:2109.00110
- CoqGym — arXiv:1905.09381
- LeanDojo — arXiv:2306.15626
- M1 — arXiv:2504.10449
- Scaling Reasoning without Attention — arXiv:2505.22425
- Thinking Slow Fast — arXiv:2502.20339

**Citations explicitly DROPPED from revision-0 (because the load they bore is no longer claimed):**
- arXiv:2402.12875 was cited as a TC⁰-escape mechanism for constant-K depth iteration. That claim is dropped per C2.

**Citations whose interpretation is corrected:**
- arXiv:2503.10799 (FPR): the contraction is on the depth-iteration solver, not on the function class. FPR-Mamba does solve A_5 / S_5 (Fig 4). The function class at the fixed point is *dense* linear RNN, strictly more expressive than diagonal Mamba.
- arXiv:2510.04871 (TRM): deep supervision trains the iteration to be attractor-seeking; TRM is not "unconstrained" relative to FPR in any sense that matters for the hypothesis.

Internal references:
- Gap-finder-1 output: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/gap-finder-1/output.md` (Gaps 2 and 7)
- Gap-finder-2 output: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/gap-finder-2/output.md` (Gap 5)
- Red-team round 1 critique: `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-2/output.md`
- Revision-0 (this hypothesis): `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-2/output.md`
