# Red-Team Critique of H2 — Revision 1

## 1. Verdict

**REJECT (revision-2)**

The smith made substantial, *honest* repairs to the round-1 critical objections. C1 (FPR misrepresented) and C2 (TC⁰-escape over-extrapolation) are now properly retracted with the load-bearing claims dropped — this is rare and admirable in revision. However, **the revision introduces a new critical defect** while attempting to fix C3, and the resulting hypothesis has weakened to a degree that it is approaching but not yet at the boundary of "let's run the experiment and see." The new defect is a citation misuse of arXiv:2511.07384 that exactly mirrors the original C1 defect: the "published recipe" the smith adopts to justify an unprecedented training protocol is being applied to architectures that paper does not study and to a recursion modality (TRM-style two-state z_L/z_H with deep supervision; FPR-style transition-mixer iteration) the paper does not implement. This is a critical-severity issue that needs surgery before approval.

The orchestrator asked specifically: is this still a hypothesis in the falsification sense? My answer: **marginally yes**, because the synthetic probe gating prediction (≥10 absolute points at L=200, sign TBD) commits to a *magnitude* even with no a priori sign, and the chain-length × variant interaction prediction (P2 ratio ≥ 1.5×, ratio gap on long ≥ 1.3× ratio gap on short) supplies a directional prediction at a granularity. But the magnitude prediction at L=200 lacks proper grounding (see §4) and I worry the smith has anchored to "≥10 points" without a power calculation justifying that specific threshold over, say, 5 or 20.

Verdict line at end of file.

## 2. Round-1 critical objections — status check

| Round-1 objection | Status in revision-1 | My assessment |
|---|---|---|
| **C1.** FPR mischaracterized (contraction is on solver, not function class; FPR-Mamba solves S_5/A_5) | Acknowledged, retracted, mechanism rewritten | **Properly addressed.** The §3 discussion now correctly states the contraction is on the depth-iteration solver and that the converged fixed point parameterizes a dense linear RNN. The "FPR cannot represent permutation composition" claim is gone. |
| **C2.** CoT-Solves-Serial doesn't generalize to constant-K depth iteration | Acknowledged, retracted, TC⁰-escape framing dropped | **Properly addressed.** §3 now explicitly says "K passes of a TC⁰ block at fixed K stays in TC⁰." The expressivity-class argument is gone. |
| **C3.** Frozen Mamba + TRM wrapper is unprecedented training recipe | Replaced with claim that "Retrofitted Recurrence (arXiv:2511.07384) provides the recipe" | **NOT addressed; transformed into a new C-grade defect.** See §3.1 below. The cited recipe applies to a different architecture class (Geiping-style Prelude→RecurrentBlock→Coda, single-state iteration) than either TRM-style (two-state z_L/z_H, deep supervision) or FPR-style (transition-mixer-only iteration). The smith has substituted the appearance of precedent for an actual precedent. |
| **C4.** Magnitude calibration extrapolated from regular-language to Lean | Replaced absolute predictions with ratio predictions; added Lean pilot | **Substantially addressed.** The pilot mechanism is sound. The headline ratio prediction (≥1.5×) is base-rate-robust. **However**, the synthetic probe magnitude (≥10 absolute points at L=200) is itself a new uncalibrated absolute number — see §4. |
| **I1.** Mamba has no Lean pretraining | Folded into C3/C4: continued pretraining on math+code corpus per Retrofitted Recurrence | Addressed at the level of the proposal (every arm gets the same continued pretraining). Practical concern remains: even with continued pretraining, Mamba on Lean is unprecedented and the pilot may return near-zero on L_long. |
| **I2.** F1 statistical underpowering | Bin sizes increased to ≥100; ratio threshold; CoqGym backfills | Addressed. Power calc is stated for F2 (1.5× ratio at α=0.05 with ≥100 problems). |
| **I3.** F3 unfalsifiable fallback | Fallback removed, F3 restated cleanly | **Addressed.** The §8 R3 retreat is gone; F3 now reads as a clean falsification. |
| **I4.** TRM also attractor-seeking | Acknowledged; mechanism story no longer claims TRM is unconstrained | **Addressed.** §3 explicitly states both variants are trained-to-be-attractor-seeking. |
| **I5.** Param-count confound | F6 substrate-control added (Llama-3.2-1B + TRM-style retrofit) | Addressed in principle; F6 is structurally clean but operationally ambiguous (see §3.5). |
| **S1.** Adjacent prior art (arXiv:2511.07384, arXiv:2511.08577) | Cited and folded into mechanism | Addressed; the smith engages with the looped-Transformer literature. |

**Net:** All four CRITICAL items from round 1 are addressed at the level of citation interpretation, but C3's repair introduces a new critical defect (§3.1).

## 3. Mechanism / Citation critique on the *new* claims

### 3.1 Retrofitted Recurrence recipe — citation MISAPPLIED [CRITICAL — NEW DEFECT]

I read arXiv:2511.07384 (McLeish et al., Nov 2025) §3 (Experimental Setup, Model Definition) and §4 (Training Recurrent Language Models) verbatim. The actual recipe is:

> "Given vocabulary set V, for an input sequence x ∈ V^n and a number of recurrences r, the model output distribution p is defined as follows.
> Prelude: e = P(x)
> Recurrent Block: s_0 ∼ N(0, σ²)^{n×h}, s_i = R(e, s_{i−1}) for i ∈ {1, ..., r}
> Coda: p = C(s_r)"

— arXiv:2511.07384 §3, Model Definition

**This is the Geiping et al. (2025) Huginn-0125 architecture**: a single-state iteration s_i = R(e, s_{i−1}) where R is a *set of unique transformer blocks* (not a single iterated layer), and the iteration is over the entire residual stream after a "prelude" of dedicated transformer layers. The iteration is depth-recurrent over a **stack of transformer layers**, evaluated through truncated backprop over the last 8 iterations.

This is **NEITHER**:
- **TRM-style** (arXiv:2510.04871, §4): TRM iterates a learned block over **two distinct latent states** z_L (low-level) and z_H (high-level "answer estimate"), with f_L and f_H sharing weights, and uses **deep supervision** with explicit cross-entropy at each recursion step. The iteration structure is z_L ← f_L(z_L + z_H + x) repeated n times, then z_H ← f_H(z_L + z_H). There is no "prelude/coda" decomposition, no s_i = R(e, s_{i−1}) form, and no Poisson-Lognormal sampling of recurrence count from a curriculum. This is a structurally different recursion modality.

- **FPR-style** (arXiv:2503.10799, §4.2): FPR iterates **only the transition mixer Q_t** of a Mamba scan, with the iteration index ℓ being a Banach-fixed-point solver index — not a stack-of-layers depth index. The hidden state H_t^ℓ at each ℓ is not "the residual stream after r transformer blocks" but "the SSM matrix state after ℓ rounds of solving the implicit transition mixer equation."

**What this means for the hypothesis:**

The smith claims "all backbone params receive gradients during continued pretraining (a) … curriculum: linear schedule of mean recurrence count from 1 → 8 over the first 75% of continued pretraining (b) … Muon optimizer (c)" and presents this as a *published* recipe. But:

1. The published recipe (a)-(c) was demonstrated only on the Geiping architecture, applied to *Transformer* substrates (TinyLlama, OLMo, Llama-3.2). The smith is proposing to apply it to (i) a Mamba substrate (never published) AND (ii) two recursion modalities (TRM-style and FPR-style) the paper does not study.

2. In particular, TRM's deep-supervision objective **is not** in the Retrofitted Recurrence training scheme. Deep supervision per TRM §2.4 involves auxiliary cross-entropy losses at intermediate recursion steps. Retrofitted Recurrence has only the final cross-entropy at s_r. Whether the smith intends to keep TRM's deep supervision (which contradicts "the recipe") or drop it (which contradicts "TRM-style") is **unspecified**.

3. The Poisson-Lognormal r-sampling and truncated 8-step backprop in Retrofitted Recurrence are designed for the Geiping single-state iteration. They are not natively compatible with FPR's solver-iteration semantics (which use a residual-norm convergence check, see arXiv:2503.10799 D.3 heuristics) nor with TRM's two-state block iteration with deep supervision.

**Severity: CRITICAL.** The original C3 defect ("frozen Mamba + TRM wrapper is unprecedented") was correctly diagnosed. The fix attempts to invoke a published precedent that does not actually cover either of the two architectures the hypothesis is about. This is the same shape of error as the original C1 (claiming a paper's result supports something it does not). The smith needs to:
- Either explicitly document which parts of Retrofitted Recurrence transfer (the broad "continued pretraining + curriculum + Muon" outline) vs. which are reinvented for TRM-style and FPR-style (deep supervision retention; the actual recursion semantics; gradient flow through depth iteration), and concede the experiment includes a substantive recipe-engineering contribution; OR
- Acknowledge that the recipe transfer is an open assumption and state R2 (recipe transfer fails) as a non-trivial risk that the pilot will detect, *without* claiming this is a "published" recipe.

The current language ("This is no longer a novel training recipe — it is the published recipe re-applied to a Mamba substrate") is misleading and should be retracted.

### 3.2 FPR Fig 4 magnitude grounding for Stage 1 prediction — IMPORTANT

The smith's Stage 1 prediction table claims:

| L | TRM-Mamba(K=8) | FPR-Mamba(K=8) | Mamba-K=1 |
|---|---|---|---|
| 50 | TBD | ≥ 70% (per FPR Fig 4) | ~5% |
| 200 | TBD | TBD | ~1% (chance 1/120) |

I verified arXiv:2503.10799 Appendix D.2 (Experimental Details, State Tracking):

> "We train the model for sequence length 16 on the train sample, and evaluate for sequence lengths 2 through 50 on the test sample."

— arXiv:2503.10799 §D.2

The main A_5/S_5 setup in FPR has a **maximum eval length of 50**, not 200. There is a **separate** experiment in §E.2 with train length 128, eval up to 512 — but this uses a different setup with 1-layer FP-Mamba2 at sequence length 128. The smith's prediction "FPR-Mamba ≥ 70% at L=50 with train length 16 setup" is plausible from FPR Fig 4 numbers, but **the L=200 prediction has no published anchor** in the train-length-16 regime. The smith correctly marks the FPR-Mamba L=200 cell as TBD, but then asserts ≥10 absolute points difference at L=200 as a headline. Without grounding in either FPR's published data or a calibration argument, the choice of "10 points" (vs. say 5 or 20) is arbitrary.

**Recommendation:** The smith should either (a) ground the threshold in a power-calculation that says "at the n=2000 per L bin, the minimum detectable effect at α=0.05 with 80% power is ≈X absolute points; we set the threshold at 2X for safety," (b) train-length 128 instead of 16 (matching FPR's E.2 setup which has published L=512 numbers), or (c) acknowledge the threshold is a soft pre-registration to be revised by the pilot.

**Severity: IMPORTANT.** The Stage 1 prediction is the gating prediction. If its magnitude threshold is arbitrary, the gate is itself underjustified.

### 3.3 The "iteration scope" argument now does not predict a sign — degenerate hypothesis check

The smith honestly admits in §2: "I do **not** predict in advance which variant wins." The only structural directional prediction is in §3 point 3:

> "Empirically, on tasks that are search-like (Lean proof search; iterative answer refinement on Sudoku/Maze where TRM was published) rather than scan-like (regular-language state tracking where FPR was published), TRM's z_H channel may carry more useful signal per iteration. This is a directional hypothesis (if anything, TRM > FPR on Lean), but I do not predict the direction with confidence."

This is a soft directional intuition rather than a load-bearing prediction. **Is the hypothesis still a hypothesis?**

The orchestrator's test cases: "It predicts a specific magnitude of distinguishability" — yes, ≥10 points at L=200. "It commits to a direction-of-difference at SOME granularity" — partially yes, the *L_long-vs-L_short ratio gap* commitment (≥1.3× larger ratio on long chains than short) is a chain-length-by-variant *interaction*, which is a legitimate directional prediction even when the marginal sign is unspecified. "The non-additivity claim survives" — yes, the L_long ratio ≥ 1.5× and L_short ratio ≤ 1.15 prediction is non-trivial and could fail.

**My assessment:** The hypothesis is at the lower boundary of "still a hypothesis." It survives the test, barely. The directional commitment lives in the **chain-length interaction**, not in the marginal difference. This is acceptable but the smith should make this more explicit: the *headline* prediction is the interaction, not the L=200 magnitude. Currently §2 reads as if the marginal prediction is headline.

### 3.4 P2 chain-length interaction prediction: under-specified vs. parametric noise

The smith predicts "pass@1(V_win) / pass@1(V_loss) ≥ 1.5 on L_long" and "≤ 1.15 on L_short" and the gap of these gaps is ≥ 1.3×. With pass-rates measured as ratios on bins of ≥100, and pilot-measured base rates not yet known, there is a real possibility the L_short bin has very high pass-rates (near 1.0) where ratio compression makes "≤ 1.15" trivially satisfied, or near-zero rates where ratio is undefined. The smith should specify (a) what happens if Mamba-K=1 on L_short is already near-saturating (e.g., 80% pass@1) — the ratio prediction becomes meaningless, and (b) what happens if any arm hits a floor (<1%) on L_long — the ratio is unstable.

**Severity: IMPORTANT.** This is a falsification-criterion-operationalization concern that the smith should address with a pre-registered fallback (e.g., shift to log-odds-ratio if base rate is in the saturating regime; report odds ratios with Wilson CIs).

### 3.5 F6 substrate-control: ambiguity in fallback

The smith writes:

> "If a parameter- and FLOP-matched Llama-3.2-1B + TRM-style retrofit ... shows the same directional gap on L_long with a similar ratio, the SSM-specific framing ... is falsified. (Note: this requires also implementing FPR-style on a Transformer, which is non-trivial; if intractable, F6 is reduced to 'TRM-style retrofit on Llama-3.2 produces similar L_long pass-rate to TRM-Mamba' as a weaker check on substrate-specificity.)"

This is the right shape of falsification, but the fallback ("if intractable") makes F6 partially unfalsifiable: the smith effectively reserves the right to weaken F6 if implementation difficulty arises. The eval-designer will not know whether F6 fires under the original or fallback specification until midway through the experiment. The smith should either commit to running both retrofits on the Transformer substrate (even at lower scale) or specify what counts as "intractable" pre-registered (e.g., "if the FPR-on-Transformer pilot diverges in 10k steps, fall back").

**Severity: SUGGESTION.** Not blocking but should be tightened.

### 3.6 Mechanism §3 point 3 — informal language

> "TRM's z_H channel may carry more useful signal per iteration."

This is the only mechanism-level argument left for why the variants would differ, and it's "may." The smith earlier said honestly that the strong mechanism story is dead. But the rephrasing as "this is exactly what the experiment measures" pushes the burden of justification onto the experiment without supplying any independent argument.

This is fine *if* the synthetic probe is accepted as a clean test, which it more or less is (FPR's Fig 4 gives the FPR-Mamba arm; we just need the TRM-Mamba arm). But the smith should be honest in §2 that the mechanism story is a "soft directional hint, with the experiment as the actual evidence" rather than implying §3 supplies a real prediction.

**Severity: SUGGESTION.**

## 4. Falsifiability assessment (revision-1)

| Criterion | Operationalizable? | Concerns |
|---|---|---|
| F1 (synthetic probe equivalence — primary gate) | Yes; threshold ≤5 absolute pts at L=200, n=2000 per bin | Magnitude threshold (10 pts) is uncalibrated against any prior. Power calc states SE on diff is 1.4 pts, so 5 pts is detectable, but "why 10?" is not justified. |
| F2 (Lean ratio prediction) | Yes; ratio ≤1.15 with 95% CI excluding 1.5 | Operationally fragile if base rate saturates or floors (§3.4). Power calc cited but assumes pass-rates ≥5% (pilot must confirm). |
| F3 (RWKV-7 dominance) | Yes; clean | Fixed from round 1. Good. |
| F4 (non-monotonicity in K) | Yes | Reasonable. K=4 best by ≥3 pts AND K=8 worse by ≥3 pts is a tight specification. |
| F5 (retrieval-dependence) | Yes | OK. Attribution test, not primary. |
| F6 (substrate control) | Partially | Fallback reservation makes the criterion soft. Should be tightened. |

**Overall:** The falsification suite is the strongest part of the hypothesis (as in round 1). F1 with the synthetic-probe gate is genuinely powered to kill the hypothesis cheaply. The main weaknesses are (a) F1's threshold is uncalibrated and (b) F6's fallback is not pre-registered.

## 5. Strongest counter-argument (steelman)

**"The revised H2 has retreated to compute-scaling claims dressed up as architecture comparison."**

Specifically:
1. The mechanism story is now: "two iteration scopes will produce empirically distinguishable accuracy curves on a long-chain task." This is a near-tautological claim — *any* two non-identical architectures with sufficient training will produce empirically distinguishable accuracy curves on *some* benchmark.
2. The interesting question is whether the distinction *predicts* something (a direction, a mechanism, a generalizability claim). The smith disclaims the direction. The mechanism is "iteration scope" without a prediction of which scope wins. The generalizability claim (F6) covers Transformer-Llama only.
3. The synthetic probe's 10-point gating threshold is the only quantitative load-bearing prediction left, and it's uncalibrated.

Under this steelman, the experiment is still useful — it would establish *empirically* whether TRM-style and FPR-style retrofits are equivalent on iterated computation on a Mamba substrate. But it's borderline whether this rises to a Tier-1 novel result (the spec novelty target was workshop-publishable). The honest framing might be: "this is a methodology paper documenting the comparison, not a hypothesis paper."

**Counter-counter:** The chain-length-by-variant interaction prediction (P2 ratio gap on long ≥ 1.3× ratio gap on short) is a non-trivial directional commitment. It predicts that the difference between the variants *grows with chain length*, which is mechanism-consistent with the iteration-scope-difference story (more iteration matters more on harder problems). This is a genuine prediction and survives.

So the steelman dings the marginal prediction but the interaction prediction holds up.

## 6. Severity-tagged objections

### Critical (must fix before approval)

**C1 (revision-1).** The Retrofitted Recurrence (arXiv:2511.07384) recipe applies to a different architecture class (Geiping/Huginn Prelude→RecurrentBlock→Coda, single-state s_i = R(e, s_{i−1})) than either TRM-style (two-state z_L/z_H with deep supervision) or FPR-style (transition-mixer-only Banach-fixed-point iteration). The smith's claim "this is no longer a novel training recipe — it is the published recipe re-applied to a Mamba substrate" is false — the recipe has been published only for one recursion modality applied to Transformer substrates. Apply this recipe to TRM-style or FPR-style on Mamba is **doubly novel** (new recursion modality + new substrate). The smith must either retract the "published recipe" framing and explicitly state R2 (recipe transfer fails) as a real risk that the pilot is designed to detect, or document specifically how Retrofitted Recurrence's training scheme is being adapted to TRM-style (deep supervision retained or dropped?) and FPR-style (Banach convergence semantics under Poisson-Lognormal r-sampling?). (See §3.1.)

### Important (should fix)

**I1 (revision-1).** The Stage 1 magnitude threshold (≥10 absolute points at L=200) is uncalibrated. Why 10 and not 5 or 20? With n=2000 per bin and SE on the difference of 1.4 points, the minimum detectable effect at α=0.05, power 0.8 is ~4 points — so a threshold at 10 is conservative but unjustified. Provide either a power-calculation-based justification, or anchor the threshold in published FPR data at a comparable L (which requires running at train-length-128 setup, where FPR has published numbers up to L=512). (See §3.2.)

**I2 (revision-1).** The L_long / L_short ratio prediction is operationally fragile under base-rate saturation or floor effects. Specify a pre-registered fallback for: (a) Mamba-K=1 on L_short is ≥80% (ratio compression makes ≤1.15 trivially satisfied); (b) any arm <1% on L_long (ratio unstable). Recommend log-odds-ratio with Wilson CIs as the robust alternative. (See §3.4.)

**I3 (revision-1).** F6 substrate-control fallback ("if intractable") makes the criterion partially unfalsifiable. Pre-register what counts as "intractable" (e.g., FPR-on-Transformer pilot loss diverges in 10k steps; or training cost exceeds 2× budget). Otherwise an eval-designer can quietly invoke the fallback when convenient. (See §3.5.)

**I4 (revision-1).** The hypothesis statement in §2 leads with the marginal prediction (≥10 points at L=200, sign TBD). The actually-load-bearing prediction is the chain-length-by-variant interaction (P2 ratio gap ≥1.3×). The smith should restructure §2 to lead with the interaction and present the marginal as a gating prerequisite, since this is the structure of the contribution. Otherwise the hypothesis reads as "two architectures differ on a benchmark" which the steelman in §5 correctly flags as near-tautological.

### Suggestion (nice to have)

**S1.** §3 point 3 ("TRM's z_H channel may carry more useful signal per iteration") is the only remaining mechanism-level argument for why variants would differ. Make it explicit that this is a soft hint, not a load-bearing claim, and that the hypothesis's evidentiary burden lives in the experiment design rather than the mechanism.

**S2.** Cite Reasoning with Latent Thoughts (arXiv:2502.17416) and Adaptive Loops and Memory in Transformers (arXiv:2603.08391) which are looped-transformer papers studying what the iteration buys. These provide additional context that "iteration scope difference matters" is a plausible empirical claim.

**S3.** The pilot magnitude (~80 H100 hours) and synthetic gate (~40 H100 hours) are well-specified. Recommend the eval-designer surface a *single decision tree* showing kill points so the experimental commitment is clear: probe → kill or pilot → kill or main → result.

## 7. Recommendation to hypothesis-smith

The revision is substantively improved: C1, C2, I3, I4 are properly addressed; the falsification suite is sharpened; the synthetic probe gating is the right shape. **The fix needed for approval is non-cosmetic but tractable** — primarily, retracting the "published recipe" framing for Retrofitted Recurrence and acknowledging the recipe-transfer-to-Mamba is itself part of what the experiment risks testing.

Specific revisions for round 2:

1. **Fix C1 (revision-1).** In §C3 paragraph and §3 mechanism discussion, replace "this is no longer a novel training recipe — it is the published recipe re-applied to a Mamba substrate" with: "we adapt Retrofitted Recurrence's continued-pretraining+curriculum+Muon outline to two recursion modalities the paper does not study (TRM-style and FPR-style) on a substrate the paper does not study (Mamba). Whether the broad recipe (curriculum schedule, optimizer, FLOP accounting) transfers is part of what the pilot tests." Document explicitly: deep supervision retention/dropping for TRM-style; Banach convergence semantics for FPR-style under Poisson-Lognormal r-sampling.

2. **Fix I1 (revision-1).** Justify the 10-point threshold either by power calculation, by anchoring in FPR published numbers at a matching setup (i.e., train length 128 instead of 16, where FPR has L=512 published), or by pre-registering the threshold as soft (revisable by the synthetic pilot).

3. **Fix I2 (revision-1).** Add a pre-registered fallback for ratio prediction under saturation/floor.

4. **Fix I3 (revision-1).** Pre-register what counts as F6 fallback being invoked.

5. **Strengthen I4 (revision-1).** Restructure §2 to lead with the chain-length-by-variant interaction prediction, since that's where the directional commitment actually lives.

If these fixes are applied, the hypothesis is approvable. **The hypothesis is not in KILL territory** — the gap survives, the falsification suite is operational, and the synthetic probe gate provides a genuine cheap kill at ~40 H100-hours. The remaining defects are about citation discipline and pre-registration tightness, not about the underlying scientific shape of the inquiry.

---

REJECT (revision-2)
