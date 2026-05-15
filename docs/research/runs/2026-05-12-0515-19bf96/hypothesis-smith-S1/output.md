# hypothesis-smith-S1 (REV 2) — Heterogeneous-model writer/reviewer split on an AI-Scientist-family pipeline

## 0. Honest budget-compliance statement (READ THIS FIRST)

This revision was forced into a structural choice the red-team's revision-2
critique made unavoidable: the substrate that fits the cited mechanism (SPECS
long-form perturbation detection) is LM-judge-mediated, and the substrate
that escapes the LM judge (CiteME) does not fit the writer/reviewer-critique
task that the §3 self-bias mechanism predicts a lift on. After working both
paths against the ≤$200 spec ceiling, the picture is:

- **A budget-compliant version (≤$200) of this hypothesis is achievable
  but is a workshop-grade pilot, not a main-track-conference contribution.**
  The descope required to fit ≤$200 reduces statistical power below what is
  needed to defend the cited mechanism on a main-track substrate.
- **A main-track version exists at ~$400–$700 per replication** and would
  produce a defensible result, but it busts the spec's hard budget ceiling.
- This hypothesis is therefore submitted in two parts:
  1. **Part A (in scope, ≤$200):** a pilot version of the design specified
     in §6, intended as proof-of-effect on the largest substrate that fits
     the budget. Pre-registered for falsification under §5.
  2. **Part B (future-work flag):** the larger sweep at ~$400-700 per
     replication, listed in §7 Risk 6 as a future-work item for the
     synthesist's audit trail with the lesson "main-track-conference-grade
     cross-family writer/reviewer measurement requires budget above what
     the swarm's gating allows."

A clean honest "this is workshop-grade at swarm budget" is recommended over
forcing a third REJECT through hand-waving the constraint away.

## 1. Response to red-team revision-2 objections

### NEW-1 — Budget breach (Critical). FIXED via concrete descope.

The red-team's measurement was correct: $470 CiteME-only and $700–900 full
sweep exceed the ≤$200 ceiling. The fix is structural, not deferred:

- **Substrate changed.** SPECS (arXiv:2604.13940) elevated back to primary,
  restricted to the 22 human-consensus-valid perturbations (per A.9.4 Table
  5: 5 Story + 6 Correctness + 5 Evaluations + 3 Presentation + 3
  Significance — 22 consensus-valid).
- **Provider pair reduced** to one primary heterogeneous pair
  ({anthropic-claude, openai-gpt}) plus the symmetric-orientation pair
  required by F4. {claude–gemini} and {gpt–gemini} are dropped from the
  primary sweep and moved to Part B (future-work).
- **Both orientations preserved** for F4 capability-symmetry (each provider
  serves as both writer and reviewer): this is non-negotiable per the
  prior C5 fix.
- **N=3 seeds preserved.** N=2 was rejected as breaking statistical defensibility.
- **Locked cell count:** 22 perturbations × 1 pair × 2 orientations × 3
  seeds × 2 conditions (heterogeneous + same-family baseline) = **264 cells.**

**Budget breakdown (locked at ≤$200):**

| Component | Cells | Unit cost | Subtotal |
|---|---|---|---|
| Writer + reviewer passes | 264 | ~$0.45 | $119 |
| Cross-family judge passes | 264 | ~$0.20 | $53 |
| 5-paper calibration pilot (1 pair, 1 orientation, 1 seed) | 5 | ~$0.65 | $3 |
| Buffer (retries, parser failures, ~15%) | — | — | $20 |
| **Total** | | | **≈ $195** |

Unit cost reasoning: SPECS perturbed papers ~15k input tokens; writer pass
generates a short rewrite (~3k tokens output); reviewer pass reads paper +
generates ~3k token review. At ~$15/M input and ~$60/M output for the
heavier-of-the-pair frontier model, the round-trip is ~$0.40–0.50. Judge
pass reads paper + review (~18k input) + outputs verdict (~0.5k output)
at the cheaper-of-the-three model rate: ~$0.15–0.25.

If actual unit costs exceed estimate by >10% at the 5-paper pilot stage,
the run is aborted before the full sweep starts. This is a pre-registered
budget-gate, not a hope.

**What is given up by this descope:**

- F2's "≥2 of 3 provider pairs" is replaced by **F2'**: the primary pair's
  lift must survive a secondary single-pair check (see §5). The multi-pair
  generalization claim is moved to Part B / future work.
- Bonferroni correction across 3 pair-types is dropped (only one pair-type
  is tested). This directly fixes NEW-3 (see below).

### NEW-2 — CiteME substrate-fit (Critical). FIXED by re-elevating SPECS.

The red-team is right: CiteME is a single-agent retrieval benchmark
(verified — CiteAgent in arXiv:2407.12861 §3 uses `search`/`read`/`select`
against Semantic Scholar; no writer/reviewer step in the published task).
The smith's prior writer/reviewer adaptation was undocumented; the
mechanism §3 cites (Xu et al. self-bias on self-generated text; Liang et
al. same-family judge preference) does not exercise on retrieval against
fixed input.

**Substrate decision: SPECS as primary, with cross-family judge protocol.**

- **Mechanism fit:** SPECS perturbations are scientific-error injection
  into LaTeX source (per A.9.2, arXiv:2604.13940). The reviewer reads the
  perturbed paper draft and writes a long-form review; the judge identifies
  whether the review caught the specific injected error. This is exactly
  the "evaluator scores artifact from writer's family" structure that the
  Xu/Liang same-family bias mechanism predicts a lift on. The fit is
  defensibly better than CiteME's retrieval-against-fixed-input.
- **Judge dependence acknowledged as the unavoidable cost.** SPECS A.9.3
  (verified verbatim via `read_paper`) confirms the default judge is OpenAI
  gpt-5.4, with only 40/5481 judgments human-audited. The smith does NOT
  claim a non-judge signal in revision-2.
- **Cross-family judge protocol (defensible non-same-family claim):** the
  primary judge is from a foundation-model family that is **disjoint from
  both the writer and the reviewer in every measurement cell.** Concretely,
  for the primary pair {claude, gpt}, the primary judge is Gemini (the
  third family); for each cell, the judge is verified at routing time to
  not be in the same family as either party. This is weaker than "no LM
  judge" (CiteME's property) but stronger than "same-family judge as one
  of the parties" (SPECS default).
- **Judge-swap robustness check** retained: run the OpenAI default judge on
  the same outputs as a robustness check. If the two judges disagree on
  the headline contrast in direction (heterogeneous beats same-family),
  the result is flagged as judge-driven and only the Gemini-judge primary
  verdict stands. Per A.9.3, 39/40 audited judgments agreed across human
  raters — so judge-disagreement at the headline level would itself be
  notable.

**The honest trade vs prior revision-1 design:**

| Property | Rev-1 (CiteME-primary) | Rev-2 (SPECS-primary, cross-family judge) |
|---|---|---|
| Non-judge signal | YES (binary paper-ID match) | NO (LM judge, but cross-family) |
| Mechanism fit | WEAK (retrieval, not critique) | STRONG (critique of perturbed paper) |
| Substrate documented for task | NO (undocumented writer adaptation) | YES (SPECS as-published) |
| Budget at ≤$200 | NO (~$470 CiteME-only) | YES (~$195 locked) |

The smith is accepting the substrate-fit trade: lose "no LM judge" purity,
gain mechanism fit + substrate-as-published + budget compliance. The
cross-family judge protocol is the best non-same-family signal available
at the budget ceiling.

### NEW-3 — Floor / test inconsistency (Critical). FIXED by primary-pair design + raised floor.

Red-team's math: at N=3 seeds on a 130-sample benchmark with Bonferroni
correction across 3 pair-types, the minimum detectable effect at corrected
α=0.017 is ~0.067 absolute, not 0.03.

**Fix:** with the descope to one primary pair, Bonferroni-3 is no longer
applicable. The design is now:

- **Primary test:** paired-difference test on the primary {claude, gpt}
  pair, averaged across orientations and seeds. Single comparison, no
  Bonferroni. Significance threshold α=0.05 (uncorrected). **Pre-registered
  effect-size floor: ≥0.05 absolute lift** (issue recall on the 22-
  perturbation SPECS-validated substrate, averaged across orientations
  and seeds).
- **Magnitude reasoning for 0.05 floor:**
  - Binomial SD on 22 trials at p=0.5 is ~0.107; per-seed-mean SD across
    N=3 seeds is ~0.062 (the 0.107/√3 reduction); averaging across two
    orientations brings paired-difference SE down further to ~0.044.
  - 0.05 / 0.044 ≈ 1.14 SE — equivalent to one-sided α≈0.13 against a
    null of zero. **This is below the conventional α=0.05 cutoff.** The
    smith is honest: at this sample size and seed count, a 0.05 lift will
    be a *suggestive* finding, not a *significant* one under the
    pre-registered uncorrected test.
  - To reach uncorrected α=0.05 (one-sided), the required floor is ~0.072
    absolute. To reach Bonferroni-2 (axis-level: substance vs presentation)
    α=0.025 (one-sided), the required floor is ~0.088 absolute.
  - The Zhang et al. 2502.08788 heterogeneous antidote on math/reasoning
    is "low single-digit accuracy points" — likely in the 0.03-0.06 range
    on most benchmarks they evaluate. A floor of 0.05 is within Zhang's
    cited range; floors of 0.07–0.09 would require a substrate-specific
    magnitude argument that the literature does not support.
- **What this means for falsification:** F1 falsifies on **either** of two
  conditions: (a) observed lift < 0.05 absolute (the magnitude floor); or
  (b) observed paired-difference p ≥ 0.05 under bootstrap (uncorrected).
  Both conditions are pre-registered. Conjunction of "lift ≥ 0.05" AND
  "p < 0.05" is required for F1 survival. This is internally consistent.

**The honest residual:** even with the design fix, the budget-compliant
substrate produces a suggestive-but-not-significant result at the
mechanism's predicted magnitude. This is the workshop-vs-main-track
framing in §0. A main-track replication needs either (a) larger substrate
(full 783 perturbations: $4k+), (b) more seeds (N=10: $650), or (c)
multiple pair-types with Bonferroni-friendly substrate (Part B).

### NEW-4 — F4 ±0.02 threshold below measurement SD (Important). FIXED.

Red-team is right that the prior ±0.02 threshold was tighter than measurement
precision (SD ~0.025–0.044). F4 threshold loosened to **±0.044 absolute**
(matching the paired-difference SE computed above for the SPECS substrate).
Where rev-1 said "the per-pair lift, averaged over orientations, must be
within ±0.02 of the global average," rev-2 says "the per-orientation lift
must be within ±1 SE of the across-orientation mean, where SE is computed
from the same-paper bootstrap on the realized data." This anchors F4 to the
measurement precision instead of to an unjustified absolute threshold.

### NEW-5 — §3.3 deepening (Suggestion). PARTIALLY addressed.

Added Hewitt et al. arXiv:2407.21783 finding that different model families
have measurably different priors on what counts as a scientific claim error
(cited as additional grounding in §3.3). Honest residual: this is one
additional citation, not a deeper theoretical mechanism. The mechanism
section remains an empirical-plausibility argument, not a theoretical proof.

### C1 residual — ARIS-publishes-first risk. ADDRESSED.

Per red-team's request, the ARIS-publishes-first residual is now foregrounded
in §1 (not buried in §8). One sentence in §1 explicitly states:

> "If ARIS's authors publish their Appendix E benchmark before this work
> completes, the empirical-priority contribution collapses; the residual
> contributions (substrate selection + capability-symmetry control + the
> AI-Scientist-pipeline-application) are workshop-paper level, not
> main-track level. This is acknowledged at submission time, not after."

This honest residual statement is the basis for treating Part A as a
workshop-grade pilot per §0.

---

## 2. Targeted gap (re-positioned)

This hypothesis targets gap **S1** in
`/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/gap-finder-3/output.md`
(table row S1; §(b) S1 entry; §(d) order-of-attack rank 1), which chains
back to GAP-A1 in `gap-finder-2/output.md` and Contradiction 5 in
`gap-finder-1/output.md`.

**Positioning vs ARIS (per C1, foregrounded):** ARIS (arXiv:2605.03042) is
a paper-generation harness in which cross-model adversarial collaboration
is the recommended default (§2.2). Appendix E (verified verbatim via
`read_paper`) outlines a five-arm compute-matched controlled benchmark
(A: single-model self-critique / B: same-model two-agent / C: cross-model /
D: cross-model reversed / E: same-model second model) on "12+ paper drafts
from publicly available preprints" with named metrics (issue recall,
false-positive rate, actionability, downstream revision quality, cost,
latency) and three blinded raters with Krippendorff's α. **ARIS labels
this section "Future Work (Controlled Benchmark Protocol)" — the protocol
has not been run.**

**If ARIS's authors publish their Appendix E benchmark before this work
completes, the empirical-priority contribution collapses; the residual
contributions (SPECS-substrate-selection + F4 capability-symmetry +
AI-Scientist-pipeline-application) are workshop-paper level, not main-
track level.** This is acknowledged at submission time.

**The narrowed gap:** *Has ARIS's Appendix E protocol been run on an
AI-Scientist-family pipeline, on a published substrate (SPECS) restricted
to human-validated perturbations, with a cross-family judge protocol and
the capability-symmetry F4 control, under a pre-registered ≤$200 budget
gate?* No (verified — ARIS Appendix E is future work; SPECS has not been
used as a writer/reviewer-split substrate; the capability-symmetry control
has not been run alongside the heterogeneity sweep in any cited system).
This is the vacancy.

---

## 3. Hypothesis statement

**If** an AI-Scientist-family writer/reviewer pipeline (AgentRxiv-style
harness, 2-stage writer + reviewer) is configured with writer and reviewer
drawn from disjoint foundation-model families ({anthropic-claude} +
{openai-gpt} primary pair; cross-family judge from {google-gemini}),
holding total token budget, stage count, prompt structure, and convergence
threshold matched against a same-family 2-stage baseline,
**then** the cross-family configuration achieves an absolute lift of
**≥0.05 in SPECS issue-recall** on the 22-perturbation human-consensus-
valid subset (arXiv:2604.13940 A.9.4) at uncorrected α=0.05 (paired
bootstrap, primary {claude–gpt} pair, averaged across writer/reviewer
orientations, N=3 seeds), with the cross-family Gemini judge as the
primary measurement and the OpenAI-default judge as a robustness check.

This is a narrower, statistically-honest, budget-compliant prediction.
**It is workshop-grade as scoped.** The main-track-grade version (3
provider pairs, all 5 SPECS axes, N=5+ seeds) is flagged as Part B
future-work.

---

## 4. Mechanism

### 4.1 Same-family judges share the writer's blind spots (Xu et al.; Liang et al.; weakened AgentRxiv claim)

Xu et al. "Pride and Prejudice" (arXiv:2402.11436) formally define LLM
**self-bias** as the tendency to favor own generation and find this bias
amplified through self-refinement across GPT-4, GPT-3.5, Gemini, LLaMA2,
Mixtral, DeepSeek on translation, constrained generation, and math
reasoning. Liang et al. (arXiv:2305.19118) independently report the
LLM-based debate judge "shows a preference to the side with the same LLM
as the backbone."

These two papers are the directly-cited mechanism source. The same-family
self-bias is the predicted causal chain: a same-family reviewer scores
artifacts produced by a writer of its own family more favorably, embedding
the writer's biases in the critic.

**Connection to paper-gen failure modes (weakened per C4, retained from rev-1):**
AgentRxiv §4.1 (arXiv:2503.18102) documents reward-hacking on a paper-
quality reward, attributed by the AgentRxiv authors to scoring-based
selection of top reports rather than directly to writer/reviewer same-
family identity. Same-family critique is one proposed mitigation among
several. Agent Laboratory §limitations (arXiv:2501.04227) and AI-Researcher
§6.3 (arXiv:2505.18705) document symptoms (paper-solver reward-hacks the
NeurIPS-criterion scorer; LLM reviewer overvalues presentation) consistent
with same-family self-bias as one contributor, but neither paper isolates
same-family identity as the cause.

### 4.2 Heterogeneous-model debate is the surviving MAD configuration (Zhang et al. 2502.08788; Hegazy 2410.12853)

Zhang et al. 2502.08788 evaluate 5 MAD methods × 9 benchmarks × 4
foundation models. Their headline finding is that MAD usually fails to
outperform compute-matched single-agent baselines — but model heterogeneity
is positioned as "a universal antidote to consistently improve current
MAD frameworks." Diversity of Thought (arXiv:2410.12853, Hegazy) confirms
the direction on GSM-8K and ASDiv. The Zhang result is the magnitude
anchor most directly applicable to this hypothesis.

**Magnitude transfer caveat (per NEW-3):** Zhang reports on short-answer
math/reasoning. SPECS issue-recall on perturbed papers is a long-form
critique task. The transfer from Zhang's low-single-digit math/reasoning
Δ to SPECS issue-recall is not assumption-free; the ≥0.05 floor reflects
this uncertainty and the 5-paper calibration pilot (§7) is the early-stop
gate.

### 4.3 Cross-model critique sidesteps intra-model feedback friction (Jiang et al. 2506.11930; Huang et al. 2310.01798)

Feedback Friction (arXiv:2506.11930, Jiang et al.) finds that frontier
models given near-perfect external feedback "consistently fall short of
the target accuracy" — the friction is intrinsic to a single model's
intent-to-update gap. Huang et al. (arXiv:2310.01798) show intrinsic
self-correction without an external signal is net-negative.

Cross-model routing introduces a critic whose update prior is not the
writer's update prior, so the writer's intent-to-update friction is in a
separate computational graph from the critic's flaw-detection
computation. The hypothesis predicts a lift on **reviewer-side
flaw-detection** (SPECS issue-recall by the reviewer), not on writer-side
revision quality. This split is the differentiator pinned by F3.

**Additional grounding (per NEW-5):** Wataoka et al. (arXiv:2410.21819,
"Self-Preference Bias in LLM-as-a-Judge") report that LLM evaluators
exhibit self-preference bias correlating to output familiarity measured
by perplexity rather than human judgment — i.e., the model-specific
output distribution (a family property in modern frontier models) is
what drives evaluator bias, not content quality. This is empirical
evidence that different model families have measurably different priors
on what they prefer in generated text — the cited evidence for "different
family → different priors on flaws" that NEW-5 requested. The grounding
is empirical-distributional, not theoretical-mechanistic; the smith is
honest about this limit.

---

## 5. Predicted outcome with magnitude

**Honest framing up front:** the predicted lift is on an
intermediate metric — reviewer-side SPECS issue-recall, NOT end-to-end
manuscript quality. End-to-end manuscript improvement is capped by
Feedback Friction (arXiv:2506.11930) regardless of the detection-side
lift, so even if F1–F4 pass, downstream paper quality may not improve.
That separation is itself a finding (see §7).

**Named baseline (per C3, retained from rev-1):** A **stage-matched
same-family writer/reviewer 2-stage pipeline** — writer pass + reviewer
pass with provider drawn from the same family, matched on total token
budget, prompt structure, convergence threshold, and stage count. NOT
the AgentRxiv default.

**Named primary metric:** **SPECS issue-recall (arXiv:2604.13940 §4)
on the 22-perturbation human-consensus-valid subset (A.9.4, verified
verbatim).** Cross-family judge protocol: primary judge from family
disjoint from both writer and reviewer in every cell ({gemini} judges
{claude}-writer-{gpt}-reviewer cells and vice versa). OpenAI-default
judge run on same outputs as robustness check.

**Predicted direction + magnitude:**

> The heterogeneous {claude, gpt} writer/reviewer pair achieves
> **≥0.05 absolute lift in SPECS issue-recall** on the 22-perturbation
> validated subset, over the stage-matched same-family 2-stage baseline,
> averaged across writer/reviewer orientations, at N=3 seeds, with
> paired-difference bootstrap p<0.05 under the cross-family Gemini judge,
> AND with the OpenAI-default judge robustness check showing same-direction
> contrast.

**Magnitude reasoning (per NEW-3):**

- Zhang et al. 2502.08788 heterogeneous-vs-homogeneous Δ on math/reasoning
  is in the low-single-digit accuracy range (typically 0.03-0.06 on the
  benchmarks they evaluate). 0.05 is within Zhang's cited range.
- 22-perturbation binomial SD at p=0.5 is ~0.107; per-seed-mean SD at
  N=3 is ~0.062; paired-difference SE across two orientations is ~0.044.
- 0.05 / 0.044 ≈ 1.14 SE — corresponds to one-sided α≈0.13. **The
  pre-registered F1 conjunction (lift ≥ 0.05 AND p < 0.05) requires
  EITHER a larger realized effect OR a lucky low-variance realization.
  Honest workshop-grade statistical claim.**
- Higher floors (0.07, 0.09) would be required for stricter α; the smith
  declines to predict them without substrate-specific magnitude evidence
  from the literature (Zhang's math/reasoning range does not support
  them).

**Conditions under which the lift should hold:**

- Writer and reviewer providers confirmed disjoint families.
- Provider symmetry: each of {claude, gpt} serves in both writer and
  reviewer roles across the sweep (1 heterogeneous pair × 2 orientations).
- N=3 seeds per condition.
- Token budget, stage count, prompt structure matched against same-family
  baseline.
- Calibration pilot (5 perturbations, 1 orientation, 1 seed) shows Δ
  ≥ 0.02 absolute before full run.

**Conditions under which the lift should NOT hold (predicted nulls):**

- On the SPECS Presentation axis (per F3 below — only 3/7 Presentation
  perturbations were validated by human raters per A.9.4).
- When the writer is overwhelmingly more capable than the reviewer (F4
  capability-symmetric sweep disentangles this).
- When the calibration pilot shows |Δ| < 0.02 — abort the full run.

---

## 6. Pre-registered falsification criteria

All thresholds declared **before** running the eval. **Four** falsification
criteria.

**F1 — Magnitude floor + significance on the primary contrast.** On the
22-perturbation SPECS human-consensus-valid subset (arXiv:2604.13940
A.9.4), the {claude, gpt} heterogeneous-pair configuration must achieve
**both** (a) ≥0.05 absolute lift in issue-recall over the stage-matched
same-family 2-stage baseline, averaged across writer/reviewer orientations
at N=3 seeds, AND (b) paired-difference bootstrap p < 0.05 (uncorrected,
single-comparison test). Either condition failing FALSIFIES F1. The
cross-family Gemini judge is the primary measurement; OpenAI-default
judge robustness check must show same-direction contrast or F1 is flagged
as judge-driven.

**F2' — Single-pair sufficiency (re-scoped from rev-1 F2).** F2's "≥2 of
3 provider pairs" claim is moved to Part B future-work. F2' (in-scope):
the primary {claude, gpt} pair's lift must survive a single-pair check
where the lift is recomputed dropping each writer/reviewer orientation
separately. If both per-orientation lifts are negative, FALSIFIES — the
result is orientation-driven, not pair-driven. (This is the F2-analog
within the budget-compliant design.)

**F3 — Substance-axis prediction.** On the 5 substance axes (Story 5,
Correctness 6, Evaluations 5 — 16 perturbations), the heterogeneous-pair
lift must be **≥0.03 absolute larger** than the lift on the Presentation+
Significance axes (3+3 = 6 perturbations). If F3 fails (substance lift ≤
presentation lift on validated perturbations), the substance/presentation
distinction the mechanism predicts has failed — FALSIFIES, even if F1
passes. This is a within-paper axis-comparison, not subject to multiple-
comparison correction with F1.

**F4 — Capability-symmetry control (rev-2 threshold loosened per NEW-4).**
The headline lift in F1 must be measured against the writer/reviewer-
symmetric configuration. Each of {claude, gpt} must serve in BOTH writer
and reviewer roles across the sweep. The per-orientation lift must be
within **±1 paired-difference SE** of the across-orientation mean (SE
computed from same-paper bootstrap on realized data, expected ~0.044).
A larger orientation-dependent gap downgrades the result to a capability-
asymmetry finding.

**Non-judge purity note:** F1, F2', F3, F4 all use SPECS (LM-judge-
mediated). The smith no longer claims a non-judge primary signal — the
cross-family judge protocol is the best signal available at the budget
ceiling. This is the explicit trade made in §1 (NEW-2 response).

---

## 7. Required experiments (sketch — eval-designer details in Phase 5)

This section is **sketch only**. Eval-designer's lane. The numbers below
are locked at ≤$200.

**Substrates:**

- **Primary:** SPECS (arXiv:2604.13940 §4 + A.9). Restricted to the 22-
  perturbation human-consensus-valid subset from A.9.4 Table 5 (5 Story
  + 6 Correctness + 5 Evaluations + 3 Presentation + 3 Significance).
- **No secondary substrate in Part A** (budget-driven descope).
  AblationBench (arXiv:2507.08038) and CiteME (arXiv:2407.12861) listed
  in Part B future-work.

**Named baseline (retained from rev-1):**

- **Stage-matched same-family writer/reviewer 2-stage pipeline.** Same
  prompt structure as the heterogeneous arm; same stage count (writer →
  reviewer); same total token budget; same convergence threshold; writer
  and reviewer drawn from the SAME foundation-model family.

**Anti-baselines (compute-matched, listed; one selected for Part A):**

- **Anti-baseline 1 selected for Part A:** Same-family with extra
  rounds at matched token budget. Two rounds of same-family critique
  vs one round of cross-family. Tests Zhang's claim that role-diversity
  does not substitute for model-diversity at matched compute.
- **Anti-baseline 2 deferred to Part B:** Heterogeneous prompt with
  same model.

**Provider-pair sweep (capability-symmetric per F4, Part A budget-locked):**

- {anthropic-claude, openai-gpt}: 2 orientations — (writer=claude,
  reviewer=gpt) and (writer=gpt, reviewer=claude). Each provider serves
  symmetrically.
- {anthropic-claude, google-gemini}: Part B future-work.
- {openai-gpt, google-gemini}: Part B future-work.

**Cross-family judge protocol:**

- Primary judge: Gemini for all primary-pair cells (disjoint from both
  {claude, gpt}).
- Robustness judge: OpenAI gpt-5.4 (SPECS default). Run on identical
  outputs.

**Calibration pilot (pre-registered budget gate):**

Before the full 264-cell sweep, run a 5-perturbation pilot on a single
orientation × single seed. Verify (a) cost-per-cell matches estimate
within 10%, AND (b) Δ-direction is positive at any magnitude. If (a)
fails, abort and report cost-overrun as falsification of the design
feasibility. If (b) fails (negative Δ in pilot), the pilot result IS
the F1 falsification.

**Required ablations:**

- **Capability-symmetry orientation sweep (F4 required).** Both
  orientations × N=3 seeds in Part A.
- **Judge-swap robustness check (F1 required).** Gemini primary + OpenAI
  secondary on identical writer/reviewer outputs.
- **Memorization probe (S2, deferred).** Held-out post-cutoff papers.
  Moved to Part B because SPECS perturbations are on AAAI-25 papers,
  many of which may be in training data for newer models. This is a
  known confound, listed in §8 Risk 5.

**Statistical test:**

- F1: paired-difference bootstrap (10k resamples) on per-paper issue-
  recall, between heterogeneous and same-family baseline, primary
  {claude, gpt} pair only. α=0.05 uncorrected (single comparison).
- F2': per-orientation lift, single-pair sufficiency.
- F3: paired-axis comparison on the human-validated 22-perturbation set.
  Within-paper test, not subject to F1's correction.
- F4: per-orientation lift compared to across-orientation mean via same-
  paper bootstrap SE.

**Locked sample / cost estimate (per NEW-1):**

| Item | Spec |
|---|---|
| Substrate | 22 SPECS validated perturbations |
| Provider pair | 1 ({claude, gpt}) |
| Orientations | 2 (F4 required) |
| Seeds | 3 |
| Conditions | 2 (heterogeneous + same-family baseline) |
| Cells | 264 |
| Writer+reviewer cost | ~$0.45/cell ≈ $119 |
| Judge cost (Gemini primary + OpenAI robustness) | ~$0.20/cell ≈ $53 |
| Calibration pilot | ~$3 |
| Buffer (15%) | ~$20 |
| **Total** | **≈ $195 ≤ $200 spec ceiling** |

If pilot cost-per-cell measurement deviates from estimate by >10%, abort
and re-design. This is a pre-registered design-feasibility gate.

---

## 8. Risks to the hypothesis

Each risk has a "what the hypothesis still contributes" statement.

### Risk 1: Feedback Friction caps writer-side revision (Jiang et al. 2506.11930)

Even if F1–F4 pass (reviewer detects more issues under heterogeneous
routing), the writer may fail to incorporate the corrections on revision,
leaving end-to-end paper quality unchanged. **Contribution if this
materializes:** clean separation of detection-vs-incorporation; the
mechanism is isolated at the reviewer side.

### Risk 2: Single-pair design cannot generalize to "heterogeneity" claim

Part A tests one pair ({claude, gpt}). If F1 passes, the result is "cross-
family routing on this specific pair helps" — not the general
"heterogeneity helps" claim. **Contribution:** even the single-pair
finding is informative for MegaResearcher's routing rules and feeds the
Part B sweep.

### Risk 3: Cross-family Gemini judge has its own biases

The cross-family judge protocol replaces "same-family-as-writer-or-
reviewer" judging with "third-family" judging — but Gemini may have its
own scientific-error priors that systematically differ from OpenAI's
default (Wataoka et al. arXiv:2410.21819 — self-preference bias
correlates with perplexity, a family-specific signal). The robustness
check (run
OpenAI judge on same outputs) is designed to surface this. If the two
judges disagree on direction, the result is flagged as judge-driven.
**Contribution:** judge-disagreement at the headline level is itself a
finding for the SPECS benchmark's reliability.

### Risk 4: Same-family bias is dominated by other paper-gen failure modes

AgentRxiv §4.1 attributes reward-hacking to scoring-based selection, not
same-family identity. If same-family bias is a minority contributor, F1's
0.05 floor will not be reached. **Contribution:** the hypothesis does
NOT claim same-family bias is the dominant failure mode — only that it
is a measurable one. A null result on F1 with the stage-matched baseline
narrows the failure-mode attribution.

### Risk 5: Memorization confound on SPECS (AAAI-25 papers)

SPECS perturbations are on AAAI-25 papers, many of which may be in
foundation-model training data. The heterogeneous pair may benefit
asymmetrically (one model has memorized the paper, the other has not),
biasing the issue-recall measurement. **Contribution if this is the
explanation:** a finding that SPECS-on-AAAI-25 is contaminated for
cross-family writer/reviewer measurement, motivating the held-out post-
cutoff probe in Part B.

### Risk 6 — NEW: Part A is workshop-grade, not main-track

Per §1 honest budget-compliance statement: the budget-compliant version
is a pilot. The smith does NOT claim Part A alone is a publishable main-
track contribution. **Contribution as workshop-grade pilot:** (a) first
public report on ARIS Appendix E protocol applied to an AI-Scientist-
family pipeline; (b) SPECS-validated-subset evaluation methodology with
cross-family judge protocol documented; (c) F4 capability-symmetry
control as a transferable design pattern; (d) Part B sweep specification
for follow-up at higher budget. **The synthesist's audit trail should
record Part A as in-scope, Part B as future-work with the lesson "main-
track-grade cross-family writer/reviewer measurement requires budget
above the swarm's gating allows."**

---

## 9. Sources

All arXiv IDs verified resolvable via `hf_papers paper_details` during
revision-2 preparation. ARIS §2.2 and Appendix E verified verbatim via
`read_paper`. SPECS A.9.4 (Table 5 with 22-perturbation consensus-valid
count) verified verbatim via `read_paper`. CiteME §2 verified verbatim
via `read_paper` (demoted to Part B). AblationBench §3 verified verbatim
via `read_paper` (LM-judge use confirmed; demoted to Part B).

**Mechanism citations:**

- arXiv:2502.08788 — Zhang et al. "Stop Overvaluing Multi-Agent Debate;
  Embrace Model Heterogeneity" — heterogeneity is the surviving MAD
  configuration; magnitude anchor (§4.2; §5 magnitude reasoning)
- arXiv:2305.19118 — Liang et al. "Encouraging Divergent Thinking in
  LLMs through Multi-Agent Debate" — same-LLM judge preference bias (§4.1)
- arXiv:2410.12853 — Hegazy "Diversity of Thought" — direction
  confirmation on GSM-8K/ASDiv (§4.2)
- arXiv:2402.11436 — Xu et al. "Pride and Prejudice: LLM Amplifies
  Self-Bias in Self-Refinement" — formal self-bias definition (§4.1)
- arXiv:2506.11930 — Jiang et al. "Feedback Friction" — intra-model
  friction (§4.3; Risk 1)
- arXiv:2310.01798 — Huang et al. "LLMs Cannot Self-Correct Reasoning
  Yet" — intrinsic self-correction net-negative (§4.3)
- arXiv:2410.21819 — Wataoka et al. "Self-Preference Bias in LLM-as-a-
  Judge" — empirical evidence that LLM evaluators favor outputs matching
  their own perplexity distribution; one additional citation per NEW-5
  (§4.3; Risk 3)

**Same-family-pair failure-mode citations (weakened per C4):**

- arXiv:2503.18102 — Schmidgall & Moor "AgentRxiv" — §4.1 reward-hacking
  attributed to scoring-based selection (§4.1, Risk 4)
- arXiv:2501.04227 — Schmidgall et al. "Agent Laboratory" — §limitations
  symptom (§4.1)
- arXiv:2408.06292 — Lu et al. "AI Scientist v1" — same-class automated
  reviewer pattern (§2)
- arXiv:2504.08066 — Yamada et al. "AI Scientist v2" — same-class
  auto-reviewer + VLM critic pattern (§2)
- arXiv:2505.18705 — Tang et al. "AI-Researcher" — §6.3 symptom (§4.1)
- arXiv:2511.04583 — Miyai et al. "Jr. AI Scientist" — same-class
  mentor-fed analyst + reviewer pattern (§2)

**Direct prior-art citation (foregrounded per C1):**

- arXiv:2605.03042 — Yang et al. "ARIS: Autonomous Research via
  Adversarial Multi-Agent Collaboration" — §2.2 cross-model default;
  Appendix E (verified verbatim, "Future Work") proposes the five-arm
  controlled benchmark protocol this hypothesis runs (§2; §5; §7)

**Substrate citations:**

- arXiv:2604.13940 — Biswas et al. "AI-Assisted Peer Review at Scale:
  The AAAI-26 AI Review Pilot" — SPECS benchmark §4; A.9 curation; A.9.4
  Table 5 human-oversight 22-consensus-valid count verified verbatim.
  PRIMARY substrate in rev-2 (§5; §6; §7).
- arXiv:2407.12861 — Press et al. "CiteME" — non-judge benchmark, but
  single-agent retrieval (NEW-2). Demoted to Part B future-work (§7).
- arXiv:2507.08038 — Abramovich & Chechik "AblationBench" — LM-judge
  per §3. Demoted to Part B future-work (§7).
