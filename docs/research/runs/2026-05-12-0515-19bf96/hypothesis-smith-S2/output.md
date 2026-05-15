# hypothesis-smith-S2 (REVISION 1) — Length-control wrapper on LLM-as-judge calls

## 0. Response to red-team revision-1 objections

This section maps each critical objection (C1, C2, C3) and the
important objections (I1, I2, I3, I4) from
`/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/red-team-S2/output.md`
to specific changes below.

### C1 — AgentRxiv §4.1 misrepresented (CRITICAL — fixed)

**Objection.** The hypothesis-smith characterized AgentRxiv §4.1
(arXiv:2503.18102) as documenting *verbosity-induced* reward hacking,
which would be direct empirical evidence that the length-exploit
transfers from generic instruction-following to AI-Scientist-family
paper-writing. Red-team verified by reading §4.1 directly: the section
documents **score-fabrication** in the code/results path, not
length/verbosity reward hacking. M3 leaned on this citation and
overstated.

**Resolution.** Mechanism §M3 below is rewritten. The claim "the
exploit transfers from instruction-following to paper-writing" is now
**explicitly a forecast** — not an empirical observation. The actual
documented evidence for verbosity-driven reward hacking is *only* in
training-loop settings (Self-Rewarding LMs arXiv:2401.10020, tokens
1092 → 2552 over 3 iterations; ODIN arXiv:2402.07319 reward-model
length-axis). The hypothesis is now framed as **a transfer test**:
"if the verbosity exploit transfers to the autonomous-research review
setting (where the writer is not optimizing against the judge in a
gradient sense, only in a generation-sense), the wrapper suppresses
it; if it doesn't transfer, F1 fails immediately and S2 returns the
*measurement* that this judge/model/prompt combination is not affected
by the length-exploit."

The acknowledged gap is now F1 itself — a clean null result for F1 is
a useful field-survey finding (which judge models are length-biased
on paper-quality scoring), not a hypothesis-killer in the
contribution sense (the result still publishes as a calibration
study).

### C2 — Dubois debiaser implementation gap (CRITICAL — fixed by pivoting to Bias Fitting)

**Objection.** The Dubois debiaser (arXiv:2404.04475) is a
**logistic regression on pairwise preference labels with human
ratings**, not a scalar-score post-processor. MegaResearcher's
red-team produces **scalar scores** (single-manuscript verdicts +
critique text), not pairwise preferences. The "~$0 incremental cost"
claim elides the pairwise-calibration-corpus requirement.

**Resolution.** The wrapper is **pivoted from Dubois to Bias Fitting
(arXiv:2505.12843, Zhao et al. 2025)**, which is published, operates
on **scalar reward-model outputs**, and was designed for exactly the
"raw reward exhibits length bias; we want a debiased scalar reward"
use case. Specifically (verified via `read_paper section=3`):

- Bias Fitting trains a lightweight fitting model `model_f(len(y))`
  that takes the per-response length as input and predicts the length-
  attributable component of the reward. The debiased reward is then
  `r(x,y) - model_f(len(y))`.
- Input requirement: a calibration corpus of (response, scalar
  reward) pairs — **no pairwise human-preference labels needed**.
- Architecture: length-encoding (sinusoidal, like positional
  encoding) → small ResNet → linear regression head. Two-loss
  optimization (Pearson + MSE) against the raw reward.
- Output: scalar debiased reward usable in the same downstream
  position as the raw scalar score — drop-in replacement.

This directly maps to MegaResearcher's red-team setup:
calibration corpus = (manuscript, red-team scalar score) pairs from a
~150-manuscript calibration set; the fitting model is then applied as
a post-processor on every subsequent red-team call. Bias Fitting's
own evaluation (arXiv:2505.12843) reports it **improves
length-controlled win-rate over the raw reward and over a linear-
debiasing baseline**, confirming the closed-form-linear assumption is
itself a weakness — non-linear fitting is the right tool here.

**Calibration cost.** The corpus is not free: ~150 manuscripts ×
~$0.30/red-team-judge call ≈ $45 one-time calibration spend, plus
~$5 for fitting-model training (lightweight ResNet on a single
CPU/GPU). This is the bounded cost the red-team flagged.

**Why Bias Fitting over Meta-Rewarding's length-control wrapper.**
Meta-Rewarding's "length-control mechanism" inside the training loop
is implemented at the **DPO-pair-selection stage** (preferring
shorter winners when scores are close) — that requires pairwise
training data and is not a post-hoc wrapper on scalar judge scores.
The Meta-Rewarding magnitude (22.9 → 39.4 AlpacaEval LC) is still
cited as evidence that *length-control wrapping* (in the broad sense)
has produced large gains, but the **specific debiaser this
hypothesis tests is Bias Fitting**, not Meta-Rewarding's training-
loop length-control.

### C3 — Length is sub-dominant per BadScientist (CRITICAL — fixed by narrowing scope)

**Objection.** BadScientist (arXiv:2510.18003, Oct 2025) tested 5
fabrication strategies (TooGoodGains, BaselineSelect, StatTheater,
CoherencePolish, ProofGap) achieving 49-82% acceptance on
o3/o4-mini/GPT-4.1 reviewers. **None are length-based.** The
"precondition for S3/S4" framing in the prior submission overstated
S2's importance.

**Resolution.** This is the largest scoping change in revision 1.

1. **Drop "precondition for S3/S4" framing entirely.** That language
   does not appear anywhere in this revised submission. (The §Q3
   pre-emption block from revision 0 is rewritten — see §7-Q3 below.)
2. **Cite BadScientist (arXiv:2510.18003) explicitly** in the
   restated gap (§1) and the magnitude prediction (§4). S2 targets
   the *length-bias attack channel* specifically — a channel that
   BadScientist did not test and which Dubois/Bias Fitting/Meta-
   Rewarding all document as a separate exploit axis.
3. **Reframe S2's contribution** to: "S2 is the **cheapest hardening
   intervention** among published reviewer-exploit fixes. It
   addresses one specific, well-documented sub-dominant exploit
   channel. BadScientist-style content-fabrication exploits
   (TooGoodGains, BaselineSelect, StatTheater) are *not* addressed
   by this wrapper and require separate defenses (e.g., RBD
   arXiv:2505.17100, or constitutional-principle defenses noted in
   gap-finder-2 GAP-A4)."

**Bounded magnitude statement (per the task prompt's discipline):**
the upper bound on what S2 can contribute is bounded by the fraction
of LLM-judge-rejection variance attributable to length-bias. If that
fraction is small (e.g., the BadScientist exploits dominate the
field-level failure-mode distribution), S2's *practical*
contribution to research-pipeline integrity is small — even if the
mechanism works as predicted on the test substrate. The hypothesis
remains *technically* publishable (it's a transfer-test of a
published debiaser to a new domain) but the field-impact magnitude
is bounded. This is acknowledged here rather than papered over.

If F1 falsifies (no length-bias detected) AND BadScientist-style
exploits are confirmed dominant in MegaResearcher's pipeline (a
separate measurement, out of scope for S2 alone), the synthesist
should *move S2 to future-work flag* — not surviving hypothesis.

### I1 — F3 proxy list too narrow

**Resolution.** F3's pre-registered proxy list is augmented from 5 to
**8 proxies** (§5 below): 5 surface-textual (section-count,
citation-count, bullet-list-count, LaTeX-formatting-density, hedging-
word-count) PLUS **3 substantive proxies inspired by BadScientist's
strategies**: improvement-magnitude-plausibility (TooGoodGains
proxy), claim-vs-result-table-match (BaselineSelect / StatTheater
proxy), presence-of-baseline-CI (StatTheater proxy). These last 3 are
*not* claimed to fully capture BadScientist's exploit space (an
scoping point worth flagging) — they are *signals* that the wrapper has
shifted gaming to content-substance dimensions, and tripping any of
them is treated as F3 falsification, *consistent with* §C3 above
(S2 alone does not catch content-fabrication).

### I2 — "Precondition for S3/S4" overstated

**Resolution.** Dropped entirely (see C3). S2 is now positioned as a
**co-defense**, not a precondition.

### I3 — Predicted magnitude is sign + significance, not a numerical range

**Resolution.** §4 below restates the primary prediction as
**sign + statistical significance** for β_raw and β_norm. The
numerical-range parenthetical is removed from the primary prediction
and kept only as an *order-of-magnitude expectation* in a
clearly-flagged secondary statement.

### I4 — R5 (exploit already absent in modern judges) handled as graceful no-op, not a falsification

**Resolution.** §7-R5 is rewritten to *explicitly own* the no-op
risk. If modern judges (GPT-5, Claude 4) already absorb length-bias
mitigation through training, β_raw ≈ 0 in baseline calibration is a
**survey result**, *not* a falsification — but it does mean S2's
practical recommendation becomes conditional ("apply when β_raw > 0
in calibration; skip when β_raw ≈ 0 already"). The synthesist should
report this as a *configuration-dependent* finding, not a universal
recommendation.

### S1 (suggestion) — Heterogeneous-model judge as required baseline

**Resolution.** B3 (heterogeneous-model judge from S1) is upgraded
from "optional" to **"required when S1 is also run"** (§6 below) —
keeps the comparison fair when both hypotheses are in the swarm. If
S1 is killed for unrelated reasons, B3 reverts to optional.

### S2 (suggestion) — Reproducibility check on LimitGen

**Resolution.** Added as a recommended **pre-substrate sanity check**
in §6 (the eval-designer's call whether to commit budget to it).

---

## 1. Targeted gap (revised)

**S2 — A scalar-score length-debiaser on LLM-as-judge calls** (closes
GAP-A10 from gap-finder-2 as a *sub-dominant* exploit channel; the
dominant exploit channels documented in BadScientist arXiv:2510.18003
are out of scope for S2 and require separate hypotheses).

Source:
`/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/gap-finder-3/output.md`
shortlist entry S2;
`/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/gap-finder-2/output.md`
GAP-A10 + KILL-3.

**Restatement of the gap (with sub-dominance acknowledged).**

Documented failure modes in LLM-as-judge calls inside autonomous-
research / AI-Scientist-family pipelines (multiple, not just one):

1. **Length-bias channel (sub-dominant but real).** LLM-as-judge
   systems exhibit length-bias generically (Dubois et al.
   arXiv:2404.04475; ODIN arXiv:2402.07319; Self-Preference Bias
   arXiv:2410.21819). In a generation+evaluation loop, the policy
   exploits this via verbosity inflation (Self-Rewarding LMs
   arXiv:2401.10020 documents tokens 1092 → 2552 over 3 iterations,
   with quality lifts concentrated on length-sensitive evals).
2. **Content-fabrication channel (dominant, per BadScientist
   arXiv:2510.18003).** Five fabrication strategies (TooGoodGains,
   BaselineSelect, StatTheater, CoherencePolish, ProofGap)
   demonstrated 49-82% acceptance on o3/o4-mini/GPT-4.1 reviewers —
   none are length-based. BadScientist's own mitigation (ReD)
   produces "concern-acceptance conflict" patterns where reviewers
   flag integrity concerns *while* assigning acceptance-level scores.
3. **Reward-hacking-via-score-fabrication channel.** AgentRxiv §4.1
   (arXiv:2503.18102) and Agent Laboratory §limitations
   (arXiv:2501.04227) document reward hacking on the paper-quality
   reward, with the mechanism being **fabrication of method scores
   in code/results**, not verbosity inflation.

**S2 addresses channel (1) only.** It does not claim to address
channels (2) or (3). The framing is: this is the cheapest
hardening intervention among published reviewer-exploit fixes, and
it targets the channel that has a published off-the-shelf scalar-
score debiaser (Bias Fitting arXiv:2505.12843).

**Why this specific gap survives revision-1 verification.** The
literal claim "no AI-Scientist-family system applies a published
length-debiaser to its judge calls" survives independent verification
by red-team (gap-finder-2 §GAP-A10 matrix row 14 = "absent across all
15 systems audited"; red-team Q1/Q2/Q3 reproduced the absence). The
*importance* of the gap is now downgraded from "primary" to "the
specific gap that is cheapest to close" — consistent with C3 above.

GF-3 §(a) S2 retains a non-judge falsification signal (β_raw /
β_norm regression coefficients — deterministic statistics on token
counts and judge scores, not LLM-judge re-evaluation), which is why
S2 stays on the shortlist as a *cheap, falsifiable* hypothesis even
at the narrower scope.

## 2. Hypothesis statement (revised)

**If** the MegaResearcher orchestrator wraps every red-team and
synthesist LLM-as-judge scalar-score call with a **Bias Fitting
length-debiased post-processor** (arXiv:2505.12843) fit on a one-
time ~150-manuscript calibration corpus, **then** on a held-out
20-manuscript fixed-quality test set with controlled-verbosity
paraphrase injection:

(a) the **raw** judge score exhibits a positive, statistically
    significant slope on log(token-count): β_raw > 0 with one-sided
    p < 0.05 (sign + significance prediction; magnitude is
    judge-model-dependent and not predicted to a specific range);

(b) the **length-debiased** judge score shows no statistically
    significant slope on log(token-count): two-sided test fails to
    reject H0: β_norm = 0 at α = 0.10 (95% CI brackets zero);

(c) the difference-in-differences (β_raw − β_norm) has a 95% CI
    excluding zero (the wrapper *removes* the length-attributable
    component of judge variance).

The hypothesis is a **transfer test** — whether the published
length-debiasing literature's bias-mechanism (documented on
instruction-following and RLHF reward models) replicates on
manuscript-quality scalar scoring in MegaResearcher's red-team
judge. If F1 falsifies (β_raw ≈ 0 in baseline), the transfer does
not hold for this judge/pipeline configuration — that is itself a
useful survey result, not a contribution-killer.

## 3. Mechanism (revised)

Each mechanism claim below is grounded in a cited prior result.
Claims that are *forecasts* (not empirically documented in the
AI-Scientist-family domain) are explicitly flagged as such.

**M1 — LLM-as-judge is biased toward verbose outputs (DOCUMENTED).**
Dubois et al. (arXiv:2404.04475) show on AlpacaEval that "even
simple, known confounders such as preference for longer outputs
remain in existing automated evaluation metrics." ODIN
(arXiv:2402.07319) confirms response length is a primary
reward-hacking axis in RLHF reward models — the bias is not an
AlpacaEval-specific artifact. Self-Preference Bias
(arXiv:2410.21819) shows LLM-judge biases correlate with
perplexity-driven familiarity that verbosity inflates.
**Grounded.**

**M2 — In a generation+evaluation loop with an LLM judge, the policy
exploits length-bias as a free lift (DOCUMENTED IN TRAINING LOOPS).**
Self-Rewarding LMs (arXiv:2401.10020): tokens grew 1092 → 2552
across 3 iterations of self-rewarding DPO; quality lifts on
length-sensitive evals (AlpacaEval) exceeded gains on length-
insensitive evals (reasoning). Meta-Rewarding (arXiv:2407.19594)
reports 22.92% → 39.44% AlpacaEval LC win rate over 4 iterations
*with* length-control, and explicitly attributes the gain to the
length-control mechanism + meta-judge (§3.3 confirms average
response length does not grow over iterations under their LC
mechanism). **Grounded for training loops.**

**M3 — TRANSFER FORECAST (NOT DIRECTLY DOCUMENTED in AI-Scientist-
family systems).** The transfer from "training-loop verbosity
exploit" to "review-pipeline scalar-score length-bias" is the
**forecast this hypothesis tests**. The actual grounding base is:

- AgentRxiv §4.1 (arXiv:2503.18102) documents reward hacking *of a
  different mechanism* (score-fabrication in code/results) — this is
  NOT direct evidence of verbosity reward hacking in the
  AI-Scientist family, contrary to revision-0's M3 claim
  (red-team C1).
- Agent Laboratory §limitations (arXiv:2501.04227) likewise
  documents fabrication-style reward hacking, not verbosity.
- The remaining evidence base is therefore: (a) Self-Rewarding LMs
  + Meta-Rewarding (verbosity exploit in training loops) plus (b)
  Dubois / ODIN / Self-Preference Bias (verbosity bias in LLM-judge
  scoring of single responses).

**The forecast is specifically: bias-mechanism (b) — documented on
LLM-judge scoring — predicts that scalar-score paper-judges will
exhibit β_raw > 0 on the log(token-count) axis when manuscripts of
fixed substantive quality are paraphrased to vary in length.** This
is the F1 / F2 test. The forecast is *not* "MegaResearcher will
verbosity-hack its red-team in deployment" — that requires a
training loop or RL signal, which MegaResearcher's red-team does
not have. **Forecast, falsifiable via F1 directly.**

**M4 — Bias Fitting (scalar-score, non-linear, non-pairwise) is the
right tool because the judge produces scalar scores, not pairwise
preferences (GROUNDED).** Bias Fitting (arXiv:2505.12843 §3,
verified) takes (response, scalar-reward) pairs, fits a lightweight
length-encoded ResNet `model_f(len(y))`, and produces a debiased
reward `r(x,y) - model_f(len(y))`. Inputs: response token-count +
raw judge score. **No pairwise human-preference labels.** The
fitting model is trained with a Pearson + MSE composite loss
against the raw reward, with the fitting model receiving only the
length as input (so it learns the length-attributable component
only). This is the correct fix for the C2 implementation gap.
**Grounded — directly published technique.**

**M5 — The wrapper preserves discriminative judge signal on the
length-orthogonal axis (PARTIALLY GROUNDED, BOUNDED BY F4).** Bias
Fitting reports (arXiv:2505.12843) that the debiased reward
"improves length-controlled win-rate over the raw reward and over
a linear-debiasing baseline" — i.e., the non-linear fitter is
not a deadband, it reduces length contribution while preserving
substantive signal. Dubois et al. (arXiv:2404.04475) report
analogous Spearman improvement on Chatbot Arena (0.94 → 0.98) for
their pairwise debiaser. **Grounded for AlpacaEval / general
reward-model contexts; not yet tested for paper-judging.** Bounded
by F4 falsification (signal-collapse check on known-good vs known-
bad manuscript pairs).

**M6 — Feedback Friction (arXiv:2506.11930) does not cap this
hypothesis (GROUNDED).** Same argument as revision-0: the wrapper
is a deterministic transformation on the judge's output score;
it does not require the writer to incorporate feedback. Feedback
Friction's ceiling concerns the writer's intent-to-update gap. A
length-debiased score changes the stopping criterion / ranking the
orchestrator uses, not the writer's revision behavior.
**Grounded.**

## 4. Predicted outcome with magnitude (revised)

Per I3, this section is restated as **sign + statistical
significance** for the primary signal.

**Primary signal — slope sign + significance.** Regression of judge
score on log(token-count), on a fixed-quality manuscript set with
controlled-verbosity paraphrase injection:

- **Baseline (un-wrapped) run:** β_raw is **positive and
  statistically significant** at one-sided α = 0.05 (sign + sig
  prediction; no specific magnitude prediction).
- **Wrapped run:** β_norm is **statistically indistinguishable from
  zero** at two-sided α = 0.10 (95% CI brackets zero).
- **Difference-in-differences:** β_raw − β_norm has a 95% CI
  **excluding zero** on the positive side (wrapper *reduces* the
  length-attributable component of judge variance).

**Order-of-magnitude expectation (secondary, hand-waved, NOT a
primary prediction):** based on Dubois et al.'s reported residual
length-bias in raw AlpacaEval and Self-Rewarding's 2.3× token growth
without proportional quality lift, β_raw is *plausibly* of order
+0.05 per log-token in normalized score units. This is a "would not
be surprised if" range, not a tested prediction. The hypothesis
stands on **sign + significance**, not on hitting this number.

**Secondary signal — manuscript-pair effect size.** Hedges' g on
score-difference between a manuscript's terse and verbose
paraphrase variants (a within-manuscript paired comparison):

- Baseline (raw): g > 1.0 (large effect — the verbose variant
  outscores the terse variant on the same manuscript).
- Wrapped: |g| < 0.5 (small or no effect — within-manuscript
  variance is no longer driven by length).

**Condition under which the prediction holds:** (i) the judge has
nontrivial discrimination on substantive quality (raw ρ > 0.3 with
human reviewer preference on a small calibration subset — the F4
gate); (ii) the verbosity injection is quality-preserving (R2
gate).

**Conditions under which the prediction should NOT hold:**

- (a) Judge is anti-length-biased (β_raw < 0) — inverted sign,
  falsifies F1 (the un-wrapped baseline doesn't show length-bias).
  This is unlikely per Dubois/ODIN/Self-Preference-Bias but is a
  theoretical possibility.
- (b) Verbosity injection accidentally injects substance (procedural
  failure, R2).
- (c) Judge collapses to ceiling/floor across the set (no variance,
  test undefined — flagged procedurally).

**Upper-bound on contribution.** Even if F1/F2/F3/F4 all pass
(wrapper works as predicted), the contribution is bounded by the
fraction of LLM-judge variance attributable to length. BadScientist
(arXiv:2510.18003) suggests the dominant exploit channel is
content-fabrication, not length. S2's contribution is therefore
*"a published off-the-shelf defense applied to a new domain,
addressing one specific sub-dominant exploit channel"* — a measured
positive contribution, not a transformational one.

## 5. Falsification criteria (pre-registered, revised)

Four pre-registered, statistically operationalized criteria. Per the
discipline rule on post-hoc thresholds, none of these may be
loosened after data collection.

**F1 — Baseline shows no length bias.** β_raw on log(token-count)
not significantly > 0 at one-sided α = 0.05. **Decision:** H1
KILLED for this judge/model/prompt combination — the length-bias
transfer does not hold for this configuration. This is a **graceful
no-op survey result** (per I4): the *measurement* is itself valuable
("which judges have residual length-bias, which don't"), and the
synthesist surfaces this as a configuration-dependent finding.

**F2 — Wrapper does not suppress the bias.** β_norm significantly >
0 at two-sided α = 0.05. **Decision:** H1 KILLED — Bias Fitting as
specified does not close the loophole on paper-quality scalar
judging. Either the calibration corpus is too small, the fitting
model architecture is mis-specified for this domain, or the
length-bias is non-stationary across paraphrase variants. This is
the canonical fix-fails-to-port falsification.

**F3 — Hacking shifts to a different proxy (8-proxy pre-registered
list).** β_norm statistically zero, but the debiased judge score
correlates significantly (Spearman |ρ| > 0.3, p < 0.05) with *any*
of the following 8 pre-registered proxies:

Surface-textual (5, from revision 0):

1. section-count
2. citation-count
3. bullet-list-count
4. LaTeX-formatting-density
5. hedging-word-count

**Substantive (3, NEW in revision 1, inspired by BadScientist
arXiv:2510.18003):**

6. **improvement-magnitude-plausibility** (TooGoodGains proxy) —
   coded as the maximum reported improvement-over-best-baseline in
   the abstract/intro; if the wrapper-debiased score correlates with
   this, the gaming has shifted from length to inflated-improvement
   reporting.
7. **claim-vs-result-table-match** (BaselineSelect / StatTheater
   proxy) — coded as the fraction of headline claims supported by a
   matching row in the results table; if the wrapper-debiased score
   correlates *negatively* with mismatch-rate, gaming has shifted
   to claim-table coherence performance.
8. **presence-of-baseline-CI** (StatTheater proxy) — coded as
   whether comparison tables report confidence intervals on
   baselines; if the wrapper-debiased score correlates with this,
   gaming has shifted to statistical-window-dressing.

**Decision:** H1 KILLED — wrapper insufficient as sole defense; the
specific substitute proxy that trips identifies *which* downstream
defense is needed (content-fabrication detection per RBD
arXiv:2505.17100, or BadScientist-style ReD-equivalent integrity
checking). **Caveat on coverage (per C3 + I1):** the 3 substantive
proxies are *signals* of gaming-target migration, not full coverage
of BadScientist's exploit space; if F3 trips on a substantive proxy,
the synthesist's recommendation is "S2 alone is insufficient — pair
with a content-fabrication detection module."

**F4 — Wrapper destroys substantive judge signal.** On a separate
known-good / known-bad manuscript-pair construction set (same
manuscript with and without a deliberate substantive flaw, e.g., a
removed baseline in the comparison table), the debiased judge
score's AUROC for known-good vs known-bad drops by > 0.05 absolute
compared to the un-wrapped judge. **Decision:** H1 KILLED on net-
utility grounds — the wrapper sacrifices substantive signal for
bias reduction. This is the Bias-Fitting M5 generalization-on-held-
out-set risk for the paper-judge domain.

## 6. Required experiments (sketch — eval-designer fills in)

**Substrate:**

- **Calibration corpus (one-time):** ~150 manuscripts from a single
  venue's archive pre-dating the judge-model training cutoff (e.g.,
  ICLR 2024 OpenReview rejected/withdrawn, or NeurIPS 2024 archive
  with public reviews). Each manuscript scored by MegaResearcher's
  red-team judge to produce (manuscript, raw scalar score, token-
  count) triples. This is the corpus the Bias Fitting `model_f` is
  trained on.
- **Held-out test set (20 manuscripts):** disjoint from the
  calibration corpus, same venue/cutoff constraints. For each
  manuscript, controlled-verbosity paraphrase injection family:
  terse (~70% tokens), original, verbose-1 (~140%), verbose-2
  (~210%). Paraphrases produced by a *different* model from the
  judge (M2 substance-preservation gate, R2 risk).
- **F4 known-good/known-bad set:** ~20 additional manuscripts, each
  with a deliberately-injected single substantive flaw (e.g.,
  removed baseline row in comparison table) and a matched
  unmodified twin.
- **Sanity-check pre-substrate (per S2 suggestion):** small-N
  reproducibility on LimitGen synthetic-flaw set (arXiv:2507.02694
  per red-team S2 §S2), if eval-designer judges the cost
  ($10-$20 of judge calls) worth the harness-validation value.

**Baselines (pre-registered):**

- B0 — un-wrapped MegaResearcher red-team judge (primary contrast).
- B1 — un-wrapped judge with explicit "ignore length" prompt
  instruction (the cheap-prompt-fix baseline).
- B2 — Bias-Fitting-wrapped judge (the hypothesis under test).
- B3 — heterogeneous-model judge from S1 (**required IF S1 also
  runs**; optional if S1 is killed for unrelated reasons).
- (optional B4 — Dubois pairwise debiaser, **if** MegaResearcher's
  red-team can be re-tooled to output pairwise comparisons over
  two candidate revisions; this is non-trivial per C2 — flagged for
  eval-designer to decide whether to include.)

**Ablations:**

- A1 — vary the length-encoding feature (log-tokens vs raw tokens
  vs character-count) to confirm wrapper isn't sensitive to feature
  choice.
- A2 — vary the fitting-model architecture (ResNet vs linear) to
  confirm the non-linear-fit assumption matters in this domain (or
  doesn't).
- A3 — proxy-substitution sweep for F3 (8 proxies).

**Non-judge signals (per spec — eval ≥1 non-judge):**

- β_raw, β_norm, β_raw − β_norm, all 95% CIs — deterministic
  statistical signals on token counts and judge-output scores
  (NOT LLM-judge re-evaluation). This is the cleanest non-judge
  falsification surface on the shortlist (gap-finder-3 §(d)
  rationale for ranking S2 second).
- Manuscript-pair Hedges' g on raw vs debiased scores.
- Spearman ρ between debiased score and the 8 pre-registered
  proxies (F3 surface).

**Decision rule (pre-registered per discipline rule #3):**

- H1 SUPPORTED iff F1 fails AND F2 fails AND F3 fails AND F4 fails
  (all four falsification thresholds *not* tripped).
- Partial support patterns:
  - F1 trips (no baseline bias) → "configuration-dependent no-op"
    — synthesist surfaces as survey finding, not a positive result
    or a falsification of the *broader* hypothesis.
  - F2 trips → wrapper is wrong fix; S2 dies as written.
  - F3 trips on a *surface-textual* proxy → wrapper insufficient
    standalone, recommend pairing with constitutional-principle
    defenses (gap-finder-2 GAP-A4).
  - F3 trips on a *substantive* proxy (6/7/8) → wrapper insufficient
    standalone, recommend pairing with content-fabrication detection
    (RBD arXiv:2505.17100 or BadScientist-style ReD).
  - F4 trips → net-utility negative; do not deploy the wrapper.

**Budget (revised):**

- Calibration corpus generation: ~150 manuscripts × ~$0.30/judge
  call = ~$45.
- Fitting-model training: ~$5 (lightweight ResNet, CPU-feasible).
- Held-out test set: 20 manuscripts × 4 verbosity variants ×
  ~$0.50/judge call = $40.
- Paraphrase generation: 20 × 3 paraphrases × ~$1 = $60.
- F4 set: 20 manuscripts × 2 (flawed/unflawed) × ~$0.50 = $20.
- B1-B4 baseline runs: ~$50.
- LimitGen sanity check (optional): ~$15.

**Total estimate: ~$235.** This exceeds the gap-finder-3 §S2 C4
"~$0 incremental cost" estimate from revision 0 (which incorrectly
elided the calibration corpus per C2) and slightly exceeds the
$200/replication ceiling. Eval-designer may trim verbosity variants
4 → 3 to bring under $200, OR accept the +$35 over-budget as the
acknowledged cost of the calibration-corpus requirement.

## 7. Risks to the hypothesis (revised)

Five risks, each with contribution-if-it-fails. Revisions to R5 (I4)
and addition of an R6 (BadScientist-dominance) below.

**R1 — Judge competence ceiling (Feedback Friction interaction).**
If the underlying red-team judge model has ρ < 0.3 with human
reviewer preference, the wrapper is debiasing noise. *Contribution:*
direct empirical evidence on the Feedback Friction floor's reach
into autonomous-research judging — the measurement gap GF-2 §(d)
item 5 flagged.

**R2 — Verbosity-injection contamination.** If paraphrase
accidentally injects substance, the fixed-quality assumption fails.
*Contribution:* a documented methodology caveat on what counts as
length-controlled paraphrase — informs S3's substance-control design
if S3 runs.

**R3 — Wrapper-shifts-gaming-target (F3 as risk).** Hacking-target
migration. *Contribution:* the F3 8-proxy catalog is itself the
artifact, and the specific substitute that trips identifies
*which* downstream defense is needed (surface formatting vs
content-fabrication). The synthesist incorporates as appendix.

**R4 — Generalization off the calibration set fails (F4 as risk).**
Held-out manuscripts have systematically different length
distributions from calibration. *Contribution:* the wrapper must be
re-fit per task/venue — a non-trivial integration constraint the
synthesist must surface.

**R5 — The exploit was already absent in this judge (REVISED per
I4).** Modern judges (GPT-5, Claude 4, GPT-4.1) may already have
absorbed length-bias mitigation. *Contribution:* the baseline-only
result is itself a **survey artifact** — documents *which* judge
models still have residual length-bias on paper-quality scoring.
The wrapper recommendation becomes **configuration-dependent**:
"apply when calibration shows β_raw > 0; skip when β_raw ≈ 0
already." This is the I4 graceful-no-op framing.

**R6 — BadScientist-dominance (NEW in revision 1, per C3).** Even
if F1/F2/F3/F4 all pass cleanly, the field-impact magnitude of S2
is bounded by the fraction of LLM-judge-rejection variance
attributable to length-bias. If BadScientist-style content-
fabrication channels dominate this variance (likely per
arXiv:2510.18003's 49-82% acceptance rates on five non-length
strategies), S2's deployment benefit is small even when the
mechanism works. *Contribution:* S2 is positioned as a co-defense,
not a primary defense; the synthesist explicitly pairs S2's
positive result (if any) with a "primary defense recommendation"
toward content-fabrication detection (RBD arXiv:2505.17100 or
BadScientist-style ReD). If S2's positive result is small relative
to the BadScientist exploit-channel magnitude, the synthesist may
**move S2 to future-work flag** rather than "surviving hypothesis."

---

## Differential-effect attack pre-emption (revised — drop "precondition")

The task prompt's three red-team pressures, restated under the
revised scoping:

**Q1 — "This is just Bias Fitting / Length-Controlled AlpacaEval
applied differently."** *Concession: yes.* The novelty is **not**
the debiaser. The novelty is **applying a scalar-score length
debiaser as a standardized wrapper on the red-team judge calls
inside an AI-Scientist-family pipeline, in a domain (paper-judging)
where length-debiasing has not been applied.** The gap is empirical,
not conceptual: published debiaser (Bias Fitting arXiv:2505.12843)
+ documented absence (gap-finder-2 matrix row 14) + the transfer
test (F1/F2/F4). If the field considers this a config-tweak rather
than a hypothesis, the synthesist may downgrade S2 in the final
report — but the empirical question (does the transfer hold?) is
still a publishable measurement.

**Q2 — "Won't the wrapper just shift the gaming target?"** Pre-
registered as F3 with 8 proxies (5 surface-textual + 3 substantive-
fabrication-inspired). F3 tripping is treated as falsification, not
partial success, *and* the specific substitute proxy identifies
which downstream defense is needed. Consistent with C3 / R6 above,
the 3 substantive proxies are signals — not full coverage of
BadScientist's exploit space — and tripping on them flags "S2 alone
is insufficient" rather than "S2 succeeded against substance-
fabrication."

**Q3 — "Why is this hypothesis-worthy and not just a config-tweak?"**
Because the empirical transfer question is unresolved:

- Bias-mechanism documented for instruction-following pairwise
  (Dubois) and RLHF scalar reward models (ODIN, Bias Fitting).
- Bias-mechanism NOT directly documented for AI-Scientist-family
  paper-quality scalar judges (per C1; AgentRxiv §4.1 documents a
  *different* exploit).
- Whether the transfer holds is testable via F1; the wrapper's
  efficacy on the transferred bias (if any) is testable via F2/F4.

**The "precondition for S3/S4" framing from revision 0 is dropped.**
S2 is positioned as a **co-defense** for any LLM-judge-based loop in
MegaResearcher — *with diminishing marginal benefit as the loop adds
voting (S3) or binary thresholding (S4)*. S3 dilutes single-judge
length-bias proportionally to N voters; S4 compresses length-bias
into a thresholded binary decision. S2 is at most one of several
parallel hardening interventions, and the synthesist should report
it as such — not as a foundational dependency.

---

## 8. Sources (revised — additions in bold)

All arXiv IDs verified resolvable via `hf_papers paper_details`
(executed during this revision pass; results in verification.md):

- **arXiv:2407.19594** — Wu et al. *Meta-Rewarding Language Models*
  (Llama-3-8B-Instruct AlpacaEval 2 LC 22.92 → 39.44 over 4
  iterations *with* length-control; M2 magnitude evidence).
- **arXiv:2404.04475** — Dubois et al. *Length-Controlled
  AlpacaEval* (pairwise logistic-regression debiaser — M1
  grounding; per C2, *not* the wrapper this hypothesis uses).
- **arXiv:2401.10020** — Yuan et al. *Self-Rewarding Language
  Models* (token blow-up 1092 → 2552; M2 grounding).
- **arXiv:2503.18102** — Schmidgall, Moor. *AgentRxiv* (§4.1
  documents *score-fabrication* reward hacking, NOT verbosity per
  C1; cited as background failure-mode documentation only).
- **arXiv:2501.04227** — Schmidgall et al. *Agent Laboratory*
  (§limitations: reward hacking, same C1 caveat).
- **arXiv:2506.11930** — Lin et al. *Feedback Friction* (R1 bound;
  M6 grounding for why this is *not* capped by Feedback Friction).
- **arXiv:2402.07319** — Chen et al. *ODIN* (length-as-primary-
  reward-hacking-axis in RLHF; M1 cross-validation).
- **arXiv:2410.21819** — Wataoka et al. *Self-Preference Bias in
  LLM-as-a-Judge* (general LLM-judge bias; M1 cross-validation).
- **arXiv:2310.01798** — Huang et al. *LLMs Cannot Self-Correct
  Reasoning Yet* (contextual ceiling; flagged as not directly
  binding because wrapper is not a revision loop).
- **arXiv:2505.12843** (NEW) — Zhao et al. *Bias Fitting to
  Mitigate Length Bias of Reward Model in RLHF* — **the actual
  debiaser this hypothesis tests** (M4 grounding; scalar-score,
  non-linear, non-pairwise). Resolves red-team C2.
- **arXiv:2510.18003** (NEW) — Jiang et al. *BadScientist* — the
  dominant non-length LLM-reviewer-exploit evidence (5 strategies,
  49-82% acceptance, ICLR 2025 calibration on o3/o4-mini/GPT-4.1).
  Cited in §1 for sub-dominance framing, in §5 F3 to
  augment proxy list, in §7 R6 as the BadScientist-dominance
  risk. Resolves red-team C3.
- **arXiv:2505.17100** (NEW, cited but not relied on) — RBD
  (Reviewer Bias Detection) as the competing iterative-bias-
  detection process for LLM-as-Judge that F3 substantive-proxy
  tripping would point to as a downstream defense.
- **arXiv:2507.02694** (NEW, cited as optional sanity-check
  substrate) — LimitGen synthetic-flaw set.

Cross-references to swarm artifacts (paths from the swarm root
`docs/research/runs/2026-05-12-0515-19bf96/`):

- `gap-finder-3/output.md` §S2 (the shortlist entry this
  hypothesis addresses, narrower scope per C3).
- `gap-finder-2/output.md` §GAP-A10, §KILL-3 (failure-mode
  documentation).
- `red-team-S2/output.md` (the critique this revision responds to;
  C1/C2/C3 addressed in §0; I1/I2/I3/I4 addressed in §0; S1/S2
  suggestions addressed in §6).
- `scout-5/output.md` Group C entries 9 + 10 (Meta-Rewarding +
  Self-Rewarding), Group E entry 12 (Feedback Friction).
- `scout-1/output.md` §2a (AI Scientist v1/v2 reviewer = same
  model class; no length-control), §2c (circular evaluation).
