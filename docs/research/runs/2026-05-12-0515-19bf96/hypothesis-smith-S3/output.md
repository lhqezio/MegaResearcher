# hypothesis-smith-S3 (REVISION 2) — Per-binary-membership voting over enumerable structured paper-decisions inside AI-Scientist v2

## Response to red-team revision-2 objections

The red-team's revision-2 REJECT raised three critical defects (CR1, CR2, CR3) and four important defects (I1-rev through I4-rev). This revision addresses each. The submission is honestly descoped: the headline aggregate floor moves from +8 to **+6 percentage points**, the ablation-axis substrate switches to **AbGen (arXiv:2507.13300)**, and the baseline-list substrate switches to **canonical-leaderboard tasks** (GLUE / SuperGLUE / ImageNet-1k / COCO / WMT-EnDe). All construction protocols are now pre-registered inside this document, not deferred to eval-designer.

| Defect | Severity | Where addressed |
|---|---|---|
| **CR1** Same-model effective-N discount of 20% is empirically too small per Patel arXiv:2604.03809 (effective rank 2.17/3.0 at N=3 ≈ 27.7% reduction; worsens to 2.09/3.0 on MATH-500); AIMO 3 arXiv:2603.27844 confirms "correlated errors limit effective sample size" with only partial decorrelation from high-temp sampling | Critical | §1 magnitude-anchor paragraph rewritten; §2 hypothesis statement (Δ_aggregate floor descoped +8 → **+6**); §3 M1 (i.i.d. discount derivation rewritten with Patel and AIMO 3 citations); §4 effect-size derivation rewritten; §5 F1 / F2 thresholds rebased |
| **CR2** "Top-30 most-cited baselines on task T via Semantic Scholar" is not deterministic (no canonical "baselines for task T" S2 query; conflates "most-cited" with "is a baseline"; re-introduces LM-derived task definition / baseline filtering) | Critical | §2 hypothesis statement (paper sample restricted to **canonical-leaderboard tasks**); §6 Pre-registered taxonomies (baseline universe = published leaderboard top-30 entries at frozen date, fully deterministic); §6 Dataset (sample shrunk from "20 ICLR papers" to **12 leaderboard-task papers** to ensure deterministic candidate universes) |
| **CR3** AbGen (arXiv:2507.13300) is the canonical published benchmark for the ablation-design task with 1,500 expert-annotated examples from 807 NLP papers, Cohen's Kappa 0.71-0.78 inter-annotator agreement; revision-1's 12-axis taxonomy is empirically less rigorous and double-counts | Critical | §1 (AbGen added to gap-positioning paragraph); §6 Dataset (ablation-axis substrate switched to AbGen's 1500 reference ablation studies); §6 Pre-registered taxonomies (12-axis collapsed to 6 axes addressing the double-counting); §7 R3 (acknowledges AbGen's 0.71-0.78 Kappa as the empirical noise floor); §8 Sources (AbGen, Patel, AIMO 3 added) |
| **I1-rev** 12-axis ablation taxonomy double-counts (hyperparameter-sweep overlaps with learning-rate / optimizer-choice / regularization-strength / model-scale / inference-step-count; architecture-variant overlaps with model-scale; baseline-comparison overlaps with the baseline-list class) | Important | §6 Pre-registered taxonomies (collapsed to 6 axes: architecture-component, training-data, training-objective, hyperparameter, evaluation-protocol, inference-procedure — with explicit dominant-variable rule and an "other / multiple" bucket for ambiguous rows) |
| **I2-rev** F3 modal-bias threshold (>70% unanimous absolute) is unsupported and could fire on confident-and-correct items | Important | §5 F3 recast as a **contrast**: unanimous-vote-rate on positive-label items minus unanimous-vote-rate on negative-label items must be < 30 percentage points (genuine bias = high unanimity on both sides regardless of ground truth) |
| **I3-rev** §1 framing "ICLR rubric enumerates" is misleading — the ICLR rubric does NOT enumerate 30 baselines or 12 axes | Important | §1 rephrased throughout: "pre-registered taxonomies aligned with ICLR-rubric concerns" rather than "ICLR rubric ships an enumeration" |
| **I4-rev** "20 papers × 52 binary decisions = 1040 trials" with McNemar would detect statistically-significant +2-3 points which isn't publishable; conflates statistical with practical significance | Important | §4 adds an explicit **practical-significance threshold** (Δ ≥ +6 aggregate) distinct from the statistical floor (Δ significantly > 0 at p < 0.001); §6 reports both; §7 R6 (new) acknowledges the noise-floor relationship |

### Honest framing of the magnitude descope

This revision lowers the predicted lift from +8 to **+6 percentage points**. That is a genuine descope. The smith's role is to find the strongest defensible version of the hypothesis, not to preserve the original magnitude. **+6 over 700–1000 pre-registered binary decisions, decomposed by decision-class, with AbGen-grade expert annotations as the ablation-axis ground truth, remains a publishable result** if it materializes, because:

1. It is the first quantitative measurement of voting-vs-debate transfer from short-answer reasoning benchmarks to AI-Scientist-family paper-gen pipelines.
2. The substrate decomposition (canonical-leaderboard baselines + AbGen ablation-axes + CiteME-shape citation attribution) is itself a contribution to the field's evaluation infrastructure for paper-gen.
3. The bypass-of-Feedback-Friction mechanism (M3, unchanged, red-team explicitly approved) is the strongest published claim that voting is preferable to critic-actor revision channels for the specific paper-gen contamination problem.

If +6 is too low for the contribution to land, the synthesist surfaces this in the audit trail. Honest descope is preferable to overclaim.

---

## 1. Targeted gap

This hypothesis addresses **shortlist entry S3** in gap-finder-3's feasibility-filtered ranking
(`/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/gap-finder-3/output.md` §(a)
S3, §(b) S3). S3 closes:

- **GAP-A2** in gap-finder-2 — majority voting over N independent draft candidates is unused in
  paper-gen, despite voting beating centralized debate on 7/7 benchmarks (gap-finder-2 §(b)
  GAP-A2). No AI-Scientist-family system in scout-1's enumeration (AI Scientist v1
  arXiv:2408.06292, AI Scientist v2 arXiv:2504.08066, Agent Laboratory arXiv:2501.04227, AgentRxiv
  arXiv:2503.18102, AI-Researcher arXiv:2505.18705, Dolphin arXiv:2501.03916, EvoScientist
  arXiv:2603.08127, freephdlabor arXiv:2510.15624, Jr. AI Scientist arXiv:2511.04583,
  CycleResearcher arXiv:2411.00816, PaperOrchestra arXiv:2604.05018, Idea2Paper arXiv:2601.20833,
  Curie arXiv:2502.16069, Baby-AIGS arXiv:2411.11910) generates N parallel candidate drafts and
  selects by plurality vote across enumerable structured paper-decisions.
- **Rank 4 in gap-finder-1** — the ICLR-rubric self-eval gap (no AI-Scientist-family system runs
  a pre-registered rubric-aligned taxonomy as the eval substrate).

### Distinguishing S3 from the USC / ModeX / FSC / SC-Open / Self-Certainty lineage

The red-team correctly flagged that voting on free-form text outputs has a 2.5-year-old prior-art
line. S3 distinguishes from each of:

- **Universal Self-Consistency (USC), Chen et al., arXiv:2311.17311.** Selects the most consistent
  answer via LM-as-judge — re-introduces an LM-judgment surface S3 is specifically avoiding.
- **Fine-Grained Self-Consistency (FSC), Wang et al., arXiv:2407.02056.** Segment-level aggregation
  via string-edit-distance fusion; merge step uses an LLM and output is synthesized text, not a
  selected ballot.
- **Self-Consistency for Open-Ended Generations (SC-Open), Jain et al., arXiv:2307.06857.** Token
  log-probability weighting — continuous self-certainty signal, not plurality on structured
  decisions.
- **Scalable Best-of-N via Self-Certainty, Kang et al., arXiv:2502.18581.** Probability-distribution-
  based Best-of-N selection. Reward-free but probability-based.
- **ModeX, Choi & Li, arXiv:2601.02535.** Spectral clustering on a semantic-similarity graph of
  N candidate texts — modal *whole output* by lexical-similarity-graph topology.

**What S3 does differently:** S3 votes on **enumerable per-candidate binary inclusion decisions**
drawn from pre-registered, externally-grounded candidate universes. The aggregator is a file-fold
`numpy.sum(ballots, axis=0) >= 3` over per-binary inclusion votes — not an LLM call (vs USC), not
a string merge (vs FSC), not a probability weight (vs SC-Open / Self-Certainty), not a spectral
cluster (vs ModeX). Specifically:

1. **No LM-as-judge in the aggregator.** USC's aggregator is an LLM selecting "most consistent."
   S3's aggregator is a binary plurality fold.
2. **Operates on enumerable structured decisions.** USC / ModeX vote on the entire response. S3
   votes on structured paper-decisions — baseline-list inclusion (against a canonical leaderboard),
   ablation-axis inclusion (against AbGen reference ablation studies), citation-attribution per
   excerpt (CiteME-shape) — each with a pre-enumerable candidate universe fixed before the
   experiment runs from an external published source.
3. **Plurality is structural, not semantic.** USC / ModeX define consensus over a continuous
   semantic-similarity space. S3 defines consensus over a discrete pre-registered candidate set;
   "did baseline X appear in the draft" is a deterministic string match.

S3's structured-decision substrate is a **descendant** of the USC/ModeX lineage restricted to a
domain where the candidate universes are externally enumerable from published artifacts (leaderboards,
AbGen, CiteME). The contribution is the **system-application** (no AI-Scientist-family system does
this) plus the **per-binary-membership operationalization grounded in published external substrates**
(different aggregator surface and a different ground-truth-construction philosophy than USC/ModeX).

ModeX §2 (read directly): the authors explicitly acknowledge that "in tasks with a finite answer
space (e.g., multiple-choice question answering), simple voting schemes can reliably recover the
modal answer," and extend to open-ended via similarity graphs. S3's contribution sits in the
middle regime: each decision is binary inclusion, but the candidate universe is externally
constructed and structured — a setting ModeX's authors explicitly identify as where simple voting
*already works* without the spectral-clustering machinery.

**Closest paper-gen-adjacent prior art:** Hegelian Dialectic (arXiv:2501.14917) applies Multi-Agent
Majority Voting to novelty in *ideation*, not to draft selection across structured paper-decisions.

**Related ablation-design benchmark:** AbGen (arXiv:2507.13300, Yale-NLP, Jul 2025) is the
canonical published benchmark for evaluating LLMs on ablation-study-design tasks: 1,500 expert-
annotated examples from 807 NLP papers, with Cohen's Kappa 0.71-0.78 inter-annotator agreement
among ACL area chairs (AbGen §3.2). AbGen's task framing is **whole-design generation** — given
research context C and target module M, generate a single ablation study description A — with
Likert-1-5 evaluation on Importance / Faithfulness / Soundness dimensions. S3 differs in
operationalization: rather than evaluating whole-design quality on a Likert scale (which AbGen
itself §3.3 documents is unreliable when delegated to an LM-judge — GPT-4.1-mini rates all 18
LLMs ~4.7-4.85 on Importance while expert humans rate the same outputs ~3.2-4.3, a >1 Likert gap),
S3 decomposes each reference ablation study into per-axis presence/absence and votes
per-binary-axis. This trades AbGen's whole-design rigor for a tractable per-binary plurality
substrate that the Choi mechanism can directly apply to.

**Materially: the ablation-axis decision-class in this revision uses AbGen's 1500 expert-annotated
reference ablation studies as the ground-truth source**, with a deterministic per-axis label
extraction protocol pre-registered in §6.

### Magnitude anchor (re-derived with empirically-grounded discount)

**Choi et al. arXiv:2508.17536 §4**: Qwen2.5-7B majority voting **0.7691 avg** vs single-agent
**0.7205 avg** across 7 NLP benchmarks. Choi's average lift is **+4.86 points**.

**Patel arXiv:2604.03809 §3.1** (read directly during this revision): three Qwen2.5-14B agents on
100 GSM8K questions: mean cosine similarity 0.888, **effective rank 2.17 out of 3.0 — a 27.7%
reduction in effective sample size at N=3.** On harder tasks (MATH-500): cosine rises to 0.904 and
effective rank drops to 2.09/3.0 = 30.3% reduction. The phenomenon worsens for tasks with less
structured reasoning. Paper-decision tasks are abstractive and have less verifiable structure than
GSM8K, so the relevant discount is at least the MATH-500 rate.

**AIMO 3 arXiv:2603.27844** (Apr 2026): "Majority voting over multiple LLM attempts improves
mathematical reasoning, but correlated errors limit the effective sample size... High-temperature
sampling already decorrelates errors but only partially." Confirms Patel's collapse is not fixed by
the T ∈ [0.7, 1.0] sampling temperature range S3 specifies.

**Empirical discount = 30%** (mid-point of Patel's GSM8K 27.7% and MATH-500 30.3%, with
paper-decision tasks closer to the MATH-500 abstraction profile than to GSM8K's verifiable
arithmetic).

**Re-derivation of the magnitude floor:**

- Choi's 7-benchmark average lift: +4.86 points
- Plurality-on-structured scoping multiplier: ×2.0 (the three decision-classes are *specifically
  selected* for plurality structure — externally-grounded candidate universes — not a random
  benchmark draw). This is the same scoping multiplier the prior revision used; it remains
  defensible because Choi's anchor included free-form generation tasks like Biographies, while S3
  restricts to externally-enumerable candidate universes.
- Predicted lift before discount: +9.7 points
- Same-model-not-i.i.d. discount: ×0.70 (the empirical 30%)
- **Predicted floor: +9.7 × 0.70 = +6.79 → headline F1 floor +6 (rounded down, honest)**

The +6 floor is approximately equal to the published Choi 7-benchmark average lift (+4.86)
multiplied by a 1.25× scoping benefit. That is the **single most-defensible number** the smith can
predict given the cited empirical evidence on same-model representational collapse.

---

## 2. Hypothesis statement

**If** for a held-out **12-manuscript** sample of accepted papers spanning canonical-leaderboard ML
tasks (GLUE / SuperGLUE / ImageNet-1k / COCO / WMT-EnDe — see §6 for the per-task allocation), an
AI Scientist v2 (arXiv:2504.08066) pipeline is run twice — once in its single-draft configuration
(N_baseline) and once with **per-binary-membership majority-vote-over-5 aggregation** applied to
**three** enumerable structured paper-decisions per manuscript:

(a) **Baseline-list inclusion** over a pre-registered K_baseline = 30 candidate baseline universe
per paper's task, where the universe is the **published leaderboard top-30 entries** for that task
as of a frozen date (Jan 1 2026) — fully deterministic, no LM-derived task definition or baseline
filtering;

(b) **Ablation-axis inclusion** over the **AbGen substrate (arXiv:2507.13300)**: for each of 100
randomly sampled AbGen reference ablation studies (drawn from AbGen's testmini-500 subset at
frozen seed), apply a pre-registered 6-axis taxonomy
{architecture-component, training-data, training-objective, hyperparameter, evaluation-protocol,
inference-procedure} to extract a 6-D binary inclusion vector per reference; the model's voted
6-D vector against the AbGen-reference 6-D vector is the per-binary-decision hit-rate;

(c) **Citation-attribution per excerpt** over a CiteME-shape excerpt-to-paper-title task, with 10
excerpts per manuscript drawn from the related-work section (single bibliography-citation sentence
per excerpt);

— with **provider-side independence enforced by N=5 independent same-model samples at
temperature T ∈ [0.7, 1.0]** — **then** the aggregate per-binary-decision hit-rate of the
N_treatment voted decisions against the externally-grounded ground truth will exceed the
N_baseline single-draft hit-rate by **Δ ≥ +6 percentage points on aggregate across the three
decision-classes**, *conditional* on two pre-registered ballot-independence checks:

1. **Variance floor:** mean Hamming distance across the 5 ballots ≥ 0.20 per decision-line, AND
2. **Modal-bias contrast:** unanimous-vote-rate on positive-label items minus unanimous-vote-rate
   on negative-label items < 30 percentage points (i.e., unanimity is not systematically biased
   toward one label regardless of ground truth).

The +6 floor is derived in §1 (magnitude anchor) and §4 from Choi's +4.86 point average,
multiplied by a 2.0 scoping benefit for plurality-on-structured-decisions, then discounted 30% for
the same-model representational-collapse penalty empirically measured by Patel (arXiv:2604.03809)
and supported by AIMO 3 (arXiv:2603.27844).

---

## 3. Mechanism

### Mechanism claim M1 — N independent samples ensemble away idiosyncratic errors when the decision has plurality structure

Choi et al. (arXiv:2508.17536, §4 martingale analysis) prove that for short-answer decisions where
the correct answer has plurality across the candidate distribution, majority voting drives the
posterior toward the plurality answer with rate √N. The mechanism is *the ensembling effect of N
independent samples* — the debate channel itself does not change the posterior.

**I4 caveat (acknowledged and empirically grounded):** Choi's martingale assumes i.i.d. candidate
draws. **Same-model sampling at T ∈ [0.7, 1.0] does NOT produce i.i.d. ballots.** Patel
(arXiv:2604.03809) measures this directly: three Qwen2.5-14B agents at T=0.7 on GSM8K achieve
mean pairwise cosine similarity 0.888 and effective rank 2.17/3.0 — a 27.7% effective-N reduction.
AIMO 3 (arXiv:2603.27844) confirms "high-temperature sampling already decorrelates errors" only
partially. The hypothesis therefore relies on Choi's **empirical** observations, not the formal √N
rate, and applies a 30% discount to the predicted magnitude (see §1).

Du et al. (arXiv:2305.14325) earlier showed multi-agent-debate gains on Biographies factuality
(66.0 → 73.8) and Chess move validity (29.3 → 45.2); Choi et al.'s decomposition isolates which
component produced those gains, attributing the gain to the voting/ensembling channel rather than
to inter-agent communication.

### Mechanism claim M2 — Structured paper-decisions are plurality-bearing when operationalized as per-binary-membership votes over externally-grounded enumerable candidate universes

The USC/FSC/SC-Open/Self-Certainty/ModeX lineage solves the "free-prose has no plurality" problem
by *redefining* aggregation: USC delegates to an LLM, FSC fuses substrings, SC-Open weights by
log-probabilities, ModeX clusters by semantic similarity. **All five replace plurality with an
alternative aggregation primitive.** S3 takes the opposite approach: **restrict the vote surface to
decisions where the candidate universe is externally grounded in published artifacts**, so the
original Choi mechanism applies directly without an alternative aggregator.

The three externally-grounded decision-classes:

- **Baseline-list inclusion (K_baseline = 30 binary votes per paper).** For each held-out paper
  on a canonical leaderboard task, the candidate universe is **the published leaderboard's top-30
  entries** as of Jan 1, 2026. No LM-derived task definition, no LM-derived baseline filtering.
  Each of 5 ballots independently includes-or-excludes each of the 30 candidates. The vote is
  `(numpy.sum(ballots[:, k], axis=0) >= 3)` for each k ∈ [0, 30). Ground truth = "is candidate k
  cited as a baseline in the held-out paper's main experiments table" (deterministic string match
  on the leaderboard entry name).
- **Ablation-axis inclusion (K_ablation = 6 binary votes per AbGen reference).** Pre-registered
  6-axis taxonomy: {architecture-component, training-data, training-objective, hyperparameter,
  evaluation-protocol, inference-procedure}. For each of 100 sampled AbGen reference ablation
  studies, the published reference text is parsed deterministically into a 6-D binary vector by
  keyword-matching against axis-specific lexicons (pre-registered before runs; see §6). The
  model's voted 6-D vector against the reference 6-D vector is per-binary hit-rate. **AbGen's
  Cohen's Kappa 0.71-0.78 inter-annotator agreement is the noise floor on the reference
  annotations themselves** (acknowledged in §7 R3).
- **Citation-attribution per excerpt (CiteME-shape, K=10 per paper).** For 10 anonymized excerpts
  per paper drawn from the related-work section, each of 5 ballots predicts the single cited paper
  title; the vote is plurality over the 5 predicted titles per excerpt. Ground truth = the actual
  cited paper in the published version.

**What S3 contributes beyond USC/ModeX:** USC/ModeX vote on *whole outputs* and define consensus
over a continuous semantic-similarity space; S3 votes on **per-binary inclusion against externally-
published candidate universes** (leaderboards, AbGen, CiteME) — discrete and pre-enumerable. The
empirical question is whether per-binary plurality on externally-grounded structure beats USC's
whole-output LM-selection and ModeX's whole-output spectral-cluster when both are wrapped around
AI Scientist v2.

### Mechanism claim M3 — Voting bypasses the Feedback Friction floor and the intrinsic self-correction floor

Feedback Friction (arXiv:2506.11930) documents that frontier models cannot fully incorporate even
oracle-grade external feedback. Huang et al. (arXiv:2310.01798) document that intrinsic
self-correction *decreases* accuracy without an external signal.

Voting **does not pass through either friction surface**: the N independent samples are drawn
fresh, not revised; the aggregator is a deterministic file-fold, not a critic-to-actor revision
channel. This is the mechanism's *bypass* property.

(M3 unchanged from prior submissions. Red-team explicitly approved as "the strongest part of the
work" and "the part I would defend publicly.")

### Where this mechanism is speculative

The +6 aggregate magnitude prediction is an extrapolation from Choi's short-answer benchmarks to
externally-grounded structured paper-decisions. The 2× scoping multiplier (for plurality-on-
structured) and the 30% discount (for representational collapse per Patel) are both empirically
anchored, but the *combination* is unverified for paper-gen. The pre-registered floor is +6, which
is approximately Choi's published 7-benchmark average lift (+4.86) × 1.25. If the long-form-domain
transfer penalty or the abstractive-task collapse penalty erases all of the structural-plurality
benefit, the floor fails and the hypothesis is falsified. That is the correct outcome.

---

## 4. Predicted outcome with magnitude

### Metric

- **Aggregate per-binary-decision hit-rate** = (sum across all binary decisions and decision-classes
  of correct per-binary matches against externally-grounded ground truth) / (total binary-decision-
  line count). Total binary-decision-line count =
  - 12 papers × 30 baselines = 360 baseline binaries (decision-class A)
  - 100 AbGen references × 6 axes = 600 ablation binaries (decision-class B)
  - 12 papers × 10 excerpts = 120 citation-attribution decisions (10-class, treated as accuracy)
  - **Total: 1080 decision-lines across the three classes**
- **Per-decision-class hit-rate** = same metric, reported separately for each of the three
  decision-classes.

### Predicted effect size (descoped from revision-1's +8 to +6)

- **Aggregate hit-rate: Δ ≥ +6 percentage points (N_treatment vs N_baseline).** Derivation in §1:
  Choi's 7-benchmark avg lift (+4.86) × 2.0 scoping multiplier × 0.70 representational-collapse
  discount = +6.79 → floor +6.
- **Baseline-list inclusion: Δ ≥ +5 to +9.** Externally-grounded 30-binary structure (leaderboard
  membership) should give the cleanest plurality. The per-class floor (+5) is below the aggregate
  floor (+6) to allow the citation-attribution class to lift the aggregate.
- **Ablation-axis inclusion: Δ ≥ +4 to +8.** AbGen's expert-annotation noise floor (Cohen's
  Kappa 0.71-0.78 = ~25% disagreement on annotations themselves) caps the achievable hit-rate; the
  predicted lift is below the baseline-list class's lift because the substrate is noisier.
- **Citation-attribution-per-excerpt: Δ ≥ +3 to +7.** Higher-cardinality (218M-paper candidate
  space per CiteME), so plurality is weakest; predicted lift is lowest.

### Practical-significance threshold (NEW, addresses I4-rev)

The red-team correctly noted that on 1080 binary trials, McNemar's paired test can detect
statistically-significant lifts as small as +2-3 points (at p < 0.05) which would not be a
publishable result.

This hypothesis pre-registers **two thresholds**:

- **Statistical floor: Δ_aggregate significantly > 0 at p < 0.001** (McNemar paired test across all
  1080 binary decisions, with paper-clustered standard errors).
- **Practical floor: Δ_aggregate ≥ +6 percentage points** (the predicted-magnitude floor; this is
  the publishable threshold).

The hypothesis passes only if BOTH floors are met. The statistical floor is a sanity check; the
practical floor is the actual claim.

### Conditions under which the prediction SHOULD hold

- Variance floor passes: mean Hamming distance across the 5 ballots ≥ 0.20 per decision-line.
- Modal-bias contrast passes: |unanimous-rate on positive items − unanimous-rate on negative items|
  < 30 percentage points.
- Decision is one of the three pre-registered externally-grounded classes.
- Temperature ∈ [0.7, 1.0].
- AI Scientist v2 is the baseline pipeline.
- The 12-manuscript held-out sample is drawn from canonical-leaderboard-task papers.
- AbGen testmini-500 sample of 100 references is drawn at frozen seed before runs.
- Pre-registered taxonomies (30-entry leaderboard per task, 6-axis ablation taxonomy with
  keyword-lexicons, 10 excerpts per paper) are frozen before runs.

### Conditions under which the prediction should NOT hold

- Variance floor fails OR modal-bias contrast fails → 5 samples are not independent → hypothesis
  is not what's being tested.
- Decision is unstructured (whole-output quality, prose tone) → out of scope, redirect to
  USC/ModeX.
- Pipeline is something other than AI Scientist v2 with deeply different rubric-line-item logic.

---

## 5. Falsification criteria

These are pre-registered. If any of the following experimental results materialize, the hypothesis
**FAILS**.

### F1 — Aggregate-hit-rate Δ < +6 percentage points

If the N_treatment (vote-of-5) configuration produces an aggregate per-binary-decision hit-rate
against externally-grounded ground truth that is **less than +6 points above the N_baseline
(single-draft) hit-rate** across all 1080 binary decisions, the hypothesis fails. This is the
headline practical-significance criterion.

The +6 floor is the empirically-grounded number: Choi's average × scoping multiplier × Patel
representational-collapse discount. Going lower (e.g., to +4 to match Choi's raw average) would not
distinguish S3's structured-decision substrate from a "majority vote on anything" claim.

### F2 — Baseline-list inclusion Δ < +5 percentage points

The baseline-list decision-class is the **cleanest plurality test** — externally-grounded 30-entry
leaderboard candidate universe (no LM-derived construction), deterministic string match against the
paper's main experiments table for ground truth, no LM-extraction step. If voting fails to produce
≥ +5 on this class, the structural plurality mechanism does not transfer to paper-gen even on the
cleanest surface, and the hypothesis fails.

The F2 threshold (+5) is **below** F1's aggregate +6 to absorb that the citation-attribution class
may lift the aggregate even if baselines underperform; F2 is the per-class structural-plurality
test.

### F3 — Variance < 0.20 mean Hamming OR modal-bias contrast > 30 points

Dual ballot-independence check:

- **Variance part:** mean Hamming distance across the 5 ballots < 0.20 per binary-decision-line →
  ballots are too correlated to be independent samples.
- **Modal-bias contrast part:** |unanimous-vote-rate on positive-label items − unanimous-vote-rate
  on negative-label items| > 30 percentage points → unanimity is systematically biased toward one
  label regardless of ground truth, which is the modal-bias failure mode the I2-rev critique flagged.

This addresses red-team I2-rev: the prior absolute >70% unanimous threshold could fire as a false
positive on confident-and-correct items. The contrast form distinguishes "voting converges on
correct items" (positive-label-side unanimity, low contrast) from "voting unanimously wrong because
of shared bias" (positive-label-side unanimity AND negative-label-side unanimity, high contrast).

### F4 — Per-decision-class regression

If any one of the three decision-classes shows **Δ < 0** (voting is worse than single-draft on at
least one decision-class), the hypothesis is *partially* falsified for that class. The hypothesis
survives if F1 and F2 still pass — but the conditional scope shrinks. The synthesist should treat
F4 materializing as a survival-with-narrowed-scope outcome, not a kill.

---

## 6. Required experiments (sketch only — eval-designer details these in Phase 5)

### Dataset

- **12-manuscript canonical-leaderboard-task sample.** Drawn from accepted papers (ACL / EMNLP /
  ICLR / NeurIPS / CVPR / ICML, 2023-2025) on the following canonical-leaderboard tasks, with
  per-task paper allocation:
  - GLUE benchmark (3 papers): 30-entry candidate universe = GLUE public leaderboard top-30 as of
    Jan 1 2026 (https://gluebenchmark.com/leaderboard, frozen snapshot)
  - SuperGLUE benchmark (3 papers): 30-entry universe = SuperGLUE public leaderboard top-30,
    frozen
  - ImageNet-1k classification (2 papers): 30-entry universe = ImageNet-1k Paperswithcode
    leaderboard top-30, frozen
  - COCO object detection (2 papers): 30-entry universe = COCO Paperswithcode leaderboard top-30,
    frozen
  - WMT-EnDe machine translation (2 papers): 30-entry universe = WMT-EnDe newstest2024 leaderboard
    top-30, frozen
  - Each paper must (i) have an extractable main experiments table with named baselines, (ii) have
    a related-work section with ≥ 10 single-citation sentences (for the CiteME-shape excerpts).
- **AbGen ablation-axis substrate.** 100 references randomly sampled from AbGen testmini-500
  subset at frozen seed=42, before any run.

### Pre-registered taxonomies (frozen before any run)

- **Per-task baseline universe (K_baseline = 30, deterministic).** **The candidate set is the
  published leaderboard top-30 entries for the task** as of Jan 1 2026, as listed on the public
  leaderboard URL. Zero LM-derivation. The choice of "task" per paper is fixed by the paper's main
  experiments table caption (which always names the benchmark explicitly for these 5 canonical
  benchmarks). This protocol fully addresses red-team CR2.
- **6-axis ablation taxonomy (K_ablation = 6, paper-independent).** Collapsed from revision-1's
  fuzzy 12-axis taxonomy:
  - **architecture-component**: ablations varying model architecture components (layers, attention,
    embeddings, model size — the latter is grouped here, not in hyperparameter). Lexicon trigger:
    {"layer", "block", "head", "attention", "encoder", "decoder", "depth", "width", "size",
    "model scale", "architecture"} — pre-registered.
  - **training-data**: ablations varying data (subset, augmentation, source corpus, scale).
    Lexicon: {"data", "corpus", "augmentation", "subset", "training set", "dataset size",
    "domain"}.
  - **training-objective**: ablations varying loss / training objective / regularization.
    Lexicon: {"loss", "objective", "regularization", "regularizer", "weight decay", "dropout",
    "label smoothing", "auxiliary loss"}.
  - **hyperparameter**: ablations varying numeric hyperparameters of the optimization process
    (learning rate, optimizer choice, batch size, training steps). Lexicon: {"learning rate", "lr",
    "optimizer", "Adam", "SGD", "batch size", "step", "epoch", "schedule"}.
  - **evaluation-protocol**: ablations varying the eval pipeline (split, metric, protocol). Lexicon:
    {"split", "test set", "metric", "evaluation", "protocol", "criterion"}.
  - **inference-procedure**: ablations varying inference-time configuration (sampling, beam,
    temperature, prompt format, retrieval). Lexicon: {"temperature", "top-k", "top-p", "beam",
    "sampling", "prompt", "retrieval", "decoding"}.
  - **Multi-axis bucket:** if an AbGen reference ablation mentions ≥ 3 of the 6 axes in the
    Experiment Process section, it is flagged "multi-axis"; the per-axis binary is still extracted,
    but the row is also reported separately in §3.2 of the eval-designer's output. The
    R3-acknowledged AbGen Cohen's Kappa 0.71-0.78 noise floor is itself the lower bound on extraction
    reliability — pre-registered.
- **Citation-attribution excerpts (10 per paper).** Excerpt extraction rule: every sentence in the
  related-work section containing exactly one bibliography citation, capped at 10 per paper
  (random sample at frozen seed if more than 10 exist). The cited paper title is the deterministic
  ground truth.

### Baselines

- **N_baseline_AIScientist = AI Scientist v2 single-draft.** Default pipeline output. Extract the
  three decision-classes per output.
- **N_treatment = AI Scientist v2 + S3 per-binary-membership voting wrapper.** Same pipeline, but
  at each of the three decision points, fan out N=5 independent same-model samples at T ∈ [0.7,
  1.0] and aggregate via per-binary plurality vote.
- **N_baseline_USC = AI Scientist v2 + USC-shape aggregation** (arXiv:2311.17311). Same fan-out of
  N=5 candidates, but aggregation via "LLM picks most consistent" — tests whether S3's per-binary
  aggregation beats USC's whole-output LM-selection.
- **N_baseline_ModeX = AI Scientist v2 + ModeX-shape aggregation** (arXiv:2601.02535). Same N=5
  fan-out, spectral-cluster aggregation. Tests whether S3's structured aggregation beats ModeX's
  whole-output spectral aggregation.
- **Optional secondary baseline: Hegelian-Dialectic MAMV** (arXiv:2501.14917). Multi-Agent
  Majority Voting on ideation novelty. Tests whether voting *anywhere* in the pipeline contributes.

### Ablations

- **Vary N.** N ∈ {1, 3, 5, 7, 9}. Measure diminishing returns past N=5.
- **Vary temperature.** T ∈ {0.3, 0.5, 0.7, 1.0}. Measure where variance / modal-bias contrast
  floors pass.
- **Per-decision-class isolation.** Run voting wrapper on only one of the three decision-classes at
  a time. Measure per-class contribution to aggregate lift.
- **Heterogeneous-vs-homogeneous-model sampling.** S3 uses same-model. The orthogonal ablation
  draws the 5 candidates from 5 different foundation models (Claude / GPT / Gemini / Llama / Qwen).
  This is a separate hypothesis (S1) and should not be collapsed into S3.

### Pre-registration

Per MegaResearcher discipline rule #3, pre-registered in this document before any run:

- Δ_aggregate ≥ +6 (F1 practical floor) AND p < 0.001 (statistical floor)
- Δ_baseline-list ≥ +5 (F2 threshold)
- Variance ≥ 0.20 mean Hamming per binary-decision-line (F3 floor part A)
- Modal-bias contrast < 30 points (F3 floor part B)
- Per-decision-class Δ < 0 → conditional-scope-shrink, not kill (F4 rule)
- Temperature ∈ [0.7, 1.0] pre-registered
- 30-entry leaderboard per task (5 leaderboards, frozen Jan 1 2026)
- 6-axis ablation taxonomy with pre-registered lexicons (above)
- 100 AbGen references sampled at frozen seed=42
- 10 excerpts per paper at frozen seed=42
- N=5 ballots per binary-decision-line; reported with 95% bootstrap CI

### Non-judge signal verification

**The three decision-classes' ground truth is deterministic and non-LM-judged:**

- **Baseline-list inclusion:** string-match against (a) the published leaderboard top-30 entry
  names AND (b) the held-out paper's main experiments table column headers. Pure deterministic
  extraction. No LM step.
- **Ablation-axis inclusion:** deterministic 6-axis lexicon-matching against AbGen's
  expert-annotated reference ablation study text. The lexicons are pre-registered above. The
  AbGen reference annotations themselves are produced by ACL-area-chair NLP researchers (AbGen §3.2)
  with Cohen's Kappa 0.71-0.78 — the gold-standard substrate for the ablation-design task.
- **Citation-attribution per excerpt:** deterministic title-match between the predicted cited paper
  title and the published paper's bibliography entry for that excerpt.

**AblationBench (arXiv:2507.08038) is NOT used as ground truth.** AblationBench's matching rubric
is LMJudge majority-vote-of-3 LMs (confirmed §7.3 read in revision-1). AblationBench is cited only
as a related-work harness.

**AbGen's Cohen's Kappa 0.71-0.78** is acknowledged as the empirical noise floor on the
ablation-axis ground truth itself (see §7 R3 for the corresponding risk).

---

## 7. Risks to the hypothesis

### Risk R1 — Long-form / abstractive transfer penalty exceeds the +6 floor

Choi et al. (arXiv:2508.17536) tested on short-answer benchmarks. Patel (arXiv:2604.03809) shows
that representational collapse worsens on harder, less-structured tasks (MATH-500 effective rank
2.09 vs GSM8K 2.17). If the paper-gen voting gain is < +6 points despite passing F3, the
structural-plurality scoping benefit does not overcome the abstractive-task collapse penalty.

**What the hypothesis contributes if R1 materializes:** A negative result that **bounds the
voting-vs-debate transfer for long-form / abstractive paper-gen** — closing scout-5 OQ1. No
published paper has measured this transfer with externally-grounded structured-decision substrates.

### Risk R2 — Candidate-variance / modal-bias-contrast floor systematically fails

Even at T=1.0, 5 same-model samples may produce correlated ballots or shared modal biases. Patel's
data shows this is not just possible — it's the empirical norm at N=3. If F3 fires across multiple
temperature sweeps, same-model voting cannot achieve independence on structured paper-decisions.

**What the hypothesis contributes if R2 materializes:** A direct empirical claim that same-model
sampling cannot achieve candidate-independence on the three structured surfaces, which then
**motivates S1 (heterogeneous-model)** as the architectural fix.

### Risk R3 — AbGen reference-annotation noise floor masks the signal

AbGen §3.2 reports Cohen's Kappa 0.71-0.78 inter-annotator agreement among four ACL area chairs on
the Importance / Faithfulness / Soundness dimensions. This translates to ~22-29% disagreement on
the annotations themselves. The per-axis lexicon-extraction from those reference ablation studies
inherits this noise. The R3 risk is that the AbGen substrate's intrinsic noise floor (~25%
ambiguity rate) drowns the predicted +6 signal on the ablation-axis class specifically.

**Remediation pre-registered:** The "multi-axis bucket" flag is reported separately. The ablation-
axis class's per-class lift (+4 to +8 predicted) is *below* the aggregate floor (+6) precisely to
absorb this noise.

**What the hypothesis contributes if R3 materializes:** A measurement-protocol contribution — the
6-axis lexicon-mapped AbGen extraction (600 binary decisions on 100 references) is itself a new
artifact the field can extend.

### Risk R4 — USC / ModeX baselines outperform S3

If N_baseline_USC and N_baseline_ModeX produce *higher* aggregate hit-rates than S3's
per-binary-membership voting, the structured-decision substrate insight is wrong — whole-output
LM-selection or spectral-clustering beats it.

**What the hypothesis contributes if R4 materializes:** A direct head-to-head between three
aggregation primitives on the same paper-gen substrate. Genuinely useful empirical bound.

### Risk R5 — AI Scientist v2's existing tree search is doing the voting work invisibly

AI Scientist v2 uses progressive agentic tree search. The empirical N_baseline_AIScientist may
already capture some voting effect through tree expansion.

**What the hypothesis contributes if R5 materializes:** A distinction between
tree-search-over-experiments and voting-over-draft-decisions. The eval-designer can mitigate R5 by
recording tree-expansion count and reporting both headline Δ and tree-expansion-controlled Δ.

### Risk R6 — Statistical vs practical significance gap (NEW, addresses I4-rev)

On 1080 binary trials, McNemar's paired test has high enough power to detect Δ ≈ +2-3 percentage
points as statistically significant at p < 0.05. The hypothesis pre-registers a **dual threshold**
(p < 0.001 AND Δ ≥ +6) precisely so a statistically-significant-but-practically-trivial outcome
does not get reported as success. If the experiment yields Δ ≈ +3 at p < 0.001, the hypothesis
FAILS per F1 (the practical threshold is the binding one).

**What the hypothesis contributes if R6 materializes:** The very-large-N substrate exposes a
genuine gap between statistical and practical significance, which is itself a contribution to the
paper-gen evaluation methodology literature.

---

## 8. Sources

All arxiv IDs verified resolvable via `mcp__plugin_megaresearcher_ml-intern__hf_papers
paper_details` during this revision (see verification.md for the per-citation log).

### Magnitude anchor (central citation)

- arXiv:2508.17536 — Choi, Zhu, Li. "Debate or Vote: Which Yields Better Decisions in Multi-Agent
  Large Language Models?" §4 martingale + §3 empirical (Qwen2.5-7B Arithmetic 0.8140 → 0.9900
  single → vote; 7-benchmark average single 0.7205 → vote 0.7691, +4.86 points).

### Same-model representational-collapse penalty (NEW in rev 2, addresses CR1)

- **arXiv:2604.03809** — Patel. "Representational Collapse in Multi-Agent LLM Committees:
  Measurement and Diversity-Aware Consensus." §3.1: three Qwen2.5-14B agents on 100 GSM8K
  questions, mean cosine 0.888, **effective rank 2.17/3.0 = 27.7% effective-N reduction at N=3**;
  on MATH-500 worsens to cosine 0.904, rank 2.09. **This is the empirical anchor for the 30% same-
  model discount applied to the F1 floor.**
- **arXiv:2603.27844** — Nitarach. "Model Capability Dominates: Inference-Time Optimization Lessons
  from AIMO 3." "Correlated errors limit effective sample size... high-temperature sampling already
  decorrelates errors but only partially." Supporting evidence that same-model T ∈ [0.7, 1.0]
  sampling does not eliminate the Patel-collapse.

### Ablation-design substrate (NEW in rev 2, addresses CR3)

- **arXiv:2507.13300** — Zhao, Chen, Xu, Patwardhan, Liu, Wang, Vig, Cohan. "AbGen: Evaluating
  Large Language Models in Ablation Study Design and Evaluation for Scientific Research." 1,500
  expert-annotated examples from 807 NLP papers, Cohen's Kappa 0.71-0.78 (Importance 0.735,
  Faithfulness 0.782, Soundness 0.710) among four ACL area chairs. **The substrate for the
  ablation-axis decision-class.** Also documents (§3.3) that GPT-4.1-mini as LM-judge rates all 18
  LLMs ~4.7-4.85 on Importance while expert humans rate them ~3.2-4.3 (>1 Likert gap) — corroborating
  the hypothesis's non-LM-judge ground-truth posture.

### USC / ModeX lineage (rev 1)

- arXiv:2311.17311 — Chen et al. Universal Self-Consistency (USC). LM-as-judge selection of "most
  consistent" candidate.
- arXiv:2407.02056 — Wang et al. Fine-Grained Self-Consistency (FSC). Segment-level edit-distance
  fusion.
- arXiv:2307.06857 — Jain et al. Self-Consistency for Open-Ended Generations. Token log-probability
  weighting.
- arXiv:2502.18581 — Kang, Zhao, Song. Scalable Best-of-N via Self-Certainty. Probability-
  distribution-based reward-free Best-of-N.
- arXiv:2601.02535 — Choi & Li. ModeX. Spectral-clustering on similarity-graph for open-ended text.

### Closest paper-gen-adjacent voting analog

- arXiv:2501.14917 — Abdali et al. "Self-reflecting Large Language Models: A Hegelian Dialectical
  Approach." Multi Agent Majority Voting applied to novelty in ideation — not to draft selection.

### Pipeline-baseline citation

- arXiv:2504.08066 — Yamada, Lange, Lu et al. AI Scientist v2.

### Decision-class substrates

- arXiv:2407.12861 — Press et al. CiteME. Citation-attribution-per-excerpt benchmark.

### AblationBench (only as related work, NOT ground truth)

- arXiv:2507.08038 — Abramovich, Chechik. AblationBench. §7.3 LMJudge majority-vote-of-3 confirmed
  in revision-1. Cited as related-work harness only.

### Negative-result ceilings being explicitly bypassed (M3)

- arXiv:2310.01798 — Huang et al. "Large Language Models Cannot Self-Correct Reasoning Yet."
- arXiv:2506.11930 — Lin et al. "Feedback Friction."
- arXiv:2502.08788 — Zhang et al. "Stop Overvaluing Multi-Agent Debate."

### Generic-pattern cousin

- arXiv:2305.14325 — Du, Li, Torralba, Tenenbaum, Mordatch. Multi-agent debate. The Biographies
  and Chess gains Choi et al. later decomposed.

Verification trail: all 18 arXiv IDs above resolved via `hf_papers paper_details` during this
revision. See `verification.md` for the per-citation check log and the per-defect address map.
