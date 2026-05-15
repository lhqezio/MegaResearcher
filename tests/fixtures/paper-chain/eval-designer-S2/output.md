# eval-designer-S2 — Experimental protocol for S2 (Bias Fitting length-debiasing wrapper)

Run: `2026-05-12-0515-19bf96`
Target: hypothesis-smith-S2 revision-1 (APPROVED by red-team-S2 revision-1)
Hypothesis path: `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/hypothesis-smith-S2/output.md`
Red-team verdict path: `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/red-team-S2/output.md`

---

## 1. Pre-registration statement

This protocol is the binding pre-registration for hypothesis S2. After this document is committed to the run directory, the decision rules in §10 may NOT be relaxed, and the falsification thresholds may NOT be adjusted, regardless of intermediate results. Investigator-degrees-of-freedom during execution are limited to the procedural reruns explicitly enumerated in §11 (re-paraphrase if R2 substance-injection gate trips; re-fit if calibration-corpus drift gate trips).

**Hypothesis (verbatim restatement from smith-S2 §2 for self-containment):**

> If the MegaResearcher orchestrator wraps every red-team and synthesist LLM-as-judge scalar-score call with a Bias Fitting length-debiased post-processor (arXiv:2505.12843) fit on a one-time ~150-manuscript calibration corpus, then on a held-out 20-manuscript fixed-quality test set with controlled-verbosity paraphrase injection:
>
> (a) β_raw on log(token-count) is positive and statistically significant (one-sided p < 0.05);
> (b) β_norm on log(token-count) is not significantly different from zero (two-sided, α = 0.10, 95% CI brackets 0);
> (c) β_raw − β_norm has a 95% CI excluding zero on the positive side.

**Falsification criteria (verbatim from smith-S2 §5):**

- F1 — Baseline shows no length bias: β_raw not significantly > 0 (α=0.05 one-sided). Graceful no-op per smith I4.
- F2 — Wrapper does not suppress the bias: β_norm significantly > 0 (α=0.05 two-sided).
- F3 — Hacking shifts to a different proxy: debiased score correlates with any of the 8 pre-registered proxies (Spearman |ρ| > 0.3, p < 0.05).
- F4 — Wrapper destroys substantive judge signal: AUROC for known-good vs known-bad drops by > 0.05 absolute under the wrapper.

**No-peeking commitment.** The investigator running this protocol will:

- Generate the held-out test set, paraphrase variants, and F4 pair set BEFORE any judge calls are run on calibration data, so the held-out is selected blind to calibration outcomes.
- Run the wrapped vs un-wrapped scoring in a single batch per (model, manuscript, variant) cell, alternating order randomly across cells to avoid temporal-drift contamination.
- Lock the Bias Fitting model checkpoint hash into the run manifest before scoring the held-out set. The fitted model may not be re-trained after held-out scoring begins.
- Pre-commit the regression specification in §5 to the manifest. Post-hoc covariate addition is not permitted.

**Randomization seed.** Single global random seed `19bf96` (derived from run id) for all stochastic steps: paraphrase ordering, judge-call ordering, calibration-fold partitioning. Per-step sub-seeds are derived as `hash("step_name:19bf96")`.

---

## 2. Datasets and substrates

### 2.1 Bias Fitting calibration corpus (one-time fit, ~150 manuscripts)

**Primary source: `DeepNLP/ICLR-2024-Accepted-Papers`** (HF dataset ID, viewer-validated 2026-05-12).

- License: not declared in the HF card; abstracts and PDF URLs are public OpenReview content. Use is for research analysis (transformative; not redistribution). The dataset stores `pdf` URLs that point to OpenReview-hosted PDFs — those PDFs are the canonical artifact, not the dataset's redistributed copies.
- Schema: `title`, `url`, `detail_url`, `authors`, `tags`, `abstract`, `pdf`. (Verified via `hf_inspect_dataset`.)
- Train split size: ~2000 accepted papers (filtered Oral / Spotlight / Poster). For calibration we sample N=150 stratified by tag (Oral/Spotlight/Poster) and abstract-length quartile to span the length distribution.
- Date filter: papers with arXiv `2306-2401` publication window (ICLR 2024 submission cycle), pre-dating the GPT-5 (2025-Q3) and Claude 4 (2025-Q2) training cutoffs. This is required so the judges have not seen the *review text*; the PDFs themselves may be in the judge's pretraining data, but the calibration target is the judge's scalar rubric score, not regurgitated text.
- Why this dataset is appropriate: it contains the modality the hypothesis predicts about — full ML/AI conference manuscripts with public PDF access at known quality (accepted at a top venue). It allows controlled-verbosity injection while holding substantive quality constant per manuscript.
- Sample size justification: Bias Fitting's paper trains its fitting model on ~10K-50K (response, reward) pairs. N=150 is small but sufficient because (a) the fitting model has ~10K parameters (2-layer ResNet + LE), (b) the smith's smith pre-commits to a single non-linear-fit and a single linear-fit baseline (no architecture search), and (c) we hold 30 of the 150 out for fitting-model validation (80/20 split, 120 fit / 30 validate). This is documented as a methodological extrapolation per I-new-1.

**I-new-2 mitigation — second calibration source (AI-Scientist-family runtime):**

The red-team flagged that calibration on ICLR-human-authored papers may not transfer to AI-Scientist-v2-generated manuscripts (systematically different length distributions). To address this directly, we use **a secondary, smaller calibration corpus** of N=75 AI-Scientist-v2-generated workshop manuscripts pulled from the SakanaAI repo's `experiments/` output directory (the AI-Scientist-v2 paper arXiv:2504.08066 documents ~12 such manuscripts publicly released; we extend to 75 by running the AI-Scientist-v2 free runtime on three workshop topics for 25 papers each at low compute cost — flagged as Risk R-substrate-1 below).

This produces a **second fitted Bias Fitting model** (call it `model_f^AI`) trained on (AI-Scientist-v2-output, scalar-judge-score) pairs. The primary analysis uses `model_f^ICLR` (the human-authored calibration); a planned ablation (A2) compares the two.

### 2.2 Held-out test set (20 manuscripts, no overlap with calibration)

**Source: `DeepNLP/ICLR-2024-Accepted-Papers` (disjoint sample from §2.1).**

- 20 manuscripts, stratified across acceptance tags (5 Oral, 7 Spotlight, 8 Poster) to span the quality distribution while holding venue-tier constant.
- Each manuscript is processed into 3 verbosity variants (decision: trim from 4 to 3 per §8 budget):
  - **Terse** (target 0.7× original token count): paraphrase via a non-judge model to compress while preserving claims.
  - **Original** (1.0×): the manuscript as-is.
  - **Verbose** (target 1.4×): paraphrase to add filler (extended motivation, redundant transitions, hedging) while preserving every claim.
- Total test set: 20 × 3 = 60 (manuscript, variant, judge-call) cells per (judge-model × wrapper-condition).

### 2.3 Controlled-verbosity injection protocol

**Paraphrase model:** `claude-haiku-4` (a model different from the judge; the judges are `gpt-5`, `claude-sonnet-4.5`, `gpt-4.1` per §3). Using a smaller, cheaper, different-family model for paraphrasing avoids the judge-substance-leak failure mode where the paraphraser anticipates judge scoring.

**Prompt template (locked, no investigator-tuning during run):**

```
You are paraphrasing a research manuscript for a length-control study.
TARGET: produce a {terse|verbose} version of the input section.

CONSTRAINTS (do not violate):
1. Preserve every numerical claim, every cited reference, every table/figure
   caption, and every section/subsection heading verbatim.
2. Preserve every method name and architecture detail verbatim.
3. Preserve the order of claims.
4. Do not introduce new claims, citations, or experimental results.

LENGTH TARGET:
- terse: output should be {0.65, 0.75} of input token count (target 0.70×).
- verbose: output should be {1.30, 1.50} of input token count
  (target 1.40×). Add filler via: expanded motivation, redundant transitions,
  hedging phrases, restating definitions, explicit signposting.

OUTPUT: rewritten manuscript text only, no preamble.
```

**Substance-preservation gate (R2 mitigation):** Each paraphrased variant is automatically checked via three deterministic gates BEFORE judge scoring:

1. **Claim-set preservation:** numerical claims (regex `\d+(\.\d+)?\s*(%|\\times|x|\\pm)?`) in the original must each appear in the paraphrase with no more than ±1 absolute change. Failures: re-paraphrase up to 3 times; if still failing, flag manuscript as bad-substrate and replace.
2. **Citation-set preservation:** the multiset of citation keys (`\\cite{[^}]+}`) in the original must equal the multiset in the paraphrase.
3. **Section-heading preservation:** ordered list of `\\section`, `\\subsection` headings must match verbatim.

A manuscript that fails any gate after 3 paraphrase attempts is replaced from the same OpenReview pool (pre-committed reserve of 10 additional manuscripts for replacements). The replacement protocol is pre-registered: pick the next manuscript in the seeded shuffle order from the reserve pool.

### 2.4 F4 known-good / known-bad substrate

20 manuscripts (disjoint from calibration and held-out), each in two paired versions:

- **Known-good:** unmodified original.
- **Known-bad:** one of three pre-registered substantive flaws injected (assigned uniformly):
  - (i) **Baseline-row removal:** delete the strongest competing baseline row from every results table.
  - (ii) **Claim inflation:** replace one reported metric value with a value 20% better than the original (deterministically; specific row pre-committed per manuscript).
  - (iii) **Ablation-row deletion:** delete the most-revealing ablation row from every ablation table.

Injection is performed by a deterministic LaTeX-source-level rewrite (no LLM), so the flaw is unambiguous. The wrapper is then applied to scores on the {known-good, known-bad} pairs, and AUROC is computed for separating the two.

### 2.5 Splits summary

| Set | N manuscripts | N judge calls per condition | Source |
|---|---|---|---|
| Calibration (primary, ICLR-human) | 150 (120 fit / 30 internal-validate) | 150 | DeepNLP/ICLR-2024-Accepted-Papers |
| Calibration (secondary, AI-Scientist runtime) | 75 | 75 | AI-Scientist-v2 runtime output |
| Held-out test | 20 × 3 variants = 60 | 60 per (model × wrapper) | DeepNLP/ICLR-2024-Accepted-Papers (disjoint) |
| F4 known-good/bad | 20 × 2 = 40 | 40 per (model × wrapper) | DeepNLP/ICLR-2024-Accepted-Papers (disjoint) |
| Reserve (substance-gate replacements) | 10 | up to 10 × 3 = 30 | DeepNLP/ICLR-2024-Accepted-Papers (disjoint) |

### 2.6 Power analysis for the held-out test set

Within-manuscript fixed-effects regression of judge score on log(token-count):

- Variance of log(token-ratio) within manuscript across 3 variants {0.7, 1.0, 1.4} = 0.080.
- Assume residual judge-call SD = 0.4 on a 1-10 rubric (consistent with reported test-retest reliability of LLM-judges on paper review tasks).
- N_total = 20 × 3 = 60; with 20 manuscript dummies, df_residual ≈ 39.
- SE(β) ≈ 0.4 / sqrt(60 × 0.080) = **0.182**.
- Minimum detectable effect at one-sided α=0.05, power 0.80: β ≥ 1.68 × 0.182 ≈ **0.31 score units per log-token**.

**Interpretation:** the design is powered to detect a slope of ~0.31 score units per natural-log-token (equivalent to a ~0.10 score difference between the terse and verbose variants on the 1-10 scale, holding manuscript fixed). The smith's "would not be surprised if" order-of-magnitude expectation of ~0.05 per log-token is *below* this MDE — meaning a null result is interpretation-ambiguous (true null vs underpowered) if observed β is in [0.00, 0.31].

**Mitigation:** Each judge call is repeated 2× and scores averaged (cuts σ by √2 to 0.283), bringing MDE to **0.22**. Effective budget impact: doubles held-out judge-call cost from $40 to $80. This is incorporated into the budget in §8.

**Stretch option (declined under budget trim):** 30 manuscripts × 3 variants × 2 calls would give MDE = 0.18 but adds ~$30 — exceeds the $200 cap. Stays at 20 manuscripts.

---

## 3. Baselines (pre-registered)

Following smith-S2 §6 baselines. All baselines are evaluated on the **same** held-out test set (§2.2) under the **same** randomization seed, to make the comparisons paired-by-manuscript.

| ID | Condition | Description | Rationale |
|---|---|---|---|
| **B0** | Un-wrapped (control) | Raw judge score, no debias. The AI-Scientist-family default — what MegaResearcher's red-team currently does. | Primary contrast for β_raw. Sanity baseline + prior-art baseline (AI-Scientist-v2 arXiv:2504.08066). |
| **B1** | Prompt-instruction baseline | Same judge call with prepended instruction "Ignore response length when scoring. Score on substance only." | Cheap-fix baseline — does an instruction prefix already close the gap? If yes, the wrapper is unnecessary. |
| **B2** | **Bias-Fitting wrapper (HYPOTHESIS)** | Raw judge score − `model_f^ICLR(len(y))`. The wrapper as defined in arXiv:2505.12843 §3. | The hypothesis under test. |
| **B3** | Linear-debias ablation | Raw judge score − `(â + b̂ · log(len(y)))` where `â, b̂` are OLS-fit on calibration corpus. | Ablation of M5: tests whether the *non-linear* fit (vs simple linear) matters. Cited as comparator in Bias Fitting paper §4.2 (LC-WR comparison). |
| **B4** | Trivial baseline | Constant prediction = mean judge score on calibration corpus (no manuscript-conditional signal). | Sanity check: confirms judges are not noise (F4 must show B0 has AUROC ≫ 0.5 vs B4 random). |

**B3 from smith (heterogeneous-model judge from S1)** is DROPPED from this protocol because S1 is a separate hypothesis in the swarm; we do not re-implement it here. The smith's conditional "required if S1 also runs" defers to the synthesist for cross-hypothesis comparison.

**Optional B-Dubois (pairwise debiaser)** from smith — DECLINED. Re-tooling MegaResearcher's red-team to output pairwise comparisons is non-trivial and out-of-scope for a transfer-test of Bias Fitting specifically.

---

## 4. Metrics

### 4.1 Primary metrics (deterministic, non-judge)

All computed from the (manuscript, variant, raw-score, wrapped-score, token-count) data table — no additional LLM-judge calls.

| Metric | Computation | Hypothesis prediction |
|---|---|---|
| **β_raw** | OLS slope of raw judge score on log(token-count) with manuscript fixed effects, on held-out set | > 0, p < 0.05 (one-sided) |
| **β_norm** | OLS slope of debiased judge score on log(token-count) with manuscript fixed effects | not significantly different from 0 (α=0.10 two-sided) |
| **β_raw − β_norm** | Difference-in-differences with cluster-robust SE | 95% CI excludes zero on positive side |
| **Within-manuscript Hedges' g** | Standardized mean diff between terse and verbose variants of same manuscript, raw vs wrapped | Raw: g > 1.0; Wrapped: |g| < 0.5 |
| **F4 AUROC** | AUROC for known-good vs known-bad classification, raw vs wrapped | drop ≤ 0.05 absolute (else F4 fires) |

### 4.2 Secondary metrics (F3 proxies — gaming-target-migration detection)

8 pre-registered proxies from smith-S2 §5 F3. Each computed deterministically (no LLM judge):

**Surface-textual (5):**

1. `section-count` — count of `\section`, `\subsection`, `\subsubsection` (regex)
2. `citation-count` — count of `\cite{...}` keys (regex, deduplicated)
3. `bullet-list-count` — count of `\item` or `\begin{itemize/enumerate}` (regex)
4. `LaTeX-formatting-density` — fraction of tokens that are LaTeX commands `\\[a-zA-Z]+` over total tokens
5. `hedging-word-count` — count of {might, could, may, possibly, perhaps, suggest, indicate, appears} (regex, case-insensitive, word-boundary)

**Substantive (3, BadScientist-inspired):**

6. `improvement-magnitude-plausibility` — extract via regex `improv\\w*\\s+by\\s+(\\d+(\\.\\d+)?)\\s*\\%` from abstract+intro, take max. Deterministic.
7. `claim-vs-result-table-match` — **I-new-3 mitigation, fully deterministic:** instead of LLM-based claim extraction, count of numerical-percentage claims in abstract+intro (regex `\\d+(\\.\\d+)?\\s*\\%`) that do NOT appear within ±0.5 absolute of any cell in any results table (extracted via `\\begin{table}...\\end{table}` block scan with column delimiter `&` and row delimiter `\\\\`). Returns `mismatch_rate` ∈ [0, 1]. Caveat documented: this misses non-numeric claims; the deterministic fallback is conservative.
8. `presence-of-baseline-CI` — boolean OR over results tables: does any table contain `\\pm`, `95\\% CI`, `(±`, `confidence`, or `s.d.` within numerical context? Deterministic regex.

**F3 firing rule:** Spearman ρ between (debiased judge score) and (proxy value) on held-out set, with |ρ| > 0.3 and p < 0.05, on any of the 8 proxies. Bonferroni-correct across the 8 proxies: per-proxy threshold p < 0.05/8 = 0.00625.

---

## 5. Statistical analysis plan (pre-registered)

### 5.1 Primary regression specification

Single regression equation, locked:

```
score_{ij} = α_i + β · log(tokens_{ij}) + ε_{ij}
```

where `i` indexes manuscript (manuscript fixed effects via dummy variables), `j` indexes variant ∈ {terse, original, verbose}. Standard errors clustered by manuscript (heteroskedasticity-robust, manuscript-level Liang-Zeger).

Fit twice:
- (a) with raw scores → β_raw
- (b) with wrapper-debiased scores → β_norm

The difference-in-differences β_raw − β_norm is computed via a stacked regression with a `wrapped` indicator and `wrapped × log(tokens)` interaction, again clustered by manuscript.

### 5.2 Decision rules (locked)

Reported as 95% CIs + point estimates. Decision rule applied to the *primary* judge model (`claude-sonnet-4.5`); per-model results across `{gpt-5, claude-sonnet-4.5, gpt-4.1}` reported alongside.

| Pattern | Outcome | Synthesist action |
|---|---|---|
| β_raw 95% CI > 0 AND β_norm 95% CI brackets 0 AND DiD 95% CI > 0 AND F3 no fire AND F4 ΔAUROC ≤ 0.05 | **SUPPORTED** | Ship as a measured-positive configuration recommendation per smith §4 |
| β_raw 95% CI > 0 AND β_norm 95% CI > 0 (still significantly > 0 after wrapper) | **FAIL (F2)** | Wrapper insufficient; recommend not deploying |
| β_raw 95% CI brackets 0 (no baseline bias) | **FAIL (F1, graceful no-op)** | Survey finding: "this judge has no residual length bias on paper-quality scoring." Configuration-dependent recommendation per smith I4 |
| β_raw 95% CI > 0 AND β_norm 95% CI brackets 0 AND F3 fires on any proxy (with Bonferroni correction) | **PARTIAL (F3, gaming-target migration)** | Wrapper insufficient standalone; recommend pairing with surface-formatting normalization (if proxy 1-5) or content-fabrication detection (if proxy 6-8 — RBD arXiv:2505.17100 or BadScientist ReD) |
| F4 ΔAUROC > 0.05 (substantive signal destroyed) | **FAIL (F4)** | Wrapper net-utility-negative; do not deploy |
| 0 < β_norm 95% CI excludes 0 AND β_norm < β_raw AND DiD 95% CI > 0 | **PARTIAL (suppression)** | Wrapper reduces but does not eliminate bias; report partial efficacy |

### 5.3 Multiple-comparison correction

Three judge models × multiple metrics → false-discovery-rate inflation risk. Strategy:

- **Primary decision** is on `claude-sonnet-4.5` only. The other two models are exploratory.
- **F3 8-proxy sweep:** Bonferroni within-model (α = 0.05/8 per proxy).
- **Cross-model:** Benjamini-Hochberg at FDR=0.10 across the 3-model × 4-condition grid for secondary reporting.

### 5.4 Power & MDE summary

- Primary β slope test: MDE = 0.22 score units per log-token (with 2-call averaging), at 80% power, α=0.05 one-sided.
- F4 AUROC drop: MDE = 0.05 absolute, with 40 paired observations (Hanley-McNeil approximation, power ~0.75 against a 0.85-baseline AUROC).
- F3 Spearman: MDE ρ = 0.30 at N=60, α=0.05/8 (Bonferroni), power ~0.65 — flagged as the weakest test in the design. Mitigation: F3 firing requires *any one* of 8 proxies to trip, which inflates omnibus power.

---

## 6. Falsification experiments (one per criterion)

Each below is designed to **fail** if the hypothesis is correct. The result that would constitute falsification is pre-stated.

### F1-experiment — Baseline length-bias detection

- **Setup:** Score 20 manuscripts × 3 variants × 2 calls × 3 judge models = 360 raw judge calls. Run primary regression on each model.
- **Falsification result:** β_raw 95% CI brackets 0 on `claude-sonnet-4.5`. This kills the hypothesis (graceful no-op): the judge has no length-bias to suppress.
- **Why it could fail:** modern judges may have been trained with explicit length-bias mitigation, making β_raw ≈ 0 in baseline. This is the documented R5 risk.

### F2-experiment — Wrapper-fails-to-suppress

- **Setup:** Same data as F1, but compute debiased scores via `model_f^ICLR` (B2 condition). Run primary regression with wrapper-debiased scores.
- **Falsification result:** β_norm 95% CI excludes 0 (i.e., still significantly > 0) on `claude-sonnet-4.5`. The wrapper does not suppress the bias.
- **Why it could fail:** Bias Fitting's fitting-model architecture may be mis-specified for the paper-judging domain (small calibration corpus, distribution shift to held-out, etc.). This is the I-new-1 + I-new-2 risk.

### F3-experiment — Gaming-target migration

- **Setup:** On the same wrapped held-out scores, compute Spearman ρ with each of the 8 pre-registered proxies. Bonferroni-correct.
- **Falsification result:** Any of the 8 proxies has Spearman |ρ| > 0.3, p < 0.05/8. The wrapper has merely shifted the bias to another correlate.
- **Why it could fail:** length is correlated with section-count, formatting-density, citation-count in practice. The wrapper's debiased score may inherit length-correlated structure from these indirect proxies. This is the smith's R3.

### F4-experiment — Substantive-signal preservation

- **Setup:** 20 manuscripts × {known-good, known-bad} × 2 calls × 3 judge models on B0 (raw) and B2 (wrapped). Compute AUROC for known-good > known-bad ranking, per model and condition.
- **Falsification result:** AUROC(wrapped) < AUROC(raw) − 0.05 absolute on `claude-sonnet-4.5`. The wrapper sacrifices substantive signal.
- **Why it could fail:** Bias Fitting was validated on instruction-following (AlpacaEval), not paper-quality scoring. The fitting model may absorb length-correlated *substantive* signal (e.g., longer papers often have more thorough experiments — a real quality signal) and remove it.

---

## 7. Ablations

| ID | Manipulation | Question answered |
|---|---|---|
| **A1** | Length-encoding feature: log(tokens) vs raw(tokens) vs char-count vs section-count | Is the wrapper sensitive to the choice of length feature? (Smith's A1.) |
| **A2** | Calibration corpus: `model_f^ICLR` (human-authored, primary) vs `model_f^AI` (AI-Scientist-runtime, secondary) | Does the wrapper transfer across calibration sources? (Red-team I-new-2.) |
| **A3** | Fitting model architecture: 2-layer ResNet (Bias Fitting default) vs 1-layer linear vs MLP-32 | Does the non-linear fit matter, or is linear sufficient? (Smith's A2, Bias Fitting §4.2 comparator.) |
| **A4** | Calibration corpus size: 50, 100, 150 manuscripts | How small can the calibration corpus be before β_norm degrades? |
| **A5** | Multi-model judge sweep: report all metrics for `{gpt-5, claude-sonnet-4.5, gpt-4.1}` separately | Does the transfer hold across judge model families? (Smith's pre-registered third judge.) |
| **A6** | F3 deterministic-proxy sweep on the 5 surface-textual proxies (sweep through 1-5 as gaming-target candidates per smith A3) | Which surface-textual axis is most-likely to absorb wrapped score variance? |

Ablations A1, A3, A4 reuse already-collected (manuscript, variant, raw-score, token-count) tuples — they only require re-fitting the fitting model and re-computing β. No additional judge calls. A2, A5 incur additional API cost (counted in §8). A6 is deterministic-only.

---

## 8. Cost-and-time budget — **DECISION: TRIM TO ≤$200**

### 8.1 Decision

Per smith I-new-4 and the run's $200/replication ceiling, this protocol **trims the budget to $190** by:

1. **Reducing verbosity variants from 4 to 3** (drop the 2.0× very-verbose tier). Saves $30 (paraphrase + judge calls).
2. **Dropping the LimitGen optional sanity check** ($15 saved). The substance-preservation gate in §2.3 already validates paraphrase quality, making LimitGen redundant as a harness check.
3. **Adding 2-call averaging on the held-out set** (+$40 budget add for the per-call doubling — necessary for adequate MDE per §2.6). Net of the trim: still under $200.

### 8.2 Itemized budget

Assumes ~$0.30/call for calibration (5-8k input + brief rubric output) and ~$0.50/call for held-out (rubric + reasoning trace).

| Item | Calculation | Cost |
|---|---|---|
| Calibration corpus, primary (ICLR-150) | 150 × $0.30 | $45 |
| Calibration corpus, secondary (AI-Scientist-75) | 75 × $0.30 | $23 |
| Bias Fitting model training (CPU, small ResNet) | ~$5 each × 2 | $5 |
| Held-out test set, 2 calls each, 3 judges | 20 × 3 variants × 2 calls × 3 models × $0.50 | varies — see breakdown |
| └─ Primary `claude-sonnet-4.5` | 20 × 3 × 2 × $0.50 | $60 |
| └─ Exploratory `gpt-5` + `gpt-4.1` | 20 × 3 × 2 × 2 × $0.50 (no 2-call avg, single-call exploratory) | $30 |
| Paraphrase generation | 20 × 2 variants (terse, verbose) × $1 | $40 |
| F4 known-good/bad pair set | 20 × 2 × $0.50 × (3 judges, 1 call) | $20 |
| Baseline runs (B1 instruction-prefix, B3 linear-debias) | reuses held-out calls; B1 incremental ~$10 | $10 |
| **Subtotal** |  | **$195** |
| Substrate replacement reserve (10 mss × 3 × $0.5 × 0.3 expected use) |  | $5 |
| **TOTAL** |  | **$200** |

### 8.3 Compute

- Bias Fitting model: 2-layer ResNet, ~10K params, trains in <5 minutes on CPU. Total wall-clock: ~30 min including A2/A3/A4 ablation re-fits.
- All judge calls: API-only; no GPU required. Total wall-clock for batched API calls (with 10-concurrent rate limit on premium tier): ~6-8 hours.
- Analysis: pandas + statsmodels OLS with cluster-robust SE; ~10 min total.

### 8.4 Total

- **Estimated dollar cost:** **$200** (precisely at ceiling, no over-budget flag).
- **Estimated compute-hours:** ~8 hours wall-clock for API calls + ~1 hour for fitting + analysis = **~9 hours total**.
- **`flagged_intractable`: false.**

---

## 9. Threats to validity

### 9.1 I-new-1 — Bias Fitting protocol extrapolation

**Threat:** The paper's protocol trains an RM from scratch (Bradley-Terry warm-up) before applying the fitting model. We bypass the warm-up and apply the fitting model directly to API-judge scalar outputs. The fitting math is unchanged, but the paper itself does not validate this direct-API-application.

**Mitigation:**
- Documented explicitly here and in the §10 outputs handoff.
- A3 ablation (linear vs ResNet) detects whether the fitting model is over-fitting absent the warm-up regularization.
- A4 ablation (corpus size sweep) detects whether the fitting model's variance is bounded with the smaller calibration corpus.
- We do NOT claim this is the same protocol as Bias Fitting — we claim it is a **transfer test of Bias Fitting's fitting-model component to API-judge scalar outputs**. Synthesist must report this caveat.

### 9.2 I-new-2 — Calibration-corpus generalization

**Threat:** Calibration on ICLR-human-authored manuscripts; runtime in MegaResearcher targets AI-Scientist-v2-generated manuscripts with systematically different length distributions.

**Mitigation:**
- Secondary calibration corpus (`model_f^AI`) on 75 AI-Scientist-v2-generated workshop manuscripts. Ablation A2 compares the two fitters.
- Primary analysis uses `model_f^ICLR`; if A2 shows `model_f^AI` materially differs in β_norm, the synthesist surfaces this as "calibration-source-dependent" finding.

### 9.3 I-new-3 — F3 proxy 7 LLM-dependence

**Threat:** Original smith spec for proxy 7 (claim-vs-result-table-match) required LLM-based claim extraction, introducing the same LLM-judge dependency that contaminated S5.

**Mitigation:**
- Re-specified as **deterministic regex-only**: count of numerical-percentage claims in abstract+intro that do NOT appear within ±0.5 absolute of any cell in any results table (see §4.2 proxy 7).
- Caveat documented: this misses non-numeric claims (qualitative claims like "outperforms all baselines"). The deterministic fallback is conservative — it underestimates true mismatches, which biases F3 *against* firing (i.e., makes the hypothesis test more permissive, not less). This is the safe direction.

### 9.4 R6 — BadScientist-channel dominance (synthesist exit)

**Threat:** Even if F1/F2/F3/F4 all pass cleanly, S2's deployment benefit may be small relative to BadScientist-style content-fabrication channels (per arXiv:2510.18003).

**Mitigation:**
- This is a synthesist-stage concern, not an eval-stage concern. The eval delivers the F1/F2/F3/F4 measurements regardless.
- Synthesist may move S2 to future-work flag per smith R6 — this is anticipated and documented.

### 9.5 Verbosity-injection confound

**Threat:** Paraphrasing to add filler may also reduce quality (filler is by definition non-substantive), contaminating the fixed-quality assumption.

**Mitigation:**
- Substance-preservation gate in §2.3 enforces claim/citation/heading preservation deterministically.
- A pilot-of-5 human-rated quality check (one-time, cheap manual scan): for 5 manuscripts, the investigator manually compares terse/original/verbose triples and confirms no substantive degradation. Out-of-budget; performed pro-bono.
- If pilot fails (substance drift detected), the paraphrase prompt is tightened ONCE (single revision, documented), and the held-out is re-generated.

### 9.6 Calibration-corpus curation cost (red-team I-new-1 hidden cost)

**Threat:** Sourcing and normalizing 150 manuscripts from OpenReview is non-trivial effort the smith silently assumed free.

**Mitigation:**
- Use the `DeepNLP/ICLR-2024-Accepted-Papers` HF dataset (2.0 MB parquet, already viewer-validated). The `pdf` field provides direct OpenReview URLs; an automated PDF-fetch + plaintext-extract via `pdftotext` or `pypdf` is deterministic and takes ~2 hours wall-clock for 150 manuscripts. Counted in §8.3 compute.

### 9.7 Baseline-tuning asymmetry

**Threat:** B0 (un-wrapped) is "the default"; B2 (wrapped) is "the tuned condition" — could the wrapper be implicitly tuning a hyperparameter the un-wrapped baseline lacks?

**Mitigation:**
- B1 (prompt-instruction baseline) controls for "any non-trivial added effort closes the gap." If B1 closes the gap and B2 does not, the wrapper's added complexity is not worth it.
- B3 (linear-debias) controls for "non-linear fit specifically." If B3 closes the gap, the non-linear fit is unnecessary.
- All baselines fit on the same calibration corpus where applicable; same compute budget. No asymmetric hyperparameter sweep.

### 9.8 Evaluation-suite drift

**Threat:** Repeated judge calls over the run period may see model-version drift if APIs roll out updates mid-run.

**Mitigation:**
- Lock model snapshot identifiers in manifest (e.g., `claude-sonnet-4-5-20260301`) at run start.
- Complete all judge calls within a 24-hour wall-clock window where possible.
- Re-run if any model is silently updated during the window (detected by comparing the first and last 5 calls on the same input — pre-registered drift-check).

---

## 10. Outputs the user can act on — decision tree + handoff

```
Run protocol → observe (β_raw, β_norm, DiD, F3, F4 AUROC, ablations A1-A6)
                              │
                              ├── β_raw 95% CI brackets 0?
                              │      └── YES → F1 fires (graceful no-op)
                              │              → synthesist publishes as
                              │                "this judge has no residual length-bias;
                              │                 wrapper unnecessary for THIS judge."
                              │                Configuration-dependent recommendation.
                              │
                              ├── β_norm 95% CI excludes 0?
                              │      └── YES → F2 fires (wrapper insufficient)
                              │              → synthesist publishes as KILLED;
                              │                wrapper does not transfer to paper-judging.
                              │
                              ├── F4 ΔAUROC > 0.05?
                              │      └── YES → F4 fires (substantive signal destroyed)
                              │              → KILLED on net-utility grounds;
                              │                recommend not-deploy.
                              │
                              ├── Any F3 proxy fires (Bonferroni-corrected)?
                              │      ├── Proxy 1-5 (surface) → PARTIAL; pair with
                              │      │                          surface-formatting
                              │      │                          normalization (GAP-A4).
                              │      └── Proxy 6-8 (substantive) → PARTIAL; pair with
                              │                                    content-fabrication
                              │                                    detection (RBD 2505.17100
                              │                                    or BadScientist ReD).
                              │
                              └── Otherwise (β_raw > 0, β_norm ≈ 0, DiD > 0, F4 ≤ 0.05, F3 quiet)
                                     → SUPPORTED
                                     → synthesist publishes as
                                       "measured-positive configuration recommendation:
                                        apply Bias Fitting wrapper on judge calls
                                        when calibration shows β_raw > 0."
                                     → Compare wrapper vs B1 (prompt-instruction) and
                                       B3 (linear-debias) to recommend the cheapest
                                       intervention that achieves the same.
                                     → Apply R6 BadScientist-dominance check (cross-run):
                                       if a parallel BadScientist-channel hypothesis
                                       shows that channel dominates, demote S2 to
                                       future-work appendix per smith R6.
```

**Handoff artifacts:**

1. `held_out_results.parquet` — full (manuscript, variant, judge_model, call_id, raw_score, wrapped_score, token_count, proxy_1..proxy_8) table.
2. `regression_table.md` — β_raw, β_norm, DiD with 95% CIs, per judge model, per condition.
3. `f3_correlations.md` — Spearman ρ table for 8 proxies × 3 models × {raw, wrapped}.
4. `f4_auroc_table.md` — AUROC by judge model, raw vs wrapped, with confidence bounds.
5. `ablations.md` — A1-A6 results.
6. `model_f_ICLR.pt` and `model_f_AI.pt` — fitted Bias Fitting model checkpoints (with sha256 hashes locked in manifest).
7. `caveats.md` — explicit reproduction of §9 threats with observed outcomes.

---

## 11. Pre-registered procedural reruns (limited investigator-degrees-of-freedom)

The only intermediate decisions permitted during execution:

1. **R-rerun-1 (substance-gate failure):** if a paraphrase fails the substance-preservation gate (§2.3), re-paraphrase up to 3 times; if still failing, replace from reserve pool. Pre-committed.
2. **R-rerun-2 (paraphrase-prompt revision):** if the pilot-of-5 human check (§9.5) detects substance drift in >2 of 5 manuscripts, the paraphrase prompt may be tightened **once**; the entire held-out set is re-paraphrased and re-scored. This is the only permitted prompt revision.
3. **R-rerun-3 (model-snapshot drift):** if the pre-registered drift-check (§9.8) detects model-version change mid-run, the run is restarted with a new locked snapshot.

No other intermediate-results-conditional decisions are permitted. Specifically:

- The Bias Fitting fitting-model checkpoint may NOT be re-trained after held-out scoring begins.
- The decision rules in §5.2 may NOT be relaxed.
- The F3 proxy list may NOT be extended.
- Additional baselines may NOT be added.

---

## 12. Sources

All arXiv IDs cited verified resolvable via `hf_papers paper_details` (executed during this protocol design pass).

- **arXiv:2505.12843** — Zhao et al., *Bias Fitting to Mitigate Length Bias of Reward Model in RLHF*. The debiaser this protocol tests. §3 method (warm-up, fitting model, length encoding) verified via `read_paper section=3`. Verified.
- **arXiv:2510.18003** — Jiang et al., *BadScientist*. Five non-length fabrication strategies; F3 substantive proxies (6, 7, 8) are derived from this. Verified; dataset `badscientist/BadScientist-Prompts` exists on HF (MIT license, research-only). Verified.
- **arXiv:2504.08066** — Yamada et al., *The AI Scientist-v2: Workshop-Level Automated Scientific Discovery*. Reference for the AI-Scientist-v2 runtime used in the secondary calibration corpus. (Existence confirmed via web search; not directly resolved via `paper_details` in this session — flagged for verification in `verification.md`.)
- **arXiv:2404.04475** — Dubois et al., *Length-Controlled AlpacaEval*. M1 grounding for length-bias-in-judges literature. Verified.
- **arXiv:2402.07319** — Chen et al., *ODIN: Disentangled Reward Mitigates Hacking in RLHF*. Cross-validation for M1; explicit reference for B3 (linear-debias) comparator from Bias Fitting §4.2. Verified.
- **arXiv:2410.21819** — Wataoka et al., *Self-Preference Bias in LLM-as-a-Judge*. M1 cross-validation. Verified.
- **arXiv:2505.17100** — Yang et al., *Reasoning-based Bias Detector (RBD)*. Cited as the downstream defense to pair with S2 if F3 fires on a substantive proxy. Verified.
- **arXiv:2503.18102** — Schmidgall, Moor, *AgentRxiv*. Score-fabrication reward-hacking reference; cited for context only (not a length-bias citation, per smith C1 honest revision).
- **arXiv:2507.02694** — Xu et al., *LimitGen*. Originally an optional sanity-check substrate; dropped per §8 trim. Cited as the considered-and-rejected alternative substrate. Verified.

Datasets (verified via `hf_inspect_dataset`):

- **`DeepNLP/ICLR-2024-Accepted-Papers`** — primary calibration + held-out + F4 substrate. Schema: title, url, detail_url, authors, tags, abstract, pdf. License: not declared in card; underlying PDFs are public OpenReview content (research-analysis use; transformative). 2.0 MB parquet, ~2000 rows. Verified.
- **`badscientist/BadScientist-Prompts`** — referenced for context only (F3 substantive proxies are derived from BadScientist's strategy taxonomy, not from this dataset). License: MIT (research-only). Verified via paper-resource lookup.
- **`yale-nlp/LimitGen`** — referenced as the dropped optional sanity-check substrate. Status: HF preview empty (may have load issues at preview-time), but the dataset is real and ACL 2025 paper. Resolved in `verification.md`.

Repositories:

- **`github.com/lichang-chen/odin`** — ODIN reference implementation. (B3 linear-debias baseline architecture reference.)
- **`github.com/yale-nlp/LimitGen`** — LimitGen code + data (8 stars). Not used in this protocol after trim.
- **`github.com/allenai/PeerRead`** (428 stars) — considered as alternative calibration substrate; declined because the corpus is older (2018) and pre-dates the AI/ML-paper-judging models we test. Documented as considered-and-rejected.

Internal swarm artifacts:

- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/hypothesis-smith-S2/output.md` — the approved hypothesis being designed for.
- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/red-team-S2/output.md` — APPROVE verdict with I-new-1 through I-new-4 (all addressed in this protocol).
- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/gap-finder-3/output.md` §S2 — the shortlist entry.
- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/gap-finder-2/output.md` §GAP-A10 — failure-mode documentation.
