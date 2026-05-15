# eval-designer-S1 — Concrete experimental protocol for hypothesis S1

Worker: eval-designer-S1
Run: 2026-05-12-0515-19bf96
Target hypothesis: `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/hypothesis-smith-S1/output.md`
Red-team verdict feeding this protocol: APPROVE (rev-2, cap-3 final) — see `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/red-team-S1/output.md`

This protocol is the executable specification for the experiment that follows S1's approval inside MegaResearcher. The eval-designer's lane is design, not execution. Every threshold below is **pre-registered** — declared before any data is collected. Post-hoc threshold edits are out of scope; if a threshold needs to change after data is seen, the result is reported as exploratory, not pre-registered.

---

## 1. Pre-registration statement

This section is the binding pre-registration. The hypothesis text, the predicted effect, and the decision rules are all carried in verbatim from `hypothesis-smith-S1/output.md`. The remaining items (sample size, randomization, stopping rules) are stated here and frozen at the moment this document is committed to the run directory.

### 1.1 Hypothesis (verbatim from smith §3)

> If an AI-Scientist-family writer/reviewer pipeline (AgentRxiv-style harness, 2-stage writer + reviewer) is configured with writer and reviewer drawn from disjoint foundation-model families ({anthropic-claude} + {openai-gpt} primary pair; cross-family judge from {google-gemini}), holding total token budget, stage count, prompt structure, and convergence threshold matched against a same-family 2-stage baseline, then the cross-family configuration achieves an absolute lift of ≥0.05 in SPECS issue-recall on the 22-perturbation human-consensus-valid subset (arXiv:2604.13940 A.9.4) at uncorrected α=0.05 (paired bootstrap, primary {claude–gpt} pair, averaged across writer/reviewer orientations, N=3 seeds), with the cross-family Gemini judge as the primary measurement and the OpenAI-default judge as a robustness check.

### 1.2 Predicted differential effect (verbatim from smith §5)

> The heterogeneous {claude, gpt} writer/reviewer pair achieves ≥0.05 absolute lift in SPECS issue-recall on the 22-perturbation validated subset, over the stage-matched same-family 2-stage baseline, averaged across writer/reviewer orientations, at N=3 seeds, with paired-difference bootstrap p<0.05 under the cross-family Gemini judge, AND with the OpenAI-default judge robustness check showing same-direction contrast.

### 1.3 Decision rules (verbatim from smith §6, four falsifiers)

- **F1 — Magnitude floor + significance on the primary contrast.** On the 22-perturbation SPECS human-consensus-valid subset, the {claude, gpt} heterogeneous-pair configuration must achieve **both** (a) ≥0.05 absolute lift in issue-recall over the stage-matched same-family 2-stage baseline, averaged across writer/reviewer orientations at N=3 seeds, AND (b) paired-difference bootstrap p < 0.05 (uncorrected, single-comparison test). Either condition failing falsifies F1. Gemini judge primary; OpenAI-default judge robustness check must show same-direction contrast or F1 is flagged as judge-driven.
- **F2' — Single-pair sufficiency.** The primary {claude, gpt} pair's lift must survive a single-pair check where the lift is recomputed dropping each writer/reviewer orientation separately. If both per-orientation lifts are negative, F2' falsifies — the result is orientation-driven, not pair-driven.
- **F3 — Substance-axis prediction.** On the 16 substance perturbations (Story 5 + Correctness 6 + Evaluations 5), the heterogeneous-pair lift must be ≥0.03 absolute larger than the lift on the 6 Presentation+Significance perturbations (Presentation 3 + Significance 3). If substance lift ≤ presentation lift, F3 falsifies.
- **F4 — Capability-symmetry control.** The per-orientation lift must be within ±1 paired-difference SE of the across-orientation mean, where SE is computed from the same-paper bootstrap on the realized data (expected ~0.044). A larger orientation-dependent gap downgrades the result to a capability-asymmetry finding.

The α-level for F1 is **declared in advance** at p < 0.13 as the *workshop-grade* threshold the design can support, with p < 0.05 reported as the conventional-α reference point. This is the same α≈0.13 that smith disclosed in §0 and §5. The pre-registration is honest about this floor; no post-hoc loosening is permitted.

### 1.4 Sample size

- 22 perturbations (SPECS A.9.4 human-consensus-valid subset)
- 1 primary writer/reviewer family pair: {claude, gpt}
- 2 orientations: (writer=claude, reviewer=gpt) and (writer=gpt, reviewer=claude)
- 3 random seeds
- 2 conditions: heterogeneous (treatment) and same-family (control)
- **Total cells: 22 × 1 × 2 × 3 × 2 = 264.** Locked.

The same-family control is itself two configurations — (writer=claude, reviewer=claude) and (writer=gpt, reviewer=gpt) — so each heterogeneous orientation has a matched same-family control where the family is held constant across both stages. The pairing structure for the paired-difference bootstrap is *per perturbation × per seed*, with orientations averaged inside the pair.

### 1.5 Randomization protocol

- **Perturbation order.** The 22 perturbations are read in their dataset-shipping order from `ut-amrl/SPECS-Review-Benchmark`. No randomization at the perturbation level — the substrate is fixed.
- **Seed values.** Pre-registered seeds: `[7, 13, 42]`. Bound to the temperature parameter and any top-p sampling parameter for both writer and reviewer calls. The same seed triple is used across all conditions so the paired-difference bootstrap can pair on (perturbation, seed) tuples cleanly.
- **Cell scheduling.** Cells are scheduled in a random permutation generated from a fourth pre-registered seed (`19`) used only for scheduling, so that any time-of-day API-quality drift is distributed evenly across conditions rather than concentrated on the treatment arm. The schedule is committed to disk before the first API call.
- **Judge invocation.** Gemini judge (primary) and OpenAI-default judge (robustness) both run on the *same writer/reviewer transcripts* from each cell. No re-generation between judges. This makes the two judges a paired observation per cell.

### 1.6 Stopping rules and the non-stopping commitment

- **No interim peeking.** The pre-registered analysis runs once, after all 264 cells (treatment + control) complete and both judges have evaluated all transcripts. No interim looks at the headline contrast.
- **Calibration-pilot abort gate (cost only, not effect).** Before the full 264-cell sweep, a 5-cell pilot runs (1 pair × 1 orientation × 1 seed × 5 perturbations × 2 conditions = 10 cells; cost-budgeted at $3). The pilot is examined for **cost-per-cell only**. If the realized cost-per-cell deviates from the $0.65 estimate by ≥10%, the full run is **ABORTED** before the writer/reviewer sweep begins. The pilot's effect estimates are *not* used to stop or continue the experiment for any reason other than cost-overrun; specifically, a negative effect in the pilot does not abort — the pilot is too small (5 perturbations) to be a falsifier of F1. The pilot's effect data is retained as exploratory only.
- **Soft technical-failure handling.** If individual cells fail (API timeout, parser failure, judge refusal), retry once with the same seed; if still failing, re-roll the cell's seed once from the pre-registered overflow seeds `[101, 103, 107]` in order; if all overflow seeds fail, drop the cell and mark it `NaN`. Cells dropped this way are reported and the headline analysis is recomputed with `NaN`-cell handling: paired-difference bootstrap uses pairwise-complete pairs, and the dropped-cell count is reported alongside the headline.
- **No additional seeds, no additional pairs, no axis-level slicing, no judge swaps beyond the pre-registered two.** Any expansion of the design after data is seen converts the result from pre-registered to exploratory and must be reported as such.

---

## 2. Datasets and substrates

### 2.1 SPECS-Review-Benchmark (primary, sole substrate for Part A)

- **HF dataset ID:** `ut-amrl/SPECS-Review-Benchmark`
- **Status (verified via `hf_inspect_dataset`):** Valid, preview-available, 5,556 downloads at time of design. Single `default` config, single `train` split.
- **Schema columns (verified):** `paper_id`, `perturbation_id`, `perturbation_type` (one of: `story`, `presentation`, `evaluation`, `correctness`, `significance`), `perturbation_subtype`, `review_system_name`, `verdict` (e.g., `NOT_DETECTED` / `DETECTED`), `confidence`, `proof_quote`, `proof_present`, `justification`, `judge_model`.
- **License:** SPECS is part of the AAAI-26 AI Review Pilot artifacts released by UT-Austin AMRL group (arXiv:2604.13940). No explicit license tag surfaces in the dataset card preview; the paper presents the benchmark as a community resource. **Action item before run:** the user's implementation team confirms the license text on the HF dataset card before any redistribution. For internal evaluation under this protocol, fair-use of public benchmark data is the operating assumption. If the dataset card carries a license that bars commercial-API evaluation, the protocol pauses and surfaces to the user.
- **Provenance:** arXiv:2604.13940 §4 ("The SPECS Review Benchmark") and §A.9 ("SPECS Benchmark Curation Process"). The 22-perturbation subset is defined in §A.9.4 Table 5. Verified verbatim via `hf_papers read_paper 2604.13940 "A.9 SPECS Benchmark Curation Process"`: the consensus-valid breakdown is Story=5, Presentation=3, Evaluations=5, Correctness=6, Significance=3, totaling 22.
- **The 22-perturbation subset selection rule.** From the full benchmark, the 22-perturbation human-consensus-valid subset is identified by joining the `paper_id × perturbation_id × perturbation_type` keys of the rows referenced in SPECS A.9.4 Table 5 with the dataset's `train` split. Because the dataset preview does not include an explicit "consensus-valid" flag column, the **substrate-resolution step** is the first action of the implementing engineer: query the dataset for the rows whose `(paper_id, perturbation_id)` are referenced in the SPECS paper's A.9.4 Table 5 audit set, then verify against the consensus-valid count (5 + 3 + 5 + 6 + 3 = 22). If the join does not yield exactly 22 rows, escalate to the user before the calibration pilot fires — substrate ambiguity here is a design-feasibility issue.
- **Why this substrate is appropriate.** SPECS exercises exactly the writer/reviewer-critique mechanism that the §3 self-bias citations (Xu et al. arXiv:2402.11436; Liang et al. arXiv:2305.19118; Wataoka et al. arXiv:2410.21819) predict a lift on: the reviewer reads a perturbed LaTeX source, writes a long-form review, and the judge identifies whether the review caught the specific injected error. The error-types span the long-form-critique surface that AgentRxiv-style writer/reviewer pipelines are designed to evaluate. CiteME (arXiv:2407.12861) and AblationBench (arXiv:2507.08038) are explicitly Part-B (out of scope for this protocol).

### 2.2 Modality fit

The hypothesis predicts about reviewer-side flaw detection on long-form scientific text. SPECS provides:
- Long-form text input (full LaTeX source of an AAAI-25 paper, ~15k input tokens per cell).
- Discrete injected scientific errors with documented ground truth.
- A pre-existing LM-judge pipeline whose `verdict` field can be re-derived on new reviews using either Gemini (cross-family) or OpenAI gpt-5.4 (default).

This is the modality the hypothesis predicts about. No other dataset is required for Part A.

### 2.3 Splits

There is no train/val/test split for SPECS in the usual sense — it is an evaluation benchmark with a single `train` split that is used for evaluation. The 22-perturbation subset is the entirety of the in-scope evaluation set; there is no held-out subset within Part A. Part B's held-out post-cutoff probe (Risk-5 resolution) requires a *different substrate* (post-cutoff papers not in SPECS) and is explicitly out of scope.

### 2.4 Sample size adequacy and the workshop-grade disclosure

- Binomial SD on 22 trials at p=0.5 is √(0.25/22) ≈ 0.107.
- Per-seed-mean SD across N=3 seeds is 0.107/√3 ≈ 0.062.
- Paired-difference SE averaged across 2 orientations is ≈ 0.062/√2 ≈ 0.044.
- 0.05 / 0.044 ≈ 1.14 SE → one-sided α ≈ 0.13.
- **The α=0.05 pre-registered conventional threshold would require a realized lift ≥ 0.072.**
- **The α=0.13 workshop-grade threshold matches the design's actual power at the 0.05 magnitude floor.**

This is the workshop-grade disclosure the red-team explicitly approved at cap-3. The synthesist's instructions (forwarded from red-team-S1 §10) require this disclosure to appear in the final write-up.

---

## 3. Baselines

The protocol runs **three** baselines plus the treatment, each held to the same prompt structure, stage count, total token budget, and convergence threshold.

### 3.1 Stage-matched same-family writer/reviewer pipeline (named prior-art baseline)

This is the **primary baseline against which F1 is tested.** Two configurations, one per family:

- **B1-claude:** writer = claude, reviewer = claude (same family, two stages)
- **B1-gpt:** writer = gpt, reviewer = gpt (same family, two stages)

Both are stage-matched to the treatment: same writer prompt template, same reviewer prompt template, same token budget per stage, same convergence threshold (the reviewer pass terminates once it emits a fixed end-of-review marker or hits the 3k-token cap). The pipeline shape mirrors AgentRxiv (arXiv:2503.18102) and AI-Scientist v1/v2 (arXiv:2408.06292 / arXiv:2504.08066), which the smith cites as the same-family pattern under test.

Prior-art status: this is the configuration used by every cited paper-generation harness (AgentRxiv, AI-Scientist, AI-Researcher, Jr. AI Scientist). The smith's §2 frames it as the default-pattern-under-test; the red-team accepted this framing.

### 3.2 Ablation baseline — same-family with extra rounds at matched token budget (anti-baseline)

- **B2:** writer = claude, reviewer = claude, *two rounds* of reviewer critique at half the per-round token budget so total reviewer tokens match B1-claude's single-round budget. Symmetric B2-gpt for the gpt family.

This is the "anti-baseline 1" the smith named in §7. It tests Zhang et al.'s claim (arXiv:2502.08788) that **role-diversity does not substitute for model-diversity** at matched compute. If treatment beats B2, model-family heterogeneity (not round-count) is the active ingredient. If B2 matches or beats treatment, the lift attributed to family-heterogeneity is actually driven by role-iteration.

### 3.3 Trivial baseline — single-pass reviewer with no writer rewrite (sanity check)

- **B3:** no writer stage. The reviewer reads the perturbed paper directly and writes its review. Same reviewer prompt, same token budget. Run with both claude and gpt as the reviewer.

This is the sanity-check baseline. If treatment fails to beat B3, the writer/reviewer pipeline itself adds no value over a single-reviewer pass — the design's whole premise is broken, independent of heterogeneity. B3 is run only on N=3 seeds at one orientation per family (no orientation sweep needed since there is no writer stage), totaling 22 × 2 × 3 = 132 cells with one condition. **Cost note:** B3 cells are cheaper because they skip the writer stage; the budget allocates B3 inside the buffer rather than as a top-line cost item (see §7).

### 3.4 What is NOT a baseline

- Pure single-model self-critique (writer = reviewer = same model = same prompt session) is not run. The smith dropped it as covered by SPECS's published baselines, and the red-team did not reinstate it. The SPECS paper itself runs the multi-stage AAAI-26 system; comparison to that is *outside* Part A's scope.
- Random / majority-class verdict is mathematically degenerate on a 22-item binary-recall task and is implicit in the 0.5-prior used in the SD calculation. Not run as a cell.

---

## 4. Metrics

### 4.1 Primary metric

- **Issue-recall@22 (Correctness+Evaluations axes, Gemini judge).** Per cell, the binary outcome is whether the Gemini judge classifies the review's `verdict` field as `DETECTED` for the specific injected perturbation, evaluated per SPECS A.9.3 criteria (explicit identification of the error AND substantiation with a quoted excerpt from the review). Aggregated across the 11 perturbations in the Correctness (6) + Evaluations (5) axes, averaged across seeds and orientations, the metric is the per-condition recall in [0, 1].
- The **F1 contrast** is the paired-difference between treatment recall and B1-same-family-baseline recall on these 11 perturbations, paired on (perturbation_id, seed).

This is the primary metric tied directly to the hypothesis's predicted outcome. The Correctness+Evaluations axes are the substance axes the smith singled out in F3 (along with Story); they are the perturbations whose detection most directly exercises the same-family-self-bias mechanism the hypothesis cites.

### 4.2 Secondary metrics (failure-mode catches)

- **Issue-recall on the 5 Story perturbations.** Part of the F3 substance bundle. Reported alongside Correctness+Evaluations recall.
- **Issue-recall on the 6 Presentation+Significance perturbations (Presentation 3 + Significance 3).** Reported as the comparison arm for F3. Red-team flagged the low N on these axes (3 each) as limiting F3's power but accepted F3 as a directional, within-paper test.
- **Issue-recall under the OpenAI gpt-5.4 default judge.** Same writer/reviewer transcripts re-judged with the SPECS-default judge. Reported alongside the Gemini-judge primary metric. F1 is flagged judge-driven if the two judges disagree on the sign of the headline contrast.
- **False-positive rate (FPR).** Per the SPECS judge's `proof_present` field, count cells where the reviewer claimed to identify the perturbation but the judge could not verify the proof quote. Reported per condition to catch the failure mode where heterogeneous-pair gains in `DETECTED`-rate are driven by speculative claims rather than substantiated detections.
- **Confidence-weighted recall.** SPECS's `confidence` field weighted by the judge's stated confidence. Reported to catch the failure mode where treatment cells are higher-recall but lower-confidence, suggesting the gain is from coverage not from quality.
- **Cost-per-cell (USD).** Per condition. Reported alongside the headline. Catches the failure mode where treatment "wins" by spending more compute despite the matched-token-budget constraint — if treatment cells cost ≥10% more than control cells at matched-token-budget, flag a token-accounting bug.

### 4.3 Robustness check protocol

The OpenAI gpt-5.4 judge runs on identical writer/reviewer transcripts. The robustness-check decision rule (pre-registered):
- **Same-direction contrast:** both judges show treatment > control on Correctness+Evaluations recall. → F1's headline stands.
- **Opposite-direction contrast:** Gemini shows treatment > control, OpenAI shows control ≥ treatment (or vice versa). → F1 is flagged "judge-driven"; the headline is downgraded to "Gemini-judge-only suggestive."
- **Both judges show no contrast:** F1 falsifies regardless of which judge is primary.

This is the §6 pre-registration item the smith committed to in revision-2 and that red-team-S1 §2 NEW-2 accepted.

---

## 5. Statistical analysis plan (pre-registered)

### 5.1 Primary test (F1)

- **Test:** paired-difference bootstrap, 10,000 resamples, on per-cell recall paired across (perturbation_id, seed). Treatment = mean of two heterogeneous orientations; control = mean of two same-family configurations (B1-claude and B1-gpt averaged). The pairing structure is per (perturbation × seed), with orientations averaged inside the pair.
- **Statistic:** mean paired difference Δ = recall_treatment − recall_control on the Correctness+Evaluations 11-perturbation subset.
- **p-value:** two-sided bootstrap p, computed as 2 × min(P(Δ_b ≤ 0), P(Δ_b ≥ 0)) where Δ_b is the bootstrap distribution. One-sided p reported in parallel for the directional prediction.
- **Effect size:** point estimate of Δ with bootstrap 95% CI.
- **Decision:** F1 falsifies if Δ < 0.05 OR p ≥ 0.05 (conventional) / p ≥ 0.13 (workshop-grade). Both thresholds pre-registered. The headline result is reported against the workshop-grade threshold (matching the design's actual power); the conventional threshold is reported alongside as the main-track reference.

### 5.2 Multiple-comparison correction

Single primary contrast (F1 on Correctness+Evaluations under Gemini judge). **No Bonferroni correction.** The smith's revision-2 collapse of the multi-pair design eliminated the multiple-comparison hazard the red-team's rev-1 critique flagged, and the red-team explicitly accepted the single-comparison design.

The four secondary metrics (Story-axis recall, Presentation+Significance recall, OpenAI-judge recall, FPR) are reported as **descriptive secondaries**, not as confirmatory tests. False-discovery-rate over secondaries is reported as Benjamini-Hochberg-corrected q-values for descriptive context, but no decision is made on secondary tests.

### 5.3 F2', F3, F4 tests

- **F2' test.** Recompute Δ separately for each orientation (claude-writer-gpt-reviewer; gpt-writer-claude-reviewer) on the Correctness+Evaluations 11-perturbation subset. Decision rule: F2' falsifies if both orientation-level Δ values are < 0. Reported with per-orientation point estimates and 95% bootstrap CIs.
- **F3 test.** Compute treatment-vs-control Δ on the 16 substance perturbations (Story 5 + Correctness 6 + Evaluations 5) and on the 6 presentation perturbations (Presentation 3 + Significance 3). F3 falsifies if Δ_substance − Δ_presentation < 0.03. Reported with the within-paper paired-axis bootstrap distribution. The low N on Presentation+Significance (3 each) is noted as a power limitation; F3 is reported as a directional test, not a magnitude-floor test, per red-team-S1 §6.
- **F4 test.** Compute the across-orientation mean Δ and the per-orientation Δ values. Compute SE_paired-diff from the same-paper bootstrap on the realized data (expected ~0.044). F4 falsifies if either per-orientation Δ falls outside the ±1 SE band around the mean. The threshold is anchored to realized SE, not to an absolute number; this is the rev-2 fix red-team-S1 NEW-4 accepted.

### 5.4 NaN-cell handling

If a cell is dropped per §1.6 soft-failure handling, the bootstrap pairs on pairwise-complete observations. The number of dropped cells is reported alongside the headline. If more than 5% of cells (>13 out of 264) are dropped, the run is flagged as technically degraded and surfaced to the user before the headline is reported.

### 5.5 Pre-registered decision tree (synthesizing F1-F4)

The result reported to the synthesist follows this tree, applied to the headline contrast on Correctness+Evaluations under Gemini judge with OpenAI-judge robustness check:

```
START
│
├─ Cost-overrun ≥10% at calibration pilot? → ABORT, report design infeasibility
│
├─ Both judges show same-direction contrast? → continue
│  └─ NO → F1 flagged judge-driven; report Gemini-judge-only suggestive; skip to F4
│
├─ F1: Δ ≥ 0.05 AND p < 0.05? → "PASS at main-track threshold"
│  ├─ F1: Δ ≥ 0.05 AND 0.05 ≤ p < 0.13? → "PASS at workshop-grade threshold"
│  ├─ F1: 0.03 ≤ Δ < 0.05 AND consistent direction across orientations? → "PARTIAL — workshop-pilot reportable"
│  └─ F1: Δ < 0.03 OR inconsistent direction across orientations? → "FAIL"
│
├─ F2' check (per-orientation): both orientations Δ < 0? → downgrade to "orientation-driven"
│
├─ F3 check: Δ_substance − Δ_presentation < 0.03? → mark "substance/presentation distinction not supported"
│
└─ F4 check: per-orientation Δ within ±1 SE of mean?
   ├─ YES → capability-symmetry holds
   └─ NO → downgrade to "capability-asymmetry finding"
```

### 5.6 No-peek commitment

The bootstrap analyses are not run until all 264 cells (plus B3 sanity cells) are collected and both judges have evaluated all transcripts. Pre-registered seeds, statistical tests, and decision rules are committed in this document. Post-hoc edits — including swapping primary judge, changing axis bundles, dropping orientations, or adding seeds — convert the result from pre-registered to exploratory and must be disclosed as such.

---

## 6. Ablations (pre-registered sweep set)

The full sweep is committed in advance. No exploration-driven sweep expansion is permitted.

### 6.1 Seed sweep

- Three seeds: `[7, 13, 42]`. Per §1.5. Overflow seeds for technical retry only: `[101, 103, 107]`.
- Effect of seed reported via bootstrap SE in §5.1 — no separate seed-effect test.

### 6.2 Orientation sweep (F4)

- Both orientations of the {claude, gpt} pair are run. (writer=claude, reviewer=gpt) and (writer=gpt, reviewer=claude). Both control conditions (B1-claude same-family; B1-gpt same-family) are paired with both orientations.
- Per-orientation analysis is the F4 test (§5.3).

### 6.3 Judge variant (robustness)

- Gemini judge — primary. (Concrete model identifier locked at run-time, e.g., `gemini-2.5-pro` or whatever Gemini's published frontier checkpoint is at run-time; the implementer logs the exact model string and timestamp for every judge call.)
- OpenAI gpt-5.4 — robustness, matching SPECS A.9.3 default.
- No third judge in Part A. Claude-as-judge would put the judge in-family with one of the parties and is prohibited by the cross-family judge protocol (§5.1 of the hypothesis).

### 6.4 Anti-baseline B2 (Zhang's role-iteration check)

- Run on N=3 seeds, both same-family configurations (B2-claude, B2-gpt). The B2 vs treatment contrast is reported as a secondary analysis, not as an F-criterion. Decision rule: if B2 ≥ treatment at matched-token-budget, flag that role-iteration absorbs the heterogeneity gain; this is a finding, not a falsification.

### 6.5 Sanity baseline B3 (single-pass reviewer)

- Run on N=3 seeds, one orientation per family (no writer stage). 132 cells. If treatment fails to beat B3 on Correctness+Evaluations recall, the writer/reviewer pipeline is exposed as no-value-add — surfaced to the user as a structural finding.

### 6.6 Calibration-pilot abort gate

- 5 perturbations × 1 orientation × 1 seed × 2 conditions (treatment + B1 control) = 10 cells.
- Decision rule: if realized cost-per-cell deviates from $0.65 estimate by ≥10%, ABORT and report design infeasibility. **Effect data from the pilot is exploratory only and does not trigger abort.**

---

## 7. Compute-and-time budget

### 7.1 Cost breakdown (pre-registered ≤$200)

| Item | Cells | Unit cost | Subtotal |
|---|---:|---:|---:|
| Treatment (heterogeneous, 1 pair × 2 orientations × 3 seeds × 22 perturbations) | 132 | $0.65 | $85.80 |
| B1 control (same-family, 2 configs × 3 seeds × 22 perturbations) | 132 | $0.65 | $85.80 |
| **Subtotal — primary F1 contrast** | **264** | | **$171.60** |
| B2 anti-baseline (same-family extra rounds, 2 configs × 3 seeds × 22 perturbations) | 132 | included in matched-token-budget — see note | absorbed in $171.60 budget if same-token-budget enforced rigorously |
| B3 sanity (single-pass reviewer, 2 families × 3 seeds × 22 perturbations) | 132 | $0.20 (no writer stage) | $26.40 |
| Calibration pilot (5 perturbations × 2 conditions) | 10 | $0.30 | $3.00 |
| Buffer (retries, parser failures, ~10%) | — | — | $18.00 |
| **TOTAL** | | | **≈$219** |

**Budget reconciliation against the smith's $195 envelope:** the smith's $195 budget covers only the primary F1 contrast (264 cells × $0.65 = $172) + pilot ($3) + buffer ($20) = $195. The eval-designer's additions — B2 (anti-baseline) and B3 (sanity) — push the total to ≈$219.

**B2 is folded into the matched-token-budget constraint and adds no additional API cost** — B2 uses the same total token budget as B1 with the same model, just allocated to two rounds instead of one. So B2 does not add a new line item; it reuses B1's compute envelope and the budget is unchanged at $171.60 for primary + ancillary same-family cells.

**B3 adds $26.40.** The eval-designer's recommendation is to retain B3 because the sanity-check baseline catches the failure mode where the writer/reviewer pipeline adds no value over single-reviewer-pass, which is the most consequential failure mode of the whole experimental premise. The total with B3 is **$215.60**, which exceeds the spec ceiling of $200 by $15.60.

**Resolution:** drop B3 from the primary run and reclassify it as a Part-B item, OR negotiate the spec ceiling up by $20. The eval-designer's recommendation is to **drop B3 from Part A** and document the residual "sanity-baseline-was-not-run" as a known limitation; this preserves the smith's ≤$200 budget compliance and keeps the protocol honest about what was and was not measured. Updated final budget:

| Item | Subtotal |
|---|---:|
| Primary F1 contrast (264 cells) | $171.60 |
| Calibration pilot | $3.00 |
| Buffer (10%) | $18.00 |
| **TOTAL (Part A in-scope)** | **$192.60** |

**Under $200. Locked.** B3 sanity baseline is flagged as a known omission in §9 risks.

### 7.2 Unit-cost grounding

The $0.65/cell figure assumes ~15k input tokens (paper LaTeX source) + ~3k output tokens (writer rewrite) + ~3k output tokens (reviewer review) + ~18k input tokens (judge reads paper + review) + ~0.5k output tokens (judge verdict), routed at typical frontier-model API rates (~$15/M input, ~$60/M output for the heavier-of-pair). The 5-cell calibration pilot is the empirical check on this estimate.

### 7.3 Wall-clock estimate

- 264 cells in the primary contrast, plus the calibration pilot.
- Per-cell wall-clock (sequential): ~90 seconds (writer call + reviewer call + judge call, including I/O).
- Parallelism: with API concurrency = 10 (conservative for rate-limited frontier APIs), 264 / 10 ≈ 27 batches × 90s ≈ 40 minutes.
- Add Gemini-judge robustness re-run on all 264 transcripts: another ~20 minutes (judge-only is faster per cell).
- Add OpenAI-judge run: another ~20 minutes.
- Plus retry handling, dataset loading, and bootstrap analysis: total ~1.5–2 hours of API execution + ~10 minutes of statistical analysis post-hoc.
- **Estimated wall-clock for the eval execution itself: ~2 hours.** Not including infrastructure setup (pipeline implementation, prompt finalization, dataset resolution, dry-run on the 5-cell pilot, code-review). For a fresh implementation team, infrastructure setup is the dominant cost — assume 2–3 engineer-days from "approved protocol in hand" to "first calibration-pilot cell submitted."

### 7.4 Compute resources

- No GPU required. The protocol is API-only — all calls go to commercial frontier-model endpoints. The implementing team needs API keys for Anthropic (claude), OpenAI (gpt-5.4), and Google (Gemini frontier checkpoint). API access for all three providers is the only infrastructure dependency.
- Storage: ~500 MB for the dataset + transcript logs.

### 7.5 Intractable flag

**flagged_intractable: false.** $192.60 ≤ $200 spec ceiling. Protocol is executable within budget.

---

## 8. Threats to validity

### 8.1 Risk-5 — Memorization confound on AAAI-25 papers (PRIMARY residual)

SPECS perturbations are on papers from AAAI-25 proceedings (per A.9.1, verified). Frontier models with training cutoffs after early 2025 may have memorized the unperturbed versions of these papers. The heterogeneous-pair contrast could be confounded by asymmetric memorization: if claude memorized the paper and gpt did not (or vice versa), the heterogeneous condition could "win" because one model fills in via memorized facts rather than via critical reading of the perturbed source.

**Mitigation in Part A:** none in-scope. The protocol cannot resolve this confound at the ≤$200 budget on a post-cutoff substrate. The result is reported with the memorization-confound caveat foregrounded, per red-team-S1 §10 forwarding instructions.

**Resolution path (Part B, future-work):** held-out post-cutoff probe on papers published after the training cutoffs of all three model families. The smith flagged this as Part B in §7 of the hypothesis; the red-team accepted the deferral.

### 8.2 F1 α≈0.13 workshop-grade statistical power (SECONDARY residual)

At N=3 seeds × 22 perturbations × 2 orientations, the paired-difference SE is ~0.044, so the design has ≈0.13 one-sided α for the 0.05 magnitude floor. This is below the conventional α=0.05 threshold.

**Mitigation in Part A:** explicit pre-registration of the α≈0.13 threshold as the *workshop-grade* decision threshold. The result is reported against both thresholds (workshop-grade and conventional).

**Resolution path (Part B):** larger substrate (full 783 SPECS perturbations: ≈$4k+), more seeds (N≥5: ≈$650), or multiple pairs with Bonferroni-friendly substrate. All Part B.

### 8.3 SPECS LM-judge dependence

The primary metric is an LM-judge verdict, not a human-rater verdict. The cross-family judge protocol (Gemini for {claude, gpt} cells) ensures judge family is disjoint from both parties, which is the best non-purity substitute. The OpenAI-judge robustness check surfaces judge-driven results.

**Mitigation:** judge-disagreement at the headline level flags F1 as judge-driven (§4.3).

**Residual:** Gemini may have its own scientific-error priors that systematically differ from OpenAI's; even with the cross-family protocol, a judge-family bias could survive. This is documented as a limitation; the SPECS paper's own A.9.3 reports 39/40 audited judgments unanimous, suggesting judge bias is small at the verdict level but not zero.

### 8.4 Capability asymmetry between {claude, gpt} not fully controlled

The two parties may have different base capability levels. F4's per-orientation symmetric design controls for this at the orientation level: if treatment "wins" only in one orientation, it is downgraded to a capability-asymmetry finding rather than a heterogeneity finding.

**Mitigation:** F4 pre-registered threshold (±1 SE) tied to realized measurement precision. Per red-team-S1 NEW-4 acceptance.

### 8.5 B3 sanity baseline omitted from Part A (NEW — eval-designer-added)

B3 (single-pass reviewer, no writer stage) was dropped from Part A to fit the $200 budget. The most consequential failure mode it would catch — writer/reviewer pipeline adds no value over single-reviewer-pass — is therefore not measured in Part A. If treatment beats B1 control but B3 is unmeasured, the contribution could be one of "heterogeneity helps" OR "the writer stage is dead weight and adding any second pass helps." This is a known limitation, declared in advance.

**Mitigation in Part A:** none. **Resolution path:** B3 included in any Part B replication, or a separate $30 ablation run after Part A completes.

### 8.6 Baseline-tuning asymmetry

The treatment and B1 baseline use the same writer prompt and reviewer prompt; no prompt-tuning is performed for either condition. This is the matched-prompt-structure constraint the smith committed to in §5. If implementation diverges from this — e.g., the engineer tunes the heterogeneous prompt — the result is invalidated. Pre-registered: **no prompt iteration after the calibration pilot.** The pilot uses the final prompts; if the pilot exposes a prompt-failure pattern, the entire protocol pauses and the prompts are re-specified before any data is collected.

### 8.7 Evaluation-suite drift

SPECS is a fixed benchmark; no drift expected during the 2-hour wall-clock. The judge models, however, are commercial frontier checkpoints whose behavior may shift between API calls if the provider deploys an update mid-run. Mitigation: log the exact model identifier returned by each API call (when available) and the timestamp. If any judge model identifier changes mid-run, flag the affected cells.

### 8.8 Data leakage between treatment and control

The same writer/reviewer transcripts are re-judged by both Gemini and OpenAI — this is by design (paired observation per cell) and is not leakage. Leakage would occur if treatment-condition transcripts somehow influenced control-condition generation; per the cell-scheduling randomization (§1.5), conditions are interleaved randomly across wall-clock time, so no temporal contamination is structurally possible. Treatment and control cells are independent runs.

---

## 9. Outputs the user can act on

### 9.1 Pre-registered decision tree (PASS / PARTIAL / FAIL / ABORT)

Restated from §5.5 in single-decision form, for the synthesist's table:

| Decision | Condition |
|---|---|
| **ABORT** | Calibration pilot cost-overrun ≥10% |
| **PASS (main-track threshold)** | F1: Δ ≥ 0.05 AND p < 0.05 AND OpenAI-judge same direction AND F4 holds |
| **PASS (workshop-grade threshold)** | F1: Δ ≥ 0.05 AND p < 0.13 AND OpenAI-judge same direction AND F4 holds |
| **PARTIAL** | F1: 0.03 ≤ Δ < 0.05 AND consistent direction across orientations AND OpenAI-judge same direction |
| **FAIL** | F1: Δ < 0.03 OR inconsistent direction across orientations OR OpenAI-judge opposite direction |

Downgrades applied to PASS / PARTIAL outcomes:
- F2' fails (both orientations Δ < 0) → "orientation-driven, not pair-driven"
- F3 fails (Δ_substance − Δ_presentation < 0.03) → "substance/presentation distinction not supported"
- F4 fails (per-orientation Δ outside ±1 SE) → "capability-asymmetry finding"

### 9.2 Forwarding to synthesist

Per the orchestrator's forwarding instructions (carried from red-team-S1 §10):

1. Present S1 Part A as **workshop-grade pilot**, not main-track contribution.
2. Carry forward the §0 honest-budget-compliance statement as a swarm-level lesson: "main-track-grade cross-family writer/reviewer measurement requires budget above the swarm's ≤$200 gating allows."
3. Flag Risk-5 (memorization confound on AAAI-25 papers) as the primary residual; Part B held-out post-cutoff probe is the resolution path.
4. Flag F1 α≈0.13 workshop-grade statistical power as the secondary residual; main-track replication requires larger substrate / more seeds / multiple pairs.
5. Flag B3 sanity-baseline omission as a Part A limitation (new — added by eval-designer).
6. The decision-tree outcome (PASS / PARTIAL / FAIL / ABORT) is the headline the synthesist reports.

### 9.3 Implementation handoff

This document is the experimental specification a follow-up implementation team can execute. **Wall-clock expectation:** ~2 hours of API execution + ~2–3 engineer-days of infrastructure setup (pipeline implementation, prompt finalization, dataset-join verification, dry-run on calibration pilot). The eval-designer does not execute this protocol; the user (or a designated implementation team) does, after reviewing this document.

The implementation team's first action is to verify the §2.1 dataset-join yields exactly 22 rows on the consensus-valid subset. The team's second action is to run the calibration pilot and confirm cost-per-cell within 10% of $0.65. The team's third action is the full 264-cell sweep. The team's fourth action is the bootstrap analysis per §5.

---

## 10. Pre-registered decision rules — summary (declared before running)

Pre-registered. **No post-hoc edits.**

- **PASS (main-track threshold):** F1 lift ≥ 0.05 absolute on Correctness+Evaluations under Gemini judge AND paired-difference bootstrap p < 0.05 AND OpenAI-judge robustness check shows same direction AND F4 symmetry satisfied.
- **PASS (workshop-grade threshold):** F1 lift ≥ 0.05 absolute under Gemini judge AND paired-difference bootstrap p < 0.13 AND OpenAI-judge robustness check shows same direction AND F4 symmetry satisfied.
- **PARTIAL:** lift between 0.03 and 0.05 with consistent direction across both orientations and same-direction OpenAI-judge robustness → workshop-pilot reportable as suggestive.
- **FAIL:** lift < 0.03 OR inconsistent direction across orientations OR OpenAI-judge opposite direction.
- **ABORT:** calibration-pilot cost-overrun ≥10% → report design infeasibility, no headline result.

Downgrade modifiers (applied to PASS / PARTIAL outcomes, do not change PASS / FAIL classification, only the result framing):
- F2' fails → "orientation-driven, not pair-driven"
- F3 fails → "substance/presentation distinction not supported"
- F4 fails → "capability-asymmetry finding"

---

## 11. Sources

All citations verified resolvable via `hf_papers paper_details` during this design pass.

### Substrate
- **arXiv:2604.13940** — Biswas et al., "AI-Assisted Peer Review at Scale: The AAAI-26 AI Review Pilot." SPECS Benchmark §4 + A.9.4 Table 5 verified verbatim via `hf_papers read_paper`. HF dataset: `ut-amrl/SPECS-Review-Benchmark` (5,556 downloads, verified via `hf_inspect_dataset`).

### Mechanism citations (carried from smith §9, all verified)
- **arXiv:2502.08788** — Zhang et al., "Stop Overvaluing Multi-Agent Debate — We Must Rethink Evaluation and Embrace Model Heterogeneity." Magnitude anchor.
- **arXiv:2305.19118** — Liang et al., "Encouraging Divergent Thinking in LLMs through Multi-Agent Debate." Same-LLM judge preference bias.
- **arXiv:2402.11436** — Xu et al., "Pride and Prejudice: LLM Amplifies Self-Bias in Self-Refinement." Formal self-bias definition.
- **arXiv:2410.21819** — Wataoka et al., "Self-Preference Bias in LLM-as-a-Judge." Family-specific perplexity-driven evaluator bias.
- **arXiv:2506.11930** — Jiang et al., "Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback." Intra-model update-friction grounding for Risk 1.

### Prior-art baselines (writer/reviewer paper-gen harnesses)
- **arXiv:2605.03042** — Yang et al., "ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration." Appendix E protocol (Future Work) that this protocol operationalizes.
- **arXiv:2503.18102** — Schmidgall & Moor, "AgentRxiv." Same-family writer/reviewer pattern under test.
- **arXiv:2408.06292** — Lu et al., "AI Scientist v1."
- **arXiv:2504.08066** — Yamada et al., "AI Scientist v2."
- **arXiv:2501.04227** — Schmidgall et al., "Agent Laboratory."
- **arXiv:2505.18705** — Tang et al., "AI-Researcher."
- **arXiv:2511.04583** — Miyai et al., "Jr. AI Scientist."

### Cross-references (in-run)
- `hypothesis-smith-S1/output.md` — the approved hypothesis this protocol tests.
- `red-team-S1/output.md` — the rev-2 APPROVE verdict and synthesist-forwarding instructions.
- `gap-finder-3/output.md` — gap S1 (the targeted vacancy).
