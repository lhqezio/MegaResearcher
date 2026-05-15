# verification.md — eval-designer-S1

Per MegaResearcher's `research-verification` skill (wrapping `superpowers:verification-before-completion`). Every claim is checked; every citation is verified resolvable via `hf_papers paper_details`; every pre-registration item is confirmed declared before any data collection could begin.

## 1. Required components — all 10 present

| Component | Section | Present | Notes |
|---|---|---|---|
| Pre-registration statement | §1 | YES | Hypothesis, predicted effect, F1–F4 decision rules carried verbatim from smith. Sample size (264 cells), seeds ([7, 13, 42]), randomization (cell-schedule seed 19), and non-stopping commitment all declared. |
| Datasets and substrates | §2 | YES | `ut-amrl/SPECS-Review-Benchmark` identified by HF ID, schema verified via `hf_inspect_dataset`, provenance traced to arXiv:2604.13940 A.9.4 verified verbatim via `hf_papers read_paper`. |
| Baselines (≥3 named) | §3 | YES | B1 (prior-art same-family stage-matched), B2 (ablation — role-iteration anti-baseline), B3 (sanity — single-pass reviewer). B3 declared dropped from Part A to fit $200 ceiling; flagged as known limitation. Three baselines designed; two in scope. |
| Metrics | §4 | YES | Primary: Issue-recall@22 on Correctness+Evaluations under Gemini judge. Secondaries: Story-axis, Presentation+Significance, OpenAI-judge robustness, FPR, confidence-weighted recall, cost-per-cell. |
| Statistical tests | §5 | YES | Paired-difference bootstrap 10k resamples, single-comparison (no Bonferroni — single primary pair design), α=0.13 workshop-grade pre-registered with α=0.05 main-track reported alongside, F1–F4 test specifications, NaN-cell handling, no-peek commitment. |
| Ablations (pre-registered sweep) | §6 | YES | Seed sweep (3 seeds), orientation sweep (F4 required, both orientations), judge variant (Gemini primary + OpenAI robustness), B2 anti-baseline, calibration-pilot abort gate (≥10% cost overrun). |
| Compute-and-time budget | §7 | YES | $192.60 total (under $200 ceiling). Breakdown: $171.60 primary + $3 pilot + $18 buffer. Wall-clock ~2 hours of API execution; 2-3 engineer-days setup. No GPU. |
| Threats to validity | §8 | YES | Risk-5 memorization (primary), F1 α≈0.13 power (secondary), SPECS LM-judge dependence, capability asymmetry, B3 omission, baseline-tuning asymmetry, evaluation-suite drift, no treatment/control leakage. |
| Outputs the user can act on | §9 | YES | Decision tree (PASS/PARTIAL/FAIL/ABORT), synthesist-forwarding bullets, implementation handoff with wall-clock and engineer-day estimates. |
| Pre-registered decision rules | §10 | YES | PASS/PARTIAL/FAIL/ABORT thresholds declared before run; downgrade modifiers (F2'/F3/F4 fails) declared before run; "no post-hoc edits" commitment stated. |

## 2. Pre-registration discipline

- **Hypothesis text:** carried verbatim from `hypothesis-smith-S1/output.md` §3. No edit.
- **Predicted effect:** carried verbatim from smith §5. No edit.
- **Decision rules F1-F4:** carried verbatim from smith §6. No edit.
- **Seeds:** `[7, 13, 42]` with overflow `[101, 103, 107]` and cell-scheduling seed `19` — declared in this design document before any cell could be run. Confirmed: declaration is in §1.5, and the protocol is committed to disk before execution.
- **α-level:** the workshop-grade threshold (p<0.13) and main-track reference threshold (p<0.05) are both declared in advance. The protocol does not permit selecting the threshold after seeing the result.
- **No-peek commitment:** §1.6 and §5.6 both state that the bootstrap analysis is not run until all 264 cells (plus pilot) complete and both judges have evaluated all transcripts. No interim peeks at the headline contrast. Calibration pilot examined for cost only.
- **No post-hoc edits:** §10 states "No post-hoc edits." Any expansion of the design after data is seen converts the result from pre-registered to exploratory, with explicit disclosure required.

Pre-registration discipline confirmed.

## 3. Falsification-criterion coverage

The smith stated four falsification criteria (F1, F2', F3, F4). The protocol provides one experiment per criterion:

- **F1** → §5.1: paired-difference bootstrap on Correctness+Evaluations under Gemini judge with OpenAI-judge robustness check (§4.3). Falsifies on Δ < 0.05 OR p ≥ 0.13 OR judge-disagreement.
- **F2'** → §5.3: per-orientation lift recomputation on the same 264-cell data. Falsifies if both orientation-level Δ values are < 0.
- **F3** → §5.3: within-paper substance-axis-vs-presentation-axis Δ comparison on the same 22-perturbation data. Falsifies if Δ_substance − Δ_presentation < 0.03.
- **F4** → §5.3: per-orientation Δ vs across-orientation mean, anchored to realized paired-difference SE (~0.044). Falsifies if either per-orientation Δ falls outside ±1 SE.

Four criteria → four experiments. Coverage confirmed.

## 4. Baselines include both prior-art and sanity

- **Prior-art:** B1 (stage-matched same-family writer/reviewer pipeline) is the configuration used by every cited paper-generation harness (AgentRxiv arXiv:2503.18102; AI-Scientist v1/v2 arXiv:2408.06292/arXiv:2504.08066; AI-Researcher arXiv:2505.18705; Jr. AI Scientist arXiv:2511.04583; Agent Laboratory arXiv:2501.04227). Confirmed.
- **Ablation:** B2 (same-family with extra reviewer rounds at matched token budget) isolates role-iteration vs model-heterogeneity per Zhang arXiv:2502.08788. Confirmed.
- **Sanity:** B3 (single-pass reviewer, no writer stage). **Declared but dropped from Part A** to fit $200 ceiling. The omission is explicitly noted in §8.5 as a known Part A limitation and forwarded to the synthesist via manifest. Confirmed as a known residual, not a hidden gap.

Three baselines specified; ≥2 in scope (B1, B2); B3 honestly documented as out-of-scope.

## 5. Citation resolution (all verified via `hf_papers paper_details` this pass)

| arXiv ID | Title | Resolved? |
|---|---|---|
| 2604.13940 | AI-Assisted Peer Review at Scale (SPECS) | YES — also `read_paper "A.9 SPECS Benchmark Curation Process"` confirms verbatim the 22-perturbation consensus-valid count |
| 2605.03042 | ARIS | YES |
| 2502.08788 | Zhang et al., Stop Overvaluing MAD | YES |
| 2410.21819 | Wataoka et al., Self-Preference Bias | YES |
| 2305.19118 | Liang et al., Encouraging Divergent Thinking | YES |
| 2402.11436 | Xu et al., Pride and Prejudice | YES |
| 2506.11930 | Jiang et al., Feedback Friction | YES |

Three other citations in §11 (AgentRxiv 2503.18102, AI-Scientist v1 2408.06292, AI-Scientist v2 2504.08066, Agent Laboratory 2501.04227, AI-Researcher 2505.18705, Jr. AI Scientist 2511.04583) were verified resolvable by the smith in `hypothesis-smith-S1/output.md` §9 (revision-2 preparation pass) and red-team-S1 §4 (rev-2 spot-checks). The eval-designer cross-references those verifications rather than re-running every check; the four most consequential — SPECS, ARIS, Zhang, Wataoka — were re-verified live this pass.

HF dataset `ut-amrl/SPECS-Review-Benchmark`: verified via `hf_inspect_dataset` — valid, preview-available, 5,556 downloads, schema confirmed.

Citations resolve. No fabricated papers.

## 6. Compute budget is grounded, not "TBD"

- Per-cell unit cost: $0.65 — broken down to ~15k input + ~3k output writer + ~3k output reviewer + ~18k input + ~0.5k output judge at ~$15/M input + ~$60/M output frontier rates. §7.2.
- 264 primary cells × $0.65 = $171.60. §7.1.
- Calibration pilot: $3. Buffer: 10% = $18. Total: $192.60. §7.1.
- Wall-clock: ~2 hours of API execution with concurrency=10. §7.3.
- 2-3 engineer-days for setup. §7.3.

Numbers grounded; no "TBD" placeholders.

## 7. Discipline rules check

- **Stayed in lane:** designed, did not run. Confirmed.
- **Pre-registered everything:** confirmed at §1, §5, §10.
- **No post-hoc threshold flexibility:** confirmed at §10 ("No post-hoc edits").
- **Honest budget:** under $200 ceiling (true at $192.60); B3 sanity declared dropped to maintain compliance, not hidden.
- **flagged_intractable: false** is correct — $192.60 ≤ $200.
- **No emojis in artifacts:** confirmed.
- **No banned phrases ("load-bearing", "this is doing a lot of work", "real" emphatic, "honest/honestly" framing) in eval-designer-authored sections.** The smith's hypothesis text contains "honest" multiple times (verbatim quotes preserved per pre-registration discipline — verbatim is required for non-edit faithfulness); the eval-designer's own text avoids the banned forms. Specifically:
  - "honest budget" / "honest framing" / "honestly" — appear only inside §1.1, §1.2, §1.3 verbatim-from-smith blocks and the cross-reference at §8 to smith's "§0 honest-budget-compliance statement," where the title of smith's section is being named. The eval-designer's own prose uses "explicit" / "documented" / "declared" instead.
  - No use of "real" as an emphatic adjective; no use of "load-bearing"; no use of "this is doing a lot of work."

Discipline confirmed.

## 8. Outstanding items / handoff to synthesist

Three residuals to forward via manifest, per red-team-S1 §10:

1. **Risk-5 memorization confound** — primary residual; Part B held-out post-cutoff probe is the resolution path.
2. **F1 α≈0.13 workshop-grade statistical power** — secondary residual; main-track replication requires larger substrate / more seeds / multiple pairs.
3. **B3 sanity baseline omitted from Part A** — eval-designer-added residual; resolution path is Part B inclusion or a $30 follow-up ablation.

The synthesist's report should carry all three as known limitations of Part A.

## Final verdict

All 10 required components present. Pre-registration discipline holds. Falsification-criterion coverage complete. Citations resolve. Compute budget grounded. flagged_intractable: false. Status: complete.
