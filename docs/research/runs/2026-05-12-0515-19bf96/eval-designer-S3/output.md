# Eval Design — Hypothesis S3 (Majority-Voting on Structured Paper Decisions)

**Worker:** eval-designer-S3 (retry 1)
**Run:** 2026-05-12-0515-19bf96
**Target hypothesis:** S3
**Smith file:** `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/hypothesis-smith-S3/output.md`
**Red-team status:** approved with revision-2 objections I-A through I-D, addressed below

---

## §1 Pre-registration statement

**Hypothesis (verbatim):**

> AI Scientist v2 wrapped with majority-vote-over-5 aggregation on 3 structured paper-decision classes lifts aggregate decision-hit-rate by ≥+6 pp over the single-draft baseline, evaluated across 1080 binary decisions total (12 papers × 30 baselines = 360 baseline-list decisions; 100 AbGen references × 6 axes = 600 ablation-axis decisions; 12 papers × 10 excerpts × 1 cluster-membership flag = 120 related-work decisions).

**Decision rules (frozen before any data is drawn):** see §10.

**No-peeking commitment.** This protocol is committed to the run directory before any candidate is sampled from AI Scientist v2. Decision thresholds, dataset partitions, the 6-axis ablation taxonomy, the leaderboard-entry list, and the CiteME excerpt list are all hash-locked at pre-flight time (per S4's hash-protocol pattern). Mid-run threshold changes are forbidden; any deviation requires writing a `protocol-deviation.md` and aborting the result as exploratory.

**Frozen artifacts at pre-flight:**

1. `frozen-baselines.json` — 12 papers × 30 leaderboard entries (5 leaderboards × top-30 from paperswithcode.com snapshot 2026-01-01)
2. `frozen-axes.json` — 6 ablation-axis taxonomy (architecture, hyperparameter, dataset-subset, optimizer, scale, regularization) with extraction lexicon per axis
3. `frozen-excerpts.json` — 12 papers × 10 CiteME-shape attribution excerpts
4. `frozen-seeds.json` — voter seeds {42, 99, 101, 137, 211}; replication seed {42}; eval seed {42}
5. `frozen-prompts.json` — single-draft prompt B0, vote-of-5 prompt B1 at T=0.7, B2 at T=0.5 (dropped per §7 trim), B3 at T=1.0

Hashes of all five files written to `manifest.yaml` at pre-flight and verified unchanged at result-reporting time.

---

## §2 Datasets and substrates

### 2.1 Baseline-list class (360 binary decisions)

- **Source:** paperswithcode.com per-leaderboard archive snapshot, frozen 2026-01-01
- **Leaderboards (5):** GLUE, SuperGLUE, ImageNet-1k (classification), COCO (detection mAP), WMT-EnDe (translation)
- **Volume:** top-30 entries per leaderboard × 5 = 150 candidate baselines per paper
- **Paper sample:** 12 papers (covering NLP, vision, multimodal, translation, RL) sampled stratified across 2024-2025 venue mix from AbGen testmini-500's paper pool (intersected with papers that have ≥1 leaderboard-applicable benchmark)
- **Per-paper task:** AI Scientist v2 produces a "claimed baselines" list. Each of the 30 top entries on the relevant leaderboard is a binary decision: included / not-included in the model's emitted baseline set. Ground truth = the accepted-version baseline set from the paper's published Methods section, extracted deterministically by regex (paperswithcode.com format is structured).
- **Decision count:** 12 × 30 = **360**
- **Licence:** paperswithcode.com data is CC-BY-SA 4.0; snapshot redistribution within the eval is permitted under research use. Recorded in `frozen-baselines.json` provenance field.

### 2.2 Ablation-axis class (600 binary decisions)

- **Source:** AbGen testmini-500 (arXiv:2507.13300), sample of N=100 references drawn at seed=42 with the dataset's provided preprocessing script (`abgen/preprocess.py`, deterministic per seed)
- **Per-reference task:** for each of the 6 pre-registered ablation axes (architecture, hyperparameter, dataset-subset, optimizer, scale, regularization), did AI Scientist v2 emit an ablation along that axis? Binary, extracted by the pilot-validated lexicon (§6.2).
- **Decision count:** 100 × 6 = **600**
- **Ground truth:** AbGen's released axis labels per reference (the dataset provides per-axis presence flags).
- **Licence:** AbGen is released under CC-BY 4.0 per its arxiv companion repo. Verified via `hf_inspect_dataset` in prior worker passes.

### 2.3 Related-work-cluster class (120 binary decisions)

- **Source:** CiteME-shape (arXiv:2407.12861) attribution excerpts, 10 per paper × 12 papers
- **Per-excerpt task:** for each cited work, AI Scientist v2 emits a cluster-membership label (which thematic cluster the citation belongs to, from a frozen 5-cluster taxonomy derived from the paper's actual related-work section structure). Binary decision: emitted cluster == ground-truth cluster?
- **Decision count:** 12 × 10 = **120**
- **Ground truth:** the paper's published related-work section's section-heading-to-citation mapping, extracted deterministically.
- **Licence:** CiteME under CC-BY 4.0 per arxiv companion.

### 2.4 Total

**1080 binary decisions** across 3 classes. Statistical power calculation (§5.2) confirms this sample size detects Δ ≥ +6 pp at α=0.001 with power ≥0.90 under reasonable correlation assumptions.

---

## §3 Baselines

| ID | Description | Purpose | In budget |
|----|-------------|---------|-----------|
| **B0** | AI Scientist v2 single-draft (no voting), T=0.7 | Prior-art baseline; the unwrapped substrate from arXiv:2504.08066 | yes |
| **B1** | Majority-vote-over-5 at T=0.7 | **Primary treatment.** Five independent candidates at seeds {42, 99, 101, 137, 211}, per-decision majority vote (3-of-5 = positive). | yes |
| **B2** | Majority-vote-over-5 at T=0.5 | Variance-floor test (red-team I-C). **Dropped from budget run** — substituted by direct Hamming-distance reporting on B1 (see §7). | NO (dropped) |
| **B3** | Majority-vote-over-5 at T=1.0 | Variance-floor pass at higher diversity. The contrast B1 vs B3 isolates the temperature-as-diversity-driver claim. | yes |
| **B-trivial** | Random binary decision at empirical class prior | Sanity baseline; the floor any non-trivial method must clear. | yes (cheap) |
| **B-ablation** | Single-draft B0 with 5× compute = 5 sequential drafts at T=0.7, last-draft-wins (no voting) | Isolates voting from raw compute spend — controls the "you just gave it 5× compute" objection | yes |

**Strongest prior-art baseline cited:** AI Scientist v2 single-draft (arXiv:2504.08066, Lu et al., "The AI Scientist v2: Workshop-Level Automated Scientific Discovery"). This is the unmodified substrate; outperforming it with the voting wrapper is the claim.

---

## §4 Metrics

### 4.1 Primary metric

**Aggregate decision-hit-rate (DHR):** proportion of the 1080 binary decisions where the system's emitted label matches the deterministic ground-truth label.

DHR computed three ways:
- DHR-aggregate (across all 1080)
- DHR-per-class (baseline-list / ablation-axis / related-work, separately)
- DHR-per-paper (per the 12 papers, for stratified bootstrap)

**Predicted effect:** B1 (vote-of-5, T=0.7) − B0 (single-draft) ≥ +6 pp on DHR-aggregate.

### 4.2 Secondary metrics (catch failure modes red-team flagged)

- **Agreement-rate (Hamming) across N=5 voters:** mean pairwise agreement across the 1080 decisions. Variance proxy. Failure mode flagged: if agreement-rate at T=0.7 is < 0.20, voting is dominated by noise and the variance-floor (I-C) has been hit even without the T=0.5 arm.
- **Per-class DHR delta:** if Δ is concentrated in one class (e.g., +15 pp on baseline-list, ~0 elsewhere), the aggregate claim is misleading and we report per-class.
- **Extraction-accuracy (pilot only):** the I-A objection. Pilot ≥0.85 or ABORT (§6.2).
- **Bootstrap CI on Δ:** 10,000 stratified bootstrap resamples by (paper × decision-class). Report 99% CI on Δ.

---

## §5 Statistical analysis plan

### 5.1 Pre-registered test

**Paired McNemar test** across the 1080 paired decisions (B0 vs B1, same decision, same ground truth).

- Null: P(B1 correct, B0 wrong) = P(B0 correct, B1 wrong)
- Alt: P(B1 correct, B0 wrong) > P(B0 correct, B1 wrong)
- α = 0.001 (Bonferroni-conservative for multiple secondary tests below)

**Dual threshold:** the hypothesis is supported only if BOTH:
1. McNemar p < 0.001 (statistical)
2. Δ_DHR ≥ +6 pp (practical)

Either alone is insufficient. The +6 pp practical floor blocks "statistically significant but trivially small" outcomes; the p<0.001 blocks "looks big but n is small per class."

### 5.2 Power calculation

Under assumption ρ = 0.6 paired correlation (B0 and B1 share substrate, so highly correlated errors expected), McNemar power at α=0.001 to detect Δ=+6 pp on n=1080 paired decisions is ~0.94. At Δ=+3 pp, power drops to ~0.55 — hence +6 is the practical-floor pre-registered cutoff; sub-6 outcomes are treated as inconclusive-or-workshop-tier per §10.

### 5.3 Multiple-comparison strategy

Primary test is one comparison: B1 vs B0 on DHR-aggregate. Secondary tests (per-class, B1 vs B3, B1 vs B-ablation) are reported but Holm-Bonferroni adjusted across the secondary family at α=0.05. False-discovery-rate control via BH at q=0.05 on the secondary family. Primary result does not depend on secondary outcomes.

### 5.4 Bootstrap variance estimate

10,000 stratified bootstrap resamples, stratified by (paper × decision-class), reporting 99% CI on Δ_DHR. This substitutes for full replication arm I-B that was trimmed for budget (§7); the bootstrap captures within-paper and within-class variance without re-running the model. Limitation: bootstrap captures sampling variance, NOT run-to-run model variance — flagged in §8 as a budget-driven limitation.

---

## §6 Ablations and validation arms

### 6.1 Replication arm (red-team I-B — partially descoped)

- **Original design:** full eval at seeds 42 and 99
- **Budget-trimmed design:** single seed (42) for the full eval; bootstrap variance estimate from §5.4 substitutes for the run-to-run variance estimate
- **Justification:** Patel §3.2 Table 5 shows up to 6-point run-to-run variance at same seed. A single seed risks reporting noise. The bootstrap captures sampling variance but not model-stochastic variance. **Limitation flagged in §8.** If the user prefers full replication, +$54 to budget.

### 6.2 Pilot study (red-team I-A — must complete BEFORE main run)

- **Design:** 20 AbGen references sampled at seed=99 (disjoint from the main-run seed=42 sample). Two independent human annotators (or two independent LLM judges with disagreement adjudication, depending on user preference) label each of 6 axes per reference. Compute extraction-lexicon accuracy against this gold standard.
- **Accept threshold:** ≥0.85 per-axis accuracy AND ≥0.85 aggregate accuracy. AbGen's reported Kappa 0.71-0.78 was Likert; binary lexicon extraction should be at least as agreeable. If pilot fails, the ablation-axis class (600 of the 1080 decisions) is excluded and the hypothesis is re-scoped to baseline-list + related-work only (480 decisions; power calculation re-run, +6 pp practical floor retained).
- **ABORT condition:** if pilot extraction accuracy <0.85 AND re-scoped sample (480) drops power below 0.80 for Δ=+6, abort the entire S3 run and surface to user.

### 6.3 Temperature sweep (red-team I-C — partially descoped)

- **Original design:** T=0.5, 0.7, 1.0
- **Budget-trimmed design:** T=0.7 (primary B1) and T=1.0 (B3); T=0.5 dropped
- **Substitute for the dropped T=0.5 arm:** the variance-floor test (does the voter ensemble actually disagree enough to gain from voting?) is conducted by reporting Hamming agreement-rate at T=0.7. If Hamming agreement < 0.20 the variance floor is hit; if Hamming agreement > 0.80 the voters are near-identical and voting cannot help (a different failure mode). The expected window for non-trivial voting gain is roughly 0.40-0.70 mean pairwise agreement.
- **Flag:** dropping T=0.5 reduces our ability to claim the temperature curve is monotone in the predicted direction. Reported as future-work in §8.

### 6.4 Stratified bootstrap (red-team I-D)

- **Design:** 10,000 bootstrap resamples stratified by (paper × decision-class), reporting Δ_DHR per stratum and aggregate. Addresses the 12-paper sample-bias objection by quantifying how much Δ depends on the specific paper sample.
- **Failure flag:** if any single paper contributes >40% of the aggregate Δ, the claim is paper-dependent and we re-report scoped to the 11 papers excluding the outlier.

### 6.5 Compute-controlled ablation (B-ablation)

- **Design:** B-ablation = 5 sequential drafts at T=0.7, take the last one (no voting). Same compute as B1 but no aggregation.
- **Purpose:** isolates voting-as-aggregation from raw-compute-spend. If B1 beats B-ablation by ≥+3 pp, the aggregation step is doing work beyond compute.

---

## §7 Cost-and-time budget

### 7.1 Per-decision call cost

- AI Scientist v2 candidate generation is structured-decision emission (not full paper draft) — per-decision call ~0.5 LLM calls (one call emits a structured list covering ~30 decisions, then per-decision extraction)
- Per-call cost ~$0.02 (Sonnet-class structured output)
- Per-1080-decision-replication candidate-cost = 1080 × 0.5 × $0.02 = $10.80 per candidate
- N=5 candidates per replication → **$54 per replication arm**

### 7.2 Original (un-trimmed) cost

- Pilot: $10 (20 examples × 6 axes × 2 annotators × $0.04)
- Main runs: (T=0.5 + T=0.7 + T=1.0) × 2 seeds × $54 = $324
- Sanity baselines (B-trivial, B-ablation): ~$25
- **Total un-trimmed: ~$359 — OVER $200 BUDGET**

### 7.3 Trimmed cost (within budget)

| Item | Cost | Status |
|------|------|--------|
| Pilot (20-example AbGen extraction validation) | $10 | required |
| B0 single-draft at T=0.7, seed=42 | $11 | required |
| B1 vote-of-5 at T=0.7, seed=42 | $54 | required |
| B3 vote-of-5 at T=1.0, seed=42 | $54 | required |
| B-ablation (5 sequential drafts, no vote) at T=0.7 | $11 | required |
| B-trivial (random at class prior) | $0 | required |
| Extraction lexicon application (deterministic, no LLM call) | $0 | required |
| **Total** | **~$140** | within $200 budget |

**Trims declared:**
- Dropped T=0.5 arm (substituted by Hamming-direct reporting on B1's voter ensemble) — flagged as I-C residual limitation
- Dropped replication seed=99 (substituted by 10k stratified bootstrap on seed=42 results) — flagged as I-B residual limitation

**Wall-clock:** ~4 hours on a single Sonnet API endpoint at 50 req/min throughput (1080 × 0.5 × 5 voters / 50 = ~54 minutes per voter run, × 3 main runs serial + parallelism = ~3-4 hours wall-clock).

**`flagged_intractable`: false**

---

## §8 Threats to validity

| ID | Threat | Mitigation in this protocol | Residual risk |
|----|--------|----------------------------|---------------|
| I-A | AbGen Kappa 0.71-0.78 was Likert, not binary lexicon | Pilot study §6.2 with ≥0.85 accept threshold and abort/rescope rule | Pilot itself uses N=20 — small. Mitigated by re-scope-on-fail rather than continue-anyway. |
| I-B | Run-to-run variance up to 6 pp at same seed (Patel §3.2 Table 5) | Stratified bootstrap §5.4 for sampling variance; **single seed only for model-stochastic variance** | Real model-stochastic variance not captured. **Budget-driven limitation.** If primary result Δ is in the 6-9 pp window, it is one model-variance sigma from null; report as inconclusive-pending-replication rather than confirmed. |
| I-C | T=0.7 scoping multiplier under-defended; predicted variance floor at T=0.5 untested | Hamming-direct reporting on B1; B3 (T=1.0) contrast | T=0.5 monotone-curve claim cannot be made from this design. Future work. |
| I-D | 12-paper sample bias | Stratified bootstrap by paper × class; outlier-paper exclusion rule (>40% Δ contribution) | 12 papers is small; the >40% rule is a heuristic. |
| I-E | Cherry-picking the 6-axis taxonomy post-hoc | 6-axis taxonomy pre-registered in `frozen-axes.json` at pre-flight with hash | None if pre-flight hash matches post-run hash. Verified in `verification.md`. |
| I-F | Ground-truth extraction is itself noisy | Paperswithcode.com data is structured (CC-BY-SA 4.0 datasheets); AbGen ships per-axis labels; CiteME ships cluster labels. All extraction is regex/lookup, not LLM-judged. | Some papers have ambiguous baseline lists; manually flag and exclude papers with ambiguous ground truth at pre-flight. |
| I-G | Voting-vs-compute confound | B-ablation (5 sequential drafts, no vote, same compute) | If B-ablation matches B1, the claim collapses — pre-registered as kill-criterion in §10. |
| I-H | Extraction-lexicon overfitting to AbGen test set | Pilot draws seed=99, main run seed=42, disjoint samples | Lexicon could still be tuned to AbGen-style references in general. Generalization is future work. |

---

## §9 Outputs the user can act on

### Decision tree (post-run)

```
Pilot extraction accuracy ≥ 0.85?
├── No → ABORT or RESCOPE to 480-decision (drop ablation-axis class)
└── Yes
    └── Primary McNemar p < 0.001 AND Δ_DHR ≥ +6 pp?
        ├── Yes
        │   └── B1 beats B-ablation by ≥ +3 pp? (voting > compute)
        │       ├── Yes → PASS-main-track (full claim supported)
        │       └── No  → PASS-workshop (compute confound; voting marginal)
        └── No
            └── McNemar p < 0.001 AND +3 ≤ Δ_DHR < +6?
                ├── Yes → PASS-workshop (effect smaller than predicted)
                └── No  → FAIL (or VARIANCE-FLOOR if Hamming < 0.20)
```

### Handoff to synthesist

The synthesist should include S3 in the "borderline-main-track" tier with the +6 pp prediction quantified. If the primary test passes at main-track threshold, S3 is the headline finding alongside whichever other hypotheses cleared. If it lands at workshop threshold, S3 becomes the supplementary contribution.

---

## §10 Pre-registered decision rules

| Outcome | McNemar p | Δ_DHR | Hamming at T=0.7 | Pilot accuracy | Voting > compute (B1 − B-ablation) | Track |
|---------|-----------|-------|------------------|----------------|-----------------------------------|-------|
| **PASS-main-track** | < 0.001 | ≥ +6 pp | 0.20 ≤ H ≤ 0.80 | ≥ 0.85 | ≥ +3 pp | Main |
| **PASS-workshop (effect-size)** | < 0.001 | +3 ≤ Δ < +6 pp | 0.20 ≤ H ≤ 0.80 | ≥ 0.85 | ≥ +1 pp | Workshop |
| **PASS-workshop (compute-confound)** | < 0.001 | ≥ +6 pp | 0.20 ≤ H ≤ 0.80 | ≥ 0.85 | < +3 pp | Workshop |
| **FAIL** | ≥ 0.001 OR Δ < +3 pp | — | — | ≥ 0.85 | — | Killed |
| **VARIANCE-FLOOR** | any | any | H < 0.20 | ≥ 0.85 | — | Killed (with note: voters disagreed below threshold) |
| **HOMOGENEITY-FLOOR** | any | any | H > 0.80 | ≥ 0.85 | — | Killed (with note: voters too similar) |
| **ABORT-pilot** | — | — | — | < 0.85 (and re-scoped power < 0.80) | — | Surface to user |
| **RESCOPE-pilot** | — | — | — | < 0.85 (re-scoped power ≥ 0.80) | — | Run on 480 decisions, retain +6 pp floor |

**Falsification criteria (per smith's hypothesis):**

- **F1 (effect-size floor):** Δ_DHR < +3 pp → falsified. Hypothesis predicts ≥+6; +3 is the practical-irrelevance floor.
- **F2 (statistical floor):** McNemar p ≥ 0.001 even at Δ ≥ +6 pp → unsupported (could be sample-size artifact).
- **F3 (variance-floor):** Hamming agreement-rate at T=0.7 < 0.20 → falsified. Voters disagree more than noise; voting cannot help.
- **F4 (compute-confound):** B-ablation matches B1 (Δ < +3 between them) → the claim of aggregation-as-mechanism is falsified, even if B1 beats B0.
- **F5 (pilot-extraction):** lexicon accuracy < 0.85 → ablation-axis class is unmeasurable; either rescope or abort.

Each falsification criterion has a corresponding experiment in §6 designed to *fail* the hypothesis if it holds. State these as pre-registered kill-criteria: any of F1–F5 firing returns S3 to FAIL or RESCOPE.

---

## Sources

- AI Scientist v2: arXiv:2504.08066 (Lu et al., 2025)
- AbGen testmini-500: arXiv:2507.13300 (CC-BY 4.0)
- CiteME-shape attribution: arXiv:2407.12861 (CC-BY 4.0)
- Patel run-to-run variance: arXiv:2604.03809 §3.2 Table 5
- Choi structured-output reliability: arXiv:2508.17536
- McNemar test: Edwards (1948) — standard paired binary test
- Holm-Bonferroni: Holm (1979) — standard FWER control
- BH FDR: Benjamini-Hochberg (1995) — standard FDR control
- Paperswithcode.com leaderboard data: CC-BY-SA 4.0
