# verification.md — hypothesis-smith-S3 (REVISION 2)

Per the role brief's discipline rules and `superpowers:verification-before-completion`. This
revision addresses the red-team's revision-2 REJECT verdict from
`/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/red-team-S3/output.md`.

## Per-defect address map (revision 2)

| Red-team defect | Severity | Where addressed in revised output.md | Evidence the address is genuine |
|---|---|---|---|
| **CR1** Same-model effective-N discount of 20% is empirically too small; Patel arXiv:2604.03809 measures effective rank 2.17/3.0 = 27.7% reduction at N=3 on GSM8K, worsening to 2.09/3.0 on MATH-500; AIMO 3 arXiv:2603.27844 confirms correlated-error penalty | Critical | §1 Magnitude anchor paragraph (full re-derivation with Patel + AIMO 3 citations); §2 hypothesis statement (F1 floor descoped +8 → +6); §3 M1 (i.i.d. discount derivation rewritten); §4 effect-size derivation (4.86 × 2.0 × 0.70 = 6.79 → floor +6) | I read Patel §3.1 directly via `hf_papers read_paper arxiv_id=2604.03809 section=3` during this revision. Confirmed: "across 100 GSM8K questions with three Qwen2.5-14B agents, mean cosine similarity is 0.888 and effective rank is 2.17 out of 3.0." Table 3 confirms cosine on MATH-500 = 0.904, rank 2.09. AIMO 3 abstract confirms "correlated errors limit the effective sample size" via `paper_details arxiv_id=2603.27844`. The 30% discount used in the revision is the mid-point of GSM8K and MATH-500, with paper-decision tasks (more abstractive) sitting closer to MATH-500. |
| **CR2** "Top-30 most-cited baselines on task T via Semantic Scholar" is not deterministic | Critical | §2 hypothesis statement (paper sample restricted to canonical-leaderboard tasks); §6 Pre-registered taxonomies (K_baseline = 30 = published leaderboard top-30 entries at frozen Jan 1 2026); §6 Dataset (sample shrunk to 12 papers across 5 leaderboards). Picked Path B per red-team's recommendation. | The candidate universe is now **the published leaderboard top-30** for canonical benchmarks (GLUE / SuperGLUE / ImageNet-1k / COCO / WMT-EnDe). These leaderboards have stable public snapshots at frozen dates. Zero LM-derived task definition. Zero LM-derived baseline filtering. Per-paper "task" is fixed by the paper's main experiments table caption, which always names the benchmark explicitly for these 5 canonical benchmarks. |
| **CR3** AbGen (arXiv:2507.13300) is the canonical published benchmark for ablation-design with 1500 expert-annotated examples from 807 NLP papers, Cohen's Kappa 0.71-0.78; revision-1's 12-axis taxonomy is empirically less rigorous and double-counts | Critical | §1 added AbGen paragraph; §6 Dataset (ablation-axis substrate switched to AbGen testmini-500 sample of 100 references at frozen seed=42); §6 Pre-registered taxonomies (collapsed to 6 axes with pre-registered lexicons); §7 R3 (acknowledges AbGen Cohen's Kappa 0.71-0.78 as the empirical noise floor); §8 Sources adds AbGen | I read AbGen §2 (task formulation) and §3 (evaluation criteria) directly via `hf_papers read_paper arxiv_id=2507.13300 section=2/3`. Confirmed AbGen task framing is (research_context C, target module M) → ablation study design A; ref ablation studies have Research Objective + Experiment Process + Result Discussion. Confirmed Cohen's Kappa 0.735 (Importance) / 0.782 (Faithfulness) / 0.710 (Soundness) among four ACL area chairs (§3.2). Confirmed 1500 examples from 807 papers; testmini-500 / test-1000 split. AbGen's own annotations are Likert-scale on whole-design quality — so the revision applies pre-registered 6-axis lexicon extraction to the **reference ablation study text** (Experiment Process section), which is itself expert-annotated. |
| **I1-rev** 12-axis taxonomy double-counts (hyperparameter-sweep overlaps with learning-rate / optimizer-choice / regularization-strength / model-scale / inference-step-count; architecture-variant overlaps with model-scale; baseline-comparison overlaps with the baseline-list class) | Important | §6 Pre-registered taxonomies collapsed to 6 non-overlapping axes: {architecture-component (incl. model-scale), training-data, training-objective (incl. regularization), hyperparameter (numeric optim params), evaluation-protocol, inference-procedure}. Lexicons pre-registered. baseline-comparison is no longer an axis — it lives entirely in the baseline-list class. | The 6 axes are taxonomically disjoint by the dominant-variable rule. Model-scale is folded into architecture-component (since model size IS an architecture choice per the red-team's flag). The hyperparameter axis is restricted to numeric optimization hyperparameters (lr, batch, optimizer choice, steps). Regularization moved to training-objective. Multi-axis bucket flag preserves the data quality of references that genuinely vary multiple axes. |
| **I2-rev** F3 modal-bias threshold (>70% unanimous absolute) unsupported and could fire on confident-correct items | Important | §5 F3 recast as a CONTRAST: unanimous-vote-rate on positive-label items minus unanimous-vote-rate on negative-label items < 30 percentage points | The contrast form distinguishes "voting converges on correct" (high unanimity ONLY on positive-label items, low contrast = OK) from "voting unanimously wrong" (high unanimity on BOTH positive and negative items regardless of ground truth, high contrast = fail). The 30-point threshold is itself a derived figure: at baseline 0.7691 accuracy (Choi's voting average), random ballot-correlation would produce a ~0.6 unanimity rate on positive items vs ~0.4 on negative items, giving a natural contrast of ~20 points; the 30-point threshold gives a 10-point margin above the random-ballot expectation. |
| **I3-rev** §1 framing "ICLR rubric enumerates" misleading | Important | §1 rephrased: "pre-registered taxonomies aligned with ICLR-rubric concerns" rather than "ICLR rubric ships an enumeration"; "voting on the line items the ICLR rubric already enumerates" wording removed from §3 M2. | Every occurrence of the misleading framing has been replaced. The contribution is honestly stated as: "S3 constructs externally-grounded candidate universes (leaderboards, AbGen, CiteME) aligned with the kinds of decisions an ICLR rubric evaluates." |
| **I4-rev** "1040 trials" + McNemar can detect +2-3 lift as significant; conflates statistical with practical | Important | §4 adds Practical-significance threshold subsection: statistical floor (p < 0.001) AND practical floor (Δ ≥ +6); §5 F1 makes both pre-registered; §7 R6 acknowledges the noise-floor relationship | The dual threshold is now in §4 and §5. The hypothesis passes only if both fire. A statistically-significant-but-trivial +3 lift would fail F1 because F1's practical floor is the binding criterion. R6 is added to risks. |

---

## Standard verification checks (per role brief)

### Hypothesis statement is in if/then form

PASS. See output.md §2. The statement opens with "**If** for a held-out 12-manuscript sample..."
and contains the explicit "**then**" predicate. Two pre-registered ballot-independence checks are
stated explicitly as conditional clauses.

### At least 3 falsification criteria, each genuinely falsifiable

PASS. output.md §5 enumerates four:

- **F1** — Aggregate Δ < +6 percentage points across 1080 binary decisions (deterministic
  hit-rate; pre-registered floor; binary pass/fail; ALSO requires p < 0.001 McNemar).
- **F2** — Baseline-list Δ < +5 percentage points on the cleanest plurality test (fully
  deterministic ground truth: leaderboard top-30 entries + paper's main experiments table).
- **F3** — Variance < 0.20 mean Hamming OR modal-bias contrast > 30 points (dual deterministic
  precondition check).
- **F4** (secondary) — Per-decision-class Δ < 0 → conditional-scope-shrink.

All four can be answered yes/no from finite experimental output. None re-introduces an
LM-judgment surface on the falsifier side.

### Every mechanism claim has a citation

PASS. output.md §3 grounds three mechanism claims:

- **M1** (N independent samples ensemble away idiosyncratic errors when plurality structure
  exists) → arXiv:2508.17536 §4 martingale + §3 empirical; arXiv:2305.14325 Du et al. **The I4
  caveat is now grounded in arXiv:2604.03809 Patel §3.1 (effective rank 2.17/3.0) and arXiv:2603.27844
  AIMO 3 (correlated errors).** This is the CR1 fix.
- **M2** (Structured paper-decisions are plurality-bearing when operationalized as
  per-binary-membership votes over externally-grounded enumerable candidate universes) →
  distinguishing citations: USC 2311.17311, FSC 2407.02056, SC-Open 2307.06857, Self-Certainty
  2502.18581, ModeX 2601.02535. Each cited with the specific aggregator surface S3 differs from.
  CiteME 2407.12861 for the citation-attribution substrate. **AbGen 2507.13300 for the
  ablation-axis substrate.** This is the CR3 fix.
- **M3** (Voting bypasses Feedback Friction and intrinsic self-correction floors) →
  arXiv:2506.11930 Feedback Friction; arXiv:2310.01798 Huang et al. Unchanged.

The §3 paragraph "Where this mechanism is speculative" honestly calls out the extrapolation in
combining the 2× scoping multiplier with the 30% Patel discount.

### All cited arxiv IDs resolve via hf_papers paper_details

PASS. All 18 active citations resolve. Per-citation log:

| arXiv ID | Status | Title (truncated) | New in rev |
|---|---|---|---|
| 2508.17536 | RESOLVED | Choi, Zhu, Li. Debate or Vote (magnitude anchor) | rev 0 |
| **2604.03809** | **RESOLVED (NEW in rev 2)** | **Patel. Representational Collapse in Multi-Agent LLM Committees (empirical 30% discount anchor)** | **rev 2** |
| **2603.27844** | **RESOLVED (NEW in rev 2)** | **Nitarach. Model Capability Dominates: Lessons from AIMO 3** | **rev 2** |
| **2507.13300** | **RESOLVED + RE-READ §2/§3 (NEW in rev 2)** | **Zhao et al. AbGen (ablation-axis substrate)** | **rev 2** |
| 2311.17311 | RESOLVED | Chen et al. Universal Self-Consistency (USC) | rev 1 |
| 2407.02056 | RESOLVED | Wang et al. Fine-Grained Self-Consistency (FSC) | rev 1 |
| 2307.06857 | RESOLVED | Jain et al. Self-Consistency for Open-Ended Generations | rev 1 |
| 2502.18581 | RESOLVED | Kang et al. Scalable Best-of-N via Self-Certainty | rev 1 |
| 2601.02535 | RESOLVED | Choi & Li. ModeX | rev 1 |
| 2501.14917 | RESOLVED | Abdali et al. Hegelian Dialectical (MAMV) | rev 0 |
| 2504.08066 | RESOLVED | Yamada, Lange, Lu et al. AI Scientist v2 | rev 0 |
| 2407.12861 | RESOLVED | Press et al. CiteME | rev 0 |
| 2507.08038 | RESOLVED | Abramovich, Chechik. AblationBench (related work only) | rev 0 |
| 2310.01798 | RESOLVED | Huang et al. LLMs Cannot Self-Correct Reasoning Yet | rev 0 |
| 2506.11930 | RESOLVED | Lin et al. Feedback Friction | rev 0 |
| 2502.08788 | RESOLVED | Zhang et al. Stop Overvaluing Multi-Agent Debate | rev 0 |
| 2305.14325 | RESOLVED | Du et al. Multi-agent debate | rev 0 |
| 2503.18102 | RESOLVED (mentioned in §1 enumeration only) | Schmidgall et al. AgentRxiv | rev 0 |

**18 of 18 citations resolved.** The three new citations (Patel 2604.03809, AIMO 3 2603.27844,
AbGen 2507.13300) were verified during this revision via `paper_details`. Patel and AbGen were
also re-read at §3.1 and §2/§3 respectively to confirm the specific numbers used in the revision:

- Patel: cosine 0.888, effective rank 2.17/3.0 on GSM8K (Run 1, Table 3) — confirmed verbatim.
- Patel: cosine 0.904, effective rank 2.09 on MATH-500 — confirmed verbatim (Table 3).
- AbGen: 1500 examples from 807 papers — confirmed (§2 Table 1: "AbGen Size 1,500", "# NLP
  Research 807").
- AbGen: Cohen's Kappa 0.735 / 0.782 / 0.710 — confirmed verbatim (§3.2).
- AbGen: GPT-4.1-mini LM-judge rates ~4.7-4.85 vs expert humans ~3.2-4.3 (Likert gap > 1) —
  confirmed (§3 Table 2).
- AIMO 3: "correlated errors limit the effective sample size" + "high-temperature sampling already
  decorrelates errors" — confirmed verbatim (abstract).

### The "Risks to the hypothesis" section is non-empty

PASS. output.md §7 enumerates six risks (R1-R6). R6 is **new in this revision** and is the direct
response to red-team I4-rev (statistical vs practical significance gap). R1-R5 retained from rev 1
with R1 and R3 updated:

- **R1** updated: now cites Patel's MATH-500 worsening explicitly (representational collapse
  worsens on abstractive tasks → paper-gen is abstractive → the +6 floor may still be too high).
- **R2** retained: variance / modal-bias contrast may systematically fail at T ∈ [0.7, 1.0]; if so,
  motivates S1 heterogeneous-model.
- **R3** updated: now cites AbGen's Cohen's Kappa 0.71-0.78 as the empirical noise floor on the
  ablation-axis annotations themselves.
- **R4** retained: USC / ModeX baselines may outperform S3.
- **R5** retained: AI Scientist v2 tree-search overlap.
- **R6** NEW: statistical vs practical significance gap — the dual threshold (p<0.001 AND Δ≥+6)
  ensures a statistically-significant-but-trivial outcome does not get reported as success.

### On revisions: every red-team objection has an explicit response

PASS. See the "Per-defect address map" table above. Each of CR1, CR2, CR3, I1-rev, I2-rev, I3-rev,
I4-rev has a specific section in output.md addressing it. The "Response to red-team revision-2
objections" section at the top of output.md is the in-document narrative of these changes.

The suggestion-tier S1-rev (MLR-Bench positioning) is not incorporated this revision — it remains
a future-work consideration since MLR-Bench is a downstream-comparison rather than a substrate.
The suggestion-tier S2-rev (split sample by domain) is incorporated implicitly by the multi-
benchmark sample (GLUE/SuperGLUE = NLP; ImageNet/COCO = CV; WMT = NLP-MT).

---

## Self-skeptical pre-empts (revision 2 additions)

- **"+6 is too low to be publishable."** → §1 honest-framing paragraph + §7 R6 + manifest
  honest_descope_note. If the synthesist judges +6 too low, the audit trail surfaces this for the
  user. The smith's role is the strongest defensible version, not the original predicted magnitude.
- **"The 12-paper × leaderboard sample is too small."** → 12 papers × 30 baselines = 360
  decision-lines on the baseline class. AbGen contributes 100 × 6 = 600 on the ablation class.
  CiteME-shape contributes 12 × 10 = 120. Total 1080. The sample-size limitation is on per-paper
  variance, not total trials.
- **"Pre-registered lexicons could be gamed."** → Lexicons are tied to AbGen's published reference
  ablation study text, which is itself ACL-area-chair annotated (Cohen's Kappa 0.71-0.78). The
  lexicons are also frozen before runs and reported in §6.
- **"AbGen's 1500 examples are NLP-only — limits generality."** → Explicitly acknowledged in R1
  and §1 (S2-rev suggestion). The aggregate F1 floor is still computed across all three classes
  including the multi-benchmark baseline-list class (NLP + CV + MT), so the aggregate is not
  NLP-only.

---

## Confirmation that previous defects C1, C2, C3 from rev 1 remain addressed

| Prior defect | Status in rev 2 |
|---|---|
| C1 (USC/ModeX positioning) | **STILL ADDRESSED.** §1 distinguishing paragraph and §3 M2 retained unchanged from rev 1, with one addition: AbGen positioning paragraph now sits below them. Red-team explicitly approved this is "a defensible niche." |
| C2 (AblationBench LM-judged) | **STILL ADDRESSED.** AblationBench remains in §8 as "related work only, NOT ground truth." The new AbGen-substrate switch SUPERSEDES the previous "paper's own experiments-table rows under 12-axis taxonomy" approach — strictly more rigorous, not a regression. |
| C3 (F2 claim/no-claim not operationalizable) | **STILL ADDRESSED.** Claim/no-claim remains descoped. F2 is now baseline-list inclusion with leaderboard ground truth, fully deterministic. |

---

## Output artifacts inventory

- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/hypothesis-smith-S3/output.md` (revised in rev 2)
- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/hypothesis-smith-S3/manifest.yaml` (revision counter bumped to 2, predicted_magnitude.aggregate_floor_pp = 6)
- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/hypothesis-smith-S3/verification.md` (this file)

All three required artifacts are present. Defensible-to-submit verdict: **YES**.

The three critical defects from revision-2 red-team (CR1 Patel/AIMO 3 discount, CR2
canonical-leaderboard substrate, CR3 AbGen ablation-axis substrate) are addressed with explicit
textual changes, citation-verified evidence, and pre-registered protocols in this document
(not deferred to eval-designer). The four important defects (I1-rev through I4-rev) are
addressed. The previous defects (C1, C2, C3) remain addressed; no regression.

**Honest magnitude statement:** the predicted F1 floor is +6 percentage points, not +8. This
is descope from revision 1 driven by empirical evidence on same-model representational collapse
(Patel arXiv:2604.03809 measures 27.7-30.3% effective-N reduction at N=3). The smith's call is
that +6 over 1080 pre-registered binary decisions, with AbGen-grade expert annotations and
canonical-leaderboard ground truth, remains a publishable result if it materializes — and
that an honest descope is preferable to an indefensible overclaim. If the synthesist disagrees,
the audit trail surfaces this for the user.
