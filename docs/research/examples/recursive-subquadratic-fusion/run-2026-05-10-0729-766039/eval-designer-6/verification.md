# verification.md — eval-designer-6 self-check

**For:** `output.md` and `manifest.yaml` in `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/eval-designer-6/`
**Hypothesis:** H6-SUB rev-1 (`hypothesis-smith-6/revision-1/output.md`)
**Verification scheme:** superpowers:verification-before-completion + project-specific eval-designer checks per spec.

---

## Required-section presence

| Required section (per task contract) | Present in output.md | Status |
|---|---|---|
| Hypothesis being tested (restated, including falsification criteria) | §0 | PASS |
| Datasets — primary + secondary, with HF ID, license, splits, size, motivation | §1.1 (CRUXEval-X primary), §1.2 (CRUXEval-O secondary), §1.5 (pretraining corpus); §1.3 specifies AST extraction script (red-team I5 fix); §1.4 dry-run d-distribution | PASS — primary + secondary + pretraining; AST extraction committed |
| Backbones (≥3 baselines + treatment), exact configs, params, training tokens, source repos | §2.1 B1 TTT-Linear treatment, §2.2 B2 LaCT replication, §2.3-§2.5 B3-B5b controls, §2.7 B6-B8 trivial sanity + prior-art anchor; §2.6 TRM construction; §2.8 summary table | PASS — 7 baselines (B3, B4, B5a, B5b, B6, B7, B8) including (a) prior-art anchor B8, (b) ablation B5a/B5b, (c) trivial sanity B6 — exceeds ≥3 requirement |
| Metrics — primary + secondary, formal definitions | §3.1 M_primary (formal r definition), §3.2-§3.5 four secondary metrics (M_sec1-M_sec4), §3.6-§3.8 three diagnostics (M_diag1-M_diag3), §3.9 trivial-floor sanity | PASS — 1 primary + 4 secondary + 4 diagnostic = 9 metrics with formal definitions |
| Statistical analysis plan (pre-registered): test, alpha, effect size, power, std, sample size | §5.1 hierarchical problem-clustered bootstrap (red-team I1 fix), §5.2 power analysis at 5/10/15 seeds, §5.3 Bonferroni α=0.01, §5.4 escalation decision tree, §5.5 inconclusive handling, §5.6 pre-registration record | PASS — pre-registration explicit and committed before any run; staged escalation rule frozen |
| Falsification experiments — at least one per criterion | §4 table (F1, F2-η0, F2-FLA, F3, F4, F5 — 6 experiments mapped to specific arms × cohorts × thresholds); F2 split into F2-η0 and F2-FLA per red-team I2 | PASS — 6 falsification experiments, each with kill threshold and direction |
| Ablations | §6 (A1 reset-vs-persist primary, A2 gradient-flow per red-team I4, A3 η-sweep, A4 d-bucket post-hoc, A5 persist-with-detach) | PASS — 5 pre-registered ablations |
| Compute budget — GPU-hours, dataset prep, wall-clock, by phase | §8.1 Phase A, §8.2 ablations, §8.3 F3 inference, §8.4 escalation phases, §8.5 totals (~5,096 hr Phase A + ablations + F3), §8.6 reduction options | PASS — full FLOP arithmetic; flagged_intractable: true for full Phase A |
| Cheaper falsification path | §7.1 Cheap-Path-A (~150 GPU-hr open-weights + light fine-tune), §7.2 Cheap-Path-B (~5 GPU-hr η-sweep) | PASS — 150 GPU-hr cheap path with kill-switch r ≥ 0.7 |
| Risks to the experiment | §9.1-§9.8 (8 risks: data leakage, baseline-tuning asymmetry, TRM-at-125M failure, bimodality, cross-language correlation, η-recipe mismatch, eval-suite drift, no-CoT-helpful-sub-split) with explicit mitigations | PASS — 8 distinct experiment-level risks, each with mitigation |
| Sources (arxiv IDs + dataset/repo IDs) | §10.1 hypothesis-cited (9 papers), §10.2 added (arXiv:2604.21106 per red-team S1), §10.3 datasets (4 with HF IDs + licenses), §10.4 reference impls (4 GitHub repos), §10.5 to-be-committed eval scripts | PASS |

**All required sections present.** Output complies with task contract.

---

## Spec-specific checks (per project research plan and task contract)

| Check | Status | Evidence |
|---|---|---|
| Every dataset is a real HF dataset with verified license | PASS | All 4 datasets verified Status=Valid via `hf_inspect_dataset`: `xhwl/cruxeval-x` (MIT), `cruxeval-org/cruxeval` (MIT), `HuggingFaceFW/fineweb-edu` (ODC-BY-1.0), `codeparrot/github-code-clean` (Apache-2.0/MIT/BSD permissive subset). |
| Statistical analysis plan is pre-registered, not post-hoc | PASS | §5.4 pre-registered staged escalation rule (5 → 10 → 15 seeds). §5.5 pre-registered inconclusive-result handling. §5.6 pre-registration record file at `preregistration.json` to be committed before Phase A starts; SHA-1 hashes of `d_extract.py` and `eval_h6sub_v1.py` included. The phrase "pre-registered" is used 11+ times throughout output.md, each tied to a specific committed decision. |
| At least one falsification experiment per criterion in the hypothesis | PASS | Hypothesis §5 lists F1-F5; this design provides one experiment per F-criterion in §4 table. F2 is *split* into F2-η0 and F2-FLA per red-team I2 fix (the η=0 case collapses to position-wise MLP, not linear attention; FLA-stationary is the true static-linear-attention distinguisher). All 5 hypothesis F-criteria mapped to experiments with kill thresholds. |
| Baselines include both prior-art and a sanity baseline | PASS | Prior-art anchor B8 (Pythia-160M ≈ 24%, GPT-Neo-125M ≈ 27% from CRUXEval Table 2). Sanity baseline B6 (random-token, analytic floor ~1%). Memorize-and-retrieve floor B7 (NN-retrieval). Plus ablation baselines B5a (η=0) and B5b (FLA-stationary). |
| Compute budget estimate is grounded (not "TBD") | PASS | §8.1 full FLOP arithmetic: 125M × 50B × 6 FLOPs/param/token = 3.75e19 FLOPs/run; 700 TFLOPs effective on H100; 15 hr per K=1 run; TRM-K multipliers explicit; Phase A 120 runs × ~27 hr avg = ~3,240 GPU-hr. Total = ~5,096 hr (Phase A + ablations + F3). Cheap-Path-A = ~150 hr. No "TBD" entries. |
| Design for falsification, not confirmation | PASS | §0 restates falsification thresholds in advance. §4 maps each F-criterion to a kill threshold direction (e.g., F1 falsified if 95% lower-CI on r ≥ 0.8). §5.4 pre-registered escalation tree includes "REPORT inconclusive" — explicitly does NOT allow post-hoc threshold relaxation. §5.5 frozen pre-registration. |
| Pre-register the decision rule | PASS | §5.4 frozen decision tree before Phase A; §5.6 pre-registration record file with SHA-1 hashes ensures the AST extraction script and eval script cannot drift after pre-registration. |
| Honest compute estimates | PASS | §8.5 summary table shows Phase A + ablations + F3 = 5,096 GPU-hr; full Phase A + ablations + F3 + Phase B + Phase C = 8,336 GPU-hr. Both **explicitly flagged over the 2,000-hr fence**. `flagged_intractable: true` set in manifest. §8.6 reduction options enumerated with trade-offs (R4 drops FLA-stationary which would un-fix red-team I2). Recommendation to user: explicit approval needed OR accept cheap-path-only kill-test with Ouro caveat. |
| Stay in lane — design only, no running, no synthesis | PASS | This document is purely design. No code is written and run; the AST extraction script in §1.3 is *committed* but not executed (the dry-run is to be performed before Phase A starts, in §1.4 wording). No synthesis-style claims about *expected* results beyond the predictions explicitly carried from the hypothesis. |

---

## Discipline-rule checks

| Discipline rule | Status | Evidence |
|---|---|---|
| Design for falsification, not confirmation | PASS | F1, F2-η0, F2-FLA, F3, F4, F5 all have falsification *directions* committed in §4 table. F1 has both an upper-bound (support) and lower-bound (falsify) decision rule. The 6 experiments are designed to fail if the hypothesis is correct: F2 specifically is the steelman test ("if TTT-Linear K=4 retains advantage over η=0 and FLA-stationary at K=4, mechanism is wrong"). |
| Pre-register the decision rule | PASS | All thresholds (r=0.5, r=0.8, |Δ|=2.0, +0.2 CoT shift, +2.0 LaCT gap) committed in §0/§4 before any run. Bonferroni α=0.01 in §5.3. Hierarchical bootstrap protocol fully specified in §5.1. Escalation rules in §5.4 are frozen. |
| Honest compute estimates | PASS | Estimate is OVER the 2,000-hr fence and is *explicitly flagged as such* in both `output.md` §8.5 and `manifest.yaml` `flagged_intractable: true`. Cheap-Path-A is in-fence at 150 hr but does NOT kill the hypothesis at the predicted r ≈ 0.33 — only the trivial-null r ≈ 1.0. This nuance is documented in `manifest.yaml` `cheaper_falsification_path_kill_test_status: PARTIAL`. |
| Stay in lane | PASS | No experiment run; no synthesis. Eval scripts committed as design artifacts but not executed. AST dry-run pre-registered for execution before Phase A — not part of this design document. |

---

## Red-team round-2 residual issues handled (I1-I5)

| Residual issue | Severity | Handled in eval-design? | How |
|---|---|---|---|
| I1 — 19-language CRUXEval-X expansion overcounts effective N | Important | YES | §5.1 hierarchical bootstrap with base-problem-level clustering (resamples problems-with-all-19-languages-as-units). §5.2 power analysis uses ~310 base problems as effective N (not 5,890 item-instances). §5.4 staged seed escalation 5 → 10 → 15 with explicit triggers. |
| I2 — η=0 collapses to position-wise MLP, not linear attention | Important | YES | §2.5 SPLITS the F2 distinguisher into B5a (η=0, position-wise MLP) AND B5b (FLA-stationary, true static linear attention). §4 splits F2 into F2-η0 and F2-FLA. Mechanism interpretation table for the 4 (F2-η0, F2-FLA) outcomes. |
| I3 — M2 over-generalises TRM §4.5 | Important | YES | §2 prelude reframes M2 to depend only on "TRM re-runs the backbone-induced operator." §3.7 M_diag2 anchors saturation-of-recurrence to arXiv:2604.21106's φ ≈ 0.46 recurrence-equivalence exponent — not TRM §4.5. |
| I4 — K_arch axis ambiguity vs TRM T/n/N_sup | Important | YES | §2.6 explicitly commits K_arch = T (full recursion processes), n=4, N_sup=1. PRIMARY gradient-flow = TRM-faithful "gradient through last process only." A2 ablation runs full-backprop variant at K=4 × 5 backbones × 5 seeds (25 runs). |
| I5 — AST depth-extraction unspecified | Important | YES | §1.3 commits the exact Python `ast` extraction script (counts For, While, If, IfExp, FunctionDef, Lambda, comprehension nesting; excludes Call, With, Try). For non-Python languages, uses shared `id` to look up the CRUXEval-O Python source. §1.4 commits the dry-run d-distribution to be reported before Phase A. |

---

## Assumptions and limitations explicitly disclosed

- **Phase A budget OVER the 2,000-hr fence.** `flagged_intractable: true` in manifest. Cheap-Path-A is the realistic in-fence kill-test for the trivial-null case but is NOT a sufficient kill-test for the predicted r ≈ 0.33 vs threshold 0.5 distinction (per Ouro caveat).
- **Effective N at d ≥ 3 is ~310 base problems**, not 5,890 item-instances; this is the I1-corrected value driving §5.2 power analysis. Within-seed binomial std ≈ 2.6 abs pts; SEM-on-r at 5 seeds ≈ ±0.86; bootstrap CI width ≈ ±1.7. Ambiguous zone (r ∈ [0.5, 0.8]) handled by escalation.
- **F2 threshold loosened from hypothesis's 1.5 to 2.0** to address red-team's "marginally powered" finding. F2 escalation triggers at 10 seeds (B1 vs B5a, B1 vs B5b) from Phase A onwards, with further escalation to 15 seeds if 10-seed CI spans the threshold.
- **AST extraction script is for the shared CRUXEval-O Python source**, not per-language AST. Justification: CRUXEval-X language translations are mechanically generated from CRUXEval-O; d is well-defined per base problem.
- **Pretraining corpus uses permissive sub-configs of github-code-clean only** to avoid copyleft license leakage. This excludes GPL/AGPL Java/C++ — possibly biasing the code-domain coverage. Mitigation: report code-language coverage of the actual pretraining mixture as a diagnostic alongside main results.
- **The ablation A2 (full-backprop) costs ~1,250 GPU-hr** (large fraction of the ablation budget) because full-backprop scales as ×K_arch in compute. Acceptable because it tests whether F1's outcome depends on the gradient-flow choice (red-team I4).

---

## Overall verdict

**PASS.**

All required sections present. Every dataset is a real HF dataset with verified license. Statistical analysis plan is pre-registered (Bonferroni α=0.01 across F1-F5, hierarchical problem-clustered bootstrap with 10,000 iterations, staged seed escalation 5/10/15 with frozen triggers, inconclusive-result handling, pre-registration record file with SHA-1 hashes of eval scripts). At least one falsification experiment per F-criterion (F1, F2-η0, F2-FLA, F3, F4, F5 — F2 split into F2-η0 and F2-FLA per red-team I2 fix). Baselines include parent-architecture treatment (B1 TTT-Linear), strong frontier prior-art anchor (B8 Pythia-160M / GPT-Neo-125M cited), trivial sanity (B6 random-token analytic), and ablation baselines (B5a η=0, B5b FLA-stationary). Compute budget grounded (not TBD; full FLOP arithmetic in §8). All five red-team round-2 residual issues (I1-I5) addressed in design.

`flagged_intractable: true` — Phase A primary at ~3,240 GPU-hr is over the 2,000-hr fence. Cheap-Path-A at ~150 GPU-hr is the in-fence kill-test for the trivial-null case (additive composition) but per the Ouro caveat is NOT a sufficient kill-test for the predicted r ≈ 0.33 vs threshold 0.5 distinction. Design honestly recommends user explicit approval for over-fence Phase A budget OR acceptance of cheap-path-only kill-test.

The pivot in the hypothesis from variance-amplification to sub-additive redundancy is reflected in the eval design's switch from std-of-std (revision-0) to first-moment ratio of means (revision-1), which gains the sample-efficiency that revision-0 lacked. The design's load-bearing additions beyond the hypothesis are: (a) the FLA-stationary backbone B5b that fixes red-team I2 (η=0 ≠ static linear attention), (b) the AST extraction script in §1.3 that fixes I5, (c) the gradient-flow ablation A2 that fixes I4, and (d) the hierarchical problem-clustered bootstrap in §5.1 that fixes I1.
