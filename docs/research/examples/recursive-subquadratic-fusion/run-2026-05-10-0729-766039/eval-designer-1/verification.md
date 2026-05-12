# verification.md — eval-designer-1 self-check

**For:** `output.md` and `manifest.yaml` in `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/eval-designer-1/`
**Hypothesis:** H1-FB rev-1 (`hypothesis-smith-1/revision-1/output.md`)
**Verification scheme:** superpowers:verification-before-completion + project-specific eval-designer checks per spec.

---

## Required-section presence

| Required section (per task contract) | Present in output.md | Status |
|---|---|---|
| Hypothesis being tested (restated, including falsification criteria) | §0 | PASS |
| Datasets — primary + ≥1 OOD, with HF ID, license, splits, size, motivation | §1.1 (primary BABILong), §1.2 (OOD-1 RULER), §1.3 (OOD-2 GSM8K + MBPP), §1.4 (pretraining corpus) | PASS — 1 primary + 3 OOD (RULER, GSM8K, MBPP) |
| Backbones (≥2 baselines + treatment), exact configs, params, training tokens, source repos | §2.1 treatment arms, §2.2 baselines (B1 parent-arch, B4 frontier, B5 trivial), §2.3 bonus arms, §2.4 summary table | PASS — 5 baselines (B1-B5), exceeds the ≥2 requirement and the spec's ≥3 baselines requirement |
| Metrics — primary + secondary, formal definitions | §3.1 (M_primary with formal def), §3.2 (M_sec1-M_sec6 with formal defs), §3.3 (failure-mode mapping), §3.4 (F4' calibration pilot) | PASS |
| Ablations (≥1 diagnostic) | §4.1 (compressed-branch), §4.2 (K-depth), §4.3 (block-depth), §4.4 (length × backbone), §4.5 (latent vs CoT) | PASS — 5 diagnostic ablations |
| Statistical analysis (pre-registered): test, alpha, effect size, power, std, sample size | §5.1 (primary test, decision rule), §5.2 (secondary tests), §5.3 (effect, σ, power calc), §5.4 (multi-comparison), §5.5 (pre-registration commitment) | PASS |
| Budget — GPU-hours, dataset prep, wall-clock, by arm | §7.1 (full), §7.2 (cheap-A), §7.3 (cheap-B), §7.4 (summary table) | PASS |
| Cheaper falsification path — clear single experiment, kill threshold, expected output | §7.2 + manifest `cheaper_falsification_path_*` fields; F3 ≥ +3.0 support / ≤ +1.0 falsification at 350M/16K stated | PASS for F3+F4'; explicit honesty that cheap path is *not* a F2 kill-test (F2 kill-test = full 1B run, which is itself in-fence) |
| Falsification trace per F1-F4' | §6 table (each F-criterion → arms involved → kill threshold → support threshold) | PASS — all 4 F-criteria mapped to specific experiments |
| Risks & mitigations | §8.1-§8.8 (8 risks: leakage, baseline-tuning asymmetry, eval-suite drift, K=1 implicit recurrence, first-token bias, A2 broken-model, K=6 instability, length-sweep saturation) | PASS |
| License table | §9 (12 resources, all licenses verified) | PASS |
| Sources (arxiv IDs + dataset/repo IDs) | §10 (16 papers, 7 datasets, 6 repos) | PASS |

**All required sections present.** Output complies with task contract.

---

## Spec-specific checks (per project research plan)

| Check from project spec | Status | Evidence |
|---|---|---|
| Primary dataset (in-distribution) | PASS | BABILong primary + 1k subsample (§1.1) |
| ≥1 OOD dataset | PASS | 3 OOD: RULER (single-hop control), GSM8K (short-context), MBPP (program-synthesis short-context) |
| ≥2 baselines, including parent-architecture-only and strong frontier | PASS | B1 = parent-architecture FLOP-matched K=1 NSA-fb; B4 = strong frontier FLOP-matched dense; B5 = trivial sanity |
| Primary + secondary metrics | PASS | M_primary defined formally; 6 M_sec metrics defined |
| ≥1 diagnostic ablation | PASS | 5 diagnostic ablations enumerated in §4 |
| Pre-registered statistical analysis (test, α, effect size, power) | PASS | §5.1-§5.5; α=0.0125 Bonferroni; effect size +5 abs pts; σ=0.020; power=0.95 under predicted effect; pre-registered statement at §5.5 |
| Budget estimate — GPU-hours, dataset prep, wall-clock | PASS | §7 — 1100-1395 hr full, 250 hr cheap-A, 50 hr cheap-B; wall-clock 10 days on 8× H100, 3 days on 32× H100 |
| Frontier-scale designs whose minimum falsifying experiment exceeds 2000 GPU-hours must include a cheaper falsification path | PASS BUT WITH A NUANCE | Full F2 experiment is **1100-1395 hours**, in-fence. So the spec's "must include cheaper falsification path" requirement is technically not triggered. We provide cheap-path A and B anyway (per hypothesis §6a), and verification.md explicitly notes that cheap path A is a sufficient kill-test only for F3 + F4', NOT for F2 — but the F2 kill-test is the full 1B run itself, which is in-fence. **No intractability flag needed.** |
| License per dataset | PASS | §9 license table covers all 7 datasets and 6 reference repos. All research-permissive (Apache-2.0, MIT, CC-BY-3.0/4.0, ODC-BY-1.0, DeepSeek License). |

---

## Discipline-rule checks

| Discipline rule | Status | Evidence |
|---|---|---|
| Design for falsification, not confirmation — experiment can produce result that disconfirms | PASS | §6 explicitly states pre-registered kill thresholds for F1-F4'. §5.1 specifies "Hypothesis FALSIFIED on F2 if bootstrap 95%-CI upper bound on F2_DiD ≤ +2 OR same-sign positive lifts." |
| Pre-register the decision rule (not post-hoc) | PASS | §5.5 explicit pre-registration commitment statement. All thresholds (F1, F2, F3, F4', α, σ, sample size) pinned before any run. |
| Honest compute estimates | PASS | §7 shows full FLOP arithmetic (2.55e20 per 1B/50B-token run × 8 = 2.04e21 → 810 hr base × 1.25 margin = 1215 hr per main bundle); no "TBD" entries. |
| Stay in lane — design only, do not run experiments, do not write synthesis | PASS | This document is purely design. No code written; no synthesis-style claims about *expected* results beyond what hypothesis pre-registers; explicit pre-registration framing. |
| Verify each dataset is real HF dataset | PASS | All 7 datasets verified via `hf_inspect_dataset` calls; 5 returned Status=Valid; 2 had alternative HF IDs (LongBench → Xnhyacinth/LongBench, MBPP → google-research-datasets/mbpp) which resolved to valid datasets. |

---

## Red-team residual issues handled

| Residual issue from red-team round 2 | Severity | Handled in eval-design? | How |
|---|---|---|---|
| N1 — Ouro §4 / Table 5 mis-cited for FLOP-match | Important | YES | §0 RT issues, §1 (header), §2.2, §10 — all FLOP-match references replaced with arXiv:2410.20672 (Relaxed Recursive Transformers) and arXiv:2604.21106 (Iso-Depth Scaling Laws). |
| N2 — PLT mechanism heterogeneity | Suggestion | YES | C2 PLT-G-SWA consistency arm added to §2.3 and the arm table §2.4; not load-bearing for F1-F4'. |
| N3 — F4' Jaccard threshold uncalibrated | Suggestion | YES | §3.4 pilot-calibration rule using cheap-path-A models; threshold becomes max(0.5×δ, 0.05) or directional fallback. |
| N4 — central-claim contribution-magnitude drift | Suggestion | NOTED | Acknowledged in §0 carry-forward; no methodological consequence — noted for synthesis. |

---

## Assumptions and limitations explicitly disclosed

- Cross-seed σ at 1B/50B tokens is extrapolated from Pythia-1B and OLMo-1B replications. If observed σ exceeds 0.02, power-degraded inference is reported (§5.3).
- Cheap path A is a F3 kill-test, not an F2 kill-test. F2 kill-test = full 1B run, in-fence.
- 350M cheap path is at-floor on qa3-5 at L=64K (per BABILong §3.1); cheap path runs at L=16K only.
- B5 trivial baseline is computed analytically; no compute spent.
- The PLT consistency arm (C2) adds 1 pretraining run beyond the original 7; included in the 8-run total in §7.1.

---

## Overall verdict

**PASS.**

All required sections present. Every dataset is a real HF dataset with verified license. Statistical analysis plan is pre-registered with explicit α, effect size, σ, power, and decision rules committed before any run. At least one falsification experiment per F-criterion (F1, F2, F3, F4'). Baselines include parent-architecture-only (B1), strong frontier prior-art (B4), and trivial sanity (B5). Compute budget is grounded with full arithmetic (no TBD), within the 2000 GPU-hour fence at 1100-1395 hours full + 300 hours cheap paths = 1495-1695 hours total. Cheap path A is honestly characterized as a kill-test for F3 + F4' but not for F2 (with the explicit note that the F2 kill-test, the full 1B run, is itself in-fence and therefore no separate cheaper-F2-path is needed).

Residual red-team issues (N1, N2, N3) are addressed in the design; N4 is noted as a synthesis-stage concern with no methodological consequence.

`flagged_intractable: false` — under fence.
