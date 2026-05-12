# Verification — eval-designer-4 for H4

Per the discipline rules in the eval-designer agent prompt, this document verifies that:

- Every dataset is a real HF dataset (cited by ID with licence noted).
- Statistical analysis plan is pre-registered, not post-hoc.
- At least one falsification experiment per criterion in the hypothesis.
- Baselines include both prior-art and a sanity baseline.
- Compute budget estimate is grounded (not "TBD").

---

## 1. Datasets — verified real and on HF

Each verified via `mcp__plugin_megaresearcher_ml-intern__hf_inspect_dataset` on 2026-05-10:

| Dataset HF ID | Status | Licence | Used for |
|---|---|---|---|
| `simonjegou/ruler` | Valid (configs 4096, 8192, 16384) | Apache-2.0 | Primary RULER benchmark; configs ≤16K verified, **32K regenerated locally** via `https://github.com/NVIDIA/RULER` (Apache-2.0). The need to regenerate is explicitly called out in §2.1 of output.md. |
| `google/frames-benchmark` | Valid (default/test, 824 items) | Apache-2.0 | Secondary multi-hop benchmark |
| `dgslibisey/MuSiQue` | Valid (default/train, default/validation) | CC-BY-4.0 | Secondary benchmark (demoted from primary per revision-1 §6) |
| `hotpotqa/hotpot_qa` | Valid (distractor train/validation) | CC-BY-SA-4.0 | F-Calib pilot for threshold calibration (red-team I5 fix) |
| `HuggingFaceFW/fineweb-edu` | Valid (114 configs, 4.3 TB) | ODC-By-1.0 | Pretraining corpus |
| `togethercomputer/RedPajama-Data-1T` | Valid (arxiv, wikipedia slices) | mixed permissive | Long-context up-training |

**Status: PASS.** All 6 datasets exist on HF, schemas inspected, licences noted.

---

## 2. Pre-registration — not post-hoc

§6 of `output.md` pre-registers:
- Decision rules for F-Calib, F1, F2, F3, F4, F-Self vs A, F-StopGrad — all written *before* any training run.
- Per-cell σ budget (≤0.6 EM with 10-seed escalation to 15 seeds at >0.6 EM, halt at >1.0 EM).
- Holm-Bonferroni family-wise α=0.05 across {F1, F2, F3, F4, F-Self vs A}.
- F-StopGrad treated as a separate family with TOST at ±0.7 EM equivalence band.
- F-Calib threshold derived from HotpotQA pilot fraction + 5pp (not a chosen number).
- eval-A/eval-B 50/50 split with seed=1 fixed.
- Hyperparameters fixed across R1/R2-A/R2-self/R2-A_stopgrad to prevent asymmetric tuning (R-tuning-1 mitigation).

§6.5 commits to a separate `preregistration.md` file written before training the first cell.

**Status: PASS.** Decision rules and thresholds pre-committed in writing.

---

## 3. Falsification experiments — one per criterion

Hypothesis revision-1 §5 lists six falsification criteria + F-Calib gate:

| Criterion | Falsification experiment | Section |
|---|---|---|
| F-Calib (mandatory pre-experiment gate) | §7.1 F-Calib on NSA-mini, calibrated against HotpotQA pilot per I5 fix | §7.1 |
| F1 (R2-A > R1 by ≥1.5 EM on RULER VT) | §7.2: R0/R1/R2-A trained 10 seeds, paired bootstrap | §7.2 |
| F2 (single-hop NIAH, R2-A − R0 ≥1.5 EM at 60–90% baseline) | §7.3: NIAH single-1, context-length-tuned to non-saturated band | §7.3 |
| F3 (signal-decoding, |w_c|/std ≥ 1.5) | §7.4: post-hoc weight z-score | §7.4 |
| F4 (collapse under dense top-n ≥ 80%) | §7.5: high top-n cell at 410/512 blocks | §7.5 |
| F-Self vs A (R2-A > R2-self by ≥1.0 EM) | §7.6: R2-self vs R2-A | §7.6 |
| F-StopGrad (interaction vs feature, TOST ±0.7 EM) | §7.7: gradient-stopped c_{t,k} vs flowing | §7.7 |

Plus a diagnostic for I4 (compression-branch attention probe) at §7.8 — addresses the "compression-branch-fails-on-multi-hop is implicit extrapolation" residual concern from red-team round 2.

**Status: PASS.** Every criterion has a designed experiment; each experiment has a pre-registered fail condition.

---

## 4. Baselines — both prior-art and sanity

§4 specifies five baselines covering the three required categories:

- **Strongest prior-art**: Adaptive Loops (arXiv:2603.08391) — dense-attention adaptive looping with halting; matched-FLOPs comparison.
- **Architectural ablations of the proposed technique**: R2-A-blind, R2-B, R2-self, R2-A_stopgrad, R2-A-random-NSA.
- **Trivial / sanity baselines**:
  - Trivial-1: copy first noun phrase (heuristic floor).
  - Trivial-2: BM25 nearest-neighbor sentence retrieval.
  - R0-K1: NSA with K=1 (no recursion) — tests whether recursion is needed.

**Status: PASS.** Prior-art baseline (Adaptive Loops) cited with arxiv ID; sanity baselines (BM25, heuristic) included; recursion-removed baseline (R0-K1) tests recursion necessity.

---

## 5. Compute budget — grounded, not "TBD"

§9 provides:
- Per-cell training cost computed from token-FLOPs ≈ 2 × 350M × 30B = 2.1e19, divided by 8 H100s × 990e12 × 0.5 MFU ≈ 340 H100-hours/cell.
- Total maximalist ladder: 13 cells × 10 seeds = 44,200 H100-hours = ~$132,600 at $3/H100-hour.
- **`flagged_intractable: true`** in manifest because the maximalist budget exceeds typical academic limits.
- Staged ladder with explicit gates (Stage 0 ~100 H100-hr, Stage 0.5 ~150 H100-hr, Stage 2 ~800 H100-hr, Stage 3 ~17,000 H100-hr, Stage 4 ~23,800 H100-hr).
- Cheapest falsification path (Stages 0+0.5+2) ≈ 1,050 H100-hours = ~$3,150.

**Status: PASS.** Budget is computed (FLOPs × MFU × cell count × seed count), not asserted; intractable flag set; cheaper falsification path provided.

---

## 6. Red-team round-2 residual concern coverage

| Concern | Where addressed |
|---|---|
| I1 (c-A regressor feature blindness inheritance) | §4.2 R2-A-blind ablation; §8 A1 ablation isolates blindness-immune features |
| I2 (double-discrete-routing training stability) | §3.4 stability protocol with Gumbel-softmax warmup, gate-collapse monitoring, pre-registered fallback |
| I3 (KV caching strategy unspecified) | §3.3 commits to recursion-wise caching as primary; recursive-sharing reported as ablation A6 |
| I4 (compression-branch-fails-on-multi-hop is implicit extrapolation) | §7.8 compression-branch attention probe diagnostic; explicit risk row R-implicit-extrapolation in §10 |
| I5 (F-Calib threshold ≥30% unanchored) | §6.4 replaces fixed 30% threshold with HotpotQA-pilot-fraction + 5pp anchored measurement |

**Status: PASS.** All five residual concerns from red-team round 2 are addressed in the design.

---

## 7. Self-checks (discipline rules)

- **Designed for falsification, not confirmation**: every F-criterion has a pre-registered fail condition; F-StopGrad is *expected to potentially fire*, with the design pre-committing to the partial-falsification interpretation.
- **Pre-registered decision rule**: §6, §6.4, §6.5.
- **Honest compute estimates**: maximalist plan flagged intractable; staged ladder shows what is actually executable.
- **In-lane**: this document designs experiments. Implementation, training, and synthesis are downstream.

**Overall status: PASS — ready for synthesist.**
