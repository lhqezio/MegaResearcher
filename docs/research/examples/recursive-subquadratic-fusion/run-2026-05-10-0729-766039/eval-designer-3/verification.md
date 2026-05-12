# Verification — eval-designer-3

Pre-completion checklist per the eval-designer protocol.

## 1. Every dataset is a real HF dataset (cited by ID with license noted)

| Dataset | HF ID | Verified? | License | Note |
|---|---|---|---|---|
| NoLiMa | `amodaresi/NoLiMa` | YES — `hf_inspect_dataset` returned valid schema (single train split, 5.4 MB, text column) | CC-BY-NC-4.0 (Adobe Research GitHub) | **Flagged non-commercial** in §2.1. Acceptable for academic protocol. |
| RULER | `simonjegou/ruler` | YES — `hf_inspect_dataset` returned valid schema (configs at 4096/8192/16384) | Apache-2.0 (NVIDIA upstream) | OOD primary. |
| Needle Threading | `jonathan-roberts1/needle-threading` | YES — `hf_inspect_dataset` returned valid schema (5 configs incl. Multi_Threads) | MIT/Apache-class per HF card; flagged for verification at run-time | Auxiliary OOD. |
| BABILong (fallback) | `RMT-team/babilong` | YES — `hf_inspect_dataset` returned valid schema (255 config/split combos) | Apache-2.0 | Listed as fallback if EverMemBench unavailable. |
| Synthetic NIAH (head-level anchor) | (procedural per arXiv:2404.15574 §2) | n/a — generated procedurally | n/a | Not an HF dataset; procedural generation explicitly stated. |
| EverMemBench-S (conditional) | unverified | NO — `web_search` returned only paper-abstract sites; no HF mirror | TBD | **Flagged in §2.5 as desirable-but-conditional.** Plan does not block on availability; RULER + Needle Threading carry OOD load. |

✓ Every committed dataset is verified by HF dataset ID with license noted. EverMemBench-S is explicitly flagged as conditional and not load-bearing.

## 2. Statistical analysis plan is pre-registered, not post-hoc

✓ §6.1 commits the F2 decision rule before data observation: point estimate ≥ +0.05, 95% BCa bootstrap CI excludes 0, F6 fires as predicted, F1 does not fire. ✓ §6.2 pre-commits the BH-FDR family. ✓ §6.5 pre-commits no-early-stopping. ✓ §6.6 explicitly marks pre-registration; deviations must be flagged by the synthesist as post-hoc.

## 3. At least one falsification experiment per criterion in the hypothesis

The hypothesis defines five falsification criteria (F1, F2 primary, F3, F5, F6) plus the auxiliary sharpness diagnostic. The eval design has:

| Criterion | Dedicated experiment in §7? |
|---|---|
| F1 (NSA absolute Δ ≥ −0.10) | YES — read off F2 grid |
| F2 (primary differential ≥ +0.05) | YES — full 5-substrate × 4-K head-level grid |
| F3 (task-level Δ ≥ +3 pp on NoLiMa) | YES — same grid evaluated on NoLiMa @ 32K |
| F3-RULER (corroboration) | YES — same grid on RULER NIAH-13 |
| F5 (ordering NSA > Quest ≥ DSA > MoBA) | YES — rank-order from F2 grid |
| F6 (compression-branch is load-bearing) | YES — BB-NSA with `g_cmp=0` clamp |
| Sharpness diagnostic | YES — entropy probe at all K (auxiliary, not falsification) |

✓ Six falsification experiments cover five falsification criteria. F3 has dual experiments (NoLiMa + RULER) for corroboration.

## 4. Baselines include both prior-art and a sanity baseline

| Class | Baselines |
|---|---|
| Prior-art (strongest) | B1 — G&A retrieval-head distillation (arXiv:2602.11374); B2 — SSA sparse-to-standard alignment (arXiv:2511.20102); B3 — dense + K=4 two-pass CoT (Attrieval-equivalent, arXiv:2503.09819) |
| Ablations | B4 — NSA-noCmp (`g_cmp=0`, =F6); B5 — hidden-state replay (no textual draft) |
| Trivial / sanity | B6 — random-text re-feed; B7 — K=1 / no-recursion floor |

✓ Three prior-art baselines, two ablations, two sanity baselines (7 total, exceeds the ≥3 minimum).

## 5. Compute budget estimate is grounded (not "TBD"; range if uncertain)

✓ §9.1 provides full-scale estimate (≈ 2.9e21 FLOP ≈ 8K — 16K H100-hours at 5 seeds), explicitly flagged as exceeding the 2000-GPU-hour fence. ✓ §9.2 provides cheap-path estimate (≈ 1200 H100-hours) with line-item breakdown. ✓ Manifest reports 1200 with `flagged_intractable: true` to signal the full-scale path requires user decision. Per project guidance ("Frontier-scale OK; >2000 GPU-hour designs need cheaper path"), the cheap path is the recommended primary execution and is documented in the hypothesis itself as the intended scale.

## 6. Falsifiability discipline (designed for falsification, not confirmation)

The design **can fail** in the following non-vacuous ways:
- F2 differential could be 0 or negative — directly falsifies.
- F5 ordering could collapse with > 1 swap — falsifies the M2a + M2b decomposition.
- F6 could show NSA-noCmp recovers like full NSA — directly falsifies M2a-as-load-bearing.
- F3 could fail despite F2 holding — head-level effect doesn't propagate (interesting partial failure).
- B3 dense + K=4 could show large recovery — undermines the "sparse × K interaction" story.

✓ The design is genuinely falsifiable; we do not have a path where any plausible result confirms.

## 7. Risks documented honestly (§10)

✓ Ten risks explicitly enumerated, each with a mitigation. The two strongest (D3 routing-quality confound; D4 two-pass-CoT overlap) are explicitly addressed by F6 (disambiguator) and B3 (dense-K control), respectively, both in the experimental grid.

## 8. Red-team round-2 residual issues addressed

- **I6 (Huginn over-interpretation)** — addressed in §11 Sources note: framing softened to "task-dependent effects of depthwise recursion."
- **S6 (Fixed-Point RNNs as analogy)** — addressed in §11 Sources note: marked as analogy.
- **I7 (Attrieval not cited)** — added to §11 Sources and used as B3 baseline reference.
- **I8 (Dreamer / SpiralFormer not cited)** — added to §11 Sources as adjacent prior art.
- **I9 (TRM-framing collapse)** — addressed in §10 D4 with B3 (dense-K control) experimental remediation.
- **I10 (SSA missing baseline)** — addressed by adding B2 SSA as a competing baseline.
- **Round-2 steelman (routing-quality confound)** — addressed in §10 D3 with F6 as disambiguator and a routing-quality probe per arXiv:2511.11571.

## Final status

✓ All required gates pass. The design is falsification-focused, statistically pre-registered, dataset-grounded with HF IDs and licenses, baseline-rich, ablation-rich, compute-honest (with the full-path intractability flag triggered on the manifest), and addresses every Important / Suggestion item from the round-2 red-team critique.

Ready for synthesist hand-off.
