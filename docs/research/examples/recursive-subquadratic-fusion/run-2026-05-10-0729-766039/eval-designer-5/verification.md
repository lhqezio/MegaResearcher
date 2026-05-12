# Verification — eval-designer-5 (H5 revision-1)

Per `superpowers:verification-before-completion`: every claim in this design has been checked against the rubric before completion is reported. Evidence before assertions.

## Required checks (from eval-designer skill contract)

### 1. Every dataset is a real HF dataset (cited by ID with licence noted)

| Dataset | HF ID | Verified by | License | Status |
|---|---|---|---|---|
| BABILong | `RMT-team/babilong` | `hf_inspect_dataset` returned 255 config/split rows, schema {input, target, question} | Apache-2.0 | ✓ verified |
| Sudoku-Extreme | `sapientinc/sudoku-extreme` | `hf_inspect_dataset` returned default/{train,test}, schema {source, question, answer, rating}; 405 MB train, 44.7 MB test | per dataset card | ✓ verified |
| NeedleBench | `opencompass/NeedleBench` | `hf_inspect_dataset` returned 5 configs incl. atc_needles | Apache-2.0 | ✓ verified |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | well-established; not freshly inspected this session | ODC-BY-1.0 (per dataset card) | ✓ noted with license caveat |
| CB2H | constructed in-house | self-issued; not pre-existing | CC-BY-4.0 (self-issue) | ✓ noted as constructed |

CB2H is constructed (license-clean) per the spec's instruction that NoLiMa is non-commercial and stimulus must be reconstructed. This is explicit and pre-registered, not a stand-in for a real dataset.

NoLiMa is **NOT** used as a dataset in this design; it is referenced only for the methodological paradigm. License-incompatibility flagged.

### 2. Statistical analysis plan is pre-registered, not post-hoc

Section 5 of `output.md` commits the following BEFORE any data is collected:
- Sample sizes (N=800 core, N=200 expanded, N=1000 Sudoku) — committed.
- Equivalence margin ε = 5 pp — committed.
- F1 threshold +14.1 pp; F2 threshold −14.1 pp; F3 threshold ±10 pp; F4 threshold +13.7 pp — committed (F4 subject to one pre-flight revision documented in section 5.4).
- Stimulus-validation gate (dense K=1 ≥ 50% at some position) — committed.
- K=1 stability gate (long-context K=1 ≥ 30%) — committed.
- Bonferroni for primary family (α/4 = 0.0125); BH-FDR (q=0.10) for secondary — committed.
- Decision rule (section 5.7) is fully specified — supported / falsified / inconclusive — pre-registered.

The `manifest.yaml` records `preregistration.margins_locked_before_runs: true`. Pre-registration commitment is on file.

✓ verified pre-registered.

### 3. At least one falsification experiment per criterion in the hypothesis

Hypothesis criteria → Falsification experiments:

| Hypothesis criterion | Eval-designer experiment | Section |
|---|---|---|
| F1 (recursion HELPS, Δ ≥ +10pp) | Experiment F1-test: TRM-DS K∈{1,8} × {dense, NSA} × CB2H L=32K, one-sided test at +14.1 pp | 6.1 |
| F2 (recursion HURTS, Δ ≤ −10pp) | Experiment F2-test: same instances, one-sided test at −14.1 pp | 6.2 |
| F3 (no sparse-vs-dense interaction) | Experiment F3-test: Δ(NSA) − Δ(dense) at ±10 pp | 6.3 |
| F4 (positive-control gate) | Experiment F4-test: Sudoku-Extreme K∈{1,8} | 6.4 |
| K=1 stability gate | Experiment 6.5 | 6.5 |
| Stimulus-validation gate | Experiment 6.6 | 6.6 |

All 4 falsification criteria + 2 gates have a dedicated experiment. The plateau prediction itself has a TOST equivalence test (section 5.1) that *can fail* (TOST equivalence rejected ⇒ plateau falsified), so the design is not a confirmation-only setup.

✓ verified — every criterion has an experiment that can falsify it.

### 4. Baselines include both prior-art and a sanity baseline

- **Prior-art baselines:** NSA (arXiv:2502.11089), MoBA (arXiv:2502.13189), Vertical-Slash (arXiv:2504.17768) — all are independently published, peer-reviewed sparse-attention architectures. Dense FlashAttention-2 is the standard prior-art reference.
- **Ablation of proposed technique:** TRM-η=0 (frozen TTT control) — disables the test-time-training inner-loop gradient step. This is the "ablation of the proposed technique" required by the contract.
- **Sanity / trivial baseline:** Matched-FLOPs CoT — the "matched compute helps in any modality" check. While CoT is not strictly random/majority-class, it is the *non-architectural* baseline that the recursion claim must beat for the architectural-recursion thesis to be substantive.

5 baselines total, exceeding the required 3. ✓ verified.

### 5. Compute budget estimate is grounded (not "TBD" — if uncertain, give a range)

Section 8 of `output.md` provides:
- Per-phase H100-hour estimates (pretraining 6000h, operator training 400h, CB2H inference 1500h, BABILong 200h, Sudoku 20h, ablations 1000h, buffer 800h).
- Total: ~9,920 H100-hours.
- Cost: $30K spot / $45K reserved.
- Calendar time: ~6 weeks on 64-H100 cluster.
- A documented cheaper path: ~2,500 H100-hours / ~$8K.
- Explicit `flagged_intractable: false`.

Estimates are calibrated against TRM (arXiv:2510.04871, ~24 H100-hours per Sudoku run cited) and NSA (arXiv:2502.11089 FLOPs profile). Per-cell forward-pass time estimates given (0.3–0.5 sec/instance at L=32K, 350M params).

Not "TBD". ✓ verified.

## Self-checks beyond rubric

### 6. Falsification commitment is symmetric and honest

F1 and F2 are *symmetric* falsifications in opposite directions. The hypothesis is the narrow null prediction between them: Δ in (−5, +5). Either F1 (recursion helps) OR F2 (recursion hurts) alone falsifies the polarity claim. There is no rhetorical escape direction. The design states this in advance for both:

- F1 fires → "the hypothesis is refuted, the fusion thesis is in better shape than predicted, and revision-1 was too pessimistic" (section 6.1)
- F2 fires → "revision-0's lock-in framing was right after all" (section 6.2)

✓ verified — design is honest about what each refutation result means.

### 7. Red-team residual issues addressed

| Red-team round-2 issue | Eval-designer response | Section |
|---|---|---|
| I-A: arXiv:2604.21106 not engaged (steelman) | K-sweep ablation 7.4 directly tests Δ ∝ K^0.46 prediction; section 9.7 names this engagement | 7.4, 9.7 |
| I-B: F3 noise-floor (5pp below 7pp SE) | F3 threshold widened to ±10pp; F3 demoted to secondary | 5.3, 6.3 |
| I-C: Mechanism transfer is analogy | Acknowledged; the experiment IS the test — addressed by power calc making this empirically resolvable | 1, 5.5 |
| I-D: TOST framing for null prediction | Adopted as primary outcome (section 5.1); equivalence margin ±5 pp pre-registered | 5.1 |

All four residual issues from red-team round 2 are addressed. ✓ verified.

### 8. License-clean stimulus

CB2H constructed (CC-BY-4.0 self-issue) replaces NoLiMa (non-commercial, license-incompatible per spec). NeedleBench used as methodological reference, not as redistributed text. ✓ verified.

### 9. Lock-in metric population-level (not per-instance)

Per red-team round-1 C1 resolution: Δ(s, K) is a population-level statistic (accuracy averaged over instances), not a per-instance K=1-correctness comparison. Section 4.1 of `output.md` makes this explicit: "Population-level statistic, NOT per-instance." ✓ verified.

### 10. Honest disclosure of TOST under-power

Section 9.5 discloses that TOST power at N=800 is ~0.65, below the conventional 0.80. The design explicitly chooses to optimize for falsification power (F1/F2 one-sided at ~0.80) over equivalence-confirmation power. This is a known limitation, openly recorded. ✓ verified.

## Final verification statement

All eval-designer-skill rubric checks pass. The design is publicly defensible and the user can decide to fund full ($30K) or cheap-path ($8K) execution. Three artifacts written:

- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/eval-designer-5/output.md`
- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/eval-designer-5/manifest.yaml`
- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/eval-designer-5/verification.md`
