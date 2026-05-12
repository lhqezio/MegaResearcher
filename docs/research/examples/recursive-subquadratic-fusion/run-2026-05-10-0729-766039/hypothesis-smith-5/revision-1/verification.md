# Verification — H5 revision-1

Required checks per `superpowers:verification-before-completion`:

## 1. Hypothesis statement is in if/then form

**PASS.** Section 2 begins with "**If** a TRM-style architecturally recursive operator ... is composed with a subquadratic attention backbone ... and run on a position-controlled long-context reasoning stimulus ..., **then** the K-vs-1 accuracy gain Δ(s, K) ... will be statistically indistinguishable from zero ..." The structural form is explicit, with bolded If/then markers.

## 2. At least 3 falsification criteria, each genuinely falsifiable

**PASS.** Four criteria in section 5:
- **F1** — Δ(s=0.5, K=8) ≥ +10 pp (recursion materially helps under sparse + long-context). Direct measurement, calibrated threshold (~2 difference SEs).
- **F2** — Δ(s=0.5, K=8) ≤ −10 pp (recursion materially hurts; revision-0 lock-in framing was right). Symmetric falsification in opposite direction.
- **F3** — Sparse-vs-dense interaction in long-context Δ(s=0.5, 8) − Δ(s=1.0, 8) ∈ [−5, +5] pp (no meaningful interaction; M2 falsified).
- **F4** — Positive-control gate: Δ_train(K=8) ≤ +10 pp on Sudoku-Extreme (operator broken; experiment cannot test H5 — pre-registered as kill, not falsification).

F1 and F2 are *symmetric* falsifications in opposite directions — there is no rhetorical escape. The plateau prediction commits to a narrow interval Δ ∈ (−3, +3); either F1 or F2 alone falsifies cleanly.

## 3. Every mechanism claim has a citation

**PASS.** Section 3 audit:

- **M1** (deep-supervision-trained recursion contracts; no spurious attractors): cited to Parcae arXiv:2604.12946 sec. 3 (spectral radius < 1 contraction; LTI dynamical-system formulation), TRM arXiv:2510.04871 sec. 4.1 (deep supervision pushes z_H toward correct y), arXiv:2510.04871 Table 1 (87.4% on Sudoku-Extreme = empirically convergent), Huginn arXiv:2507.02199 sec. 3.4 (plateau evidence: 4 → 32 steps gives 3.11 → 4.93), Huginn sec. 3.2 (rank trajectory non-smoothness).
- **M2** (weight-tied iterated A^t → dominant eigenvector; first-token-sink under causal/sliding-window mask): cited to arXiv:2502.01951 Theorems 4.1, 4.2 with explicit acknowledgment that weight-tied recursion is a special case of the time-varying theorem (per red-team I3) — qualitative direction authorized, quantitative prediction tagged hypothesis-derived.
- **M3** (long-context OOD is the regime where M1+M2 manifest as plateau): cited to arXiv:2512.13898 (Score Dilution at Test Time), arXiv:2602.15028 (Long Context, Less Focus), arXiv:2510.04871 sec. 5 (TRM training-distribution success).

The Tunnel Vision distinction: cited to arXiv:2509.04475 sec. 2.2.

## 4. All cited arxiv IDs resolve via hf_papers paper_details

**PASS.** Verified during research:
- 2510.04871 (TRM) — resolved, sec. 4.1, 4.2, 5 read.
- 2604.12946 (Parcae) — resolved, sec. 3 read.
- 2507.02199 (Huginn) — resolved, sec. 3.2, 3.4 read.
- 2502.01951 (Position-Bias Emergence) — resolved, sec. 4 read.
- 2601.18401 (Superlinear Multi-Step Attention) — resolved, paper_details read.
- 2407.11963 (NeedleBench) — resolved, paper_details read.
- 2509.04475, 2601.10679, 2410.04422, 2502.05167, 2502.11089, 2504.17768, 2307.03172, 2512.13898, 2602.15028, 2509.07980, 2412.19707, 2504.07052 — verified during revision-0; carried forward.

## 5. The "Risks to the hypothesis" section is non-empty

**PASS.** Section 7 lists 5 distinct risks: (1) TRM doesn't transfer to long context at all (mitigated by K=1 stability gate), (2) deep supervision generalizes farther than predicted (genuine falsification path via F1), (3) plateau confounded with floor effect (mitigated by stimulus-validation gate), (4) sparsity-pattern dependence (mitigated by 2-pattern design), (5) plateau confounded with no-useful-work-at-any-K (mitigated by dense reference). The "even if this risk materializes, the result is informative" hedge from revision-0 has been **removed entirely** per red-team I4.

## 6. On revisions: every red-team objection has an explicit response

**PASS.** "Changes from revision-0" section at top of `output.md` addresses each objection:

**Critical (all addressed):**
- C1 (broken lock-in metric L) — Lock-in metric *abandoned*; replaced with population-level Δ(s, K) accuracy delta that is well-defined regardless of K=1 correctness.
- C2 (deep supervision misread as commitment device) — Acknowledged as misreading; commitment-device framing dropped; mechanism rebuilt around what deep supervision *actually does* (push toward correct y, dissolve wrong directions during training).
- C3 (HRM-attractor mechanism doesn't transfer to TRM) — Polarity flipped (path (c) the red-team explicitly offered). New mechanism positively grounded in Parcae spectral-radius contraction and Huginn empirical plateau.

**Important (all addressed):**
- I1 (F1 noise-floor calibration) — Power calculation included (200 instances/cell; per-cell SE ~3.5 pp; difference SE ~5 pp; thresholds at 2× SE).
- I2 (Parcae + Superlinear citations) — Parcae load-bearing for M1; Superlinear engaged in section 1.
- I3 (Theorem 4.2 → recursion extension) — Now argued explicitly: weight-tied iterated stochastic matrix → dominant eigenvector by Perron-Frobenius. Quantitative prediction tagged hypothesis-derived.
- I4 (asymmetric-value hedge) — Removed entirely. Falsification is falsification.
- I5 (α-exponent extrapolation) — Moved to secondary diagnostic, tagged hypothesis-derived.
- I6 (Risk 2 interpretability-killing) — Pre-registered K=1 stability gate at 30% accuracy; if gate fails, experiment yields different finding, not falsification.

**Suggestions:**
- S1 (Tunnel-Vision naming conflation) — Renamed to "Latent Plateau, Not Lock-In."
- S2 (NoLiMa license-clean construction unbounded) — Bounded scope: explicit template family (5 attributes, 2-hop chain, fixed paragraph schema), referenced to NeedleBench Ancestral Trace Challenge (arXiv:2407.11963) for paradigm.
- S3 (Engage 2603.21676, 2512.19941) — Partial: did not engage these directly given scope budget. The flipped polarity makes them less central; the load-bearing engagement is with Parcae + Huginn.
- S4 (K* prediction over-specific) — Dropped entirely.

## Honesty disclosure

The polarity flip means revision-1 is making a *different* prediction from revision-0, not a refinement of it. This is honest given that the red-team correctly identified that revision-0's mechanism story was wrong (C2, C3). The new prediction (latent plateau) is narrower and more falsifiable: F1 and F2 are symmetric falsifications, and the plateau interval (−3, +3) pp leaves no rhetorical escape direction. If both F1 and F2 fail and Δ lands cleanly in (−3, +3), the hypothesis is confirmed; if either fires, it is refuted; there is no third "still informative" outcome.

The mechanism is grounded in directly-cited prior art (Parcae spectral-radius, Huginn empirical plateau, Position-Bias Emergence weight-tied corollary) rather than in the HRM-attractor inheritance argument that was the central weakness of revision-0.

## Verdict

All six required checks pass. Submission is defensible.
