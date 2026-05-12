# Verification — hypothesis-smith-5

Self-check applying `superpowers:verification-before-completion` plus the spec's H5-specific verification surface.

---

## Required-section checks

| Check | Status | Evidence |
|---|---|---|
| Hypothesis statement is in if/then form | PASS | output.md §2: "If [TRM-style architecturally recursive operator composed with subquadratic attention on position-controlled long-context stimulus], then [for K >= K\*, recursive operator locks into spurious latent fixed point that increases with K]." |
| At least 3 falsification criteria, each genuinely falsifiable | PASS | output.md §5 enumerates 4 criteria (F1, F2, F3, F4), each with metric + numeric threshold + direction |
| Every mechanism claim has a citation | PASS | output.md §3: M1 cites 2601.10679 §3.1, §4.4, §5 + 2510.04871 §4.1; M2 cites 2502.01951 Theorems 4.1, 4.2; M3 cites 2512.13898, 2410.04422, 2602.15028; CoT-distinction paragraph cites 2509.04475 §2.2 + 2510.04871 §4.1 |
| All cited arxiv IDs resolve via hf_papers paper_details | PASS | Verified during draft: 2510.04871, 2509.04475, 2307.03172, 2410.04422, 2502.01951, 2502.05167, 2601.10679, 2512.13898, 2602.15028, 2504.07052, 2412.19707, 2509.07980, 2504.17768, 2502.11089 all returned valid metadata |
| "Risks to the hypothesis" section is non-empty (forces honest self-critique) | PASS | output.md §7 lists 5 risks (deep-supervision dominance, TRM transfer failure, opposite-direction interaction, lock-in vs convergence confound, sparsity-pattern dependence) and for each states what the experiment still contributes if the risk materializes |

## H5-specific spec checks

| Check | Status | Evidence |
|---|---|---|
| Cited throughout (not bolted on) | PASS | Citations appear in §1 (gap), §2 (hypothesis), §3 (mechanism — every M-claim grounded), §4 (predicted outcome conditions), §5 (falsification metrics), §6 (experiments), §7 (risks), §8 (table) |
| Falsification has metric + threshold + direction (each criterion) | PASS | F1: metric d^2L/dsdK, threshold <+5pp, direction positive; F2: metric U-delta, threshold <=0, direction positive (deepening); F3: metric Δα, threshold <=+0.02, direction positive growth; F4: metric max-s lock-in increment, threshold <=+5pp, direction positive at some s. Each is metric+threshold+direction. |
| Recursion vs CoT distinction is explicit and engaged | PASS | output.md §3 dedicates a labeled paragraph "Why this is qualitatively distinct from CoT Tunnel Vision" arguing (a) CoT is discrete-token-commit lock-in, (b) recursion is continuous-latent attractor lock-in, (c) the continuous-vs-discrete distinction protects against token-sampling commitment but NOT against attractor-dynamics lock-in. The hypothesis is built on this distinction rather than collapsing the two phenomena. |
| Non-additive prediction | PASS | output.md §2 final paragraph: "multiplicative interaction term s × K with a positive coefficient that exceeds the sum of the marginal s-only and K-only effects. A purely additive failure-mode model is insufficient." Prediction §4.1 quantifies as ∂²L/∂s∂K positive; falsification F1 is specifically this coefficient. |
| Predicts on (lock-in rate, U-curve depth, super-linearity exponent) as functions of (s × K_arch) | PASS | output.md §4 predicts L(s, K), U(s, K), α(s, K) — exactly the three metrics specified in the gap-finder-2 #7 table row. K\* threshold is also predicted as a function of s. |
| Strong-falsification axis embraced | PASS | output.md §1 explicitly attacks "the comforting null." §5 anti-falsifiability hedge prevention: "if all four criteria fail, the hypothesis is wrong — and that is the spec's strong-falsification axis paying off." §7 Risk 3 explicitly notes the opposite-polarity (recursion-helps-under-sparsity) outcome and that it falsifies cleanly. |
| Architectural recursion (not CoT/agent) — discipline rule | PASS | All predicted phenomena are about K passes within a single forward pass (TRM-style), not about token-level CoT or agent-level scaffolding. The CoT-only baseline at matched FLOPs is included as a *control* (§6) precisely to separate the two. |
| YAGNI / scale plausibility | PASS | Experimental sketch §6 uses K ∈ {1, 2, 4, 8, 16}, s ∈ {0.1, 0.25, 0.5, 1.0}, context lengths {8K, 32K, 128K} — all within published TRM/NSA budgets. No exotic compute requirements. License-clean stimulus reconstruction is bounded synthetic generation. |
| Citation discipline (no uncited mechanism claim) | PASS | The CoT-distinction paragraph is the most novel synthesis but each component claim (CoT discrete-commit; latent-continuous; attractor existence; deep-supervision-as-implicit-commitment; sparse-mask center-node bias) is individually cited. |

## Revision response check (Phase 4)

| Check | Status |
|---|---|
| On revisions: every red-team objection has an explicit response | N/A — initial submission, revision = 0 |

## PASS / FAIL

**PASS** — all required checks satisfied. Hypothesis is falsifiable on the strong-falsification axis the spec requires; mechanism is grounded in cited prior work; the recursion-vs-CoT distinction is explicit and load-bearing; predictions are quantitative on the exact (lock-in, U-curve, super-linearity) × (s × K) surface specified by gap-finder-2 #7.

**Asymmetric value confirmed:** confirmation of H5 reshapes the fusion thesis (identifies a specific architectural countermeasure target); refutation of H5 delivers a positive separation result (architectural recursion does NOT inherit CoT lock-in under sparse attention) which directly strengthens the rest of the hypothesis-smith pool.
