# Verification — hypothesis-smith-2 (H2: TC⁰ escape via external recursion on SSMs)

Self-check against the hypothesis-smith discipline rules and the H2 assignment contract.

## Format checks

- [x] **If/then form.** Section 2 begins "If [intervention] … then [predicted observable]" with explicit comparators (Mamba+TRM-recursion vs Fixed-Point-RNN-Mamba) and quantitative thresholds (≥ +5 absolute points on L_long, ≤ +2 absolute points on L_short).
- [x] **Single hypothesis.** One comparative claim about the (mechanism × chain-length) interaction. Not a bag of sub-claims.
- [x] **Mechanism cited.** Every mechanism claim in §3 has a citation (arXiv:2404.08819, arXiv:2412.06148, arXiv:2410.03810, arXiv:2402.12875, arXiv:2503.10799, arXiv:2510.04871, arXiv:2412.19350, arXiv:2411.12537). The load-bearing claim — "Fixed-Point-RNN iteration is contraction-bounded; TRM is not" — is grounded in arXiv:2503.10799 (its contraction-map convergence requirement) and arXiv:2510.04871 (no such constraint in TRM).
- [x] **Predicted outcome with magnitude.** Section 4 gives a per-arm × per-stratum point-estimate table with ±CI ranges. The headline prediction is +5 absolute points on L_long and ≤ 2 absolute points on L_short.
- [x] **Falsification criteria with metric + threshold + direction.** Five criteria (F1–F5) all give a metric, a numeric threshold, and a direction (e.g., "≤ +1 absolute point" not "the experiment fails").

## Discipline checks specific to this run

- [x] **Architectural recursion vs internal-state-transition recursion explicitly distinguished.** Section 3 has a dedicated subsection "Why Fixed-Point-RNN internal iteration is *different*" with three structural differences (object refined, contraction constraint, iteration scope). The contraction-vs-non-contraction distinction is the load-bearing argument and is itself testable via F1.
- [x] **Recursion vs CoT distinction maintained.** Section 3 final paragraph explicitly cites M1 (arXiv:2504.10449), Scaling Reasoning without Attention (arXiv:2505.22425), Thinking Slow Fast (arXiv:2502.20339) as the *output-CoT* alternatives and states test-time-sampling is held constant across arms. The hypothesis is about in-forward-pass depth, not token generation.
- [x] **Recursion vs agent scaffolding distinction.** Section 3 final paragraph: "no external proof-search scaffolding ... output token sequence is a flat tactic chain ... only the per-token forward-pass compute differs."
- [x] **Non-additive prediction.** Section 2 names the (mechanism × chain length) interaction; section 4 gives the ANOVA-style prediction; section 3 §"Why the interaction is non-additive" provides the mechanism reason.
- [x] **YAGNI fence.** Manifest flags `yagni_compliant: true`. No new pretraining (Mamba checkpoint frozen), no sparse-attention primitive, no kernel work, no AGI claims. Only wrapper params trained.
- [x] **Coherence (recursion accessing tokens the sparse pattern excludes).** Not applicable — this hypothesis uses dense Mamba (not a sparse-attention backbone). No structural coherence violation.
- [x] **Scale plausibility.** Mamba 2.8B + TRM wrapper at K_arch ≤ 8 stays within single-A100 forward-pass budget; LeanDojo execution is the dominant cost; PutnamBench + miniF2F-Lean4 ≈ 884 problems at pass@8 ≈ 7000 forward passes per arm × 5 arms × 5 K-values is ~175k forward passes. Tractable.
- [x] **Cheaper falsification path.** Section 7 specifies an S_5 permutation-composition synthetic probe at < 1% of Lean compute that kills the contraction-vs-non-contraction argument independently.

## Citation checks

All arxiv IDs in §9 were resolved via `mcp__plugin_megaresearcher_ml-intern__hf_papers paper_details` during this session. Specifically verified live during preparation:

- 2510.04871 (TRM) — resolved, confirmed.
- 2503.10799 (Fixed-Point RNNs) — resolved, confirmed contraction-map / fixed-point claim.
- 2503.14456 (RWKV-7 Goose) — resolved, confirmed TC⁰-escape claim via vector-valued ICL-rate.
- 2412.19350 (SD-SSM) — resolved, confirmed dense-transition fix.
- 2402.12875 (CoT Inherently Serial) — resolved, confirmed depth-buys-expressivity result.
- 2404.08819 (Illusion of State) — resolved, confirmed TC⁰ floor for diagonal SSMs.
- 2412.06148 (Computational Limits Mamba) — resolved, confirmed DLOGTIME-uniform TC⁰.
- 2411.12537 (Negative Eigenvalues) — resolved, confirmed parity-fix-via-eigenvalues.
- 2410.03810 (Mamba COPY bound) — resolved.
- 2407.11214 (PutnamBench) — resolved, confirmed Lean 4 / Apache-2.0+MIT.
- 2109.00110 (miniF2F) — resolved, confirmed Lean 4 port via lean-dojo/minif2f-lean4.
- 1905.09381 (CoqGym) — resolved.
- 2306.15626 (LeanDojo) — resolved, confirmed retrieval-K knob exists (ReProver premise selection).
- 2312.00752 (Mamba), 2405.21060 (Mamba-2), 2502.05171 (Huginn) — resolved.

## Risks self-critique

- [x] Risks section non-empty (§8 has four risks: R1 retrieval-bound failure, R2 compute-matching confound, R3 RWKV-7 dominance, plus the unnumbered TRM-objective-transfer caveat). For each risk the contribution-if-it-materializes is named, so the experiment is informative regardless of outcome.

## Final result

**PASS.**

The hypothesis is in if/then form, makes a quantitative comparative non-additive prediction with five falsification criteria each at metric+threshold+direction, every mechanism claim is cited, the architectural-vs-internal-iteration distinction is the load-bearing claim and is itself testable, and a cheaper falsification path is provided. Ready for red-team Phase 4.
