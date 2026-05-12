# Verification — hypothesis-smith-6 (H6-VAR)

Self-check per the spec's verification gate. Each item is checked against `output.md` and `manifest.yaml` in this directory.

## Required checks

### 1. Hypothesis statement is in if/then form
**PASS.** Section 2 of `output.md` is structured as a single "If [TRM-recursion wrapped on TTT vs Mamba vs dense at matched params and ≥5 seeds, evaluated on CRUXEval-O split by program-depth d], then (a) TTT × TRM-recursion shows variance-amplifying middle band at K∈{2,4} with std ≥2.0× K=1 baseline; (b) controls show monotone non-decreasing means and flat std; (c) interaction term ≥1.5× larger marginal." Single explicit if/then with three labeled quantified consequents.

### 2. At least 3 falsification criteria, each genuinely falsifiable, each with metric + threshold + direction
**PASS.** Six criteria provided (F1-F6, five primary plus η=0 mechanism check):
- F1: metric = σ(TTT, K=4); threshold = < 2.0 × σ(TTT, K=1); direction = below-threshold falsifies.
- F2: metric = TTT std-lift minus Mamba std-lift; threshold = < +1.5 abs points; direction = below-threshold or flip-sign falsifies.
- F3: metric = σ(TTT, K=4) minus midpoint of σ(TTT, K=1) and σ(TTT, K=16); threshold = < +1.0 abs points; direction = below-threshold falsifies (rules out monotone-in-K alternatives).
- F4: metric = σ(TTT, K=4, d≥4) minus σ(TTT, K=4, d=1); threshold = < +1.5; direction = below-threshold falsifies (depth-axis specificity).
- F5: metric = σ(TTT, K=4, no-CoT) minus σ(TTT, K=4, CoT-helpful); threshold = < 0; direction = below-threshold falsifies (CoT-confound control).
- F6: metric = σ(η=0 TTT, K=4) compared to σ(TTT, K=4); threshold = >= σ(trained-TTT); direction = static-weight matching trained-weight falsifies the M1/M3 mechanism.

Each criterion specifies an empirical observable (across-seed std), a quantitative cutoff in absolute accuracy points, and a falsifying direction. None reduces to "the experiment fails." No single criterion is a marginal-only test — F1 is the load-bearing variance criterion, F2 forces the non-additivity, F3 forces the non-monotonicity, F4-F6 isolate mechanism from confounds.

### 3. Every mechanism claim has a citation
**PASS.** Section 3 sub-claims:
- M1 (TTT inner-loop produces non-stationary fast weights along sequence): cited to TTT 2407.04620 §2.1, Figure 4 (explicit non-convergence at T=2048); LaCT 2505.23884 (chunked stabilization motivation); Longhorn 2407.14207 (SSM = amortized online learner).
- M2 (TRM outer loop reads fast-weight state K times within forward pass; refined (y,z) per iteration): cited to TRM 2510.04871 §4.1 full-recursion definition and §4.2 (y,z) reinterpretation.
- M3a (stale fast-weight reads under deep supervision): cited to TRM 2510.04871 §2.4 deep supervision and §4.1 full back-prop replacing 1-step gradient; Bai et al. 2019 fixed-point assumption (cited via TRM's own citation, which TRM removes — argument is that TTT does NOT satisfy that assumption).
- M3b (over-fitted fast-weights when inner loop converges too fast): cited to LaCT 2505.23884 (the over-fitting failure mode that motivates LaCT).
- M4 (CRUXEval as test surface; CoT-helpful/harmful subsplit; per-instance program-recursion-depth as program AST invariant): cited to CRUXEval 2401.03065 §1, structural-recursion negative baseline 2401.12947, looped-models-match-deeper-unlooped 2502.17416, CRUXEval-X 2408.13001.
- M5 (non-additivity argument): cited to TTT 2407.04620 Figure 4 (K=1 stable across seeds), TRM 2510.04871 Table 1 (K=4 Mamba/dense stable across seeds).
- Speculative-but-flagged claim about bimodal seed distribution: explicitly listed in §7 R2 as a risk, not a prediction load-bearing on falsification.

No mechanism claim is left ungrounded.

### 4. All cited arxiv IDs resolve via hf_papers paper_details
**PASS.** During hypothesis preparation, paper_details was successfully called on:
- 2510.04871 (TRM), 2407.04620 (TTT), 2401.03065 (CRUXEval), 2401.12947 (structural recursion), 2505.23884 (LaCT), 2407.14207 (Longhorn), 2502.05171 (Huginn), 2506.21734 (HRM), 2510.25741 (Ouro), 2507.10524 (MoR), 2502.17416 (Latent Thoughts), 2408.13001 (CRUXEval-X), 2311.12424 (Looped Transformers), 2410.01405 (Expressive Power Looped), 2504.05298 (One-Minute Video TTT).
- 2602.21204 (TTT-as-linear-attention) cited from gap-finder-1's verified bibliography. 16 IDs in `manifest.yaml#cited_arxiv_ids`.

### 5. Risks-to-the-hypothesis section is non-empty
**PASS.** Section 7 lists five explicit risks (R1: scale dependence may push effect above 125M; R2: bimodal seed distribution rather than continuous broadening; R3: CRUXEval may saturate the variance signal at 800 items; R4: the two recursions may not interact at all; R5: Mamba may not be a clean stationary control). Each risk includes an explicit "what the work still contributes if R*" — honest self-critique with positive contribution claims grounded in the spec's stated need to fill the empty design-grid cell.

### 6. Sequence-time vs depth-time distinction explicit
**PASS.** Section 1 (Targeted gap) defines the two axes verbatim from the gap-finder. Section 3 (M2, M3) names the substrate over which they interact (the fast-weight state W_t), the rate at which each operates (TTT updates per token along sequence-time; TRM reads K times along depth-time within one forward pass), and the rate-mismatch that drives the destructive interaction. Mamba control isolates "no fast-weight inner loop" from "depth-time outer loop"; dense-softmax control isolates "no sequence-time updates at all." η=0 TTT control (F6) isolates "fast weights present but frozen" — the cleanest test of M1.

### 7. Recursion-vs-CoT distinction maintained
**PASS.** Section 4 specifies "no text-CoT in primary condition" explicitly. Section 5 F5 is a dedicated CoT-confound control: variance lift must be larger under no-CoT than under CoT-helpful sub-split (CRUXEval has both, arXiv:2401.03065 §5). Mechanism (M4) explicitly distinguishes program-recursion-depth d (program AST invariant, not text emission) from text-CoT (external chain-length axis). The hypothesis's load-bearing prediction lives in the latent-iteration-only condition; CoT existence in the data is treated as a confound to control, not a feature to exploit.

### 8. Non-additive interaction predicted (not sum of two known effects)
**PASS.** This is the structural property of the hypothesis: the same TRM operator and the same TTT layer, when each is run alone (TTT at K=1, TRM-on-Mamba/dense at K=4), are stable across seeds (M5: cited to TTT Figure 4 and TRM Table 1). The variance lift at TTT × K=4 cannot arise as a sum of these — it is a *multiplicative interaction term*. F2 directly tests this: (TTT std lift at K=4) minus (Mamba std lift at K=4) must be ≥+1.5 to survive. An additive composition predicts F2 ≈ 0; the hypothesis predicts F2 ≥ +3.0. Sign flips also falsify F2.

### 9. YAGNI fence respected
**PARTIAL with mitigation.** The full experiment in §6 estimates ≈3,000-4,000 GPU-hours, above the 2,000-GPU-hour fence. §6a provides a kill-switch cheaper falsification path at ≈50 GPU-hours on a single 8×H100 node, runnable on existing open-weights TTT/LaCT/Mamba checkpoints with TRM-style test-time wrapping. Explicit kill-thresholds (σ lift < +1.0 dead) and explicit caveats about probe weakness (test-time-only ≠ pre-training, per Ouro). The cheaper path is the gate; the full experiment is conditional on a positive cheaper-path result. This is the same YAGNI-respecting pattern as hypothesis-smith-1.

### 10. Single hypothesis (per output contract)
**PASS.** One hypothesis (H6-VAR). The spec offered three candidate directions; this hypothesis chose the *destructive-interference* direction, motivated by (a) the non-additive prediction requirement (the redundancy direction can be additively explained as "K_arch absorbs depth requirement"; the destructive direction cannot), and (b) the mechanism-grounding requirement (TTT Figure 4 and LaCT motivation give explicit citations for fast-weight non-stationarity).

### 11. Cheaper falsification path provided
**PASS.** Section 6a specifies a single test-time wrapping experiment at ≈50 GPU-hours, with explicit kill-switch (σ lift < +1.0 dead) and explicit caveat about probe weakness vs pre-training.

### 12. Architectural-coherence rule respected
**PASS.** The hypothesis predicts a *failure* mode that is a feature of architectural coherence, not a violation of it. TTT and TRM are both valid stand-alone architectures; the prediction is about what happens at their composition boundary, with controls (Mamba, dense, η=0 TTT) that systematically vary which property of TTT is relevant. The destructive-interaction signal is the *predicted variable*, not a hidden assumption.

### 13. Scale plausibility
**PASS.** Anchored to TTT paper's 125M parameter scale (arXiv:2407.04620 main experiments), TRM's 5-27M scale (arXiv:2510.04871, Table 1), CRUXEval-scale evaluations on 125M-class open models (35-40% pass@1 range per arXiv:2401.03065 §1). 5 seeds × 5 K_arch × 4 backbones × 125M params is computationally aggressive but achievable; the cheaper path drops this to inference-time evaluation on existing checkpoints, ~50 GPU-hours.

## Final verdict

**PASS.** Submitting.

## Notes for red-team round 1 (anticipated objections)

The strongest red-team angles I can predict and how this hypothesis is positioned to absorb them:

- **"σ across 5 seeds is too noisy to draw conclusions."** Acknowledged in R3. Pre-registered sample-size analysis recommended; CRUXEval-X (15K items) as 19-language replicate provides higher-power follow-up. The 2.0× and 4× thresholds in F1 are deliberately generous to absorb expected sampling variance.

- **"Variance differences could come from any training-instability source, not the M3a/M3b mechanism."** Addressed by F4 (depth-axis specificity), F5 (CoT-confound control), and F6 (η=0 frozen-fast-weight test). If σ at η=0 TTT × K=4 matches σ at trained TTT × K=4, the mechanism is wrong and the work outputs a clean negative result on the destructive-interaction prediction.

- **"Why CRUXEval and not BABILong / proof search?"** CRUXEval gives per-instance program-recursion-depth as a *program AST invariant*, which is the gap-finder-1 framing of "matching K_arch to depth requirement." Long-context proof search confounds context length with reasoning depth; CRUXEval at 3-13 lines holds context fixed (low) and varies depth — exactly the H6 test surface.

- **"Test-time-only TRM wrapping in §6a is a weak probe."** Explicitly flagged; treated as a one-way kill-switch (positive = strong evidence; null = suggestive but inconclusive), per the Ouro arXiv:2510.25741 caveat that iteration must be baked in at pre-training.

- **"TRM has only been demonstrated on small fixed-context puzzles (Sudoku, Maze, ARC), not on Python execution prediction."** Per arXiv:2510.04871 §4.5, TRM's attention-on variant works (74.7% on Sudoku-Extreme); Reasoning with Latent Thoughts (arXiv:2502.17416) generalizes the looped-recursion result to multi-task reasoning. The architectural recipe transfers; whether it transfers to CRUXEval at 125M is part of the experimental contribution, not a prerequisite assumption.

- **"Mamba is not a fully stationary control because Mamba's hidden state is also context-dependent."** Acknowledged R5; mitigated by adding the η=0 TTT control (F6), which is provably equivalent to fixed-`W` linear attention per arXiv:2602.21204.
