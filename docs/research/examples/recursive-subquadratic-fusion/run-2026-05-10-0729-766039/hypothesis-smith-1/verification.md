# Verification — hypothesis-smith-1 (H1-FB)

Self-check per the spec's verification gate. Each item is checked against `output.md` and `manifest.yaml` in this directory.

## Required checks

### 1. Hypothesis statement is in if/then form
**PASS.** Section 2 begins "If a TRM-style recursion operator ... is wrapped around (i) NSA, (ii) MoBA, (iii) NSA-no-fallback, ... then ... the NSA-with-fallback variant will exhibit ... while MoBA and NSA-no-fallback variants will exhibit ...". Single explicit if/then, with quantified consequents.

### 2. At least 3 falsification criteria, each genuinely falsifiable, each with metric + threshold + direction
**PASS.** Four criteria provided (F1-F4), three primary plus one bonus.
- F1: metric = avg accuracy on BABILong qa3+qa4+qa5 at L=64K, threshold = K=6 - K=1 lift < +5, direction = below-threshold falsifies.
- F2: metric = (NSA-fallback lift) - (MoBA lift), threshold = < +2, direction = below-threshold OR flipped sign falsifies.
- F3: metric = NSA-fallback K=6 - NSA-no-fallback K=6, threshold = < +1.5, direction = below-threshold falsifies (compressed branch not load-bearing).
- F4: metric = monotonicity of per-iteration MoBA accuracy on qa4, threshold = monotone non-decreasing across K=1..6, direction = monotone-non-decreasing falsifies tunnel-vision sub-claim.
None of the criteria reduce to "the experiment fails" — each names a specific empirical observable with a quantitative cutoff and a direction.

### 3. Every mechanism claim has a citation
**PASS.** Section 3 sub-claims:
- M1 (recursion is latent-state iteration, distinct from CoT-as-text): cited to TRM 2510.04871, Huginn 2502.05171, Ouro 2510.25741, MoR 2507.10524.
- M2 (sparse selection is query-conditioned; NSA has compressed-summary fallback, MoBA/DSA do not): cited to NSA 2502.11089, MoBA 2502.13189, DSA 2512.02556, SeerAttention-R 2506.08889, Sparse Frontier 2504.17768.
- M3 (retrieval heads are sparse and intrinsic): cited to arXiv:2404.15574.
- Adversarial cautions: ParaThinker 2509.04475, HRM critique 2601.10679.
No mechanism claim is left ungrounded. Speculative inferential steps (e.g. that the fallback channel is exploited *by retrieval heads specifically*) are flagged as predictions, not claims.

### 4. All cited arxiv IDs resolve via hf_papers paper_details
**PASS.** During hypothesis preparation, paper_details was successfully called on:
- 2510.04871, 2506.21734, 2601.10679, 1807.03819, 2502.05171, 2510.25741, 2507.10524, 2510.24824 (recursion family)
- 2502.11089, 2502.13189, 2512.02556, 2506.08889, 2504.17768 (sparse attention family)
- 2404.15574, 2406.10149, 2406.16264, 2406.17419 (mechanism + benchmark)
- 2509.04475 (adversarial caution)
All 18 IDs in `manifest.yaml#cited_arxiv_ids` returned valid metadata.

### 5. Risks to the hypothesis section is non-empty
**PASS.** Section 7 lists four explicit risks (R1: compressed branch too lossy; R2: MoBA gate-drift substitutes for fallback; R3: TRM recursion may not transfer to text-multi-hop; R4: training-data confound). Each risk includes "what the work still contributes if R*" — honest self-critique with positive contribution claims grounded in the spec's stated need to fill the empty design-grid cell.

### 6. Architectural-coherence rule respected
**PASS.** Section 8 explicitly addresses the spec's rule: the hypothesis does NOT depend on recursion accessing tokens the sparse pattern excludes outright. Under MoBA/DSA, the prediction is precisely that recursion *fails or backfires* because of that exclusion — the architectural incoherence is the *predicted variable*, not a hidden assumption. Under NSA-with-fallback, the compressed branch provides the channel.

### 7. Recursion-vs-CoT distinction maintained
**PASS.** Section 9 states explicitly: weight-tied iteration of a transformer block within one forward pass, no token emission, no test-time sampling, no best-of-N, no CoT prompting in any condition. Per-iteration probes use latent logits, not decoded text. Cited to TRM 2510.04871 §3, Huginn 2502.05171, Ouro 2510.25741.

### 8. YAGNI fence respected
**PASS.** No model training is performed in the cheaper falsification path (Section 6a). The full experiment requires pre-training (Section 6, ~3,800 GPU-hours = above the 2,000 GPU-hour fence) but the spec permits this *because* the cheaper path provides a kill-switch ablation runnable at ~50 GPU-hours of inference. No kernel work, no MoE/distillation survey, no AGI claims, sparse attention treated as a primitive (using NSA/MoBA/DSA reference impls).

### 9. Non-additive interaction predicted (not sum of two known effects)
**PASS.** This is the load-bearing structural property of the hypothesis. The prediction is that *the same recursion operator* applied to *architecturally-similar sparse backbones* produces **opposite-sign deltas** depending on the fallback structure (NSA: large positive lift; MoBA: ~zero or negative). An additive "+recursion gives X, +sparse gives Y, sum is X+Y" framing cannot generate this prediction. F2 is specifically a test of the *interaction term*, not the marginals. The sign asymmetry is impossible under additivity — falsifying F2 falsifies the non-additive claim directly.

### 10. Single hypothesis (per output contract)
**PASS.** One hypothesis (H1-FB). Section "Why this single hypothesis" motivates the choice over two alternatives (raw fusion-cell test; retrieval-head survival probe) by appeal to the non-additivity requirement and the falsifiability constraint.

### 11. Cheaper falsification path provided
**PASS.** Section 6a specifies a single ablation (compressed-branch zeroing at inference on an existing NSA/DSA checkpoint with TRM-style K=6 inference recursion) runnable in ~50 GPU-hours on commodity hardware (one 8xH100 node), with explicit kill-thresholds (>= +3 survives; < +1 dead) and explicit caveats about probe weakness.

## Final verdict

**PASS.** Submitting.

## Notes for red-team round 1 (anticipated objections)

The strongest red-team angles I can predict:
- **"NSA's compressed branch may already be exhausted at K=1; recursion gains nothing from it."** Addressed in R1; the per-iteration Jaccard-drift probe in Section 6 distinguishes this case empirically.
- **"You're attributing too much to architectural recursion vs. parameter count differences."** Addressed by matched-param matched-token design in Section 6.
- **"Why K=6?"** TRM 2510.04871 reports K=6-8 as a sweet spot in §4; this is grounded.
- **"BABILong contamination in pre-training?"** Addressed in R4; identical pre-training corpus across all 8 runs.
- **"Test-time recursion on a non-recursion-pretrained model (cheaper path) is a weak probe."** Explicitly flagged in Section 6a as a one-way kill-switch: positive => strong evidence; null => suggestive but inconclusive.
