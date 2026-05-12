# Verification — H4 revision-1

## Discipline checklist (`superpowers:verification-before-completion`)

- [x] **Hypothesis statement is in if/then form.** §2: "If we train an NSA backbone ... [comparing R0..R3], then on a multi-hop ... benchmark, R2 outperforms R1 by ≥1.5 EM ..."
- [x] **At least 3 falsification criteria, each genuinely falsifiable.** Six in §5: F-Calib, F1, F2, F3, F4, F-Self-vs-A, F-StopGrad. Each specifies direction, threshold, statistical test (paired bootstrap n=10 seeds, α=0.05 two-sided), and a measurable quantity.
- [x] **Every mechanism claim has a citation.** Verified per-claim:
  - M1 (NSA three branches): NSA 2502.11089 §3.2; SSA 2511.20102 Appendix F (compression-branch failure to extrapolate).
  - M2 (MoR gate inherits subset-variance): MoR 2507.10524 §2.2.1 Eq. 2.1.
  - M3 (δ-proxy): SSA 2511.20102 Theorem 1, Definition 1, §5; Prism 2602.08426 §1 (RoPE spectral features).
  - M4 (miscalibration via Gradient Update Deficiency): SSA 2511.20102 Proposition 4.1.
  - M5 (multi-hop dependence): Retrieval Head 2404.15574; SCBench 2412.10319; Latent Multi-Hop 2402.16837.
  - M-Steelman: SSA 2511.20102 Appendix F empirically rebuts the "redundant coverage" claim.
- [x] **All cited arxiv IDs resolvable.** Verified during research: 2511.20102, 2602.08426, 2510.01585, 2603.08391, 2507.10524, 2602.07150, 2402.16837, 2502.11089, 2510.04871 all returned valid `paper_details` / `read_paper` responses. Older citations (ACT 1603.08983, PonderNet 2107.05407, etc.) were verified in revision-0.
- [x] **Risks section non-empty.** Six risks (A–F) in §7; each names a contribution-under-failure.
- [x] **Every red-team objection has explicit response.**
  - **C1 (self-justification of c_{t,k})** — Addressed in §3 M3 (δ-proxy reformulation, c-A operationalization with full-attention supervision) and M4 (explicit Gradient Update Deficiency claim grounded in SSA Prop 4.1). Also F-Self-vs-A and F-Calib give two independent falsification surfaces for this claim.
  - **C2 (F-thresholds at noise floor)** — Addressed in §4 (≥10 seeds, paired bootstrap, per-cell std budget ≤0.6 EM, RULER-multi-hop primary), §5 (thresholds raised: F1 0.5→1.5, F2 1.0→1.5, F3 1.0→1.5, F4 1.0→1.5).
  - **C3 (TRM-PonderNet recipe conflation)** — Addressed by committing to MoR's expert-choice routing (2507.10524 §2.2.1) per red-team option (c). All references to "TRM-style + PonderNet" removed; recipe is now MoR-expert-choice with sigmoid gate `g_t^r = σ(θ_r^T H_t^r)`.
  - **I1 (smoke test not falsification)** — Replaced with F-Calib pre-test (real falsification gate).
  - **I2 (M1 overstated)** — M1 fully restated to acknowledge three-branch architecture; variance argument re-derived as subset-conditional.
  - **I3 (M4/R3 misuses MoEUT/SUT)** — F5 R3-ordering downgraded to exploratory contrast; no longer a falsification.
  - **I4 (scale/cost)** — §8 adds explicit tier gate: F-Calib → Path 2 (~1 GPU-week) → from-scratch only if Path 2 R2-A − R1 ≥ 0.5 EM.
  - **I5 (PonderNet per-sequence)** — Resolved by C3 fix (MoR replaces PonderNet entirely).
  - **I6 (F2 NIAH at saturation)** — F2 calibrated to 60–90% baseline band, not 95–99%.
  - **I7 (MuSiQue@32K contrived)** — Primary benchmark switched to RULER-multi-hop; MuSiQue@32K demoted to secondary.
  - **S1 (drop M6)** — M6 Tunnel Vision dropped.
  - **S2 (cite ReSSFormer / Adaptive Loops)** — Both cited in §1 with explanations of why neither subsumes Gap 4.
  - **S3 (calibration pre-test)** — F-Calib is now a mandatory pre-experiment gate.

## Hypothesis-smith self-review questions

**Q1: Do I believe this hypothesis is defensible?**
Yes. The C1 fix via δ-proxy + SSA's documented Gradient Update Deficiency is a real, citable miscalibration claim. The C2 fix puts statistical thresholds above the noise floor with a pre-registered paired bootstrap. The C3 fix commits to a single, published halting recipe (MoR). The hypothesis is now structurally falsifiable: F-Calib alone can kill it before the expensive run, F-Self-vs-A directly tests the C1 claim, and F-StopGrad directly tests the strong-interaction (M4-strong) framing.

**Q2: What is the residual risk?**
The biggest residual risk is Risk A (steelman wins): F-Calib reveals NSA is well-calibrated on multi-hop, and the hypothesis dies. This is a *desirable* outcome by the discipline rules — falsification is non-negotiable, and a clean kill before the from-scratch run is the best efficient outcome. The contribution-under-failure (a measurement of NSA's effective miscalibration on multi-hop) is itself useful.

**Q3: Is the hypothesis still novel after citing ReSSFormer?**
Yes. ReSSFormer (2510.01585) instantiates R0 (fixed K, no per-token halting). Our R1/R2/R3 cells extend it in a direction (per-token depth halting jointly trained with NSA's per-token block selection) that ReSSFormer does not address. Adaptive Loops (2603.08391) does adaptive looping on dense attention, leaving the sparse-attention case untouched. The Gap 4 cell remains unoccupied.

**Q4: Is the proposed architecture coherent (not a chimera)?**
Yes. The recipe is now: NSA backbone (2502.11089) + MoR expert-choice routing (2507.10524 §2.2.1) for per-token depth + a learned δ-proxy regressor ĉ_φ trained via SSA-style 5%-full-attention mixing (2511.20102) + MoR's recursion-wise KV caching. Each component is published; the novelty is in the **coupling** (the C2 dropped-mass conditioning of the depth gate), which is the gap-finder's identified gap.

## Verdict

`SUBMIT (revision-1)`. Three artifacts produced:
- `output.md` (hypothesis with revision-1 changes documented at top)
- `manifest.yaml` (revision: 1, with red-team-objection accounting)
- `verification.md` (this document)
