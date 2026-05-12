# Red-team H4 revision-1: verification-before-completion

## Independent literature queries (≥3 required)

**Query 4** (round-2): `MoR mixture of recursions sparse attention NSA combination` — 10 results.
- Returned: MoR (2507.10524), MoSA (2505.00315), MoA (2406.14909), NSA (2502.11089), unrelated MoE.
- **None combine MoR's expert-choice recursive routing with NSA-style block selection + dropped-mass-conditioned halting.**

**Query 5** (round-2): `dropped attention mass conditioning halting depth gate full attention proxy supervision` — 10 results.
- Returned: gated attention (2505.06708), attention dropout (2310.18738), saliency-based hallucination work, miscellaneous.
- **No paper uses dropped attention mass as a feature for depth-routing.**

**Query 6** (round-2): `recursive transformer adaptive computation sparse attention long context reasoning halting` — 8 results.
- Returned: ReSSFormer (fixed-K), Sparse Frontier (no recursion), Adaptive Loops (dense-attention).
- **No paper does adaptive depth halting + sparse attention jointly.**

Combined with gap-finder's 2 queries and round-1's 3 queries, **6 independent queries** failed to surface a paper occupying the cell.

**Verdict: gap_claim_survives = true.**

## Citation spot-checks (≥3 required, focused on revision-1's load-bearing additions)

**(a) SSA Proposition 4.1** — verified verbatim from arXiv:2511.20102 §4.2. Quote includes "For any token j in a dropped block, its gradient is zero."

**(b) SSA Theorem 1** — verified verbatim. Quote includes "‖h_full(t) − h_sparse(t)‖ ≤ δ(t) (max ‖v(j)‖ + ‖h_sparse(t)‖)" and "This bound scales linearly with δ(t)."

**(c) SSA Appendix F** — verified verbatim. Quote: "replacing all three with full attention (fa+fa+fa) yields extremely high perplexity (191.3)... only the selection module extrapolates well to full attention, and the compression and sliding window modules both fail."

**(d) MoR §2.2.1 Expert-choice routing** — verified. `g_t^r = G(θ_r^T H_t^r)` with sigmoid/tanh activation, β-percentile thresholding, hierarchical filtering. Smith's recipe matches the paper.

**(e) ReSSFormer §3** — verified. R2MU iterates K times (fixed). ASAM uses sparsemax + top-k + MoE. No per-token depth halting. Smith's "≈ R0 baseline" framing is correct.

**(f) Adaptive Loops via paper_details** — verified. Adaptive per-layer looping with halting on dense attention, gated memory.

**(g) Prism via paper_details** — verified subject. Prism is about improving block-sparse importance estimation via spectral-aware coarse attention. Smith's use as a feature source for c-A regressor is a downstream application; flagged as Suggestion S1.

All revision-1 load-bearing citations check out. The miscalibration claim is grounded in real, citable theorems and empirical results.

## Verdict-severity consistency check

Verdict: APPROVE.
Critical objections: 0.
Important objections: 5 (I1–I5, none block APPROVE; all are eval-designer carry-overs).
Suggestion objections: 4 (S1–S4).

Round-1 Critical issues:
- C1 (self-justification) → Resolved via SSA-grounded miscalibration claim + δ-proxy regressor.
- C2 (noise-floor thresholds) → Resolved via 10-seed minimum, σ ≤0.6 budget, thresholds raised to 1.5 EM, RULER primary.
- C3 (TRM/PonderNet conflation) → Resolved via MoR expert-choice routing commitment.

APPROVE with 0 Critical objections is consistent. Important and Suggestion objections carry forward to eval-designer for Phase 5 specification, not back to hypothesis-smith for Phase 3 revision.

## Reframe check (orchestrator's flag)

The orchestrator asked whether the revision morphs the hypothesis from "halting × sparse routing coupling is ill-defined" to "can a learned δ-proxy regressor improve halting?"

Finding: the revision narrows the *same* gap (Gap 4: per-token depth halting × per-token sparse routing × jointly trained × multi-hop) by specifying a precise mechanism (NSA's selection-branch miscalibration enables dropped-mass signal to inform halting). The δ-proxy regressor is the *signal* the coupling needs, not a different gap. F-Calib gates the entire claim before from-scratch training; F-Self-vs-A directly tests whether c-A is the C1-fix it claims to be; F-StopGrad distinguishes interaction from feature.

The reframe is a sharpening, not a pivot. Cell coverage unchanged.

## Files produced

- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-4/revision-1/output.md`
- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-4/revision-1/manifest.yaml`
- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-4/revision-1/verification.md`
