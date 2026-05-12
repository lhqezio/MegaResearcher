# Verification before completion — red-team-5 revision-1

## Independent literature queries (≥3 required)

**Query A:** `architectural recursion sparse attention long context reasoning out of distribution` — 10 hits. Closest: NSA (2502.11089), MoBA (2502.13189), DELTA (2510.09883), AsyncTLS (2604.07815), Superlinear Multi-Step Attention (2601.18401). None composes architectural recursion × sparse attention × long-context OOD reasoning. Gap holds.

**Query B:** `depth recurrent transformer long context plateau out-of-distribution generalization` — 10 hits. Hit on arXiv:2510.14095 ("Unlocking OOD Generalization in Transformers via Recursive Latent Space Reasoning"). Read sec. 3 directly: task is modular arithmetic on computational graphs (≤128 nodes), not long-context retrieval. Different task domain. Gap survives but should be cited for breadth.

**Query C:** `looped transformer recurrence depth GSM8K plateau scaling steps` — 8 hits. Hit on arXiv:2604.21106 ("How Much Is One Recurrence Worth? Iso-Depth Scaling Laws"). Establishes recurrence-equivalence exponent of 0.46 across 116 pretraining runs. Contradictory in spirit to strict plateau prediction; not cited by smith. Documented as Important objection I-A.

**Query D (additional):** `tiny recursive model TRM long context needle haystack subquadratic` — 10 hits. No new compositions of TRM × long-context × sparse. Gap holds.

**Query E (additional):** `deep supervision iterative refinement out-of-distribution context length transfer` — 8 hits. All on continued-pretraining for context extension; none on architectural recursion under OOD long-context.

## Citation spot-checks (≥3 required)

**1. Parcae (arXiv:2604.12946) sec. 3** — read directly.
- Smith's claim: "spectral radius < 1 → contraction toward fixed point" is load-bearing for M1.
- Paper actually says: ρ(A̅) < 1 is the LTI stability condition; ρ ≥ 1 implies divergence; empirically verified across runs.
- **Verdict: accurate quote, with acknowledged analogy gap (Parcae is not deep-supervised; smith flags this).**

**2. Huginn (arXiv:2507.02199) sec. 3.4** — read directly.
- Smith's claim: "4→32 steps yields 3.11→4.93 on GSM8K, vs CoT 24.87 — recursion plateaus where CoT scales."
- Paper verbatim: "increasing the number of recurrent steps from 4 to 32 leads to only modest gains in accuracy (from 3.11 to 4.93), and performance plateaus thereafter. In contrast, Huginn with explicit CoT achieves significantly higher accuracy (24.87/38.13)."
- **Verdict: numbers exactly correct.**

**3. TRM (arXiv:2510.04871) sec. 4.1** — read directly.
- Smith's revised claim: deep supervision pushes z_H toward correct y at every supervision step, dissolves wrong directions during training (opposite of revision-0's commitment-device claim).
- Paper verbatim: "Through deep supervision, the models learns to take any (z_L, z_H) and improve it through a full recursion process, hopefully making z_H closer to the solution."
- **Verdict: smith's revised reading is correct; revision-0 was wrong; the retraction is appropriate.**

**4. Position-Bias Emergence (arXiv:2502.01951)** — re-checked from round 1.
- Smith's M2 now explicitly argues the Perron-Frobenius extension to weight-tied recursion: "iterated A^t is the t-th power of a single stochastic matrix ... converges to its dominant left eigenvector."
- Quantitative claim is correctly tagged "hypothesis-derived, not theorem-derived."
- **Verdict: correctly framed.**

## Verdict severity check

Verdict: APPROVE
Critical objections: 0
Important objections: 4 (I-A, I-B, I-C, I-D)
Suggestions: 2 (S-A, S-B)

APPROVE with 0 Critical objections is consistent. APPROVE with 4 Important objections is acceptable when Important objections are about tightening (not soundness) — the eval-designer can address these without changing the hypothesis structure.

## Sanity checks

- Did I verify load-bearing citations? Yes (Parcae sec. 3, Huginn sec. 3.4, TRM sec. 4.1).
- Did I run independent literature queries? Yes (5 queries documented).
- Did I steelman the opposing position? Yes — three prongs (recurrence-equivalence exponent 0.46; TRM's 7.8% on ARC-AGI-2; Huginn's plateau is regime-specific to suppressed-CoT arithmetic).
- Did I check that all round-1 critical objections are resolved? Yes — C1 (lock-in metric abandoned), C2 (deep-supervision misread retracted), C3 (polarity flipped per round-1's offered path c). Plus I1–I6 all addressed.
- Is the verdict severity-consistent? Yes.
- Is the gap claim genuinely surviving independent re-verification? Yes, with acknowledgment that two adjacent papers (2510.14095, 2604.21106) narrow the framing and should be engaged.
