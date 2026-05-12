# Verification record — red-team-5 critique of H5

## Independent literature queries (≥3 required)

1. `recursive transformer sparse attention long context lost in middle` — 10 results. None pairs architectural recursion with sparse attention on a long-context Lost-in-the-Middle stimulus. Closest hits: Sparse Frontier (2504.17768), DAM (2506.11104), HSA (2511.23319), Found in the Middle (2406.16008), DHSA (2510.24606).

2. `looped transformer attractor latent fixed point lock-in` — 10 results. Surfaced **Parcae (2604.12946)** (looped-LM stability via spectral norms — directly adjacent to H5's attractor claim) and **Block-Recurrent Dynamics in Vision Transformers (2512.19941)** (Transformer depth as flow). Smith does not engage with these.

3. `tunnel vision latent reasoning recursive depth lock-in` — 10 results. Surfaced **Depth-Recurrent Attention Mixtures (2601.21582)** and **Thinking Deeper, Not Longer (2603.21676)**. Both are conceptually adjacent.

4. `iterative refinement recursion long context position bias amplification depth` — 10 results. No direct intersection.

5. `recursion depth multi-pass attention sparse needle haystack interaction` — 10 results. Surfaced **Superlinear Multi-Step Attention (2601.18401)** — the closest published thing to a multi-step subquadratic architecture. Smith does not cite it.

**Conclusion:** Gap technically survives but is narrower than presented. Three closely adjacent papers (Parcae 2604.12946, Superlinear Multi-Step Attention 2601.18401, Thinking Deeper Not Longer 2603.21676) must be engaged with in revision.

## Citation spot-checks (≥3 required)

1. **arXiv:2502.01951 Theorem 4.1.** Read §4.1 directly. Theorem statement matches smith's representation: P^(t)(z_i=1|X^(0)) → 1 with rate C(1−(j−1)ϵ)^t under causal mask. ✓

2. **arXiv:2502.01951 Theorem 4.2.** Read §4.1 directly. Theorem statement matches smith's representation: sliding-window mask of width w yields rate (1−(j−1)ϵ^⌈(N−1)/(w−1)⌉)^(t/(2⌈(N−1)/(w−1)⌉)). ✓ but the leap from "iterative layer depth" to "weight-tied K-pass recursion" is asserted not proven (Important issue I3).

3. **arXiv:2601.10679 §4.4.** Read §4 directly. Spurious-fixed-point attractor description matches smith. ✓ but the smith glosses over the mechanism (HRM relies on fixed-point assumption + 1-step IFT gradient; TRM removes both) — Critical issue C3.

4. **arXiv:2510.04871 §2.4 / §4.1 (deep supervision).** Read §2 and §4 directly. The smith's "implicit commitment device reinforcing early wrong direction" framing **directly contradicts** §4.1 ("model learns to take any (z_L, z_H) and improve it ... making z_H closer to the solution") — Critical issue C2.

## Verdict-severity consistency check

- 3 Critical objections logged.
- 6 Important objections logged.
- 4 Suggestion-tagged objections.
- Verdict: `REJECT (revision-1)`.
- An APPROVE with 3 Critical objections would be invalid; verdict matches severity.
- A KILL is not appropriate because the gap claim survives, the failure modes are addressable in revision, and the hypothesis targets a real empty cell.

## Discipline self-check

- I steelmanned the position that the hypothesis is *wrong* (§6 of output): "either H5 cannot be measured (Risk 2 hard) or the empirical sign goes the *opposite* direction (recursion helps)."
- I would defend a `REJECT (revision-1)` verdict publicly: the citation discipline issue on deep supervision, the operational issue on L(s,K), and the unaddressed mechanism gap on HRM→TRM transfer are concrete and not performative skepticism.
- I would NOT defend an APPROVE on this hypothesis as currently written.
