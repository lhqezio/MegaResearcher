# Verification — red-team-2

## Verification-before-completion checks

### Independent literature queries (≥ 3 required)

I ran the following 7 independent literature queries to verify the gap claim and surface adjacent prior art:

1. `hf_papers search query="TRM tiny recursive model SSM Mamba"` — confirms TRM has not been applied to SSM. Result: 10 papers, top hit is TRM 2510.04871 itself; no SSM application.
2. `hf_papers search query="Mamba PutnamBench Lean formal proof"` — confirms no Mamba+PutnamBench paper exists. Top theorem-proving hits are all Transformer-based (Hilbert, Seed-Prover, MA-LoT, DeepSeek-Prover, ReProver/LeanDojo, Lean Copilot, Lean-STaR, InternLM2.5-StepProver).
3. `hf_papers search query="SSM linear attention theorem proving Lean miniF2F"` — same conclusion: no SSM-based prover.
4. `hf_papers search query="Mamba RWKV linear RNN PutnamBench miniF2F CoqGym formal"` — same.
5. `hf_papers search query="recursive depth iteration state-space model SSM Mamba theorem proving reasoning"` — surfaces Retrofitted Recurrence (2511.07384) as adjacent prior art (depth-recurrent retrofit on pretrained Transformers).
6. `hf_papers search query="depth recurrent looped pretrained Mamba SSM retrofit"` — confirms depth-recurrent retrofitting on Transformers but not on SSMs.
7. `hf_papers search query="looped transformer iteration constant depth TC0 latent reasoning"` — surfaces large body of looped-transformer papers (2502.17416, 2602.11451, 2511.08577 "Think-at-Hard" with 110 upvotes, 2603.21676, 2603.08391, 2507.06203 survey on Latent Reasoning).
8. `hf_papers search query="HRM hierarchical reasoning fixed point ARC-AGI Sudoku"` — surfaces critical paper arXiv:2601.10679 ("Are Your Reasoning Models Reasoning or Guessing?") which mechanistically analyzes HRM and finds "guessing behavior rather than true reasoning, with failures stemming from fixed point violations."
9. `web_search "Mamba SSM theorem proving Lean formal LeanDojo PutnamBench evaluation"` — corroborates: no Mamba-on-LeanDojo evaluation in the public record.
10. `web_search "TRM tiny recursive arxiv 2510.04871 Lean formal proof"` — TRM has no Lean application.

**Conclusion:** Gap survives in narrow form; the smith missed Retrofitted Recurrence (2511.07384) and the broader looped-transformer literature as adjacent prior art that should have been engaged.

### Citation spot-checks (≥ 3 required)

I read sections of the following primary citations:

1. **Fixed-Point RNNs (arXiv:2503.10799)** — sections 3 ("Fixed-Points as an RNN Layer"), 4 ("Fixed-Point Mamba"), 5 ("Discussion"). Verdict: **MISREPRESENTED**. The contraction is on the depth-iteration solver (Theorem 3.1), not on the function class. The converged fixed point parameterizes a dense linear RNN. Section 4's Figure 4 shows FP-Mamba solves A_5 and S_5 (permutation composition) — directly contradicting the hypothesis's central claim. This is Critical objection C1.

2. **TRM (arXiv:2510.04871)** — sections 4 ("Tiny Recursion Models") and 4.1 ("No fixed-point theorem required"). Verdict: **PARTIALLY MISREPRESENTED**. TRM removes the fixed-point theoretical assumption, but deep supervision still pushes recursion toward a target solution; functionally attractor-seeking. The "unconstrained" framing is misleading. This is Important objection I4.

3. **CoT-Solves-Serial (arXiv:2402.12875)** — section 1 (Introduction). Verdict: **OVER-EXTRAPOLATED**. The result is for *Transformers with autoregressive token emission* (T = number of emitted tokens scaling with sequence length), not for any-substrate constant-K depth iteration without emission. K=6 forward iterations of a TC⁰ block is still in TC⁰. This is Critical objection C2.

4. **Illusion of State (arXiv:2404.08819)** — section 5 ("Extending the Expressive Power of SSMs"). Verdict: **FAITHFUL but MISLEADING**. The paper does establish diagonal SSMs are in TC⁰. But the §5 escapes it endorses are nonlinearity (RNN-SSM) and input-dependent transition matrices (IDS4) — NOT external recursion. The hypothesis's mechanism story is not endorsed by this paper.

5. **RWKV-7 (arXiv:2503.14456)** — paper details / abstract. Verdict: **FAITHFUL**. RWKV-7 does claim TC⁰ escape. F3 is a real risk.

6. **Retrofitted Recurrence (arXiv:2511.07384)** — section 1 (Introduction). Verdict: **MISSING from hypothesis**. This is concurrent published work (Nov 2025) that retrofits depth-recurrence onto pretrained Transformer LMs. Highly adjacent to H2's claim. The hypothesis does NOT engage with it. (See Suggestion S1.)

### Verdict-severity consistency check

Verdict: REJECT (revision-1)
Critical objections: 4 (C1, C2, C3, C4)
Important objections: 5 (I1, I2, I3, I4, I5)
Suggestions: 4 (S1, S2, S3, S4)

**Consistency:** A REJECT-revision-1 with 4 Critical objections is consistent. APPROVE would require all Criticals resolved; KILL would require the gap to collapse OR no falsification criteria to be salvageable. Here the gap survives (so not KILL), but the mechanism story has 2 fundamental misreadings of cited papers that cannot be approved as-is. The falsification suite is salvageable, the architectural plausibility needs work, and the magnitudes need calibration. This is exactly revision-1 territory.

### Process notes

- I did NOT take any cited claim on trust. Every load-bearing citation was checked against paper sections.
- I did surface stronger counter-arguments than the hypothesis acknowledges (steelman §6) — the strongest is "this is a compute-scaling claim with a confused mechanism story."
- I did NOT perform performative skepticism — every objection has a specific citation or quantitative concern attached.
- I would defend this REJECT publicly: the contraction-misreading and the TC⁰-K-passes issue are both citationally grounded.
