# Consolidated Gaps — Phase 2 Output

Run: 2026-05-10-0729-766039
Phase 2 status: COMPLETE — 2/2 gap-finders PASS

## Source artifacts

- [gap-finder-1/output.md](gap-finder-1/output.md) — 10 architecture-side gaps + 4 discarded
- [gap-finder-2/output.md](gap-finder-2/output.md) — 9 architecture × evaluation mismatches + 3 discarded

## Selection for Phase 3 (hypothesis-smiths)

Six hypotheses will be forged. Each combines architecture-side gaps (gap-finder-1, prefix `A`) with their corresponding architecture × evaluation mismatch (gap-finder-2, prefix `B`) where natural.

| Hypothesis | Composes gaps | Falsification surface |
|---|---|---|
| **H1 — Sparse fusion lift** | A1 + B1 + B4 | BABILong (k_reasoning × L_haystack); NoCha + Loong (R/R+/R++) under (NSA/MoBA/DSA × K_arch) |
| **H2 — TC⁰ escape via recursion on SSMs** | A2 + A7 + B5 | PutnamBench / miniF2F / CoqGym proof success vs K_arch stratified by tactic-chain length, vs Fixed-Point RNNs |
| **H3 — Retrieval-head re-formation** | A8 + B9 | Retrieval-head retention score under (sparse pattern × K_arch) on NoCha + Beyond-the-Needle |
| **H4 — Joint halting × sparse routing** | A4 | Halting-signal coherence under (NSA/MoBA/DSA × per-token halting) on BABILong / Math-500; degeneracy boundary |
| **H5 — Tunnel-Vision under fusion** | B7 | (Tunnel-Vision lock-in rate, Lost-in-the-Middle U-curve depth, Hyper-multi-step super-linearity) on (sparse pattern × K_arch) |
| **H6 — TRM vs TTT (depthwise vs sequence-time)** | A6 + B8 | CRUXEval program-loop-depth × K_arch diagonal-advantage signature; redundancy/complementarity ablation |

## Remaining gaps (not selected for Phase 3 in this run)

The following gaps are valid but de-prioritized to keep the hypothesis-smith fan-out tractable. They are documented for the synthesist's future-work section.

- **A3** — DEQ-style implicit-fixed-point depth never combined with subquadratic operator inside f.
- **A5** — Looped-Transformers-as-Computer construction's sparsity-ratio break point unmeasured.
- **A9** — DSA's lightning indexer is itself O(L²); recursive amortization of routing across loops untried.
- **A10** — Four-way distinction (recursion refines: SSM state / latent answer / memory tokens / input embeddings) never posed for sub-quadratic substrates.
- **B2** — Michelangelo / LSQ open replication × recursion-depth K under SubQ.
- **B3** — Sparse Frontier method-by-task taxonomy × chain-length-scaling regime.
- **B6** — SWE-bench-Verified / Pro × architectural recursion at edit-LOC quartiles. (Subsumed partially by H1's R++ tests but not deep on long-horizon edits.)

These will be flagged to the synthesist as "explored gaps not pursued in this run" so the audit trail is complete.
