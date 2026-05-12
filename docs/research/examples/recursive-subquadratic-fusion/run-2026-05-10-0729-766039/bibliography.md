# Consolidated Bibliography — Phase 1 Output

Run: 2026-05-10-0729-766039
Phase 1 status: COMPLETE — 5/5 scouts PASS

| Scout | Sub-topic | Output |
|---|---|---|
| scout-1 | Architectural recursion / iterative-depth networks | [output.md](scout-1/output.md) — 32 entries |
| scout-2 | Subquadratic sparse-attention transformers + formal regime | [output.md](scout-2/output.md) — 26 entries (1 industrial blog flagged) |
| scout-3 | State-space and linear-attention subquadratic backbones | [output.md](scout-3/output.md) — 31 entries |
| scout-4 | Long-context reasoning benchmarks + failure modes | [output.md](scout-4/output.md) — 37 entries (21 benchmarks + 16 failure-mode papers) |
| scout-5 | Math / proof / program-synthesis benchmarks | [output.md](scout-5/output.md) — 32 entries |

**Total entries surveyed:** ~158 across all scouts.

Headline cross-scout signals (for gap-finder context):

- **Scout-1 ↔ Scout-2/3 fusion-axis signals.**
  - Parallel Loop Transformer (PLT, arXiv:2510.24824) is the *only* paper found that explicitly couples loop recursion with a subquadratic attention pattern (gated sliding-window). It does not analyze whether dropped tokens are exactly the ones late-loop iterations need.
  - Mixture-of-Recursions (MoR, arXiv:2507.10524) is the closest existing fusion: adaptive recursion + KV reuse, but still dense attention.
  - Huginn (arXiv:2502.05171) and Ouro / LoopLM (arXiv:2510.25741) are the only open pretrained looped LMs >1B scale; both use dense attention.
  - HRM mechanistic critique (arXiv:2601.10679) shows the recursive fixed point is fragile even with dense attention.
  - **No literal product** of TRM-style architectural recursion × SSM/linear-attention backbone exists in the surveyed literature. Closest: TTT (test-time training inner-opt), Zamba (share-and-reuse), M1 (sequence-time CoT on Mamba), RMT (segment-level), Fixed-Point RNNs (mathematical fixed-point recursion on linear RNN).

- **Scout-2 / "what is lost" axis.**
  - The "what is lost / dropped / compressed" annotation varies sharply: NSA preserves a compressed-summary fallback; MoBA does not; DSA (DeepSeek-V3.2, arXiv:2512.02556) drops below top-k entirely.
  - Hidden-quadratic warning: DSA's indexer is itself O(L²) (HISA paper). Many production "sparse attention" pipelines have a quadratic step.
  - The Sparse Frontier (2504.17768) and Retrieval Head (2404.15574) are the most precise existing language for the spec's "where fusion becomes incoherent" question.

- **Scout-3 / state-tracking limits.**
  - Five papers (arXiv:2404.08819, 2412.06148, 2411.12537, 2412.19350, 2410.03810) jointly establish that diagonal-state SSMs (Mamba/RWKV-pre-7/GLA) live in TC⁰ and fail at compositional state tracking.
  - **Fixed-Point RNNs** (arXiv:2503.10799) is the most fusion-thesis-relevant paper found: iterating a diagonal linear RNN K times converges to a dense linear RNN, recovering state-tracking. This is "architectural recursion on a linear-RNN backbone" already, but the recursion is a mathematical fixed-point rather than a TRM-style learned operator. **The fusion thesis must explicitly contrast itself against this.**

- **Scout-4 / failure modes most relevant to fusion.**
  - Lost-in-the-Middle (2307.03172), NoLiMa (2502.05167), Hyper-multi-step (2410.04422), Position Bias under sparse masks (2502.01951), Context Denoising Training (2510.05862), Score Dilution at Test Time (2512.13898).
  - **Tunnel Vision / ParaThinker (arXiv:2509.04475) cuts AGAINST naive recursion** — it documents test-time-thinking failure modes that get worse with naive depth. Hypothesis-smiths must not propose recursion-only solutions to long-context failures without engaging with this paper.
  - License flags requiring spec attention: NoLiMa is non-commercial (Adobe Research License); Michelangelo is closed eval; NoCha redistributes annotations only. LongBench / SWE-bench / SWE-bench-Verified / LiveCodeBench HF cards lack explicit license tags (inherit upstream).

- **Scout-5 / the (chain × context) plane.**
  - **None of the 9 chain-length-scaling papers studies a subquadratic-attention model.** The cleanest open-literature gap for the fusion thesis.
  - The (chain-length × context-length) plane is almost never measured jointly. Closest cases: InftyThink (arXiv:2503.06692), SWE-bench oracle-vs-full-repo, LeanDojo retrieval-K, BigCodeBench `Hard`.
  - License gaps: FrontierMath (gated); MATH-500 / SWE-bench-Verified / LeanDojo HF dataset cards lack explicit license fields (cite upstream repos).
  - PutnamBench-hard / FrontierMath-tier-4 do not publicly tabulate proof-term or CoT length distributions for solved instances; any "chain length >50" threshold needs to be operationalized via a separate length-stratified split.
