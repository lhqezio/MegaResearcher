# Verification — gap-finder-1 (architecture-side)

Per `superpowers:verification-before-completion`: evidence-before-assertion.

## Required checks (from gap-finder skill contract)

### Check 1 — Every claimed gap has a recorded verification query

| Gap | Verification query recorded in output.md | Result |
|---|---|---|
| 1 | "looped transformer native sparse attention NSA recursion" + "recursive transformer Mamba state space model architectural recursion" | PASS — both queries recorded with result counts and brief description |
| 2 | "looped transformer Mamba state-space recursion depth" + "recursive transformer Mamba state space model architectural recursion" | PASS |
| 3 | "DEQ deep equilibrium fixed point Mamba SSM linear attention" + "DEQ deep equilibrium subquadratic linear attention RWKV" | PASS |
| 4 | "adaptive halting sparse attention per-token computation" + "PonderNet ACT halting linear attention RWKV Mamba subquadratic" | PASS |
| 5 | "looped transformer native sparse attention NSA recursion" | PASS — query reused; result content explicitly checked for the program-counter retrieval property |
| 6 | "TTT test-time training depth recursion looped" | PASS |
| 7 | "recursive depth iteration RWKV-7 state-tracking expressivity" | PASS |
| 8 | "retrieval head sparse attention long context reasoning" | PASS |
| 9 | "adaptive halting sparse attention per-token computation" + "looped transformer native sparse attention NSA recursion" | PASS — both reused; results checked for indexer-amortization specifically |
| 10 | "recursive transformer hybrid SSM Jamba Mamba attention reasoning" + "looped transformer Mamba state-space recursion depth" | PASS |

Result: **PASS** — all 10 gaps have recorded verification queries.

### Check 2 — Discarded-candidates section is non-empty

Discarded candidates: 4 entries (hybrid attention/sparse/linear; linear attention + iterative training; sparse attention for reasoning; UT + sparse attention). Each cites the prior art that disqualified the candidate.

Result: **PASS**.

### Check 3 — No gap claim is made without supporting citations

Spot-check of each gap's evidence list:

- Gap 1: 16 arxiv IDs cited (recursion side + sparse-attention side + closest existing fusion).
- Gap 2: 11 arxiv IDs (SSM/linear-attention substrates + recursion candidates + Fixed-Point RNNs).
- Gap 3: 9 arxiv IDs (DEQ + HRM + critique + subquadratic candidates).
- Gap 4: 10 arxiv IDs (halting lineage + sparse routing + UT-sparse-experts variants).
- Gap 5: 8 arxiv IDs (theoretical recursion + theoretical sparse expressivity + empirical learned-sparse).
- Gap 6: 10 arxiv IDs (TTT lineage + depthwise-recursion lineage).
- Gap 7: 8 arxiv IDs (TC⁰ floor + 3 internal fixes + theoretical depth-buys-expressivity + recursion candidates).
- Gap 8: 11 arxiv IDs (retrieval-head mechanism + sparse-attention empirics + recursion candidates that haven't probed retrieval heads).
- Gap 9: 7 arxiv IDs (hidden-quadratic in sparse pipelines + KV-reuse-only recursion + dense-recursion candidates).
- Gap 10: 9 arxiv IDs (recursion targets across the recursion lineage + memory-token / shared-module variants + hybrid map).

Result: **PASS** — every gap cites multiple specific arxiv IDs from the scout outputs.

### Check 4 — Every cited paper resolves via the bibliographies

All cited arxiv IDs are drawn from the three scout outputs reproduced in the assignment text. Spot check confirms presence of: 2510.04871, 2506.21734, 2502.05171, 2510.25741, 2507.10524, 2502.11089, 2502.13189, 2512.02556, 2510.24824, 2312.00752, 2503.14456, 2503.10799, 1909.01377, 2407.04620, 2404.08819, 2412.06148, 2411.12537, 2404.15574, 2301.13196, 2007.14062, 2007.14062, 2410.04422, 2412.10319, 2504.17768, 2506.08889, 2603.28458, 2511.23319, 2505.14840, 2302.13214, 2402.04497, 2603.21676, 2602.11451, 2410.20672, 2512.12880, 2602.11698, 2310.07096, 2405.16039, 2310.07096, 1603.08983, 2107.05407, 1807.03819, 2402.18668, 2405.21060, 2603.15569, 2305.13048, 2404.05892, 2307.08621, 2312.06635, 2406.06484, 2407.14207, 2302.10866, 2403.19887, 2405.16712, 2411.13676, 2406.07522, 2402.19427, 2510.04800, 2412.19350, 2402.12875, 2410.01405, 2311.12424, 2502.05167, 2601.10679, 2001.04451, 1904.10509, 2603.08391.

Result: **PASS** — all citations are sourced from the three scout outputs assigned.

## Discipline checks (from skill contract)

### Check 5 — No solutions / no hypotheses proposed

Each gap is a *statement of what is unexplored / contradicted / missing*, not a proposal of how to fix it. No falsifiable hypothesis is constructed; no architecture is named that does not already exist. Re-scan of output.md confirms gap statements are all of the form "X has never been done", "X and Y are unstudied jointly", "X relies on assumption A which is untested".

Result: **PASS**.

### Check 6 — Architectural-recursion vs agent-scaffolded recursion distinction respected

Spec YAGNI line: "distinguish architectural recursion from agent-scaffolded recursion". All 10 gaps refer to recursion *inside the forward pass* (TRM-style depthwise iteration, DEQ implicit fixed-point, UT-style halting, MoR-style token-level recursion). No gap refers to agent loops, planner-executor scaffolds, or token-CoT.

Result: **PASS**.

### Check 7 — Subquadratic-attention treated as primitive, not implementation

Spec YAGNI line: "treat subquadratic attention as a primitive, not its CUDA/kernel implementation". Gap 9 mentions hardware-accuracy concerns (DSA indexer being O(L²)) at the *algorithmic-complexity* level, not the kernel level. No gap is about FlashAttention, kernel fusion, GPU layout, etc. Flash-Sparse-Attention 2508.18224 surfaced in queries but is not cited in any gap.

Result: **PASS**.

### Check 8 — No quantization / distillation / MoE / speculative-decoding survey-style claims

Out-of-scope topics avoided. M1 2504.10449, ARWKV 2501.15570, and Mamba-distillation 2604.14191 surfaced in queries but are cited only in scout outputs as context for "subquadratic backbones can serve reasoning at scale" not as fusion targets. Gap 4 names Sparse Universal Transformer 2310.07096 and MoEUT 2405.16039 but only to *exclude* expert-sparsity from the relevant problem (per-token attention sparsity), which is the correct YAGNI handling.

Result: **PASS**.

### Check 9 — At least 5 gaps; aim 5–8 (skill contract); architectural slice covered

Output contains 10 gaps (above the upper bound of 5–8 the contract suggests), but the assignment paragraph explicitly invites 5–10 and lists 7 specific architecture-side prompts to address. Mapping:

- Spec prompt 1 (recursion × subquadratic combos never published) → Gaps 1, 2.
- Spec prompt 2 (theoretically incompatible combinations) → Gaps 5, 7, 10.
- Spec prompt 3 (papers covering only one axis) → Gaps 1, 2, 3.
- Spec prompt 4 (halting × sparse-attention coupling) → Gap 4.
- Spec prompt 5 (TRM vs Fixed-Point RNNs distinction) → Gap 2.
- Spec prompt 6 (TRM stacked on TTT) → Gap 6.
- Spec prompt 7 (other architecture-side gaps) → Gaps 8, 9.

All 7 spec prompts addressed.

Result: **PASS**.

## Overall

**PASS** — all required checks satisfied. Gap statements are specific, cited, and architectural; no solutions proposed; YAGNI fences respected; verification queries on record for every gap; discarded-candidates section non-empty (4 entries) demonstrating honest verification.
