# Verification — Red-Team-4 on H4

## Verification-before-completion checklist

### (1) Independent literature queries

Three independent literature searches were run beyond the gap-finder's two:

- **Q1**: `PonderNet ACT halting NSA sparse attention selection joint training` → 10 results. No direct match for joint per-token depth halting + per-token NSA-style block routing. Returned NSA itself, attention-sparsification variants (BLASST, NOSA, Twilight, SpargeAttn).
- **Q2**: `looped recursive transformer NSA block sparse multi-hop reasoning` → 10 results. Surfaced two near-neighbors the gap-finder missed: ReSSFormer (2510.01585, recursive + sparse + multi-hop QA, fixed K) and Adaptive Loops and Memory in Transformers (2603.08391, adaptive looping + halting on dense attention). Neither subsumes Gap 4 but both should be cited.
- **Q3**: `joint depth halting attention sparsity per-token reasoning early exit` → 10 results. All decoding-time early-exits or PonderNet-on-dense (PALBERT). None instantiate joint training of depth halting with attention sparsity.
- **Q4** (`coverage signal early exit halting attention sparsity routing recursion`) → 10 results, surfaced Informed Routing 2510.13831 (different problem: execute-or-approximate per token, not depth × sparsity coupling).
- **Q5** (`adaptive computation time token selection block sparse coverage`) → 10 results, all attention-sparsification or vision-token-pruning. Not relevant.

Net: gap_claim_survives = true; smith should add ReSSFormer + Adaptive Loops as prior art.

### (2) Citation spot-checks (≥ 3 required, did 6)

| # | Citation | Method | Result | Severity |
|---|---|---|---|---|
| 1 | NSA 2502.11089 §3.3.2 (importance scores) | `read_paper` §3 + §3.3 | Confirmed: p_t^slc derived from compressed-branch softmax. c_{t,k} is computable without extra forward passes. **But the score is what NSA used to select, so c_{t,k} → 1 by construction unless NSA is miscalibrated.** | Critical |
| 2 | TRM 2510.04871 §3.2 + Figure 3 pseudocode | `read_paper` §3 | TRM uses ACT Q-learning halting (`q_hat = Q_head(y)`, BCE loss against `(y_hat == y_true)`), NOT PonderNet λ_n. **Smith conflates two recipes.** | Important |
| 3 | PonderNet 2107.05407 §2.2 | `read_paper` §2 | λ_n is per-sequence in original; smith uses "PonderNet" to mean "per-token PonderNet-style." Terminology, not load-bearing. | Important |
| 4 | ParaThinker / Tunnel Vision 2509.04475 | `paper_details` | Tunnel Vision is about sequential CoT generation hitting a ceiling, NOT about depthwise recursion. **Smith stretches citation.** | Suggestion |
| 5 | MoR 2507.10524 §3 | `read_paper` §3 | MoR uses expert-/token-choice routing for adaptive depth, not PonderNet. Smith's analogy framing is fair. | OK |
| 6 | LoopFormer 2602.11451, HRM critique 2601.10679, Retrieval Head 2404.15574, SCBench 2412.10319 | `paper_details` | Existence and rough framing verified for all four. SCBench's specific 2-6 EM swing magnitude not verified to §-level. | OK / Suggestion |

### (3) Severity-verdict consistency

Verdict: `REJECT (revision-1)`. Critical count: 3. Important count: 7. Suggestion count: 3.

REJECT (revision-1) is consistent with 3 Criticals + 7 Importants — the hypothesis has serious structural issues (self-justification of c_{t,k}, falsification thresholds at noise floor, recipe conflation) but the gap is real and the mechanism core is grounded. KILL would require either gap-collapse (rejected by Q1-Q3) or unrecoverable mechanism failure (the smith can plausibly reframe the c_{t,k} claim as a feature/regularization signal rather than an interaction signal).

APPROVE would be inconsistent with 3 Criticals: an APPROVE with falsification thresholds at the noise floor would amount to signing off on a structurally unfalsifiable hypothesis.

### (4) Steelman performed

Yes — see §6 of output.md. The steelman: NSA's three-branch architecture already encodes coverage info via the compressed and sliding-window branches; explicit conditioning is solving a non-existent problem; R2 ≈ R1 ± noise. The smith's Risk A partly acknowledges this, but does not commit to a discriminative experiment (gradient-stopped c_{t,k} ablation, recommended in I1).

### (5) Verification artifacts

- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-4/output.md` — full critique (this is the deliverable for the orchestrator).
- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-4/manifest.yaml` — machine-readable summary.
- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-4/verification.md` — this file.

All three files written; verdict line `REJECT (revision-1)` present at end of output.md.
