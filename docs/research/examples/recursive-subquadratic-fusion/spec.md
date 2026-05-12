# Recursive Reasoning on Subquadratic-Attention Backbones — Research Spec

**Status:** draft
**Created:** 2026-05-10
**Novelty target:** hypothesis

## Question

Can architectural recursion — a small network applied recursively within a single forward pass to gain depth-of-reasoning, in the style of Tiny Recursive Model (TRM; Jolicoeur-Martineau, arXiv:2510.04871) — be productively layered on top of subquadratic-attention backbones in the style of SubQ (Subquadratic, 2026) and the formal subquadratic-attention regime characterized by Gupta, Huang, Saha, Xu, Ye (arXiv:2505.14840), so that a system simultaneously benefits from (a) parameter-efficient deep iteration and (b) sub-quadratic context scaling? Concretely: where on the (reasoning-depth × context-length) plane does this fusion uniquely outperform either parent technique alone for long-context reasoning (multi-document QA, code-repository understanding, long-horizon agent traces) and multi-step math / proof / program synthesis, and where does it become incoherent or fail (e.g., when sparse attention drops the very tokens recursion needs to keep referring back to)?

## Modalities and domain

Primary task families in scope:

- **Long-context reasoning over text/code**
  - Code-repository understanding (whole-repo Q&A, cross-file refactoring, bug localization). Reference benchmarks: SWE-bench / SWE-bench-Verified, RepoBench, Long Code Arena, LiveCodeBench long-context splits.
  - Multi-document QA over heterogeneous corpora. Reference benchmarks: ∞Bench, LongBench v2, HELMET, RULER, NoCha, Loong, NIAH variants beyond simple needle retrieval.
  - Long-horizon agent traces (open-ended tool-use trajectories where the model must reference earlier tool outputs many turns later). Reference: AgentBench long-horizon splits, τ-bench, ScienceAgentBench, BrowseComp-Long.
- **Multi-step math / proof / program synthesis**
  - Olympiad-style math with extended scratchpads. Reference: PutnamBench, miniF2F, MathArena, FrontierMath, OlympiadBench.
  - Formal proof assistants where the proof term is long. Reference: Lean / Isabelle / Coq corpora released through LeanDojo, Mathlib4, ProofNet.
  - Program synthesis with rich specification context. Reference: BigCodeBench, APPS hard-split, CRUXEval, LiveCodeBench.

Short-context puzzle benchmarks (ARC-AGI, Sudoku, mazes) appear only as ablation/sanity reference to TRM's reported regime, not as primary targets — we want tasks where both depth-of-reasoning and long working memory are first-class.

The architectural primitives in scope:

- **Recursion lineage:** TRM (arXiv:2510.04871), HRM (Wang et al., 2025) and its earlier hierarchical variants, Universal Transformer (Dehghani et al., 2018) as classic prior art, looped/iterative transformers (Giannou et al., 2023; Yang et al., 2024), implicit-depth / DEQ models (Bai et al., 2019).
- **Subquadratic-attention lineage:** SubQ (Subquadratic blog, 2026), the formal regime in arXiv:2505.14840, sparse-attention families (Longformer, BigBird, Reformer, StreamingLLM, native-sparse-attention work from 2024–2026), low-rank / kernel approximations (Performer, Linformer) only as relevant prior art, hierarchical attention (e.g., HEDGE / hierarchical sparse routing).
- **State-space and linear-attention adjacents** (Mamba, Mamba-2, RWKV, RetNet, Hyena, Based, GLA) — cited only when they materially inform the choice of subquadratic backbone or interact with recursion in a way relevant to the hypotheses.

## Constraints

- Open datasets and open-weights models only — anything cited in eval designs must have an arxiv ID, HF dataset ID, or a public repository with a commit SHA.
- Every claim in any worker output cites a paper, dataset, or repository. Industrial systems whose research papers are not yet public (notably SubQ) may be cited via official blog posts and product pages, but the citation must explicitly note "industrial blog, no peer-reviewed paper."
- Recency window: 2023–present for fusion-architecture and long-context-reasoning literature. Foundational classics older than 2023 may be cited where they directly inform the lineage (e.g., Universal Transformer, Linformer, Performer, original Transformer for the quadratic baseline).
- No GPU spend in this run. Eval-designer may **design** real multi-GPU training runs (including frontier-scale settings up to a 100M–7B-parameter subquadratic-recursive backbone on long-context math/code benchmarks) and must include explicit budget estimates (GPU-hours, dataset preparation cost, wall-clock projection), but the swarm does not execute them.
- No deadline; single swarm run in standard mode.
- No proprietary corpora. License of every cited dataset stated in eval designs.

## Success criteria

A run is successful if **all** of the following hold:

- **Hypothesis count.** At least three (3) hypotheses survive the red-team critique loop. Each surviving hypothesis explicitly states:
  - The claimed gap (with citations to the prior art that does **not** address it).
  - The proposed mechanism (why architectural recursion + subquadratic attention should interact in the predicted way).
  - The predicted outcome (a quantitative or qualitative prediction precise enough to be wrong).
  - **Falsification criteria** — the specific empirical observation that would refute the hypothesis. A hypothesis whose falsification criterion is "the experiment fails" is not acceptable; it must name the metric, the threshold, and the direction.
- **Eval design depth.** For each surviving hypothesis the eval-designer produces a protocol containing:
  - At least one primary dataset and one out-of-distribution dataset.
  - At least two baselines (one parent-architecture-only baseline — e.g., a pure subquadratic backbone without recursion or a pure recursive model without subquadratic attention — and one strong frontier baseline reasonable for the task).
  - Primary metrics, secondary metrics, and at least one diagnostic ablation.
  - A pre-registered statistical-analysis plan with the test, the alpha level, the effect size of interest, and a power consideration.
  - A budget estimate (GPU-hours, dataset prep cost, wall-clock projection).
- **Audit trail.** Every hypothesis killed during red-team is preserved in the final synthesist document with the red-team verdict and a one-sentence reason. The reader must be able to see what was considered and why it was dropped.
- **Synthesist document.** Markdown, 6–10 pages including references, with these sections present:
  1. Problem framing and the (depth × context) plane.
  2. Surviving hypotheses with falsification criteria.
  3. Per-hypothesis eval designs.
  4. Killed-hypothesis audit trail.
  5. YAGNI fence reflection — explicit statement of what was **not** addressed and why, mirroring the spec.
- **Citation discipline.** Every claim cites a paper (arxiv ID), a dataset (HF ID / official URL), or a repository (commit SHA). Red-team flags any uncited claim and either kills the host hypothesis or sends it back for revision.

## Out of scope (YAGNI fence)

The following are explicitly **not** addressed by this run. The synthesist's final document must reflect this fence.

- **Training a model.** The swarm produces a research direction with eval protocols. No model is trained, fine-tuned, or evaluated empirically in this run.
- **Hardware-level optimizations.** No CUDA kernel writing, no FlashAttention internals, no rederivation of the subquadratic-attention algorithm. Subquadratic attention is treated as a *primitive*; we reason about its abstract behavior (sparsity pattern, compute scaling, what it can and cannot represent), not its implementation.
- **Unrelated efficiency methods.** Quantization, distillation, mixture-of-experts routing, and speculative decoding are mentioned only when they directly interact with the fusion thesis. No general efficiency survey.
- **General post-Transformer architecture survey.** Mamba/SSM/RWKV/RetNet families are cited where they inform the choice of subquadratic backbone or where their behavior under recursion would change a hypothesis. We do not attempt to compare them comprehensively.
- **AGI / consciousness / general-intelligence claims.** TRM's headline numbers on ARC-AGI are treated as a benchmark result for a small architecture, not as evidence about general intelligence. No commentary on AGI timelines, "reasoning emergence," or related debates.
- **SubQ as a commercial product evaluation.** Public technical claims from Subquadratic's blog are treated as one data point in an architecture lineage, not as a buyer's guide. No comparison of SubQ to other vendors as products.
- **Agent-scaffolded "recursion".** Recursion in this project refers strictly to **architectural recursion within a single forward pass** (a learned operator applied to its own output for K steps before producing a final answer). Tool-using agents that call themselves, recursive prompting strategies, and chain-of-thought-as-recursion are out of scope and the synthesist must distinguish them from the architectural sense.

## Custom workers

None — using the bundled six (literature-scout, gap-finder, hypothesis-smith, red-team, eval-designer, synthesist).

## Decisions locked in

- 2026-05-10 · Novelty target is `hypothesis` · User wants testable hypotheses with falsification criteria, accepting the red-team-loop cost.
- 2026-05-10 · Primary task domains are long-context reasoning (B) AND multi-step math / proof / program synthesis (C) · The fusion claim is most non-trivial where both depth and long working memory matter; short-context puzzle benchmarks are ablation/sanity references only.
- 2026-05-10 · Eval-designer authorized to design (not execute) real multi-GPU training runs with budget estimates · User wants concrete experiments designed at realistic scale.
- 2026-05-10 · Industrial blog citations allowed for systems without published papers (notably SubQ), but must be marked as such · The SubQ technical claims are load-bearing for the subquadratic-attention lineage; refusing the citation would distort the literature map.
- 2026-05-10 · "Recursion" means architectural recursion within forward pass · Distinguished from agent-scaffolded recursive prompting; the synthesist must keep these separate.
