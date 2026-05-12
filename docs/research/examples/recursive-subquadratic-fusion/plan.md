# Recursive Reasoning on Subquadratic-Attention Backbones — Research Plan

> **For agentic workers:** This is a MegaResearcher swarm plan. The terminal action is the user running `/research-execute /Users/ggix/research/docs/research/plans/2026-05-10-recursive-subquadratic-fusion-plan.md`. The orchestrator (the `executing-research-plan` skill) reads the *Swarm decomposition* section and dispatches workers in waves. Do not execute this plan as code.

**Goal:** Run a swarm that produces ≥3 falsifiable hypotheses (with eval protocols) on whether architectural recursion (TRM-style) layered on subquadratic-attention backbones (SubQ / Gupta-et-al-style) yields a system that simultaneously gains parameter-efficient depth-of-reasoning and sub-quadratic context scaling on long-context reasoning and multi-step math/proof/program-synthesis tasks.

**Architecture:** Six-phase MegaResearcher swarm. Phase 1 partitions the literature into five non-overlapping scout assignments covering the recursion lineage, sparse-attention subquadratic methods, SSM/linear-attention adjacents, long-context reasoning benchmarks, and math/proof/program-synthesis benchmarks. Phase 2 cross-reads the bibliographies for gaps. Phases 3–5 are orchestrator-computed (one smith per gap, one red-team per smith with up to three revision rounds, one eval-designer per surviving hypothesis). Phase 6 emits a single 6–10-page synthesist document with surviving hypotheses, killed-hypothesis audit trail, and YAGNI fence reflection.

**Tech Stack:** MegaResearcher worker subagents (literature-scout, gap-finder, hypothesis-smith, red-team, eval-designer, synthesist) dispatched via the `research-swarm` orchestrator. Tool access: `hf_papers`, `web_search`, `github_examples`, `hf_inspect_dataset`, `hf_repo_files`, `github_read_file`, `WebFetch`. Output written under `docs/research/runs/<timestamp>/` in this repo.

---

## Context

The approved spec lives at `docs/research/specs/2026-05-10-recursive-subquadratic-fusion-spec.md` and is the source of truth for *why* this run exists, what is in scope, and what success looks like. This plan is the *how*.

The three seed sources the user identified during brainstorming are:

- **TRM** (Tiny Recursive Model, Jolicoeur-Martineau, arXiv:2510.04871) — 7M-parameter recursive network reaches 45% on ARC-AGI-1 with ~1k training examples. Demonstrates that recursive *depth-of-reasoning* via a tiny operator can substitute for parameter scale on hard reasoning puzzles.
- **Gupta, Huang, Saha, Xu, Ye** (arXiv:2505.14840) — characterizes when subquadratic attention is achievable for arbitrary temperature; gives an $\tilde{O}(n^{2-1/d}\,\mathrm{polylog}(B))$ algorithm for constant head dimension and matching SETH-based hardness.
- **SubQ** (Subquadratic blog, 2026) — productized subquadratic LLM with sparse attention claimed 52× faster than FlashAttention, ~1000× attention-compute reduction at frontier scale, 12M-token context.

The fusion thesis is that these two efficiency axes (recursion-for-depth, sub-quadratic-for-length) are *orthogonal* and can be combined. The plan is structured to test that thesis rather than assume it.

## Approach

The highest-leverage decision in any MegaResearcher run is **scout partitioning** (Phase 1). Bad scout coverage → bad gap maps → bad hypotheses.

The partitioning chosen below splits the topic along two dimensions:

- **Architecture vs. evaluation.** Scouts 1–3 cover architectures (recursion lineage, sparse-attention, SSM/linear-attention adjacents). Scouts 4–5 cover the evaluation surface (long-context reasoning benchmarks, math/proof/program-synthesis benchmarks).
- **Recursion vs. subquadratic vs. SSM-adjacent on the architecture side.** This split matters because the fusion thesis is about combining recursion with subquadratic attention; SSMs/linear-attention are a *third* family of subquadratic backbones with very different cache behavior, and a hypothesis-smith may want to argue the fusion works *uniquely* well (or poorly) on one of these families.

Phase 2 has two gap-finders. Gap-finder A looks for architecture-side gaps (which (recursion × subquadratic-backbone) combinations have never been tried, and where prior art only covers one axis). Gap-finder B looks for architecture-evaluation mismatches (where a benchmark exposes a failure mode that the fusion uniquely addresses, or where an existing fusion-style architecture has never been tested on the right benchmark).

Phases 3–6 follow the standard contract.

## Critical files

The orchestrator and synthesist read these:

- `docs/research/specs/2026-05-10-recursive-subquadratic-fusion-spec.md` — source of truth for success criteria, YAGNI fence, citation discipline.
- `docs/research/plans/2026-05-10-recursive-subquadratic-fusion-plan.md` — this file.
- `docs/research/runs/<timestamp>/` — the orchestrator creates this directory at run time. Each worker writes its `output.md`, `manifest.yaml`, `verification.md` to a worker-specific subdirectory.

---

## Swarm decomposition

### Phase 1 — literature-scout dispatches

Five scouts, non-overlapping. All scouts may use `hf_papers`, `web_search`, `github_examples`; all must produce annotated bibliographies with arxiv IDs, HF dataset/model IDs, and repo commit SHAs where available.

**Scout 1 — Architectural recursion / iterative-depth networks.**
Sub-topic: networks that gain effective reasoning depth by applying a learned operator recursively within a single forward pass. Focus list (non-exhaustive): TRM (arXiv:2510.04871) and its precursor HRM (Wang et al. 2025); Universal Transformer (Dehghani et al. 2018) and follow-ups; looped/iterative transformers (Giannou et al. 2023; Yang et al. 2024 latent-reasoning loops); implicit-depth / Deep Equilibrium Models (Bai et al. 2019) and any 2024–2026 transformer-DEQ hybrids; ALBERT-style shared-weight stacking only as relevant prior art; halting / adaptive computation (Graves 2016, PonderNet, and 2024+ continuations); CoCoNUT-style continuous-thought / latent-reasoning architectures. Recency: 2023–present primary, classics older where directly cited. Constraint: distinguish architectural recursion (a learned operator applied to its own output K times within forward pass) from agent-scaffolded recursive prompting — the latter is out of scope per the spec.

**Scout 2 — Subquadratic sparse-attention transformers and the formal subquadratic regime.**
Sub-topic: transformer attention algorithms with provably sub-quadratic compute in sequence length. Focus list: SubQ (Subquadratic blog, 2026) — note explicitly as industrial blog, no peer-reviewed paper; Gupta-Huang-Saha-Xu-Ye (arXiv:2505.14840) and the surrounding hardness literature (Alman & Song lines on hardness of attention); native-sparse-attention 2024–2026 work (DeepSeek's NSA, MoBA, sparse-attention training-time methods); classic sparse attention (Longformer, BigBird, Reformer) as prior art; StreamingLLM / sliding-window inference; hierarchical / routing sparsity; FlashAttention only mentioned where it interacts with sparsity (we are not surveying FlashAttention internals). Recency: 2023–present primary. Constraint: the spec treats subquadratic attention as a *primitive* — capture what each method drops/compresses and what its compute scaling actually is, not its CUDA-level implementation.

**Scout 3 — State-space and linear-attention subquadratic backbones.**
Sub-topic: non-attention or linear-attention architectures that achieve sub-quadratic context scaling. Focus list: Mamba (Gu & Dao 2023) and Mamba-2; RWKV-4/5/6/7; RetNet; Hyena and the Hyena-DNA / StripedHyena lineage; Based; GLA (Gated Linear Attention); 2024–2026 hybrid architectures (Jamba, Zamba, Hymba, Samba) that mix SSM with attention. For each, capture: (a) how the model maintains state over long context, (b) what is known about its reasoning-depth limitations (e.g., the State-Tracking Limits of SSM line of work), (c) any prior attempts to combine SSMs with iterative or recursive heads. Recency: 2023–present.

**Scout 4 — Long-context reasoning benchmarks and documented failure modes.**
Sub-topic: benchmarks designed to exercise reasoning over long context, and the empirical failure modes documented for current long-context models. Focus list: RULER (Hsieh et al. 2024) and the needle-variants beyond simple NIAH; ∞Bench; LongBench / LongBench v2; HELMET; NoCha (novel-challenge long-context); BABILong; Loong; BrowseComp / BrowseComp-Long; SWE-bench and SWE-bench-Verified; RepoBench; Long Code Arena; LiveCodeBench long-context splits; agent-trace benchmarks (τ-bench, ScienceAgentBench, AgentBench long-horizon splits). Capture documented failure modes: lost-in-the-middle, attention dilution, retrieval-vs-reasoning trade-off, position-bias, context-window-vs-effective-context gaps. The output should make it possible for a hypothesis-smith to claim "fusion would specifically address failure mode X in benchmark Y."

**Scout 5 — Math / formal proof / program synthesis benchmarks with long reasoning chains.**
Sub-topic: benchmarks where reasoning-chain depth is the binding constraint, and where long context is also typically required (proof state, library context, repository context). Focus list: PutnamBench; miniF2F; MathArena; FrontierMath; OlympiadBench; LeanDojo and Mathlib4-based evaluation; ProofNet; CoqGym; BigCodeBench; APPS hard split; CRUXEval; LiveCodeBench reasoning splits; HumanEval-Pro / EvalPlus where reasoning depth is the bottleneck. For each capture: typical chain length / proof-term length, whether long-context model performance degrades with chain length, license. The output should let a hypothesis-smith ground a falsifiable prediction about chain-length × context-length scaling.

### Phase 2 — gap-finder dispatches

Two gap-finders. Each must produce a list of gaps (not hypotheses); each gap must cite specific scout outputs.

**Gap-finder A — Architecture-side gaps.**
Slice: scout-1 + scout-2 + scout-3 outputs.
Focus: unexplored intersections in the (recursion × subquadratic-backbone) plane. Specifically:
- Which combinations have *never* been published? (e.g., TRM-style recursion on Mamba; recursive operator on native-sparse-attention backbone; DEQ-style implicit depth on RWKV).
- Which subquadratic backbones are theoretically incompatible with recursion in non-obvious ways (e.g., does a sparse-attention pattern that drops mid-context tokens make recursive refinement of those tokens impossible? does an SSM's compressed state lose the very information recursion would refine?).
- Where prior art covers only one axis: papers that propose recursion only on dense quadratic attention, or papers that propose subquadratic attention but only with single-pass decoding.
- Halting / adaptive-computation behavior under sub-quadratic attention — is the halting signal itself well-defined when attention is sparse?

**Gap-finder B — Architecture × evaluation mismatches.**
Slice: scout-4 + scout-5 outputs, cross-referenced with scout-1 + scout-2 + scout-3.
Focus:
- Long-context benchmarks where current models fail in a way recursion would specifically address (e.g., chain-of-thought-over-long-context tasks where models degrade in proof-search depth as context length grows).
- Math/proof benchmarks where chain length and context length both bind — and where SSM/linear-attention models are known to struggle with state tracking, suggesting recursion-on-SSM might either rescue or further break the architecture.
- Benchmarks that distinguish "needle retrieval" from "reasoning over the needle" — the fusion thesis predicts asymmetric behavior on these.
- Specific empirical claims in scouts 1–3 that have *never* been tested on the benchmarks in scouts 4–5 (e.g., has TRM ever been evaluated on a long-context math benchmark? has any recursive transformer been tested on SWE-bench-Verified?).

### Phase 3 — hypothesis-smith dispatches

Computed dynamically by the orchestrator: one smith per gap surfaced in Phase 2. Each smith must produce a hypothesis containing claimed gap, mechanism, predicted outcome, and **falsification criteria with metric, threshold, and direction** (per spec — "the experiment fails" is not acceptable). Each claim must be cited.

Project-specific guidance for smiths:

- The fusion thesis is asymmetric: recursion adds depth, subquadratic attention adds width. A good hypothesis predicts a *non-additive* interaction (synergy or interference), not just a sum of two known effects. "Both work" is not novel.
- Distinguish wall-clock compute reduction from compute-per-token reduction; many subquadratic methods recover dense behavior asymptotically and a fusion claim must be specific about which regime it advantages.
- A hypothesis that depends on the recursion accessing tokens the sparse pattern dropped is structurally incoherent — flag this case explicitly if it arises and either reformulate or kill.

### Phase 4 — red-team critique loop

Computed dynamically: one red-team per hypothesis, with up to three revision rounds (revised hypothesis fires a fresh red-team; same number of rounds gates).

Project-specific critique focus:

- **Citation discipline (load-bearing).** Every claim must cite an arxiv ID, HF ID, or repo commit SHA. Industrial-blog citations (notably for SubQ) are allowed but must be flagged as such. Uncited claims invalidate the hypothesis.
- **Falsifiability.** The falsification criterion must name the metric, the threshold, and the direction. "Performance does not improve" is unacceptable; "accuracy on PutnamBench-easy with K=8 recursion steps does not exceed the K=1 baseline by ≥3 percentage points at p<0.05" is acceptable.
- **Architectural coherence.** Reject hypotheses that rely on the recursion attending to tokens the chosen sparse pattern excludes, unless the hypothesis explicitly addresses how those tokens are recovered.
- **Distinguishing recursion senses.** Reject any hypothesis that conflates architectural recursion with chain-of-thought, agent loops, or test-time-scaling-via-sampling. The spec is explicit on this.
- **Scale plausibility.** Reject hypotheses whose mechanism only manifests at frontier scale (≥70B parameters) without justifying why a smaller-scale eval would still falsify them. Eval-designer cannot run frontier-scale experiments; the hypothesis must be falsifiable at a budget the eval-designer will respect.

### Phase 5 — eval-designer dispatches

Computed dynamically: one designer per surviving hypothesis. Each protocol must include: primary dataset (in-distribution), at least one OOD dataset, two baselines (one parent-architecture-only, one strong frontier baseline), primary + secondary metrics, ≥1 diagnostic ablation, pre-registered statistical analysis (test, alpha, effect size of interest, power consideration), and a budget estimate (GPU-hours, dataset prep, wall-clock).

Project-specific budget guidance:

- The spec authorizes designs at frontier scale (up to ~7B-parameter subquadratic-recursive backbones) with explicit budget estimates. Do not self-censor into toy experiments.
- Designs whose minimum falsifying experiment exceeds **2,000 GPU-hours** must include a *cheaper falsification path* — a smaller-scale ablation that would still kill the hypothesis if it failed. The synthesist will use these cheaper paths in the final document.
- Each design must state its license for every dataset (CC-BY, MIT, Apache-2.0, custom), per spec.

### Phase 6 — synthesist

Single dispatch. The synthesist reads the spec, the plan, all worker outputs (every literature-scout, gap-finder, every hypothesis-smith including revisions, every red-team verdict, every eval-designer protocol) and produces the run's primary deliverable.

Project-specific synthesis requirements:

1. Document is markdown, **6–10 pages including references** (per spec). Exceeding 10 pages is a failure mode; trim before delivering.
2. Required sections, in order:
   1. Problem framing — recap the (depth × context) plane and where the fusion thesis lives on it.
   2. Surviving hypotheses — each with falsification criteria stated verbatim.
   3. Per-hypothesis eval designs — including the cheaper-falsification path where applicable.
   4. Killed-hypothesis audit trail — every rejected hypothesis preserved with the red-team verdict and a one-sentence reason.
   5. YAGNI fence reflection — explicit statement of what was *not* addressed, mirroring the spec's out-of-scope list and confirming each item was respected.
3. Every claim cites its source (arxiv ID / HF ID / commit SHA / industrial blog with flag).
4. The recursion/agent-recursion distinction must be reasserted in the synthesis — this is a known confusion point.

### Custom worker dispatches

None — the spec did not define custom workers.

### Parallelism budget

`MEGARESEARCHER_MAX_PARALLEL = 4`

Phase 1 has 5 scouts → runs as 4-then-1 (or whatever the orchestrator's scheduler chooses). Phase 2 has 2 gap-finders → runs in parallel. Phases 3–5 are dynamic and parallelism is bounded by the same budget.

### Estimated total runtime + token budget

Honest estimate, assumptions stated:

| Phase | Workers | Avg tokens / worker | Subtotal |
|---|---|---|---|
| 1 (scouts) | 5 | 35,000 | 175,000 |
| 2 (gap-finders) | 2 | 30,000 | 60,000 |
| 3 (smiths) | ~5 (one per gap) | 25,000 | 125,000 |
| 4 (red-team incl. revisions) | ~10 (avg ~2 rounds per hypothesis) | 25,000 | 250,000 |
| 5 (eval-designers) | ~3–4 (survivors only) | 50,000 | 175,000 |
| 6 (synthesist) | 1 | 80,000 | 80,000 |
| Orchestrator overhead | — | — | ~75,000 |
| **Total** | **~26 worker dispatches** | | **~940,000 tokens** |

Round up to **~1.0M tokens** for budgeting headroom. This is for a single full run.

Wall-clock estimate, assuming the swarm runs at MAX_PARALLEL=4 and average worker latency ~7 min:

- Phase 1: 5 scouts in 2 batches → ~16 min
- Phase 2: 2 gap-finders parallel → ~7 min
- Phase 3: ~5 smiths in 2 batches → ~14 min
- Phase 4: red-team revisions are *serial within a hypothesis* (smith → red-team → smith → red-team), parallel across hypotheses → ~25 min
- Phase 5: ~3 eval-designers parallel → ~10 min
- Phase 6: synthesist → ~10 min
- **Total wall-clock: ~80–100 min** for a typical run.

Worst-case wall-clock (red-team rejects every hypothesis at every round, all hypotheses use full 3 revision rounds): ~140 min.

---

## Verification

Before reporting the run complete, the `research-verification` skill (wrapping `superpowers:verification-before-completion`) must confirm:

- [ ] Every surviving hypothesis has falsification criteria with metric, threshold, and direction (not "the experiment fails").
- [ ] Every claim in worker outputs cites an arxiv ID, HF ID, repo commit SHA, or flagged industrial blog.
- [ ] Killed-hypothesis audit trail is non-empty (if Phase 4 produced no kills it should be obvious in the manifest; the synthesist must still acknowledge the absence).
- [ ] Synthesist document is between 6 and 10 pages including references.
- [ ] Synthesist document includes all six required sections in order.
- [ ] YAGNI fence reflection is present and addresses each spec out-of-scope item by name.
- [ ] Architectural recursion vs agent-scaffolded recursion distinction is explicitly maintained in the final document.
- [ ] Eval designs whose minimum falsifying experiment exceeds 2,000 GPU-hours include a cheaper-falsification path.

---

## Self-review notes

Spec coverage scan (run inline, no gaps found):

- Spec § Question → addressed by all six phases; Phase 2 explicitly maps the (depth × context) plane.
- Spec § Modalities → scouts 4 and 5 are partitioned along this axis (long-context reasoning vs math/proof/program-synthesis).
- Spec § Constraints (open data, citation discipline, no-GPU-spend, frontier-scale eval design allowed) → enforced in red-team focus and eval-designer guidance.
- Spec § Success criteria (≥3 surviving hypotheses with falsification, eval protocols, audit trail, 6–10 page synthesis) → encoded in Phases 4, 5, 6 and the verification checklist.
- Spec § YAGNI fence (no training, no kernel work, no MoE/distillation/quantization survey, no AGI claims, no SubQ commercial eval, no agent-recursion conflation) → enforced as critique focus in Phase 4 and required reflection in Phase 6.
- Spec § Custom workers → none required.

Type/term consistency: "architectural recursion within a single forward pass" used consistently; "subquadratic-attention backbone" used consistently; "fusion thesis" used as the umbrella term consistently; "recursion vs agent-scaffolded recursion" distinction reasserted in red-team focus and synthesist requirements.

Placeholder scan: no `<...>`, "TBD", "TODO", "implement later", or vague-handling-language remains.
