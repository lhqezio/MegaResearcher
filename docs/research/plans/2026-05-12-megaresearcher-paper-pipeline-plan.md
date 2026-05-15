# Augmenting MegaResearcher into an End-to-End Paper Pipeline — Research Plan

**Status:** draft
**Created:** 2026-05-12
**Spec:** `docs/research/specs/2026-05-12-megaresearcher-paper-pipeline-spec.md`
**Run mode:** MegaResearcher swarm, novelty target = `hypothesis`

## Context

This plan executes the hypothesis-target swarm defined in the spec. The output is a research punch list — surviving hypotheses, falsification criteria, and eval designs — describing which architectural changes would close the gap between MegaResearcher and a system capable of producing main-track-conference-grade ML papers.

The plan does not implement the augmentation. It does not run the eval designs. It produces the document that downstream implementation work will draw on.

Project guardrails to read alongside this plan:
- `CLAUDE.md` (project-level) — MegaResearcher's discipline rules, the no-nested-dispatch constraint, citation discipline.
- The spec — the YAGNI fence and success criteria that gate verification.

## Approach

Run a full 6-phase swarm. Novelty target = `hypothesis` means all phases fire: literature-scout → gap-finder → hypothesis-smith → red-team (with revision loop) → eval-designer → synthesist.

The decisive design choices:

1. **Phase 1 partition is by *capability axis*, not by *system*.** A scout-per-system partition would produce six redundant summaries of overlapping pipelines; a capability-axis partition surfaces what each system does *well* and *badly* on a fixed set of dimensions (manuscript drafting, peer review, experiment execution, memory, multi-agent critique, long-context coherence). This is the partition the gap-finder needs to do its job.

2. **Phase 2 runs three gap-finders along distinct cuts:** capability gaps (which dimensions are systematically thin across all systems), architectural-pattern gaps (which orchestration patterns from generic multi-agent work have not been applied to paper generation), and feasibility-filtered gaps (which subset can be closed inside MegaResearcher's constraints — single-session orchestrator, no nested dispatch, file-based artifact passing).

3. **Red-team focus is on "hypothesis vs. existing-system" differentiation.** "Add a reviewer worker" is not a hypothesis when AI-Scientist already has one — the hypothesis must specify the *differential* mechanism and predicted differential outcome. This is the most common failure mode for this topic and red-team is told to attack it specifically.

4. **Eval-designer constraint: no venue submissions, no GPU-training, ≤$200 API spend per replication.** Designs that bust the budget are flagged for the user, not silently included.

## Critical files

- **Spec:** `docs/research/specs/2026-05-12-megaresearcher-paper-pipeline-spec.md`
- **Run directory (created by orchestrator):** `docs/research/runs/2026-05-12-<HHMM>-<short-hash>/`
- **Final deliverable:** `docs/research/runs/<run-id>/output.md` (synthesist's position-paper-style document)
- **Verification report:** `docs/research/runs/<run-id>/verification-report.md`
- **Plugin agents reference:** `agents/literature-scout.md`, `agents/gap-finder.md`, `agents/hypothesis-smith.md`, `agents/red-team.md`, `agents/eval-designer.md`, `agents/synthesist.md`
- **MCP tool surface:** `mcp/ml-intern/` — `hf_papers`, `web_search`, `github_examples`, `hf_repo_files`, `hf_inspect_dataset`, `hf_docs_explore`, `hf_docs_fetch`, `github_read_file`, `github_list_repos`.

## Swarm decomposition

### Phase 1 — literature-scout dispatches

Six scouts, partitioned by capability axis. Each scout produces an annotated bibliography (≥6 citations, focus on 2023–2026 unless tracing the origin of a pattern requires an older anchor paper) per the worker contract.

**scout-1 — End-to-end autonomous-research systems**
Sub-topic: full pipelines that take a topic and produce a paper-shaped output. AI-Researcher (arXiv:2505.18705), AI Scientist v1 (Sakana, 2024) and v2 (2025), Agent Laboratory (Schmidgall et al.), ResearchAgent, Virtual Lab (Stanford), Coscientist (Boiko et al.), Aviary, Curie, Genesis-Flow. Anything else in the same family.
Focus: 2023–2026. For each system extract — the agent roster, the loops present (draft-review, propose-experiment-analyze, critique-revise), the artifact format passed between agents, the claimed evaluation, and the explicitly-stated limitations / future work. This scout is the spine of the whole run.
Tools: `hf_papers` (queries: "autonomous research agent", "AI scientist paper generation", "multi-agent scientific discovery LLM", "automated research pipeline"), `github_examples` for the corresponding code repos, `web_search` for project pages and follow-up blog posts.

**scout-2 — Manuscript drafting and document-scale coherence**
Sub-topic: how LLM systems write long, structured, multi-section documents. Hierarchical drafting (outline → section → paragraph), abstract-conclusion-results consistency checking, cross-section reference resolution, citation-anchored writing, retrieval-augmented manuscript generation, document-scale RLHF.
Focus: 2023–2026. Both research-paper-specific and general long-document work that transfers. Include AI-Researcher's three-phase documentation approach, AI-Scientist's writeup module, Storm (Stanford), Wikipedia-style automated synthesis, and any document-scale coherence benchmarks.
Tools: `hf_papers` (queries: "long document generation LLM coherence", "hierarchical document generation", "automated scientific writing", "outline-to-paper generation"), `github_examples` for any reference implementations.

**scout-3 — Automated peer review and paper-quality evaluation**
Sub-topic: LLM-as-reviewer for scientific papers. ICLR-bench / OpenReview-trained reviewers, AI Scientist's review module, rubric-based critique, LLM judge agreement with human reviewers, presentation-vs-content failure modes, peer-review reform proposals using LLMs.
Focus: 2023–2026. Include both the systems that *use* an LLM reviewer in a pipeline and the standalone benchmarks (ReviewerBench, Scientist-Bench review track, etc.). Surface the known overweighting of presentation by LLM judges — this is the paper-evaluation crisis point.
Tools: `hf_papers` (queries: "LLM peer review automated", "AI reviewer scientific paper", "reviewer agent ICLR", "paper evaluation benchmark"), `web_search` for ARR / OpenReview tooling, `github_examples` for review-agent implementations.

**scout-4 — Experiment execution and verification in agent systems**
Sub-topic: agents that actually run code, execute experiments, and verify results — not just propose them. SWE-agent, OpenHands, Claude Code itself, AutoML agents, MLE-Bench, Vercel Sandbox / Modal / E2B / Daytona for sandboxed execution, reproducibility checks, citation-claim verification.
Focus: 2023–2026. The hypothesis-smith will use this to ground proposals for an "experimentalist" worker. Emphasis on what fails — long-horizon coherence loss, premature task completion, hallucinated success.
Tools: `hf_papers` (queries: "code agent benchmark", "MLE bench autonomous", "scientific experiment agent", "reproducibility LLM"), `github_examples` for the agent frameworks, `web_search` for sandboxed-execution platforms used in agent research.

**scout-5 — Multi-agent critique, debate, and revision loops**
Sub-topic: generic patterns for multi-agent self-improvement applied to text — debate, reflection, constitutional AI, multi-agent finetuning, self-rewarding, tree-of-thought-style revision, adversarial collaboration. NOT paper-generation specific; this scout surfaces patterns that *could* be applied.
Focus: 2023–2026. Look for empirical results — what magnitude of improvement does adding a debate/reflection/revision loop produce on which tasks. The hypothesis-smith uses this to forecast the differential effect of adding revision loops to MegaResearcher.
Tools: `hf_papers` (queries: "multi agent debate LLM", "self refine reflection", "constitutional AI", "multi-agent improvement"), `github_examples` for reference implementations.

**scout-6 — Memory and state management for long agent workflows**
Sub-topic: how long-running agent systems manage memory beyond the context window. Hierarchical memory (MemGPT, A-MEM), structured-artifact passing (the MegaResearcher pattern), KV-cache management, external scratchpads, vector memory, episodic memory, work-product handoff between agents.
Focus: 2023–2026. The constraint set in the spec is "no nested dispatch, file-based artifact passing" — this scout surfaces alternatives within those constraints, and surfaces what AI-Researcher specifically flagged as a gap (memory degrading to abstract summaries).
Tools: `hf_papers` (queries: "agent memory long horizon", "MemGPT hierarchical memory", "structured memory LLM agent", "artifact passing multi agent"), `github_examples` for memory frameworks.

### Phase 2 — gap-finder dispatches

Three gap-finders, non-overlapping focus axes.

**gap-finder-1 — Capability gaps across the genre**
Slice: scout-1, scout-2, scout-3, scout-4 outputs (the paper-generation-specific scouts).
Focus: build a matrix — rows = systems from scout-1 (AI-Researcher, AI Scientist v1, v2, Agent Lab, ResearchAgent, Virtual Lab, etc.), columns = capabilities (manuscript drafting, peer review loop, experiment execution, statistical rigor, related-work map, ablation discipline, citation verification, theoretical reasoning, ICLR-rubric self-evaluation). Cells = "strong / weak / absent" with citation pointer. Output: ranked list of capabilities systematically thin or absent across the genre.

**gap-finder-2 — Architectural-pattern gaps**
Slice: scout-5 (multi-agent patterns), scout-6 (memory), cross-referenced with scout-1 (what paper-gen systems actually do).
Focus: which proven multi-agent orchestration patterns from generic work (debate, reflection, constitutional AI, hierarchical memory, structured artifact passing) have *not* been applied to scientific-paper generation? Each unexplored intersection is a candidate gap. Score by (a) magnitude of effect the pattern shows on adjacent tasks and (b) plausibility of transfer to paper generation.

**gap-finder-3 — Feasibility-filtered gaps**
Slice: outputs of gap-finder-1 and gap-finder-2 (this gap-finder runs in the same wave but reads the others' outputs after they complete; if running strictly in parallel, instead reads all six scout outputs and applies the feasibility filter independently — see Parallelism budget below).
Focus: of the gaps surfaced, which can be closed inside MegaResearcher's hard constraints — single-session orchestrator, no nested dispatch, file-based artifact passing, off-the-shelf model APIs only, no fine-tuning, ≤$200 API spend per replication of the eval? Output a shortlist of feasibility-passing gaps, ranked by expected impact-per-implementation-cost. Gaps that fail feasibility are still listed with the reason — they may matter as future-work flags.

### Phase 3 — hypothesis-smith dispatches

Computed dynamically by the orchestrator from gap-finder-3's feasibility-passing shortlist (one smith per gap on the shortlist). Expected: 5–8 smiths.

Each hypothesis must specify the augmentation (which worker(s), which loop, which artifact format), the mechanism, the predicted differential outcome vs. a named prior-art baseline (not vs. "no system"), pre-registered falsification criteria, and citations grounding the mechanism in prior art.

### Phase 4 — red-team critique loop

Computed dynamically (one red-team per hypothesis, revision loop capped at 3 attempts).

Project-specific critique focus:
- **Differential-effect attack.** "Add a peer-reviewer worker" is not a hypothesis if AI Scientist already has one — the hypothesis must specify what the proposed worker does *differently* and why that difference produces a measurable differential outcome. Red-team attacks every hypothesis on this point first.
- **Falsifiability attack.** The falsification criteria must be runnable inside the spec's constraints (no venue submissions, ≤$200/replication). "Improved paper quality" is not falsifiable; "≥0.5-point improvement in mean LLM-judge novelty score on a 20-paper held-out sample under rubric R" is.
- **Implementability attack.** Verify the proposed augmentation actually fits MegaResearcher's architecture (single-session orchestrator, no nested dispatch, file-based artifact passing). Hypotheses requiring nested dispatch, custom runtimes, or fine-tuning fail.
- **LLM-judge-overweighting-presentation attack.** scout-3 will surface this; red-team must apply it. A hypothesis whose only eval is an LLM judge that rewards presentation cannot survive — the hypothesis must triangulate with at least one non-judge signal (reproducibility check, citation-graph fidelity, ablation completeness, statistical-test correctness, blinded human spot-check on a small sample).

### Phase 5 — eval-designer dispatches

Computed dynamically (one designer per surviving hypothesis). Expected: 3–6 designers.

Constraints:
- **No venue submission.** Use LLM-judge panels with disclosed rubrics + at least one non-judge signal per the red-team rule above.
- **No GPU training.** Off-the-shelf model APIs only.
- **API spend ≤ $200 per replication of the eval.** Designs that exceed must be flagged for user approval; do not silently include intractable experiments in the synthesis.
- **Pre-registered decision rules.** Named thresholds, declared before running. Post-hoc thresholds violate the spec.
- **Named baselines.** Every comparison is against a specific prior system (AI-Researcher, AI Scientist v2, etc.) or against a specific ablation of MegaResearcher itself. "Compared to no system" is not a baseline.

### Phase 6 — synthesist

Single dispatch. Inputs: spec + all worker outputs from Phases 1–5 (including killed hypotheses and revision history).

Synthesis-specific requirements:
- **Length:** ≤ 12 pages.
- **Format:** position-paper style — (1) Introduction (the gap and why it matters), (2) Related work (the six-scout digest, organized by capability axis), (3) Gap analysis (the matrices from gap-finder-1 and -2, refined), (4) Proposed architecture (the augmented MegaResearcher with each surviving hypothesis as a labeled subsystem), (5) Hypotheses table (one row per surviving hypothesis: augmentation / mechanism / predicted differential effect / falsification criterion / eval design / named baseline), (6) Eval designs (full per-hypothesis protocols from Phase 5), (7) Threats to validity (LLM-judge overweighting, sample-size limitations on held-out paper sets, etc.), (8) Audit trail (every killed hypothesis with the lesson it contributes — non-negotiable per MegaResearcher discipline), (9) YAGNI fence reflection (explicit mirror of the spec's Out-of-scope list).
- **Citation discipline:** every claim backed by an arXiv ID, HF Papers entry, or DOI. The verification.md artifact must list every cited paper with retrieval status.
- **No new claims.** The synthesist composes from existing worker outputs. Does not introduce hypotheses or evaluations not produced by Phases 3 / 5.

### Custom worker dispatches

None.

### Parallelism budget

`MEGARESEARCHER_MAX_PARALLEL = 4` (default).

Phase 1 (6 scouts) runs in two waves of 3. Phase 2 (3 gap-finders) — note that gap-finder-3 ideally reads gap-finder-1 and -2's outputs; orchestrator should run gap-finder-1 and -2 in one wave (with one slot free), then gap-finder-3 in a second wave after the first two complete. If wall-clock matters more than gap-finder-3 dependency, fall back to all three in parallel and have gap-finder-3 read scout outputs directly. Phase 3 (5–8 smiths) runs in 2–3 waves. Phase 4 fires one red-team per smith output, sequentially per hypothesis (revision loop is intrinsically sequential), but different hypotheses' loops run in parallel up to the budget. Phase 5 (3–6 designers) runs in 1–2 waves. Phase 6 is single-dispatch.

### Estimated total runtime + token budget

| Phase | Workers | ~Tokens / worker | Subtotal |
|---|---|---|---|
| 1 — literature-scout | 6 | 30k | 180k |
| 2 — gap-finder | 3 | 40k | 120k |
| 3 — hypothesis-smith | 6 (est.) | 30k | 180k |
| 4 — red-team (with ~1.5 avg revisions) | 6 × 2.5 calls | 25k | 375k |
| 5 — eval-designer | 4 (est.) | 45k | 180k |
| 6 — synthesist | 1 | 70k | 70k |
| Orchestrator overhead | — | — | 100k |
| **Total (estimate)** | | | **~1.2M** |
| **Total (rounded up for safety)** | | | **~1.6M** |

Wall-clock estimate at parallelism = 4: ~2.5–4 hours total. Phase 1 = two waves × ~15 min. Phase 2 = two waves × ~15 min (per the dependency above). Phase 3 = 2–3 waves × ~10 min. Phase 4 = ~30–60 min (the revision loop dominates; sequential per hypothesis). Phase 5 = ~15–25 min. Phase 6 = ~10–15 min.

Assumptions: average scout produces ~8–12 citations with detailed annotations (this topic has more interlocking systems than a typical literature query); average hypothesis-smith reads ~20 citations across two gap-finder outputs; red-team reads the hypothesis + cited prior art (~15 papers); eval-designer reads the hypothesis + falsification criteria + named baselines; synthesist reads all worker outputs plus the spec.

The Phase-4 line is the largest single bucket and the most variable — if hypotheses are weak, revision counts climb. If average revisions exceed 2.5, escalate to user per the spec discipline.

## Verification

The run is "done" when:

1. **Bibliography count:** ≥ 35 unique citations across the six scout outputs, primarily 2023–2026, all retrievable via `hf_papers` / arXiv / DOI. Verified by spot-checking 5 random citations.
2. **Capability matrix present:** gap-finder-1's output contains a system × capability matrix with ≥ 80% of cells populated, citing supporting evidence.
3. **Feasibility filter applied:** gap-finder-3 names the feasibility constraints from the spec and applies them explicitly; every gap on its shortlist passes the four constraints (no nested dispatch, file-based artifact passing, off-the-shelf APIs, ≤$200/replication).
4. **Hypotheses survive with falsification criteria:** ≥ 3 surviving hypotheses, each with pre-registered falsification criteria using named thresholds and at least one non-judge signal.
5. **Differential-effect attack survived:** every surviving hypothesis names a specific prior system or MegaResearcher ablation as its baseline; none are "compared to no system."
6. **Audit trail:** synthesist names every killed / revised hypothesis with the lesson each contributes. Empty or one-line audit trails fail.
7. **YAGNI fence reflected:** explicit section in synthesist output mirroring spec's Out-of-scope list.
8. **No invented citations:** verification.md confirms every cited paper resolves on arXiv or HF Papers; any that fail to resolve are dropped or replaced.
9. **Synthesist document is in position-paper format** with all nine sections per Phase 6 spec.
10. **Spec success criteria all checked.**

If any of 1–10 fail, re-dispatch the failing worker(s) with a course-correction prompt. Do not paper over verification failures.

## Decisions locked in

- 2026-05-12 · Six scouts partitioned by capability axis (not by system) · A system-per-scout partition would produce overlapping summaries; a capability-axis partition surfaces what each system does well and badly on a fixed set of dimensions — the right input for the gap-finders.
- 2026-05-12 · Three gap-finders along distinct cuts (capability, architectural-pattern, feasibility) · Each addresses a different question; feasibility-filter is downstream of the other two.
- 2026-05-12 · Red-team's first attack is differential-effect · The most common failure mode for this topic is "add a worker that prior art already has." Red-team is told to attack this first.
- 2026-05-12 · Eval-designer budget ceiling $200/replication · Forces designs to be implementable in follow-up work; designs that exceed flag to the user, not silently included.
- 2026-05-12 · Synthesist in position-paper format · The deliverable mirrors the form of the research goal (a paper), even though this run produces a research-direction document, not a paper itself.
- 2026-05-12 · Token budget rounded to ~1.6M · Phase-4 revision loop is the largest unknown; safety margin reflects that.

## Next step

Run `/research-execute docs/research/plans/2026-05-12-megaresearcher-paper-pipeline-plan.md` to dispatch the swarm. Review this plan before running.
