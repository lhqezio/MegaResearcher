# Augmenting MegaResearcher into an End-to-End Paper Pipeline — Research Spec

**Status:** draft
**Created:** 2026-05-12
**Novelty target:** hypothesis

## Question

What architectural changes to a multi-agent research orchestrator — specifically MegaResearcher's design (single-session orchestrator dispatching leaf workers in waves, no nested dispatch, file-based artifact passing between workers, mandatory audit trail of killed hypotheses) — would close the gaps between current autonomous-research systems (AI-Researcher [arXiv:2505.18705], AI Scientist v1/v2, Agent Laboratory, ResearchAgent, Virtual Lab, Coscientist, and contemporaries) and producing professional-rigor papers? "Professional-rigor" is operationalized as the main-track conference bar (ICLR / NeurIPS / ACL): a novel contribution, multiple baselines, ablation study, statistical significance, related-work that survives expert review, and a manuscript a senior researcher in the field could put their name on without embarrassment. The output of this swarm is a punch list of testable architectural hypotheses, each tied to a specific augmentation MegaResearcher could adopt, with eval designs that allow the augmentation to be implemented and measured in follow-up work.

## Modalities and domain

- **Systems studied:** multi-agent LLM research-automation pipelines published 2023–2026 — AI-Researcher (HKU, 2025), AI Scientist v1 (Sakana, 2024) and v2 (2025), Agent Laboratory, ResearchAgent, Virtual Lab, Coscientist (Boiko et al.), and related orchestrators. Both academic and industry systems in scope.
- **Architectural focus:** which agent roles exist, what loops are present (critique-revise, draft-review, propose-experiment-analyze), how artifacts pass between agents, how the system handles long-horizon coherence, how memory/context is managed across stages, how the system self-evaluates.
- **Output focus:** the synthesist produces a position-paper-style document — sections include intro, related-work map, gap analysis, proposed architecture for the augmented MegaResearcher, hypotheses table, eval designs, threats to validity, audit trail of killed hypotheses.
- **Application area:** scholarly NLP and ML research-paper generation. Cross-domain findings (chemistry-paper generation in Coscientist, biology in Virtual Lab) included only where the architectural insight generalizes to ML-paper generation.

## Constraints

- **No GPU spend during scoping.** This is research-direction work; no training, no fine-tuning, no large-scale eval runs in this swarm.
- **Citation discipline.** Every cited paper must resolve via `hf_papers paper_details` or its arXiv ID must be verifiable. If a paper does not resolve, it does not exist for purposes of this swarm — drop the claim or replace the citation.
- **Open-access preferred.** Where a paywalled paper is the only source for a non-trivial claim, flag it; do not rely on it without surfacing the dependency.
- **Implementable in a Claude Code subagent architecture.** Every proposed augmentation must be expressible as: a single-session orchestrator dispatching leaf-worker subagents in waves, with file-based artifact passing between workers, no nested dispatch. Hypotheses that require nested-agent dispatch, a custom runtime, or fine-tuned models are out of scope for this run (but may be flagged as future work).
- **Testable without closed venues.** Falsification criteria must not require submitting a paper to a real conference. Use proxy measures — LLM-judge panels with disclosed rubrics, blinded human expert review on a held-out sample, reproducibility checks, citation-graph fidelity, etc. Disclose the proxy's limits.
- **Main-track bar, not best-paper bar.** Aiming at ICLR/NeurIPS/ACL accept threshold, not oral / best-paper / domain-prestige bar.

## Success criteria

- **At least 3 surviving hypotheses** that pass red-team critique (up to 3 revision rounds each, per MegaResearcher's discipline rules). Each surviving hypothesis must specify:
  - The augmentation it proposes (which new worker(s), which loop(s), which artifact format)
  - The mechanism — why it would close the gap
  - The predicted outcome — direction and rough magnitude of expected effect
  - **Falsification criteria** — pre-registered decision rules with named thresholds (e.g., "if peer-review-style scores on the held-out 20-paper sample do not improve by ≥0.5 points on average vs the no-revision-loop baseline, the hypothesis is falsified")
  - The eval design — datasets, baselines (named prior systems), metrics, ablations, statistical tests
- **Full audit trail** of every killed or revised hypothesis with the lesson each contributes — no silent rejections.
- **Synthesist document under 12 pages** in the position-paper-style format above.
- **YAGNI fence reflected** in the synthesist document — explicit list of what was deliberately not addressed.
- **Every claim cited.** Uncited assertions are a failure mode the verification step must flag.

## Out of scope (YAGNI fence)

- **Implementing the augmentation.** This run produces a research-direction document, not code. Implementation is separate follow-up work.
- **Running the eval designs.** The eval-designer produces protocols. Actually executing experiments on real paper-generation systems is downstream of this swarm.
- **Domain-specific paper-quality criteria.** "What makes a biology paper good vs an ML paper good vs an HCI paper good" is not in scope. The bar is ML-conference-style rigor.
- **Publishing logistics.** Venue selection, formatting (LaTeX templates, page limits per venue), camera-ready prep, IP / authorship / ethics of AI-authored papers (this is a separate research conversation worth having later but not here).
- **Paywalled-only literature.** If the only sources for a claim are behind paywalls we cannot verify, drop the claim.
- **Training new models / fine-tuning.** Architecture changes only, using off-the-shelf model APIs.
- **Top-tier / best-paper / oral-acceptance bar.** Main-track-accept bar is the ceiling for this run.
- **Changes to MegaResearcher's existing workers in this swarm run.** The bundled six dispatch as-is for this run; proposed new workers are outputs (hypotheses), not workers used by this run.
- **Cost / pricing analysis** of the proposed augmentations. Token cost may be noted as a constraint when relevant to feasibility, but cost-optimization research is not the target.
- **Comparison to non-LLM research-automation systems** (older expert systems, symbolic AI scientific-discovery work like AM/Eurisko). Modern LLM-era systems only.

## Custom workers

None — using the bundled six (literature-scout, gap-finder, hypothesis-smith, red-team, eval-designer, synthesist).

**Note on synthesist format:** the synthesist is instructed in the plan to write the final document in position-paper style (intro, related work, gap analysis, proposed architecture, hypotheses table with falsification criteria, eval designs, threats to validity, audit trail of killed hypotheses, YAGNI fence reflection), not the default generic markdown brief. This is a formatting instruction, not an architectural change to the worker.

## Decisions locked in

- 2026-05-12 · Topic seeded by AI-Researcher paper (arXiv:2505.18705) · User wants to eventually augment MegaResearcher into an end-to-end pipeline; this swarm produces the research punch list, not the implementation.
- 2026-05-12 · Novelty target = `hypothesis` · User wants testable design changes with falsification criteria, not just a literature map. Triggers mandatory red-team loop.
- 2026-05-12 · Bar = main-track conference (ICLR/NeurIPS/ACL) framed as professional rigor · Concrete enough for eval design; avoids best-paper bar which is out of reach for current systems.
- 2026-05-12 · No custom workers this run · User clarified: don't change MegaResearcher's architecture mid-research. Augmentations are the swarm's output, not its workers.
- 2026-05-12 · Synthesist writes position-paper-style output · So the deliverable is in the same format as the target output (a paper), even though this run's output is a research-direction document, not a paper itself.
- 2026-05-12 · Modern LLM-era systems only (2023–2026) · Excludes symbolic AI / expert-systems lineage to keep the scope tractable.
