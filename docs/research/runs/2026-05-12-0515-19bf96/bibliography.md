# Consolidated Bibliography — Phase 1 Outputs

**Run:** 2026-05-12-0515-19bf96
**Total verified citations across all 6 scouts:** ~141 distinct papers (some overlap across scouts on anchor systems like AI Scientist v1/v2, AI-Researcher).

## Scout outputs (read these for full annotations)

| Scout | Sub-topic | Citations | Output path |
|---|---|---|---|
| scout-1 | End-to-end autonomous-research systems | 27 (14 detailed) | `docs/research/runs/2026-05-12-0515-19bf96/scout-1/output.md` |
| scout-2 | Manuscript drafting and document-scale coherence | 26 | `docs/research/runs/2026-05-12-0515-19bf96/scout-2/output.md` |
| scout-3 | Automated peer review and paper-quality evaluation | 28 | `docs/research/runs/2026-05-12-0515-19bf96/scout-3/output.md` |
| scout-4 | Experiment execution and verification in agent systems | 24 | `docs/research/runs/2026-05-12-0515-19bf96/scout-4/output.md` |
| scout-5 | Multi-agent critique, debate, and revision loops | 18 | `docs/research/runs/2026-05-12-0515-19bf96/scout-5/output.md` |
| scout-6 | Memory and state management for long agent workflows | 18 | `docs/research/runs/2026-05-12-0515-19bf96/scout-6/output.md` |

## Anchor citations every gap-finder should know

- **arXiv:2505.18705** — AI-Researcher (HKU). Spec's named anchor. §6.2 names the memory gap; §6.1–6.3 are the cleanest published statement of the gaps MegaResearcher is targeting.
- **arXiv:2408.06292** — AI Scientist v1 (Sakana). The first published full pipeline; failure-mode catalog still active.
- **arXiv:2504.08066** — AI Scientist v2. Workshop-level acceptance claim.
- **arXiv:2501.04227** — Agent Laboratory. Multi-stage assistant architecture.
- **arXiv:2503.18102** — AgentRxiv. §4 has the most-cited concrete failure-mode catalogue (hallucination, reward hacking, impossible plans, LaTeX failures).
- **arXiv:2509.08713** — "Hidden Pitfalls of AI Scientist Systems" (Luo/Kasirzadeh/Shah). Names the four pitfalls: benchmark selection, data leakage, metric misuse, post-hoc bias.
- **arXiv:2511.04583** — Jr. AI Scientist. Recent (2025) variant.

## Critical negative results & ceilings (anchor these for the differential-effect attack)

- **arXiv:2310.01798** — Huang et al. "LLMs Cannot Self-Correct Reasoning Yet." Intrinsic self-correction without external signal is net-negative.
- **arXiv:2502.08788** — Zhang et al. "Stop Overvaluing MAD." Most MAD gains evaporate vs matched-compute CoT + self-consistency.
- **arXiv:2508.17536** — Choi et al. "Debate or Vote?" Majority voting beats inter-agent debate on 7/7 NLP benchmarks; centralized MAD is the *worst* configuration.
- **arXiv:2506.11930** — Feedback Friction. Frontier models cannot fully incorporate even oracle-grade feedback. Caps revision-loop hypotheses.

## Trustworthy and untrustworthy evaluation proxies

**Known-untrustworthy (do NOT use alone in falsification criteria):**
1. Single-shot LLM-as-judge review score — beaten by title+abstract baseline (Höpner 2503.05712)
2. GPT-4 self-report from authors — fails BadScientist (2510.18003)
3. Self-evaluation by same-family model in closed loop (AI Scientist's pattern)
4. Reviewer-reviewer overlap rate alone — gameable (2509.19326)
5. PeerRead-trained acceptance prediction — venue/year drift

**Triangulation candidates (use ≥1 non-judge signal):**
- Reproducibility checks (CORE-Bench 2409.11363)
- Citation-graph fidelity (CiteAudit 2602.23452)
- SPECS-Review-Benchmark flaw-injection axes (CC-BY-4.0 dataset)
- Blinded human spot-check on small sample

## Architectural precedents for MegaResearcher pattern

- **arXiv:2602.01566** — FS-Researcher (Zhu et al.). Closest published precedent to the MegaResearcher architecture (file-system-based dual-agent framework, persistent workspace, stateless agents).
- **arXiv:2508.00031** — Git Context Controller. Maps to the audit-trail rule with version-controlled context (COMMIT/BRANCH/MERGE).

## Stateless-dispatch + file-handoff *compatible* memory systems (in-scope for hypotheses)

A-MEM (2502.12110), MIRIX (2507.07957), AriGraph (2407.04363), G-Memory (2506.07398), GAM (2511.18423), FS-Researcher (2602.01566), Git Context Controller (2508.00031), MemoryOS (2506.06326, tier-adapted), L2MAC (2310.02003, adapted).

## *Incompatible* memory systems (future-work flags only)

KVFlow (2507.07400), KVCOMM (2510.12872), SCBench (2412.10319), and the RL-trained memory controllers (MemPO, DeltaMem, Mem-T, MemGen).

## Systems named in spec but NOT arXiv-resolvable (excluded per discipline rule)

- Coscientist (Boiko et al., Nature 2023) — paywalled, not in hf_papers index
- Virtual Lab (Swanson/Zou et al., Nature 2025) — paywalled, not in hf_papers index
- Genesis-Flow — could not resolve any system by this name
- Carl (Autoscience) / Zochi (Intology) — commercial, no arXiv entry

## High-yield open questions surfaced across scouts

1. Citation verification as pre-flight gate vs post-hoc check (scout-4 OQ#3)
2. Tree-search vs wave-orchestrator vs linear-pipeline — no head-to-head (scout-4 OQ#5)
3. Cross-section coherence in autonomous-research-system outputs — literature does not measure (scout-2 OQ family)
4. Magnitude of presentation-overweighting on a common axis (scout-3 OQ#3)
5. Rebuttal handling — near-absent in current pipelines (scout-3 OQ#4)
6. AI-Researcher's "abstraction drift" and UltraHorizon's "in-context locking" — not unified in corpus (scout-6 OQ#6)
7. Audit-trail completeness as a memory property — unmeasured (scout-6 OQ#7)
8. Whether MegaResearcher's "same base model, different prompts" counts as heterogeneous-debate (scout-5 OQ#4)
9. No paper evaluates revision loops directly on whole-paper-grade output — forecasting gap (scout-5 OQ#6)
