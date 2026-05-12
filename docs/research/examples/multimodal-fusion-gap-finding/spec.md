# Multi-modal AI Fusion for Situational Awareness — Gap-Finding Landscape Spec

**Status:** draft
**Created:** 2026-05-10
**Novelty target:** gap-finding

## Question

Survey 2024–2026 multi-modal fusion architectures relevant to fusing ≥2 of {EO/IR, SAR, RF/SIGINT, audio, text intel, telemetry, sonar} for ISR/C2 situational awareness. Identify under-explored intersections — both unexplored modality pairings and capability gaps against the IDEaS desired-outcome dimensions (spatiotemporal alignment, uncertainty propagation and confidence scoring, policy-aware provenance across classification levels, SWaP-aware edge deployment, operator-facing explainability). Output a gap map plus a shortlist of three candidate architecture families that could plausibly be brought to TRL 4–5 within twelve months by a small team using only open or synthetic data, each annotated with an open-data path, a baseline reference, a SWaP profile, an explainability story, and named technical risks. Output is intended to feed the IDEaS Competitive Projects proposal in the TRL 4–5 / $1.5M / 12-month band (deadline 2026-06-02).

## Modalities and domain

Multi-modal sensor and intelligence fusion for ISR (Intelligence, Surveillance, Reconnaissance) and C2 (Command and Control). Modality set in scope:

- **EO/IR** — electro-optical and infra-red imagery and video
- **SAR** — synthetic aperture radar
- **RF / SIGINT** — radio-frequency signals and signal intelligence
- **Audio** — speech, acoustic event, environmental
- **Text intel** — operational reports, OSINT, structured intelligence text
- **Telemetry** — platform sensor streams (IMU, GPS, mission-system feeds)
- **Sonar** — maritime acoustic (passive and active)

Application contexts surveyed wide (per the IDEaS call):

1. Joint ISR fusion for Arctic operations (satellite imagery + RF + telemetry)
2. Real-time multi-domain threat assessment (EO video + SIGINT + text intel)
3. Tactical-edge fusion on wearables under degraded connectivity (audio + video + sensor)
4. Maritime task group anomaly detection (sonar + RF + visual + uncertainty)
5. Airborne multi-sensor stealth/spoof detection (radar + EO/IR + telemetry)

Operational context: real-time, explainable, policy-aware decision support; outputs targeted at CAF ISR platforms, C2 systems, and tactical-edge deployments. The capability is positioned by DND as a contrast to traditional rule-based fusion: AI-driven architectures that learn complex relationships across modalities, propagate uncertainty, and deliver policy-aware, explainable outputs.

## Constraints

- **No classified data** of any kind. Open or synthetic datasets only.
- **Stable identifiers** for every dataset cited: HF dataset name + revision, arXiv ID, or DOI.
- **Licence flagged** per dataset. Anything more restrictive than CC-BY needs an explicit flag in the bibliography.
- **No GPU spend** during this scoping run. CPU-only baselines if any sandboxed exploration is performed.
- **No HF Jobs** submissions. **No HF Spaces** creation. Only local Bash sandboxing for any code execution.
- **All citations** must be retrievable via `hf_papers`, arXiv, or Semantic Scholar (the ml-intern MCP surface). No invented citations — flag and skip if not retrievable.
- **Reproducibility**: every claim is backed by a stable identifier; any numeric result is seeded; the run reproduces from a clean checkout via documented `uv run` commands.
- **Output discipline**: bibliography, gap map, and candidate shortlist live under `scoping/outputs/` with inline citations; the swarm run also writes to `docs/research/runs/<run-id>/` per MegaResearcher convention.
- **No external engagement**. The swarm never contacts DND, CAF, or any third party. All correspondence is the user's.

## Success criteria

1. **Annotated bibliography** — at least 25 citations, all dated 2024–2026, all retrievable via `hf_papers` or arXiv, grouped by modality pair (e.g., "EO+RF", "EO+text", "SAR+telemetry"). Each entry: title, arXiv ID, year, modality pair(s), one-sentence relevance note.
2. **Gap map** — explicit list of under-explored or thinly-explored intersections within the modality set, each scored against the five IDEaS desired-outcome capabilities (spatiotemporal alignment, uncertainty propagation, policy-aware provenance, SWaP-aware edge deployment, explainability). Score scale defined inline by the gap-finder worker.
3. **Three-candidate shortlist** of architecture families plausibly reachable to TRL 4–5 in 12 months by a small team using only open / synthetic data. Each candidate documents:
   - At least one open dataset path with stable identifier and licence
   - At least one baseline or reference implementation (paper + repo where available)
   - A SWaP profile: parameter count, expected inference cost, edge-deployment story
   - An explainability story: what does the operator see, and how is it derived?
   - Named technical risks with severity
4. **Synthesist document** ≤ 8 pages, every claim cited, with:
   - Surviving-vs-killed audit trail (which gaps were considered, which kept, which dropped, why)
   - The YAGNI fence reflected explicitly
   - A "what would change our mind" section noting evidence that would invalidate the shortlist
5. **No invented citations.** Verification step confirms every cited paper resolves on arXiv or HF Papers.

Red-team approval is **not applicable** for this run — `gap-finding` does not invoke the hypothesis-smith / red-team loop. The synthesist's audit trail is the substitute discipline mechanism.

## Out of scope (YAGNI fence)

- **Hypothesis generation with falsification criteria** — explicitly deferred to a later phase once a candidate architecture is selected.
- **Eval-design experiments** — `gap-finding` does not trigger the eval-designer worker.
- **Classical / rule-based fusion methods** — surveyed only where directly compared as baselines in AI fusion papers; not a primary subject.
- **Operational concept of operations (ConOps), wargaming integration, doctrine drafting** — proposal-stage work, not landscape research.
- **Cost modeling, team narrative, partner mapping, letter-of-support drafting** — proposal-stage work.
- **Procurement / supplier selection.**
- **Eligibility review** — confirmed separately by the user.
- **Submission to CanadaBuys.**
- **Direct engagement with DND, CAF, or any other government body.**
- **GPU-bound experiments, training-job submissions, HF Jobs orchestration, HF Spaces deployment** — deferred to a later compute-approved phase.
- **Picking one application example** — wide-scope is intentional this run; narrowing to one of the five IDEaS application contexts is a follow-up phase decision informed by the gap map.

## Custom workers

None — using the bundled six (literature-scout, gap-finder, hypothesis-smith, red-team, eval-designer, synthesist).

For this run, only **literature-scout**, **gap-finder**, and **synthesist** will fire. The hypothesis-smith, red-team, and eval-designer workers idle when novelty target = `gap-finding`.

## Decisions locked in

- 2026-05-10 · TRL band 4–5 · honest target given 12-month build window and no prior code.
- 2026-05-10 · Tooling: ml-intern via the local MCP server at `ml_intern_cc/` (CC-native, not standalone CLI).
- 2026-05-10 · Novelty target: `gap-finding` · proposal-shaped output with lower nonsense risk; hypothesis-smithing deferred to a Phase 2 once a candidate is picked.
- 2026-05-10 · Scope: wide across all 7 modalities and all 5 IDEaS application examples · narrowing first would defeat the point of gap-finding.
- 2026-05-10 · Outputs land under `scoping/outputs/` (existing project convention) and `docs/research/runs/<run-id>/` (MegaResearcher convention).
