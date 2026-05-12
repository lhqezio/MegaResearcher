# Multi-modal AI Fusion for Situational Awareness — Research Plan

**Status:** draft
**Created:** 2026-05-10
**Spec:** `docs/research/specs/2026-05-10-multimodal-fusion-gap-finding-spec.md`
**Run mode:** MegaResearcher `research-swarm`, novelty target = `gap-finding`

## Context

This plan executes the gap-finding landscape survey defined in the spec. The output feeds the IDEaS Competitive Projects proposal (TRL 4–5 / $1.5M / 12-month band; submission deadline 2026-06-02).

The plan does not produce hypotheses, eval designs, or proposal copy. It produces a literature-grounded gap map and a three-candidate buildable shortlist. Hypothesis-smithing, eval design, and proposal writing are explicitly downstream.

Project guardrails to read alongside this plan:
- `scoping/BRIEF.md` — single source of truth for the challenge.
- `scoping/CONSTRAINTS.md` — data, compute, reproducibility, output discipline.
- `ml_intern_cc/server.py` — research-tool surface area available to workers.

## Approach

Run a 6-phase research-swarm. With novelty target = `gap-finding`, only Phases 1, 2, and 6 fire (literature-scout, gap-finder, synthesist). Phases 3, 4, 5 (hypothesis-smith, red-team, eval-designer) idle.

The decisive design choice is the Phase-1 partition. With 7 modalities and 5 IDEaS application contexts, naive "one scout per modality" misses the cross-cutting capability axes (spatiotemporal alignment, uncertainty propagation, policy-aware provenance, SWaP, explainability) that the IDEaS rubric scores against. The plan partitions Phase 1 by **modality pair × application affinity** (5 scouts) plus **one cross-cutting capability scout** (1 scout) so the gap-finder receives both axes.

Phase 2 runs three gap-finders along non-overlapping focus axes: modality-pair gaps, capability gaps, and TRL-buildability filter.

Phase 6 produces the deliverable per the spec's success criteria.

## Critical files

- **Spec:** `docs/research/specs/2026-05-10-multimodal-fusion-gap-finding-spec.md`
- **Run directory (created by orchestrator):** `docs/research/runs/2026-05-10-<HHMM>-<short-hash>/`
- **Final deliverable:** `docs/research/runs/<run-id>/output.md` (synthesist) — copy to `scoping/outputs/01-landscape/` per project convention after the run.
- **Project brief:** `scoping/BRIEF.md`
- **Research constraints:** `scoping/CONSTRAINTS.md`
- **MCP tool surface:** `ml_intern_cc/server.py` — exposes `hf_papers`, `hf_inspect_dataset`, `hf_docs_explore`, `hf_docs_fetch`, `hf_repo_files`, `github_examples`, `github_list_repos`, `github_read_file`, `web_search`.

## Swarm decomposition

### Phase 1 — literature-scout dispatches

Six scouts. Five modality-axis + one cross-cutting capability-axis. Each scout produces an annotated bibliography (~5+ citations, 2024–2026) per the worker contract.

**scout-1 — EO/IR + SAR fusion**
Sub-topic: cross-modal fusion of electro-optical/infrared imagery with synthetic aperture radar.
Focus: 2024–2026 only. Cross-attention transformers, joint EO-SAR encoders, contrastive pretraining for satellite and airborne platforms. Spatiotemporal alignment under cloud cover and night/IR. Application affinity: Arctic ISR, airborne stealth/spoof detection, maritime visual.
Tools: `hf_papers` (queries: "EO SAR fusion", "multimodal remote sensing 2025", "cross-attention satellite imagery"), `github_examples` for reference implementations, `hf_inspect_dataset` for OpenSAR-Ship, BigEarthNet, Sentinel-1/2 collocated sets.

**scout-2 — RF / SIGINT fused with imagery or text**
Sub-topic: RF / signal intelligence fused with imagery or with text intel.
Focus: 2024–2026 only. RadioML-derived backbones, joint signal-image training, modulation classification + visual context, spectrum-imagery alignment, RF + OSINT correlation. Application affinity: multi-domain threat assessment, Arctic ISR.
Tools: `hf_papers` (queries: "RF SIGINT deep learning fusion 2025", "modulation classification multimodal", "signal imagery joint embedding"), `web_search` for RadioML-2018+ updates, `github_examples` for open RF + vision pipelines.

**scout-3 — Audio + video + sensor fusion at the edge**
Sub-topic: tactical-edge wearable fusion under SWaP and degraded-connectivity constraints.
Focus: 2024–2026 only. Multi-modal models < 100M params, on-device inference, audio-visual event detection, IMU/sensor fusion, federated / intermittent-connectivity training, model compression. Explicitly include explainability for operator-worn systems. Application affinity: tactical-edge wearables.
Tools: `hf_papers` (queries: "edge multimodal fusion 2025", "tinyml audio video fusion", "wearable activity recognition multimodal"), `github_examples` for on-device inference frameworks, `hf_inspect_dataset` for AudioSet, Ego4D, EPIC-KITCHENS where relevant.

**scout-4 — Text intel fused with sensor or imagery**
Sub-topic: bridging unstructured text intel and OSINT with sensor or imagery streams.
Focus: 2024–2026 only. Vision-language models for OSINT + EO, knowledge graph integration, entity resolution across reports and sensor data, event correlation (ACLED/GDELT-class), report → sensor cueing. Application affinity: multi-domain threat assessment.
Tools: `hf_papers` (queries: "vision language OSINT 2025", "multimodal knowledge graph defense", "entity resolution sensor text"), `web_search` for recent ACLED/GDELT applied work, `hf_inspect_dataset` for any open OSINT-imagery paired sets.

**scout-5 — Sonar + maritime multi-modal anomaly detection**
Sub-topic: maritime fusion across sonar, RF, AIS, and visual.
Focus: 2024–2026 only. Hydroacoustic deep learning, AIS + sonar + RF + visual fusion, anomaly detection with uncertainty scoring, OpenSAR-Ship and similar maritime datasets. Application affinity: maritime task group anomaly detection.
Tools: `hf_papers` (queries: "maritime multimodal fusion 2025", "AIS sonar deep learning", "ship anomaly detection multimodal"), `web_search` for open sonar dataset releases, `hf_inspect_dataset` for maritime-relevant datasets.

**scout-6 — Cross-cutting capability axes (capability-axis scout, not modality-axis)**
Sub-topic: methods for the five IDEaS desired-outcome capabilities, surveyed across modalities.
Focus: 2024–2026 only. Five sub-buckets:
1. Spatiotemporal alignment across heterogeneous streams (asynchronous timestamps, varying spatial resolution).
2. Uncertainty propagation and confidence scoring across modalities (Bayesian deep learning, evidential learning, conformal prediction in fusion).
3. Policy-aware fusion and AI provenance / lineage tracking across classification levels.
4. SWaP-aware edge deployment (model compression, quantization, distillation specific to multimodal).
5. Operator-facing explainability for fusion outputs.
Tools: `hf_papers` (one query per sub-bucket), `web_search` for non-arxiv methods (NIST, IEEE), `github_examples` for reference implementations of each.

### Phase 2 — gap-finder dispatches

Three gap-finders. Non-overlapping focus axes.

**gap-finder-1 — Modality-pair gaps**
Slice: all six scout outputs.
Focus: which modality pairs in {EO/IR, SAR, RF/SIGINT, audio, text intel, telemetry, sonar} are well-served by 2024–2026 work, which are under-served, and which are unexplored. Score every pair against the five IDEaS capability dimensions. Output a matrix: rows = modality pairs (or triples present in the literature), columns = capability dimensions, cells = "served / thin / absent" with citations.

**gap-finder-2 — Capability gaps within active modality work**
Slice: all six scout outputs (especially scout-6).
Focus: across the modality work that does exist, which IDEaS capability dimensions are systematically under-addressed? Likely-thin a priori: policy-aware provenance and SWaP-aware edge deployment. Confirm or refute against the literature. Output a ranked list of capability gaps with citations.

**gap-finder-3 — TRL-buildability filter**
Slice: scout-1, scout-2, scout-3, scout-6 outputs.
Focus: of the architectures surveyed, which have all three of (i) open dataset access, (ii) open baseline / reference implementation, and (iii) small-team / 12-month / no-classified-data feasibility? Output a candidate list ranked by buildability with named open data + named baseline + footprint estimate per candidate. This is the upstream of the synthesist's three-candidate shortlist.

### Phase 3 — hypothesis-smith dispatches

**Idle for this run.** Novelty target = `gap-finding`. The orchestrator must not invoke hypothesis-smith.

### Phase 4 — red-team critique loop

**Idle for this run.** No hypotheses to critique. The synthesist's audit trail substitutes for the red-team discipline mechanism.

### Phase 5 — eval-designer dispatches

**Idle for this run.** No surviving hypotheses to design experiments for.

### Phase 6 — synthesist

Single dispatch. Inputs: spec + all six scout outputs + all three gap-finder outputs.

Synthesis-specific requirements:
- **Length:** ≤ 8 pages.
- **Structure:** must include (i) gap map (matrix from gap-finder-1, refined), (ii) three-candidate shortlist with the per-candidate fields specified in the spec's success criteria #3, (iii) audit trail of which gaps were considered and why each was kept or dropped, (iv) YAGNI fence reflected explicitly per the spec, (v) a "what would change our mind" section noting evidence that would invalidate any of the three shortlisted candidates.
- **Citation discipline:** every claim must be backed by an arXiv ID, HF Papers entry, or DOI. The verification.md artifact must list every cited paper with its retrieval status.
- **YAGNI fence:** explicit section reflecting the spec's Out-of-scope list. The synthesist must not drift into proposal copy, ConOps, or hypothesis-shaped claims.

### Custom worker dispatches

None.

### Parallelism budget

`MEGARESEARCHER_MAX_PARALLEL = 4` (default).

Phase 1 (6 scouts) runs in two waves of ~3 each. Phase 2 (3 gap-finders) runs in one wave. Phase 6 (synthesist) is single-dispatch.

### Estimated total runtime + token budget

| Phase | Workers | ~Tokens / worker | Subtotal |
|---|---|---|---|
| 1 — literature-scout | 6 | 25k | 150k |
| 2 — gap-finder | 3 | 30k | 90k |
| 6 — synthesist | 1 | 40k | 40k |
| Orchestrator overhead | — | — | 50k |
| **Total (estimate)** | | | **~330k** |
| **Total (rounded up for safety)** | | | **~450k** |

Wall-clock estimate at parallelism=4: ~45–75 minutes total. Phase 1 = two waves × ~10–15 min = 20–30 min. Phase 2 = single wave × ~10–15 min. Phase 6 = ~5–10 min.

Assumptions: average scout produces ~6–10 citations with brief annotations; average gap-finder reads ~30 citations; synthesist reads all 9 worker outputs plus the spec. Tool calls per worker dominate token usage; each `hf_papers` call returns 2–4k tokens of paper metadata.

## Verification

The run is "done" when:

1. **Bibliography count:** ≥ 25 unique citations across the six scout outputs, all 2024–2026, all retrievable via `hf_papers` or arXiv. Verified by spot-checking 5 random citations.
2. **Gap map present:** `gap-finder-1` output contains a modality-pair × capability-dimension matrix with ≥ 80% of cells populated (served / thin / absent + citation pointer).
3. **Shortlist completeness:** synthesist's three-candidate shortlist has all five fields per candidate (open dataset, baseline reference, SWaP profile, explainability story, named risks).
4. **Audit trail:** synthesist names ≥ 3 considered-but-killed gaps with explicit reasons.
5. **YAGNI fence:** explicit section in synthesist output mirroring spec's Out-of-scope list.
6. **No invented citations:** verification.md confirms every cited paper resolves on arXiv or HF Papers.
7. **Spec success criteria 1–5 all checked.**

If any of 1–7 fail, re-dispatch the failing worker(s) with a course-correction prompt. Do not paper over verification failures in the synthesist.

## Decisions locked in

- 2026-05-10 · Six scouts, three gap-finders, one synthesist · five-modality-axis + one capability-axis Phase 1 partition gives the gap-finder both axes; three gap-finders cover modality-pair, capability, and TRL-buildability without overlap.
- 2026-05-10 · Parallelism = 4 (default) · no reason to deviate; the swarm size fits.
- 2026-05-10 · Token budget rounded to ~450k · safety margin against scout/gap-finder verbosity drift.
- 2026-05-10 · Verification gates Spec success criteria 1–5 plus three swarm-mechanics gates (bibliography count, audit-trail completeness, no invented citations).

## Next step

Run `/research-execute docs/research/plans/2026-05-10-multimodal-fusion-gap-finding-plan.md` to dispatch the swarm. Do not run it without reviewing the plan first.
