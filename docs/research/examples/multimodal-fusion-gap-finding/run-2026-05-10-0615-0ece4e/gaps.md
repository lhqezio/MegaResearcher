# Consolidated Gap Map — run 2026-05-10-0615-0ece4e

Phase 2 complete. Three gap-finder outputs available for Phase 6 (synthesist).

For this run novelty target = `gap-finding`, so Phases 3 (hypothesis-smith), 4 (red-team), and 5 (eval-designer) are intentionally skipped. The synthesist composes the final research-direction document directly from scout + gap-finder outputs.

## Gap-finder outputs

### gap-finder-1

\`\`\`yaml
role: gap-finder
slice: modality-pair-gaps (all six scout outputs scout-1..scout-6, full)
gaps_count: 10
discarded_count: 5
verification_tool: mcp__ml-intern__web_search + WebFetch (mcp__ml-intern__hf_papers blocked by parameter-deserialization bug, documented per spec workaround)
output_path: /Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/gap-finder-1/
artifacts:
  - output.md
  - manifest.yaml
  - verification.md
modality_pairs_evaluated:
  - EO/IR + SAR
  - EO/IR + text-intel
  - EO/IR + RF/SIGINT
  - EO/IR + audio
  - EO/IR + telemetry
  - EO/IR + sonar
  - SAR + text-intel
  - SAR + RF/SIGINT
  - SAR + audio
  - SAR + sonar
  - SAR + telemetry
  - RF/SIGINT + audio
  - RF/SIGINT + text-intel
  - RF/SIGINT + telemetry
  - audio + telemetry
  - audio + sonar
  - sonar + AIS
  - sonar + EO/IR + AIS (maritime tri-modal)
  - sonar + RF + visual + AIS (maritime quad-modal)
  - EO/IR + SAR + AIS (dark-vessel)
  - text-intel + event-streams + EO/IR (joint forecasting)
capability_dimensions_scored:
  - spatiotemporal alignment
  - uncertainty propagation
  - policy-aware provenance
  - SWaP-aware edge deployment
  - operator explainability
\`\`\`

Output: `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/gap-finder-1/output.md`

### gap-finder-2

\`\`\`yaml
role: gap-finder
slice: capability-axis gaps across all six scout outputs (scout-1 EO/SAR, scout-2 RF/SIGINT, scout-3 tactical-edge wearable, scout-4 text intel, scout-5 sonar/maritime, scout-6 cross-cutting capability — primary)
gaps_count: 8
discarded_count: 4
\`\`\`

Output: `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/gap-finder-2/output.md`

### gap-finder-3

\`\`\`yaml
role: gap-finder
variant: gap-finder-3-trl-buildability-filter
slice:
  - scout-1 (EO/IR + SAR)
  - scout-2 (RF/SIGINT)
  - scout-3 (audio + video + sensor at edge)
  - scout-6 (cross-cutting capability axes)
  - excluded: scout-4 (text/OSINT), scout-5 (sonar/maritime) — per assignment
gaps_count: 0   # n/a — this gap-finder produces a candidate-shortlist artefact, not a gap list
candidates_count: 8
candidates_top_tier_count: 4   # A or A-rated: 1 (TerraMind), 2 (DOFA + M4-SAR), 3 (Galileo), 4 (SmolVLM + COMODO)
candidates_second_tier_count: 4   # B or B-rated: 5 (RFML), 6 (SoftFormer/DDFM), 7 (OV-AVE on SmolVLM), 8 (EarthMind)
discarded_count: 8
discarded_breakdown:
  - SkySense — fails criterion (i): non-commercial research-only weights
  - RingMoE — fails criteria (ii) and (iii): no public reference impl, infeasible at 14.7B for small team
  - FastVLM — fails criterion (i): research-only model licence (verified via LICENSE_MODEL fetch)
  - ImageBind — fails criterion (i): CC-BY-NC 4.0 (verified)
  - Ego4D / Ego-Exo4D as deployment training data — fails criterion (i): custom signed research-only licence
  - RadioML 2018.01A as deployment training data — fails criterion (i): CC-BY-NC-SA 4.0 (verified)
  - DynamicEarthNet as primary training data — fails criterion (i): CC-BY-NC-SA
  - WavesFM / RF-GPT / PReD direct re-implementation — fails criterion (ii): no public reference impl confirmed
verification_method: WebFetch on github.com / huggingface.co / mendeley.com / deepsig.ai (mcp__ml-intern__hf_papers wrapper broken)
output_path: /Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/gap-finder-3/
artifacts:
  - output.md
  - manifest.yaml
  - verification.md
\`\`\`

Output: `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/gap-finder-3/output.md`

## Convergence signal

gap-finder-1 (modality-pair) and gap-finder-2 (capability) — independent agents, partly overlapping scout slices — converged on:
- **No calibrated cross-modal uncertainty in any 2024–2026 EO/SAR foundation model** (gap-finder-1 G4 ↔ gap-finder-2 G5/G8)
- **No on-device SWaP measurements for fused EO+SAR/EO+RF detectors** (gap-finder-1 G7 ↔ gap-finder-2 G3)
- **No paper jointly delivers calibrated multi-modal uncertainty + operator-facing explainability** (gap-finder-2 G8 implicit in gap-finder-1 across multiple cells)
- **Cross-classification-level provenance is modality-agnostic and unimplemented in fusion stacks** (gap-finder-1 G6 ↔ gap-finder-2 G1/G2)

These are the strongest signals for the synthesist to lean on.

## TRL-buildability shortlist input (gap-finder-3)

gap-finder-3 produced a ranked 8-candidate list with explicit licence-driven discards. The synthesist picks the final 3 from this list per the spec`s success criterion #3.

gap-finder-3`s recommended top-3 (synthesist guidance):
- **TerraMind** (Apache-2.0 weights + CC-BY-4.0 MMEarth) — Arctic ISR / cloud-cover cross-modal completion
- **DOFA + M4-SAR** (MIT + CC-BY-4.0) — EO/SAR object detection (AGPL-3.0 dataset is a flag)
- **Galileo + Discounted Belief Fusion** (MIT + MIT) — temporal-axis EO/SAR with uncertainty
- **SmolVLM-256M + COMODO** (Apache-2.0) — tactical-edge wearable

Disqualified (licence/size) and recorded in gap-finder-3 discarded section: FastVLM, ImageBind, RadioML-2018.01A, SkySense, RingMoE.
