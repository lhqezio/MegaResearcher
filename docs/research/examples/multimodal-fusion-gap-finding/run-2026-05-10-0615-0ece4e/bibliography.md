# Consolidated Bibliography — run 2026-05-10-0615-0ece4e

Phase 1 complete. Six literature-scout outputs available for Phase 2 (gap-finders).

## Per-scout summaries

- **scout-1** — EO/IR + SAR fusion — cross-attention transformers, joint EO-SAR encoders, contrastive pretraining; spatiotemporal alignment under cloud/night; satellite, airborne, maritime, Arctic ISR application affinity
  - papers: 22 · datasets: 11 · implementations: 12 · open questions: 8
  - Output: `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/scout-1/output.md`

- **scout-2** — RF / SIGINT fused with imagery or text intel (2024-2026); RadioML-derived backbones, spectrogram-image alignment, RF + VLM integration, RF + telemetry / RF + EO multi-sensor fusion
  - papers: 25 · datasets: 8 · implementations: 12 · open questions: 9
  - Output: `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/scout-2/output.md`

- **scout-3** — Tactical-edge wearable audio+video+sensor fusion under SWaP and degraded-connectivity (2024-2026 multimodal fusion under ~100M params, on-device inference, federated / intermittent connectivity, compression, and operator-facing explainability).
  - papers: 31 · datasets: 11 · implementations: 10 · open questions: 8
  - Output: `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/scout-3/output.md`

- **scout-4** — scout-4 — bridging unstructured text intel and OSINT with sensor or imagery streams (EO/IR primary, SAR adjacent), 2024–2026
  - papers: 26 · datasets: 9 · implementations: 12 · open questions: 10
  - Output: `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/scout-4/output.md`

- **scout-5** — Sonar + maritime multi-modal anomaly detection (AIS / SAR / RF / visual / sonar; 2024-2026)
  - papers: 26 · datasets: 13 · implementations: 11 · open questions: 8
  - Output: `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/scout-5/output.md`

- **scout-6** — Cross-cutting capability axes (spatiotemporal alignment, uncertainty propagation, policy-aware provenance, SWaP-aware edge deployment, operator-facing explainability) surveyed across modalities for 2024-2026
  - papers: 41 · datasets: 5 · implementations: 6 · open questions: 8
  - Output: `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/scout-6/output.md`

## Totals

- **Papers:** 171 (spec floor: 25)
- **Datasets:** 57 (with licence flags)
- **Reference implementations:** 63
- **Open questions for Phase 2:** 51

## Phase 1 runtime notes

- All 6 scouts independently identified an `hf_papers` parameter-deserialization bug in the running MCP server.
- All 6 scouts worked around it via WebFetch against `arxiv.org/abs/<id>` and `huggingface.co/papers/<id>`. Spec explicitly permits this fallback.
- Fix committed in MegaResearcher 3819dd4; takes effect on next MCP server start.
- Phase 2 prompts will pre-bake the workaround note so gap-finders skip rediscovery.
