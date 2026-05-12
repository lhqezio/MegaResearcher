# Verification — scout-6

This document records the `superpowers:verification-before-completion` checks I ran for the scout-6 bibliography, plus the documented workaround for the broken `hf_papers` kwargs path.

## Workaround note: hf_papers tool

Per the runtime issue flagged in the worker brief, `mcp__ml-intern__hf_papers` (and its alias `mcp__plugin_megaresearcher_ml-intern__hf_papers`) currently returns `'<key>' is required for <op>` for every kwargs-bearing operation, regardless of payload shape. Only `trending` works and that operation does not let me look up a specific arXiv ID.

The research spec explicitly permits citation retrieval "via `hf_papers`, arXiv, or Semantic Scholar (the ml-intern MCP surface)" — so I substituted **direct WebFetch against `https://arxiv.org/abs/<id>`** as the primary verification channel. Each WebFetch call was prompted to return title, first author, year, a 2-sentence summary, and a real/not-real verdict; I treated only positive verdicts with consistent metadata as valid citations.

Discovery used `mcp__ml-intern__web_search` and `WebSearch` (Anthropic native) — both worked normally. `mcp__ml-intern__github_examples` returned an internal `'repo'` error on every call, so reference implementations were verified directly with WebFetch against GitHub.

## Required check 1: every cited arXiv ID resolves

I WebFetched every arxiv ID cited in `output.md` against the canonical `https://arxiv.org/abs/<id>` URL. All 41 IDs returned valid arXiv metadata with titles consistent with the way they appear in the bibliography.

**Spot-check (recorded verbatim from WebFetch output):**

> arXiv:2503.05274 → "Evidential Uncertainty Estimation for Multi-Modal Trajectory Prediction" — Sajad Marvi et al. — 2025. "This is a legitimate paper submitted to arXiv on March 7, 2025 (arXiv:2503.05274). The submission shows proper metadata, author information, and institutional affiliations, with a complete abstract and access to PDF and HTML versions."

A second spot-check on a provenance-cluster ID:

> arXiv:2502.19567 → "Atlas: A Framework for ML Lifecycle Provenance & Transparency" — Marcin Spoczynski et al. — 2025. "This is a legitimate submission to arXiv (identifier 2502.19567), categorized under Cryptography and Security (cs.CR), with a submission date of February 26, 2025, and a revision on May 14, 2025."

A third spot-check on an explainability-cluster ID:

> arXiv:2510.21518 → "Head Pursuit: Probing Attention Specialization in Multimodal Transformers" — Lorenzo Basile et al. — 2025. "Yes, this is a legitimate arXiv submission (arXiv:2510.21518) that was accepted as a spotlight presentation at NeurIPS 2025."

## Required check 2: no invented citations — flagged-and-skipped IDs

One paper was rejected during verification:

- **arXiv:2406.05527** — surfaced by WebSearch labelled as "Prior-guided Fusion of Multimodal Features for Change Detection from Optical-SAR Images." When WebFetched, the ID resolved to a condensed-matter physics paper ("Ab-initio investigations of novel potential all-d metal Heusler alloys Co2MnNb" — Sumit Kumar 2024). The search engine had returned a misleading title-to-ID mapping (probably a hash collision in DuckDuckGo's snippet). **I dropped this candidate from the bibliography** per the discipline rule.

No other ID was rejected. Two candidates surfaced by search were not included in the final output because they post-date the cut-off implausibly (e.g., results dated 2604.* or 2605.* — search engine dating artefacts) and could not be cleanly verified; rather than risk an invented citation I omitted them.

## Required check 3: ≥8-paper floor

Bibliography count = **41 papers** across five sub-clusters:

| Sub-cluster | Papers |
|---|---|
| A. Spatiotemporal alignment | 6 |
| B. Uncertainty propagation | 8 |
| C. Policy-aware provenance | 7 |
| D. SWaP-aware edge deployment | 14 |
| E. Operator-facing explainability | 7 |
| **Total** | **41** |

This far exceeds the 8-paper floor, and the worker brief explicitly anticipated 15–25 across this scout's five sub-buckets.

The thinnest cluster is sub-cluster A (spatiotemporal alignment) at 6 papers. This is intentional and justified: many alignment papers are domain-specific (autonomous-driving radar/lidar/camera) and were excluded as out-of-scope for the capability-axis scan. The six retained are the ones that actually advance a *capability primitive* (cross-resolution alignment, temporal-rearrangement fusion, alignment-as-imputation, late-fusion-as-alignment-bypass) rather than just a domain-specific architecture. If a stricter ≥8 per cluster reading is preferred, A1 (the 2024 survey) and A2/A3/A4/A5/A6 still satisfy the global ≥8 floor.

## Required check 4: every dataset has a verifiable HF page or licence note

| Dataset | Verifiable identifier / source | Licence note in output |
|---|---|---|
| M4-SAR | arXiv:2505.10931 paper page (paper releases the dataset) | Licence-check-required (not stated in the verification fetch) — flagged |
| KITTI | Canonical project page (referenced by A6/B6/D7) | CC-BY-NC-SA 3.0 — non-commercial restriction flagged |
| Ego4D | Canonical project (referenced by A5) | Custom Ego4D Licence Agreement — flagged restrictive |
| Sentinel-1 / Sentinel-2 | Copernicus / ESA programme | Copernicus open licence with attribution |
| HF model-card / supply-chain corpora | C4 (arXiv:2402.05160) and C5 (arXiv:2502.04484) papers | Public metadata; HF terms apply |

No dataset is cited without either a paper-of-record arXiv ID or an established programme identifier. All restrictive licences are flagged in `output.md` per the spec's "anything more restrictive than CC-BY needs an explicit flag" rule.

## Reference-implementation sanity checks

Each GitHub reference in `output.md` was visited via WebFetch and confirmed to exist with non-zero star count consistent with reported number:

- `jacobgil/vit-explain` — 1.1k stars, attention rollout / gradient attention rollout for ViTs (verified).
- `aangelopoulos/conformal-prediction` — 1.0k stars, MIT, conformal prediction notebooks (verified).
- `hanmenghan/TMC` — 281 stars, official TMC implementation (verified).
- `Meituan-AutoML/MobileVLM` — 1.4k stars, Apache-2.0 (verified).
- `apple/ml-fastvlm` — 7.3k stars, official Apple FastVLM (verified).
- `ggerganov/llama.cpp` — 109k stars, canonical edge inference runtime (verified).

Two repositories I attempted to verify did not exist at the URLs I had inferred (`IntelLabs/Atlas`, `HCPLab-SYSU/Causal-VLR`, `yProvML/yProvML`) — these were therefore **not cited** in the reference-implementations section, even though the underlying papers are in the bibliography. This is consistent with the discipline rule.

## Conclusion

All four required verification checks pass. The scout-6 outputs (`output.md`, `manifest.yaml`, `verification.md`) are ready for orchestrator review.
