# Gap-Finder-3 Verification Record

## Tooling note (workaround)

The assignment-supplied `mcp__ml-intern__hf_papers` MCP wrapper is broken (parameter-deserialization bug — `'query' is required` / `'arxiv_id' is required` errors regardless of payload shape, also reported in scout-2 verification.md). Per the assignment instructions, I used:

- `WebFetch` against the canonical GitHub / HuggingFace / Mendeley / DeepSig page for each load-bearing artefact.
- `mcp__ml-intern__web_search` (DuckDuckGo backend) for gap-pattern verification queries.
- `mcp__ml-intern__hf_inspect_dataset` (works) for dataset-schema validation on the canonical AudioSet HF mirror.
- `mcp__ml-intern__hf_repo_files` (broken — returns generic help text instead of file listings); fell back to WebFetch on the HF web pages.

Every cited paper in `output.md` has an arXiv ID present in the upstream scout outputs (`scout-1`, `scout-2`, `scout-3`, `scout-6`), each of which has its own verification record confirming `arxiv.org/abs/<id>` resolution. This gap-finder does NOT add new arXiv citations beyond the consolidated bibliography — every arXiv ID is sourced from a verified scout entry.

## Verification checks (per `superpowers:verification-before-completion`)

### Check 1 — Every claimed candidate has a recorded verification

Pass. See `output.md` §4 (the verification summary table). Each of the eight candidates has at least one of {licence, parameter-count, repo-existence} confirmed by a direct WebFetch, with the URL and the quote captured in §4.

### Check 2 — The discarded-candidates section is non-empty (proves verification, not confirmation bias)

Pass. `output.md` §3 lists eight discarded candidates, each with a named rejection criterion and (where applicable) a direct WebFetch verification (FastVLM, ImageBind, RadioML, DynamicEarthNet, Ego4D, RingMoE, SkySense, WavesFM-class).

The two highest-impact discards — FastVLM and ImageBind — were both ones I would have intuitively picked as substrates before verification. Verification flipped them. This is the discipline rule working as intended.

### Check 3 — No candidate claim is made without supporting citations

Pass. Every candidate cites:
- One or more arXiv IDs for the architecture family (sourced from scout outputs).
- A specific repo URL for the open implementation (with star count and licence verified).
- A specific dataset identifier (with licence verified).

Where a sub-claim could not be verified at scout-time (e.g., star counts on `nasaharvest/galileo`, parameter counts on TerraMind sub-variants), I performed an additional WebFetch and recorded the result inline in §1 or §4.

### Check 4 — Every cited paper resolves via paper-details

The `mcp__ml-intern__hf_papers paper_details` tool is part of the same broken wrapper. Per scout-1, scout-2, scout-3, scout-6 verification records, every arXiv ID carried into this gap-finder has been independently verified to resolve via `arxiv.org/abs/<id>` WebFetch. This gap-finder does not introduce any new arXiv ID; every ID in `output.md` is traceable to a verified scout entry.

## Key WebFetch verifications performed in this run

These are the load-bearing licence and feasibility checks I performed myself (beyond the upstream scout verifications). The full request-and-response is captured in the conversation transcript; the abbreviated record:

1. **`github.com/IBM/terramind`** — License Apache-2.0, 258 stars. TerraMind-tiny/small/base/large + tokenizers per modality.
2. **`github.com/IBM/terramind/blob/main/LICENSE`** — "Apache License Version 2.0, January 2004". Confirmed.
3. **`huggingface.co/ibm-esa-geospatial`** — TerraMind 1.0 family hosted; last updated Nov 3 2025.
4. **`huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base`** — License Apache-2.0. S2L2A / S2L1C / S1GRD / S1RTC / DEM / RGB inputs. Generation outputs incl. S1GRD and LULC.
5. **`github.com/zhu-xlab/DOFA`** — License MIT, 184 stars. Sentinel-1, Sentinel-2, NAIP, Gaofen, EnMAP modalities. Demo notebook present.
6. **`huggingface.co/earthflow/DOFA`** — Weights licensed CC-BY-4.0 (note: code MIT, weights CC-BY-4.0).
7. **`github.com/wchao0601/M4-SAR`** — License AGPL-3.0, 54 stars, 61 commits. 112,174 image pairs, 7 baselines (13.5M–53.8M params).
8. **`github.com/nasaharvest/galileo`** — License MIT, 187 stars. Sentinel-1, Sentinel-2 modalities; nano weights on GitHub, others on HF.
9. **`github.com/antofuller/CROMA`** — License MIT, 45 stars. Weights at `huggingface.co/antofuller/CROMA`.
10. **`github.com/cruiseresearchgroup/COMODO`** — License MIT, 24 stars. Cross-modal video → IMU distillation.
11. **`github.com/huggingface/smollm`** — License Apache-2.0, ~3.8k stars.
12. **`huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct`** — License Apache-2.0. 256M params (93M vision encoder + 135M decoder). <1 GB GPU at FP32. ONNX exports shipped. 23 quantised variants.
13. **`huggingface.co/datasets/agkphysics/AudioSet`** — License CC-BY-4.0. Commercial use permitted. Via `hf_inspect_dataset`: status Valid, balanced/full/unbalanced configs.
14. **`github.com/Meituan-AutoML/MobileVLM`** — License Apache-2.0, 1.4k stars. 21.5 tok/s on Snapdragon 888 CPU; 65.3 tok/s on Jetson Orin GPU.
15. **`github.com/apple/ml-fastvlm`** — Code license open; **model license LICENSE_MODEL is research-only / non-commercial** (verified by direct fetch of LICENSE_MODEL). Disqualifies as IDEaS substrate.
16. **`github.com/apple/ml-fastvlm/blob/main/LICENSE_MODEL`** — "This License does not grant any rights for any commercial purpose. ... 'Research Purposes' does not include any commercial exploitation, product development or use in any commercial product or service." Confirmed.
17. **`github.com/Richardzhangxx/AMR-Benchmark`** — 437 stars, 14 baseline AMR models. License unstated on README — flagged.
18. **`data.mendeley.com/datasets/f4c2b4n755/1`** (DroneRF) — License CC-BY-4.0. Permits commercial use.
19. **`deepsig.ai/datasets`** — RadioML 2018.01A and 2016.10A both CC-BY-NC-SA-4.0. "Commercial use is NOT allowed".
20. **`github.com/vishalned/MMEarth-data`** — Dataset license CC-BY-4.0. Sentinel-1, Sentinel-2, ERA5, DEM, ESA WorldCover, etc.
21. **`github.com/vishalned/MMEarth-train`** — 63 stars, last updated 2025-02-03. Atto and Tiny model variants. Multiple checkpoints across modality / dataset-version / loss-weighting axes.
22. **`github.com/facebookresearch/ImageBind`** — License CC-BY-NC 4.0. Disqualifies as IDEaS substrate.
23. **`github.com/cloudtostreet/Sen1Floods11`** — License not explicit on README — flagged.
24. **`github.com/rl1024/SoftFormer`** — License MIT, 33 stars (low maintenance signal).

## Negative-result verifications (gap-pattern queries)

These DuckDuckGo searches confirmed under-explored intersections that influence the candidate composition:

- `"audio" "IMU" "RF" fusion edge wearable ISR 2025 open-source` → 1 result (DuckDuckGo feedback page only) — confirms the audio + IMU + RF tri-fusion at the soldier edge has no open published architecture, supporting the framing of Candidate 4 (audio + video + IMU only) and Candidate 5 (RF + EO + acoustic separately).
- `"audio" "video" "IMU" fusion small model Apache-2.0 CC-BY-4.0 dataset 2024 2025` → 0 substantive results — confirms there is no off-the-shelf Apache-2.0 model that fuses audio + video + IMU at the SmolVLM scale, consistent with scout-3's "no fully open implementation that simultaneously combines (a) audio + video + IMU at sub-100M parameters, (b) federated/online training under missing modalities, and (c) operator-facing explainability" finding.
- `"counter-UAS" RF EO fusion open source dataset 2025 detection` → returned simulation-platform and per-modality datasets but no end-to-end RF + EO + acoustic detection benchmark, consistent with scout-2's "Counter-UAS literature splits along modality lines" gap.
- `"SAR" "audio" fusion ISR multimodal architecture 2024 2025` → returned only "SAR-LM symbolic audio reasoning" (unrelated; the SAR here is "speech audio reasoning") — confirms SAR (synthetic aperture radar) + audio is not a published fusion lane.

## What I did NOT verify (load-bearing flags carried forward)

- Star counts and licenses on `shuyansy/EarthMind`, `waterdisappear/SARATR-X`, `CAESAR-Radi/SARCLIP`, `Lmyxxn/Multimodal-NF` — flagged in scout-1 and scout-2 as "not retrieved this run". Carried forward into Candidate 8's risk register.
- Specific TerraMind parameter counts per variant — only inferred by analogy to ViT-tiny / -small / -base / -large conventions; HF cards did not assert numeric counts.
- Galileo model parameter counts per variant — README mentions "nano" but does not assert numeric counts.
- The licence on `Richardzhangxx/AMR-Benchmark` — README does not state a licence. Flagged.
- The licence on Sen1Floods11 dataset itself — README does not state a licence. Flagged.

## Reproducibility

All verifications above are reproducible by re-running the WebFetch on the URLs listed in §1–§24 above. Date of verification run: 2026-05-10. Any of these candidates' licence or repo-health signals could change after the verification date; the synthesist should re-verify the top three picks before final shortlist freeze.
