# Scout-1 Annotated Bibliography — EO/IR + SAR Fusion

## 1. Scope

Cross-modal fusion of electro-optical / infrared (EO/IR) imagery with synthetic aperture radar (SAR) for satellite and airborne ISR applications, focusing on 2024–2026 work covering cross-attention transformers, joint EO–SAR encoders, contrastive pretraining, and spatiotemporal alignment under cloud cover, night-time, and other degraded conditions. Also explicitly relevant to: Arctic ISR (sea-ice / open-water under cloud and polar night), maritime visual + SAR ship recognition, and airborne stealth/spoof detection involving radar with EO/IR.

**Narrowing decisions made:**
- Excluded SAR-only foundation models *unless* they are precursors that anchor a cross-modal stack (kept SARCLIP and SARATR-X because they appear as the "SAR-side" half of paired EO/SAR pipelines). Single-sensor SAR detection work without an explicit EO bridge is left to other scouts and reviews.
- Excluded EO-only foundation models (e.g. Prithvi-EO-2.0) unless they have a published SAR pathway. Multi-sensor models (DOFA, Galileo, TerraMind, SkySense, RingMoE, MMEarth, CROMA) explicitly carry SAR+optical modalities and are kept.
- Allowed two pre-2024 anchors (CROMA, SkySense) because they are the canonical reference for contrastive radar-optical pretraining and multimodal RS foundation models respectively, and 2024–2026 work cites them as the baseline.
- Airborne stealth/spoof and EO/IR multi-sensor papers (UAV identification, spoof tracking) are kept because the assignment names this as an application affinity even though the radar in question is more often dismount-radar than SAR proper.

## 2. Key papers

### 2a. EO–SAR foundation / pretraining models (joint encoders, contrastive, masked-AE)

1. **CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders** — Fuller, Millard, Green — arXiv: **2311.00566** — NeurIPS 2023. Pretrains on Sentinel-1 + Sentinel-2 with a dual-encoder + masked autoencoder + cross-modal contrastive objective; introduces spatial-attention adjustments that let pretrained models scale to larger images at inference. *Why it matters:* CROMA is the canonical contrastive radar-optical pretraining baseline cited by virtually every 2024–2026 EO/SAR fusion paper; the 2024 ecosystem still measures itself against CROMA's frozen-encoder probes.

2. **DOFA — Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation** — Xiong, Wang, Zhang et al. (TUM) — arXiv: **2403.15356** — 2024. A wavelength-conditioned dynamic hypernetwork lets a single backbone ingest Sentinel-1 SAR, Sentinel-2 multispectral, NAIP RGB, Gaofen, and EnMAP hyperspectral; pretrained on five EO modalities and shown to generalize to unseen sensors. *Why it matters:* DOFA's "one-for-all" design is the most flexible 2024 SAR+EO encoder publicly available with weights, and it is the top candidate as a frozen backbone for an Arctic ISR proposal that may need to swallow heterogeneous Canadian sensor data (e.g. RCM SAR + Sentinel-2 + WorldView).

3. **MMEarth: Exploring Multi-Modal Pretext Tasks for Geospatial Representation Learning** — Nedungadi et al. — arXiv: **2405.02771** — ECCV 2024. Releases a 1.2M-location global pretraining dataset pairing Sentinel-1, Sentinel-2 L1C, Sentinel-2 L2A, ERA5, biome, climate-zone, elevation, and land-cover; trains a Multi-Pretext Masked Autoencoder that beats ImageNet- and SatMAE-pretrained MAEs on downstream tasks. *Why it matters:* The MMEarth release pairs an open-source SAR+optical pretraining corpus with a published recipe, removing one of the biggest blockers for groups starting cold in EO/SAR fusion.

4. **TerraMind: Large-Scale Generative Multimodality for Earth Observation** — Jakubik et al. (IBM + ESA) — arXiv: **2504.11171** — ICCV 2025. Any-to-any generative foundation model spanning nine geospatial modalities (S-1 GRD, S-1 RTC, S-2 L1C/L2A, DEM, LULC, NDVI, S-2 RGB, captions); introduces "Thinking-in-Modalities" that synthesises intermediate modalities at inference. *Why it matters:* TerraMind is the first foundation model to demonstrate generative cross-modal completion (e.g. synthesise plausible optical from SAR when clouds are present) at scale with a permissive licence — directly relevant to spatiotemporal alignment under cloud / polar-night conditions.

5. **Galileo: Learning Global & Local Features of Many Remote Sensing Modalities** — Tseng, Cartuyvels et al. (NASA Harvest) — arXiv: **2502.09356** — ICML 2025. Multimodal transformer trained with dual global+local contrastive losses on optical, SAR, elevation, and weather time-series; outperforms specialist models across 11 benchmarks including crop mapping and flood detection. *Why it matters:* Galileo is the strongest 2025 baseline for *time-series* EO/SAR fusion (the spatiotemporal alignment leg of the IDEaS spec), and weights/code are fully open.

6. **SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery** — Guo et al. (Wuhan / Ant Group) — arXiv: **2312.10115** — CVPR 2024. Billion-scale model pretrained on 21.5M temporal sequences combining Sentinel-1, Sentinel-2 and high-res optical with a factorized spatiotemporal encoder, multi-granularity contrastive learning, and geo-context prototypes. *Why it matters:* SkySense remains the highest-capacity public EO/SAR encoder and is the natural ceiling reference; weights are non-commercial-research-only, which is itself a YAGNI flag for any commercial proposal.

7. **RingMoE: Mixture-of-Modality-Experts Multi-Modal Foundation Models for Universal Remote Sensing Image Interpretation** — Bi et al. — arXiv: **2504.03166** — 2025. 14.7B-parameter MoE foundation model with modality-specific experts spanning optical, SAR, and multi-spectral; reports SoTA on 23 benchmarks and demonstrates 1B-parameter compressed deployment variant. *Why it matters:* RingMoE shows that a single MoE backbone can host EO and SAR experts simultaneously with controllable inference cost — relevant to SWaP-aware edge deployment trade-off conversations.

### 2b. Cross-attention / transformer fusion architectures

8. **M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR Fusion Object Detection** — Wang, Lu, Li, Yang, Luo — arXiv: **2505.10931** — 2025. Introduces 112,184 precisely aligned optical-SAR image pairs (~1M oriented instances, 6 categories) plus a benchmarking toolkit integrating six SoTA fusion methods, plus E2E-OSDet, an end-to-end fusion detection framework. *Why it matters:* M4-SAR is the first standardized, large-scale optical-SAR fusion *detection* benchmark — without it, every fusion paper was reporting numbers on its own splits. Fully open and HF/Kaggle hosted.

9. **EarthMind: Leveraging Cross-Sensor Data for Advanced Earth Observation Interpretation with a Unified Multimodal LLM** — Shu, Ren, Xiong et al. (Trento + TUM + Berlin) — arXiv: **2506.01667** — 2025. Vision-language MLLM with hierarchical cross-modal attention (HCA) that fuses optical and SAR features and aligns them with language queries; ships FusionEO (30K pairs, multi-task annotations) and EarthMind-Bench (2,841 pairs, expert-annotated). *Why it matters:* This is one of the very few open multimodal LLMs that explicitly accepts SAR as an input modality and produces operator-readable language outputs — directly relevant to the IDEaS explainability dimension.

10. **SoftFormer: SAR-optical fusion transformer for urban land use and land cover classification** — Liu, Ling, Zhang (HKU) — ISPRS Journal of Photogrammetry & Remote Sensing, vol. 218, 2024 (no arXiv). Shifted-window transformer with a soft-fusion module that mixes SAR and optical features at multiple stages; reports SoTA on Sentinel-1/2 urban classification. *Why it matters:* Provides the most concrete, code-released cross-attention recipe for SAR+optical and is widely cited as the reference architecture for transformer-based fusion in 2024–2025; open-source PyTorch implementation. (Cited via DOI / journal — spec permits non-arXiv when canonical and stable; flagged in verification as non-arXiv.)

11. **M²CD: A Unified MultiModal Framework for Optical-SAR Change Detection with Mixture of Experts and Self-Distillation** — Liu et al. — arXiv: **2503.19406** — 2025. Adds Mixture-of-Experts modules into a Siamese change-detection backbone plus an "Optical-to-SAR guided" path and self-distillation to bridge the modality gap. *Why it matters:* Change detection is the IDEaS-relevant operationalisation of "what's different now under cloud cover?"; M²CD is the cleanest 2025 MoE recipe for cross-modal change tasks.

12. **M³amba: CLIP-driven Mamba Model for Multi-modal Remote Sensing Classification** — Cao et al. — arXiv: **2503.06446** — 2025. Replaces self-attention with a Mamba state-space backbone, conditions modality adapters on CLIP text features, and runs across hyperspectral + SAR + RGB; reports ≥5.98% average gain over SoTA on multi-modal classification. *Why it matters:* Mamba's linear-time scaling is a SWaP-aware alternative to quadratic cross-attention — relevant when the proposal needs an edge-deployable encoder.

### 2c. Cross-modal alignment, registration, translation, language-bridged

13. **Multi-Resolution SAR and Optical Remote Sensing Image Registration: A Review, Datasets, and Future Perspectives** — Zhang et al. — arXiv: **2502.01002** — 2025. Releases the MultiResSAR dataset (>10K cross-resolution pairs), benchmarks 16 SoTA registration methods, and identifies that performance degrades sharply with increasing resolution mismatch. *Why it matters:* Cleanly maps the spatiotemporal-alignment failure modes that any EO/SAR ISR pipeline will hit; explicitly recommends noise suppression and 3D geometric fusion as open problems.

14. **CLOSP: A Unified Semantic Space for SAR, MSI, and Text in Remote Sensing** — Cambrin, Vaiani et al. — arXiv: **2507.10403** — 2025. Aligns Sentinel-1, Sentinel-2, and free-text into a shared semantic space with text as the connector; releases GeoCLOSP variants and shows large gains on cross-modal retrieval. *Why it matters:* Text-bridged alignment offers a path around paired-data scarcity (the dominant blocker for EO+SAR pretraining outside Sentinel) and gives operators a query-by-text interface — explainability-relevant.

15. **SARCLIP: A Vision Language Foundation Model for Semantic Understanding and Target Recognition in SAR Imagery** — Ma, Wang, Liu et al. — arXiv: **2510.22665** — 2025. SAR-specific CLIP variant trained on SARCLIP-1M (>1M SAR–text pairs) with a domain-transfer training strategy; SoTA zero-shot SAR classification, retrieval, captioning, and semantic localisation. *Why it matters:* Provides the SAR half of any paired SAR-EO-text retrieval pipeline; pairs naturally with CLOSP / CLIP-style optical foundation models.

16. **SARATR-X: Toward Building A Foundation Model for SAR Target Recognition** — Li, Yang et al. — arXiv: **2405.09365** — 2024. Self-supervised foundation model for SAR target recognition built on top of HiViT and trained with masked image modelling on a curated SAR corpus. *Why it matters:* The de facto SAR-side encoder used to bootstrap SAR-EO fusion stacks where one wants to freeze a strong unimodal SAR backbone before stitching in optical features.

17. **Accelerating Diffusion for SAR-to-Optical Image Translation via Adversarial Consistency Distillation** — Bai et al. — arXiv: **2407.06095** — 2024. Distils a diffusion SAR→optical translator with consistency + adversarial losses; reports a ~131× inference speedup with preserved quality on SEN12 / GF3. *Why it matters:* SAR-to-optical translation under cloud is an alternative to direct fusion; this paper makes that pathway tractable on the edge by collapsing the diffusion steps.

### 2d. Cross-modal Re-ID and downstream maritime / disaster / Arctic applications

18. **Cross-modal Ship Re-Identification via Optical and SAR Imagery: A Novel Dataset and Method** — Wang et al. — arXiv: **2506.22027** — ICCV 2025. Releases the HOSS ReID dataset (LEO optical + SAR ship pairs) and proposes TransOSS, a ViT that learns modality-invariant features via contrastive learning. *Why it matters:* The first open cross-modal ship Re-ID benchmark — directly relevant to maritime task-group anomaly detection in the IDEaS spec, where the same hull may be observed by RADARSAT-2 and a separate optical pass.

19. **IceDiff: High Resolution and High-Quality Sea Ice Forecasting with Generative Diffusion Prior** — Xu et al. — arXiv: **2410.09111** — 2024. Two-stage forecasting framework: a vision transformer produces a coarse sea-ice prediction, then a diffusion prior super-resolves it to 6.25 km × 6.25 km; first method to deliver that resolution operationally. *Why it matters:* Arctic sea-ice forecasting is the canonical Arctic ISR fusion task; IceDiff sits at the SAR + multispectral + reanalysis junction and is one of the few open 2024 baselines with weights.

20. **Leveraging AI multimodal geospatial foundation models for improved near-real-time flood mapping at a global scale** — Tulbure et al. — arXiv: **2512.02055** — 2025. Fine-tunes TerraMind on 85 global flood events using Sentinel-1 SAR + Sentinel-2 optical; finds that multimodal foundation models *can* enhance NRT flood mapping but classical CNNs remain competitive on certain metrics. *Why it matters:* The most rigorous 2025 evaluation of when EO/SAR foundation models actually help vs. when a small custom CNN suffices — important YAGNI evidence for any TRL-4 proposal.

### 2e. UAV / airborne stealth-spoof multimodal sensor fusion (radar + EO/IR)

21. **Intelligent Multimodal Multi-Sensor Fusion-Based UAV Identification, Localization, and Countermeasures for Safeguarding Low-Altitude Economy** — Tao et al. — arXiv: **2510.22947** — 2025. Combines RF spectral analysis, radar detection, and EO identification in an end-to-end UAV management framework with both soft-kill and hard-kill countermeasures. *Why it matters:* Direct analogue for the IDEaS "airborne stealth/spoof detection (radar + EO/IR + telemetry)" application; provides a baseline for what a multi-sensor fusion stack looks like in 2025.

22. **SpoofTrackBench: Interpretable AI for Spoof-Aware UAV Tracking and Benchmarking** — Le and Le — arXiv: **2510.22726** — 2025. A reproducible benchmark for testing adversarial robustness of localization/tracking under radar spoofing, with two tracker baselines and visualisations of how spoofing distorts trajectories. *Why it matters:* The only 2025 open spoof-aware tracking benchmark; gives a concrete adversarial evaluation harness for the airborne stealth/spoof IDEaS context.

## 3. Datasets

| Dataset | Identifier | Modalities | Licence | Note |
|---|---|---|---|---|
| **M4-SAR** | github.com/wchao0601/M4-SAR (HF + Kaggle + Baidu mirrors); paper arXiv:2505.10931 | Sentinel-1 SAR (VV+VH) + Sentinel-2 optical, 112K paired patches, ~1M oriented instances, 6 classes | Released for research; specific licence not asserted in repo readme — **flag** | First standardized optical-SAR fusion detection benchmark |
| **MMEarth** | vishalned.github.io/mmearth ; HF dataset name **earthflow/MMEarth** ; paper arXiv:2405.02771 | Sentinel-1 GRD + Sentinel-2 L1C/L2A + ERA5 + biome + climate + land-cover + DEM, 1.2M global locations | CC-BY-4.0 (per project page) | Largest open multimodal pretraining corpus with SAR included |
| **SEN12MS** | TUM portal; mirrored at multiple HF repos including torchgeo/sen12ms (inspector reports schema issues — **flag**, fall back to upstream TUM mirror) | Sentinel-1 SAR (dual-pol) + Sentinel-2 (13 bands) + MODIS land-cover, 180K triples | CC-BY-4.0 | Canonical SAR+optical paired benchmark, used by virtually every 2024 fusion paper |
| **SEN1-2** | arXiv:1807.01569 (paper); various GitHub mirrors | Sentinel-1 SAR + Sentinel-2 optical, ~282K paired patches | CC-BY-SA-4.0 | Older but canonical paired SAR-optical for translation tasks |
| **HOSS ReID** | Released with arXiv:2506.22027; hosted by paper authors | Optical + SAR LEO ship pairs | Research-only; specific licence in repo — **flag** | First cross-modal ship Re-ID dataset |
| **FusionEO + EarthMind-Bench** | Released with arXiv:2506.01667; github.com/shuyansy/EarthMind | Optical + SAR pairs with VL annotations (30K + 2,841 expert-annotated) | Research; licence in repo — **flag** | First cross-sensor MLLM benchmark |
| **MultiResSAR** | Released with arXiv:2502.01002 | Multi-resolution SAR + optical pairs (~10K) | Research; specific licence in repo — **flag** | Targeted at SAR-optical *registration* under resolution mismatch |
| **DFC2023** | grss-ieee.org Data Fusion Contest 2023; CodaLab competition 8987 | Sentinel-1 SAR + RGB optical + DSM, building footprints | IEEE GRSS contest licence — **flag** (research only) | Canonical multimodal building extraction + height fusion benchmark |
| **DynamicEarthNet** | arXiv:2203.12560; via GEO-Bench-2 | Daily Planet multispectral, monthly Sentinel-1 + Sentinel-2 over 75 sites | CC-BY-NC-SA — **flag** (non-commercial) | Daily multi-spectral with weekly SAR; canonical dense-time-series benchmark |
| **BigEarthNet (v1/v2)** | HF: blanchon/BigEarthNet (inspector reports schema issues — **flag**, fall back to bigearth.net upstream); v2 also released as BigEarthNet-MM with S-1 + S-2 paired tiles | Sentinel-1 SAR + Sentinel-2 multispectral, 590K patches, 19 classes | CC-BY-4.0 | Largest open multilabel SAR+optical land-cover benchmark |
| **SARCLIP-1M** | github.com/CAESAR-Radi/SARCLIP; paper arXiv:2510.22665 | SAR + text pairs, >1M | Research; specific licence in repo — **flag** | SAR-side text-pair pretraining corpus, complement to optical-text corpora |

## 4. Reference implementations

| Repo | Paper | Stars | Licence | What it implements |
|---|---|---|---|---|
| antofuller/CROMA | arXiv:2311.00566 | 45 | MIT | Dual encoder + masked AE + contrastive Sentinel-1 / Sentinel-2 pretraining; pretrained base+large weights on HF |
| zhu-xlab/DOFA | arXiv:2403.15356 | 184 | MIT | Wavelength-conditioned dynamic hypernetwork over multiple EO sensors including S-1; HF weights + TorchGeo integration |
| IBM/terramind | arXiv:2504.11171 | 258 | Apache-2.0 | Any-to-any generative EO model across 9 modalities incl. S-1 GRD/RTC and S-2; integrates with TerraTorch |
| Jack-bo1220/SkySense | arXiv:2312.10115 | 68 | Non-commercial research-only | Factorized spatiotemporal encoder for optical+SAR; example detection + segmentation pipelines |
| nasaharvest/galileo (per ICML 2025 / OpenReview) | arXiv:2502.09356 | (open-sourced via NASA; star count not retrieved) | Apache-2.0 (per project page) | Multi-modal masked / contrastive pretraining over optical, SAR, DEM, weather time-series |
| wchao0601/M4-SAR | arXiv:2505.10931 | 54 | Research; check repo | E2E-OSDet + 6 baseline fusion detectors; benchmarking toolkit |
| shuyansy/EarthMind | arXiv:2506.01667 | (released with paper; star count not retrieved this run) | Research; check repo | Hierarchical Cross-modal Attention (HCA) MLLM over optical + SAR; FusionEO + EarthMind-Bench |
| rl1024/SoftFormer | ISPRS-JPRS 218, 2024 | (small, single-author repo; star count not retrieved this run) | MIT (per repo) | Shifted-window transformer + soft-fusion module for SAR+optical urban classification |
| waterdisappear/SARATR-X | arXiv:2405.09365 | (active; star count not retrieved this run) | Research; check repo | SAR foundation model (HiViT MIM) suitable as the SAR leg of an EO/SAR stack |
| CAESAR-Radi/SARCLIP | arXiv:2510.22665 | (released with paper; star count not retrieved this run) | Research; check repo | SAR-text contrastive pretraining + SARCLIP-1M release |
| Zhaozixiang1228/MMIF-DDFM | arXiv:2303.06840 (DDFM, ICCV 2023 oral) | (canonical for diffusion-based multi-modal fusion; star count not retrieved) | MIT (per repo) | Diffusion-based multi-modality image fusion (infrared+visible) — relevant as canonical fusion baseline |
| AICyberTeam/DFC2023-baseline | DFC2023 contest (no arXiv) | (research; star count not retrieved) | Research | Baseline for multimodal building extraction fusion contest |

## 5. Open questions you noticed

These are gaps observed while reading; they are not hypotheses or proposed experiments — they are flagged for the gap-finder.

- **Cross-modal pretraining is dominated by Sentinel-1/2 collocations.** Almost every 2024–2026 EO/SAR foundation model (CROMA, DOFA, Galileo, MMEarth, TerraMind, SkySense, RingMoE) trains on the same Sentinel pair. Whether these encoders transfer to Canadian RADARSAT-Constellation Mission (RCM) C-band quad-pol or to airborne X-band SAR (e.g. ICEYE micro-sats) is essentially untested in the open literature.

- **Spatiotemporal alignment is the bottleneck under cloud / polar night.** Zhang et al. (2025, arXiv:2502.01002) explicitly flag that registration accuracy collapses with resolution mismatch; nothing in the 2024–2026 fusion literature has *jointly* trained a model that handles both registration uncertainty and downstream task uncertainty.

- **Uncertainty propagation across modalities is treated as an afterthought.** None of the SoTA EO/SAR foundation papers surveyed (CROMA, DOFA, MMEarth, Galileo, TerraMind, SkySense, RingMoE) reports calibrated cross-modal uncertainty. Tulbure et al. (2025, arXiv:2512.02055) is one of the only papers to discuss when foundation-model fusion *is* and *isn't* worth the cost — and they don't propagate uncertainty either.

- **Generative cross-modal completion vs. discriminative cross-modal fusion is bifurcated.** TerraMind's any-to-any thinking-in-modalities is the only public model that lets one synthesise SAR-from-optical or optical-from-SAR at scale; the discriminative fusion pipelines (EarthMind, M4-SAR, SoftFormer, M²CD) treat this as out of scope. No open work surveyed jointly evaluates whether synthesised SAR (from generative completion) beats real SAR for downstream detection in cloudy regions.

- **Maritime + Arctic + airborne stealth/spoof are siloed in the open literature.** HOSS ReID (maritime ship), IceDiff (Arctic sea ice), and SpoofTrackBench / Tao et al. 2025 (airborne UAV) all live in different sub-communities with different evaluation protocols; no shared multi-context EO/SAR/RF/IR benchmark exists.

- **Operator-facing explainability of cross-modal models is barely addressed.** Only EarthMind (arXiv:2506.01667) and CLOSP (arXiv:2507.10403) explicitly produce language-mediated outputs for SAR+optical fusion; the heavy-weight foundation models (CROMA, DOFA, SkySense, TerraMind, Galileo, RingMoE) report patch-level metrics with no published probe that maps internal cross-modal attention to operator-readable rationales.

- **SWaP-aware deployment is mostly aspirational.** RingMoE claims a 1B compressed variant; Mamba-based M³amba claims linear scaling; no surveyed paper reports real on-device inference numbers for an EO+SAR fused detector in a power envelope characteristic of an airborne or tactical edge device.

- **Reference-implementation health is uneven.** The strongest numerical baselines (SkySense, RingMoE) carry the most restrictive licences; the most permissively licenced models (DOFA-MIT, TerraMind-Apache-2.0, CROMA-MIT, MMIF-DDFM-MIT) are 2024-class and have less capacity. There is no high-capacity Apache-2.0 EO+SAR foundation model with calibrated cross-modal uncertainty as of 2026-05-10.

## 6. Sources

### arXiv IDs cited (verified via WebFetch on arxiv.org/abs and/or huggingface.co/papers, see verification.md)

- 2311.00566  (CROMA)
- 2403.15356  (DOFA)
- 2405.02771  (MMEarth)
- 2504.11171  (TerraMind)
- 2502.09356  (Galileo)
- 2312.10115  (SkySense)
- 2504.03166  (RingMoE)
- 2505.10931  (M4-SAR)
- 2506.01667  (EarthMind)
- 2503.19406  (M²CD)
- 2503.06446  (M³amba)
- 2502.01002  (Multi-resolution SAR-optical registration review)
- 2507.10403  (CLOSP)
- 2510.22665  (SARCLIP)
- 2405.09365  (SARATR-X)
- 2407.06095  (SAR→optical diffusion distillation)
- 2506.22027  (HOSS ReID / TransOSS)
- 2410.09111  (IceDiff)
- 2512.02055  (TerraMind flood-mapping eval)
- 2510.22947  (UAV multimodal identification)
- 2510.22726  (SpoofTrackBench)
- 2303.06840  (DDFM, canonical multi-modality fusion baseline)
- 2203.12560  (DynamicEarthNet)
- 1807.01569  (SEN1-2 dataset)

### Non-arXiv canonical references (flagged in verification)

- Liu, Ling, Zhang. *SoftFormer: SAR-optical fusion transformer for urban land use and land cover classification.* ISPRS Journal of Photogrammetry & Remote Sensing, vol. 218, 2024. (DOI via journal; code at github.com/rl1024/SoftFormer)
- IEEE GRSS Data Fusion Contest 2023 (DFC2023): grss-ieee.org/community/technical-committees/2023-ieee-grss-data-fusion-contest/

### Repos cited

- github.com/antofuller/CROMA
- github.com/zhu-xlab/DOFA
- github.com/IBM/terramind
- github.com/Jack-bo1220/SkySense
- github.com/wchao0601/M4-SAR
- github.com/shuyansy/EarthMind
- github.com/rl1024/SoftFormer
- github.com/waterdisappear/SARATR-X
- github.com/CAESAR-Radi/SARCLIP
- github.com/Zhaozixiang1228/MMIF-DDFM
- github.com/AICyberTeam/DFC2023-baseline
- github.com/AutoLab-SAI-SJTU/MambaFusion (referenced incidentally)

### HF dataset / model handles cited

- earthflow/DOFA  (HF model)
- earthflow/MMEarth  (HF dataset)
- DarthReca/CLOSP-Visual  (HF model, GeoCLOSP-RN variant)
- MBZUAI/TerraFM  (HF model, mentioned during search but not used as primary citation)
- blanchon/BigEarthNet  (HF dataset; inspector flagged schema issue, fall back to bigearth.net)
- torchgeo/sen12ms  (HF dataset; inspector flagged schema issue, fall back to TUM mirror)
