# Scout-5 Annotated Bibliography — Sonar + Maritime Multi-Modal Anomaly Detection

## 1. Scope

Survey of 2024–2026 deep-learning literature on maritime fusion across at least two of {sonar, AIS, RF, EO/IR, SAR, telemetry, chart}, with an emphasis on (a) hydroacoustic ATR, (b) AIS+vision/SAR fusion for dark-vessel and trajectory-anomaly detection, and (c) explainability and uncertainty scoring relevant to a Maritime Task Group anomaly-detection use case.

Narrowing decisions:
- **Acoustic-only RF emitter ID** is largely covered by other scouts; only one canonical RF/SEI paper retained because it is the only one that explicitly uses an AIS RFF dataset.
- Since true maritime *quad-modal* (sonar + RF + visual + AIS) systems are essentially absent in the open literature, this bibliography intentionally over-samples each pairwise combination (acoustic-optical, AIS-visual, AIS-SAR, AIS-LLM, optical-SAR-AIS triples) so the gap-finder has the building blocks for the missing intersections.
- xView3-SAR (2022) is included because it is the canonical open benchmark every 2024–2026 SAR/AIS dark-vessel paper still trains on.
- DeepSORVF/FVessel (Feb 2023) is included for the same canonical reason — almost every 2024+ AIS-visual fusion paper benchmarks on it.

## 2. Key papers

### 2A. Multimodal maritime fusion (≥2 sensors)

**Multimodal and Multiview Deep Fusion for Autonomous Marine Navigation**
- arXiv: `2505.01615` · 2025 · Dagdilelis, Grigoriadis, Galeazzi
- Cross-attention transformer fuses multiview RGB + LWIR + sparse LiDAR into a BEV scene; X-band radar and ENC chart used as training-time auxiliaries. Validated on real sea trials in adverse weather.
- *Why it matters:* one of the very few open papers doing >2 fused maritime modalities at the architecture level; demonstrates the BEV-as-fusion-target pattern that anomaly detectors could overlay.

**Real-Time Fusion of Visual and Chart Data for Enhanced Maritime Vision**
- arXiv: `2507.13880` · 2025 · Kreis, Kiefer
- Transformer end-to-end network detects buoys in video, queries chart positions, predicts confidence-scored matches. Beats ray-casting and YOLOv7 baselines on real coastal video.
- *Why it matters:* unusual modality pair (visual + symbolic chart) and explicitly outputs per-detection confidence — a building block for explainable anomaly scoring.

**Graph Learning-Driven Multi-Vessel Association: Fusing Multimodal Data for Maritime Intelligence (GMvA)**
- arXiv: `2504.09197` · 2025 · Lu, Yang, Yang, Ding, Weng, Liu
- Graph-attention model fuses AIS + CCTV vessel trajectories with a dedicated MLP-based **uncertainty fusion module** (MLP-UMF) and Hungarian-algorithm matching. Reports robustness in dense traffic and missing-data regimes.
- *Why it matters:* first 2025 work to wire uncertainty propagation directly into the multimodal association step — directly relevant to the spec's uncertainty-propagation IDEaS dimension.

**Asynchronous Trajectory Matching-Based Multimodal Maritime Data Fusion (DeepSORVF) + FVessel benchmark**
- arXiv: `2302.11283` · 2023 (canonical, kept) · Guo, Liu, Qu, Lu, Zhu, Lv (IEEE TITS 2023)
- DL framework that matches asynchronous AIS and video tracks under occlusion in inland waterways; releases FVessel dataset (26 videos + paired AIS, multiple weather).
- *Why it matters:* the canonical open benchmark for AIS-visual fusion that nearly all 2024+ inland/coastal fusion papers benchmark against.

**Enhancing Maritime Domain Awareness on Inland Waterways: A YOLO-Based Fusion of Satellite and AIS for Vessel Characterization**
- arXiv: `2510.11449` · 2025 · Agorku, Hernandez, Hames, Wagner
- Satellite optical + AIS fusion with YOLOv11 on 5,973 sq mi of Lower Mississippi; classifies vessel type, barge attributes, status, direction (F1 > 91%); flags AIS-discrepant ("non-cooperative") vessels.
- *Why it matters:* concrete worked example of using AIS+optical to surface "AIS-off" anomalies, and reports spatial-transferability metrics (98%) — directly aligned with anomaly use case.

### 2B. Sonar / Hydroacoustic deep learning

**Sonar-based Deep Learning in Underwater Robotics: Overview, Robustness and Challenges**
- arXiv: `2412.11840` · 2024 · Aubard, Madureira, Teixeira, Pinto
- First comprehensive survey of sonar-based DL through the lens of robustness — covers classification, detection, segmentation, SLAM, datasets, simulators, NN verification, adversarial attacks. Explicitly identifies sim-to-real and dataset scarcity as open gaps.
- *Why it matters:* the canonical 2024 systematization of sonar-DL; cites every dataset and method a maritime fusion candidate would need to baseline against.

**Sonar Image Datasets: A Comprehensive Survey of Resources, Challenges, and Applications**
- arXiv: `2510.03353` · 2025 · Gomes, Almeida, Moreira, Quiroz, Xavier, Soares, Brião, Oliveira, Drews-Jr (SIBGRAPI 2025)
- Catalogues open SSS / FLS / SAS / MBES datasets through 2025 with annotation details and a chronological timeline. Identifies still-missing dataset types.
- *Why it matters:* one-stop dataset map for the gap-finder when scoring SWaP/data-availability axes; flags that no public dataset combines sonar with AIS.

**Synth-SONAR: Sonar Image Synthesis with Enhanced Diversity and Realism via Dual Diffusion Models and GPT Prompting**
- arXiv: `2410.08612` · 2024 · Natarajan, Basha, Nambiar
- Hierarchical dual-diffusion architecture with GPT-conditioned coarse/fine prompts produces diverse synthetic sonar imagery; first GPT-prompted sonar synthesis system.
- *Why it matters:* enables synthetic sonar augmentation under the spec's "open or synthetic data only" constraint; pairs naturally with edge-deployable detectors.

**Syn2Real Domain Generalization for Underwater Mine-like Object Detection Using Side-Scan Sonar**
- arXiv: `2410.12953` · 2024 · Agrawal, Sikdar, Makam, Sundaram, Besai, Gopi (IEEE 2025)
- DDPM/DDIM-generated synthetic SSS + real data improves Mask-RCNN AP by ~60 % over real-only training for mine-like objects.
- *Why it matters:* concrete recipe for the synthetic→real pipeline a sonar-anomaly TRL 4–5 build would rely on.

**Underwater object detection in sonar imagery with detection transformer and Zero-shot neural architecture search (NAS-DETR)**
- arXiv: `2505.06694` · 2025 · Gu, Tang, Cao, Yu
- DETR + zero-shot NAS over CNN-Transformer backbones using a max-entropy fitness; deformable-attention decoder; SOTA on standard sonar benchmarks at low compute.
- *Why it matters:* demonstrates SWaP-aware sonar ATR pathway — small backbones discovered without GPU NAS, directly relevant to edge constraints.

**Adversarial multi-task underwater acoustic target recognition: towards robustness against various influential factors**
- arXiv: `2411.02848` · 2024 · Xie, Xu, Ren, Li
- Multi-task adversarial learning that conditions UATR on auxiliary tasks (source range, depth, wind speed); SOTA on ShipsEar 12-class.
- *Why it matters:* explicitly models environmental influencing factors as separate auxiliary outputs — a pathway for explainability that says "I am uncertain because conditions look like X."

**Guiding the underwater acoustic target recognition with interpretable contrastive learning**
- arXiv: `2402.12658` · 2024 · Xie, Ren, Xu (OCEANS 2023-Limerick)
- Class-activation analysis shows recognition models attend to low-frequency line spectra and high-frequency periodic modulation; dual-encoder contrastive scheme improves cross-database generalization.
- *Why it matters:* the rare paper that explicitly looks under the hood of UATR explainability — a needed input for any operator-facing maritime anomaly system.

**Cross-Domain Knowledge Transfer for Underwater Acoustic Classification Using Pre-trained Models**
- arXiv: `2409.13878` · 2024 (v2 2025) · Mohammadi, Kelhe, Carreiro, Van Dine, Peeples
- Compares ImageNet vs. PANNs pre-training for passive sonar classification; ImageNet-pretrained models slightly outperform audio-pretrained PANNs; analyses sampling-rate effects.
- *Why it matters:* useful baseline-selection guidance when bootstrapping a sonar leg under no-GPU-spend scoping constraints.

**Automated data curation for self-supervised learning in underwater acoustic analysis**
- arXiv: `2505.20066` · 2025 · Hummel, Bhulai, Ghani, van der Mei
- Fully-automated SSL curation pipeline for passive acoustic monitoring; integrates ship-identity data with hydrophone recordings via hierarchical clustering.
- *Why it matters:* path to weak labels for an open-data sonar leg without manual annotation cost.

### 2C. AIS / SAR anomaly detection and trajectory modelling

**AIS-LLM: A Unified Framework for Maritime Trajectory Prediction, Anomaly Detection, and Collision Risk Assessment with Explainable Forecasting**
- arXiv: `2508.07668` · 2025 · Park, Jung, Seo, Choi, Cho, Park, Choi
- Fuses AIS time-series with an LLM via cross-modal alignment; multi-task decoder produces trajectory, anomaly score, collision risk and a natural-language rationale.
- *Why it matters:* combines anomaly detection with operator-facing textual explanations — the spec's explainability dimension is uncommon to find delivered end-to-end.

**AIS Data-Driven Maritime Monitoring Based on Transformer: A Comprehensive Review**
- arXiv: `2505.07374` · 2025 · Xie, Tu, Fu, Yuan, Han
- Survey of transformer-based AIS monitoring (trajectory prediction, behavior detection/prediction); curates and statistically analyses public AIS datasets.
- *Why it matters:* canonical 2025 entry-point for AIS-only methods, useful when comparing pure-AIS baselines vs. AIS-fused approaches.

**Outlier detection in maritime environments using AIS data and deep recurrent architectures**
- arXiv: `2406.09966` · 2024 · Maganaris, Protopapadakis, Doulamis
- Encoder-decoder RNN (best: bidirectional GRU with recurrent dropout) reconstructs AIS motion patterns; anomaly score = reconstruction error.
- *Why it matters:* a clean baseline for the "AIS-only" arm of any fusion ablation — needed under the spec's "compare against single-modality baseline" pattern.

### 2D. SAR / cross-modal SAR foundations and ship Re-ID

**xView3-SAR: Detecting Dark Fishing Activity Using Synthetic Aperture Radar Imagery**
- arXiv: `2206.00897` · 2022 (canonical, kept) · Paolo, Lin, Gupta, Goodman, Patel, Kuster, Kroodsma, Dunnmon
- Releases ~1,000 analysis-ready Sentinel-1 SAR scenes with 243k AIS-anchored annotations (vessel/non-vessel, length, fishing-vs-not), plus co-registered bathymetry and wind.
- *Why it matters:* the canonical open benchmark for SAR dark-vessel detection — every 2024+ paper in this space trains/evaluates on it.

**SARATR-X: Toward Building A Foundation Model for SAR Target Recognition**
- arXiv: `2405.09365` · 2024 (TIP 2025) · Li, Yang, Hou, Liu, Liu, Li
- Self-supervised foundation model trained on 0.18M unlabelled SAR samples across multiple benchmarks; covers ships, vehicles, aircraft; competitive with fully supervised baselines in few-shot regimes.
- *Why it matters:* the strongest open SAR backbone available — natural feature provider for any SAR leg of a fusion model under "open data only."

**Cross-modal Ship Re-Identification via Optical and SAR Imagery: A Novel Dataset and Method (HOSS ReID + TransOSS)**
- arXiv: `2506.22027` · 2025 (ICCV 2025) · Wang, Li, Yang, Liu, Lv, Zhou
- Releases HOSS ReID (paired optical-SAR ship images across satellites/times) and TransOSS, a ViT-based contrastive ReID baseline for modality-invariant features.
- *Why it matters:* directly enables the optical-SAR re-association step that any maritime task-group anomaly system needs to keep tracks consistent across passes.

**SMART-Ship: A Comprehensive Synchronized Multi-modal Aligned Remote Sensing Targets Dataset and Benchmark for Berthed Ships Analysis**
- arXiv: `2508.02384` · 2025 · Fan, Guo, Zhang, Qi, Huang, Mao, Suo, Jiang, Liu, He
- 1,092 synchronized 5-modality image sets (visible/SAR/panchromatic/multi-spectral/NIR) over 38,838 ships, week-bounded acquisition windows, polygon + change-mask + fine-grained class annotations.
- *Why it matters:* one of the most complete open multi-modal aligned ship datasets — useful for SAR/optical/multi-spectral fusion ablations under the no-classified-data constraint.

### 2E. Acoustic-Optical underwater fusion (cross-modal AUV perception)

**Learning-Based Leader Localization for Underwater Vehicles With Optical-Acoustic-Pressure Sensor Fusion**
- arXiv: `2502.20817` · 2025 · Yang, Sha, Zhang
- Trimodal fusion of optical (high-res), acoustic (long-range), pressure (environmental) for AUV leader localization; outperforms single- and dual-modality variants.
- *Why it matters:* concrete demonstration that >2 underwater modalities can be jointly trained at TRL ~4 scale — directly mirrors the maritime task-group fusion intent.

### 2F. Explainability and review

**Explainable AI for Maritime Autonomous Surface Ships (MASS): Adaptive Interfaces and Trustworthy Human-AI Collaboration**
- arXiv: `2509.15959` · 2025 · Zhang, Xu
- Reviews 100 studies on transparency in maritime autonomy; proposes an adaptive transparency framework coupling operator-state estimation with explainable decision support and uncertainty indicators.
- *Why it matters:* directly maps onto the IDEaS "operator-facing explainability" dimension; useful catalogue of what explainability features have already been tried in maritime systems.

**A Comprehensive Survey on Underwater Acoustic Target Positioning and Tracking: Progress, Challenges, and Perspectives**
- arXiv: `2506.14165` · 2025 · Yang et al. (19 authors)
- Reviews 180+ papers (2016–2025) on underwater acoustic positioning/tracking organized by target scale, sensor mode, and collaboration; flags federated learning, blockchain, embodied AI and LLMs as emerging directions.
- *Why it matters:* canonical 2025 systematization of the underwater-acoustic side that the bibliography would otherwise be missing.

**A Survey on SAR Ship Classification using Deep Learning**
- arXiv: `2503.11906` · 2025 · Awais, Reggiannini, Moroni, Salerno
- Categorizes SAR-ship-classification DL methods by architecture, feature integration, SAR-attribute usage, fine-tuning, and explainability; identifies dataset and standardization gaps.
- *Why it matters:* the 2025 survey to use when sanity-checking that a chosen SAR baseline is representative.

## 3. Datasets

| Dataset | Modalities | Open? / Licence | Stable identifier | Notes |
|---|---|---|---|---|
| **xView3-SAR** | Sentinel-1 SAR + AIS labels + bathymetry + wind | Yes; CC-BY 4.0 (per dataset card on iuu.xview.us) | arXiv:2206.00897, https://iuu.xview.us/ | 991 scenes, 243k objects; canonical dark-vessel benchmark |
| **OpenSARShip 1.0/2.0** | Sentinel-1 SAR (with AIS-derived ship-type labels) | Yes; research use (terms on `opensar.sjtu.edu.cn`) — **flag: not CC-BY**, custom academic licence | http://opensar.sjtu.edu.cn/ | 11,346 chips, 17 ship types |
| **FUSAR-Ship** | Gaofen-3 SAR + AIS matchup | Yes; research-use only (Fudan EMW Lab) — **flag: not CC-BY** | DOI 10.1007/s11432-019-2772-5 | 5k+ ship chips with AIS, 15 categories/98 sub-categories |
| **HRSID** | High-res SAR (optical-style box/segmentation labels) | Yes; research-use | https://github.com/chaozhong2010/HRSID | 5,604 images, 16,951 ship instances |
| **SMART-Ship** | Visible + SAR + panchromatic + multi-spectral + NIR | Yes; check Github release for licence — **flag: not yet on HF** | arXiv:2508.02384 | 1,092 synchronized image sets, 38,838 ships |
| **HOSS ReID** | Optical + SAR cross-time ship pairs | Yes; via paper's GitHub | arXiv:2506.22027 | First open optical-SAR ship Re-ID dataset |
| **DeepShip** | Passive sonar (4 ship classes; 47h of audio) | Yes; research-use; **flag: not CC-BY**, requires citation | https://github.com/irfankamboh/DeepShip | 47h, 265 ships, Cargo/Passenger/Tanker/Tugboat |
| **ShipsEar** | Passive sonar (12 ship classes) | Yes; research-use; **flag: custom licence** | http://atlanttic.uvigo.es/underwaternoise/ | 2,223 clips, used by ~every UATR paper |
| **SeafloorAI / SeafloorGenAI** | Side-scan sonar + bathymetry + slope/rugosity + 7M Q/A pairs | Yes (NeurIPS 2024 D&B track); raw data CC-BY (USGS/NOAA) | arXiv:2411.00172, https://sites.google.com/udel.edu/seafloorai/home | 696k sonar images, 827k masks; the largest open SSS+VL set |
| **FVessel** | AIS + dome-camera video, inland waterway | Yes; research-use | https://github.com/gy65896/FVessel | 26 videos, multi-weather; canonical AIS-video benchmark |
| **eyesofworld/AIS_Dataset (HF)** | AIS-only (MMSI, position, SOG, COG, heading, type, status) | HF dataset; licence unspecified on card — **flag: confirm before any redistribution** | https://huggingface.co/datasets/eyesofworld/AIS_Dataset | Verified via `hf_inspect_dataset`; 10-column AIS schema |
| **Global Fishing Watch — Sentinel-1 Vessel Detections** | SAR-derived detections + AIS matches | Yes; CC-BY-NC 4.0 (per GFW dataset terms) — **flag: NC restricts commercial reuse** | https://globalfishingwatch.org/datasets-and-code/ | Industrial vessel detections 2017–present |
| **xView3 environmental rasters** | Bathymetry + wind co-registered to SAR | Yes; bundled with xView3-SAR licence | iuu.xview.us | Useful as auxiliary modality |

## 4. Reference implementations

| Repo | Tied to paper | Modalities | Notes |
|---|---|---|---|
| `allenai/sar_vessel_detect` | xView3 SAR baseline (`2206.00897`) | SAR | AI2 Skylight team's xView3 inference + training pipeline |
| `allenai/vessel-detection-sentinels` | xView3 lineage | SAR (S1) + EO/IR (S2) | Production-ish multi-sensor vessel detection on Sentinels |
| `DIUx-xView/xView3_fourth_place` | xView3 (`2206.00897`) | SAR | Public 4th-place solution; reproducible training |
| `waterdisappear/SARATR-X` | SARATR-X (`2405.09365`) | SAR | Foundation-model weights + curated ATR data |
| `irfankamboh/DeepShip` | DeepShip dataset paper | Passive sonar | Reference loader and baseline |
| `gy65896/FVessel` | DeepSORVF (`2302.11283`) | AIS + video | Benchmark dataset + readme |
| `gy65896/DeepSORVF` | DeepSORVF (`2302.11283`) | AIS + video | Method code for asynchronous trajectory matching |
| `QuJX/AIS-Visual-Fusion` | AIS-visual fusion line | AIS + video | Companion implementation for inland waterway surveillance |
| `eyesofworld/Maritime-Monitoring` | AIS Transformer survey (`2505.07374`) | AIS | Companion code & curated AIS data |
| `chaozhong2010/HRSID` | HRSID dataset | SAR | Loader for high-res SAR ship images |
| `jasonmanesis/Satellite-Imagery-Datasets-Containing-Ships` | meta-index | SAR + EO | Curated index of ship satellite datasets — useful jumping-off point |

(GitHub star counts could not be fetched in this run because the `github_examples` MCP path returned an error; the repos above were located via web search and are flagged for the synthesist to crosscheck.)

## 5. Open questions noticed (flagged, not hypothesised)

1. No open paper in the 2024–2026 window jointly fuses **sonar + RF + visual** at the architecture level — the closest tri-modal underwater work is optical+acoustic+pressure (`2502.20817`), and the closest surface tri-modal is RGB+LWIR+LiDAR (`2505.01615`).
2. Uncertainty propagation across fused modalities is rare; **GMvA (`2504.09197`)** is one of the only papers with a dedicated uncertainty fusion module, and it operates on AIS+CCTV only.
3. **AIS-LLM (`2508.07668`)** is the only paper found that produces operator-facing natural-language explanations alongside anomaly scores; this leaves a clear gap for sonar/RF anomaly explanations.
4. **No public dataset combines sonar with AIS or with RF.** SeafloorAI is the largest open sonar+text set but lacks any AIS or RF channel; xView3 is SAR+AIS but no sonar.
5. Sonar-DL robustness surveys (`2412.11840`, `2510.03353`) explicitly flag that simulation-to-reality gap and lack of standardised benchmarks remain unsolved — gating any TRL 4–5 sonar leg.
6. SWaP/edge-deployment is rarely reported. **NAS-DETR (`2505.06694`)** reports compute-budget-aware sonar ATR; **xView3 FPGA work** (arXiv:2507.04842, untriaged) hints at edge SAR; no fusion paper combines edge inference with uncertainty propagation.
7. Cross-modal Ship Re-ID datasets exist for optical+SAR (`2506.22027`) but not for optical+sonar or AIS+sonar — implying no public way to evaluate persistent identity across an above-water/below-water boundary.
8. Foundation-model coverage is asymmetric: SAR has SARATR-X, SARVLM, FUSAR-KLIP, but sonar has no comparable foundation model; underwater-acoustic SSL curation (`2505.20066`) is the closest.

## 6. Sources (flat list of arxiv IDs and URLs cited above)

ArXiv IDs:
- `2206.00897` — xView3-SAR
- `2302.11283` — DeepSORVF / FVessel
- `2402.12658` — Interpretable contrastive UATR
- `2405.09365` — SARATR-X
- `2406.09966` — AIS outlier detection (RNN)
- `2409.13878` — Cross-domain transfer for UATR
- `2410.08612` — Synth-SONAR
- `2410.12953` — Syn2Real SSS mine detection
- `2411.00172` — SeafloorAI
- `2411.02848` — Adversarial multi-task UATR
- `2412.11840` — Sonar-DL robustness survey
- `2502.20817` — Optical-acoustic-pressure tri-modal fusion
- `2503.11906` — SAR ship classification survey
- `2504.09197` — GMvA (graph + uncertainty AIS-CCTV fusion)
- `2505.01615` — Multimodal/multiview marine navigation
- `2505.06694` — NAS-DETR for sonar
- `2505.07374` — AIS Transformer survey
- `2505.20066` — Automated SSL curation for underwater acoustics
- `2506.14165` — Underwater acoustic positioning/tracking survey
- `2506.22027` — HOSS ReID + TransOSS
- `2507.13880` — Visual + chart fusion
- `2508.02384` — SMART-Ship
- `2508.07668` — AIS-LLM
- `2509.15959` — XAI for MASS
- `2510.03353` — Sonar dataset survey
- `2510.11449` — YOLO satellite + AIS inland waterway

URLs:
- https://iuu.xview.us/ (xView3 challenge / dataset portal)
- http://opensar.sjtu.edu.cn/ (OpenSARShip)
- https://github.com/chaozhong2010/HRSID (HRSID)
- https://github.com/irfankamboh/DeepShip (DeepShip)
- https://github.com/allenai/sar_vessel_detect (xView3 baseline)
- https://github.com/allenai/vessel-detection-sentinels (Sentinel-1/2 vessel detection)
- https://github.com/DIUx-xView/xView3_fourth_place (xView3 4th-place solution)
- https://github.com/waterdisappear/SARATR-X (SARATR-X)
- https://github.com/gy65896/FVessel (FVessel benchmark)
- https://github.com/gy65896/DeepSORVF (DeepSORVF method)
- https://github.com/QuJX/AIS-Visual-Fusion (AIS-Visual fusion)
- https://github.com/jasonmanesis/Satellite-Imagery-Datasets-Containing-Ships (dataset index)
- https://huggingface.co/datasets/eyesofworld/AIS_Dataset (HF AIS dataset)
- https://sites.google.com/udel.edu/seafloorai/home (SeafloorAI site)
- https://globalfishingwatch.org/datasets-and-code/ (Global Fishing Watch open data)
