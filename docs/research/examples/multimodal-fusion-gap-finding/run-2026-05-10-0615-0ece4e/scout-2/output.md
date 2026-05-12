# Scout-2 Annotated Bibliography: RF / SIGINT fused with imagery or text

## 1. Scope

This sub-topic surveys 2024–2026 work that fuses **radio-frequency / signals-intelligence streams** with at least one of {EO/IR or other imagery, text intel, or imagery-shaped wireless representations like spectrograms / CSI maps treated as images} for downstream classification, identification, sensing, or reasoning tasks. Application affinity: real-time multi-domain threat assessment (EO video + SIGINT + text intel), Arctic ISR (satellite + RF + telemetry), counter-UAS / counter-drone fusion, and emerging wireless foundation models that ingest joint signal-image inputs.

**Narrowing decisions made (and why):**
- I include **single-modality RF foundation / RFML papers** only where they are canonical pre-conditions for multimodal RF fusion (e.g., RadioML-derived backbones, IQ-stream encoders that are explicitly intended to be plugged into multimodal stacks). I exclude single-modality RFML work that has no multimodal hook.
- I include **camera-radar autonomous-driving fusion** only briefly, and only where the architectural pattern (4D radar tensor + image cross-attention) transfers to wider RF/SIGINT use. Active radar is not the same as passive RF/SIGINT, so I keep this category small.
- I treat **RF sensing for human activity / behavior** as relevant when the work explicitly fuses RF with imagery or text — RF-only activity recognition is excluded.
- The MCP wrapper for `mcp__ml-intern__hf_papers` was returning `'query' is required` / `'arxiv_id' is required` errors regardless of how the kwargs payload was shaped, so verification was performed by direct WebFetch against `arxiv.org/abs/<id>` for every cited ID. This is documented in `verification.md`.

## 2. Key papers

### 2a. Multimodal wireless foundation models (RF + image-like inputs in a single backbone)

**Multimodal Wireless Foundation Models** — arXiv 2511.15162 — 2025 — Aboulfotouh, Abou-Zeid (et al.).
Introduces the first wireless foundation model that ingests **both raw IQ streams and image-like wireless modalities** (spectrograms, CSI maps) under a single masked-wireless-modeling self-supervised objective. Evaluates on five downstream tasks including human-activity sensing, RF signal classification, and device fingerprinting. *Why it matters for the spec's gap-finding target:* this is the cleanest 2025 example of a unified RF + spectrogram-image encoder. Any "RF + EO" candidate architecture for the IDEaS proposal will be benchmarked against, or built on top of, this style of multimodal masked-wireless pretraining.

**6G WavesFM: A Foundation Model for Sensing, Communication, and Localization** — arXiv 2504.14100 — 2025 — Aboulfotouh et al.
A unified Vision-Transformer backbone with task-specific heads and LoRA adapters that processes **spectrograms, CSI, and OFDM resource grids** across four downstream applications (positioning, channel estimation, activity sensing, RF signal classification). Shares 80 % of parameters across tasks. *Why it matters:* shows the parameter-efficient adaptation route from a single image-shaped wireless backbone to multiple SIGINT-adjacent heads — directly relevant to SWaP-aware tactical-edge candidates.

**Building 6G Radio Foundation Models with Transformer Architectures** — arXiv 2411.09996 — 2024 — Aboulfotouh et al.
Earlier paper from the same group: ViT plus Masked Spectrogram Modeling for self-supervised RF pretraining, evaluated on CSI-based human activity sensing and spectrogram segmentation. *Why it matters:* the canonical 2024 reference for "treat the spectrogram like an image and apply ViT pretraining to RF" — useful baseline citation for any spectrogram-as-image fusion candidate.

**LWM-Spectro: A Foundation Model for Wireless Baseband Signal Spectrograms** — arXiv 2601.08780 — 2026 — Kim et al.
Transformer model that learns IQ-baseband signals as spectrograms with masked modeling, contrastive learning, and a mixture-of-experts head. Reports strong modulation classification and SNR/mobility recognition. *Why it matters:* a 2026 contender to the WavesFM family; the MoE head is the architectural ingredient missing from earlier RF foundation models and could be repurposed for modality-conditional routing in fused stacks.

### 2b. RF + vision-language model integration (RF interpreted by VLM/LLM stacks)

**PReD: An LLM-based Foundation Multimodal Model for Electromagnetic Perception, Recognition, and Decision** — arXiv 2603.28183 — 2026 — Han et al.
1.3 M-sample dataset and model covering signal detection, modulation recognition, parameter estimation, protocol recognition, RF fingerprinting, and anti-jamming decision-making across **time-domain waveforms, spectrograms, and constellation diagrams** routed through an LLM backbone. *Why it matters:* the closest published analogue to "operator-facing LLM that reasons about RF + text intel together" — directly informs the explainability story for any IDEaS candidate that wants natural-language operator output.

**Seeing Radio: From Zero RF Priors to Explainable Modulation Recognition with Vision Language Models** — arXiv 2601.13157 — 2026 — Zou et al.
Converts RF signals into **time-series, spectrogram, and joint visualizations**, then parameter-efficient-fine-tunes an off-the-shelf VLM on a 57-class benchmark, reaching ~90 % with **human-readable rationales**. *Why it matters:* demonstrates that off-the-shelf VLM weights transfer to RF when the input is reshaped as imagery; provides an explainability pattern (rationale generation) directly applicable to the IDEaS explainability success criterion.

**RF-GPT: Teaching AI to See the Wireless World** — arXiv 2602.14833 — 2026 — Zou et al.
Builds a radio-frequency language model: complex IQ → time-frequency tokens → decoder-only LLM, trained on ~12 K synthetic RF scenes and ~625 K instruction examples covering six wireless technologies. Benchmarks include modulation classification, wireless-technology recognition, and 5G NR information extraction. *Why it matters:* this is the most explicit "RF + text intel" instruction-tuned model in the 2026 corpus, and the synthetic-instruction corpus is a template for an open Arctic-ISR or counter-UAS analogue.

**Large Language Model-Driven Distributed Integrated Multimodal Sensing and Semantic Communications** — arXiv 2505.18194 — 2025 — Peng et al.
Proposes an "RF-vision fusion network (RVFN)" with cross-attention between RF and camera features, plus a LLM-based semantic transmission network. Evaluates on synthetic multi-view RF-visual datasets. *Why it matters:* demonstrates a working cross-attention recipe for RF + vision under an LLM-driven controller — applicable to the EO video + SIGINT + text intel application context the spec names.

**MMSense: Adapting Vision-based Foundation Model for Multi-task Multi-modal Wireless Sensing** — arXiv 2511.12305 — 2025 — Li et al.
Unified foundation model that converts **image, radar, LiDAR, and textual data** into vision-compatible formats, with adaptive modality fusion and instruction-driven task adaptation atop a vision-language backbone. *Why it matters:* shows the "everything-into-VLM" approach with radar in the mix — a useful reference for spec applications where active radar streams are part of the SIGINT-adjacent fusion set.

### 2c. Modulation recognition and emitter ID with multimodal feature representations

**MCANet: A Coherent Multimodal Collaborative Attention Network for Advanced Modulation Recognition in Adverse Noisy Environments** — arXiv 2510.18336 — 2025 — Jiang et al.
Multimodal deep-learning framework for AMR under low-SNR conditions; the "multi-modality" here is **multi-representation of the same RF signal** (time-domain, frequency-domain, constellation), fused via collaborative attention. *Why it matters:* a strong 2025 baseline for the "spectrogram + raw IQ + constellation" three-stream fusion pattern — a useful comparative reference for any candidate that promises low-SNR robustness.

**Enhancing Automatic Modulation Recognition With a Reconstruction-Driven Vision Transformer Under Limited Labels** — arXiv 2508.20193 — 2025 — Ahmadi et al.
ViT framework combining supervised, self-supervised, and reconstruction objectives for AMR; tested on RML2018.01A. Achieves ResNet-comparable accuracy with only 15-20 % labels. *Why it matters:* establishes the limited-label regime that any open-data IDEaS candidate must operate in; the reconstruction objective is an ingredient for the SAR/EO-RF cross-modal pretext-task design.

**AI/ML-Based Automatic Modulation Recognition: Recent Trends and Future Possibilities** — arXiv 2502.05315 — 2025 — Jafarigol et al.
Comparative review of high-performance AMR models on RadioML-2016A, with discussion of architectures and training strategies. *Why it matters:* the canonical 2025 RFML-baseline survey — needed for the gap-finder's claim that single-modality AMR is well-explored and the unexplored intersections lie elsewhere.

**FLAME: A Federated Learning Approach for Multi-Modal RF Fingerprinting** — arXiv 2503.04136 — 2025 — Kianfar et al.
Federated-learning framework for device identification across multiple access points using **multiple complementary RF representations** beyond raw IQ; reports up to 30 % accuracy gain vs. IQ-only baselines. *Why it matters:* most direct 2025 evidence that "multi-representation RF fingerprinting" beats single-IQ — useful for arguments that the multimodal-RF lane is the lower-risk side of the design space.

**SEI-SHIELD: Robust Specific Emitter Identification Under Label Noise Via Self-Supervised Filtering and Iterative Rescue** — arXiv 2605.04721 — 2026 — Zhang et al.
Specific-emitter-identification framework for noisy-label settings using self-supervised filtering. *Why it matters:* SEI is the operationally-relevant SIGINT cousin to AMR; this paper anchors the SEI half of the SIGINT bibliography for the gap map.

### 2d. Cross-layer / cross-sensor RF fusion (RF + telemetry, RF + radar + EO)

**Securing UAV Communications by Fusing Cross-Layer Fingerprints (SecureLink)** — arXiv 2511.05796 — 2025 — Huang et al.
Fuses **physical-layer RF fingerprints with application-layer MEMS sensor fingerprints** (accelerometers, gyros, barometers) via attention networks plus a one-class SVM. 98.61 % closed-world / 97.54 % open-world authentication. *Why it matters:* a clean 2025 example of RF + telemetry cross-layer fusion at the platform edge — directly applicable to the Arctic-ISR application context (satellite + RF + telemetry).

**Intelligent Multimodal Multi-Sensor Fusion-Based UAV Identification, Localization, and Countermeasures for Safeguarding Low-Altitude Economy** — arXiv 2510.22947 — 2025 — Tao et al.
Integrated counter-UAS pipeline combining **RF spectral feature analysis, radar detection, and electro-optical identification** at the detection level, plus multi-sensor fusion at the localization level. Discusses both soft-kill and hard-kill countermeasures at the architecture level. *Why it matters:* the most explicit RF + radar + EO fusion stack in the 2025 corpus — direct architectural reference for the "real-time multi-domain threat assessment" application context.

**RF-Behavior: A Multimodal Radio-Frequency Dataset for Human Behavior and Emotion Analysis** — arXiv 2511.06020 — 2025 — Zuo et al.
Multimodal dataset with 13 radars (ground + ceiling), 6–8 RFID tags per participant, LoRa devices, IMUs, and 24 IR cameras across 44 participants. *Why it matters:* the cleanest open multimodal RF + IR + telemetry dataset of 2025; a candidate base dataset for any IDEaS-shaped fusion experiment that needs synchronized RF + imaging + telemetry.

**Multimodal-NF: A Wireless Dataset for Near-Field Low-Altitude Sensing and Communications** — arXiv 2603.28280 — 2026 — Li et al.
Synchronizes high-fidelity near-field CSI plus wireless labels with **RGB images, LiDAR point clouds, and GPS** — explicitly framed for low-altitude (drone-altitude) scenarios. *Why it matters:* directly applicable to Arctic ISR (low-altitude satellite + RF + telemetry context) and to multi-sensor counter-UAS — and has GPS as an explicit tie to spatiotemporal alignment.

**Vision-Language-Model-Guided Differentiable Ray Tracing for Fast and Accurate Multi-Material RF Parameter Estimation** — arXiv 2601.18242 — 2026 — Kang et al.
Uses a VLM to parse scene images, infer material categories, and seed an ITU-R material prior, then runs differentiable ray tracing for RF parameter estimation. 2–4× faster convergence, 10–100× lower error. *Why it matters:* a 2026 demonstration that scene-imagery understanding can be plugged into RF physics models — relevant to any candidate that wants explainable RF-EO fusion via physics-grounded priors.

### 2e. Spectrum prediction / preprocessing (spectrogram-as-image lineage, secondary references)

**Spectrum Prediction With Deep 3D Pyramid Vision Transformer Learning** — arXiv 2408.06870 — 2024 — Pan et al.
3D Swin Transformer plus pyramid design for spectrum prediction, with transfer learning. *Why it matters:* canonical 2024 reference for "spectrogram is a spatiotemporal image" — important for any candidate whose pretext task is spectrum forecasting.

**Optimal Preprocessing for Joint Detection and Classification of Wireless Communication Signals in Congested Spectrum Using Computer Vision Methods** — arXiv 2408.06545 — 2024 — Kang et al.
Optimizes STFT parameters for applying YOLO and DETR to RF spectrograms in congested spectrum. *Why it matters:* establishes the preprocessing hyperparameter floor for any "spectrogram + CV detector" candidate — the unsexy but load-bearing reference.

**IQFM: A Wireless Foundational Model for I/Q Streams in AI-Native 6G** — arXiv 2506.06718 — 2025 — Mashaal, Abou-Zeid.
Contrastive SSL on raw IQ for modulation classification, AoA, beam prediction, RF fingerprinting. Single-modality but explicitly designed as a plug-in encoder. *Why it matters:* useful as the IQ-only baseline against which any multimodal RF + image candidate must demonstrate gain.

**Self-Supervised Radio Representation Learning: Can we Learn Multiple Tasks?** — arXiv 2509.03077 — 2025 — Kanu et al.
Momentum-contrast SSL on radio signals for AoA + AMR. *Why it matters:* establishes that the SSL recipe transfers across RF tasks even single-modally, providing a baseline "no-fusion" lower bound for any fusion-gain claim.

### 2f. Auxiliary / context references

**GNSS/GPS Spoofing and Jamming Identification Using Machine Learning and Deep Learning** — arXiv 2501.02352 — 2025 — Ghanbarzade, Soleimani.
ML / DL on GNSS data for jamming and spoofing detection. *Why it matters:* relevant context for any Arctic-ISR or stealth/spoof-detection candidate that needs to integrate GNSS-integrity signals into an RF fusion stack.

**Large Multi-Modal Models (LMMs) as Universal Foundation Models for AI-Native Wireless Systems** — arXiv 2402.01748 — 2024 — Xu et al.
Position paper proposing LMMs that handle multi-modal sensing data, causal reasoning, RAG, and neuro-symbolic reasoning for wireless systems. *Why it matters:* often-cited 2024 framing paper for "LMMs as the substrate for wireless multimodal fusion" — useful for the synthesist's framing chapter even though it's high-level.

## 3. Datasets

| Dataset | HF / canonical name | Modalities | Licence | Notes |
|---|---|---|---|---|
| **RadioML 2018.01A** | DeepSig open data; not on HF as a hosted dataset (DeepSig hosts directly) | IQ, 24 modulation types, ~2 M samples, HDF5 | **CC BY-NC-SA 4.0** (flag: NonCommercial — restrictive vs. CC-BY) | Canonical RFML benchmark; used by 2508.20193 (Ahmadi), 2510.18336 (MCANet), 2502.05315 (Jafarigol). |
| **RadioML 2016.10A / 2016.04C** | DeepSig open data | IQ, 11 modulations | **CC BY-NC-SA 4.0** (flag: NonCommercial) | Older but still benchmarked widely; explicitly labelled "early academic work" by DeepSig with known errata. |
| **RF-Behavior** | arXiv 2511.06020 release; HF dataset page not yet confirmed at scout time | Radar (13×) + RFID + LoRa + IMU + IR cameras | Listed in the paper; licence to verify on release | Strong candidate base for IDEaS-shaped RF + IR + telemetry fusion experiments. **Flag: licence not confirmed at scout time, must be verified before any download/use.** |
| **Multimodal-NF** | arXiv 2603.28280 release dataset | CSI + RGB + LiDAR + GPS | Listed in paper; licence to verify | Drone-altitude RF + visual + GPS — direct match for Arctic-ISR application context. **Flag: licence not confirmed at scout time.** |
| **DroneRF / DroneDetect** | Not on HF; original DroneRF is on Mendeley Data | Raw RF captures of drone control + downlink | DroneRF — CC BY 4.0 (per Mendeley page); DroneDetect — academic-use | Used by 2507.14592 (Liu et al.). |
| **Katherinezml/radar_jamming_and_communication_modulation_dataset** | HF: `Katherinezml/radar_jamming_and_communication_modulation_dataset` | Radar jamming + communication modulation samples (~3.59 GB) | **Licence not specified on HF page** (flag: unknown) | Updated Mar 2025; HF viewer currently unavailable. **Flag: cannot use without contacting uploader for licence.** |
| **0x70DA/drone-spectrogram (v1/v2/v3)** | HF: `0x70DA/drone-spectrogram` family | Drone spectrograms | Licence not visible in HF listing (flag: unknown) | Small (4–8 K rows); useful for sandbox experiments only. **Flag: licence to verify.** |
| **DeepSig synthetic IQ generators (GNU Radio)** | DeepSig-released GNU Radio flowgraphs | Synthetic IQ; user-generated | Tooling typically GPL (GNU Radio); generated data carries no upstream licence | Used by 2503.04136 (FLAME) and others to extend training corpora. |

## 4. Reference implementations

| Repo | Stars | Tied to | Notes |
|---|---|---|---|
| `Richardzhangxx/AMR-Benchmark` | 437 | RadioML-family AMR | Unified implementation of several baseline DL models for AMR; the de-facto baseline-collection repo. |
| `iyytdeed/Automatic-Modulation-Classification` | 250 | Master's-thesis baselines | Multiple architectures benchmarked on RadioML 2016.10A. |
| `wzjialang/MCLDNN` | 133 | IEEE-WCL paper on spatiotemporal multi-channel AMR | Last updated June 2024; clean PyTorch implementation. |
| `WestdoorSad/IQFormer` | 85 | "IQFormer: Multi-Modality Fusion for AMR" (paper appears in Wireless Networks; arXiv ID could not be confirmed via WebFetch — see verification.md) | Updated June 2025. **Flag: paper retrieval failed, repo may stand alone.** |
| `Richardzhangxx/PET-CGDNN` | 80 | Parameter-efficient AMR | Lightweight architecture. |
| `Patrick-Nick/CDSCNN` | 33 | Complex-valued depthwise-separable CNN for AMC | Useful CV-style RF baseline. |
| `drexelwireless/RadioML` | 14 | Original RadioML residual-network style | Older but canonical reproducer. |
| `iksnagreb/radioml-transformer` | 1 | Transformer baseline for RadioML | Small, but a clean transformer reference. |
| `Lexicon121/Strix-Interceptor` | 42 | Open-source counter-UAS / counter anti-drone RF | C++; system-level rather than ML. |
| `khalidt/Anti-Drone` | 0 | Vision + RF + acoustic UAV detection toolkit (research-grade) | Updated Nov 2025; explicitly multi-modal scope. |
| `QinHaoXiang1996/Spectrum-Sensing-...` (DL spectrum sensing) | 43 | DL-based spectrum sensing on GNU Radio | Useful for synthetic-data pipelines. |
| `rishhabhnaik/End-to-end-OFDM` | 43 | GNU Radio + USRP X310 + NN | Hardware-loop OFDM example. |

No public reference implementation was confirmed for the 2025–2026 RF-foundation-model papers (Aboulfotouh's WavesFM family, RF-GPT, PReD, MMSense). Several promise code releases in the abstract; status to be re-checked at synthesist time.

## 5. Open questions you noticed

(These are gaps spotted **while reading**. They are flagged for the gap-finder, not solved here.)

- **RF + EO satellite imagery** as a fused pair is essentially absent from the 2024–2026 corpus. Multimodal-NF (2603.28280) ships RGB + LiDAR + GPS + CSI, but it is near-field / low-altitude. There is no open paper and no open dataset for "satellite EO + co-located passive RF" at the orbital scale the IDEaS Arctic-ISR application calls for.
- **RF + text-intel (OSINT) fusion** is essentially absent. RF-GPT (2602.14833) and PReD (2603.28183) instruction-tune LLMs on RF, but the text side is synthetic instruction data describing the RF — not external operator reports / OSINT correlated with live RF. No paper retrievable in 2024–2026 explicitly fuses RF observations with structured text-intel feeds.
- **Uncertainty propagation across RF + image fusion** is rare. Most fusion works (2510.22947, 2511.05796) report point estimates; very few propagate calibrated confidence from RF to imagery branches and back, which is one of the IDEaS desired-outcome capabilities.
- **SWaP-aware tactical-edge deployment of RF foundation models.** WavesFM (2504.14100) shares 80 % of parameters across heads, but no paper retrievable in scope reports actual edge-device latency / power numbers for RF + image fusion. Big open question.
- **RF + audio/sonar fusion for maritime ISR** has no open paper retrievable in the 2024–2026 window. The closest is RF-Behavior (2511.06020) which fuses RF + IR cameras + IMU but not audio.
- **Specific-emitter identification (SEI) fused with operator/text intelligence** is essentially absent. SEI-SHIELD (2605.04721) is RF-only; PReD (2603.28183) is the only retrievable work that touches RF fingerprinting in a multimodal LLM stack, but does not explicitly tie SEI to text intel.
- **Counter-UAS literature splits along modality lines.** Tao et al. (2510.22947) integrate RF + radar + EO conceptually; Jahan et al. (2603.08208) fuse only thermal + visual; UAV-MM3D (2511.22404) is a synthetic dataset including radar but not deep RF spectral analysis. There is no open paper demonstrating end-to-end counter-UAS fusion across **passive RF spectral analysis + EO video + text intel** simultaneously.
- **Licence story for canonical RFML data is restrictive.** RadioML 2018.01A is CC BY-NC-SA 4.0 — non-commercial. Any IDEaS proposal that wants TRL 4–5 deployment needs to justify either (a) staying within the academic-research carve-out of the licence or (b) using DeepSig's GNU Radio flowgraphs to regenerate equivalent data under a different licence. This is a real provenance question, not just a flag.
- **Reproducibility of RF foundation models is poor.** Of the 2024–2026 RF-foundation-model papers retrieved (2511.15162, 2504.14100, 2411.09996, 2601.08780), none has a confirmed public reference implementation as of scout-time. The bibliographic foundation is solid but the implementation foundation is thin.

## 6. Sources

All cited arXiv IDs (verified individually via `arxiv.org/abs/<id>` WebFetch — see `verification.md`):

- 2402.01748 — Large Multi-Modal Models (LMMs) as Universal Foundation Models for AI-Native Wireless Systems
- 2408.06545 — Optimal Preprocessing for Joint Detection and Classification of Wireless Communication Signals in Congested Spectrum Using Computer Vision Methods
- 2408.06870 — Spectrum Prediction With Deep 3D Pyramid Vision Transformer Learning
- 2411.09996 — Building 6G Radio Foundation Models with Transformer Architectures
- 2501.02352 — GNSS/GPS Spoofing and Jamming Identification Using Machine Learning and Deep Learning
- 2502.05315 — AI/ML-Based Automatic Modulation Recognition: Recent Trends and Future Possibilities
- 2503.04136 — FLAME: A Federated Learning Approach for Multi-Modal RF Fingerprinting
- 2504.14100 — 6G WavesFM: A Foundation Model for Sensing, Communication, and Localization
- 2505.18194 — Large Language Model-Driven Distributed Integrated Multimodal Sensing and Semantic Communications
- 2506.06718 — IQFM: A Wireless Foundational Model for I/Q Streams in AI-Native 6G
- 2508.20193 — Enhancing Automatic Modulation Recognition With a Reconstruction-Driven Vision Transformer Under Limited Labels
- 2509.03077 — Self-Supervised Radio Representation Learning: Can we Learn Multiple Tasks?
- 2510.18336 — MCANet: A Coherent Multimodal Collaborative Attention Network for Advanced Modulation Recognition
- 2510.22947 — Intelligent Multimodal Multi-Sensor Fusion-Based UAV Identification, Localization, and Countermeasures
- 2511.05796 — Securing UAV Communications by Fusing Cross-Layer Fingerprints (SecureLink)
- 2511.06020 — RF-Behavior: A Multimodal Radio-Frequency Dataset for Human Behavior and Emotion Analysis
- 2511.12305 — MMSense: Adapting Vision-based Foundation Model for Multi-task Multi-modal Wireless Sensing
- 2511.15162 — Multimodal Wireless Foundation Models
- 2601.08780 — LWM-Spectro: A Foundation Model for Wireless Baseband Signal Spectrograms
- 2601.13157 — Seeing Radio: From Zero RF Priors to Explainable Modulation Recognition with Vision Language Models
- 2601.18242 — Vision-Language-Model-Guided Differentiable Ray Tracing for Fast and Accurate Multi-Material RF Parameter Estimation
- 2602.14833 — RF-GPT: Teaching AI to See the Wireless World
- 2603.28183 — PReD: An LLM-based Foundation Multimodal Model for Electromagnetic Perception, Recognition, and Decision
- 2603.28280 — Multimodal-NF: A Wireless Dataset for Near-Field Low-Altitude Sensing and Communications
- 2605.04721 — SEI-SHIELD: Robust Specific Emitter Identification Under Label Noise Via Self-Supervised Filtering and Iterative Rescue

Datasets & dataset hubs:
- DeepSig open data (RadioML 2016.10A, 2016.04C, 2018.01A): https://www.deepsig.ai/datasets — CC BY-NC-SA 4.0
- HF: `Katherinezml/radar_jamming_and_communication_modulation_dataset`
- HF: `0x70DA/drone-spectrogram` (v1/v2/v3)
- DroneRF (Mendeley Data) — referenced by 2507.14592

GitHub:
- https://github.com/Richardzhangxx/AMR-Benchmark
- https://github.com/iyytdeed/Automatic-Modulation-Classification
- https://github.com/wzjialang/MCLDNN
- https://github.com/WestdoorSad/IQFormer
- https://github.com/Richardzhangxx/PET-CGDNN
- https://github.com/Patrick-Nick/CDSCNN
- https://github.com/drexelwireless/RadioML
- https://github.com/iksnagreb/radioml-transformer
- https://github.com/Lexicon121/Strix-Interceptor
- https://github.com/khalidt/Anti-Drone
- https://github.com/QinHaoXiang1996 (Dataset Generation for DL-based Spectrum Sensing)
- https://github.com/rishhabhnaik (End-to-end OFDM with DL)

Skipped (could not retrieve via arXiv WebFetch verification — flagged not invented):
- "IQFormer: A Novel Transformer-Based Model With Multi-Modality Fusion for Automatic Modulation Recognition" — appears in Wireless Networks / IEEE; no arXiv ID resolves. The repo `WestdoorSad/IQFormer` exists but the arXiv version is not retrievable.
- "RFSensingGPT: A Multi-Modal RAG-Enhanced Framework for Integrated…" — IEEE-only as of scout time; no arXiv ID resolves.
- "MAFFNet: a multi-modal adaptive feature fusion net for signal…" — Springer-only; no arXiv ID resolves.
- "Modulation recognition method based on multimodal features" (Frontiers) — journal-only; no arXiv ID resolves.
- "Automatic modulation recognition using vision transformers with cross…" (Wireless Networks) — journal-only; no arXiv ID resolves.

Per discipline rule "if a paper couldn't be retrieved, you flag and skip", these are flagged here and not used as citations in the body.
