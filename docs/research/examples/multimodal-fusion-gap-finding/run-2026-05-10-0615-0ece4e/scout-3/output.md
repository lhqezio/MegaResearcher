# Literature Scout 3 — Tactical-Edge Wearable Multimodal Fusion under SWaP and Degraded Connectivity

## 1. Scope

Survey of 2024–2026 multimodal fusion architectures and supporting techniques relevant to **tactical-edge wearable systems fusing audio + video + IMU/sensor streams under strict SWaP (Size, Weight, Power) and degraded-connectivity constraints**, with explicit attention to (a) on-device inference at sub-100M to ~1B parameter regimes, (b) audio-visual event detection, (c) IMU/sensor fusion and cross-modal distillation, (d) federated / online learning under intermittent connectivity and missing modalities, (e) compression (quantization, pruning, distillation) tailored to multimodal models, and (f) explainability for operator-worn systems. The IDEaS application affinity is soldier situational awareness (audio + video + sensor fusion on body-worn devices).

**Narrowing decisions:**
- Excluded medical-only XAI work unless the technique transfers cleanly to operator-facing wearables (kept the SHAP/concept-based XAI papers since they generalize).
- Treated tactical-grade biometric/military-only papers cautiously: only kept arxiv-resolvable items.
- Included two slightly-out-of-window references: Project Aria (2023) and ImageBind (2023) — both are canonical for the egocentric-multimodal data and IMU-binding lines and are referenced as upstream anchors only; not counted in the "≥8 recent" floor.
- Allowed a small number of papers that target ~1B parameters (rather than strictly <100M) because the on-device tactical-edge community now considers 0.5B–4B with INT4/INT8 quant a viable mobile-side budget; flagged when this is the case.

## 2. Key Papers

Grouped by sub-cluster.

### 2A. Cross-modal distillation and IMU↔video bridges (the closest match to the operator-wearable use case)

**[1] COMODO: Cross-Modal Video-to-IMU Distillation for Efficient Egocentric Human Activity Recognition** — Baiyu Chen et al., 2025. arXiv: **2503.07259**.
Self-supervised cross-modal distillation: a frozen video encoder supervises an IMU encoder via a dynamic instance queue; no labels required, and inference at runtime only needs the IMU stream. Matches or exceeds fully supervised baselines while being far cheaper and privacy-preserving. **Why it matters:** addresses the SWaP-driven case where the operator-worn system must run on IMU only (camera off, or dropped to save power) but was *trained* on richer multimodal data. This is exactly the "video as a knowledge source, IMU as the deployable sensor" pattern the spec needs.

**[2] XTinyHAR / Tiny Inertial Transformer (multimodal teacher → unimodal student via cross-modal KD)** — companion line to COMODO, surfaced via arXiv 2508.12213 survey (Cai et al., 2025) and the Nature Sci Reports 2025 instance. Although the Nature paper itself is non-arXiv, the survey resolves and the COMODO paper covers the same primitive. **Why it matters:** confirms the cross-modal-distillation-for-edge pattern is a live family, not a single paper.

**[3] SensorLM: Learning the Language of Wearable Sensors** — Yuwei Zhang et al., 2025. arXiv: **2506.09108**.
Sensor-language foundation model: hierarchical caption generation tied to wearable signals; a 59.7M-hour, 103k-person sensor-language dataset; CLIP/CoCa-style architectures extended to sensor streams. **Why it matters:** opens an explainability path where the operator sees natural-language descriptions of what the wearable just inferred — directly relevant to the "explainability story" capability.

### 2B. Wearable / egocentric activity recognition under SWaP

**[4] TinierHAR: Towards Ultra-Lightweight Deep Learning Models for Efficient Human Activity Recognition on Edge Devices** — Sizhen Bian et al., 2025. arXiv: **2507.07949**.
Residual depthwise-separable convolutions + GRUs + temporal aggregation; 2.7× fewer parameters than TinyHAR, 43.3× fewer than DeepConvLSTM, 6.4×–58.6× MAC reduction, no F1 loss across 14 HAR datasets. **Why it matters:** sets the SWaP baseline for IMU-only wearable HAR. Any tactical fusion candidate must beat or at least match this for the "no-camera, no-radio" degraded-connectivity mode.

**[5] WatchHAR: Real-time On-device Human Activity Recognition System for Smartwatches** — Taeyoung Yeon et al., 2025. arXiv: **2509.04736**.
On-device fusion of audio + motion sensors; 9.3 ms event detection, 11.8 ms multimodal classification, >90% accuracy across 25+ activity classes; 5× faster than baselines. **Why it matters:** demonstrates audio + IMU fusion at sub-12 ms latency on a smartwatch — the closest commercial proxy for a soldier-borne wrist module.

**[6] A Wearable Multi-Modal Edge-Computing System for Real-Time Kitchen Activity Recognition** — Mengxi Liu et al., 2024. arXiv: **2409.06341**.
Six sensors (IMUs, thermal cameras, etc.) across two MCUs; 184.5 KB compact model; 87.83% accuracy across 15 activities at 25.26 ms inference. **Why it matters:** rare end-to-end wearable system with thermal + IMU + multi-sensor fusion that fits within microcontroller budgets — concrete ConOps-relevant proof point.

**[7] Towards Generalizable Human Activity Recognition: A Survey** — Yize Cai et al., 2025. arXiv: **2508.12213**.
229 papers and 25 datasets reviewed; categorizes IMU HAR by model-centric (pre-training, end-to-end, LLM-based) vs. data-centric (multimodal, augmentation) approaches. **Why it matters:** provides the gap-finder with a structured map of what's been tried in IMU HAR, and explicitly flags domain shift and generalization as the open problems — exactly what tactical operators face.

**[8] A Survey on Multimodal Wearable Sensor-based Human Action Recognition** — Jianyuan Ni et al., 2024. arXiv: **2404.15349**.
Companion survey: inter-multimodal (vision + non-visual) vs. intra-multimodal (non-visual only). **Why it matters:** anchors the canonical taxonomy this scout's gap analysis sits in.

**[9] Scaling Wearable Foundation Models (LSM)** — Girish Narayanswamy et al., 2024. arXiv: **2410.13638**.
40M hours of HR, HRV, electrodermal, accelerometer, skin temp, altimeter from 165k people; scaling laws for sensor foundation models; downstream activity recognition. **Why it matters:** the only large-scale "wearable foundation model" paper with explicit scaling-laws data — informs the 100M parameter budget decision (i.e., what capability you give up at that size vs. a few-B model).

### 2C. Audio-visual event detection at the edge

**[10] Towards Open-Vocabulary Audio-Visual Event Localization** — Jinxing Zhou et al., 2024. arXiv: **2411.11278**.
Open-vocabulary AVE: 24,800-video OV-AVEBench (46 seen + 21 unseen categories); training-free ImageBind path + a fine-tuned variant. **Why it matters:** tactical scenarios are inherently open-vocabulary (you cannot enumerate every event a soldier might encounter); this paper is the cleanest 2024–2025 reference for open-set audio-visual detection.

**[11] Detect Any Sound: Open-Vocabulary Sound Event Detection with Multi-Modal Queries** — Pengfei Cai et al., 2025. arXiv: **2507.16343**.
DASM: query-based audio detection using text or audio prompts; dual-stream decoder separating recognition and localization; 7.8 PSDS gain in open-vocabulary settings. **Why it matters:** audio-only complement to [10]; tactical relevance is high (operator queries "any sound matching this acoustic signature?").

**[12] SoundingActions: Learning How Actions Sound from Narrated Egocentric Videos** — Changan Chen et al., 2024. arXiv: **2404.05206**.
33k Ego4D clip annotations; trains audio-action alignment from in-the-wild narrated egocentric video. **Why it matters:** operationalizes the egocentric audio-visual signal that a body-worn system actually produces.

### 2D. Robustness to missing modalities / degraded sensors

**[13] Resilient Sensor Fusion under Adverse Sensor Failures via Multi-Modal Expert Fusion (MoME)** — Konyul Park et al., 2025. arXiv: **2503.19776**.
Three parallel expert decoders (camera-only, LiDAR-only, fused) with an Adaptive Query Router; SOTA on nuScenes-R across beam reduction, sensor dropout, FoV limits. **Why it matters:** the routing-by-expert pattern is directly portable from autonomous driving to wearable audio-visual-IMU under intermittent sensor availability.

**[14] Robust Multimodal Learning via Cross-Modal Proxy Tokens** — Md Kaykobad Reza et al., 2025. arXiv: **2501.17823** (TMLR).
Cross-modal proxy tokens approximate the missing modality's class token using only the available modality; LoRA adapters in frozen encoders. **Why it matters:** lightweight, training-time-cheap mitigation for the "camera dropped" case — fits under the "no GPU spend" constraint of the IDEaS scoping run.

**[15] Multimodal Learning with Uncertainty Quantification based on Discounted Belief Fusion** — Grigor Bezirganyan et al., 2024. arXiv: **2412.18024**.
Order-invariant evidence fusion + conflict-based discounting; reallocates uncertainty mass when one modality becomes unreliable. **Why it matters:** the explicit uncertainty-propagation primitive that the IDEaS spec calls out as a desired-outcome capability — and one of the few that handles modality conflict.

### 2E. Federated / online learning under intermittent connectivity

**[16] Multimodal Online Federated Learning with Modality Missing in Internet of Things (MMO-FL)** — Heqiang Wang et al., 2025. arXiv: **2505.16138**.
Dynamic, decentralized multimodal learning for IoT edge devices; Prototypical Modality Mitigation algorithm compensates for missing modalities. **Why it matters:** directly addresses the "degraded connectivity" half of the spec — operator devices come and go from the network, modalities drop in and out.

**[17] Mitigating Modality Quantity and Quality Imbalance in Multimodal Online Federated Learning** — Heqiang Wang et al., 2025. arXiv: **2508.11159**.
Follow-up to [16] introducing the Modality Quantity and Quality Rebalanced (QQR) prototype-based algorithm; theoretical analysis of how modality QQI harms learning. **Why it matters:** quantifies the failure modes that arise when only a fraction of soldiers' devices have functioning microphones, etc.

### 2F. On-device / SWaP-aware multimodal language and vision-language models

**[18] SmolVLM: Redefining small and efficient multimodal models** — Andrés Marafioti et al., 2025. arXiv: **2504.05299**.
256M to 2.2B parameter VLMs; 256M variant uses <1 GB GPU memory at inference and beats much larger competitors; aggressive tokenization + curated data. **Why it matters:** the most concrete "VLM that actually fits on a wearable companion device" reference — a candidate substrate for the tactical-edge candidate architecture.

**[19] AndesVL Technical Report: An Efficient Mobile-side Multimodal Large Language Model** — Zhiwei Jin et al., 2025. arXiv: **2510.11496**.
0.6B–4B mobile MLLMs; 6.7× decoding speedup on MediaTek Dimensity 9500; 30.9% memory reduction; 1.8 bits-per-weight quantization; QALFT (quantization-aware LoRA fine-tuning). **Why it matters:** state-of-the-art mobile SoC demonstration with explicit quantization integration — provides a SWaP profile target for the candidate shortlist. **Note:** above the strict <100M cap, but with 1.8 bpw quantization an INT4-equivalent footprint of a 0.6B model is ~135 MB, which is realistic for tactical wearable companion compute.

**[20] Mobile-VideoGPT: Fast and Accurate Model for Mobile Video Understanding** — Abdelrahman Shaker et al., 2025. arXiv: **2503.21782**.
Sub-1B-parameter video-MLLM; 0.5B variant at 46 tokens/s; attention-based frame selection + token projector dropping redundant visual content. **Why it matters:** establishes that on-device *video* understanding is now plausible — the audio + video fusion direction stops being aspirational for the wearable form factor.

**[21] MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases** — Zechun Liu et al., 2024. arXiv: **2402.14905**.
125M and 350M LLMs; deep-thin architectures, embedding sharing, grouped-query attention; comparable to LLaMA-2 7B on API calling. **Why it matters:** the canonical "deep-thin sub-billion LLM" reference; many of the multimodal edge papers cite this for the language backbone of a wearable system.

**[22] Vision-Language Models for Edge Networks: A Comprehensive Survey** — Ahmed Sharshar, Latif U. Khan, Waseem Ullah, Mohsen Guizani, 2025. arXiv: **2502.07855**.
Pruning, quantization, distillation, hardware co-design for VLMs at the edge. **Why it matters:** the most up-to-date scaffolding paper — the gap-finder can cross-reference its taxonomy of compression techniques against what tactical-edge work has actually adopted.

### 2G. Compression — distillation and quantization specific to audio / multimodal

**[23] Quantization for OpenAI's Whisper Models: A Comparative Analysis** — Allison Andreyev et al., 2025. arXiv: **2503.09905**.
INT4/INT5/INT8 on Whisper variants; 19% latency reduction, 45% size reduction with preserved WER on LibriSpeech. **Why it matters:** Whisper-class ASR is the obvious speech component of any tactical wearable (operator commands, comms transcript, acoustic intel); this paper gives concrete numbers for INT8 deployment.

**[24] EAT: Self-Supervised Pre-Training with Efficient Audio Transformer** — Wenxi Chen et al., 2024. arXiv: **2401.03497**.
~15× pretraining speedup vs. existing audio SSL; competitive AudioSet/ESC-50/SPC-2 results. **Why it matters:** efficient audio backbone upstream of any audio-visual fusion model that has to ship.

### 2H. Edge platforms, multimodal TinyML systems

**[25] BioGAP-Ultra: A Modular Edge-AI Platform for Wearable Multimodal Biosignal Acquisition and Processing** — Sebastian Frey et al., 2025. arXiv: **2508.13728**.
GAP9 SoC, up to 32.2 GMAC/s on-device; synchronized EEG/EMG/ECG/PPG acquisition; 8.6–32.8 mW envelopes across head/wrist/chest form factors. **Why it matters:** open-source hardware reference for a wearable multi-sensor edge AI platform — informs the "what does a soldier-grade SWaP envelope actually look like" question.

**[26] A Multicore and Edge TPU-Accelerated Multimodal TinyML System for Livestock Behavior Recognition** — Qianxue Zhang et al., 2025. arXiv: **2504.11467**.
Accelerometer + vision multimodal fusion on edge TPU; 270× model-size reduction; <80 ms latency. **Why it matters:** another concrete multimodal-tinyML system; demonstrates that multimodal TinyML is no longer a research toy.

**[27] From Tiny Machine Learning to Tiny Deep Learning: A Survey** — Shriyank Somvanshi et al., 2025. arXiv: **2506.18927**.
Quant/pruning/NAS, MCU-to-NPU hardware, deployment frameworks, AutoML; explicit forward-looking section on neuromorphic + foundation models for the edge. **Why it matters:** the broad reference frame for the tactical-edge SWaP candidate.

### 2I. Multimodal HAR + interpretability

**[28] Multimodal Fusion and Interpretability in Human Activity Recognition: A Reproducible Framework for Sensor-Based Modeling** — Yiyao Yang et al., 2025. arXiv: **2510.22410**.
CMU-MMAC: video + audio + RFID synchronized; early/late/hybrid LSTM fusion comparison (late wins); PCA + t-SNE interpretability; RFID adds >50% accuracy. **Why it matters:** rare reproducible reference framework that combines multimodal fusion *and* interpretability *and* is wearable-relevant.

**[29] Rethinking Explainability in the Era of Multimodal AI** — Chirag Agarwal, 2025. arXiv: **2506.13060**.
Argues unimodal explanations misrepresent cross-modal influence; proposes three principles (modality influence via ablation, joint predictive power capture, consistency under cross-modal perturbation). **Why it matters:** sets the bar for what an *operator-grade* multimodal explanation must show — directly informs the explainability story the candidate architectures will need.

**[30] Explainable AI Using Inherently Interpretable Components for Wearable-based Health Monitoring** — Maurice Kuschel et al., 2026. arXiv: **2603.12880**.
"Inherently Interpretable Components" embed domain concepts directly into the model architecture; performance preserved; tested on wearable health monitoring (state assessment, seizure detection). **Why it matters:** rare wearable-specific XAI paper that does *not* rely on post-hoc methods — the operator-facing explainability path.

**[31] Wearable Sensor-Based IoT-XAI Framework for Predicting Freezing of Gait in Parkinson's Disease** — Biplov Paneru et al., 2025. arXiv: **2507.01068**.
ESP-32 microcontroller + LoRa; XGBoost/CatBoost/Extra-Trees; SHAP attribution surfaces "GYR SI degree" as top predictor. **Why it matters:** operationally relevant: shows tabular SHAP-style XAI can run end-to-end on a constrained edge node — a fallback path if neural-XAI is too heavy.

### 2J. Canonical anchors (older but indispensable)

**[A] Project Aria: A New Tool for Egocentric Multi-Modal AI Research** — Jakob Engel et al., 2023. arXiv: **2308.13561**.
Meta's egocentric multimodal data-recording device; the substrate for nearly all 2024–2026 egocentric multimodal work referenced above. *Outside the 2024–2026 window but called out as the canonical hardware/data reference per the spec's "older paper if canonical" allowance.*

**[B] ImageBind: One Embedding Space To Bind Them All** — Rohit Girdhar et al., 2023. arXiv: **2305.05665**.
Six-modality joint embedding (image, text, audio, depth, thermal, IMU) trained with image-paired data only. *Outside the window but cited by [10] (Open-Vocabulary AVE) as the training-free backbone, and is the reason a single joint embedding for tactical multimodal queries is even plausible.*

## 3. Datasets

| Dataset | HF / Identifier | Modalities | Licence | Notes for tactical-edge use |
|---|---|---|---|---|
| **AudioSet** | `agkphysics/AudioSet` (HF, verified via `hf_inspect_dataset`) | Audio (10s clips), labels | CC-BY 4.0 (audio derivatives), source YouTube TOS apply — **flag** | The de facto large-scale acoustic event corpus; balanced split is ~24 GB. |
| **Ego4D** | github.com/facebookresearch/Ego4d (591 stars, MIT) — data via signed Ego4D licence | Egocentric video, audio, narrations, some IMU | **Restrictive** Ego4D licence (research-only, signed agreement) — **flag** | Foundational for egocentric audio-visual fusion; not openly redistributable. |
| **Ego-Exo4D** | ego-exo4d-data.org; arXiv: **2311.18259** | First + third-person video, audio, IMU, gaze | Ego4D licence (signed) — **flag** | Adds skilled-activity third-person view; useful as training-time supervisor for an IMU-only deployment (the COMODO pattern). |
| **EPIC-KITCHENS-100** | epic-kitchens.github.io (HF dataset listing exists but not always loaded — `EPIC-KITCHENS-100/EPIC-KITCHENS-100` returned no schema in inspector) | Egocentric video, audio | CC-BY-NC 4.0 — **flag** (non-commercial) | Standard benchmark for egocentric action recognition. **Verification note:** HF dataset page exists but `hf_inspect_dataset` returned empty schema; download from project site is the canonical route. |
| **Project Aria Open Datasets** | explorer.projectaria.com | RGB, slam cameras, IMU, audio, eye gaze | Aria research licence — **flag** | Closest physical analogue to a body-worn sensor pack. |
| **OV-AVEBench** | from arXiv: **2411.11278** (released alongside paper) | Audio + video, 24,800 clips, 67 categories | Per paper release (likely research) — **flag** | Newest open-vocabulary audio-visual benchmark. |
| **VAAR** | from arXiv: 2510.13630 (paper withdrawn — **DO NOT USE** until v2 reappears) | Audio + video, 3,000 clips, 10 anomaly classes | n/a | Flagged because the paper was withdrawn for additional analysis. |
| **CMU-MMAC** | kitchen.cs.cmu.edu | Video + audio + RFID + IMU | Free for research use — **flag** | The interpretability paper [28] uses this; provides multimodal HAR with synchronized streams. |
| **A Multiclass Acoustic Drone Dataset** | from arXiv: **2509.04715** | Audio (32 drone classes, 16,000 s) | Per paper release | Tactical-relevant: supports drone-acoustic detection retraining. |
| **LSM (Wearable Foundation Model corpus)** | from arXiv: **2410.13638** (Google) | HR, HRV, EDA, accelerometer, skin temp, altimeter | **Not openly released** — proprietary Fitbit/Pixel dataset — **flag** | Useful only as a benchmark reference, not a path; an operator-grade equivalent does not exist openly. |
| **SensorLM corpus** | from arXiv: **2506.09108** | Wearable signals + language | **Not openly released** at scale | Same caveat as LSM. |

## 4. Reference Implementations

| Repo | Stars | Licence | Tied to paper |
|---|---|---|---|
| github.com/cruiseresearchgroup/COMODO | 24 | MIT | [1] COMODO (2503.07259) |
| github.com/facebookresearch/Ego4d | 591 | MIT | Ego4D dataset tooling |
| github.com/facebookresearch/ImageBind | ~9,000 | **CC-BY-NC 4.0** — non-commercial flag | [B] ImageBind |
| github.com/huggingface/smollm | ~3,800 | Apache-2.0 | [18] SmolVLM and the SmolVLM2 video variant; ONNX checkpoints + WebGPU demo |
| github.com/itsjunwei/Realtime-SELD-Edge | (unverified — surfaced via search) | (per repo) | Real-time SELD on edge — companion to [11] |
| github.com/teco-kit/ISWC22-HAR | (unverified, from the TinyHAR predecessor of TinierHAR [4]) | per repo | TinyHAR — direct ancestor of [4] |
| huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct | n/a (HF model card) | Apache-2.0 | [18] SmolVLM video variant; 1.38 GB GPU RAM at video inference |
| huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct | n/a | Apache-2.0 | [18] |
| github.com/rh20624/Awesome-IMU-Sensing | (unverified — survey meta-repo) | per repo | Companion to [7] survey |
| github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey | (unverified) | per repo | Index for efficient-foundation-model line, including [22] |

**Implementation gaps observed:** there is **no fully open implementation** that simultaneously combines (a) audio + video + IMU at sub-100M parameters, (b) federated/online training under missing modalities, and (c) operator-facing explainability. The COMODO repo is the closest open thing for the IMU-from-video-distillation primitive; everything beyond that has to be glued together.

## 5. Open Questions Noticed (flagged, not answered)

These were spotted while reading. They are gaps to record — not hypotheses.

- **Q1.** Cross-modal distillation (e.g. COMODO [1]) demonstrates video → IMU label-free transfer at the activity-recognition level. It is unclear how well this extends to *event-level* tactical-relevant signals (gunshot, vehicle approach) where the temporal structure differs from coarse activities and the IMU signal is much weaker.

- **Q2.** Open-vocabulary audio-visual event localization [10] uses ImageBind, which is **CC-BY-NC** — no operational pathway exists. The 2024–2026 corpus has no open-licence drop-in replacement at comparable capability for the IMU branch. This is a concrete licence gap, not just a research gap.

- **Q3.** Federated multimodal work [16, 17] addresses modality missing as a *steady-state* condition with prototype mitigation, but does not address the *transient* connectivity pattern of tactical operators (radio silence, sudden burst sync, MANET partitions). The benchmarks are simulated IID/non-IID, not link-failure models.

- **Q4.** The mobile MLLMs ([18], [19], [20]) are all benchmarked on consumer mobile SoCs (Dimensity 9500, etc.) — no paper in this batch reports on tactical-grade SoCs (e.g., NVIDIA Jetson Orin Nano-class or specialized DSP envelopes), so the SWaP numbers do not transfer cleanly.

- **Q5.** Explainability work splits cleanly into "tabular SHAP on a microcontroller" [31] and "multimodal cross-modal influence principles" [29] with **no paper bridging the two for an operator-facing wearable** that produces real-time, on-device, multimodal explanations a soldier can act on.

- **Q6.** Uncertainty quantification under degraded sensors ([13], [15]) is mature for autonomous driving and medical imaging but **the audio-IMU pair specifically** is not addressed in any 2024–2026 paper found here — even though it is the most likely dyadic combination after a tactical operator drops the camera.

- **Q7.** The wearable foundation model line ([9] LSM, [3] SensorLM) trains on **proprietary** corpora at the 40M-hours / 100k-people scale. There is no obvious open path to that scale of multi-sensor wearable data — which constrains the "open-data only" candidate-shortlist constraint of the IDEaS spec.

- **Q8.** Quantization papers for audio [23] are Whisper-specific. No published 2024–2026 paper covers **joint quantization of audio + vision + IMU encoders** in a single fused model — each branch is quantized in isolation in the surveyed literature.

## 6. Sources

All citations below resolved via arxiv.org/abs/{id} (verified via WebFetch). Spot-checked via `mcp__plugin_megaresearcher_ml-intern__hf_papers paper_details` — see verification.md for tool-status note.

- arXiv: 2503.07259 — COMODO
- arXiv: 2411.11278 — Open-Vocabulary AVE Localization
- arXiv: 2505.16138 — MMO-FL
- arXiv: 2503.19776 — MoME
- arXiv: 2504.05299 — SmolVLM
- arXiv: 2510.22410 — Multimodal Fusion + Interpretability HAR
- arXiv: 2503.09905 — Whisper Quantization
- arXiv: 2410.13638 — Scaling Wearable Foundation Models (LSM)
- arXiv: 2510.11496 — AndesVL
- arXiv: 2503.21782 — Mobile-VideoGPT
- arXiv: 2501.17823 — Cross-Modal Proxy Tokens
- arXiv: 2402.14905 — MobileLLM
- arXiv: 2507.07949 — TinierHAR
- arXiv: 2509.04736 — WatchHAR
- arXiv: 2508.12213 — Generalizable HAR Survey
- arXiv: 2502.07855 — VLMs for Edge Networks Survey
- arXiv: 2506.09108 — SensorLM
- arXiv: 2509.04715 — Drone Acoustic Dataset
- arXiv: 2603.12880 — XAI Inherently Interpretable Components
- arXiv: 2508.13728 — BioGAP-Ultra
- arXiv: 2409.06341 — Wearable Multi-Modal Edge Kitchen
- arXiv: 2412.18024 — Discounted Belief Fusion
- arXiv: 2401.03497 — EAT
- arXiv: 2506.18927 — TinyML→TinyDL Survey
- arXiv: 2504.11467 — Multicore Edge-TPU TinyML Livestock
- arXiv: 2508.11159 — Modality Quantity/Quality Imbalance MMO-FL
- arXiv: 2507.16343 — Detect Any Sound (DASM)
- arXiv: 2506.13060 — Rethinking Explainability Multimodal
- arXiv: 2404.05206 — SoundingActions
- arXiv: 2507.01068 — Wearable Sensor IoT XAI Parkinson
- arXiv: 2404.15349 — Multimodal Wearable Sensor HAR Survey
- arXiv: 2308.13561 — Project Aria (canonical anchor, 2023)
- arXiv: 2305.05665 — ImageBind (canonical anchor, 2023)
- arXiv: 2311.18259 — Ego-Exo4D (dataset reference)

GitHub repos: facebookresearch/ImageBind (~9k), facebookresearch/Ego4d (591), cruiseresearchgroup/COMODO (24), huggingface/smollm (~3.8k).

HF datasets verified via `hf_inspect_dataset`: `agkphysics/AudioSet` (Valid). `EPIC-KITCHENS-100/EPIC-KITCHENS-100` schema empty in inspector — flagged.
