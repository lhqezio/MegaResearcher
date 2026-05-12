# Gap-Finder-1 — Modality-Pair Gaps Across the Multimodal-Fusion Landscape

## 1. Slice scope

Read in full:

- `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/scout-1/output.md` (EO/IR + SAR; 24 arXiv IDs)
- `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/scout-2/output.md` (RF / SIGINT + imagery/text; 25 arXiv IDs)
- `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/scout-3/output.md` (audio + video + sensor at the tactical edge; 31 arXiv IDs + 2 anchors)
- `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/scout-4/output.md` (text-intel + imagery; 27 arXiv IDs)
- `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/scout-5/output.md` (sonar / maritime; 27 arXiv IDs)
- `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/scout-6/output.md` (cross-cutting capability axes; 41 arXiv IDs)

Modality set in scope: `{EO/IR, SAR, RF/SIGINT, audio, text-intel, telemetry, sonar}`. Capability dimensions: `{spatiotemporal alignment, uncertainty propagation, policy-aware provenance, SWaP-aware edge deployment, operator explainability}`.

## 2. Modality-pair × capability matrix

Cell legend: **served** = ≥2 well-cited 2024–2026 papers explicitly addressing the cell with code/dataset releases; **thin** = at most one paper, or only adjacent / partial coverage; **absent** = no retrievable open paper or dataset in 2024–2026 found in scout output or my verification queries.

| Pair | Spatiotemporal alignment | Uncertainty propagation | Policy-aware provenance | SWaP-aware edge | Operator explainability |
|---|---|---|---|---|---|
| **EO/IR + SAR** | served — CROMA 2311.00566; M4-SAR 2505.10931; X-STARS 2405.09922; MultiResSAR 2502.01002 | thin — Tulbure 2512.02055 (when-helps eval, no UQ); evidential-fusion line (2412.18024, 2503.05274) not yet applied to EO/SAR FMs | absent — no EO/SAR FM ships PROV-JSON/Atlas-style classification-aware lineage (Atlas 2502.19567 + yProv4ML 2507.01078 are modality-agnostic) | thin — RingMoE 2504.03166 1B compressed; M³amba 2503.06446 linear-time; no on-device numbers reported for EO+SAR fusion | thin — EarthMind 2506.01667; CLOSP 2507.10403; SkySense / Galileo / DOFA / TerraMind expose patch metrics only |
| **EO/IR + text-intel** | served — TEOChat 2410.06234 (temporal); UrbanCross 2404.14241; LHRS-Bot 2402.02544 | thin — VHM 2403.20213 hallucination control; conflict UQ in 2506.14817 (text/event-only); Calibration on MLLMs 2505.03788 | thin — agent topologies (GeoLLM-Squad 2501.16254, RS-Agent 2406.07089) and KG survey 2402.05391 expose lineage hooks but no open EO-VLM ships them | thin — Falcon 2503.11070 sub-1B; GeoLLaVA 2410.19552 PEFT; SmolVLM 2504.05299; no SWaP-vs-accuracy frontier study for EO-VLMs | served — GeoChat 2311.15826; SkySenseGPT 2406.10100 (scene graph); SNIFFER 2403.03170; CrisisKAN 2401.06194 |
| **EO/IR + RF/SIGINT** | absent — Multimodal-NF 2603.28280 is near-field/low-altitude only; no satellite-EO + co-located passive-RF dataset or paper | absent | absent | absent | absent |
| **EO/IR + audio** | served (egocentric) — OV-AVE 2411.11278; SoundingActions 2404.05206; WatchHAR 2509.04736; DASM 2507.16343 | thin — Discounted Belief Fusion 2412.18024 not specifically validated on audio-visual at edge; Any2Any 2411.10513 KITTI-only | absent | served — WatchHAR 2509.04736 sub-12 ms; SmolVLM2 2504.05299; AndesVL 2510.11496; FastVLM 2412.13303; Mobile-VideoGPT 2503.21782 | thin — Multimodal Fusion + Interpretability HAR 2510.22410; Rethinking Explainability 2506.13060 (principles only) |
| **EO/IR + telemetry (IMU/GPS)** | served — RGB+LiDAR 2504.19002; Aria 2308.13561 substrate | thin — Conformal mm-Reg 2410.19653; Evidential trajectory 2503.05274 | absent | served — TinierHAR 2507.07949; LiteVLM 2506.07416; BioGAP-Ultra 2508.13728 | thin — XAI Inherently Interpretable Components 2603.12880; Wearable IoT-XAI 2507.01068 (tabular only) |
| **EO/IR + sonar** | thin — only AONeuS-style optical-sonar surface reconstruction (cited in scout-5 verification), no ATR/anomaly fusion paper found | absent | absent | absent | absent |
| **SAR + text-intel** | thin — EarthDial 2412.15190 carries SAR as passenger modality; no SAR-native VLM in GeoChat mold | absent | absent | absent | thin — SARCLIP 2510.22665 caption only; CLOSP 2507.10403 retrieval; no SAR-VLM with rationale generation |
| **SAR + RF/SIGINT** | absent | absent | absent | absent | absent |
| **SAR + audio / SAR + sonar / SAR + telemetry** | absent / absent / thin (telemetry as platform metadata in M4-SAR 2505.10931) | absent | absent | absent | absent |
| **RF/SIGINT + audio** | absent — closest is RF-Behavior 2511.06020 (RF + IR + IMU but no audio) | absent | absent | absent | absent |
| **RF/SIGINT + text-intel** | thin — RF-GPT 2602.14833 and PReD 2603.28183 both use *synthetic* RF-instruction text, not OSINT; no paper fuses live RF observations with external operator reports | absent | absent | absent | thin — Seeing Radio 2601.13157 (rationales for modulation only) |
| **RF/SIGINT + telemetry** | served — SecureLink 2511.05796 (RF + MEMS); Tao 2510.22947 (RF + radar + EO); Multimodal-NF 2603.28280 (CSI + GPS + LiDAR + RGB) | thin — point-estimate authentication scores in 2511.05796 / 2510.22947 | absent | absent | absent |
| **Audio + telemetry (IMU)** | served — WatchHAR 2509.04736; COMODO 2503.07259 (video→IMU but applicable); kitchen multi-sensor 2409.06341 | absent — no 2024–2026 paper specifically propagates uncertainty for audio+IMU pair (scout-3 Q6) | absent | served — WatchHAR 2509.04736; Whisper-Q 2503.09905; EAT 2401.03497 | thin — IoT-XAI Parkinson 2507.01068; XAI Inherently Interpretable 2603.12880 |
| **Audio + sonar** (above-water acoustic + below-water acoustic) | absent — no surveyed paper joins above-water acoustic event detection with passive sonar streams | absent | absent | absent | absent |
| **Sonar + AIS** | thin — xView3 2206.00897 (SAR+AIS, not sonar); AIS-LLM 2508.07668 (AIS only); no sonar+AIS paired dataset (explicitly flagged in scout-5 dataset survey 2510.03353) | thin — GMvA 2504.09197 has uncertainty fusion module but for AIS+CCTV only | absent | thin — NAS-DETR 2505.06694 sonar-only edge | served — AIS-LLM 2508.07668; XAI for MASS 2509.15959; Interpretable contrastive UATR 2402.12658 |
| **Sonar + EO/IR + AIS (maritime tri-modal)** | thin — multi-modal/multi-view marine navigation 2505.01615 (RGB + LWIR + LiDAR, no sonar/AIS); SMART-Ship 2508.02384 (5 satellite modalities, no sonar/AIS) | thin — GMvA 2504.09197 (AIS+CCTV only) | absent | absent | thin — XAI for MASS 2509.15959 review; AIS-LLM 2508.07668 |
| **Sonar + RF + visual + AIS (maritime quad-modal)** | absent | absent | absent | absent | absent |
| **EO/IR + SAR + AIS (dark-vessel)** | served — xView3 2206.00897; allenai/vessel-detection-sentinels; YOLO+AIS inland 2510.11449 | thin — outlier RNN 2406.09966; no calibrated cross-modal UQ on dark-vessel | absent | thin — xView3 FPGA work referenced (2507.04842, untriaged) | thin — agency-driven event narratives (Global Fishing Watch) but no architecture-level rationale generation |
| **Text-intel + event streams (ACLED/GDELT) + EO/IR (joint forecasting)** | absent — Nemkova 2505.09852 RAG (text only); von der Maase 2506.14817 spatial U-Net (event grids only); STFT-VNNGP 2506.20935 (events only). No joint text+event+imagery forecasting paper found | absent | absent | absent | absent |

## 3. Discrete gaps

Each gap is stated as a *claim about the literature* per the gap-finder contract: type, statement, evidence (with arXiv IDs), verification query, and why it matters for the spec.

---

### Gap 1 — EO + passive RF/SIGINT at orbital scale is unexplored

- **Type:** Unexplored intersection.
- **Statement:** No open 2024–2026 paper or dataset jointly trains or evaluates a fusion model on co-located satellite EO imagery and passive RF/SIGINT collected at orbital scale; the closest analogue is near-field, low-altitude (Multimodal-NF; arXiv 2603.28280) or terrestrial RF + IR (RF-Behavior; arXiv 2511.06020).
- **Evidence:**
  - EO-side foundation models all train on Sentinel-1+2 collocations — CROMA (arXiv 2311.00566), DOFA (arXiv 2403.15356), MMEarth (arXiv 2405.02771), SkySense (arXiv 2312.10115), TerraMind (arXiv 2504.11171), Galileo (arXiv 2502.09356), RingMoE (arXiv 2504.03166). None ingests RF.
  - RF-side foundation models all stay within the wireless modality stack — Multimodal Wireless FM (arXiv 2511.15162), 6G WavesFM (arXiv 2504.14100), 6G Radio FM (arXiv 2411.09996), LWM-Spectro (arXiv 2601.08780). None ingests EO.
  - The only RF+image fused datasets are near-field — Multimodal-NF (arXiv 2603.28280, drone-altitude CSI+RGB+LiDAR+GPS) and RF-Behavior (arXiv 2511.06020, indoor RF+IR+IMU).
- **Verification query:** WebSearch `"satellite EO RF SIGINT fusion Arctic ISR 2024 2025 deep learning multimodal"` — 10 results returned; only the 2021 Vakil/Liu IEEE survey (Oakland) explicitly addresses passive-RF + EO and pre-dates the 2024–2026 window. WebSearch `"passive RF satellite EO geostationary ISR fusion deep learning paper 2025"` — 10 results, none are 2024–2026 satellite-EO + passive-RF fusion papers (only one 2021 survey + EO-only spatiotemporal-fusion work). Result: zero in-scope hits ⇒ supports the gap.
- **Why it matters for the spec:** The IDEaS Arctic-ISR application context (satellite imagery + RF + telemetry) is one of the spec's five named scenarios. An empty cell here is the most operationally consequential gap on the matrix.

---

### Gap 2 — RF/SIGINT + text-intel (OSINT) fusion is essentially absent (only synthetic instruction-text exists)

- **Type:** Unexplored intersection.
- **Statement:** No open 2024–2026 paper fuses live RF/SIGINT observations with external operator reports or OSINT text streams; the two papers that combine RF with language (RF-GPT; arXiv 2602.14833 and PReD; arXiv 2603.28183) use *synthetic instruction-style descriptions of the RF*, not real text-intel feeds correlated with live signals.
- **Evidence:**
  - RF-GPT (arXiv 2602.14833) — instruction corpus is ~625K synthetic examples generated to describe RF.
  - PReD (arXiv 2603.28183) — 1.3M-sample dataset of RF features + LLM, but the text side is curated descriptions, not OSINT.
  - SEI-SHIELD (arXiv 2605.04721) — RF-only specific-emitter ID with no text component.
  - On the text-intel side, GDELT/ACLED-class fusion stops at imagery — Nemkova et al. (arXiv 2505.09852) RAG over ACLED/GDELT/news; Tarekegn (arXiv 2406.10552) GDELT clustering. None reach the RF stack.
- **Verification query:** WebSearch `"RF SIGINT text intel OSINT fusion arxiv 2025 multimodal LLM"` — 10 results; none describe live-RF + OSINT-text fusion. Top hit (arXiv 2506.04788, "Towards LLM-Centric Multimodal Fusion") is a generic survey of 125 MLLMs that does not name RF↔text-intel as an instantiated pair. Result: zero hits ⇒ supports the gap.
- **Why it matters for the spec:** Real-time multi-domain threat assessment (EO video + SIGINT + text intel) is the second IDEaS application context; the SIGINT↔text-intel leg of that triple is the limiting cell.

---

### Gap 3 — Sonar paired with any non-acoustic surface-domain modality (RF, AIS, EO/IR) has no open dataset or fused-architecture paper

- **Type:** Unexplored intersection (and confirmed missing-dataset).
- **Statement:** No open 2024–2026 dataset or fusion model couples sonar with RF, AIS, EO/IR, or SAR. The Sonar Image Datasets survey (arXiv 2510.03353) explicitly catalogues every open sonar dataset through 2025 and confirms none combines sonar with AIS or RF; the closest tri-modal underwater fusion is optical+acoustic+pressure (arXiv 2502.20817) which has no surface-domain bridge, and the closest surface tri-modal is RGB+LWIR+LiDAR (arXiv 2505.01615) which has no sonar.
- **Evidence:**
  - Sonar Image Datasets survey, arXiv 2510.03353 — explicitly states no public dataset combines sonar with AIS.
  - Optical-acoustic-pressure trimodal AUV, arXiv 2502.20817 — underwater only.
  - Multimodal/multiview marine navigation, arXiv 2505.01615 — RGB + LWIR + LiDAR + radar + ENC chart, no sonar.
  - SMART-Ship, arXiv 2508.02384 — 5 satellite modalities (visible/SAR/panchromatic/multispectral/NIR) but zero sonar and zero AIS.
  - xView3-SAR, arXiv 2206.00897 — SAR + AIS but no sonar.
  - SeafloorAI, arXiv 2411.00172 — sonar + bathymetry + text but no AIS/RF.
- **Verification query:** WebSearch `"sonar AIS fusion deep learning 2024 2025 maritime anomaly detection"` — 10 results, all of which fuse AIS with radar/EO/CCTV/LiDAR but never with sonar. WebSearch `"sonar RF visual fusion underwater surface multimodal benchmark 2025 anomaly"` — 10 results: AONeuS (acoustic-optical underwater), underwater SLAM, optical-sonar — all underwater or lab settings; none cross the above-water/below-water boundary with RF or AIS. Result: zero in-scope hits ⇒ supports the gap.
- **Why it matters for the spec:** Maritime task-group anomaly detection (sonar + RF + visual + uncertainty) is the fourth IDEaS application context. The cell that would underwrite this scenario is empty.

---

### Gap 4 — No EO/SAR foundation model (or any cross-modal foundation model in scope) ships calibrated cross-modal uncertainty

- **Type:** Untested assumption (the assumption being that current foundation-model fusion is well-calibrated, which is asserted nowhere and tested nowhere).
- **Statement:** Every 2024–2026 EO/SAR cross-modal foundation model surveyed reports patch-level metrics without calibrated cross-modal uncertainty; the evidential-fusion line (arXiv 2412.18024, 2409.00755, 2408.13123, 2503.05274) and the conformal-fusion line (arXiv 2410.19653, 2411.10513) are developed in parallel sub-fields and have not been applied to a SAR+optical foundation model in any retrievable paper.
- **Evidence:**
  - EO/SAR FMs without UQ: CROMA (2311.00566), DOFA (2403.15356), MMEarth (2405.02771), Galileo (2502.09356), SkySense (2312.10115), TerraMind (2504.11171), RingMoE (2504.03166), EarthMind (2506.01667), M4-SAR (2505.10931), SARATR-X (2405.09365). Tulbure flood-mapping eval (2512.02055) reports performance but no calibrated UQ.
  - Cross-modal UQ techniques exist but separately: Discounted Belief Fusion (2412.18024), TUNED (2409.00755), Evidential Partial Multi-View (2408.13123), Evidential Trajectory (2503.05274), Conformal multimodal regression (2410.19653), Any2Any (2411.10513), MLLM-grounding calibration (2505.03788).
  - Scout-1 explicitly flags this (open question: "Uncertainty propagation across modalities is treated as an afterthought").
- **Verification query:** WebSearch `"EO SAR foundation model uncertainty calibration evidential fusion 2024 2025"` — 10 results; the SAR/EO foundation-model results (DOFA, FAST-EO project page, GRSS DFC2025) discuss fusion but explicitly *not* UQ. The evidential-fusion results are all medical/AV (medical image segmentation, fundus disease, object detection in autonomous vehicles). Cross-product is empty. WebSearch `"conformal prediction evidential fusion combined multimodal 2024 2025"` — 10 results: HyperDUM (CVPR 2025), conformalized multiview, EsurvFusion (medical) — none on EO/SAR. Result: supports the gap.
- **Why it matters for the spec:** Uncertainty propagation is one of the five IDEaS desired-outcome capabilities and is a load-bearing requirement for the whole proposal narrative ("AI-driven architectures that... propagate uncertainty"). The core EO/SAR backbone family does not deliver it.

---

### Gap 5 — Joint text-intel + structured-event (ACLED/GDELT) + EO imagery forecasting is unexplored

- **Type:** Unexplored intersection.
- **Statement:** No 2024–2026 paper jointly forecasts on all three of {OSINT/text-intel, structured event streams (ACLED/GDELT), satellite imagery}; the literature splits cleanly into (a) text + structured-event RAG, (b) gridded-event-only spatial forecasting, and (c) imagery + ACLED point-prediction — but never the triple.
- **Evidence:**
  - Text + structured event only — Nemkova et al. arXiv 2505.09852 ("Do LLMs Know Conflict?") uses RAG over ACLED+GDELT+news; no imagery.
  - Structured event only, gridded — von der Maase arXiv 2506.14817 (MC-Dropout LSTM-U-Net on conflict event grids); no imagery, no text.
  - Geopolitical event forecasting — STFT-VNNGP arXiv 2506.20935 (Sparse TFT + GP, GDELT only); no imagery.
  - Imagery + ACLED only — CNN-on-Landsat-8 + ACLED conflict-fatality risk in Nigeria (MDPI Remote Sensing 2024, surfaced via verification search, doi 10.3390/rs16183411) — point-estimate, no text-intel.
  - Tarekegn arXiv 2406.10552 — GDELT clustering with LLM; no imagery.
  - Governing Automated Strategic Intelligence arXiv 2509.17087 — taxonomy paper that *names* the gap but does not fill it.
- **Verification query:** WebSearch `"text intel" OR "OSINT" AND "ACLED" AND "satellite imagery" joint forecasting deep learning 2024 2025` — 10 results, none combine all three; WebSearch `conflict forecasting satellite imagery ACLED GDELT joint multimodal arxiv 2024 2025` — 10 results: 2505.09852 (text+events only), 2506.20935 (events only), MDPI 2024 (imagery+events only). Result: supports the gap.
- **Why it matters for the spec:** The Real-Time Multi-Domain Threat Assessment application context (EO video + SIGINT + text intel) overlaps directly with conflict forecasting; the empty cell here means the most plausible academic-domain analogue still doesn't exist.

---

### Gap 6 — Cross-classification-level / cross-modality lineage is asserted but not implemented

- **Type:** Missing baseline (the obviously-stronger baseline being a fusion stack that records *per-modality* PROV-JSON / Atlas-style lineage including classification level; current ML-provenance frameworks are modality-agnostic and current EO/SAR fusion stacks ship no lineage at all).
- **Statement:** The 2024–2026 ML-provenance frameworks (Atlas arXiv 2502.19567; yProv4ML arXiv 2507.01078; HASC arXiv 2509.20394) are modality-agnostic and do not demonstrate per-modality classification-level lineage; the multimodal fusion architectures surveyed (EO/SAR, RF, audio-IMU, sonar, text-intel) do not implement any of them. The intersection cell — provenance-instrumented multimodal fusion at classification boundaries — is empty.
- **Evidence:**
  - Provenance frameworks: Atlas (2502.19567) — fully attestable ML pipelines, modality-agnostic; yProv4ML (2507.01078) — PROV-JSON + energy metrics; HASC (2509.20394) — Hazard-Aware System Cards.
  - Federated multimodal that touch provenance adjacency: FedEPA (2504.12025), SHIFT (2506.05683) — both assume horizontal client splits, not classification-level splits.
  - Empirical evidence of insufficiency: Liang et al. (2402.05160) on 32K HF model cards documents systematic gaps; Stalnaker et al. (2502.04484) on 760K models + 175K datasets surfaces licensing/supply-chain opacity.
  - Scout-6 explicitly flags this (open question: "Differential privacy under multi-modal fusion at classification boundaries... not addressed").
- **Verification query:** WebSearch `"multimodal classification level provenance differential privacy fusion ISR 2024 2025"` — 10 results; the multimodal-DP work is biomedical or sentiment, the multimodal-fusion work has no classification-level lineage hooks. WebSearch `"cross-classification level fusion provenance lineage SECRET UNCLAS multimodal 2025"` — 10 results, all generic multimodal fusion / medical, none on classification-level provenance. Result: supports the gap.
- **Why it matters for the spec:** Policy-aware provenance across classification levels is an explicit IDEaS desired-outcome capability and a CAF/DND policy requirement. The empty cell means there is *no* drop-in baseline.

---

### Gap 7 — No reported on-device latency/power numbers for EO+SAR fused detectors at edge

- **Type:** Untested assumption (papers claim SWaP-aware-ness without providing a measurement).
- **Statement:** No 2024–2026 paper reports actual on-device inference latency or power for an EO+SAR fused detector in a tactical-edge envelope (Jetson Orin / Kria / GAP9-class). Claims of efficiency (RingMoE arXiv 2504.03166 1B compressed; M³amba arXiv 2503.06446 linear-time) are made without measurements; the on-device benchmark community focuses on either single-modality SAR (NAS-DETR arXiv 2505.06694) or VLMs on consumer phones (AndesVL arXiv 2510.11496; LiteVLM arXiv 2506.07416) — never on EO+SAR fusion.
- **Evidence:**
  - Claimed-efficient EO/SAR FMs without on-device numbers: RingMoE (2504.03166), M³amba (2503.06446), Falcon (2503.11070).
  - On-device VLM benchmarks (no SAR) on consumer SoCs: AndesVL on Dimensity 9500 (2510.11496); LiteVLM on NVIDIA embedded (2506.07416); FastVLM on Apple Silicon (2412.13303); SmolVLM (2504.05299); OmniVLM (2412.11475).
  - On-device single-modality SAR: NAS-DETR (2505.06694).
  - Scout-1 explicitly flags this; scout-3 Q4 corroborates ("no paper in this batch reports on tactical-grade SoCs").
- **Verification query:** WebSearch `"edge deployment SAR optical fusion FPGA Jetson power latency benchmark 2024 2025"` — 10 results, all generic Edge AI hardware survey content; no SAR+optical fused detector latency/power numbers found. Result: supports the gap.
- **Why it matters for the spec:** SWaP-aware edge deployment is one of the five IDEaS desired-outcome capabilities; without published on-device numbers for the fused stack, the candidate-architecture shortlist would inherit untested SWaP claims.

---

### Gap 8 — Audio + IMU pair lacks calibrated uncertainty propagation (the most-likely degraded-mode dyad has no UQ)

- **Type:** Unexplored intersection (within the wearable / tactical-edge sub-cluster).
- **Statement:** No 2024–2026 paper specifically propagates uncertainty across the audio + IMU pair, even though it is the most likely sub-modality combination for a tactical operator who has dropped the camera (the SWaP-default "no video, no comms" mode). The evidential / conformal multimodal UQ work (arXiv 2412.18024, 2503.05274, 2410.19653, 2411.10513) is developed on AV / medical / sentiment benchmarks and the audio-IMU fusion work (WatchHAR arXiv 2509.04736, COMODO arXiv 2503.07259, kitchen multi-sensor arXiv 2409.06341, AudioIMU 2022) reports point-estimate accuracies only.
- **Evidence:**
  - Audio + IMU fusion at the edge but no UQ: WatchHAR (2509.04736), COMODO (2503.07259), kitchen multi-sensor MCU (2409.06341), TinierHAR (2507.07949), AudioIMU (2022 / earlier).
  - UQ methods on other modality pairs only: Discounted Belief Fusion (2412.18024), Evidential Trajectory (2503.05274), Conformal mm-Reg (2410.19653), Any2Any (2411.10513), Uncertainty-Resilient Cross-Modal (2511.15741).
  - Scout-3 Q6 explicitly flags this gap.
- **Verification query:** WebSearch `"audio IMU fusion uncertainty quantification wearable 2024 2025 arxiv"` — 10 results: AudioIMU (2022, no UQ), generalizable HAR survey 2508.12213, IMU cross-modal transfer survey 2403.15444, sensor-fusion review 2412.05895 (multi-sensor health, Bayesian fusion is medical/clinical). No paper hits the audio+IMU + UQ + tactical-edge intersection. Result: supports the gap.
- **Why it matters for the spec:** Tactical-edge fusion on wearables under degraded connectivity is the third IDEaS application context; the audio+IMU dyad is the most plausible degraded-mode for that scenario, and lacks the UQ primitive the spec explicitly calls out.

---

### Gap 9 — No SAR-native VLM exists at GeoChat-class maturity (asymmetry between EO-VLM and SAR-VLM development)

- **Type:** Domain-transfer gap.
- **Statement:** EO-side VLMs reached production-class maturity by 2024 (GeoChat arXiv 2311.15826; LHRS-Bot arXiv 2402.02544; SkySenseGPT arXiv 2406.10100; VHM arXiv 2403.20213; TEOChat arXiv 2410.06234; Falcon arXiv 2503.11070; GeoVLM-R1 arXiv 2509.25026), and a parallel event-camera VLM exists (EventGPT arXiv 2412.00832), but no SAR-native VLM in the GeoChat mold has been released — SAR is treated only as a passenger modality (EarthDial arXiv 2412.15190, EarthMind arXiv 2506.01667) or a separate caption-only encoder (SARCLIP arXiv 2510.22665).
- **Evidence:**
  - Mature EO-VLMs: 2311.15826, 2402.02544, 2403.20213, 2406.10100, 2410.06234, 2503.11070, 2509.25026 — six independent releases over 18 months.
  - SAR-as-passenger: EarthDial (2412.15190), EarthMind (2506.01667).
  - SAR-encoder-only / caption-only: SARCLIP (2510.22665), CLOSP (2507.10403), SARATR-X (2405.09365 — backbone, not VLM).
  - Scout-4 Q10 explicitly flags this.
- **Verification query:** WebSearch `"SAR optical foundation model Canadian RADARSAT-2 RCM transfer X-band quad-pol 2024 2025"` — 10 results: SARATR-X (encoder, not VLM); RADARSAT mission descriptions; cross-frequency calibration for soil moisture (not VLM). No GeoChat-class SAR-native VLM identified. Result: supports the gap.
- **Why it matters for the spec:** Operator-facing explainability is one of the five IDEaS desired-outcome capabilities; without a SAR-native VLM, the IDEaS Arctic-ISR scenario (which by geography is SAR-dominant under cloud cover and polar night) inherits no rationale-generation primitive.

---

### Gap 10 — Cross-sensor SAR transfer (Sentinel-1 → RADARSAT-Constellation Mission / X-band micro-SAR) is essentially untested in the open literature

- **Type:** Domain-transfer gap.
- **Statement:** Every 2024–2026 EO/SAR cross-modal foundation model is pretrained on Sentinel-1 + Sentinel-2 collocations; whether the resulting representations transfer to Canadian RADARSAT-Constellation Mission C-band quad-pol or to airborne X-band SAR (e.g., ICEYE, Capella) is not benchmarked in any retrievable paper. Scout-1 names this; the SARATR-X dataset includes RADARSAT-2 *images* but not as a transfer-evaluation split, and no transfer-learning paper explicitly evaluates on RCM.
- **Evidence:**
  - Sentinel-1+2 pretrained: CROMA (2311.00566), DOFA (2403.15356), MMEarth (2405.02771), TerraMind (2504.11171), Galileo (2502.09356), SkySense (2312.10115), RingMoE (2504.03166).
  - Cross-sensor alignment work targets resolution mismatch within Sentinel: X-STARS (2405.09922); MultiResSAR (2502.01002).
  - SARATR-X (2405.09365) corpus includes RADARSAT-2 + TerraSAR-X + Sentinel-1 mixed but does not publish a held-out RCM transfer evaluation.
  - Scattering Prompt Tuning (CVPR 2024 Workshop) — fine-tunes a foundation model on SAR objects, not specifically RCM cross-sensor.
- **Verification query:** WebSearch `"RADARSAT RCM SAR foundation model fine-tune 2024 2025 cross-sensor transfer"` — 10 results: only SARATR-X mentions RADARSAT-2 (training corpus, not a held-out RCM transfer eval); cross-frequency soil-moisture calibration (Springer 2026) is L-band/C-band physics, not foundation-model transfer. Result: supports the gap.
- **Why it matters for the spec:** Arctic ISR for CAF specifically depends on RADARSAT-Constellation Mission imagery; if the foundation-model bibliography assumes Sentinel-1 transfer holds, that assumption is presently untested.

---

## 4. Discarded candidate gaps (rejected after verification)

These were considered but verification queries showed they are *not* gaps.

### Discarded 1 — "No 2024–2026 work on multimodal masked autoencoder pretraining for EO+SAR"

- **Why I considered it:** Scout-1 lists CROMA, MMEarth, and TerraMind, and I initially wondered if the joint-MAE recipe was thinly explored.
- **Why I rejected it:** CROMA (arXiv 2311.00566) and MMEarth (arXiv 2405.02771) are explicit MAE-style EO+SAR pretrained models with code and weights; Galileo (arXiv 2502.09356) extends this to multi-modal masked + contrastive joint training over time series. TerraMind (arXiv 2504.11171) demonstrates any-to-any generative completion. This cell is *served*, not absent. Verification: scout-1 already enumerates 6 well-cited 2024–2026 papers in this exact lane.

### Discarded 2 — "No 2024–2026 work on conformal prediction for multimodal models"

- **Why I considered it:** Wanted to check whether conformal calibration was missing from the surveyed corpus.
- **Why I rejected it:** Conformal Prediction for Multimodal Regression (arXiv 2410.19653) and Any2Any (arXiv 2411.10513) are both 2024 publications addressing multimodal conformal calibration, and HyperDUM (CVPR 2025, surfaced via WebSearch verification) extends this. Verification query: WebSearch `"conformal prediction evidential fusion combined multimodal 2024 2025"` returned 10 results showing this is an active and well-served line. The gap is the *combination* of conformal with evidential on EO/SAR foundation models (Gap 4 above), not the existence of multimodal conformal work.

### Discarded 3 — "No interpretable / inherently explainable multimodal architectures in the 2024–2026 corpus"

- **Why I considered it:** Wanted to confirm the operator-explainability cell wasn't simply empty across the board.
- **Why I rejected it:** I2MoE (arXiv 2505.19190, ICML 2025), KAN-MCP (arXiv 2504.12151), GMAR (arXiv 2504.19414), Head Pursuit (arXiv 2510.21518, NeurIPS 2025 spotlight), ConceptAttention (arXiv 2502.04320, ICML 2025 oral), and Multimodal MXAI Review (arXiv 2412.14056) all populate this cell. Scout-6 explicitly catalogues them. The cell is *served* in general; the gap is in specific modality pairs (e.g., SAR-VLM rationales, audio+IMU explanations) — Gap 9 and adjacent rather than the cell as a whole.

### Discarded 4 — "Federated learning for multimodal sensor fusion is unexplored"

- **Why I considered it:** Spec mentions degraded connectivity / tactical-edge.
- **Why I rejected it:** MMO-FL (arXiv 2505.16138) and the QQR follow-up (arXiv 2508.11159), FedEPA (arXiv 2504.12025), SHIFT (arXiv 2506.05683), and FLAME (arXiv 2503.04136 — federated multimodal RF fingerprinting) populate this cell. Scouts 2, 3, and 6 each independently surface federated-multimodal work. The cell is *served*, not absent. The remaining gap is *transient-connectivity* federated multimodal at classification boundaries (subsumed in Gap 6 above).

### Discarded 5 — "RGB+LiDAR autonomous-driving fusion has no 2024–2026 baselines"

- **Why I considered it:** As an adjacent baseline for RGB+telemetry tactical fusion.
- **Why I rejected it:** MoME (arXiv 2503.19776), RGB+LiDAR robust fusion (arXiv 2504.19002), Cross-Modal Proxy Tokens (arXiv 2501.17823) all address this and are well-cited. Discarded.

---

## 5. Summary

The matrix produces 18 modality-pair × capability cells flagged "absent" across 10 distinct gap claims. The most operationally consequential absent cells map directly onto the IDEaS application contexts: EO + RF (Arctic-ISR), RF + text-intel (real-time multi-domain threat), sonar + RF + visual + AIS (maritime task group). The cross-cutting capability gaps — uncalibrated cross-modal UQ, missing on-device SWaP measurements, and absent classification-aware provenance — apply across nearly every modality pair surveyed.

These gaps are stated as claims about the literature only; hypotheses, candidate architectures, and falsification criteria are out of scope for this gap-finder run (the spec's novelty target is `gap-finding`).
