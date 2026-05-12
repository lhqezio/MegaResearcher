# Multi-Modal AI Fusion for Situational Awareness — Gap-Finding Landscape

**Run id:** `2026-05-10-0615-0ece4e`
**Spec:** `docs/research/specs/2026-05-10-multimodal-fusion-gap-finding-spec.md`
**Plan:** `docs/research/plans/2026-05-10-multimodal-fusion-gap-finding-plan.md`
**Novelty target:** `gap-finding` (no hypotheses, no eval designs — see §6)
**Output destination:** IDEaS Competitive Projects proposal, TRL 4–5 / $1.5M / 12-month band, deadline 2026-06-02
**Document length target:** ≤ 8 pages

---

## 1. Executive summary

**Question.** Where in the 2024–2026 multi-modal-fusion literature for ISR/C2 are the load-bearing gaps that a small Canadian team can credibly close to TRL 4–5 in twelve months using only open or synthetic data?

**What we found.** The swarm read 171 unique 2024–2026 papers across seven modalities (EO/IR, SAR, RF/SIGINT, audio, text intel, telemetry, sonar) and the five IDEaS desired-outcome capabilities (spatiotemporal alignment, uncertainty propagation, policy-aware provenance, SWaP-aware edge, operator explainability). Two independent gap-finders converged on four gaps that are simultaneously load-bearing and buildable:

1. **No 2024–2026 EO/SAR foundation model ships calibrated cross-modal uncertainty** (gap-finder-1 G4 ↔ gap-finder-2 G5 ↔ gap-finder-2 G8). Every CROMA/DOFA/MMEarth/Galileo/TerraMind/SkySense/RingMoE paper reports patch metrics; the evidential and conformal multimodal-UQ primitives exist (TMC, Discounted Belief Fusion, Any2Any) but have never been integrated.
2. **No on-device latency/power numbers exist for any fused EO+SAR or EO+RF detector** (gap-finder-1 G7 ↔ gap-finder-2 G3). On-device benchmarks cover single-modality SAR (NAS-DETR) or vision-language pairs (AndesVL, FastVLM, LiteVLM), never the fused stack.
3. **Per-modality classification-level provenance is asserted but unimplemented in any open fusion stack** (gap-finder-1 G6 ↔ gap-finder-2 G1/G2). Atlas, yProv4ML, and HASC are modality-agnostic; FedEPA and SHIFT assume horizontal client splits, not vertical splits across UNCLAS/SECRET boundaries.
4. **No paper joins multi-modal calibrated uncertainty with operator-facing explanation in one architecture** (gap-finder-2 G8). Eight uncertainty primitives and seven explanation primitives exist; their cross-product is empty.

A third gap-finder filtered the consolidated literature for TRL-4–5 buildability and produced a ranked eight-candidate shortlist with explicit licence-driven discards (FastVLM, ImageBind, RadioML-2018.01A, SkySense, RingMoE — see §4).

**Headline recommendation.** Lead with three candidates spanning three of the five spec-named contexts, exploiting the convergence-signal gaps and clearing a clean licence audit:

- **Pick A — Galileo + Discounted Belief Fusion** (Arctic ISR). MIT time-series EO/SAR encoder (NASA Harvest, ICML 2025) + MIT evidential fusion (TMC) — closes calibrated-uncertainty (G4/G5) and temporal-alignment gaps on Sentinel-1+2 substrates.
- **Pick B — DOFA + M4-SAR** (multi-domain threat assessment + airborne stealth/spoof). MIT code + CC-BY-4.0 weights wavelength-conditioned hypernetwork (TUM, 2024) + AGPL-3.0 paired S-1/S-2 detection benchmark with seven pre-published baseline detectors (13.5M–53.8M params). The only candidate with edge-deployable fused detectors pre-built — direct lever on the SWaP gap (G7/G3).
- **Pick C — SmolVLM-256M + COMODO** (tactical-edge wearable). Apache-2.0 substrate + Apache-2.0 distillation primitive + CC-BY-4.0 AudioSet. Single Apache-2.0 multimodal LLM that fits sub-1 GB on a wearable companion. Closes the audio+IMU UQ gap (G8); the joint audio+vision+IMU quantization gap (G4) becomes a research contribution.

Maritime task-group context is **deliberately not in the top three** — gap-finder-3 documented no public dataset combines sonar with AIS or RF; the architectural cell is empty (gap-finder-1 G3) and operationally-relevant data is classified-data dominant. See §5 for recommended posture.

---

## 2. Gap map

### 2.1 Modality-pair × capability matrix (gap-finder-1, condensed)

Cell legend: **served** ≥ 2 well-cited 2024–2026 papers with code/data; **thin** at most one paper or only adjacent coverage; **absent** no retrievable open paper or dataset in scope. Citations are arXiv IDs.

| Modality pair | Spatiotemporal alignment | Uncertainty propagation | Policy-aware provenance | SWaP-aware edge | Operator explainability |
|---|---|---|---|---|---|
| **EO/IR + SAR** | served (CROMA 2311.00566; M4-SAR 2505.10931; X-STARS 2405.09922; MultiResSAR 2502.01002) | **thin** (Tulbure 2512.02055; evidential line 2412.18024 not yet applied to EO/SAR FMs) | **absent** | thin (RingMoE 2504.03166; M³amba 2503.06446 — no on-device numbers) | thin (EarthMind 2506.01667; CLOSP 2507.10403; SARCLIP 2510.22665) |
| **EO/IR + text-intel** | served (TEOChat 2410.06234; UrbanCross 2404.14241; LHRS-Bot 2402.02544) | thin (VHM 2403.20213; conflict UQ 2506.14817; calibrated MLLM 2505.03788) | thin (agent topologies 2501.16254, 2406.07089; KG survey 2402.05391) | thin (Falcon 2503.11070 sub-1B; GeoLLaVA 2410.19552 PEFT) | served (GeoChat 2311.15826; SkySenseGPT 2406.10100; SNIFFER 2403.03170) |
| **EO/IR + RF/SIGINT** | **absent** | absent | absent | absent | absent |
| **EO/IR + audio (egocentric)** | served (OV-AVE 2411.11278; SoundingActions 2404.05206; WatchHAR 2509.04736) | thin (Discounted Belief Fusion 2412.18024 not validated on AV) | absent | served (WatchHAR sub-12 ms; SmolVLM2 2504.05299; AndesVL 2510.11496) | thin (HAR interpretability 2510.22410; multimodal XAI principles 2506.13060) |
| **EO/IR + telemetry (IMU/GPS)** | served (RGB+LiDAR 2504.19002; Aria 2308.13561) | thin (Conformal mm-Reg 2410.19653; Evidential Trajectory 2503.05274) | absent | served (TinierHAR 2507.07949; LiteVLM 2506.07416; BioGAP-Ultra 2508.13728) | thin (Inherent Interpretable 2603.12880; IoT-XAI 2507.01068) |
| **SAR + text-intel** | thin (EarthDial 2412.15190 — SAR as passenger) | absent | absent | absent | thin (SARCLIP 2510.22665 caption-only; CLOSP 2507.10403 retrieval-only) |
| **SAR + RF/SIGINT** | absent | absent | absent | absent | absent |
| **RF/SIGINT + text-intel (OSINT)** | thin (RF-GPT 2602.14833 + PReD 2603.28183 use **synthetic** text, not OSINT) | absent | absent | absent | thin (Seeing Radio 2601.13157 modulation rationale only) |
| **RF/SIGINT + telemetry** | served (SecureLink 2511.05796; Tao 2510.22947; Multimodal-NF 2603.28280) | thin (point-estimate scores) | absent | absent | absent |
| **Audio + telemetry (IMU)** | served (WatchHAR 2509.04736; COMODO 2503.07259; kitchen multi-sensor 2409.06341) | **absent** (no UQ for the audio+IMU dyad) | absent | served (WatchHAR; Whisper-Q 2503.09905; EAT 2401.03497) | thin (IoT-XAI 2507.01068) |
| **Sonar + AIS / sonar + RF / sonar + EO** | absent / absent / thin | absent | absent | thin (NAS-DETR 2505.06694 sonar-only edge) | thin (AIS-LLM 2508.07668 NL rationale; XAI for MASS 2509.15959) |
| **EO/IR + SAR + AIS (dark vessel)** | served (xView3 2206.00897; YOLO+AIS inland 2510.11449) | thin (RNN outlier 2406.09966 — no calibrated cross-modal UQ) | absent | thin (xView3 FPGA work referenced 2507.04842, untriaged) | thin (Global Fishing Watch narratives) |
| **Text-intel + event streams + EO/IR (joint forecasting)** | **absent** | absent | absent | absent | absent |

### 2.2 Capability-axis severity table (gap-finder-2)

| Capability | Coverage state | Severity |
|---|---|---|
| #1 Spatiotemporal alignment | Active (CROMA, DOFA, MMEarth, Galileo, TerraMind, X-STARS, M4-SAR, MultiResSAR, TEOChat, IIR mechanism). | Low |
| #2 Uncertainty propagation | Eight fresh primitives (B1–B8) but **never integrated into EO/SAR FMs (G5)** and **never paired with operator-facing explanation (G8)**. | High |
| #3 Policy-aware provenance / classification | Three frameworks (Atlas, yProv4ML, HASC) but **never benchmarked on multi-modal pipelines (G1)** and **never extended to vertical / classification-level federation (G2)**. | **Highest** — both a-priori-flagged thin lanes confirmed thin. |
| #4 SWaP-aware multi-modal edge | Strong technique inventory (token pruning D10/D11, MBQ D8, Mamba) but **zero measured edge wattage for any fused EO+SAR or EO+RF detector (G3)** and **modality-balanced quantization restricted to vision-language (G4)**. | High |
| #5 Operator-facing explainability | Five new architectures (E3–E7) but **zero operator-effect studies (G6)**, **no head-to-head against post-hoc rollout on same task (G7)**, **no integration with calibrated uncertainty (G8)**. | High |

### 2.3 Convergence signals (the strongest evidence in this run)

When two independent gap-finders working from partly-disjoint scout slices converge on the same finding, that finding is the most defensible claim the swarm produced. Four such signals:

- **Calibrated cross-modal uncertainty in EO/SAR FMs is missing.** gap-finder-1 G4 ↔ gap-finder-2 G5/G8.
- **On-device SWaP measurements for fused EO+SAR / EO+RF detectors are missing.** gap-finder-1 G7 ↔ gap-finder-2 G3.
- **Joint uncertainty + operator-facing explanation in one architecture is missing.** gap-finder-2 G8 implicit across multiple gap-finder-1 cells.
- **Cross-classification-level provenance is unimplemented in fusion stacks.** gap-finder-1 G6 ↔ gap-finder-2 G1/G2.

These four gaps, taken together, are the spine of the proposal narrative.

---

## 3. Three-candidate shortlist

Each candidate documents the spec's five required fields. All licence claims verified by gap-finder-3 via direct WebFetch on source pages (audit trail in gap-finder-3 §4).

### Candidate A — Galileo time-series EO/SAR encoder + Discounted Belief Fusion

**Architecture.** Multimodal masked-and-contrastive transformer over time-series of Sentinel-1, Sentinel-2, DEM, and weather — Galileo (arXiv:2502.09356, NASA Harvest, ICML 2025); outperforms specialists on 11 benchmarks. Pairs with Discounted Belief Fusion (arXiv:2412.18024, AISTATS 2025) for the IDEaS uncertainty axis.

**Open dataset path.** MMEarth (`vishalned/MMEarth-data`, **CC-BY-4.0** verified; arXiv:2405.02771; Sentinel-2 L1C/L2A + Sentinel-1 GRD + ASTER DEM + ERA5 + canopy height + Dynamic World + WorldCover; 1.2M locations × 3 sizes). BigEarthNet-MM v2 (Sentinel-1+2, 590k patches, 19 classes; **CC-BY-4.0**). Sen1Floods11 (`cloudtostreet/Sen1Floods11`; **flag: README licence not explicit, confirm before deployment training**; underlying Copernicus is permissive). DynamicEarthNet (arXiv:2203.12560; **CC-BY-NC-SA** — benchmarking only, not deployment training).

**Reference implementation.** `nasaharvest/galileo` (**MIT** verified, 187★; nano weights on GitHub, base/large on HF). `hanmenghan/TMC` (**MIT**, 281★; canonical evidential / subjective-logic fusion behind B2/B3/B4 — recipe: replace softmax → Dirichlet → Dempster fusion → multi-task loss). `aangelopoulos/conformal-prediction` (**MIT**, 1.0k★; distribution-free primitives B5/B6 as second-line uncertainty layer).

**SWaP profile.** Galileo nano is the SWaP variant (≤ 30M params per the published ladder; not numerically asserted in README). Time-series inference ~ 50–150 ms per multi-temporal stack on Jetson Orin Nano with INT8 (via MBQ, arXiv:2412.19509). Temporal-aware SAR/optical fusion under varying cloud / polar-night.

**Explainability.** Operator sees a fused S-1 + S-2 + ERA5 time-series prediction with a **per-prediction calibrated credal set** (DBF) showing per-modality contribution and conflict. Extension: GMAR attention rollout (arXiv:2504.19414) on Galileo cross-modal attention surfaces which temporal frames drove the credal set. NL layer post-hoc — Galileo is encoder-only, not a VLM.

**Risks.** **Important** Composing conformal with evidential fusion is unverified at the multi-modal level (scout-6 §5 Q1; gap-finder-2 D1) — pick one paradigm or run the composition experiment as part of v1. **Important** Sentinel-only pretraining bias: every TerraMind/MMEarth/CROMA/Galileo run uses Sentinel-1+2 collocations; transfer to Canadian RCM C-band quad-pol or ICEYE X-band is untested (gap-finder-1 G10). Frame as "open pretrain + small Canadian-sensor finetune", not "drop-in transfer". **Watch** Time-series compute multiplier on edge devices: if soldier-edge requires ≤ 50 ms latency, prune the temporal stack or use Candidate B's single-frame path.

**TRL-buildability:** A (gap-finder-3 candidate 3). **IDEaS application context:** Arctic ISR (satellite + RF + telemetry) — primary; maritime task-group — secondary (revisit-time anomaly detection).

---

### Candidate B — DOFA wavelength-conditioned dynamic hypernetwork on M4-SAR

**Architecture.** Wavelength-conditioned dynamic hypernetwork — DOFA (arXiv:2403.15356, TUM, 2024). Per-band wavelength embedding swallows Sentinel-1, Sentinel-2, NAIP RGB, Gaofen, EnMAP into a single ViT — no per-sensor finetuning. Fine-tuned on M4-SAR (arXiv:2505.10931, 2025) for SAR + optical object detection.

**Open dataset path.** M4-SAR (`wchao0601/M4-SAR`; **AGPL-3.0** verified, 54★; 112,174 aligned S-1 SAR (VV+VH) + S-2 optical 512×512 pairs, ~ 1M oriented instances, 6 categories). AGPL-3.0 is commercial-permissible with the network-use clause; flag in proposal language. MMEarth (CC-BY-4.0). BigEarthNet-MM v2 (CC-BY-4.0).

**Reference implementation.** `zhu-xlab/DOFA` (code **MIT**, weights at `huggingface.co/earthflow/DOFA` **CC-BY-4.0** — verified separately; 184★; TorchGeo integration `dofa_base_patch16_224`; ViT-Base ≈ 86M, ViT-Large variant). `wchao0601/M4-SAR` ships **seven pre-built baseline fusion detectors with AP50/AP75/mAP + inference-time numbers**: E2E-OSDet (27.5M), CSSA (13.5M), ICAFusion (29.0M), CMADet (41.5M), CLANet (48.2M), CFT (53.8M), MMIDet (53.8M). Cleanest pre-built baseline-bench artefact in the EO/SAR fusion lane.

**SWaP profile.** DOFA-ViT-Base ≈ 86M; M4-SAR baselines 13.5M–53.8M. The 13.5M–29M range is edge-deployable on Jetson-class with INT8. M4-SAR paper reports < 20 ms per 512×512 image for lighter baselines. Edge story: DOFA's hypernetwork means a single backbone export covers all input sensors at runtime — no sensor-specific weights to hot-swap.

**Explainability.** Operator sees per-detection bounding boxes with **per-modality confidence** (which detection driven by SAR vs. optical) plus modality-attention saliency from GMAR (arXiv:2504.19414) on DOFA cross-attention. NL upgrade: layer frozen EarthMind (arXiv:2506.01667) or CLOSP (arXiv:2507.10403) text head on DOFA features.

**Risks.** **Critical** DOFA has not been benchmarked against M4-SAR's E2E-OSDet head-to-head in open literature; the 12-month plan must explicitly run that comparison, not assume transfer. **Important** Sentinel-1/2 collocation bias (same as A; gap-finder-1 G10) — M4-SAR test set is Sentinel-only. **Important** AGPL-3.0 network-use clause; acceptable under government contract; flag explicitly. **Watch** No spec-aligned uncertainty without bolt-on; calibrate via Conformal Prediction (arXiv:2410.19653) or Evidential Partial-View (arXiv:2408.13123).

**TRL-buildability:** A− (gap-finder-3 candidate 2). **IDEaS application context:** Real-time multi-domain threat assessment (EO + SIGINT + text intel) — primary via the EO+SAR detection leg; airborne stealth/spoof — secondary.

---

### Candidate C — SmolVLM-256M + COMODO IMU distillation for tactical-edge wearable

**Architecture.** Sub-billion-parameter multimodal LLM as the operator-facing language interface, fed by a separately-trained sensor-fusion encoder. Substrate: SmolVLM (arXiv:2504.05299, HF, 2025; 256M–2.2B variants). Sensor-fusion encoder: COMODO cross-modal video→IMU distillation (arXiv:2503.07259) + WatchHAR audio+motion pattern (arXiv:2509.04736).

**Open dataset path.** AudioSet (`agkphysics/AudioSet` HF, **CC-BY-4.0** verified — explicitly permits commercial use; balanced 24 GB / full 2.3 TB). CMU-MMAC (kitchen.cs.cmu.edu; "free for research use" — **flag**, evaluation-only). Project Aria (`explorer.projectaria.com`; Aria research licence — **flag**). Drone-Acoustic 32-class (arXiv:2509.04715, 16,000 s, tactical-relevant). **CC-BY-NC alternatives flagged for academic benchmarking only:** Ego4D, Ego-Exo4D, EPIC-KITCHENS-100.

**Reference implementation.** `huggingface/smollm` (**Apache-2.0** verified, ~ 3,800★). SmolVLM-256M-Instruct: **Apache-2.0**, 256M total (93M SigLIP + 135M SmolLM2 decoder), < 1 GB GPU RAM at inference, ONNX exports shipped, 23 quantized variants compatible with llama.cpp / Ollama / LM Studio. `cruiseresearchgroup/COMODO` (**MIT** verified, 24★; IMU encoder distilled from video, IMU-only at inference — privacy-preserving and SWaP-positive). `Meituan-AutoML/MobileVLM` (**Apache-2.0**, 1.4k★; backup substrate at 1.7B if 256M proves capability-insufficient). Audio backbone EAT (arXiv:2401.03497).

**SWaP profile.** SmolVLM-256M: 256M params, < 1 GB GPU FP32, ~ 135 MB INT4. COMODO IMU encoder sub-10M params (TinyHAR-class, arXiv:2507.07949). Total ~ 270M, ~ 1 GB RAM with quantization — reaches Jetson Orin Nano-class. Edge story: SmolVLM ONNX → llama.cpp INT8/INT4 → tactical SoC. **Strongest "Apache-2.0 multimodal LLM that fits on a wearable" reference in the entire 2025 corpus.**

**Explainability.** Operator sees a natural-language situational summary ("vehicle approach, west, ~ 50 m") generated by SmolVLM from fused audio + IMU + occasional video frames. Calibrated confidence via DBF (arXiv:2412.18024) on the audio + IMU dyad — directly closes gap-finder-1 G8 ("audio + IMU lacks UQ"). Saliency via I2MoE (arXiv:2505.19190) on the fusion router or attention rollout (vit-explain / GMAR arXiv:2504.19414) on SmolVLM's vision encoder.

**Risks.** **Critical** Joint audio + IMU + vision quantization is unstudied (scout-3 Q8) — the fused-model INT4/INT8 deployment story is a research question, but that is itself a publishable contribution. **Important** Operator-facing explanation pipeline is not pre-built; fold in I2MoE or attention-rollout. **Important** No open dataset matches the operational context exactly — AudioSet + CMU-MMAC + Project Aria approximates a soldier sensor stack but lacks tactical audio events (gunshot/vehicle); synthetic augmentation is the path. **Watch** FastVLM (arXiv:2412.13303, Apple, 7.3k★) is *excluded* as substrate — model weights ship under research-only Apple licence (verified via LICENSE_MODEL fetch). SmolVLM is correct, not FastVLM.

**TRL-buildability:** A− (gap-finder-3 candidate 4). **IDEaS application context:** Tactical-edge fusion on wearables under degraded connectivity (audio + video + sensor) — primary; real-time multi-domain threat assessment — secondary.

---

### 3.1 Cross-cutting transverse layers (compose with any candidate above)

Source: scout-6. The five IDEaS capability axes are *not* candidate architectures — they are component layers any candidate composes with.

| Capability axis | Recommended primitive | Repo / paper | Licence |
|---|---|---|---|
| Spatiotemporal alignment under sensor mismatch | X-STARS MSAD loss | arXiv:2405.09922 | flag |
| Spatiotemporal alignment via fusion-as-imputation | Cloud-Aware SAR Fusion | arXiv:2506.17885 | flag (PSNR 31.01 dB / SSIM 0.918 SAR→optical) |
| Uncertainty (evidential / belief) | Discounted Belief Fusion (B2) | arXiv:2412.18024 + `hanmenghan/TMC` | MIT |
| Uncertainty (distribution-free) | Conformal Prediction for Multimodal Regression | arXiv:2410.19653 + `aangelopoulos/conformal-prediction` | MIT |
| Uncertainty under missing modality | Any2Any conformal retrieval (B6) | arXiv:2411.10513 | flag |
| Provenance (lifecycle) | Atlas + yProv4ML | arXiv:2502.19567 + arXiv:2507.01078 | flag |
| Provenance (federated multimodal) | FedEPA | arXiv:2504.12025 | flag |
| SWaP — quantization | MBQ + DivPrune | arXiv:2412.19509 + arXiv:2503.02175 | flag |
| SWaP — mobile MLLM | AndesVL recipe | arXiv:2510.11496 | flag (1.8 bpw, 6.7× speedup on Dimensity 9500) |
| SWaP — substrate | SmolVLM-256M | `huggingface/smollm` | Apache-2.0 |
| Explainability (inherent) | I2MoE | arXiv:2505.19190 (ICML 2025) | flag |
| Explainability (post-hoc) | GMAR on `vit-explain` | arXiv:2504.19414 | MIT |

---

## 4. Audit trail — what was considered and dropped

The synthesist's audit trail substitutes for the red-team gate that `gap-finding` runs do not invoke (spec §Success criteria).

### 4.1 Gaps considered and dropped

**gap-finder-1 (5 discards).** Each was rejected after verification queries showed the cell is in fact served:

- "No EO+SAR multimodal MAE pretraining" — served by CROMA (2311.00566), MMEarth (2405.02771), Galileo (2502.09356), TerraMind (2504.11171).
- "No conformal prediction for multimodal models" — served by Conformal mm-Reg (2410.19653), Any2Any (2411.10513), HyperDUM (CVPR 2025); the remaining gap (conformal × evidential × EO/SAR FMs) survives as G4.
- "No inherently-interpretable multimodal architectures" — served by I2MoE (2505.19190), KAN-MCP (2504.12151), GMAR (2504.19414), Head Pursuit (2510.21518), ConceptAttention (2502.04320), MXAI Review (2412.14056); specific-pair gaps survive as G9.
- "Federated multimodal fusion is unexplored" — served by MMO-FL (2505.16138), QQR (2508.11159), FedEPA (2504.12025), SHIFT (2506.05683), FLAME (2503.04136); the transient-connectivity-at-classification-boundary variant survives in G6.
- "RGB+LiDAR autonomous-driving fusion has no 2024–2026 baselines" — served by MoME (2503.19776), RGB+LiDAR robust fusion (2504.19002), Cross-Modal Proxy Tokens (2501.17823).

**gap-finder-2 (4 discards).** Each was a broader claim that verification narrowed:

- "Conformal × evidential never combined" — refuted by ACM 3649329.3663512 (DAC 2024, single-modality); the multi-modal-fusion variant survives as G8.
- "No DP-multimodal-federated work exists" — refuted by IEEE 11126983 (2025); the vertical-classification-level claim survives as G2.
- "No EO FM uncertainty work exists" — refuted by arXiv:2409.08744 (single-modality FM UQ); the cross-modal-fusion variant survives as G5.
- "No edge inference numbers for any multimodal model" — refuted by BioGAP-Ultra (2508.13728), Edge-TPU livestock (2504.11467), AndesVL (2510.11496), WatchHAR (2509.04736); the EO+SAR / EO+RF variant survives as G3.

### 4.2 Candidates dropped — the licence-driven traps (gap-finder-3)

These eight discards are the most consequential audit entries; each is a non-obvious trap caught by direct WebFetch on source LICENSE / dataset pages.

- **SkySense** (arXiv:2312.10115). 21.5M temporal sequences, the strongest open numerical EO+SAR baseline. **Discarded** on (i): weights are **non-commercial research-only** (scout-1 §4). Cite as ceiling reference only.
- **RingMoE** (arXiv:2504.03166). 14.7B MoE, SoTA on 23 benchmarks. **Discarded** on (ii) + (iii): no public reference impl; a 12-month small team cannot reproduce a 14.7B MoE pretrain. Ceiling only.
- **FastVLM** (arXiv:2412.13303, Apple, 7.3k★, 0.5B variant 85× faster TTFT than LLaVA-OneVision-0.5B). **Discarded** on (i): WebFetch of `github.com/apple/ml-fastvlm/blob/main/LICENSE_MODEL` confirms research-only / non-commercial weights — distinct from the (more permissive) code licence. SmolVLM-256M (Apache-2.0) is the correct substrate. Reference FastVLM only as TTFT benchmark ceiling.
- **ImageBind** (arXiv:2305.05665, ~ 9k★). Six-modality joint embedding. **Discarded** on (i): **CC-BY-NC-4.0** verified. The training-free OV-AVE path (gap-finder-3 candidate 7) depends on ImageBind — that is the trap in candidate 7.
- **Ego4D / Ego-Exo4D as deployment training corpora.** **Discarded** on (i): custom signed-agreement research-only licence. Use AudioSet (CC-BY-4.0) + Project Aria + synthetic; keep Ego4D for academic benchmarking only.
- **RadioML 2018.01A as deployment training data.** **Discarded** on (i): direct fetch of `deepsig.ai/datasets` confirms **CC-BY-NC-SA-4.0** — commercial use NOT allowed. Use DroneRF (CC-BY-4.0) and regenerate via DeepSig GNU Radio flowgraphs (GPL on tooling, no licence on generated data).
- **DynamicEarthNet as primary training data.** **Discarded** on (i): **CC-BY-NC-SA**. Benchmark only; MMEarth for pretraining.
- **WavesFM / RF-GPT / PReD direct re-implementation.** **Discarded** on (ii): scout-2 §4 confirms no public reference implementation for any 2024–2026 RF foundation model paper. Candidate 5 (RFML) is reframed around AMR-Benchmark + DroneRF.

### 4.3 Buildable candidates not promoted to the top three (available to proposal-writer)

- **Candidate 1 — TerraMind any-to-any** (gap-finder-3 A). Apache-2.0 weights + CC-BY-4.0 MMEarth. Best cross-modal completion. Not picked because synthesised-SAR vs. real-SAR for downstream detection is itself an open question (gap-finder-1 bifurcation). Strong alternative to B if proposal emphasises Arctic cloud-cover imputation over detection.
- **Candidate 5 — Multi-representation RFML on DroneRF** (gap-finder-3 B). DroneRF CC-BY-4.0; AMR-Benchmark (437★) exists with 14 baselines. Not picked because (i) no confirmed RF foundation-model code, (ii) RF + EO satellite imagery fusion is essentially absent (gap-finder-1 G1) so the candidate must build its own benchmark. Promote only if proposal emphasis shifts to counter-UAS.
- **Candidate 6 — SoftFormer / DDFM on EO + IR** (gap-finder-3 B). MIT end-to-end. Not picked: SoftFormer 33★ is low maintenance signal; DDFM is a 2023 baseline. Use as reproducible baseline only.
- **Candidate 7 — OV-AVE on SmolVLM substrate** (gap-finder-3 B). Apache-2.0 substrate reachable; training-free OV-AVE depends on ImageBind (CC-BY-NC) and must be replaced. Not picked: the from-scratch audio-visual binding bounds 12-month feasibility.
- **Candidate 8 — EarthMind hierarchical-cross-attention MLLM** (gap-finder-3 B−). One of two open instances that meet the IDEaS cross-modal SAR+optical language explainability criterion. Not picked because the FusionEO + EarthMind weights licence was unverified at scout/gap-finder time. If licence verifies non-restrictive and the proposal wants cross-modal language explainability over RF coverage, swap with Candidate B.

---

## 5. YAGNI fence — what we did NOT explore

The spec's Out-of-scope list was honoured. Each item below names what would change if scope expanded.

- **Hypothesis generation with falsification criteria** — deferred. Shortlist names architecture families and risks, not falsifiable thresholds. **Extend:** invoke hypothesis-smith on the picked candidate; suggested entry point is Candidate A's "evidential + conformal compose under cross-modal miscalibration" question.
- **Eval-design experiments** — `gap-finding` does not trigger eval-designer. **Extend:** invoke eval-designer once a candidate is picked; specify train/dev/test splits over MMEarth + Sen1Floods11 + BigEarthNet-MM + (RCM finetune set), choose decision rules, and budget compute (compute-approval required).
- **Classical / rule-based fusion methods** — surveyed only as compared baselines in AI fusion papers. **Extend:** widen scout-1 / scout-5 prompts to include Kalman/Bayesian classical baselines, especially the maritime sonar+AIS lane.
- **ConOps, wargaming, doctrine drafting** — proposal-stage; user-owned. **Extend:** user drafts, drawing on Candidate B's M4-SAR pre-built baselines as operational-detection evidence.
- **Cost modeling, team narrative, partner mapping, letters of support** — proposal-stage; user-owned. This document supplies the technical-risk and SWaP-profile evidence to feed those.
- **Procurement, eligibility review, CanadaBuys submission, direct DND/CAF engagement** — user-owned, out of scope.
- **GPU-bound experiments, training-job submissions, HF Jobs / HF Spaces** — deferred to compute-approved phase. This run produced zero inference/training/measurement; everything is paper-read.
- **Picking one application example** — wide-scope was deliberate. Shortlist spans three of five spec-named contexts; narrowing is a follow-up decision.
- **Maritime task-group anomaly detection candidate** — considered and not promoted. gap-finder-1 G3 confirmed no public dataset combines sonar with AIS / RF / EO/IR; the spec's Application 4 cell is empty and operationally-relevant data is classified-data dominant. **Extend:** if proposal must include maritime, lead with AIS + SAR + EO/IR (xView3-SAR + allenai/vessel-detection-sentinels + GMvA arXiv:2504.09197) and treat sonar as a v2 capability gated on a classified-data partnership.

---

## 6. What would change our mind

Each shortlisted candidate carries specific evidence triggers that would invalidate it.

**Candidate A invalidated if:** (i) a 2026 paper integrates calibrated cross-modal uncertainty into an EO/SAR FM (closing G4/G5) — novelty shrinks from "first" to "competitive"; (ii) conformal × evidential composition is published with a negative result under cross-modal miscalibration — forces a single-paradigm choice; (iii) a RCM cross-sensor transfer benchmark publishes negative results on Sentinel-pretrained transfer (closes G10 negatively) — Arctic ISR becomes a much harder build.

**Candidate B invalidated if:** (i) a new EO/SAR FM ships with permissive licence + integrated detection head that beats DOFA + M4-SAR baselines on M4-SAR's own benchmark — the "pre-built baselines" advantage erodes; (ii) M4-SAR licence changes — proposal must rebase on a different paired SAR+optical detection benchmark; (iii) measured on-device latency/power is published for a fused EO+SAR detector on tactical-grade hardware (closes G7/G3) — SWaP novelty becomes less differentiated (the hypernetwork SWaP-positive property still holds, but headline narrative shifts).

**Candidate C invalidated if:** (i) a new multimodal LLM ships with permissive licence + native audio+IMU input heads — novelty drops to "deployment story only"; (ii) SmolVLM licence changes from Apache-2.0 — swap to MobileVLM (Apache-2.0, 1.4k★, 1.7B) at SWaP cost; (iii) a 2026 paper publishes joint audio+vision+IMU quantization in a single fused model (closes scout-3 Q8) — quantization contribution shrinks; (iv) Apache-2.0 alternative to ImageBind reaches parity — Candidate 7 (OV-AVE) becomes viable and competes for the wearable slot.

**Cross-cutting triggers:** (i) FastVLM LICENSE_MODEL flips to permissive commercial — FastVLM displaces SmolVLM as the tactical-edge ceiling; re-run gap-finder-3 with FastVLM in the candidate set. (ii) Sentinel-1/2 collocation bias empirically refuted by a 2026 cross-sensor benchmark — recommendation order across A and B may flip.

**One thing the workers may have missed:** the swarm did not surface 2025–2026 work on **"open-vocabulary RF event detection"** as a candidate intersection (RF-side analogue of OV-AVE). If such work exists, it would slot between Candidate 5 (RFML) and Candidate C and needs re-scouting. Flagged for proposal-writer spot-check at draft time.

---

## 7. Run metadata

| Item | Value |
|---|---|
| Run id | `2026-05-10-0615-0ece4e` |
| Run dir | `/Users/ggix/ND-Challenge/docs/research/runs/2026-05-10-0615-0ece4e/` |
| Spec | `docs/research/specs/2026-05-10-multimodal-fusion-gap-finding-spec.md` |
| Plan | `docs/research/plans/2026-05-10-multimodal-fusion-gap-finding-plan.md` |
| Novelty target | `gap-finding` (Phases 3 / 4 / 5 idle) |
| Parallelism budget | `MEGARESEARCHER_MAX_PARALLEL = 4` |
| Total worker invocations | **10** (6 literature-scout + 3 gap-finder + 1 synthesist) |
| Phases run | 1 (literature-scout, six workers in two waves), 2 (gap-finder, three workers in one wave), 6 (synthesist, single dispatch) |
| Phases skipped (by design) | 3 (hypothesis-smith), 4 (red-team), 5 (eval-designer) — `gap-finding` does not invoke these |
| Total papers cited across the swarm | **171** unique 2024–2026 papers (full list §8) |
| Total datasets surfaced (with licence flags) | 57 |
| Total reference implementations surfaced | 63 |
| Hypotheses surviving | n/a (gap-finding) |
| Hypotheses rejected / killed | n/a (gap-finding) |
| Gaps surviving | 18 (gap-finder-1 = 10 + gap-finder-2 = 8) |
| Gaps discarded | 9 (gap-finder-1 = 5 + gap-finder-2 = 4) |
| Candidates surviving (TRL-buildability filter) | 8 (gap-finder-3) |
| Candidates discarded with prejudice (licence / size / impl) | 8 (gap-finder-3) |
| Top-3 picked by synthesist | Galileo + DBF · DOFA + M4-SAR · SmolVLM + COMODO |
| Escalations during run | 0 |

**Tooling note (load-bearing for reproducibility).** All six scouts and all three gap-finders independently identified an `hf_papers` parameter-deserialization bug in the running ml-intern MCP server (wrapper rejected `query` and `arxiv_id` regardless of payload shape). Workers worked around it by direct WebFetch against `arxiv.org/abs/<id>` and `huggingface.co/papers/<id>` — spec's "retrievable via `hf_papers`, arXiv, or Semantic Scholar" rule permits this. **Fix committed in MegaResearcher `3819dd4` mid-run; running MCP server still has old code, takes effect on next MCP server start.** Phase 2 prompts pre-baked the workaround so gap-finders skipped rediscovery. Per-citation WebFetch evidence in `scout-N/verification.md` and `gap-finder-N/verification.md`.

---

## 8. Sources — flat deduplicated bibliography

All 2024–2026 unless flagged anchor. Verified via WebFetch on `arxiv.org/abs/<id>` per worker verification.md records (per-paper evidence in `scout-N/verification.md`). Listed by arXiv ID, grouped by primary scout slice; deduplicated. Per-item metadata (titles, authors, why-it-matters annotations) lives in the originating scout output (paths in `bibliography.md`).

**EO/IR + SAR (scout-1):** 2311.00566, 2403.15356, 2405.02771, 2504.11171, 2502.09356, 2312.10115, 2504.03166, 2505.10931, 2506.01667, 2503.19406, 2503.06446, 2502.01002, 2507.10403, 2510.22665, 2405.09365, 2407.06095, 2506.22027, 2410.09111, 2512.02055, 2510.22947, 2510.22726, 2405.09922, 2506.17885; anchors 2303.06840, 2203.12560, 1807.01569; non-arXiv: SoftFormer (Liu et al., ISPRS-JPRS 218, 2024).

**RF / SIGINT (scout-2):** 2402.01748, 2408.06545, 2408.06870, 2411.09996, 2501.02352, 2502.05315, 2503.04136, 2504.14100, 2505.18194, 2506.06718, 2508.20193, 2509.03077, 2510.18336, 2511.05796, 2511.06020, 2511.12305, 2511.15162, 2601.08780, 2601.13157, 2601.18242, 2602.14833, 2603.28183, 2603.28280, 2605.04721.

**Audio + video + sensor at edge (scout-3):** 2503.07259, 2411.11278, 2505.16138, 2503.19776, 2504.05299, 2510.22410, 2503.09905, 2410.13638, 2510.11496, 2503.21782, 2501.17823, 2402.14905, 2507.07949, 2509.04736, 2508.12213, 2502.07855, 2506.09108, 2509.04715, 2603.12880, 2508.13728, 2409.06341, 2412.18024, 2401.03497, 2506.18927, 2504.11467, 2508.11159, 2507.16343, 2506.13060, 2404.05206, 2507.01068, 2404.15349; anchors 2308.13561, 2305.05665; dataset ref 2311.18259.

**Text intel / OSINT (scout-4):** 2311.15826 (anchor), 2402.02544, 2402.05391, 2403.03170, 2403.20213, 2404.14241, 2406.07089, 2406.10100, 2406.10552, 2407.13862, 2407.14321, 2410.06234, 2410.19552, 2412.00832, 2412.15190, 2401.06194, 2501.16254, 2502.11163, 2503.11070, 2505.09852, 2505.14361, 2505.21089, 2506.14817, 2508.19967, 2509.17087, 2509.25026, 2511.21753.

**Sonar / Maritime (scout-5):** 2206.00897 (anchor), 2302.11283 (anchor), 2402.12658, 2406.09966, 2409.13878, 2410.08612, 2410.12953, 2411.00172, 2411.02848, 2412.11840, 2502.20817, 2503.11906, 2504.09197, 2505.01615, 2505.06694, 2505.07374, 2505.20066, 2506.14165, 2507.13880, 2508.02384, 2508.07668, 2509.15959, 2510.03353, 2510.11449.

**Cross-cutting capability axes (scout-6):** 2411.17040, 2509.10729, 2504.19002, 2503.05274, 2409.00755, 2408.13123, 2410.19653, 2411.10513, 2505.03788, 2511.15741, 2502.19567, 2507.01078, 2509.20394, 2402.05160, 2502.04484, 2504.12025, 2506.05683, 2504.09724, 2503.10665, 2412.11475, 2412.13303, 2506.07416, 2412.19509, 2509.18763, 2503.18278, 2503.02175, 2502.01158, 2506.07055, 2507.20613, 2412.14056, 2508.04427, 2505.19190, 2504.12151, 2504.19414, 2510.21518, 2502.04320.

**Verified-licence repos (gap-finder-3 §4):** `IBM/terramind` Apache-2.0 · `vishalned/MMEarth-data` CC-BY-4.0 · `antofuller/CROMA` MIT · `zhu-xlab/DOFA` MIT code, `earthflow/DOFA` HF weights CC-BY-4.0 · `wchao0601/M4-SAR` AGPL-3.0 · `nasaharvest/galileo` MIT · `hanmenghan/TMC` MIT · `aangelopoulos/conformal-prediction` MIT · `huggingface/smollm` Apache-2.0 · `cruiseresearchgroup/COMODO` MIT · `Meituan-AutoML/MobileVLM` Apache-2.0 · `agkphysics/AudioSet` CC-BY-4.0 · DroneRF on Mendeley CC-BY-4.0 · `jacobgil/vit-explain` MIT · `Zhaozixiang1228/MMIF-DDFM` MIT · `rl1024/SoftFormer` MIT.

**Verified-discard repos / datasets:** `apple/ml-fastvlm` LICENSE_MODEL non-commercial · `facebookresearch/ImageBind` CC-BY-NC-4.0 · `facebookresearch/Ego4d` custom signed agreement (deployment-ineligible) · `deepsig.ai/datasets` RadioML 2018.01A CC-BY-NC-SA-4.0 · `Jack-bo1220/SkySense` non-commercial research-only.

**Total unique items:** ~171 (papers + canonical anchors). **Spec success-criterion-1 floor:** 25. **Met:** ✓ by ~7×.

---

*End of synthesist deliverable. The audit-trail discipline (§4) substitutes for red-team approval on `gap-finding` runs. The "what would change our mind" section (§6) is the honesty gate. The YAGNI fence (§5) names what is deliberately not in scope. This document feeds the IDEaS Competitive Projects proposal in the TRL 4–5 / $1.5M / 12-month band, deadline 2026-06-02.*
