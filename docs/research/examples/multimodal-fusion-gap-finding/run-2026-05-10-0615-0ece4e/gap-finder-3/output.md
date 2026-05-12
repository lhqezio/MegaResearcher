# Gap-Finder-3 — TRL-Buildability Filter

**Role:** TRL-buildability filter for the synthesist's three-candidate shortlist
**Run:** `docs/research/runs/2026-05-10-0615-0ece4e/`
**Slice:** scout-1 (EO/IR + SAR), scout-2 (RF/SIGINT), scout-3 (audio + video + sensor at edge), scout-6 (cross-cutting capability axes). Scout-4 (text/OSINT) and scout-5 (sonar/maritime) are de-prioritized per the assignment because (a) text-fusion is heavily dominated by closed-source LLM substrates and (b) open-water sonar is operationally classified-data dominant, making either path a poor TRL-4–5 fit under the spec's no-classified-data constraint.

This document is **not** a list of gaps in the literature — that role is filled by gap-finder-1 and gap-finder-2. This document treats the IDEaS spec's three-candidate shortlist as the artefact, and acts as the *buildability filter* over the consolidated bibliography. Each candidate below is a **named architecture family + reference implementation** that has all three of:

  (i)   Open dataset access with stable identifier and CC-BY-or-more-permissive licence (or a clearly flagged exception).
  (ii)  Open baseline / reference implementation (paper + repo with a verifiable star count or recent commit).
  (iii) Small-team / 12-month / no-classified-data feasibility for TRL 4–5 (parameter count, expected inference cost, edge-deployment story).

For the synthesist: a TRL-buildability score of **A** means all three criteria are met cleanly with permissive licensing; **B** means feasible but with a single load-bearing risk (e.g., one dataset on CC-BY-NC, or one repo with low maintenance signal); **C** means feasible only after substantial engineering or licensing work. The synthesist should pick three of {A, B} candidates, not C.

The IDEaS five capability axes (spatiotemporal alignment, uncertainty propagation, policy-aware provenance, SWaP-aware edge, explainability) are covered by **transverse component layers** (sub-cluster B/C/D/E from scout-6) that any candidate inherits — these are listed at the end so the synthesist can assemble a candidate's full stack.

---

## 1. Ranked candidate list

### Candidate 1 — TerraMind-base (or -tiny) any-to-any EO/SAR foundation backbone, fine-tuned on Sen1Floods11 + MMEarth for cross-modal completion + change detection

- **Architecture family:** Any-to-any generative multimodal foundation model spanning Sentinel-1 (SAR GRD/RTC) + Sentinel-2 (L1C/L2A) + DEM + LULC + RGB. Derives from TerraMind (arXiv:2504.11171, IBM + ESA, ICCV 2025) with the "Thinking-in-Modalities" pretext (synthesise intermediate modalities at inference). Anchored by CROMA (arXiv:2311.00566) as the canonical contrastive radar-optical pretraining baseline.
- **Open dataset path:**
  - **MMEarth** — `vishalned/MMEarth-data` (GitHub) with three sizes (1.2M @ 128px / 1.2M @ 64px / 100k @ 128px). **Licence: CC-BY-4.0** (verified via README.md fetch). Modalities: Sentinel-2, Sentinel-1, ASTER DEM, Dynamic World, ESA WorldCover, ERA5, canopy height. arXiv:2405.02771.
  - **Sen1Floods11** — `cloudtostreet/Sen1Floods11` (GitHub) — paired Sentinel-1 (VV+VH) + Sentinel-2 (13 bands) + flood-water masks. ~14 GB. Used by TerraMind authors as their finetune target. Licence not explicit on the GitHub README — **flag: licence to confirm**, but the Copernicus / Sentinel data underneath is permissive.
  - **BigEarthNet-MM v2** — Sentinel-1 + Sentinel-2 paired tiles, 590k patches, 19 classes. **Licence: CC-BY-4.0** (per scout-1 entry).
- **Open baseline / reference implementation:**
  - `IBM/terramind` — **Apache-2.0** (verified via LICENSE fetch), 258 stars, last commit Nov 3 2025 (HF model cards updated same date). Provides TerraMind 1.0 in tiny / small / base / large variants on `huggingface.co/ibm-esa-geospatial`. Integrates with `TerraTorch`. Pretrained tokenizers per modality (S2L2A, S2L1C, S1GRD, S1RTC, DEM, NDVI) shipped separately — directly relevant to the "synthesise SAR-from-optical" cross-modal completion pattern under cloud cover.
  - Companion: `antofuller/CROMA` (MIT, 45 stars) and `vishalned/MMEarth-train` (license-shown, 63 stars, last commit Feb 3 2025) for self-pretraining sandboxes if frozen-backbone evaluation is preferred.
- **SWaP profile estimate:**
  - TerraMind-tiny is the appropriate edge variant; TerraMind-base for an airborne / shipboard deployment. Parameter counts not numerically asserted on the HF cards but the tiny → base ladder follows ViT-tiny (~5M) to ViT-base (~86M) convention; expect ~5M / ~25M / ~100M / ~300M for tiny/small/base/large.
  - Inference: ViT-base on 128-px tiles is sub-100 ms on a Jetson Orin Nano-class device; tiny variant is sub-50 ms. Fine-tuning a TerraMind-base on Sen1Floods11 fits comfortably on a single 24 GB GPU (within the post-scoping compute envelope, not this run).
  - Edge story: HF Apache-2.0 weights → llama.cpp / ONNX export → INT8 quantisation per scout-6/D8 (arXiv:2412.19509 MBQ, modality-balanced quantization). The Apache-2.0 licence on weights and the CC-BY-4.0 licence on MMEarth means the entire stack is dual-use and re-distributable.
- **TRL-buildability score: A.** Permissive licensing top-to-bottom on the canonical path. The single load-bearing risk is dataset-side: TerraMind-flagged finetunes were trained primarily on Sen1Floods11, where the README licence is not explicit (flagged above). Replacing Sen1Floods11 with MMEarth's flood subset or BigEarthNet-MM removes that risk.
- **Named technical risks:**
  - **Critical** — *Domain-transfer risk to Canadian sensors.* Every TerraMind / MMEarth / CROMA pretraining run uses Sentinel-1/Sentinel-2 collocations; transfer to RADARSAT-Constellation Mission C-band quad-pol or ICEYE X-band is essentially untested in open literature (this is the gap-finder-1 finding). For an IDEaS Arctic ISR scenario the proposal must explicitly frame "open-data pretraining + small Canadian-sensor finetune" as the test, not "drop-in transfer".
  - **Important** — *Generative completion vs. discriminative fusion is bifurcated* (gap-finder-1 finding; also visible in scout-1 §5). TerraMind is the only open candidate that does cross-modal generation, and "synthesised SAR from optical" has not been directly evaluated against "real SAR" for downstream detection in cloudy regions — a clean falsification target for Phase 2.
  - **Watch** — *No calibrated cross-modal uncertainty out-of-the-box.* Layer in Discounted Belief Fusion (B2 / arXiv:2412.18024) or Conformal Prediction for Multimodal Regression (B5 / arXiv:2410.19653) per scout-6.

### Candidate 2 — DOFA wavelength-conditioned dynamic hypernetwork, fine-tuned on M4-SAR for optical-SAR object detection

- **Architecture family:** Wavelength-conditioned dynamic hypernetwork (one backbone, all sensors) — DOFA (arXiv:2403.15356, TUM, 2024). Uses a per-band wavelength embedding to swallow Sentinel-1, Sentinel-2, NAIP RGB, Gaofen, and EnMAP into a single ViT, with no per-sensor finetuning needed.
- **Open dataset path:**
  - **M4-SAR** — `wchao0601/M4-SAR` (GitHub, AGPL-3.0, 54 stars, 61 commits, paper arXiv:2505.10931). 112,174 precisely aligned Sentinel-1 SAR (VV+VH) + Sentinel-2 optical 512×512 image pairs, ~1M oriented instances, 6 categories. **Licence: AGPL-3.0** (verified via repo fetch) — copyleft but commercial-permissible with the AGPL caveat that derivative inference services must be source-released.
  - Backup / pretraining: **MMEarth** (CC-BY-4.0 — same as Candidate 1).
- **Open baseline / reference implementation:**
  - `zhu-xlab/DOFA` — **MIT** (per repo) with weights at `huggingface.co/earthflow/DOFA` and `XShadow/DOFA` (HF model card cites **CC-BY-4.0** for weights — verified via fetch). 184 stars, demo notebook (`demo.ipynb`) and TorchGeo integration (`dofa_base_patch16_224`). DOFA-ViT-Base ≈ 86M parameters (standard ViT-B count; explicit numerics not asserted on HF page). DOFA-ViT-Large variant available.
  - `wchao0601/M4-SAR` ships **E2E-OSDet (27.5M)**, **CSSA (13.5M)**, **ICAFusion (29.0M)**, **CMADet (41.5M)**, **CLANet (48.2M)**, **CFT (53.8M)**, **MMIDet (53.8M)** — seven baseline fusion detectors with reported AP50/AP75/mAP and inference-time numbers. This is the cleanest baseline-bench artefact in the entire EO/SAR fusion lane.
- **SWaP profile estimate:**
  - DOFA-ViT-Base ≈ 86M params; the M4-SAR baselines run from 13.5M (CSSA) to 53.8M (MMIDet). The 13.5M–29M range is comfortably edge-deployable on Jetson-class hardware with INT8 quantisation. Inference on 512×512 patches at <20 ms per image is plausible for the lighter baselines per the M4-SAR paper's reported numbers.
  - Edge story: DOFA's hypernetwork design means a single backbone export covers all input sensors at runtime — a SWaP-positive property because no sensor-specific weights need to be hot-swapped on a constrained device.
- **TRL-buildability score: A−.** All three criteria met. The "−" is for AGPL-3.0 on M4-SAR, which is open but creates a commercial-deployment caveat (network use clause). For an IDEaS Competitive Project this is acceptable — the project itself is under government contract — but proposal language should explicitly address the AGPL-3.0 clause.
- **Named technical risks:**
  - **Critical** — *DOFA was not benchmarked against M4-SAR's E2E-OSDet head-to-head in the open literature.* A 12-month plan must explicitly compare the frozen-DOFA-backbone + lightweight detection head against the M4-SAR-bundled baselines, not assume transfer.
  - **Important** — *Sentinel-1/2 collocation bias* — same gap as Candidate 1. The M4-SAR test set is Sentinel-only; Canadian-sensor transfer remains a gap.
  - **Watch** — *No spec-aligned uncertainty story without bolt-on.* M4-SAR baselines report point estimates; calibrate via Conformal Prediction (arXiv:2410.19653) or Evidential Deep Partial Multi-View (arXiv:2408.13123) per scout-6.

### Candidate 3 — Galileo time-series EO/SAR encoder + Discounted Belief Fusion uncertainty layer for Arctic spatiotemporal alignment

- **Architecture family:** Multimodal masked-and-contrastive transformer over **time-series** of Sentinel-1, Sentinel-2, DEM, and weather — Galileo (arXiv:2502.09356, NASA Harvest, ICML 2025). Outperforms specialist models on 11 benchmarks including crop mapping and flood detection. Pairs naturally with Discounted Belief Fusion (B2 / arXiv:2412.18024, AISTATS 2025) for the uncertainty-propagation IDEaS axis.
- **Open dataset path:**
  - **MMEarth** — Sentinel-1 + Sentinel-2 + ERA5 weather + DEM time-series substrate. **CC-BY-4.0** (confirmed).
  - **Sen1Floods11** for downstream evaluation (flagged-licence as in Candidate 1).
  - **DynamicEarthNet** (arXiv:2203.12560) — daily Planet + monthly S-1 + S-2 over 75 sites — flagged **CC-BY-NC-SA** (non-commercial, scout-1 §3). Use for benchmarking only, not deployment training.
- **Open baseline / reference implementation:**
  - `nasaharvest/galileo` — **MIT** (per repo), 187 stars (verified via fetch). Nano weights on GitHub directly; other sizes on Hugging Face. Pretraining recipe described in arXiv:2502.09356.
  - `hanmenghan/TMC` — **281 stars** (per scout-6), the canonical evidential / subjective-logic multi-view fusion implementation behind B2/B3/B4. Provides the four-step recipe (replace softmax → Dirichlet → Dempster fusion → multi-task loss).
  - `aangelopoulos/conformal-prediction` — **1.0k stars** — distribution-free uncertainty primitives (B5/B6) as the second-line uncertainty layer.
- **SWaP profile estimate:**
  - Galileo nano is the SWaP-relevant variant (parameter count not numerically asserted on the README, but "nano" implies ≤30M); base/large sizes are HF-hosted. Time-series inference is more compute-intensive than single-frame; expect ~50–150 ms per multi-temporal stack on Jetson Orin Nano with INT8.
  - Edge story: Galileo's main edge story is **temporal-aware** SAR/optical fusion — relevant to Arctic ISR where the same scene is revisited under varying cloud / polar-night conditions and the spatiotemporal-alignment capability axis (scout-6 sub-cluster A) is the dominant value-add.
- **TRL-buildability score: A.** All three criteria met. Galileo is the strongest 2025 baseline for the temporal-axis use of EO/SAR fusion, MIT-licensed code, MMEarth-compatible. The Discount-Fusion uncertainty layer is a separate well-tested codebase; integration risk is bounded.
- **Named technical risks:**
  - **Important** — *Composing conformal calibration with evidential fusion is unverified* (gap-finder finding from scout-6 §5: "Does conformal calibration compose with evidential fusion?"). The proposal must pick one paradigm or explicitly run the composition experiment.
  - **Important** — *Sentinel-only pretraining bias* — same as Candidates 1 and 2. Shared cross-cutting risk.
  - **Watch** — *Time-series compute multiplier on edge devices* — if the soldier-edge or shipboard target requires ≤50 ms latency, the temporal stack length must be pruned (or DOFA's single-frame approach used instead).

### Candidate 4 — SmolVLM-256M (Apache-2.0) tactical-edge vision-language operator interface, fed by a multimodal sensor fusion head distilled per COMODO

- **Architecture family:** Sub-billion-parameter multimodal LLM as the operator-facing language interface, with a separately-trained sensor-fusion encoder feeding it. Substrate: SmolVLM (arXiv:2504.05299, HF, 2025) — 256M to 2.2B variants. Sensor-fusion encoder: COMODO cross-modal video-to-IMU distillation (arXiv:2503.07259) + WatchHAR audio+motion fusion pattern (arXiv:2509.04736).
- **Open dataset path:**
  - **AudioSet** — `agkphysics/AudioSet` HF dataset, **CC-BY-4.0** (verified by fetch — explicitly permits commercial use). Balanced split 24 GB; full unbalanced 2.3 TB. Used for any audio-event-detection pretext.
  - **CMU-MMAC** (kitchen.cs.cmu.edu) — synchronized video + audio + RFID + IMU. "Free for research use" — **flag**, treat as evaluation-only, not deployment training data.
  - **Project Aria** — `explorer.projectaria.com` — RGB + SLAM + IMU + audio. Aria research licence — **flag**.
  - **CC-BY-NC alternatives flagged**: Ego4D, EPIC-KITCHENS-100. Use only for non-commercial benchmarking.
- **Open baseline / reference implementation:**
  - `huggingface/smollm` — **Apache-2.0** (verified by fetch), ~3.8k stars. SmolVLM-256M-Instruct variant (verified by fetch on `huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct`): **Apache-2.0**, 256M parameters total (93M vision encoder SigLIP-based + SmolLM2-135M decoder), <1 GB GPU RAM at inference, **ONNX exports already shipped** (vision_encoder.onnx, embed_tokens.onnx, decoder_model_merged.onnx), 23 quantised variants compatible with llama.cpp / Ollama / LM Studio.
  - `cruiseresearchgroup/COMODO` — **MIT** (verified by fetch), 24 stars. Provides the IMU-encoder-from-video knowledge-distillation primitive — IMU runs alone at inference, video is teacher-only.
  - `Meituan-AutoML/MobileVLM` — **Apache-2.0**, 1.4k stars, 21.5 tok/s on Snapdragon 888 CPU and 65.3 tok/s on Jetson Orin GPU at the 1.7B size. Backup substrate if SmolVLM's 256M proves capability-insufficient.
- **SWaP profile estimate:**
  - SmolVLM-256M: 256M params, <1 GB GPU at FP32, ~135 MB at INT4. Reachable on a Jetson Orin Nano-class soldier-companion device.
  - COMODO IMU encoder is sub-10M params (lightweight TinyHAR-class architecture descended from arXiv:2507.07949). Total stack: ~270M params, ~1 GB RAM target with quantisation.
  - Edge story: SmolVLM ONNX export → llama.cpp INT8 / INT4 → run on tactical SoC. The 256M variant is the single strongest "Apache-2.0 multimodal LLM that fits on a wearable" reference in the entire 2025 corpus.
- **TRL-buildability score: A−.** Apache-2.0 substrate, CC-BY-4.0 for the headline pretraining data (AudioSet). The "−" is because the *fusion* of {audio, video, IMU} into SmolVLM's input space is not a published recipe — the SmolVLM input head is image+text only as shipped. The candidate requires a small adapter module bridging COMODO IMU embeddings + audio embeddings (e.g., from EAT, arXiv:2401.03497) into SmolVLM tokens. This is engineering risk, not licensing or compute risk.
- **Named technical risks:**
  - **Critical** — *Audio + IMU joint quantisation is unstudied* (scout-3 Q8 finding). The candidate's INT4/INT8 deployment story for the *fused* model is a research question, not a known recipe.
  - **Important** — *Operator-facing explanation pipeline is not pre-built.* Fold in I2MoE (arXiv:2505.19190, ICML 2025) or attention-rollout (vit-explain, 1.1k stars) per scout-6 sub-cluster E.
  - **Important** — *No open dataset matches the operational context exactly.* AudioSet + CMU-MMAC + Project Aria approximates a soldier-borne sensor stack but does not include tactical audio events (gunshot, vehicle, drone-acoustic — the latter via arXiv:2509.04715 32-class drone-acoustic dataset). Synthetic augmentation is the path here.
  - **Watch** — *FastVLM (arXiv:2412.13303, Apple, 7.3k stars) is excluded as substrate because the model weights ship under a research-only / non-commercial Apple licence* (verified via LICENSE_MODEL fetch). SmolVLM is the correct substrate, not FastVLM.

### Candidate 5 — Multi-representation RFML stack (spectrogram + raw IQ + constellation) with WavesFM ViT backbone for counter-UAS RF + EO

- **Architecture family:** Multi-representation RF fusion (spectrogram + raw IQ + constellation diagram, fused via collaborative attention) atop a 6G-WavesFM-style ViT backbone. Anchored by MCANet (arXiv:2510.18336, 2025) for the multi-representation pattern, 6G-WavesFM (arXiv:2504.14100, 2025) for the spectrogram-as-image backbone with LoRA adapters, and Tao et al. (arXiv:2510.22947) for the RF + radar + EO architectural reference.
- **Open dataset path:**
  - **DroneRF** (Mendeley `data.mendeley.com/datasets/f4c2b4n755/1`) — **CC-BY-4.0** (verified by fetch — permits commercial use). Raw RF captures of drone control + downlink, the canonical open RF-only counter-UAS dataset.
  - **Drone-Acoustic 32-class dataset** (arXiv:2509.04715) — drone audio signatures, 16,000 s. Useful as second modality.
  - **Multimodal-NF** (arXiv:2603.28280, GitHub `Lmyxxn/Multimodal-NF`) — synchronised CSI + RGB + LiDAR + GPS at drone altitude. Licence not asserted in scout-2 — **flag**.
  - **CC-BY-NC alternatives flagged**: RadioML 2018.01A (`CC-BY-NC-SA-4.0`, verified via DeepSig fetch) — **NOT compatible with a commercial / dual-use IDEaS deployment** unless DeepSig commercial licence is obtained. RadioML 2016.10A same restriction. *Use only for academic baselining; regenerate equivalent training data via DeepSig's GNU Radio flowgraphs (GPL on tooling, no licence on generated data) for any deployment training.*
- **Open baseline / reference implementation:**
  - `Richardzhangxx/AMR-Benchmark` — 437 stars, 14 baseline AMR models implemented (CNN1, CNN2, MCNET, IC-AMCNET, ResNet, DenseNet, GRU, LSTM, DAE, MCLDNN, CLDNN, CLDNN2, CGDNet, PET-CGDNN). License not stated on README — **flag**.
  - No public reference implementation is confirmed for WavesFM, RF-GPT, or PReD as of scout-2 time (scout-2 §4) — *this is the load-bearing risk*. Path A: contact authors / wait for release. Path B: re-implement from paper. Path C: stop at the AMR-Benchmark + spectrogram-as-image substrate and skip foundation-model pretraining for v1.
  - `khalidt/Anti-Drone` — vision + RF + acoustic UAV detection toolkit (research-grade, last updated Nov 2025), explicit multi-modal scope.
- **SWaP profile estimate:**
  - Lightweight AMR baselines (PET-CGDNN, MCLDNN) are sub-1M params; ViT-based spectrogram backbones are ~25–86M.
  - Edge: spectrogram + raw IQ inference on Jetson Orin Nano is well-precedented in the AMR community at <20 ms per inference window.
- **TRL-buildability score: B.** Open dataset is CC-BY-4.0 (DroneRF) and AGPL-style permissive code base exists. The "B" rather than "A" reflects (i) the absence of a confirmed open reference implementation for the foundation-model side (scout-2 §5: "Reproducibility of RF foundation models is poor"), and (ii) the canonical RFML benchmark RadioML being CC-BY-NC-SA, forcing a regeneration step before a commercial deployment training run.
- **Named technical risks:**
  - **Critical** — *No confirmed reference implementation for any 2024–2026 RF foundation model paper* (WavesFM, LWM-Spectro, PReD, RF-GPT, MMSense). Either re-implement, wait, or skip. This is a load-bearing engineering risk.
  - **Critical** — *RF + EO satellite-imagery fusion is essentially absent* from open literature (scout-2 §5). For Arctic ISR (the spec's named application context), this candidate cannot draw on prior open work and must build its own benchmark — a 12-month-feasible but substantial deliverable.
  - **Important** — *RadioML licence forces dataset regeneration* before any commercial / dual-use deployment training. This is a known, manageable engineering step; flag in proposal language.
  - **Watch** — *Counter-UAS literature splits along modality lines* (scout-2 §5). End-to-end RF + EO + text-intel fusion is unpublished; the candidate should explicitly scope to RF + EO + acoustic for v1 and treat text-intel as a Phase-3 extension.

### Candidate 6 — Cross-attention SoftFormer (or DDFM) on aligned EO + IR image pairs, with M4-SAR as the SAR extension

- **Architecture family:** Shifted-window transformer + soft-fusion (SoftFormer, ISPRS-JPRS 218, 2024, `rl1024/SoftFormer`, MIT, 33 stars). The cleanest small-footprint cross-attention recipe for SAR + optical fusion. Companion: DDFM (arXiv:2303.06840), the canonical diffusion-based multi-modal fusion (infrared + visible) — `Zhaozixiang1228/MMIF-DDFM` (MIT). For the SAR extension: M4-SAR (above).
- **Open dataset path:**
  - **M4-SAR** — AGPL-3.0 (confirmed). Same licence caveat as Candidate 2.
  - **MMEarth** — CC-BY-4.0.
  - **TNO Image Fusion** / **RoadScene** / **MSRS** — canonical IR + visible fusion datasets used by DDFM and successors. Licences vary by source — **flag for batch verification**.
- **Open baseline / reference implementation:**
  - `rl1024/SoftFormer` — **MIT**, 33 stars, PyTorch.
  - `Zhaozixiang1228/MMIF-DDFM` — **MIT**, the canonical IR + visible diffusion fusion baseline. Reference for the airborne stealth/spoof IDEaS application context where IR + EO is the dominant pair, not SAR.
  - `wchao0601/M4-SAR` — for the SAR-extension head.
- **SWaP profile estimate:**
  - SoftFormer is small (single-author repo, no parameter count published — likely <50M based on shifted-window backbones at this scale). DDFM is mid-size. Both are runnable on a Jetson Orin Nano class device with INT8.
- **TRL-buildability score: B.** All three criteria met but with three small-team risks: (i) SoftFormer's 33-star repo is a low-maintenance signal, (ii) DDFM is from 2023 and is the *canonical baseline* not a frontier model, (iii) the IR + visible fusion lane is more publication-mature than SAR + optical, which is a YAGNI plus for proposal de-risking.
- **Named technical risks:**
  - **Important** — *SoftFormer maintenance signal is weak.* Single-author repo, 33 stars, no recent commit signal extracted. The proposal should treat SoftFormer as a *reproducible baseline*, not as production substrate.
  - **Important** — *DDFM is a 2023 baseline*, not the 2025 frontier. Use it as the floor against which a new candidate is measured, not as the candidate itself.
  - **Watch** — *IR + visible fusion is well-published.* The novelty contribution must come elsewhere (uncertainty layer, deployment story, or operator-facing explainability) — pure architecture novelty here is a hard sell.

### Candidate 7 — Open-Vocabulary Audio-Visual Event Localization on a SmolVLM substrate, with AudioSet as the open pretraining corpus

- **Architecture family:** Open-vocabulary audio-visual event localization — OV-AVE (arXiv:2411.11278, 2024) + DASM (arXiv:2507.16343, 2025) — substrate-detached so the deployment-time backbone can be SmolVLM (Apache-2.0) rather than ImageBind (CC-BY-NC, see "Discarded" below). Companion: SoundingActions (arXiv:2404.05206) for egocentric audio-action alignment.
- **Open dataset path:**
  - **AudioSet** — `agkphysics/AudioSet` HF, **CC-BY-4.0** (verified). Balanced split 24 GB.
  - **OV-AVEBench** — released with arXiv:2411.11278. 24,800 videos, 67 categories. Licence per paper release — **flag**.
  - **Drone-Acoustic 32-class** (arXiv:2509.04715) — tactical-relevant acoustic signatures.
- **Open baseline / reference implementation:**
  - `huggingface/smollm` — Apache-2.0 (above).
  - The OV-AVE method itself relies on ImageBind in the training-free pathway — this is the **trap** in this candidate (see "Discarded candidates" Q2).
  - DASM (arXiv:2507.16343) is the audio-only complement; reference implementation status not confirmed in scout-3.
- **SWaP profile estimate:** SmolVLM-256M backbone (per Candidate 4) plus a lightweight audio encoder (EAT, arXiv:2401.03497, ~15× pretraining speedup). Total ~280M params, sub-1 GB at INT8.
- **TRL-buildability score: B.** Apache-2.0 substrate is reachable; the *training-free* OV-AVE path uses ImageBind (CC-BY-NC) and must be replaced. AudioSet for the audio side is CC-BY-4.0.
- **Named technical risks:**
  - **Critical** — *Reference implementation depends on ImageBind, which is CC-BY-NC* (Q2 in scout-3). The candidate must commit to re-training the audio-visual binding from scratch on Apache-2.0 substrate, OR demonstrate that a CC-BY-NC training artefact can be lawfully replaced by an Apache-2.0 distillate. This is non-trivial and bounds 12-month feasibility.
  - **Important** — *Tactical audio events are not well-covered* by AudioSet alone. Augment with Drone-Acoustic (arXiv:2509.04715) and synthetic gunshot / vehicle datasets.
  - **Watch** — *Open-vocabulary metrics are nascent.* OV-AVEBench is the only open benchmark; it is paper-fresh and may not be a stable target.

### Candidate 8 — EarthMind hierarchical-cross-attention MLLM as the operator-facing language interface for SAR + optical, on FusionEO

- **Architecture family:** Vision-language MLLM with Hierarchical Cross-modal Attention (HCA) over Sentinel-1 + Sentinel-2 — EarthMind (arXiv:2506.01667, Trento + TUM + Berlin, 2025). Bundled FusionEO (30K paired multi-task annotated) and EarthMind-Bench (2,841 expert-annotated).
- **Open dataset path:**
  - **FusionEO + EarthMind-Bench** — released with paper at `github.com/shuyansy/EarthMind`. **Licence flagged in scout-1 §3** as research-only / per-repo — must be verified at synthesist time.
  - **MMEarth** — CC-BY-4.0 — for any open-licence pretraining substrate.
- **Open baseline / reference implementation:**
  - `shuyansy/EarthMind` — referenced in scout-1 §4. Star count and licence not retrieved at scout time — **flag for synthesist-side verification**.
- **SWaP profile estimate:** MLLM-class — likely 2–7B params, edge-borderline. INT4 quantisation per AndesVL (arXiv:2510.11496, 1.8 bits-per-weight) brings 7B → ~1.6 GB which is plausible on a Jetson Orin AGX (not Nano-class).
- **TRL-buildability score: B−.** Mature in the literature, but *the licence and repo health are unverified at scout time* — that is a load-bearing risk. Promoted from "discarded" to "candidate" because the explainability story (operator-facing language outputs from cross-modal SAR+optical) is one of only two open-literature instances that meet the IDEaS explainability criterion.
- **Named technical risks:**
  - **Critical** — *Licence on FusionEO and EarthMind weights unverified at scout time.* If non-commercial, this candidate drops to "discarded" with prejudice.
  - **Important** — *MLLM-class compute footprint pushes the SWaP envelope.* Either accept Jetson-Orin-AGX-class hardware, or invest in aggressive distillation per scout-6 D8/D9/D12.
  - **Watch** — *Domain bias, same as Candidates 1–3.*

---

## 2. Transverse component layers

The five IDEaS capability axes are *not* candidate architectures — they are component layers that any of the eight candidates above can compose with. Source: scout-6.

| Capability axis | Recommended primitive | Repo / paper | Licence | Notes |
|---|---|---|---|---|
| Spatiotemporal alignment under sensor mismatch | X-STARS MSAD loss | arXiv:2405.09922 | per repo — **flag** | For multi-resolution SAR/optical |
| Spatiotemporal alignment via fusion-as-imputation | Cloud-Aware SAR Fusion | arXiv:2506.17885 | per repo — **flag** | PSNR 31.01 dB / SSIM 0.918 SAR→optical reconstruction |
| Uncertainty propagation (evidential / belief) | Discounted Belief Fusion (B2) | arXiv:2412.18024 + `hanmenghan/TMC` (281 stars) | **MIT** (TMC) | Order-invariant; handles modality conflict |
| Uncertainty propagation (distribution-free) | Conformal Prediction for Multimodal Regression | arXiv:2410.19653 + `aangelopoulos/conformal-prediction` (1.0k stars) | **MIT** | Drop-in calibration on top of any backbone |
| Uncertainty under missing modality | Any2Any conformal retrieval (B6) | arXiv:2411.10513 | per repo — **flag** | Matches complete-modality on KITTI |
| Policy-aware provenance (lifecycle) | Atlas (arXiv:2502.19567) + yProv4ML (arXiv:2507.01078) | per repo | **flag** | Atlas leverages trusted hardware; yProv4ML is PROV-JSON-based |
| Policy-aware provenance (federated multimodal) | FedEPA (arXiv:2504.12025) | per repo | **flag** | Privacy-preserving cross-classification fusion |
| SWaP-aware deployment (compression) | MBQ modality-balanced quantisation (arXiv:2412.19509, CVPR 2025) + DivPrune (arXiv:2503.02175, CVPR 2025) | per paper | **flag** | 4.4–11.6% gain over modality-agnostic PTQ |
| SWaP-aware deployment (mobile MLLM) | AndesVL (arXiv:2510.11496) recipe | per paper | **flag** | 6.7× speedup on Dimensity 9500; 1.8 bits-per-weight |
| SWaP-aware deployment (substrate) | SmolVLM-256M | `huggingface/smollm` (3.8k stars) | **Apache-2.0** | <1 GB GPU at FP32; ONNX shipped |
| Operator-facing explainability (inherent) | I2MoE (arXiv:2505.19190, ICML 2025) | per paper | **flag** | Local + global cross-modal interaction interpretability |
| Operator-facing explainability (post-hoc) | GMAR (arXiv:2504.19414) on `vit-explain` (1.1k stars) | **MIT** | Drop-in attention rollout |

Three of the eight candidates above (1, 2, 3) sit cleanly at the EO/IR + SAR axis; one (4) at the wearable audio + video + IMU axis; one (5) at the RF + EO + acoustic axis; one (6) at the IR + visible (with SAR extension) axis; one (7) at the audio + video axis; one (8) at the EO + SAR with language explainability axis. The synthesist's three-candidate shortlist should pick **one** from each of {EO/SAR-foundation, tactical-edge wearable, RF-fusion} to span the IDEaS application contexts the spec names.

---

## 3. Discarded candidates

These were considered seriously and rejected because they fail at least one of the three buildability criteria. Citing each rejection explicitly is required by the gap-finder discipline.

### Discarded — SkySense

- **Why considered:** Strongest open-literature numerical baseline for EO + SAR + temporal fusion (scout-1 §6, arXiv:2312.10115, CVPR 2024). 21.5M temporal sequences, factorised spatiotemporal encoder, multi-granularity contrastive learning.
- **Rejection criterion (i — open-data-licence):** Scout-1 §4 records the SkySense weights as **"non-commercial research-only"**. This is the load-bearing fail. An IDEaS Competitive Project that intends a TRL-4–5 dual-use deliverable cannot ship a SkySense-derived model.
- **Verdict:** Not in shortlist. The ceiling reference for capability comparisons only.

### Discarded — RingMoE

- **Why considered:** 14.7B-parameter MoE foundation model with modality-specific experts (optical + SAR + multispectral); SoTA on 23 benchmarks; demonstrated 1B compressed deployment variant — relevant to SWaP-aware story (scout-1 §4, arXiv:2504.03166).
- **Rejection criterion (ii — open implementation) and (iii — feasibility):** No public reference implementation confirmed at scout-1 time; scout-1 §6 explicitly notes "the strongest numerical baselines (SkySense, RingMoE) carry the most restrictive licences". Even if licensed permissively, a 12-month small-team project cannot reasonably reproduce a 14.7B MoE pretraining run from scratch.
- **Verdict:** Not in shortlist. Cite as ceiling reference only.

### Discarded — FastVLM

- **Why considered:** 7.3k-star Apple repo (the highest star count of any tactical-edge VLM candidate), iOS demo app already shipped, 0.5B variant with 85× faster TTFT than LLaVA-OneVision — apparently the perfect tactical-wearable substrate.
- **Rejection criterion (i — open-data licence on weights):** Verified by direct fetch of `github.com/apple/ml-fastvlm/blob/main/LICENSE_MODEL`: **"This License does not grant any rights for any commercial purpose. ... 'Research Purposes' does not include any commercial exploitation, product development or use in any commercial product or service."** This is a research-only model licence on the *weights*, separate from the (more permissive) code licence.
- **Verdict:** Not in shortlist. SmolVLM-256M (Apache-2.0) is the correct tactical-edge VLM substrate. Reference FastVLM only as a benchmark ceiling for TTFT comparisons.

### Discarded — ImageBind as the joint-embedding substrate

- **Why considered:** Six-modality joint embedding (image, text, audio, depth, thermal, IMU) with ~9k stars on `facebookresearch/ImageBind` — would unify scout-3's audio + video + IMU + thermal stack in a single embedding.
- **Rejection criterion (i — licence):** Verified: **CC-BY-NC 4.0**. Non-commercial. The OV-AVE training-free pathway (Candidate 7) uses ImageBind, which is the *trap* in that candidate. Operational pathway requires re-training or distillation onto an Apache-2.0 / MIT substrate.
- **Verdict:** Discarded as substrate. SmolVLM (Apache-2.0) is the closest open-licence substitute; CLOSP / SARCLIP for SAR-text alignment are licence-flagged but worth verifying separately.

### Discarded — Ego4D / Ego-Exo4D as deployment training corpora

- **Why considered:** The canonical egocentric multimodal datasets, used by nearly every scout-3 paper (COMODO, OV-AVE, SoundingActions). 591 stars on `facebookresearch/Ego4d` (MIT on the *tooling*).
- **Rejection criterion (i — data licence):** **Custom Ego4D Licence Agreement, signed-agreement-required, research-only.** Cannot be openly redistributed; cannot underpin a TRL-4–5 deployment training run that produces commercially-redistributable artefacts.
- **Verdict:** Discarded for deployment training. Use AudioSet (CC-BY-4.0) + Project Aria (Aria research licence — flagged) + synthetic augmentation. Ego4D / Ego-Exo4D remain valid for academic benchmarking and methods-paper validation.

### Discarded — RadioML 2018.01A as the deployment training dataset

- **Why considered:** Canonical RFML benchmark (24 modulation types, ~2M samples), used by Ahmadi 2025 (arXiv:2508.20193), MCANet (arXiv:2510.18336), Jafarigol 2025 (arXiv:2502.05315).
- **Rejection criterion (i — licence):** Verified by direct fetch of `deepsig.ai/datasets`: **CC-BY-NC-SA 4.0**. Non-commercial. *"Commercial use is NOT allowed under this license."*
- **Verdict:** Discarded for deployment training. Use DroneRF (CC-BY-4.0) for the open RF benchmark, regenerate equivalent synthetic IQ via DeepSig's GNU Radio flowgraphs (GPL on tooling) for any training data needed beyond DroneRF, and contact DeepSig for commercial-licence pricing only if the project requires the canonical 24-modulation taxonomy at scale.

### Discarded — DynamicEarthNet as a primary training dataset

- **Why considered:** Daily Planet multispectral + monthly Sentinel-1 + Sentinel-2 over 75 sites — the canonical dense-time-series benchmark for Galileo-class candidates (Candidate 3).
- **Rejection criterion (i — licence):** **CC-BY-NC-SA**. Non-commercial restriction.
- **Verdict:** Discarded for deployment training. Use as benchmark only. MMEarth (CC-BY-4.0) for any time-series pretraining.

### Discarded — Pure WavesFM / RF-GPT / PReD direct re-implementation

- **Why considered:** The 2024–2026 RF foundation model family (WavesFM arXiv:2504.14100, RF-GPT arXiv:2602.14833, PReD arXiv:2603.28183, MMSense arXiv:2511.12305) is the strongest published RF-multimodal substrate.
- **Rejection criterion (ii — open implementation):** Scout-2 §4 explicitly records "**No public reference implementation was confirmed for the 2025–2026 RF-foundation-model papers**". Verification at the WebFetch level confirms — none of these papers' GitHub repositories are public-and-pretrained at the time of this gap-finder run.
- **Verdict:** Not eligible as a *primary* substrate without re-implementing from paper. Candidate 5 reframes around AMR-Benchmark + DroneRF + a re-pretrained ViT spectrogram backbone, with WavesFM-class methods as v2 enhancement once code releases.

---

## 4. Verification summary table

Every candidate's three load-bearing claims are verified below. Verification queries used WebFetch on the actual GitHub / Mendeley / HF pages because the `mcp__ml-intern__hf_papers` tool wrapper is broken (parameter-deserialization bug). See `verification.md` for the full record.

| Candidate | Open-dataset claim | Verified by | Result |
|---|---|---|---|
| 1 (TerraMind) | MMEarth is CC-BY-4.0 | WebFetch on `github.com/vishalned/MMEarth-data` | "The license for the data is CC BY 4.0." Confirmed. |
| 1 (TerraMind) | TerraMind weights are Apache-2.0 | WebFetch on `github.com/IBM/terramind/blob/main/LICENSE` and HF model card `ibm-esa-geospatial/TerraMind-1.0-base` | "Apache License Version 2.0". Confirmed. |
| 2 (DOFA) | M4-SAR is AGPL-3.0 | WebFetch on `github.com/wchao0601/M4-SAR` | "AGPL-3.0 license". Confirmed. |
| 2 (DOFA) | DOFA weights on HF are CC-BY-4.0 | WebFetch on `huggingface.co/earthflow/DOFA` | "CC-BY-4.0 (not MIT)". Confirmed (note: code is MIT, weights are CC-BY-4.0). |
| 3 (Galileo) | `nasaharvest/galileo` is MIT | WebFetch on `github.com/nasaharvest/galileo` | "License: MIT license. Star Count: 187 stars". Confirmed. |
| 4 (SmolVLM) | SmolVLM-256M-Instruct is Apache-2.0 | WebFetch on `huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct` | "Apache 2.0 ✓ Confirmed. 256 Million (0.3B) parameters. <1GB GPU RAM". Confirmed. |
| 4 (SmolVLM) | AudioSet on HF is CC-BY-4.0 | WebFetch on `huggingface.co/datasets/agkphysics/AudioSet`; also `hf_inspect_dataset` | "License: CC-BY-4.0. Commercial use is permitted." Confirmed. |
| 4 (SmolVLM) | COMODO is MIT, IMU-only at inference | WebFetch on `github.com/cruiseresearchgroup/COMODO` | "License: MIT. Star Count: 24 stars". Inference-time IMU-only path inferred from paper, not explicit in README. |
| 5 (RFML) | DroneRF is CC-BY-4.0 | WebFetch on `data.mendeley.com/datasets/f4c2b4n755/1` | "Licensed under CC BY 4.0. Permits commercial use." Confirmed. |
| 5 (RFML) | RadioML is CC-BY-NC-SA-4.0 (excluded) | WebFetch on `deepsig.ai/datasets` | "CC BY-NC-SA 4.0. Commercial use is NOT allowed." Confirmed (drives the dataset to "discarded" for deployment training). |
| 5 (RFML) | AMR-Benchmark exists with 14 baselines | WebFetch on `github.com/Richardzhangxx/AMR-Benchmark` | "437 stars. 14 deep learning models implemented." Confirmed (license unstated — flag). |
| Discard FastVLM | LICENSE_MODEL is research-only | WebFetch on `github.com/apple/ml-fastvlm/blob/main/LICENSE_MODEL` | "This License does not grant any rights for any commercial purpose." Confirmed. |
| Discard ImageBind | License is CC-BY-NC | WebFetch on `github.com/facebookresearch/ImageBind` | "CC-BY-NC 4.0 license". Confirmed. |

---

## 5. Synthesist guidance — recommended top-3 picks for the IDEaS shortlist

This is *advisory*, not binding — the synthesist owns the final pick.

- **Pick A (highest-confidence): Candidate 1 (TerraMind any-to-any) or Candidate 2 (DOFA + M4-SAR).** These are the two cleanest A-rated EO/SAR candidates and cover the spec's named Arctic ISR (Application 1) and airborne stealth/spoof (Application 5) contexts. TerraMind is the better cross-modal completion story (cloud / polar-night); DOFA + M4-SAR is the better detection / object-recognition story.
- **Pick B (tactical-edge wearable): Candidate 4 (SmolVLM + COMODO).** This is the single Apache-2.0 candidate that fits the soldier-borne SWaP envelope and explicitly addresses Application 3 (tactical-edge fusion under degraded connectivity). The audio-IMU joint-quantisation gap (scout-3 Q8) is a research contribution in itself.
- **Pick C (RF/SIGINT-anchored): Candidate 5 (multi-representation RFML on DroneRF + WavesFM-style backbone).** Covers Application 2 (multi-domain threat assessment) via counter-UAS. The "no confirmed RF-foundation-model code" risk is real but bounded — the candidate is buildable from AMR-Benchmark + DroneRF without WavesFM if needed.

**Alternative composition:** if the synthesist values cross-modal language explainability over RF coverage, swap Candidate 5 for Candidate 8 (EarthMind), conditional on the EarthMind licence being verified non-restrictive.

**Do NOT pick** Candidates 6 (SoftFormer/DDFM — repo health concerns) or 7 (OV-AVE — ImageBind dependency) as primary substrates. Use them only as baselines.

---

## 6. Sources cited (arXiv IDs, all verified via direct WebFetch on `arxiv.org/abs/<id>` per scout-{1,2,3,6} verification records)

EO/SAR — 2311.00566, 2403.15356, 2405.02771, 2502.09356, 2504.11171, 2505.10931, 2506.01667, 2506.17885, 2405.09922, 2410.09111.
RF/SIGINT — 2402.01748, 2411.09996, 2502.05315, 2503.04136, 2504.14100, 2509.03077, 2510.18336, 2510.22947, 2511.05796, 2511.06020, 2602.14833, 2603.28183, 2603.28280, 2511.12305.
Audio + video + IMU — 2305.05665, 2308.13561, 2401.03497, 2402.14905, 2404.05206, 2404.15349, 2410.13638, 2411.11278, 2501.17823, 2503.07259, 2503.21782, 2504.05299, 2506.09108, 2507.07949, 2507.16343, 2508.12213, 2509.04715, 2509.04736, 2510.11496.
Capability axes — 2402.05160, 2408.13123, 2409.00755, 2410.19653, 2411.10513, 2411.17040, 2412.11475, 2412.13303, 2412.14056, 2412.18024, 2412.19509, 2502.04484, 2502.19567, 2503.02175, 2503.05274, 2503.10665, 2503.18278, 2503.19776, 2504.09724, 2504.12025, 2504.12151, 2504.19002, 2504.19414, 2505.03788, 2505.16138, 2505.19190, 2506.05683, 2506.07055, 2506.07416, 2506.13060, 2506.18927, 2507.01078, 2507.20613, 2508.04427, 2509.10729, 2509.18763, 2509.20394, 2510.21518, 2511.15741.

Datasets and repos with verified licences — see §4 verification table.
