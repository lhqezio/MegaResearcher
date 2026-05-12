# Gap-Finder-2 — Capability gaps within active modality work

## 1. Slice scope

This worker reads ALL six scout outputs in full, with primary weight on scout-6 (the cross-cutting capability scout). The aim is *capability-axis* gaps — i.e., across the multi-modal fusion work that exists, which of the five IDEaS desired-outcome capabilities are systematically under-addressed?

The five capabilities (per the spec):

1. **Spatiotemporal alignment across heterogeneous streams**
2. **Uncertainty propagation and confidence scoring across modalities**
3. **Policy-aware fusion + AI provenance / lineage tracking across classification levels**
4. **SWaP-aware edge deployment for multi-modal systems**
5. **Operator-facing explainability for fusion outputs**

Files read in full:

- `docs/research/runs/2026-05-10-0615-0ece4e/scout-1/output.md` — EO/IR + SAR fusion
- `docs/research/runs/2026-05-10-0615-0ece4e/scout-2/output.md` — RF/SIGINT + imagery / text
- `docs/research/runs/2026-05-10-0615-0ece4e/scout-3/output.md` — Tactical-edge wearable (audio + video + IMU under SWaP)
- `docs/research/runs/2026-05-10-0615-0ece4e/scout-4/output.md` — Text intel + sensor / imagery
- `docs/research/runs/2026-05-10-0615-0ece4e/scout-5/output.md` — Sonar + maritime multi-modal
- `docs/research/runs/2026-05-10-0615-0ece4e/scout-6/output.md` — Cross-cutting capability scout (primary)

The a-priori thin capabilities (per the assignment) are #3 (policy-aware provenance) and #4 (SWaP-aware edge for multi-modal). Verification queries below confirm both, and surface additional, less-anticipated gaps in capabilities #2 and #5.

## 2. Gaps (ranked by severity vs. the spec's novelty target)

### Gap G1 — Per-modality classification-level lineage in multi-modal fusion provenance

**Capability dimension affected:** #3 (policy-aware fusion / AI provenance / lineage across classification levels).

**Type:** Untested assumption (the provenance frameworks all assume a uniform-classification pipeline) crossed with an unexplored intersection (provenance frameworks × multi-modal fusion × classification-level annotation).

**Statement.** No 2024–2026 paper in the surveyed corpus implements per-modality classification-level lineage in a fused ML pipeline — i.e., the open provenance frameworks treat the pipeline as a single trust domain, never tagging which fused token / feature derived from a UNCLAS source vs. a more restricted source, and no fusion architecture exposes that lineage to a downstream policy enforcer.

**Evidence.**

- Scout-6 §C cites the three open provenance candidates: **Atlas (arXiv:2502.19567)**, **yProv4ML (arXiv:2507.01078)**, and **HASC / "Blueprints of Trust" (arXiv:2509.20394)**. None of the three is benchmarked on a multi-modal pipeline; each treats lineage at the dataset/model-card grain. Scout-6's own §5 open-question 3 records this directly: *"Provenance frameworks (Atlas, yProv4ML, HASC) do not specifically target multimodal fusion pipelines. … whether per-modality lineage … can be expressed in PROV-JSON or HASC is not demonstrated in any 2024–2026 paper I found."*
- Scout-4 §5 open-question 6: *"Provenance is asserted in survey/policy papers (arXiv:2509.17087, arXiv:2402.05391) but not implemented in any open EO-VLM stack reviewed."*
- Scout-1, scout-2, scout-3, scout-5 — none reports any provenance / lineage feature at all in the fusion architectures listed.

**Verification query.** `mcp__ml-intern__web_search "policy-aware AI provenance multi-modal fusion classification level lineage 2025"` and `"PROV-JSON multimodal fusion classification clearance level lineage per-modality"` and `"multi-classification level intelligence fusion AI bell-lapadula clearance information flow"`. Combined results: **17 hits, zero relevant**. Top hits are general AI-governance blog posts (davidvonthenen.com, NetApp, manifestcyber.com), the IDEaS challenge page itself (canada.ca), the Multimodal Alignment & Fusion survey (which has no provenance content), and CISSP study materials for Bell-LaPadula. No paper retrievable that combines (multi-modal fusion) × (per-modality provenance) × (classification-level annotation). Strong support for the gap.

**Why it matters for the spec.** Capability #3 is the most differentiated capability in the IDEaS call (and is one of two flagged a-priori as thin). A candidate architecture that exposes per-modality classification-level lineage in PROV-JSON to a downstream policy enforcer would be a clean novelty target.

---

### Gap G2 — Cross-classification-level fusion in federated / DP multimodal stacks

**Capability dimension affected:** #3 (policy-aware fusion / classification levels).

**Type:** Untested assumption — every federated-multimodal paper assumes horizontal client splits; vertical splits across classification levels are never tested.

**Statement.** All 2024–2026 multimodal federated-learning papers retrieved assume **horizontal client splits** (different clients hold the same modalities for different samples); no paper addresses **vertical / cross-classification-level splits**, where one party holds UNCLAS modalities and another holds higher-classified modalities and the joint inference must respect a Bell-LaPadula-style information-flow constraint.

**Evidence.**

- Scout-6 §C cites **FedEPA (arXiv:2504.12025)** and **SHIFT (arXiv:2506.05683)** — both are horizontal federated multimodal frameworks. Scout-6's §5 open-question 6 states: *"The federated-multimodal papers (C6, C7) assume horizontal client splits; cross-classification-level fusion (e.g., UNCLAS partner ingest + SECRET native streams) is not addressed."*
- The DP-multimodal line (verification surfaced **IEEE 11126983**, "Differential Privacy for Multi-Modal Federated Learning With Modality…", 2025) does add a privacy budget but still assumes horizontal partitioning of clients holding the same set of modalities.

**Verification query.** `mcp__ml-intern__web_search "cross-classification level multi-modal fusion UNCLAS SECRET federated differential privacy"`. **8 hits.** All are horizontal-split federated multimodal works (FedAFD arXiv:2603.04890, the survey at sciencedirect S1566253524003543, IEEE 11126983 DP-multimodal-FL, etc.). None addresses vertical-classification-level partition. Supports the gap.

**Why it matters for the spec.** ISR/C2 in CAF context inherently spans classification boundaries (NATO-partner UNCLAS feeds + Canadian SECRET sensor data). This gap targets exactly the IDEaS "policy-aware" axis and is contiguous with G1 (lineage) on the proposal side.

---

### Gap G3 — No reported edge-device SWaP measurements for any fused EO/IR + SAR (or EO/IR + RF) detector

**Capability dimension affected:** #4 (SWaP-aware multi-modal edge deployment).

**Type:** Missing baseline / untested assumption — every EO+SAR fusion paper claims "edge-friendly" or "real-time" without reporting on-device wattage or latency on a real edge SoC.

**Statement.** No 2024–2026 paper in the surveyed corpus reports inference latency *and* power on a tactical-grade edge SoC (Jetson Orin / Coral / GAP9 / FPGA tactical) for a **fused** EO + SAR or EO + RF detector — every multi-modal fusion paper reports server-GPU FLOPS or generic parameter counts, while every reported edge-SWaP measurement covers a **single-modality** detector.

**Evidence.**

- Scout-1 §5 open-question 7: *"SWaP-aware deployment is mostly aspirational. RingMoE claims a 1B compressed variant; Mamba-based M³amba claims linear scaling; no surveyed paper reports real on-device inference numbers for an EO+SAR fused detector in a power envelope characteristic of an airborne or tactical edge device."* Cited candidates: **RingMoE (arXiv:2504.03166)**, **M³amba (arXiv:2503.06446)**, **DOFA (arXiv:2403.15356)**, **TerraMind (arXiv:2504.11171)** — none reports edge wattage.
- Scout-2 §5 open-question 4: *"SWaP-aware tactical-edge deployment of RF foundation models. WavesFM ([2504.14100](https://arxiv.org/abs/2504.14100)) shares 80% of parameters across heads, but no paper retrievable in scope reports actual edge-device latency / power numbers for RF + image fusion."*
- Scout-5 §5 open-question 6: *"NAS-DETR ([2505.06694](https://arxiv.org/abs/2505.06694)) reports compute-budget-aware sonar ATR; xView3 FPGA work hints at edge SAR; no fusion paper combines edge inference with uncertainty propagation."*
- Counter-examples that confirm the asymmetry: Scout-3 §2H lists **BioGAP-Ultra (arXiv:2508.13728)** and **the multicore Edge-TPU livestock paper (arXiv:2504.11467)** — both report wattage envelopes but on a *consumer-grade or wearable-only* multimodal stack, not on a tactical EO/SAR/RF stack. Scout-6 §D lists **AndesVL (arXiv:2510.11496)**, **FastVLM (arXiv:2412.13303)**, **LiteVLM (arXiv:2506.07416)** — all report edge numbers but only for vision-language pairs, never for EO+SAR or EO+RF.

**Verification query.** `mcp__ml-intern__web_search "SWaP edge deployment multi-modal fusion EO SAR RF latency power benchmark 2025"`, `"edge tactical Jetson Orin SAR EO RF fused inference watt benchmark 2024 2025"`, `"SAR optical fusion edge inference power watts Jetson real-time multimodal 2024"`, `"on-device multimodal fusion EO SAR text wattage power characterization paper 2025"`. Combined results: **~32 hits.** None is a fused EO+SAR or EO+RF model with measured edge wattage. Hits split into (a) single-modality SAR-on-edge (e.g., MDPI 2072-4292/17/13/2168, ACM 3769102.3772713 — both SAR-only YOLO / FPGA), (b) generic Jetson benchmark pages (developer.nvidia.com), and (c) commercial product blurbs (EDGETAK, Curium Labs, Syslogic). The only fused-multimodal-on-edge paper (Stanford CS231N final report) is unpublished coursework. Strong support for the gap.

**Why it matters for the spec.** The IDEeAS spec explicitly calls out SWaP-aware edge deployment as one of the five desired outcomes. A candidate that ships a **measured** EO+SAR or EO+RF SWaP profile — even on a Jetson Orin Nano — would close one of the most concrete and externally checkable gaps in the field.

---

### Gap G4 — Modality-balanced quantization is unstudied outside vision-language

**Capability dimension affected:** #4 (SWaP-aware edge).

**Type:** Domain-transfer gap — the technique exists for VLMs but has never been tested on non-language modalities.

**Statement.** **MBQ (arXiv:2412.19509, CVPR 2025)** demonstrates that re-weighting visual vs. textual tokens during quantization calibration recovers 4.4–11.6% accuracy over modality-agnostic PTQ for VLMs; no 2024–2026 paper in scope tests modality-balanced quantization for non-language modalities (RF spectrograms, SAR, sonar, IMU, telemetry), all of which exhibit similar inter-modality loss-sensitivity heterogeneity.

**Evidence.**

- Scout-6 §D8 cites MBQ explicitly. Scout-6 §5 open-question 4: *"Modality-balanced quantisation (D8) is shown only on VLMs (vision + text). Its applicability to non-language modalities (RF spectrograms, sonar, telemetry) is unstudied."*
- Scout-3's quantization references — **Whisper INT4 (arXiv:2503.09905)**, **AndesVL 1.8 bpw (arXiv:2510.11496)** — all quantize each branch in isolation. Scout-3 §5 open-question 8: *"No published 2024–2026 paper covers joint quantization of audio + vision + IMU encoders in a single fused model — each branch is quantized in isolation in the surveyed literature."*
- Scout-2 lists **WavesFM (arXiv:2504.14100)** and **IQFM (arXiv:2506.06718)** for RF — neither studies modality-balanced quantization across an RF + image stack.

**Verification query.** `mcp__ml-intern__web_search "modality-balanced quantization spectrogram SAR sonar non-language modality 2025"`. **8 hits.** Every hit is the MBQ paper itself or its CVPR mirror / GitHub repo (thu-nics/MBQ) or unrelated SAR-image quantization (mdpi.com 2072-4292/17/3/557 — that one quantizes raw SAR pixel intensities, not the model). Zero hits on extending MBQ-style calibration to non-language modalities. Supports the gap.

**Why it matters for the spec.** A fused EO+SAR+RF stack on a tactical edge device under INT4 PTQ will hit exactly the inter-modality loss-sensitivity asymmetry MBQ documents. Demonstrating modality-balanced quantization on SAR or RF would be a low-risk, narrowly-scoped novelty contribution attached to a SWaP-aware candidate.

---

### Gap G5 — Cross-modal calibrated uncertainty is missing from every EO/SAR foundation model

**Capability dimension affected:** #2 (uncertainty propagation and confidence scoring).

**Type:** Untested assumption — every EO/SAR multi-modal foundation model assumes that downstream tasks can call calibrated confidence "later" without architectural support; none reports calibrated cross-modal uncertainty as a primary metric.

**Statement.** None of the 2024–2026 EO/SAR multi-modal foundation models surveyed (CROMA, DOFA, MMEarth, Galileo, TerraMind, SkySense, RingMoE) reports calibrated cross-modal uncertainty (ECE, Brier, conformal coverage, or evidential credal sets) — and the only EO foundation-model uncertainty study available (arXiv:2409.08744, "Uncertainty and Generalizability in Foundation Models for Earth Observation") evaluates **single-modality** Sentinel-1 *or* Sentinel-2 backbones, not the **fused** dual-modality regime.

**Evidence.**

- Scout-1 §5 open-question 3 (verbatim): *"Uncertainty propagation across modalities is treated as an afterthought. None of the SoTA EO/SAR foundation papers surveyed (CROMA, DOFA, MMEarth, Galileo, TerraMind, SkySense, RingMoE) reports calibrated cross-modal uncertainty."* Citations: **CROMA arXiv:2311.00566**, **DOFA arXiv:2403.15356**, **MMEarth arXiv:2405.02771**, **Galileo arXiv:2502.09356**, **TerraMind arXiv:2504.11171**, **SkySense arXiv:2312.10115**, **RingMoE arXiv:2504.03166**.
- Scout-6 §B lists six new uncertainty primitives (B1–B8) — evidential / discounted-belief / conformal — but **none** has been integrated into an EO/SAR foundation-model fine-tune in the open literature. Scout-6 §5 open-question 8: *"Reference implementations of evidential multimodal fusion in 2024–2025 do not appear consolidated — most link back to the 2021 TMC repo."*
- Verification of arXiv:2409.08744 via WebFetch: confirms the paper studies "eight existing FMs on either Sentinel 1 or Sentinel 2 as input data" — i.e., **single-modality**. Does not address fused / cross-modal calibration.

**Verification query.** `mcp__ml-intern__web_search "remote sensing foundation model uncertainty calibration Sentinel-1 Sentinel-2 confidence 2025"` and `"evidential multimodal fusion remote sensing EO SAR radar 2024 2025 calibration reliability"`. Combined: **16 hits**, dominated by single-modality flood-mapping uncertainty studies, ESA Sentinel-2 radiometric uncertainty tools, and unrelated medical/lidar-hyperspectral fusion. Zero hits on "calibrated uncertainty propagation across SAR + optical fused FM downstream." Supports the gap.

**Why it matters for the spec.** Capability #2 is one of the five IDEaS desired outcomes. An open weights EO+SAR encoder that ships calibrated cross-modal credal sets (or conformal coverage) would be the first of its kind and is reachable to TRL 4 with off-the-shelf primitives (TMC, conformal-prediction notebooks, B1–B6 of scout-6).

---

### Gap G6 — No empirical operator-effect study for fusion explainability in ISR

**Capability dimension affected:** #5 (operator-facing explainability).

**Type:** Untested assumption — the entire MXAI literature for fusion claims operator benefit; none has been measured on ISR or C2 operators.

**Statement.** Across the 2024–2026 corpus, no paper reports a controlled human-subjects study measuring whether any multi-modal fusion explanation method (attention rollout, KAN-style splines, I2MoE local/global interpretations, conceptual-saliency, inherent-interpretability components) actually improves operator decision quality, latency, or calibration on ISR/C2 tasks.

**Evidence.**

- Scout-6 §5 open-question 7 (verbatim): *"Empirical evidence on operator-facing explanations in real ISR/C2 user studies is absent from the surveyed work. The systematic review (E2 = arXiv:2508.04427) explicitly flags the lack of cross-modal evaluation standards; no paper measures whether attention rollouts or KAN/MoE explanations actually improve operator decisions."*
- Cited methods that lack human-subjects validation: **I2MoE arXiv:2505.19190**, **KAN-MCP arXiv:2504.12151**, **GMAR arXiv:2504.19414**, **Head Pursuit arXiv:2510.21518**, **ConceptAttention arXiv:2502.04320**, **MXAI review arXiv:2412.14056**, **PRISMA review arXiv:2508.04427**.
- Scout-3 §5 open-question 5 corroborates: *"Explainability work splits cleanly into 'tabular SHAP on a microcontroller' [31] and 'multimodal cross-modal influence principles' [29] with no paper bridging the two for an operator-facing wearable that produces real-time, on-device, multimodal explanations a soldier can act on."*
- Scout-5 cites **AIS-LLM arXiv:2508.07668** (produces NL rationales for AIS anomalies) but reports no operator study.
- Scout-1 cites **EarthMind arXiv:2506.01667** and **CLOSP arXiv:2507.10403** as the only EO/SAR + language explainability efforts; neither runs an operator study.

**Verification query.** `mcp__ml-intern__web_search "operator user study explainability ISR multi-modal fusion decision quality"`, `"operator decision-making study explainable AI ISR attention rollout user experiment effect size"`, `"defense ISR analyst study explainability situational awareness AI experimental evaluation"`. Combined: **~24 hits.** Top hits are general XAI human-subjects studies in non-defense domains (industrial robotics on freederia.com, eye-tracking recommendation studies on Springer, the ISR-23 INFORMS paper on explainable AI in non-fusion settings, RAND's "Improving Sense-Making with Artificial Intelligence" RRA3152-1 which is a policy paper, not a controlled study), or commercial vendor case studies (i3ca.com ISAAC-ISR). No retrievable paper combines (multi-modal fusion explanation method) × (controlled human-subjects experiment) × (ISR/C2 task). Supports the gap.

**Why it matters for the spec.** The synthesist needs a "what would change our mind" hook for capability #5; the absence of empirical operator-effect data is itself the most decision-relevant evidence here. A candidate that pre-registers an operator study (even on an unclassified analogue task) is differentiated.

---

### Gap G7 — Inherent-interpretability fusion architectures unbenchmarked against post-hoc rollout on the same task

**Capability dimension affected:** #5 (operator-facing explainability).

**Type:** Missing baseline — inherent-interpretability papers (I2MoE, KAN-MCP) and post-hoc rollout papers (GMAR, Head Pursuit, ConceptAttention) report on disjoint task suites.

**Statement.** Inherent-interpretability fusion architectures (**I2MoE arXiv:2505.19190**, **KAN-MCP arXiv:2504.12151**) and post-hoc attention-rollout / saliency methods (**GMAR arXiv:2504.19414**, **Head Pursuit arXiv:2510.21518**, **ConceptAttention arXiv:2502.04320**) have never been benchmarked head-to-head on the same multi-modal fusion task with the same evaluation suite (faithfulness, fidelity, explanation completeness), so the explanation-quality vs. accuracy frontier between the two paradigms is empirically unknown.

**Evidence.**

- Scout-6 §5 open-question 5 (verbatim): *"Inherent-interpretability architectures (I2MoE, KAN-MCP) have not been benchmarked against post-hoc rollout methods on the same fusion task. No paper in scope quantifies the explanation-fidelity vs. accuracy trade-off across the two paradigms."*
- The two inherent-interpretability papers report on sentiment / interaction-pattern benchmarks; the post-hoc rollout papers report on ImageNet classification or medical VQA. No overlapping benchmark surfaces in the 2024–2026 corpus.

**Verification query.** `mcp__ml-intern__web_search "interpretable fusion architecture vs post-hoc attention rollout same task accuracy fidelity 2025"` and `"attention rollout inherent interpretability KAN multimodal fusion benchmark comparison fidelity"`. Combined: **16 hits.** The closest hit is the 2024 ScienceDirect S0004370224001152 ("Assessing fidelity in XAI post-hoc techniques") — which is a *general* post-hoc-fidelity comparison, not a head-to-head with KAN/MoE-style inherent-interpretability fusion. The other relevant hit is the Springer "Inherently Interpretable Machine Learning: A Contrasting Paradigm" (s12599-025-00964-0) — a paradigm review without head-to-head benchmarks. ResearchGate publication 370922025 is an Empirical Comparison of Interpretable Models to Post-Hoc Explanations but in tabular ML, not multimodal fusion. Supports the gap.

**Why it matters for the spec.** The spec's success criteria require an "explainability story" per candidate; choosing inherent vs. post-hoc without head-to-head evidence is exactly the kind of decision a benchmark would crystallise. Mirrors the spec's "what would change our mind" requirement.

---

### Gap G8 — No paper combines multi-modal fusion uncertainty with operator-facing explanation in a single architecture

**Capability dimension affected:** #2 (uncertainty propagation) crossed with #5 (operator-facing explainability).

**Type:** Unexplored intersection — uncertainty primitives X have been built; operator-facing explanation primitives Y have been built; no paper joins X and Y in a single fusion stack.

**Statement.** Across all six scout outputs, no 2024–2026 paper combines a calibrated multi-modal uncertainty primitive (evidential, conformal, discounted-belief) with an operator-facing explanation primitive (rollout, KAN, I2MoE, NL rationale) in a single fusion architecture — every existing fusion stack ships at most one of the two.

**Evidence.**

- Uncertainty side: Scout-6 §B lists eight primitives (**B1: arXiv:2503.05274 evidential trajectory**, **B2: arXiv:2412.18024 discounted-belief**, **B3: arXiv:2409.00755 TUNED**, **B4: arXiv:2408.13123 partial-view evidential**, **B5: arXiv:2410.19653 conformal-multimodal**, **B6: arXiv:2411.10513 Any2Any conformal**, **B7: arXiv:2505.03788 calibrated MLLM**, **B8: arXiv:2511.15741 consistency-as-uncertainty**); none ships an operator-facing explanation alongside.
- Explanation side: Scout-6 §E (E3 I2MoE arXiv:2505.19190, E4 KAN-MCP arXiv:2504.12151, E5 GMAR arXiv:2504.19414, E6 Head Pursuit arXiv:2510.21518, E7 ConceptAttention arXiv:2502.04320), scout-1 EarthMind arXiv:2506.01667, scout-5 AIS-LLM arXiv:2508.07668 — none reports calibrated uncertainty over the explanation outputs.
- Closest near-miss: **GMvA arXiv:2504.09197** (scout-5) wires an "uncertainty fusion module" into AIS+CCTV vessel association but produces no explanation. **AIS-LLM arXiv:2508.07668** produces NL rationales for AIS anomalies but no calibrated uncertainty.

**Verification query.** `mcp__ml-intern__web_search "conformal prediction evidential deep learning combined multimodal fusion uncertainty"` and `"conformal evidential fusion combined multimodal coverage guarantees 2024 2025 arxiv"`. Combined: **16 hits.** The closest hit is **ACM 3649329.3663512** ("Conformal Inference meets Evidential Learning: Distribution-Free…", 2024) — but this is a *single-modality* DAC workshop paper that integrates the two paradigms, not in a multi-modal fusion architecture, and not paired with an operator-facing explanation. Other hits (Dual-level Deep Evidential Fusion sciencedirect S1566253523004293, MDPI 2504-446X/10/2/130 evidential drone object detection) ship one primitive (uncertainty) without the other (explanation). The gap holds: **uncertainty** *combined with* **operator-facing fusion explanation** in one architecture is unbuilt.

**Why it matters for the spec.** Two of the five IDEaS desired outcomes (#2 uncertainty, #5 explainability) are routinely shipped separately. A candidate that ships both — even minimally — would tick two capability boxes simultaneously and would be unique in the corpus.

---

## 3. Discarded candidate gaps

### Discarded D1 — "Conformal calibration on evidential fusion has never been studied"

**Initial framing.** Scout-6 §5 open-question 1 mused: *"Does conformal calibration compose with evidential fusion?"* — I considered claiming this as a clean unexplored intersection.

**Why discarded.** Verification query `"conformal evidential fusion combined multimodal coverage guarantees 2024 2025 arxiv"` surfaced **ACM 3649329.3663512 — "Invited: Conformal Inference meets Evidential Learning: Distribution-Free…"** (DAC 2024). Reading the abstract confirms they **do** combine the two paradigms (in a single-modality setting). The intersection is not unexplored at the *paradigm* level; only the multi-modal-fusion application is unexplored. To avoid double-counting with G8 (which is a strictly broader claim — uncertainty *plus* operator-facing explanation), this candidate is dropped.

### Discarded D2 — "No multi-modal federated learning paper uses differential privacy"

**Initial framing.** Reading scout-6 §C and §5 open-question 6 led me to consider claiming that DP and multimodal-federated had never been combined.

**Why discarded.** Verification query `"cross-classification level multi-modal fusion UNCLAS SECRET federated differential privacy"` surfaced **IEEE 11126983 — "Differential Privacy for Multi-Modal Federated Learning With Modality…"** (2025). DP has been combined with multimodal FL. The remaining gap is the more specific claim that the DP-multimodal-FL line still assumes horizontal partitioning — that narrower claim survives as G2.

### Discarded D3 — "Uncertainty quantification on EO foundation models is unstudied"

**Initial framing.** Scout-1's §5 open-question 3 says no EO/SAR foundation-model paper reports calibrated cross-modal uncertainty; I almost claimed this as "no paper has ever reported uncertainty on EO FMs."

**Why discarded.** Verification surfaced **arXiv:2409.08744 — "Uncertainty and Generalizability in Foundation Models for Earth Observation"** (2024). WebFetch confirms it studies eight FMs on Sentinel-1 *or* Sentinel-2 (single-modality) — so the claim "no EO FM uncertainty work exists" is false at the single-modality grain. The cross-modal-fusion variant of the claim survives as G5; the broader claim is dropped.

### Discarded D4 — "No paper has reported edge inference for any multi-modal fusion model on a real SoC"

**Initial framing.** A reading of scout-3 led me to consider claiming that no paper reports edge wattage for any multi-modal stack at all.

**Why discarded.** Scout-3 already cites concrete counter-examples: **BioGAP-Ultra arXiv:2508.13728** (8.6–32.8 mW envelopes), **multicore Edge-TPU livestock arXiv:2504.11467** (<80ms), **AndesVL arXiv:2510.11496** (Dimensity 9500 6.7×), **WatchHAR arXiv:2509.04736** (sub-12ms smartwatch). These cover wearable / consumer-mobile multi-modal stacks. The narrower claim — that no such measurement exists for the **EO+SAR** or **EO+RF** ISR-relevant modality pairs — is the actual gap and survives as G3.

---

## 4. Capability-coverage summary

| Capability | Surveyed coverage | Severity of gap |
|---|---|---|
| #1 Spatiotemporal alignment | Active and well-resourced (CROMA, DOFA, MMEarth, Galileo, TerraMind, X-STARS, M4-SAR, MultiResSAR, TEOChat) — covered by gap-finder-1/-3, not the principal lane here. | **Low** for this scout. |
| #2 Uncertainty propagation | Six fresh primitives (B1–B8) but **never integrated into EO/SAR FMs** (G5) and **never paired with operator-facing explanation** (G8). | **High**. |
| #3 Policy-aware provenance / classification | Three open frameworks (Atlas, yProv4ML, HASC) — **never benchmarked on multi-modal pipelines** (G1) and **never extended to vertical / classification-level federation** (G2). | **Highest** — both flagged a-priori thin lanes confirmed. |
| #4 SWaP-aware multi-modal edge | Strong technique inventory (token pruning, LoRA, INT4 quant, Mamba) but **zero measured edge wattage for any fused EO+SAR or EO+RF detector** (G3) and **modality-balanced quantization restricted to vision-language** (G4). | **High**. |
| #5 Operator-facing explainability | Five new architectures (E3–E7) but **zero operator-effect studies** (G6), **no head-to-head against post-hoc rollout on same task** (G7), and **no integration with calibrated uncertainty** (G8). | **High**. |

The two a-priori-flagged thin lanes (capabilities #3 and #4) are confirmed thin. Capabilities #2 and #5 are also thin once the lens is "is the primitive integrated with the rest of the fusion stack?" rather than "does the primitive exist standalone?".
