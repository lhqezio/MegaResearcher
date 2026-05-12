# Scout-4 Annotated Bibliography — Text Intel Fused with Sensor or Imagery

## 1. Scope

This sub-topic covers 2024–2026 research on bridging unstructured **text intel / OSINT** with **sensor or imagery streams** (primarily EO/IR satellite imagery, secondarily SAR and event-camera or social-media imagery). It targets the IDEaS application context "real-time multi-domain threat assessment (EO video + SIGINT + text intel)" and the broader gap of report-to-sensor cueing, entity resolution across reports and imagery, and event correlation against ACLED/GDELT-class structured event streams.

Narrowing decisions:

- **Stayed inside text↔imagery / text↔EO**, not pure imagery↔SAR or audio fusion (those are other scouts' lanes).
- **Excluded SIGINT-RF↔text** because there is essentially no public 2024–2026 work pairing radio-frequency signal data directly with text intel; flagged as a thin-prior-art region in the open-questions section.
- **Pre-2024 work cited only when canonical** (e.g., GeoChat 2023-11 because it is the dominant baseline that 2024–2025 EO-VLM papers compare against; xBD 2019 only mentioned as the referenced damage benchmark, not added to the bibliography).
- **No invented citations.** Every arXiv ID below was verified via WebFetch on arxiv.org or huggingface.co/papers — see `verification.md`. The MCP `mcp__ml-intern__hf_papers` tool's `paper_details` and `search` operations errored on every invocation in this run, so verification fell back to direct WebFetch on the arXiv abstract pages, which the discipline rules explicitly permit ("retrievable via hf_papers, arXiv, or Semantic Scholar").

## 2. Key papers

### 2a. Vision-language foundation models for EO / OSINT-grade imagery

These papers establish the modern baseline for text-conditioned reasoning over satellite imagery and are the natural starting point for any text-intel↔EO fusion architecture.

1. **GeoChat: Grounded Large Vision-Language Model for Remote Sensing** — arXiv:2311.15826 (2023, CVPR 2024) — Kuckreja, Danish, Naseer et al. The first grounded VLM for remote sensing: handles high-resolution imagery, region-level conversation, visual grounding, and spatially-anchored object references. Built on LLaVA-1.5; trained on a 318k instruction dataset. *Why it matters:* canonical baseline that every 2024–2026 EO-VLM compares against; sets the architectural template (CLIP encoder + LLaVA-style projector + LLM) most candidate fusion architectures will build on. Older than the 2024–2026 fence but explicitly canonical, so retained per the spec's "older paper as canonical reference" allowance.

2. **TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data** — arXiv:2410.06234 (2024, ICLR 2025) — Irvin, Liu et al. First VLM that ingests *temporal* sequences of EO imagery and reasons about change. Trained on TEOChatlas (554k examples across fMoW, xBD, S2Looking, QFabric). Beats Video-LLaVA, GeoChat, GPT-4o and Gemini 1.5 Pro on temporal EO tasks. *Why it matters:* directly addresses the spatiotemporal-alignment IDEaS dimension and is the only open-weights VLM with native temporal reasoning over satellite data — the prime candidate for a "text query → temporal EO answer" cueing architecture.

3. **EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues** — arXiv:2412.15190 (2024, CVPR 2025) — Soni, Dudhane, Debary et al. Conversational VLM for EO that handles *multi-spectral, multi-temporal, multi-resolution* imagery (RGB, SAR, NIR, infrared) via an 11M instruction dataset; tested on 44 downstream sets. *Why it matters:* the only public 2024 VLM that natively crosses modality boundaries inside the imagery stack (optical↔SAR↔IR), making it a strong base for a fusion architecture that needs to swallow heterogeneous sensor inputs alongside text.

4. **Falcon: A Remote Sensing Vision-Language Foundation Model (Technical Report)** — arXiv:2503.11070 (2025) — Yao, Xu, Yang et al. Unified VLM for 14 remote-sensing tasks (classification, detection, segmentation) at 0.7B params, trained on the 78M-sample Falcon_SFT dataset, evaluated on 67 datasets. *Why it matters:* SWaP-friendly footprint (sub-1B params) makes it a candidate for tactical-edge deployment, which other EO-VLMs in the 7–13B class are not.

5. **VHM: Versatile and Honest Vision Language Model for Remote Sensing Image Analysis** — arXiv:2403.20213 (2024, AAAI 2025) — Pang, Weng, Wu et al. Pairs a captioning dataset with a deliberately-deceptive QA dataset to suppress hallucination on remote-sensing imagery. *Why it matters:* squarely targets the IDEaS "operator-facing explainability + uncertainty" dimension by reducing false affirmations — directly relevant to confidence scoring on EO-derived intelligence claims.

6. **LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model** — arXiv:2402.02544 (2024, ECCV 2024) — Muhtar, Li, Gu et al. Builds LHRS-Align/LHRS-Instruct datasets by pairing OpenStreetMap (Volunteered Geographic Information) attributes with satellite imagery; uses curriculum learning. *Why it matters:* the **most directly text-intel-relevant** EO-VLM in this list — VGI is structurally similar to OSINT text in that it is loosely-curated, geotagged, and globally available. Demonstrates a tractable text-intel↔imagery alignment recipe at internet scale.

7. **SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding** — arXiv:2406.10100 (2024) — Luo, Pang, Zhang et al. Introduces FIT-RS (1.8M instruction samples) with explicit scene-graph and object-relation reasoning tasks; FIT-RSFG / FIT-RSRC benchmarks. *Why it matters:* scene-graph supervision is the cleanest known route to *structured, queryable* outputs an analyst can audit — relevant to provenance and explainability dimensions.

8. **GeoVLM-R1: Reinforcement Fine-Tuning for Improved Remote Sensing Reasoning** — arXiv:2509.25026 (2025) — Fiaz, Debary, Fraccaro et al. Applies RL with task-specific rewards as a post-training step for EO-VLMs across detection, captioning, and change detection. *Why it matters:* the first published recipe for closing the reasoning gap on EO with RL-style post-training rather than more SFT data — relevant if the candidate architecture must handle adversarial or out-of-distribution OSINT-cued queries.

9. **GeoLLaVA: Efficient Fine-Tuned Vision-Language Models for Temporal Change Detection in Remote Sensing** — arXiv:2410.19552 (2024) — Elgendy, Sharshar, Aboeitta et al. Uses LoRA / QLoRA on Video-LLaVA and LLaVA-NeXT-Video to detect landscape change between paired video frames. *Why it matters:* shows that **PEFT-only adaptation** of an off-the-shelf video VLM can hit useful temporal-change scores (BERT 0.864 / ROUGE-1 0.576), which is the cheapest path to a text-intel-cued temporal monitor for a small team.

10. **Vision-Language Modeling Meets Remote Sensing: Models, Datasets and Perspectives** — arXiv:2505.14361 (2025, IEEE GRSM) — Weng, Pang, Xia. Survey of contrastive learning, instruction tuning, and text-conditioned generation for EO. *Why it matters:* the most current published landscape map for the EO-VLM sub-space — useful for the gap-finder worker as a sanity-check counterpart.

### 2b. Knowledge graphs, entity resolution, and cross-modal alignment

11. **Knowledge Graphs Meet Multi-Modal Learning: A Comprehensive Survey** — arXiv:2402.05391 (2024) — Chen, Zhang, Fang et al. 300+ paper survey of KG-driven multi-modal learning, multi-modal KG construction, multi-modal entity alignment, and KG completion. *Why it matters:* the canonical map of the multi-modal-KG sub-field — the discipline closest to the spec's "policy-aware provenance" outcome dimension because KGs make provenance edges explicit.

12. **Multi-Agent Geospatial Copilots for Remote Sensing Workflows** — arXiv:2501.16254 (2025) — Lee, Paramanayakam, Karatzas et al. GeoLLM-Squad framework using AutoGen-style sub-agents for orchestrated EO workflows; +17% agentic correctness over single-agent baselines. *Why it matters:* the cleanest published demonstration of an agent-decomposed pipeline that could front-end a text-intel↔EO fusion stack, with explicit task delegation amenable to provenance auditing.

13. **RS-Agent: Automating Remote Sensing Tasks through Intelligent Agent** — arXiv:2406.07089 (2024) — Xu, Yu, Mu et al. Central-controller LLM + dynamic toolkit + Solution Space + Knowledge Space; >95% planning accuracy across 18 remote-sensing tasks. *Why it matters:* an alternative agent topology to GeoLLM-Squad with explicit knowledge-base hooks — makes it easier to ingest text-intel embeddings as a retrieval knowledge source.

14. **UrbanCross: Enhancing Satellite Image-Text Retrieval with Cross-Domain Adaptation** — arXiv:2404.14241 (2024) — Zhong, Hao, Yan et al. Geo-tagged cross-domain dataset across three countries; SAM-augmented features; +10% retrieval, +15% with domain adaptation. *Why it matters:* the most rigorous 2024 study of **text↔satellite image retrieval** under domain shift — directly solves the report-to-sensor cueing primitive (analyst types description, system retrieves matching imagery).

### 2c. Event-stream / OSINT / conflict-text fusion with imagery

15. **Large Language Model Enhanced Clustering for News Event Detection** — arXiv:2406.10552 (2024) — Tarekegn. LLM-based pipeline (keyword extraction → embedding → summarization → topic labeling) on the GDELT corpus, with a new Cluster Stability Assessment Index. *Why it matters:* the only 2024 paper with a clean recipe for getting from raw GDELT to LLM-clusterable events — the upstream half of any GDELT↔imagery fusion candidate.

16. **Do Large Language Models Know Conflict? Investigating Parametric vs. Non-Parametric Knowledge of LLMs for Conflict Forecasting** — arXiv:2505.09852 (2025) — Nemkova, Lingareddy, Ray Choudhury et al. Compares LLM forecasting from pretrained weights vs. RAG over ACLED/GDELT and recent news, evaluated on 2020–2024 conflict regions. *Why it matters:* the cleanest empirical result that **structured event data + RAG beats parametric knowledge** for conflict prediction — i.e., text-intel fusion is necessary, not optional.

17. **Next-Generation Conflict Forecasting: Unleashing Predictive Patterns through Spatiotemporal Learning** — arXiv:2506.14817 (2025) — von der Maase. Monte-Carlo-Dropout LSTM-U-Net producing 36-month-ahead spatial forecasts of state-based, non-state, and one-sided violence with calibrated uncertainty. *Why it matters:* delivers the **uncertainty-propagation** dimension explicitly via MC dropout — a rare published example of calibrated uncertainty on conflict-event outputs that a multimodal fusion candidate could mimic or extend.

18. **Governing Automated Strategic Intelligence** — arXiv:2509.17087 (2025) — Kruus, Thakur, Khoja et al. Empirical study + taxonomy of multimodal foundation models that synthesize satellite imagery, location, social media, and documents for intelligence questions. *Why it matters:* the most current published treatment of **the exact use case** — open-source multimodal automation of intelligence analysis — including taxonomy of question types and capability determinants. Useful as a framing reference for the synthesist.

19. **Extracting Disaster Impacts and Impact Related Locations in Social Media Posts Using Large Language Models** — arXiv:2511.21753 (2025) — Hameed, Ranathunga, Prasanna et al. Fine-tunes LLMs to pull impact-and-location pairs from disaster tweets (F1 0.69 impact / 0.74 location). Treats social media as "geo-sensors." *Why it matters:* the report-to-sensor cueing primitive in the OSINT direction — converts unstructured social text into geotagged structured records ready for sensor cueing.

20. **CrisisKAN: Knowledge-infused and Explainable Multimodal Attention Network for Crisis Event Classification** — arXiv:2401.06194 (2024) — Gupta, Saini, Kundu. Cross-attention over image+text+Wikipedia knowledge with Grad-CAM explainability on CrisisMMD. *Why it matters:* the most directly ISR-relevant **multi-modal-with-explanation** crisis paper of 2024; touches the explainability dimension head-on.

### 2d. OSINT misinformation / verification (relevant to provenance)

21. **SNIFFER: Multimodal Large Language Model for Explainable Out-of-Context Misinformation Detection** — arXiv:2403.03170 (2024, CVPR 2024) — Qi, Yan, Hsu et al. Two-stage instruction-tuning on InstructBLIP with external knowledge retrieval; +40% over baselines on out-of-context misinformation; produces natural-language explanations. *Why it matters:* the canonical 2024 architecture for **provenance-aware text↔image consistency checking** — the primitive that any OSINT fusion stack needs to filter adversarial inputs.

22. **Multimodal Misinformation Detection using Large Vision-Language Models (LVLM4FV)** — arXiv:2407.14321 (2024) — Tahmasebi, Müller-Budack, Ewerth. Zero-shot LVLM-based fact-verification with re-ranked multimodal evidence retrieval; addresses incomplete ground-truth annotation. *Why it matters:* zero-shot framing means it can drop into a pipeline **without bespoke labeled data** — relevant to the open-data constraint in the spec.

### 2e. Geolocation as a text-intel↔imagery primitive

23. **Enhancing Worldwide Image Geolocation by Ensembling Satellite-Based Ground-Level Attribute Predictors** — arXiv:2407.13862 (2024) — Bianco, Eigen, Gormish. New Recall-vs-Area metric; ensembles GeoEstimation + GeoCLIP + ESA satellite-attribute predictors; particularly improves underrepresented (non-urban) regions. *Why it matters:* the strongest 2024 result on **text-free image-to-location**, which is one of two halves of report-to-sensor cueing (the other is location-to-image).

24. **AI Sees Your Location, But With A Bias Toward The Wealthy World (VLM geolocation evaluation)** — arXiv:2502.11163 (2025) — Huang, Huang, Liu et al. Evaluates 4 VLMs on 1,200 geo-tagged images: 53.8% city accuracy overall, but −12.5% in lower-income regions and −17% in sparsely-populated areas. *Why it matters:* surfaces the geographic-bias risk in any VLM-based geolocation primitive — material for the red-team / synthesist on policy-aware deployment.

25. **Assessing the Geolocation Capabilities, Limitations and Societal Risks of Generative Vision-Language Models** — arXiv:2508.19967 (2025) — Grainge, Waheed, Stilgoe et al. Evaluates 25 VLMs on geolocation across four datasets; 61% accuracy on social-media-style imagery. *Why it matters:* corroborates and extends 2502.11163 with broader model coverage; a useful counterpart citation for any candidate that uses VLM geolocation as a building block.

### 2f. Adjacent (event-camera streams)

26. **EventGPT: Event Stream Understanding with Multimodal Large Language Models** — arXiv:2412.00832 (2024) — Liu, Li, Zhao et al. First multimodal LLM specifically for event-camera (DVS) data; three-stage training (RGB → synthetic events → real events). *Why it matters:* an *adjacent* example of a non-RGB sensor modality being adapted into a VLM via a custom encoder + spatio-temporal aggregator. Pattern is directly transferable to other ISR sensors (SAR, radar) the spec cares about.

## 3. Datasets

All HF dataset names confirmed via WebFetch unless flagged. **Licence flags** marked where stricter than CC-BY.

| Dataset (HF or canonical name) | Description | Licence | Verified |
|---|---|---|---|
| `MBZUAI/GeoChat_Instruct` | 318k remote-sensing instruction-tuning pairs (LRBEN, NWPU, SOTA, SIOR, FAST) | Apache 2.0 | Yes |
| `jirvin16/TEOChatlas` | 554k temporal EO instructions over fMoW + xBD + S2Looking + QFabric | Apache 2.0 | Yes |
| `danielz01/fMoW` | Functional Map of the World — temporal satellite + textual metadata | Not explicitly stated; **gated access (login + agreement required)** — flag | Yes |
| `Junjue-Wang/DisasterM3` (GitHub-hosted with HF paper page 2505.21089) | 26,988 bi-temporal optical+SAR images, 123k instructions, 36 disasters, 5 continents | **CC BY-NC-SA 4.0** — non-commercial — flag | Yes |
| `om-ai-lab/RS5M` (GitHub) | 5M remote-sensing image-text pairs (CC source images, repackaged) | Mixed (per upstream Conceptual Captions / LAION); **review case-by-case** — flag | Yes |
| `CrisisMMD v2.0` | ~16k tweets + images, 7 disasters, 2017 (Alam, Ofli, Imran) | "Terms of use" agreement; **no permissive licence stated; no HF mirror** — flag | Yes |
| GDELT 1.0 / 2.0 | Global news event database (CSV / BigQuery / JSON APIs) | "100% free and open" per project; effectively open — verify per record | Yes |
| ACLED | Curated armed-conflict event dataset, near-real-time | Registration required; **non-permissive academic / research licence** — flag | Yes |
| GeoChat instruction sub-corpora (`LRBEN`, `NWPU_captions`, `SOTA`, `SIOR`, `FAST`) | Component datasets aggregated under GeoChat_Instruct | Apache 2.0 (as released by MBZUAI) | Yes (via parent) |

The MCP `hf_inspect_dataset` tool was attempted but the `hf_papers` family of tools all errored on parameter validation in this run; HF dataset pages were verified directly via WebFetch.

## 4. Reference implementations

GitHub stars and licence verified via WebFetch.

| Repo | Paper | Stars | Licence |
|---|---|---|---|
| `mbzuai-oryx/GeoChat` | arXiv:2311.15826 | 714 | not stated in repo content (paper-side Apache 2.0) |
| `ermongroup/TEOChat` | arXiv:2410.06234 | 141 | Apache 2.0 |
| `opendatalab/VHM` | arXiv:2403.20213 | 116 | Apache 2.0 |
| `Luo-Z13/SkySense-Chat` (a.k.a. SkySenseGPT) | arXiv:2406.10100 | 142 | Apache 2.0 |
| `NJU-LHRS/LHRS-Bot` | arXiv:2402.02544 | 188 | Apache 2.0 |
| `Junjue-Wang/DisasterM3` | arXiv:2505.21089 | 121 | CC BY-NC-SA 4.0 (data); code part Apache-style |
| `MischaQI/Sniffer` | arXiv:2403.03170 | 84 | BSD-3-Clause |
| `mustansarfiaz/GeoVLM-R1-Toolkit` | arXiv:2509.25026 | (referenced; live, star count not extracted) | not extracted |
| `om-ai-lab/RS5M` | arXiv:2306.11300 (RS5M/GeoRSCLIP) | (live, not extracted) | not extracted |
| `chen-yang-liu/Text2Earth` | text-driven RS image generation | (live, not extracted) | not extracted |
| `siruzhong/MM24-UrbanCross` | arXiv:2404.14241 | (live, not extracted) | not extracted |
| `AWCXV/TextFusion` | TextFusion / Information Fusion 2025 | (live, not extracted) | not extracted |

`hf_papers find_models` and `find_datasets` operations errored on parameter validation; repo metadata was verified via WebFetch on github.com pages. The four "(not extracted)" rows resolve to live repos but star/licence detail wasn't pulled in this run — flagged for the synthesist if needed.

## 5. Open questions you noticed

(Flagged for the gap-finder, not proposed as hypotheses.)

1. **No public 2024–2026 paper pairs RF/SIGINT signals with text intel.** Every multimodal "intel" paper in this lane stops at imagery + text (or imagery + imagery + text). The text↔SIGINT crossing appears effectively unaddressed in open arXiv literature.
2. **Knowledge graph + EO-VLM integration is asserted but not benchmarked.** The KG/multi-modal survey (arXiv:2402.05391) reviews general multi-modal KG work; no 2024–2026 paper found that publishes a *benchmark* combining EO imagery, structured text-intel events (ACLED/GDELT), and a queryable KG on a shared evaluation set.
3. **GDELT/ACLED-class events are forecasted from text alone or imagery alone, almost never jointly.** The closest is arXiv:2505.09852 (RAG over ACLED+GDELT+news) — but it does not ingest imagery. arXiv:2506.14817 ingests gridded conflict history but not imagery or text. The text+event+imagery joint-forecasting cell looks empty.
4. **VLM geolocation has known geographic and economic biases (arXiv:2502.11163 / 2508.19967).** No 2024–2026 paper proposes a *bias-aware* report-to-sensor cueing pipeline that accounts for these biases when issuing imagery requests.
5. **Entity resolution across reports and sensor records is largely treated as a pure-text problem.** Multi-modal entity alignment surveys exist, but none of the 2024–2026 papers found explicitly resolve "the entity referred to in this OSINT post" against "the object detected in this satellite frame at coordinates X" with calibrated confidence.
6. **Provenance is asserted in survey/policy papers (arXiv:2509.17087, arXiv:2402.05391) but not implemented in any open EO-VLM stack reviewed.** Architecture-level provenance edges (which token came from which source, classification level) appear absent from the open codebases surveyed.
7. **SWaP for EO-VLMs is dominated by 7B+ models.** Falcon (arXiv:2503.11070) is the rare sub-1B example; GeoLLaVA shows PEFT works, but there is no published, principled SWaP-vs-accuracy frontier study for EO-VLMs at the tactical edge.
8. **Temporal reasoning is monomodal.** TEOChat reasons across satellite frames; GeoLLaVA across video frames; no paper found that reasons across *interleaved* OSINT-text events and EO frames on a shared temporal axis.
9. **Open-data constraint binds tightly on disaster/conflict imagery datasets.** DisasterM3 is CC BY-NC-SA (non-commercial); CrisisMMD has only "terms of use"; ACLED is registration-gated. The IDEaS open-data success criterion will need to navigate this carefully.
10. **Event-stream sensors (DVS, arXiv:2412.00832) have a working VLM recipe, but no analog exists for SAR-VLMs at comparable maturity.** EarthDial touches SAR but as a passenger modality; a SAR-native VLM in the GeoChat mold appears absent.

## 6. Sources

All arXiv IDs:

- 2311.15826 — GeoChat (canonical baseline)
- 2402.02544 — LHRS-Bot
- 2402.05391 — KG meets multi-modal learning (survey)
- 2403.03170 — SNIFFER
- 2403.20213 — VHM
- 2404.14241 — UrbanCross
- 2406.07089 — RS-Agent
- 2406.10100 — SkySenseGPT
- 2406.10552 — LLM-enhanced clustering for GDELT
- 2407.13862 — Worldwide image geolocation ensembling
- 2407.14321 — LVLM4FV multimodal misinformation
- 2410.06234 — TEOChat
- 2410.19552 — GeoLLaVA
- 2412.00832 — EventGPT
- 2412.15190 — EarthDial
- 2401.06194 — CrisisKAN
- 2501.16254 — GeoLLM-Squad / Multi-Agent Geospatial Copilots
- 2502.11163 — VLM geolocation bias study
- 2503.11070 — Falcon
- 2505.09852 — LLMs Know Conflict (parametric vs RAG)
- 2505.14361 — VLM × Remote Sensing survey
- 2505.21089 — DisasterM3
- 2506.14817 — Next-generation conflict forecasting
- 2508.19967 — Geolocation capabilities & societal risks
- 2509.17087 — Governing Automated Strategic Intelligence
- 2509.25026 — GeoVLM-R1
- 2511.21753 — Disaster impact extraction from social media

URLs explicitly used for dataset / repo verification:

- https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct
- https://huggingface.co/datasets/jirvin16/TEOChatlas
- https://huggingface.co/datasets/danielz01/fMoW
- https://github.com/mbzuai-oryx/geochat
- https://github.com/ermongroup/TEOChat
- https://github.com/opendatalab/VHM
- https://github.com/Luo-Z13/SkySense-Chat
- https://github.com/NJU-LHRS/LHRS-Bot
- https://github.com/Junjue-Wang/DisasterM3
- https://github.com/MischaQI/Sniffer
- https://www.gdeltproject.org/data.html
- https://acleddata.com/data/
- https://crisisnlp.qcri.org/crisismmd
