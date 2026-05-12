# Scout-6 — Cross-cutting capability axes (capability-axis scout)

## 1. Scope

Methods for the five IDEaS desired-outcome capabilities, surveyed *across* modalities (a horizontal scan, not a per-modality vertical). Coverage is restricted to 2024–2026 work and is organised into five sub-clusters: spatiotemporal alignment, uncertainty propagation and confidence scoring, policy-aware fusion / AI provenance, SWaP-aware multimodal edge deployment, and operator-facing explainability.

Narrowing decisions:
- Excluded purely single-modality compression / interpretability / uncertainty work — only kept methods that either explicitly target multi-modal architectures or provide a primitive (e.g., attention rollout) widely re-used in multi-modal pipelines.
- For provenance, I include both the "ML lifecycle" provenance frameworks (Atlas, yProv4ML, AI System Cards) and the federated/DP-multimodal sub-thread, because the IDEaS "policy-aware provenance across classification levels" capability spans both axis-tracking and privacy-preserving lineage.
- For SWaP-aware, I deliberately scoped to multimodal/VLM-specific compression, not generic LLM quantization papers.
- "Operator-facing explainability" interpreted broadly to include both inherently-interpretable architectures and post-hoc primitives (rollout/grad-rollout) usable on multi-modal transformers.

## 2. Key papers

### Sub-cluster A — Spatiotemporal alignment across heterogeneous streams

**A1. Multimodal Alignment and Fusion: A Survey** — arXiv:2411.17040 — 2024 — Songtao Li et al.
Comprehensive survey of multi-modal alignment and fusion, organised by structural and methodological dimensions and explicitly addressing cross-modal misalignment, asynchrony, and computational efficiency. Accepted to IJCV 2025. Useful as a canonical taxonomy reference for the capability axis even though it is itself multi-modality-spanning.

**A2. Cross-sensor self-supervised training and alignment for remote sensing (X-STARS)** — arXiv:2405.09922 — 2024 — Valerio Marsocci et al.
Introduces X-STARS, a self-supervised framework with a Multi-Sensor Alignment Dense (MSAD) loss that aligns representations across remote-sensing sensors with very different resolutions; supports continual pre-training when adapting a low-resolution-pretrained model to a new sensor. Directly addresses the "varying spatial resolution" branch of the alignment capability — relevant to the EO/IR + SAR pairing in the IDEaS spec.

**A3. M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR Object Detection** — arXiv:2505.10931 — 2025 — Chao Wang et al.
Releases a benchmark of 112k aligned optical-SAR pairs and proposes an Interleaved Input Rearrangement (IIR) mechanism that positions optical and SAR features adjacently along a temporal axis to enable parameter sharing under non-simultaneous acquisition. Concrete instance of "modality-agnostic alignment" mechanics applied to a SAR+EO pair the spec explicitly cares about.

**A4. Cloud-Aware SAR Fusion for Enhanced Optical Sensing in Space Missions** — arXiv:2506.17885 — 2025 — Trong-An Bui et al.
Attention-driven feature fusion mechanism that combines SAR and optical streams to reconstruct cloud-free optical imagery (PSNR 31.01 dB / SSIM 0.918). Demonstrates a fusion-as-imputation framing where alignment robustness substitutes for a missing-modality assumption — relevant to the Arctic ISR application context.

**A5. Using LLMs for Late Multimodal Sensor Fusion for Activity Recognition** — arXiv:2509.10729 — 2025 — Ilker Demirel et al.
Uses an LLM as a late-fusion arbiter over modality-specific predictions (audio + motion), achieving zero/one-shot performance on Ego4D without joint training or alignment. Notable for the "alignment by abstention" insight: avoids tight temporal alignment by deferring to a high-level reasoner — an interesting capability-axis pattern.

**A6. Deep Learning-Based Multi-Modal Fusion for Robust Robot Perception and Navigation** — arXiv:2504.19002 — 2025 — Delun Lai et al.
Lightweight RGB + LiDAR fusion architecture with adaptive fusion and time-series modelling, evaluated on KITTI (3.5% / 2.2% accuracy gains, real-time). Useful as a small-team baseline for spatiotemporally-aligned EO + telemetry fusion in tactical scenarios.

### Sub-cluster B — Uncertainty propagation and confidence scoring

**B1. Evidential Uncertainty Estimation for Multi-Modal Trajectory Prediction** — arXiv:2503.05274 — 2025 — Sajad Marvi et al.
Applies evidential deep learning to multi-modal trajectory prediction so that aleatoric and epistemic uncertainty are propagated end-to-end across heterogeneous predictions. Concrete instantiation of evidential fusion for a sequence-prediction setting structurally similar to ISR track fusion.

**B2. Multimodal Learning with Uncertainty Quantification based on Discounted Belief Fusion** — arXiv:2412.18024 — 2024 — Grigor Bezirganyan et al.
Order-invariant evidence fusion with a discounting mechanism that detects unreliable modalities; outperforms prior art at distinguishing conflicting vs. harmonious sources. AISTATS 2025. Closest to a drop-in fusion primitive for a multi-source CONOPS where modality reliability varies (e.g., SIGINT degraded in jamming).

**B3. Trusted Unified Feature-Neighborhood Dynamics for Multi-View Classification** — arXiv:2409.00755 — 2024 — Haojian Huang et al.
TUNED model integrating local and global feature-neighborhood structure with evidential deep learning + a selective Markov random field for cross-view uncertainty. AAAI 2025. Demonstrates how subjective-logic fusion scales to many views — directly applicable to fusing >2 ISR sources.

**B4. Evidential Deep Partial Multi-View Classification With Discount Fusion** — arXiv:2408.13123 — 2024 — Haojian Huang et al.
Targets *incomplete* multi-view data (a recurring ISR condition) using K-means imputation plus a Conflict-Aware Evidential Fusion Network. Useful for the missing-modality story under degraded connectivity.

**B5. Conformal Prediction for Multimodal Regression** — arXiv:2410.19653 — 2024 — Alexis Bose et al.
Extends distribution-free conformal prediction to multimodal regressors by harvesting internal multimodal-fused features for non-conformity scoring. First clean recipe for adding calibrated prediction intervals on top of an existing fusion backbone.

**B6. Any2Any: Incomplete Multimodal Retrieval with Conformal Prediction** — arXiv:2411.10513 — 2024 — Po-han Li et al.
Reframes incomplete-modality retrieval as binary classification with conformal calibration on cross-modal encoders; matches complete-modality baselines on KITTI. Demonstrates conformal calibration acting as the "missing-modality handler."

**B7. Calibrating Uncertainty Quantification of Multi-Modal LLMs using Grounding** — arXiv:2505.03788 — 2025 — Trilok Padhi et al.
Combines self-consistency with grounding-model temperature scaling to calibrate MLLM confidence on visual QA / medical tasks. Brings post-hoc calibration to MLLM-style fusion, relevant where text-intel + EO are fused via an MLLM.

**B8. Uncertainty-Resilient Multimodal Learning via Consistency-Guided Cross-Modal Transfer** — arXiv:2511.15741 — 2025 — Hyo-Jeong Jang.
Cross-modal semantic-consistency objective that down-weights unreliable modalities; demonstrated on affect recognition and BCI tasks. The consistency-as-uncertainty heuristic is interesting because it does not need explicit per-modality variance heads.

### Sub-cluster C — Policy-aware fusion, AI provenance, and lineage tracking

**C1. Atlas: A Framework for ML Lifecycle Provenance & Transparency** — arXiv:2502.19567 — 2025 — Marcin Spoczynski et al.
End-to-end "fully attestable ML pipelines" combining trusted hardware with an open provenance specification; supports verifiable model authenticity while protecting data confidentiality. Direct precursor for an architecture with audit-trails baked in across classification levels.

**C2. yProv4ML: Effortless Provenance Tracking for Machine Learning Systems** — arXiv:2507.01078 — 2025 — Gabriele Padovani et al.
Captures provenance in PROV-JSON with minimal code modification, including hyperparameters, dataset metadata, and energy-efficiency metrics. Practical lightweight option for adding lineage to a small-team training pipeline.

**C3. Blueprints of Trust: AI System Cards for End-to-End Transparency and Governance (HASC)** — arXiv:2509.20394 — 2025 — Huzaifa Sidhpurwala et al.
Hazard-Aware System Card augments standard system cards with safety records, AI Safety Hazard IDs, and pipeline-as-code generation; complements ISO standards for accountability. Maps cleanly onto a CAF/DND-style policy-aware reporting requirement.

**C4. What's documented in AI? Systematic Analysis of 32K AI Model Cards** — arXiv:2402.05160 — 2024 — Weixin Liang et al.
Empirical study of 32k Hugging Face model cards revealing systematic gaps in environmental-impact, limitations, and provenance fields. Useful as a baseline showing why manual model-card discipline is insufficient at scale (motivates Atlas/yProv4ML).

**C5. An Empirical Analysis of Machine Learning Model and Dataset Documentation, Supply Chain, and Licensing Challenges on Hugging Face** — arXiv:2502.04484 — 2025 — Trevor Stalnaker et al.
Examines 760k models and 175k datasets, surfacing licensing inconsistencies and supply-chain opacity in the open-model ecosystem. Direct evidence that a policy-aware fusion stack cannot blindly trust upstream HF artefacts — relevant to the spec's classification-aware requirement.

**C6. FedEPA: Enhancing Personalization and Modality Alignment in Multimodal Federated Learning** — arXiv:2504.12025 — 2025 — Yu Zhang et al.
Federated multimodal framework with personalized aggregation and unsupervised modality alignment for label-scarce regimes. Federated learning is the canonical "no-data-leaves-the-classification-domain" pattern; this is a recent multimodal instantiation.

**C7. Multi-Modal Multi-Task Federated Foundation Models for Next-Generation Extended Reality Systems** — arXiv:2506.05683 — 2025 — Fardis Nadimi et al.
SHIFT framework for privacy-preserving distributed multi-modal foundation models in AR/VR/MR with sensor-diversity / hardware-constraint / interactivity factors. Adjacent capability-axis evidence: federated + multimodal at the foundation-model scale.

### Sub-cluster D — SWaP-aware multimodal edge deployment

**D1. Vision-Language Models for Edge Networks: A Comprehensive Survey** — arXiv:2502.07855 — 2025 — Ahmed Sharshar et al.
IEEE IoTJ-accepted survey of VLM-on-edge techniques (pruning, quantisation, distillation, federated learning) with deployment cases in healthcare and environmental monitoring. Canonical literature anchor for the SWaP-aware sub-cluster.

**D2. A Survey on Efficient Vision-Language Models** — arXiv:2504.09724 — 2025 — Gaurav Shinde et al.
Complementary survey focused on compact VLM architectures, frameworks, and performance–memory trade-offs. Useful for technique-level decomposition.

**D3. Small Vision-Language Models: A Survey on Compact Architectures and Techniques** — arXiv:2503.10665 — 2025 — Nitesh Patnaik et al.
Specific to small (sVLM) family — transformer-based and Mamba-based designs, distillation, lightweight attention. Tactical-edge wearable scenario maps onto sVLMs almost directly.

**D4. OmniVLM: A Token-Compressed, Sub-Billion-Parameter Vision-Language Model** — arXiv:2412.11475 — 2024 — Wei Chen et al.
968M-parameter VLM with a token-compression mechanism (729 → 81 visual tokens). Represents the "sub-1B parameter, edge-runnable" frontier circa late 2024.

**D5. AndesVL Technical Report: An Efficient Mobile-side Multimodal Large Language Model** — arXiv:2510.11496 — 2025 — Zhiwei Jin et al.
0.6B–4B mobile MLLM suite with detailed quantisation / LoRA / speculative decoding stack achieving 6.7× speedup on phone hardware. Most concrete recent recipe for a mobile-grade multimodal stack with full deployment notes.

**D6. FastVLM: Efficient Vision Encoding for Vision Language Models** — arXiv:2412.13303 — 2024 — Pavan Kumar Anasosalu Vasu et al. (Apple)
Hybrid FastViTHD encoder; 3.2× TTFT improvement at parity, 85× over LLaVA-OneVision-0.5B for the smallest variant. CVPR 2025. Open-source weights + iOS demo make it credible as an edge baseline.

**D7. LiteVLM: A Low-Latency Vision-Language Model Inference Pipeline for Resource-Constrained Environments** — arXiv:2506.07416 — 2025 — Jin Huang et al.
Patch selection + token reduction + speculative decoding; 2.5× end-to-end latency reduction on NVIDIA embedded hardware. Targets robotics / autonomous-driving edge — closest existing analogue to a tactical wearable.

**D8. MBQ: Modality-Balanced Quantization for Large Vision-Language Models** — arXiv:2412.19509 — 2024 — Shiyao Li et al. (CVPR 2025)
Recognises that visual and textual tokens have different loss sensitivities and re-weights them during quantisation calibration; up to 4.4–11.6% accuracy gains over modality-agnostic PTQ. The "modality-imbalance" framing is itself the capability-axis insight.

**D9. Bi-VLM: Pushing Ultra-Low Precision Post-Training Quantization Boundaries in Vision-Language Models** — arXiv:2509.18763 — 2025 — Xijun Wang et al.
Sub-2-bit PTQ for VLMs using Gaussian-quantile partitioning + saliency-aware quantisation; 3–47% gains over prior art on VQA. Useful upper bound on how aggressively a multimodal model can be compressed.

**D10. TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal VLM** — arXiv:2503.18278 — 2025 — Cheng Yang et al. (CVPR 2025)
Formulates visual-token pruning as an optimisation problem (feature similarity + spatial distance) compatible with FlashAttention; no retraining needed. A drop-in efficiency primitive.

**D11. DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models** — arXiv:2503.02175 — 2025 — Saeed Ranjbar Alvar et al. (CVPR 2025)
Casts token pruning as a Max-Min Diversity Problem; SOTA on 16 image and video-language datasets without fine-tuning. Complements TopV; the two together represent the leading 2025 visual-token-pruning recipes.

**D12. MIND: Modality-Informed Knowledge Distillation Framework for Multimodal Clinical Prediction Tasks** — arXiv:2502.01158 — 2025 — Alejandro Guerra-Manzanares et al. (TMLR 2025)
Distils multiple pre-trained unimodal teachers into a smaller multimodal student; demonstrated across clinical tasks combining time series and imaging. Recipe transfers cleanly to ISR settings where each modality has a strong unimodal teacher.

**D13. A Layered Self-Supervised Knowledge Distillation Framework for Efficient Multimodal Learning on the Edge (LSSKD)** — arXiv:2506.07055 — 2025 — Tarique Dahri et al.
Auxiliary classifiers on intermediate features generate self-supervised signal — no pre-trained teacher required. Especially attractive when no foundation-model teacher exists for an exotic modality (sonar, RF).

**D14. Enhancing Large Multimodal Models with Adaptive Sparsity and KV Cache Compression** — arXiv:2507.20613 — 2025 — Te Zhang et al.
Tree-structured Parzen Estimator dynamically tunes per-layer pruning ratios and KV-cache quantisation; outperforms SparseGPT/Wanda. Late-2025 SOTA on adaptive multimodal compression.

### Sub-cluster E — Operator-facing explainability for fusion outputs

**E1. A Review of Multimodal Explainable Artificial Intelligence: Past, Present and Future** — arXiv:2412.14056 — 2024 — Shilin Sun et al.
Reviews MXAI across four eras (traditional ML → generative LLMs); categorises methods, metrics, and datasets and discusses transparency-and-trust gaps. Canonical anchor for the sub-cluster.

**E2. Decoding the Multimodal Maze: A Systematic Review on the Adoption of Explainability in Multimodal Attention-based Models** — arXiv:2508.04427 — 2025 — Md Raisul Kibria et al.
PRISMA review of 2020–2024 explainability work in attention-based multimodal models. Documents the field's heavy reliance on attention-based explanations and the lack of standard cross-modal evaluation — useful as a methodological-gap reference.

**E3. I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts** — arXiv:2505.19190 — 2025 — Jiayi Xin et al. (ICML 2025)
End-to-end MoE that explicitly models heterogeneous multi-modal interactions and exposes both local (sample) and global (dataset) interpretations. Among the cleanest "inherently interpretable fusion" architectures of 2025.

**E4. Towards Explainable Fusion and Balanced Learning in Multimodal Sentiment Analysis (KAN-MCP)** — arXiv:2504.12151 — 2025 — Miaosen Luo et al.
Combines Kolmogorov-Arnold Networks (interpretable splines) with a Pareto multi-modal balancing scheme; provides visual explanations of cross-modal interactions. KAN-style architectures are an underexplored route to inherent interpretability in fusion.

**E5. GMAR: Gradient-Driven Multi-Head Attention Rollout for Vision Transformer Interpretability** — arXiv:2504.19414 — 2025 — Sehyeong Jo et al.
Re-weights heads in attention rollout by gradient-based importance scores. While ViT-targeted, the rollout primitive is inherited by every multi-modal transformer fusion stack and is the workhorse behind operator-facing saliency.

**E6. Head Pursuit: Probing Attention Specialization in Multimodal Transformers** — arXiv:2510.21518 — 2025 — Lorenzo Basile et al. (NeurIPS 2025 spotlight)
Identifies and ranks specialised attention heads in unimodal and multimodal transformers; shows that editing 1% of heads suppresses or enhances target concepts. A causal-style alternative to vanilla saliency for operator explanations.

**E7. ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features** — arXiv:2502.04320 — 2025 — Alec Helbling et al. (ICML 2025 oral)
Re-uses DiT attention parameters to produce sharp concept saliency maps without retraining. Generalises to any multi-modal transformer with text conditioning.

## 3. Datasets

Note: this scout's sub-buckets are mostly methodological. Primary datasets surfaced via the cited papers:

- **M4-SAR** (`M4-SAR/M4-SAR` per HF-style naming used in the paper; arXiv:2505.10931). 112k aligned optical-SAR pairs, ~1M instances. Licence: not stated in the search return — flag as **licence-check-required** before use.
- **KITTI** — ubiquitous benchmark referenced in A6, B6, D7. CC-BY-NC-SA 3.0 — **non-commercial restriction flagged** for IDEaS proposal use.
- **Ego4D** — referenced in A5 (LLM late fusion). Licensed under the **Ego4D Licence Agreement (custom, non-CC)** — flag as restrictive.
- **Sentinel-1 / Sentinel-2** — referenced as multimodal SAR/optical pretraining substrate in A2, A4 (X-STARS, Cloud-Aware). Copernicus open licence — permissive but with attribution.
- **Hugging Face Hub model-card / supply-chain corpora** referenced in C4 (arXiv:2402.05160) and C5 (arXiv:2502.04484). Public metadata, redistribution governed by HF terms.

No dataset directly underpins sub-clusters B, C (other than HF corpora), D, or E in a way the IDEaS proposal would consume — this scout's value is method-side, not data-side.

## 4. Reference implementations

- **vit-explain** — `https://github.com/jacobgil/vit-explain` — 1.1k stars. Implements attention rollout and gradient attention rollout for ViTs. Direct primitive behind E5 (GMAR is a re-weighted variant of this rollout family).
- **conformal-prediction (Angelopoulos)** — `https://github.com/aangelopoulos/conformal-prediction` — 1.0k stars. Reference notebooks for distribution-free uncertainty quantification across image classification, regression, segmentation; the canonical primitive used in B5/B6.
- **Trusted Multi-View Classification** — `https://github.com/hanmenghan/TMC` — 281 stars. Official implementation of the foundational TMC paper (ICLR 2021) and the TPAMI 2022 extension; the four-step recipe (replace softmax → Dirichlet → Dempster fusion → multi-task loss) underlies B2/B3/B4.
- **MobileVLM** — `https://github.com/Meituan-AutoML/MobileVLM` — 1.4k stars, Apache-2.0. 1.7B/3B mobile-deployable VLM; supports llama.cpp inference. Strong baseline for the SWaP sub-cluster.
- **FastVLM** — `https://github.com/apple/ml-fastvlm` — 7.3k stars. Apple's official PyTorch + Apple-Silicon export of FastVLM (D6); includes an iOS demo app — closest to the tactical-wearable form factor.
- **llama.cpp** — `https://github.com/ggerganov/llama.cpp` — 109k stars. Inference runtime supporting MobileVLM and many quantised multimodal stacks; canonical edge-deployment substrate.

## 5. Open questions you noticed

- **Does conformal calibration compose with evidential fusion?** Sub-cluster B presents two parallel uncertainty paradigms (evidential / subjective-logic vs. distribution-free conformal); no surveyed paper combines them. The literature does not establish whether conformal sets layered on top of evidential fusion preserve coverage when the evidence is itself miscalibrated.
- **Is there a "missing-modality" alignment recipe that transfers across modality pairs?** A3 (M4-SAR / IIR), A5 (LLM late fusion), B4 (Discount Fusion) and B6 (Any2Any) all handle missing-modality but in disjoint ways; no head-to-head comparison surfaced.
- **Provenance frameworks (Atlas, yProv4ML, HASC) do not specifically target multimodal fusion pipelines.** They treat the pipeline generically; whether per-modality lineage (e.g., classification level of the SAR feed vs. text intel feed) can be expressed in PROV-JSON or HASC is not demonstrated in any 2024–2026 paper I found.
- **Modality-balanced quantisation (D8) is shown only on VLMs (vision + text).** Its applicability to non-language modalities (RF spectrograms, sonar, telemetry) is unstudied.
- **Inherent-interpretability architectures (I2MoE, KAN-MCP) have not been benchmarked against post-hoc rollout methods on the same fusion task.** No paper in scope quantifies the explanation-fidelity vs. accuracy trade-off across the two paradigms.
- **Differential privacy under multi-modal fusion at classification boundaries.** The federated-multimodal papers (C6, C7) assume horizontal client splits; cross-classification-level fusion (e.g., UNCLAS partner ingest + SECRET native streams) is not addressed.
- **Empirical evidence on operator-facing explanations in real ISR/C2 user studies is absent from the surveyed work.** The systematic review (E2) explicitly flags the lack of cross-modal evaluation standards; no paper measures whether attention rollouts or KAN/MoE explanations actually improve operator decisions.
- **Reference implementations of evidential multimodal fusion in 2024–2025 do not appear consolidated** — most link back to the 2021 TMC repo. Whether the newer methods (B1–B4) have publicly released, maintainable code is uncertain from the scan.

## 6. Sources

- arXiv:2411.17040 — `https://arxiv.org/abs/2411.17040`
- arXiv:2405.09922 — `https://arxiv.org/abs/2405.09922`
- arXiv:2505.10931 — `https://arxiv.org/abs/2505.10931`
- arXiv:2506.17885 — `https://arxiv.org/abs/2506.17885`
- arXiv:2509.10729 — `https://arxiv.org/abs/2509.10729`
- arXiv:2504.19002 — `https://arxiv.org/abs/2504.19002`
- arXiv:2503.05274 — `https://arxiv.org/abs/2503.05274`
- arXiv:2412.18024 — `https://arxiv.org/abs/2412.18024`
- arXiv:2409.00755 — `https://arxiv.org/abs/2409.00755`
- arXiv:2408.13123 — `https://arxiv.org/abs/2408.13123`
- arXiv:2410.19653 — `https://arxiv.org/abs/2410.19653`
- arXiv:2411.10513 — `https://arxiv.org/abs/2411.10513`
- arXiv:2505.03788 — `https://arxiv.org/abs/2505.03788`
- arXiv:2511.15741 — `https://arxiv.org/abs/2511.15741`
- arXiv:2502.19567 — `https://arxiv.org/abs/2502.19567`
- arXiv:2507.01078 — `https://arxiv.org/abs/2507.01078`
- arXiv:2509.20394 — `https://arxiv.org/abs/2509.20394`
- arXiv:2402.05160 — `https://arxiv.org/abs/2402.05160`
- arXiv:2502.04484 — `https://arxiv.org/abs/2502.04484`
- arXiv:2504.12025 — `https://arxiv.org/abs/2504.12025`
- arXiv:2506.05683 — `https://arxiv.org/abs/2506.05683`
- arXiv:2502.07855 — `https://arxiv.org/abs/2502.07855`
- arXiv:2504.09724 — `https://arxiv.org/abs/2504.09724`
- arXiv:2503.10665 — `https://arxiv.org/abs/2503.10665`
- arXiv:2412.11475 — `https://arxiv.org/abs/2412.11475`
- arXiv:2510.11496 — `https://arxiv.org/abs/2510.11496`
- arXiv:2412.13303 — `https://arxiv.org/abs/2412.13303`
- arXiv:2506.07416 — `https://arxiv.org/abs/2506.07416`
- arXiv:2412.19509 — `https://arxiv.org/abs/2412.19509`
- arXiv:2509.18763 — `https://arxiv.org/abs/2509.18763`
- arXiv:2503.18278 — `https://arxiv.org/abs/2503.18278`
- arXiv:2503.02175 — `https://arxiv.org/abs/2503.02175`
- arXiv:2502.01158 — `https://arxiv.org/abs/2502.01158`
- arXiv:2506.07055 — `https://arxiv.org/abs/2506.07055`
- arXiv:2507.20613 — `https://arxiv.org/abs/2507.20613`
- arXiv:2412.14056 — `https://arxiv.org/abs/2412.14056`
- arXiv:2508.04427 — `https://arxiv.org/abs/2508.04427`
- arXiv:2505.19190 — `https://arxiv.org/abs/2505.19190`
- arXiv:2504.12151 — `https://arxiv.org/abs/2504.12151`
- arXiv:2504.19414 — `https://arxiv.org/abs/2504.19414`
- arXiv:2510.21518 — `https://arxiv.org/abs/2510.21518`
- arXiv:2502.04320 — `https://arxiv.org/abs/2502.04320`
- `https://github.com/jacobgil/vit-explain`
- `https://github.com/aangelopoulos/conformal-prediction`
- `https://github.com/hanmenghan/TMC`
- `https://github.com/Meituan-AutoML/MobileVLM`
- `https://github.com/apple/ml-fastvlm`
- `https://github.com/ggerganov/llama.cpp`
