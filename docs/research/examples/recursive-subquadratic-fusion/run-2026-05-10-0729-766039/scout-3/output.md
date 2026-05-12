# Scout 3 — State-space and linear-attention subquadratic backbones

## 1. Scope

Map the non-attention / linear-attention subquadratic architecture family — specifically state-space models (SSMs), linear-attention models, long-convolution models, and modern hybrids — with emphasis on (i) their state-update mechanism (analogue of "what attention preserves"), (ii) compute scaling, (iii) reasoning-depth / state-tracking limitations, and (iv) any prior combination with iterative or recursive heads, in service of the fusion thesis (TRM-style architectural recursion stacked on a subquadratic backbone).

Narrowing decisions:
- Treat each architecture as a primitive; no kernel-implementation depth.
- Distinguish *architectural recursion* (a learned operator applied K times in the forward pass — TRM-style) from *internal recurrence* (the SSM's own state update). Both exist in this family; only the former is the target of the fusion thesis.
- Recency: 2023–present (with one 2018 anchor for Universal Transformers as the conceptual root of architectural recursion).
- Vision-only and domain-specific Mamba variants are excluded unless they bear directly on state-tracking or recursion combination.

---

## 2. Key papers

### 2.1 Foundational subquadratic backbones (the building blocks the fusion thesis would compose with)

**Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (arXiv:2312.00752, 2023)
*Albert Gu, Tri Dao.*
Introduces selective SSM (S6) where the A, B, C matrices and discretization Δ are input-dependent, enabling content-based reasoning. State-update mechanism: a discretized linear ODE x' = Ā·x + B̄·u with input-dependent Ā, B̄; channel-wise (diagonal) sequence mixer. Compute scaling: linear (O(N) sequence length, with hardware-aware parallel scan). Why it matters: it is the canonical "what subquadratic backbone do we recurse on top of?" candidate; its diagonal state mixing is the load-bearing weakness that several state-tracking papers exploit.

**Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2)** (arXiv:2405.21060, 2024)
*Tri Dao, Albert Gu.*
Establishes the SSD (state-space duality) framework: SSMs and a class of linear attention are structured semiseparable matrices. Mamba-2 simplifies Mamba's selective scan into a matrix-multiply-friendly form (~2-8× faster) at the cost of slightly less expressive scalar-times-identity transitions. Compute: O(N·d²) sequential or chunkwise O(N·d) — same linear scaling as Mamba-1. Why it matters: this is the theoretical bridge. If recursion stacks on Mamba, it stacks on SSD-class linear attention — the fusion thesis can target SSD as the unified backbone abstraction.

**Mamba-3: Improved Sequence Modeling using State Space Principles** (arXiv:2603.15569, 2026)
*Aakash Lahoti, Kevin Y. Li, Berlin Chen, et al. (incl. Tri Dao, Albert Gu).*
Latest Mamba revision: complex-valued state-update rule (improves state tracking — directly addressing 2411.12537's complaint), trapezoidal discretization, multi-input multi-output (MIMO) variant. Why it matters: shows the SSM line is actively addressing state-tracking via richer eigenstructure rather than via stacking recursion — relevant baseline for the fusion thesis.

**RWKV: Reinventing RNNs for the Transformer Era** (arXiv:2305.13048, 2023)
*Bo Peng, Eric Alcaide, Quentin Anthony, et al.*
Linear RNN with attention-like channel-mix and time-mix; trains in parallel via WKV reformulation, runs as an RNN at inference. State: time-decayed exponential moving average over channel-projected keys/values. Compute: linear training, O(1) per-token inference. Why it matters: foundational non-attention subquadratic family that runs models up to 14B; reasoning quality lags Transformers especially on math.

**Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence (RWKV-5/6)** (arXiv:2404.05892, 2024)
*Bo Peng, Daniel Goldstein, Quentin Anthony, et al.*
Upgrades RWKV-4 with multi-headed matrix-valued states (richer than vector state) and data-dependent recurrence — closing the expressivity gap to Mamba. Compute: linear. Why it matters: documents a clear architectural arc — vector state → matrix state — driven by reasoning failures of vector-state RNNs.

**RWKV-7 "Goose" with Expressive Dynamic State Evolution** (arXiv:2503.14456, 2025)
*Bo Peng, Ruichong Zhang, Daniel Goldstein, et al. (incl. William Merrill).*
Generalises the delta rule with vector-valued in-context learning rates and a relaxed value-replacement rule. Authors explicitly claim RWKV-7 escapes the TC⁰ ceiling that constrains Transformers and Mamba (i.e., it can solve state-tracking tasks Transformers cannot, while remaining parallelizable). Why it matters: this is the strongest 2025 non-attention contender that *purports to address* state-tracking expressivity directly in the backbone — relevant comparator and possibly the cleanest baseline for "do you still need recursion on top?"

**Retentive Network: A Successor to Transformer for Large Language Models (RetNet)** (arXiv:2307.08621, 2023)
*Yutao Sun, Li Dong, Shaohan Huang, et al.*
Retention mechanism with three computational forms: parallel (training), recurrent (inference), and chunkwise-recurrent (long-context). State: exponentially decayed key-value matrix per head with multi-scale γ across heads. Compute: linear. Why it matters: established the "parallel-train / recurrent-infer" template that GLA, RWKV-5+, and Mamba-2 all inherit.

**Hyena Hierarchy: Towards Larger Convolutional Language Models** (arXiv:2302.10866, 2023)
*Michael Poli, Stefano Massaroli, Eric Nguyen, et al.*
Replaces attention with implicitly parameterised long convolutions interleaved with data-controlled gating. State: no explicit state — long convolution captures the global structure; the "memory" is the kernel itself. Compute: O(N log N) (FFT). Why it matters: a third way to subquadratic — distinct from SSMs and linear attention — and notably weaker on associative recall than either.

**HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution** (arXiv:2306.15794, 2023)
*Eric Nguyen, Michael Poli, Marjan Faizi, et al.*
Hyena applied to genomics with up to 1M-token context. Why it matters: demonstrates Hyena scales to ultra-long sequences in a non-language modality — establishes the operator's long-context credentials but also its empirical limit on language tasks (which led to StripedHyena hybrids).

**Simple linear attention language models balance the recall-throughput tradeoff (BASED)** (arXiv:2402.18668, 2024)
*Simran Arora, Sabri Eyuboglu, Michael Zhang, et al.*
Hybridises Taylor-series-feature linear attention with sliding-window attention. State: matrix-valued KV running sum + small SWA cache. Compute: linear-with-window. Why it matters: rigorously characterises the "recall–throughput" Pareto frontier — quantifies what linear-state architectures lose at fixed memory; central to thinking about whether recursion can recover lost recall.

**Gated Linear Attention Transformers with Hardware-Efficient Training (GLA)** (arXiv:2312.06635, 2023)
*Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim.*
Linear attention with data-dependent gating (per-element decay), reformulated as a 2D matrix-valued RNN. State: gated outer-product running sum. Compute: linear with chunkwise FlashAttention-style kernel. Why it matters: the immediate generalisation of RetNet and a key reference architecture for state-update *richness*.

**Parallelizing Linear Transformers with the Delta Rule over Sequence Length (DeltaNet)** (arXiv:2406.06484, 2024)
*Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim.*
Replaces the additive outer-product update with a delta rule: K-dependent "write a key, evict the previous one." State: Householder-structured matrix. Compute: linear. Why it matters: empirically the strongest associative-recall linear attention before RWKV-7; basis for Gated DeltaNet, Kimi Linear, etc.

**Longhorn: State Space Models are Amortized Online Learners** (arXiv:2407.14207, 2024)
*Bo Liu, Rui Wang, Lemeng Wu, et al.*
Recasts SSM updates as steps of an online regression, deriving an implicit-update SSM. State: implicit fixed-point of an online learner. Why it matters: a minor but conceptually important paper — frames SSM updates as solutions to an inner optimisation, which is one mathematical lens onto recursion-on-SSM.

**Test-Time Training (TTT) layers: Learning to (Learn at Test Time)** (arXiv:2407.04620, 2024)
*Yu Sun, Xinhao Li, Karan Dalal, et al.*
The hidden state IS a small ML model; the per-token update is a self-supervised gradient step on that model. Compute: linear. Why it matters: this is the closest existing architecture to "architectural recursion inside a subquadratic backbone" — every token application is a learned inner optimisation step — though the recursion is *along the sequence* (one step per token), not stacked depthwise like TRM. Critical comparator for the fusion thesis.

### 2.2 Hybrid SSM + attention architectures (the "what's already been tried" anchor)

**Jamba: A Hybrid Transformer-Mamba Language Model** (arXiv:2403.19887, 2024)
*Opher Lieber, Barak Lenz, Hofit Bata, et al. (AI21).*
Interleaves blocks of Mamba and Transformer layers, with MoE in some layers; 52B total / 12B active params, 256K context. State: per-layer choice of attention vs Mamba state. Compute: dominantly linear with periodic quadratic blocks. Why it matters: the production-scale data point that hybridisation works; the hybrid layout (one attention block every ~7 SSM blocks) is now a de-facto template.

**Zamba: A Compact 7B SSM Hybrid Model** (arXiv:2405.16712, 2024)
*Paolo Glorioso, Quentin Anthony, Yury Tokpanov, et al. (Zyphra).*
Mamba backbone with a *single shared attention module* called from multiple positions. Why it matters: parameter-efficiency move that prefigures TRM-style weight-tied recursion — a single shared module is invoked many times. The shared module is attention, not a learned operator, but the architectural move (share-and-reuse) is structurally analogous.

**Hymba: A Hybrid-head Architecture for Small Language Models** (arXiv:2411.13676, 2024)
*Xin Dong, Yonggan Fu, Shizhe Diao, et al. (NVIDIA).*
Hybrid *intra-layer*: each layer has parallel attention heads and SSM heads fused via meta-tokens. Compute: linear-dominated with partial sliding-window attention. Why it matters: shows intra-layer fusion (parallel) is a viable alternative to inter-layer fusion (Jamba, Zamba). Fusion thesis must choose: which slot does recursion wrap?

**Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling** (arXiv:2406.07522, 2024)
*Liliang Ren, Yang Liu, Yadong Lu, et al. (Microsoft).*
Layer-wise interleaving of Mamba and Sliding Window Attention; extrapolates well beyond training length. Why it matters: directly tests "compress-via-SSM, attend-locally" — the design pattern most compatible with adding a recursion head on top of the compressed state.

**Griffin: Mixing Gated Linear Recurrences with Local Attention** (arXiv:2402.19427, 2024)
*Soham De, Samuel L. Smith, Anushan Fernando, et al. (Google DeepMind).*
Hawk (RNN-only) and Griffin (RNN + local attention) families. Why it matters: shows gated linear recurrences match Mamba and that local-attention hybridisation matches Llama-2 with 6× fewer training tokens. RecurrentGemma productionised this.

**Hybrid Architectures for Language Models: Systematic Analysis and Design Insights** (arXiv:2510.04800, 2025)
*Sangmin Bae, Bilge Acun, Haroun Habeeb, et al.*
Most recent systematic study of hybrid SSM-attention design space (inter- vs intra-layer fusion, ratios, placement). Why it matters: the empirical map of where attention is load-bearing in hybrids — directly tells the fusion-thesis designer where in the stack a recursion head most needs to sit.

### 2.3 State-tracking expressivity limits (the load-bearing field for the fusion thesis)

**The Illusion of State in State-Space Models** (arXiv:2404.08819, 2024)
*William Merrill, Jackson Petty, Ashish Sabharwal.*
Proves Mamba-style SSMs (with diagonal/scalar transition) live in TC⁰ — no better than Transformers — and cannot solve permutation composition (S₅), parity, code evaluation, or entity tracking in one forward pass. Why it matters: the canonical reference for "SSMs are not magically more expressive than Transformers." Establishes the gap that recursion (architectural depth in forward pass) is hypothetically meant to fill.

**The Computational Limits of State-Space Models and Mamba via the Lens of Circuit Complexity** (arXiv:2412.06148, 2024)
*Yifang Chen, Xiaoyu Li, Yingyu Liang, Zhenmei Shi, Zhao Song.*
Independent confirmation: poly-precision constant-depth SSMs and Mamba sit in DLOGTIME-uniform TC⁰. Cannot solve arithmetic-formula evaluation, boolean-formula evaluation, or permutation composition. Why it matters: theoretical lower bound is now overdetermined — both Merrill et al. and this paper agree. Fusion thesis must argue that adding K iterations of an architectural recursion lifts effective depth out of TC⁰ within the same parameter budget.

**Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues** (arXiv:2411.12537, 2024)
*Riccardo Grazzi, Julien Siems, Jörg K. H. Franke, Arber Zela, Frank Hutter, Massimiliano Pontil.*
Shows that Mamba/RWKV/GLA/mLSTM/DeltaNet's state-transition matrices being constrained to non-negative eigenvalues is *the* reason they fail parity in one pass. Allowing negative eigenvalues (or identity-minus-rank-1 transitions) unlocks parity and improves code/math benchmarks. Why it matters: this is a backbone-internal fix to state-tracking — a competitor to "add recursion on top." Critical for the fusion thesis to engage with: maybe the right move is richer transitions, not stacked recursion.

**On the Expressiveness and Length Generalization of Selective SSMs on Regular Languages** (arXiv:2412.19350, 2024)
*Aleksandar Terzić, Michael Hersche, Giacomo Camposampiero, et al. (IBM).*
Distinguishes commutative vs non-commutative regular-language tasks: diagonal selective SSMs handle commutative ones, but non-commutative tasks need dense (non-diagonal) transition matrices. Proposes SD-SSM with full softmax-selected transition matrices and shows perfect length generalisation on FSA emulation. Why it matters: pinpoints exactly where the diagonal-state assumption breaks; this is the same expressivity surface where recursion would have to demonstrate it adds something beyond just "use dense transitions."

**Fixed-Point RNNs: Interpolating from Diagonal to Dense** (arXiv:2503.10799, 2025)
*Sajad Movahedi, Felix Sarnthein, Nicola Muca Cirone, Antonio Orvieto.*
Most directly relevant paper to the fusion thesis: parameterises a *dense* linear RNN as the fixed-point of K iterations of a *diagonal* linear RNN. Iterating the parallelizable diagonal recurrence K times in the forward pass converges to the dense recurrence — reaches state-of-the-art on state-tracking benchmarks. Why it matters: this *is* "architectural recursion stacked on a linear-RNN backbone" for the express purpose of recovering state-tracking expressivity. Closest existing work to the fusion thesis; the fusion-thesis designer must explicitly contrast TRM-style recursion (inputs-and-state-iteratively-refined) with fixed-point recursion (state-iteratively-refined-toward-a-mathematically-defined-attractor).

**Can Mamba Always Enjoy the "Free Lunch"?** (arXiv:2410.03810, 2024)
*Ruifeng Ren, Zhicong Li, Yong Liu.*
Proves Mamba (constant-size state) cannot solve COPY at scale and is theoretically limited on a class of dynamic-programming tasks; quantifies the overhead of Chain-of-Thought to recover. Why it matters: complements the Merrill bound with task-level evidence, and explicitly shows CoT is the workaround Mamba uses — an interesting analogue of "sequence-time recursion" the fusion thesis wants to *avoid* by doing recursion in latent space.

### 2.4 Non-attention reasoning models (test-time-compute on subquadratic backbones — overlap with fusion thesis)

**M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models** (arXiv:2504.10449, 2025)
*Junxiong Wang, Wen-Ding Li, Daniele Paliotta, Daniel Ritter, Alexander M. Rush, Tri Dao.*
Distills a Transformer-trained reasoning model into a hybrid Mamba (M1) and continues with RL on AIME/MATH; matches or beats DeepSeek-R1-Distill at much higher generation throughput. Why it matters: existence proof that subquadratic backbones can serve long-CoT reasoning at scale — and that the path taken so far is *sequence-time* CoT recursion, not architectural recursion. The fusion thesis differentiates by asking "can we do less generation and more in-pass recursion?"

**Scaling Reasoning without Attention** (arXiv:2505.22425, 2025)
*Xueliang Zhao, Wei Wu, Lingpeng Kong.*
Builds an attention-free reasoning model on Mamba-2 SSD layers with a curriculum FT strategy; competitive on AIME-24/25 and LiveCodeBench. State: SSD running matrix. Why it matters: the most direct claim that pure-SSM models can do hard reasoning. Important as "what does the no-recursion baseline already get you?" anchor.

**Thinking Slow, Fast: Scaling Inference Compute with Distilled Reasoners** (arXiv:2502.20339, 2025)
Mamba-style models distilled from Transformers, scaled at inference. Why it matters: confirms the test-time-compute-on-Mamba pattern is robust, not an M1 fluke.

### 2.5 Architectural recursion / looped depth (the recursion side of the fusion)

**Universal Transformers** (arXiv:1807.03819, 2018)
*Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Łukasz Kaiser.*
Original weight-tied looped Transformer with adaptive computation time. Why it matters: conceptual root of architectural recursion — applies the same Transformer block K times in the forward pass with a halting mechanism. Pre-dates SSMs but defines the genus.

**Reasoning with Latent Thoughts: On the Power of Looped Transformers** (arXiv:2502.17416, 2025)
Looped Transformers match deeper non-looped ones on reasoning tasks at a fraction of parameters. Why it matters: the modern theoretical case for architectural recursion as parameter-efficient depth; analogue of TRM's claim within the Transformer family.

**Scaling Latent Reasoning via Looped Language Models (LoopLM)** (arXiv:2510.25741, 2025)
Pretrained looped LMs with entropy-regularised loop objective; demonstrates iterative-computation pretraining at scale. Why it matters: most recent (and highest-upvoted) demonstration that looping works at LM-pretraining scale on Transformer backbones. Fusion thesis: does the same lift hold over an SSM backbone?

**Less is More: Recursive Reasoning with Tiny Networks (TRM)** (arXiv:2510.04871, 2025)
*Alexia Jolicoeur-Martineau.*
The fusion-thesis target architecture. 7M-parameter, two-layer network recursed many times, beats much larger LLMs on Sudoku/Maze/ARC-AGI. Why it matters: defines what "architectural recursion" means in the fusion thesis. State-of-art evidence that K iterations of a small operator can substitute for parameters.

**Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models** (arXiv:2510.14961, 2025)
*Jonas Geiping, Xinyu Yang, Guinan Su.*
Parallelises sampling for recurrent-depth (looped/universal) Transformers. Why it matters: one of the few recent papers explicitly connecting recurrent-depth (architectural recursion) to alternative compute paradigms; relevant to inference-cost story for the fusion thesis.

**Block-Recurrent Transformers** (arXiv:2203.07852, 2022)
Recurrent transformer over blocks with linear long-context behaviour. Why it matters: predecessor blueprint where a transformer block carries a recurrent state — relevant to "what does recursion look like when the carrier is a transformer block."

**Recurrent Memory Transformer (RMT) — Scaling Transformer to 1M tokens and beyond** (arXiv:2304.11062, 2023)
*Aydar Bulatov, Yuri Kuratov, Mikhail Burtsev.*
Adds learned memory tokens that are recurrently passed across segments. Why it matters: a different flavour of recursion — segment-level, carrying explicit memory tokens. Distinct from TRM-style depthwise recursion but relevant to "what state does recursion preserve over long context."

---

## 3. Datasets

This sub-topic is architecture-side and therefore only weakly dataset-driven. The relevant open datasets are the ones already used by the cited reasoning/state-tracking work:

- **MATH** (Hendrycks et al.) — used by M1, Scaling-Reasoning-without-Attention, RWKV-7. HF: `hendrycks/competition_math`. Licence: MIT.
- **AIME-24 / AIME-25** — math competition problems used by M1, Thinking-Slow-Fast, Scaling-Reasoning-without-Attention. Multiple HF mirrors (`HuggingFaceH4/aime_2024`, etc.); source AMC public competition data.
- **GSM8K** (Cobbe et al.) — math word problems. HF: `openai/gsm8k`. Licence: MIT.
- **LiveCodeBench** — code reasoning, contamination-controlled. HF: `livecodebench/code_generation`. Licence: research/CC-BY-NC.
- **The Pile** — Hyena/Mamba/RWKV pretraining default. HF: `EleutherAI/pile`. Licence: mixed; copyright concerns flagged, use of subsets recommended.
- **SlimPajama-627B** / **RedPajama-v2** — modern open pretraining corpora used by Mamba-2, Jamba ablations, Hymba. HF: `cerebras/SlimPajama-627B`, `togethercomputer/RedPajama-Data-V2`. Licences: ODC-BY / Apache 2.0.
- **Long Range Arena (LRA)** — original SSM benchmark, still used for state-tracking ablations. HF: `tomg-group-umd/long-range-arena`. Licence: Apache 2.0.
- **GenomicBenchmarks** / **Nucleotide Transformer benchmarks** — used by HyenaDNA. HF: `katielink/genomicbenchmarks`, `InstaDeepAI/nucleotide_transformer_downstream_tasks`. Licences: research / MIT.
- **Zoology synthetic-tasks** (associative recall, MQAR) — used by BASED, Mamba-2, GLA for backbone state-capacity probing. HF: `huaXiaKyrie/up` (linked off Mamba paper) and HazyResearch's zoology repo. Licence flag: please check repo (likely Apache 2.0).
- **ARC-AGI-1 / ARC-AGI-2** (Chollet) — TRM target benchmark; would be reused for any fusion-thesis recursion+SSM eval. Licence: Apache 2.0 / public.

Open-data status for all of the above is good; no closed datasets are required to reproduce the relevant ablations.

---

## 4. Reference implementations

- **state-spaces/mamba** — github.com/state-spaces/mamba — 18,207 stars. Canonical Mamba/Mamba-2 implementation with selective scan CUDA kernel.
- **BlinkDL/RWKV-LM** — github.com/BlinkDL/RWKV-LM — 14,517 stars. Canonical RWKV-4/5/6 reference; RWKV-7 reference at github.com/RWKV/RWKV-LM (62 stars at the new org).
- **HazyResearch/hyena-dna** — github.com/HazyResearch/hyena-dna — 784 stars. Reference Hyena training code (the language-model Hyena lives in HazyResearch/safari).
- **microsoft/Samba** — github.com/microsoft/Samba — 958 stars. Microsoft's Mamba+SWA hybrid; trained checkpoints + training code.
- **NVlabs/hymba** — github.com/NVlabs/hymba — 212 stars. NVIDIA's hybrid-head SLM.
- **berlino/gated_linear_attention** — github.com/berlino/gated_linear_attention — 107 stars. Reference GLA (the maintained fork lives at fla-org/flash-linear-attention with broader coverage).
- **hazyresearch/based** — github.com/hazyresearch/based — 252 stars. BASED reference.
- **google-deepmind/recurrentgemma** — github.com/google-deepmind/recurrentgemma — 674 stars. Productionised Griffin.
- **test-time-training/ttt-lm-jax** — github.com/test-time-training/ttt-lm-jax — 460 stars. TTT-Linear / TTT-MLP.
- **jxiw/M1** — github.com/jxiw/M1 — 47 stars. Hybrid-Mamba reasoning model; distillation+RL pipeline.
- **ibm/selective-dense-state-space-model** — github.com/ibm/selective-dense-state-space-model — 16 stars. SD-SSM with dense transition matrices for FSA emulation.
- **MoonshotAI/Kimi-Linear** — github.com/MoonshotAI/Kimi-Linear — 1,385 stars. State-of-art DeltaNet-derived hybrid (2025).
- **SamsungSAILMontreal/TinyRecursiveModels** — github.com/SamsungSAILMontreal/TinyRecursiveModels — 6,496 stars. TRM reference; the recursion side of the fusion thesis.

A maintained collection of linear-attention / SSM kernels and training code is at **fla-org/flash-linear-attention**, which covers GLA, RetNet, RWKV-6/7, DeltaNet, Mamba-2, Gated DeltaNet, and several hybrids — useful single dependency for a fusion-thesis prototype.

---

## 5. Open questions noticed (flagging only — no hypothesis or experiment design)

- **Q1 (most load-bearing).** Does TRM-style architectural recursion compose with a *diagonal-state* SSM (Mamba/Mamba-2) in a way that lifts effective depth out of TC⁰, or do you need to first switch to negative-eigenvalue/dense-transition variants (per 2411.12537, 2412.19350)? Equivalently: is the diagonal-state assumption a hard floor, or is it removable by enough K iterations of a recursion head?
- **Q2.** Fixed-Point RNNs (2503.10799) is *already* "K iterations of a diagonal linear-RNN converging to a dense linear-RNN." How does TRM-style recursion (which iterates a *learned operator* over a *latent thought*) differ formally and empirically from that fixed-point construction? Is one a strict superset?
- **Q3.** TTT (2407.04620) already runs an inner optimisation per token. Is TRM-style recursion stacked on TTT redundant, complementary, or in tension? Are they two axes of "internalised computation" (sequence-time vs depth-time)?
- **Q4.** All hybrid-SSM hard-reasoning results (M1, Scaling-Reasoning-without-Attention, Thinking-Slow-Fast) achieve hardness via long *output* CoT. None achieves it via depth-time recursion in the forward pass. Is this an artefact of inheritance (they distill from Transformers) or a real architectural barrier?
- **Q5.** RWKV-7 claims expressivity beyond TC⁰ in a single forward pass via richer state evolution. If true, does it eliminate the motivation for stacking recursion on top, or does recursion still buy parameter-efficiency at fixed expressivity?
- **Q6.** Mamba-3's complex-valued state updates and Hybrid-Architecture-Systematic-Analysis's design map are both 2025–2026. What is the *current* state-of-the-art subquadratic backbone for hard reasoning, and does that change which backbone the fusion thesis should target?
- **Q7.** Recurrent Memory Transformer carries an explicit memory token across segments; SSMs carry an implicit hidden state. When recursion is layered on top, *what state object is being refined* — the SSM hidden state, the memory tokens, the input embeddings, or a separate latent? The fusion thesis is silent on this and the answer matters.
- **Q8.** Gap: no paper found that combines *TRM-style architectural recursion* with *any* SSM backbone. Adjacent works exist (Fixed-Point RNNs as fixed-point recursion on linear RNNs; LoopLM/Universal-Transformers as architectural recursion on attention; M1 as test-time-compute on Mamba) but the literal product is empty. This is consistent with the fusion-thesis novelty claim.
- **Q9.** Hyena/Hyena-DNA use long convolutions, not state — recursion on top of a stateless long-conv backbone would have very different semantics. Should Hyena even be considered a valid fusion target?
- **Q10.** All cited state-tracking-limit papers are theoretical or use synthetic FSAs. Do the bounds bite empirically on math/proof/program-synthesis at the scales covered by the fusion thesis? The Mamba-can't-COPY result (2410.03810) suggests yes; the M1 success suggests not necessarily. Empirical scoping is open.

---

## 6. Sources

- arXiv:2312.00752 — Mamba — https://arxiv.org/abs/2312.00752
- arXiv:2405.21060 — Mamba-2 / SSD — https://arxiv.org/abs/2405.21060
- arXiv:2603.15569 — Mamba-3 — https://arxiv.org/abs/2603.15569
- arXiv:2305.13048 — RWKV-4 — https://arxiv.org/abs/2305.13048
- arXiv:2404.05892 — Eagle/Finch (RWKV-5/6) — https://arxiv.org/abs/2404.05892
- arXiv:2503.14456 — RWKV-7 — https://arxiv.org/abs/2503.14456
- arXiv:2307.08621 — RetNet — https://arxiv.org/abs/2307.08621
- arXiv:2302.10866 — Hyena — https://arxiv.org/abs/2302.10866
- arXiv:2306.15794 — HyenaDNA — https://arxiv.org/abs/2306.15794
- arXiv:2402.18668 — BASED — https://arxiv.org/abs/2402.18668
- arXiv:2312.06635 — GLA — https://arxiv.org/abs/2312.06635
- arXiv:2406.06484 — DeltaNet — https://arxiv.org/abs/2406.06484
- arXiv:2407.14207 — Longhorn — https://arxiv.org/abs/2407.14207
- arXiv:2407.04620 — TTT layers — https://arxiv.org/abs/2407.04620
- arXiv:2403.19887 — Jamba — https://arxiv.org/abs/2403.19887
- arXiv:2405.16712 — Zamba — https://arxiv.org/abs/2405.16712
- arXiv:2411.13676 — Hymba — https://arxiv.org/abs/2411.13676
- arXiv:2406.07522 — Samba — https://arxiv.org/abs/2406.07522
- arXiv:2402.19427 — Griffin/Hawk — https://arxiv.org/abs/2402.19427
- arXiv:2510.04800 — Hybrid Architectures Systematic Analysis — https://arxiv.org/abs/2510.04800
- arXiv:2404.08819 — Illusion of State in SSMs — https://arxiv.org/abs/2404.08819
- arXiv:2412.06148 — Computational Limits of SSMs / Mamba — https://arxiv.org/abs/2412.06148
- arXiv:2411.12537 — Negative-Eigenvalue State-Tracking — https://arxiv.org/abs/2411.12537
- arXiv:2412.19350 — Selective SSMs on Regular Languages — https://arxiv.org/abs/2412.19350
- arXiv:2503.10799 — Fixed-Point RNNs — https://arxiv.org/abs/2503.10799
- arXiv:2410.03810 — Can Mamba Always Enjoy the "Free Lunch"? — https://arxiv.org/abs/2410.03810
- arXiv:2504.10449 — M1 — https://arxiv.org/abs/2504.10449
- arXiv:2505.22425 — Scaling Reasoning without Attention — https://arxiv.org/abs/2505.22425
- arXiv:2502.20339 — Thinking Slow, Fast — https://arxiv.org/abs/2502.20339
- arXiv:1807.03819 — Universal Transformers — https://arxiv.org/abs/1807.03819
- arXiv:2502.17416 — Reasoning with Latent Thoughts (Looped Transformers) — https://arxiv.org/abs/2502.17416
- arXiv:2510.25741 — LoopLM — https://arxiv.org/abs/2510.25741
- arXiv:2510.04871 — TRM — https://arxiv.org/abs/2510.04871
- arXiv:2510.14961 — Parallel Samplers for Recurrent-Depth Models — https://arxiv.org/abs/2510.14961
- arXiv:2203.07852 — Block-Recurrent Transformers — https://arxiv.org/abs/2203.07852
- arXiv:2304.11062 — RMT — https://arxiv.org/abs/2304.11062
- github.com/state-spaces/mamba
- github.com/BlinkDL/RWKV-LM
- github.com/RWKV/RWKV-LM
- github.com/HazyResearch/hyena-dna
- github.com/microsoft/Samba
- github.com/NVlabs/hymba
- github.com/berlino/gated_linear_attention
- github.com/hazyresearch/based
- github.com/google-deepmind/recurrentgemma
- github.com/test-time-training/ttt-lm-jax
- github.com/jxiw/M1
- github.com/ibm/selective-dense-state-space-model
- github.com/MoonshotAI/Kimi-Linear
- github.com/SamsungSAILMontreal/TinyRecursiveModels
- github.com/fla-org/flash-linear-attention
