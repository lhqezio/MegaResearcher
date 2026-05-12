# Scout 1 — Architectural Recursion / Iterative-Depth Networks

## 1. Scope

**Sub-topic (one sentence).** Networks that gain effective reasoning depth by applying a learned operator (often a single shared block) recursively within a single forward pass — i.e. *architectural* recursion, not agent-scaffolded prompting.

**Narrowing decisions.**
- I include only architectures where a learned function `f` is applied K times to its own output (or jointly to its output and a context) inside one forward pass. ALBERT-style cross-layer weight tying is included only as direct prior art for recursive transformers (the operator is "applied K times" by virtue of being shared).
- I exclude pure test-time-scaling-via-CoT, recursive prompting, MCTS / parallel sampling, and "recursive program calls between LM invocations" (e.g. arXiv:2603.02112 *Recursive Models for Long-Horizon Reasoning* and arXiv:2603.20105 *Y-Combinator for LLMs*). Those papers use the word "recursive" but the recursion is **outside** the forward pass; the spec's YAGNI fence rules them out. Where they are flagged below, it is to mark the conflation explicitly.
- For each entry I capture the four signals the assignment requested: (i) recursion mechanism, (ii) what the operator attends to per step, (iii) reasoning-depth claim and where it breaks down, (iv) compatibility / silence on subquadratic attention.

A note on terminology used throughout: "weight-tied loop" and "looped transformer" and "recursive transformer" and "depth-recurrent transformer" are treated as overlapping labels for the same family — these papers reuse a single block / stack `K` times. "Architectural recursion" is my umbrella term per the spec.

---

## 2. Key papers

### 2A. The TRM / HRM lineage (immediate ancestors of the spec's recursion side)

**1. Less is More: Recursive Reasoning with Tiny Networks (TRM)**
Jolicoeur-Martineau. arXiv:**2510.04871**, 2025. *(spec's anchor.)*
**Mechanism.** A 2-layer network is applied recursively K times within one forward pass, alternating updates of a "reasoning state" `z` and an "answer state" `y` (so `z ← f(z, y, x)` then `y ← g(z, y)` repeated). Trained with deep supervision per recursion step rather than via a 1-step implicit-fixed-point gradient (this is the explicit improvement over HRM).
**Attends to.** Each step reads the static problem encoding `x`, the current `y`, and the previous `z`. No external KV cache; the operator's "context" is a fixed-size tensor.
**Depth claim and break.** 7M-parameter TRM beats Gemini-2.5-Pro / o3-mini on ARC-AGI-1/2 puzzle tasks; the paper does **not** test long-text reasoning, multi-doc QA, or sequence lengths beyond puzzle grids. Authors flag that recursion benefits saturate without their MLP-only inner block on grid tasks (attention hurts on ARC-AGI but helps on Sudoku).
**Subquadratic-attention compatibility.** Silent. The released model uses small dense self-attention inside the block; nothing in the architecture forbids substituting a subquadratic operator, but no experiment is reported.
**Code/weights.** github.com/SamsungSAILMontreal/TinyRecursiveModels (6.5k★). HF datasets: `emiliocantuc/sudoku-extreme-1k-aug-1000`. Community weights: `wtfmahe/Samsung-TRM`.

**2. Hierarchical Reasoning Model (HRM)**
Wang, Li, Sun, Chen, Liu, Wu, Lu, Song, Yadkori. arXiv:**2506.21734**, 2025.
**Mechanism.** Two coupled recurrent modules at different timescales: a "low-level" module updates many steps per outer "high-level" step, both weight-shared across recursions. Trained with a one-step approximate gradient through the implicit fixed point (DEQ-style), not BPTT.
**Attends to.** Inner module attends to its own state plus a slow-varying high-level vector; high-level attends to converged low-level state and original input embedding.
**Depth claim and break.** ~27M params, ~1k training examples, beats much larger LLMs on ARC-AGI / Maze / Sudoku. Subsequent work (arXiv:2601.10679, see entry 14) shows HRM frequently fails on *trivial* instances of the same puzzle class, indicating the "reasoning" is partly solver-shaped fixed-point matching rather than algorithmic.
**Subquadratic-attention compatibility.** Silent. Inner module uses standard attention over a small token grid; no test of how it scales when the grid is replaced by a 100k-token document.

**3. Deep Improvement Supervision (TRM training improvement)**
Asadulaev, Banerjee, Karray, Takac. arXiv:**2511.16886**, 2025.
**Mechanism.** Reframes TRM's recursive iterations as classifier-free guidance / implicit policy improvement; introduces a halting head and a CFG-style training signal across recursion depth.
**Attends to.** Same as TRM (`z`, `y`, `x`).
**Depth claim and break.** Reduces forward passes ~10× while matching ARC accuracy. Still puzzle-only; no long-context test.
**Subquadratic compat.** Silent.

### 2B. Looped / recursive transformers (explicit fusion candidates)

**4. Universal Transformers**
Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser. arXiv:**1807.03819**, 2018. *(canonical pre-2023 reference.)*
**Mechanism.** Apply one transformer encoder block K times with shared weights, plus per-position ACT-style adaptive halting (each token can stop independently). Backprop is BPTT through K applications.
**Attends to.** Full self-attention over the input sequence at every recursion step (no compression of context across recursions).
**Depth claim and break.** Beats vanilla Transformer on LAMBADA and bAbI; struggles to scale to LM-grade parameter counts because layer-sharing reduces parameter–compute ratio (revisited by MoEUT).
**Subquadratic compat.** Silent (predates most subquadratic work). Attention is dense inside every recursion step — directly relevant: the operator re-attends to the same long sequence K times, multiplying any per-step attention cost.

**5. Looped Transformers as Programmable Computers**
Giannou, Rajput, Sohn, Lee, Lee, Papailiopoulos. arXiv:**2301.13196**, 2023.
**Mechanism.** A constant-depth transformer wrapped in an outer loop is shown by *construction* to emulate a Turing-complete instruction-set computer (program counter, branches, function calls, in-context backprop).
**Attends to.** A "punchcard" input that doubles as program + memory; full attention to the whole strip every step.
**Depth claim and break.** Theoretical, not empirical — establishes that weight-tied loop + attention can express any algorithm given enough loop iterations.
**Subquadratic compat.** Their construction relies on attention precisely retrieving program-counter cells; sparse-by-default attention that drops "unimportant" tokens would by definition kill the construction. **First clear flag** that subquadratic attention may interact badly with looped programs that re-reference any sequence position.

**6. Looped Transformers are Better at Learning Learning Algorithms**
Yang, Lee, Nowak, Papailiopoulos. arXiv:**2311.12424**, 2023.
**Mechanism.** Trains a small transformer block in a loop (K fixed at train, optionally varied at test) on in-context regression tasks; matches deeper non-looped baselines at a fraction of params.
**Attends to.** ICL-style: the same prompt sequence at every loop iteration; the loop *implements* an iterative learner like gradient descent.
**Depth claim and break.** The loop count behaves like solver iterations; performance saturates when loop depth exceeds the implicit algorithm's natural step count. Synthetic regression only.
**Subquadratic compat.** Silent.

**7. Reasoning with Latent Thoughts: On the Power of Looped Transformers**
Saunshi, Dikkala, Li, Kumar, Reddi. arXiv:**2502.17416**, 2025.
**Mechanism.** Theory + experiments: many reasoning problems have high *depth* requirement but low *parameter* requirement; looping a k-layer block K times mimics a kK-layer stack on hop-induction, addition, math. Adds a regularization that makes the loop count behave like a "thinking budget".
**Attends to.** Same input every iteration; latent state evolves across iterations.
**Depth claim and break.** Ablation shows benefit on serial / multi-hop tasks; on retrieval-heavy or pure-bandwidth tasks, loops do not substitute for parameters.
**Subquadratic compat.** Silent. Maps cleanly to subquadratic backbones in principle but not tested.

**8. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn-3.5B)**
Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein. arXiv:**2502.05171**, 2025.
**Mechanism.** A recurrent block is iterated arbitrarily many times at test-time (e.g. 4 → 64 iterations) on top of an embedded prefix; works without CoT-style training data, with small context window. 3.5B-param decoder pretrained from scratch with random per-step iteration count to allow inference-time depth scaling.
**Attends to.** A latent state initialized from sequence embeddings; standard self-attention over the current token positions inside each recurrent block.
**Depth claim and break.** Performance on GSM8k / MMLU improves monotonically with iteration count up to ~32, then saturates. Failure mode: latent state can collapse if init is poor, mitigated by random-truncation training.
**Subquadratic compat.** Silent. Released model uses dense attention; one of the most natural fusion candidates because depth scaling is already orthogonal to context length. Code: github.com/seal-rg/recurrent-pretraining (883★). Weights: `tomg-group-umd/huginn-0125`. Dataset: `tomg-group-umd/huginn-dataset`.

**9. Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer (Huginn analysis)**
Lu, Yang, Lee, Li, Liu. arXiv:**2507.02199**, 2025.
**Mechanism.** Probes Huginn-3.5B (entry 8) with Logit Lens / Coda Lens.
**Findings.** Recurrence depth helps marginally; the model does **not** show interpretable step-by-step latent CoT in arithmetic — gains plateau and probes disagree across recurrence steps.
**Why it matters.** Strong negative-result reference: depth-recurrent gains are not automatically "deeper reasoning" in any human-interpretable sense. Critical for spec's claim about "depth of reasoning".
**Subquadratic compat.** Silent.

**10. Scaling Latent Reasoning via Looped Language Models (Ouro / LoopLM)**
Zhu, Wang, Hua, et al. arXiv:**2510.25741**, 2025.
**Mechanism.** Pretrains 1.4B / 2.6B "Ouro" decoders with iterative computation built into pre-training — entropy-regularized objective for per-token learned depth allocation, plus a "Thinking" variant.
**Attends to.** Standard causal attention over context; the loop iterates the entire stack.
**Depth claim and break.** Strong reasoning gains versus same-flop baselines, particularly on knowledge-manipulation. Authors note pretraining cost is non-trivial; per-token halting is learned but coarse.
**Subquadratic compat.** Silent. Uses dense attention. Weights publicly available: `ByteDance/Ouro-1.4B`, `ByteDance/Ouro-2.6B`, plus `-Thinking` variants.

**11. Mixture-of-Recursions (MoR)**
Bae, Kim, Bayat, Kim, Ha, Schuster, Fisch, Harutyunyan, Ji, Courville et al. arXiv:**2507.10524**, 2025.
**Mechanism.** A shared stack of layers is reused across recursions, but a lightweight router decides per-token how many recursion steps to spend; KV pairs are also shared across recursion steps for the same token.
**Attends to.** Standard attention; recursion repeats the whole stack with shared K and V cache reuse.
**Depth claim and break.** Improves perplexity at matched FLOPs; cuts prefill latency by reducing redundant KV computation.
**Subquadratic compat.** **Most direct touch-point.** MoR explicitly notes that quadratic attention dominates cost at long context and shares KV across recursion steps as a partial mitigation. This is the closest existing work to the spec thesis but does not actually swap in a subquadratic attention operator. Code: github.com/raymin0223/mixture_of_recursions (571★).

**12. Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA**
Bae, Fisch, Harutyunyan, Ji, Kim, Schuster. arXiv:**2410.20672**, 2024.
**Mechanism.** Convert an existing pretrained LLM into a "Recursive Transformer" by tying layers, then add layer-specific LoRA to relax the tying; introduces "Continuous Depth-wise Batching" for inference parallelism across recursion steps.
**Attends to.** Standard attention.
**Depth claim and break.** Recovers most of the original LLM's quality at much lower parameter count by reusing the same block multiple times. Recovery degrades on knowledge-heavy tasks.
**Subquadratic compat.** Silent — but the "Continuous Depth-wise Batching" trick is interesting because it pipelines independent recursion-depth budgets, suggesting a path to mix recursion with subquadratic streaming attention.

**13. Improving Recursive Transformers with Mixture of LoRAs (MoL)**
Nouriborji, Rohanian, Rohanian. arXiv:**2512.12880**, 2025.
**Mechanism.** Inserts LoRA experts inside the FFN of a parameter-shared recursive transformer; experts are token-conditional, restoring expressivity lost to layer tying.
**Attends to.** Standard attention.
**Depth claim and break.** Closes most of the quality gap to a non-recursive baseline at equal parameters; tested on GLUE / SQuAD-v2 / BEIR — encoder tasks, not long-context generation.
**Subquadratic compat.** Silent.

**14. Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of HRM**
Ren, Liu. arXiv:**2601.10679**, 2026.
**Mechanism (under analysis).** Probes HRM iterates and shows: failure on extremely simple instances; multiple fixed points; "grokking" dynamics during training. Proposes input-perturbation augmentation to push toward a unique fixed point.
**Why it matters.** Critical falsification reference: the operator's iteration trajectory is fragile, and the recursion may be doing pattern-completion rather than search. Anyone fusing recursion with subquadratic attention must consider whether sparse-attention noise destabilizes the same fixed-point property.

**15. SpiralFormer: Looped Transformers Can Learn Hierarchical Dependencies via Multi-Resolution Recursion**
Yu, Shu, Wang, Zhang, Wu, Wu, Long, Chen, Xu, Su et al. arXiv:**2602.11698**, 2026.
**Mechanism.** Recursion at multiple sequence resolutions — early loops operate over compressed/coarse tokens, later loops over fine tokens. Functional specialization across loop iterations.
**Attends to.** Resolution-dependent token grids per loop step.
**Depth claim and break.** Beats fixed-resolution looped transformers at matched compute on reasoning tasks.
**Subquadratic compat.** **Notable.** Multi-resolution attention is inherently subquadratic-flavored; this is the most explicit existing fusion of recursion with cost-controlled attention, although it is not framed as a subquadratic-attention contribution.

**16. LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation**
Jeddi, Ciccone, Taati. arXiv:**2602.11451**, 2026.
**Mechanism.** Variable-length trajectory training with shortcut-consistency regularization — the loop count becomes adjustable at inference without retraining.
**Attends to.** Standard attention.
**Depth claim and break.** Adapts compute per query under different budgets; trades off accuracy gracefully when loop budget is small.
**Subquadratic compat.** Silent.

**17. Parallel Loop Transformer (PLT)**
Wu, Chen, Luo, Yan, Yu, Xia, Zhang, Zhan, Zhong, Zhou et al. arXiv:**2510.24824**, 2025.
**Mechanism.** Cross-Loop Parallelism: parallelize execution across loop iterations using Gated Sliding-Window Attention, so additional loops do not incur sequential latency.
**Attends to.** **Sliding-window attention** within each loop step — i.e. a *subquadratic* operator. This is the only paper in the corpus that explicitly couples loop recursion with a subquadratic attention pattern.
**Depth claim and break.** Maintains accuracy of serial loop execution while removing the latency penalty.
**Subquadratic compat.** **Highly relevant.** Direct evidence that sliding-window attention can host loop recursion without collapse; missing analysis of *which* tokens get dropped across loops and whether information needed at late loops survives the window.

**18. LoopViT: Scaling Visual ARC with Looped Transformers**
Shu, Qiu, Zhu, Chen, Liu, Yang. arXiv:**2602.02156**, 2026.
**Mechanism.** Vision transformer with weight-tied recurrence on a Hybrid (local conv + global attention) Block, plus a predictive-entropy Dynamic Exit.
**Attends to.** Local conv + global attention over visual tokens, repeated K times.
**Depth claim and break.** New SOTA on visual ARC-AGI at small param counts.
**Subquadratic compat.** Local convolution is linear in tokens; global attention block is dense, so this is a hybrid not a clean subquadratic. Useful as a precedent for "recursion + non-uniform attention pattern".

**19. Sparse Universal Transformer (SUT)**
Tan, Shen, Chen, Courville, Gan. arXiv:**2310.07096**, 2023.
**Mechanism.** UT with sparse mixture-of-experts inside the shared block, plus a stick-breaking dynamic halting mechanism.
**Attends to.** Standard attention; expert routing per token.
**Depth claim and break.** Compositional generalization on CFQ / WMT'14; halting reduces compute.
**Subquadratic compat.** "Sparse" here is over experts, not over tokens — does **not** address attention cost. But the halting-mechanism + recursion combo is a building block.

**20. MoEUT: Mixture-of-Experts Universal Transformers**
Csordás, Irie, Schmidhuber, Potts, Manning. arXiv:**2405.16039**, 2024.
**Mechanism.** Universal Transformer + MoE, with novel layer-norm placement and grouping schemes to fix the parameter-compute ratio of UTs. ~1B-param MoE-UT competitive with non-shared baseline.
**Attends to.** Standard attention; MoE routing.
**Depth claim and break.** Closes the practical gap that prevented UT from scaling. Compositional generalization on PIQA, BLiMP.
**Subquadratic compat.** Silent.

### 2C. Implicit-depth and DEQ-flavored architectures

**21. Deep Equilibrium Models (DEQ)**
Bai, Kolter, Koltun. arXiv:**1909.01377**, 2019. *(classic, directly informs HRM's 1-step gradient.)*
**Mechanism.** Treats infinite-depth weight-tied stack as the fixed point of `z = f(z, x)`; finds the equilibrium by root-finding (Broyden / Anderson) and backprops via implicit differentiation, bypassing BPTT memory costs.
**Attends to.** `f` can be any differentiable operator including a transformer block.
**Depth claim and break.** Constant-memory training of "infinite-depth" sequence models. Convergence not guaranteed; instability at scale.
**Subquadratic compat.** Silent in 2019. The implicit-gradient trick *should* compose with any subquadratic operator inside `f`.

**22. Adaptive Computation Time (ACT) for RNNs**
Graves. arXiv:**1603.08983**, 2016. *(classic.)*
**Mechanism.** A halting unit emits a probability per step; cumulative probability triggers stop. Differentiable, deterministic, no extra noise.
**Why it's still cited.** Direct ancestor of UT halting, PonderNet, MoR routing, LoopFormer elastic depth — every adaptive-recursion paper inherits the mechanism.

**23. PonderNet: Learning to Ponder**
Banino, Balaguer, Blundell. arXiv:**2107.05407**, 2021.
**Mechanism.** Reformulates ACT as a probabilistic per-step halting that yields a proper distribution over computation lengths; trained end-to-end on prediction accuracy + computation cost.
**Why it matters.** Standard reference for "halting head" in TRM-style recursion.

### 2D. Negative results and theory

**24. On Expressive Power of Looped Transformers (timestep encoding)**
Xu, Sato. arXiv:**2410.01405**, 2024.
**Mechanism (theory).** Establishes approximation rate of looped transformers via a modulus-of-continuity argument; identifies a fundamental expressiveness ceiling without a per-step timestep encoding.
**Why it matters.** Theoretical floor on how much depth-of-recursion can substitute for parameter count; predicts known empirical saturation. Direct caveat for the spec.

**25. Transformer-Based Models Are Not Yet Perfect At Learning to Emulate Structural Recursion**
Zhang, Tigges, Zhang, Biderman, Raginsky, Ringer. arXiv:**2401.12947**, 2024.
**Why it matters.** Empirically, transformers fail at *structural* recursion (recursion over data structures such as ASTs) even when given many examples. **Critical:** "architectural recursion" (loop the operator) is not the same as "the model has learned the concept of recursion" — papers sometimes conflate these.

**26. Chain of Thought Empowers Transformers to Solve Inherently Serial Problems**
Z. Li, H. Liu, D. Zhou, T. Ma. arXiv:**2402.12875**, 2024.
**Mechanism (theory).** Constant-depth transformers + CoT can express problems beyond TC0/AC0 — i.e. serial computation requires *some* form of depth, whether unrolled in tokens or in loops.
**Why it matters.** Theoretical support for the proposition that depth-of-iteration buys real expressivity, but does not distinguish architectural recursion from token-CoT — relevant context for the spec's separation.

### 2E. Adjacent / related (use cautiously)

**27. Intra-Layer Recurrence in Transformers for Language Modeling**
Nguyen, Lin. arXiv:**2505.01855**, 2025.
**Mechanism.** Apply recurrence selectively to specific layers of a stack rather than the whole stack — finer-grained control of which sub-operators are looped.
**Why it matters.** Shows partial-recursion ablation surface; useful for designing the fusion experiment.

**28. Adaptive Loops and Memory in Transformers: Think Harder or Know More?**
Frey, Shomali, Bashir, Berghaus, Koehler, Ali. arXiv:**2603.08391**, 2026.
**Mechanism.** Adaptive per-layer looping + gated memory bank; tries to disentangle "more iterations" from "more storage".
**Why it matters.** Direct probe of the same question the spec asks: does extra recursion compensate for limited per-layer state? Suggests both are needed.

**29. Depth-Recurrent Attention Mixtures (Dreamer)**
Knupp, Metzen, Bohn, Groh, Kersting. arXiv:**2601.21582**, 2026.
**Mechanism.** Combines sequence attention, depth attention, and sparse expert attention in a depth-recurrent block; explicitly studies the hidden-size bottleneck of UT-like recursion.
**Why it matters.** **Notable.** "Depth attention" itself is a primitive that mixes information across recursion steps — relevant if the fusion needs to remember earlier loop iterates as a sequence.
**Subquadratic compat.** Sparse expert attention is closer to the subquadratic regime than dense.

**30. Recurrent-Depth VLA**
Tur, Naghiyev, Fang, Tsai, Duan, Fox, Krishna. arXiv:**2602.07845**, 2026.
**Mechanism.** Recurrent action head trained with truncated BPTT; latent convergence as a stopping signal; constant memory regardless of recurrence depth.
**Why it matters.** Recursion in a different modality (VLA), reinforcing that the recursion idea generalizes; uses truncated BPTT — gradient method choice.

**31. Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization**
Chen. arXiv:**2603.21676**, 2026.
**Mechanism.** Shared-weight transformer block iterated K times in latent space; adds "silent thinking" objective, LayerScale init, identity-biased recurrence to stabilize deep recurrence.
**Why it matters.** Most recent crystallization of the design pattern; reports that compositional generalization patterns differ from token-CoT.

**32. A Survey on Latent Reasoning**
Zhu, Peng, Cheng, Qu et al. arXiv:**2507.06203**, 2025.
**Why it matters.** Standard survey reference; partitions latent reasoning into activation-based recurrence, hidden-state propagation, and infinite-depth via masked diffusion — taxonomy that subsumes the architectures above. Uses term "infinite-depth latent reasoning".

---

## 3. Datasets

Open and HF-hosted, relevant to architectural-recursion research (especially as benchmarks against which the spec's fusion would be evaluated):

| Name | HF ID | License | Use |
|---|---|---|---|
| Sudoku-Extreme (TRM-style) | `emiliocantuc/sudoku-extreme-1k-aug-1000` | not declared on card — **flag for license check before use** | TRM/HRM baseline puzzle |
| HRM-dataset | `ThomasHeim/HRM-dataset` | not declared on card — **flag** | HRM training |
| Huginn pretraining mixture | `tomg-group-umd/huginn-dataset` | dataset card cites SmolLM-Corpus + ProofPile-2 + StarCoder etc. — derivative; check upstream | Recurrent-depth pretraining |
| ARC-AGI v1/v2 | not on HF as a single official dataset; obtain from `github.com/fchollet/ARC-AGI-1` (ARC license) | ARC-AGI license (research use, attribution) | TRM/HRM/LoopViT eval |
| GSM8K | `openai/gsm8k` | MIT | Reasoning eval used by Huginn / Ouro |
| MMLU | `cais/mmlu` | MIT | Reasoning eval used by Ouro / MoR |
| LongBench v2 | `THUDM/LongBench-v2` | MIT | Long-context reasoning — useful for the spec's fusion question |
| Loogle | `bigai-nlco/LooGLE` | Apache-2.0 | Long-doc multi-hop QA |
| RULER | `nvidia/RULER` | Apache-2.0 | Long-context probe across lengths |
| BABILong | `RMT-team/babilong` | Apache-2.0 | Long-context bAbI extension — directly suits "depth × context-length" plane |
| ProofPile-2 | `EleutherAI/proof-pile-2` | mixed (subsets vary) | Math/proof pretraining for recursion-on-math experiments |

(I have not run a hardness assertion on each card; HF dataset license fields are sometimes empty even when the data is openly distributed. License flagged where missing.)

---

## 4. Reference implementations

Public, recursion-relevant repos with star counts. Star counts captured from search-tool output, not re-verified at the commit-SHA level (no GPU spend / no external clone).

| Repo | What it implements | Stars (approx.) |
|---|---|---|
| `SamsungSAILMontreal/TinyRecursiveModels` | TRM (arXiv:2510.04871) — 2-layer recursive operator with deep supervision | 6,496 |
| `seal-rg/recurrent-pretraining` | Huginn / recurrent-depth pretraining (arXiv:2502.05171) | 883 |
| `raymin0223/mixture_of_recursions` | MoR (arXiv:2507.10524) — token-level adaptive recursion + KV sharing | 571 |
| `multimodal-art-projection/LatentCoT-Horizon` | Survey companion repo (arXiv:2507.06203) — curated latent-reasoning code | 385 |
| `wenquanlu/huginn-latent-cot` | Huginn probing (arXiv:2507.02199) | 18 |
| `robertcsordas/moeut` | MoEUT (arXiv:2405.16039) | 92 |
| `armenjeddi/loopformer` | LoopFormer (arXiv:2602.11451) | 19 |
| `WenjieShu/LoopViT` | Loop-ViT (arXiv:2602.02156) | 40 |
| `leiay/looped_transformer` | Yang et al. looped-ICL (arXiv:2311.12424) | 43 |
| `shawntan/SUT` | Sparse Universal Transformer (arXiv:2310.07096) | 20 |
| `ant-8/Layer-Recurrent-Transformers` | Intra-Layer Recurrence (arXiv:2505.01855) | 6 |
| `szq0214/sret` | Sliced Recursive Transformer (arXiv:2111.05297) | 66 |
| `yuedajiong/looped-transformer` | Reference for Giannou et al. (arXiv:2301.13196) | 4 |
| `tinkoff-ai/palbert` | PALBERT — ALBERT + PonderNet (arXiv:2204.03276) | 37 |
| `ceyzaguirre4/adaptive_computation` | ACT (Graves, arXiv:1603.08983) | 4 |

Pretrained weights worth noting (open):
- `tomg-group-umd/huginn-0125` — 3.5B recurrent-depth checkpoint
- `ByteDance/Ouro-1.4B`, `ByteDance/Ouro-2.6B` (+ `-Thinking` variants) — pretrained looped LMs
- `Ex0bit/hrm-demo` — HRM checkpoint
- Community TRM weights: `wtfmahe/Samsung-TRM`

---

## 5. Open questions noticed (flag only — no hypotheses)

While reading the corpus I noticed these gaps. Per role discipline I am not proposing hypotheses, just flagging.

1. **Token-drop interaction.** No paper in this corpus actually evaluates a recursive operator on top of a token-dropping or sparse / sliding-window subquadratic attention beyond PLT (entry 17). PLT uses *gated* sliding-window — but does not analyze which tokens are dropped at each loop and whether late-loop computations need access to dropped tokens.
2. **Long-context evaluation absent.** TRM, HRM, Huginn, Ouro, MoR, LoopFormer all benchmark on short-context reasoning (puzzles, GSM8K, MMLU, BLiMP). None evaluate on multi-doc QA, code-repo understanding, or long-horizon agent traces — which is exactly the regime the spec targets.
3. **Operator's per-step receptive field.** It is unclear in most papers whether the recursive operator can access *new* parts of the long input at later loop steps (e.g. via routing) or only re-attends to the same pool.
4. **Gradient pathway.** Papers split between BPTT (UT, MoEUT, PLT), 1-step approximate gradient (HRM, DEQ-style), and explicit deep supervision (TRM). No head-to-head comparison of these gradient schemes on a single backbone, let alone with subquadratic attention.
5. **Halting × subquadratic interaction.** Per-token halting (UT, MoR, LoopFormer) plus per-token sparsification (subquadratic attention) jointly route at the token level. Whether the two routers must be coupled, or can be learned independently, is not addressed.
6. **Fixed-point fragility under sparse attention.** HRM's mechanistic critique (entry 14) shows the recursive fixed point is fragile even with dense attention; introducing approximation noise from a subquadratic operator may amplify "guessing" behavior. No measurement exists.
7. **Depth-recurrent latent CoT fails interpretability probes (entry 9).** This raises the question of whether depth-recurrent gains are real reasoning gains or a different phenomenon — important to distinguish before claiming long-context reasoning improvement.
8. **Ouro / LoopLM is the only recent open-pretrained looped LM at >1B scale, but with dense attention.** No pretrained looped LM exists with a subquadratic backbone, which is precisely the empirical gap the spec identifies.
9. **Theory ceiling (entry 24).** Approximation-rate ceiling for looped transformers is established without timestep encoding; whether a subquadratic backbone changes that ceiling is open.
10. **"Recursive" word collision.** Three different research lines call themselves "recursive" — architectural (this scout), agent-scaffolded (arXiv:2603.02112, arXiv:2603.20105), and structural-recursion-as-task (arXiv:2401.12947). Anyone reading the literature without this distinction will conflate them; the spec must be explicit.

---

## 6. Sources

All arxiv IDs and HF/GitHub URLs cited above. Each was retrieved through `mcp__plugin_megaresearcher_ml-intern__hf_papers` (search + paper_details).

- arXiv:2510.04871 — TRM — https://arxiv.org/abs/2510.04871 — github.com/SamsungSAILMontreal/TinyRecursiveModels
- arXiv:2506.21734 — HRM — https://arxiv.org/abs/2506.21734
- arXiv:2511.16886 — Deep Improvement Supervision — https://arxiv.org/abs/2511.16886
- arXiv:1807.03819 — Universal Transformers — https://arxiv.org/abs/1807.03819
- arXiv:2301.13196 — Looped Transformers as Programmable Computers — https://arxiv.org/abs/2301.13196
- arXiv:2311.12424 — Looped Transformers Better at Learning Algorithms — https://arxiv.org/abs/2311.12424 — github.com/leiay/looped_transformer
- arXiv:2502.17416 — Reasoning with Latent Thoughts (looped) — https://arxiv.org/abs/2502.17416
- arXiv:2502.05171 — Huginn / Recurrent-Depth Latent Reasoning — https://arxiv.org/abs/2502.05171 — github.com/seal-rg/recurrent-pretraining
- arXiv:2507.02199 — Decoding the Depth-Recurrent Transformer — https://arxiv.org/abs/2507.02199 — github.com/wenquanlu/huginn-latent-cot
- arXiv:2510.25741 — Ouro / LoopLM — https://arxiv.org/abs/2510.25741
- arXiv:2507.10524 — MoR — https://arxiv.org/abs/2507.10524 — github.com/raymin0223/mixture_of_recursions
- arXiv:2410.20672 — Relaxed Recursive Transformers — https://arxiv.org/abs/2410.20672
- arXiv:2512.12880 — MoL for Recursive Transformers — https://arxiv.org/abs/2512.12880
- arXiv:2601.10679 — Mechanistic analysis of HRM — https://arxiv.org/abs/2601.10679
- arXiv:2602.11698 — SpiralFormer — https://arxiv.org/abs/2602.11698
- arXiv:2602.11451 — LoopFormer — https://arxiv.org/abs/2602.11451 — github.com/armenjeddi/loopformer
- arXiv:2510.24824 — Parallel Loop Transformer (PLT) — https://arxiv.org/abs/2510.24824
- arXiv:2602.02156 — LoopViT — https://arxiv.org/abs/2602.02156 — github.com/WenjieShu/LoopViT
- arXiv:2310.07096 — Sparse Universal Transformer — https://arxiv.org/abs/2310.07096 — github.com/shawntan/SUT
- arXiv:2405.16039 — MoEUT — https://arxiv.org/abs/2405.16039 — github.com/robertcsordas/moeut
- arXiv:1909.01377 — Deep Equilibrium Models — https://arxiv.org/abs/1909.01377
- arXiv:1603.08983 — ACT — https://arxiv.org/abs/1603.08983
- arXiv:2107.05407 — PonderNet — https://arxiv.org/abs/2107.05407
- arXiv:2410.01405 — Expressive Power of Looped Transformers — https://arxiv.org/abs/2410.01405
- arXiv:2401.12947 — Structural Recursion in Transformers — https://arxiv.org/abs/2401.12947
- arXiv:2402.12875 — CoT Solves Inherently Serial Problems — https://arxiv.org/abs/2402.12875
- arXiv:2505.01855 — Intra-Layer Recurrence — https://arxiv.org/abs/2505.01855
- arXiv:2603.08391 — Adaptive Loops and Memory — https://arxiv.org/abs/2603.08391
- arXiv:2601.21582 — Depth-Recurrent Attention Mixtures (Dreamer) — https://arxiv.org/abs/2601.21582
- arXiv:2602.07845 — Recurrent-Depth VLA — https://arxiv.org/abs/2602.07845
- arXiv:2603.21676 — Thinking Deeper Not Longer — https://arxiv.org/abs/2603.21676
- arXiv:2507.06203 — Survey on Latent Reasoning — https://arxiv.org/abs/2507.06203 — github.com/multimodal-art-projection/LatentCoT-Horizon
- arXiv:2510.14961 — Efficient Parallel Samplers for Recurrent-Depth Models — https://arxiv.org/abs/2510.14961
- arXiv:2204.03276 — PALBERT — https://arxiv.org/abs/2204.03276 — github.com/tinkoff-ai/palbert
- arXiv:2111.05297 — Sliced Recursive Transformer — https://arxiv.org/abs/2111.05297

Out-of-scope-but-flagged (architectural-vs-scaffolded confusion):
- arXiv:2603.02112 — Recursive Models for Long-Horizon Reasoning — agent-scaffolded recursion, **flagged out of scope**
- arXiv:2603.20105 — Y-Combinator for LLMs — λ-calculus-as-prompting, **flagged out of scope**
