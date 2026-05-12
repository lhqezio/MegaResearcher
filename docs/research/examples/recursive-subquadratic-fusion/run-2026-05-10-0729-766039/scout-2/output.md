# Scout-2 — Subquadratic / Sparse-Attention Transformers (Annotated Bibliography)

## 1. Scope

Sub-topic: **transformer attention algorithms with provably or empirically sub-quadratic compute in sequence length, treated as a *primitive*** — i.e., we record sparsity pattern, scaling, and what information is lost or compressed, but do NOT inspect CUDA kernels, FlashAttention internals, or memory-IO tricks.

Narrowing decisions:
- Linear / kernel approximations (Performer, Linformer) included only as prior-art reference, not as primary fusion candidates, since the spec frames "sparse attention dropping tokens" as the load-bearing failure mode for recursion.
- KV-cache eviction methods (H2O, Quest) included because they *implement* a sparsity decision at inference time, even though they keep the underlying full attention math during prefill.
- Mamba / SSMs noted only as a contrastive baseline (sub-quadratic but *not* attention).
- "What is lost" annotation prioritised — recursion can only refine what attention preserves, so this is the load-bearing axis for the fusion thesis.

## 2. Key Papers

### 2A. Industrial / production claim (flagged: industrial blog, no peer-reviewed paper)

**SubQ — "Introducing SubQ"** (Subquadratic Inc.; 2026-05-05; URL: https://subq.ai/introducing-subq).
*Industrial blog, no peer-reviewed paper, no preprint.*
Claims a "fully subquadratic architecture" where attention compute grows linearly with context length; "52× faster than FlashAttention" with "63% less compute"; "~1000× attention compute reduction vs. frontier models"; demonstrates a 12 M-token research context window. Published benchmark numbers: RULER-128K 95.0 (vs. Claude Opus 4.6 94.8), MRCR v2 65.9 (vs. Opus 4.7 32.2 / GPT 5.5 74 / Gemini 3.1 Pro 26.3), SWE-Bench Verified 81.8.
**Sparsity pattern:** "sparse attention" — pattern not disclosed in the blog. **Scaling:** linear in n. **Empirical reasoning:** strong on RULER (multi-hop retrieval) and SWE-Bench (code reasoning); MRCR is a multi-needle retrieval benchmark.
**What is lost:** un-stated. The blog does not characterise which token-pair interactions are dropped, nor whether the sparsity is data-dependent. *This is the central fusion-thesis blind spot for SubQ.*
Why it matters: the spec explicitly names SubQ as a fusion-thesis target and demands every claim be flagged as industrial. All future scouts and gap-finders should treat the 12M / 52×/ 1000× numbers as advertised, not verified.

### 2B. Theoretical foundations of subquadratic attention

**Subquadratic Algorithms and Hardness for Attention with Any Temperature** — Gupta, Huang, Saha, Xu, Ye (arXiv:2505.14840; May 2025).
*Could not be retrieved via hf_papers (no HF mirror); arxiv abstract fetched directly. Flagged but retained because it is the spec's named anchor.*
Algorithm: Õ(n^(2−1/d) · polylog B) for constant head dimension d = O(1), where B bounds entry magnitude. Removes the bounded-entry restriction of Alman-Song. Hardness: for d = 2^(Θ(log* n)), attention requires n^(2−o(1)) time under SETH; for d = poly(n), the standard quadratic algorithm is optimal.
**Sparsity pattern:** none — this is a *dense* attention algorithm that achieves subquadratic time via algebraic structure (low-rank decomposition + polynomial method). **Scaling:** Õ(n^(2−1/d)). For d = 1 → Õ(n), d = 2 → Õ(n^1.5), d = 4 → Õ(n^1.75).
**What is lost:** ε additive approximation (1/poly(n) error); *no tokens are dropped.* This is the formal regime in which attention can be made subquadratic without sparsifying — useful upper bound for the spec to delineate "principled subquadratic" from "heuristic sparse drop".
Why it matters: the spec quotes this paper as the theoretical regime characterisation. Any fusion-thesis claim that "subquadratic = drops tokens recursion needs" must reckon with the fact that *some* subquadratic regimes (low constant d, bounded entries) preserve the full attention output up to ε.

**Fast Attention Requires Bounded Entries** — Alman, Song (arXiv:2302.13214; NeurIPS 2023).
*Not on HF Papers; verified via arxiv.org.*
Sharp threshold at B = Θ(√log n): for d = O(log n) and B = o(√log n), an n^(1+o(1)) algorithm achieves 1/poly(n) additive error; at B = Θ(√log n) and assuming SETH, no truly subquadratic algorithm exists.
**Sparsity pattern:** none (theoretical). **Scaling:** n^(1+o(1)) in the bounded regime, n^(2−o(1)) lower bound otherwise.
**What is lost:** the bound on entry magnitude is the load-bearing assumption — softmax temperature implicitly controls B. Larger temperatures → larger B → forced into quadratic regime. *Recursive reasoning that sharpens attention distributions (lower-temperature ensembles) may push the system across this hardness boundary.*
Why it matters: canonical theoretical reference for "why subquadratic works at all". Establishes that the SubQ-style claim is feasible only because LLM softmax temperatures keep entries bounded.

**The Fine-Grained Complexity of Gradient Computation for Training Large Language Models** — Alman, Song (arXiv:2402.04497; NeurIPS 2024).
Extends the 2302.13214 dichotomy from forward to backward pass: gradient computation for attention is almost-linear in the bounded-entry regime, conditionally quadratic-required (under SETH) outside.
**What is lost:** same boundary as forward; no extra information lost. **Why it matters for the spec:** if a fusion architecture *trains* a recursive head over a subquadratic attention substrate, the gradient cost of the substrate scales the same way as inference. Bounds the joint training cost of (TRM-style recursion) ⊕ (subquadratic attention).

### 2C. Classic sparse-attention prior art (canonical references)

**Generating Long Sequences with Sparse Transformers** — Child, Gray, Radford, Sutskever (arXiv:1904.10509; 2019).
**Sparsity pattern:** factorised fixed sparse patterns (strided + local). **Scaling:** O(n√n). **What is lost:** any token pair not in the union of the two factor strides; the assumption is that two-hop reachability through the strided layer recovers what's missing — an *implicit* recursive-reasoning argument decades before TRM.
Why it matters: the strided + local factorisation is the philosophical ancestor of every fixed-pattern sparse method. Its two-hop-reachability argument is the strongest pre-existing analogue to "use depth to recover what sparsity drops".

**Longformer: The Long-Document Transformer** — Beltagy, Peters, Cohan (arXiv:2004.05150; 2020).
**Sparsity pattern:** sliding window (local) + dilated window + task-specific global tokens (e.g., [CLS]). **Scaling:** O(n · w) where w is window size; effectively O(n) for fixed w. **What is lost:** mid-range token-pair interactions outside the window; *non-global, non-local long-range dependencies are entirely dropped.* Compensated by stacking layers (depth-grows-receptive-field argument).
Why it matters: still the dominant fixed-pattern baseline in the literature. Its "depth recovers receptive field" argument is the same shape as the spec's "recursion recovers what attention drops" question.

**Big Bird: Transformers for Longer Sequences** — Zaheer et al. (arXiv:2007.14062; 2020).
**Sparsity pattern:** random + window + global tokens. Proven universal-approximator and Turing-complete. **Scaling:** O(n) with constant random-attention sample count. **What is lost:** any interaction missed by the random sample — bounded-error in expectation, but tail interactions can vanish. Universal-approximation proof requires Ω(n) total non-zeros, asymptotically.
Why it matters: the formal universal-approximation theorem is the closest classical result to "sparse attention preserves enough to support arbitrary computation, given enough depth". Directly addresses the spec's coherence-boundary question.

**Reformer: The Efficient Transformer** — Kitaev, Kaiser, Levskaya (arXiv:2001.04451; 2020).
**Sparsity pattern:** locality-sensitive hashing — tokens routed to the same hash bucket attend, others do not. Data-dependent. **Scaling:** O(n log n). **What is lost:** any pair whose hashes diverge despite semantic similarity; multiple-round hashing and chunking partly mitigate. *First major data-dependent sparsity scheme.*
Why it matters: prototype for "the sparsity pattern depends on the content". Recursion on top of LSH-attention is a degenerate case the fusion thesis must handle.

### 2D. 2024–2026 trainable / native sparse attention

**Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (NSA)** — Yuan et al., DeepSeek (arXiv:2502.11089; Feb 2025).
**Sparsity pattern:** dynamic hierarchical — three branches per query: (i) compressed coarse-block summary, (ii) selected fine-grained blocks, (iii) sliding-window local. End-to-end trained. **Scaling:** O(n · k) for top-k block selection with k constant. **What is lost:** non-selected blocks contribute *only* through their compressed summary; query-specific fine-grained interactions are limited to top-k blocks. Trainability mitigates the "wrong block selected" failure mode dense-trained-then-sparse-tested incurs.
Why it matters: NSA is the first production-grade *natively trainable* sparse attention — the closest open-weights analogue to what SubQ is doing. Direct fusion candidate for TRM-style recursion.

**MoBA: Mixture of Block Attention for Long-Context LLMs** — Lu et al., Moonshot AI (arXiv:2502.13189; Feb 2025).
**Sparsity pattern:** MoE-style — each query routes to top-k blocks via learned gating; gradient flows through router. Can fall back to dense attention when needed. **Scaling:** O(n · k) for top-k blocks. **What is lost:** un-selected blocks contribute zero (no compressed summary, unlike NSA). Tail-routing errors compound across layers.
Why it matters: contrasts with NSA — same compute budget, no compression channel. Sharp test for the fusion thesis: if MoBA's routing drops a token recursion needs, there's no fallback path; whereas NSA's coarse channel may give recursion a starting point.

**DeepSeek-V3.2 — DeepSeek Sparse Attention (DSA)** — DeepSeek-AI (arXiv:2512.02556; Dec 2025).
**Sparsity pattern:** lightning indexer scores every historical token, computes attention only over top-k. Token-level (not block-level) sparsity. **Scaling:** core attention O(L · k); but the indexer itself is O(L²). **What is lost:** non-top-k tokens are entirely dropped from the value-mixing step (the indexer still saw them).
Why it matters: the most recent open-weights production sparse-attention model with strong reasoning numbers (claims gold-medal IMO/IOI). Demonstrates that the "drop everything outside top-k" pattern can support reasoning at the SOTA frontier. Directly comparable to TRM-style recursive refinement.

**HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention** — Xu et al. (arXiv:2603.28458; 2026).
Addresses the O(L²) indexer bottleneck in DSA: hierarchical block-then-token index reduces selection to subquadratic. **Sparsity pattern:** two-stage (block filter → token refine). **Scaling:** sub-quadratic for indexer; core attention same as DSA. **What is lost:** blocks pruned in stage 1 cannot recover at stage 2 — irreversible coarse-grained dropping.
Why it matters: shows the practical engineering boundary — once you make the indexer subquadratic too, you've introduced a *second* layer of dropped tokens. Doubly-relevant to the fusion thesis: recursion has to refine across *two* sparsity hierarchies.

**SeerAttention-R: Sparse Attention Adaptation for Long Reasoning** — Gao et al. (arXiv:2506.08889; June 2025).
Self-distilled gating tailored for autoregressive decoding of *reasoning* models. **Sparsity pattern:** learned block selection (no query pooling, vs. SeerAttention v1 which used it). **Scaling:** O(n · k). **What is lost:** non-selected blocks dropped at decode; the gating is trained to preserve reasoning-critical dependencies via self-distillation from a dense teacher.
Why it matters: *first paper to explicitly target sparse attention adaptation for long-chain reasoning*. The closest existing analogue to "sparse attention that protects what reasoning needs". The self-distillation idea is a direct precursor to recursion-aware sparsity training.

**Every Token Counts: Generalizing 16M Ultra-Long Context (HSA)** — Hu et al. (arXiv:2511.23319; Nov 2025).
**Sparsity pattern:** Hierarchical Sparse Attention — three properties claimed: sparsity, random-access flexibility, length generalization. 8B MoE model. **Scaling:** sub-quadratic (hierarchical block-tree). **What is lost:** off-tree tokens are dropped; random-access flexibility is the explicit countermeasure to the "fixed-pattern misses what you need" failure.
Why it matters: pushes context to 16M and explicitly engineers for *random access* — the property recursion most needs (referring back to arbitrary earlier tokens).

### 2E. Inference-time sparsity / KV-cache compression

**Efficient Streaming Language Models with Attention Sinks (StreamingLLM)** — Xiao, Tian, Chen, Han, Lewis (arXiv:2309.17453; Sept 2023).
**Sparsity pattern:** sliding window + first-k "sink" tokens. Sink tokens preserved permanently; everything between sink and window is *evicted*. **Scaling:** O(n · w) for window size w; constant-memory at inference. **What is lost:** all token-key/value information outside the window∪sink. The "attention sink" finding shows that without the sink, fluency collapses entirely.
Why it matters: demonstrates that *some* tokens are load-bearing for the model in ways the model's gradient learned but the architecture didn't make explicit. Recursion in this regime can only refer back to window∪sink.

**Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference** — Tang et al. (arXiv:2406.10774; June 2024).
**Sparsity pattern:** query-aware top-k page selection from KV cache. Pages = blocks of contiguous tokens. **Scaling:** O(n) preselect + O(k · d) attention. **What is lost:** non-top-k pages dropped per-query; the selection signal is min/max over keys-in-page. Cross-query reuse is limited.
Why it matters: the sparse-frontier benchmark (2504.17768) names Quest as the strongest "robust decoding" baseline. Direct candidate for the inference-side sparsity primitive in a fusion architecture.

**H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs** — Zhang et al. (arXiv:2306.14048; June 2023).
**Sparsity pattern:** keep recent + heavy-hitter (high cumulative attention score) tokens, evict the rest. **Scaling:** O(n) effective with constant cache budget. **What is lost:** evicted-then-needed tokens are unrecoverable. Heavy-hitter heuristic can mis-identify under distribution shift.
Why it matters: foundational eviction baseline. The "heavy hitter" structure is exactly the kind of attention pattern recursion would also produce — interaction worth studying.

### 2F. Linear / low-rank approximations (prior art only)

**Performer: Rethinking Attention with Performers** — Choromanski et al. (arXiv:2009.14794; 2020).
**Sparsity pattern:** none — kernel approximation via random features (FAVOR+). Computes a *dense* low-rank approximation. **Scaling:** O(n · r) for random-feature dimension r. **What is lost:** estimation noise; tail interactions noisier than dominant ones. *No tokens are dropped.*
Why it matters: contrastive prior art — recursion atop kernelised attention is a fundamentally different fusion regime than recursion atop sparse-drop. Useful for the spec's "where does fusion become incoherent" question: probably not here, since nothing is dropped.

**Linformer: Self-Attention with Linear Complexity** — Wang et al. (arXiv:2006.04768; 2020).
**Sparsity pattern:** none — projects K and V to a fixed lower dimension k. **Scaling:** O(n · k). **What is lost:** rank-truncation error; fixed-projection cannot adapt to query-specific information needs (a known failure mode for retrieval tasks).
Why it matters: same as Performer — contrast case for the fusion thesis. Linformer's *fixed* projection is the closest classical analogue to a *non-data-dependent* subquadratic primitive; recursion atop it has different incoherence boundaries than recursion atop top-k.

**LongNet: Scaling Transformers to 1,000,000,000 Tokens** — Ding et al. (arXiv:2307.02486; July 2023).
**Sparsity pattern:** dilated attention with exponentially expanding fields across heads. **Scaling:** linear in n. **What is lost:** any pair outside the union of dilated fields; coverage approaches full only by mixing many heads.
Why it matters: represents the "geometric coverage" branch of sparse attention. Co-occurs with the spec's lineage but is different in spirit from data-dependent top-k methods.

### 2G. Empirical analyses — what sparse attention loses for reasoning

**The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs** — Nawrot, Li, Huang, Ruder, Marchisio, Ponti (arXiv:2504.17768; April 2025).
The largest-scale isoFLOPs comparison of training-free sparse-attention methods to date (Qwen 2.5, Llama 3.1, Gemma 3; 4B–72B; 16K–128K; sparsity ≤ 0.95). Three findings: (1) sparse attention enables larger sparse models to beat smaller dense ones at equal FLOPs; (2) method choice is task-dependent — fine-grained Vertical-Slash for retrieval, Block-Sparse for reasoning/aggregation, Quest robust at decode; (3) longer sequences tolerate higher sparsity; fixed-budget methods are sub-optimal.
**What is lost (meta-finding):** the mapping from sparsity-pattern → task-degradation is non-uniform; reasoning tasks degrade differently from retrieval tasks. *Direct evidence that "what sparse attention drops" matters per-task.* Critical for the fusion thesis's "where it becomes incoherent" question.

**NoLiMa: Long-Context Evaluation Beyond Literal Matching** — Modarressi et al. (arXiv:2502.05167; Feb 2025).
Benchmark removes literal-match exploit from needle-in-haystack tests, forcing models to use the underlying attention mechanism for fact chaining and in-context reasoning. *Performance degrades sharply with context length even for state-of-the-art 128K–1M models.*
**What is lost:** the implicit — non-literal — attention pathways are the first to fail under length stress. Sparse attention that drops tokens with no literal match risks compounding this.
Why it matters: the spec's coherence-boundary hypothesis ("recursion needs to keep referring back to tokens sparse attention drops") needs an evaluation harness — NoLiMa is the canonical one for chained reasoning.

**Retrieval Head Mechanistically Explains Long-Context Factuality** — Wu, Wang, Xiao, Peng, Fu (arXiv:2404.15574; April 2024).
Identifies a small set of "retrieval heads" — sparse, intrinsic, dynamically-activated, causal — responsible for arbitrary-position fact retrieval. Disabling these heads degrades reasoning and increases hallucination.
**What is lost:** any sparse-attention pattern that drops a retrieval head's preferred tokens *for the query that would have used them* destroys long-context factuality. *This is the most precise mechanistic statement of the fusion-thesis failure mode.*
Why it matters: gives a concrete unit ("retrieval head") to argue about when the fusion thesis fails. A recursion-on-sparse system must verify retrieval heads still fire correctly through the sparse substrate.

**Hyper-multi-step: The Truth Behind Difficult Long-context Tasks** — Yu (arXiv:2410.04422; Oct 2024).
Decomposes long-context difficulty into *multi-matching retrieval* and *logic-based retrieval* — both manifestly multi-step in the underlying attention computation.
**What is lost:** sparse-attention single-pass loses the multi-step refinement needed for these tasks. Direct argument for *why* recursion would help.
Why it matters: independent argument for the spec's hypothesis — long-context hard tasks are "hyper-multi-step", which is exactly what TRM-style recursion targets.

**SCBench: A KV Cache-Centric Analysis of Long-Context Methods** — Li et al. (arXiv:2412.10319; Dec 2024).
Multi-request lifecycle benchmark: cache generation, compression, retrieval, loading. Finds sparse-attention KV-compression methods are *more robust under cache reuse* than gated linear RNNs and Mamba-attention hybrids on multi-turn tasks.
**What is lost (with reuse):** distribution shift across turns — the sparsity pattern that was "right" for turn 1 may drop a turn-3 needle.
Why it matters: extends the fusion thesis from single-shot to multi-turn. Recursion + sparse attention may be *more* coherent across turns than dense Mamba alternatives.

### 2H. Contrast: the non-attention subquadratic baseline

**Mamba: Linear-Time Sequence Modeling with Selective State Spaces** — Gu, Dao (arXiv:2312.00752; Dec 2023).
**Sparsity pattern:** none — recurrent state-space model with input-dependent selection. **Scaling:** O(n). **What is lost:** information squeezed through fixed-dimension hidden state; long-range exact retrieval degrades vs. attention.
Why it matters: the strongest non-attention sub-quadratic baseline. The fusion thesis needs to explain *why* sparse-attention + recursion is preferable to Mamba (which is already sub-quadratic and has explicit recurrence). Useful for spec's "where fusion uniquely outperforms" axis.

## 3. Datasets

All open-licensed unless noted. License flags as best ascertainable from HF dataset cards.

| Dataset | HF / Source | License | Use |
|---|---|---|---|
| RULER (NIAH + multi-hop) | https://github.com/hsiehjackson/ruler | Apache-2.0 (repo) | Long-context retrieval & multi-hop tracing — the headline benchmark in SubQ's claims. |
| LongBench v2 | https://huggingface.co/datasets/THUDM/LongBench-v2 | MIT (per repo) | Long-context multi-task; multi-doc QA, code, in-context learning. |
| NoLiMa | https://github.com/adobe-research/NoLiMa | per repo (Adobe Research; check before redistribution) | Long-context fact-chaining without literal-match leakage. |
| Needle Threading | (arXiv:2411.05000) | per repo | Near-million-scale haystack reasoning. |
| ImpliRet | https://github.com/zeinabtaghavi/impliret | per repo | Implicit-fact retrieval (temporal + arithmetic). |
| SCBench | (arXiv:2412.10319) | per repo | KV-cache lifecycle multi-turn. |
| 100-LongBench | (arXiv:2505.19293) | per repo | Length-controllable disentanglement of long-context skill. |

License flag: NoLiMa's distribution terms should be checked on the Adobe repo card before any derivative use.

## 4. Reference Implementations

| Method | Repo | Stars | Paper |
|---|---|---|---|
| Native Sparse Attention (NSA) reference | https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention | 615 | 2508.18224 (FSA implementation of NSA) |
| MoBA | https://github.com/moonshotai/moba | 2113 | 2502.13189 |
| FlashMoBA | https://github.com/mit-han-lab/flash-moba | 248 | 2511.11571 |
| MoA | https://github.com/thu-nics/MoA | 157 | 2406.14909 |
| Quest | https://github.com/mit-han-lab/quest | 384 | 2406.10774 |
| H2O | https://github.com/FMInference/H2O | 517 | 2306.14048 |
| SeerAttention | https://github.com/microsoft/seerattention | 203 | 2410.13276 |
| Sparse-Frontier benchmark suite | https://github.com/PiotrNawrot/sparse-frontier | 122 | 2504.17768 |
| Retrieval-head probe | https://github.com/nightdessert/retrieval_head | 238 | 2404.15574 |
| RULER benchmark | https://github.com/hsiehjackson/ruler | 1532 | 2404.06654 |
| NoLiMa benchmark | https://github.com/adobe-research/NoLiMa | 193 | 2502.05167 |
| IndexCache (DSA acceleration) | https://github.com/THUDM/IndexCache | 97 | 2603.12201 |
| Mamba reference | https://github.com/state-spaces/mamba | 18207 | 2312.00752 |
| TRM (the spec's recursion anchor) | https://github.com/SamsungSAILMontreal/TinyRecursiveModels | 6496 | 2510.04871 |

## 5. Open Questions Flagged (no hypothesis-proposing — just gaps observed)

1. **No public technical disclosure of SubQ's sparsity pattern.** All "what is lost" reasoning about SubQ is unverifiable from the blog. (Spec already flags this as industrial; recording for downstream awareness.)
2. **No paper directly studies recursion-style depth applied to *natively trainable* sparse-attention substrates.** TRM operates on dense small models; NSA/MoBA/DSA papers don't analyse what happens when a recursive head is layered on top.
3. **Retrieval-head behaviour under sparse attention is uncharacterised.** Wu et al. (2404.15574) identifies retrieval heads in dense models; whether NSA/DSA preserve, displace, or destroy retrieval-head structure is open.
4. **The Gupta et al. (2505.14840) Õ(n^(2−1/d)) regime is *dense* subquadratic — no published architecture instantiates it.** All practical subquadratic LLMs use sparsity instead. The fusion thesis may have a different shape in the dense-subquadratic regime than in the sparse regime.
5. **Sparse-Frontier (2504.17768) found "longer sequences tolerate higher sparsity"** — but no analysis of whether longer *reasoning chains* tolerate higher sparsity. Reasoning length is independent of context length and may invert the trend.
6. **MoBA vs. NSA contrast** — MoBA has no compressed-summary fallback for un-routed blocks; NSA does. No direct study of how this asymmetry affects recursive refinement.
7. **The Alman-Song bounded-entry boundary moves with softmax temperature** — but recursive reasoning models often use sharpened (low-temperature) attention. Whether deep recursion pushes a system across the SETH boundary is open.
8. **No benchmark separates "context-length stress" from "reasoning-depth stress."** RULER, NoLiMa, NeedleThreading scale context; AIME / ARC-AGI / Sudoku scale depth. The fusion thesis's two-axis claim has no canonical evaluation surface.
9. **DSA's indexer is itself O(L²)** (HISA paper, 2603.28458) — so production "sparse attention" pipelines have a hidden quadratic step. Any fusion-thesis claim that the substrate is sub-quadratic should cite which indexer.
10. **SCBench (2412.10319) finds sparse attention more robust than Mamba hybrids on multi-turn** — but no recursion-on-mamba-vs-recursion-on-sparse comparison exists.

## 6. Sources (flat list of every URL / arxiv ID cited)

- https://subq.ai/introducing-subq (industrial blog, no peer-reviewed paper)
- arXiv:2505.14840 (Gupta, Huang, Saha, Xu, Ye — Subquadratic Algorithms and Hardness)
- arXiv:2302.13214 (Alman, Song — Fast Attention Requires Bounded Entries)
- arXiv:2402.04497 (Alman, Song — Fine-Grained Complexity of Gradient Computation)
- arXiv:1904.10509 (Child et al. — Sparse Transformer)
- arXiv:2004.05150 (Beltagy et al. — Longformer)
- arXiv:2007.14062 (Zaheer et al. — BigBird)
- arXiv:2001.04451 (Kitaev et al. — Reformer)
- arXiv:2502.11089 (Yuan et al. — NSA)
- arXiv:2502.13189 (Lu et al. — MoBA)
- arXiv:2512.02556 (DeepSeek-AI — DeepSeek-V3.2 / DSA)
- arXiv:2603.28458 (Xu et al. — HISA)
- arXiv:2603.12201 (Bai et al. — IndexCache)
- arXiv:2506.08889 (Gao et al. — SeerAttention-R)
- arXiv:2410.13276 (Gao et al. — SeerAttention)
- arXiv:2511.23319 (Hu et al. — HSA / Every Token Counts 16M)
- arXiv:2510.24606 (Xiong et al. — DHSA on-device)
- arXiv:2604.07394 (Qiu et al. — Flux Attention hybrid)
- arXiv:2309.17453 (Xiao et al. — StreamingLLM)
- arXiv:2406.10774 (Tang et al. — Quest)
- arXiv:2306.14048 (Zhang et al. — H2O)
- arXiv:2009.14794 (Choromanski et al. — Performer)
- arXiv:2006.04768 (Wang et al. — Linformer)
- arXiv:2307.02486 (Ding et al. — LongNet)
- arXiv:2504.17768 (Nawrot et al. — Sparse Frontier)
- arXiv:2502.05167 (Modarressi et al. — NoLiMa)
- arXiv:2404.15574 (Wu et al. — Retrieval Head)
- arXiv:2410.04422 (Yu — Hyper-multi-step)
- arXiv:2412.10319 (Li et al. — SCBench)
- arXiv:2404.06654 (Hsieh et al. — RULER)
- arXiv:2312.00752 (Gu, Dao — Mamba)
- arXiv:2510.04871 (Jolicoeur-Martineau — TRM, spec anchor)
- GitHub: hsiehjackson/ruler · adobe-research/NoLiMa · moonshotai/moba · mit-han-lab/quest · FMInference/H2O · microsoft/seerattention · PiotrNawrot/sparse-frontier · nightdessert/retrieval_head · THUDM/IndexCache · state-spaces/mamba · SamsungSAILMontreal/TinyRecursiveModels · Relaxed-System-Lab/Flash-Sparse-Attention · mit-han-lab/flash-moba · thu-nics/MoA
