# Hypothesis H3 — Retrieval-head re-formation under sparse attention × recursion is conditional on the existence of an all-tokens fallback channel

## 1. Targeted gap

This hypothesis addresses **Gap 9 from gap-finder-2** (`/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/gap-finder-2/output.md#9`), which is the same intersection captured by **Gap 8 in gap-finder-1** (`/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/gap-finder-1/output.md#41` ff., the "Retrieval-head behavior under sparse attention is uncharacterized" gap). Compositional anchor: A8 (retrieval-head structure) × B9 (sparse attention × recursion-depth instrument).

Restated: Retrieval Head (arXiv:2404.15574) shows a small, sparse, intrinsic set of attention heads is mechanistically responsible for arbitrary-position fact retrieval and that disabling them collapses NIAH and degrades CoT (arXiv:2404.15574, §3.1, §4.1, §4.3). Retrieval-Aware Distillation for Transformer-SSM Hybrids (arXiv:2602.11374) confirms this structure is fragile under naive sub-quadratic conversion — its "Gather-and-Aggregate" heads must be explicitly preserved when distilling Transformers into SSMs. RazorAttention (arXiv:2407.15891) further confirms retrieval-head specialness by exploiting it for KV compression. NoLiMa (arXiv:2502.05167; non-commercial license — flagged) and Hyper-multi-step (arXiv:2410.04422) extend the picture to multi-step retrieval. Sparse Frontier (arXiv:2504.17768) ablates sparse-attention patterns on retrieval/reasoning workloads but does not measure retrieval-head retention; SeerAttention-R (arXiv:2506.08889) self-distills sparse gating for reasoning models but does not test retrieval-head behavior across recursive loop iterations. No paper has measured retrieval-head retention as a function of (sparse-attention pattern × architectural-recursion depth K).

## 2. Hypothesis statement

**If** a TRM-style architectural recursion (K passes within a forward pass, refining a latent answer state and inputs as in arXiv:2510.04871) is layered on a transformer backbone whose attention has been replaced with NSA (arXiv:2502.11089, which retains a compressed-summary branch over all blocks plus a sliding-window branch in addition to top-k selection), **then** retrieval-head retention score (per the copy-paste detection method of arXiv:2404.15574, §2) measured at the deepest recursion pass K=K\* will be ≥ 0.85× the dense-attention retention baseline at K=1, **and** the recovery will be monotonic in K (retention at K=4 > retention at K=2 > retention at K=1 by ≥ 0.05 absolute, at fixed parameter count and fixed sparsity ratio). **In contrast**, on a MoBA backbone (arXiv:2502.13189) at the same sparsity ratio, retrieval-head retention will *not* recover with K (Δretention from K=1 to K=4 < 0.02 absolute), because MoBA has no all-tokens fallback channel — non-selected blocks contribute exactly zero (g_i=0 for blocks outside top-k, arXiv:2502.13189, §2.2). The differential effect (NSA-with-recursion minus MoBA-with-recursion) at K=4 will exceed the differential at K=1 by ≥ 0.10 absolute retrieval-head retention.

This is a non-additive, conditional prediction: the *interaction* between recursion and sparsity pattern is the load-bearing claim, not either factor alone.

## 3. Mechanism

The mechanism rests on three cited claims, none speculative.

**Claim M1 — retrieval heads are a copy-paste circuit identifiable by attention-mass concentration on input-position needles.** arXiv:2404.15574 §2 defines the retrieval score as the frequency with which a head's argmax attention probability lands on the needle token whose value is then emitted; the threshold for "retrieval head" is 0.1 frequency, and the same heads appear universally across model families (Llama-2, Mistral, Mixtral, Yi, Qwen — arXiv:2404.15574, Table 1, §3.1). The instrument transfers across architectures because it operates at the level of attention probability assignment, not on a specific architectural prior.

**Claim M2 — NSA's compression branch preserves a coarse-grained signal over *all* blocks, including blocks not selected by top-k.** arXiv:2502.11089 §2.3 and Figure 2 specify three parallel branches: (a) compressed attention over coarse-pooled blocks (every block contributes), (b) selected attention over top-k blocks, (c) sliding attention over a local window. All three are concatenated/gated into the head output. Therefore information about an arbitrary needle position is *never zeroed* at the head input — it survives at compressed-block resolution even when not selected. arXiv:2502.11089 §2.2 explicitly cites this concern, noting that post-hoc top-k pruning leaves "retrieval heads in pretrained models vulnerable to pruning during inference," and the native compression branch is one of the design responses. MoBA (arXiv:2502.13189, §2.2 Eq. 5) defines g_i = 0 for blocks outside top-k, with no fallback channel: information from un-routed blocks is structurally zero.

**Claim M3 — recursive depth refines a latent state by re-querying the substrate K times, allowing later passes to issue queries that depend on intermediate results from earlier passes.** TRM (arXiv:2510.04871, §3) recurses a small two-layer block over a latent answer state z and inputs x for K iterations, with each pass refining z based on the previous pass; the same block is re-applied with the same parameters but different state. Because z is updated between passes, the *query distribution at pass k+1 is different from pass k*, so different blocks may be top-k selected at different passes. This iteration is what gives TRM its expressivity advantage on Sudoku/Maze/ARC-AGI (arXiv:2510.04871, §4) — not extra parameters, but a fresh query at each pass.

**Composition (M1+M2+M3 → predicted outcome).** Under NSA + recursion: at pass 1, top-k may miss the block containing the needle, but the compression branch (M2) still passes a coarse-grained signal about that block to the head. The head output, which contributes to the residual stream feeding the latent state z, therefore carries weak-but-nonzero information about the needle. At pass 2, the refined z (M3) issues a new query whose distribution is conditioned on what was retrieved at pass 1 — including the weak compressed-branch signal. With a sharper query, the top-k selection at pass 2 may now include the needle block. The retrieval-head copy-paste behavior (M1) thus has an **information path** to re-form across passes: the compression branch carries enough signal for the recursion to *reconstruct* the retrieval target on a later pass. Under MoBA + recursion: if the needle block is missed at pass 1, no information about that block reaches the residual stream (M2 fails — g_i=0). The refined z at pass 2 cannot be conditioned on what pass 1 didn't see; the query refinement is blind to the missed block. Recursion has **no information path to access the dropped tokens**, and retention cannot recover.

This satisfies the architectural-coherence flag from the spec: the mechanism does *not* propose that recursion attends to fully dropped tokens. It specifies the path explicitly — the NSA compression branch is the load-bearing fallback channel; MoBA has none.

A subsidiary mechanism claim, weaker but cited: arXiv:2602.11374 ("Gather-and-Aggregate" heads in Transformer→SSM distillation) provides direct evidence that retrieval-head function is preserved only when *some* explicit channel for arbitrary-position access is retained; their distillation ablations show that hybrid models without preserved G&A heads collapse on retrieval tasks, paralleling the predicted MoBA failure mode.

## 4. Predicted outcome with magnitude

**Primary metric.** Retrieval-head retention score, measured per arXiv:2404.15574 §2 (copy-paste frequency with threshold 0.1), computed on the dense-attention pretrained reference and on the sparse-attention variant at each recursion pass k ∈ {1,…,K}. We define **retention(k)** = (number of heads with retrieval score ≥ 0.1 at pass k that were retrieval heads in the dense baseline) / (number of dense-baseline retrieval heads).

**Conditions where the hypothesis must hold:**
- Backbone: a transformer in the 1-3B parameter range with a known retrieval-head set (e.g., Llama-2-7B-class family per arXiv:2404.15574 Table 1; or a smaller model with retrieval heads identified by the same procedure).
- Sparsity: ratio held constant across NSA and MoBA at ≈ 0.10 (similar to NSA's reported configuration in arXiv:2502.11089).
- Context length: 32K–64K (within NSA/MoBA reported regimes).
- Recursion: TRM-style outer-loop K∈{1,2,4,8} on the latent answer state, parameters tied across passes, no extra training of the recursion (zero-shot recursion or minimal adapter — YAGNI fence).
- Probe: NoLiMa (arXiv:2502.05167; license-flagged for non-commercial use) for non-literal-match retrieval, and Beyond the Needle's Illusion / EverMemBench-S (arXiv:2601.20276) for adversarial NIAH at decoupled access vs. use.

**Predicted magnitudes:**
- NSA + K=1: retention ≈ 0.55–0.75 of dense baseline (sparsity damages retrieval-head function on first pass; consistent with the "70% coverage by top-20% attention" datapoint in arXiv:2502.11089 §2.2 citing Chen et al. 2024b).
- NSA + K=4: retention ≥ 0.85 of dense baseline (≥ 0.10 absolute recovery from K=1 to K=4).
- MoBA + K=1: retention ≈ 0.45–0.65 of dense baseline (similar or slightly worse than NSA, since fewer information channels survive).
- MoBA + K=4: retention within ±0.02 of MoBA + K=1 (no recovery).
- DSA + K=4 (per arXiv:2512.02556's lightning-indexer top-k routing without compression branch): retention within ±0.03 of DSA + K=1 (no recovery; MoBA-class behavior).
- Quest + K=4 (per arXiv:2406.10774's query-aware page-level top-k): retention recovery 0.02–0.05 absolute (intermediate; query-aware selection benefits more from refined queries than block-static MoBA, but the absence of an all-tokens compression channel still bounds recovery).

**Conditions under which the hypothesis must NOT hold (negative predictions):**
- If sparsity ratio is set to 1.0 (full attention) on NSA: K should not improve retention (the compression branch is redundant; no information was lost on pass 1).
- If the retrieval task is *single-step retrieval at distance < sliding-window size*, NSA's sliding branch already solves it on pass 1 and K should not help.
- On a non-retrieval task (e.g., simple language-modeling perplexity on in-distribution short context), neither sparsity-pattern × K interaction should appear; retention is undefined.

## 5. Falsification criteria

We require **three independent falsification paths**, each with a metric, threshold, and direction.

**F1 (primary, retention monotonicity).** *Metric:* retrieval-head retention score per arXiv:2404.15574 §2. *Threshold:* On NSA, if retention(K=4) − retention(K=1) < +0.05 absolute averaged over five seeds, the hypothesis is falsified. *Direction:* increasing K must increase NSA retention; if it does not, M3 (recursion refines the query and re-engages the compression channel) is wrong.

**F2 (differential test, the non-additive prediction).** *Metric:* (NSA retention(K=4) − NSA retention(K=1)) − (MoBA retention(K=4) − MoBA retention(K=1)). *Threshold:* if this differential is < +0.05 absolute, the hypothesis is falsified. *Direction:* NSA must recover *more* with K than MoBA does. If both recover equally, the architectural-coherence claim (compression branch is the load-bearing fallback channel) is wrong; either both work for some other reason or recovery is an illusion of the metric.

**F3 (downstream task transfer).** *Metric:* NoLiMa accuracy delta (arXiv:2502.05167) and EverMemBench-S "evidence-access" metric (arXiv:2601.20276 — separated from "evidence-use" by their decoupled diagnostic protocol). *Threshold:* on NoLiMa, if NSA + K=4 does not improve over NSA + K=1 by ≥ 3 points on accuracy at 32K context, OR if MoBA + K=4 improves over MoBA + K=1 by ≥ 3 points (which would contradict the prediction that MoBA cannot recover), F3 fires. *Direction:* retrieval-head retention must propagate to task-level retrieval performance, and the NSA/MoBA asymmetry must persist at the task level.

**Optional F4 (mechanism check).** *Metric:* gradient or attention-flow probe — at NSA pass 2, measure whether the top-k selection at pass 2 includes the needle block more often than at pass 1, conditional on the needle block being non-selected at pass 1. *Threshold:* if top-k at pass 2 covers needle ≤ 1.05× top-k at pass 1 (i.e., < 5% relative increase), the M3 mechanism (refined query re-selects relevant blocks) is wrong even if F1 holds. *Direction:* this ensures any retention recovery is attributable to query refinement re-engaging the substrate, not to a confounder in the metric.

If F1 fires, the hypothesis is fully falsified. If F2 fires, the architectural-coherence rationale is wrong (recursion may help both equally — would be a separate, weaker hypothesis). If F3 fires, the result is mechanistically interesting but not task-relevant. If F4 fires alongside F1 holding, the metric is suspect.

## 5a. Cheaper falsification path

The full hypothesis requires four sparse-attention variants × four K values × two probe benchmarks × five seeds, which is a Phase-5-scale evaluation (eval-designer's job). For early kill:

**Cheapest test (single experiment, ≈ a small training-free run).** Fix one mid-size pretrained dense transformer with retrieval heads identified per arXiv:2404.15574. Apply NSA-style attention swap *post-hoc* (no retraining — a sparse mask emulating NSA's three-branch structure, including compression) on inference; compare against MoBA-style mask (top-k blocks only, no compression). Run TRM-style outer-loop wrapper at K=1 vs K=4 (unparameterized — just re-feed the latent answer state through the same model) on a 50-sample NoLiMa subset. Measure retrieval-head retention.

**Single-number kill condition.** If on this minimal setup, NSA-with-K=4 retention does not exceed NSA-with-K=1 retention by ≥ 0.03 absolute on 50 NoLiMa samples (1-sigma noise floor estimated from arXiv:2404.15574's ≈ 600-instance protocol, scaled down), the full Phase-5 evaluation should not be funded. This costs <1% of the full design and uses only training-free post-hoc sparsity, which is acknowledged-imperfect (per arXiv:2502.11089 §2.2 "post-hoc top-k cannot fully exploit native sparse structure") but is sufficient to bound the *direction* of the K-effect.

**Why the cheap test is informative even though post-hoc sparsity is imperfect.** The hypothesis claims the *information channel* (NSA's compression branch) is what enables recovery. Post-hoc swaps preserve that channel structurally even though the network was not trained to use it. If recovery cannot be detected even post-hoc, the channel-existence claim is suspect. If it is detected, the full Phase-5 evaluation under native training is justified.

## 6. Required experiments (sketch only — eval-designer details these)

- **Datasets.** NoLiMa (arXiv:2502.05167, license-flagged); EverMemBench-S / Beyond the Needle's Illusion (arXiv:2601.20276); a small synthetic NIAH set following arXiv:2404.15574 §2 protocol (1K–50K context, 10 depths × 20 lengths, ≈ 600 samples per model).
- **Backbones.** A common pretrained dense transformer (Llama-2-7B-class or smaller, with published retrieval-head set per arXiv:2404.15574 Table 1) with attention swapped for: NSA (arXiv:2502.11089), MoBA (arXiv:2502.13189), DSA (arXiv:2512.02556), Quest (arXiv:2406.10774). Sparsity ratio held constant.
- **Recursion wrapper.** TRM-style outer loop (arXiv:2510.04871) at K ∈ {1,2,4,8}, no extra training (YAGNI fence: no kernel work, no novel training).
- **Baselines.** Dense-attention K=1 (upper bound on retention); each sparsity × K=1 (recursion-off baseline); each sparsity × K=4 (treatment).
- **Ablations.** (a) Strip NSA's compression branch but keep selected + sliding (predicted: behaves like MoBA — recovery collapses; this is the load-bearing ablation). (b) Replace TRM-style state recursion with naive output-replay (no z update — predicted: no recovery on any sparsity pattern, since query is unchanged across passes).
- **Probes.** Retrieval-head retention per arXiv:2404.15574 §2; attention-flow probe on top-k selection across passes (F4); task-level NoLiMa / EverMemBench-S accuracy (F3).

## 7. Risks to the hypothesis

**R1. Native vs post-hoc sparsity confound.** NSA, MoBA, DSA require *native* trained-from-scratch sparsity to fully exhibit their behavior (arXiv:2502.11089 §2.2). Post-hoc attention swaps may exhibit pathologies that contaminate the K-axis effect. *If R1 materializes:* the hypothesis still contributes by specifying which channel structure permits retrieval-head re-formation in principle; the empirical test simply needs to be done with native-trained models (a heavier eval-designer ask but tractable; e.g., the published NSA/MoBA checkpoints if available).

**R2. K=1 retention may already be at ceiling.** If sparse-attention pretraining preserves enough retrieval-head function on first pass that K>1 has no headroom (retention(K=1) ≈ 0.95 of dense), F1 fires trivially without informing about the mechanism. *If R2 materializes:* the hypothesis still contributes a stronger claim — sparse attention does *not* destroy retrieval heads at training, so the spec's fusion thesis has a weaker constraint than feared. We would refine to: pose the test under stress (longer context, harder NoLiMa distractor settings, adversarial NIAH from arXiv:2601.20276) where retention(K=1) drops and headroom appears.

**R3. Recursion may destroy retrieval heads rather than re-form them.** If the latent state update at each pass smears probability mass and degrades the sharp argmax that defines retrieval heads (arXiv:2404.15574's metric is argmax-based), retention could *decrease* with K on all sparsity patterns. *If R3 materializes:* the hypothesis is falsified, and that is itself the gap-finder's anticipated negative outcome — strong evidence that the spec's fusion thesis has a structural problem on long-context retrieval. The negative result is publishable as a constraint on the fusion design space.

**R4. The compression branch may be too coarse.** NSA's compression pools blocks before contributing; if the per-block resolution loses needle-level detail, M2 may fail in practice even though it holds in principle. *If R4 materializes:* a sharper variant of the hypothesis — that *block-pool granularity* (a tunable knob in NSA) gates retrieval-head re-formation — becomes the contribution, with the same falsification logic applied to the compression-block size axis.

**R5. Dataset contamination on NoLiMa.** If the model's pretraining saw NoLiMa-style needles, retention measurements may overestimate generalization. *If R5 materializes:* fall back to EverMemBench-S (arXiv:2601.20276), which uses a 326M-token MemoryBank specifically to evade contamination.

## 8. Sources

- arXiv:2404.15574 — Wu et al., "Retrieval Head Mechanistically Explains Long-Context Factuality" (retrieval-head detection and metric; baseline retention; downstream effects on NIAH and CoT). Verified via paper_details.
- arXiv:2502.11089 — Yuan et al., "Native Sparse Attention" (NSA's three-branch design including compression; explicit acknowledgment of retrieval-head vulnerability under post-hoc top-k). Verified via paper_details and read_paper §2.
- arXiv:2502.13189 — Lu et al., "MoBA: Mixture of Block Attention" (top-k gating; g_i=0 for non-selected blocks; no fallback channel). Verified via paper_details and read_paper §2.
- arXiv:2510.04871 — Jolicoeur-Martineau, "Tiny Recursive Model" (TRM-style K-pass recursion, latent answer state refinement, parameter-tied passes). Verified via paper_details.
- arXiv:2602.11374 — Bick et al., "Retrieval-Aware Distillation for Transformer-SSM Hybrids" (Gather-and-Aggregate heads; explicit preservation under sub-quadratic conversion). Verified via paper_details.
- arXiv:2407.15891 — Tang et al., "RazorAttention" (retrieval heads exploited for KV compression — independent confirmation of head specialness). Verified via paper_details.
- arXiv:2502.05167 — Modarressi et al., "NoLiMa" (non-literal-match long-context probe; non-commercial license — flagged). Verified via paper_details.
- arXiv:2410.04422 — Yu, "Hyper-multi-step" (long-context tasks decompose into multi-step retrieval). Verified via paper_details.
- arXiv:2504.17768 — Nawrot et al., "Sparse Frontier" (sparse-attention pattern × task ablations; baseline for the (sparse pattern × task) axis without retrieval-head retention probe). Verified via paper_details.
- arXiv:2506.08889 — Gao et al., "SeerAttention-R" (self-distilled sparse gating for reasoning; baseline for sparse × reasoning without recursion-pass probe). Verified via paper_details.
- arXiv:2406.10774 — Tang et al., "Quest" (query-aware top-k page selection; baseline for query-aware sparsity). Verified via paper_details.
- arXiv:2512.02556 — DeepSeek-AI, "DeepSeek-V3.2" (DSA / lightning indexer; baseline for indexed sparsity without compression branch). Verified via paper_details.
- arXiv:2601.20276 — Lin et al., "Beyond the Needle's Illusion" / EverMemBench-S (decoupled evidence-access vs. evidence-use; 326M-token MemoryBank for adversarial NIAH). Verified via paper_details.
