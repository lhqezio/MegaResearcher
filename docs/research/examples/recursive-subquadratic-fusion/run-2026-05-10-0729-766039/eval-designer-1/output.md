# Eval design for H1-FB rev-1 — Compressed-Summary Fallback as a Sufficient Recursion Substrate

**Worker:** eval-designer-1
**For hypothesis:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-1/revision-1/output.md` (H1-FB rev-1, red-team APPROVE verdict at round 2)
**Pre-registration date:** 2026-05-10. The decision rules in §6 are committed before any run executes.

---

## 0. Hypothesis being tested (restatement, with falsification criteria)

**Central claim.** Wrap a TRM-style weight-tied K=6 latent-recursion operator (arXiv:2510.04871 §3) around three natively-trainable sparse-attention backbones and compare against FLOP-matched non-shared-deeper K=1 baselines (per the standard iso-depth recipe in arXiv:2410.20672 / arXiv:2604.21106; **not** Ouro §4 — see N1 below):

- **NSA-fb** = NSA arXiv:2502.11089 (compressed + selected + sliding-window branches; the compressed branch is the fallback channel).
- **NSA-no-fb** = NSA with the compressed branch zeroed at *both train and eval time* (training-aware ablation).
- **MoBA** = MoBA arXiv:2502.13189 (top-k block gate, no fallback channel of any kind).

The hypothesis predicts on BABILong qa3+qa4+qa5 (multi-hop, k≥3) at L=64K, 1B-parameter scale:

- **F1 (necessary).** `(NSA-fb K=6) − (NSA-fb K=1) ≥ +6.0` absolute points; falsified if observed < +3.0.
- **F2 (central, sufficient — non-additive DiD with sign asymmetry).**
  `[(NSA-fb K=6) − (NSA-fb K=1)] − [(MoBA K=6) − (MoBA K=1)] ≥ +5.0` AND
  `(NSA-fb K=6 − K=1) > 0` AND `(MoBA K=6 − K=1) ≤ 0` AND
  K=1 calibration `|NSA-fb K=1 − MoBA K=1| ≤ 2`. Falsified if DiD < +2 OR same-sign positive both sides.
  If K=1 calibration fails (gap > 2), test is **inconclusive**, not falsified.
- **F3 (causal role of compressed branch).** `(NSA-fb K=6) − (NSA-no-fb K=6) ≥ +4.0`; falsified if Δ ≤ +1.5.
- **F4' (mechanistic Jaccard probe).** `Jaccard_NSA-fb(I_1, I_6) ≤ Jaccard_MoBA(I_1, I_6) − 0.10` on selected blocks at L=64K on qa3+qa4+qa5. Threshold is acknowledged as uncalibrated (red-team N3); §3.4 of this protocol pins it down with a pilot calibration before main run.

**Residual red-team issues to handle in this design (carried forward):**
- **N1 (Important, citation).** Ouro §4 / Table 5 mis-cited for FLOP-match. This eval-design adopts arXiv:2410.20672 (Relaxed Recursive Transformers, §3-§4 non-tied vs tied at matched FLOPs) as the protocol citation, with arXiv:2604.21106 (Iso-Depth Scaling Laws) as secondary. Ouro citations dropped from FLOP-match recipe.
- **N3 (Suggestion, F4' Jaccard threshold uncalibrated).** Mitigated by §3.4 pilot anchoring.
- **N4 (Suggestion, contribution-magnitude drift).** No methodological consequence; preserved as a synthesis note.
- **N2 (Suggestion, PLT mechanism heterogeneity).** Mitigated by adding a PLT-G-SWA arm to the secondary comparison sweep so the "fallback-type-matters" sub-claim has a direct test rather than relying on PLT's published numbers.

---

## 1. Datasets

Every dataset is a real Hugging Face dataset; license verified at the `hf_inspect_dataset` call below.

### 1.1 Primary (in-distribution): BABILong

- **Name / HF ID:** `RMT-team/babilong` (full benchmark for eval; predictions; 255 config/split rows covering 0k…128K) and `RMT-team/babilong-1k-samples` (1000-example subset per (length × qa) cell, used as the *primary statistical-power test set* per arXiv:2406.10149 §3 reporting convention).
- **License:** Apache-2.0 (per `huggingface.co/datasets/RMT-team/babilong` dataset card; underlying bAbI tasks from Facebook's `bAbI-tasks` are CC-BY-3.0; the BABILong benchmark wrappers/PG19 haystack composition are released under the booydar/babilong repo's Apache-2.0 license per arXiv:2406.10149).
- **Verified via:** `hf_inspect_dataset RMT-team/babilong` and `RMT-team/babilong-1k-samples` — both return Status=Valid with viewer enabled, schema `{input: string, target: string, question: string}`. Configs include `0k, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k`; splits are `qa1…qa20` per BABILong arXiv:2406.10149 §2.
- **Why this dataset.** BABILong is the unique published benchmark that pairs (a) controllable haystack length (key axis for the sparse-attention prediction; below 16K the sparse pattern reduces to near-dense) with (b) reasoning-depth control via the bAbI qa1-qa20 task families (qa3 = three-supporting-fact, qa4 = two-argument relations, qa5 = three-argument relations are k=3-hop multi-hop tasks; qa1 = single-fact, k=1, used as the negative control). This is exactly the modality the hypothesis predicts about: long context × multi-hop reasoning × sparse attention. No other public benchmark gives this orthogonality.
- **Splits used.**
  - **Calibration (gating)**: `16k/qa3` 1k samples — used pre-main to verify `|NSA-fb K=1 − MoBA K=1| ≤ 2 abs points` matched-sparsity calibration. If it fails, sparsity hyperparameters are re-tuned and the calibration re-run; main run is gated on calibration pass.
  - **Primary test**: `64k/qa3`, `64k/qa4`, `64k/qa5` 1k samples each (3000 total examples per arm) — F1, F2, F3 are computed on this.
  - **Length sweep (secondary)**: `4k, 16k, 32k, 64k, 128k` × `qa3, qa4, qa5` 1k samples each — produces a length × backbone × K interaction surface. Tests R1 (compressed-branch bandwidth threshold).
  - **Negative-control test**: `64k/qa1` 1k samples — single-needle retrieval, all variants near-ceiling expected; sanity check that an architectural difference is not driving generic capacity.
- **Sample sizes.** 1000 examples per (length × qa × arm) cell. With 5-class chance ≈ 20% accuracy on bAbI single-token answers, σ̂ ≈ 0.013 at p=0.5. A +5-point DiD detection at α=0.05 two-sided across 4 arms × 4 K-settings (8 conditions for the DiD) requires N ≈ 800 per arm to detect d=0.10 in difference-in-differences with power 0.80; 1000 satisfies this with margin (see §5 power calc).

### 1.2 OOD-1 (single-hop retrieval control): RULER

- **Name / HF ID:** `simonjegou/ruler` (community Parquet redistribution of NVIDIA/RULER per arXiv:2404.06654).
- **License:** Apache-2.0 (NVIDIA/RULER GitHub `LICENSE` at github.com/NVIDIA/RULER is Apache-2.0; the simonjegou redistribution preserves it).
- **Verified via:** `hf_inspect_dataset simonjegou/ruler` returns Status=Valid; configs `4096, 8192, 16384` × split `test`; schema `{context, question, answer_prefix, answer, task, max_new_tokens}`; tasks include `niah_single_1` etc.
- **Why.** RULER's NIAH (needle-in-a-haystack) is **single-hop** retrieval; the hypothesis explicitly predicts NSA-fb and MoBA should be near-ceiling on this and the F2 sign asymmetry should *not* appear. This is a *built-in null case* (per §4 of the hypothesis) — if F2 sign asymmetry emerges on RULER NIAH, that would suggest a generic NSA-vs-MoBA capacity gap rather than a multi-hop-specific recursion-fallback effect. Confounds-test.
- **Splits used.** `4096/test`, `8192/test`, `16384/test`. RULER does not publish a 64K split; we extend by *resampling the depth-32K NIAH templates* up to 64K following the RULER repo's `synthetic.py` synthesis script (Apache-2.0, github.com/NVIDIA/RULER) — this is the standard practice in published RULER evaluations beyond the redistributed lengths. Implementation note: licensing of the synthesized 64K extension follows Apache-2.0.
- **Sample sizes.** 500 examples × 5 NIAH subtasks × 2 lengths {16K, 64K} = 5000 evaluations per arm. NIAH is exact-match string scoring; σ̂ at p=0.9 is ≈ 0.013; +2 points in NIAH is detectable.

### 1.3 OOD-2 (short-context recursion-only control): GSM8K + MBPP

- **GSM8K:** HF ID `openai/gsm8k`, MIT license (per dataset card); verified `hf_inspect_dataset openai/gsm8k` Status=Valid; schema `{question, answer}`. Use `main/test` split (1319 examples).
- **MBPP:** HF ID `google-research-datasets/mbpp`, CC-BY-4.0 (per dataset card); verified `hf_inspect_dataset google-research-datasets/mbpp` Status=Valid; configs `full, sanitized`; use `sanitized/test` (257 examples) for held-out evaluation.
- **Why.** Both are **short-context** (well under 8K, well under any sparse-pattern engagement length). They isolate "recursion benefit at matched compute on text" *independent* of the haystack-length axis. If neither NSA-fb nor MoBA shows recursion lift on GSM8K/MBPP, then the recursion machinery itself is broken (training failure mode); if both show similar lift here, the F2 contrast is genuinely length-conditional. This addresses hypothesis R3 ("TRM may not transfer to text-domain at all").
- **Sample sizes.** GSM8K test = 1319; MBPP sanitized test = 257; pass@1 reported. σ̂ at p=0.4 is ≈ 0.014 on GSM8K, ≈ 0.031 on MBPP.

### 1.4 Pretraining corpus (no-eval, reported for licensing only)

- **HF ID:** `HuggingFaceFW/fineweb-edu` (default config). Verified Status=Valid. License: ODC-BY-1.0 (per dataset card). Corpus is permissive for research pretraining.
- **Use.** All 8 backbone runs see an identical 50B-token sample of fineweb-edu (deterministic seed) so that per-arm differences cannot be attributed to corpus drift (R6 in hypothesis §7).

---

## 2. Backbones (treatment arms + baselines)

All eight runs share: tokenizer (Llama-3 32K BPE, public), optimizer (AdamW, β1=0.9, β2=0.95, wd=0.1, peak LR=3e-4 with cosine decay to 3e-5), training data (50B tokens of fineweb-edu, fixed shuffle seed), training context length (16K with rope-base extension to 128K eval per RoPE-NTK), batch size (4M tokens), gradient accumulation, init scheme. Only the attention module and the recursion topology differ.

### 2.1 Treatment arms (the H1-FB pivot)

**Arm A1 — NSA-fb K=6 (recursion).** Reference implementation: `fla-org/native-sparse-attention` (Apache-2.0, https://github.com/fla-org/native-sparse-attention) or `tilde-research/nsa-impl` (MIT) for Triton kernels per arXiv:2502.11089 §4.1. Three branches (compressed block-size 32, selected top-k=16 blocks, sliding window 512). 1B parameters total (16 transformer blocks, hidden 2048, FF 5504, 32 heads). **Recursion topology:** the last 4 blocks are weight-tied per the TRM scheme (arXiv:2510.04871 §3) and applied K=6 times within one forward pass before the LM head; no token emission; no auxiliary supervision at intermediate iterations (consistent with §9 of the hypothesis). Reference recursion implementation: SamsungSAILMontreal/TinyRecursiveModels (Apache-2.0, https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

**Arm A2 — NSA-no-fb K=6 (recursion, compressed branch zeroed).** Same as A1 except the compressed-branch logits are masked to −∞ at *both* train and eval time. This is the training-aware F3 ablation (not a destructive post-hoc edit). Per red-team RT-4 fix in hypothesis revision: this is *not* an inference-time hack — the model is trained from scratch with the compressed branch removed.

**Arm A3 — MoBA K=6 (recursion).** Reference implementation: `MoonshotAI/MoBA` (Apache-2.0 per repo, https://github.com/MoonshotAI/MoBA) per arXiv:2502.13189. Top-k gate calibrated so that *block-density at the matched 64K length matches NSA-fb's selected-branch density* (per matched-sparsity calibration in §3.2). 1B parameters total, same block layout as A1, K=6 weight-tied recursion in last 4 blocks.

### 2.2 Baselines

**Baseline B1 — Strong frontier (parent-architecture-only): NSA-fb K=1 with non-shared deeper FLOP-matched stack.** Per RT-5 fix and N1: the K=1 baseline is **not** the same model with K=1; it is a *non-shared deeper* variant with 4 additional non-tied transformer blocks of the same width that match the K=6 model's total *inference FLOPs* per forward pass. This is the standard iso-depth comparison in arXiv:2410.20672 §3-§4 (Relaxed Recursive Transformers) and arXiv:2604.21106 (Iso-Depth Scaling Laws). Same recipe applied to NSA-no-fb (B2), MoBA (B3), and Dense (B4-strong-prior).

**Baseline B4 — Strong prior-art frontier: dense softmax K=1 with non-shared deeper FLOP-matched stack.** Same 1B-parameter, FLOP-matched depth as A1's K=6 footprint, with full softmax attention. This is the published-frontier sanity baseline: dense attention has no fallback question (it sees everything once); F1 floors the recursion lift on dense for reference. Cites NSA paper Tab 1/2 as anchor (NSA already +1.3 to +3.2 over dense single-pass; see hypothesis §RT-1).

**Baseline B5 — Trivial sanity baseline: random-answer + majority-class.** For BABILong, the bAbI answer space has 6-12 unique tokens per qa-task; random-uniform over the answer vocabulary and majority-class (most-common gold answer in the qa-train slice) provide chance/floor anchors. Reported per (qa, length) cell. Sanity-checks that the trained models are above floor (per RT-9 / hypothesis §R4 noise-floor concern at the originally-considered 350M scale, now mitigated by 1B-scale + this explicit floor reporting).

### 2.3 Bonus / consistency arms (not load-bearing for F1-F3)

**Arm C1 — DSA inference-time recursion (cheap path B, ~50 GPU-hours).** DeepSeek-V3.2-Exp open weights (arXiv:2512.02556) with K=6 inference-time recursion in the last block. Weak signal, framed as a *bonus prediction* per hypothesis §6a. Not used in F1-F3 falsification logic; reported as supplementary evidence for the "no-fallback predicts flat lift" mechanism.

**Arm C2 — PLT-G-SWA K=6 (per N2 mitigation).** A from-scratch 1B PLT-style backbone with G-SWA per arXiv:2510.24824 §3.1 (per-loop dedicated KV, sliding-window). This is added so the "fallback-type-matters" claim is testable directly on our recipe rather than relying on PLT's published 680M-active numbers. Pre-registered as a *consistency arm* (no F-falsifier depends on it; if PLT-G-SWA shows similar lift to NSA-fb, the broader fallback-channel claim is corroborated; if dramatically different, both NSA-fb and PLT may be operating via mechanism heterogeneity, which would require synthesis-stage interpretation).

### 2.4 Total arm count

| ID | Arm | Sparse pattern | Recursion | FLOP-match baseline | Purpose |
|---|---|---|---|---|---|
| A1 | NSA-fb K=6 | compressed+selected+sliding | K=6, last 4 blocks tied | B1 | Treatment |
| A2 | NSA-no-fb K=6 | selected+sliding (no compressed) | K=6, last 4 blocks tied | B2 | F3 ablation |
| A3 | MoBA K=6 | top-k gate, no fallback | K=6, last 4 blocks tied | B3 | F2 contrast |
| B1 | NSA-fb K=1 | compressed+selected+sliding | K=1, +4 non-tied blocks | — | F1, F2 anchor |
| B2 | NSA-no-fb K=1 | selected+sliding | K=1, +4 non-tied blocks | — | F3 anchor |
| B3 | MoBA K=1 | top-k gate | K=1, +4 non-tied blocks | — | F2 anchor |
| B4 | Dense K=1 (deep) | full softmax | K=1, +4 non-tied blocks | — | Frontier prior-art |
| B5 | Trivial baseline | n/a | n/a | — | Random + majority |
| C1 | DSA K=6 inference (open weights) | lightning-indexer top-k | K=6 inference only | — | Bonus prediction |
| C2 | PLT-G-SWA K=6 | per-loop G-SWA | K=6 last 4 blocks tied | dense-deep (B4) | N2 consistency |

8 from-scratch pretraining runs (A1, A2, A3, B1, B2, B3, B4, C2) + C1 (no training, inference only). B5 is computed analytically from BABILong gold answer distributions.

**At least 3 baselines requirement:** B1 (parent-architecture prior-art / FLOP-matched K=1 of the treatment), B4 (frontier prior-art / dense), B5 (trivial sanity) — three baselines.

---

## 3. Metrics

### 3.1 Primary metric (tied to F1, F2, F3)

**M_primary = exact-match accuracy on BABILong qa3+qa4+qa5, length 64K, averaged**.

Formal definition. Let $A_{x, K, qa, L}$ be exact-match accuracy on (arm $x$, recursion depth $K$, qa task, length $L$):
$$A_{x, K, qa, L} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat y_i = y_i]$$
where $\hat y_i$ is the greedy-decoded first-token answer (single-token bAbI answers). Define
$$\bar A_{x, K, L} = \frac{1}{3} \sum_{qa \in \{3,4,5\}} A_{x, K, qa, L}.$$

Then:
- $F1 := \bar A_{NSA\text{-}fb, K=6, 64K} - \bar A_{NSA\text{-}fb, K=1, 64K}$
- $F2_{DiD} := (\bar A_{NSA\text{-}fb, K=6, 64K} - \bar A_{NSA\text{-}fb, K=1, 64K}) - (\bar A_{MoBA, K=6, 64K} - \bar A_{MoBA, K=1, 64K})$
- $F2_{cal} := |\bar A_{NSA\text{-}fb, K=1, 64K} - \bar A_{MoBA, K=1, 64K}|$
- $F2_{signA} := \bar A_{NSA\text{-}fb, K=6, 64K} - \bar A_{NSA\text{-}fb, K=1, 64K}$
- $F2_{signB} := \bar A_{MoBA, K=6, 64K} - \bar A_{MoBA, K=1, 64K}$
- $F3 := \bar A_{NSA\text{-}fb, K=6, 64K} - \bar A_{NSA\text{-}no\text{-}fb, K=6, 64K}$

### 3.2 Secondary metrics (catch failure modes)

- **M_sec1 — Length sweep (bandwidth-threshold catch).** $\bar A_{x, K, L}$ for $L \in \{4K, 16K, 32K, 64K, 128K\}$. The compressed-branch bandwidth concern (R1) predicts F2's DiD should *grow* with L. If F2 DiD is large at 16K but flat at 64K, the compressed branch is bandwidth-limited as suspected.
- **M_sec2 — RULER NIAH single-hop accuracy.** As §1.2 — null-case check.
- **M_sec3 — GSM8K / MBPP recursion-lift check.** Per §1.3 — confirms recursion machinery itself is intact text-domain.
- **M_sec4 — F4' inter-iteration Jaccard drift (mechanism probe).**
$$J_{x, qa, L} = \frac{1}{H \cdot N_q} \sum_{h, n} \frac{|S_1^{(h, n)} \cap S_6^{(h, n)}|}{|S_1^{(h, n)} \cup S_6^{(h, n)}|}$$
where $S_t^{(h, n)}$ is the set of selected (top-k) block IDs in head $h$ at query position $n$ at iteration $t$. Computed for arms A1 (NSA-fb K=6) and A3 (MoBA K=6) on a held-out 200-example sample of qa3+qa4+qa5 at L=64K. F4' criterion: $J_{NSA\text{-}fb} \leq J_{MoBA} - 0.10$.
- **M_sec5 — Per-iteration attention-pattern entropy.**
$$H_t = -\sum_{j} p_t^{(j)} \log p_t^{(j)}$$
over normalized attention weights at iteration $t$. Predicted to *decrease* with $t$ on NSA-fb (refinement) and stay flat on MoBA (no productive refinement channel). Reported as a sanity diagnostic.
- **M_sec6 — Trivial-baseline floor.** B5 (random + majority-class) per cell, to confirm trained models are above-floor.

### 3.3 Why these metrics catch the failure modes red-team flagged

- **RT-2 (additivity).** F2_DiD with calibration constraint F2_cal directly tests interaction term, not joint outcomes. Sign-asymmetry constraint forecloses scaled-additive null.
- **RT-5 (compute-match).** B1, B2, B3, B4 are all FLOP-matched non-shared-deeper baselines per arXiv:2410.20672 — recursion lift cannot be confounded with depth.
- **R1 (bandwidth).** M_sec1 length sweep directly probes the bandwidth threshold; if compressed branch is bandwidth-saturated at L=64K, M_sec1 surfaces it as a non-monotonic curve.
- **R3 (transfer to text).** M_sec3 (GSM8K, MBPP) detects whether the recursion machinery works at all on text.
- **N3 (Jaccard threshold uncalibrated).** Mitigation in §3.4.

### 3.4 F4' Jaccard threshold pilot calibration (N3 mitigation)

The 0.10 threshold is *not* anchored in published prior work (red-team N3). To pin it down before the main run, we add a pre-main pilot:

- **Pilot.** On the 350M cheap-path-A models (cheap path A from hypothesis §6a), measure $J_{NSA\text{-}fb}(I_1, I_6)$ and $J_{NSA\text{-}no\text{-}fb}(I_1, I_6)$ on `16k/qa3` 1k samples. Compute the gap $\delta = J_{NSA\text{-}no\text{-}fb} - J_{NSA\text{-}fb}$.
- **Pre-registered rule.** If pilot $\delta \geq +0.05$, we adopt the F4' threshold as `Jaccard_NSA-fb ≤ Jaccard_MoBA − max(0.5 × δ, 0.05)` on the main 1B run. If $\delta < +0.05$, the F4' threshold is weakened to a *directional* prediction ("NSA-fb's Jaccard < MoBA's Jaccard with bootstrapped CI excluding zero"). This is a *partial-falsifier degradation*, explicitly pre-registered, that addresses N3's calibration concern.
- The pilot is part of cheap-path-A's existing 250 GPU-hour budget; no incremental compute.

---

## 4. Ablations (diagnostic isolation)

The four arms in §2 already constitute the principal ablation grid. The diagnostic ablations called out below isolate which *factor* drives the F2/F3 outcome.

### 4.1 Compressed-branch ablation (already part of F3)

A1 vs A2 (NSA-fb vs NSA-no-fb, both K=6, FLOP-matched) — isolates the compressed branch as the load-bearing channel for the recursion lift. **Predicted:** A1 − A2 ≥ +4 abs points on qa3+qa4+qa5 at 64K (F3). **If null:** the compressed branch is not load-bearing; M2 mechanism is wrong but the broader "any fallback" claim could still survive via PLT-G-SWA arm.

### 4.2 Recursion-depth ablation

For arm A1 only (cheap, since the model is already trained): vary inference-time K ∈ {1, 2, 4, 6, 8} on the *same* trained K=6 weights. Predicted: monotone improvement up to K=6 (training distribution), plateau or degradation at K=8 (out-of-distribution). Identifies whether K=6 is on a saturation or on a slope.

### 4.3 Recursion-block depth ablation

A1' = NSA-fb with weight-tied recursion in the last *2* blocks instead of last 4. Same training tokens. Predicted: recursion lift roughly tracks scaled-down K=6, mechanism preserved. Tests whether the lift is "block-count-dependent" or a generic property of weight-tied iteration. Optional, reduces to one extra 1B run if budget permits (see §7).

### 4.4 Length × backbone interaction (already M_sec1)

Length sweep is itself a diagnostic ablation: it isolates whether F2 is length-dependent (predicted) or length-independent (would suggest a generic NSA-MoBA capacity gap, partial steelman).

### 4.5 Recursion-vs-CoT distinction (architectural-coherence check)

Per hypothesis §9: F4'/M_sec4 (Jaccard) and M_sec5 (per-iteration entropy) probe directly that the recursion is operating in *latent space* (attention pattern shifts across iterations) and not via any text emission. No CoT prompting in any condition. If M_sec5 is flat (no entropy change across iterations), the recursion is not refining latent state and the entire F2 outcome is suspect.

---

## 5. Statistical analysis plan (pre-registered)

**Pre-registration statement.** This analysis plan is committed before any model is trained. Decision rules below are not modifiable post-hoc.

### 5.1 Primary test (F2)

- **Estimator.** $F2_{DiD}$ as defined in §3.1, computed per-seed.
- **Replication design.** $S = 3$ seeds for each of the 8 main pretraining runs (24 total runs). The recursion-lift differential per backbone is computed over seeds; $F2_{DiD}$ is averaged across seeds, with seed-clustered standard error.
- **Test.** Two-sided paired bootstrap (10000 resamples, sample-clustered at the example level *within* a seed and seed-clustered *across* seeds; this hierarchical bootstrap accounts for both data sampling variance and seed variance per the recommendation in arXiv:2305.18486 / common ML eval practice). Test statistic: $F2_{DiD} \geq +5.0$.
- **α (Type-I).** α = 0.05 two-sided primary; α = 0.0125 after Bonferroni correction across the four falsification criteria F1, F2, F3, F4' (m = 4). FDR is *not* applied because the criteria are pre-specified and ordered (Bonferroni is more conservative and apt for confirmatory pre-registration).
- **Decision rule (pre-registered).**
  - **Hypothesis SUPPORTED on F2** if all of: (i) bootstrap 95%-CI lower bound on $F2_{DiD} \geq +5.0$ Bonferroni-adjusted (bootstrap p < 0.0125); (ii) sign asymmetry holds in the *point estimate* and the sign of $F2_{signA}$ has 95%-CI lower bound > 0 AND sign of $F2_{signB}$ has 95%-CI upper bound ≤ 0; (iii) $F2_{cal} \leq 2$ in point estimate.
  - **Hypothesis FALSIFIED on F2** if: (i) bootstrap 95%-CI upper bound on $F2_{DiD} \leq +2$, OR (ii) $F2_{signA} \leq 0$ at lower CI bound AND $F2_{signB} \geq 0$ at upper CI bound (same-sign positive lifts).
  - **Inconclusive** if $F2_{cal} > 2$ (calibration failure — re-run with re-tuned sparsity hyperparameters before re-evaluating).

### 5.2 Secondary tests

- **F1.** Bootstrap CI on $\bar A_{NSA\text{-}fb, K=6} - \bar A_{NSA\text{-}fb, K=1}$. Threshold +6.0 for support, +3.0 for non-falsification, < +3.0 for falsification. Bonferroni-adjusted α = 0.0125.
- **F3.** Bootstrap CI on $\bar A_{NSA\text{-}fb, K=6} - \bar A_{NSA\text{-}no\text{-}fb, K=6}$. Threshold +4.0 for support, +1.5 for non-falsification, < +1.5 for falsification. Bonferroni-adjusted α = 0.0125.
- **F4'.** Per §3.4 calibration-aware threshold; one-sided bootstrap on the Jaccard difference. Bonferroni-adjusted α = 0.0125.

### 5.3 Effect-size of interest and power

- **Effect size of interest.** F2 DiD = +5.0 absolute points on accuracy ratio in [0, 1] = +0.050 on the proportion scale. (Anchor: PLT +6.1 absolute on 680M MoE at 150B tokens, arXiv:2510.24824 Tab 2; predicted NSA-fb lift bracket +6 to +10 minus MoBA non-positive ≈ +5 lower bound.)
- **Assumed within-seed σ.** From bAbI single-token answer dispersion: σ̂ ≈ 0.013 at p=0.5 with N=1000 examples (binomial). Cross-seed σ from prior text-domain replications at 1B scale (Pythia-1B, OLMo-1B): σ ≈ 0.010 on multi-task averaged accuracy.
- **Combined σ on $F2_{DiD}$.** Four model trainings × hierarchical: $\sigma_{F2_{DiD}} = \sqrt{4 \cdot (\sigma_{seed}^2 + \sigma_{sample}^2/N)} \approx \sqrt{4 \cdot (0.010^2 + 0.013^2/1000)} \approx 0.020$ (point estimate).
- **Power.** With 3 seeds and N=1000 per cell, the minimum detectable effect at α=0.0125, power=0.80 is roughly $1.96 \cdot \sigma_{F2_{DiD}} / \sqrt{S} = 1.96 \cdot 0.020 / \sqrt{3} \approx 0.023$ on the proportion scale = 2.3 absolute points. The hypothesis predicts +5 absolute points, comfortably above MDE. **Power for F2 ≥ 0.95** under the predicted effect.
- **Honest caveat.** Cross-seed σ at 1B/50B tokens is somewhat extrapolated; if observed cross-seed σ exceeds 0.02, we will *upfront* report power-degraded inference and reduce confidence in the SUPPORT verdict accordingly. We will *not* increase seeds post-hoc unless the entire 1B re-run is justifiable from budget.

### 5.4 Multiple-comparison strategy

- Bonferroni-corrected α = 0.0125 across F1, F2, F3, F4' (m = 4).
- Length sweep (M_sec1) is *exploratory* (not pre-registered with thresholds beyond directional predictions); reported as descriptive.
- RULER, GSM8K, MBPP secondary tests are *exploratory* with descriptive thresholds (NIAH near-ceiling, recursion lifts in [+1, +6] absolute) and are not corrected.

### 5.5 Pre-registration commitment

Decision rules in §5.1-§5.4 are committed in this document. Any deviation (e.g., changing α, threshold, or test) post-hoc will be flagged as exploratory in the final synthesist report.

---

## 6. Falsification trace (each F-criterion → which experiment yields the metric)

| Falsifier | Source of metric | Arms involved | Pre-registered kill threshold | Pre-registered support threshold |
|---|---|---|---|---|
| **F1** | BABILong qa3+qa4+qa5, L=64K, M_primary | A1 (NSA-fb K=6) vs B1 (NSA-fb K=1 deep) | $\bar A_{A1} - \bar A_{B1} < +3.0$ | $\bar A_{A1} - \bar A_{B1} \geq +6.0$ |
| **F2** | BABILong qa3+qa4+qa5, L=64K, M_primary | A1, B1, A3, B3 with calibration | $F2_{DiD} \leq +2$ at upper CI OR same-sign positive | $F2_{DiD} \geq +5$ at lower CI AND signs asymmetric AND $F2_{cal} \leq 2$ |
| **F3** | BABILong qa3+qa4+qa5, L=64K, M_primary | A1 (NSA-fb K=6) vs A2 (NSA-no-fb K=6) | $\bar A_{A1} - \bar A_{A2} \leq +1.5$ | $\bar A_{A1} - \bar A_{A2} \geq +4.0$ |
| **F4'** | M_sec4 Jaccard at L=64K on qa3+qa4+qa5 | A1, A3 (selected-block IDs across iterations) | $J_{A1} - J_{A3} \geq -0.05$ (or $\geq -0.5\delta$ from pilot) | $J_{A1} - J_{A3} \leq -0.10$ (or $\leq -\delta$ from pilot) |

**Each F-criterion is yielded by an experiment that is *designed to fail* if the hypothesis is wrong.** F1 fails if the NSA-fb backbone simply cannot use the compressed branch under recursion; F2 fails if the interaction term is null (additive marginals only) or sign-symmetric; F3 fails if the compressed branch is incidental to the lift; F4' fails if the mechanism is not attention-pattern drift on NSA. *The experiment can fail.* The authors of this design state in advance what would constitute falsification, before any run is executed.

---

## 7. Compute budget (honest)

GPU = NVIDIA H100 80GB SXM, assumed sustained 700 TFLOPs effective on the 1B-scale FlashAttention-2 / NSA / MoBA Triton kernels. All arithmetic below shows derivation.

### 7.1 Full experiment (8 pretraining runs, 1B params, 50B tokens each)

| Component | FLOPs | H100-hours | Notes |
|---|---|---|---|
| 1B model × 50B tokens × 6 FLOPs/token (Chinchilla forward+backward, sparse-attn-corrected to 0.85x of dense) | 2.55e20 | 101 hr | Per single 1B run. |
| × 8 from-scratch runs (A1, A2, A3, B1, B2, B3, B4, C2) | 2.04e21 | 810 hr | 50% margin → 1215 hr realistic. |
| Eval pass (BABILong 4K-128K × 8 runs × 1k samples × 6 qa-tasks + RULER + GSM8K + MBPP) | ~3e19 | 12 hr | Inference-only, batched. |
| Mechanistic probes (Jaccard + entropy across 200 examples × 6 iters × 2 arms) | ~1e19 | 8 hr | Negligible. |
| Hyperparameter search (sparsity-density calibration on `16k/qa3` for F2_cal) | ~5e19 | 60 hr | Two backbones × ~5 sparsity points. |
| Storage / overhead / checkpointing / restart loss | — | 100 hr | 8% slack. |
| **Subtotal (full)** | — | **~1100 hr realistic, ~1395 hr 1.25× margin** | Under 2000-GPU-hour fence. |

### 7.2 Cheap path A (NSA-only 350M F3 + F4' pilot, ~250 GPU-hours)

Per hypothesis §6a, two 350M NSA models × 5B tokens. 1B → 350M scales FLOPs ~0.35x; 50B → 5B scales tokens 0.1x. Net per-run ≈ 101 × 0.35 × 0.1 = 3.5 hr. ×2 runs = 7 hr training. Eval pass and Jaccard pilot ≈ 30 hr. **Under 250 GPU-hours easily** — actual estimate ≈ 100-150 hr; the 250 hr bound is generous.

This cheap path is **a genuine kill-test for F3 and a calibration test for F4'**, but explicitly **not** a kill-test for F2 (the central falsifier). It is a cost-efficient pre-1B go/no-go: if A1 − A2 < 0 at 350M/16K, the F3 mechanism is doubtful enough that the full 1B experiment should be reconsidered.

### 7.3 Cheap path B (DSA inference-only on open weights, ~50 GPU-hours)

Inference only on DeepSeek-V3.2-Exp open weights, batch-12 K=6 recursion at L=32K on BABILong qa3+qa4+qa5 1k samples + DSA K=1 baseline. Inference estimate: 32K × 1000 × 3 qa × 2 K-settings × 30B-active-params × 2 FLOPs/param/token / 700e12 ≈ 24 hr. Allow 50 hr for retries and tuning. **Under fence.**

### 7.4 Total compute summary

| Bundle | GPU-hours |
|---|---|
| Cheap path A (F3 + F4' pilot at 350M) | ~150 (250 budgeted) |
| Cheap path B (DSA inference) | ~50 |
| Full 1B experiment (F1, F2, F3, F4') | ~1100 (1395 with margin) |
| **Combined total** | **~1495 (1695 with all slack)** |

**Wall-clock projection.** On 8× H100 cluster: cheap paths complete in ~3 days; full experiment in ~7 days. Sequential conservative wall-clock = 10 days. With 32× H100 (4 nodes), cheap paths in 1 day, full in 2 days — total ~3 days.

**`flagged_intractable: false`.** Under the 2000 GPU-hour fence even at 1.25× margin.

---

## 8. Risks to the experiment (what could make the result misleading)

### 8.1 Data leakage

- **Risk.** BABILong is built on bAbI (Facebook, 2015) + PG19 haystacks; bAbI is in many pretraining corpora. If FineWeb-edu happens to contain bAbI prompts, all arms benefit equally, but the threshold may be deflated.
- **Mitigation.** Decontaminate FineWeb-edu against the BABILong gold-answer set via 13-gram filtering (standard SLED protocol). Report contamination rate. Pre-registered: if contamination > 0.1% of training corpus, re-shuffle to a non-contaminated subset.

### 8.2 Baseline-tuning asymmetry

- **Risk.** NSA's reference Triton kernels are mature (`fla-org/native-sparse-attention`), MoBA's reference (`MoonshotAI/MoBA`) is also mature, but tuning hours invested in one arm but not the other could create a confound.
- **Mitigation.** Pre-registered hyperparameter sweep budget per arm: 100 GPU-hours each on `16k/qa3` calibration. No arm receives more sweep time. Sparsity densities matched at calibration.

### 8.3 Evaluation-suite drift

- **Risk.** RMT-team/babilong has been updated post-publication (255 config/split rows currently visible). Drift between paper-time and our eval-time data could change accuracy levels.
- **Mitigation.** Pin the dataset commit hash at evaluation time. Report the commit hash in the synthesist's final report.

### 8.4 Recursion-K=1 being effectively K=6 in disguise

- **Risk.** If the K=1 baseline (B1) "discovers" iterative refinement during training (e.g., implicit recurrence via residual stream), the FLOP-matched comparison no longer isolates explicit weight-tied recursion.
- **Mitigation.** Probe B1 for *implicit recurrence* via M_sec5 entropy at intermediate layers (not iterations); flag if the deeper non-tied stack shows iteration-like attention-pattern dynamics. Reported as a confound check.

### 8.5 Single-token bAbI answers being too sensitive to first-token bias

- **Risk.** bAbI answers are single-tokens (e.g., "bathroom"); accuracy depends only on the first decoded token. Different arms may have different first-token-bias from their own training distributions.
- **Mitigation.** Compute both M_primary (first-token exact match) and a string-normalized variant (lower-cased, stripped). Report both.

### 8.6 Compressed-branch zeroing might break the model entirely (A2 distribution shift)

- **Risk.** Even with training-aware A2 (RT-4 fix), zeroing the compressed branch from random init may produce a model that fails on basic LM perplexity, not specifically on multi-hop. If A2 is broken at K=1, the F3 contrast is uninterpretable.
- **Mitigation.** Pre-registered A2 sanity gate: if A2's K=1 perplexity on FineWeb-edu held-out is > 1.3× A1's K=1 perplexity, the A2 model is considered broken and F3 is reported as inconclusive (not falsifying).

### 8.7 K=6 pretraining instability

- **Risk.** Weight-tied K=6 from-scratch can be unstable (TRM uses K=4–8 with careful init). Unstable training inflates seed variance and reduces F2 power.
- **Mitigation.** Use TRM's published init recipe (arXiv:2510.04871 §4) and scale auxiliary supervision off (per §9 of hypothesis). Run at S=3 seeds; if seed variance σ > 0.03 on $\bar A_{x, K, L}$, increase S to 5 (within +30% compute slack budgeted).

### 8.8 Non-monotone length sweep signaling bandwidth saturation rather than no-effect

- **Risk.** F2 DiD might be +5 at 16K and 0 at 64K (compressed branch saturates). This is *not* falsification of F2 at 64K; rather it is partial support with a bandwidth caveat (R1 in hypothesis).
- **Mitigation.** Explicitly pre-registered: F2 is evaluated *at L=64K* for the falsification decision, but the full length sweep is reported and discussed. A "16K-only F2 support" is a different finding (consistent with M2 mechanism but with R1 risk realized) and synthesist must report it as such.

---

## 9. License table

| Resource | HF / GitHub ID | License | Verified |
|---|---|---|---|
| BABILong | `RMT-team/babilong` | Apache-2.0 (BABILong wrappers); CC-BY-3.0 (underlying bAbI tasks) | hf_inspect_dataset Status=Valid; arXiv:2406.10149 license footer |
| BABILong-1k | `RMT-team/babilong-1k-samples` | Apache-2.0 | hf_inspect_dataset Status=Valid |
| RULER | `simonjegou/ruler` (redistribution); upstream `NVIDIA/RULER` | Apache-2.0 | hf_inspect_dataset Status=Valid; NVIDIA/RULER LICENSE |
| GSM8K | `openai/gsm8k` | MIT | hf_inspect_dataset Status=Valid; dataset card |
| MBPP | `google-research-datasets/mbpp` | CC-BY-4.0 | hf_inspect_dataset Status=Valid; dataset card |
| LongBench (optional supplementary) | `Xnhyacinth/LongBench` | MIT (THUDM upstream) | hf_inspect_dataset Status=Valid |
| FineWeb-edu | `HuggingFaceFW/fineweb-edu` | ODC-BY-1.0 | hf_inspect_dataset Status=Valid; dataset card |
| NSA reference (Triton kernels) | `fla-org/native-sparse-attention` | Apache-2.0 | github.com/fla-org/native-sparse-attention LICENSE |
| NSA reference (alt) | `tilde-research/nsa-impl` | MIT | github LICENSE |
| MoBA reference | `MoonshotAI/MoBA` | Apache-2.0 | github.com/MoonshotAI/MoBA LICENSE |
| TRM recursion reference | `SamsungSAILMontreal/TinyRecursiveModels` | Apache-2.0 | github LICENSE |
| DeepSeek-V3.2-Exp open weights (cheap-path B only) | DeepSeek-V3.2-Exp model card | DeepSeek License (model weights, research-permissive) | DeepSeek-AI release notes for V3.2 |

All data and reference-implementation licenses are research-permissive. No restricted-license items in the critical path.

---

## 10. Sources (arxiv IDs + repo / dataset IDs, every citation in this document)

**Papers (arxiv):**
- TRM — `arXiv:2510.04871` — Less is More: Recursive Reasoning with Tiny Networks
- HRM — `arXiv:2506.21734`
- Universal Transformer — `arXiv:1807.03819`
- Huginn — `arXiv:2502.05171`
- Huginn Logit-Lens / Coda-Lens probe critique — `arXiv:2507.02199`
- Ouro — `arXiv:2510.25741` (cited only as one of the recursion-pretraining literature; *not* cited for FLOP-match per N1 fix)
- MoR — `arXiv:2507.10524`
- Parallel Loop Transformer (PLT) — `arXiv:2510.24824`
- NSA — `arXiv:2502.11089`
- MoBA — `arXiv:2502.13189`
- DSA / DeepSeek-V3.2-Exp — `arXiv:2512.02556`
- SeerAttention-R — `arXiv:2506.08889`
- Sparse Frontier — `arXiv:2504.17768` (Jaccard / pattern-drift methodology only)
- Retrieval Head — `arXiv:2404.15574` (auxiliary M3, not load-bearing)
- BABILong — `arXiv:2406.10149`
- RULER — `arXiv:2404.06654`
- Relaxed Recursive Transformers — `arXiv:2410.20672` (FLOP-match recipe, replacing the mis-cited Ouro reference per N1)
- Iso-Depth Scaling Laws for Looped LMs — `arXiv:2604.21106` (secondary FLOP-match citation)

**Datasets (HF):**
- `RMT-team/babilong`, `RMT-team/babilong-1k-samples`, `simonjegou/ruler`, `openai/gsm8k`, `google-research-datasets/mbpp`, `Xnhyacinth/LongBench`, `HuggingFaceFW/fineweb-edu`.

**Repos (GitHub):**
- `fla-org/native-sparse-attention`, `tilde-research/nsa-impl`, `MoonshotAI/MoBA`, `SamsungSAILMontreal/TinyRecursiveModels`, `NVIDIA/RULER`, `booydar/babilong`.
