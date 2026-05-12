# Red-Team Critique — H3 (round 1)

**Critiquing:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-3/output.md`
**Gap targeted:** A8 × B9 (gap-finder-1 §41 + gap-finder-2 §9)
**Revision round:** 1

---

## 1. Verdict

**REJECT (revision-1)**

The gap-claim survives independent verification, the mechanism's *direction* is intuitive, and the falsification structure is mostly operationalizable. But several load-bearing details are unjustified or internally inconsistent: the cross-architecture retrieval-head identity is not handled, the Quest prediction is ad hoc relative to the stated mechanism, the noise floor on the cheap test is below the kill threshold, the architectural-coherence story is weakened by competing prior art (G&A distillation), and the recursion is treated as a turn-key wrapper when adapting TRM's `(x, y, z)`-recursion to a long-context transformer is nontrivial. Each is fixable. None is fatal.

---

## 2. Gap re-verification

I ran four independent literature queries to test whether the (sparse-attention × recursion × retrieval-head retention) intersection is genuinely unexplored.

| # | Query | Top hits | Verdict |
|---|---|---|---|
| Q1 | `retrieval head sparse attention recursion depth` | 2404.15574, 2412.04757, 2602.11374, 2407.15891, 2410.10819, 2510.20787, 2603.28458; no recursion×retrieval-head paper | clean |
| Q2 | `native sparse attention retrieval head retention long context` | 2404.15574, 2511.00819 (NSA optimization), 2502.09647 (long-ctx head id), 2503.23306 (focus directions), 2511.23319 (HSA); no joint study | clean |
| Q3 | `recursive transformer iterative refinement long context retrieval` | RMT (2304.11062), Landmark Attention (2305.16300), ATLAS (2505.23735), RCC (2406.06110); these are token-level recurrence, NOT the depthwise architectural recursion (TRM/HRM/Ouro) the hypothesis means | clean for depthwise recursion × retrieval-head |
| Q4 | `attention masking retrieval head NSA pretrained model evaluation` | DeCoRe (2410.18860, masks retrieval heads at decode-time, not under sparse attention); AttentionInfluence (2505.07293, masks for data selection); 2404.15574 itself | clean |
| Q5 | `Tiny Recursive Model TRM long context language model` | **Recursive Language Models (2512.24601, 96 upvotes)** and **SRLM (2603.15653, 12 upvotes)** are inference-time programmatic recursion, not architectural; TRM itself does not couple to long-context | partly relevant, see §7-O5 |

**Gap claim survives.** No paper measures retrieval-head retention as a function of (sparse-attention pattern × architectural-recursion depth K). The closest reference points (DuoAttention 2410.10819, RazorAttention 2407.15891, DeCoRe 2410.18860, G&A 2602.11374) all touch retrieval heads under modifications of the dense-attention substrate, none touch the joint regime.

`gap_claim_survives: true`

---

## 3. Citation spot-checks

I verified four cited papers against what they actually claim:

### 3.1 arXiv:2404.15574 — Retrieval Head (read §2 directly)

What the paper says: retrieval score is the **frequency** of a head's copy-paste behavior — specifically, score = |g_h ∩ k| / |k| where g_h is the set of needle tokens for which the head's argmax-attention position lands inside the needle index range *and* the attended token equals the currently generated token. Threshold = 0.1. Total ≈ 600 instances per model.

What the hypothesis claims: same definition, same threshold, same protocol. **Verified accurate.**

**However:** the metric is an *argmax* over the head's attention probability distribution. Under NSA, the attention output for a given query is `gated_combination(compressed_attn, selected_attn, sliding_attn)` — there are **three** softmaxes and a learned gate. The "head's argmax attention position" is not directly defined; one must specify whether the argmax is taken over the gated mixture, over the selected branch alone, etc. The hypothesis does not specify this, which is a non-trivial measurement decision (Important).

### 3.2 arXiv:2502.11089 — NSA (read §2 directly)

What the paper says (§2.2): the *motivation* for NSA's compression branch is precisely that "top 20% attention can only cover 70% of total attention scores, rendering structures like retrieval heads in pretrained models vulnerable to pruning during inference."

What the hypothesis claims (§3 M2): "NSA §2.2 explicitly cites this concern, noting that post-hoc top-k pruning leaves retrieval heads in pretrained models vulnerable to pruning during inference." **Verified accurate quote.**

**However:** §2 of NSA does **not** itself contain Figure 2 / the three-branch architectural diagram (that's §3). The hypothesis cites "§2.3 and Figure 2" for the three-branch design — Figure 2 is referenced in §2 of the read I performed (which is the "Rethinking Sparse Attention" framing section, not the architecture section). The architectural specification is in §3, "NSA Methodology." This is a minor citation-locator inaccuracy, not a substantive misrepresentation. (Suggestion.)

### 3.3 arXiv:2502.13189 — MoBA (read §2 directly)

What the paper says (§2.2, eq. 5): `g_i = 1 if s_i ∈ Topk(...), 0 otherwise`. Selected blocks alone contribute; non-selected blocks are excluded entirely. There is no compression / pooling / fallback channel.

What the hypothesis claims: same. **Verified accurate.**

The MoBA-vs-NSA architectural asymmetry the hypothesis leans on is real. Good.

### 3.4 arXiv:2510.04871 — Tiny Recursive Model (read §3 directly)

What the paper says (§3, code excerpt):
```python
def latent_recursion(x, y, z, n=6):
    for i in range(n):
        z = net(x, y, z)
    y = net(y, z)
    return y, z
```
TRM's recursion operates on three explicit tensors: `x` (input), `y` (current answer), `z` (latent reasoning state). The `net` is a small two-layer block. ARC-AGI/Sudoku/Maze are puzzle outputs where `y` is a small grid-shaped tensor; `z` is the same shape as `y`. The recursion is depthwise (same parameters reapplied) and operates *on the answer state*, not on the residual stream of a transformer.

What the hypothesis claims (§3 M3): "TRM recurses a small two-layer block over a latent answer state z and inputs x for K iterations, with each pass refining z based on the previous pass; the same block is re-applied with the same parameters but different state." **Verified accurate description of TRM as published.**

**However:** the hypothesis then proposes "TRM-style outer-loop K∈{1,2,4,8} on the latent answer state" applied to a 1-3B-parameter transformer with retrieval-head structure on 32-64K-token context. TRM's architecture is *not* a transformer wrapping a long-context backbone — it is a tiny custom 2-layer block whose `(x, y, z)` operate on puzzle-shaped grids. There is no published recipe for "TRM-style recursion on a long-context transformer," and the operationalization of `y` (answer state) and `z` (latent) for a long-context retrieval task is a substantial design step the hypothesis treats as obvious. (Critical — see §4.)

---

## 4. Mechanism critique

### 4.1 M1 (retrieval-head metric is universal across architectures) — partially supported

arXiv:2404.15574 Table 1 covers Llama-2, Mistral, Mixtral, Yi, Qwen — all decoder-only dense transformers with **the same attention substrate**. The "universality" claim is over different *training regimes* (pretrain, length-extension, SFT, RLHF, MoE upcycling) on **dense** attention. The claim that the instrument transfers to NSA/MoBA — which have *different attention substrates* (gated 3-branch / top-k block) — is **not** something arXiv:2404.15574 supports. The hypothesis assumes transferability; that assumption is itself a research question.

### 4.2 M2 (compression branch as load-bearing fallback channel) — direction correct, magnitude unjustified

NSA's compression branch is real (verified, §3.3 above). It pools tokens within a block before attending. But for a single-token needle in a 32-64K context, pooled at compression-block resolution, the "weak-but-nonzero" signal claim is qualitatively correct but quantitatively unspecified. R4 acknowledges this; the hypothesis does not.

A finite version of M2 should specify: under NSA's reported compression block size (typically 32 tokens per compression unit per their config), what fraction of needle-token information survives pooling? And: at what context length does the pooled signal drop below a recoverable threshold? Without this, M2 is a "in principle there is signal, in practice we'll see" claim — which is fine *if* the falsification criterion is set to fail when the signal isn't enough. F1 (Δretention ≥ 0.05) does this implicitly, so M2's vagueness is recoverable. (Important.)

### 4.3 M3 (recursion refines the query, re-engaging the substrate) — most fragile

The TRM recursion update is `z = net(x, y, z)`. For a long-context transformer, the analog is not specified. Three plausible operationalizations:
- (a) **Output-conditioned recursion**: feed the previous pass's final hidden state back as additional context. This re-uses the dense KV-cache and *does* refine the query distribution at each pass.
- (b) **Latent-state recursion**: maintain a separate latent state `z` that gets updated and broadcast. Closer to TRM but requires architectural modification.
- (c) **Naive output-replay**: re-feed the model's own output as new input. This is what the cheap kill-test in §5a actually proposes ("just re-feed the latent answer state through the same model").

Only (a) and (b) plausibly produce *different attention queries* at pass 2 vs pass 1. (c) does not change the queries at all — and (c) is what the cheap test does. **The cheap test cannot test M3.** This is a critical incoherence in the hypothesis: the cheap kill-test does not actually exercise the mechanism it is meant to falsify. (Critical.)

### 4.4 Quest prediction (intermediate recovery 0.02-0.05) — ad hoc

Quest (arXiv:2406.10774) selects top-k pages query-aware, with no compression branch. Per the stated mechanism, Quest should behave like MoBA: g=0 for non-selected pages, no fallback channel, no information path for recursion to use. The hypothesis predicts intermediate recovery, justified by "query-aware selection benefits more from refined queries than block-static MoBA."

This argument is plausible at the *selection-policy* level: if the query is sharper at pass 2, Quest's selector picks better pages, while MoBA's may not. But this means **query-refinement alone — without a compression-channel fallback — can produce recovery**. If so, the load-bearing claim ("compression branch is what enables recovery") is at least partially wrong: query refinement is a separate mechanism that contributes independently. The mechanism story should be reframed as "two channels (compression-branch signal + query refinement) together drive recovery," with predictions disentangled. As stated, NSA's expected recovery (≥0.10) and Quest's (0.02-0.05) cannot both follow from M2 alone. (Critical.)

### 4.5 Cross-architecture retrieval-head identity — undefined

The hypothesis defines `retention(k) = (heads with retrieval score ≥ 0.1 at pass k *that were retrieval heads in the dense baseline*) / (number of dense-baseline retrieval heads)`. This explicitly indexes against the *dense* model's head set.

In a post-hoc swap (the cheap test), the layer/head indices are aligned because the weights are unchanged — only the attention mask differs. The metric is well-defined.

In a natively-trained NSA model (the full eval), the layer/head indices may align by construction (if NSA's architecture preserves head count per layer), but the *function* of each head is determined by training. There is no reason to expect head (L=12, h=4) in NSA-pretrained to be the same retrieval head as (L=12, h=4) in dense-pretrained. The metric as defined may produce near-zero retention in NSA simply because retrieval has been re-allocated to *different heads* — not because retrieval-head function was lost.

Fix: define an architecture-specific retrieval-head set (re-run the 600-instance protocol on each architecture independently), then measure retention *within architecture* across K. The cross-architecture comparison should be at the *task-level* (NoLiMa accuracy), not at the *head-set-overlap* level. (Critical.)

---

## 5. Falsifiability assessment

| Criterion | Operationalizable? | Notes |
|---|---|---|
| F1: Δretention(NSA, K=4 vs K=1) ≥ 0.05 abs over 5 seeds | **Yes**, but threshold is uncalibrated to noise. arXiv:2404.15574's 600-instance protocol gives stable averages "after a few samples"; the variance across seeds at the full protocol scale is unspecified, and at 50-sample (cheap test) scale, variance grows by factor √12. Threshold ≥ 0.03 on 50 samples is likely **at or below** the 1-sigma noise floor. (Important.) |
| F2: differential ≥ +0.05 abs | **Yes**, well-defined contrast. Same noise concern at 50-sample scale. Threshold appears reasonable for full-protocol scale. |
| F3: NoLiMa Δaccuracy ≥ 3 pts at 32K | **Yes**, NoLiMa scores are well-defined. 3-point absolute is plausibly above noise (NoLiMa reports 5-10 point gaps between configurations). |
| F4: top-k at pass 2 covers needle > 1.05× pass 1 | **Operationalizable but tautological as a mechanism check.** F4 fires only if F1 holds without F4 — i.e., if recovery happens without the predicted top-k re-selection. This is more a "consistency" check than an independent mechanism test. A genuine mechanism check would zero out the compression-branch contribution at pass 1 and measure whether recovery still occurs (this is the §6 ablation (a), which is stronger than F4 and should be promoted to a falsification path). (Suggestion.) |

The cheap test's threshold (Δ ≥ 0.03 on 50 NoLiMa samples) is the weakest link. With 50 samples, retention scores are integer-valued / very-low-resolution averages over a small head set — variance can easily exceed 0.03 absolute. The kill condition risks being noise-dominated. (Important.)

---

## 6. Strongest counter-argument (steelman)

**The strongest case that H3 is wrong:** retrieval-head function is not "reconstructible" by recursion in the way M3 suggests. Retrieval heads work by sharp argmax concentration on a single needle position. Recursion's update step (whether TRM-style or output-replay) typically *smears* probability mass — each pass blends the previous pass's information distribution into the next, so the argmax becomes flatter, not sharper. In the limit K → ∞, recursion converges to a distribution that is the fixed point of the update operator; that fixed point is not generally the sharp delta-function on the needle position that defines a retrieval head.

The hypothesis flags this as R3 ("recursion may destroy retrieval heads"), but does not engage with the prior literature on it. arXiv:2503.10799 (Fixed-Point RNNs) and arXiv:2601.10679 (HRM critique) both argue that depthwise recursion converges to fixed points that are *averaged* over the substrate's behaviors. arXiv:2502.05171 (Huginn) shows that loop-recurrent transformers exhibit *worse* needle-retrieval than their non-recurrent baselines at long context.

If the steelman is right, the prediction inverts: NSA + recursion should *worsen* retention with K, MoBA + recursion should *also* worsen, and the differential could even go in the wrong direction (NSA worsens *more* because there's more compression-branch noise to smear). The hypothesis's positive-effect prediction relies on the recursion *sharpening* rather than smearing the query; this is asserted but not supported by direct prior art.

The strongest defense: TRM specifically demonstrates *sharpening* on Sudoku/Maze/ARC (where the answer is a discrete grid). But puzzle answer-sharpening over a small grid is structurally different from needle-position sharpening over a 32K-token context. The transferability is the load-bearing assumption.

**The hypothesis-smith should engage R3 more substantively in the revision** — specifically by citing the Huginn smearing observation (or its absence) and committing to a falsification path that distinguishes "smearing" from "no signal." Otherwise R3 is a hand-wave on the strongest objection.

---

## 7. Severity-tagged objections

### Critical (must fix)

- **C1.** The cheap kill-test (§5a) does not exercise M3. Output-replay through the same model does not change attention queries at pass 2; it cannot test the "refined query re-engages substrate" mechanism. Either rewrite the cheap test to use latent-state injection (closer to TRM operationalization (b)) or acknowledge that the cheap test only falsifies the *output*-side correlation, not the mechanism, and adjust the kill condition's interpretation accordingly.

- **C2.** Cross-architecture retrieval-head identity is undefined. The retention metric as written assumes head-index alignment across architectures, which is invalid for natively-trained NSA/MoBA. Specify: (i) for the post-hoc swap (cheap test), use dense-baseline head indices; (ii) for the full eval, re-identify retrieval heads independently per architecture and compare *within-architecture* retention as a function of K, with the cross-architecture comparison happening at the task-level (NoLiMa) only.

- **C3.** Quest prediction is ad hoc relative to the stated mechanism. M2 says "compression branch is the fallback channel"; Quest has no compression branch; M2 alone predicts MoBA-class behavior. Either (a) reframe the mechanism as multi-channel (compression-signal + query-refinement) with predictions decomposed, or (b) drop Quest from the prediction and treat it as exploratory. As written, the Quest prediction *contradicts* the stated mechanism.

- **C4.** TRM operationalization on a long-context transformer is non-trivial and unspecified. TRM's `(x, y, z)` recursion is designed for puzzle-shaped answer tensors; the hypothesis treats applying it to a 1-3B-param transformer with 32-64K context as a turn-key wrapper. Specify: what is `y`? what is `z`? where does the latent state live in the transformer's residual stream? This is a research design step, not a known recipe.

### Important (should fix)

- **I1.** Retrieval-head metric requires re-operationalization for NSA's three-branch attention. The argmax over which softmax? Specify the choice and justify it. (Probably: argmax over the gated mixture's effective key set, i.e., compute the per-key contribution to the output and take the position with maximum contribution.)

- **I2.** Cheap-test noise floor is uncalibrated. With 50 NoLiMa samples vs 600-instance protocol, sigma scales by √12 ≈ 3.46. The 0.03 threshold is plausibly below 1-sigma. Either (a) increase sample count to ≥150, (b) raise the threshold, or (c) explicitly compute and report the noise floor under the cheap-test protocol and adjust threshold accordingly.

- **I3.** M2 is qualitative on signal strength. NSA's compression block size (per their reported config, typically 32 tokens) determines how much per-token information survives pooling. Add a quantitative version: for context length L, block size B, the per-needle signal-to-noise after compression is ~1/B. Specify the regime (B, L) where this is or isn't sufficient, and pre-register the falsification regime accordingly.

- **I4.** Competing prior-art treatment of arXiv:2602.11374 (Gather-and-Aggregate). The hypothesis cites G&A as supporting evidence, but it is also a *competing approach*: G&A heads can be explicitly preserved by distillation, no recursion needed. The hypothesis must address: why is recursion needed if G&A heads can be preserved by simpler means? A reasonable answer exists (G&A preservation works for Transformer→SSM; the question is preservation under sparse-attention pretraining, where the mechanism may differ), but this needs to be stated.

- **I5.** R3 (recursion may destroy retrieval heads) needs substantive engagement. Cite Huginn (2502.05171) or Fixed-Point RNNs (2503.10799) on smearing under depthwise recursion, and commit to a measurement (head-level argmax sharpness as a function of K) that distinguishes sharpening from smearing. Currently R3 is a single sentence; it is the strongest objection and deserves more.

### Suggestion (nice to have)

- **S1.** Citation-locator §3 M2: the three-branch architectural diagram is in NSA §3, not §2. Minor.

- **S2.** F4 should be replaced or supplemented by the §6 compression-branch ablation (a), which is a stronger mechanism check.

- **S3.** Consider arXiv:2512.24601 (Recursive Language Models, 96 upvotes, Dec 2025) and arXiv:2603.15653 (SRLM) as related work. Both treat long-context-as-environment programmatically — different from architectural recursion (and thus not direct competitors), but they should be cited as adjacent prior art and their non-overlap with the proposed approach explained.

- **S4.** The "intermediate-Quest" prediction (if retained) should specify a directional ordering: NSA > Quest > MoBA ≈ DSA. Currently it's a magnitude prediction; an ordering is more robustly testable.

- **S5.** F4's "≤ 1.05×" is awkward. State as "if the conditional needle-coverage rate at pass 2 minus pass 1 is < 0.05 absolute, M3 is wrong." Same content, clearer threshold.

### Counts
- Critical: 4
- Important: 5
- Suggestion: 5

---

## 8. Recommendation to hypothesis-smith

The hypothesis is conceptually well-formed, addresses a real gap, and has the structure of a falsifiable scientific claim. The revision must address:

1. **(C1, C4)** Specify the recursion operationalization concretely. What does "TRM-style outer loop on the latent answer state" mean for a 1-3B transformer at 32-64K context? Pick one of: (a) output-conditioned re-feed (then admit the cheap test exercises this), (b) latent-state injection at a specific layer (then specify the layer and the state shape), (c) full TRM-style architecture replacement (then admit this is heavier than YAGNI permits). Make the cheap kill-test consistent with the chosen operationalization.

2. **(C2)** Re-define retention to be within-architecture. Move cross-architecture comparisons to the task-level metric (NoLiMa).

3. **(C3)** Either decompose the mechanism into compression-channel + query-refinement and predict each contribution separately, or drop Quest from the prediction. The current Quest prediction is internally inconsistent with M2.

4. **(I2, I5)** Calibrate the noise floor for the cheap test (or remove it). Engage R3 with citations and a sharpness-vs-smearing measurement.

5. **(I4)** Position arXiv:2602.11374 (G&A) as competing baseline, not co-evidence. State why recursion is needed when G&A distillation already preserves retrieval heads.

If the hypothesis-smith addresses C1-C4 plus I2 and I4-I5, this becomes a strong, defensible hypothesis. The core insight — that NSA's compression branch and MoBA's lack thereof should produce a recursion-recovery asymmetry — is novel, falsifiable, and well-grounded in cited mechanism. The revision is bounded and tractable.

If the revision still cannot disentangle "compression channel" from "query refinement" (C3) and cannot specify the recursion operationalization (C1, C4), the hypothesis collapses into "NSA is better than MoBA at long context, and recursion may help" — which is neither novel nor mechanistically interesting. So those two are the most important to fix.

---

APPROVE | **REJECT (revision-1)** | KILL (irrecoverable)

REJECT (revision-1)
