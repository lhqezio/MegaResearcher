# Hypothesis H3 (revision 1) — Architectural recursion plus an all-tokens fallback channel produces a non-additive recovery of long-context retrieval, decomposable into a compression-signal effect and a query-refinement effect

## Revision response (red-team round 1)

This revision addresses the four Critical objections (C1, C2, C3, C4) and the strongest counter-argument (R3 / I5). Important and Suggestion items are absorbed inline. Where the red-team was right, the hypothesis is changed; where I disagree, the disagreement is grounded in citation.

**C1 — "Cheap kill-test inconsistent with mechanism."** Accepted. The original cheap test re-fed the latent answer state through the same model with no change to the attention queries, so it could not exercise M3 (refined-query mechanism). I have replaced the cheap kill-test with a concrete operationalization (Option (a) — output-conditioned re-feed) that *does* change the query distribution at pass 2: at pass k+1, the model's pass-k draft answer is appended to the input as an "answer-so-far" prefix and the full forward pass is rerun. Because the input sequence at pass k+1 differs from pass k by the appended draft, the query vectors at every position computed at pass k+1 differ from those at pass k. This is the minimal, training-free instantiation of TRM-style recursion that is *guaranteed* to perturb the query distribution; it does not require architectural modification to the backbone. (See §3 M3 below for the explicit mapping of TRM's `(x, y, z)` onto a long-context transformer.)

**C2 — "Retention metric assumes head-index alignment across architectures."** Accepted. The original metric — "fraction of *dense*-baseline retrieval heads still firing in the sparse variant" — is invalid for natively-trained NSA/MoBA models, where retraining redistributes head function across indices. Revised metric is **within-architecture retention**: independently re-run the 600-instance protocol of arXiv:2404.15574 §2 on each architecture (dense-pretrained, NSA-pretrained, MoBA-pretrained, etc.) to identify *that architecture's* retrieval-head set H_A at K=1, then measure how many of H_A still satisfy the score-≥-0.1 criterion at K=k>1. Cross-architecture comparison is moved to the task level (NoLiMa accuracy and EverMemBench-S evidence-access). This eliminates the head-index-alignment confound and tracks the structurally meaningful question: "does recursion *preserve* an architecture's own retrieval circuitry across passes?"

**C3 — "Quest prediction contradicts mechanism."** Accepted. The original prediction placed Quest as "intermediate" recovery without justifying why a no-compression-channel design should recover at all. I have decomposed M2 into two cited sub-mechanisms:

- **M2a (compression-signal channel).** NSA's compressed-attention branch passes a coarse-grained signal from *every* block (arXiv:2502.11089 §3, Eq. 7-9). At pass 1 even un-selected blocks contribute weak nonzero information to the head output. Without this channel (MoBA, DSA, Quest), un-selected blocks are zeroed at pass 1.
- **M2b (query-refinement channel).** TRM-style recursion alters the input fed to the model at pass k+1 (output-conditioned re-feed; see M3), producing a different query distribution at every layer at pass k+1 than at pass k. This re-runs *the selection step itself*, regardless of whether a compression channel exists.

These are independent additive contributions to retention recovery. NSA gets both M2a and M2b; Quest gets only M2b (its top-k re-selects on a refined query but un-selected pages were zeroed at pass 1, so re-selection has no signal residue to amplify); MoBA gets effectively only M2b but with a coarser block-static gate that is *less* responsive to query refinement (arXiv:2502.13189 §2.2 — gating is at block level by mean-pooled key, so a small query change must change *which* block is in top-k to have any effect; Quest's per-page query-aware top-k arXiv:2406.10774 §3 has finer granularity and is more responsive to query refinement); DSA's lightning-indexer (arXiv:2512.02556) is between MoBA and Quest. The revised ordering prediction — **NSA > Quest ≥ DSA > MoBA at K=4 retention recovery** — follows from the decomposition.

**C4 — "TRM puzzle-shape recursion mapping unspecified."** Accepted. TRM's `(x, y, z)` tensors are puzzle-grid-shaped and cannot be ported directly to a 32-64K-token transformer. I have replaced the under-specified "TRM-style outer loop" with an explicit mapping (§3 M3, "TRM operationalization on a long-context transformer"). In short: `x` = the original prompt token IDs; `y` = the model's draft answer tokens (the span generated at the answer position, of small fixed length L_y, e.g., 32-128 tokens); `z` = the final-layer hidden states at the answer-span positions, shape (L_y, d_model). The recursion `z = net(x, y, z), y = net(y, z)` becomes `(y, z) = full_forward(concat(x, y_prev_tokens))` re-running the *unmodified* backbone with the previous pass's draft answer appended to the prompt. This is option (a) from the red-team's analysis (output-conditioned re-feed), which the red-team itself notes "plausibly produces different attention queries at pass 2 vs pass 1." It is YAGNI-compliant (no parameters added, no training, no architectural modification) and operationally identical to the cheap kill-test, so the cheap test now does exercise the mechanism. The cost is that this is a *weaker* form of TRM than the published version (no separate latent z-stream, just answer-span re-feed), and the hypothesis therefore makes a *correspondingly weaker* prediction (see magnitudes below — predicted recovery is reduced from the original).

**R3 / I5 — "Recursion may smear rather than sharpen."** Engaged substantively in §3 M3 below and in §7 (Risks). The red-team is right that arXiv:2502.05171 (Huginn / Recurrent-Depth) shows loop-recurrent transformers can degrade on needle-style retrieval, and arXiv:2503.10799 (Fixed-Point RNNs) frames depthwise recursion as fixed-point iteration where the limit is the attractor of the update operator, not generally a sharp delta. **I do not have direct prior-art support for the claim that output-conditioned re-feed *sharpens* retrieval-head argmaxes.** TRM's sharpening evidence is on small puzzle grids (arXiv:2510.04871 §4) and does not transfer mechanistically to 32K-token attention. I therefore (i) downgrade the directional prediction's confidence — the hypothesis's *primary* prediction is now the **NSA-vs-MoBA differential** (F2), not the absolute monotonicity (F1), because the differential survives even if both architectures smear, as long as NSA smears *less* due to compression-branch resilience; (ii) add an explicit sharpness probe (entropy of the retrieval head's attention distribution as a function of K) so a smearing-not-sharpening outcome can be diagnosed even when retention drops; (iii) accept that the directional sign of the absolute K-effect is uncertain and let F2 (differential) carry the load. This is a real concession to the red-team and weakens the hypothesis from "recursion improves retention" to "recursion's effect on retention is *more positive (or less negative)* under NSA than under MoBA." That weaker prediction is still novel, mechanistically grounded, and falsifiable.

**Important fixes absorbed.** I1: argmax over the gated-mixture's effective per-key contribution (per-key attribution of the head output, take argmax position) — specified in §4 below. I2: cheap test sample size raised to 200 NoLiMa items with retention-noise-floor pre-computation; threshold raised correspondingly. I3: M2a now states quantitative compression-pooling regime (NSA compression block size = 32 per arXiv:2502.11089 §3 Eq. 7; per-needle SNR ≈ 1/32 at the compression-block level). I4: G&A heads (arXiv:2602.11374) repositioned from "co-evidence" to "competing baseline / orthogonal mechanism" — see §3 M2 footnote and §6 ablation. I5: see above. S1, S2, S4, S5 incorporated.

**Suggestions absorbed but worth flagging.** S3: arXiv:2512.24601 (Recursive Language Models) and arXiv:2603.15653 (SRLM) are programmatic / inference-time recursion, not architectural — cited in §1 to clarify non-overlap.

## Changes from revision-0

- **Hypothesis statement (§2)**: primary prediction switched from "absolute monotonic NSA recovery" to "NSA-vs-MoBA differential is positive at K=4 minus K=1." Quest replaced with explicit ordering NSA > Quest ≥ DSA > MoBA. Magnitudes adjusted for output-conditioned re-feed (weaker than full TRM).
- **Mechanism (§3)**: M2 split into M2a (compression-signal) + M2b (query-refinement); M3 made explicit with TRM-to-transformer mapping.
- **Predicted outcome (§4)**: within-architecture metric defined; magnitudes adjusted; ordering prediction added; sharpness probe added.
- **Falsification (§5)**: F2 promoted to primary; F1 downgraded to "consistency check"; new F5 (ordering); F4 replaced by promoted compression-branch ablation; sharpness probe (for distinguishing smearing-from-no-signal) added.
- **Cheap path (§5a)**: now uses output-conditioned re-feed (consistent with M3); sample size 200; threshold recomputed.
- **Risks (§7)**: R3 expanded with Huginn/Fixed-Point-RNN citations and concrete contingency.
- **Sources (§8)**: added arXiv:2502.05171, arXiv:2503.10799.

---

## 1. Targeted gap

This hypothesis addresses **Gap 9 from gap-finder-2** (`/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/gap-finder-2/output.md#9`), same intersection captured by **Gap 8 in gap-finder-1** (`/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/gap-finder-1/output.md#41` ff., the "Retrieval-head behavior under sparse attention is uncharacterized" gap). Compositional anchor: A8 (retrieval-head structure) × B9 (sparse attention × recursion-depth instrument).

Restated: Retrieval Head (arXiv:2404.15574) shows a small, sparse, intrinsic set of attention heads is mechanistically responsible for arbitrary-position fact retrieval (§3.1, §4.1, §4.3). Retrieval-Aware Distillation for Transformer-SSM Hybrids (arXiv:2602.11374) confirms retrieval-head function is fragile under naive sub-quadratic conversion — its "Gather-and-Aggregate" heads must be explicitly preserved when distilling Transformers into SSMs. RazorAttention (arXiv:2407.15891) further confirms retrieval-head specialness by exploiting it for KV compression. NoLiMa (arXiv:2502.05167; non-commercial license — flagged) and Hyper-multi-step (arXiv:2410.04422) extend the picture to multi-step retrieval. Sparse Frontier (arXiv:2504.17768) ablates sparse-attention patterns on retrieval/reasoning workloads but does not measure retrieval-head retention; SeerAttention-R (arXiv:2506.08889) self-distills sparse gating for reasoning models but does not test retrieval-head behavior across recursive loop iterations. Programmatic / inference-time recursion (Recursive Language Models arXiv:2512.24601; SRLM arXiv:2603.15653) is orthogonal — it treats long-context as a queryable environment via tool-use, not via depthwise architectural recursion in the forward pass — and therefore does not address this gap. No paper has measured retrieval-head retention as a function of (sparse-attention pattern × architectural-recursion depth K).

## 2. Hypothesis statement

**If** a TRM-style architectural recursion is operationalized as output-conditioned re-feed (K passes within a forward sequence — the model's pass-k draft answer is appended to the input prompt and the unmodified backbone is rerun, per the explicit mapping in §3 M3) layered on a transformer backbone whose attention has been replaced with NSA (arXiv:2502.11089, retains compressed-attention + selected-attention + sliding-attention branches), **then** the *differential* in within-architecture retrieval-head retention recovery between NSA and MoBA (arXiv:2502.13189) at K=4 minus K=1 will be positive and at least 0.05 absolute, in favor of NSA. Formally:

```
[ retention_NSA(K=4) − retention_NSA(K=1) ]  −  [ retention_MoBA(K=4) − retention_MoBA(K=1) ]   ≥   +0.05
```

where `retention_A(k)` is computed within architecture A using A's own retrieval-head set (re-identified per arXiv:2404.15574 §2 protocol on each architecture).

**Secondary directional prediction (ordering).** At K=4 within-architecture retention recovery, ranked: **NSA > Quest ≥ DSA > MoBA**. Quest (arXiv:2406.10774) gets the query-refinement effect (M2b) without the compression-signal effect (M2a); DSA (arXiv:2512.02556) similar; MoBA gets the smallest query-refinement effect because its block-mean-pool gating is least responsive to query change.

**Cross-architecture (task-level) prediction.** On NoLiMa (arXiv:2502.05167) at 32K context, NSA + K=4 minus NSA + K=1 accuracy delta exceeds MoBA + K=4 minus MoBA + K=1 accuracy delta by ≥ 3 percentage points.

This is a non-additive, conditional prediction: the *interaction* between recursion and sparsity-pattern is the load-bearing claim, not either factor alone. The magnitude and sign of the absolute monotonic K-effect on either architecture is *not* the primary prediction — that effect could in principle be negative under recursion-induced smearing (R3), but the differential should still favor NSA.

## 3. Mechanism

The mechanism rests on three cited claims, none speculative; M2 is decomposed into M2a (compression-signal) + M2b (query-refinement) per the red-team's C3 objection.

**Claim M1 — retrieval heads are a copy-paste circuit identifiable by attention-mass concentration on input-position needles.** arXiv:2404.15574 §2 defines retrieval score as the frequency with which a head's argmax attention probability lands on the needle token whose value is then emitted; threshold = 0.1; same heads appear universally across model families (arXiv:2404.15574, Table 1, §3.1). The instrument is *not* assumed to transfer across architectures with different attention substrates — the revised metric (§4) re-runs the protocol per architecture.

**Claim M2a — compression-signal channel.** NSA (arXiv:2502.11089 §3, Eq. 7-9) defines three parallel branches gated together: (i) compressed-attention `Attn(q, K_cmp, V_cmp)` where `K_cmp = phi(K_blocks)` is a learned per-block pooling (§3 Eq. 7), with reported block size ≈ 32 tokens (§3 / config); (ii) selected-attention over top-k blocks; (iii) sliding-window-attention. All three are summed via learned gates `g_cmp + g_sel + g_swa = 1`. Therefore information about *every* block (including non-top-k blocks) reaches the head output via path (i), at compression-block resolution — per-needle SNR ≈ 1/(block-size) = 1/32 in the worst case where the needle is one of 32 pooled tokens. By contrast, MoBA (arXiv:2502.13189 §2.2 Eq. 5) sets `g_i = 0` for blocks outside top-k, with no compression / pooling / fallback path — un-selected blocks contribute exactly zero. Quest (arXiv:2406.10774 §3) selects top-k *pages* using a per-page max-key heuristic but has no compression branch — un-selected pages contribute zero. DSA / lightning-indexer (arXiv:2512.02556) similar: indexed top-k routing without compression branch.

**Claim M2b — query-refinement channel (independent of M2a).** Output-conditioned re-feed (§3 M3 below) changes the input prompt at pass k+1 by appending the pass-k draft answer. Every transformer layer's queries at pass k+1 are computed from a different residual stream than at pass k (because the position-(L_x + i) tokens are now the draft-answer tokens, not absent). This re-runs the *selection step* (whatever it is for the architecture: top-k blocks for MoBA/NSA-selected-branch, top-k pages for Quest, lightning-indexer top-k for DSA) at pass k+1 with a *different* query distribution. **Key point:** M2b operates on whatever sparse-selection mechanism the architecture has; it does not depend on a compression channel. M2b is what allows Quest, DSA, and even MoBA to exhibit *some* recovery at K>1.

**Claim M3 — TRM-style recursion mapped onto a long-context transformer.**

TRM's recursion as published (arXiv:2510.04871 §3):
```python
def latent_recursion(x, y, z, n=6):
    for i in range(n):
        z = net(x, y, z)
    y = net(y, z)
    return y, z
```
where `x` = puzzle inputs, `y` = current answer grid, `z` = latent reasoning state (same shape as `y`).

For a long-context transformer doing retrieval at 32-64K context, the explicit mapping is:

- `x` ← original prompt token IDs (length L_x, the question + context).
- `y` ← the model's draft answer tokens at pass k (length L_y, e.g., 32-128 tokens). At pass 1, `y_0` is the empty string (model generates from `x` alone).
- `z` ← the final-layer hidden states at the answer-span positions of pass k, shape (L_y, d_model). Implicit in the forward pass; not separately stored unless used for analysis.
- `net(x, y, z)` ← `full_forward(concat(x, "\\nAnswer-so-far: ", y_tokens, "\\n"))`. The unmodified backbone is rerun on the concatenated sequence; the new draft answer at the (extended) answer position is the `y_{k+1}`.

Equivalently in the TRM grammar: the for-loop `z = net(x, y, z)` is replaced with `y, z = full_forward(concat(x, "Answer-so-far: ", y))`. The `z` update happens implicitly inside the forward pass; the answer update `y = net(y, z)` is the final-step generation. This is option (a) from the red-team's analysis (output-conditioned re-feed). It is the **weakest** form of TRM-style recursion — it does not maintain a separate latent stream, so the recursion only refines the query through the input-token modification — but it has three advantages:

1. **Implementable on any pretrained transformer with no architectural change** (YAGNI fence).
2. **Guaranteed to change the query distribution** at every layer at pass k+1 (because the input sequence is different).
3. **The cheap kill-test (§5a) and the full eval use the same operationalization**, so the cheap test exercises M3, not just M2a (resolves C1).

**Why a stronger recursion (option (b) with separate latent z-stream) is not used in this revision.** Option (b) requires architectural modification (an injected latent stream broadcast at a chosen layer) and out-of-distribution training to make use of it — both are outside the YAGNI fence and would conflate "TRM works" with "the revised architecture works." Output-conditioned re-feed is parameter-free and architecture-free.

**Composition (M1 + M2a + M2b + M3 → predicted outcome).**

- **NSA + recursion**: at pass 1, top-k may miss the needle block, but compressed-attention (M2a) carries a coarse signal (~1/32 SNR) into the residual stream at the answer span. At pass 2, the appended draft answer (which contains some echo of the coarse signal — a partial-match output, or even a confident-wrong output that is *informative* about the model's residual-stream state) shifts the queries (M2b). Top-k selection at pass 2 may now include the needle block; selected-attention now contributes full-resolution signal. Retention recovers.
- **MoBA + recursion**: at pass 1, the needle block is either selected (retrieval succeeds, no recovery needed) or not selected (zero contribution to head output, M2a fails). At pass 2, the draft answer reflects only what was selected at pass 1; query refinement (M2b) operates on a residual stream that has no echo of the missed block. The new top-k may *happen* to land on the needle block (some recovery), but only by chance or by general distractor-suppression (the draft answer's confident-wrong content shifts queries away from distractors). Recovery is bounded by query-refinement-only contribution.
- **Quest + recursion**: similar to MoBA on the M2a axis (no compression channel), but Quest's per-page query-aware top-k (arXiv:2406.10774 §3, Eq. 5) is more responsive to query change than MoBA's mean-pool gating, so M2b contributes more. Intermediate recovery.
- **DSA + recursion**: similar to Quest; lightning-indexer is a learned top-k selector, slightly more responsive to query refinement than MoBA's static block-mean-pool but lacks query-aware page-level granularity of Quest.

**Mechanism for steelman R3 (smearing). Engaged.** Huginn (arXiv:2502.05171) shows that loop-recurrent transformers can underperform on needle-style retrieval relative to non-recurrent baselines at long context. Fixed-Point RNNs (arXiv:2503.10799) formalizes depthwise recursion as fixed-point iteration where the attractor depends on the operator's spectral structure, with no a priori reason for the attractor to be a sharp delta on the needle position. Output-conditioned re-feed at K=2 may produce a draft answer whose content is *less* sharply localized than the K=1 output; if the K=2 query is computed from this draft, the queries are flatter, and retrieval-head argmaxes flatten. **The hypothesis does not assume sharpening occurs; it assumes the smearing is *less severe* under NSA than under MoBA, because NSA's compression channel (M2a) provides a non-zero signal residue that anchors the query refinement.** The directional prediction (F1) is downgraded to a consistency check. The differential prediction (F2) carries the load and is robust to symmetric smearing.

**Note on G&A (arXiv:2602.11374).** Repositioned. G&A heads can be explicitly preserved by distillation; that is a *competing approach* to retrieval-head retention, not co-evidence. The hypothesis is asking a different question: *given* a sparse-attention substrate (NSA/MoBA) where retrieval-head function may be fragile, can architectural recursion recover it? G&A distillation requires (i) a known-good retrieval-head set in a teacher and (ii) explicit preservation supervision; recursion proposes to recover it without supervision, by exploiting M2a + M2b. The two are alternatives, not co-supports. The hypothesis is preferred when distillation is unavailable (e.g., training NSA from scratch).

## 4. Predicted outcome with magnitude

**Primary metric.** Within-architecture retrieval-head retention. For each architecture A ∈ {dense, NSA, MoBA, Quest, DSA}, independently run the arXiv:2404.15574 §2 protocol (≈ 600 instances, 10 depths × 20 lengths) at K=1 to identify A's retrieval-head set H_A := {heads with retrieval score ≥ 0.1 at K=1 on A}. Then at K ∈ {2, 4, 8}, measure `retention_A(K) := |{h ∈ H_A : score(h, K) ≥ 0.1}| / |H_A|`. By construction `retention_A(1) = 1.0`.

**Argmax-over-NSA-mixture (resolves I1).** For NSA's gated three-branch attention, the head's "argmax position" is defined as `argmax_p Σ_branches g_branch · attn_branch(q, position p)` — the position whose total contribution to the gated head output (across all three branches) is maximal. This collapses the gated mixture into a single per-key attribution and makes the metric well-defined.

**Sharpness probe (R3 diagnostic).** At each K and architecture, also compute the entropy `H(retrieval-head attention distribution)` averaged over the head set H_A and over needles. If retention drops with K but entropy *also* increases, the cause is smearing (R3). If retention drops with K but entropy stays flat, the cause is selection failure (M2a/M2b not delivering signal).

**Conditions where the hypothesis must hold:**
- Backbones: pretrained models in the 1-3B parameter range, one per attention scheme. NSA from arXiv:2502.11089's released checkpoints if available, else trained-from-scratch matched to the published recipe; same for MoBA (arXiv:2502.13189 released) and Quest (arXiv:2406.10774 — applied post-hoc on a dense base, since Quest is inference-only). DSA from arXiv:2512.02556 (DeepSeek-V3.2 lightning-indexer scaled down).
- Sparsity: ratio held constant at ≈ 0.10 across architectures where adjustable.
- Context length: 32K-64K.
- Recursion: output-conditioned re-feed (§3 M3), K ∈ {1, 2, 4, 8}, no extra training.
- Probes: NoLiMa (license-flagged); EverMemBench-S / Beyond the Needle's Illusion (arXiv:2601.20276); synthetic NIAH per arXiv:2404.15574 §2.

**Predicted magnitudes (revised down for output-conditioned re-feed).**

Within-architecture retention at K=4 vs K=1, expected absolute recovery:

| Arch | K=1 (by def.) | K=4 (predicted) | Recovery (K=4 − K=1) |
|------|---------------|-----------------|----------------------|
| Dense (full attention) | 1.0 | 0.95-1.0 | ≈ 0 (no headroom) |
| NSA | 1.0 | 0.92-1.0 | +0.0 to +0.08 (from M2a + M2b) |
| Quest | 1.0 | 0.85-0.95 | -0.05 to +0.05 (M2b only) |
| DSA | 1.0 | 0.83-0.95 | -0.07 to +0.05 (M2b only) |
| MoBA | 1.0 | 0.80-0.90 | -0.10 to +0.0 (M2b weakest) |

**Note on the negative recovery values.** Because retention(K=1) = 1 by construction (heads identified at K=1), retention can only decrease or stay at 1.0 with K. The M2a + M2b mechanism predicts that NSA loses retention more *slowly* with K than MoBA does. The differential prediction is therefore:

```
F2_primary:    [retention_NSA(4) − retention_NSA(1)]  −  [retention_MoBA(4) − retention_MoBA(1)]   ≥   +0.05 absolute
```

Equivalently: NSA retains at least 5% more of its initial retrieval-head set at K=4 than MoBA retains of its initial set.

**Cross-architecture task-level prediction (NoLiMa).** Identifying retrieval heads at K=1 sets a per-architecture floor, but the task-level signal is what matters. NoLiMa accuracy at 32K context, K=4 minus K=1, ranked: NSA delta > MoBA delta by ≥ 3 percentage points. (3 points is plausibly above noise — NoLiMa reports 5-10 point gaps between configurations, arXiv:2502.05167.)

**Conditions under which the hypothesis must NOT hold (negative predictions):**
- Sparsity ratio = 1.0 (full attention) on NSA: K should not improve retention (no signal was lost; M2a is redundant).
- Single-step retrieval at distance < sliding-window size: NSA's sliding branch already solves it on pass 1; K should not help.
- Non-retrieval task (in-distribution short-context perplexity): no sparsity × K interaction; retention undefined.

## 5. Falsification criteria

We require **three independent falsification paths**, each with a metric, threshold, and direction. Metric design now reflects the within-architecture redefinition (C2) and the differential-as-primary shift (R3).

**F2 (PRIMARY — differential, the non-additive prediction).** *Metric:* `[retention_NSA(K=4) − retention_NSA(K=1)] − [retention_MoBA(K=4) − retention_MoBA(K=1)]`, within-architecture as defined in §4. *Threshold:* if this differential is < +0.05 absolute averaged over five seeds (± 1 sigma), the hypothesis is falsified. *Direction:* NSA's recursion-induced retention loss must be *less negative* (or recovery more positive) than MoBA's by at least 5 absolute percentage points at K=4 vs K=1. This is the load-bearing prediction; it survives even under symmetric smearing (R3) because the differential isolates the architectural-asymmetry contribution.

**F5 (ordering — secondary).** *Metric:* rank-order of `retention_A(K=4) − retention_A(K=1)` across A ∈ {NSA, Quest, DSA, MoBA}. *Threshold:* if the predicted ordering NSA > Quest ≥ DSA > MoBA is violated by more than one swap (e.g., MoBA > Quest is one swap; MoBA > NSA is two swaps and falsifies), the mechanism decomposition (M2a + M2b) is wrong. *Direction:* compression-having architectures must recover more than compression-lacking architectures; query-refinement-responsive selectors (Quest) more than block-mean-pool gates (MoBA).

**F3 (downstream task transfer).** *Metric:* NoLiMa (arXiv:2502.05167) accuracy delta and EverMemBench-S "evidence-access" metric (arXiv:2601.20276). *Threshold:* on NoLiMa at 32K context, if (NSA + K=4 − NSA + K=1) − (MoBA + K=4 − MoBA + K=1) < +3 percentage points, F3 fires. *Direction:* differential at the task level must mirror differential at the head level.

**F1 (consistency check, demoted from primary).** *Metric:* `retention_NSA(K=4) − retention_NSA(K=1)`. *Threshold:* if this is *more negative* than -0.10 absolute (NSA lost 10+ percentage points of its retrieval-head set with recursion), the recursion is doing significant harm to NSA — even if F2 still holds (i.e., MoBA was harmed even more), the absolute claim that recursion is "useful" is wrong, and the hypothesis is interestingly partially-falsified: differential exists but absolute effect is negative. *Direction:* recursion should not catastrophically destroy NSA's retrieval heads.

**F6 (mechanism check — promoted from old F4 + §6 ablation).** *Metric:* compression-branch ablation. Train (or apply post-hoc) NSA with the compressed-attention branch zeroed (gates `g_cmp = 0`), keeping only selected-attention + sliding-attention. Measure `retention_NSA-noCmp(K=4) − retention_NSA-noCmp(K=1)`. *Threshold:* this differential should approximately equal the MoBA differential (within ±0.03 absolute); if it remains as positive as full NSA's, M2a was *not* the load-bearing mechanism — recovery came from M2b alone, and the architectural-asymmetry rationale is wrong. *Direction:* turning off the compression branch must convert NSA's recovery profile to MoBA-class. This is a stronger mechanism check than the original F4 (which was tautological per the red-team's I5).

**Sharpness diagnostic (auxiliary, not a falsification path on its own).** Track entropy of retrieval-head attention distributions across K. If retention drops *and* entropy increases, smearing (R3) is the cause; if retention drops *and* entropy is flat, M2a/M2b is failing. This does not falsify the hypothesis — it diagnoses *why* a falsification fired.

If F2 fires, the hypothesis is fully falsified (the architectural-asymmetry rationale is wrong). If F5 fires, the mechanism decomposition is wrong. If F3 fires, the head-level effect doesn't propagate to task-level. If F1 fires, the absolute claim is wrong but differential may still hold (partially falsified). If F6 fires, M2a is not load-bearing.

## 5a. Cheaper falsification path (revised for C1, I2)

The full hypothesis requires four sparse-attention variants × four K values × two probes × five seeds — Phase-5-scale. For early kill:

**Cheapest test (single experiment, ≈ a small training-free run).** Take one mid-size pretrained dense transformer (e.g., a 1-3B model with retrieval heads identified per arXiv:2404.15574). Apply two post-hoc attention masks at inference:
- *NSA-emulation mask*: top-k block attention + sliding window + a coarse compressed-attention branch implemented as mean-pooled key/value over each block, gated by learned-or-fixed weights.
- *MoBA-emulation mask*: top-k block attention + sliding window only, no compression branch.

Run **output-conditioned re-feed** (§3 M3) at K=1 vs K=4 on a 200-sample NoLiMa subset (raised from 50 per I2). At each K, measure the *post-hoc* retrieval-head retention (using the dense baseline's head set, since the post-hoc swap preserves head indices — this is the one case where dense-indexed retention is well-defined). The cheap test is now consistent with the M3 mechanism: re-feed changes the input prompt at K=2, which changes the queries at every layer.

**Single-number kill condition (revised threshold).** If on this minimal setup, the differential

```
[retention_NSA-mask(K=4) − retention_NSA-mask(K=1)] − [retention_MoBA-mask(K=4) − retention_MoBA-mask(K=1)] < +0.05 absolute
```

over 200 NoLiMa samples (× 5 seeds for re-feed sampling temperature), the full Phase-5 evaluation should not be funded.

**Noise-floor calibration (resolves I2).** On 200 samples instead of 600, 1-sigma of retention-score variance scales by √(600/200) ≈ 1.73× the published protocol's per-instance variance. Per arXiv:2404.15574's §3 they report retention scores stable to ~0.01 at 600 samples, so at 200 samples 1-sigma is ~0.017. The differential 1-sigma is √2 × 0.017 ≈ 0.024. The +0.05 threshold is therefore ~2.1-sigma — adequate but not generous. To strengthen, the cheap test runs 5 seeds (re-feed sampling temperature ≠ 0 for the draft answer to inject some variance) and reports the across-seed mean ± SEM. If the lower end of the ± SEM band crosses below 0, the differential is not significant and the hypothesis is *not* killed but is "inconclusive at cheap scale" — full eval needed.

**Why post-hoc swap is informative even though imperfect.** The hypothesis claims the *information channel* (NSA's compression branch) is what enables differential recovery. Post-hoc swaps preserve channel structure even though the network was not trained to use it. If recovery cannot be detected even post-hoc, the channel-existence claim is suspect at training time too. If detected, the full Phase-5 eval under native training is justified. This argument was already in revision-0 and stands.

**Acknowledged caveat.** Output-conditioned re-feed via post-hoc attention swap is the weakest version of the recursion mechanism. If F2 differential is exactly at the noise floor at the cheap scale, this is not a strong kill — the full eval may surface a stronger effect under native training. The cheap test is for *strong negative results*, not for marginal ones.

## 6. Required experiments (sketch only — eval-designer details these)

- **Datasets.** NoLiMa (arXiv:2502.05167, license-flagged); EverMemBench-S / Beyond the Needle's Illusion (arXiv:2601.20276); synthetic NIAH per arXiv:2404.15574 §2.
- **Backbones.** Mid-size (1-3B) transformers with the following attention substrates:
  - Dense (control)
  - NSA (arXiv:2502.11089)
  - MoBA (arXiv:2502.13189)
  - Quest applied post-hoc (arXiv:2406.10774, inference-only by design)
  - DSA / lightning-indexer (arXiv:2512.02556)
  Sparsity ratio held constant. Within-architecture retrieval-head sets re-identified per arXiv:2404.15574 §2.
- **Recursion wrapper.** Output-conditioned re-feed (§3 M3) at K ∈ {1, 2, 4, 8}, no parameters added, no training (YAGNI fence).
- **Baselines.** Dense + K=1 (within-architecture floor); each sparse arch + K=1 (within-architecture baseline by definition retention = 1.0); each sparse arch + K=4 (treatment).
- **Ablations.**
  - **(a, F6)** Strip NSA's compression branch (g_cmp = 0), keep selected + sliding. Predicted: retention recovery profile collapses to MoBA-class.
  - **(b)** Replace output-conditioned re-feed with naive output-replay (re-feed final hidden states without textual draft). Predicted: no recovery on any sparse arch (M3 fails — input sequence unchanged, queries unchanged).
  - **(c)** Run G&A distillation (arXiv:2602.11374) on the same backbone independently. Used as a *competing baseline* (not co-evidence) — establishes how much retrieval-head retention is achievable without recursion, when retention is supervised explicitly.
- **Probes.**
  - Within-architecture retrieval-head retention (M1 + per-architecture H_A).
  - Sharpness probe (entropy of retrieval-head attention distribution).
  - Per-K top-k overlap with needle block (auxiliary mechanism check).
  - Task-level NoLiMa / EverMemBench-S accuracy.

## 7. Risks to the hypothesis

**R1. Native vs post-hoc sparsity confound.** NSA, MoBA, DSA require *native* trained-from-scratch sparsity to fully exhibit their behavior (arXiv:2502.11089 §2.2). Post-hoc attention swaps may exhibit pathologies that contaminate the K-axis effect. *If R1 materializes:* the hypothesis still contributes by specifying which channel structure permits retrieval-head re-formation in principle; the empirical test simply needs to be done with native-trained models. The cheap test (§5a) explicitly acknowledges this caveat.

**R2. K=1 retention may already be at ceiling.** With the within-architecture redefinition, retention(K=1) = 1.0 by construction; the question is how much K>1 *erodes* it. If sparse-attention pretraining produces a small, stable retrieval-head set that is robust to recursion, K>1 may not erode any architecture meaningfully (all K=4 retentions ≈ 0.98), and F2 differential is too small to measure. *If R2 materializes:* the hypothesis's mechanistic claim is unfalsified-at-this-protocol. We refine to a stress-test variant: longer context (128K), harder NoLiMa distractor settings, adversarial NIAH from arXiv:2601.20276.

**R3 (expanded — strongest counterargument). Recursion may smear rather than sharpen.** Huginn / Recurrent-Depth (arXiv:2502.05171) reports loop-recurrent transformers *underperforming* non-recurrent baselines on needle-style retrieval at long context. Fixed-Point RNNs (arXiv:2503.10799) frames depthwise recursion as fixed-point iteration where the limit is the operator's attractor — generally not a sharp delta on a needle position. Output-conditioned re-feed at K=2 may produce a draft answer whose content is *less* sharply localized than the K=1 output, and feeding this back may flatten queries.

*Engagement.* I do not have direct prior-art support for re-feed *sharpening* retrieval-head argmaxes at long context — TRM's sharpening is on small grids (arXiv:2510.04871 §4), not transformer-scale residual streams. The hypothesis therefore (i) does not predict sharpening; (ii) makes the *differential* (F2, NSA-vs-MoBA) the primary prediction, which survives symmetric smearing; (iii) instruments the sharpness probe to diagnose whether smearing is driving any retention loss; (iv) keeps F1 only as a "consistency check" so a smearing-only outcome partially-falsifies but does not fully falsify the hypothesis.

*If R3 materializes fully (smearing dominates and is symmetric across architectures):* F1 fires, F2 does not — the hypothesis is partially falsified (architectural asymmetry was real but absolute effect was negative). This is still a publishable result: it constrains the spec's fusion design space by showing recursion-as-implemented (output-conditioned re-feed) is contraindicated for sparse-attention long-context retrieval. The fix would be a stronger recursion (option (b), separate latent stream) which is a different research project.

*If R3 materializes asymmetrically (NSA smears less than MoBA because compression channel anchors the residual stream):* this is exactly the F2 prediction — it is *the* hypothesis confirmed.

**R4. The compression branch may be too coarse.** NSA's compression pools blocks (block size ≈ 32 per arXiv:2502.11089 §3) before contributing; per-needle SNR after pooling is ~1/32 in the worst case. If this is below the threshold for query-refinement to amplify in subsequent passes, M2a fails in practice. *If R4 materializes:* a sharper variant of the hypothesis — that *block-pool granularity* (a tunable knob) gates retrieval-head re-formation — becomes the contribution, with the same falsification logic on the compression-block-size axis.

**R5. Output-conditioned re-feed is too weak a recursion.** Compared to TRM's published `(x, y, z)` recursion with a separate latent z-stream, output-conditioned re-feed only refines the query through input-token modification, no separate latent broadcast. Effects may be small. *If R5 materializes (small effects across the board, F2 below 0.05):* the cheap test is inconclusive; full eval may need a stronger recursion (option (b)) which is outside the YAGNI fence — at that point the hypothesis converts to a YAGNI-bound negative result: "the minimum-cost recursion does not produce the predicted effect; a heavier-architecture recursion is needed."

**R6. Dataset contamination on NoLiMa.** If the model's pretraining saw NoLiMa-style needles, retention measurements may overestimate generalization. *If R6 materializes:* fall back to EverMemBench-S (arXiv:2601.20276), which uses a 326M-token MemoryBank specifically to evade contamination.

## 8. Sources

- **arXiv:2404.15574** — Wu et al., "Retrieval Head Mechanistically Explains Long-Context Factuality" (retrieval-head detection, metric, baseline retention, downstream effects on NIAH and CoT). Verified via paper_details.
- **arXiv:2502.11089** — Yuan et al., "Native Sparse Attention" (NSA's three-branch design including compression; §3 architectural specification; explicit retrieval-head vulnerability claim in §2.2). Verified via paper_details and read_paper §2 + §3. (Citation locator corrected per S1.)
- **arXiv:2502.13189** — Lu et al., "MoBA: Mixture of Block Attention" (top-k gating; g_i=0 for non-selected blocks; no fallback channel). Verified via paper_details and read_paper §2.
- **arXiv:2510.04871** — Jolicoeur-Martineau, "Tiny Recursive Model" (TRM-style K-pass recursion; explicit `(x, y, z)` recipe). Verified via paper_details and read_paper §3. Mapping onto a long-context transformer specified explicitly in §3 M3 (per C4).
- **arXiv:2602.11374** — Bick et al., "Retrieval-Aware Distillation for Transformer-SSM Hybrids" (Gather-and-Aggregate heads; explicit preservation under sub-quadratic conversion). Verified via paper_details. Repositioned from co-evidence to *competing baseline* per I4.
- **arXiv:2407.15891** — Tang et al., "RazorAttention" (retrieval heads exploited for KV compression — independent confirmation of head specialness). Verified via paper_details.
- **arXiv:2502.05167** — Modarressi et al., "NoLiMa" (non-literal-match long-context probe; non-commercial license — flagged). Verified via paper_details.
- **arXiv:2410.04422** — Yu, "Hyper-multi-step" (long-context tasks decompose into multi-step retrieval). Verified via paper_details.
- **arXiv:2504.17768** — Nawrot et al., "Sparse Frontier" (sparse-attention pattern × task ablations; baseline for the (sparse pattern × task) axis without retrieval-head retention probe). Verified via paper_details.
- **arXiv:2506.08889** — Gao et al., "SeerAttention-R" (self-distilled sparse gating for reasoning; baseline for sparse × reasoning without recursion-pass probe). Verified via paper_details.
- **arXiv:2406.10774** — Tang et al., "Quest" (query-aware top-k page selection; no compression channel — exemplar of M2b-only architecture). Verified via paper_details.
- **arXiv:2512.02556** — DeepSeek-AI, "DeepSeek-V3.2" (DSA / lightning indexer; baseline for indexed sparsity without compression branch). Verified via paper_details.
- **arXiv:2601.20276** — Lin et al., "Beyond the Needle's Illusion" / EverMemBench-S (decoupled evidence-access vs. evidence-use; 326M-token MemoryBank for adversarial NIAH). Verified via paper_details.
- **arXiv:2502.05171** — Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (Huginn; loop-recurrent transformer; cited in R3 for prior evidence that depth-recursion can degrade needle retrieval). Verified via paper_details. **NEW in revision-1.**
- **arXiv:2503.10799** — Movahedi et al., "Fixed-Point RNNs: Interpolating from Diagonal to Dense" (depthwise recursion as fixed-point iteration; cited in R3 for the smearing/attractor framing). Verified via paper_details. **NEW in revision-1.**
- **arXiv:2512.24601** — "Recursive Language Models" (programmatic / inference-time recursion via tool-use; cited in §1 as orthogonal prior art per S3). Note: not directly read; cited via the red-team's attribution.
- **arXiv:2603.15653** — "SRLM" (programmatic recursion; cited in §1 as orthogonal prior art per S3). Note: not directly read; cited via the red-team's attribution.
