# Red-Team Critique — H3 (round 2, revision-1)

**Critiquing:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-3/revision-1/output.md`
**Round-1 critique:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-3/output.md`
**Original hypothesis:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-3/output.md`
**Gap targeted:** A8 × B9 (gap-finder-1 §41 + gap-finder-2 §9)
**Revision round:** 2

---

## 1. Verdict

**APPROVE**

The smith engaged with all four Critical objections honestly. The hypothesis has been substantially tamed — primary prediction is now a *differential* (F2: NSA-vs-MoBA Δretention at K=4 vs K=1) rather than the original absolute monotonicity claim, and the smith explicitly concedes both that the directional sign is uncertain (R3) and that "output-conditioned re-feed" is "the weakest form of TRM-style recursion." The result is a *narrower but more defensible* hypothesis. F2 is operationalizable, F6 (compression-branch ablation) is a genuinely strong mechanism check, the within-architecture retention metric is well-defined, and the gap claim survives. The remaining issues (Huginn over-interpretation in R3, Quest > DSA ordering being thin, framing collision with two-pass CoT / Self-Notes / Attrieval) are Important but not Critical, and the surviving primary prediction is robust to all of them.

I would defend this hypothesis publicly. The honest concession that the absolute K-effect could go either way under R3 — combined with the differential being the load-bearing prediction — is exactly the discipline a red-team should reward. APPROVE.

---

## 2. Round-1 critical-objection status

### C1 — Cheap kill-test inconsistent with mechanism — **RESOLVED (with caveat)**

The new operationalization (output-conditioned re-feed: append pass-k draft answer to prompt, rerun forward pass) does change attention queries at K=2 — this is mathematically definitional, since `Q = X·W_Q` and `X` differs at K=2 from K=1 by the appended draft tokens. The cheap test now operationally matches the full eval (both use output-conditioned re-feed with post-hoc / native NSA masks respectively). C1 is resolved at the consistency level.

**Caveat (not a re-rejection).** The C1 fix collapses the recursion onto something **operationally indistinguishable from two-pass CoT** or one iteration of Self-Notes (arXiv:2305.00833) / RAT (arXiv:2403.05313) / Attrieval (arXiv:2503.09819). The smith acknowledges this implicitly ("It is YAGNI-compliant ... no parameters added, no training, no architectural modification") and explicitly ("This is the **weakest** form of TRM-style recursion"). The architectural-recursion framing is now mostly *labelling* — there is no separate latent stream, no parameter sharing across passes (because the same model is run twice), no TRM-specific mechanism beyond "rerun with answer-so-far appended." This is a substantial deflation of the original "TRM-style architectural recursion" framing, and the smith owns it. Rated as Important (see §7).

### C2 — Cross-architecture retention metric — **RESOLVED**

Within-architecture retention is a clean fix. Each architecture's H_A is identified independently at K=1, then `retention_A(K) = |H_A still firing at K| / |H_A|`. By construction `retention_A(1) = 1`, so the metric measures the *erosion* of an architecture's own retrieval circuitry across recursion passes. The cross-architecture comparison is moved to NoLiMa task accuracy, which is the right level. C2 is resolved.

The K=1 = 1.0 by-construction property does **not** make the differential vacuous — different architectures will erode at different rates with K, and that erosion-rate gap is what F2 measures.

### C3 — Quest contradicts mechanism — **RESOLVED**

The M2a / M2b decomposition is principled:
- M2a (compression-signal channel): NSA's compressed-attention branch carries weak nonzero signal from every block, including non-top-k blocks. Verified against arXiv:2502.11089 §3 — the three-branch design with gated combination is correctly characterized.
- M2b (query-refinement channel): output-conditioned re-feed changes queries → re-runs sparse selection. Architecture-agnostic.

The ordering NSA > Quest ≥ DSA > MoBA falls out of the decomposition: NSA gets both M2a + M2b; Quest/DSA/MoBA get only M2b at varying responsiveness. **One concern:** the *internal* ordering of Quest vs DSA vs MoBA on M2b responsiveness is asserted by analogy (Quest's per-page query-aware top-k > DSA's lightning-indexer > MoBA's block-mean-pool gate). I find this plausible — Quest's selection function is explicitly query-aware (arXiv:2406.10774 §3) while MoBA's gate is `mean-pool(K_block) · query` (arXiv:2502.13189 §2.2 Eq. 4-5) which has lower per-query sensitivity — but this specific ordering has no direct prior art. F5 graceful-degradation (one-swap allowed before falsification) handles this honestly.

### C4 — TRM puzzle-shape recursion mapping — **RESOLVED**

The explicit `(x, y, z) → (prompt tokens, draft answer tokens, answer-span hidden states)` mapping with `net = full_forward(concat(x, y))` is concrete and implementable. The smith correctly notes this is option (a) from my round-1 analysis (output-conditioned re-feed) and explicitly chooses it over option (b) (separate latent stream) because option (b) would require architectural modification.

**Caveat folded into C1 above.** Under this mapping, the "z update happens implicitly inside the forward pass" — i.e., there is no explicit z update, just the answer-span being recomputed at every pass. This is an honest concession that the mapping makes TRM into something closer to two-pass autoregressive generation than a depth-recurrent architecture. Rated as Important.

### R3 / I5 — Recursion may smear — **RESOLVED (with caveat)**

The smith engaged R3 substantively in revision §3 and §7. Concessions: (i) no direct prior art that output-conditioned re-feed *sharpens* retrieval-head argmaxes at long context; (ii) directional sign of absolute K-effect uncertain; (iii) primary prediction shifted to F2 differential which survives symmetric smearing; (iv) sharpness probe (entropy of retrieval-head attention distribution) added as a diagnostic. This is exactly the response a red-team should reward.

**Caveat.** The Huginn citation (arXiv:2502.05171) is *partially* over-interpreted. Reading Huginn §5.3 directly ("Where does recurrence help most?"), the paper actually shows recurrence helps *more* on tasks that require integrating context, with the saturation depth scaling with the number of few-shot examples. There's no direct "loop-recurrent transformers degrade on needle-style retrieval" claim that I can find in §5.3. The smith's framing — "Huginn ... shows loop-recurrent transformers can underperform non-recurrent baselines on needle-style retrieval at long context" — is not a quote, and I cannot find it in §5.3. The Fixed-Point RNN citation (arXiv:2503.10799) is correct in spirit (depthwise recursion as fixed-point iteration), but Movahedi et al.'s paper is about *linear* RNN parameterizations (interpolating diagonal-to-dense), not transformer depth recursion specifically. Rated as Important — the citations are real and broadly relevant, but the specific claims attributed to them are loose. The smith should soften the framing to "depthwise recursion has been shown to converge to operator attractors that are not generally sharp deltas," with Fixed-Point RNNs cited for the framing and Huginn cited only for the broader fact that recurrence has task-dependent effects.

---

## 3. Independent gap re-verification (round 2)

I ran four new queries focused on whether the C1 fix introduces a collision with prior CoT / iterative-prompting work that the original gap claim missed.

| # | Query | Top hits | Verdict |
|---|---|---|---|
| Q6 | `self-refine iterative chain of thought retrieval long context` | RAT (2403.05313, 9 upvotes), Attrieval (2503.09819), LongRePS (2502.20790), Self-Notes (2305.00833), RetroRAG (2501.05475), RecaLLM (2604.09494) | These cover the *iterative-prompting* axis but none touches retrieval-head retention under sparse-attention substrates |
| Q7 | `latent recursion long context retrieval head architectural recursion depth` | Retrieval Head (2404.15574), DuoAttention (2410.10819), **SpiralFormer (2602.11698)**, **Depth-Recurrent Attention Mixtures (Dreamer, 2601.21582)** — Dreamer combines depth-recurrence + sparse expert attention | Dreamer is the closest collision; needs spot-check |
| Q8 | `CoT sparse attention NSA MoBA retrieval head test-time` | Kinetics (2506.05333), MoSA (2505.00315), FlashMoBA (2511.11571), **SSA (2511.20102, 28 upvotes)** which aligns sparse with full attention on NSA/MoBA | SSA is a competing approach (alignment-based) for the same problem (NSA/MoBA fragility) but not via recursion |
| Q9 | `draft answer prepend re-feed multi-pass attention sparse` | DraftAttention (2505.14708, video diffusion), Delta Attention (2505.11254), FlexPrefill (2502.20766) | None on text-LLM two-pass CoT × sparse attention |
| Q10 | `recurrent depth needle haystack retrieval` | NoLiMa, NeedleChain, EverMemBench-S, RULER — all benchmarks; no recurrent-depth study | Confirms no prior art on recurrent-depth × NIAH |

**Dreamer (arXiv:2601.21582) is the strongest potential collision.** It combines depth-recurrence + sparse expert attention. From the abstract, however: it focuses on "scaling efficiency" via sequence-attention + depth-attention + sparse-expert-attention as a *jointly trained architecture*, not on retrieval-head retention as a function of K under existing sparse-attention substrates (NSA/MoBA/Quest/DSA). Its sparse expert attention is closer to MoE-style routing than to NSA-style three-branch design. It does not measure retrieval-head retention. Spot-checked via paper_details — confirms the abstract framing. **Adjacent prior art, not direct collision.** The smith should cite it in §1 to clarify non-overlap, similar to how SRLM/Recursive Language Models were handled.

**SSA (arXiv:2511.20102, 28 upvotes) is an important adjacent baseline** the smith does not cite. SSA explicitly addresses sparse-attention fragility on NSA/MoBA via output-feature alignment with full attention — a *direct competing approach* to retrieval-head preservation. SSA does not use recursion. The smith should add this as a competing baseline alongside G&A. Suggestion-tier.

**Gap claim survives** — narrower than originally claimed, but the specific intersection (sparse-attention pattern × architectural-recursion-K × retrieval-head retention) is genuinely unmeasured. Adjacent prior art exists in:
- Iterative prompting / CoT for long-context retrieval (RAT, Attrieval, Self-Notes, LongRePS, RetroRAG) — different lens (task-level accuracy, not head-level retention)
- Sparse-attention output alignment (SSA) — different mechanism (alignment training, not recursion)
- Depth-recurrent architectures (Huginn, Dreamer, SpiralFormer) — different substrate (custom recurrent backbones, not NSA/MoBA wraps)

`gap_claim_survives: true` (narrower scope; smith should cite Dreamer + SSA as adjacent prior art for completeness).

---

## 4. Citation spot-checks (round 2 — focusing on revision-1 additions)

### 4.1 arXiv:2502.05171 — Huginn / Recurrent-Depth (§5.3 read directly)

**What the paper actually says (§5.3 "Where does recurrence help most?"):** The recurrent model outperforms its non-recurrent twin "with an especially pronounced advantage on harder tasks" (ARC challenge). On easier recall tasks (SciQ), gains are smaller. **Crucially:** "without few-shot examples to consider, the model saturates in compute around 8-12 iterations. However, when more context is given, the model can reason about more information in context, which it does, saturating around 20 iterations if 1 example is provided, and 32 iterations, if 25-50 examples are provided." This shows recurrence *helps more* with more context, not less. There is also a Table 5 result showing the recurrent model closes the gap to OLMo-2 in the open-book setting where a relevant fact is provided.

**What the smith claims (§3 M3 / R3):** "Huginn (arXiv:2502.05171) shows loop-recurrent transformers can underperform non-recurrent baselines on needle-style retrieval at long context."

**Verdict: PARTIAL MISREPRESENTATION.** I do not find a Huginn passage that supports "loop-recurrent transformers underperform non-recurrent baselines on needle-style retrieval at long context." Huginn §5.3 supports the *opposite* read in some respects (recurrence helps more with more context). The Fixed-Point RNNs framing (recursion as fixed-point iteration converging to an operator's attractor) is more directly supportive of the smith's R3 framing. Rated **Important** (not Critical) — the smith's R3 argument doesn't actually depend on Huginn; the Fixed-Point RNN citation alone carries the argument. The smith should soften the Huginn framing to "Huginn explores how depthwise recursion interacts with context length and shows task-dependent effects" rather than "underperforms on needle-style retrieval."

### 4.2 arXiv:2503.10799 — Fixed-Point RNNs (paper_details verified)

**What the paper actually says:** "We investigate parameterizations of a large class of dense linear RNNs as fixed-points of parallelizable diagonal linear RNNs." This is about linear RNN parameterizations interpolating diagonal-to-dense for state-tracking expressivity, not about transformer depth recursion or retrieval.

**What the smith claims:** "Fixed-Point RNNs (arXiv:2503.10799) frames depthwise recursion as fixed-point iteration where the limit is the attractor of the update operator, not generally a sharp delta."

**Verdict: SOFT MISREPRESENTATION.** The paper does treat recursion as fixed-point iteration in the linear-RNN context, but the framing "depthwise recursion converges to an operator's attractor that is not generally a sharp delta on a needle position" is the smith's *extrapolation* of the Fixed-Point RNN logic to a different setting (transformer depth recursion on long-context retrieval). The extrapolation is *plausible* but not directly supported by Movahedi et al. — their paper does not study transformer depth recursion. Rated **Suggestion** — the smith should explicitly mark this as an extrapolation from linear-RNN theory, e.g., "by analogy to Fixed-Point RNNs' framing of recursion as fixed-point iteration..." The argument doesn't require the specific paper to support it directly; it's a standard fixed-point-iteration intuition.

### 4.3 arXiv:2510.04871 — TRM (§3 read directly)

**What the paper actually says (§3, code):** TRM's recursion is `z = net(x, y, z)` for n iterations, then `y = net(y, z)`. The function `net` is a small two-layer block (MLP-like), not a long-context transformer. TRM is *not* designed as an outer-loop wrapper around a pretrained transformer — it's a custom architecture for puzzle-grid problems (Sudoku, Maze, ARC-AGI).

**What the smith claims (§3 M3):** TRM's `(x, y, z)` mapping onto a long-context transformer becomes `(y, z) = full_forward(concat(x, y_prev_tokens))`. The smith explicitly acknowledges this is the "**weakest** form of TRM-style recursion — it does not maintain a separate latent stream."

**Verdict: HONEST CONCESSION.** The smith correctly characterizes TRM and correctly notes that the mapping is a deflation: there is no separate latent stream `z`, no fixed-point iteration in the formal TRM sense, no parameter-tied recursion (because the same transformer is rerun, not a small recursive block). This is a substantial deflation, but the smith owns it. Rated **Important** for the labelling — the resulting method is operationally closer to two-pass CoT than to TRM — but not a misrepresentation.

### 4.4 arXiv:2503.09819 — Attrieval (§3.2 + §4 read directly)

**What the paper actually says:** Attrieval is a training-free method that uses attention weights from generated CoT tokens to retrieve facts from long context, then incorporates retrieved facts for reasoning. The mechanism is identical-in-spirit to the smith's "M2b query-refinement channel": the model generates tokens (the smith's "draft answer"), and those tokens' attention to input is used to surface facts. Attrieval uses Llama-3.1-8B / Llama-3.2-3B / Qwen2.5-3B at 0K-32K context on Deduction / MuSiQue / HotpotQA. **Attrieval does not measure retrieval-head retention** and does not use sparse-attention substrates.

**Verdict: NOT CITED — IMPORTANT GAP.** Attrieval is the closest published method to what the smith proposes (output-conditioned re-feed using draft-answer attention as a retrieval signal), and the smith does not cite it. Rated **Important**: the smith should cite Attrieval in §1 as adjacent prior art (similar lens — using generated tokens to refine retrieval — but at task-level, not at retrieval-head-retention level, and not under sparse-attention substrates). This strengthens the gap claim by making the non-overlap explicit.

---

## 5. Mechanism critique (round 2)

### 5.1 M2a (compression-signal channel) — solid

The argument that NSA's compressed-attention branch passes a coarse signal from every block (including non-top-k blocks) is well-grounded. NSA §3 Eq. 7-9 specifies the three-branch design with gated combination. Per-needle SNR ≈ 1/32 at the compression-block-size of 32 is a quantitatively defensible claim. The contrast with MoBA (g_i = 0 for non-top-k) and Quest (no compression branch) is real and verified in round 1.

**F6 (compression-branch ablation) is a strong mechanism check.** Stripping NSA's compression branch should make NSA's recovery profile collapse to MoBA-class. If it doesn't, M2a is not load-bearing — that's a meaningful, single-experiment falsification. This is a substantial improvement over the original F4 (which I called "tautological" in round 1).

### 5.2 M2b (query-refinement channel) — definitional, but the leverage is thin

The M2b claim is **definitional**: appending tokens to the input changes the input, which changes the queries `Q = X·W_Q` at every layer. This is true by construction. The substantive question is whether the *change* in queries is large enough to *change which top-k blocks get selected*. The smith does not give a quantitative bound for this — at small block size and small query change, top-k selection may be stable across passes, in which case M2b contributes nothing.

This is not a critical objection: F2 (differential) doesn't require M2b to drive recovery — it requires NSA's recovery to exceed MoBA's. M2a alone is sufficient for that. M2b is the *only* channel for MoBA, so MoBA's recovery (or lack thereof) is the M2b signal.

### 5.3 The "TRM-style recursion" framing collapse — Important

Under the smith's revision-1 mapping, "TRM-style architectural recursion" reduces to "rerun a pretrained transformer with `Answer-so-far: <pass-k draft>` appended to the prompt." This is operationally identical to:
- Two-pass CoT prompting
- One iteration of Self-Notes (arXiv:2305.00833)
- One iteration of RAT-style draft revision (arXiv:2403.05313, but without external retrieval)
- One iteration of LongRePS-style answer refinement (arXiv:2502.20790)
- One pass of Attrieval-style attention-guided retrieval (arXiv:2503.09819) without the explicit fact-extraction step

The smith honestly concedes this ("the weakest form of TRM-style recursion ... it does not maintain a separate latent stream, so the recursion only refines the query through the input-token modification"). The hypothesis label "architectural recursion" is now somewhat misleading — the substance is "two-pass CoT" with the lens being retrieval-head retention under sparse-attention substrates.

**This does not invalidate the hypothesis** — the prediction is still falsifiable (F2 differential, F6 ablation) and the mechanism is still grounded (M2a is real, F6 directly tests it). But the *novelty framing* should be updated. The novel contribution is not "TRM applied to long context" but rather "the asymmetric robustness of NSA's compression-channel architecture under iterative query refinement, measured at the retrieval-head level." That's a genuine but narrower contribution. Rated Important.

### 5.4 The Quest/DSA/MoBA ordering on M2b — thin but graceful

The ordering NSA > Quest ≥ DSA > MoBA in §2 / F5 rests on the M2b-responsiveness ordering Quest > DSA > MoBA. The smith's argument:
- Quest: per-page query-aware top-k (arXiv:2406.10774 §3) — high responsiveness to query change.
- DSA: lightning-indexer learned top-k (arXiv:2512.02556) — moderate responsiveness.
- MoBA: block-mean-pool gating (arXiv:2502.13189 §2.2 Eq. 4-5) — low responsiveness because the gate is a static function of block content.

This is plausible but has no prior art directly comparing M2b-responsiveness across these. F5's graceful-degradation policy (one swap allowed before falsification) handles this honestly — if MoBA happens to outperform DSA on M2b, that's one swap; the prediction survives. If MoBA outperforms NSA, that's two swaps — the mechanism decomposition is wrong. This is reasonable.

---

## 6. Falsifiability assessment (round 2)

| Criterion | Operationalizable? | Notes |
|---|---|---|
| F2 (PRIMARY: differential) | **Yes**, well-defined contrast. Threshold +0.05 absolute. Within-architecture metric is well-defined since `retention_A(1) = 1` by construction. | The differential can in principle be 0 if both architectures preserve their own retrieval heads at K=4 (R2 risk acknowledged). The smith handles this by stress-testing at longer context if needed. |
| F5 (ordering) | **Yes**, finite rank-order. One-swap tolerance is reasonable. | Less rigorous than F2 but useful as a mechanism-decomposition check. |
| F3 (NoLiMa task-transfer) | **Yes**, NoLiMa has 5-10 point gaps between configurations per arXiv:2502.05167; +3 point threshold is plausibly above noise. | License-flagged for non-commercial. Acceptable. |
| F1 (consistency check) | **Yes**, demoted appropriately. | The smith correctly uses this as a *partial* falsification — F1 fires means NSA's absolute K=4 retention is catastrophically lower than K=1, which would be interesting in its own right. |
| F6 (mechanism check / compression-branch ablation) | **Yes**, single-experiment falsification of M2a as load-bearing. | The strongest mechanism check of all. If `retention_NSA-noCmp(K=4) − retention_NSA-noCmp(K=1)` ≈ MoBA's differential within ±0.03, M2a is confirmed; if it remains as positive as full NSA's, M2a was not load-bearing. |
| Sharpness diagnostic | Auxiliary — diagnostic only, not falsification. | Useful for distinguishing R3-mediated (smearing) from M2a/M2b-mediated (selection failure) retention drops. |

**All five falsification paths plus the diagnostic are genuinely operationalizable.** The cheap-test noise floor (1-sigma ~ 0.024 on the differential at 200 samples) makes the +0.05 threshold ~2.1-sigma — adequate but not generous. The smith handles this with multi-seed reporting and an explicit "inconclusive at cheap scale" outcome. Reasonable.

**No criterion collapses to unfalsifiable hand-waving.** Compare to a hypothetical "the technique is shown to be useless in all contexts" — F2/F5/F3/F6 all reduce to numerical thresholds on observable quantities.

---

## 7. Severity-tagged objections (round 2)

### Critical (must fix)

**None.** All four round-1 Critical objections are resolved at the level required.

### Important (should fix)

- **I6 (NEW).** Huginn citation in R3 / §3 M3 is partially over-interpreted. Soften the framing — Huginn does not directly support "loop-recurrent transformers underperform on needle-style retrieval at long context." The Fixed-Point RNN citation alone carries the smearing-as-fixed-point argument; Huginn should be cited only for "depthwise recursion has task-dependent effects on retrieval-style benchmarks" or similar.

- **I7 (NEW).** Attrieval (arXiv:2503.09819) is the closest published method to the smith's output-conditioned re-feed and is not cited. Add it to §1 as adjacent prior art with explicit non-overlap statement (different lens — task-level vs head-level; different substrate — dense attention, not sparse).

- **I8 (NEW).** Dreamer (arXiv:2601.21582 — Depth-Recurrent Attention Mixtures, May 2026) and SpiralFormer (arXiv:2602.11698) are recent depth-recurrent architectures with sparse expert attention. Cite as adjacent prior art for non-overlap — they jointly train depth-recurrent + sparse attention, while the smith proposes post-hoc / native-trained sparse attention with output-conditioned re-feed (no joint training of the recursion).

- **I9 (NEW).** The "TRM-style architectural recursion" framing is partly *labelling* — under revision-1's mapping, it operationally collapses to two-pass CoT with answer-prepended re-input. The smith should reframe the novelty as "the asymmetric robustness of NSA's compression-channel under iterative query refinement, measured at the retrieval-head level" rather than "TRM applied to long-context retrieval." This is honest framing, not a re-rejection.

- **I10 (NEW).** SSA (arXiv:2511.20102, 28 upvotes) is a competing approach to NSA/MoBA fragility (output-alignment training, not recursion). Cite alongside G&A as a competing baseline in §6 ablation (c) or in §1.

### Suggestion (nice to have)

- **S6.** Mark Fixed-Point RNN extrapolation explicitly as an analogy in R3 — the paper is about linear-RNN parameterizations, not transformer depth recursion. The smith should write "by analogy to the fixed-point-iteration framing of Fixed-Point RNNs (arXiv:2503.10799)" rather than implying direct support.

- **S7.** For the cheap test, consider adding one more architecture (e.g., a sparsity-ratio sweep on the NSA-emulation mask alone) so the post-hoc setup has more axes than just NSA-vs-MoBA. This increases the chance of catching a confound.

- **S8.** The recursive-language-models / SRLM citations in §1 are still not directly verified by the smith (acknowledged in verification.md). Either fetch them or remove.

### Counts (round 2)
- Critical: 0
- Important: 5
- Suggestion: 3

---

## 8. Strongest counter-argument (round 2 steelman)

**The strongest case that H3-revision-1 is wrong:** the F2 differential could be driven by a confound unrelated to the M2a/M2b mechanism — namely, **MoBA's known suboptimal training-aware routing**. Recent work (FlashMoBA / Optimizing Mixture of Block Attention, arXiv:2511.11571) explicitly identifies MoBA's gating signal-to-noise ratio as a bottleneck — raw MoBA without the FlashMoBA optimizations gates poorly, making it especially fragile to query distribution shifts. If the F2 differential is observed, the smith would attribute it to "compression channel preserves recovery"; the alternative explanation is "MoBA's gate is bad and any iterative query refinement separates NSA from MoBA, regardless of mechanism."

**This is exactly what F6 is designed to disambiguate.** Stripping NSA's compression branch and checking that NSA-noCmp's recovery profile collapses to MoBA-class would *confirm* the M2a-as-load-bearing claim. If NSA-noCmp still recovers better than MoBA, the differential was driven by something other than M2a — likely the routing-quality confound. F6 is therefore a critical experiment, not optional. The smith should highlight F6 as load-bearing for ruling out the routing-quality confound, not just as a mechanism check.

**The smith handles this implicitly** — F6 is the strongest mechanism check, and its failure mode is exactly "M2a is not load-bearing" — but I'd suggest making the routing-quality alternative explanation explicit in §7 as R7 or an extension of R5. (Suggestion-tier.)

---

## 9. Recommendation to hypothesis-smith

The hypothesis is now defensible, falsifiable, and mechanism-grounded. APPROVE.

If the smith chooses to revise further (not required for this approval), the priorities are:

1. **(I6, S6)** Soften the Huginn / Fixed-Point RNN framings. Huginn's §5.3 does not directly support "underperforms on needle-style retrieval"; Fixed-Point RNNs is about linear RNNs not transformers.

2. **(I7, I8, I10)** Add adjacent-prior-art citations: Attrieval (2503.09819), Dreamer (2601.21582), SpiralFormer (2602.11698), SSA (2511.20102). These don't change the hypothesis — they tighten the gap claim.

3. **(I9)** Reframe the novelty slightly. Instead of "TRM applied to long-context retrieval," lead with "asymmetric robustness of NSA's compression-channel architecture under iterative query refinement at the retrieval-head level." This is a more accurate description of the contribution under the revision-1 mapping.

4. **(Steelman)** Make the routing-quality alternative explanation explicit and note that F6 is the disambiguator. This isn't required — the experiment design already handles it — but stating it explicitly closes a small remaining loophole.

The hypothesis is ready for Phase 5 (eval-designer) as-is.

---

APPROVE | REJECT (revision-2) | KILL (irrecoverable)

**APPROVE**
