# Verification — Red-Team H3, round 2 (revision-1)

## Discipline checks (per skill spec)

### 1. Independent literature queries (≥ 3 required)

**PASS — 5 new independent queries run via hf_papers search:**

- **Q6** (`self-refine iterative chain of thought retrieval long context`): surfaced RAT (2403.05313, 9 upvotes), Attrieval (2503.09819), LongRePS (2502.20790), Self-Notes (2305.00833), RetroRAG (2501.05475), RecaLLM (2604.09494). None measures retrieval-head retention or uses sparse-attention substrates. Adjacent but non-overlapping.

- **Q7** (`latent recursion long context retrieval head architectural recursion depth`): surfaced Retrieval Head (2404.15574), DuoAttention (2410.10819), **SpiralFormer (2602.11698)**, **Dreamer / Depth-Recurrent Attention Mixtures (2601.21582)**. Dreamer is the closest depth-recurrent + sparse-attention work; spot-checked via paper_details — it focuses on jointly trained scaling efficiency, does not measure retrieval-head retention.

- **Q8** (`CoT sparse attention NSA MoBA retrieval head test-time`): surfaced Kinetics (2506.05333), MoSA, FlashMoBA (2511.11571), **SSA (2511.20102, 28 upvotes)**. SSA explicitly addresses NSA/MoBA fragility via output-feature alignment (a competing approach); does not use recursion.

- **Q9** (`draft answer prepend re-feed multi-pass attention sparse`): surfaced video-diffusion sparse-attention work (DraftAttention, Re-ttention) and inference acceleration (Delta Attention, FlexPrefill). No text-LLM two-pass CoT × sparse attention work.

- **Q10** (`recurrent depth needle haystack retrieval`): surfaced NoLiMa, NeedleChain, EverMemBench-S, RULER (benchmarks only); no recurrent-depth × NIAH study.

**Conclusion: gap claim survives, narrower than originally claimed.** Three new adjacent-prior-art citations surfaced (Attrieval, Dreamer, SSA) that should be added to the smith's §1 — flagged as Important objections I7, I8, I10 (not Critical because the specific intersection — sparse-attention × architectural-recursion × retrieval-head retention — remains unmeasured).

### 2. Citation spot-checks (≥ 3 required)

**PASS — 4 citations spot-checked via paper_details and read_paper:**

- **arXiv:2502.05171 (Huginn).** Read §5.3 directly. Found PARTIAL MISREPRESENTATION: smith claims Huginn shows "loop-recurrent transformers can underperform non-recurrent baselines on needle-style retrieval at long context." §5.3 does not directly support this. Huginn actually shows recurrence saturation depth scales with context complexity (more few-shots → more recurrence helps), and recurrent model closes gap to OLMo-2 in open-book setting (Table 5). Rated Important — smith should soften.

- **arXiv:2503.10799 (Fixed-Point RNNs).** Verified via paper_details. Paper is about linear-RNN parameterizations (interpolating diagonal-to-dense for state-tracking). Smith's "depthwise recursion as fixed-point iteration with attractor that's not a sharp delta" is a *plausible extrapolation* of the framing to transformer depth recursion, but Movahedi et al. don't study transformers. Rated Suggestion — smith should mark as analogy.

- **arXiv:2510.04871 (TRM).** Read §3 directly. Confirmed TRM's `(x, y, z)` recursion uses a small two-layer block, not a long-context transformer. Smith's mapping `(y, z) = full_forward(concat(x, y_prev_tokens))` is correctly characterized as "the **weakest** form of TRM-style recursion" (smith's own words). Rated Important for the labelling issue (operationally closer to two-pass CoT than to TRM) — but smith owns it and the prediction is still falsifiable.

- **arXiv:2503.09819 (Attrieval).** Read §3.2 + §4 directly. Found NOT CITED — IMPORTANT GAP. Attrieval's mechanism (use attention weights from generated CoT tokens to retrieve facts) is the closest published method to smith's output-conditioned re-feed. Smith should cite it as adjacent prior art with explicit non-overlap statement.

### 3. Verdict matches severity of objections

**PASS.** Verdict: APPROVE. Objection counts: 0 Critical, 5 Important, 3 Suggestion. APPROVE with zero Critical objections is consistent. The 5 Important objections (I6 Huginn, I7 Attrieval, I8 Dreamer/SpiralFormer, I9 framing, I10 SSA) are all "should fix" tier — they tighten the gap claim and adjust framing but do not invalidate the hypothesis. They could be addressed in a future revision but do not block Phase 5.

The smith's revision-1 honestly addresses all four round-1 Critical objections:
- C1 (cheap test inconsistent with mechanism) — RESOLVED via output-conditioned re-feed which provably perturbs queries.
- C2 (cross-architecture retention undefined) — RESOLVED via within-architecture metric.
- C3 (Quest contradicts mechanism) — RESOLVED via M2a/M2b decomposition.
- C4 (TRM puzzle-shape unmapped) — RESOLVED via explicit `(x, y, z) → (prompt, draft, hidden states)` mapping.

R3 / I5 (recursion may smear) — RESOLVED with substantive engagement, citation, and graceful-degradation falsification design (F2 differential survives symmetric smearing).

## Specific concerns from prompt evaluation

The prompt asked four critical things to check:

1. **Is output-conditioned re-feed actually meaningful test of architectural-recursion mechanism, or rephrased CoT?**
   - **Answer: closer to two-pass CoT than to architectural recursion.** Smith owns this ("weakest form of TRM-style recursion"). The hypothesis label is partly aspirational; the *prediction* and *falsification* still work without the architectural-recursion label. Rated Important (I9), not Critical.

2. **Does M2a + M2b decomposition make principled predictions or are orderings ad hoc?**
   - **Answer: M2a is principled and citationally grounded; M2b is definitional; the internal ordering Quest > DSA > MoBA on M2b-responsiveness is plausible-by-analogy with no direct prior art.** F5's one-swap-tolerance handles this honestly. Acceptable.

3. **Within-arch retention K=1=1.0 by construction. Falsifiable?**
   - **Answer: yes.** The differential (how much retention erodes from K=1 to K=4 for NSA vs MoBA) is well-defined. F2 directly tests this. F6 (compression-branch ablation) provides a strong mechanism check independent of K.

4. **Has TRM mapping become token-CoT?**
   - **Answer: yes, mostly.** Smith owns this. The architectural-recursion framing is partly labelling. The substance is "two-pass CoT × sparse attention substrate × retrieval-head retention metric." Smith should reframe the novelty (I9) but the falsifiability and mechanism-grounding survive.

## Assessment of the revision

Substantial honest engagement with all round-1 Critical objections. Primary prediction shifted from absolute monotonicity to differential, which is a real concession and a more defensible claim. Sharpness probe added for R3 diagnosis. F6 promoted to a primary mechanism check (it was implicitly there in revision-0 as ablation (a); now it's in the falsification list).

The hypothesis is narrower than originally — it's now "NSA + two-pass CoT vs MoBA + two-pass CoT, measured at retrieval-head level" rather than "TRM applied to long-context retrieval." That's still a novel, mechanistically grounded, falsifiable hypothesis. APPROVE is warranted.

## Outputs produced

- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-3/revision-1/output.md` (round-2 critique)
- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-3/revision-1/manifest.yaml`
- `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/red-team-3/revision-1/verification.md` (this file)

## Ready to advance to Phase 5?

**YES.** Hypothesis is approved. Eval-designer should pick up `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-3/revision-1/output.md` as the canonical version for experiment design. The 5 Important objections (I6-I10) can be folded in by the smith between Phase 4 and Phase 5 if desired, but do not block.
