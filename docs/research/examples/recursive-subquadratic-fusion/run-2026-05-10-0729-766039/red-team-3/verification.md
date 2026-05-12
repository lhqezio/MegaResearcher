# Red-Team Verification — H3

## Verification-before-completion checklist

### 1. Independent literature queries (≥3 required)

Five independent queries run via `mcp__plugin_megaresearcher_ml-intern__hf_papers search`:

1. `retrieval head sparse attention recursion depth` — 10 results; closest hits are Retrieval Head 2404.15574, Ltri-LLM 2412.04757, Retrieval-Aware Distillation for SSMs 2602.11374, RazorAttention 2407.15891, DuoAttention 2410.10819. None measure retention as a function of (sparse-pattern × recursion-depth K).

2. `native sparse attention retrieval head retention long context` — 10 results; Optimizing-NSA (2511.00819), Adaptive-Long-Context-Head-Identification (2502.09647), Focus Directions (2503.23306), HSA (2511.23319). None couple retrieval-head retention with recursion or NSA-vs-MoBA contrast.

3. `recursive transformer iterative refinement long context retrieval` — 10 results; RMT (2304.11062), Landmark Attention (2305.16300), ATLAS (2505.23735), RCC (2406.06110). All are token-level recurrence, not the depthwise architectural recursion the hypothesis means. Distinct gap area.

4. `attention masking retrieval head NSA pretrained model evaluation` — 8 results; DeCoRe (2410.18860, decode-time retrieval-head masking, not under sparse attention), AttentionInfluence (2505.07293, masks for data selection). None hit the joint regime.

5. `Tiny Recursive Model TRM long context language model` — 8 results; **Recursive Language Models (2512.24601, 96 upvotes, post-cutoff)** and **SRLM (2603.15653)** are programmatic / inference-time recursion, not architectural. TRM itself does not appear in long-context retrieval evaluations.

**Conclusion:** Gap claim survives independent verification. No paper measures retrieval-head retention under (sparse-attention × architectural-recursion) joint regime. Two adjacent papers (RLM, SRLM) exist that the hypothesis-smith should cite as related work but do not collapse the gap (they are inference-time programmatic, not architectural).

### 2. Citation spot-checks (≥3 required)

Four spot-checks performed via `paper_details` and `read_paper`:

1. **arXiv:2404.15574 §2** — read directly. Retrieval-score formula and 600-instance / 0.1-threshold / argmax-on-needle protocol all verified accurate. Universality claim (Table 1) covers dense-attention models only; cross-architecture transfer is **not** supported by the cited paper. Hypothesis assumes transfer; assumption is itself a research question (flagged Critical C2).

2. **arXiv:2502.11089 §2** — read directly. The cited "70% coverage by top-20% attention, retrieval heads vulnerable to post-hoc pruning" claim is in §2.2 verbatim and accurately quoted. The three-branch architectural design is in §3 / Figure 2, not §2.3 as the hypothesis states (minor citation-locator error, flagged Suggestion S1).

3. **arXiv:2502.13189 §2** — read directly. MoBA's `g_i = 1` for top-k blocks and `g_i = 0` otherwise (eq. 5) is verbatim. The "no fallback channel" claim is structurally accurate.

4. **arXiv:2510.04871 §3** — read directly. TRM's `latent_recursion(x, y, z, n=6)` and `deep_recursion` pseudocode is verbatim. Confirms the hypothesis's description of TRM-style recursion. **Also confirms:** TRM's recursion operates on puzzle-shaped (x, y, z) tensors, not transformer residual streams. Adapting to long-context transformer is non-trivial design step (flagged Critical C4).

### 3. Verdict-severity match

Critical: 4. Important: 5. Suggestion: 5.

Verdict: **REJECT (revision-1)**.

A REJECT is appropriate when there are unresolved Critical objections that the hypothesis-smith can plausibly address in revision. KILL would be appropriate only if the gap collapsed (it didn't), the falsification structure were unsalvageable (it isn't), or no falsifiable mechanism could be specified (one was, just imperfectly). Three of the four Critical objections (C1, C3, C4) are about specification gaps — solvable. C2 (cross-architecture identity) is a substantive metric redefinition but well-bounded.

APPROVE would require zero Critical objections. Not possible here: C1, C3 are internal inconsistencies; C2 is a metric flaw; C4 is a missing operationalization. Each is independently sufficient grounds for revision.

Verdict matches severity: pass.

### 4. Steelman quality check

§6 of `output.md` constructs the strongest opposing case (recursion-induced smearing on retrieval heads) and engages with it specifically: cites Huginn 2502.05171 and Fixed-Point RNNs 2503.10799 as evidence that depthwise recursion may smear rather than sharpen, points out the hypothesis-smith's R3 is a single hand-waved sentence, and notes that TRM's puzzle-grid sharpening is not obviously transferable to long-context needle-position sharpening. This is a non-trivial counter-argument that the hypothesis-smith must engage in the revision (flagged I5).

Steelman quality: pass.

### 5. Discipline check

- Performative agreement avoided.
- No "great hypothesis"-style flattery.
- All objections grounded in either citation evidence (C2, C3, I3, I4) or internal-inconsistency analysis (C1, C3, F4 critique).
- Tone is rigorous, not gratuitously hostile.

Pass.
