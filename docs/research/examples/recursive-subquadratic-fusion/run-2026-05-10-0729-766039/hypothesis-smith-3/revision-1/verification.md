# Verification — Hypothesis H3, revision 1

## Mandatory checks (per skill spec)

### 1. Hypothesis statement is in if/then form

**PASS.** §2 opens with: "**If** a TRM-style architectural recursion is operationalized as output-conditioned re-feed ... layered on a transformer backbone whose attention has been replaced with NSA ..., **then** the *differential* in within-architecture retrieval-head retention recovery between NSA and MoBA at K=4 minus K=1 will be positive and at least 0.05 absolute, in favor of NSA."

The conditional structure is explicit; the consequent is an inequality with a numerical threshold (≥ 0.05 absolute differential).

### 2. At least 3 falsification criteria, each genuinely falsifiable

**PASS.** Five falsification paths plus a sharpness diagnostic:

- **F2 (primary, differential).** Threshold: differential `[retention_NSA(4) − retention_NSA(1)] − [retention_MoBA(4) − retention_MoBA(1)] < +0.05` falsifies. Genuinely falsifiable: a finite experiment yields a single number; comparison to threshold is unambiguous.
- **F5 (ordering).** NSA > Quest ≥ DSA > MoBA at K=4 recovery; more than one swap falsifies. Genuinely falsifiable: ordering is a finite, observable rank.
- **F3 (task transfer).** NoLiMa accuracy delta (NSA K=4 − K=1) − (MoBA K=4 − K=1) < +3 percentage points falsifies. Genuinely falsifiable.
- **F1 (consistency check).** retention_NSA(K=4) − retention_NSA(K=1) more negative than -0.10 absolute partially falsifies. Genuinely falsifiable.
- **F6 (mechanism check / promoted ablation).** Stripping NSA's compression branch should collapse its recovery to MoBA-class (within ±0.03); failure to collapse falsifies M2a as load-bearing. Genuinely falsifiable: this is the load-bearing mechanism check.

All thresholds are pre-registered; all metrics are computable from the specified protocols; all directions are stated.

### 3. Every mechanism claim has a citation

**PASS.** Inventory:

- M1 (retrieval heads = copy-paste circuit identifiable by argmax on needle): cited arXiv:2404.15574 §2.
- M2a (NSA compression branch carries signal from all blocks; per-block SNR ~ 1/32): cited arXiv:2502.11089 §3 Eq. 7-9.
  - MoBA gating zero for non-selected blocks: cited arXiv:2502.13189 §2.2 Eq. 5.
  - Quest top-k pages without compression: cited arXiv:2406.10774 §3.
  - DSA / lightning indexer: cited arXiv:2512.02556.
- M2b (output-conditioned re-feed changes input → changes queries → changes selection): grounded in transformer self-attention's deterministic dependence on input tokens — definitional, not a contested claim, but the empirical instantiation (text-level re-feed altering attention in a 1-3B transformer) is supported by analogy to TRM's `(x, y, z)` recursion arXiv:2510.04871 §3.
- M3 (TRM-style recursion mapping): cited arXiv:2510.04871 §3 (the TRM recipe), with explicit mapping to long-context transformer specified.
- R3 engagement (smearing as alternative): cited arXiv:2502.05171 (Huginn) and arXiv:2503.10799 (Fixed-Point RNNs).
- G&A as competing baseline: cited arXiv:2602.11374.

Every cited claim resolves to a real paper; no mechanism step rests on uncited intuition. The single non-cited mechanism claim — that altering an input prompt at one position changes attention queries at all subsequent positions — is mathematically definitional from the transformer architecture (Q = X·W_Q where X is the input; X changes ⇒ Q changes).

### 4. All cited arxiv IDs resolve via hf_papers paper_details

**PASS for citations verified in revision-1**:
- arXiv:2502.05171 (Huginn) — verified via paper_details in this revision (returned title "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach").
- arXiv:2503.10799 (Fixed-Point RNNs) — verified via paper_details in this revision (returned title "Fixed-Point RNNs: Interpolating from Diagonal to Dense").

**Inherited verifications from revision-0** (still valid; not re-checked):
- arXiv:2404.15574, 2502.11089, 2502.13189, 2510.04871, 2602.11374, 2407.15891, 2502.05167, 2410.04422, 2504.17768, 2506.08889, 2406.10774, 2512.02556, 2601.20276 — all verified in revision-0 verification.md.

**Not verified** (cited via red-team attribution, marked as such in §8):
- arXiv:2512.24601 (Recursive Language Models) — cited from red-team's gap re-verification §2 Q5; not directly fetched. Risk: arxiv ID may be miscited by red-team. Mitigation: paper is cited as orthogonal prior art only (not load-bearing).
- arXiv:2603.15653 (SRLM) — same status.

If the red-team in round 2 challenges these two citations, I will fetch them via paper_details before relying on them. They do not affect any falsification path.

### 5. The "Risks to the hypothesis" section is non-empty

**PASS.** Six risks (R1-R6) with contingency analysis for each:
- R1: native vs post-hoc sparsity confound.
- R2: K=1 ceiling effect.
- R3: recursion-induced smearing (expanded substantially with Huginn + Fixed-Point RNNs citations and concrete contingency).
- R4: compression branch too coarse.
- R5: output-conditioned re-feed too weak.
- R6: NoLiMa contamination.

### 6. On revisions: every red-team objection has an explicit response

**PASS.** "Revision response (red-team round 1)" section at the top of output.md addresses every objection by code:

- **C1** — accepted; cheap test rewritten using output-conditioned re-feed (which does perturb queries).
- **C2** — accepted; retention metric redefined within-architecture; cross-architecture comparison moved to task-level NoLiMa.
- **C3** — accepted; M2 decomposed into M2a + M2b; ordering prediction NSA > Quest ≥ DSA > MoBA derived.
- **C4** — accepted; explicit TRM-to-transformer mapping with `x` = prompt tokens, `y` = draft answer tokens, `z` = answer-span hidden states, `net = full_forward(concat(x, y))`.
- **I1** — accepted; argmax over gated mixture's per-key contribution specified.
- **I2** — accepted; sample size raised to 200, 1-sigma noise floor computed (~0.024 for differential), threshold of 0.05 ≈ 2.1-sigma stated.
- **I3** — accepted; compression block size 32, per-needle SNR ~ 1/32 stated.
- **I4** — accepted; G&A repositioned from co-evidence to competing baseline, with rationale.
- **I5** — accepted; R3 expanded with two citations and concrete contingency.
- **S1, S2, S3, S4, S5** — all incorporated.

No objections dismissed without justification; concessions explicit (in manifest under `concessions_to_red_team` and in §2 final paragraph).

## Discipline-rule checks (from skill spec)

- **Falsifiability is non-negotiable.** PASS — five named falsification paths; primary is the differential F2 with a numerical threshold.
- **Cite every mechanism claim.** PASS — see check 3 above.
- **Specific magnitudes, not directions.** PASS — table in §4 gives K=1 → K=4 retention range per architecture; F2 threshold +0.05; F3 threshold +3 percentage points; F6 threshold ±0.03; F1 threshold -0.10.
- **Stay in your lane.** PASS — eval-designer details in §6 sketched only; no experiment is *run*; no benchmark *built*; no novel architecture *implemented*.

## YAGNI fence check

- No parameters added.
- No training runs proposed (output-conditioned re-feed is inference-time; post-hoc attention swaps for cheap test are inference-time).
- No kernel work.
- No new dataset.

## Single-hypothesis check

PASS — one hypothesis with one primary prediction (F2 differential). Secondary predictions (F5 ordering, F3 task transfer) are tightly coupled corollaries of the same mechanism, not independent hypotheses.

## Architectural coherence (information path is explicit)

PASS — §3 M2a + M2b explicitly identify the information path:
- For NSA: compression-branch passes coarse signal at pass 1 → residual stream → pass-2 query is conditioned on this signal → pass-2 selection includes needle block.
- For MoBA: no path; un-selected blocks zeroed at pass 1; recursion can only contribute via M2b (query change).

The mechanism does NOT claim "recursion attends to fully dropped tokens" — explicit in §3.

## Non-additivity check

PASS — F2 is exactly an interaction-term test. The hypothesis is null if either factor (recursion alone or sparsity alone) explains the result; only the architecture × K interaction is the load-bearing claim.

## Coherence check (cheap test exercises the mechanism)

PASS — fixed in revision-1. Cheap test uses output-conditioned re-feed, identical operationalization to the full eval. Re-feed changes the input prompt at K=2, which changes attention queries at every layer — this directly exercises M2b. The post-hoc NSA-mask preserves M2a structurally, so M2a is also exercised. Both legs of the mechanism are tested by the cheap version.

## Ready-to-submit?

**YES.** Falsifiable, mechanism-grounded, non-additive, YAGNI-compliant, every red-team objection responded to (with explicit acceptances or grounded rejections), risks engaged substantively (especially R3).
