# Verification — Hypothesis H2 revision-1

This verification follows `superpowers:verification-before-completion` discipline. Each check is run against the revised `output.md` at this path; failures would block submission.

## Structural checks

- [x] **Hypothesis statement is in if/then form.** §2 begins "If [retrofit recipe] is applied … then [synthetic probe shows ≥10pt directional gap and Lean ratio ≥1.5×]." Two-stage chained prediction; the second stage is explicitly conditional on the first.
- [x] **At least 3 falsification criteria, each genuinely falsifiable.** §5 has 6 criteria (F1–F6). Each is metric + threshold + direction with a specific quantitative test. F1 is the gating falsification with explicit power note (n=2000 per L bin, SE ≈ 1.4 points on the difference). F3 has the fallback narrative removed per I3. F6 is new per I5.
- [x] **Every mechanism claim has a citation.** §3 cites FPR (arXiv:2503.10799) §3 / §4.2 / Eq. 12-14, TRM (arXiv:2510.04871) §4 / §4.1 / §4.3, Illusion of State (arXiv:2404.08819), Retrofitted Recurrence (arXiv:2511.07384) §4.2 / §4.3, and explicitly disclaims earlier expressivity-class claims. No uncited mechanism claims remain.
- [x] **All cited arxiv IDs were resolved via `hf_papers paper_details`.** During revision, I called paper_details on the new citations: 2511.07384 (resolved, 19 upvotes, McLeish et al, Nov 2025) and 2511.08577 (resolved, 110 upvotes, Fu et al, Nov 2025). I also re-read FPR §4 and TRM §4.1 directly to verify the corrected interpretations.
- [x] **Risks section is non-empty and honest.** §8 contains R1–R5, each with what the hypothesis still contributes. R4 (RWKV-7 dominance) explicitly removes the revision-0 fallback ("head-to-head is the contribution") and concedes it's a real falsification.

## Red-team objection responses (per spec requirement on revisions)

| Objection | Severity | Response location | Resolution |
|---|---|---|---|
| C1 — FPR contraction misrepresented | CRITICAL | "Changes from revision-0" §C1; §3 (mechanism rewrite) | Dropped contraction-vs-no-contraction structural argument; acknowledged FPR-Mamba solves S_5 per Fig 4. |
| C2 — CoT-Solves-Serial over-extrapolated | CRITICAL | "Changes" §C2; §3 ("What I am explicitly NOT claiming") | Dropped TC⁰-escape framing; explicit acknowledgment that constant-K composition stays in TC⁰. |
| C3 — Frozen Mamba unprecedented | CRITICAL | "Changes" §C3; §2; §6 (training protocol) | Replaced "frozen Mamba + TRM wrapper" with Retrofitted Recurrence continued-pretraining (all params receive gradients, curriculum 1→8, Muon optimizer). |
| C4 — Magnitude calibration | CRITICAL | "Changes" §C4; §2 (ratio prediction); §4 (pilot) | Switched to ratio prediction (1.5×); added mandatory ~80 H100-hour Lean pilot to lock magnitudes. |
| I1 — Mamba has no Lean pretraining | IMPORTANT | "Changes" §I1; folded into §6 | Continued pretraining on Nemotron-CC-Math-v1 + Mathlib fine-tuning per Retrofitted Recurrence §4.3.2. |
| I2 — F1 statistical underpowering | IMPORTANT | "Changes" §I2; §5 F1 (with explicit power note); §6 (bin sizes ≥100 on L_long) | Synthetic probe gates Lean; ratio-based threshold; bin sizes increased; CoqGym backfills L_long. |
| I3 — F3 fallback narrative | IMPORTANT | "Changes" §I3; §5 F3 ("No fallback narrative."); §8 R4 | Fallback removed; F3 is now a clean falsification; R4 explicitly accepts it as a real falsification. |
| I4 — TRM is also attractor-seeking | IMPORTANT | "Changes" §I4; §3 ("What I am explicitly NOT claiming") | Acknowledged TRM §4.1 verbatim quote about deep supervision goal; revised mechanism makes no "TRM is unconstrained" claim. |
| I5 — Param-count confound | IMPORTANT | "Changes" §I5; §5 F6 (NEW) | New falsification F6 — substrate-specificity test via Llama-3.2-1B + TRM-style retrofit. |
| S1 — Adjacent prior art | SUGGESTION | "Changes" §S1; §1 (gap framing); §6 (Llama comparator); §9 | Retrofitted Recurrence and Think-at-Hard added as first-class precedents and baselines. |

All 10 listed objections (4 critical + 5 important + 1 suggestion explicitly engaged with) have explicit responses. No objection is dismissed without a documented reason.

## Discipline rule checks

- [x] **Falsifiability is non-negotiable.** F1 explicitly kills the hypothesis at the cheap synthetic-probe stage; F2/F3/F6 each kill different aspects. No "this is interesting either way" hand-wave remains in §8 R1–R5.
- [x] **Cite every mechanism claim.** §3 paragraph by paragraph: each architectural description cites the relevant paper section. No uncited "plausibly" claims survive (the load-bearing "TRM unconstrained" claim was removed; the load-bearing "FPR contraction-bounded" claim was removed).
- [x] **Specific magnitudes, not directions.** §4 specifies a ≥10-point directional gap on Stage 1 (sign TBD) and a ratio ≥1.5× on Stage 2 (V_win/V_loss). The pilot will lock the absolute base rates; pre-registered thresholds are stated as ratios.
- [x] **Stay in your lane.** I forge a hypothesis only. I do not prescribe eval-designer's full experimental design (compute budget per arm, training-recipe hyperparameters beyond what Retrofitted Recurrence already published). I name the comparator, dataset, and stratification requirements only.

## Architectural-coherence check (per discipline rules)

- [x] **Single revised hypothesis.** One claim with two chained stages (synthetic gate, Lean ratio).
- [x] **Architectural recursion only.** TRM-style and FPR-style retrofits both modify the per-token forward pass via depth-recurrence; output token sequence remains a flat tactic chain across all arms.
- [x] **YAGNI.** No new training corpora invented (uses Nemotron-CC-Math-v1 + Mathlib, both published). No new benchmarks (PutnamBench + miniF2F + CoqGym, all standard). No new baseline architectures (TRM, FPR-Mamba, RWKV-7, Llama-3.2 + TRM-retrofit all published or directly published-recipe-on-published-checkpoint).
- [x] **Citation discipline.** Every claim has an arxiv ID. Re-interpretations of revision-0 citations are explicitly noted in §9.
- [x] **Non-additive prediction.** §4 predicts a chain-length-dependent ratio: K-vs-1 gap on L_long ≥ 2× K-vs-1 gap on L_short. This is a (mechanism × chain length) interaction, not a main effect.
- [x] **Cheaper falsification path.** §7 makes the synthetic probe both the cheapest falsification AND the gating prerequisite for Lean compute (50× compute reduction if F1 fires).

## Honest-weakening declaration

Per spec ("If after honest engagement the mechanism story breaks irreparably … it is acceptable to write a SHARPLY-REDUCED hypothesis … but flag this clearly. A weaker honest hypothesis is better than a strong false one."):

**This is a sharply-reduced hypothesis relative to revision-0.** Revision-0's mechanism — TC⁰ escape via constant-K depth iteration with TRM unconstrained vs FPR contraction-bounded — is irreparably broken (C1 and C2 both lethal against the cited papers). The revised hypothesis:

1. Makes **no expressivity-class claim** distinguishing TRM from FPR.
2. Makes **no a priori prediction of sign** (does TRM beat FPR or vice versa) — the experiment is designed to *find out*.
3. Predicts only that the variants are **empirically distinguishable** in a chain-length-dependent way under matched compute (≥10 points at synthetic L=200; ≥1.5× ratio on Lean L_long if Stage 1 passes).
4. **Gates the expensive Lean experiments behind the cheap synthetic probe.**

This is honest. Revision-0's strong claims were false. The revised claim is what is actually defensible after reading the cited papers carefully.

## Submission decision

All required checks pass. The hypothesis is falsifiable (F1 kills it cheaply; F2–F6 kill specific attribution and substrate-specificity claims), every mechanism claim is cited and consistent with the source paper, every red-team objection has an explicit response, and the honest weakening relative to revision-0 is flagged. Submitting.
