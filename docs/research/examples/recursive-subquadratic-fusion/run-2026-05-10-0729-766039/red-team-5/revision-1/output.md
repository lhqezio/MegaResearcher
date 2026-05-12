# Red-team critique of H5 (revision-1) — "Latent Plateau, Not Lock-In"

**Critiquing:** `/Users/ggix/research/docs/research/runs/2026-05-10-0729-766039/hypothesis-smith-5/revision-1/output.md`
**Targeting gap:** B7 (gap-finder-2 #7) — (architectural recursion × Tunnel-Vision under SubQ) intersection.
**Revision round:** 2 (revising round-1 critique)

---

## 1. Verdict

`APPROVE`

The polarity flip is mechanistically defensible. All three Critical objections from round 1 (C1, C2, C3) are addressed. All six Important objections (I1–I6) are addressed. Load-bearing citations are verified accurate. Falsification is a genuine commitment with symmetric thresholds. The hypothesis is publicly defensible.

I am leaving this APPROVE rather than waiting for a second revision because:

1. The remaining issues (I document four below) are *Important* but not *Critical* — they affect tightness, not soundness, and the eval-designer can incorporate them at the experiment-design stage.
2. The smith has been honest about the limitations of mechanism transfer (explicitly tagging M2's quantitative claim as "hypothesis-derived, not theorem-derived") and has dropped the "asymmetric value" hedge in body sections.
3. Multiple revisions on the same hypothesis hit diminishing returns; the swarm budget is better spent on Phase 5.

---

## 2. Round-1 critical objections — verification of resolution

### C1 — Lock-in metric L broken (when K=1 wrong, can't measure lock-in into wrong)
**Status: RESOLVED.** The smith abandons the lock-in metric entirely. The new primary outcome is Δ(s, K) = accuracy(s, K) − accuracy(s, K=1), a population-level statistic that is well-defined regardless of per-instance K=1 correctness. Section 6 explicitly states: "No lock-in metric, no per-instance K=1-baseline comparison. This is a population-level statistic, robust to per-instance K=1 wrongness." Clean resolution.

### C2 — Deep supervision misread
**Status: RESOLVED.** Section "Changes from revision-0" explicitly acknowledges: "revision-0's claim that deep supervision is an 'implicit commitment device reinforcing wrong direction' misread arXiv:2510.04871 sec. 4.1. The TRM training objective pushes z_H toward the correct y at every supervision step." I verified TRM sec. 4.1 directly: the smith's revised reading is correct ("Through deep supervision, the models learns to take any (z_L, z_H) and improve it through a full recursion process, hopefully making z_H closer to the solution"). The commitment-device framing is dropped.

### C3 — HRM-attractor mechanism doesn't transfer to TRM
**Status: RESOLVED via polarity flip.** The smith takes the path the round-1 critique explicitly offered (path c): flip polarity from "TRM inherits HRM-style lock-in" to "TRM escapes Tunnel Vision but exhibits a different failure mode (plateau)." The new mechanism is built from positive evidence about successful deep-supervised recursion (Parcae stability, Huginn empirical plateau) rather than from forcing HRM's failure mode onto TRM. This is mechanistically cleaner.

### I4 — Asymmetric-value hedge
**Status: MOSTLY RESOLVED.** The explicit "asymmetric value: confirmation reshapes; refutation strengthens" sentence from revision-0 is gone. Section 5 commits to F1/F2 as symmetric falsifications: "If F1 fires, revision-1 was too pessimistic about recursion's OOD transfer ... If F2 fires, revision-0's lock-in framing was right and revision-1's flip was wrong. Either is a clean refutation." Mild residual hedge in section 3 ("Either off-axis result is itself diagnostic") but defensible — that's an interpretive note, not a falsification escape.

### I5 — α-exponent prediction not citation-grounded
**Status: RESOLVED.** Smith moves α-exponent prediction to "Secondary diagnostics (not load-bearing for falsification)" and tags it as "hypothesis-derived, not citation-grounded."

---

## 3. Independent gap re-verification

Three independent literature queries:

**Query A — `architectural recursion sparse attention long context reasoning out of distribution`** (10 hits): closest hits are NSA (2502.11089), MoBA (2502.13189), DELTA (2510.09883), AsyncTLS (2604.07815). None composes architectural recursion with sparse attention on long-context OOD reasoning. Gap holds.

**Query B — `depth recurrent transformer long context plateau out-of-distribution generalization`** (10 hits): hit on **arXiv:2510.14095 ("Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning")**. I read sec. 3 — task is GSM8K-style modular arithmetic on computational graphs (≤128 nodes), NOT long-context retrieval. Different task domain. Gap survives but the smith should cite this for breadth.

**Query C — `looped transformer recurrence depth GSM8K plateau scaling steps`** (8 hits): hit on **arXiv:2604.21106 ("How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models")**. This paper establishes a recurrence-equivalence exponent of 0.46 — looping a block r times is equivalent in validation loss to r^0.46 unique blocks. **Important:** this is a published quantitative result that recurrence DOES provide measurable capacity gains (non-trivial, not plateau). The smith does NOT cite this. It is contradictory in spirit (though not in scope — 2604.21106 is on validation loss in pretraining, not OOD long-context retrieval). The gap survives but the smith should engage.

**Verdict on gap claim.** The gap claim survives — no published work tests TRM-style architectural recursion × subquadratic attention × long-context OOD. But the gap is *narrower* than the bibliography acknowledges: 2510.14095 studies recursive latent reasoning under OOD (different task), and 2604.21106 quantifies how much one recurrence is worth (different setting). Both narrow the framing.

---

## 4. Citation spot-checks

### Spot-check 1: Parcae (arXiv:2604.12946) sec. 3 — spectral-radius < 1 contraction claim

I read sec. 3 directly. Parcae explicitly:
- Recasts looped LM forward pass as a non-linear time-variant dynamical system: h_t = A̅·h_{t−1} + B̅·e + R̅(h_{t−1}, e).
- Linearizes to discrete LTI form: h_{t+1} = A̅·h_t + B̅·e.
- States that ρ(A̅) < 1 is the stability condition; ρ ≥ 1 implies divergence.
- Empirically verifies: divergent runs learn ρ(A̅) ≥ 1; convergent runs maintain ρ(A̅) < 1.

**The smith's representation is accurate.** ✓

**HOWEVER, an important nuance.** Parcae's argument is about *training stability* (avoiding residual explosion during pretraining), not specifically about "how does a deep-supervision-trained operator behave at inference." The smith bridges this with: "TRM-style recursion, while not architecturally identical (TRM has explicit y, z separation per arXiv:2510.04871 sec. 4.2), is a *successful instance* of deep-supervision-trained recursion that empirically converges (arXiv:2510.04871 Table 1 — 87.4% on Sudoku-Extreme)." This is a fair acknowledgment of the inferential leap — the smith is using Parcae as evidence that *stable* looped LMs satisfy ρ < 1 (so contraction is plausible), and TRM is empirically stable. That's defensible mechanism-by-analogy, with the analogy explicitly flagged. Not perfect, but honest.

### Spot-check 2: Huginn (arXiv:2507.02199) sec. 3.4 — recurrent-step plateau claim

I read sec. 3.4 directly. Verbatim: "increasing the number of recurrent steps from 4 to 32 leads to only modest gains in accuracy (from 3.11 to 4.93), and performance plateaus thereafter. In contrast, Huginn with explicit CoT achieves significantly higher accuracy (24.87/38.13)."

**The smith's quoted numbers are exactly correct.** ✓ The plateau interpretation is faithful to the paper's own framing.

**HOWEVER, the same nuance applies as Spot-check 1.** Huginn is a depth-recurrent transformer pretrained on standard LM loss — NOT a deep-supervision-toward-correct-y operator like TRM. Whether Huginn's plateau transfers to TRM is the empirical question H5 asks. This is acknowledged: section 1 says Huginn is "the most directly evidential prior" rather than asserting equivalence.

### Spot-check 3: TRM (arXiv:2510.04871) sec. 4.1 — deep supervision dynamics

I read sec. 4.1 directly. Verbatim: "Through deep supervision, the models learns to take any (z_L, z_H) and improve it through a full recursion process, hopefully making z_H closer to the solution. This means that by the design of the deep supervision goal, running a few full recursion processes (even without gradients) is expected to bring us closer to the solution."

**The smith's reading is correct** — this contradicts revision-0's commitment-device claim (which the smith now retracts). Note: TRM's own paper says recursion "brings us closer to the solution" — meaning more passes HELP. This is the training-distribution claim. The smith's prediction is specifically about OOD long-context, where this training-distribution behavior may not transfer. The hypothesis is honest about this distinction.

### Spot-check 4: Position-Bias Emergence (arXiv:2502.01951) Theorems 4.1–4.2 — extension to weight-tied recursion

This was checked in round 1 and remains a known nuance. The smith now explicitly frames the application: "under weight-tying, the iterated A^t is the t-th power of a single stochastic matrix, which by Perron-Frobenius converges to its dominant left eigenvector regardless of t-time-variance. The qualitative direction (first-token sink under causal/sliding-window mask) follows. The *quantitative* prediction is now framed as a hypothesis-derived corollary, not a theorem-derived fact." This is a fair reading: Perron-Frobenius authorizes the qualitative direction (iterated stochastic matrix → dominant eigenvector); the quantitative rate under weight-tying is an extension. The tagging is correct.

---

## 5. Mechanism critique

**M1 (deep-supervision-trained recursion contracts under bounded ρ; not instance-specific spurious attractors).** The argument is positively grounded in Parcae sec. 3 and Huginn sec. 3.4. The mechanism transfer (Parcae's stability + Huginn's empirical plateau → TRM should plateau on OOD) is acknowledged as analogical: TRM is "a successful instance of deep-supervision-trained recursion that empirically converges." The argument structure is: *if* TRM is in the same class of stable looped operators as Parcae, *and* Huginn (a closely related depth-recurrent operator) plateaus on OOD GSM8K, *then* the prediction is plausible. This is an inductive-by-analogy argument, not a deductive one. Acceptable for a hypothesis but the analogy is an empirical claim that the experiment would test.

**M2 (weight-tied recursion → first-token-sink contraction).** Section 3 now explicitly argues the Perron-Frobenius extension. The qualitative direction is correctly authorized; the quantitative claim is correctly tagged hypothesis-derived. The mechanism is internally coherent. The interaction prediction (sparse blocks recursion utility, dense allows partial compensation) is a substantive claim grounded in M2's qualitative direction.

**M3 (long-context OOD is the regime where M1+M2 manifest as plateau).** This is the load-bearing composition. The argument is: on training distribution, deep-supervision aligns iterates with task structure → recursion materially helps. On OOD long-context, iterates have no aligned target → contract toward content-blind first-token-sink direction → no improvement, but also no instance-specific wrongness (no lock-in). This is internally consistent with M1 and M2.

**Strongest mechanism critique I can construct:** TRM achieves 7.8% on ARC-AGI-2 (an OOD benchmark by design). This is not zero, suggesting TRM's deep supervision DOES produce some OOD transfer. Risk 2 in the smith's section 7 acknowledges this. If the deep-supervision objective transfers even partially to long-context OOD inputs, F1 could fire (Δ ≫ 0). The smith treats this as the meaningful empirical question that warrants the experiment, which is fair.

---

## 6. Falsifiability assessment

**F1 (Δ ≥ +10pp on long-context sparse): operationalizable.** Threshold (+10pp) is ~2× the difference SE (~5pp) at the planned sample size of 200 instances per cell. Detectable.

**F2 (Δ ≤ -10pp): symmetric to F1.** Operationalizable. Strong commitment.

**F3 (sparse-vs-dense interaction outside [-5pp, +5pp]): operationalizable.** Per the smith's power calculation, interaction-term SE ≈ 7pp. The 5pp threshold is below 1 SE, which means the falsification threshold is *not* at 2 SE for this metric. **Important issue:** F3 may fire on noise. Smith should either widen F3 to ±10pp or tag it as a secondary metric.

**F4 (positive-control gate fails): operationalizable.** Pre-registered.

**The "null prediction" structural concern.** The hypothesis predicts |Δ| ≤ 3pp (with 95% CI containing zero). F1 fires at +10pp, F2 at -10pp. There is a gray zone of |Δ| ∈ (3pp, 10pp) where neither the prediction (|Δ| ≤ 3pp) nor falsification (|Δ| ≥ 10pp) cleanly fires. With per-cell SE ≈ 3.5pp and difference SE ≈ 5pp, an observed Δ = 7pp would be ~1.4 SE away from zero — not clearly null, not clearly not-null.

**Mitigating factor:** The smith's commitment-stated criterion is "95% CI containing zero" — which IS a well-defined statistical test. If the 95% CI contains zero, the prediction holds; if the CI excludes zero AND the point estimate exceeds 10pp, F1 fires. Between those, the result is "directionally suggestive but inconclusive" — a real outcome that should be allowed in honest experiments. This is acceptable.

**A rigorous version would use TOST (two one-sided tests) for equivalence:** test whether |Δ| < some equivalence margin δ. Smith doesn't go this far but could be asked to in eval-designer.

**Overall:** F1 and F2 are clean, symmetric, well-thresholded. F3 has a noise-floor issue. F4 is a setup gate. The hypothesis is genuinely falsifiable. **Important suggestion for eval-designer:** consider TOST framing for the null prediction itself; widen F3 threshold.

---

## 7. Strongest counter-argument (steelman)

**Steelman: Recurrence is genuinely worth ~r^0.46 unique blocks (per arXiv:2604.21106). At K=8, that's ~2.6× equivalent depth. Even on OOD long-context, this should produce a measurable Δ — not plateau.**

The strongest opposing case has three prongs:

1. **Recurrence has measurable empirical value.** arXiv:2604.21106 establishes a recurrence-equivalence exponent of 0.46 across 116 pretraining runs. Even granting that this is on-distribution validation loss rather than OOD long-context, the result that recurrence has a non-trivial scaling exponent suggests recurrence does compute *something* useful. If the trained operator's recursion buys real capacity, applying it to OOD inputs should produce a measurable (perhaps degraded) improvement — not exact zero.

2. **TRM partially transfers to OOD already.** TRM's 7.8% on ARC-AGI-2 (a task it wasn't trained on) is non-trivial — strictly above random. This suggests deep-supervision-trained iterates DO carry some general-purpose refinement capacity that survives modest OOD shift.

3. **The plateau in Huginn is regime-specific.** Huginn's 4→32 plateau (3.11→4.93) is on GSM8K with suppressed CoT, where the model is fundamentally being asked to do internalized arithmetic without the chain. The plateau may be specific to "math you can't do without working memory" — not generic OOD transfer. Long-context retrieval is a different task profile (re-attention, not multi-step computation).

If any of these holds — and at least one likely does — F1 fires (Δ ≫ 0). The plateau prediction is then refuted.

This is the right outcome for a hypothesis on the strong-falsification axis: refutation by F1 firing would be a *positive* result for the fusion thesis (architectural recursion DOES help on long-context OOD). The smith acknowledges this in Risk 2. The hypothesis remains valuable because — *if* the steelman's prediction is right — knowing that experimentally is a substantive finding.

**Bottom line on the steelman:** the opposing case is real and credible. The hypothesis commits to F1/F2 falsification, so it would lose if the steelman is right. This is the falsification commitment doing its job.

---

## 8. Severity-tagged objections

### Critical (must fix before APPROVE) — none.

All three critical objections from round 1 are resolved.

### Important (should fix; eval-designer should address)

**I-A: Missed citation — arXiv:2604.21106 ("Iso-Depth Scaling Laws for Looped Language Models").** This paper directly quantifies the value of recurrence (exponent 0.46). It is contradictory in spirit to a strict plateau prediction and should be engaged with. The smith's claim is OOD-specific, but this paper sets a baseline expectation that deserves explicit positioning.

**I-B: F3 noise-floor issue.** F3 threshold (5pp on the interaction term) is below the smith's own stated interaction SE (~7pp). Falsification could fire on noise. Either widen F3 to ±10pp or demote F3 from primary falsification.

**I-C: Mechanism transfer is analogy-by-related-architecture.** Parcae is not deep-supervised; Huginn is not deep-supervised in the TRM sense. Both are used as evidence for what TRM-style recursion will do. The smith acknowledges this (M1's "successful instance of deep-supervision-trained recursion") but the bridge is not airtight. *Mitigation: this is exactly what the experiment tests* — so a slightly loose mechanism is acceptable for a hypothesis that is itself the empirical test.

**I-D: TOST framing for null prediction.** A genuine null prediction (|Δ| ≤ 3pp) is harder to confirm than a directional one. The smith's "95% CI contains zero" is OK but a TOST equivalence test against ±5pp bounds would be cleaner. Defer to eval-designer.

### Suggestion (nice to have)

**S-A: Engage with arXiv:2510.14095 (Recursive Latent Space Reasoning for OOD).** Different task, but conceptually adjacent. One-paragraph positioning would strengthen the bibliography.

**S-B: Consider what would happen if Δ_train(K=8) on Sudoku-Extreme is negative or small under the matched-parameter K=1-vs-K=8 framing.** TRM's published numbers compare different (T,n) configurations, not K=1-vs-K=8 at matched T,n — so the smith's "expected delta should be at least 20pp" assumption deserves verification before pre-registration.

---

## 9. Recommendation to hypothesis-smith and eval-designer

**To smith:** No revision required. The hypothesis is technically defensible. Optional improvements: add citation 2604.21106 to bibliography with positioning, consider widening F3, consider TOST for null prediction.

**To eval-designer:**
1. Verify that TRM's K=1-vs-K=8 matched-parameter Sudoku-Extreme delta is ≥ 20pp BEFORE locking in F4 threshold. If not, F4 will trivially fail and the experiment becomes a setup-validation rather than a hypothesis test.
2. Consider TOST equivalence testing for the null prediction: |Δ| < 5pp at α=0.05.
3. Widen F3 threshold to ±10pp or demote to secondary metric.
4. Engage with 2604.21106's recurrence-equivalence exponent: if this paper's scaling holds at K=8, Δ should be non-zero on the matched-distribution control. The smith's positive-control prediction (Δ_train ≥ 20pp) is consistent with that, but the long-context prediction (Δ ≈ 0) is implicitly a claim that recurrence's value collapses entirely under sparse + long-context. This is a sharp commitment worth flagging in pre-registration.

---

## 10. Verdict (final)

`APPROVE`

The polarity flip is mechanistically defensible. C1, C2, C3 from round 1 are resolved. I1–I6 are addressed. Load-bearing citations (Parcae sec. 3, Huginn sec. 3.4, TRM sec. 4.1) are verified accurate. F1/F2 are symmetric strong-falsification commitments. The remaining issues are Important (not Critical) and can be tightened at the eval-designer stage.

I would defend this hypothesis publicly. APPROVE.
