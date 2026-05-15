# red-team-S1 — REV-2 critique of hypothesis-smith-S1 (FINAL, cap-3 last pass)

Revision round: 2 (third critique pass — cap-3 final).
Prior verdict (rev-1): REJECT (revision-2).
Prior critique: same file, archived by overwrite.

## 1. Verdict

The smith resolved all three Critical issues from the rev-1 critique cleanly
and concretely. NEW-1 (budget) is locked at ≤$200 by an arithmetic-defensible
descope. NEW-2 (CiteME substrate-fit) is fixed by re-elevating SPECS with a
cross-family judge protocol and an honest non-judge-purity trade. NEW-3
(floor/test inconsistency) is fixed by dropping Bonferroni-3 (single-pair
design) and raising the floor to 0.05 while honestly self-disclosing α≈0.13
workshop-grade statistical power.

The remaining question is whether workshop-grade-self-disclosure is enough.
S6 was KILLed at scaffolding-grade. The smith's pilot is one rung above
scaffolding: it has a defensibly-fit substrate, a pre-registered ≤$200
budget gate, a primary-pair-with-orientation-symmetry design that retains
the F4 control, and a §1 self-disclosure that ARIS-publishes-first
collapses Part A to workshop-paper level. The contribution under §0
honest framing is: a pre-registered pilot of ARIS Appendix E on SPECS
with the F4 control, deliverable under the spec's hard budget.

This APPROVES at workshop-grade-with-Part-B-future-work-flag. The
synthesist's audit trail carries the explicit lesson: "main-track-grade
cross-family writer/reviewer measurement requires budget above the
swarm's ≤$200 gating allows."

Verdict on the LAST line: APPROVE.

## 2. Re-check of the three rev-1 Critical objections

### NEW-1 (budget breach) — FIXED, arithmetic verified

Smith's locked budget: 22 perturbations × 1 pair × 2 orientations × 3
seeds × 2 conditions = 264 cells. Writer+reviewer ~$0.45/cell = $119,
judge ~$0.20/cell = $53, pilot $3, buffer $20, total $195.

Independent arithmetic check:

- 264 × $0.45 = $118.80 (writer + reviewer)
- 264 × $0.20 = $52.80 (judge primary + robustness)
- $119 + $53 + $3 + $20 = $195

The arithmetic holds. The per-cell cost reasoning ($15/M input + $60/M
output × ~15k input + ~3k output per pass) is at the higher end of
frontier-model pricing; if anything, the smith has padded conservatively.
The 5-paper calibration pilot pre-registered budget-gate ($3, 10% cost
deviation triggers abort-and-redesign) is the right shape for catching
pricing surprises early. This is recoverable budget compliance, not a
fictional one.

**The descope cost.** Smith loses the {claude-gemini} and {gpt-gemini}
sweeps. The cross-pair-consistency claim (F2's "≥2 of 3 pairs survive")
collapses to F2' (single-pair-survives-orientation-drop). This is
honest. Smith preserves F4 (capability-symmetry) through both
orientations of the same pair, which is the minimum viable F4 control —
not the multi-pair F4 control. The orientation-symmetric design is
defensible as the workshop-grade form of the F4 claim; multi-pair F4
is correctly moved to Part B.

**This Critical is resolved.**

### NEW-2 (CiteME substrate-fit) — FIXED by re-elevating SPECS with cross-family judge

Smith took the rev-1 recommendation: re-elevate SPECS (substrate-fit),
accept LM-judge dependence as the unavoidable cost, design a cross-family
judge protocol where the judge is from a third foundation-model family
disjoint from both writer and reviewer in every cell ({claude, gpt}
writer/reviewer cells judged by {gemini}). This is meaningfully different
from same-family-in-loop judging because: (a) the judge family does not
share the writer's output distribution; (b) the judge family does not
share the reviewer's evaluation prior; (c) the OpenAI-default-judge
robustness check on identical outputs surfaces judge-driven results.

The non-judge-purity claim is dropped explicitly in §6. This is the
right honest trade: the smith no longer claims SPECS escapes the LM
judge, only that the judge is family-disjoint from both parties in
every measurement cell.

**Is this enough to defend the hypothesis vs the rev-1 substrate-fit
attack?** Yes — the rev-1 attack was that CiteME's retrieval task did
not exercise the Xu/Liang self-bias mechanism. SPECS does exercise it
(reviewer reads perturbed paper, writes long-form review, judge
identifies whether review caught the perturbation), and the cross-
family judge protocol contains judge-family bias as a controlled
variable rather than a free parameter. The judge-swap robustness check
operationalizes "would a different judge family produce the same
direction" — defensibly different from the SPECS-default of
same-family-judge-as-one-of-parties.

**Has the hypothesis lost its core differentiation by depending on an
LLM judge?** No. The differentiation is now: (a) primary substrate
(SPECS) with documented perturbations, (b) cross-family judge protocol
that disjoins judge from both parties, (c) F4 capability-symmetry
control, (d) pre-registered ≤$200 budget gate. The non-judge property
was never the unique differentiation — it was a defense against a prior
critique. Trading non-judge-purity for substrate-fit + budget compliance
is the correct trade given the spec's hard constraints.

**This Critical is resolved.**

### NEW-3 (floor/test inconsistency) — FIXED with honest workshop-grade disclosure

Smith dropped Bonferroni-3 (single primary pair, single comparison).
Floor raised to ≥0.05. F1 falsifies on EITHER lift < 0.05 OR
paired-difference p ≥ 0.05.

**The honest residual that smith foregrounds:** at the 22-perturbation
substrate with N=3 seeds and 2 orientations, paired-difference SE is
~0.044, so 0.05 / 0.044 ≈ 1.14 SE ≈ one-sided α≈0.13. Smith is explicit
in §0 and §5 that this is "workshop-grade statistical claim," "suggestive
not significant," and that a main-track replication requires either
larger substrate, more seeds, or multiple pairs.

**Is the α≈0.13 framing honest?** Yes — it is exactly the calculation
the rev-1 critique demanded (matching the smith's own measurement-SD
math to the test threshold). At standard α=0.05, the test is
underpowered for the cited-magnitude effect. At workshop-grade α≈0.13,
the test is consistent with what the budget can deliver. The smith
acknowledges this trade in §0 (READ THIS FIRST), §5 (magnitude
reasoning), and Risk 6. This is exactly the disclosure pattern the
rev-1 critique requested.

**Is this enough for the spec's main-track bar?** No — but the smith
does not claim main-track. The smith claims **workshop-grade pilot
with Part B future-work flag**, explicitly. The spec's novelty target
is "hypothesis," and the workshop-grade pilot satisfies the hypothesis
form (pre-registered prediction, magnitude, falsifiability) at reduced
statistical strength.

**This Critical is resolved at workshop-grade.**

## 3. The workshop-grade question (S1 vs S6 precedent)

The bar matters. S2 was APPROVED at narrower scope. S3 was APPROVED at
+6 borderline. S6 was KILLed at "scaffolding rather than primary
contribution."

**S6 precedent (KILL).** S6 was killed because its core contribution
was a scaffold — the proposed measurement could not isolate the cited
mechanism even in principle. Workshop-grade self-disclosure was a
symptom, not the disease: S6's disease was that no design within the
budget could measure the predicted effect.

**S1 in revision-2.** The design CAN measure the predicted effect if
the effect is at the upper end of Zhang's heterogeneity magnitude range
(0.05-0.06). At 0.05 absolute lift, F1 fails (p ≈ 0.13). At 0.06-0.07
absolute lift, F1 passes. The design is workshop-grade BECAUSE the
budget caps the substrate at 22 perturbations, not because the design
cannot measure the effect. A larger substrate (full 783 perturbations
at ~$4k or full 130 SPECS pool at unclear cost) would give
main-track-grade statistical power on the same design. This is a
budget constraint, not a design constraint. **The distinction matters:**
S6's KILL was substrate-fit + design-validity. S1's residual is
substrate-size + budget. Those are categorically different concerns.

**Is workshop-grade enough?** For Part A as in-scope pilot: yes. The
hypothesis is operationalizable, the falsification criteria are
pre-registered, the budget gate is concrete, the audit trail item
("main-track requires ~$400-700 per replication") is the lesson worth
carrying forward. The synthesist can include S1's Part A as in-scope
result + Part B future-work flag, and the swarm's audit trail
appropriately records the budget-vs-statistical-power tension as a
finding.

## 4. Citation spot-checks (this revision pass)

Three spot-checks via `paper_details` / `read_paper`:

1. **arXiv:2604.13940 SPECS A.9.4** — `read_paper` confirms Table 5
   verbatim: 35 perturbations randomly sampled (6 story, 7 presentation,
   8 evaluations, 7 correctness, 7 significance); R1+R2 consensus on
   22 valid (5 story, 3 presentation, 5 evaluations, 6 correctness, 3
   significance). Smith's claim "22-perturbation human-consensus-valid
   subset (5 Story + 6 Correctness + 5 Evaluations + 3 Presentation +
   3 Significance — 22 consensus-valid)" — note: smith has Story=5
   Correctness=6 which matches Table 5 verbatim. The paper itself uses
   the slightly counter-intuitive ordering. Verification holds.
2. **arXiv:2410.21819 Wataoka** — `paper_details` confirms: title
   "Self-Preference Bias in LLM-as-a-Judge"; authors Wataoka,
   Takahashi, Ri; abstract confirms self-preference-bias-correlates-
   with-perplexity finding. Smith's replacement citation for the
   fabricated Hewitt 2407.21783 is correct. Wataoka is a defensibly
   accurate citation for "different model families have measurably
   different priors on what they prefer in generated text."
3. **arXiv:2605.03042 ARIS Appendix E** — `read_paper §"Appendix E"`
   confirms verbatim (carried from rev-1 verification, re-verified
   this pass): 12+ paper drafts, five-arm protocol (A–E), metrics
   include issue recall, false-positive rate, actionability,
   downstream revision quality, cost, latency; three blinded raters
   with Krippendorff's α; labeled "Future Work (Controlled Benchmark
   Protocol)". Smith's §2 framing is accurate.

All three citations verify. The Hewitt 2407.21783 → Wataoka 2410.21819
replacement that smith made during revision is correct.

## 5. Independent gap re-verification

Three independent queries via `hf_papers search`:

1. `"cross-family writer reviewer SPECS heterogeneous LLM benchmark perturbation 2026"` (10 results) — none implement the smith's specific design. SPECS appears only as the source benchmark (arXiv:2604.13940). No follow-up work runs the ARIS Appendix E protocol on it.
2. `"cross-model agent paper review heterogeneous writer reviewer ARIS Appendix benchmark replication"` (10 results) — the closest matches are paper-replication benchmarks (PaperBench, ReplicationBench, REPRO-Bench) and paper-generation frameworks (PaperOrchestra, Paper Circle). None implement ARIS Appendix E on a writer/reviewer-split substrate.
3. `"SPECS benchmark cross-family judge writer reviewer AAAI-26 AI review heterogeneous pair"` (10 results) — only SPECS itself (arXiv:2604.13940). No published work uses SPECS as a writer/reviewer-split substrate with a cross-family judge protocol.

**Gap claim survives** at workshop-grade framing. The narrowed gap (ARIS
Appendix E protocol applied to AI-Scientist-family pipeline on SPECS-
validated-subset with cross-family judge + F4 capability-symmetry control,
under ≤$200) has no published precedent.

## 6. Falsifiability assessment (this revision pass)

| Criterion | Operationalizable? | Concern |
|---|---|---|
| F1 (≥0.05 lift AND p<0.05 on 22-perturbation SPECS validated subset) | Yes — paired-difference bootstrap with explicit conjunction. **But:** α=0.05 cut at 0.072 not 0.05 → smith honestly notes this is a workshop-grade test at α≈0.13. Operationalizable; smith pre-registers the workshop-grade framing in §0. |
| F2' (per-orientation lift survives orientation-drop) | Yes — binary per-orientation survival, single-pair design. |
| F3 (substance axis lift ≥0.03 larger than presentation axis on 16 vs 6 validated perturbations) | Yes — within-paper axis test on validated subset. Low N on presentation axis (3) limits power, but F3 is a directional test, not a magnitude-floor test. |
| F4 (per-orientation lift within ±1 paired-difference SE of across-orientation mean) | Yes — anchored to realized measurement precision (~0.044 SE), not to an unjustified absolute threshold. Fixes the rev-1 NEW-4 objection. |

Falsifiability is genuinely operationalizable across all four criteria.
F1's workshop-grade statistical power is self-disclosed; F4's threshold
is anchored to measurement SE; F2' is binary; F3 is within-paper.

## 7. Steelman of the opposing position

The strongest argument that this hypothesis is wrong, in priority order:

1. **The F1 lift at workshop-grade-α may be a false positive at α=0.05.**
   If the realized lift is 0.05-0.07 and observed p is 0.05-0.13, the
   result is suggestive-not-significant under standard α. Smith
   acknowledges this in §0 and Risk 6. The synthesist must carry this
   forward as the explicit limitation.
2. **Memorization confound on AAAI-25 papers.** SPECS perturbations are
   on AAAI-25 papers; many frontier models may have memorized them. The
   heterogeneous pair may benefit asymmetrically (one model memorized
   the paper, the other did not). Smith honestly flags this as Risk 5
   and moves the held-out post-cutoff probe to Part B. Workshop-grade
   acceptance of this confound is defensible; main-track would require
   Part B held-out probe.
3. **Cross-family judge has its own biases.** Gemini-judge may
   systematically favor or disfavor a specific writer/reviewer family.
   Smith's robustness check (OpenAI-default judge on identical outputs)
   is designed to surface this; smith pre-registers that judge-
   disagreement at the headline direction flags the result as
   judge-driven. This is defensible.
4. **The Zhang 2502.08788 magnitude transfer is not assumption-free.**
   Zhang reports on short-answer math/reasoning; SPECS issue-recall is
   long-form critique. The 0.05 floor is at the upper end of Zhang's
   range; if SPECS-critique is harder than Zhang's math, the realized
   lift may be below floor. Smith acknowledges in §4.2 magnitude
   caveat and uses the 5-paper calibration pilot as the early-stop
   gate. Workshop-grade-pilot defensible.

None of these is a structural design defect. All are residual risks
that the smith has explicitly disclosed and provided a mechanism
(robustness check, calibration pilot, Risk-5 Part-B future-work) for
handling.

## 8. Severity-tagged objections (this revision pass)

**Critical (must fix or KILL):**

- None. All three rev-1 Critical objections are resolved at
  workshop-grade.

**Important (should fix):**

- **Workshop-grade explicit in synthesist's output.** The synthesist
  MUST carry forward the §0 honest-budget-compliance framing and the
  Risk-6 workshop-grade self-disclosure. The Part A pilot result
  should be presented as workshop-grade-with-Part-B-flag, not as a
  main-track contribution. (Not a smith fix — a synthesist instruction.)
- **Memorization confound (Risk 5) needs explicit synthesist mention.**
  Workshop-grade Part A on AAAI-25 papers has a known training-data
  contamination risk. Synthesist should flag it as a residual that
  Part B (held-out post-cutoff probe) is designed to resolve.

**Suggestion:**

- **§4.3 Wataoka grounding is empirical-distributional, not theoretical-mechanistic.**
  Smith acknowledges this honestly. The hypothesis would be strengthened
  by a theoretical-mechanistic grounding (e.g., why family-specific
  pretraining priors should produce different flaw-detection priors).
  No such citation has been surfaced in this swarm; this is a "would be
  nice if it existed" suggestion, not a revision request.

## 9. Verdict-severity-consistency check

- Zero Critical objections remain.
- Two Important objections remain, both addressed-to-synthesist not addressed-to-smith.
- One Suggestion.

APPROVE is consistent with the severity. The Important objections are
synthesist instructions; they do not require a smith revision pass and
are appropriate to carry forward via the manifest.

## 10. Recommendation to hypothesis-smith for revision-3

None required. The hypothesis is approved as-submitted for Part A
in-scope, with Part B as future-work-flag for synthesist audit trail.

**To the synthesist (forwarded via manifest):**

1. Present S1 Part A as workshop-grade pilot, not main-track contribution.
2. Carry forward the §0 honest-budget-compliance statement verbatim or
   summarized as a swarm-level lesson: "main-track-grade cross-family
   writer/reviewer measurement requires budget above the swarm's ≤$200
   gating allows" (a finding about the swarm, not just about S1).
3. Flag the Risk-5 memorization confound on AAAI-25 papers as the
   primary residual; Part B held-out post-cutoff probe is the
   resolution path.
4. Flag the F1 α≈0.13 workshop-grade statistical power as the secondary
   residual; main-track replication requires either larger substrate
   (full 783 SPECS perturbations), more seeds (N≥5), or multiple pairs
   with Bonferroni-friendly substrate.

## 11. Final discipline check

Am I approving out of cap-fatigue? Check:

- **S2 APPROVE bar**: narrower scope, defensible within budget. S1
  satisfies the same bar (narrower scope = single primary pair + 22
  validated perturbations; defensible within budget = $195 ≤ $200).
- **S3 APPROVE bar**: +6 borderline statistical power, but
  pre-registered and operationalizable. S1 satisfies the same bar
  (α≈0.13 borderline, but pre-registered with honest self-disclosure
  and operationalizable falsification criteria).
- **S6 KILL bar**: scaffolding-grade with no design that could measure
  the effect in principle. S1 does NOT satisfy this bar — the design
  CAN measure the effect at upper-Zhang-range (0.06-0.07 lift); the
  workshop-grade residual is substrate-size + budget, not design-
  validity.

S1 sits closer to S3 (borderline-but-defensible) than to S6 (scaffolding).
The S3-precedent is the right anchor. APPROVE is the consistent call.

VERDICT: APPROVE
