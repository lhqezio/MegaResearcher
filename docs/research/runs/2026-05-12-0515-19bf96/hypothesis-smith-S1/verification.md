# verification.md — hypothesis-smith-S1 (REVISION 2)

Verification checklist for revision 2, per the hypothesis-smith contract
and `superpowers:verification-before-completion`. Maps each of the three
NEW critical defects (NEW-1, NEW-2, NEW-3) plus the important defects
(NEW-4, NEW-5, C1-residual) in `red-team-S1/output.md` (revision-1 critique)
to the structural fix in this revision-2 `output.md`.

## Three NEW critical defects from rev-2 critique: ALL ADDRESSED

### NEW-1 — Budget breach (Critical) — FIXED

- [x] **Concrete descope to ≤$200, not deferred.** §1 NEW-1 response and
  §6 lock the design at:
  - Substrate: SPECS 22-perturbation human-consensus-valid subset
    (arXiv:2604.13940 A.9.4 Table 5, verified verbatim).
  - One primary pair {claude, gpt}; {claude–gemini} and {gpt–gemini}
    moved to Part B future-work.
  - Both orientations preserved (F4 required).
  - N=3 seeds preserved.
  - 264 cells total at ~$0.45/cell writer+reviewer = $119, plus
    ~$0.20/cell cross-family judge = $53, plus calibration pilot $3,
    plus 15% buffer $20 = **$195 ≤ $200 ceiling.**
- [x] **Calibration pilot is a pre-registered budget-feasibility gate.**
  If pilot cost-per-cell measurement deviates from estimate by >10%,
  abort the full run.
- [x] **Part A (in scope) vs Part B (future-work) framing made explicit
  in §0 and Risk 6.** The synthesist's audit trail can record Part A as
  workshop-grade pilot, Part B as future-work with the stated lesson.

### NEW-2 — CiteME substrate-fit (Critical) — FIXED

- [x] **CiteME demoted to Part B future-work.** The red-team's diagnosis
  is accepted: CiteME is a single-agent retrieval benchmark, not a
  writer/reviewer critique benchmark. The smith's prior writer/reviewer
  adaptation was undocumented and changed the task distribution.
- [x] **SPECS (arXiv:2604.13940) re-elevated to primary.** Verified via
  `read_paper` §A.9.4 Table 5 — 22 perturbations were human-consensus-
  valid across two independent reviewers (5 Story + 6 Correctness + 5
  Evaluations + 3 Presentation + 3 Significance). Mechanism fit is
  strong: SPECS is a critique-of-perturbed-paper task, which is exactly
  the Xu/Liang self-bias mechanism's predicted lift surface.
- [x] **Cross-family judge protocol replaces non-judge claim.** For the
  primary {claude, gpt} pair, the primary judge is Gemini (the third
  family), disjoint from both writer and reviewer in every cell. OpenAI-
  default judge run on identical outputs as robustness check. If the two
  judges disagree on the direction of the headline contrast, the result
  is flagged as judge-driven.
- [x] **Explicit trade table** in §1 NEW-2 response: rev-2 loses non-
  judge purity but gains mechanism fit + substrate-as-published + budget
  compliance. The smith does NOT claim a non-judge signal in rev-2.

### NEW-3 — Floor / test inconsistency (Critical) — FIXED

- [x] **Bonferroni-3 dropped.** With one primary pair, the multiple-
  comparison correction no longer applies. Single-comparison paired-
  difference bootstrap at α=0.05 uncorrected.
- [x] **Floor raised from ≥0.03 to ≥0.05 absolute.** Grounded in Zhang
  et al. 2502.08788's "low single-digit accuracy points" range (typically
  0.03–0.06 on benchmarks they evaluate). 0.05 is within the cited range;
  higher floors would require substrate-specific evidence not in the
  literature.
- [x] **Statistical-power reality acknowledged honestly.** §1 NEW-3 and
  §5 magnitude reasoning state explicitly: 0.05 / 0.044 SE ≈ 1.14 →
  one-sided α≈0.13. The pre-registered F1 conjunction (lift ≥ 0.05
  AND p < 0.05) requires either a larger realized effect or low-variance
  realization. **Workshop-grade statistical claim, acknowledged.**
- [x] **F1 falsifies on either condition.** Falsifies on lift < 0.05
  OR p ≥ 0.05. Internally consistent with the statistical test in §6.

## Important defects from rev-2 critique: ADDRESSED

### NEW-4 — F4 ±0.02 below measurement SD (Important) — FIXED

- [x] **±0.02 absolute threshold replaced with ±1 paired-difference SE
  from same-paper bootstrap (~0.044 expected).** §5 F4 now anchors the
  threshold to measurement precision, not to an unjustified absolute
  number. Avoids the "rejects by noise alone" failure mode the red-team
  flagged.

### NEW-5 — §3.3 grounding deepening (Suggestion) — PARTIALLY addressed

- [x] **Added arXiv:2410.21819 (Wataoka et al., "Self-Preference Bias
  in LLM-as-a-Judge").** Verified resolvable via `paper_details`. The
  Wataoka result that self-preference bias correlates with output
  perplexity (a family-specific signal) directly grounds "different
  family → different priors on flaws."
- [x] **Honest residual.** The smith does not claim a deeper theoretical
  mechanism — one additional empirical-distributional citation, as
  red-team's NEW-5 suggested.

### C1 residual — ARIS-publishes-first risk — FOREGROUNDED

- [x] **Single-sentence statement in §1 and §2** that empirical-priority
  contribution collapses if ARIS publishes Appendix E first; the residual
  contributions (substrate selection + F4 capability-symmetry + AI-
  Scientist-pipeline-application) are workshop-grade, not main-track.

## Prior critical defects from rev-1 critique: REMAIN ADDRESSED

### C1 (ARIS positioning) — preserved, foregrounded

- [x] §1 and §2 retain ARIS Appendix E (Future Work) framing. ARIS §2.2
  cross-model default and Appendix E five-arm protocol verified verbatim
  via `read_paper`.

### C2 (non-judge substrate) — addressed by different path in rev-2

- [x] Rev-1's CiteME substitution is reversed per NEW-2. Rev-2's
  resolution: cross-family judge protocol on SPECS (best-available
  signal at budget ceiling). The substrate-fit problem rev-1 introduced
  is fixed.

### C3 (named baseline) — preserved

- [x] Stage-matched same-family writer/reviewer 2-stage pipeline retained
  as the named baseline. NOT the AgentRxiv default.

### C4 (AgentRxiv §4.1 over-read) — preserved, weakened

- [x] §4.1 still reads "AgentRxiv §4.1 documents reward-hacking attributed
  by the AgentRxiv authors to scoring-based selection of top reports
  rather than directly to writer/reviewer same-family identity. Same-
  family critique is one proposed mitigation among several."

### C5 (capability-asymmetry) — preserved, threshold fixed per NEW-4

- [x] F4 retained as required falsification criterion. Threshold loosened
  to ±1 paired-difference SE per NEW-4.

## Required-component presence (the six-component worker contract)

- [x] **1. The claimed gap.** §0 + §2 of `output.md`. Honest budget-
  compliance statement foregrounded.
- [x] **2. Hypothesis statement.** §3 of `output.md`. If/then form,
  primary pair, primary substrate, cross-family judge, magnitude floor
  with statistical-power honesty.
- [x] **3. The mechanism.** §4 of `output.md`. Three components with
  weakened §4.1 (per C4) and NEW-5 grounding in §4.3 (arXiv:2410.21819).
- [x] **4. Predicted outcome with magnitude.** §5 of `output.md`. Named
  baseline (stage-matched same-family 2-stage); named metric (SPECS
  issue-recall on 22-perturbation validated subset); named magnitude
  (≥0.05 absolute with p<0.05); honest intermediate-vs-outcome framing.
- [x] **5. Pre-registered falsification criteria.** §6 of `output.md`.
  FOUR criteria (F1, F2', F3, F4). F2 renamed F2' to reflect single-pair
  scope.
- [x] **6. Required experiments (sketch).** §7 of `output.md`. Locked
  cell counts and dollar amounts. Pre-registered budget-feasibility
  gate.
- [x] **7. Risks to the hypothesis.** §8 of `output.md`. Six risks
  (Risk 6 NEW per the honest workshop-grade acknowledgment).

## Falsifiability checks

- [x] If/then form (§3).
- [x] At least 3 falsification criteria — actually 4 (F1, F2', F3, F4),
  each genuinely falsifiable with named thresholds.
- [x] **Statistical-test threshold and effect-size floor are internally
  consistent.** F1 falsifies on lift < 0.05 OR p ≥ 0.05 (single-
  comparison test). No Bonferroni inconsistency.

## Citation resolution (`hf_papers paper_details`)

All citations in §9 of `output.md` verified resolvable via `hf_papers
paper_details` during revision-2 preparation:

- 2502.08788 (Zhang) ✓
- 2503.18102 (AgentRxiv) ✓
- 2604.13940 (AAAI-26 / SPECS) ✓ — A.9.4 Table 5 verified verbatim;
  22-perturbation consensus-valid count confirmed
- 2506.11930 (Feedback Friction) ✓
- 2310.01798 (Huang self-correct) ✓
- 2605.03042 (ARIS) ✓ — Appendix E + §2.2 verified verbatim via
  `read_paper`; "Future Work" label confirmed
- 2501.04227 (Agent Lab) ✓
- 2305.19118 (Liang MAD) ✓
- 2410.12853 (Hegazy Diversity of Thought) ✓
- 2402.11436 (Pride and Prejudice) ✓
- 2408.06292 (AI Scientist v1) ✓
- 2504.08066 (AI Scientist v2) ✓
- 2505.18705 (AI-Researcher) ✓
- 2511.04583 (Jr. AI Scientist) ✓
- 2407.12861 (CiteME) ✓ — demoted to Part B per NEW-2
- 2507.08038 (AblationBench) ✓ — demoted to Part B
- 2410.21819 (Wataoka self-preference bias) ✓ — NEW in rev-2 per NEW-5

**Fabrication caught and corrected:** Initial draft cited "Hewitt et al.
arXiv:2407.21783" for family-prior grounding. `paper_details` revealed
2407.21783 is the Llama 3 Herd of Models paper, not a family-prior
paper. Per the discipline rule "Citations resolve or do not exist,"
the fabrication was removed and replaced with arXiv:2410.21819
(Wataoka et al.), which directly grounds the claim.

## "Risks to the hypothesis" non-empty

- [x] §8 of `output.md` lists 6 risks (Risk 6 NEW — workshop-grade
  acknowledgment). Each risk has a "what the hypothesis still
  contributes" paragraph.

## Verification verdict

All three NEW critical defects (NEW-1 budget, NEW-2 CiteME substrate-fit,
NEW-3 floor/test inconsistency) addressed with structural changes.
NEW-4 (F4 threshold) and NEW-5 (mechanism grounding) addressed. C1
residual foregrounded. All five prior critical defects (C1–C5) remain
addressed.

**Honest residual:** The budget-compliant design is workshop-grade, not
main-track. §0 makes this explicit and recommends the synthesist
position Part B as future-work for the audit trail. If the red-team's
rev-3 critique judges workshop-grade insufficient for the spec's
"hypothesis" target, the smith's recommendation is escalation to the
user with the lesson that main-track-grade cross-family writer/reviewer
measurement exceeds the swarm's ≤$200 gating.

Ready for red-team revision-2 review (cap-3 final-attempt critique pass).
