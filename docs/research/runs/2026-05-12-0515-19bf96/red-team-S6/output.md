# red-team-S6 — Adversarial critique of S6 revision-1 (Ablation-coverage flag-handoff worker)

## 1. Verdict

**VERDICT: KILL (irrecoverable)**

The smith addressed three of the four prior Critical objections honestly (C2 metric, C3 LM-judge framing, C4 AbGen). But the revision's load-bearing maneuver — reframing the mechanism from "rubric promotion" to "file-handoff substrate" — does not survive scrutiny. The file-handoff-vs-in-context distinction collapses under the diagnostic arm the smith himself added: the smith concedes in R3 that if the two arms produce indistinguishable F1@5 the C1 collapse re-materializes at the architectural level, and there is no published reason to expect them to differ. The smith further concedes in §0 that "+3 F1pp is not a main-track-conference primary contribution by itself" and recommends the synthesist position S6 as scaffolding. The honest framing the smith offers IS the kill rationale: a hypothesis whose author asks the synthesist to position it as scaffolding rather than primary contribution should be killed and flagged as future-work, not run through Phase 5 to produce an eval-designer document for a workshop-grade lift on top of an already-stretched mechanism.

The kill is reinforced by three new defects the revision introduces: (i) a second metric mis-statement — the smith repeats the "38% on AuthorAblation" figure in §1 that the actual AblationBench Table 4 contradicts (max LM-Planner F1@5 = 0.31, not 0.38); (ii) the magnitude derivation in §4 chains three speculative multiplicands (0.50 flag precision × 0.60 incorporation × headroom) that compound estimation error, yielding a "minimum-defensible +3 F1pp" that has no published anchor; (iii) the Feedback Friction qualitative-only retreat in step (b2) does not survive the paper's own scope disclaimer (Feedback Friction §3.1 explicitly excludes subjective/structured-output tasks because "another LLM to evaluate more subjective tasks like instruction following or translation could lead to issues like reward hacking and unreliable assessments" — ablation generation is exactly such a task).

The mechanism on which the +3 F1pp prediction rests is at this point three layers removed from any direct citation: (1) AblationBench measured rubric-promotion lift on a different transition (Agent-Planner → LM-Planner, not LM-Planner → LM-Planner+wave-2), (2) Feedback Friction measured F3 > F1 in a domain whose authors warn against generalizing, (3) the file-handoff-vs-in-context arm itself tests an architectural distinction the smith cannot point to any prior work establishing as material. The hypothesis cannot recover from this within one more revision round.

The honest-kill option in the prompt is the right call.

---

## 2. Re-check of the four prior Critical defects

### C1 (mechanism collapse) — NOT fully addressed; the reframe is windowdressing

The smith reframes the mechanism from "promote rubric to first-class prompt" (which IS LM-Planner) to "cross-wave file-handoff of named missing-ablation IDs through MegaResearcher's stateless-leaf-dispatch architecture." On the surface this dodges C1. But the diagnostic arm the smith adds in §6 ("file-handoff vs in-context") is itself the proof that the reframe is hollow:

- **The arm tests whether file-handoff differs from in-context concatenation.** Concatenating the rubric-output into wave-1's eval-designer prompt IS the LM-Planner pattern with a slightly larger context window. AblationBench §4.2 already measured LM-Planner with the full paper-text + rubric-shaped prompt; there is no published reason — none surfaced by the smith and none I can find — to expect file-handoff to produce a different F1@5 than in-context concatenation when the receiving worker is also an LM call.
- **The smith's own R3 explicitly admits this risk:** "If the two variants produce statistically indistinguishable F1@5, the file-handoff substrate adds no value beyond the rubric content itself — meaning the C1 collapse re-materializes at the architectural level." This is the smith conceding that the reframe MAY be vacuous and offering "a measurement of the architectural-equivalence claim" as the consolation contribution. A consolation contribution is not a hypothesis worth Phase-5 design.
- **What's actually being tested:** the +3 F1pp prediction is for the two-wave configuration over the strongest LM-Planner baseline. If file-handoff is equivalent to in-context concatenation, then the two-wave configuration is just LM-Planner-plus-more-context, and the +3 F1pp prediction has no mechanism. If file-handoff DOES differ from in-context, the smith owes a citation establishing why — and there isn't one.

**Verdict on C1:** The reframe is plausible-sounding but mechanism-empty. The smith dressed the same prompt-engineering experiment in MegaResearcher-specific terminology ("stateless-leaf-dispatch", "cross-wave"), but the underlying intervention is putting a rubric-output into a downstream worker's input, which is the unmodified prior-art pattern. **Critical, unresolved.**

### C2 (metric substitution) — Addressed honestly but with a residual error

The smith locked the primary metric to F1@5 and reduced the predicted Δ from +10pp recall to +3 F1pp. That part is correct.

**Residual error:** §1 still claims "best-performing LM system reaches 38% on AuthorAblation." Table 4 of AblationBench (verified via `read_paper` §7) shows max F1@5 on AuthorAblation is **0.31 (GPT-4o LM-Planner)**, with Claude 3.5 Sonnet LM-Planner at 0.30. The 38% figure is Claude 3.5 Sonnet's **recall@5**, not F1@5. The smith repeats the F1-vs-recall conflation he claimed to have fixed, two paragraphs after acknowledging it. This is a careless repeat that suggests the C2 acknowledgment wasn't internalized.

**Verdict on C2:** Largely addressed but a residual statistic-quote error in §1 shows the same defect creeping back in. **Important, partially unresolved.**

### C3 (non-judge claim) — Addressed honestly; F2 elevation is correct but F2 may be entangled with F1

The smith concedes the non-judge framing was wrong and elevates the deterministic file-diff incorporation rate (F2) to co-primary. This is the cleanest of the four fixes.

**However**, F2 and F1 are not as independent as "both must pass" implies. The wave-2 eval-designer that produces the incorporated entries IS what is evaluated by the AblationBench LMJudge for F1@5. If wave-2 incorporates flag X as a named entry in the protocol, and the LMJudge then matches that entry against a GT ablation, F2 and F1 are measuring overlapping events. The F2 component "incorporated flag must hit a GT entry the wave-1 protocol missed" (F2c) explicitly couples F2 to GT-hit recovery — the same recall surface F1 depends on. The two metrics share a denominator (named entries in wave-2's protocol) and a numerator-component (GT-hit). "Both must pass" is therefore not the independent dual-criterion the smith claims; it is a correlated dual-criterion.

**Verdict on C3:** Honesty restored on framing, but the deterministic co-primary is partially entangled with the LM-judge-mediated primary. The smith's "both must pass" presentation overstates the independence. **Important.**

### C4 (AbGen omission) — Addressed correctly

I verified AbGen §3.2 directly. The human-evaluation Likert protocol exists exactly as the smith describes (1-5 across importance/faithfulness/soundness; inter-annotator Cohen's Kappa 0.71-0.78). The reference (original human-written) ablation studies score 4.77 average; best LLM (DeepSeek-R1-0528) scores 4.11. A +0.10 lift on this 1-5 Likert scale represents ~15% of the LLM-to-reference gap — operationally meaningful. AbGen's §3.3 GPT-4.1-mini automated judge is correctly flagged as unreliable by AbGen itself.

F5 ("AbGen non-transfer") is a real falsifier in the sense that the smith pre-registers a directional threshold (Δ ≥ +0.10 on human-eval Overall). But it's classified as supplementary in F5, which the smith explicitly says "is not counting toward 3-criterion floor." Demoting AbGen to supplementary defeats the purpose of having a robustness substrate: if the F1@5 lift on AblationBench fails to transfer to AbGen's human eval, that's exactly the substrate-portability falsification AbGen's existence enables. Calling it "not a F1-falsification but a substrate-portability limitation" is hedging.

**Verdict on C4:** Citation added correctly; the human-eval column is correctly identified as the right AbGen surface. But F5's demotion to supplementary undercuts its role. **Important.**

---

## 3. Citation spot-checks (new, on revision-1 surface)

### Spot-check #1 — AblationBench Table 4 numbers in revision §1

**Smith's claim (§1):** "AblationBench... best-performing LM system reaches 38% on AuthorAblation, below human-level."

**What Table 4 actually says (verified via `read_paper`):** Max LM-Planner F1@5 on AuthorAblation = 0.31 (GPT-4o). Max recall@5 = 0.38 (Claude 3.5 Sonnet). The 38% figure is recall, not F1.

**Assessment: ERROR repeats the C2 defect.** Two paragraphs after the smith acknowledges F1 vs recall@5 confusion as C2, he repeats it in §1. This is the kind of mistake that suggests the metric distinction isn't internalized.

### Spot-check #2 — AbGen §3.2 human-eval protocol

**Smith's claim:** "AbGen's human-evaluation protocol (Likert 1-5 across Importance / Faithfulness / Soundness, §3.2) on a small expert-annotated re-evaluation subset."

**What §3.2 says (verified):** Exactly this. Three criteria, Likert 1-5, with the protocol of first scoring without reference then adjusting after seeing reference. Inter-annotator agreement Cohen's Kappa 0.735/0.782/0.710 on a sample of 40 outputs.

**Assessment: ACCURATE.** The smith's AbGen characterization is faithful.

### Spot-check #3 — Feedback Friction §3.1 scope statement

**Smith's claim (step b2):** "Feedback Friction... demonstrates this qualitatively across multiple model families: Strong-Model Reflective Feedback (F3-shape) outperforms Binary Correctness Feedback (F1-shape) on the integration step, regardless of magnitude."

**What §3.1 says (verified via `read_paper`):** "We employ nine diverse tasks for evaluation, **deliberately choosing objective tasks with clear ground-truth answers** to ensure reliable evaluation of feedback incorporation. Using another LLM to evaluate more subjective tasks like instruction following or translation could lead to issues like reward hacking and unreliable assessments."

**Assessment: SCOPE-MISMATCH.** Feedback Friction's authors explicitly scoped their study to objective ground-truth tasks BECAUSE they consider LLM-evaluation on subjective tasks unreliable. Ablation generation — judged by an LMJudge with 0.74 F1 against humans — is precisely the class of task the Feedback Friction authors excluded. The smith's "qualitative direction" import survives in the sense that "named feedback might outperform binary feedback" is plausible folk-wisdom, but the smith is citing a paper whose authors explicitly disclaimed this generalization. The retreat from importing magnitudes does not address the underlying scope mismatch.

### Spot-check #4 — AblationBench §4.2/§7.4 confirming the LM-Planner identity

**Smith's claim (§3 step a):** "AblationBench §7.4 reports LM-Planner... outperforms Agent-Planner (SWE-agent multi-step) by 8 F1 percentage points on AuthorAblation with Claude 3.5 Sonnet (p=0.009, paired t-test)."

**What §7.4 says (verified):** Exactly this. "Using Claude 3.5 Sonnet, the best-performing agent model, LM-Planner attains an F1 score that is 8 percentage points higher, while being 2.35× cheaper; this difference is statistically significant (p=0.009, paired t-test on F1 scores)."

**Assessment: ACCURATE.** The headline citation that grounds the smith's concession is correctly characterized. This actually reinforces the kill: the +8 F1pp lift is the entire published-evidence anchor for rubric-promotion's effect, and the smith correctly notes S6 cannot claim it. What remains as the S6-specific lift (+3 F1pp from "file-handoff") has no comparable anchor.

---

## 4. Mechanism critique (revision-1 surface)

The revised mechanism has three steps. Each remains defective.

**Step (a) — Rubric-shaped single-call prompts are the strongest known scaffold.** The smith correctly demotes this to "published baseline, not a contribution." This is accurate but vacuous as a mechanism claim — it now does nothing more than identify the prior-art ceiling. No critique here other than to note that step (a) of the "mechanism" is now just citing the baseline.

**Step (b) — The file-handoff substrate carries named flags across waves.** This is the entire load of the contribution claim. Three sub-claims, each defective:

- **(b1) "No published AI-Scientist-family pipeline does this."** Survives the gap-finder re-verification — true. But "no one has tried this" is not a mechanism; it's an empty-cell claim. The hypothesis needs to also explain *why* file-handoff should produce a measurable Δ over what LM-Planner alone produces. (b1) doesn't.
- **(b2) "Structured named feedback is qualitatively easier to incorporate than free-form feedback."** Borrowed from Feedback Friction, which itself disclaims this scope (spot-check #3). The smith's retreat from importing magnitudes is correct, but the qualitative direction-claim is borrowed from a paper whose authors explicitly excluded the target domain because they considered LLM-evaluation on it unreliable. The mechanism-claim is structurally analogous to "stars guide ship navigation in clear weather; therefore stars likely also guide ship navigation in fog" — the source authority does not extend to the target domain.
- **(b3) "The file-handoff substrate is the specific MegaResearcher constraint being tested."** This is the most surface-level part of the reframe. MegaResearcher's leaf workers being stateless is a swarm-implementation detail; it constrains *how* you build a multi-wave system but says nothing about *whether* that multi-wave system should lift F1@5 over an LM-Planner baseline. The smith implicitly conflates "MegaResearcher has this constraint" with "this constraint is what produces the lift." The lift, if it exists, comes from the second wave being able to incorporate named flags — but in-context concatenation can do this too, which is exactly what the smith's own diagnostic arm acknowledges.

**Step (c) — No AI-Scientist-family baseline has the integration mechanism.** Survived O8 with a softened claim that the eval-designer should "treat this as a measurement open question, not a settled claim." Honest, but now step (c) is also no longer a load-bearing mechanism claim — it's an acknowledged measurement open question.

**Aggregate mechanism assessment:** After the revision, steps (a) and (c) are both demoted from mechanism claims to context claims. The entire mechanism is concentrated in step (b), and step (b) rests on a domain transfer (b2) the source paper disclaims, plus an architectural assertion (b3) that the smith's own diagnostic arm exists to test, with no prior establishing it should produce a measurable effect. **This is not a mechanism; it is a hopeful experimental setup.**

---

## 5. Falsifiability assessment

| Criterion | Falsifiable? | Operationalizable? | Notes |
|---|---|---|---|
| F1 (F1@5 ≥ max(B1,B2,B3) + 0.03) | Yes | Yes | Now correctly metric-matched. Threshold is small but pre-registered with a CI bound. |
| F2 (≥3 flags, ≥60% incorporation, ≥50% GT-hit precision) | Yes | Yes | The precision-of-flags floor addresses O6. But F2 is partially entangled with F1 via the shared GT-hit denominator (see C3 critique). The "all three must pass" composition is appropriately stringent — possibly *too* stringent: ≥50% of incorporated flags hitting GT entries the wave-1 protocol missed is a high bar given AblationBench's LMJudge precision is only 0.76. The F2 threshold has no published anchor; it could plausibly be set anywhere from 30% to 70%. |
| F3 (AuthorAblation vs ReviewerAblation rubric sweep) | Yes | Yes | Narrowed correctly to two arms. ReviewerAblation as control is a sensible choice. |
| F4 (cost-budget breach) | Yes | Yes | Simple cost accounting. |
| F5 (AbGen non-transfer) | Yes | Yes | But demoted to supplementary. If AbGen non-transfer doesn't count as a falsification (and the smith says it "is not a F1-falsification"), then F5 is decorative. |

**Falsifiability assessment summary:** The criteria are individually operationalizable. The bigger issue is that F1 is the only criterion whose threshold ties directly to the hypothesis's central claim (file-handoff produces a measurable lift). F2's three-component composite is appropriately tight but partially correlated with F1. F3 tests a rubric-side ablation. F5 is demoted. So the hypothesis effectively stands or falls on a +3 F1pp lift over the strongest of three LM-Planner-class baselines, judged by AblationBench's own LMJudge with documented 0.74 F1 against humans.

**Severity:** the +3 F1pp threshold, with the LMJudge's ~26% disagreement-with-humans noise floor, means that a +3pp signal is roughly within the LMJudge's measurement noise. This is a thresh-noise mismatch the eval-designer would need to address (e.g., via paired-bootstrap CI), but the underlying signal is already small relative to the measurement instrument's noise.

---

## 6. Strongest counter-argument (steelman the kill)

The strongest case for killing S6 rather than revising again is structural: **after honest concessions, S6 is asking the swarm to spend Phase-5 design effort and Phase-5 budget on measuring whether putting a rubric-output in a downstream worker's input file produces a different result than putting it in the downstream worker's context window.** That is not a research question that merits a main-track contribution, and even the smith asks the synthesist to position it as scaffolding rather than primary contribution.

The steelman of the "still worth running" position:

1. *Negative results have value.* If the file-handoff arm produces no Δ over the in-context arm, that IS a measurement of the architectural-equivalence claim, which is informative about MegaResearcher's design.
2. *The +3 F1pp threshold is pre-registered, not retrofitted.* Pre-registration discipline is a contribution regardless of outcome.
3. *AbGen-transfer is a useful side-finding.* Whether the lift transfers across benchmarks is a methodologically interesting question independent of the primary claim.

The case against, which I find decisive:

1. *Negative results have value only when the positive prediction was non-trivially likely.* The smith provides no prior establishing that file-handoff should differ from in-context concatenation. A negative result on a question with no published prior is just "we tried this thing and it didn't work" — not a contribution.
2. *Pre-registration is a methods discipline, not a contribution in its own right.* Pre-registering a +3 F1pp threshold for a measurement that is in the LMJudge's noise floor doesn't elevate the experiment.
3. *AbGen-transfer is interesting but small-bore.* It's a methodological side-finding; it's not S6's primary claim.

The kill rationale: a hypothesis that the smith asks the synthesist to demote to scaffolding, whose mechanism rests on a domain-transferred direction-claim from a paper that disclaims the target domain, whose architectural-equivalence diagnostic arm risks vacating the contribution entirely if it succeeds, and whose +3 F1pp threshold sits inside the LMJudge's noise floor — is not a hypothesis. It's a system-integration measurement framed as a hypothesis.

The synthesist should include S6 as a future-work flag: "MegaResearcher's file-handoff substrate has not been measured against in-context-pass alternatives on the ablation-coverage task; future work should design an experiment isolating the architectural contribution." That is the honest framing the smith himself gestures toward in §0's "Honest disclosure: smaller magnitude, narrower scope" paragraph.

---

## 7. New attack axes raised by the prompt

### 7a. Is +3 F1pp honest enough? Or does it self-defeat?

The smith openly admits in §0: "A +3 F1pp lift is **not a main-track-conference primary contribution by itself**." This is honesty in the right direction. But honest disclosure that the lift is not main-track is not the same as honest disclosure that the hypothesis is worth designing for. A small lift on a reasonable benchmark is still publishable as a system-integration paper. A small lift whose mechanism doesn't survive the diagnostic arm the smith himself added is not. The honest disclosure is necessary but not sufficient — the smith's self-disclosure should have triggered a self-kill, not a self-rescue via "the synthesist should position S6 as scaffolding."

### 7b. Should we just kill and flag for synthesist?

**Yes.** This is the right call.

The instructions explicitly authorize "A clean KILL with lesson is BETTER than a thin APPROVE that produces a Phase-5 design for a workshop-grade contribution." The smith's self-positioning as "necessary scaffolding rather than primary contribution" IS the cue for the kill option. Approving would commit the swarm to producing an eval-designer document that the smith himself characterizes as scaffolding-grade, which wastes the eval-designer's effort on what should be a future-work flag.

### 7c. Is the file-handoff-vs-in-context distinction sharp enough?

**No.** Two grounds:
1. *The interventions are functionally identical at the LLM-call level.* A wave-2 eval-designer reading `ablation-coverage.yaml` from a file is, at the level of what the LM sees, identical to a wave-2 eval-designer with the same YAML content concatenated into its prompt context. The only material difference is that file-handoff allows the worker to be invoked statelessly (no need to carry state from wave 1) — but this is a swarm-implementation property, not a model-behavior property.
2. *The smith's own R3 admits this risk.* The smith's R3 contribution-if-materialized is "a measurement of the architectural-equivalence claim" — i.e., the smith already concedes the experiment may show the distinction doesn't matter, and offers a consolation framing.

### 7d. F2 hardening — overly stringent?

The composite threshold (≥3 flags AND ≥60% incorporation AND ≥50% GT-hit precision) is plausibly hard. Without published prior on either incorporation rate or GT-hit precision in this substrate, the smith cannot defend why these three thresholds should jointly clear at the predicted lift level. The thresholds are reasonable-sounding but arbitrary. A more disciplined design would either (a) lower-bound at exploratory thresholds (e.g., 30% incorporation as a "any signal at all" floor) or (b) anchor the thresholds to LMJudge-precision (e.g., GT-hit precision must exceed LMJudge baseline precision = 0.76, which would make the threshold ~50% incorporation × 0.76 = ~38% GT-hit). The smith does neither and just sets the thresholds at round numbers. **Suggestion-tier, not critical.**

### 7e. PaperRecon parallel — missed by the smith

`arXiv:2604.01128` PaperRecon (15 upvotes, Agent4Science-UTokyo) exists as a parallel paper-evaluation benchmark with 51 AI-written papers, evaluating presentation and hallucination dimensions. The smith does not cite it. PaperRecon does not directly evaluate ablation coverage, but it DOES evaluate AI-written-paper quality at the paper level — exactly the surface S6 ultimately bears on (the question "do better ablations make the paper better?"). The smith's gap re-verification searches did not surface this. The omission is not as critical as the AbGen omission was (PaperRecon doesn't specifically measure ablation coverage), but it indicates the smith's search coverage is still incomplete after revision-1. **Important.**

---

## 8. Severity-tagged objections

**Critical (must fix to APPROVE, none of which can be addressed within revision-2 budget):**

1. **Mechanism reframe is window-dressing.** The file-handoff-substrate-vs-in-context-concatenation distinction has no published prior establishing it should produce a measurable Δ. The smith's own diagnostic arm exists because he cannot anchor the distinction to any cited result. (§3 step b3; §6 diagnostic arm; smith's own R3 admission.)

2. **+3 F1pp threshold has no published anchor.** The derivation in §4 chains three speculative multiplicands (0.30 baseline recall × 0.50 flag-precision × 0.60 incorporation), each unanchored. "+3 F1pp is the pre-registered minimum-defensible threshold" is asserted, not derived.

3. **Feedback Friction scope mismatch.** The §3.2 qualitative direction-claim is borrowed from a paper whose §3.1 explicitly disclaims generalization to subjective/LLM-judge-evaluated tasks. The smith's retreat from importing magnitudes did not address the underlying scope mismatch.

**Important (would fix in any other revision context but irrelevant for KILL):**

4. **F1@5 vs recall@5 conflation repeats in §1.** "38% on AuthorAblation" is recall@5, not F1@5. The smith repeats the same defect he claims to have fixed in C2.

5. **F2 and F1 are entangled.** Both depend on GT-hit recovery in wave-2's revised protocol. "Both must pass" overstates independence.

6. **F5 (AbGen non-transfer) demoted to supplementary.** AbGen-transfer is exactly the substrate-portability falsification AbGen's existence enables; making it non-counting toward the falsification floor is hedging.

7. **PaperRecon (`arXiv:2604.01128`) not cited.** Parallel paper-quality benchmark missed in gap re-verification.

**Suggestion (would address only in deep revision):**

8. F2 thresholds are arbitrary round numbers; could be anchored to LMJudge precision floor.

9. The "ID space" resolution in O9 (deterministic slugs from component-name + removal-or-modification-type) is reasonable for AblationBench-mapped papers but ambiguous for AbGen and out-of-substrate cases. The smith doesn't specify how slug collisions are resolved or how synonymous component-names are normalized.

---

## 9. Recommendation to hypothesis-smith

**Recommendation: stop revising. Accept the KILL.**

The smith's revision was technically careful in three of four C-objection responses, but the cumulative effect is a hypothesis that the smith himself describes as "necessary scaffolding rather than primary contribution" and asks the synthesist to demote. The honest disclosure should have triggered a self-kill in revision-1, not a continued rescue attempt. There is no path to revision-2 that recovers a mainline contribution from the file-handoff-vs-in-context architectural distinction without finding a published prior establishing the distinction matters — and the smith has not found one in two attempts.

**Lesson to flag to the synthesist:** The audit trail should record:

> S6 (ablation-coverage flag-handoff worker) was killed after revision-1 because the mechanism collapsed into a system-integration measurement of whether MegaResearcher's stateless-leaf-dispatch + file-handoff architecture is equivalent to single-wave in-context-concatenation on the ablation-coverage task. The underlying gap (no AI-Scientist-family pipeline integrates an external ablation-coverage rubric as a worker) is real and survives re-verification; AblationBench and AbGen provide the substrate. But the specific MegaResearcher-architectural contribution does not survive the diagnostic arm. Future work: an experiment isolating file-handoff from in-context-pass on a multi-wave research pipeline, with a published prior establishing why the distinction should produce a measurable Δ before designing the experiment. Substrate: AblationBench AuthorAblation + AbGen testmini human-eval. The +3 F1pp lift the smith pre-registered as minimum-defensible is within AblationBench's LMJudge ~26%-disagreement-with-humans noise floor and would require either a larger-sample design or a more stringent measurement instrument.

This lesson IS a contribution. It documents an architectural-evaluation question the swarm couldn't answer in this run and points future-work design at the missing piece.

---

## 10. Verdict line

VERDICT: KILL (irrecoverable)
