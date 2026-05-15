# Red-team critique of Hypothesis S4 — REVISION 1

**Worker:** red-team-S4 (revision-1 pass)
**Run:** 2026-05-12-0515-19bf96
**Critiquing:** `hypothesis-smith-S4/output.md` (revision 1)
**Revision round:** 1 (second red-team look, smith already revised once)

---

## 1. Verdict

**VERDICT: REJECT (revision-2)**

The smith addressed all three prior Critical defects honestly and well. C1 (EvoScientist) is now framed correctly. C2 (SPECS) is dropped, not patched-over. C3 (F1 contamination) is disclosed openly with a frozen-hash protocol. The revised hypothesis is much cleaner than the initial submission.

But independent verification of the revision's new surface area uncovers **one new Critical defect** and **two new Important defects** that the smith did not engage with — none of which is unrecoverable, but all of which the smith must address before a clean APPROVE.

The single new Critical: **The Last Human-Written Paper (arXiv:2604.24658) is dramatically misrepresented as a "position paper that names the concept; S4 is the measurement protocol."** Direct read of §2.2 and §5.2 shows that 2604.24658 ALSO specifies (a) a structured schema for dead-end nodes with the exact field shape `(hypothesis, failure mode, lesson)` S4 proposes, (b) the **ARA Seal Level 1** machine-verifiable credential which is precisely the deterministic-schema-conformance measurement protocol S4 claims as its primary contribution, and (c) an open-source Live Research Manager / Ara Compiler agent skill that populates the structured dead-end ledger automatically. The smith's revised gap-claim still does not survive against this paper at the depth I read.

This is recoverable in one more revision: the differentiator must be the **binary-deterministic-signal triggering** (which 2604.24658 does not specify — Ara's dead-end nodes are populated by the Live Research Manager via LLM event-routing on conversation traces, not by deterministic signals like `citation_resolves=false` or `ablation_present_in_table=false`). But the smith's current framing of 2604.24658 as a position paper is wrong, and the gap-claim conjunction "structured schema + measurement protocol + deterministic" needs to be sharpened to "binary-deterministic-signal-triggered structured dead-end ledger" — the *measurement protocol* leg of the conjunction is occupied by Ara Seal Level 1.

Plus two Important defects: the F1 sample-size-driven MDE turns out to be very large (~+30pp at the smith's stated N=20 paired manuscripts), and the F2 incorporation-rate measurement has a same-model contamination problem the smith did not raise.

---

## 2. Re-check of three prior Critical defects

### C1 (EvoScientist gap-claim) — ADDRESSED

I re-read EvoScientist §3.5 directly via `hf_papers read_paper arxiv_id=2603.08127 section=3.5`. The smith's revised quote is verbatim accurate:

> "When experiments are complete, the EMA compares the proposed method against baselines based on W and **uses an LLM-based analysis to judge whether the proposal fails** … we use the idea validation analysis to update the ideation memory."

The smith's revised gap framing — "EvoScientist has a persistent ideation memory keyed on an LLM-judge-mediated failure signal; S4's contribution is the binary-deterministic-signal scoping discipline + schema discipline" — is defensible. Signal-determinism + schema-enforced scoping is a **genuine** distinction, not window-dressing.

There is a small residual: EvoScientist §3.5 also has a **rule-based** failure trigger ("If the engineer cannot find any executable code within the pre-defined budget (rule-based), we treat the proposal as failed"). So EvoScientist partially has deterministic signals too — but only on the experiment-execution surface, not on the prose-decision surface S4 targets. The smith should briefly acknowledge this in §7 pre-emption. Minor, not Critical.

**Verdict on C1:** Addressed.

### C2 (SPECS citation) — ADDRESSED

I re-ran `hf_papers search "SPECS controlled flaw injection peer review benchmark" limit=10`. SPECS-Review-Benchmark does not appear. Top results are SWE-bench Verified, SEC-bench, AnistBench, etc. — none is SPECS. The smith correctly removed the citation without substituting a paper-thin replacement.

**Verdict on C2:** Addressed.

### C3 (F1 contamination) — PARTIALLY ADDRESSED, important residual

The smith's frozen-hash protocol works at the level of *making the F1 string-match procedure deterministic ex post the artifact*. SHA-256 + pre-registration commits the artifact and prevents post-hoc key tuning. The disclosure "the lesson keys were originally derived with LM-judge assistance from AblationBench" is genuinely transparent.

But the more serious concern about "frozen LM-judge output used as a deterministic key" remains. Reviewers at a main-track venue will read: "the key set against which we measure recovery is itself an LM-judge output, frozen at a single timestamp." The deterministic step adds reproducibility but does not add validity — a different LM-judge run on the same AblationBench labels would yield a different key set, and S4 has no defense against the claim that the experiment is measuring agreement-with-the-frozen-judge rather than agreement-with-ground-truth. The smith's acknowledgment is correct but the validity question is not solved by hashing.

**A stronger move the smith does not consider:** ground the lesson keys in the AblationBench *human-annotator* labels, not the LM-judge derivatives. AblationBench AuthorAblation has hand-labeled flawed-baseline manuscripts. The smith could pre-register keys derived from the human label column rather than from the LM-judge column. This would be a clean, defensible non-judge ground truth. If the AblationBench schema doesn't support that, the smith should say so explicitly and weaken F1 to "agreement with a frozen LM-judge baseline" rather than "lesson recovery."

**Verdict on C3:** Hash protocol is fine; framing of the resulting metric as "lesson recovery" still overstates what F1 measures. Important, not Critical (it's a metric-interpretation issue, not a protocol violation).

### Hash slot pre-registration check

The frozen-hash slot in §5 F1 currently reads `[HASH-TBD-PREREGISTERED-AT-EVAL-DESIGN-TIME]`. This is a paper-thin disclaimer until the eval-designer actually populates it. The smith correctly defers the hash to the eval-designer dispatch step, which is the right architectural decision — but in the current document, the slot is a literal placeholder. This is acceptable *if* the eval-designer is required to populate before any experiment runs, and the smith's §5 wording does require this. Note: the orchestrator must verify the hash is actually populated before Phase 5 fires. Not a defect; a process risk.

---

## 3. Attack on the NEW surface (post-revision)

### Critical defect — The Last Human-Written Paper (2604.24658) misrepresented

I read 2604.24658 §1, §2, §3, §4, §5 directly. The smith's framing — "names the failure mode (Storytelling Tax) as a research-artifact problem; S4 is the measurement protocol for the artifact" — understates this paper by an order of magnitude. Here is what 2604.24658 actually proposes:

1. **Ara protocol** (§2.2) — a complete file-system schema for research artifacts with four layers (Cognitive `/logic`, Physical `/src`, Exploration Graph `/trace`, Evidence `/evidence`). The Exploration Graph is `exploration_tree.yaml`, a "nested YAML tree with five typed node kinds — question, decision, experiment, **dead_end**, pivot — where nesting encodes parent→child edges and an `also_depends_on` field captures convergence points." The `dead_end` node payload is `(Hypothesis, failure mode, lesson)`. This is **almost identical** to S4's ledger schema `(hypothesis_id, verbal_reflection, lesson)` minus the binary-signal field. The `decision` event type has structured payload `(Choice, alternatives, evidence)`, parallel to what S3 (structured-decision substrate) targets.

2. **ARA Seal Level 1** (§5.2) — "verifies that the artifact is well-formed and internally consistent: the directory ontology exists, **all structured files conform to their schema** (each claim carries Statement, Status, Falsification criteria, and Proof; each heuristic carries Rationale, Sensitivity, and Bounds), and **all cross-layer references resolve** (experiment IDs in claims.md point to valid entries in experiments.md, code references trace to implementations in /src)." This is a **deterministic, machine-verifiable, schema-conformance credential** — the exact "measurement protocol for an audit-trail artifact" the smith claims is S4's primary contribution.

3. **ARA Seal Level 2** (§5.2) — Rigor Auditor evaluates "falsifiability quality (criteria are actionable, non-tautological, scope-matched, and independently testable), methodological rigor (baseline adequacy, ablation coverage, statistical reporting, and metric-claim alignment)." This explicitly checks several of S4's binary signal types (baseline adequacy, ablation coverage, falsification criteria).

4. **Live Research Manager** (§3) — open-source agent skill (github.com/Orchestra-Research/Agent-Native-Research-Artifact, 132 stars) that auto-populates the Ara structure including dead_end nodes, with a three-stage pipeline (Context Harvester → Event Router → Maturity Tracker). It writes typed events with provenance tags. So the artifact-population side is already implemented, not just specified.

5. **Ara Compiler** (§4) — accepts trajectory logs (RE-Bench traces) and seeds `/trace` with dead-end nodes the PDF omits. The smith's "no surveyed system produces this exact conjunction of properties" claim in §1 is hard to defend against this.

The smith's gap-claim conjunction is **"binary-deterministic signal + schema-enforced scoping + append-only ledger + measurement protocol for cross-wave lesson recovery."** Of these four:

- **Schema-enforced scoping:** Ara protocol has this (§2.2 — schema-enforced four-layer structure with typed dead_end nodes).
- **Append-only ledger:** Ara's Exploration Graph is append-only and version-controlled (§3.1 P3: "version-controlled, so each milestone produces a navigable snapshot and retroactive revisions are first-class operations rather than destructive overwrites").
- **Measurement protocol:** ARA Seal Level 1 (§5.2) is precisely this — deterministic, machine-verifiable schema-conformance credential.
- **Binary-deterministic signal:** This S4 has, 2604.24658 does not. Live Research Manager populates dead_end nodes by Event Router LLM-classification of conversation traces, not by deterministic binary signals like `citation_resolves` or `ablation_present_in_table`.

**The genuine S4-vs-Ara differentiator is leg 4 only:** binary-deterministic-signal triggering. The other three legs are already published in 2604.24658. The smith's revised §1 framing must be tightened to acknowledge this — the conjunction-of-four contribution does not hold, the contribution narrows to the deterministic-signal contribution alone.

**Severity:** Critical. The "this is just a measurement protocol for the artifact 2604.24658 already proposed" attack is the strongest single objection to S4 and the smith does not pre-empt it. The contribution is narrower than the smith claims.

This is recoverable in one more revision — the deterministic-signal-trigger leg is genuinely novel and defensible against 2604.24658, but the framing must reflect that this is the entire contribution, not a four-leg conjunction.

### Important defect — F1 MDE at N=20 is large

The smith's revised §4 says: "the falsification floor is set at the smallest difference detectable at 80% power given the 20-manuscript paired sample size, computed pre-experiment by the eval-designer." This is methodologically clean — but a quick MDE check shows the floor will be **uncomfortably large**.

For a McNemar paired test with N=20 paired binary observations, alpha=0.025 (Bonferroni-corrected from 0.05 across F1+F2), power=0.80:

- Required discordant pairs for detection at 80% power is ~10. With N=20 manuscripts and an expected discordant rate of ~50%, this means the minimum detectable difference in marginal proportions is approximately **+30 to +35 percentage points**.
- If the unit of pairing is "binary signal kill" rather than "manuscript" (i.e., ~5 kills × 20 manuscripts = 100 paired observations), MDE drops to roughly +12-15pp — much better, but still substantial.

The smith does not specify the unit of pairing. **This matters a lot.** "20-manuscript paired sample" is ambiguous between N=20 (manuscripts as units) and N=~100 (kills as units, manuscripts as clusters with potential intra-cluster correlation requiring a different test like cluster-bootstrap McNemar).

Two consequences:

1. **If unit = manuscript and N=20, MDE ≈ +30pp.** The smith dropped the +25pp prediction precisely because it was anchored to Reflexion's task-success-rate transfer. But the sample-size-driven floor lands close to where the dropped prediction was. The smith should explicitly compute MDE in §4 and disclose: "the floor lands at ~+30pp; if the empirical effect is below this, F1 fails to falsify in either direction (insufficient power)." That is a power-analysis honesty issue and the smith doesn't engage with it.

2. **If unit = kill and N=100, the McNemar test assumes independence which is violated by intra-manuscript clustering.** A cluster-bootstrap McNemar (Donner & Klar, or paired GEE) is needed. The smith does not specify this.

**Severity:** Important. The MDE turns out either embarrassingly large (in which case F1 is weak as a falsifier) or requires a more involved statistical procedure than McNemar (in which case the eval-designer has more work than the smith implies). The smith should pick one and disclose.

### Important defect — F2 incorporation rate same-model contamination

F2 measures whether the next-wave hypothesis-smith's output.md file-diff addresses each flagged lesson. The next-wave smith is the same model class (likely the same Anthropic model checkpoint) as the prior wave's smith and red-team. **The next-wave smith is also prompted with the ledger as part of its read-side input** — but it ALSO has access to its own prior outputs as worker artifacts in `docs/research/runs/<run-id>/hypothesis-smith-SN/output.md`.

Two confounds:

1. **A smith without the ledger but WITH access to its own prior output.md may still address the same lessons** — because the lessons are in some sense "what a competent smith would naturally avoid on the second pass." If the no-ledger baseline reads the prior output.md as part of its context (which the orchestrator typically allows for cross-wave continuity), F2's signal is the difference between "smith sees ledger + prior output" vs "smith sees only prior output." This difference may be very small even if the ledger has real epistemic content — the prior output already encodes most of what was wrong by what was kept out of it. The smith doesn't specify whether the no-ledger baseline has access to its own prior output.md.

2. **The next-wave smith may demonstrate ledger-incorporation as a prompt-following behavior rather than a genuine epistemic update.** When the system prompt says "read the rejected-hypotheses ledger before forging," the smith will mention items from it in its output by simple instruction-following. The file-diff will register a touch. But this is not the same as "the smith genuinely updated its hypothesis space." F2 cannot distinguish prompt-following from epistemic update without a deeper probe.

**Severity:** Important. The smith should either (a) restrict F2 to a treatment-vs-control where the baseline does NOT have access to its prior output.md, or (b) acknowledge F2 measures "prompt-following behavior in the presence of the ledger" rather than "genuine cross-wave epistemic update," and re-frame accordingly.

### Auto Research with Specialist Agents (2605.05724) §3.3 verification — ADDRESSED

I re-read 2605.05724 §3.3 directly. The smith's quotes and framing are accurate. §3.3 explicitly says: "The run log stores hypothesis text, diff summary, score, status, timing, and crash reason… keeps failed directions visible without replaying the full transcript." The smith's differentiator — "Auto Research's run-log is keyed on a numeric score, in a training-recipe domain, with no schema separating signal from reflection" — is verifiable and accurate.

§3.3 also says "This setup also makes the research process a releasable artifact" and notes the public archive at github.com/cxcscmu/Auto-Research-Recipes. So 2605.05724 is publishing the audit-trail concept too, but the smith's framing is correct.

### Storytelling Tax citation — VERIFIED

The smith's claim that 2604.24658 names the "Storytelling Tax" is verified by direct read of §1. The abstract literally uses "Storytelling Tax, where failed experiments, rejected hypotheses, and the branching exploration process are discarded." Citation is accurate as stated. But the smith dramatically understates what else 2604.24658 proposes (see Critical defect above).

### Cross-hypothesis structural analogy acknowledgment

The smith's §3 Leg 1 and §3 closing paragraph honestly acknowledge "Reflexion is multi-trial-same-task; S4 is cross-hypothesis." This is correctly disclosed. The smith does not claim Reflexion's mechanism transfers cleanly; it claims only the qualitative prior. This is honest.

But the consequence is that **S4's empirical predictive claim has very little prior-art grounding.** The smith's §8 risk 1 names this: "the cross-hypothesis structure is structurally different from Reflexion's multi-trial-same-task setup, so even modest signal mass may not move the next wave." The smith's residual contribution argument is "the ledger format itself is a novel discipline artifact." But as shown in the Critical defect above, the ledger format is largely 2604.24658's contribution. The robust contribution actually narrows to: **deterministic-binary-signal-triggered population of an exploration-graph-style ledger, on the prose-decision surface of an autonomous-paper-generation pipeline.** That is a defensible, specific, narrow contribution. But the smith's current framing of "first deterministically verifiable audit-trail artifact for AI-Scientist-family pipelines" is still oversold.

---

## 4. Gap re-verification — does the narrowed gap survive?

**Queries I ran:**

1. `hf_papers search "binary deterministic schema audit trail rejection ledger autonomous research agent" limit=10` — Top hits are governance/financial-domain papers (Springdrift, Valori, POLARIS) plus ARIS and one EvoScientist-class system. None implements binary-deterministic-signal-triggered rejected-hypothesis ledger for paper-generation pipelines. **The narrow deterministic-signal gap survives.**

2. `hf_papers search "append-only run log failure summary agentic scientific discovery cross-wave memory" limit=10` — Surfaces Mistake Notebook Learning (2512.11485), CORRECT (2509.24088), and 2604.24658 again. Mistake Notebook Learning uses an LM-judge proxy verifier (§3.1 "Self-Evolution regime"), so its signal is LM-mediated, not binary-deterministic. CORRECT generates schemas via LLM (§3.1), also LM-mediated. **Deterministic-signal gap survives against these too.**

3. `hf_papers search "structured schema failure memory LLM agent deterministic signal trigger" limit=10` — Surfaces MAS-FIRE (2602.19843), AgentDebug (2509.25370), MemMA (2603.18718). None addresses the paper-generation surface; none uses binary-deterministic file-artifact signals to trigger memory entries.

4. `hf_papers search "research process artifact dead end branching exploration trajectory autonomous" limit=8` — Surfaces 2604.24658 (Last Human-Written Paper), AgentRxiv (2503.18102), OpenResearcher (2603.20278), Scaling Laws in Scientific Discovery (2503.22444). **2604.24658 is the closest match and is much closer than the smith's revised §1 acknowledges.**

5. `hf_papers search "ARA Seal Level 1 structural integrity schema conformance research artifact" limit=5` — Returns 2604.24658 as the dominant match. ARA Seal Level 1 is the published deterministic-schema-conformance measurement protocol.

**Gap claim survives:** **YES, but narrower than the smith claims.** The deterministic-binary-signal-trigger leg of the conjunction is genuinely unoccupied. The schema-enforced-scoping + append-only-ledger + measurement-protocol legs are largely covered by 2604.24658 (Ara protocol + ARA Seal Level 1). The smith's revised §1 still oversells.

---

## 5. Citation spot-checks

| Cited | arXiv ID | Resolves? | Matches the revised claim? |
|---|---|---|---|
| Reflexion | 2303.11366 | yes | yes — qualitative-only framing is honest |
| Huang 2310.01798 | 2310.01798 | yes | yes |
| EvoScientist | 2603.08127 | yes | yes — quote from §3.5 is verbatim accurate; framing is now correct |
| AblationBench | 2507.08038 | yes | yes — keywords confirmed include "LM-based judges"; disclosure honest |
| AI Scientist v2 | 2504.08066 | yes | yes — no-ledger baseline accurate |
| Feedback-Friction | 2506.11930 | yes | yes |
| Auto Research with Specialist Agents | 2605.05724 | yes | yes — §3.3 quote is verbatim accurate |
| **The Last Human-Written Paper** | 2604.24658 | yes | **NO — claim is "names the Storytelling Tax failure mode S4 measures." This is correct as far as it goes but dramatically understates §2.2 Ara protocol and §5.2 ARA Seal Level 1, which together overlap heavily with S4's claimed measurement contribution.** |
| SPECS-Review-Benchmark | (removed) | n/a | correctly removed |

**Critical citation finding:** 2604.24658 is misrepresented as a position paper. Direct read shows it proposes a structured schema (Ara), a deterministic verification protocol (ARA Seal Level 1), and an implementation (Live Research Manager + Ara Compiler, with public GitHub). The smith's framing must be revised.

---

## 6. Mechanism critique

**Leg 1 (Reflexion qualitative prior) — Honest.** The smith correctly demotes Reflexion from magnitude prior to qualitative prior. The structural-analogy caveat is surfaced.

**Leg 2 (Huang ceiling) — Correctly grounded.** Verified.

**Leg 3 (Schema firewall) — Partially honest, partially weak.** The write-side firewall is real (the binary-signal-enum constraint genuinely prevents prose-only triggers). The read-side residual is now acknowledged. But there is a deeper issue: the schema constraint only fires *if a binary signal is available*. If the red-team kills a hypothesis for a soft reason (e.g., "the mechanism feels unmotivated") that doesn't map to one of the five enum signals, the schema-enforced approach **silently does not record the rejection** — the entry is rejected by the writer's validator. This is correctly stated by the smith in §2 ("entries with a value outside the enum are rejected by the writer's schema validator and not appended"). But this means the ledger systematically under-records the rejections that are most epistemically interesting: the prose-based judgments that don't fit a structured trigger. The smith's framing of this as a "firewall" obscures the cost — it's also a coverage loss.

A weaker firewall (e.g., "binary signal required, but soft kill can be flagged with `binary_signal_that_killed_it: soft_kill_other` as a special enum value with structured-context rationale") might capture more of the epistemically useful rejections without losing the deterministic-signal property. The smith should consider this as a design choice and either justify the strict enum or relax it. Important, not Critical.

---

## 7. Falsifiability assessment

**F1 — Statistical significance + frozen-hash MDE.** Falsifiable. Concern is the MDE magnitude as discussed in §3 (likely +30pp at N=20 manuscripts, weakening F1 as a discriminator). Important caveat needed.

**F2 — 30% incorporation floor via file-diff.** Falsifiable. Concern is the same-model contamination as discussed in §3. Important caveat needed.

**F3 — Three-outcome asymmetry.** Falsifiable. The smith's revision genuinely surfaces the asymmetry (outcome 3B/3C falsifies novelty contribution while 3A confirms). This is a substantial improvement over the initial submission. Approved.

**Cost disclosure:** F3 cost ~$20 vs $10 single-swarm baseline is now stated. Approved.

---

## 8. Strongest counter-argument (steelman, post-revision)

The steelman against S4-revision-1 is now: **"S4 is essentially a binary-signal-trigger restriction layered on top of the Ara protocol's exploration_tree.yaml dead_end node. The novel contribution is the trigger restriction, not the ledger format. As a contribution at main-track-conference scale, the trigger-restriction is a workshop-grade idea — useful, defensible, but narrow."**

This is harder to refute than the prior steelman ("Reflexion mechanism transfer doesn't hold on cross-hypothesis structure"), because it does not require any empirical claim — it's a positioning critique. The smith's revised version actually opens up to this attack more than the original, because the original (incorrectly) claimed novelty across four legs while the revision honestly narrows the contribution. The narrow contribution is more defensible epistemically but less impressive as a main-track contribution.

The smith should explicitly own this and decide whether the narrow contribution clears the spec's "main-track-conference bar." My reading: this is borderline. A workshop on autonomous research agents would publish it; a main-track ICLR/NeurIPS/ACL submission would face pushback on "this is too narrow a delta over Ara/2604.24658." The spec frames the bar as "main-track conference (ICLR/NeurIPS/ACL)" — and on that bar, this hypothesis is in the *workshop* tier of the spec's main track, not the *novel-contribution* tier. The smith should either (a) sharpen the deterministic-signal contribution to be substantively novel (e.g., demonstrating a class of signal types Ara cannot capture by design), or (b) accept the workshop-grade framing and let the synthesist handle whether to retain at this bar.

---

## 9. Severity-tagged objections

### Critical (must fix before APPROVE)

**C1. The Last Human-Written Paper (2604.24658) is dramatically understated.** §1, §6, §7 must be rewritten to acknowledge that 2604.24658 specifies (a) the Ara protocol structured schema with typed dead_end nodes carrying (hypothesis, failure mode, lesson) — almost identical to S4's ledger schema; (b) ARA Seal Level 1, a deterministic machine-verifiable schema-conformance credential — which IS the measurement-protocol-for-an-audit-trail-artifact that the smith claims as S4's primary contribution; (c) an open-source Live Research Manager + Ara Compiler that auto-populates the dead_end ledger from conversation traces. The genuine S4 differentiator narrows to **binary-deterministic-signal triggering** alone. The "conjunction of four properties" framing does not survive.

### Important (should fix before APPROVE)

**I1. F1 MDE at N=20 manuscripts is large and the smith does not compute it.** With McNemar paired binary, alpha=0.025, power=0.80, the minimum detectable effect at N=20 paired manuscripts is approximately +30pp. The smith should either (a) compute and disclose the MDE in §4, accepting that F1 is a weak falsifier, or (b) specify a more granular unit of pairing (e.g., individual binary-signal kills as the unit, with intra-manuscript clustering handled via cluster-bootstrap) and re-state the MDE. Currently the unit of pairing is ambiguous.

**I2. F2 same-model contamination not addressed.** The next-wave hypothesis-smith is the same model class as the prior-wave writer. F2 cannot distinguish (a) genuine ledger-driven epistemic update from (b) the smith's natural tendency to address weaknesses it can detect in its own prior output without the ledger, or (c) instruction-following behavior triggered by the system prompt mentioning the ledger. The smith should either restrict F2's no-ledger baseline to not have access to prior output.md, or re-frame F2 as measuring "prompt-following behavior in the presence of the ledger" rather than "epistemic update."

**I3. C3 disclosure is honest but the "lesson recovery" metric name is misleading.** F1 measures agreement-with-frozen-LM-judge, not lesson recovery. Either ground keys in AblationBench's human-annotator labels (if available) or rename M1 to "frozen-baseline agreement rate."

**I4. Schema-firewall trade-off (write-side) under-disclosed.** The strict enum constraint means soft-kill rejections (the epistemically richest ones) are silently dropped from the ledger. The smith should either justify the strict enum or relax it with a special enum slot for soft kills.

### Suggestion (nice to have)

**S1. EvoScientist also has a rule-based deterministic signal (no-executable-code-in-budget).** §7 pre-emption should briefly acknowledge this to be fully honest — EvoScientist's deterministic-signal coverage is narrower than S4's (only on experiment execution, not on prose decisions), but it is not zero.

**S2. CORRECT (2509.24088) and Mistake Notebook Learning (2512.11485) deserve one-line mentions** as further adjacent prior art on structured-failure-memory for LLM agents — they use LM-mediated signals so S4's deterministic-signal differentiator holds, but the smith should engage with them rather than ignore.

**S3. Consider whether the contribution is genuinely main-track-grade.** Per the steelman, this is borderline workshop-vs-main-track. The smith should either sharpen or accept.

---

## 10. Recommendation to hypothesis-smith

The hypothesis is much improved from revision 1. The three prior Critical defects are addressed cleanly. But the verification of 2604.24658 surfaces a new Critical defect: that paper proposes most of what the smith claims as the four-leg novelty conjunction, leaving only the deterministic-signal-trigger leg as genuinely novel.

**Concrete revision guidance (revision-2):**

1. **Rewrite §1, §6, §7 to acknowledge 2604.24658 properly (C1).** Specifically:
   - §1 gap claim: drop the "no surveyed system produces this exact conjunction of properties" framing. Replace with: "2604.24658 publishes the Ara protocol and ARA Seal Level 1 as the structured-schema + deterministic-verification side of this space. S4's narrow contribution is the binary-deterministic-signal triggering — the part 2604.24658 does not address, which it leaves to LLM event-routing in the Live Research Manager."
   - §6 baselines: add 2604.24658 as a baseline (not just a related work). It is the closest published peer.
   - §7 pre-emption: add an attack "This is just Ara/ARA-Seal with a trigger restriction" and pre-empt it directly.

2. **Compute and disclose F1 MDE (I1).** Pick a unit of pairing, run the power calculation, state the result. If the MDE is +30pp at N=20, say so and either accept the weakness or scale up the sample.

3. **Address F2 same-model contamination (I2).** Specify the no-ledger baseline's read context (prior output.md visible or not); acknowledge the prompt-following confound or design a control for it.

4. **Rename M1 or ground keys in human labels (I3).** "Lesson recovery against frozen LM-judge baseline" is the honest metric name; either pick that or find the human-label alternative.

5. **Acknowledge the schema-firewall trade-off (I4).** Justify the strict enum, or relax it with a soft-kill slot.

6. **Optional: cite CORRECT and Mistake Notebook Learning (S2).**

7. **Optional but important: decide whether the contribution clears the main-track bar (S3).** If the smith judges yes after the narrowing, defend it explicitly. If not, frame as workshop-grade and let the synthesist decide whether to retain.

The hypothesis is **REJECT (revision-2)** — not KILL. Recoverable in one more revision. Cap is 3 revisions; this is round 2 (out of 3).

---

VERDICT: REJECT (revision-2)
