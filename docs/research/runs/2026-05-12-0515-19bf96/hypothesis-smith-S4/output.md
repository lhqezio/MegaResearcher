# Hypothesis S4 — KILL recommendation (after revision-2 deep-read of arXiv:2604.24658)

**Worker:** hypothesis-smith-S4 (revision 2, FINAL)
**Run:** 2026-05-12-0515-19bf96
**Status:** revised after red-team REJECT (round 2)
**Recommendation:** **KILL** — contribute the lesson to the audit trail rather than enter revision-3 with a workshop-grade contribution against a main-track-conference spec bar.

---

## §0 Top-line decision

After the red-team's revision-2 deep read of arXiv:2604.24658 (The Last Human-Written Paper — Liu, Pei, Huang et al.), and after my own independent re-read of §2.2 (Ara Architecture) and §5.2 (ARA Seal) via `hf_papers read_paper`, I am recommending **KILL** rather than another narrowing revision. The rationale is the spec bar, not the soundness of the narrowed hypothesis.

**The narrowed S4 is still defensible** — deterministic-file-signal-triggered dead_end-node entry is a genuinely novel architectural axis that arXiv:2604.24658 does not occupy. Ara's Live Research Manager populates dead_end nodes via LLM event-routing on conversation traces (§3); ARA Seal Level 2's Rigor Auditor (§5.2) uses an LLM-rubric-graded judge to check ablation coverage, baseline adequacy, and falsifiability quality. S4's deterministic file-signal trigger (`citation_resolves`, `ablation_present_in_table`, etc.) is the part Ara does not address.

**But the contraction is severe.** The honest framing post-deep-read is:

> S4 = Ara protocol's typed dead_end node + ARA Seal Level 1's schema-conformance + deterministic-file-signal triggers replacing Ara's LLM-event-routing on a single axis.

This is a workshop-grade architectural delta over a published 132-star-GitHub-implementation prior-art baseline. The spec's bar is **main-track ICLR/NeurIPS/ACL** (§Success criteria of `2026-05-12-megaresearcher-paper-pipeline-spec.md`). Workshop-grade does not clear that bar.

The red-team's revision-2 steelman §8 (page 200-207) made this explicit:

> "S4 is essentially a binary-signal-trigger restriction layered on top of the Ara protocol's exploration_tree.yaml dead_end node. The novel contribution is the trigger restriction, not the ledger format. As a contribution at main-track-conference scale, the trigger-restriction is a workshop-grade idea — useful, defensible, but narrow."

I cannot defensibly preserve main-track-conference magnitude after honest narrowing, and I am at the revision cap. Submitting a third revision that re-claims main-track novelty for the trigger-restriction alone would not survive red-team round 3 and would force user escalation under the discipline rule. A clean honest KILL contributes the lesson directly to the audit trail and saves the user the adjudication call — this is the same path S6 took.

The remainder of this document fulfills the required `output.md` structure for a KILL recommendation, including the lesson contribution to the audit trail per discipline rule #1 ("no silent rejections").

---

## §1 Response to red-team revision-2 objections

Mapping each objection to the disposition in this final revision.

### Critical objection — The Last Human-Written Paper (arXiv:2604.24658) dramatically understated

**Red-team finding.** Direct read of §2.2 and §5.2 shows arXiv:2604.24658 specifies: (a) Ara protocol's typed dead_end nodes with `(hypothesis, failure mode, lesson)` payload — almost identical to S4's ledger schema; (b) ARA Seal Level 1, a deterministic machine-verifiable schema-conformance credential — which is precisely the measurement-protocol-for-an-audit-trail-artifact S4 claimed as primary contribution; (c) open-source Live Research Manager + Ara Compiler that auto-populates dead_end nodes.

**Independent verification.** I re-read §2.2 and §5.2 directly via `hf_papers read_paper arxiv_id=2604.24658`. The red-team's reading is accurate and the overlap is substantial. Specifically:

From §2.2 (verbatim): "exploration_tree.yaml stores the complete research directed acyclic graph (DAG) as a nested YAML tree with five typed node kinds — question, decision, experiment, **dead_end**, pivot... dead-end nodes preserve **the hypothesis, failure mode, and lesson** that narrative papers discard."

From §5.2 (verbatim): "Level 1 – Structural Integrity verifies that the artifact is well-formed and internally consistent: the directory ontology exists, **all structured files conform to their schema** (each claim carries Statement, Status, Falsification criteria, and Proof; each heuristic carries Rationale, Sensitivity, and Bounds), and all cross-layer references resolve."

This is exactly the deterministic schema-conformance + cross-reference-resolution measurement protocol S4 proposed.

From §5.2 (Level 2, verbatim): "...falsifiability quality, checking that criteria are actionable, non-tautological, scope-matched, and independently testable... methodological rigor, covering **baseline adequacy, ablation coverage, statistical reporting, and metric–claim alignment**."

ARA Seal Level 2 covers the same content domain as four of S4's five proposed binary signals (`citation_resolves`, `ablation_present_in_table`, `baseline_in_comparison_table`, `falsification_criterion_count_gte_3`). The only S4 signal Level 2 does not specifically check is `magnitude_claim_has_citation`. The difference is that Level 2 uses an LLM-rubric-graded Rigor Auditor; S4 uses deterministic file-grep + table-parse + API-call checks.

**Disposition.** S4's contribution leg-count, after honest accounting against arXiv:2604.24658:

| S4 claimed contribution leg | Status after 2604.24658 deep-read |
|---|---|
| Schema-enforced scoping (typed dead-end nodes) | **Subsumed** by Ara protocol §2.2 |
| Append-only ledger | **Subsumed** by Ara's version-controlled exploration_tree.yaml §3.1 P3 |
| Measurement protocol for audit-trail artifact | **Subsumed** by ARA Seal Level 1 §5.2 (deterministic schema-conformance credential) |
| Binary-deterministic-signal triggering | **Genuinely novel** — Ara uses LLM event-routing (Live Research Manager §3) for dead_end population and LLM-rubric grading (Rigor Auditor §5.2 Level 2) for quality assessment. Deterministic-file-signal triggers are not in Ara. |

One out of four legs survives. The contribution narrows to: "deterministic-file-signal-triggered population of an Ara-style dead_end node, on the prose-decision surface of an AI-Scientist-family multi-wave pipeline."

This is genuinely novel but workshop-grade against the spec's main-track bar. KILL recommended.

### Important objection I1 — F1 MDE at N=20 is large and not computed

**Red-team finding.** McNemar paired test at N=20 paired manuscripts, α=0.025 (Bonferroni from 0.05 two-tail across F1+F2), power=0.80 — minimum detectable effect ≈ +30pp. If unit of pairing is "individual binary-signal kill" instead, intra-manuscript clustering invalidates McNemar's independence assumption and a cluster-bootstrap McNemar (Donner & Klar) is needed.

**Independent verification.** I confirm the red-team's MDE estimate is correct. For a paired binary McNemar test with N=20, the discordant-pair sample is the only signal source. At expected discordance rate ~40-50%, the MDE on marginal proportion difference at 80% power one-sided α=0.025 is roughly +25 to +35pp depending on assumed baseline rate. This is uncomfortably large given that the +25pp magnitude was dropped specifically because Reflexion-magnitude-transfer was indefensible — the sample-size-driven floor lands right back at the magnitude the smith was forced to abandon.

**Disposition.** Two paths existed for this defect:

1. **Increase N** beyond 20 manuscripts. Spec's budget allowance is $10/replication for the structured-only condition. Scaling to N=50 (would deliver MDE ~ +15pp) costs ~$25 marginal; eval-designer would need to flag whether this fits in the cumulative budget for the run.
2. **Change the unit of pairing** to individual binary-signal kills (~5 kills × 20 manuscripts = 100 paired observations), accept intra-manuscript correlation, and switch to cluster-bootstrap McNemar. MDE drops to ~+12-15pp, but the statistical procedure becomes more involved.

Both paths are technically available. **Neither path rescues the bigger problem** (the Critical Ara-overlap objection). Even with a properly-powered F1, the contribution remains workshop-grade. The MDE defect is not the binding constraint on the KILL decision — the Ara overlap is. Disposition: acknowledged, not fixed.

### Important objection I2 — F2 incorporation rate same-model contamination

**Red-team finding.** F2 cannot distinguish (a) ledger-driven epistemic update from (b) the smith's natural tendency to address weaknesses visible in its own prior `output.md`, or (c) prompt-following behavior. The no-ledger baseline's read-context is not specified.

**Independent verification.** Correct. The proposed fix (orchestrator's strategic guidance) is to add a control arm where the smith reads its own prior `output.md` but NOT the ledger. If incorporation rate is similar across arms, the ledger contributes nothing beyond rereading-own-output. This is a clean control and pre-registerable.

**Disposition.** This defect is **fixable in principle** — add the control arm to the eval design and pre-register the comparison. The fix would not, however, rescue the Ara overlap. Acknowledged, not fixed in this document because the KILL decision moots it.

### Important objection I3 — "Lesson recovery" metric name misleading

**Red-team finding.** F1 measures agreement-with-frozen-LM-judge, not lesson recovery. Should be renamed or grounded in AblationBench's human-annotator labels.

**Disposition.** Acknowledged. Would be fixed in revision-3 if proceeding, by renaming M1 to "frozen-baseline agreement rate" and checking whether AblationBench AuthorAblation has a human-label column. Not fixed here.

### Important objection I4 — Schema-firewall write-side coverage loss

**Red-team finding.** The strict enum constraint silently drops soft-kill rejections from the ledger, systematically under-recording the epistemically interesting prose-based rejections. Should either justify the strict enum or relax it with a `soft_kill_other` slot.

**Disposition.** Acknowledged. Would be fixed in revision-3 by adding a `soft_kill_other` enum slot with structured rationale field. Not fixed here.

### Suggestions S1, S2, S3 — acknowledged

S1 (EvoScientist rule-based signal mention), S2 (cite CORRECT and Mistake Notebook Learning), S3 (decide on workshop-vs-main-track) — all acknowledged. S3 is the decision being made in this document: workshop-grade after the contraction, KILL.

---

## §2 Targeted gap (restated, KILL framing)

The gap S4 targeted was GF-1-Rank-1 from `gap-finder-1/output.md`: "Audit trail of rejected hypotheses as architectural discipline — no measurable artifact verifying that discipline holds across waves in AI-Scientist-family pipelines."

After the revision-2 deep read of arXiv:2604.24658, this gap is **partially-but-substantially closed by the Ara protocol + ARA Seal Level 1**. The Ara protocol (§2.2) defines typed dead_end nodes with the (hypothesis, failure mode, lesson) payload, the exploration_tree.yaml format is append-only and version-controlled, and ARA Seal Level 1 (§5.2) is the deterministic, machine-verifiable schema-conformance credential.

**What remains genuinely unoccupied:** deterministic-file-signal triggering of dead_end node creation (vs Ara's LLM event-routing approach). This is a narrow architectural delta on the trigger axis.

**Gap-finder-1's framing was incomplete.** GF-1-Rank-1 surveyed AI-Scientist v2, EvoScientist, Auto Research, AgentRxiv, and others, but did not engage with arXiv:2604.24658 at the §2.2 / §5.2 depth. The gap is more contested than the gap-finder assessed. This is the meta-lesson S4 contributes to the audit trail.

---

## §3 What the audit trail receives from S4 (the KILL lesson)

Per MegaResearcher discipline rule #1, every killed hypothesis must contribute its lesson to the synthesist's final document. S4 contributes the following:

**Lesson S4-L1.** The audit-trail-as-architectural-discipline space is more contested than gap-finder-1's GF-1-Rank-1 assessed. arXiv:2604.24658's Ara protocol + ARA Seal Level 1 publish the structured schema + deterministic measurement protocol legs of the conjunction. Future work in this space should:

1. Cite arXiv:2604.24658 as the primary baseline, not as related work.
2. Target the deterministic-signal-trigger axis specifically (the leg Ara does not occupy via LLM event-routing).
3. Compare against Ara's open-source Live Research Manager directly (github.com/Orchestra-Research/Agent-Native-Research-Artifact, 132 stars as of red-team's check) on a fair head-to-head: same input traces, same dead_end node schema, deterministic-file-signal triggers vs LLM event-routing — measure trigger-precision and trigger-recall, not lesson-recovery.

**Lesson S4-L2.** Gap-finder dispatch on this run should have surfaced arXiv:2604.24658 at §2.2 / §5.2 depth. Two red-team rounds were burned because the gap-finder's prior-art coverage was sub-depth on the closest peer. Future runs should require gap-finders to do `read_paper section=...` on the top-3 candidate-peer papers, not just title-and-abstract scans, when the spec novelty bar is main-track-conference.

**Lesson S4-L3.** Reflexion-magnitude-transfer is structurally weak as a quantitative prior for cross-hypothesis (not multi-trial-same-task) settings. The smith should not anchor magnitude predictions on Reflexion's task-success-rate deltas when the proposed mechanism operates on a different metric over a different object. This lesson generalizes beyond S4 — any cross-hypothesis-memory hypothesis in this swarm should derive predictions from same-class systems (EvoScientist's documented effect sizes on ideation-memory, if any are reported) rather than from Reflexion.

**Lesson S4-L4.** For audit-trail measurement claims, the lesson keys' upstream LM-judge origin is not erased by SHA-256 hashing. The frozen-hash protocol provides reproducibility, not validity. Future audit-trail measurement work should ground keys in human-annotator labels when available, and explicitly weaken claims to "agreement with a frozen judge baseline" when not.

**Lesson S4-L5.** Sample size + statistical-test choice must be pre-flighted before the magnitude is dropped. The S4 revision-1 dropped its +25pp magnitude prediction in response to red-team I1, then deferred the magnitude floor to a power-driven MDE — but did not compute the MDE itself. The result was that the floor effectively returned to ~+30pp at N=20. Future hypotheses dropping a magnitude prediction must compute the resulting power-driven floor in the same revision pass, not defer it.

---

## §4 What S4 deliberately does NOT submit

The S4 contribution does not survive to a publishable hypothesis at the spec's main-track bar. The following are explicitly *not* claimed:

- **Not claimed:** "First deterministically verifiable audit-trail artifact for AI-Scientist-family pipelines." (Ara Seal Level 1 publishes this for general research artifacts, including agent-generated; the AI-Scientist-family restriction is too narrow a delta to carry the claim.)
- **Not claimed:** "A four-leg conjunction of properties not occupied by any surveyed system." (Three of the four legs are occupied by arXiv:2604.24658.)
- **Not claimed:** Any predicted magnitude on M1 (lesson-recovery rate) or M2 (incorporation rate). At N=20, F1 is underpowered; F2 has unaddressed same-model contamination.

---

## §5 If a future run wants to recover S4

The trigger-axis narrow contribution would clear a workshop bar (e.g., NeurIPS-workshop autonomous-research, ICLR-workshop self-improving systems). If a future run targets a workshop bar instead of main-track, the recovery path is:

1. **Reposition explicitly against Ara.** Title: "Deterministic-Signal-Triggered dead_end Population for Ara-Protocol Pipelines." Acknowledge Ara as baseline, not related work.
2. **Drop the schema-enforced-scoping claim, the append-only-ledger claim, and the measurement-protocol claim.** Retain only the deterministic-signal-trigger contribution.
3. **Re-design the eval as a head-to-head trigger-precision/recall comparison** against Ara's Live Research Manager LLM event-routing. Same input traces, same target dead_end nodes. Metrics: precision of correct dead_end-creation triggering, recall of legitimate dead_ends missed.
4. **Decouple from the AblationBench lesson-recovery framing.** Lesson-recovery was the wrong metric — what S4 actually measures is trigger-fidelity, not lesson-content-quality.
5. **Scope to N≥50** for adequate power at the +12-15pp MDE expected for this comparison; budget impact ~+$15 over the $10 baseline.
6. **Pre-register the F2 control arm** (smith reads prior output.md but not ledger) to address same-model contamination.

These are the conditions under which the trigger-axis contribution could be defended at workshop scale. It does not clear the spec's main-track bar. Hence KILL.

---

## §6 Falsification criteria for the KILL recommendation itself

To stay honest, the KILL recommendation has falsification criteria of its own:

**FK1.** If the synthesist judges that arXiv:2604.24658's overlap is overstated in this document (e.g., the synthesist's read of §2.2 and §5.2 finds that Ara's dead_end nodes are conceptually distinct from S4's ledger in a way I missed), the KILL should be reconsidered and S4 returned to revision-3.

**FK2.** If the synthesist judges that the spec's main-track bar admits trigger-axis-only contributions (because the spec is research-direction-document rather than a paper itself, and the swarm output is a punch list of testable hypotheses, not a single submission), the KILL should be reconsidered.

**FK3.** If a fourth swarm worker or human reviewer identifies a substantive prior-art gap in my deep-read of arXiv:2604.24658 (e.g., Ara's LLM event-routing approach has known failure modes that deterministic-signal-trigger would specifically address, with measurable lift), the KILL should be reconsidered.

Default disposition absent any of FK1-FK3: synthesist embeds Lessons S4-L1 through S4-L5 in the audit-trail section of the final output and S4 stops here.

---

## §7 Sources

All citations independently verified via `hf_papers paper_details` and (for the load-bearing arXiv:2604.24658 §2.2 / §5.2 claims) `hf_papers read_paper`.

- **arXiv:2604.24658** — Liu, Pei, Huang et al. *The Last Human-Written Paper: Agent-Native Research Artifacts.* §2.2 (Ara Architecture) and §5.2 (ARA Seal three-level credential) directly read and quoted. This paper is the primary reason S4 narrows to workshop-grade and is recommended KILL.
- **arXiv:2603.08127** — Lyu, Zhang, Yi et al. *EvoScientist.* §3.5 (Idea Validation Evolution, LM-judge-mediated ideation memory) retained as adjacent prior art.
- **arXiv:2605.05724** — Ning, Li, Zeng, Kang, Xiong. *Auto Research with Specialist Agents.* §3.3 (structured run-log) retained as adjacent prior art (training-recipe domain, not paper-generation).
- **arXiv:2504.08066** — Yamada, Lange, Lu et al. *The AI Scientist-v2.* No-ledger baseline.
- **arXiv:2507.08038** — Abramovich & Chechik. *AblationBench.* Would have been the lesson-key source. Disposition: not used.
- **arXiv:2303.11366** — Shinn et al. *Reflexion.* Qualitative mechanism prior. Lesson S4-L3 contributes the meta-finding that Reflexion's magnitude transfers do not hold in cross-hypothesis settings.
- **arXiv:2310.01798** — Huang et al. *LLMs Cannot Self-Correct Reasoning Yet.* Intrinsic-reflection ceiling.
- **arXiv:2506.11930** — Jiang et al. *Feedback Friction.* Would have been the F2 floor reasoning. Disposition: not used.

**Citation removed in revision-1 and confirmed-removed in revision-2:** SPECS-Review-Benchmark — arXiv:2604.13940 resolves to AAAI-26 AI Review Pilot, not SPECS. Independent searches in revisions 1 and 2 confirm no resolution.

---

## §8 Recommendation to orchestrator

**KILL S4.** Embed Lessons S4-L1 through S4-L5 in the synthesist's audit-trail section. Do not dispatch eval-designer for S4. Do not enter revision-3 (which would burn a third red-team round on a workshop-grade contribution against a main-track spec, and would force user escalation under the cap).

The clean honest KILL is the right call given the spec bar. This matches the path the orchestrator's strategic guidance noted S6 took. The audit trail is enriched by the lessons, the swarm budget is preserved for the surviving hypotheses, and the user is not asked to adjudicate a borderline contribution at the cap.
