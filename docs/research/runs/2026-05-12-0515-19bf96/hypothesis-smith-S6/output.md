# hypothesis-smith-S6 — Ablation-coverage flag-handoff worker (REVISION 1)

## 0. Response to red-team revision-1 objections

The red-team REJECT critique at `docs/research/runs/2026-05-12-0515-19bf96/red-team-S6/output.md` raised four Critical objections (C1–C4) and seven Important/Suggestion-tier objections. This section maps each to the concrete change made in the revision; the body of `output.md` is rewritten accordingly. Where I disagree with an objection, I say so explicitly and ground the disagreement.

### C1 — Mechanism collapse on Step (a): rubric-promotion is the existing LM-Planner baseline

**Concession.** The red-team is correct after direct re-read of AblationBench §4.2: the LM-Planner already takes "title, abstract, and aggregated TeX source," uses "a single CoT style prompt to an LM," and outputs "structured JSONL file containing up to k ablation entries" with "removal or modification of a component" typology. That IS the "rubric-as-prompt" structure the previous draft claimed was the novel intervention. The +8 F1pp Agent-Planner → LM-Planner lift (§7.4, p=0.009) was earned moving from a worse scaffold to the rubric-shaped prompt. Promoting "rubric to first-class worker prompt" is the published baseline, not a contribution.

**Fix applied.** The novel mechanism is reframed to the **file-handoff substrate**: the LM-Planner output (or any equivalent rubric-shaped coverage call) is written by a leaf worker to a structured artifact (`ablation-coverage.yaml`), which a *next-wave* eval-designer reads and acts on. The novelty is therefore the **cross-wave file passing of named missing-ablation IDs through MegaResearcher's stateless-leaf-dispatch architecture** — a question genuinely empty in the literature (verified per gap-finder-1 §4 item 4 "No paper I read evaluates whether file-based artifact handoff is sufficient as a memory substitute"). The predicted magnitude is reduced accordingly because the in-distribution rubric-prompt lift is already captured by LM-Planner; the additional Δ now claimed comes only from second-wave incorporation. See §2 and §4 below.

### C2 — Metric substitution (F1 → recall@5)

**Concession.** AblationBench §7.4 reports +8 percentage points on **F1@5**, not recall@5. F1 = 2·P·R/(P+R); a recall@5 lift cannot be derived from an F1 lift without knowing what precision did. The previous draft conflated the two.

**Fix applied.** Primary metric is **locked to F1@5 to match AblationBench Table 4**. The predicted Δ is also restated and reduced: where the previous draft predicted Δ ≥ +10pp on recall@5, the revised hypothesis predicts **Δ ≥ +3 F1pp** over the strongest of (LM-Planner Claude 3.5 Sonnet baseline, AI Scientist v2 at the same substrate, AI-Researcher at the same substrate) on AuthorAblation, attributable to the second-wave eval-designer incorporation step. The +3 F1pp figure is derived below (§4) with explicit reasoning from the smaller-than-baseline-promotion magnitude argument. Recall@5 is retained as a secondary diagnostic to be reported alongside, but no decision rule is pegged to it without precision-matched derivation.

### C3 — "Non-judge signal" claim is false; AblationBench AuthorAblation recall is LM-judge-mediated

**Concession.** AblationBench §4.3 makes it explicit: "we design an LM-based judge that compares the GT ablations to the generated ones and decides whether each has a match." The published 0.76/0.79/0.74 precision/recall/F1 are LMJudge-vs-human agreement, not deterministic-checker signal. Calling AblationBench recall@5 a "non-judge signal" is wrong on the substance.

**Fix applied.** Two parts.

1. The AblationBench falsifier is restated honestly: it is **an LM-judge-mediated metric with measured 0.74 F1 against humans, using AblationBench's published cross-family majority-vote-of-3 mitigation** (§4.3: ensemble of three models with positional and contextual debiasing). The LMJudge is external to MegaResearcher's writer/critic models and uses a different scaffolding than the agent under test, which is a partial — not perfect — defense against same-family contamination. This is acceptable as a non-MegaResearcher-judge signal but NOT as a deterministic-checker signal; the distinction matters and the revision states it explicitly.
2. The genuinely deterministic falsifier is **elevated to co-primary**: the file-diff incorporation rate of named flags from the checker's `ablation-coverage.yaml` into the eval-designer's revised protocol. This is a literal string/structured-field diff with no LM judge anywhere in the chain. With C3 fixed this way, the falsification design has one LM-judge-mediated primary (AblationBench F1@5) and one deterministic co-primary (file-diff F2). Both must pass for the hypothesis to survive.

### C4 — AbGen omission (arXiv:2507.13300)

**Concession.** AbGen (Zhao et al., Yale, 2025-07-13, 20 upvotes on HF) is the parallel benchmark the previous draft missed. Verified via `hf_papers paper_details` at revision time: 1,500 expert-annotated examples from 807 NLP papers, with the headline finding "automated evaluation systems may not be fully reliable for our task" (§4.2 RQ1 conclusion) and a meta-evaluation benchmark AbGen-Eval (§5) specifically built to measure LMJudge reliability against humans on this task.

**Fix applied.** AbGen is added as the second substrate in §6 and treated as a triangulating comparator:

- **AblationBench AuthorAblation (primary substrate):** retains F1@5 + LMJudge-with-cross-family-ensembling as before, with the C3 framing.
- **AbGen (robustness substrate):** uses AbGen's human-evaluation protocol (Likert 1–5 across Importance / Faithfulness / Soundness, §3.2) on a small expert-annotated re-evaluation subset. AbGen's automated GPT-4.1-mini judge (§3.3) is used as secondary, with the explicit acknowledgment from AbGen §4.2 that the LM-based eval correlations to human are weak (note: AbGen explicitly states "disparity between automated evaluation systems and human assessments").
- The hypothesis is pre-registered to require the F1@5 lift on AblationBench AuthorAblation AND directional improvement (≥ +0.10 on a 1–5 Likert overall human-evaluated score) on AbGen's testmini human-eval subset. If AbGen shows the lift is benchmark-specific, that's a documented limitation, not a kill.

AbGen's "unreliability of automated evaluation methods" finding does not directly contradict the hypothesis — it strengthens the case for using AbGen's human-eval column rather than its automated column as the AbGen-side falsifier. The revision uses the human-eval subset, not GPT-4.1-mini.

### Important/Suggestion-tier objections

**O5 (Feedback-Friction domain transfer).** Conceded. The Feedback Friction §3.2 magnitudes (+26.7% AIME, +33.3% Llama-4-Scout) are math-reasoning lifts, not paper-pipeline lifts. The revision **drops the "+2pp adjacent to AblationBench-magnitude" reasoning** and re-grounds the F2 ≥ 60% threshold as **exploratory, with no within-domain prior in the literature**. Feedback Friction is now cited only for the qualitative claim that F3-shaped (named, reasoned, structured) feedback outperforms F1-shaped (binary) feedback on incorporation, NOT for any specific numeric magnitude transferred to this substrate. See §3 step (b).

**O6 (F2 trivializability — checker could issue 1 flag and trivially pass 60%).** Conceded. F2 is reformulated with a **precision-of-flags floor**: the checker must issue at least 3 flags per paper AND ≥ 60% must be incorporated AND ≥ 50% of incorporated flags must correspond to actual missing ablations as scored by AblationBench's GT (i.e., the incorporated flag must hit a GT entry the LM-Planner baseline missed). See §5 F2.

**O7 (F3 third arm under-specified — "hand-curated rubric" not defined).** Conceded. The third arm is **dropped**. F3 now sweeps only two arms: AblationBench's AuthorAblation rubric (primary) and AblationBench's ReviewerAblation rubric (control). The "hand-curated MegaResearcher rubric" is removed from pre-registration because no construction protocol can be locked at hypothesis time without retro-fittability concerns. See §5 F3.

**O8 (AI Scientist v2 coverage-claim not spot-checked at right granularity).** Partial concession. The revision weakens Step (c) language from "AI Scientist v2 does not enforce a coverage criterion for ablations" to "AI Scientist v2's published abstract and gap-finder-1 §2 Rank 1 row report tree-search over experiments without describing an external ablation-coverage rubric; I cannot confirm the absence of an implicit-coverage register from the abstract alone, and an explicit-baseline-conditioning behavior is left as a measurement open question for eval-designer." See §3 step (c).

**O9 (ID space under-specified).** Resolved. Ablation IDs are **the ablation-name strings from AblationBench's GT schema** (AblationBench §4.1, e.g., "HybrIK Variant Comparison"). For papers not in AblationBench's GT, the checker assigns IDs by generating a short slug from the (component, removal-or-modification-type) tuple — deterministic given the checker's structured output. Specified in §6.

**O10 (Curie rigor module overlap).** Resolved. Per gap-finder-1 §1, Curie's rigor module (`arXiv:2502.16069` §3) wraps each step in a verification harness but does not check against an external coverage rubric; it is a within-experiment verification mechanism, not a cross-experiment coverage register. The hypothesis now states explicitly that Curie's rigor module overlaps zero with AblationBench's external rubric — they are orthogonal mechanisms. See §3 step (c).

**O11 (Field-weak overclaim).** Conceded. Language in §3 step (c) is softened from "the field is weak at ablations" to "no AI-Scientist-family pipeline integrates an external ablation-coverage rubric as a worker," matching the surviving claim from the red-team's gap re-verification.

### Honest disclosure: smaller magnitude, narrower scope

After all four Critical and seven Important/Suggestion objections are addressed, the predicted lift drops from Δ ≥ +10pp recall@5 to **Δ ≥ +3 F1pp on AblationBench AuthorAblation** attributable specifically to the file-handoff substrate beyond what LM-Planner alone produces. A +3 F1pp lift is **not a main-track-conference primary contribution by itself**; it is a system-integration measurement of whether a published rubric carried via file artifacts to a second wave adds anything beyond the rubric-as-prompt baseline. The synthesist should position S6 as **necessary scaffolding rather than primary contribution** — the value is in demonstrating that MegaResearcher's stateless-leaf-dispatch architecture can integrate external coverage rubrics through file handoff, not in beating AblationBench's ceiling. This honest framing is requested by the revision-1 instructions and I endorse it.

---

## 1. Targeted gap

**Source:** `docs/research/runs/2026-05-12-0515-19bf96/gap-finder-3/output.md` shortlist entry **S6** (Ablation-coverage checklist worker), itself folding:

- `docs/research/runs/2026-05-12-0515-19bf96/gap-finder-1/output.md` §2 **Rank 1 (tied): Ablation Discipline** — `S=0, W=2, A=11, U=1` across the 14-system scout-1 matrix. Only Curie (`arXiv:2502.16069`) and Jr. AI Scientist (`arXiv:2511.04583`) score even partial "W"; neither integrates an external coverage rubric.
- `docs/research/runs/2026-05-12-0515-19bf96/scout-3/output.md` §5 #2 — "No paper isolates the workshop-vs-main-track quality delta," with reviewer-flagged insufficient ablations a recurring main-track-rejection cause.

The published existence-proof of the field-level gap is **AblationBench** (`arXiv:2507.08038`, Abramovich & Chechik) — best-performing LM system reaches 38% on AuthorAblation, below human-level — and **AbGen** (`arXiv:2507.13300`, Zhao et al., Yale), with 1,500 expert-annotated examples from 807 NLP papers showing significant LLM-vs-expert gap on ablation design and flagging unreliability of automated evaluation.

**Surviving claim (post-red-team verification):** No AI-Scientist-family pipeline integrates an external ablation-coverage rubric as a worker with cross-wave file handoff. This is narrower than the original "no system coverage-checks externally" framing, but is what the gap-verification supports. The remaining empty cell is **integration**, not the *existence* of a coverage rubric (AblationBench and AbGen provide rubrics) and not the *capability* of a single LM call to use one as a prompt (LM-Planner already does this).

## 2. Hypothesis statement

**If** MegaResearcher's swarm is augmented with a leaf worker `ablation-coverage-checker` that (i) runs after the eval-designer's first-wave protocol output is written, (ii) takes as a prompt AblationBench's AuthorAblation rubric structure (a structured JSONL list of removal- or modification-type ablation entries, per AblationBench §4.2), and (iii) emits an `ablation-coverage.yaml` artifact naming each flagged missing ablation by an ID drawn from a deterministic slug schema (component-name + removal/modification-type), AND a follow-up eval-designer wave is dispatched with that artifact as input, **then** on a held-out 20-paper sample of AblationBench's AuthorAblation test set, the full two-wave configuration achieves AuthorAblation **F1@5 ≥ 0.03 above** the strongest of (B1 LM-Planner Claude 3.5 Sonnet at the same substrate, B2 AI Scientist v2 at the same substrate, B3 AI-Researcher at the same substrate), pre-registered, AND the deterministic file-diff incorporation rate of named flags from the checker into the eval-designer's revised protocol is ≥ 60% subject to the precision-of-flags floor in §5 F2.

Both clauses are binary-checkable: (a) F1@5 differential on AblationBench AuthorAblation, decided by AblationBench's published LMJudge with cross-family majority-vote-of-3; (b) deterministic file-diff incorporation rate.

## 3. Mechanism

The proposed mechanism has three steps, each grounded in cited published results. **The novelty is concentrated in step (b): the second-wave file-handoff substrate carries named missing-ablation IDs into a downstream worker that would not otherwise see them.** Steps (a) and (c) establish the inputs and the absence of equivalent integration in the baselines.

**Step (a): Rubric-shaped single-call prompts are the strongest known scaffold for ablation generation.** AblationBench §7.4 reports LM-Planner (single CoT call with structured JSONL output containing up to k ablations with removal/modification typology, §4.2) outperforms Agent-Planner (SWE-agent multi-step) by 8 F1 percentage points on AuthorAblation with Claude 3.5 Sonnet (p=0.009, paired t-test), at 2.35× lower cost (verified via `read_paper` on §4.2 and §7.4). **This is the published baseline, not a contribution.** The hypothesis takes this scaffold as a given input to the checker worker.

**Step (b): The novel claim — file-handoff carries named missing-ablation IDs across waves in a stateless-leaf-dispatch architecture.** This is the empty cell. Three sub-claims:

- (b1) **No published AI-Scientist-family pipeline does this.** Gap-finder-1 §2 Rank 1 establishes 12/14 scout-1 systems score "A" (absent) on external ablation-coverage; the remaining 2 ("W" partial) — Curie and Jr. AI Scientist — score W on *baseline-conditioning behavior*, not on external-rubric coverage with named-flag file handoff (gap-finder-1 §2 row; verified during red-team gap re-check). The integration question is empty in the literature.
- (b2) **Structured named feedback is qualitatively easier to incorporate than free-form feedback.** Feedback Friction (`arXiv:2506.11930`) §3.2 demonstrates this qualitatively across multiple model families: Strong-Model Reflective Feedback (F3-shape) outperforms Binary Correctness Feedback (F1-shape) on the integration step, regardless of magnitude. **I do not import the specific +26.7% / +33.3% numbers, which come from AIME / GPQA math-reasoning substrates** (per red-team C5 objection). The direction of the F3 > F1 effect is what the hypothesis depends on, not the magnitude.
- (b3) **The file-handoff substrate is the specific MegaResearcher constraint being tested.** MegaResearcher's leaf workers are stateless; cross-wave memory exists only through file artifacts. Gap-finder-1 §4 item 4 explicitly flags this as unmeasured: "No paper I read evaluates whether file-based artifact handoff is sufficient as a memory substitute." S6 is one test instance of that broader open question, restricted to ablation-coverage.

**Step (c): No AI-Scientist-family baseline has the integration mechanism.** Independently grounded:

- AI Scientist v2 (`arXiv:2504.08066`) — published abstract describes "progressive agentic tree-search, experiment manager agent, VLM" with no mention of an external-rubric coverage check. **Caveat per red-team O8: I cannot verify absence of an implicit-coverage register from the abstract alone**; the tree-search manager may track coverage in ways not visible from the abstract. The eval-designer should treat this as a measurement open question, not a settled claim.
- AI-Researcher (`arXiv:2505.18705`) §6.3 — Scientist-Bench reviewer is presentation-biased; no ablation-coverage step described.
- AI Scientist v1 (`arXiv:2408.06292`) — ablation grid is template-bound (cannot run beyond template support).
- AgentRxiv (`arXiv:2503.18102`) §4.1 — same-family reviewer reward-hacks; no separate ablation-coverage pass.
- Curie (`arXiv:2502.16069`) §3 — rigor module wraps individual experimental steps in within-experiment verification, **orthogonal to external cross-experiment ablation-coverage rubric** (per red-team O10).
- The "Hidden Pitfalls" survey (`arXiv:2509.08713`, Luo/Kasirzadeh/Shah) names post-hoc selection bias as a documented AI-scientist failure mode; absence of pre-emit coverage rubric is the design hole that enables this.

The chain (a)+(b)+(c) is: scaffold (a) is the same as the baseline; integration (b) is the novel claim, with the specific testable sub-claim being whether file-handoff into a second wave produces F1@5 lift beyond what LM-Planner alone produces; (c) establishes that no baseline has this integration.

## 4. Predicted outcome with magnitude

**Primary metric:** AblationBench AuthorAblation **F1@5** (the metric AblationBench Table 4 actually reports; LM-judge-mediated with 0.74 F1 against humans per §7.3, using cross-family majority-vote-of-3 with positional and contextual debiasing per §4.3).

**Substrate:**
- AblationBench AuthorAblation 20-paper held-out sample from the 83-paper test set (`arXiv:2507.08038` §4.1; public benchmark at github.com/ai-scientist-bench/ablation-bench).
- AbGen testmini subset (`arXiv:2507.13300` §2.6; 500-example testmini set) used as robustness substrate with **human evaluation column** as the AbGen-side signal (not the GPT-4.1-mini automated column, per AbGen's own §4.2 acknowledgment that automated and human evaluations disagree).

**Baseline (named-prior comparators):**
- **B1:** AblationBench's published LM-Planner with Claude 3.5 Sonnet at the same substrate — Table 4 of `arXiv:2507.08038`.
- **B2:** AI Scientist v2 (`arXiv:2504.08066`) prompted with the same paper-method-section input, evaluated by AblationBench's LMJudge.
- **B3:** AI-Researcher (`arXiv:2505.18705`) at the same substrate.

**Predicted treatment:** Two-wave MegaResearcher (eval-designer wave 1 → checker → eval-designer wave 2 with `ablation-coverage.yaml` as input) achieves **F1@5 ≥ max(B1, B2, B3) + 0.03 absolute**, with lower 95% paired-bootstrap CI bound > 0.

**Magnitude reasoning — why +3 F1pp and not more or less:**

The previous draft's +10pp recall@5 prediction collapsed under red-team C1 (rubric promotion is the baseline) and C2 (F1 ≠ recall). The corrected derivation:

1. The +8 F1pp Agent-Planner → LM-Planner lift (AblationBench §7.4, Claude 3.5 Sonnet, p=0.009) represents the magnitude of moving from a worse scaffold to the rubric-shaped prompt baseline. **The S6 augmentation does NOT add this lift on top of LM-Planner**, because LM-Planner already realizes it.
2. The *additional* lift S6 must produce comes from the second-wave eval-designer incorporation step. The mechanism is: the checker (an LM-Planner-equivalent call against AblationBench's rubric) writes named missing-ablation flags; the eval-designer in wave 2 adds these into its protocol; some fraction of those added flags hit GT ablations the wave-1 protocol missed.
3. **The headroom is bounded by the union of wave-1 misses that wave-2 can recover via named-flag handoff.** If wave-1 (an AI-Scientist-family baseline) produces 0.30 recall@5 on AuthorAblation (consistent with the 38% best-performing LM-Planner figure), and the checker correctly flags ~50% of the gaps to GT (a permissive guess given LMJudge reliability), and wave-2 incorporates ~60% of those flags (F2 threshold), then the maximum recoverable recall is ≈0.30 + (1-0.30)·0.50·0.60 = 0.51 in the most optimistic scenario. F1@5 lifts from recall lifts depend on precision behavior; assuming precision degrades modestly because wave-2 adds 3+ flagged ablations to its protocol, the realized F1@5 lift is plausibly +3 to +6 percentage points in the optimistic scenario.
4. **+3 F1pp is the pre-registered minimum-defensible threshold.** This is below the most-optimistic +6 because: (i) AblationBench §7.4 documents diminishing returns on top of the LM-Planner baseline; (ii) AblationBench §7.4 also documents that modification-type ablations are bottlenecked by domain knowledge ("relevant and feasible substitutions—often informed by domain-specific knowledge—remains a challenge for current models"), capping the second-wave incorporation's ability to convert flags into hits; (iii) the checker is itself fallible at the LMJudge-precision level (0.76 against humans), meaning some flags are spurious and cannot help.

**Honest acknowledgment per the revision-1 instructions:** A +3 F1pp lift is **not a main-track-conference primary contribution by itself**. It is a measurement of whether MegaResearcher's file-handoff architecture can carry a published rubric across waves with non-zero downstream value. The synthesist should position S6 accordingly.

**Secondary metric (deterministic co-primary per C3):** Deterministic file-diff incorporation rate. After the checker writes `ablation-coverage.yaml` flagging K missing ablations, the eval-designer in wave 2 is invoked with the artifact as input. The threshold is **≥ 60% of flagged ablations show up as named entries in the eval-designer's revised protocol** (binary file-diff over structured YAML keys, no LLM judge). Co-primary with the F1@5 lift — both must pass.

**Conditions under which the hypothesis SHOULD hold:**
- AblationBench AuthorAblation as the primary substrate, AbGen testmini human-eval column as robustness substrate.
- Claude 3.5 Sonnet or comparable strongly-grounded frontier model on the checker (AblationBench §7.4 finds Claude is best on AuthorAblation specifically because of grounding behavior).
- Same paper substrate across all arms (paired design).

**Conditions under which the hypothesis SHOULD NOT hold (pre-registered scope limits):**
- **AblationBench ReviewerAblation** — §7.4 documents that grounded models *underperform* less-grounded models here (creativity beyond explicit experiments). Mechanism does not predict the lift on ReviewerAblation; this is a pre-registered scope boundary, not a falsification.
- **Modification-type ablations** specifically — recall@5 lift may be concentrated on removal-type ablations (per §7.4). If lift is absent on modification-type, hypothesis still holds in aggregate, but eval-designer pre-registers a removal/modification split as secondary analysis.
- **AbGen human-eval column** — if the lift fails to transfer (i.e., AblationBench shows +3 F1pp but AbGen human-eval shows no directional improvement), the result is benchmark-specific; this is documented as a substrate-portability limitation, not a falsification of the F2 file-handoff claim, but is a falsification of the broader claim of robustness.

## 5. Falsification criteria

Each is sufficient on its own. All thresholds pre-registered before experiment execution.

**F1. F1@5 lift below threshold.** If on the 20-paper AuthorAblation sample, the treatment configuration achieves F1@5 < (max(B1, B2, B3) + 0.03), with lower 95% paired-bootstrap CI bound ≤ 0, the hypothesis is falsified on the primary differential-outcome criterion.

**F2. Friction-conversion failure (deterministic co-primary, addresses C3).** Falsified if any of:
- (F2a) The checker issues fewer than 3 flags per paper on average (precision-of-flags floor, addresses red-team O6 trivializability).
- (F2b) The eval-designer's wave-2 revised protocol incorporates < 60% of issued flags as named entries (deterministic file-diff over structured YAML, no LLM judge).
- (F2c) Among incorporated flags, < 50% correspond to actual missing ablations as scored by AblationBench's GT key (i.e., the incorporated flag must hit a GT entry the wave-1 protocol missed).
- All three sub-conditions must be met for F2 to pass. Any one failing falsifies the friction-conversion mechanism (Step b in §3).

**F3. No rubric works (narrowed per O7).** F3 sweeps **two arms only**: AblationBench AuthorAblation rubric (primary) vs AblationBench ReviewerAblation rubric (control). The hand-curated arm is dropped. Falsification fires only if neither rubric, swept independently, produces the predicted ≥ +3 F1pp lift.

**F4 (supplementary, not counting toward 3-criterion floor). Cost-budget breach.** If achieving the lift requires > $200/replication (C4 ceiling from gap-finder-3 §0), falsified on feasibility. Budget per §6 estimate: ~$140/replication = 4 LM-calls/paper × 20 papers × ~$2/call (one extra call for the second eval-designer wave vs the previous draft's plan).

**F5 (supplementary). AbGen non-transfer.** If AblationBench shows the F1@5 lift but AbGen testmini human-eval Overall score shows no directional improvement (Δ < +0.10 on the 1–5 Likert scale, see AbGen §3.2 / Table 2), this is documented as benchmark-specific, not a F1-falsification but a substrate-portability limitation surfacing in the synthesist's section.

## 6. Required experiments (sketch — eval-designer details)

**Substrate:**
- **Primary:** AblationBench AuthorAblation 20-paper held-out sample from the 83-paper test set (`arXiv:2507.08038` §4.1; public benchmark).
- **Robustness:** AbGen testmini 20-example sample (drawn from the 500-example testmini set, `arXiv:2507.13300` §2.6) with the AbGen human-evaluation Likert protocol (importance/faithfulness/soundness, §3.2).
- Same paper substrate across all baseline/treatment arms within each benchmark.

**Baselines (pre-registered):**
- **B1:** AblationBench LM-Planner with Claude 3.5 Sonnet (Table 4 of `arXiv:2507.08038`).
- **B2:** AI Scientist v2 (`arXiv:2504.08066`) prompted with paper title/abstract/method section; output evaluated by AblationBench LMJudge and AbGen human eval.
- **B3:** AI-Researcher (`arXiv:2505.18705`) at the same substrate.

**Treatment:**
- Two-wave MegaResearcher: (Wave 1) eval-designer produces an initial protocol; (Checker leaf) `ablation-coverage-checker` reads wave-1 protocol + paper input, writes `ablation-coverage.yaml`; (Wave 2) eval-designer is re-invoked with the YAML artifact in input context, produces revised protocol.
- IDs in `ablation-coverage.yaml`: for AblationBench substrate, IDs draw from AblationBench's GT schema ablation-name strings; for AbGen and out-of-substrate cases, IDs are deterministic slugs from (component, removal-or-modification-type) (per O9 resolution).

**Ablation arms (sweeps):**
- Rubric source (F3): AuthorAblation rubric (primary), ReviewerAblation rubric (control).
- Checker model: Claude 3.5 Sonnet (primary), GPT-4o (control), Llama 3.1 405B (less-grounded control per AblationBench §7.4).
- File-handoff vs in-context arm (key for the C1 reframe): in addition to the two-wave file-handoff configuration, run a single-wave variant where the checker's output is concatenated into the wave-1 eval-designer's prompt context directly. This isolates the file-handoff substrate's contribution from the rubric-prompt content. If the lift is the same with or without file-handoff, the contribution is attributable to the rubric content alone (i.e., the C1 collapse re-materializes); if the file-handoff variant lifts F1@5 above the in-context variant, the file-handoff substrate is the proximal mechanism. **This arm is the most diagnostic experiment in the design.**

**Falsifiers (per spec discipline rule, judge breakdown):**
- **Primary falsifier (LM-judge-mediated, externally provided):** AblationBench LMJudge for F1@5 — cross-family majority-vote-of-3 with positional and contextual debiasing (§4.3), 0.74 F1 against humans (§7.3 Table 3). External to MegaResearcher's writer/critic models.
- **Co-primary falsifier (deterministic):** file-diff named-flag incorporation rate (F2). No LLM judge anywhere in this signal.
- **Robustness falsifier:** AbGen human-eval Likert subset (F5). Human evaluators, not LM judges.

**Decision rule pre-registration:**
- Primary outcome: paired F1@5 difference with 95% CI (paired bootstrap, ≥1000 replicates).
- Threshold: Δ F1@5 ≥ +0.03 absolute, lower 95% CI bound > 0.
- Co-primary outcome F2: composite (≥3 flags/paper average AND ≥60% incorporation AND ≥50% GT-hit precision).
- Robustness outcome F5: Δ AbGen human-eval Overall ≥ +0.10 on 1–5 Likert.
- All thresholds locked at hypothesis-submission time.

## 7. Risks to the hypothesis

I list 4 distinct risks. Each ends with "what the hypothesis still contributes if this materializes."

**R1. The +3 F1pp prediction does not generalize beyond Claude 3.5 Sonnet.** AblationBench §7.4 reports the +8 F1pp baseline-promotion lift specifically for Claude 3.5 Sonnet; magnitudes vary across models. If the checker model is constrained (cost, availability), the F1@5 lift may fall below +3. **Contribution if materialized:** A +1 to +2 F1pp directional lift, with F2 co-primary still passing, is still publishable as "file-handoff substrate carries named flags into a second wave with measurable but sub-threshold value; the bottleneck is the checker's rubric-grounded generation quality, suggesting future work on rubric-grounded checker training rather than architecture."

**R2. AblationBench rubric is not the right rubric for main-track ablation coverage.** AblationBench is operationalized from a specific subset of CSR-Bench, SUPER-Expert, and PaperBench (§4.1). Its rubric may miss main-track reviewer concerns. The F3 sweep (AuthorAblation vs ReviewerAblation) is pre-emption. **Contribution if both fail:** A negative result pinning down "no AblationBench-derived rubric, even with file-handoff carrying named flags to a second wave, closes the workshop-vs-main-track delta." That's a useful finding pointing future work toward better rubric construction (gap-finder-3 §4 item 2 "no unified failure taxonomy").

**R3. File-handoff substrate is functionally equivalent to in-context concatenation.** The file-handoff vs in-context arm (§6) is designed to test this directly. If the two variants produce statistically indistinguishable F1@5, the file-handoff substrate adds no value beyond the rubric content itself — meaning the C1 collapse re-materializes at the architectural level. **Contribution if materialized:** A measurement of the architectural-equivalence claim. The synthesist documents that MegaResearcher's stateless-leaf-dispatch + file-handoff architecture is computationally equivalent (on this task) to a single-wave in-context-pass configuration. That is itself a finding about the file-handoff substrate (gap-finder-1 §4 item 4).

**R4. AbGen and AblationBench disagree on whether the lift exists.** If AblationBench shows the F1@5 lift but AbGen human-eval does not show directional improvement, the substrate-portability claim fails. **Contribution if materialized:** A documented substrate-specific finding plus a methodological observation that AblationBench's LMJudge and AbGen's human-eval column disagree even on directional improvement; this directly extends AbGen's own §4.2 finding that "current automated evaluation systems may not be fully reliable for our task." That extension is publishable as a methods/measurement paper finding.

## 8. Sources

All citations resolve via `hf_papers paper_details`. Re-verified at revision-1 submission time.

- `arXiv:2507.08038` — **AblationBench: Evaluating Automated Planning of Ablations in Empirical AI Research** — Abramovich & Chechik. Primary substrate; §4.2 LM-Planner rubric structure; §4.3 LMJudge architecture and cross-family majority-vote-of-3 mitigation; §7.4 +8 F1pp Agent-Planner → LM-Planner lift with p=0.009, grounded-model trade-off across AuthorAblation/ReviewerAblation; Table 3 §7.3 LMJudge precision 0.76 / recall 0.79 / F1 0.74 against humans; Table 4 baseline F1@5 numbers.
- `arXiv:2507.13300` — **AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research** — Zhao et al. (Yale). 1,500 expert-annotated examples from 807 NLP papers (§2.6); §3.2 human-evaluation Likert protocol on importance/faithfulness/soundness; §3.3 GPT-4.1-mini LLM-as-judge; §4.2 RQ1 finding "current automated evaluation systems may not be fully reliable for our task"; §5 AbGen-Eval meta-evaluation benchmark for LMJudge reliability. **Added in revision-1 per red-team C4.**
- `arXiv:2506.11930` — **Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback** — Jiang, Zhang, Wang, Andrews, Khashabi. §3.2 qualitative finding that F3-shaped (named, reasoned, structured) feedback outperforms F1-shaped (binary) feedback on integration across model families. **Magnitudes from AIME/GPQA NOT imported per revision-1 O5 fix.**
- `arXiv:2504.08066` — **The AI Scientist-v2** — Yamada et al. Named baseline B2; tree search over experiments, no published external-rubric coverage check at the abstract level. Specific claim weakened per O8.
- `arXiv:2505.18705` — **AI-Researcher: Autonomous Scientific Innovation** — Tang, Xia, Li, Huang. Named baseline B3; §6.3 Scientist-Bench is presentation-biased.
- `arXiv:2408.06292` — **The AI Scientist** — Lu et al. Ablation grid template-bound.
- `arXiv:2503.18102` — **AgentRxiv** — Schmidgall, Moor. §4.1 same-family reviewer reward-hacking.
- `arXiv:2509.08713` — **The More You Automate, the Less You See** — Luo, Kasirzadeh, Shah. Post-hoc selection bias as the failure mode a pre-emit coverage rubric attacks.
- `arXiv:2502.16069` — **Curie** — Kon et al. Per gap-finder-1 §1; §3 rigor module is within-experiment verification, orthogonal to external cross-experiment coverage rubric (per O10).
- `arXiv:2510.18003` — **BadScientist** — Jiang et al. Concern-acceptance conflict failure mode (referenced in revision-0 R4, retained as context but no longer load-counting for the LMJudge contamination defense, which is now grounded in AblationBench §4.3 directly).
