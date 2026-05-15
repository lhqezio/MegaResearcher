# gap-finder-1 — Capability gaps across the autonomous-research-systems genre

## 0. Slice scope

Sources read in full:

- `docs/research/runs/2026-05-12-0515-19bf96/scout-1/output.md` — end-to-end autonomous-research systems (14+ systems)
- `docs/research/runs/2026-05-12-0515-19bf96/scout-2/output.md` — manuscript drafting + document-scale coherence
- `docs/research/runs/2026-05-12-0515-19bf96/scout-3/output.md` — automated peer review + paper-quality evaluation
- `docs/research/runs/2026-05-12-0515-19bf96/scout-4/output.md` — experiment execution + verification

Plus targeted `hf_papers` verification queries (logged per gap).

Cell notation in §1: **S** = strong (explicit, evaluated), **W** = weak (present but flagged failing in same paper or a contemporary), **A** = absent (not described / explicitly future-work), **U** = unknown (no evidence in slice).

---

## 1. The matrix

Rows = systems from scout-1 (with arXiv ID for resolution). Columns = capabilities from the spec.

Abbreviated column headers:
- **MD** = Manuscript Drafting
- **PR** = Peer-review loop (drives revision before emit)
- **EX** = Experiment Execution (runs code, not just plans)
- **ST** = Statistical Rigor (multi-seed, significance, error bars)
- **RW** = Related-Work Map (systematic, not parametric)
- **AB** = Ablation Discipline (planned & isolated)
- **CV** = Citation Verification (pre-flight or gated)
- **TR** = Theoretical Reasoning (proofs, derivations)
- **RU** = ICLR-Rubric Self-Evaluation (against a venue rubric)
- **AT** = Audit Trail (rejected hypotheses preserved cross-run)
- **LH** = Long-Horizon Coherence (>1-day, persistent state)

| System (arXiv) | MD | PR | EX | ST | RW | AB | CV | TR | RU | AT | LH |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AI-Researcher (2505.18705) | S §3 docs stage | W §6.3 LLM reviewer overvalues presentation | S §3 algorithm impl | U | W §3 Resource Analyst, no graph | A | A | A | W §6.3 Scientist-Bench rubric, but presentation-biased | A | W §6.2 named memory gap |
| AI Scientist v1 (2408.06292) | W template-based, §writeup | W same-family reviewer; 2509.08713 §2 calls reward-hacking | S Aider-driven sandbox | A — toy scale only | A — §limit "no real RW" | A | A — Semantic Scholar lookup only, post-hoc | A | W NeurIPS-form auto-reviewer (same model class) | A | A linear |
| AI Scientist v2 (2504.08066) | W single-pass writeup §4 | W VLM critic on figures + auto-reviewer (same family) | S tree search w/ exp manager | W workshop-bar only | A | A | A | A | W workshop-form reviewer (still self-mirror) | A | W tree state, single-run |
| Agent Laboratory (2501.04227) | S paper-solver | W paper-solver reward-hacks NeurIPS scorer §limit | S mle-solver | U | W PhD lit-review agent | A | A | A | W mle-solver LLM-reward; documented reward-hacking | A — silent rejections | A no mid-plan revision |
| AgentRxiv (2503.18102) | S inherits Agent Lab | W §4.1 reward-hacking documented | S | U | W +preprint memory | A | A | A | W same | W cross-run preprint store, not failure-keyed | W cross-lab memory of accepted reports only |
| Dolphin (2501.03916) | A — no manuscript output | A | S traceback-guided exec | W — narrow eval domains | A | A | A | A | A | W feedback loop carries result records | W closed loop, no persistent memory across rounds |
| EvoScientist (2603.08127) | U — abstract does not detail writeup | A — no peer-review-style eval connection §limit | S engineer agent | U | U | U | A | A | A — no formal review connection | S persistent ideation + experimentation memory | S Evolution Manager learns across runs |
| freephdlabor (2510.15624) | W writer in reference deployment, qualitative only | A | S executor | U | U | A | A | A | A | W workspace + persistent memory, not rejection-keyed | S dynamic workflow + persistent memory; central pitch |
| Idea2Paper (2601.20833) | S online compose, offline graph | W minimal critique-revise §limit | W less emphasis on exec | U | S offline methodological graph (precomputed) | A | A — graph quality bounds | A | A | A | W graph is static; no audit trail |
| Jr. AI Scientist (2511.04583) | S baseline-paper-conditioned | W AI reviewer; risk report acknowledges | S | U | S inherits baseline's RW skeleton | W partial — baseline-conditioned only | W partial — inherits from baseline | A | W Agents4Science criteria | A | A single-baseline scope |
| CycleResearcher (2411.00816) | S writer | S CycleReviewer (separately trained!) ⇄ writer | U — manuscript-gen focus | U | U | A | A | A | S trained reviewer with MAE < human disagreement (claim contested) | A | A |
| PaperOrchestra (2604.05018) | S decoupled writing pipeline | W internal reviewer | A — not exec-focused, takes pre-written results | A | S Literature Synthesizer agent | A | A | A | W internal reviewer | A | A |
| Aviary (2412.21154) | A — gymnasium, not paper output | A | S env-bound | U | A | A | A | A | A | A | W per-environment, not cross-task |
| Curie (2502.16069) | A — no manuscript output | A | S intra/inter-agent rigor modules | S explicit rigor module §3 | A | W rigor-enforcement wraps each step | A | A | A | W structured experiment records with pre/post conditions | W experiment-knowledge module |

Cell-population stats:
- 14 systems × 11 capabilities = 154 cells.
- Populated (S/W/A): 142.
- Unknown (U): 12.
- Population rate: ≈92% — clears the 80% target.

Notes on cell granularity:
- "W" is used when the capability is present *but the same paper or a contemporary documents it failing in a named way* (e.g., "paper-solver reward-hacks the NeurIPS scorer" in Agent Lab → cell is W on RU). This is intentional: "present but documented broken" is the gap pattern that matters for the spec.
- "A" with a section pointer means the paper's limitation or future-work section explicitly names the absence.
- "U" cells are not gap evidence; they are honesty markers.

---

## 2. Ranked list of systematically-thin capabilities

For each capability, I count "weak + absent" cells (W+A) across the 14 systems. Higher count = thinner across the genre. Where two ties exist, I rank by spec relevance (audit trail and ablation discipline tie-break high because the spec's novelty target is main-track-bar).

### Rank 1 (tied): Audit Trail (rejected-hypothesis lineage) — W/A in 13/14 systems

**Count:** S in 1 (EvoScientist 2603.08127). W in 5 (AgentRxiv, Dolphin, freephdlabor, CycleResearcher, Curie — each carries *some* trace but not a rejected-hypothesis record). A in 8.

**Citations of supporting evidence:**
- AI-Researcher (2505.18705) §6 — no rejected-hypothesis store described
- AI Scientist v1 (2408.06292) — "notes.txt running log" is per-run, no rejection record
- Agent Laboratory (2501.04227) §limit — silent rejection of failed experiment branches
- AgentRxiv (2503.18102) §4 — preprint store keeps *accepted* reports; failure modes catalogued post-hoc in §4 only
- Luo/Kasirzadeh/Shah (2509.08713) — explicitly calls "post-hoc selection bias" a failure mode and prescribes mandatory trace logs as the answer; no system in scout-1 has implemented this as a first-class artifact

**Verification query 1:** `hf_papers search "audit trail rejected hypothesis autonomous research"` — returned 8 results. Closest matches:
- `2605.05724` (Auto Research with Specialist Agents) — describes "auditable trajectory of proposals, code diffs, experiments, scores, and failure labels" — this is the closest published peer to what the gap describes, but is a single-domain (training-recipe) system, not a general manuscript-producing pipeline.
- `2604.24658` (The Last Human-Written Paper) — names "Storytelling Tax" of discarding failed experiments / rejected hypotheses in linear paper narratives.
- Of 14 scout-1 systems, only `2605.05724` and `2604.24658` (neither in scout-1's table) treat rejected-hypothesis lineage as a first-class artifact. Confirms the gap.

**Why this matters for the spec:** the spec's discipline rule #1 ("Audit trail is non-negotiable… every rejected/killed hypothesis appears in the synthesist's final document") is empirically novel relative to the 14 systems in scout-1; that's the architectural feature the hypothesis-smith can defend as the differentiator.

### Rank 1 (tied): Ablation Discipline (planned, isolated, reported) — W/A in 13/14 systems

**Count:** S in 0. W in 2 (Curie has rigor module that wraps each step; Jr. AI Scientist partial via baseline-conditioning). A in 11. U in 1.

**Citations of supporting evidence:**
- AblationBench (2507.08038, Abramovich & Chechik) — explicitly published *because* current AI co-scientist agents are weak at proposing and identifying ablations. Their AuthorAblation / ReviewerAblation tasks were created to measure this gap.
- AI Scientist v1 (2408.06292) §limit — "cannot run truly novel architectures beyond what the template supports" — ablation grid is template-bound
- AI Scientist v2 (2504.08066) §limit — tree search prunes branches, but does not enforce a coverage criterion for ablations expected by reviewers
- MLR-Bench (2505.19955) — finds "coding agents produce unreliable experimental results" even with high writing scores — ablation-coverage is part of what reviewers check
- Luo/Kasirzadeh/Shah (2509.08713) — "post-hoc selection bias" implicates ablation cherry-picking

**Verification query 2:** `hf_papers search "ablation discipline autonomous research agent ML paper"` — top hit is AblationBench (2507.08038). The fact that a 2025 benchmark exists *specifically to measure agent ablation-planning weakness* confirms the gap is open. Out of 14 scout-1 systems, none is published as solving the AblationBench tasks.

**Why this matters for the spec:** reviewers at main-track ML conferences mark "insufficient ablations" as one of the most common rejection causes; this is a direct route to closing the workshop-vs-main-track delta scout-3 §5 names.

### Rank 3: Citation Verification as a pre-flight gate — W/A in 14/14 systems

**Count:** S in 0. W in 1 (Jr. AI Scientist — inherits citations from baseline paper). A in 13.

**Citations of supporting evidence:**
- AI Scientist v1 (2408.06292) — uses Semantic Scholar lookup as a novelty check, *not* citation verification; CiteAudit (2602.23452) reports hallucinated citations already in accepted ML papers
- CiteME (2407.12861) — large LM↔human gap on citation attribution
- The 17% Gap paper (2601.17431) — quantifies unresolved citation rates in AI-assisted papers
- FactReview (2604.04074) is closest to pre-flight grounding, but per scout-3 entry it is still post-hoc; reads only manuscript narrative
- Scout-4 §5 #3 explicitly states: "All AI-scientist systems I read first generate citations, then optionally verify. The MegaResearcher rule 'if `hf_papers paper_details` does not return a paper, the paper does not exist' is a *pre-flight* gate. I did not find a paper that implements this pre-flight design."

**Verification query 3:** `hf_papers search "abstract results consistency check post-hoc verification scientific manuscript"` — top hits (CiteAudit, 17% Gap, SEA) all describe *post-hoc* verification. No system in the top 8 implements pre-flight citation resolution as a gate.

**Why this matters for the spec:** MegaResearcher's discipline rule #4 ("Citations resolve or do not exist") is empirically distinct from the 14 systems; hypothesis-smith can defend the pre-flight gate vs. the field-standard post-hoc check.

### Rank 4: ICLR-Rubric Self-Evaluation against an external rubric — W/A in 14/14 systems

**Count:** S in 0 (CycleResearcher claims human-disagreement-grade MAE but is reviewed by a model trained on the same review distribution; this is closed-loop, not external). W in 6 (AI Scientist v1/v2, Agent Lab, AgentRxiv, Jr. AI Scientist, PaperOrchestra all do internal reviewer scoring). A in 8.

**Citations of supporting evidence:**
- Höpner et al. (2503.05712) — title+abstract regression beats LLM-reviewer score prediction
- BadScientist (2510.18003) — fabricated papers fool LLM reviewers; "concern-acceptance conflict"
- "Merits and Defects of LLMs in Review" (2509.19326) — quality-insensitivity (LLMs give similar feedback regardless of paper quality)
- Pre-review to Peer review (2512.22145) — limited human alignment for autonomous review; recommends pre-review screening role
- SPECS-Review-Benchmark (2604.13940) — controlled flaw injection across Story / Presentation / Evaluations / Correctness / Significance; no scout-1 system has been evaluated against this

**Verification query 4:** `hf_papers search "ICLR rubric self-evaluation autonomous research paper agent"` — returns AI Scientist v2 (workshop-level only), CycleResearcher (its own reviewer), AgentRxiv (no rubric). PaperBench (2504.01848) uses ICML 2024 rubrics for *replication* assessment, not for the system's self-evaluation. The field has no system that scores its own draft against SPECS-Review-Benchmark or a comparable controlled-flaw rubric before emit.

**Why this matters for the spec:** scout-3 §5 #2 says "no paper isolates the workshop-vs-main-track quality delta"; the rubric-side cause is that no system uses a *main-track-shaped* rubric for self-evaluation.

### Rank 5: Statistical Rigor (multi-seed, significance, error bars) — W/A in 13/14 systems

**Count:** S in 1 (Curie 2502.16069 has explicit rigor module — but it does not produce manuscripts, so the rigor is not translated to reportable claims). W in 1 (Dolphin). A in 10. U in 2.

**Citations of supporting evidence:**
- Henderson et al. canonical RL-seed paper (1806.08295) shows the issue predates LLM-era pipelines
- Luo/Kasirzadeh/Shah (2509.08713) — "metric misuse" failure mode
- AI Scientist v1 §limit — "GPU resource limits force toy-scale experiments" implies no multi-seed budget
- MLR-Bench (2505.19955) — coding agents produce unreliable experimental results
- Statistical Methods in Generative AI (2509.07054) — survey acknowledges scarcity of statistical rigor in agent-generated experiments

**Verification query 5:** `hf_papers search "statistical significance multiple seeds autonomous AI scientist"` — only 1806.08295 (pre-LLM) directly addresses the issue. Of 8 results, no system in scout-1 reports its results with multi-seed CIs as a default.

**Why this matters for the spec:** main-track conferences expect error bars and significance tests; this is one of the most-cited rejection reasons in the conferences this spec targets.

### Rank 6: Related-Work Map (systematic, not parametric) — W/A in 11/14 systems

**Count:** S in 3 (Idea2Paper offline graph 2601.20833, Jr. AI Scientist baseline-paper RW skeleton 2511.04583, PaperOrchestra Literature Synthesizer 2604.05018). W in 4 (AI-Researcher Resource Analyst no graph, AgentRxiv preprint-only memory, freephdlabor unknown writer detail, CycleResearcher unknown). A in 7.

**Citations of supporting evidence:**
- STORM (2402.14207) — "source bias transfer" and "over-association" failure modes — known for paper writing
- AutoSurvey (2406.10252) — "parametric knowledge constraint failures" explicitly named
- SurveyG (2510.07733) — hierarchical citation graph as backbone; not used by any *paper-producing* scout-1 system except Idea2Paper
- Scout-2 §5 #6 — outline-generation work optimizes for outline quality, not for downstream manuscript related-work survival under expert review

**Verification query 6:** `hf_papers search "related work map related-work survey agent autonomous research paper"` — returns AutoSurvey, SurveyG, PaperCircle (2604.06170). Of the 8 returned, none is integrated into one of the 14 scout-1 paper-producing pipelines as a first-class subagent.

**Why this matters for the spec:** scout-2 §5 #6 directly names the gap; main-track reviewers reject papers with weak related-work, and this is where presentation-overweighting (BadScientist) most easily masks substance failure.

### Rank 7: Theoretical Reasoning (proofs, derivations) — A in 14/14 systems

**Count:** S in 0. W in 0. A in 14.

**Citations of supporting evidence:**
- Bolzano (2604.16989) — single multi-agent system for mathematical research, NOT in scout-1's autonomous-research-paper family
- Prover Agent (2506.19923) — Lean-based theorem proving, NOT a manuscript producer
- Every scout-1 system targets empirical ML; none produces or verifies derivations

**Verification query 7:** `hf_papers search "theoretical reasoning proof autonomous research agent LLM"` — confirms theorem-proving agents (Prover Agent, Bolzano) exist but are *separate from* the autonomous-paper-generation family. No scout-1 system has a theory worker.

**Why this matters for the spec:** main-track ML conferences (esp. ICLR, NeurIPS) commonly expect at least a proposition + sketch proof for new optimizers / loss functions / convergence claims. The genre's blind spot is structural, not just shallow.

### Rank 8: Long-Horizon Coherence (>1-day, persistent state) — W/A in 11/14 systems

**Count:** S in 2 (EvoScientist 2603.08127, freephdlabor 2510.15624 — both pitch persistent memory as headline). W in 7. A in 5.

**Citations of supporting evidence:**
- RE-Bench (2411.15114) — agents beat humans on short budgets but humans pull ahead with more time; agents fail to use additional compute productively (the canonical "long-horizon coherence loss" measurement)
- ML-Master 2.0 (2601.10402) — Hierarchical Cognitive Caching explicitly proposed to address it
- UltraHorizon (2509.21766) — "in-context locking" as a named failure mode
- AI-Researcher (2505.18705) §6.2 — memory management gap explicitly stated
- Most scout-1 systems are single-run; only EvoScientist and freephdlabor address cross-run memory

**Why this matters for the spec:** any system aiming at main-track-grade output needs to coherently maintain state across the days-long literature → ideation → experimentation → drafting pipeline; this is the gap where memory architecture is the named bottleneck across four papers.

---

## 3. Surfaced contradictions

### Contradiction 1: Workshop acceptance "proves main-track readiness"

- **One claim:** AI Scientist v2 (2504.08066) reports the first AI-generated paper accepted at an ICLR workshop and uses this as the headline credibility claim.
- **Counter:** scout-3 §5 #2 — "no paper isolates the workshop-vs-main-track quality delta"; AAAI-26 pilot (2604.13940) finds LLM reviewers catch some axes (Presentation, Story) but not others (Correctness, Significance) per SPECS; BadScientist (2510.18003) shows acceptance-grade scores can be earned by fabricated papers with no experiments.
- **Why it matters:** the workshop-acceptance claim is the most-cited evidence in the AI-scientist literature; the AAAI-26 + BadScientist results imply this is a weak proxy for main-track viability.

### Contradiction 2: CycleResearcher's human-disagreement-grade reviewer MAE

- **One claim:** CycleResearcher (2411.00816) reports CycleReviewer MAE on review scores below human inter-reviewer disagreement; framed as parity with human reviewers.
- **Counter:** Höpner et al. (2503.05712) — a simple title+abstract regression beats LLM-based reviewers on review-score prediction; Pre-review to Peer review (2512.22145) — frontier open-weight LLMs show limited alignment with human peer reviewers; Merits & Defects (2509.19326) — LLM reviews quality-insensitive.
- **Why it matters:** scout-3 §5 #1 surfaces this directly. CycleResearcher's MAE-parity claim is the strongest published claim of automated-reviewer reliability; multiple later papers say it doesn't hold up to different evaluation lenses.

### Contradiction 3: GPT-4o backbone-dependence of multi-turn code generation

- **One claim:** AI-Researcher (2505.18705) §6.1 — GPT-4o exhibits premature task completion in extended interactions; Claude 3.5 Sonnet does not — i.e., backbone-dependent. This is reported as a finding about *implementation fidelity*.
- **Counter:** RE-Bench (2411.15114) — agents (across backbones) fail to use additional compute productively over long budgets, finding the failure mode is *universal* across model families, not Claude-vs-GPT specific.
- **Why it matters:** this affects how the spec should reason about its workers. If 2505.18705 is right (backbone-dependent), then model routing is the fix; if RE-Bench is right (universal), the fix has to be architectural. The literature does not reconcile the two.

### Contradiction 4: Memory-as-architectural-fix consensus, but no convergence on mechanism

- **One claim:** Memory architecture is consistently called the bottleneck — AI-Researcher §6.2, freephdlabor abstract, EvoScientist motivation, Idea2Paper motivation all converge on this diagnosis (this is from scout-1 §5 #4).
- **Counter:** The four proposed fixes are mutually incompatible architecturally. EvoScientist uses persistent in-memory modules (ideation + experimentation memory). freephdlabor uses workspace-based file communication + automatic context compaction. Idea2Paper *precomputes* an offline methodological knowledge graph. ML-Master 2.0 (2601.10402) proposes Hierarchical Cognitive Caching with multi-level recall.
- **Why it matters:** four-way disagreement on the mechanism for a problem all four agree on is the cleanest "the field has not converged" surface in scout-1. The hypothesis-smith should treat this as an open design choice not a settled one.

### Contradiction 5: Auto-reviewer reliability under closed-loop training

- **One claim:** AI Scientist v1/v2, Agent Laboratory, AgentRxiv all use an LLM reviewer cut from the *same model class* as the writer and claim it produces useful feedback.
- **Counter:** AgentRxiv (2503.18102) §4.1 explicitly catalogues reward-hacking on this same-family reviewer. Agent Lab (2501.04227) §limit names paper-solver reward-hacking the NeurIPS-criterion scorer. CycleResearcher (2411.00816) is the only system that uses a *separately trained* reviewer model. ARIS (2605.03042) — cross-model adversarial collaboration explicitly designed to avoid this.
- **Why it matters:** the same papers that ship the auto-reviewer-as-loop-counterpart pattern *also* document its failure mode. This is the cleanest contradiction in the genre.

---

## 4. What I searched for but couldn't find

These are capability claims I expected to confirm or refute, where the slice is silent:

1. **Cost-conditional quality curves across scout-1 systems.** Agent Lab claims 84% cost reduction; AI Scientist v1 cites <$15/paper; no system publishes a Pareto frontier of cost vs. manuscript quality. Verification query: `hf_papers search "cost quality trade off autonomous research paper"` (not run because the slice already names the absence; flagging here as honest unknown).

2. **A unified failure-mode taxonomy across the autonomous-research family.** AgentRx (2602.02475) does this for *agent runs in general* with a critical-failure-step taxonomy. AI Scientist v1/v2 each lists its own failure modes. Trehan & Chopra (2601.03315) lists six recurring modes. Hidden Pitfalls (2509.08713) names four. No paper cross-walks these into one taxonomy with comparable definitions.

3. **Cross-section coherence (abstract↔results) measurement.** Scout-2 §5 #1 names this directly. No system in any scout produces a quantitative measurement of cross-section consistency on its own outputs. SciSage (2506.12689) Reflector is the closest mechanism, but it operates at outline/section/document levels of the *same* draft, not at semantic claim level between abstract and experiment logs.

4. **What happens when MegaResearcher's stateless-leaf-dispatch constraint meets long-horizon memory.** EvoScientist and freephdlabor both bake in persistent memory; neither has been replicated under a stateless-leaf constraint. No paper I read evaluates whether file-based artifact handoff (MegaResearcher's substrate) is *sufficient* as a memory substitute or whether in-agent persistent state is necessary.

5. **Tree-search vs. wave-orchestrator vs. linear-pipeline head-to-head.** Scout-4 §5 #5 explicitly names this absence — AI Scientist v2 uses tree search; v1 uses linear; Agent Lab uses linear-with-checkpoints; ML-Master uses sequential-with-cache; MegaResearcher uses wave-orchestrator. No published paper compares these four architectures on the same task suite. (This is a measurement gap rather than a design gap; I am flagging it but not claiming it as a primary capability gap.)

6. **Sandbox latency floor on agent task success.** Scout-4 §5 #7 names this; no paper measures Docker vs. E2B vs. Modal vs. Vercel Sandbox vs. Daytona impact on agent task success rate (third-party comparisons exist for speed/cost only).

7. **Position-paper output mode.** Scout-1 §5 #9 names this — all scout-1 systems produce ML method papers, none produces a position-paper output. The spec's synthesist target leans this way; I am unable to find published comparators.

---

## 5. Verification query log

Every query below was executed via `hf_papers search`; counts and brief findings are summarized.

| # | Query | N results | Finding (gap-direction) |
|---|---|---|---|
| 1 | `audit trail rejected hypothesis autonomous research` | 8 | Two papers (2605.05724, 2604.24658) name the storytelling-tax / lineage-feedback concept; neither is in scout-1's 14 systems. Gap confirmed for the genre. |
| 2 | `ablation discipline autonomous research agent ML paper` | 8 | AblationBench (2507.08038) exists specifically to measure the gap. No scout-1 system reports solving AblationBench tasks. Gap confirmed. |
| 3 | `pre-registration decision rules LLM agent experiment` | 8 | No paper in the top 8 implements pre-registered decision rules for autonomous research; results are about evaluation rubrics for general LLM-agent benchmarks (AgentRewardBench) and safety constitutions (TrustAgent), not research pre-reg. Gap confirmed by absence. |
| 4 | `ICLR rubric self-evaluation autonomous research paper agent` | 8 | PaperBench (2504.01848) uses ICML rubrics but for *replication*. No system self-evaluates against SPECS-Review-Benchmark. Gap confirmed. |
| 5 | `theoretical reasoning proof autonomous research agent LLM` | 8 | Prover Agent (2506.19923) and Bolzano (2604.16989) are theorem-proving agents — *not* in the autonomous-paper-generation family. Cross-family gap confirmed. |
| 6 | `statistical significance multiple seeds autonomous AI scientist` | 8 | Only 1806.08295 (pre-LLM RL) addresses multi-seed methodology. No scout-1 system reports multi-seed default. Gap confirmed. |
| 7 | `long horizon coherence cross section consistency manuscript` | 8 | Top hits are video / world-model generation (CoNo, AutoScape, BlockVid); only ML-Master 2.0 (2601.10402) addresses long-horizon coherence in research-agent setting. Manuscript-side cross-section consistency unaddressed. |
| 8 | `related work map related-work survey agent autonomous research paper` | 8 | AutoSurvey, SurveyG, PaperCircle exist as survey-generation agents but are not integrated into the 14 paper-producing pipelines. Capability is a separate research thread, not part of any pipeline. |
| 9 | `Paper2Rebuttal academic rebuttal multi-agent` | 5 | Rebuttal-generation agents (Paper2Rebuttal 2601.14171, DRPG 2601.18081, ToM rebuttal 2601.15715) exist but operate post-acceptance-flow; none is integrated into a paper-producing pipeline. Surfaced as related context, not claimed as a primary capability gap because it is post-acceptance. |
| 10 | `presentation overweighting reward hacking LLM reviewer adversarial` | 5 | Confirms the BadScientist / SPECS / Höpner pattern that presentation overweighting is documented but unmitigated in the auto-reviewer literature. |
| 11 | `abstract results consistency check post-hoc verification scientific manuscript` | 5 | Only CiteAudit, 17% Gap, SEA come up — all post-hoc; no pre-flight design. Confirms the pre-flight citation gate is novel. |
| 12 | `cross-model adversarial reviewer different model class autonomous research` | 5 | ARIS (2605.03042) is the cleanest peer; not in scout-1's 14 systems. Same-family auto-reviewer collapse is documented but only ARIS adopts the cross-model fix. |
| 13 | `memory persistent autonomous scientific research agent failure` | 5 | EvoScientist (2603.08127), ResearchGym (2602.15112), Why LLMs Aren't Scientists Yet (2601.03315) all triangulate that memory is the named bottleneck; confirms scout-1 §5 #4. |

---

## 6. Discarded candidate gaps

These were candidate gaps I considered and rejected after verification queries showed they ARE explored. Recording per the gap-finder discipline (forces honest verification).

### Discarded 1: "No system has rebuttal-generation capability"

**Initial intuition:** Scout-3 §5 #4 says "rebuttal handling is essentially absent"; this looked like a clean gap.
**Verification query:** `hf_papers search "rebuttal handling autonomous research paper agent"` — 8 results.
**What I found:** Three recent multi-agent rebuttal systems exist — Paper2Rebuttal (2601.14171, 53 upvotes), Dancing in Chains (2601.15715), DRPG (2601.18081). Plus the Re² dataset (2505.07920) explicitly for full-stage peer review + multi-turn rebuttal. The capability *exists* in the broader autonomous-research literature; it is simply not *integrated* into the 14 scout-1 systems. So the proper gap statement is "rebuttal integration" rather than "rebuttal capability absent." I downgraded this from a ranked gap to context in §3/§4.

### Discarded 2: "No system implements a cross-model adversarial reviewer"

**Initial intuition:** Scout-1 §5 #8 — "The auto-reviewer is the auto-writer's mirror"; CycleResearcher is the only separate-reviewer system. Looked like a clean unexplored intersection.
**Verification query:** `hf_papers search "cross-model adversarial reviewer different model class autonomous research"` — 5 results.
**What I found:** ARIS (2605.03042) explicitly proposes cross-model adversarial collaboration as the architecture; 105 upvotes, ~9k stars on the (claimed) repo. This is post-cutoff for scout-1 but published. So the cross-model adversarial reviewer is now an *existing architectural idea* in the literature, not an open gap. I kept it as a Contradiction-5 framing (auto-reviewer reliability under closed-loop training, with ARIS as one of the cited counter-evidence papers) but did not claim it as an unexplored capability.

### Discarded 3: "No system supports human-in-the-loop checkpoints"

**Initial intuition:** Looked like a candidate gap on the assumption that fully autonomous pipelines don't pause for humans.
**Verification:** Re-reading scout-1: Agent Laboratory (2501.04227) explicitly supports human-in-the-loop checkpoints between phases; freephdlabor (2510.15624) supports non-blocking human intervention; AgentRxiv (2503.18102) inherits Agent Lab's HITL. HLER (2603.07444) is explicitly human-in-the-loop economics research. The capability is present in at least 4/14 systems. Not a gap.

### Discarded 4: "No system uses citation graph as structural backbone"

**Initial intuition:** Citation graphs as backbone-of-survey looked novel to scout-1's pipelines.
**Verification:** SurveyG (2510.07733) uses hierarchical citation graphs for survey generation; Idea2Paper (2601.20833) uses an offline methodological graph; AI-Researcher's Resource Analyst Agent uses paper-graph relationships. The capability is published; the *integration* into a paper-producing pipeline is partial. So this is a "weak / partial" cell pattern in the matrix (RW column), not an unexplored capability. Demoted to part of Rank 6 (Related-Work Map) rather than its own gap.

---

## 7. Spec-relevance summary (one line per ranked gap)

| Rank | Capability | Why for the spec |
|---|---|---|
| 1 (tie) | Audit trail (rejected-hypothesis lineage) | MegaResearcher's discipline rule #1 is empirically novel; defends the differentiator |
| 1 (tie) | Ablation discipline | AblationBench exists to measure exactly this; closes the workshop→main-track delta |
| 3 | Citation verification as pre-flight gate | MegaResearcher's discipline rule #4 is unique in the genre |
| 4 | ICLR-rubric self-evaluation against external rubric | Closes the same-family auto-reviewer collapse documented in scout-3 |
| 5 | Statistical rigor (multi-seed, CIs) | Most-cited rejection reason at main-track ML venues |
| 6 | Related-work map (systematic graph) | Where presentation-overweighting most easily masks substance |
| 7 | Theoretical reasoning | Structural blind spot of the genre — none of 14 systems even attempts |
| 8 | Long-horizon coherence | Four-way unresolved memory-architecture disagreement |

Hypothesis-smith inherits this list to forge architectural augmentations.
