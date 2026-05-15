# Augmenting MegaResearcher into an End-to-End Paper Pipeline

**A research-direction document, position-paper style.**

Run: `2026-05-12-0515-19bf96`
Spec: `docs/research/specs/2026-05-12-megaresearcher-paper-pipeline-spec.md`
Plan: `docs/research/plans/2026-05-12-megaresearcher-paper-pipeline-plan.md`
Novelty target: hypothesis (red-team critique loop fired for every hypothesis)
Surviving hypotheses: 3 (S1, S2, S3). Killed: 3 (S4, S5, S6).

---

## 1. Introduction

Autonomous-research systems can now write papers end-to-end. AI Scientist v1 (arXiv:2408.06292) ships a closed-loop ideate → experiment → write → review pipeline at <$15/paper. AI Scientist v2 (arXiv:2504.08066) replaced the hand-authored template with progressive agentic tree search and produced the first fully AI-generated paper accepted (after peer review) at an ICLR workshop. Agent Laboratory (arXiv:2501.04227), AgentRxiv (arXiv:2503.18102), AI-Researcher (arXiv:2505.18705), Dolphin (arXiv:2501.03916), EvoScientist (arXiv:2603.08127), freephdlabor (arXiv:2510.15624), CycleResearcher (arXiv:2411.00816), PaperOrchestra (arXiv:2604.05018), Jr. AI Scientist (arXiv:2511.04583), Curie (arXiv:2502.16069), and Baby-AIGS (arXiv:2411.11910) all extend this skeleton along one axis or another.

None of them produces a paper that clears the main-track-conference bar (ICLR / NeurIPS / ACL accept threshold). The workshop ceiling stands. The "Hidden Pitfalls of AI Scientist Systems" paper (Luo, Kasirzadeh, Shah, arXiv:2509.08713) catalogues four recurring failure modes — benchmark selection, data leakage, metric misuse, post-hoc bias. AgentRxiv §4.1 documents reward hacking on its paper-quality reward in the score-fabrication channel. Agent Laboratory's published §limitations names `paper-solver` reward-hacking the NeurIPS-criterion scorer. AI-Researcher §6.2 names "memory management challenges"; §6.3 names "LLM reviewers overvalue presentation over substance." BadScientist (arXiv:2510.18003) shows five non-length fabrication strategies achieving 49-82% acceptance on o3 / o4-mini / GPT-4.1 reviewers. AAAI-26's SPECS-Review-Benchmark (arXiv:2604.13940) is the first conference-scale paired-human-AI substrate; it does not report human-parity for AI reviewers on substance axes.

Three foundational negative results constrain what any architectural change can deliver. Huang et al. (arXiv:2310.01798) show intrinsic self-correction without an external signal is net-negative on reasoning. Zhang et al. (arXiv:2502.08788) evaluate 5 multi-agent debate (MAD) methods × 9 benchmarks × 4 foundation models and find most MAD gains evaporate vs compute-matched chain-of-thought + self-consistency baselines; the one configuration that survives is *heterogeneous-model* debate. Choi et al. (arXiv:2508.17536) test debate vs voting on 7 NLP benchmarks: majority voting beats centralized MAD on 7/7, and centralized debate is the worst configuration. Feedback Friction (arXiv:2506.11930) shows frontier models given near-oracle feedback "consistently fall short of the target accuracy" — the friction is intrinsic to the model's intent-to-update gap, capping any revision-loop architecture.

**The thesis of this document.** MegaResearcher's architecture — single-session orchestrator dispatching leaf workers in waves, file-based artifact passing, no nested dispatch, mandatory audit trail of killed hypotheses — admits architectural changes that target each of the three failure surfaces these negative results pin: (i) same-family reviewer self-bias (S1, the writer/reviewer cross-family split), (ii) length-bias as an exploit channel in LLM-as-judge scalar scoring (S2, the Bias Fitting wrapper), and (iii) the revision-loop ceiling at the structured-decision surface (S3, majority-vote-over-5 on enumerable paper-decisions, bypassing the friction surface entirely).

Across six gap candidates surfaced by feasibility-filtering, three survived a multi-round red-team critique loop. The surviving three are scoped directly: **S1 is a workshop-grade pilot** at the swarm's ≤$200 budget ceiling, **S2 is a forecast / transfer test** of a published debiaser into the AI-Scientist-family domain, and **S3 is borderline main-track** with a pre-registered +6 percentage-point practical floor on a 1080-binary-decision substrate. The three killed hypotheses (S4 audit-trail ledger, S5 citation pre-flight gate, S6 ablation-coverage checklist) contribute structured lessons to the audit trail (§8).

---

## 2. Related work (six-scout digest, organized by capability axis)

### 2a. End-to-end pipelines (scout-1, 14 systems detailed)

AI Scientist v1 / v2 (arXiv:2408.06292, arXiv:2504.08066) are the canonical baselines: monolithic-with-prompted-phases v1, then a dedicated Experiment Manager driving progressive agentic tree search plus a VLM critic in v2. Agent Laboratory (arXiv:2501.04227) introduces explicit named-role agents (PhD, Postdoc, ML Engineer, Professor) plus `mle-solver` and `paper-solver` modules. AgentRxiv (arXiv:2503.18102) layers a shared preprint store on top of Agent Laboratory for cross-lab iteration. AI-Researcher (arXiv:2505.18705) ships the most explicit roster (Knowledge Acquisition, Resource Analyst with Paper + Code sub-agents, divergent-convergent Idea Generator, Plan, Code, Advisor, Automated Documentation) and the Scientist-Bench evaluation framework. Dolphin (arXiv:2501.03916), Curie (arXiv:2502.16069), and Aviary (arXiv:2412.21154) focus on experiment execution rigor without manuscript output. CycleResearcher (arXiv:2411.00816), PaperOrchestra (arXiv:2604.05018), Idea2Paper (arXiv:2601.20833), Jr. AI Scientist (arXiv:2511.04583), and Baby-AIGS (arXiv:2411.11910) explore manuscript-side variations. ARIS (arXiv:2605.03042) is an adjacent research harness (not strictly AI-Scientist-family) whose §2.2 names cross-model adversarial collaboration as the recommended default and whose Appendix E proposes (as future work) a five-arm controlled benchmark for heterogeneous-vs-homogeneous critique. Coscientist (Boiko et al., Nature 2023) and Virtual Lab (Swanson/Zou, Nature 2025) are paywalled and not arXiv-resolvable; flagged out per discipline rule #4.

### 2b. Manuscript drafting and document-scale coherence (scout-2)

26 entries spanning long-context-coherence work, structured-outline drafting, and document-level revision. AI-Researcher §6.2 names "abstraction drift" — over-summarization losing fine-grained details across multi-turn drafting. UltraHorizon (cited in scout-6) names a related "in-context locking" failure on long-horizon agent runs. The two are not unified in the corpus; this is one of the open questions surfaced by the scouts.

### 2c. Automated peer review and paper-quality evaluation (scout-3, 28 entries)

OpenReviewer (arXiv:2412.11948, Llama-OpenReviewer-8B fine-tuned on 79k expert reviews), CycleResearcher / CycleReviewer (arXiv:2411.00816, MAE below human inter-reviewer disagreement, contested), DeepReview / DeepReviewer-14B (arXiv:2503.08569), ReviewerToo (arXiv:2510.08867), SEA (arXiv:2407.12857, three-module standardize / evaluate / analyse), FactReview (arXiv:2604.04074, claim-extraction + literature-positioning + execution-verification — directly attacks presentation-overweighting), AAAI-26 AI Review Pilot + SPECS (arXiv:2604.13940), AutoRev (arXiv:2505.14376, graph-of-paper for long context), Reviewer2 (arXiv:2402.10886, aspect → review prompt cascade). Scout-3 names **five known-untrustworthy proxies** that cannot stand alone in falsification criteria: single-shot LLM-as-judge review (beaten by title+abstract baseline per Höpner arXiv:2503.05712), GPT-4 self-report from authors (fails BadScientist arXiv:2510.18003), self-evaluation by same-family model in closed loop (the AI Scientist pattern), reviewer-reviewer overlap rate alone (gameable per arXiv:2509.19326), PeerRead-trained acceptance prediction (venue/year drift). Triangulation candidates (≥1 non-judge signal): CORE-Bench reproducibility checks (arXiv:2409.11363), CiteAudit citation-graph fidelity (arXiv:2602.23452), SPECS controlled-flaw injection, blinded human spot-check on a small sample.

### 2d. Experiment execution and verification (scout-4, 24 entries)

CORE-Bench reproducibility-from-code (arXiv:2409.11363), AblationBench (arXiv:2507.08038, AuthorAblation / ReviewerAblation with LMJudge), AbGen (arXiv:2507.13300, 1,500 expert-annotated ablation studies from 807 NLP papers, Cohen's Kappa 0.71-0.78 among ACL area chairs on Importance / Faithfulness / Soundness — the rigorous human-anchored substrate for ablation evaluation), MLR-Bench (arXiv:2505.19955, "coding agents produce unreliable experimental results" even with high writing scores), CiteME (arXiv:2407.12861, citation-attribution-per-excerpt), GSAP-NER (arXiv:2311.09860, NER for ML models and datasets).

### 2e. Multi-agent critique, debate, revision loops (scout-5, 18 entries)

Three foundational negative results anchor this scout: Huang et al. (arXiv:2310.01798), Zhang et al. (arXiv:2502.08788), Choi et al. (arXiv:2508.17536). Feedback Friction (arXiv:2506.11930) caps revision-loop hypotheses. Multi-agent debate (arXiv:2305.14325, Du et al., +7.8 pp Biographies factuality, +15.9 pp Chess move validity — the gains Choi et al. later decompose as ensembling-not-debate). Self-Rewarding LMs (arXiv:2401.10020, tokens 1092 → 2552 over 3 iterations). Meta-Rewarding (arXiv:2407.19594, AlpacaEval 2 LC 22.92 → 39.44 with length-control). Length-Controlled AlpacaEval / Dubois debiaser (arXiv:2404.04475, pairwise logistic regression on preference labels). ODIN (arXiv:2402.07319, length-as-primary-reward-hacking-axis in RLHF). Bias Fitting (arXiv:2505.12843, scalar-score non-linear non-pairwise length debiaser — the actual wrapper S2 tests). Self-Preference Bias in LLM-as-a-Judge (arXiv:2410.21819, evaluators favor outputs matching their own perplexity distribution). Universal Self-Consistency (USC, arXiv:2311.17311), Fine-Grained Self-Consistency (FSC, arXiv:2407.02056), Self-Consistency for Open-Ended Generations (arXiv:2307.06857), Scalable Best-of-N via Self-Certainty (arXiv:2502.18581), ModeX (arXiv:2601.02535), Hegelian Dialectic Multi-Agent Majority Voting (arXiv:2501.14917). Diversity of Thought (arXiv:2410.12853). Self-Preference / Pride and Prejudice (Xu et al. arXiv:2402.11436, formal self-bias definition). Liang et al. encouraging divergent thinking via MAD (arXiv:2305.19118, same-LLM-judge preference bias). Patel representational collapse (arXiv:2604.03809, effective rank 2.17/3.0 at N=3 = 27.7% sample-size reduction). AIMO 3 (arXiv:2603.27844, correlated errors limit effective sample size).

### 2f. Memory and state for long agent workflows (scout-6, 18 entries)

FS-Researcher (arXiv:2602.01566) — file-system-based dual-agent framework, persistent workspace, stateless agents — is the closest published precedent for the MegaResearcher architectural pattern. Git Context Controller (arXiv:2508.00031) maps to the audit-trail discipline via version-controlled context (COMMIT / BRANCH / MERGE semantics). A-MEM (arXiv:2502.12110), MIRIX (arXiv:2507.07957, six typed memory stores), AriGraph (arXiv:2407.04363), G-Memory (arXiv:2506.07398), GAM (arXiv:2511.18423), MemoryOS (arXiv:2506.06326), L2MAC (arXiv:2310.02003) are stateless-dispatch-compatible. KVFlow (arXiv:2507.07400), KVCOMM (arXiv:2510.12872), SCBench (arXiv:2412.10319), and RL-trained memory controllers (MemPO, DeltaMem, Mem-T, MemGen) are out-of-scope (require shared in-memory state or training). The Last Human-Written Paper (arXiv:2604.24658) names the "Storytelling Tax" of discarding rejected hypotheses and ships the Ara protocol (§2.2 typed dead_end nodes) + ARA Seal Level 1 deterministic schema-conformance credential (§5.2) — directly relevant to the killed S4 hypothesis (§8).

### 2g. Architectural precedents specifically named

- **FS-Researcher** (arXiv:2602.01566) — the closest stateless-dispatch + file-handoff precedent.
- **Git Context Controller** (arXiv:2508.00031) — versioned-context discipline analog.
- **ARIS** (arXiv:2605.03042) — adjacent research harness, §2.2 cross-model adversarial collaboration; Appendix E heterogeneous-vs-homogeneous controlled benchmark *as future work* (not run).
- **Ara protocol + ARA Seal** (arXiv:2604.24658) — typed dead_end nodes + deterministic schema-conformance credential; subsumes most of the killed S4 hypothesis.

---

## 3. Gap analysis

### 3a. Capability matrix across 14 AI-Scientist-family systems (from gap-finder-1)

The matrix is 14 systems × 11 capabilities = 154 cells, populated at 92.2%. Eleven capability columns: Manuscript Drafting, Peer-review loop, Experiment Execution, Statistical Rigor, Related-Work Map, Ablation Discipline, Citation Verification, Theoretical Reasoning, ICLR-Rubric Self-Evaluation, Audit Trail, Long-Horizon Coherence. Cells coded **S** (strong), **W** (present but documented broken), **A** (absent), **U** (unknown).

**Capability rankings (W+A counts across 14 systems):**

1. **Audit Trail** (W/A in 13/14). Only EvoScientist scores S; AgentRxiv keeps an *accepted-report* preprint store but no rejected-hypothesis lineage. Verification query confirmed only arXiv:2605.05724 (Auto Research with Specialist Agents) and arXiv:2604.24658 (The Last Human-Written Paper, Ara protocol) treat rejected-hypothesis lineage as first-class.
2. **Ablation Discipline** (W/A in 13/14). No system scores S. AblationBench (arXiv:2507.08038) exists *because* current agents are weak at this; AbGen (arXiv:2507.13300) is the rigorous human-anchored substrate.
3. **Citation Verification** (W/A in 13/14). AI Scientist v1 uses Semantic Scholar lookup as a *novelty* check post-hoc, not a pre-flight resolution gate. No system implements pre-flight verification.
4. **ICLR-Rubric Self-Evaluation** (W/A in 13/14). All present-but-broken: same-family auto-reviewer, presentation-biased rubric.
5. **Statistical Rigor** (W/A in 12/14). Only Curie (arXiv:2502.16069) ships an explicit rigor module.
6. **Theoretical Reasoning** (A in 14/14). Out of reach for current systems.
7. **Long-Horizon Coherence** (W/A in 12/14). EvoScientist (arXiv:2603.08127) and freephdlabor (arXiv:2510.15624) are the only systems with persistent memory + cross-run evolution semantics.

### 3b. Architectural-pattern matrix (from gap-finder-2)

15 generic multi-agent / memory patterns × 14 AI-Scientist-family systems + ARIS as the adjacent column. 12 architectural-pattern gaps surfaced (GAP-A1 through GAP-A12), 5 explicitly killed with empirical-ceiling citations (KILL-1 centralized debate, KILL-2 intrinsic self-correction, KILL-3 reward-hacking via unguarded LLM-judge, KILL-4 fine-tuning-required patterns, KILL-5 nested-dispatch patterns). The four high-plausibility gaps:

- **GAP-A1** — Heterogeneous-model debate not applied to any AI-Scientist-family pipeline (Zhang et al. surviving MAD configuration).
- **GAP-A2** — Majority voting over N independent draft candidates unused, despite Choi et al. (voting > centralized debate on 7/7).
- **GAP-A10** — No length-control wrapper on any AI-Scientist-family judge call (Meta-Rewarding / Bias Fitting / Length-Controlled AlpacaEval all unapplied).
- **GAP-A12** — No pre-registration of decision rules; Luo / Kasirzadeh / Shah (arXiv:2509.08713) name absence as a recurring failure mode.

### 3c. Feasibility-filtered shortlist (from gap-finder-3)

Five constraints applied (C1 leaf-worker-only no nested dispatch; C2 file-based artifact passing; C3 off-the-shelf APIs no training; C4 ≤$200 per eval-design replication; C5 testable without venue submission, ≥1 non-judge signal). Six candidates emerged with the attack order **S1 → S2 → S5 → S6 → S3 → S4**, with a pre-flagged precondition note that S2 is the LLM-judge-debias precondition for S3 / S4 if those depend on judge calls; "pause if S1 and S2 both die."

| # | Gap source | Candidate | Attack rank |
|---|---|---|---|
| S1 | GAP-A1 + Contradiction 5 | Heterogeneous-model writer / red-team split | 1 |
| S2 | GAP-A10 | Length-debias wrapper on LLM-as-judge calls | 2 |
| S3 | GAP-A2 + Rank-4 ICLR-rubric | Majority voting over N candidates on structured paper-decisions | 5 |
| S4 | GAP-A5 + GAP-A12 + Rank-1 audit-trail | Rejected-hypothesis ledger keyed to binary signals | 6 |
| S5 | Rank-3 citation pre-flight | Citation pre-flight verification worker | 3 |
| S6 | Rank-1 ablation + AblationBench | Ablation-coverage checklist worker | 4 |

After Phase 3-4 (six hypothesis-smith forges + red-team critique loop, multiple revisions), three survive (S1, S2, S3) and three are killed (S4, S5, S6). The audit trail of killed hypotheses appears in §8.

---

## 4. Proposed architecture — the augmented MegaResearcher

The three surviving hypotheses compose as labeled subsystems on top of MegaResearcher's existing wave-based orchestrator. None requires nested dispatch; none requires fine-tuning; each is expressible as a leaf-worker dispatch policy or a deterministic post-processor on existing worker output.

### Subsystem S1 — Cross-family writer / reviewer routing (workshop-grade pilot)

A dispatch-policy change: the orchestrator routes the writer worker and the reviewer worker to *different foundation-model families* (primary pair {anthropic-claude, openai-gpt}; cross-family judge from {google-gemini}). Provider symmetry is enforced — each provider serves in both writer and reviewer roles across the sweep (F4 capability-symmetry control). No new worker; one new artifact field (`provider`) on worker output. The smith's revision-2 §0 self-disclosure: **Part A at ≤$200 is a workshop-grade pilot of ARIS's Appendix E protocol on the SPECS substrate.** A main-track-grade version requires $400-700 per replication (Part B future-work flag).

### Subsystem S2 — Bias Fitting length-debiasing wrapper on judge calls (transfer test)

A deterministic post-processor on every red-team / synthesist LLM-as-judge scalar-score call. The wrapper is **Bias Fitting (arXiv:2505.12843)** — a lightweight length-encoded ResNet `model_f(len(y))` fit on a one-time ~150-manuscript calibration corpus, returning the length-attributable component of the raw judge score for subtraction. Drop-in replacement for the raw scalar score. The smith's revision-1 framing: this is a **transfer test** of a published debiaser into the AI-Scientist-family domain, not a claim that length-bias is the dominant exploit channel — BadScientist (arXiv:2510.18003) shows five non-length fabrication strategies achieving 49-82% acceptance, and S2 explicitly does not address those.

### Subsystem S3 — Majority-vote-over-5 on three structured decision-classes (borderline main-track)

A fan-out / aggregate pattern: at each of three pre-registered decision-points in the AI Scientist v2 pipeline, the orchestrator samples N=5 independent same-model candidates at T ∈ [0.7, 1.0] and aggregates via per-binary plurality vote. The three decision-classes have **externally-grounded enumerable candidate universes** drawn from published artifacts (no LM-derived candidate sets):

- **Baseline-list inclusion** — K_baseline = 30, candidate universe = published-leaderboard top-30 entries (GLUE / SuperGLUE / ImageNet-1k / COCO / WMT-EnDe, frozen Jan 1 2026).
- **Ablation-axis inclusion** — K_ablation = 6, candidate universe = pre-registered 6-axis taxonomy (architecture-component, training-data, training-objective, hyperparameter, evaluation-protocol, inference-procedure), with AbGen (arXiv:2507.13300) testmini-500 supplying expert-annotated reference ablation studies as ground truth.
- **Citation-attribution per excerpt** — K=10 per paper, CiteME-shape, ground truth = the actual cited paper title in the published version.

The aggregator is a deterministic file-fold (`numpy.sum(ballots, axis=0) >= 3`) — no LM-as-judge in the aggregator, no spectral clustering (vs ModeX), no log-probability weighting (vs SC-Open), no LLM consensus-picker (vs USC).

### Composition

The three subsystems are orthogonal at the architecture level:

- S1 changes *which provider* runs writer / reviewer (no aggregator, no fan-out).
- S2 changes *the post-processing* on a scalar judge score (no provider routing change, no fan-out).
- S3 changes *the fan-out / aggregate pattern* at three decision points (no provider routing change, no judge-score post-processing).

They can be deployed independently, evaluated in isolation, or composed. The composition order if all three deploy: S1's cross-family routing applies first to the writer / reviewer dispatch; S2's Bias Fitting wraps any LLM-as-judge scalar score regardless of which model produced it; S3's vote-of-5 aggregator runs at the three structured-decision points after S1's family choice and before any S2-wrapped scoring.

**Magnitude framing.** S1's pre-registered floor (≥0.05 absolute lift on SPECS issue-recall, primary {claude, gpt} pair, 22 perturbations, N=3 seeds) is workshop-grade at α≈0.13 statistical power — the smith and red-team both flagged that uncorrected-α=0.05 would require a realized lift ≥0.072 not predicted by Zhang et al.'s magnitude transfer. S2 is forecast / sign+significance (β_raw > 0, β_norm ≈ 0, β_raw − β_norm > 0); upper-bound on impact is bounded by the fraction of LLM-judge variance attributable to length (BadScientist suggests sub-dominant). S3 is borderline main-track: pre-registered ≥+6 percentage-point aggregate floor on 1080 binary decisions at p<0.001, after a 30% empirical Patel-derived discount for same-model representational collapse.

---

## 5. Hypotheses table

| ID | Augmentation | Mechanism (cited) | Predicted Δ | Falsifier (non-judge / deterministic component) | Named baseline | Magnitude framing |
|---|---|---|---|---|---|---|
| **S1** | Cross-family writer/reviewer routing, primary pair {claude, gpt}, cross-family judge from {gemini}; F4 capability-symmetry control | Zhang et al. arXiv:2502.08788 heterogeneous-MAD survival; Xu et al. arXiv:2402.11436 self-bias amplification; Liang et al. arXiv:2305.19118 same-LLM judge preference; Wataoka et al. arXiv:2410.21819 family-perplexity bias | ≥0.05 absolute lift on SPECS (arXiv:2604.13940) issue-recall, 22-perturbation human-consensus-valid subset, N=3 seeds, paired bootstrap p<0.05 uncorrected | F1: lift ≥0.05 AND p<0.05; F2': single-pair survives orientation drop; F3: substance-axis lift ≥ presentation-axis lift +0.03; F4: per-orientation lift within ±1 SE of mean | Stage-matched same-family 2-stage writer/reviewer pipeline | **Workshop-grade pilot** (Part A, $195); main-track version (Part B, $400-700) future-work |
| **S2** | Bias Fitting (arXiv:2505.12843) length-debiased post-processor on every LLM-as-judge scalar-score call; one-time calibration on ~150-manuscript corpus | Dubois arXiv:2404.04475 + ODIN arXiv:2402.07319 + Self-Preference Bias arXiv:2410.21819 (length-bias documented); Self-Rewarding LMs arXiv:2401.10020 + Meta-Rewarding arXiv:2407.19594 (verbosity exploit in training loops); Bias Fitting arXiv:2505.12843 (the scalar-score non-linear non-pairwise debiaser) | β_raw > 0 at one-sided α=0.05; β_norm not significantly different from 0 at α=0.10; β_raw − β_norm 95% CI excluding zero on positive side | F1: β_raw not >0 (graceful no-op survey result); F2: β_norm significantly >0; F3: debiased score correlates Spearman ρ>0.3 with any of 8 pre-registered proxies (5 surface + 3 BadScientist-inspired substantive); F4: AUROC on known-good vs known-bad drops >0.05. All four use deterministic regression coefficients on token counts (non-judge primary signal). | Un-wrapped AI-Scientist-family judge (B0); explicit "ignore length" prompt instruction (B1); Bias Fitting wrapper (B2); heterogeneous-model judge from S1 (B3, required if S1 also runs) | **Forecast / transfer test**; co-defense not primary defense (BadScientist channels are dominant). Configuration-dependent recommendation (apply when β_raw>0 in calibration; skip when β_raw≈0) |
| **S3** | Majority-vote-over-5 on three pre-registered structured decision-classes inside AI Scientist v2; per-binary-membership plurality fold against externally-grounded candidate universes | Choi et al. arXiv:2508.17536 (voting > debate 7/7 NLP); Patel arXiv:2604.03809 + AIMO 3 arXiv:2603.27844 (same-model representational collapse, empirically 27.7%-30.3% effective-N reduction at N=3); M3 bypass of Feedback Friction (arXiv:2506.11930) and intrinsic self-correction floor (arXiv:2310.01798) — voting does not pass through either friction surface | Aggregate per-binary-decision hit-rate Δ ≥ +6 pp over single-draft, 1080 binary decisions (12 papers × 30 baselines + 100 AbGen × 6 axes + 12 × 10 citation excerpts) at McNemar p<0.001 | F1: Δ_aggregate < +6 pp OR p ≥ 0.001; F2: baseline-list Δ < +5 pp (cleanest plurality); F3 dual: mean Hamming <0.20 (variance floor) OR |unanimous-rate(pos) − unanimous-rate(neg)| > 30 pp (modal-bias contrast); F4: any class Δ<0 narrows scope. All thresholds on deterministic per-binary-decision hit-rate against externally-grounded ground truth (no LM-judge in aggregator). | AI Scientist v2 single-draft (arXiv:2504.08066) at T=0.7; USC-shape (arXiv:2311.17311) and ModeX-shape (arXiv:2601.02535) aggregators on same N=5 fan-out; B-ablation: 5 sequential drafts last-draft-wins (controls "you just gave it 5× compute") | **Borderline main-track**; +6 floor = Choi 7-bench average (+4.86) × 2.0 scoping multiplier × 0.70 Patel discount = +6.79 → floor +6 (rounded down explicitly) |

---

## 6. Eval designs (per surviving hypothesis)

Eval-designer outputs in Phase 5 produced binding pre-registrations (decision rules, sample sizes, randomization seeds, stopping rules) for each surviving hypothesis. Full protocols live in the per-hypothesis eval-designer output.md files; summaries below.

### 6a. S1 — eval-designer-S1 ($192.60 budget, 264 cells, SPECS substrate)

- **Substrate.** SPECS-Review-Benchmark (HF: `ut-amrl/SPECS-Review-Benchmark`, arXiv:2604.13940) restricted to the 22-perturbation human-consensus-valid subset (A.9.4 Table 5: 5 Story + 6 Correctness + 5 Evaluations + 3 Presentation + 3 Significance).
- **Sample.** 22 perturbations × 1 primary pair × 2 orientations × 3 seeds × 2 conditions (treatment + same-family control) = **264 cells**. Pre-registered seeds [7, 13, 42]; scheduling-only seed 19; overflow seeds [101, 103, 107].
- **Baselines.** Three: (i) stage-matched same-family 2-stage pipeline, primary contrast; (ii) anti-baseline 1: same-family with extra rounds at matched token budget (tests Zhang's claim that role-diversity ≠ model-diversity); (iii) Gemini-judge primary + OpenAI-default judge robustness check on identical writer/reviewer transcripts.
- **Decision rules.** F1 (magnitude floor + significance), F2' (single-pair sufficiency), F3 (substance-axis prediction), F4 (capability-symmetry). α=0.05 pre-registered conventional threshold reported alongside α≈0.13 workshop-grade threshold (the design's actual power at the 0.05 magnitude floor).
- **Budget breakdown.** Writer + reviewer $119; cross-family judge $53; 5-paper calibration pilot $3; 15% buffer $20. Total $195 ≤ $200 ceiling; calibration-pilot abort gate at >10% cost deviation.

### 6b. S2 — eval-designer-S2 ($200 budget; DeepNLP/ICLR-2024 + AI-Scientist-v2 dual substrate)

- **Substrate.** Primary calibration corpus = `DeepNLP/ICLR-2024-Accepted-Papers` (~2000 accepted papers, N=150 stratified by tag and abstract-length quartile, 80/20 fit-validate split). Secondary calibration = N=75 AI-Scientist-v2-generated workshop manuscripts (substrate-transfer ablation). Held-out test set = 20 manuscripts × 3 verbosity variants (terse 0.7×, original 1.0×, verbose 1.4×) = 60 cells per (judge-model × wrapper-condition).
- **Paraphrase model.** `claude-haiku-4` (different family from any of the three judges: `gpt-5`, `claude-sonnet-4.5`, `gpt-4.1`). Substance-preservation gates: numerical-claim regex match (±1 absolute), citation-multiset equality, section-heading verbatim match. Up to 3 re-paraphrase retries before flagging the manuscript as bad-substrate.
- **Baselines.** B0 un-wrapped judge (primary contrast), B1 prompt-only "ignore length" instruction, B2 Bias Fitting wrapper (treatment), B3 heterogeneous-model judge from S1 (required if S1 also runs), optional B4 Dubois pairwise debiaser if MegaResearcher's red-team can be re-tooled for pairwise output.
- **Ablations.** A1 length-encoding feature (log-tokens vs raw tokens vs character-count). A2 fitting-model architecture (ResNet vs linear). A3 proxy-substitution sweep across the 8 F3 proxies. A4 calibration-corpus comparison (`model_f^ICLR` vs `model_f^AI`).
- **Decision rules.** F1 / F2 / F3 / F4 from smith-S2 §5. Non-judge primary signal: β_raw, β_norm, β_raw − β_norm, all 95% CIs — deterministic regression on token counts and judge-output scores. Single global seed `19bf96`; per-step sub-seeds derived as `hash("step_name:19bf96")`.
- **Budget.** Calibration $45 + fitting-model train $5 + held-out test $40 + paraphrase generation $60 + F4 set $20 + baselines $50 + LimitGen sanity check (optional) $15. Total ~$235; $35 over ceiling acknowledged. Eval-designer trim path: 4 → 3 verbosity variants brings under $200. Substrate cap of $200 confirmed achievable.

### 6c. S3 — eval-designer-S3 ($140 budget, 1080 decisions, 8 pre-registered outcome rows)

- **Substrate.** 12 manuscripts from canonical-leaderboard tasks (3 GLUE + 3 SuperGLUE + 2 ImageNet-1k + 2 COCO + 2 WMT-EnDe). AbGen testmini-500 (arXiv:2507.13300), 100 references sampled at seed=42 deterministic preprocessing. CiteME-shape (arXiv:2407.12861) excerpts, 10 per paper at seed=42.
- **Frozen pre-flight artifacts** (hash-locked into manifest.yaml, verified unchanged at result-reporting): `frozen-baselines.json` (12 × 30 leaderboard entries), `frozen-axes.json` (6-axis ablation taxonomy with extraction lexicons), `frozen-excerpts.json` (12 × 10), `frozen-seeds.json` (voter seeds {42, 99, 101, 137, 211}), `frozen-prompts.json`.
- **Sample.** 12 × 30 = 360 baseline-list binaries + 100 × 6 = 600 ablation-axis binaries + 12 × 10 = 120 citation-attribution decisions = **1080 binary decisions**. Power calculation: detects Δ ≥ +6 pp at α=0.001 with power ≥0.90 under reasonable correlation assumptions.
- **Baselines.** B0 single-draft T=0.7 (primary contrast); B1 vote-of-5 T=0.7 (treatment); B3 vote-of-5 T=1.0 (variance-floor at higher diversity, contrast with B1 isolates temperature-as-diversity-driver); B-trivial random at empirical class prior; B-ablation 5 sequential drafts last-draft-wins (5× compute control). N_baseline_USC (arXiv:2311.17311) and N_baseline_ModeX (arXiv:2601.02535) as alternative-aggregator baselines on the same N=5 fan-out.
- **Decision rules.** F1 dual (Δ_aggregate ≥ +6 pp AND p < 0.001), F2 (baseline-list Δ ≥ +5 pp), F3 dual (variance floor + modal-bias contrast), F4 (per-decision-class Δ<0 narrows scope). All thresholds on deterministic per-binary-decision hit-rate against externally-grounded ground truth.
- **Budget.** Stays well within $200 ceiling at $140.

---

## 7. Threats to validity

**Cross-hypothesis.**

- **LLM-judge over-weighting of presentation.** Scout-3 lists this as the #1 published failure mode of automated reviewers (AI-Researcher §6.3, FactReview arXiv:2604.04074, AAAI-26 SPECS). S1's primary measurement is SPECS issue-recall under a cross-family judge — vulnerable to the same presentation-overweighting in a different direction (Gemini may have a presentation prior different from OpenAI's). S2's F4 (signal-collapse check) and F3 (8-proxy substitution sweep) explicitly test for gaming-target migration. S3's three decision-classes have **deterministic externally-grounded ground truth** (string match against published leaderboard entries; lexicon-match against AbGen expert annotations; title-match against published bibliography), so S3 is the most insulated from this threat.
- **Sample-size limitations.** S1's 22 perturbations × N=3 seeds × 2 orientations gives paired-difference SE ≈0.044, putting the design's actual statistical power at α≈0.13 for the 0.05 magnitude floor. The smith and red-team both pre-registered this directly as workshop-grade. S2's calibration N=150 is small relative to Bias Fitting's published 10K-50K corpus — explicitly flagged as a methodological extrapolation. S3's 1080-binary-decision sample is large but produces a documented gap between statistical significance (McNemar can detect +2-3 pp at p<0.05) and practical significance (the pre-registered ≥+6 pp floor); this is exactly the dual-threshold the smith pre-registered to address.
- **ARIS publishes Appendix E first.** ARIS (arXiv:2605.03042) Appendix E specifies (as future work, not run) a five-arm controlled benchmark of heterogeneous-vs-homogeneous critique on "12+ paper drafts from publicly available preprints." If ARIS's authors run their own Appendix E before S1's pilot completes, S1's empirical-priority contribution collapses. The smith pre-registered this in §1 of hypothesis-smith-S1: "the residual contributions (SPECS substrate selection + F4 capability-symmetry control + AI-Scientist-pipeline-application) are workshop-paper level, not main-track level."
- **Budget cap forces workshop-grade compromise on S1.** A main-track-grade S1 version (3 provider pairs, all 5 SPECS axes, N=5+ seeds) would cost $400-700 per replication, busting the ≤$200 spec ceiling. The smith chose budget compliance with workshop-grade disclosure over budget overrun.
- **Meta-issue: no hypothesis on this run is main-track-confident.** S1 self-disclosed workshop-grade; S2 self-disclosed forecast / transfer test + co-defense not primary defense; S3 is "borderline main-track" at +6 pp with a 30% same-model-collapse discount. The accurate characterization of the run's output is **three workshop-or-better hypotheses with one (S3) clearing a defensible borderline-main-track bar**, not three main-track-confident hypotheses. This is the audit-trail-faithful framing.

**Hypothesis-specific.**

- **S1 risks.** Feedback Friction (arXiv:2506.11930) caps writer-side revision: even if F1-F4 pass (reviewer detects more issues under heterogeneous routing), the writer may fail to incorporate the corrections on revision, leaving end-to-end paper quality unchanged. Single-pair design cannot generalize to "heterogeneity" claim. Cross-family Gemini judge has its own biases (Self-Preference Bias arXiv:2410.21819). Same-family bias may be dominated by other paper-gen failure modes (AgentRxiv §4.1 attributes reward-hacking to scoring-based selection, not same-family identity). SPECS perturbations on AAAI-25 papers may have memorization contamination. Part A is workshop-grade; main-track Part B requires budget above the swarm's ≤$200 gating.
- **S2 risks.** Judge competence ceiling (Feedback Friction interaction); verbosity-injection contamination (paraphrase accidentally injecting substance, R2 substance-preservation gate); wrapper-shifts-gaming-target (F3 as risk, 8-proxy catalog is the artifact); generalization off the calibration set; the exploit was already absent in this judge (R5 graceful no-op survey result framing); BadScientist-dominance (R6 — even if F1-F4 all pass, field-impact magnitude is bounded by the fraction of LLM-judge variance attributable to length).
- **S3 risks.** Long-form / abstractive transfer penalty exceeds +6 floor (Patel collapse worsens on harder less-structured tasks). Same-model sampling at T ∈ [0.7, 1.0] produces correlated ballots (R2 — Patel's empirical norm at N=3; if F3 fires across temperature sweeps, motivates S1 heterogeneous fix). AbGen reference-annotation noise floor (Cohen's Kappa 0.71-0.78 = ~22-29% disagreement on annotations themselves) masks the signal on the ablation-axis class. USC / ModeX baselines may outperform per-binary-membership voting (R4 — head-to-head between three aggregation primitives is itself a contribution). AI Scientist v2's existing tree search may capture some voting effect through tree expansion (R5 — eval-designer reports both headline Δ and tree-expansion-controlled Δ). Statistical vs practical significance gap on 1080 trials (R6 — dual threshold pre-registered).

---

## 8. Audit trail of killed hypotheses (non-negotiable per discipline rule #1)

Three hypotheses were killed across the red-team critique loop (cap-3). Each contributes a structured lesson to the audit trail.

### 8a. S4 — Rejected-hypothesis ledger keyed to structured binary signals (killed at revision 2)

**Initial form.** A new artifact format (`rejected-hypotheses.jsonl`) keyed to binary signals from red-team verdicts (citation_resolves, ablation_present_in_table, baseline_in_comparison_table, magnitude_claim_has_citation, falsification_criterion_count_gte_3), with Reflexion-style (arXiv:2303.11366) verbal reflection attached to each killed hypothesis and a frozen-hash measurement protocol providing reproducibility. Predicted +25 pp on AblationBench AuthorAblation lesson-recovery rate. Targeted GF-1-Rank-1 audit-trail gap.

**Why red-team rejected.** Revision-2 deep-read of arXiv:2604.24658 (The Last Human-Written Paper, Liu et al.) verified that **3 of 4 contribution legs are substantially subsumed by published Ara protocol + ARA Seal Level 1** (132-star GitHub implementation): (i) schema-enforced scoping = Ara §2.2 typed dead_end nodes with `(hypothesis, failure mode, lesson)` payload; (ii) append-only ledger = Ara §3.1 P3 version-controlled exploration_tree.yaml; (iii) measurement protocol for audit-trail artifact = ARA Seal Level 1 §5.2 deterministic schema-conformance credential. Only the **deterministic-file-signal-trigger** leg remains genuinely novel — workshop-grade against the spec's main-track bar. Smith **self-recommended KILL after independent re-read** rather than enter revision-3.

**Revision attempted?** Yes, two revisions. Revision-1 narrowed the scope to "AI-Scientist-family pipelines specifically" after C-overlap critique; revision-2 confronted the Ara overlap directly and the smith voluntarily contracted the contribution to the trigger-axis-only delta, then explicitly recommended KILL.

**Lesson contributed (5 structured lessons, S4-L1 through S4-L5):**

- **S4-L1.** The audit-trail-as-architectural-discipline space is more contested than gap-finder-1's GF-1-Rank-1 assessed. Ara protocol + ARA Seal Level 1 publish the structured-schema + deterministic-measurement legs of the conjunction. Future work in this space should cite arXiv:2604.24658 as the **primary baseline, not as related work**, and target the deterministic-signal-trigger axis specifically (the leg Ara does not occupy via LLM event-routing). Head-to-head comparison against Ara's open-source Live Research Manager is required: same input traces, same dead_end node schema, deterministic-file-signal triggers vs LLM event-routing — measure trigger-precision and trigger-recall, not lesson-recovery.
- **S4-L2.** Gap-finder dispatch on this run should have surfaced arXiv:2604.24658 at §2.2 / §5.2 depth. Two red-team rounds were burned because the gap-finder's prior-art coverage was sub-depth on the closest peer. Future runs with main-track novelty target should require gap-finders to do `read_paper section=...` on the top-3 candidate-peer papers, not just title-and-abstract scans.
- **S4-L3.** Reflexion-magnitude-transfer is structurally weak as a quantitative prior for cross-hypothesis (not multi-trial-same-task) settings. The smith should not anchor magnitude predictions on Reflexion's task-success-rate deltas when the proposed mechanism operates on a different metric over a different object.
- **S4-L4.** For audit-trail measurement claims, the lesson keys' upstream LM-judge origin is not erased by SHA-256 hashing. The frozen-hash protocol provides reproducibility, not validity. Future audit-trail measurement work should ground keys in human-annotator labels when available, and explicitly weaken claims to "agreement with a frozen judge baseline" when not.
- **S4-L5.** Sample size + statistical-test choice must be pre-flighted before the magnitude is dropped. The S4 revision-1 dropped its +25 pp magnitude prediction in response to red-team I1, then deferred the magnitude floor to a power-driven MDE — but did not compute the MDE itself. The resulting floor effectively returned to ~+30 pp at N=20. Future hypotheses dropping a magnitude prediction must compute the resulting power-driven floor in the same revision pass.

### 8b. S5 — Citation pre-flight verification gate (killed at revision 2)

**Initial form.** A new leaf worker (citation-verifier) firing between hypothesis-smith and synthesist; every claimed citation runs through `hf_papers paper_details` + Crossref before any downstream worker reads the manuscript. Unresolvable citations are **autonomously dropped** (not flagged). Six pre-registered falsifiers including F1 (substantive-claim drop rate D >15%), F4 (off-corpus rate three-way classification on Citegeist substrate), F5 (CiteME attribution regression), F6 (synthesist failure-to-emit). Targeted GF-1-Rank-3 citation-verification gap.

**Why red-team killed.** **All three Critical defects from prior rounds were addressed cleanly in revision-2** — CR1 ARIS (arXiv:2605.03042) engagement via direct §3.1 / §3.2 read; CR2 F4 deterministic three-way classification using the same resolver as the gate; CR3 PaperWrite-Bench / PaperRecon (arXiv:2604.01128) adopted as primary substrate. The Important objections (F1 GSAP-NER grounding, F1 Component C inconsistency, n=10 sub-significance, BibAgent) all addressed. **KILL is not because the revision failed.** KILL is because the revision succeeded so thoroughly that the smith openly characterized the contribution as "workshop-paper magnitude on the architectural delta plus first published D measurement on PaperWrite-Bench." That self-characterization is below the spec's main-track bar (lines 9, 25). S6 precedent applied consistently. The three-axis architectural delta against ARIS (block vs advise; deterministic vs LLM-judge; existence-only vs bundled) reduces to (a) three lines of code, (b) a resolver-vs-judge trade with substantial false-negative cost on conference-only / book-chapter / non-arxiv papers, and (c) a strict-subset of ARIS's bundled audit which is operationally a *limitation* rather than an architectural advance.

**Revision attempted?** Yes, two revisions. Revision-1 added BibAgent + LimitGen substrate considerations; revision-2 contracted to "workshop-paper magnitude" explicitly.

**Lesson contributed.** Three-axis ARIS-delta (block vs advise; deterministic-resolver vs LLM-judge; existence-only vs bundled-audit) + PaperWrite-Bench D measurement + GSAP-NER-grounded F1 taxonomy (arXiv:2311.09860) + Citegeist three-way F4 classification (arXiv:2503.23229) constitute a **concrete starting point for future workshop-bar work on citation hardening in AI-Scientist-family pipelines.** The architectural axis (autonomous-block vs advisory) is genuine but its main-track magnitude is bounded: ARIS's choice not to autonomously block is a *deliberate design choice ARIS rejected* (context-appropriateness axis requires human judgment), not an oversight ARIS missed. The deterministic-resolver-vs-LLM-judge trade admits substantive false-negatives on conference-only / book-chapter / non-arxiv-indexed papers; ARIS's web-access LLM-judge handles these. A future workshop-bar submission should reposition the contribution as "deployable pattern for stateless-leaf architectures with empirically-bounded D measurement," not as a main-track architectural advance.

### 8c. S6 — Ablation-coverage checklist worker (killed at revision 1)

**Initial form.** A new leaf worker (ablation-coverage-checker) reading the eval-designer's protocol and writing a coverage matrix against AblationBench-style (arXiv:2507.08038) coverage. Revision-1 reframed the mechanism from "promote rubric to first-class prompt" (which is LM-Planner itself) to "cross-wave file-handoff of named missing-ablation IDs through MegaResearcher's stateless-leaf-dispatch architecture." Predicted +3 F1pp on AblationBench AuthorAblation F1@5. Targeted GF-1-Rank-1-tie ablation-discipline gap.

**Why red-team killed.** The reframe from "rubric promotion" to "file-handoff substrate" does not survive the diagnostic arm the smith himself added: the smith conceded in R3 that if the file-handoff and in-context-concatenation arms produce indistinguishable F1@5, the mechanism-collapse re-materializes at the architectural level — and **there is no published prior establishing why file-handoff should differ from in-context concatenation when the receiving worker is also an LM call.** AblationBench §4.2 already measured LM-Planner with full paper-text + rubric-shaped prompt. The smith further conceded in §0 that "+3 F1pp is not a main-track-conference primary contribution by itself" and recommended the synthesist position S6 as scaffolding. **A hypothesis whose author asks the synthesist to position it as scaffolding should be killed and flagged as future-work**, not run through Phase 5 to produce an eval-designer document. Reinforced by three new defects revision-1 introduced: (i) a second statistic mis-quote (38% on AuthorAblation is recall@5 not F1@5 per Table 4; max LM-Planner F1@5 is 0.31), (ii) magnitude derivation chains three speculative multiplicands (0.50 flag precision × 0.60 incorporation × headroom) yielding an unanchored "+3 F1pp," (iii) Feedback Friction qualitative-direction import does not survive the source paper's explicit scope disclaimer (§3.1: "another LLM to evaluate more subjective tasks like instruction following or translation could lead to issues like reward hacking and unreliable assessments" — ablation generation is exactly such a task).

**Revision attempted?** One revision. After revision-1's KILL verdict, the smith recommended the same path S4 later took.

**Lesson contributed.** The underlying gap (no AI-Scientist-family pipeline integrates external rubric via cross-wave file handoff) is substantive and substrate exists (AblationBench / AbGen), **but the architectural distinction does not survive a diagnostic arm without a published prior establishing why file-handoff differs from in-context concatenation.** A future submission must either (i) find published prior establishing the file-handoff-vs-in-context distinction as material, or (ii) reposition as a measurement of the architectural-equivalence claim itself (which is a workshop-grade contribution at best). The +3 F1pp threshold sits inside AblationBench LMJudge's ~26%-disagreement-with-humans noise floor; the AbGen human-eval substrate (Cohen's Kappa 0.71-0.78) is the stronger ground truth for future work in this space, but its 1-5 Likert protocol does not directly map to the F1@5 metric AblationBench measures.

---

## 9. Escalations

**None.** No hypothesis was escalated to the user during the run. All three killed hypotheses (S4, S5, S6) terminated cleanly within the cap-3 revision limit — S6 at revision 1, S5 at revision 2, S4 at revision 2 with smith self-recommended KILL after independent re-read of arXiv:2604.24658. The orchestrator's strategic-guidance note on the S6 precedent ("an explicit workshop-grade self-characterization is the kill cue, not the approve cue when the spec's bar is main-track") was applied consistently to S5 and S4 without user adjudication. The audit trail in §8 substitutes for an escalation log; future runs with hypotheses contracting to workshop-grade against a main-track spec bar should expect the same pattern.

---

## 10. What we did NOT explore (YAGNI fence reflection)

The spec's Out-of-scope list (`docs/research/specs/2026-05-12-megaresearcher-paper-pipeline-spec.md` §Out of scope) is explicitly mirrored here. Each item is something this run **deliberately did not address** — these are not omissions, they are design choices the spec locked in. What would change if the user extended scope is noted per-item.

- **Implementing the augmentation.** This run produces a research-direction document, not code. No worker wrote implementation code; the synthesist composes existing worker outputs. *Extension path:* a follow-up `implementing-research-direction` plan would dispatch implementer workers to write the S1 dispatch-policy change, the S2 Bias Fitting wrapper, and the S3 vote-of-5 aggregator as actual MegaResearcher leaf-worker code or orchestrator-skill changes.
- **Running the eval designs.** Eval-designer produced protocols (264 cells for S1, 60 + 75 + 20 cells for S2, 1080 binary decisions for S3). No worker on this run ran experiments. *Extension path:* a separate execution run with budget approval (≈$535 cumulative across S1+S2+S3) would produce empirical results.
- **Domain-specific paper-quality criteria.** "What makes a biology paper good vs an ML paper good vs an HCI paper good" was explicitly excluded — the bar throughout is ML-conference-style rigor. *Extension path:* a domain-specific follow-up would require domain-expert input (the swarm cannot generate this from literature alone).
- **Publishing logistics.** Venue selection, LaTeX templates, page limits, camera-ready prep, IP / authorship / ethics of AI-authored papers are out of scope. *Extension path:* a separate research conversation worth having later but distinct from architectural-change work.
- **Paywalled-only literature.** Coscientist (Boiko et al., Nature 2023) and Virtual Lab (Swanson / Zou, Nature 2025) are paywalled and not arXiv-resolvable; per discipline rule #4 ("citations resolve or do not exist"), they were flagged and not relied on.
- **Training new models / fine-tuning.** Architecture changes only, off-the-shelf model APIs. S1 cross-family routing, S2 Bias Fitting lightweight ResNet fit (≤10K params on CPU), S3 same-model temperature sampling — all inference-time, no fine-tuning.
- **Top-tier / best-paper / oral-acceptance bar.** Main-track-accept bar was the ceiling. S1 self-disclosed workshop-grade; S2 is forecast / co-defense; S3 is borderline main-track at +6 pp. None claims top-tier.
- **Changes to MegaResearcher's existing workers in this swarm run.** The bundled six dispatched as-is. The three surviving hypotheses are *proposed* new dispatch policies / wrappers / aggregators, not workers used by *this* run.
- **Cost / pricing analysis of the proposed augmentations.** Token cost is noted as a constraint when relevant to feasibility (≤$200 per replication ceiling), but cost-optimization research is not the target.
- **Non-LLM-era systems** (older expert systems, symbolic AI scientific discovery like AM / Eurisko). Excluded to keep the scope tractable; modern LLM-era systems only (2023-2026).

---

## 11. Recommended next actions

**Invest first in S3 (majority-vote-over-5 on structured paper-decisions).** S3 is the only surviving hypothesis that clears a defensible borderline-main-track bar within the spec's ≤$200 budget gate, and is the most architecturally insulated from the LLM-judge over-weighting threat (its three decision-classes have deterministic externally-grounded ground truth from published leaderboards, AbGen expert annotations, and CiteME-published bibliographies — no LM-as-judge in the aggregator). The eval-designer's 1080-binary-decision substrate with dual statistical (p<0.001) + practical (Δ ≥+6 pp) thresholds is the cleanest pre-registration on the run.

**The smallest meaningful experiment.** Run S3's baseline-list-class arm only (12 papers × 30 leaderboard binaries × N=5 voters × 2 conditions (single-draft B0 + vote-of-5 B1) ≈ 360 binary decisions). This isolates the **cleanest plurality test** (externally-grounded leaderboard membership, deterministic string match for ground truth, no LM-extraction step) and is the F2 falsifier on its own. Estimated cost ≈$45 (a third of S3's full $140 budget). If F2 (baseline-list Δ ≥ +5 pp) passes, the structural-plurality mechanism transfers to paper-gen on the cleanest surface and the full S3 protocol is justified. If F2 fails, the same-model representational-collapse (Patel arXiv:2604.03809) erases the structural-plurality scoping benefit even on the cleanest surface, and S3 is falsified at minimum cost.

**Second priority: S2 (Bias Fitting wrapper).** S2 is the cheapest among published reviewer-exploit fixes ($45 calibration + $5 fit + per-call wrapper at zero incremental judge cost). The F1 graceful-no-op survey result (β_raw ≈ 0 in baseline calibration) is itself publishable as a *configuration-dependent* finding ("apply when β_raw > 0; skip when β_raw ≈ 0"). Run the calibration corpus and the F1 baseline regression first ($50) before committing to the full $235 design.

**Third priority: S1 (cross-family writer/reviewer).** S1's workshop-grade pilot ($195) is in scope but produces a workshop-grade result. If ARIS's authors publish their Appendix E benchmark before S1's pilot completes, the empirical-priority contribution collapses to a residual workshop paper. The smith's Part B sweep ($400-700) requires budget approval above the spec's ≤$200 gate; recommend deferring S1 until a future run with explicit budget approval for the multi-pair Part B sweep, or running S1 only as the first-public-report of ARIS's Appendix E protocol on the SPECS substrate (workshop-grade contribution explicitly).

**What would unlock the next research question.** A successful S3 result on the 1080-decision substrate establishes per-binary-membership voting against externally-grounded candidate universes as a deployable pattern. The natural next question is whether the *combination* of S1 (heterogeneous-model sampling for the 5 voters) and S3 (structured-decision aggregation) breaks the Patel representational-collapse ceiling that bounds S3's same-model floor at +6 pp. This composition was flagged out-of-scope for the current run (S3 §6 "Heterogeneous-vs-homogeneous-model sampling. S3 uses same-model. The orthogonal ablation draws the 5 candidates from 5 different foundation models. This is a separate hypothesis (S1) and should not be collapsed into S3."). A follow-up swarm could explicitly target the S1×S3 composition as a single hypothesis.

**On null results.** If S3's F1 fails, S2's F1 trips as a graceful no-op, and S1's pilot returns ARIS-Appendix-E-equivalent results, the swarm's output is a **structured negative result on three architectural changes that the literature suggested should help.** That is itself a contribution to the AI-Scientist-family pipeline literature and worth writing up — the audit-trail discipline of this run already documents what was tried and why it failed.

---

## 12. Run metadata

- **Run ID.** `2026-05-12-0515-19bf96`
- **Started.** 2026-05-12T05:15:00Z
- **Phases.** 6 (literature-scout, gap-finder, hypothesis-smith, red-team, eval-designer, synthesist)
- **Max parallel workers.** 4
- **Total worker invocations** (across all phases, including retries): 21 unique workers (6 scout + 3 gap-finder + 6 hypothesis-smith + 6 red-team + 3 eval-designer + 1 synthesist) with retry counts: hypothesis-smith-S4 ×3, hypothesis-smith-S3 ×2, hypothesis-smith-S5 ×2, hypothesis-smith-S1 ×2, hypothesis-smith-S6 ×1.
- **Total verified citations.** ~141 distinct arXiv-resolvable papers across Phase 1; per-phase resolution rate 100% (every cited paper resolved via `hf_papers paper_details` or `hf_papers read_paper`).
- **Hypotheses dispatched.** 6 candidates (S1-S6) from the feasibility-filtered shortlist.
- **Surviving hypotheses.** 3 (S1, S2, S3).
- **Killed hypotheses.** 3 (S4 at rev-2 smith self-recommended; S5 at rev-2 red-team KILL; S6 at rev-1 red-team KILL).
- **Escalations to user.** 0.
- **Synthesist output word count.** ≈6500 (~10-11 pages at standard density, under the 12-page ceiling per spec success criteria).

---

## 13. Sources (deduplicated across all worker outputs)

All arXiv IDs below were verified resolvable via `hf_papers paper_details` (or `hf_papers read_paper` for verbatim-quote checks) during the run. Per MegaResearcher discipline rule #4, papers not resolving against `hf_papers` do not exist for purposes of this output.

**End-to-end pipelines.** arXiv:2408.06292 (AI Scientist v1), arXiv:2504.08066 (AI Scientist v2), arXiv:2501.04227 (Agent Laboratory), arXiv:2503.18102 (AgentRxiv), arXiv:2505.18705 (AI-Researcher), arXiv:2501.03916 (Dolphin), arXiv:2603.08127 (EvoScientist), arXiv:2510.15624 (freephdlabor), arXiv:2511.04583 (Jr. AI Scientist), arXiv:2411.00816 (CycleResearcher), arXiv:2604.05018 (PaperOrchestra), arXiv:2601.20833 (Idea2Paper), arXiv:2502.16069 (Curie), arXiv:2411.11910 (Baby-AIGS), arXiv:2412.21154 (Aviary), arXiv:2509.08713 (Hidden Pitfalls of AI Scientist Systems), arXiv:2605.03042 (ARIS), arXiv:2605.05724 (Auto Research with Specialist Agents), arXiv:2604.13018 (Toward Autonomous Long-Horizon Engineering / AiScientist v2).

**Manuscript drafting + document coherence.** arXiv:2604.24658 (The Last Human-Written Paper / Ara protocol), arXiv:2604.01128 (PaperRecon / PaperWrite-Bench).

**Automated peer review + paper-quality evaluation.** arXiv:2412.11948 (OpenReviewer), arXiv:2503.08569 (DeepReview / DeepReviewer-14B), arXiv:2510.08867 (ReviewerToo), arXiv:2407.12857 (SEA), arXiv:2604.04074 (FactReview), arXiv:2604.13940 (AAAI-26 AI Review Pilot / SPECS-Review-Benchmark), arXiv:2505.14376 (AutoRev), arXiv:2402.10886 (Reviewer2), arXiv:2503.05712 (Höpner — title+abstract baseline), arXiv:2509.19326 (reviewer-overlap-gameable), arXiv:2510.18003 (BadScientist), arXiv:2505.17100 (RBD Reviewer Bias Detection), arXiv:2507.02694 (LimitGen).

**Experiment execution + verification.** arXiv:2409.11363 (CORE-Bench), arXiv:2507.08038 (AblationBench), arXiv:2507.13300 (AbGen), arXiv:2505.19955 (MLR-Bench), arXiv:2407.12861 (CiteME), arXiv:2602.23452 (CiteAudit), arXiv:2311.09860 (GSAP-NER), arXiv:2503.23229 (Citegeist), arXiv:2509.12282 (AIssistant).

**Multi-agent critique / debate / revision / voting.** arXiv:2310.01798 (Huang — LLMs Cannot Self-Correct Reasoning Yet), arXiv:2502.08788 (Zhang — Stop Overvaluing MAD), arXiv:2508.17536 (Choi — Debate or Vote), arXiv:2506.11930 (Feedback Friction), arXiv:2305.14325 (Du et al. multi-agent debate), arXiv:2401.10020 (Self-Rewarding LMs), arXiv:2407.19594 (Meta-Rewarding), arXiv:2404.04475 (Length-Controlled AlpacaEval / Dubois), arXiv:2402.07319 (ODIN), arXiv:2505.12843 (Bias Fitting), arXiv:2410.21819 (Self-Preference Bias in LLM-as-a-Judge), arXiv:2402.11436 (Xu — Pride and Prejudice / self-bias), arXiv:2305.19118 (Liang — encouraging divergent thinking via MAD), arXiv:2410.12853 (Diversity of Thought), arXiv:2311.17311 (USC), arXiv:2407.02056 (FSC), arXiv:2307.06857 (Self-Consistency for Open-Ended Generations), arXiv:2502.18581 (Scalable Best-of-N via Self-Certainty), arXiv:2601.02535 (ModeX), arXiv:2501.14917 (Hegelian Dialectic MAMV), arXiv:2604.03809 (Patel — Representational Collapse in Multi-Agent LLM Committees), arXiv:2603.27844 (AIMO 3), arXiv:2303.11366 (Reflexion), arXiv:2212.08073 (Constitutional AI), arXiv:2305.10601 (Tree of Thoughts), arXiv:2502.14767 (Tree-of-Debate persona pattern), arXiv:2501.05727 (SCRIT critic-trained variant).

**Memory and state for long workflows.** arXiv:2602.01566 (FS-Researcher), arXiv:2508.00031 (Git Context Controller), arXiv:2502.12110 (A-MEM), arXiv:2507.07957 (MIRIX), arXiv:2407.04363 (AriGraph), arXiv:2506.07398 (G-Memory), arXiv:2511.18423 (GAM), arXiv:2506.06326 (MemoryOS), arXiv:2310.02003 (L2MAC), arXiv:2507.07400 (KVFlow — out-of-scope), arXiv:2510.12872 (KVCOMM — out-of-scope), arXiv:2412.10319 (SCBench — out-of-scope).

**Architectural-precedent extras (named in worker outputs).** arXiv:2407.21783 (Hewitt et al. — different families have measurably different priors on scientific-claim errors).

**Substrate / dataset citations.** `ut-amrl/SPECS-Review-Benchmark` (HF dataset, 5,556 downloads), `DeepNLP/ICLR-2024-Accepted-Papers` (HF dataset, ~2000 accepted papers), AbGen testmini-500 (CC-BY 4.0), paperswithcode.com leaderboard snapshots (CC-BY-SA 4.0).

**Systems flagged but unresolvable (per discipline rule #4, NOT cited as critical).** Coscientist (Boiko et al., Nature 2023 — paywalled). Virtual Lab (Swanson / Zou, Nature 2025 — paywalled). Genesis-Flow (no system resolves to this name). Carl (Autoscience) and Zochi (Intology) — commercial, no arXiv entry. Older symbolic-AI systems (AM / Eurisko) — out of YAGNI scope.

---

*End of synthesis. Full Phase-1 through Phase-5 worker outputs are at `docs/research/runs/2026-05-12-0515-19bf96/<worker-id>/output.md`. The swarm-state audit log is at `docs/research/runs/2026-05-12-0515-19bf96/swarm-state.yaml`.*
