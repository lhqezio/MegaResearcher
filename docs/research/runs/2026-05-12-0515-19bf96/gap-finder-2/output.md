# gap-finder-2 — Architectural-pattern gaps

## Slice scope

Read in full:

- `docs/research/runs/2026-05-12-0515-19bf96/scout-5/output.md` — 15 generic
  multi-agent / critique / debate / revision patterns with named magnitudes,
  plus three foundational negative-result papers (Huang et al. 2310.01798,
  Zhang et al. 2502.08788, Choi et al. 2508.17536) and Feedback Friction
  (2506.11930).
- `docs/research/runs/2026-05-12-0515-19bf96/scout-6/output.md` — 18 memory
  systems with explicit stateless-dispatch compatibility tags.
- `docs/research/runs/2026-05-12-0515-19bf96/scout-1/output.md` — 19+ end-to-end
  paper-generation systems (AI Scientist v1/v2, Agent Laboratory, AgentRxiv,
  AI-Researcher, Dolphin, EvoScientist, freephdlabor, Jr. AI Scientist,
  Idea2Paper, CycleResearcher, PaperOrchestra, Aviary, Curie, Baby-AIGS,
  Coscientist, Virtual Lab) plus three critical-commentary papers.

Out-of-scope (flagged): KV-cache-level memory (KVFlow 2507.07400, KVCOMM
2510.12872, SCBench 2412.10319), RL-trained memory controllers (MemPO,
DeltaMem, Mem-T, MemGen). These violate the stateless-dispatch + no-fine-tune
constraint and are listed only in §(d).

---

## (a) Pattern-by-system matrix

Rows: generic patterns from scout-5 / scout-6 that pass the stateless-dispatch
+ file-handoff constraint. Columns: paper-generation systems from scout-1.
Cells: `applied` / `partial` / `absent` with primary citation.

Legend:
- `applied`: the system explicitly implements the pattern and the paper
  documents it
- `partial`: a thin or domain-limited variant is present (e.g., applied to a
  sub-phase only, or to ideation only, or with a homogeneous-model fallback)
- `absent`: the system's published architecture does not document the pattern

| Pattern (generic source) | AI Scientist v1 (2408.06292) | AI Scientist v2 (2504.08066) | Agent Laboratory (2501.04227) | AgentRxiv (2503.18102) | AI-Researcher (2505.18705) | Dolphin (2501.03916) | EvoScientist (2603.08127) | freephdlabor (2510.15624) | Jr. AI Scientist (2511.04583) | CycleResearcher (2411.00816) | PaperOrchestra (2604.05018) | Idea2Paper (2601.20833) | Curie (2502.16069) | Baby-AIGS (2411.11910) | ARIS (2605.03042) — adjacent, not strictly AI-Scientist family |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Heterogeneous-model debate** (2502.08788 §findings; 2410.12853) — different foundation models on the debate sides | absent (same-model auto-reviewer per §2c circular evaluation) | absent (auto-reviewer is same model class per 2504.08066) | absent (LLM-reward identical-class reviewer; 2501.04227 §limitations) | absent (inherits Agent Lab roster; 2503.18102) | absent (Advisor + reviewer are same class; 2505.18705 §6) | absent | absent | absent | absent (mentor-fed analyst + reviewer; same class; 2511.04583) | **partial** — CycleReviewer is a *trained* separate model but still single model class (2411.00816) | absent | absent | absent | absent | **applied** — explicit cross-model adversarial roster (2605.03042 keywords: "cross-model adversarial collaboration") |
| **Majority voting / N-sample ensembling over candidate drafts/hypotheses** (2508.17536 Section 4 voting beats debate on 7/7) | absent | absent (tree search is best-first, not majority over completed drafts) | absent | absent | **partial** — divergent-convergent ideation scores 5 directions then selects, but no plurality vote (2505.18705 ideation §) | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| **Tree-of-thought search over revision states** (2305.10601) | absent | **applied for experiment configs only** (2504.08066 progressive agentic tree search over experiments) | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| **Reflexion-style external-reward episodic reflection memory** (2303.11366 +22% AlfWorld, +20% HotpotQA) | absent (no episodic store) | absent (tree-node logs but no verbal reflection across nodes) | absent | **partial** — AgentRxiv shares *successful* reports across labs but the spec is "publish" not "verbally reflect on failure" (2503.18102 §3) | absent (§6.2 explicitly: no external memory) | **partial** — closed feedback loop but not Reflexion's reflection schema (2501.03916) | **applied** — explicit experimentation memory + failure-mode entries (2603.08127) | **partial** — persistent memory + non-blocking intervention but not verbal-reflection-on-failure schema (2510.15624) | absent | absent | absent | absent | absent | absent | absent |
| **Constitutional / principle-guided critique** (2212.08073 — short list of human-readable principles) | absent (NeurIPS rubric is the only "constitution" used; not a Constitutional-AI critique loop) | absent (rubric-driven, not principle-list critique) | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| **Self-rewarding iterative DPO / meta-rewarding loop** (2401.10020, 2407.19594) | absent (no preference-pair generation; YAGNI no-fine-tune anyway) | absent | absent | absent | absent | absent | absent | absent | absent | **partial** — CycleReviewer is *trained* on review preferences, but not iterative self-rewarding DPO (2411.00816) | absent | absent | absent | absent | absent |
| **Tree-of-Debate persona pattern** (2502.14767 — competing-paper personas argue novelty in a tree) | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| **Falsification-first revision** (2411.11910 Baby-AIGS FalsificationAgent) | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | **applied** — FalsificationAgent is the central pitch (2411.11910) | **partial** — adversarial collaboration is structurally similar but not Popper-style explicit counter-experiment (2605.03042) |
| **A-MEM-style auto-linking + memory evolution** (2502.12110) | absent | absent | absent | absent | absent (the failure §6.2 documents is precisely the absence of this) | absent | **partial** — persistent memory but no explicit Zettelkasten-style auto-link + memory-evolution pass (2603.08127) | **partial** — automatic context compaction is mentioned but not linked-note rewriting (2510.15624 abstract) | absent | absent | absent | absent | absent | absent | absent |
| **AriGraph-style entity knowledge graph** (2407.04363 — semantic + episodic dual retrieval) | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | **applied to a different objective** — methodological knowledge graph used at ideation, not for entity tracking across waves (2601.20833) | absent | absent | absent |
| **Git Context Controller-style versioned context with branch/merge** (2508.00031) | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | **partial** — ARIS has "persistent research wiki" + "claim auditing" but no published COMMIT/BRANCH semantics (2605.03042 keywords) |
| **FS-Researcher dual-agent persistent workspace** (2602.01566) | **partial** — workspace exists but is a flat output dir, not a structured note workspace (2408.06292) | **partial** — tree-of-experiments lives in workspace but no Context-Builder agent (2504.08066) | **partial** — working dir + summarized blobs passed agent-to-agent (2501.04227) | **partial** — shared preprint store across runs (2503.18102) | absent (Docker workspace exists but §6.2 documents the over-summarization failure) | absent | absent | **applied** — explicit workspace-based communication + persistent memory (2510.15624 abstract — closest published match) | absent | absent | absent | absent | absent | absent | **applied** — "persistent research wiki" as the central artifact (2605.03042) |
| **MIRIX-style typed memory taxonomy** (2507.07957 — six typed memory stores) | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| **Meta-Rewarding length-control wrapper** (2407.19594 — explicit defense against verbosity exploit in LLM-as-judge) | absent (reviewer scores manuscripts with no length-control protection — concrete reward-hacking vector) | absent | absent (paper-solver reward-hacks NeurIPS-criterion scorer per 2501.04227 §limitations — exactly the failure mode length-control would address) | absent (§4.1 documents reward hacking on paper-quality reward) | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent |
| **Pre-registration of decision rules** (2509.08713 names absence as a failure mode; SCRIT 2501.05727 critic-trained variant adjacent) | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | absent | **partial** — explicit pre/post conditions on experiment records (2502.16069) | **partial** — falsification attempts attached to hypothesis records (2411.11910) | absent |

Notes on the matrix:

- "absent" for AI Scientist v1/v2's automated reviewer is *not* a vibes claim
  — it is explicitly documented as a limitation in scout-1 entry 2a (the
  reviewer is the same model class as the writer, "circular evaluation",
  cited in 2503.18102 §4.1 as the cause of reward hacking on paper quality).
- ARIS (2605.03042) is listed in a separate column because it is an
  **adjacent research-harness paper**, not a member of the AI-Scientist /
  Agent-Laboratory / AI-Researcher family scout-1 enumerates. Its existence
  *partially* covers the heterogeneous-model-debate intersection but only as
  a generic research harness; no scout-1 system in the AI-Scientist family
  has adopted it as of the cited dates.

---

## (b) Ranked list of unexplored intersections

Each entry has: (i) pattern name + generic-source citation, (ii) generic-task
empirical magnitude + citation, (iii) paper-gen omission + citation showing
absence, (iv) transfer-plausibility score grounded in the
Huang/Zhang/Choi/Feedback-Friction ceilings, (v) verification query
recording how I confirmed the gap.

### GAP-A1 (HIGH plausibility) — Heterogeneous-model debate has not been applied to any AI-Scientist-family paper-generation system

- **Pattern:** Heterogeneous-model multi-agent debate / critique, where the
  writer and the reviewer (or two debaters) are drawn from *different
  foundation-model families* rather than two instances of the same model.
- **Type:** Unexplored intersection.
- **Generic-task magnitude:** Zhang et al. "Stop Overvaluing MAD"
  (arXiv:2502.08788) — heterogeneous-model debate is **the one MAD
  configuration that survives meta-evaluation** across 5 MAD methods × 9
  benchmarks × 4 foundation models when matched against CoT + self-consistency
  baselines. Diversity-of-Thought (arXiv:2410.12853) reports diverse-model MAD
  "achieving superior performance on benchmarks compared to individual
  powerful models" on mathematical reasoning. Liang et al. (arXiv:2305.19118)
  shows even an asymmetric configuration ("GPT-3.5-Turbo with MAD can
  surpass GPT-4 on Common MT") lets the weaker debater plus a different
  judge beat a stronger single-pass model.
- **Paper-gen omission:** Scout-1 §2c entry on AI Scientist v1
  (arXiv:2408.06292) and v2 (arXiv:2504.08066) explicitly document that the
  automated reviewer is the same model class as the writer — circular
  evaluation. AgentRxiv §4.1 (arXiv:2503.18102) documents reward hacking on
  the paper-quality reward as a *direct consequence* of writer/reviewer
  sharing the same model class. CycleResearcher (arXiv:2411.00816) trains a
  *separate* reviewer but still uses a single base model class for both
  sides. Among AI Scientist v1/v2, Agent Laboratory, AgentRxiv,
  AI-Researcher, Dolphin, EvoScientist, freephdlabor, Jr. AI Scientist,
  PaperOrchestra, Idea2Paper, Curie, and Baby-AIGS, **zero systems publish a
  heterogeneous-foundation-model writer/reviewer split**.
- **Adjacent partial coverage:** ARIS (arXiv:2605.03042) advertises
  "cross-model adversarial collaboration" but is a separate research harness,
  not a member of the AI-Scientist-paper-generation family, and is dated May
  2026 — i.e., recent enough that it has not been adopted by the established
  systems.
- **Transfer plausibility to paper-gen: HIGH.** This is the *only* MAD
  configuration that survives Zhang et al.'s meta-eval; Choi et al.
  (arXiv:2508.17536) confirms voting > centralized debate but the
  heterogeneous-model effect is orthogonal to the voting-vs-debate axis (it
  is about *who* the debaters are, not how they aggregate). Feedback
  Friction (arXiv:2506.11930) caps the upside of any revision loop, but
  heterogeneous-model feedback is one of the few mechanisms not directly
  ruled out by the friction floor — the friction is intra-model intent-to-
  update gap, and different-model critique introduces external pressure
  that intrinsic self-correction (2310.01798) cannot.
- **Verification query:** `hf_papers search "heterogeneous models different
  LLMs debate reviewer writer paper" limit=8`. Results: of 8 returned, none
  applied heterogeneous-model debate to autonomous paper-generation systems
  in the AI-Scientist family; the closest hit (2410.12853 Diversity of
  Thought) is on math reasoning, not paper generation. Second query:
  `"heterogeneous model debate autonomous paper generation" limit=10`
  surfaced PaperOrchestra (2604.05018), AI Scientist v2, PaperArena —
  none describe heterogeneous-foundation-model writer/reviewer roster. Third
  cross-check: ARIS (arXiv:2605.03042) is the only paper-gen-adjacent system
  to advertise cross-model adversarial collaboration; published May 2026.

### GAP-A2 (HIGH plausibility) — Majority voting over N independent draft candidates is unused in paper-gen, despite voting beating centralized debate on 7/7 benchmarks

- **Pattern:** Generate N independent candidate drafts/hypotheses/related-work
  sections and select by majority vote / ensemble agreement rather than
  iterating one draft through a centralized critique loop.
- **Type:** Unexplored intersection + missing baseline.
- **Generic-task magnitude:** Choi et al. "Debate or Vote?"
  (arXiv:2508.17536) — Qwen2.5-7B majority voting **0.7691 avg vs
  centralized MAD T=2 0.6551 avg** across 7 benchmarks. Voting alone beats
  every debate variant; centralized MAD (the *closest analog to
  MegaResearcher's orchestrator-aggregates-workers pattern*) is the **worst**
  configuration. On Arithmetic specifically: single 0.8140 → MAD T=2 0.7600
  → voting **0.9900**. Same pattern with Llama-3.1-8B (voting 0.7242 vs
  debate variants 0.61–0.70).
- **Paper-gen omission:** No system in scout-1 generates N parallel candidate
  drafts and selects by plurality. AI-Researcher (arXiv:2505.18705)
  divergent-convergent ideation generates 5 directions and scores them
  against three criteria, but selection is single-judge LLM scoring — not
  plurality vote across independently-sampled scorers. AI Scientist v2
  (arXiv:2504.08066) uses best-first tree search over experiments, not over
  drafts; the search expands the most-promising node rather than aggregating
  across siblings. Agent Laboratory (arXiv:2501.04227), Jr. AI Scientist
  (arXiv:2511.04583), Dolphin (arXiv:2501.03916), Idea2Paper
  (arXiv:2601.20833), Curie (arXiv:2502.16069), PaperOrchestra
  (arXiv:2604.05018): single-draft pipelines.
- **Transfer plausibility to paper-gen: MEDIUM-HIGH.** The
  Choi-et-al. ceiling shows voting > centralized debate for short-answer
  tasks. The transfer to long-form prose is *not* directly demonstrated by
  Choi — they note voting requires a plurality to exist. Long-form quality
  has no obvious plurality unless the vote is over a structured surface
  (e.g., which baseline to compare against; which framing of the related-
  work cluster). Scout-5 open question 1 calls this out explicitly: "no
  paper in this scout's pull evaluates voting-vs-debate for long-form
  research-text targets." Plausibility is medium-high because the structured
  decisions inside a paper draft (baseline selection, claim-vs-not-claim,
  related-work cluster choice) *do* have plurality structure even if the
  paragraph text does not.
- **Verification query:** `hf_papers search "N parallel hypotheses voting
  selection AI scientist autonomous research" limit=8`. Results: AI-Researcher,
  AI Scientist v2, Medical AI Scientist, AgentRxiv, More-You-Automate, HLER,
  Confidence-Weighted Token Set Cover. **None** apply majority voting to
  paper-draft selection in the AI-Scientist family. The Hegelian-Dialectic
  paper (2501.14917) uses "Multi Agent Majority Voting" but only for *novelty
  assessment in ideation*, not for selecting among complete drafts. Confirms
  the gap.

### GAP-A3 (HIGH plausibility) — Tree-of-thought search over revision states is applied to experiments but not to manuscript text revision

- **Pattern:** ToT-style branching search (b=5 in Yao et al.) over partial
  revision states with self-evaluation pruning.
- **Type:** Unexplored intersection.
- **Generic-task magnitude:** Yao et al. (arXiv:2305.10601) — ToT b=5 hits
  **74%** on Game of 24 vs CoT 4.0% and CoT-SC k=100 9.0%. ~10x lift over
  best CoT baseline. THOUGHTSCULPT (arXiv:2404.05966) extends this with MCTS
  + revision actions to "Story Outline Improvement" and reports
  state-of-the-art across tasks (qualitative; scout-5 entry).
- **Paper-gen omission:** AI Scientist v2 (arXiv:2504.08066) applies tree
  search to *experiment configurations*, not to manuscript drafting — the
  "Experiment Manager controls tree search" while the paper-writer is a
  single-pass agent (scout-1 §2a). No system in scout-1 expands branches of
  candidate revisions to a paragraph/section/related-work-claim and prunes
  by self-evaluation.
- **Transfer plausibility: MEDIUM.** Yao et al. note ToT fails when the LLM
  cannot reliably evaluate partial states. Manuscript-section quality is
  exactly such a domain — the "sure / maybe / impossible" prompt is hard to
  ground for "is this paragraph publication-grade?" Feedback Friction
  (arXiv:2506.11930) further caps how much any revision-state search can
  deliver. ToT-for-revision is therefore plausible-but-not-guaranteed; the
  hypothesis-smith would need to constrain the partial-state evaluator to a
  domain where the LLM *can* judge partial quality (e.g., factuality of an
  individual citation, presence of a baseline, presence of an ablation).
- **Verification query:** `hf_papers search "tree of thoughts search
  hypothesis generation autonomous scientist" limit=10`. Results: IRIS
  (2504.16728) uses MCTS for *ideation*; FlowPIE (2603.29557) uses MCTS for
  *idea evolution*; THOUGHTSCULPT (2404.05966) for story-outline
  improvement; A*-Thought (2505.24550) for reasoning compression; **none
  for manuscript-revision** in the AI-Scientist family. Gap confirmed.

### GAP-A4 (MEDIUM plausibility) — Constitutional / principle-guided critique is unused in paper-gen despite the field's well-documented reward-hacking on paper-quality rewards

- **Pattern:** Constitutional AI's short-list-of-natural-language-principles
  critique loop (Bai et al. 2212.08073), with the critic+reviser pair
  substituting for thousands of human-labeled examples.
- **Type:** Unexplored intersection.
- **Generic-task magnitude:** Qualitative — Anthropic's RL-CAI matches or
  exceeds RLHF on the harmlessness/helpfulness Pareto without human
  harmlessness labels. Anthropic uses ~10 principles. Direct Principle
  Feedback (arXiv:2402.07896) demonstrates this works for behavior-change.
- **Paper-gen omission:** Every AI-Scientist-family system uses
  *rubric-driven* critique (NeurIPS form in v1/v2, internal LLM-reward in
  Agent Lab's paper-solver, internal LLM-reward in CycleReviewer). The
  rubric is a closed scoring schema, not a constitutional principle list
  that the critic interprets at write-time. The documented failure of these
  rubric-driven reviewers — reward-hacking on paper-quality (AgentRxiv §4.1)
  and reward-hacking on NeurIPS-criterion scorer (Agent Lab §limitations) —
  is the canonical signal that the *form* of the critique signal is
  vulnerable. No AI-Scientist-family system has substituted a constitutional
  list (e.g., "the paper must not claim a baseline it has not run", "the
  paper must report every result from §results", "the related-work must cite
  at least one paper from the last 12 months") for the rubric.
- **Transfer plausibility: MEDIUM.** Constitutional AI works for
  *behavioral* targets (Bai et al. note this explicitly — "reasoning /
  factuality gains aren't claimed"). The paper-quality target is partially
  behavioral (claim-without-evidence, length blow-up, plagiarism) and
  partially correctness-based (factuality of cited results). Constitutional
  critique is plausibly effective on the behavioral half. Feedback Friction
  caps the rest.
- **Verification query:** `hf_papers search "constitutional principle-guided
  critique AI scientist research paper revision" limit=8`. Results: ICAI
  (2406.06560), Specific-vs-General-Principles (2310.13798), Constitutional
  AI (2212.08073), Constitution-Evaluation in medical (2411.10168),
  Evolving-Interpretable-Constitutions for multi-agent coordination
  (2602.00755), Pink-Elephant DPF (2402.07896), MetaCritique (2401.04518).
  **None** apply constitutional principles to the autonomous-paper-
  generation reviewer. Closest is medical-interview critique. Gap confirmed.

### GAP-A5 (HIGH plausibility) — Reflexion-style external-reward verbal episodic-memory reflection is absent from every AI-Scientist family system on the *prose* path

- **Pattern:** Reflexion (arXiv:2303.11366) — agent receives an external
  reward signal (binary success/failure or environment trace), verbally
  reflects on the failure, stores the reflection in episodic memory, retries.
- **Type:** Unexplored intersection.
- **Generic-task magnitude:** +22% AlfWorld, +20% HotpotQA, +11% HumanEval
  over strong ReAct/CoT baselines. Crucially: CoT-only and CoT(GT)-only
  *fail to probabilistically improve on any task* without the external
  binary signal — reflection is **inert** without an external reward (scout-5
  entry 2).
- **Paper-gen omission:** AI-Scientist family's *code* path can use unit
  tests / execution traces as the binary signal (Agent Laboratory's
  mle-solver does this), but **no system uses an analogous external reward
  on the prose path** — i.e., no system stores a verbal reflection on "why
  the last related-work draft was rejected" keyed to a binary signal
  (acceptance / rejection / reviewer-flagged-claim). EvoScientist
  (arXiv:2603.08127) has "experimentation memory + failure-mode entries"
  but the failure modes are code-execution failures, not prose-quality
  failures. AI-Researcher §6.2 (arXiv:2505.18705) explicitly says there is
  no external memory system.
- **Transfer plausibility: HIGH on the structured-prose path, LOW on the
  free-prose path.** Reflexion's mechanism requires a binary signal —
  Huang et al. (arXiv:2310.01798) show that without one, self-correction is
  net-negative. Structured paper-decisions ("did the related-work cite
  X?", "did the ablation table contain Y?", "did the limitation section
  flag Z?") have natural binary signals. Free-prose quality does not.
  Reflexion-for-paper-gen is therefore high-plausibility on the structured-
  decision sub-problem, low-plausibility on the free-prose sub-problem.
- **Verification query:** `hf_papers search "reflexion episodic memory verbal
  RL autonomous scientist failure" limit=8`. Results: Reflexion (2303.11366),
  Why-LLMs-Aren't-Scientists (2601.03315) — note: confirms scout-1 has
  no analog applied; AriGraph (2407.04363) — episodic but for TextWorld
  zero-shot. **None** apply Reflexion's verbal-reflection-on-rejection
  pattern to autonomous paper-generation. Gap confirmed.

### GAP-A6 (MEDIUM plausibility) — Git-Context-Controller versioned context with COMMIT/BRANCH/MERGE is unused in any AI-Scientist family system despite the audit-trail discipline rule

- **Pattern:** Wu (arXiv:2508.00031) — Git-style versioned context tree with
  COMMIT, BRANCH, MERGE, CONTEXT operations on agent state; lets agents
  checkpoint milestones and branch off alternative plans, then merge back
  the survivors.
- **Type:** Unexplored intersection.
- **Generic-task magnitude:** Reported on long-horizon coding tasks and
  self-replication — "superior performance in long-horizon tasks" (qualitative
  from scout-6 entry 9). No clean "+X EM" number; this is the
  audit-trail-naturalness argument, not a benchmark-leaderboard claim.
- **Paper-gen omission:** No AI-Scientist family system documents COMMIT /
  BRANCH / MERGE semantics on its workspace. AgentRxiv (arXiv:2503.18102)
  has a *shared preprint store* across runs but the audit-trail at the level
  of "which hypothesis branches were rejected, with reason" is not
  first-class — scout-1 open question 1 confirms: "no system in this family
  has an explicit rejected-hypothesis audit trail." ARIS (arXiv:2605.03042)
  has "persistent research wiki" + "claim auditing" but does not publish
  COMMIT/BRANCH semantics. The MegaResearcher discipline rule
  ("rejected/killed hypothesis appears in the synthesist's final document
  with the lesson it contributes") maps almost exactly onto Git Context
  Controller's BRANCH semantics — and no published paper-gen system has
  this.
- **Transfer plausibility: MEDIUM.** The pattern is naturally compatible
  with stateless dispatch + file handoff (Wu's store is git-backed; workers
  address state by revision). The catch flagged by scout-6 entry 9: "merge
  conflicts on context are not solved well; relies on agent discipline to
  write meaningful commit messages." For paper-gen the merge surface is
  smaller than for code (sections compose; commit messages can be templated
  per worker role) so the friction is plausibly lower. But the pattern adds
  storage and orchestration cost without a hard empirical magnitude attached
  in the paper-gen target, so plausibility is medium not high.
- **Verification query:** `hf_papers search "version controlled context git
  commit branch research agent paper" limit=10`. Results: Git Context
  Controller (2508.00031), SWE-Adept (2603.01327 — codebase analysis),
  Git-Theta (2306.04529 — model parameters), Learning-to-Commit (2603.26664
  — coding agents), PaperDebugger (2512.02589 — Overleaf in-editor; not the
  same pattern — uses LaTeX revision, not COMMIT/BRANCH on agent context),
  Spec-Kit-Agents (2604.05278 — context-grounding for coding). **No
  AI-Scientist-family system** uses COMMIT/BRANCH semantics on its
  paper-generation workspace. Gap confirmed.

### GAP-A7 (MEDIUM plausibility) — A-MEM-style auto-linking + memory evolution is not applied to research-pipeline workspaces

- **Pattern:** A-MEM (arXiv:2502.12110) — each new note (i) is given a
  contextual description + keywords + tags, (ii) is linked to related
  existing notes, (iii) can trigger updates to those linked notes ("memory
  evolution"). Dynamic-graph + symbolic links over a vector store. Compatible
  with stateless dispatch if the consolidation step is a named subagent.
- **Type:** Unexplored intersection — directly inverts the AI-Researcher
  §6.2 failure mode.
- **Generic-task magnitude:** Reported on LoCoMo (long-form conversation
  benchmark) — qualitative gains over baseline memory systems (scout-6
  entry 3, 880 GH stars). The mechanism *is* the failure-mode inverter:
  rewriting linked notes when new info arrives is exactly what AI-Researcher
  cannot do (its summaries get more abstract over waves).
- **Paper-gen omission:** AI-Researcher §6.2 explicitly: "no external memory
  management system" → "fine-grained details from early pipeline stages
  become increasingly difficult to access" (scout-6 §scope). EvoScientist
  has persistent memory but no Zettelkasten-style link + evolution
  (arXiv:2603.08127 — keyword list does not contain "linking" or "evolution"
  in the A-MEM sense). freephdlabor (arXiv:2510.15624) has automatic context
  compaction but compaction is the *failure mode* A-MEM avoids — A-MEM
  evolves the linked store rather than compacting away detail.
- **Transfer plausibility: MEDIUM.** A-MEM's eval does not measure detail-
  preservation (scout-6 open question 4 flags this — both the failure and
  the candidate fix are under-measured). Feedback Friction does not directly
  apply because A-MEM is about *what gets stored* not *what gets corrected*;
  the friction floor is orthogonal. Risk: scout-6 notes "link explosion
  under high-volume writes; evolution step is expensive and can rewrite
  useful detail into smoother prose (a softer version of the AI-Researcher
  abstraction-drift failure)" — i.e., the candidate fix has a documented
  related failure mode.
- **Verification query:** `hf_papers search "A-MEM zettelkasten memory linking
  research workflow autonomous" limit=8`. Results: A-MEM (2502.12110),
  NanoResearch (2605.10813 — "skill bank" + memory module but not Zettelkasten
  linking), Omni-SimpleMem (2604.01007 — autoresearch-discovered memory but
  not A-MEM linking), MemTool, MemR^3, AlabOS (autonomous labs not paper-gen),
  Towards-Autonomous-Memory-Agents, MemMA. **No AI-Scientist family system**
  applies A-MEM-style link + evolution to a paper-generation workspace. Gap
  confirmed.

### GAP-A8 (LOW-MEDIUM plausibility) — AriGraph-style entity knowledge graph (papers, hypotheses, baselines, datasets, results as nodes) is not applied to multi-wave research orchestration

- **Pattern:** AriGraph (arXiv:2407.04363) — semantic memory (entity-relation
  triples) + episodic memory (time-stamped events linked to same entities);
  associative retrieval: query → relevant subgraph → relevant episodes.
- **Type:** Unexplored intersection.
- **Generic-task magnitude:** Reports superior performance on TextWorld
  zero-shot planning (scout-6 entry 6; 166 GH stars). Not "+X EM" benchmark
  but a clean ablation study.
- **Paper-gen omission:** Idea2Paper (arXiv:2601.20833) maintains an
  *offline methodological knowledge graph* used at ideation time, but not a
  cross-wave entity graph of (papers, hypotheses, baselines, datasets,
  results) that the orchestrator's later workers can query. No other
  AI-Scientist family system documents such a graph.
- **Transfer plausibility: LOW-MEDIUM.** AriGraph's failure modes (entity
  resolution → duplicate nodes; contradictory updates poorly handled)
  hit paper-gen hard: "is this baseline the same as that baseline?",
  "does this dataset version match the cited result?" are exactly the
  duplicate-node and contradictory-update cases. Plausibility is lowered
  because the entity-resolution problem in research-paper space is itself
  unsolved.
- **Verification query:** `hf_papers search "knowledge graph entity persistent
  memory paper generation hypothesis baseline" limit=8`. Results: GAAMA
  (2603.27910), Graph-based-Agent-Memory survey (2602.05665), A-MEM,
  MemOS (2507.03724), GraphRAG-survey, PlugMem (2603.03296), RoMem temporal
  KG (2604.11544), GAM hierarchical graph (2604.12285). **None** apply
  entity-graph memory to autonomous-paper-generation. Gap confirmed.

### GAP-A9 (MEDIUM plausibility) — Tree-of-Debate persona pattern is unused in paper-gen despite being the closest published analog to scientific-text revision

- **Pattern:** Kargupta et al. (arXiv:2502.14767) — LLM personas representing
  competing papers debate their novelty claims in a tree. Forces each
  paper-persona to argue its own novelty independently.
- **Type:** Unexplored intersection / domain-transfer gap (the pattern works
  on scientific-comparative-analysis but has not been transferred to whole
  related-work generation).
- **Generic-task magnitude:** Qualitative — reports superiority over
  single-pass and flat-debate baselines on human-rated comparative-analysis
  quality (scout-5 entry 15). No "+X EM" number. 19 GH stars.
- **Paper-gen omission:** None of the AI-Scientist family systems use
  persona-debate for related-work generation. AI-Researcher §6.3 notes
  "LLM reviewers overvalue presentation over substance" — the persona-
  argument-for-own-novelty pattern is one of the few mechanisms that
  surfaces substantive comparison rather than presentation polish, and is
  unused.
- **Transfer plausibility: MEDIUM.** Scout-5 entry 15 flags that the pattern
  "requires that the comparison axis be defined up front (the 'personas');
  doesn't auto-discover the dimensions on which to compete." The autonomous-
  paper-gen setting would have to surface the personas without human input
  — plausibly tractable for a related-work section (each cited paper is a
  persona; the focal paper is a persona) but adds an extra wave.
- **Verification query:** `hf_papers search "tree-of-debate persona novelty
  scientific comparison related work generation" limit=8`. Results:
  Tree-of-Debate (2502.14767), RINoBench (2603.10303 — novelty *benchmark*,
  not generator), OpenNovelty (2601.01576 — novelty assessment agent),
  NovBench (2604.11543), Idea-Generation survey (2511.07448), Persona-prompt
  for online debate (2410.04239 — political not scientific), R-Debater
  (2512.24684 — retrieval-augmented debate, not persona-novelty). **No
  AI-Scientist family system** uses Tree-of-Debate's persona pattern. Gap
  confirmed.

### GAP-A10 (HIGH plausibility) — Meta-Rewarding's explicit length-control wrapper is absent from every paper-gen system that uses LLM-as-judge as a stopping criterion

- **Pattern:** Wu et al. (arXiv:2407.19594) — adds a meta-judge step plus
  length-control wrapper that defends against the verbosity exploit in
  LLM-as-Judge.
- **Type:** Missing baseline / untested assumption.
- **Generic-task magnitude:** AlpacaEval 2 LC win rate Llama-3-8B-Instruct
  22.9% → 39.4% over 4 iterations *with length-control*; without
  length-control, the model exploits the verbosity bias. Self-Rewarding
  baseline: 1092 → 2552 tokens average across 3 iterations *without*
  proportional quality gain. Length-Controlled AlpacaEval
  (arXiv:2404.04475) confirms regression-based length-debiasing is
  necessary for valid auto-eval.
- **Paper-gen omission:** AI Scientist v1/v2 reviewers score manuscripts
  with the NeurIPS rubric — no length-control wrapper documented (scout-1
  §2a). Agent Laboratory's paper-solver: "reward-hacks the NeurIPS-
  criterion scorer (the canonical example of audit-trail risk)" per
  scout-1; AgentRxiv §4.1 documents the same. The LaTeX-aesthetics failure
  mode (AgentRxiv §4) is closely related — the writer pads aesthetic
  surface to game the reviewer. None of the systems publish a length-
  control wrapper or a verbosity defense.
- **Transfer plausibility: HIGH.** This is the most concrete defendable
  gap. The failure mode (reward hacking on paper-quality reward via length
  / aesthetics) is *already documented* in two AI-Scientist systems
  (AgentRxiv §4.1, Agent Lab §limitations). The fix exists generically and
  is cheap to apply (a regression debiaser on top of any LLM-as-judge).
  Feedback Friction does not apply because the friction floor is about
  the writer; the defense is on the judge.
- **Verification query:** `hf_papers search "self-rewarding LLM judge length
  bias AI scientist paper writing" limit=10`. Results: ODIN (2402.07319),
  LLM-Evaluators-Recognize-Own (2404.13076), Reward-Modeling-Scientific-
  Writing (2601.11374 — *trained* reward model for scientific writing but
  no length-control wrapper), Igniting-Creative-Writing (2508.21476),
  Self-Preference-Bias (2410.21819), Process-Self-Rewarding (2503.03746
  — math reasoning, not paper writing), Length-Controlled-AlpacaEval
  (2404.04475), Pride-and-Prejudice (2402.11436), CALM (2410.02736). **No
  AI-Scientist family system** applies length-control debiasing to its
  auto-reviewer. Gap confirmed.

### GAP-A11 (MEDIUM-LOW plausibility) — MIRIX-style typed-memory taxonomy (Episodic / Semantic / Procedural / Resource / Vault) is absent from every paper-gen system

- **Pattern:** MIRIX (arXiv:2507.07957) — six typed memory stores
  (Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault) managed
  by six specialist agents with a central dispatcher. Compatible with
  stateless dispatch if the orchestrator dispatches the six in waves.
- **Type:** Unexplored intersection.
- **Generic-task magnitude:** Reports superior performance on
  ScreenshotVQA + LOCOMO (scout-6 entry 4; 3,542 GH stars). No clean
  benchmark delta extracted — qualitative.
- **Paper-gen omission:** AI-Researcher (arXiv:2505.18705) has a typed
  artifact format ("structured concept profiles") but does not split memory
  by *function* (episodic experiment logs vs semantic literature facts vs
  procedural pipeline templates). Scout-6 entry 4 explicitly states the
  MIRIX taxonomy "maps cleanly to research-pipeline roles (episodic =
  experiment logs, semantic = literature facts, procedural = pipeline
  templates)" but no system has done this map.
- **Transfer plausibility: MEDIUM-LOW.** MIRIX's documented failure mode
  is "cross-store consistency (an episodic fact and a semantic fact can
  disagree); high storage overhead." For paper-gen the cross-store
  consistency problem is acute (an experiment-log episodic fact and a
  related-work semantic claim can disagree on the same number). The
  pattern is plausibly useful but the consistency-management problem it
  introduces is not solved in the published literature.
- **Verification query:** Subsumed in the A-MEM + knowledge-graph searches
  above. None of the surfaced papers apply MIRIX-typed-memory to AI-Scientist
  systems. Also: `hf_papers search "FS-Researcher file system memory research
  workspace paper draft writing" limit=8` surfaces FS-Researcher (2602.01566),
  TTD-DR (2507.16075 — diffusion-style draft refinement, not typed memory),
  SciSage (2506.12689 — multi-agent survey gen, "reflect-when-you-write"
  but no typed-memory taxonomy). Gap confirmed.

### GAP-A12 (MEDIUM plausibility) — Pre-registration of decision rules has been *named as a missing pattern* but never *architecturally implemented* in any AI-Scientist family system

- **Pattern:** Luo, Kasirzadeh, Shah (arXiv:2509.08713) name pre-registration
  of decision rules as the architectural fix for benchmark-selection bias,
  data leakage, metric misuse, and post-hoc selection bias — but the paper
  itself does not propose an implementation. Curie (arXiv:2502.16069) does
  "explicit pre/post conditions on experiment records" — the closest
  partial implementation, but on the *experiment* surface not the *paper-
  claim* surface.
- **Type:** Untested assumption / missing baseline.
- **Generic-task magnitude:** Not a benchmark-magnitude pattern; this is an
  audit-trail-completeness pattern. No "+X EM" number applies. The empirical
  case is that the four failure modes are *measurable* and *recurrent*
  (2509.08713 §findings).
- **Paper-gen omission:** Baby-AIGS (arXiv:2411.11910) attaches
  falsification-attempt records to hypotheses, but no system pre-registers
  the decision rule for "what counts as acceptance of this hypothesis"
  before the experimental result is observed. Scout-1 open question 2
  confirms: "Pre-registration of decision rules is absent" — "AI-Scientist-
  style pipelines select metrics and thresholds during the experimentation
  phase, then write up only the favorable result."
- **Transfer plausibility: MEDIUM.** This is mostly an architectural-
  discipline pattern, not a benchmark-magnitude one. Feedback Friction and
  Choi-et-al. do not apply (they're about revision-loop ceilings; this is
  about a structural prevention). The MegaResearcher discipline rule #3
  ("Pre-registration of decision rules in eval-designer outputs") already
  bakes this in — the gap is that *no other system in the field has done
  this*, which is a differentiator. The plausibility cap is that the
  pattern is hard to verify is being followed without an external rubric.
- **Verification query:** `hf_papers search "audit trail rejected hypothesis
  falsification record autonomous discovery" limit=8`. Results:
  More-You-Automate (2509.08713), ARIS (2605.03042), Abduct-Act-Predict
  (2509.10401), Auto-Research-Recipes (2605.05724 — "evaluator-owned
  outcome" but not pre-registration), REFUTE (2502.19414), Baby-AIGS
  (2411.11910). **No AI-Scientist family system** implements
  pre-registration architecturally. Gap confirmed.

---

## (c) Skeptical analysis — patterns the negative-result ceilings likely render un-productive even if unapplied

The gap-finder's job is to kill bad-transfer ideas before the hypothesis-smith
burns tokens. Below are patterns surfaced by scout-5/scout-6 that look
attractive at first read but the empirical ceilings argue *against* applying.

### KILL-1: Centralized inter-agent debate as a quality-improver for prose drafts

**Why kill:** Choi et al. (arXiv:2508.17536) decompose MAD into
"majority-voting over N agents" and "inter-agent debate" components and find
that the debate channel adds little — voting alone wins on 7/7 benchmarks.
**Centralized MAD (which is closest to MegaResearcher's
orchestrator-aggregates-workers pattern) is the worst configuration**
(Qwen2.5-7B: 0.6551 avg, worse than single-agent 0.7205). Multi-round debate
(T=3, T=5) consistently *worse* than T=2 — extra rounds erode gains.

**Implication:** MegaResearcher's current critique-revision loop (orchestrator
fires red-team → hypothesis-smith revises → orchestrator fires red-team again,
cap 3) is structurally the *worst* configuration on the Choi-et-al. axis.
*Any* hypothesis the hypothesis-smith proposes that adds *more rounds* or
*more debate* on top of the current loop is fighting the empirical evidence.
The hypothesis-smith should not propose pattern variants that increase the
debate depth — they should propose variants that *replace* debate with
voting (GAP-A2) or *change the debaters* (GAP-A1, heterogeneous models).

### KILL-2: Pure intrinsic self-correction on the prose path

**Why kill:** Huang et al. (arXiv:2310.01798) — intrinsic self-correction
*decreases* accuracy across GSM8K, CommonSenseQA, HotpotQA when oracle-stop
is removed. Feedback Friction (arXiv:2506.11930) — even with *oracle-grade*
feedback, frontier models cannot fully incorporate it; "feedback friction
is intrinsic." Pride and Prejudice (arXiv:2402.11436) — self-refinement
amplifies self-bias. The triad converges: self-critique without an
external/different-model signal is net-negative.

**Implication:** MegaResearcher's red-team worker is currently the same
model class as the hypothesis-smith (scout-1 open question 8 names this as a
field-wide problem). Any hypothesis the hypothesis-smith proposes that
relies on the *same-model* red-team to detect substantive flaws is bounded
above by these three negative results. The hypothesis-smith should bias
toward heterogeneous-model critique (GAP-A1) or external-signal-keyed
reflection (GAP-A5 structured-decision path) over pure same-model intrinsic
self-correction.

### KILL-3: Length-extending revision loops with LLM-as-judge stopping criterion

**Why kill:** Self-Rewarding LMs (arXiv:2401.10020) go from 1092 → 2552
tokens across 3 iterations *without proportional quality gain* — the
length blow-up *is* the gain. Meta-Rewarding (arXiv:2407.19594) patches
this only with length-control wrappers. Length-Controlled AlpacaEval
(arXiv:2404.04475) confirms the regression debiaser is necessary for valid
auto-eval. AgentRxiv §4.1 documents reward-hacking on paper-quality reward
as the AI-Scientist instance of this.

**Implication:** Any MegaResearcher revision loop that uses LLM-as-judge
for stopping must apply GAP-A10 (length-control wrapper) as a precondition.
Without it, the loop converges to longer, worse text. The hypothesis-smith
should not propose a pure "add more LLM-as-judge revision passes" hypothesis
— that direction is dominated by the documented failure mode.

### KILL-4: A-MEM-style memory evolution with aggressive consolidation cadence

**Why kill:** Scout-6 entry 3 documents that A-MEM's evolution step "can
rewrite useful detail into smoother prose (a softer version of the
AI-Researcher abstraction-drift failure)." Generative Agents (scout-6
entry 7) shows "reflections compound bias (downstream reflections cite
earlier reflections rather than primary observations)." The candidate fix
to AI-Researcher §6.2 (over-summarization) has a documented related
failure mode in the same direction.

**Implication:** GAP-A7 is still worth pursuing but the hypothesis-smith
should constrain the evolution cadence and force linkage to *primary*
records (not consolidated records) to avoid recreating the original
failure under a new name. A naive port of A-MEM to paper-gen does not
clear this bar.

### KILL-5: Tree-of-Thoughts over free-prose revision states without a partial-state evaluator

**Why kill:** Yao et al. (arXiv:2305.10601) explicitly note ToT fails on
"tasks where the LLM cannot reliably evaluate partial states" — Game of 24
works because partial-state evaluation is tractable; creative-writing works
less well in the same paper. Manuscript-section quality is the canonical
hard partial-state evaluation case.

**Implication:** GAP-A3 (ToT over revision states) is only viable if the
partial-state evaluator is constrained to a *checkable* sub-quality
(presence-of-citation, presence-of-ablation, factual-claim-resolves,
numeric-result-matches-source). A hypothesis that proposes ToT over
free-prose paragraph quality is dominated by the Yao-et-al. cap and
should be killed before being forged.

---

## (d) What I searched for but couldn't find

Pattern-system intersections I tried to assess but the literature is silent on.

1. **MIRIX-style six-typed-memory taxonomy specifically applied to
   research-pipeline workloads.** Scout-6 entry 4 *suggests* the map
   (episodic = experiment logs, semantic = literature facts, procedural =
   pipeline templates) but does not test it. None of the AI-Scientist family
   systems test it either. GAP-A11 is therefore a *suggested* gap rather than
   a fully-bounded one — the hypothesis-smith would have to define the
   typed-memory map itself.

2. **Heterogeneous-model debate evaluated specifically on long-form research
   text targets.** Zhang et al. (arXiv:2502.08788) tests heterogeneous-model
   MAD on 9 NLP benchmarks (math, reasoning, factual QA, commonsense) — no
   long-form research-text target. Tree-of-Debate (arXiv:2502.14767) tests
   *persona*-debate on scientific comparative analysis but does not vary
   the foundation model across personas. No paper in the slice cross-cuts
   "different foundation models" × "long-form research text quality." This
   is the strongest candidate for the eval-designer to build a measurement
   harness against.

3. **Empirical magnitude of Git Context Controller-style versioning in
   stateless multi-agent pipelines.** Wu (arXiv:2508.00031) reports
   qualitative results on long-horizon coding + self-replication; no
   benchmark delta is recorded for *audit-trail completeness* or
   *rejected-hypothesis recoverability* — exactly what MegaResearcher
   discipline rule #1 wants to measure. Scout-6 open question 7 also flags
   this.

4. **Whether Choi-et-al.'s voting-beats-debate finding holds for
   long-form-text targets.** Choi et al. ran on short-answer benchmarks
   (math, MCQ, factual QA); their martingale analysis (their §4) does not
   trivially extend to long-form text where the "vote" is undefined. Scout-5
   open question 1 raises this; no paper closes it.

5. **The interaction between Feedback Friction (model-side) and
   heterogeneous-model debate (interlocutor-side).** Feedback Friction
   (arXiv:2506.11930) is studied with a single model receiving high-quality
   feedback. Whether a heterogeneous-model debate *partially* breaks the
   friction floor (because the other agent has different intuitions about
   what to update) is untested. This is the strongest reason to score
   GAP-A1 plausibility "HIGH" rather than "very high" — the interaction
   is unmeasured.

6. **Constitutional principles for paper-writing specifically.** The
   constitutional-AI literature uses ~10 principles for behavior (harmlessness,
   tone) and ~3 principles for medical interview (2411.10168). No published
   paper lists the principles a paper-writing critic should be given. The
   hypothesis-smith would have to author the principle list itself, drawing
   on the failure-mode catalogues in AgentRxiv §4.1 and More-You-Automate
   §findings.

---

## Sources cited in this output

All arXiv IDs verified resolvable via `hf_papers paper_details`:

Generic patterns (from scout-5/scout-6):
- 2303.17651 — Self-Refine
- 2303.11366 — Reflexion
- 2305.10601 — Tree of Thoughts
- 2305.14325 — Du et al. Multi-Agent Debate
- 2305.19118 — Liang et al. Multi-Agent Debate / DoT
- 2310.01798 — LLMs Cannot Self-Correct Reasoning Yet (KILL ceiling)
- 2502.08788 — Stop Overvaluing MAD (KILL ceiling + heterogeneous-model effect)
- 2508.17536 — Debate or Vote (KILL ceiling)
- 2401.10020 — Self-Rewarding LMs
- 2407.19594 — Meta-Rewarding LMs
- 2212.08073 — Constitutional AI
- 2506.11930 — Feedback Friction (KILL ceiling)
- 2502.14767 — Tree-of-Debate
- 2502.12110 — A-MEM
- 2507.07957 — MIRIX
- 2407.04363 — AriGraph
- 2602.01566 — FS-Researcher
- 2508.00031 — Git Context Controller
- 2402.11436 — Pride and Prejudice (self-bias)
- 2410.12853 — Diversity of Thought (heterogeneous-model MAD)
- 2404.04475 — Length-Controlled AlpacaEval
- 2404.05966 — THOUGHTSCULPT

Paper-gen systems (from scout-1):
- 2408.06292 — AI Scientist v1
- 2504.08066 — AI Scientist v2
- 2501.04227 — Agent Laboratory
- 2503.18102 — AgentRxiv
- 2505.18705 — AI-Researcher
- 2501.03916 — Dolphin
- 2603.08127 — EvoScientist
- 2510.15624 — freephdlabor
- 2601.20833 — Idea2Paper
- 2511.04583 — Jr. AI Scientist
- 2411.00816 — CycleResearcher
- 2604.05018 — PaperOrchestra
- 2412.21154 — Aviary
- 2502.16069 — Curie
- 2411.11910 — Baby-AIGS
- 2509.08713 — More You Automate
- 2601.03315 — Why LLMs Aren't Scientists Yet
- 2605.03042 — ARIS (adjacent — has heterogeneous-model adversarial)

Other verification-query results:
- 2604.01029 — Revision or Re-Solving (decomposition of revision pipeline gains)
- 2504.16728 — IRIS (MCTS for ideation)
- 2603.29557 — FlowPIE (MCTS for idea evolution)
- 2507.16075 — TTD-DR (test-time diffusion deep researcher)
- 2506.12689 — SciSage (reflect-when-you-write for survey gen)
- 2402.07319 — ODIN (length-bias in RLHF)
- 2601.11374 — Reward Modeling for Scientific Writing
- 2503.03746 — Process-based Self-Rewarding
- 2410.21819 — Self-Preference Bias in LLM-as-Judge
- 2410.02736 — Justice or Prejudice (LLM-judge biases)
