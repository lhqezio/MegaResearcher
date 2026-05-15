# scout-6 — Memory and state management for long agent workflows

## Scope

Sub-topic: how long-running LLM-agent systems manage memory beyond a single
context window — hierarchical memory (MemGPT, A-MEM, MemoryOS), structured-
artifact passing (file-system-as-memory, version-controlled context),
KV-cache management for multi-agent workflows, external scratchpads, vector
memory, episodic memory, and work-product handoff between agents.

Narrowing decisions:

- I prioritised systems that can be implemented (or already are implemented)
  under the spec's hard constraint: **single-session orchestrator, no nested
  dispatch, file-based artifact passing between stateless worker subagents.**
  Each entry is tagged with a `statefulness requirement` field that makes
  compatibility with this constraint explicit.
- I flagged but did not deeply cover systems that require a persistent agent
  process or RL-trained memory-controller (MemPO, DeltaMem, Mem-T, MemGen) —
  they are out-of-scope for direct hypotheses under the YAGNI fence ("no
  fine-tuning") but appear in §"Open questions" as future work.
- I confirmed the AI-Researcher gap the assignment specifically asks about
  by reading arXiv:2505.18705 §6.2: AI-Researcher operates **without a
  dedicated external memory management system**, so fine-grained details
  from early pipeline stages "become increasingly difficult to access" and
  are "effectively compressed into increasingly abstract summaries that
  sacrifice specificity for brevity." This is the exact failure mode every
  paper below either attacks directly or sidesteps.

## Key papers

### Cluster A — Hierarchical / OS-inspired virtual memory

**1. MemGPT: Towards LLMs as Operating Systems** — arXiv:2310.08560 (2023)
Packer, Fang, Patil, Lin, Wooders, Gonzalez.
Introduces virtual context management: a fast in-context "main context" and
a slow "external context" with paging via LLM-issued function calls (write,
read, search). Inspired by OS hierarchical memory; the agent itself decides
what to page in and out via interrupts.
- Memory architecture: **hierarchical, OS-style paging** (main context ↔
  recall storage ↔ archival storage), self-directed via function calls.
- Statefulness: requires a persistent agent process to manage paging and
  receive interrupts. **Incompatible with stateless leaf-worker dispatch as
  written**, but the page-store / archival-store can be reduced to a file
  store readable by stateless workers — that's the design pattern MegaResearcher
  would need to abstract.
- Failure mode: documented retrieval-recall degradation when the agent
  fails to page in the right chunk; relies on the LLM's own retrieval
  judgement.
- Evaluated on: deep document analysis (multi-doc QA), and multi-session
  conversation (MSC dataset).
- Why it matters: canonical reference for "context window is not enough,
  give it a memory hierarchy." The architectural template every subsequent
  paper below either extends or critiques.

**2. MemoryOS of AI Agent** — arXiv:2506.06326 (2025)
Kang, Ji, Zhao, Bai.
Three-tier hierarchical store (short-term, mid-term, long-term personal
memory) with FIFO eviction within tiers and segmented page organisation.
Top result on LoCoMo benchmark at time of publication.
- Memory architecture: **three-tier hierarchical**, with explicit movement
  rules between tiers and segmented page organisation inside each tier.
- Statefulness: tier-movement logic runs as a continuous process; can be
  reduced to file-on-disk tiers if the move policy is stateless per call
  and decisions are based on artifact timestamps.
- Failure mode: tiering policy is rule-based and hand-tuned; brittle when
  task distribution shifts.
- Evaluated on: LoCoMo long-form conversation benchmark.
- Why it matters: provides a concrete tier-design vocabulary
  (short / mid / long) MegaResearcher can map onto its
  `runs/<run-id>/<worker>/` artifact tree.

### Cluster B — Self-organising / agentic memory

**3. A-MEM: Agentic Memory for LLM Agents** — arXiv:2502.12110 (2025)
Xu, Liang, Mei, Gao, Tan, Zhang. (880 GH stars)
Zettelkasten-inspired: each new memory note is automatically (i) given a
contextual description, keywords, and tags, (ii) linked to related existing
notes, and (iii) allowed to trigger updates to those linked notes ("memory
evolution"). Memory structure emerges from interactions rather than from a
fixed schema.
- Memory architecture: **dynamic graph of typed notes** (vector + symbolic
  links), append-mostly with controlled evolution of linked notes.
- Statefulness: the note store is a file/db that any worker can read; the
  link-update step can be run as a stateless post-hoc consolidation job.
  **Compatible with file-based handoff** if the consolidation step is a
  named subagent invoked between waves.
- Failure mode: link explosion under high-volume writes; evolution step is
  expensive and can rewrite useful detail into smoother prose (a softer
  version of the AI-Researcher abstraction-drift failure).
- Evaluated on: LoCoMo (conversational long-term memory).
- Why it matters: the closest published analogue to the structured-artifact
  pattern MegaResearcher already uses, with an explicit linking step that
  MegaResearcher currently lacks.

**4. MIRIX: Multi-Agent Memory System for LLM-Based Agents** — arXiv:2507.07957 (2025)
Wang, Chen. (3,542 GH stars)
Splits memory into six typed stores managed by six specialist agents: Core,
Episodic, Semantic, Procedural, Resource Memory, and Knowledge Vault. A
dispatcher routes writes/reads to the right store.
- Memory architecture: **modular typed multi-store**, one agent per memory
  type, central dispatcher.
- Statefulness: each memory-type agent can be a stateless leaf worker that
  reads/writes its own typed file store. **Compatible with the
  MegaResearcher architectural constraint** if "no nested dispatch" is
  interpreted as the orchestrator dispatching all six in waves rather than
  the dispatcher being a peer.
- Failure mode: cross-store consistency (an episodic fact and a semantic
  fact can disagree); high storage overhead.
- Evaluated on: ScreenshotVQA (multimodal long-form), LOCOMO.
- Why it matters: explicit memory-type taxonomy that maps cleanly to
  research-pipeline roles (episodic = experiment logs, semantic = literature
  facts, procedural = pipeline templates).

**5. General Agentic Memory Via Deep Research (GAM)** — arXiv:2511.18423 (2025)
Yan, Li, Qian, Lu, Liu. (848 GH stars, 170 upvotes)
"JIT compilation" approach: keep only a lightweight, generic memory store
offline (raw pages, minimal annotation); at query time a "researcher" agent
walks the store and assembles a task-specific context just-in-time. Reverses
the upfront-summarisation pattern that AI-Researcher §6.2 calls out.
- Memory architecture: **universal page-store + on-demand re-compilation**,
  rather than upfront structured indexing.
- Statefulness: page store is a file repo; the researcher agent is a leaf
  worker invoked per query. **Compatible with file-based handoff.**
- Failure mode: query-time cost (the researcher walks the store on every
  call); RL-trained researcher is needed for cost-quality trade-off, which
  pushes into out-of-scope fine-tuning.
- Evaluated on: deep-research-style long-horizon QA.
- Why it matters: directly attacks the "abstraction over time" failure that
  AI-Researcher identifies. If the orchestrator never produces lossy
  summaries until they are needed, the early-stage-detail-loss failure mode
  is structurally avoided.

### Cluster C — Episodic / graph memory

**6. AriGraph: Learning Knowledge Graph World Models with Episodic Memory
for LLM Agents** — arXiv:2407.04363 (2024)
Anokhin, Semenov, Sorokin, Evseev, Burtsev, Burnaev. (166 GH stars)
Builds a knowledge graph of semantic memories (entity–relation triples)
plus episodic memories (time-stamped events linked to the same entities).
Retrieval is associative: query → relevant subgraph → relevant episodes.
- Memory architecture: **knowledge graph + episodic events**, dual
  retrieval (semantic neighbourhood + episode pointers).
- Statefulness: the graph is persistent on disk and read-only from any
  worker; the writer can be a stateless consolidation pass after each
  worker wave. **Compatible with stateless dispatch.**
- Failure mode: graph schema rigidity (entity-resolution failures cause
  duplicate nodes that retrieval cannot reconcile); does not handle
  contradictory updates well.
- Evaluated on: TextWorld zero-shot planning.
- Why it matters: serves as the analogue for a "research entity graph"
  (papers, hypotheses, baselines, datasets, results) that MegaResearcher
  could maintain across waves; entities provide cross-worker anchors that
  prose summaries lose.

**7. Generative Agents: Interactive Simulacra of Human Behavior** —
arXiv:2304.03442 (2023)
Park, O'Brien, Cai, Morris, Liang, Bernstein.
The memory-stream paper: append-only log of observations, retrieved by
joint recency + importance + relevance score, with periodic "reflection"
synthesising higher-level beliefs from low-level entries. Pre-MemGPT
canonical reference for episodic-stream memory.
- Memory architecture: **append-only memory stream** with reflection-as-
  consolidation.
- Statefulness: append-only stream is a file; reflection is a periodic
  stateless pass. **Compatible.**
- Failure mode: reflections compound bias (downstream reflections cite
  earlier reflections rather than primary observations); recency weight
  drowns important-but-old facts.
- Evaluated on: small-town simulation (qualitative).
- Why it matters: the foundational pattern, and the **first explicit case
  of the AI-Researcher failure mode** — reflections progressively replace
  primary records. Knowing this is in the canonical paper grounds the gap
  AI-Researcher §6.2 reports.

### Cluster D — Structured / file-system memory (most compatible with the spec)

**8. FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks
with File-System-Based Agents** — arXiv:2602.01566 (2026)
Zhu, Xu, Du, Wang, Wang, Mao, Zhang. (29 GH stars, 52 upvotes)
Dual-agent framework where a "Context Builder" agent maintains a persistent
workspace of structured notes on the file system, and a "Researcher" agent
operates over that workspace. Designed specifically for deep-research tasks
that exceed the context window.
- Memory architecture: **file-system persistent workspace** of structured
  notes (markdown + JSON manifests).
- Statefulness: workspace is the state; agents are stateless. **Already
  matches the MegaResearcher pattern exactly.**
- Failure mode: workspace can become disorganised without a periodic
  organisational pass; retrieval over a flat workspace degrades.
- Evaluated on: long-horizon research benchmarks.
- Why it matters: the closest published precedent for the MegaResearcher
  architecture. Reading this paper carefully would be the single biggest
  win for hypothesis-smith.

**9. Git Context Controller: Manage the Context of LLM-based Agents like
Git** — arXiv:2508.00031 (2025)
Wu.
Treats agent context as a versioned hierarchy with `COMMIT`, `BRANCH`,
`MERGE`, `CONTEXT` operations. Lets agents checkpoint milestones and
branch off alternative plans, then merge back the survivors.
- Memory architecture: **version-controlled context tree** with explicit
  commit/branch semantics, on top of files.
- Statefulness: backed by an actual git-like store; workers are stateless
  and address state by revision. **Compatible with file-based handoff and
  naturally supports the audit trail the spec requires.**
- Failure mode: merge conflicts on context are not solved well; relies on
  agent discipline to write meaningful commit messages.
- Evaluated on: long-horizon coding tasks and self-replication.
- Why it matters: the audit-trail-discipline rule in CLAUDE.md is exactly
  the use case this paper addresses. A version-controlled artifact store
  would make rejected-hypothesis branches first-class.

**10. L2MAC: Large Language Model Automatic Computer for Extensive Code
Generation** — arXiv:2310.02003 (2023)
Holt, Ruiz Luyten, van der Schaar.
Stored-program automatic computer analogue for LLMs: an instruction
registry + a file store, with a control-unit LLM that executes instructions
sequentially, each instruction operating on the file store. Predates
file-system-as-memory framings.
- Memory architecture: **instruction registry + file store**, von Neumann
  style.
- Statefulness: control unit is stateful (program counter), file store is
  on disk; can be reduced to stateless steps if the program counter lives
  in the file store. **Compatible with adaptation.**
- Failure mode: instruction-fidelity loss in long programs (the control
  unit drifts from the planned sequence); explicit precursor to the
  AI-Researcher abstraction-drift failure.
- Evaluated on: extensive code generation (full codebases), book writing.
- Why it matters: the earliest explicit "files-as-memory" formalism in the
  LLM-agent literature; the failure modes documented here predict
  AI-Researcher's failures and let the synthesist cite the lineage.

### Cluster E — Diagnosis, taxonomy, and benchmarks (essential to red-team)

**11. Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of
Evaluation and System Limitations** — arXiv:2602.19320 (2026)
Jiang, Li, Wei, Yang, Kishore, Zhao, Kang, Hu, Chen, Li et al. (21 GH stars)
Empirical taxonomy paper: benchmarks for agentic memory are underscaled,
metrics are misaligned with semantic utility, performance is backbone-
dependent, and system-level costs are routinely ignored. Provides a
unified evaluation framework.
- Memory architecture: N/A (taxonomy and meta-analysis).
- Statefulness: N/A.
- Failure mode: this paper **is** the failure-mode analysis — exactly the
  empirical critique a hypothesis-smith needs to anchor falsifiable
  eval design.
- Evaluated on: cross-benchmark study.
- Why it matters: refuses naïve "system X beat system Y" claims; tells the
  red-team worker exactly which metric pitfalls to challenge.

**12. Memory in the Age of AI Agents (survey)** — arXiv:2512.13564 (2025)
Hu, Liu, Yue, Zhang, Liu, Zhu, Lin, Guo, Dou, Xi et al. (1,960 GH stars on
the linked paper list, 156 upvotes)
Latest comprehensive survey. Frames memory along forms (parametric / latent
/ token / external), functions (working / experiential / factual), and
dynamics (write / read / evolve). Lists open problems including
trustworthiness and memory-RL integration.
- Memory architecture: N/A (survey).
- Why it matters: provides the up-to-date vocabulary the synthesist will
  need to put the swarm output in dialogue with the field as of late 2025.

**13. SCBench: A KV Cache-Centric Analysis of Long-Context Methods** —
arXiv:2412.10319 (2024)
Li, Jiang, Wu, Luo, Ahn, Zhang, Abdi, Li, Gao, Yang et al.
Reframes long-context evaluation as a KV-cache lifecycle (generation →
compression → retrieval → loading) and runs it across method families
(sparse attention, KV dropping, Mamba hybrids, prompt compression).
- Memory architecture: **KV-cache-as-memory** (orthogonal axis to
  application-layer memory).
- Statefulness: KV cache is intrinsically tied to the model process and
  generally cannot survive across stateless dispatch boundaries.
  **Incompatible with file-based handoff between stateless workers**
  unless cache is serialised to disk (see KVFlow / KVCOMM below).
- Failure mode: distribution shift in attention patterns across cache
  lifecycle stages; methods that look good in single-request fail in
  reuse.
- Why it matters: tells MegaResearcher honestly that KV-cache-level memory
  is **not** in scope under the architectural constraint, while still
  giving a clean reason to cite it as future work.

### Cluster F — KV-cache for multi-agent workflows (out-of-scope but adjacent)

**14. KVFlow: Efficient Prefix Caching for Accelerating LLM-Based
Multi-Agent Workflows** — arXiv:2507.07400 (2025)
Pan, Patel, Hu, Shen, Guan, Li, Qin, Wang, Ding.
Builds an "Agent Step Graph" of upcoming agent invocations and uses it to
schedule KV-cache eviction and prefetching. Targets multi-agent serving
specifically.
- Memory architecture: **prefix-cache scheduling over an agent DAG**.
- Statefulness: cache lives in the serving runtime; requires control over
  the inference stack. **Incompatible with the spec's architectural
  constraint** (Claude Code workers do not own the cache).
- Why it matters: surfaces the most efficient pattern that the constraint
  rules out — useful as a "what would be different if the constraint were
  relaxed" hypothesis from gap-finder.

**15. KVCOMM: Online Cross-context KV-cache Communication for Efficient
LLM-based Multi-agent Systems** — arXiv:2510.12872 (2025)
Ye, Gao, Ma, Wang, Fu, Chung, Lin, Liu, Zhang, Zhuo et al. (162 GH stars)
Training-free method to reuse KV caches across agents by aligning cache
offsets via an anchor pool. Achieves speedups without quality loss.
- Memory architecture: **shared KV-cache pool with offset alignment**
  across agents.
- Statefulness: same as KVFlow — requires runtime control. **Incompatible
  with file-based handoff.**
- Why it matters: same role as KVFlow — establishes that the most
  performance-relevant memory layer is currently out of reach for
  Claude-Code-style architectures.

### Cluster G — Long-horizon benchmarks (use these for eval-designer)

**16. UltraHorizon: Benchmarking Agent Capabilities in Ultra Long-Horizon
Scenarios** — arXiv:2509.21766 (2025)
Luo, Zhang, Zhang, Wang, Qin, Lu, Ma, He, Xie, Zhou et al. (23 GH stars,
24 upvotes)
Long-horizon, partially observable benchmark explicitly testing sustained
reasoning, planning, memory management, and tool use. Documents a
phenomenon they call "in-context locking" — agents lock onto early
hypotheses and cannot revise them despite later evidence.
- Why it matters: the "in-context locking" failure mode is the empirical
  twin of AI-Researcher §6.2 abstraction drift. Gives the eval-designer
  a real benchmark to ground falsification on.

**17. Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory
in LLMs (BEAM / LIGHT)** — arXiv:2510.27246 (2025)
Tavakoli, Salemi, Ye, Abdalla, Zamani, Mitchell.
Procedurally generates up-to-10M-token coherent conversations and
introduces LIGHT — a memory framework with long-term episodic memory,
short-term working memory, and a scratchpad. Explicit ablation across
memory components.
- Why it matters: the only benchmark in the corpus where the eval design
  isolates scratchpad-vs-episodic-vs-working contributions, exactly the
  ablation hypothesis-smith will want.

## Datasets

- **LOCOMO** — long-form conversation benchmark for long-term memory.
  Used by A-MEM (2502.12110), MemoryOS (2506.06326), MIRIX (2507.07957),
  many others. HF: https://huggingface.co/datasets/snap-research/locomo
  (verify availability before use). Licence: research-use; check HF page.
- **LongMemEval** — long-term memory eval across sessions. Cited widely
  in the survey 2512.13564 and Anatomy paper 2602.19320; HF page exists
  under the upstream authors.
- **BEAM** — the long-conversation benchmark from arXiv:2510.27246.
  Up to 10M tokens, procedurally generated; check the paper's GitHub for
  the dataset card.
- **UltraHorizon** — long-horizon partially-observable tasks from
  arXiv:2509.21766. GitHub: github.com/StarDewXXX/UltraHorizon. Dataset
  artifact bundled with the repo; licence in repo.
- **MemBench** — multi-dimensional memory eval (arXiv:2506.21605).
  Surfaces factual / reflective / temporal / cross-session memory.
- **TextWorld** — text-game environment used by AriGraph (2407.04363)
  for episodic-memory evaluation. MIT licence, classic env.
- **MSC (Multi-Session Chat)** — Facebook AI long-session conversation
  benchmark used by MemGPT (2310.08560) and many follow-ups. CC-BY-NC.
- **MemoryRewardBench** — arXiv:2601.11969, evaluation of reward models
  for memory management. GitHub: github.com/LCM-Lab/MemRewardBench (9
  stars). Flagged for future work, not directly applicable here.

**Licence flags:** MSC is CC-BY-NC (commercial use restricted, fine for
academic eval). LOCOMO and BEAM appear research-use; verify on each HF
page before any pipeline integration.

## Reference implementations

- **MIRIX** — github.com/Mirix-AI/MIRIX (3,542 stars). Full multi-agent
  memory system implementation matching arXiv:2507.07957.
- **MemoryOS** — github.com/BAI-LAB/MemoryOS (1,375 stars). Reference
  implementation for arXiv:2506.06326.
- **A-MEM (Agentic Memory)** — github.com/wujiangxu/agenticmemory (880
  stars). Zettelkasten-style memory framework matching arXiv:2502.12110.
- **General Agentic Memory (GAM)** — github.com/VectorSpaceLab/general-agentic-memory
  (848 stars). JIT-compilation memory matching arXiv:2511.18423.
- **G-Memory** — github.com/bingreeky/GMemory (231 stars). Hierarchical
  multi-agent memory matching arXiv:2506.07398.
- **KVCOMM** — github.com/FastMAS/KVCOMM (162 stars). Reference KV-cache
  reuse for multi-agent workflows; flagged out-of-scope but useful.
- **AriGraph** — github.com/airi-institute/arigraph (166 stars).
  Knowledge-graph + episodic memory reference.
- **HiAgent** — github.com/hiagent2024/hiagent (52 stars). Subgoal-based
  working memory reference.
- **FS-Researcher** — github.com/Ignoramus0817/FS-Researcher (29 stars).
  **File-system-based dual-agent reference — the closest match to the
  MegaResearcher architectural pattern in the corpus.**
- **Agent-Memory-Paper-List** — github.com/Shichun-Liu/Agent-Memory-Paper-List
  (1,960 stars). Curated paper list maintained alongside the survey
  2512.13564; useful as a living index.

## Open questions you noticed

(Flagged for gap-finder and hypothesis-smith — not proposing hypotheses.)

1. **The AI-Researcher abstraction-drift failure mode (§6.2 of arXiv:2505.18705)
   is described qualitatively but never measured.** No paper in the corpus
   gives a concrete metric for "how much primary-source detail survives
   N waves of summarisation in a research pipeline." This is a measurement
   gap, not a method gap.
2. **None of the agentic-memory papers explicitly test against a
   research-pipeline workload.** Almost all evaluation lives in
   conversation (LOCOMO, MSC), embodied/textgame (TextWorld,
   ALFWorld), or coding (HumanEval). The closest is FS-Researcher
   (arXiv:2602.01566), but its eval is deep-research-QA, not paper
   production with baselines, ablations, and related work.
3. **Compatibility of MIRIX-style typed memory with stateless leaf
   dispatch is plausible but not demonstrated.** The MIRIX paper assumes
   the six memory-type agents are peers that can talk to each other;
   under MegaResearcher's no-nested-dispatch rule they'd have to be
   wave-dispatched by the orchestrator with file handoff between waves.
   No paper in the corpus tests this restricted setup.
4. **The "memory evolution" step in A-MEM (2502.12110) is the failure-mode
   inverter to AI-Researcher §6.2** — it rewrites linked notes when new
   info arrives — but A-MEM's eval does not measure detail-preservation
   either. So both the failure (AI-Researcher) and the candidate fix
   (A-MEM) are under-measured.
5. **KV-cache-level memory (KVFlow, KVCOMM, SCBench) is the highest-leverage
   layer for multi-agent efficiency but is out-of-reach for stateless
   leaf-worker architectures.** This is a constraint-cost worth surfacing:
   how much performance is left on the table by the file-handoff rule?
   Unmeasured.
6. **In-context locking (UltraHorizon, 2509.21766) and abstraction drift
   (AI-Researcher) may be the same phenomenon at different granularities.**
   No paper unifies them. Worth flagging to gap-finder.
7. **Git-style versioned context (2508.00031) maps naturally onto
   MegaResearcher's audit-trail-is-non-negotiable rule**, but no eval
   in the corpus directly measures audit-trail completeness or
   rejected-hypothesis recoverability.
8. **Memory systems that require RL-trained controllers** (MemPO 2603.00680,
   DeltaMem 2604.01560, Mem-T 2601.23014, MemGen 2509.24704) violate the
   "no fine-tuning" YAGNI clause. Flagging them so the orchestrator does
   not waste a wave on training-required hypotheses.

## Sources

Cited arXiv IDs (every entry above):

- 2505.18705 — AI-Researcher
- 2310.08560 — MemGPT
- 2506.06326 — MemoryOS
- 2502.12110 — A-MEM
- 2507.07957 — MIRIX
- 2511.18423 — General Agentic Memory (GAM)
- 2407.04363 — AriGraph
- 2304.03442 — Generative Agents
- 2602.01566 — FS-Researcher
- 2508.00031 — Git Context Controller
- 2310.02003 — L2MAC
- 2602.19320 — Anatomy of Agentic Memory
- 2512.13564 — Memory in the Age of AI Agents (survey)
- 2412.10319 — SCBench
- 2507.07400 — KVFlow
- 2510.12872 — KVCOMM
- 2509.21766 — UltraHorizon
- 2510.27246 — Beyond a Million Tokens (BEAM / LIGHT)
- 2506.07398 — G-Memory
- 2412.15266 — On the Structural Memory of LLM Agents
- 2303.11366 — Reflexion
- 2506.21605 — MemBench
- 2408.09559 — HiAgent

Additional context for "open questions" (flagged out-of-scope):

- 2603.00680 — MemPO (RL-trained, out of scope)
- 2604.01560 — DeltaMem (RL-trained, out of scope)
- 2601.23014 — Mem-T (RL-trained, out of scope)
- 2509.24704 — MemGen (parametric memory, out of scope)
- 2601.11969 — MemoryRewardBench
- 2508.08997 — Intrinsic Memory Agents
