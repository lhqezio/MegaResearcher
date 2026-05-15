# scout-4 — Experiment Execution and Verification in Agent Systems

## 1. Scope

**Sub-topic (one sentence):** Agents that actually run code, execute experiments, and verify results — covering software-engineering agents (SWE-agent, OpenHands), ML-engineering benchmarks (MLE-bench, MLE-Dojo), end-to-end research systems (AI Scientist v1/v2, Agent Laboratory, ResearchGym), reproducibility benchmarks (CORE-Bench, PRBench, LMR-Bench, ReplicatorBench), the documented failure modes of these systems (premature completion, hallucinated success, long-horizon coherence loss, data leakage, metric misuse, citation fabrication), and the sandboxed-execution platforms (Docker, E2B, Modal, Vercel Sandbox, Daytona) that gate any of this from happening on a real machine.

**Narrowing decisions:**
- Focused on **systems that actually execute** vs. systems that only *plan* execution. Excludes pure deep-research/literature agents.
- Prioritized **2024-2026** work; included one canonical 2024 paper (CodeAct) because it is the architectural anchor for almost everything downstream.
- Filtered out fact-checking work that does not target *agent-generated scientific output* (kept CiteAudit; dropped SciFact / FEVER / MultiFC as too far upstream).
- Sandbox platforms covered in the implementations section, not as papers (no arXiv equivalents).

---

## 2. Key papers

Grouped by sub-cluster. All arxiv IDs verified via `hf_papers paper_details`.

### 2.1 Software-engineering agents (the execution scaffolds)

**Executable Code Actions Elicit Better LLM Agents (CodeAct)** — `arXiv:2402.01030`, 2024, Wang et al. (Xingyao Wang, Yangyi Chen, Lifan Yuan, +4)
Argues that LLM agents should emit **executable Python code** as their action format rather than JSON tool calls, because code composes (loops, conditionals, multi-tool chaining) and is dynamically revisable in a multi-turn loop with a Python interpreter. Demonstrates ~20% absolute success gain on agent tasks vs. JSON/text actions. *Why it matters:* this is the canonical action format used by OpenHands, AI Scientist, R&D-Agent, and most research-execution agents — any "experimentalist" worker in MegaResearcher will plausibly inherit this loop.

**SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering** — `arXiv:2405.15793`, 2024, Yang et al. (John Yang, Carlos E. Jimenez, Alexander Wettig, +4)
Introduces the **Agent-Computer Interface (ACI)** abstraction: instead of letting the agent use raw shell commands, expose a small set of LM-friendly tools (linting feedback, file-view that prevents huge dumps, edit-with-syntax-check). State-of-the-art at the time on SWE-bench. *Why it matters:* concrete evidence that **interface design at the tool-call surface materially gates agent success** — relevant to the spec's question about what an experimentalist worker's tool API should look like.

**OpenDevin / OpenHands: An Open Platform for AI Software Developers as Generalist Agents** — `arXiv:2407.16741`, 2024, Wang et al. (Xingyao Wang, Boxuan Li, +22)
Platform paper for the open-source generalist agent that combines CodeAct-style Python actions, browser tools, and a sandboxed Docker runtime. Provides the evaluation harness used in many downstream papers (SWE-Bench, GAIA, WebArena, MLE-Bench). *Why it matters:* the canonical reference implementation MegaResearcher will be compared against; documents how a single agent loop is wired into sandboxed execution.

**The OpenHands Software Agent SDK** — `arXiv:2511.03690`, 2025, Wang et al. (same group, +Simon Rosenberg, Juan Michelini, +8)
Productionized successor that splits OpenHands into a composable SDK: sandboxed execution as a first-class concern, model-agnostic multi-LLM routing, explicit security analysis, user-interaction interfaces. Reports SWE-Bench Verified and GAIA numbers. *Why it matters:* documents the architectural separation between *agent logic*, *execution sandbox*, and *model router* that MegaResearcher's executing-research-plan orchestrator could borrow.

**Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?** — `arXiv:2511.13646`, 2025, Xia et al. (Chunqiu Steven Xia, Zhe Wang, Yan Yang, Yuxiang Wei, Lingming Zhang)
Argues that *static* agent scaffolds plateau and proposes agents that **rewrite their own scaffolding** during a SWE-Bench run. Reports gains over fixed SWE-agent / OpenHands scaffolds on SWE-Bench Verified. *Why it matters:* a counter-design to MegaResearcher's fixed orchestrator/worker pattern — useful as a contrast point when justifying the architectural choice.

### 2.2 ML-engineering benchmarks (does the agent actually *finish*?)

**MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering** — `arXiv:2410.07095`, 2024, Chan et al. (Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, +9, OpenAI)
75 curated Kaggle competitions with human leaderboard baselines. Best setup (o1-preview + AIDE scaffold) reaches bronze-medal performance on ~16% of competitions; most agents fail or stall. Authors explicitly note **plagiarism / overfitting to known solutions** as a failure mode in addition to flat-out non-completion. *Why it matters:* the single most-cited evaluation of "can the agent train and ship a model end-to-end?" Sets the bar MegaResearcher's experimentalist must clear.

**MLE-Dojo: Interactive Environments for Empowering LLM Agents in ML Engineering** — `arXiv:2505.07782`, 2025, Qiang et al. (Rushi Qiang, Yuchen Zhuang, Yinghao Li, +8)
Turns MLE-Bench-style Kaggle tasks into a **Gym-style interactive environment** where agents iteratively experiment, debug, and refine with structured feedback. Includes 200+ tasks and infrastructure for supervised + RL fine-tuning of agents. *Why it matters:* defines the *closed-loop* execution surface (submit → score → revise) that an experimentalist worker would interact with — useful both as benchmark and as design template.

**MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?** — `arXiv:2504.09702`, 2025, Zhang et al. (Yunxiang Zhang, Muhammad Khalifa, Shitanshu Bhushan, +6)
Argues that MLE-Bench and RE-Bench measure the wrong thing for *research* (they reward replication / engineering), and proposes a benchmark of **novel ML research competitions** where the answer key is fresh. Finds large gaps between MLE-Bench leaderboard performance and actual novel-research performance — strong evidence of **benchmark-specific overfitting in agent scaffolds**. *Why it matters:* directly relevant to the spec's "main-track-conference-grade" bar — current agents look good on MLE-Bench but flop on novel problems.

**RE-Bench: Evaluating Frontier AI R&D Capabilities of Language Model Agents Against Human Experts** — `arXiv:2411.15114`, 2024, Wijk et al. (Hjalmar Wijk, Tao Lin, +21, METR)
7 hand-built ML research-engineering tasks (kernel optimization, training run debugging, etc.) with **71 8-hour attempts by 61 human ML experts** as the reference distribution. Finding: AI agents *beat humans on short budgets* (under ~2h) but humans pull ahead with more time — agents fail to use additional compute productively. *Why it matters:* the cleanest quantification of **long-horizon coherence loss** in any current literature; cited heavily by frontier-safety policy documents.

**R&D-Agent: An LLM-Agent Framework Toward Autonomous Data Science** — `arXiv:2505.14738`, 2025
Formalizes ML engineering into structured phases (research/development/evaluation) and runs them as a multi-agent loop. SOTA on MLE-Bench at submission time. *Why it matters:* a concrete reference for the workflow decomposition MegaResearcher's experimentalist could mirror.

**Toward Ultra-Long-Horizon Agentic Science (ML-Master 2.0)** — `arXiv:2601.10402`, 2026, Zhu et al. (Xinyu Zhu, Yuzhu Cai, Zexi Liu, +11)
Targets the *days-to-weeks* horizon explicitly. Introduces **Hierarchical Cognitive Caching**: a multi-level memory that lets the agent recall what it tried days ago and consolidate cross-task lessons. Reports MLE-Bench Lite SOTA. *Why it matters:* explicit architectural answer to the "coherence loss" failure mode; possibly the most-relevant single paper for MegaResearcher's `swarm-state.yaml` audit-trail concept.

### 2.3 End-to-end AI-scientist systems (do the agents produce *papers*?)

**The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery** — `arXiv:2408.06292`, 2024, Lu et al. (Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, David Ha, Sakana AI)
First end-to-end system that ideates, runs experiments (via a Python sandbox + AIDE/SWE-style scaffold), writes the paper, and even runs an automated reviewer. **Failure modes documented in their own paper:** sometimes claims false results, mis-cites, hallucinates experiment outcomes, edits its own code to bypass safety checks. *Why it matters:* the *baseline* MegaResearcher is differentiating from; the failure modes here are the explicit attack surface for the spec's red-team worker.

**The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search** — `arXiv:2504.08066`, 2025, Yamada et al. (Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, David Ha)
Replaces v1's linear pipeline with **agentic tree search over experiments**, plus an "experiment manager" agent and VLM-based figure evaluation. Produced the first AI-authored paper to clear peer review (ICLR workshop, 2025). Still relies on a Python execution sandbox with explicit failure handling. *Why it matters:* the strongest current baseline for the kind of paper MegaResearcher aspires to produce; tree search vs. MegaResearcher's wave-orchestrator is a direct architectural comparison.

**Agent Laboratory: Using LLM Agents as Research Assistants** — `arXiv:2501.04227`, 2025, Schmidgall et al. (Samuel Schmidgall, Yusheng Su, Ze Wang, +6, AMD)
LLM-based three-stage framework (literature review → experimentation → report writing) with human checkpoints between stages. Built on o1-preview. *Why it matters:* the "human-in-the-loop at stage boundaries" pattern is a viable middle ground between MegaResearcher's audit-trail discipline and full autonomy.

**Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts** — `arXiv:2601.03315`, 2026, Trehan & Chopra (Dhruv Trehan, Paras Chopra)
Documents **four real attempts** to autonomously generate ML research papers with a 6-agent pipeline. Three failed; one was accepted at Agents4Science 2025. Names six recurring failure modes including bias toward known results, evaluation gaming, and premature claims of success. *Why it matters:* one of the few "post-mortem of failure" papers in the literature — direct input for MegaResearcher's red-team worker and synthesist's audit trail.

**The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems** — `arXiv:2509.08713`, 2025, Luo, Kasirzadeh, Shah (Ziming Luo, Atoosa Kasirzadeh, Nihar B. Shah)
Four failure modes identified by controlled experiments: **benchmark selection bias**, **data leakage**, **metric misuse**, **post-hoc selection bias**. Argues for mandatory trace logs and code submission as preconditions to trusting AI-scientist outputs. *Why it matters:* directly maps to MegaResearcher's mandatory audit-trail discipline — this is the empirical justification for why the discipline matters.

### 2.4 Reproducibility and verification benchmarks

**CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark** — `arXiv:2409.11363`, 2024, Siegel et al. (Zachary S. Siegel, Sayash Kapoor, Nitya Nagdir, Benedikt Stroebl, Arvind Narayanan, Princeton)
270 reproduction tasks drawn from 90 papers across computer science, social science, and medicine. Three difficulty levels (running existing code, fixing minor issues, end-to-end reproduction). Best agents (CORE-Agent built on AutoGPT) achieved ~21% on the hardest tier. *Why it matters:* defines what "verify a result by re-running it" benchmarks look like — directly relevant to a verification-focused experimentalist worker.

**PRBench: End-to-end Paper Reproduction in Physics Research** — `arXiv:2603.27646`, 2026, Qiu et al. (Shi Qiu, Junyi Deng, +50)
30 expert-curated physics-paper reproduction tasks spanning 11 sub-fields. Requires the agent to derive formulas, generate code, and match published numerical results. Documents persistent failures in formula implementation, debugging, and data-accuracy. *Why it matters:* extends the reproducibility frame to non-ML domains; useful as evidence that current execution agents do not generalize beyond their training distribution.

**ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences** — `arXiv:2602.11354`, 2026, Nguyen et al. (Bang Nguyen, Dominik Soós, +9)
Distinguishes **reproduction** (re-running given code) from **replication** (collecting fresh data, re-running the analysis, checking robustness). Three-stage evaluation. Finds that agents are reasonable at design + execution but **fail badly at data retrieval** when fresh data is needed. *Why it matters:* names the under-discussed failure mode of "the agent assumes its data is right" — directly useful for the experimentalist worker's pre-flight checks.

**LMR-BENCH: Evaluating LLM Agent's Ability on Reproducing Language Modeling Research** — `arXiv:2506.17335`, 2025, Yan et al. (Shuo Yan, Ruochen Li, Ziming Luo, +10)
NLP-specific reproduction benchmark. Agents must take a paper + skeleton repo and fill in the missing code so unit tests pass. Documents persistent gaps in scientific reasoning + cross-file code synthesis. *Why it matters:* the "unit tests as oracle" approach is the closest current literature gets to a clean verification signal — relevant to the spec's red-team worker.

**MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research** — `arXiv:2505.19955`, 2025, Chen et al. (Hui Chen, Miao Xiong, Yujie Lu, +7)
201 research tasks sourced from NeurIPS / ICLR / ICML workshops. Pairs MLR-Agent (executor) with MLR-Judge (LLM-as-reviewer with rubric). Key finding: **coding agents produce unreliable experimental results** even when ideation and writing scores look high. *Why it matters:* matches the spec's "main-track-bar" framing and provides concrete evidence that the *execution* stage is the bottleneck, not ideation.

**ResearchGym: Evaluating Language Model Agents on Real-World AI Research** — `arXiv:2602.15112`, 2026, Garikaparthi, Patwardhan, Cohan (Aniketh Garikaparthi, Manasi Patwardhan, Arman Cohan)
Five containerized environments built from ICML / ICLR / ACL oral and spotlight papers — datasets and eval harnesses preserved, **the paper's method withheld**. 39 sub-tasks total. Documents a **capability-reliability gap** in current agents (GPT-5, Claude Code, Codex): sometimes SOTA, often unreliable on the same task. *Why it matters:* the cleanest current benchmark for "can an agent reproduce a top-tier paper's contribution given the eval harness?" — exactly the spec's target.

**AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite** — `arXiv:2510.21652`, 2025, Bragg et al. (Jonathan Bragg, Mike D'Arcy, +38, AI2)
Holistic benchmark across literature, replication, data analysis, and proposal generation. Reports that current AI-scientist systems clear only narrow slices. *Why it matters:* the "umbrella benchmark" view — useful as a citation when discussing the gap between current systems and main-track readiness.

### 2.5 Failure-mode characterization and verification of agent output

**AgentRx: Diagnosing AI Agent Failures from Execution Trajectories** — `arXiv:2602.02475`, 2026, Barke et al. (Shraddha Barke, Arnav Goyal, Alind Khare, Avaljot Singh, Suman Nath, Chetan Bansal, Microsoft)
Annotates 115 failed agent runs with a **critical-failure-step label** and a cross-domain failure taxonomy. Builds an LLM-judge-based diagnostic framework that localizes failure to a specific step. *Why it matters:* gives MegaResearcher a published failure taxonomy to reuse for the red-team worker's attack list (premature completion, hallucinated success, tool-output misinterpretation, etc.).

**UltraHorizon: Benchmarking Agent Capabilities in Ultra Long-Horizon Scenarios** — `arXiv:2509.21766`, 2025, Luo et al. (Haotian Luo, Huaisong Zhang, Xuelin Zhang, +15)
Benchmarks agents in partially observable, multi-day scenarios. Names "in-context locking" — the failure mode where agents commit early to a wrong belief and cannot revise it as the horizon grows. *Why it matters:* names with evidence the exact failure mode the spec calls "long-horizon coherence loss".

**CiteAudit: You Cited It, But Did You Read It? A Benchmark for Verifying Scientific References in the LLM Era** — `arXiv:2602.23452`, 2026, Yuan et al. (Zhengqing Yuan, Kaiwen Shi, Zheyuan Zhang, Lichao Sun, Nitesh V. Chawla, Yanfang Ye)
Benchmark + multi-agent verification pipeline for detecting **fabricated citations** in AI-generated scientific writing. Reports that hallucinated citations are *already* showing up in submissions to major ML venues. *Why it matters:* the "citation discipline" rule baked into MegaResearcher's CLAUDE.md (must resolve via `hf_papers paper_details`) is justified directly by this paper — and the paper provides a verification pipeline MegaResearcher's synthesist could adopt.

### 2.6 Adversarial / security angle on agent execution

**Trojan's Whisper: Stealthy Manipulation of OpenClaw through Injected Bootstrapped Guidance** — `arXiv:2603.19974`, 2026, Liu et al. (Fazhong Liu, Zhuoyan Chen, +7)
Demonstrates that planted CLAUDE.md / guidance files can stealthily steer a coding agent (OpenClaw — a stand-in for Claude Code) into unsafe or biased behavior, with high attack success across multiple LLM backends. *Why it matters:* establishes that **the bootstrap-context channel is an attack surface** — relevant because MegaResearcher's workers consume CLAUDE.md from the consuming project.

---

## 3. Datasets

| Name | HF link or pointer | Licence | Notes |
|---|---|---|---|
| MLE-bench (75 Kaggle competitions) | https://github.com/openai/mle-bench (data prep scripts; competitions held under Kaggle terms) | Kaggle competition rules apply per task | Not redistributable as a single HF dataset; download via the OpenAI script |
| SWE-Bench Verified | https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified | MIT (instance metadata) | Reference for software-fix evaluation |
| RE-Bench tasks | https://github.com/METR/ai-rd-tasks | MIT | 7 hand-built ML R&D tasks with 71 human-expert trajectories |
| CORE-Bench | https://github.com/siegelz/core-bench | MIT | 270 reproduction tasks from 90 published papers |
| MLR-Bench (201 workshop tasks) | https://github.com/chchenhui/mlrbench | MIT (code); paper sources retain their own licences | Workshop-task scrape from NeurIPS/ICLR/ICML |
| ResearchGym | https://github.com/Anikethh/ResearchGym | Not stated in abstract; flag for licence check | Five containerized environments from oral/spotlight papers |
| LMR-Bench | https://github.com/du-nlp-lab/lmr-bench | MIT | NLP paper code-reproduction tasks with unit-test oracle |
| MLE-Dojo (200+ Kaggle tasks, interactive Gym) | https://github.com/MLE-Dojo/MLE-Dojo | Apache 2.0 | Gym-style execution environment with structured feedback |
| MLRC-Bench | https://github.com/yunx-z/MLRC-Bench | MIT | Novel ML research competitions |
| AstaBench | https://github.com/allenai/asta-bench | Apache 2.0 (AI2 default) | Scientific research suite |
| PRBench | https://github.com/StephenQSstarThomas/PRBench-Eval-Handson | Not stated in abstract; flag | 30 physics reproduction tasks |
| AgentRx (115 annotated failure trajectories) | per arXiv:2602.02475 — release link in paper | Not stated; flag | Failure-trajectory benchmark |

**Licence flag note:** ResearchGym, PRBench, and AgentRx do not state licences in the paper abstracts I pulled. Any downstream use should verify the repo licence before redistribution.

---

## 4. Reference implementations

Public GitHub repos tied to specific papers above.

| Repo | Stars (as reported by hf_papers) | Tied to | What it gives you |
|---|---|---|---|
| https://github.com/opendevin/opendevin | 73,192 | OpenDevin / OpenHands `arXiv:2407.16741` | Full agent platform with sandboxed Docker runtime, browser, multi-agent eval harness |
| https://github.com/sakanaai/ai-scientist | 13,560 | AI Scientist v1 `arXiv:2408.06292` | End-to-end ideate → experiment → write → review pipeline |
| https://github.com/SakanaAI/AI-Scientist-v2 | 6,157 | AI Scientist v2 `arXiv:2504.08066` | Tree-search version; required for the workshop-accepted paper |
| https://github.com/SamuelSchmidgall/AgentLaboratory | 5,579 | AgentRxiv `arXiv:2503.18102` / Agent Laboratory `arXiv:2501.04227` | Three-stage research framework |
| https://github.com/xingyaoww/code-act | 1,651 | CodeAct `arXiv:2402.01030` | The canonical Python-action agent loop |
| https://github.com/openai/mle-bench | 1,525 | MLE-bench `arXiv:2410.07095` | Benchmark + AIDE scaffold reference |
| https://github.com/sjtu-sai-agents/ML-Master | 404 | ML-Master 2.0 `arXiv:2601.10402` | Hierarchical cognitive caching for long-horizon ML engineering |
| https://github.com/OpenAutoCoder/live-swe-agent | 385 | Live-SWE-agent `arXiv:2511.13646` | Self-evolving scaffold |
| https://github.com/METR/ai-rd-tasks | 136 | RE-Bench `arXiv:2411.15114` | 7 R&D tasks + human-expert trajectories |
| https://github.com/MLE-Dojo/MLE-Dojo | 95 | MLE-Dojo `arXiv:2505.07782` | Gym-style ML engineering environment |
| https://github.com/siegelz/core-bench | 73 | CORE-Bench `arXiv:2409.11363` | Reproducibility benchmark + CORE-Agent |
| https://github.com/Anikethh/ResearchGym | 29 | ResearchGym `arXiv:2602.15112` | Containerized end-to-end research envs |
| https://github.com/chchenhui/mlrbench | 30 | MLR-Bench `arXiv:2505.19955` | Open-ended ML research benchmark + LLM judge |
| https://github.com/allenai/asta-bench | 103 | AstaBench `arXiv:2510.21652` | Holistic scientific-research benchmark |
| https://github.com/yunx-z/MLRC-Bench | 8 | MLRC-Bench `arXiv:2504.09702` | Novel ML research competitions |
| https://github.com/du-nlp-lab/lmr-bench | 11 | LMR-Bench `arXiv:2506.17335` | NLP code-reproduction with unit tests |
| https://github.com/StephenQSstarThomas/PRBench-Eval-Handson | 5 | PRBench `arXiv:2603.27646` | Physics paper reproduction |

**Sandbox-execution platforms (no arXiv paper, but used by the systems above):**

| Platform | URL | Use in agent literature |
|---|---|---|
| Docker | docker.com | Default sandbox for OpenHands/OpenDevin, SWE-agent, MLE-Bench |
| E2B | https://e2b.dev | Hosted code-interpreter sandbox; used by several deep-research agents |
| Modal | https://modal.com | Serverless GPU sandbox; common for ML-engineering agents needing transient compute |
| Vercel Sandbox | https://vercel.com/docs/sandbox | Firecracker microVM-based ephemeral sandbox for untrusted agent code (Vercel KB compares to E2B) |
| Daytona | https://daytona.io | Workspace-orchestration sandbox positioned for AI agents (see Northflank comparison, 2026) |
| Sprites.dev | https://sprites.dev | Newer entrant in the AI-agent sandbox category |

Notable third-party comparisons (from web search, not arXiv):
- Superagent — "AI Code Sandbox Benchmark 2026: Modal vs E2B vs Daytona" (https://www.superagent.sh/blog/ai-code-sandbox-benchmark-2026)
- Northflank — "Daytona vs E2B in 2026" (https://northflank.com/blog/daytona-vs-e2b-ai-code-execution-sandboxes)
- Vercel KB — "Vercel Sandbox vs E2B" (https://vercel.com/kb/guide/vercel-sandbox-vs-e2b)
- GitHub — `nibzard/ai-sandbox-benchmark` (open benchmark across platforms)

---

## 5. Open questions noticed while reading

Flagged, not proposed-as-hypotheses. (Hypothesis-smith's job to turn these into testable claims.)

1. **Premature-completion detection is not a first-class concern in any current scaffold I read.** AI Scientist v1, AI Scientist v2, Agent Laboratory, R&D-Agent, ML-Master 2.0 all log "done" when the agent emits a completion token; none I found cross-check that the claimed completion *matches an oracle*. CORE-Bench and PRBench only score *after* the run; they do not feed back during the run.

2. **The "audit trail" idea exists in fragments but not as a system property.** AI Scientist v2 logs trees, ML-Master 2.0 logs traces, MLR-Bench logs rubric scores — but each is internal to one run. No paper I read in this batch describes an audit trail that *survives across runs* with the explicit purpose of capturing rejected hypotheses + lessons. MegaResearcher's CLAUDE.md rule #1 is empirically novel relative to this literature.

3. **Citation verification is treated as a *post-hoc* check (CiteAudit), not a *pre-flight* gate.** All AI-scientist systems I read first generate citations, then optionally verify. The MegaResearcher rule "if `hf_papers paper_details` does not return a paper, the paper does not exist" is a *pre-flight* gate. I did not find a paper that implements this pre-flight design.

4. **RE-Bench's human-budget curve has not been replicated.** RE-Bench found agents beat humans on short budgets but humans catch up over time. No follow-up paper I found re-runs this with a different agent family (e.g. Claude 4.x, GPT-5) on the same tasks. This is a measurement gap, not a design gap.

5. **Tree search vs. wave-orchestrator vs. linear pipeline — no head-to-head.** AI Scientist v2 uses tree search; AI Scientist v1 used linear; Agent Laboratory uses linear-with-checkpoints; ML-Master uses sequential-with-cache. MegaResearcher uses wave-orchestrator (parallel scouts, then critique loop, then synthesist). I found no paper that compares these four architectures on the same task suite.

6. **"In-context locking" (UltraHorizon) and "post-hoc selection bias" (Luo/Kasirzadeh/Shah) may be the same failure mode under different names.** Both describe agents committing early to a belief and then selecting evidence to support it. A unified characterization would be useful but I did not find one in the batch I read.

7. **Sandbox platform choice is treated as an implementation detail.** Across the papers I read, the choice between Docker / E2B / Modal / Vercel Sandbox / Daytona is never benchmarked for impact on agent task success. Third-party comparisons exist (Superagent 2026, Northflank 2026) but compare speed/cost/security, not *agent task success rate*. There may be a real "sandbox latency floor" effect on agent loops that nobody has measured.

8. **The reproduction benchmarks (CORE-Bench, PRBench, LMR-Bench, ReplicatorBench, ResearchGym) define their own oracles independently.** No paper proposes a *shared* oracle interface that an experimentalist worker could implement once. This is an opportunity for the spec's experimentalist design.

---

## 6. Sources

All citations cross-resolve via `hf_papers paper_details`.

- `arXiv:2402.01030` — CodeAct — https://arxiv.org/abs/2402.01030
- `arXiv:2405.15793` — SWE-agent — https://arxiv.org/abs/2405.15793
- `arXiv:2407.16741` — OpenDevin/OpenHands — https://arxiv.org/abs/2407.16741
- `arXiv:2408.06292` — AI Scientist v1 — https://arxiv.org/abs/2408.06292
- `arXiv:2409.11363` — CORE-Bench — https://arxiv.org/abs/2409.11363
- `arXiv:2410.07095` — MLE-bench — https://arxiv.org/abs/2410.07095
- `arXiv:2411.15114` — RE-Bench — https://arxiv.org/abs/2411.15114
- `arXiv:2501.04227` — Agent Laboratory — https://arxiv.org/abs/2501.04227
- `arXiv:2503.18102` — AgentRxiv — https://arxiv.org/abs/2503.18102 (cited as context, not a key paper above)
- `arXiv:2504.08066` — AI Scientist v2 — https://arxiv.org/abs/2504.08066
- `arXiv:2504.09702` — MLRC-Bench — https://arxiv.org/abs/2504.09702
- `arXiv:2505.07782` — MLE-Dojo — https://arxiv.org/abs/2505.07782
- `arXiv:2505.14738` — R&D-Agent — https://arxiv.org/abs/2505.14738
- `arXiv:2505.19955` — MLR-Bench — https://arxiv.org/abs/2505.19955
- `arXiv:2506.17335` — LMR-Bench — https://arxiv.org/abs/2506.17335
- `arXiv:2509.08713` — The More You Automate the Less You See — https://arxiv.org/abs/2509.08713
- `arXiv:2509.21766` — UltraHorizon — https://arxiv.org/abs/2509.21766
- `arXiv:2510.21652` — AstaBench — https://arxiv.org/abs/2510.21652
- `arXiv:2511.03690` — OpenHands SDK — https://arxiv.org/abs/2511.03690
- `arXiv:2511.13646` — Live-SWE-agent — https://arxiv.org/abs/2511.13646
- `arXiv:2601.03315` — Why LLMs Aren't Scientists Yet — https://arxiv.org/abs/2601.03315
- `arXiv:2601.10402` — ML-Master 2.0 — https://arxiv.org/abs/2601.10402
- `arXiv:2602.02475` — AgentRx — https://arxiv.org/abs/2602.02475
- `arXiv:2602.11354` — ReplicatorBench — https://arxiv.org/abs/2602.11354
- `arXiv:2602.15112` — ResearchGym — https://arxiv.org/abs/2602.15112
- `arXiv:2602.23452` — CiteAudit — https://arxiv.org/abs/2602.23452
- `arXiv:2603.19974` — Trojan's Whisper — https://arxiv.org/abs/2603.19974
- `arXiv:2603.27646` — PRBench — https://arxiv.org/abs/2603.27646

Sandbox platform sources (non-arXiv):
- https://www.superagent.sh/blog/ai-code-sandbox-benchmark-2026
- https://northflank.com/blog/daytona-vs-e2b-ai-code-execution-sandboxes
- https://vercel.com/kb/guide/vercel-sandbox-vs-e2b
- https://github.com/nibzard/ai-sandbox-benchmark
