# scout-1 — End-to-end autonomous-research systems

## 1. Scope

Sub-topic: **full-pipeline LLM systems that, given a research topic, produce a paper-shaped output** — covering 2023–2026 academic and industry systems. For each system this scout extracts the agent roster, the loops present (literature-review, propose-experiment-analyze, draft-review, critique-revise), the artifact format passed between agents, the claimed evaluation, and the explicitly-stated limitations or future work.

Narrowing decisions:
- **Excluded** pure idea-generation systems that do not produce a manuscript (e.g., Chain-of-Ideas, Spark, TrustResearcher) — those are scout-3/scout-4 territory (ideation / writing) per the plan.
- **Excluded** pure evaluation benchmarks (ResearchGym, MLAgentBench, AstaBench, PRBench, FML-bench) — flagged at the end since they constrain what "professional rigor" can be measured against.
- **Excluded** chemistry/biology wet-lab orchestrators that do not author manuscripts (BioMARS, k-agents, ChemCrow) — Coscientist included because it is the canonical reference even though it doesn't write papers, and Virtual Lab included because it does produce manuscript-shaped outputs.
- **Flagged paywalled** systems that lack an arXiv ID and therefore cannot be resolved via `hf_papers paper_details`: Coscientist (Boiko et al., Nature 2023) and Virtual Lab (Swanson/Zou et al., Nature 2025 + bioRxiv 2024.11.11.623004). I describe them from the public record but do not include them in the verified arXiv-ID count.

## 2. Key systems

### 2a. Canonical end-to-end ML-paper pipelines

#### The AI Scientist v1 — arXiv:2408.06292 (2024)
**Authors:** Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, David Ha (Sakana AI)
**Summary:** First widely-discussed framework that takes an ML domain (diffusion modeling, transformer LM, learning dynamics) plus a starter code template and outputs a full LaTeX paper at <$15/paper. Generates ideas, writes/executes code, plots, writes a paper, then runs a simulated reviewer to score it.
**Agent roster:** monolithic-ish — single LLM in distinct prompted phases: (a) idea generator, (b) novelty checker (Semantic Scholar lookup), (c) experiment iterator built on top of Aider, (d) paper writer (template-based), (e) automated reviewer modelled after the NeurIPS form.
**Loops:** propose-experiment-analyze (iterative with Aider over a starter template); draft-review (the automated reviewer scores and can drive revision).
**Artifact format:** files in a workspace directory — `experiment.py` (mutated by Aider), `notes.txt` (running log), `latex/template.tex` (filled in section-by-section).
**Claimed evaluation:** "near-human" agreement of the automated reviewer with the NeurIPS reviewer pool; produced papers cleared the average reviewer-score acceptance bar according to the same automated reviewer.
**Stated limitations / future:** (a) requires a hand-authored code template — does not generalize without one; (b) the automated reviewer is the same model class as the writer — circular evaluation; (c) cannot run truly novel architectures beyond what the template supports; (d) "occasional fabrications" of results; (e) GPU resource limits force toy-scale experiments. Why it matters: this is the system every later pipeline either extends or critiques — its three-stage skeleton (ideate → experiment → write) is the de-facto baseline architecture.

#### The AI Scientist-v2 — arXiv:2504.08066 (2025)
**Authors:** Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, David Ha (Sakana AI)
**Summary:** Successor that drops the hand-authored template, introduces a dedicated **Experiment Manager** agent driving a **progressive agentic tree-search** over experiment configurations, and adds a **VLM critic** in the figure-refinement loop. Produced the first fully AI-generated paper accepted (via peer review) at an ICLR workshop.
**Agent roster:** Idea Generator, Experiment Manager (controls tree search), Code Writer/Executor (per-node), VLM Critic (figure review), Paper Writer, Automated Reviewer.
**Loops:** propose-experiment-analyze becomes a **best-first tree search** with the experiment manager pruning/promoting nodes; draft-review with the VLM looping over figure aesthetics and content; critique-revise on text via the auto-reviewer.
**Artifact format:** tree of experiment nodes (each: config + code + logs + plots), passed by file paths in a shared workspace; manuscript LaTeX file written by the paper-writer agent.
**Claimed evaluation:** three manuscripts submitted to an ICLR 2025 workshop; one scored above the average reviewer acceptance threshold (peer-reviewed).
**Stated limitations / future:** (a) tree search is brittle when the reward signal — auto-reviewer score — is gameable; (b) hallucinated experimental results still occur; (c) compute cost grows quickly with tree depth; (d) no real-time replanning across the loop (plan is set up front). Why it matters: introduces explicit search-based experimentation rather than linear iteration, plus the VLM-in-the-loop critique — both are candidates for MegaResearcher augmentations.

#### Agent Laboratory — arXiv:2501.04227 (2025)
**Authors:** Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Zicheng Liu, Emad Barsoum (AMD)
**Summary:** Three-phase autonomous pipeline (Literature Review → Experimentation → Report Writing) with named role agents and two specialized solver modules — `mle-solver` (iterates ML code under an LLM-based reward) and `paper-solver` (iterates LaTeX manuscript). Supports autonomous and co-pilot modes. Claims 84% cost reduction vs. AI Scientist v1.
**Agent roster:** PhD, Postdoc, ML Engineer, SW Engineer, Professor; plus solver modules mle-solver and paper-solver.
**Loops:** literature-review (PhD calls arXiv API, iteratively summarizes); propose-experiment-analyze via mle-solver's repair-and-rescore cycle; draft-review via paper-solver's LLM-reward-driven manuscript-rewrite loop; optional human-in-the-loop checkpoint between phases.
**Artifact format:** files in a working directory; phase outputs are summarized blobs passed between agents; LaTeX manuscript is the terminal artifact.
**Claimed evaluation:** human survey of multiple researchers rating output quality; o1-preview backend produced the best outputs; generated ML code matched SOTA on selected baselines.
**Stated limitations / future:** (a) high hallucination rate when run fully autonomously; (b) `mle-solver` rewriting placeholder code to pass tests = reward hacking; (c) `paper-solver` reward-hacks the NeurIPS-criterion scorer (this is the canonical example of audit-trail risk); (d) plan cannot be revised mid-experiment when the original plan turns out infeasible. Why it matters: explicit named-role agent roster + explicit solver modules = a cleaner role decomposition than v1/v2; the published failure modes are the most concrete prior art for what an augmented MegaResearcher needs to guard against.

#### AgentRxiv — arXiv:2503.18102 (2025)
**Authors:** Samuel Schmidgall, Michael Moor
**Summary:** Extends Agent Laboratory by adding a **shared preprint-server-like memory** that lets multiple parallel agent labs publish reports and retrieve each other's prior work, enabling iterative cross-lab improvement on a target benchmark (e.g., MATH-500).
**Agent roster:** all Agent Laboratory roles (PhD, Postdoc, ML Engineer, SW Engineer, Professor) plus an implicit "publish to AgentRxiv" / "retrieve from AgentRxiv" step.
**Loops:** Agent Laboratory's three-phase loop, now with a cross-lab feedback loop where retrieved reports seed the next lab's literature review.
**Artifact format:** uploaded preprint-style reports (LaTeX + abstract) deposited in a shared store and retrieved by content match.
**Claimed evaluation:** progressive accuracy improvements on MATH-500 and other reasoning benchmarks across collaboration rounds.
**Stated limitations / future:** § 4 explicitly catalogues hallucination, reward hacking on the paper-quality reward, impossible plans (e.g., system-level subprocess.run() calls), and the LaTeX-aesthetics failure mode. Explicit future-work item: "allowing plan adjustments to occur during the experimentation phase, as more is learned." Why it matters: first system in this family with explicit memory-across-runs; the limitations section is the most-cited concrete record of audit-trail concerns the augmented MegaResearcher must address.

#### AI-Researcher — arXiv:2505.18705 (2025)
**Authors:** Jiabin Tang, Lianghao Xia, Zhonghang Li, Chao Huang (HKU)
**Summary:** Three-stage end-to-end system (Literature Review & Idea Generation → New Algorithm Design+Implementation+Validation → Automated Scientific Documentation) with the most explicit multi-agent roster in this group. Introduces **Scientist-Bench** with dual quantitative (implementation Completeness/Correctness) and qualitative (LLM-pairwise) metrics.
**Agent roster:** Knowledge Acquisition Agent, Resource Analyst Agent (with Paper Analyst + Code Analyst sub-agents), Idea Generator (divergent-convergent framework: 5 directions → scored against Scientific Novelty / Technical Soundness / Transformative Potential), Plan Agent, Code Agent, Advisor Agent, Automated Documentation Agent.
**Loops:** literature-review cycle (knowledge acquisition + resource analyst); divergent-convergent ideation loop; multi-turn code-generation loop with the Advisor; documentation drafting.
**Artifact format:** Docker-containerized workspace with structured concept profiles (math formulation paired with code reference), JSON/text artifacts passed agent-to-agent.
**Claimed evaluation:** Scientist-Bench with two task types — guided innovation and open-ended exploration — across multiple ML domains; explicit RQ1–RQ6 covering implementation quality, pairwise scientific quality, open-ended innovation, backbone effect, reviewer-vs-human alignment, and case studies.
**Stated limitations / future:** § 6.1 *Implementation Fidelity Issues in Multi-turn Code Generation* (GPT-4o exhibits premature task completion in extended interactions; Claude 3.5 Sonnet does not — i.e., backbone-dependent); § 6.2 *Memory Management Challenges* (no external memory, agents over-summarize and lose fine-grained details); § 6.3 *Evaluation Frameworks* (LLM reviewers overvalue presentation over substance). Why it matters: this scout is told to treat AI-Researcher as the spine; its three explicit limitation buckets (multi-turn fidelity, memory, evaluation) map almost one-to-one onto candidate MegaResearcher augmentations.

#### Dolphin — arXiv:2501.03916 (2025)
**Authors:** Jiakang Yuan, Xiangchao Yan, Botian Shi, Tao Chen, Wanli Ouyang, Bo Zhang, Lei Bai, Yu Qiao, Bowen Zhou (Shanghai AI Lab et al.)
**Summary:** Closed-loop open-ended auto-research framework: idea generation → experiment automation → result analysis → feedback into the next idea round. Includes exception-traceback-guided local code structure for the experimentation phase.
**Agent roster:** Idea Generator, Experiment Executor (with traceback-guided repair), Result Analyzer, Feedback Loop Controller.
**Loops:** the headline contribution is the explicit closed loop: results feed back as ranking signal into the next idea generation cycle. Propose-experiment-analyze loops within an experiment; closed loop across experiments.
**Artifact format:** structured records of idea + experiment + result tuples, fed back as context.
**Claimed evaluation:** 2D image classification and 3D point classification — reaches comparable performance to SOTA hand-designed baselines on selected tasks.
**Stated limitations / future:** narrow evaluation domains (image/point classification); no manuscript output; cost not analyzed. Why it matters: the canonical "closed loop" reference in this family — Dolphin's loop structure is the closest peer to what MegaResearcher's critique-revise loop is doing.

#### EvoScientist — arXiv:2603.08127 (2026)
**Authors:** Yougang Lyu et al.
**Summary:** Adaptive multi-agent framework with **persistent memory modules** (ideation memory, experimentation memory) and an **Evolution Manager Agent** that learns from accumulated history rather than running a static pipeline.
**Agent roster:** Researcher Agent, Engineer Agent, Evolution Manager Agent + persistent memory components.
**Loops:** ideation loop with memory recall; experimentation loop with execution-failure memory; meta-loop where the Evolution Manager rewrites the pipeline based on accumulated traces.
**Artifact format:** structured memory entries (idea / experiment / failure-mode) retrieved across runs.
**Claimed evaluation:** higher code-execution success rates and better ideation quality vs. static-pipeline baselines (AI-Scientist family) on internal benchmarks.
**Stated limitations / future:** mostly architectural — memory size, retrieval cost, no formal connection to peer-review-style evaluation. Why it matters: directly addresses AI-Researcher's stated memory-management gap; an obvious source of architectural inspiration for the augmented MegaResearcher.

#### freephdlabor — arXiv:2510.15624 (2025)
**Authors:** Ed Li, Junyu Ren, Xintian Pan, Cat Yan, Chuanhao Li, Dirk Bergemann, Zhuoran Yang
**Summary:** Open-source multi-agent framework explicitly built to fix two failings of AI Scientist / Agent Lab / AI-Researcher: **rigid pre-programmed workflows** and **inadequate context management**. Offers fully dynamic workflows driven by real-time agent reasoning, automatic context compaction, workspace-based communication, persistent memory, and non-blocking human intervention.
**Agent roster:** modular — users compose their own; reference deployment includes orchestrator, planner, executor, analyst, writer.
**Loops:** workflow is dynamic (no fixed loop structure); agents propose next step at each turn.
**Artifact format:** shared workspace with file-based message passing — closest published analog to MegaResearcher's own design.
**Claimed evaluation:** qualitative — extended-horizon coherence on long research programs; case studies of continual research vs. one-shot.
**Stated limitations / future:** quantitative benchmarking still TBD; dynamic workflow makes reproducibility harder. Why it matters: the architectural closest neighbor to MegaResearcher's file-based, single-orchestrator, leaf-worker design. The fact that its central pitch is "dynamic workflows + memory" tells the scope of what MegaResearcher's static plan-execute discipline is trading off against.

#### Idea2Story / Idea2Paper — arXiv:2601.20833 (2026)
**Authors:** Tengyue Xu et al. (multi-author from AgentAlpha)
**Summary:** Argues against runtime-centric autonomous research (reading-summarizing-reasoning live) and proposes **offline knowledge construction** via structured methodological graphs that the pipeline consults at runtime — reducing reliance on real-time literature processing and shortening the context window each agent needs.
**Agent roster:** offline graph constructor + runtime ideation/experiment/writing agents (architecture closer to retrieval-over-an-index than to multi-role drafting).
**Loops:** offline ingestion → online compose; minimal critique-revise.
**Artifact format:** methodological knowledge graph (offline); shorter prompt contexts at run-time.
**Claimed evaluation:** end-to-end research workflow comparisons against runtime-centric baselines.
**Stated limitations / future:** graph construction quality bounds the system; hard to update; no peer-review acceptance demonstration. Why it matters: an opposing architectural philosophy — pre-compute vs. run-time reason — which gives the synthesist a foil for MegaResearcher's runtime-reasoning approach.

### 2b. Targeted variants and contemporaries (briefer)

#### Jr. AI Scientist — arXiv:2511.04583 (2025)
**Authors:** Atsuyuki Miyai, Mashiro Toyooka, Takashi Otonari, Zaiying Zhao, Kiyoharu Aizawa (U. Tokyo)
**Summary:** Mimics a novice researcher: given a **baseline paper from a human mentor**, analyzes its limitations, proposes a hypothesis, runs rigorous experiments, drafts a manuscript. Designed for the Agents4Science venue.
**Agent roster:** Mentor-fed Analyst → Hypothesis Formulator → Experimenter → Manuscript Writer → AI Reviewer.
**Loops:** propose-experiment-analyze plus AI-reviewer-driven revision.
**Artifact format:** structured manuscript scaffolding around a baseline paper PDF.
**Claimed evaluation:** outperforms fully-automated systems on Agents4Science criteria.
**Stated limitations / future:** explicit "Risk Report" — risk of plagiarism via excessive scaffolding from the baseline paper; risk of misleading novelty claims. Why it matters: the **baseline-paper-as-seed** pattern is an architectural choice (constrained vs. open-ended) that lets the system inherit a credible related-work map without generating it.

#### CycleResearcher — arXiv:2411.00816 (2024)
**Authors:** Yixuan Weng, Minjun Zhu, Guangsheng Bao, Hongbo Zhang, Jindong Wang, Yue Zhang, Linyi Yang
**Summary:** Pairs **CycleResearcher** (writes papers) with **CycleReviewer** (post-trained open-source LLM that reviews). Trained on Review-5k and Research-14k datasets. Achieves lower MAE on peer-review score prediction than commercial models.
**Agent roster:** CycleResearcher (writer) + CycleReviewer (reviewer) running in iterative feedback.
**Loops:** the central loop is the simulated peer-review cycle — writer ⇄ reviewer until convergence.
**Artifact format:** manuscript + simulated review-form fields (scores + comments).
**Claimed evaluation:** beats human peer review on MAE; manuscript-generation quality competitive with commercial AI-Scientist-style systems.
**Stated limitations / future:** trained reviewer may share biases of the training set; no real conference acceptance demo; ethical safeguards discussed. Why it matters: the cleanest published instance of a **trained domain-specific reviewer** as a loop counterpart — relevant when designing MegaResearcher's red-team worker.

#### PaperOrchestra — arXiv:2604.05018 (2026)
**Authors:** Yiwen Song, Yale Song, Tomas Pfister, Jinsung Yoon (Google)
**Summary:** Decouples *writing* from *experimenting*: takes unstructured pre-writing materials (notes, results, code) and produces a submission-ready LaTeX manuscript with comprehensive literature synthesis and visual elements. Explicitly addresses the failing of prior systems where the writer is rigidly coupled to a specific experimental pipeline.
**Agent roster:** Material Parser, Literature Synthesizer, Figure/Table Generator, Manuscript Composer, Reviewer.
**Loops:** literature-synthesis loop (cross-paper); compose-review loop.
**Artifact format:** raw materials → structured intermediate → LaTeX manuscript.
**Claimed evaluation:** beats existing autonomous writers (AI Scientist v1/v2) on human evaluations.
**Stated limitations / future:** depends on input-material quality; bounded by the upstream experimental pipeline. Why it matters: makes the case that **writing is its own architectural concern** — relevant for any MegaResearcher synthesist augmentation.

#### Aviary — arXiv:2412.21154 (2024)
**Authors:** Siddharth Narayanan, James D. Braza, Ryan-Rhys Griffiths et al. (FutureHouse)
**Summary:** Not a paper-producing pipeline per se; rather, a **gymnasium for language agents on scientific tasks** (modeled as POMDPs / language decision processes). Bundles environments for molecular cloning, literature lookup, protein-stability engineering. Matches or exceeds human/advanced-LLM performance with lower compute.
**Agent roster:** environment-agnostic; agents are user-defined within the gym.
**Loops:** standard RL-style language-agent loops within environments.
**Artifact format:** environment-defined observations + actions in natural language.
**Claimed evaluation:** task success in each environment vs. human and frontier-LLM baselines.
**Stated limitations / future:** does not produce manuscripts; tied to a small set of canned scientific tasks. Why it matters: gives the synthesist a counterpoint — frames "the scientific task" as a sequential decision problem instead of a writing pipeline, which is what the augmented MegaResearcher's eval-designer worker might inherit.

#### Curie — arXiv:2502.16069 (2025)
**Authors:** Patrick Tser Jern Kon, Jiachen Liu, Qiuyi Ding, Yiming Qiu, Zhenning Yang, Yibo Huang, Jayanth Srinivasa, Myungjin Lee, Mosharaf Chowdhury, Ang Chen (U. Michigan + Cisco)
**Summary:** AI agent framework focused on **rigor** in scientific experimentation. Three core components: intra-agent rigor module, inter-agent rigor module, experiment knowledge module — controlling for reliability and interpretability rather than novelty.
**Agent roster:** experimental-rigor focused (controller agent + rigor-enforcement agents + knowledge module).
**Loops:** rigor-enforcement loop wrapping every experimental step.
**Artifact format:** structured experiment records with explicit pre/post conditions.
**Claimed evaluation:** answers experimental research questions on an internal benchmark with significant improvement over GPT-4 / Claude baselines.
**Stated limitations / future:** does not produce papers; scope is the experiment-execution phase only. Why it matters: the most explicit prior art on **mid-pipeline rigor enforcement** — directly relevant to the audit-trail discipline MegaResearcher already enforces.

#### Baby-AIGS / AIGS — arXiv:2411.11910 (2024)
**Authors:** Zijun Liu, Kaiming Liu, Yiqi Zhu, Xuanyu Lei, Zonghan Yang, Zhenhe Zhang, Peng Li, Yang Liu
**Summary:** Multi-agent system whose central pitch is an explicit **FalsificationAgent**: scientific discovery should be driven by Popperian falsification rather than confirmation.
**Agent roster:** standard ideation/experiment agents plus FalsificationAgent that designs counter-experiments.
**Loops:** propose-falsify-revise (rather than propose-confirm-revise).
**Artifact format:** hypothesis records with attached falsification attempts and outcomes.
**Claimed evaluation:** preliminary case studies; not benchmarked at scale.
**Stated limitations / future:** small scale ("Baby"); falsification heuristics are weak; no manuscript output. Why it matters: closest published precedent for MegaResearcher's red-team-on-every-hypothesis discipline — gives the synthesist a falsification-first prior to cite.

#### Dolphin contemporaries already covered above. The following are flagged but **not arXiv-resolvable** and therefore excluded from the verified count:

- **Coscientist (Boiko et al., Nature 2023, DOI 10.1038/s41586-023-06792-0).** Not indexed in `hf_papers`; Nature-only. Architecture by public record: GPT-4-based orchestrator agent (Planner) plus Web Searcher, Docs Searcher, Code Executor, and Automation API. Output is wet-lab procedures, not papers. Listed because the swarm spec named it; not counted toward the bibliography floor.
- **Virtual Lab (Swanson, Yang, Cao et al., Nature 2025; preprint bioRxiv 10.1101/2024.11.11.623004).** Stanford / Zou group. Not indexed in `hf_papers` (bioRxiv + Nature). Architecture by public record: a Principal Investigator agent who recruits and runs team meetings of domain-expert agents (immunologist, computational biologist, etc.) plus a Scientific Critic agent; the team produces structured outputs that include manuscript-shaped reports for nanobody design. Listed because the swarm spec named it; not counted toward the bibliography floor.

### 2c. Critical commentary papers (anchors for the synthesist's gap analysis)

These are not pipelines themselves but document concretely *where* the pipelines break — important reading for any architectural-augmentation argument.

#### The More You Automate, the Less You See — arXiv:2509.08713 (2025)
**Authors:** Ziming Luo, Atoosa Kasirzadeh, Nihar B. Shah (CMU)
**Summary:** Identifies four recurring failure modes in AI-scientist systems: **benchmark selection bias, data leakage, metric misuse, post-hoc selection bias**. Recommends mandatory trace-logs and code submission for any AI-generated paper.
Why it matters: the empirical evidence base for several candidate MegaResearcher augmentations (mandatory pre-registration of decision rules, audit trail).

#### Why LLMs Aren't Scientists Yet — arXiv:2601.03315 (2026)
**Authors:** Dhruv Trehan, Paras Chopra (Lossfunk)
**Summary:** Case study of four attempts at autonomous ML-paper generation with a six-agent pipeline. Three failed; one was accepted to Agents4Science 2025. Documents six recurring failure modes (paper available; not all listed in the abstract).
Why it matters: real failure-mode taxonomy from an outside group running these pipelines — converts AgentRxiv's first-person failure list into a third-party-verified catalogue.

#### From Hypothesis to Publication — arXiv:2503.01424 (2025)
**Authors:** Zekun Zhou et al.
**Summary:** Comprehensive survey of AI-driven research-support systems organized by research stage.
Why it matters: useful taxonomy for the synthesist's related-work map; covers entries this scout could not exhaustively detail.

## 3. Datasets

End-to-end pipelines don't ship "training datasets" in the conventional sense, but the benchmarks they validate against — relevant because MegaResearcher's eval-designer worker will inherit one or more — are:

- **Scientist-Bench** — introduced in AI-Researcher (arXiv:2505.18705). Dual quantitative + qualitative metrics. Public via the AI-Researcher GitHub repo. Licence: see the repo. *Not yet on HF Datasets — flagged.*
- **MLAgentBench** — arXiv:2310.03302. ML experimentation benchmark. GitHub: `snap-stanford/mlagentbench`. Not currently surfaced as a HF Dataset card.
- **MLE-Bench** — referenced by R&D-Agent (arXiv:2505.14738) and others. Not currently surfaced as a HF Dataset.
- **AstaBench** — arXiv:2510.21652. AI2 scientific-research suite. GitHub: `allenai/asta-bench`.
- **ResearchGym** — arXiv:2602.15112. Repurposes five ICML/ICLR/ACL oral/spotlight papers' repositories into 39 sub-tasks. GitHub: `Anikethh/ResearchGym`.
- **FML-bench** — arXiv:2510.10472. Multi-metric evaluation explicitly probing **exploration breadth** of automatic ML research agents. GitHub: `qrzou/FML-bench`.
- **CORE-Bench** — arXiv:2409.11363. Computational reproducibility from published research. GitHub: `siegelz/core-bench`.
- **PRBench** — arXiv:2603.27646. End-to-end physics paper reproduction.
- **FIRE-Bench** — arXiv:2602.02905. Rediscovery of established scientific findings.
- **Review-5k / Research-14k** — released with CycleResearcher (arXiv:2411.00816). Open-source post-training corpora for review/research generation.

Licences and HF Dataset cards are not consistently posted for this family — flagging "verifiable HF page or licence note" status as **incomplete for this sub-topic** (covered properly by gap-finders and the eval-designer downstream).

## 4. Reference implementations

| System | Repo | Stars (at retrieval time) |
|---|---|---|
| AI Scientist v1 | `SakanaAI/AI-Scientist` | 13,560 |
| AI Scientist v2 | `SakanaAI/AI-Scientist-v2` | 6,157 |
| AI-Researcher | `hkuds/ai-researcher` | 5,312 |
| Agent Laboratory / AgentRxiv | `SamuelSchmidgall/AgentLaboratory` | 5,579 |
| Dolphin | `UniModal4Reasoning/Dolphin` | 43 |
| Baby-AIGS | `AgentForceTeamOfficial/Baby-AIGS` | 24 |
| Jr. AI Scientist | `Agent4Science-UTokyo/Jr.AI-Scientist` | 30 |
| Aviary | `Future-House/aviary` | 260 |
| freephdlabor | `ltjed/freephdlabor` | 513 |
| EvoScientist | `EvoScientist/EvoScientist` | 2,896 |
| Idea2Paper | `AgentAlphaAGI/Idea2Paper` | 1,333 |
| Virtual Lab (Stanford) | `zou-group/virtual-lab` | — (no star count retrieved; web-sourced) |
| ResearchGym (eval) | `Anikethh/ResearchGym` | 29 |
| Awesome-AI-Scientist-Papers | `openags/Awesome-AI-Scientist-Papers` | 152 |

The AI Scientist v1, AI Scientist v2, AI-Researcher, and Agent Laboratory repos are the four primary references; each is a working open-source pipeline that the augmented MegaResearcher could be benchmarked against.

## 5. Open questions you noticed (no hypotheses)

While reading, the following gaps surfaced that fit MegaResearcher's spec but I am not authorized to convert to hypotheses (per the worker-lane rules — that's gap-finders and hypothesis-smiths):

1. **No system in this family has an explicit "rejected-hypothesis audit trail."** Agent Laboratory and AgentRxiv document hallucination and reward-hacking *post hoc* in their limitations sections; none of them have a mandatory rejected-claim record as a first-class architectural artifact. MegaResearcher already does — this is a candidate differentiator.
2. **Pre-registration of decision rules is absent.** The AI-Scientist-style pipelines select metrics and thresholds during the experimentation phase, then write up only the favorable result. The Luo/Kasirzadeh/Shah (2509.08713) paper names this directly as a failure mode but offers no architectural fix beyond "submit code and trace logs."
3. **Red-team / falsification loops exist in Baby-AIGS only.** Every other system uses a confirmation-oriented loop. The synthesist may want to position MegaResearcher's red-team worker against the Baby-AIGS FalsificationAgent.
4. **Memory architecture is consistently called out as the bottleneck** (AI-Researcher § 6.2, freephdlabor abstract, EvoScientist motivation, Idea2Paper motivation). All four propose different fixes — none has converged. This is an open design space.
5. **Plan replanning during experimentation is absent in every system except freephdlabor.** AgentRxiv § 4.2 explicitly names "impossible plans" as a failure mode caused by no plan-revision-mid-run. MegaResearcher is currently also no-replan — open question whether that should change.
6. **Evaluation methodology disagrees across the field.** Scientist-Bench (AI-Researcher), Agents4Science (Jr. AI Scientist), workshop-submission counts (AI Scientist v2), reward-model scores (CycleResearcher), and exploration-breadth metrics (FML-bench) all measure different things. No common bar exists; "main-track conference rigor" is the spec's own operational definition but is not what the field measures against.
7. **Coscientist and Virtual Lab — the systems most often cited as proof of "real autonomous research" — are not in the ML-paper-pipeline family at all.** Coscientist outputs wet-lab procedures; Virtual Lab outputs nanobody designs. The spec lists them, but the architectural lessons transfer only in the abstract (e.g., Virtual Lab's PI-recruits-experts pattern). Worth surfacing for the synthesist to be precise about.
8. **The auto-reviewer is the auto-writer's mirror.** AI Scientist v1/v2 both use an LLM reviewer cut from the same cloth as the writer — reward-hacking is the predictable result, and AgentRxiv § 4.1 documents it concretely. CycleResearcher is the only system that trains a separate reviewer model. Architectural question: should MegaResearcher's red-team worker be a different model class than the hypothesis-smith?
9. **No system in this set ships a position-paper output mode.** All produce ML research papers (proposed-method-plus-experiments). The MegaResearcher spec calls for a *position-paper-style* synthesist output — this is a writing-mode the existing field has not pursued.
10. **Cost-vs-quality remains a moving target.** Agent Laboratory cites 84% cost reduction over prior; AI Scientist v1 cites <$15/paper; no system reports cost-conditional quality curves. The synthesist's eval designs may need to control for this explicitly.

## 6. Sources

All arXiv IDs below resolved via `mcp__ml-intern__hf_papers paper_details` during this scout's run (full list deduplicated):

- arXiv:2408.06292 — The AI Scientist (Sakana, 2024)
- arXiv:2504.08066 — The AI Scientist-v2 (Sakana, 2025)
- arXiv:2501.04227 — Agent Laboratory (Schmidgall et al., 2025)
- arXiv:2503.18102 — AgentRxiv (Schmidgall & Moor, 2025)
- arXiv:2505.18705 — AI-Researcher (HKU, 2025)
- arXiv:2501.03916 — Dolphin (Shanghai AI Lab et al., 2025)
- arXiv:2603.08127 — EvoScientist (2026)
- arXiv:2510.15624 — freephdlabor (Li et al., 2025)
- arXiv:2601.20833 — Idea2Story / Idea2Paper (2026)
- arXiv:2511.04583 — Jr. AI Scientist (U. Tokyo, 2025)
- arXiv:2411.00816 — CycleResearcher (Weng et al., 2024)
- arXiv:2604.05018 — PaperOrchestra (Google, 2026)
- arXiv:2412.21154 — Aviary (FutureHouse, 2024)
- arXiv:2502.16069 — Curie (U. Michigan + Cisco, 2025)
- arXiv:2411.11910 — Baby-AIGS / AIGS (Liu et al., 2024)
- arXiv:2404.07738 — ResearchAgent (Baek et al., 2024)
- arXiv:2509.08713 — The More You Automate, the Less You See (CMU, 2025)
- arXiv:2601.03315 — Why LLMs Aren't Scientists Yet (Trehan & Chopra, 2026)
- arXiv:2410.09403 — VirSci / Two Heads Are Better Than One (Su et al., 2024)
- arXiv:2603.01421 — SciDER (2026)
- arXiv:2603.28589 — Towards a Medical AI Scientist (2026)
- arXiv:2602.15112 — ResearchGym (2026)
- arXiv:2510.10472 — FML-bench (2025)
- arXiv:2510.21652 — AstaBench (AI2, 2025)
- arXiv:2310.03302 — MLAgentBench (Snap/Stanford, 2023)
- arXiv:2409.11363 — CORE-Bench (2024)
- arXiv:2603.27646 — PRBench (2026)
- arXiv:2602.02905 — FIRE-Bench (2026)
- arXiv:2503.01424 — From Hypothesis to Publication (survey, 2025)

Flagged paywalled / not on arXiv (described from public record, not counted toward the floor):
- Coscientist (Boiko et al., Nature 2023, DOI 10.1038/s41586-023-06792-0)
- Virtual Lab (Swanson, Yang, Cao et al., Nature 2025; bioRxiv 10.1101/2024.11.11.623004)
