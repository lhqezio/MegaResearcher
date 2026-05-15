# Scout-2 — Manuscript Drafting and Document-Scale Coherence

## 1. Scope

**Sub-topic (one sentence):** How LLM systems produce long, structured, multi-section documents — covering hierarchical outline-driven drafting, retrieval-augmented manuscript generation, citation-anchored writing, cross-section coherence checking, and document-scale RL — with explicit focus on systems that target research-paper-shaped artifacts (papers, surveys, Wikipedia-like articles) and general long-document techniques that transfer.

**Narrowing decisions:**
- Restricted to 2023–2026 work, with one canonical 2023 paper (FLARE) included because every subsequent retrieval-augmented long-form generator either extends or contrasts with it.
- Kept *both* "purpose-built scientific manuscript" systems (AI-Scientist v1/v2, AI-Researcher, Agent Laboratory, AutoSurvey/SurveyForge/SurveyX/SciSage/SurveyG, Meow, Citegeist) and *general long-document* systems (LongWriter, LongWriter-Zero, RAPID, FLARE, STORM, Co-STORM, LLMxMapReduce-V2) because the swarm needs to assess whether general techniques (DPO-on-long-output, plan-then-write, MapReduce convolutional scaling) transfer to paper-grade output.
- Excluded news/story/novel generation except where the coherence mechanism is directly transferable (StoryWriter excluded; CritiCS-style critic loops referenced inside SciSage entry only).
- Excluded multimodal/poster/slide systems (Auto-Slides, PosterForest, RIDGE, OmniLayout) — out of scope for manuscript text generation.
- Excluded review-generation systems pointed at the *reviewing* side (Reviewer2) because the spec's drafting question is the focus; included evaluation/benchmark papers that name what breaks (HelloBench, LongEval, SurveyBench, SPOT, "Hidden Pitfalls", "17% Gap", LCFO).

## 2. Key papers

Grouped by sub-cluster. All arXiv IDs verified via `hf_papers paper_details`.

### 2A. Outline-driven hierarchical drafting (general long-form)

**[1] Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models ("STORM")** — arXiv: `2402.14207` (2024). Shao, Jiang, Kanell, Xu, Khattab, Lam (Stanford).
STORM decomposes long-form article generation into a pre-writing stage (multi-perspective question asking, retrieval) and a writing stage (outline-then-section). Introduces FreshWiki and human evaluation by Wikipedia editors. Identified failure modes: source bias transfer and over-association of unrelated facts — both directly relevant to a paper-writing pipeline.
**Drafting architecture:** hierarchical (perspectives → outline → sections). **Coherence mechanism:** outline as scaffold; no explicit cross-section consistency pass.
**Why it matters for the spec:** the canonical "pre-writing-stage" formulation that AI-Researcher and AI-Scientist v2 implicitly inherit; the named failure modes (source bias, over-association) are exactly the kind of cross-section breakage main-track reviewers catch.

**[2] Into the Unknown Unknowns / Co-STORM** — arXiv: `2408.15232` (2024). Jiang, Shao, Ma, Semnani, Lam (Stanford).
Collaborative extension of STORM where multiple LM agents converse and a user can join the discourse. Introduces WildSeek dataset. Discourse trace becomes the artifact backbone.
**Drafting architecture:** multi-agent dialogue → discourse trace → article. **Coherence mechanism:** the shared discourse trace acts as a persistent grounding context across sections.
**Why it matters for the spec:** demonstrates that an explicit *shared evidence trace* (which MegaResearcher already has via file-based artifacts) is a coherence primitive — bears on whether the orchestrator's audit trail is reusable as a writeup substrate.

**[3] LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs** — arXiv: `2408.07055` (2024). Bai, Zhang, Lv, Zheng, Zhu, Hou, Dong, Tang, Li.
Identifies that LLMs' effective generation length is bounded by SFT example length; releases AgentWrite (plan-then-write pipeline), LongWriter-6k dataset, LongBench-Write benchmark, and DPO recipe. 1,861 GitHub stars.
**Drafting architecture:** plan → subtask decomposition → per-subtask generation → concatenation. **Coherence mechanism:** none beyond plan adherence; coherence emerges from the plan structure.
**What fails:** factual drift across subtasks, no cross-subtask consistency pass; quality degrades past plan boundary.

**[4] LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning** — arXiv: `2506.18841` (2025). Wu, Bai, Hu, Lee, Li.
RL-only (no synthetic SFT) approach to ultra-long generation, with reward models for length control, writing quality, and structural formatting. Evaluated on WritingBench and Arena-Write.
**Drafting architecture:** monolithic RL-tuned generation (no explicit outline step). **Coherence mechanism:** reward model includes "structural formatting"; document-scale, but not paper-section-aware.
**Why it matters for the spec:** direct evidence that document-scale RL can be done without SFT; relevant to the question of whether MegaResearcher should ever fine-tune (spec's YAGNI fence says no, but this is the relevant prior art for "what would it look like if we did").

**[5] RAPID: Efficient Retrieval-Augmented Long Text Generation with Writing Planning and Information Discovery** — arXiv: `2503.00751` (2025). Gu, Li, Dong, Zhang, Lv, Wang, Lian, Liu, Chen.
Three-stage: preliminary outline generation → attribute-constrained search → plan-guided article generation. Evaluated on FreshWiki-2024. Explicitly targets hallucination, topic incoherence, and latency.
**Drafting architecture:** outline → per-section retrieval (attribute-constrained) → write. **Coherence mechanism:** outline + retrieval plan act as the consistency anchor; no post-hoc cross-section pass.
**What fails:** still struggles with thematic coherence across sections when retrieved attributes overlap.

**[6] LLMxMapReduce-V2: Entropy-Driven Convolutional Test-Time Scaling for Generating Long-Form Articles from Extremely Long Resources** — arXiv: `2504.05732` (2025). Wang et al. (THUNLP).
Long-to-long generation via stacked convolutional scaling layers at test time. 874 GitHub stars.
**Drafting architecture:** map (per-chunk drafts) → convolutional reduce (cross-chunk integration). **Coherence mechanism:** convolutional reduce layer is the cross-section integration step — closest thing in the recent literature to an explicit cross-section consistency operator.
**Why it matters for the spec:** the architecture is mechanically similar to what a synthesist subagent could do over scout outputs (each scout is a "chunk").

### 2B. Scientific-manuscript-specific drafting pipelines

**[7] The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery** — arXiv: `2408.06292` (2024). Lu, Lu, Lange, Foerster, Clune, Ha (Sakana). 13,560 GitHub stars.
End-to-end pipeline: idea → experiment → writeup → automated reviewer. The writeup module uses a fixed LaTeX template and per-section generation, with the automated reviewer providing a feedback loop.
**Drafting architecture:** template-driven, section-by-section. **Coherence mechanism:** the automated reviewer scores the writeup but coherence checks are not explicitly cross-section; mostly per-section LaTeX hygiene.
**What fails:** widely documented hallucinated citations, weak related-work, results section claims not always matching the experiment logs — flagged downstream by the "Hidden Pitfalls" paper (`2509.08713`) and the evaluation in `2502.14297`.

**[8] The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search** — arXiv: `2504.08066` (2025). Yamada, Lange, Lu, Hu, Lu, Foerster, Clune, Ha (Sakana). 6,157 GitHub stars.
Adds progressive agentic tree-search, an experiment-manager agent, and Vision-Language Model integration. First fully AI-generated paper accepted at a peer-reviewed ICLR workshop (AI Safety). Removes the v1 fixed template constraint.
**Drafting architecture:** tree-search over experiment plans → writeup conditioned on accepted branch. **Coherence mechanism:** the experiment manager keeps logs the writeup module reads; coherence is enforced by shared state, not explicit consistency checking.
**What fails:** still relies on a single-pass writeup; v2 paper acknowledges its acceptance was at workshop level, not main track.

**[9] AI-Researcher: Autonomous Scientific Innovation** — arXiv: `2505.18705` (2025). Tang, Xia, Li, Huang (HKU). 5,312 GitHub stars.
End-to-end system (literature review → hypothesis → algorithm implementation → publication-ready manuscript). Documentation is the explicit final stage. Introduces Scientist-Bench (guided innovation + open-ended exploration).
**Drafting architecture:** three-phase documentation, building over implementation artifacts. **Coherence mechanism:** the implementation artifacts (code, experiment logs) serve as ground-truth substrate for writeup — but no explicit abstract↔results consistency check is described.
**Why it matters for the spec:** the assignment explicitly names this system's three-phase documentation; the gap between "documentation stage exists" and "main-track-grade documentation" is exactly the spec's territory.

**[10] Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper** — arXiv: `2511.04583` (2025). Miyai, Toyooka, Otonari, Zhao, Aizawa (UTokyo).
Mimics a novice researcher: given a baseline paper from a human mentor, formulates novel hypotheses and runs the workflow. Authors claim it outperforms fully autonomous systems and is explicit about risks (Agents4Science context).
**Drafting architecture:** baseline-paper-conditioned drafting; the mentor paper sets the writing template and related-work skeleton.
**Coherence mechanism:** structural inheritance from baseline paper.
**Why it matters for the spec:** strongest direct evidence that *grounding the manuscript in a human-authored seed paper* is a coherence/quality lever — relevant to whether MegaResearcher should require a seed paper input mode.

**[11] Agent Laboratory: Using LLM Agents as Research Assistants** — arXiv: `2501.04227` (2025). Schmidgall, Su, Wang, Sun, Wu, Yu, Liu, Liu, Barsoum (AMD).
Three-stage agentic framework (literature review → experimentation → report writing) using o1-preview. Discusses cost reductions and human-feedback loops.
**Drafting architecture:** specialized agents per stage; report-writing agent has access to prior stages' outputs.
**Coherence mechanism:** sequential agent handoff; no documented post-writing consistency check.
**Why it matters for the spec:** the cleanest published analog to MegaResearcher's worker decomposition, and the "report writing" stage's weakness is the obvious gap that motivates the swarm-augmentation question.

### 2C. Survey/literature-review generation (multi-section structured documents)

**[12] AutoSurvey: Large Language Models Can Automatically Write Surveys** — arXiv: `2406.10252` (2024). Wang, Guo, Yao, Zhang, Zhang, Wu, Zhang, Dai, Zhang, Wen et al. 467 GitHub stars.
Two-stage: chunked outline generation under context-window constraints, then section-wise writing. First systematic LLM-survey-generation evaluation framework.
**Drafting architecture:** hierarchical outline → section. **Coherence mechanism:** none beyond outline scaffold; explicitly acknowledges parametric-knowledge constraint failures.

**[13] SurveyForge: On the Outline Heuristics, Memory-Driven Generation, and Multi-dimensional Evaluation for Automated Survey Writing** — arXiv: `2503.04629` (2025). Yan, Feng, Yuan, Xia, Wang, Zhang, Bai. 331 GitHub stars.
Outline heuristics learned from human-written surveys' logical structure, plus a scholar-navigation agent for memory-driven generation. Introduces SurveyBench. Outperforms AutoSurvey on reference, outline, and content quality.
**Drafting architecture:** structure-mimicking outline → memory-augmented section drafting. **Coherence mechanism:** scholar-navigation agent acts as a coherence-and-citation anchor across sections.

**[14] SurveyX: Academic Survey Automation via Large Language Models** — arXiv: `2502.14776` (2025). Liang, Yang, Wang, Tang, Zheng, Niu, Song, Wang, Tang, Xiong et al. 971 GitHub stars. 100 upvotes.
Adds AttributeTree preprocessing, reference retrieval, and a *re-polishing* post-processing pass — explicitly a second-pass document-scale refinement step, which most prior systems lack. Claims approach to human-expert performance on content+citation.
**Drafting architecture:** preprocessing tree → outline → section → re-polish.
**Coherence mechanism:** the re-polish stage is the closest published analog to document-scale consistency editing.
**Why it matters for the spec:** strongest evidence in this cluster that a separate "polish/coherence" pass is a measurable quality lever.

**[15] SurveyG: A Multi-Agent LLM Framework with Hierarchical Citation Graph for Automated Survey Generation** — arXiv: `2510.07733` (2025). Nguyen, Nguyen, N. T., Dang, Dong, Le.
Builds a hierarchical citation graph over the candidate paper set and uses it as the structural backbone for the survey outline; multi-agent validation with LLM-as-judge.
**Drafting architecture:** citation graph → outline → section. **Coherence mechanism:** the citation graph provides cross-section structural relationships, so claims in one section can be checked against claims grounded in connected papers.

**[16] SciSage: A Multi-Agent Framework for High-Quality Scientific Survey Generation** — arXiv: `2506.12689` (2025). Shi, Kou, Li, Tang, Xie, Yu, Wang, Zhou.
"Reflect-when-you-write" paradigm: a hierarchical Reflector agent critically evaluates drafts at outline, section, and document levels. Introduces SurveyScope; reports citation F1 improvements.
**Drafting architecture:** generator + Reflector at three granularities (outline/section/document). **Coherence mechanism:** explicit document-level reflection step — closest analog in this cluster to an end-of-draft cross-section consistency referee.
**Why it matters for the spec:** the hierarchical reflector is exactly the architectural primitive the spec needs to evaluate for paper-grade drafting (does adding a document-level critique step before final emit measurably improve coherence?).

**[17] Meow: End-to-End Outline Writing for Automatic Academic Survey** — arXiv: `2509.19370` (2025). Ma, Shan, Zhao, Xu, Wang.
Metadata-driven (rather than full-text-driven) outline generation, fine-tuned with SFT + RL on a reasoning model. Reports structural fidelity and stylistic coherence improvements.
**Drafting architecture:** metadata → hierarchical outline (then outsourced to a downstream writer). **Coherence mechanism:** trained-in structural coherence via RL reward.

### 2D. Citation-anchored writing and related-work generation

**[18] Citegeist: Automated Generation of Related Work Analysis on the arXiv Corpus** — arXiv: `2503.23229` (2025). Beger, Henneking.
Dynamic RAG over arXiv corpus with embedding → summarization → filtering pipeline, explicitly designed to address hallucinated citations and lack of grounded knowledge base.
**Drafting architecture:** retrieval-anchored, paragraph-by-paragraph drafting with each paragraph tied to cited papers.
**Coherence mechanism:** citation grounding per paragraph; coherence across paragraphs is not explicitly enforced.

**[19] CiteME: Can Language Models Accurately Cite Scientific Claims?** — arXiv: `2407.12861` (2024). Press, Hochlehnert, Prabhu, Udandarao, Press, Bethge.
Benchmark of citation attribution: given an excerpt referencing a paper, can the LM identify the cited paper? Introduces CiteAgent (GPT-4o + search + read). Reveals large gap between LM and human performance.
**Why it matters for the spec:** the cleanest published quantification of how badly current systems hallucinate citations — directly relevant to the spec's "citations resolve or do not exist" discipline rule.

**[20] Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling** — arXiv: `2412.14860` (2024). Li, Ng.
Formulates attributed generation as multi-step reasoning + search; uses Self-Guided MCTS with Progress Reward Models to steer drafting toward citation-supported claims.
**Drafting architecture:** tree-search over draft continuations, with progress rewards favoring citation-grounded paths.
**Coherence mechanism:** the progress reward implicitly enforces local citation consistency.

**[21] Active Retrieval Augmented Generation (FLARE)** — arXiv: `2305.06983` (2023). Jiang, Xu, Gao, Sun, Liu, Dwivedi-Yu, Yang, Callan, Neubig (CMU). 668 GitHub stars.
Forward-looking active retrieval: at each step of long-form generation, predict the upcoming sentence, retrieve based on it if confidence is low, and regenerate. Canonical reference for *during-generation* retrieval.
**Drafting architecture:** generation with interleaved retrieval triggered by low-confidence tokens.
**Coherence mechanism:** retrieval is invoked exactly when the model is about to make an ungrounded claim — closest published analog to *just-in-time evidence anchoring* during manuscript drafting.
**Why it matters for the spec:** included as the older canonical reference because every 2024–2026 manuscript-drafting system either uses FLARE-style triggering or explicitly chooses against it.

### 2E. Benchmarks and evaluation of long-document / paper-scale generation (what fails)

**[22] HelloBench: Evaluating Long Text Generation Capabilities of Large Language Models** — arXiv: `2409.16191` (2024). Que, Duan, He, Mou, Wang, Liu, Rong, Wang, Yang, Zhang et al. Bloom's-Taxonomy-grounded benchmark; companion HelloEval is human-aligned.
**Why it matters:** the cleanest published statement of which long-generation sub-tasks current models actually fail.

**[23] LongEval: A Comprehensive Analysis of Long-Text Generation Through a Plan-based Paradigm** — arXiv: `2502.19103` (2025). Wu, Li, Qu, Ravikumar, Li, Loakman, Quan, Wei, Batista-Navarro, Lin.
Compares direct vs plan-based generation across model sizes; documents performance degradation as length grows even with planning. References LongWriter.
**Why it matters:** the comparative evidence that *plan-based ≠ coherent* — directly informs whether outline-driven is sufficient or whether a post-hoc coherence pass is required.

**[24] SurveyBench: How Well Can LLM(-Agents) Write Academic Surveys?** — arXiv: `2510.03120` (2025). Sun, Zhu, Zhou, Tong, Wang, Fu, Li, Liu, Wu.
Quiz-driven evaluation framework for LLM-generated surveys; rates outline quality, content quality, synthesis granularity, logical coherence, non-textual richness. Identifies systematic deficiencies in DeepResearch-style agents.

**[25] The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems** — arXiv: `2509.08713` (2025). Luo, Kasirzadeh, Shah.
Names four failure modes in AI-Scientist-style systems: benchmark selection, data leakage, metric misuse, post-hoc selection bias. Calls for trace logs and code submission as integrity primitives.
**Why it matters for the spec:** four of MegaResearcher's discipline rules (audit trail, citation resolution, pre-registered decision rules, no silent rejections) directly answer the pitfalls this paper documents — relevant for the synthesist's framing of why MegaResearcher's existing architecture is closer to main-track-ready than the AI-Scientist family.

**[26] Chain-of-Verification Reduces Hallucination in Large Language Models** — arXiv: `2309.11495` (2023). Dhuliawala, Komeili, Xu, Raileanu, Li, Celikyilmaz, Weston (Meta).
CoVe: draft → plan verification questions → answer independently → revise. Evaluated on longform text generation.
**Why it matters for the spec:** canonical reference for the architectural primitive "post-hoc consistency check via independent re-querying" — directly relevant to whether the synthesist or a new "writer" worker should run CoVe-style verification on draft sections before final emit.

## 3. Datasets

| Name | Source | Relevance | License/access flag |
|---|---|---|---|
| FreshWiki | from STORM (`2402.14207`) | Wikipedia-like article generation eval; basis for RAPID's FreshWiki-2024 split | Open via STORM repo; Wikipedia-derived (CC BY-SA implied for source articles) |
| WildSeek | from Co-STORM (`2408.15232`) | Open-ended information-seeking discourse | Open via Co-STORM repo |
| LongWriter-6k | from LongWriter (`2408.07055`) | SFT data for ultra-long output | Open on HF (search `THUDM/LongWriter-6k`); needs licence verification on HF page |
| LongBench-Write | from LongWriter (`2408.07055`) | Long-output evaluation | Open via THUDM/LongWriter repo |
| WritingBench, Arena-Write | from LongWriter-Zero (`2506.18841`) | RL-tuned long-output evaluation | Open via paper repos; licences not flagged in paper abstracts — needs HF check |
| HelloBench | from HelloBench (`2409.16191`) | Hierarchical long-text generation benchmark grounded in Bloom's Taxonomy | Open via `quehry/hellobench` repo (56 stars) |
| LongEval | from LongEval (`2502.19103`) | Plan-vs-direct long-text generation benchmark | Open via `Wusiwei0410/LongEval` |
| LCFO (Long Context and Long Form Output) | `2412.08268` | Gradual-summarization / summary-expansion benchmark | Open per paper; HF page should be verified before adoption |
| OARelatedWork | `2405.01930` | Related-work generation with full-texts of cited papers | Open access source data |
| CiteME | `2407.12861` | Citation-attribution benchmark | Open via `bethgelab/CiteME` (48 stars) |
| Scientist-Bench | from AI-Researcher (`2505.18705`) | Guided-innovation + open-ended exploration benchmark for autonomous research | Released with AI-Researcher repo (5.3k stars); licence governed by repo (check before use) |
| SurveyBench | from SurveyBench (`2510.03120`) and SurveyForge | Quiz-driven survey-quality evaluation | Open via SurveyForge repo (`alpha-innovator/surveyforge`, 331 stars) |
| SurveyScope | from SciSage (`2506.12689`) | Survey-evaluation suite | Open via `FlagOpen/SciSage` |

**Note:** none of these have licences that look problematic for downstream methodology research, but every dataset would need an explicit licence read before any derivative MegaResearcher artifact is shipped. The spec's "open-access preferred" constraint is satisfied for the above with the verification flags noted.

## 4. Reference implementations

| Repo | Paper | Stars | Notes |
|---|---|---|---|
| `sakanaai/ai-scientist` | AI Scientist v1 (`2408.06292`) | 13,560 | Reference for fixed-template LaTeX writeup |
| `SakanaAI/AI-Scientist-v2` | AI Scientist v2 (`2504.08066`) | 6,157 | Reference for tree-search + experiment-manager + writeup |
| `SamuelSchmidgall/AgentLaboratory` | AgentRxiv (`2503.18102`) + Agent Laboratory (`2501.04227`) | 5,579 | Three-stage agentic framework; closest analog to MegaResearcher's decomposition |
| `hkuds/ai-researcher` | AI-Researcher (`2505.18705`) | 5,312 | Three-phase documentation pipeline |
| `thudm/longwriter` | LongWriter (`2408.07055`) | 1,861 | AgentWrite plan-then-write pipeline; SFT recipe |
| `IAAR-Shanghai/SurveyX` | SurveyX (`2502.14776`) | 971 | Includes the re-polish post-processing stage |
| `thunlp/LLMxMapReduce` | LLMxMapReduce-V2 (`2504.05732`) | 874 | Map-reduce-style convolutional long-form generation |
| `jzbjyb/flare` | FLARE (`2305.06983`) | 668 | Active retrieval reference implementation |
| `ltjed/freephdlabor` | freephdlabor (`2510.15624`) | 513 | Dynamic-workflow multi-agent framework; partial analog to MegaResearcher |
| `autosurveys/autosurvey` | AutoSurvey (`2406.10252`) | 467 | Chunked outline + section-wise survey generation |
| `alpha-innovator/surveyforge` | SurveyForge (`2503.04629`) | 331 | Memory-driven survey generation; ships SurveyBench |
| `Babelscape/FENICE` | FENICE (`2403.02270`) | 30 | NLI+claim-extraction factuality metric; transferable to abstract-vs-results checking |
| `Agent4Science-UTokyo/Jr.AI-Scientist` | Jr. AI Scientist (`2511.04583`) | 30 | Baseline-paper-conditioned drafting |
| `bethgelab/CiteME` | CiteME (`2407.12861`) | 48 | Citation-attribution benchmark + CiteAgent |
| `quehry/hellobench` | HelloBench (`2409.16191`) | 56 | Hierarchical long-text evaluation suite |
| `chenneking/citegeist` | Citegeist (`2503.23229`) | 1 | RAG-over-arXiv for related-work generation |
| `FlagOpen/SciSage` | SciSage (`2506.12689`) | 10 | Reflect-when-you-write hierarchical critic |

## 5. Open questions I noticed while reading

Flagging only — not proposing hypotheses. Each is something the literature *doesn't* clearly answer for the spec's main-track-bar target.

1. **No published manuscript-generation system runs an end-to-end abstract↔results consistency check.** SciSage's Reflector is the closest, but reflects "at outline, section, and document levels" without specifying that abstract claims must be triangulated against logged experiment results. Whether a CoVe-style (`2309.11495`) post-hoc verification specifically targeted at abstract/conclusion claims would change main-track viability is unmeasured.
2. **The "re-polishing" step in SurveyX (`2502.14776`) is the only explicit document-scale post-pass in the cluster, and its contribution is reported in aggregate.** No ablation isolates how much of SurveyX's gain over AutoSurvey is the re-polish vs the AttributeTree preprocessing. Direct empirical evidence on whether a separate coherence-pass worker pays off is missing.
3. **Citation-grounded drafting (Citegeist, Think&Cite, CiteME) is per-paragraph, never per-document.** No system in this slice enforces that the related-work section's claims about a cited paper *match* the way that paper is later referenced in the discussion section. Cross-section citation consistency is unaddressed.
4. **None of the AI-Scientist-family papers report manuscript-quality ablations.** v1, v2, and AI-Researcher all describe pipelines and report success rates, but do not vary the writeup-module architecture (template vs free-form, single-pass vs multi-pass, with vs without retrieval) and measure the manuscript-quality delta. The gap between "we have a writeup stage" and "the writeup stage's architecture matters" is unevaluated.
5. **Baseline-paper conditioning (Jr. AI Scientist, `2511.04583`) outperforms fully autonomous systems, but the contribution of the baseline-paper conditioning to *manuscript quality specifically* (vs idea quality) is not isolated.** Whether the writeup gets better because it inherits structure from a real paper or because the underlying research is better is unclear.
6. **Most outline-generation work (Meow, SurveyForge, AutoSurvey) optimises for *outline quality* rather than the downstream manuscript quality the outline produces.** End-to-end optimisation from outline → draft → reviewer-grade output is largely unmeasured; the assumption that better outlines yield better papers is taken for granted.
7. **Document-scale RLHF for *research papers* doesn't exist in the published literature.** LongWriter-Zero (`2506.18841`) is the closest, but its reward model targets generic long-form quality (length control, structural formatting, writing quality), not paper-grade rubrics (novelty articulation, baseline-comparison rigor, ablation completeness, related-work survival under expert review). Whether a reward model trained on accept/reject signals from a venue-shaped rubric would help is open.
8. **The "Hidden Pitfalls" paper (`2509.08713`) lists four failure modes that are largely *about the experimental process*, but the manuscript is where those pitfalls become visible to reviewers.** No work in this slice studies whether a manuscript-stage verifier (reading the trace + the draft together) can catch experimental-stage pitfalls before submission. The link between trace-log discipline and manuscript-stage integrity checking is unexplored.
9. **SurveyBench, HelloBench, LongEval, LCFO measure different things and don't compose.** There is no single document-scale coherence benchmark that a paper-pipeline could optimise against without picking a partial proxy.
10. **The "17% Gap" paper (`2601.17431`) quantifies unresolved-citation rates in AI-assisted survey papers but the unresolved-citation rate for fully autonomous *research papers* (AI Scientist v1/v2 outputs) has not been independently measured at scale.** Anecdotal evidence is plentiful; systematic measurement is missing.

## 6. Sources

All arXiv IDs below were verified via `mcp__ml-intern__hf_papers paper_details` (15 spot-checks) and `search` results during this scout.

- arXiv:2402.14207 — STORM — https://arxiv.org/abs/2402.14207
- arXiv:2408.15232 — Co-STORM — https://arxiv.org/abs/2408.15232
- arXiv:2408.07055 — LongWriter — https://arxiv.org/abs/2408.07055
- arXiv:2506.18841 — LongWriter-Zero — https://arxiv.org/abs/2506.18841
- arXiv:2503.00751 — RAPID — https://arxiv.org/abs/2503.00751
- arXiv:2504.05732 — LLMxMapReduce-V2 — https://arxiv.org/abs/2504.05732
- arXiv:2408.06292 — AI Scientist v1 — https://arxiv.org/abs/2408.06292
- arXiv:2504.08066 — AI Scientist v2 — https://arxiv.org/abs/2504.08066
- arXiv:2505.18705 — AI-Researcher — https://arxiv.org/abs/2505.18705
- arXiv:2511.04583 — Jr. AI Scientist — https://arxiv.org/abs/2511.04583
- arXiv:2501.04227 — Agent Laboratory — https://arxiv.org/abs/2501.04227
- arXiv:2503.18102 — AgentRxiv — https://arxiv.org/abs/2503.18102
- arXiv:2406.10252 — AutoSurvey — https://arxiv.org/abs/2406.10252
- arXiv:2503.04629 — SurveyForge — https://arxiv.org/abs/2503.04629
- arXiv:2502.14776 — SurveyX — https://arxiv.org/abs/2502.14776
- arXiv:2510.07733 — SurveyG — https://arxiv.org/abs/2510.07733
- arXiv:2506.12689 — SciSage — https://arxiv.org/abs/2506.12689
- arXiv:2509.19370 — Meow — https://arxiv.org/abs/2509.19370
- arXiv:2503.23229 — Citegeist — https://arxiv.org/abs/2503.23229
- arXiv:2407.12861 — CiteME — https://arxiv.org/abs/2407.12861
- arXiv:2412.14860 — Think&Cite — https://arxiv.org/abs/2412.14860
- arXiv:2305.06983 — FLARE — https://arxiv.org/abs/2305.06983
- arXiv:2409.16191 — HelloBench — https://arxiv.org/abs/2409.16191
- arXiv:2502.19103 — LongEval — https://arxiv.org/abs/2502.19103
- arXiv:2510.03120 — SurveyBench — https://arxiv.org/abs/2510.03120
- arXiv:2509.08713 — Hidden Pitfalls of AI Scientist Systems — https://arxiv.org/abs/2509.08713
- arXiv:2309.11495 — Chain-of-Verification — https://arxiv.org/abs/2309.11495
- arXiv:2412.08268 — LCFO — https://arxiv.org/abs/2412.08268 (cited in datasets)
- arXiv:2405.01930 — OARelatedWork — https://arxiv.org/abs/2405.01930 (cited in datasets)
- arXiv:2510.15624 — freephdlabor — https://arxiv.org/abs/2510.15624 (cited in reference implementations)
- arXiv:2403.02270 — FENICE — https://arxiv.org/abs/2403.02270 (cited in reference implementations)
- arXiv:2601.17431 — The 17% Gap — https://arxiv.org/abs/2601.17431 (cited in open questions)
- arXiv:2502.14297 — Sakana AI Scientist evaluation — https://arxiv.org/abs/2502.14297 (cited in AI Scientist entry)
