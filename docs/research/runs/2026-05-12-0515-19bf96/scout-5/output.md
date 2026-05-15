# scout-5 — Multi-agent critique, debate, and revision loops

## Scope

Generic patterns for multi-agent self-improvement applied to text — debate, reflection, constitutional AI, multi-agent finetuning, self-rewarding, tree-of-thought-style revision, adversarial collaboration. NOT paper-generation specific. Focus 2023–2026, with bias toward empirical magnitude-of-improvement results and null/negative findings that constrain forecasting.

Narrowing decisions: (a) dropped strictly safety/RLHF-style constitutional work that doesn't transfer to research-text revision; (b) prioritized papers that report named numbers on reasoning/factuality/quality benchmarks over pure architectural proposals; (c) deliberately included three papers (Huang et al. 2310.01798, Zhang et al. 2502.08788, Choi et al. 2508.17536) whose findings are *negative* about debate/self-correction, because the hypothesis-smith's forecast must account for null results, not just headlines.

## Key papers

### Group A — Foundational refinement / reflection loops

**1. Self-Refine: Iterative Refinement with Self-Feedback**
arXiv: 2303.17651 (2023). Madaan, Tandon, Gupta, et al.
Same LLM proposes an output, critiques itself, then refines, looping until a stop criterion. No training, no extra data.
- **Pattern:** reflection / iterative self-feedback
- **Task evaluated on:** Code Optimization, Constrained Generation, Dialogue Response Generation, Sentiment Reversal, Acronym Generation, Math Reasoning. Seven tasks total across GPT-3.5, ChatGPT, GPT-4.
- **Magnitude:** GPT-4 + Self-Refine improves Code Optimization from 27.3% to 36.0% (+8.7 absolute). Dialogue Response Generation preference score: 25.4% → 74.6% (+49.2). Across all preference-based tasks, gains of ~20–50 absolute points.
- **What didn't transfer:** Math Reasoning shows only modest gains (~5% with oracle hint, less without). Authors quote: "ChatGPT feedback for 94% of math instances is 'everything looks good'" — the critic fails to find errors.
- **Mechanism stated by authors:** Multiple-pass generation allows the model to "explore the space of possible outputs" and fix misses, but only when error-detection is itself a tractable subtask. Math fails because errors are nuanced and the critic can't localize them.

**2. Reflexion: Language Agents with Verbal Reinforcement Learning**
arXiv: 2303.11366 (2023). Shinn, Cassano, Labash, Gopinath, Narasimhan, Yao.
Agent receives an external reward signal (binary success/failure or environment trace), verbally reflects on the failure, stores the reflection in episodic memory, retries.
- **Pattern:** reflection with external reward + episodic memory
- **Task evaluated on:** AlfWorld (134 household-task envs), HotpotQA (Wikipedia QA), HumanEval / MBPP / LeetcodeHard (code).
- **Magnitude:** +22% on AlfWorld (130/134 tasks solved), +20% on HotpotQA, +11% on HumanEval over strong ReAct / CoT baselines. ReAct-only plateaus at ~22% hallucination rate with "no signs of long-term recovery"; Reflexion + ReAct climbs across 12 trials.
- **What didn't transfer:** Pure CoT-only and CoT(GT)-only "fail to probabilistically improve on any tasks" between trials — i.e., without the external binary signal, the reflection loop is inert. CoT(GT) still misses 39% of HotpotQA, Reflexion closes ~14 of those points.
- **Mechanism stated by authors:** Verbal reflection acts as "self-hints" that distill long failed trajectories into a few sentences carried forward in context — a form of in-context credit assignment that substitutes for weight updates.

**3. Encouraging Divergent Thinking in LLMs through Multi-Agent Debate**
arXiv: 2305.19118 (2023). Liang et al.
Two agents argue in a "tit-for-tat" state with a judge; designed specifically to break the "Degeneration-of-Thought" (DoT) failure mode of self-reflection — where an LLM that has committed to a wrong answer cannot self-correct.
- **Pattern:** adversarial debate / divergent thinking
- **Task evaluated on:** Commonsense Machine Translation (Common MT), Counter-Intuitive Arithmetic Reasoning.
- **Magnitude:** "GPT-3.5-Turbo with MAD can surpass the performance of GPT-4 on Common MT" — i.e., debate lets a weaker model beat a much stronger single-pass model on the target task.
- **What didn't transfer:** Authors find the LLM-based judge "shows a preference to the side with the same LLM as the backbone" — i.e., self-bias when judging debates. Adaptive break and modest tit-for-tat are required; over-aggressive disagreement degrades quality.
- **Mechanism stated by authors:** Self-reflection alone suffers Degeneration-of-Thought (LLM locks into its first answer); a second agent breaks the lock via three routes — corrects distorted thinking, complements rigidity, supplies external feedback.

### Group B — Multi-agent debate (the canonical thread + critique)

**4. Improving Factuality and Reasoning in Language Models through Multiagent Debate**
arXiv: 2305.14325 (2023). Du, Li, Torralba, Tenenbaum, Mordatch. 528 GitHub stars.
Multiple independent LLM instances generate candidate answers, then iteratively read each other's responses and revise. Society-of-minds framing.
- **Pattern:** multi-agent debate (consensus-seeking, parallel agents)
- **Task evaluated on:** Arithmetic (6-number two-digit expressions), GSM8K, Chess move prediction, Biographies (factuality), MMLU.
- **Magnitude:** Biographies factuality 66.0 → 73.8 (+7.8). MMLU 63.9 → 71.1 (+7.2). Chess move validity 29.3 → 45.2 (+15.9). Reflection alone: smaller gains (~+2–9 pts) and sometimes regression. Debate is "compatible with CoT" — gains are additive.
- **What didn't transfer:** All examples evaluated with chatGPT; gains shrink as the underlying model improves (later work shows this collapses entirely; see #6, #7, #10).
- **Mechanism stated by authors:** "All models can initially be wrong but arrive at the correct answer through the debate process" — debate amplifies agreement on majority-evidence answers and prunes idiosyncratic errors.

**5. Tree of Thoughts: Deliberate Problem Solving with LLMs**
arXiv: 2305.10601 (2023). Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan. 5,944 GitHub stars.
Structured search (BFS/DFS) over a tree of intermediate thoughts, with the LLM acting as both proposer and self-evaluator of partial states.
- **Pattern:** tree-of-thought search with self-evaluation
- **Task evaluated on:** Game of 24, Creative Writing, Mini Crosswords.
- **Magnitude:** Game of 24: IO prompting 7.3%, CoT 4.0%, CoT-SC (k=100) 9.0%, ToT (b=1) 45%, ToT (b=5) **74%**. CoT best-of-100 only reaches 49%. ~10x lift over CoT on this task.
- **What didn't transfer:** Tasks where the LLM cannot reliably evaluate partial states (the "sure/maybe/impossible" prompt fails when the domain has no clean partial-evaluation signal). Cost grows with branching factor; appendix shows GPT-3.5 results are much weaker.
- **Mechanism stated by authors:** Token-level left-to-right decoding cannot do lookahead; ToT externalizes lookahead and pruning as an explicit search the LLM can guide.

**6. Large Language Models Cannot Self-Correct Reasoning Yet**
arXiv: 2310.01798 (2023). Huang et al.
Direct empirical attack on the self-correction premise. Distinguishes "oracle-stop" self-correction (use ground-truth to know when to stop refining — the implicit setting in Self-Refine / Reflexion on reasoning) from "intrinsic" self-correction (model decides when to stop).
- **Pattern:** reflection / self-correction (critique, not novel method)
- **Task evaluated on:** GSM8K, CommonSenseQA, HotpotQA across GPT-3.5 and GPT-4.
- **Magnitude:** Intrinsic self-correction *decreases* accuracy across all benchmarks tested (Tables 3–6). Oracle-stop variants get the headline gains; without oracle, performance drops. Also shows multi-agent debate does not outperform self-consistency at matched call budget.
- **What didn't transfer:** Self-correction without an external signal is net-negative on reasoning. This invalidates a large fraction of the headline numbers in Group A when re-evaluated honestly.
- **Mechanism stated by authors:** LLMs cannot reliably detect their own reasoning errors; revision then degrades a correct answer as often as it fixes a wrong one. Gains in prior work come from leakage of ground-truth into the stopping criterion.

**7. Stop Overvaluing Multi-Agent Debate — We Must Rethink Evaluation and Embrace Model Heterogeneity**
arXiv: 2502.08788 (2025). Zhang, Cui, Chen, Wang, Zhang, Wang, Wu, Hu.
Systematic re-evaluation of 5 representative MAD methods × 9 benchmarks × 4 foundation models.
- **Pattern:** multi-agent debate (meta-evaluation)
- **Task evaluated on:** 9 NLP benchmarks (math, reasoning, factual QA, commonsense).
- **Magnitude:** MAD "often fails to outperform single-agent baselines" once Chain-of-Thought and Self-Consistency baselines are matched at equal compute. The headline result is the *absence* of a robust gain. Heterogeneous-model debate (mixing different foundation models as debaters) is the one configuration where MAD survives.
- **What didn't transfer:** Same-model MAD on standard reasoning benchmarks. The implication: most published debate gains were against weak baselines.
- **Mechanism stated by authors:** Homogeneous-debate agents agree too quickly; diversity (model heterogeneity, not just role heterogeneity) is what matters for the debate effect.

**8. Debate or Vote: Which Yields Better Decisions in Multi-Agent LLMs?**
arXiv: 2508.17536 (2025). Choi, Zhu, Li. 73 GitHub stars.
Decomposes MAD into "majority voting over N agents" and "inter-agent debate", isolates the contribution of each.
- **Pattern:** multi-agent debate (component-decomposition critique)
- **Task evaluated on:** Arithmetic, GSM8K, MMLU (Pro-Med, Formal-Logic), HellaSwag, CommonsenseQA, HH-RLHF (7 benchmarks).
- **Magnitude (Qwen2.5-7B):** Single agent 0.7205 avg, Decentralized MAD T=2 0.7377, Sparse MAD T=2 0.7330, Majority Voting **0.7691**. Voting alone beats every debate variant on the average. Centralized MAD T=2 0.6551 (worse than single-agent). On Arithmetic specifically: single 0.8140, MAD T=2 0.7600, voting 0.9900. Same pattern with Llama-3.1-8B: voting 0.7242, debate variants 0.61–0.70.
- **What didn't transfer:** Multi-round debate (T=3, T=5) consistently *worse* than T=2 — extra rounds erode gains. Centralized MAD (which is closest to MegaResearcher's orchestrator-aggregates-workers pattern) is the worst configuration.
- **Mechanism stated by authors:** Most of MAD's headline gain is the ensembling effect of N independent samples; the debate communication channel adds little and can subtract. The martingale analysis (Section 4) formalizes why iterated debate doesn't improve expected correctness over the initial distribution.

### Group C — Self-rewarding, meta-rewarding, training-based loops

**9. Self-Rewarding Language Models**
arXiv: 2401.10020 (2024). Yuan, Pang, Cho, Sukhbaatar, Xu, Weston. 153 upvotes.
Same model is both policy and reward model (LLM-as-a-Judge prompt). Iterative DPO using self-generated preferences from a Llama-2-70B seed.
- **Pattern:** self-rewarding / iterative DPO
- **Task evaluated on:** AlpacaEval 2.0, MT-Bench, ARC, HellaSwag, GSM8K, MMLU, Natural Questions.
- **Magnitude:** AlpacaEval 2.0 win rate vs GPT-4-Turbo: Iter1 9.94% → Iter2 15.38% → Iter3 **20.44%** (Llama-2-70B base). Iter3 beats Claude 2 (17.19%), Gemini Pro (16.85%), GPT-4 0613 (15.76%). Head-to-head vs SFT baseline: 49.2% wins → 62.5% wins by Iter3.
- **What didn't transfer:** GSM8K and reasoning categories show *flat* or *negative* gains across iterations (Table 2 MT-Bench Math/Code/Reasoning: SFT 3.93 → M3 4.17, a much smaller delta than the writing/STEM category 8.60 → 9.10). NLP benchmarks (ARC, HellaSwag, MMLU) are flat-to-down from M1 to M3. Authors explicitly note self-rewarding mainly helps the model "better utilize its existing knowledge."
- **Mechanism stated by authors:** The model's reward-modeling ability and its instruction-following ability co-improve when both share weights, because the LLM-as-a-Judge head is itself an instruction-following task.

**10. Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge**
arXiv: 2407.19594 (2024). Wu, Yuan, Golovneva, Xu, Tian, Jiao, Weston, Sukhbaatar.
Adds a *meta-judge* step: the model judges its own judging. Targets the saturation problem in Self-Rewarding.
- **Pattern:** self-rewarding with meta-critique
- **Task evaluated on:** AlpacaEval 2 (length-controlled), Arena-Hard.
- **Magnitude:** AlpacaEval 2 LC win rate Llama-3-8B-Instruct: ~22.9% → 39.4% (4 iterations); Arena-Hard +8.5 points. Beats Self-Rewarding ablation at every iteration.
- **What didn't transfer:** Without length-control, the model exploits the verbosity bias of LLM-as-a-Judge (the original Self-Rewarding paper also shows length blow-up: M1 1092 → M3 2552 tokens average).
- **Mechanism stated by authors:** Iterating only the policy saturates the reward signal; iterating both policy *and* judge breaks the saturation because the judge keeps gaining discriminative power.

### Group D — Constitutional AI and principle-guided critique

**11. Constitutional AI: Harmlessness from AI Feedback**
arXiv: 2212.08073 (2022). Bai, Kadavath, Kundu, Askell, et al. (Anthropic).
Train a harmless assistant using only a list of natural-language principles + self-critique. Two phases: SL-CAI (model critiques + revises its own responses according to principles) and RL-CAI (preference model trained on AI-judged pairs).
- **Pattern:** constitutional / principle-guided self-critique
- **Task evaluated on:** Harmlessness vs helpfulness Pareto frontier; Anthropic helpful-base assistant.
- **Magnitude:** Not a benchmark-leaderboard paper. The reported result is qualitative — RL-CAI matches or exceeds RLHF on the harmlessness/helpfulness Pareto without any human-labeled harmlessness data, using ~10 principles. (No single "+X EM" number — this is the canonical "doesn't fit the format" entry that the hypothesis-smith still needs to know exists.)
- **What didn't transfer:** Reasoning / factuality gains aren't claimed. Constitutional principles work for *behavioral* targets (harmlessness, tone) better than for *correctness* targets.
- **Mechanism stated by authors:** A short list of human-readable principles is enough to bootstrap a critique model; the critic + reviser pair is the substitute for thousands of human-labeled examples.

### Group E — Limits and resistance

**12. Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback**
arXiv: 2506.11930 (2025). Lin et al. (JHU CLSP).
Controlled experiment: give Claude 3.7 / Llama-3.3-70B / Llama-4 *high-quality external feedback* on wrong answers and let them retry up to 10 times.
- **Pattern:** external feedback / revision loop (limits study)
- **Task evaluated on:** AIME 2024, MATH-500, TriviaQA, PopQA, MMLU, MMLU-Pro, GPQA, two synthetic multiplication tasks.
- **Magnitude:** Even with the *best* feedback mechanism (GPT-4.1-mini reflective feedback with ground-truth access), all frontier models "consistently fall short of the target accuracy" — i.e., they fail to incorporate feedback even when it is clear and correct. Models claim to understand feedback >95% of the time but fail to actually update.
- **What didn't transfer:** Sampling-based mitigations partially help but never close the gap. More confident models are *more* resistant to correction.
- **Mechanism stated by authors:** "Feedback friction" is intrinsic — there is a gap between stated intent to update and actual behavior. Not solved by better prompts or more rounds.

**13. Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning**
arXiv: 2505.24726 (2025). 282 upvotes.
Trains the model to write *better self-reflections* via RL, rewarding reflections that, when conditioned on for a retry, lead to a correct answer.
- **Pattern:** reflection trained by RL (not just prompted)
- **Task evaluated on:** Math equation writing, function-calling tasks (sparse-reward, limited feedback).
- **Magnitude:** Authors report consistent gains when the verifier is sparse / binary — the RL signal teaches the model to write reflections that are actually useful at retry time, not just plausible. (Specific deltas reported on equation-writing and function-calling: +X% improvements; see paper Table 2 — couldn't fully pull the number cleanly here, flagged in verification.md.)
- **What didn't transfer:** Requires a binary outcome signal to compute the reward. Domains without a verifier (research-paper quality, prose) cannot use this directly.
- **Mechanism stated by authors:** Default LLM reflections are linguistically fluent but operationally inert. Training reflections against retry-success teaches the model to write reflections that change the retry distribution.

### Group F — Adjacent: critique-trained models and the critic-vs-policy split

**14. Enabling Scalable Oversight via Self-Evolving Critic (SCRIT)**
arXiv: 2501.05727 (2025). 72 upvotes.
Train a critic via synthetic-data + self-validation. Specifically targets the *critic* side of the critic-actor pair.
- **Pattern:** self-evolving critic / multi-agent finetuning
- **Task evaluated on:** Critique-correction benchmarks, error-identification benchmarks (CriticBench-class).
- **Magnitude:** Authors report "significant improvements" — paper figures show critic accuracy jumping ~10–20 points across iterations on critique-correction and error-identification benchmarks. (Couldn't pull cleaner per-benchmark numbers in this scout pass; flagged.)
- **What didn't transfer:** Requires reference solutions for contrastive bootstrapping; doesn't claim generalization to open-ended quality judgments.
- **Mechanism stated by authors:** Contrastive synthetic data (good critique vs bad critique on the same target) + self-validation lets the critic improve without human labels. Argues that critic capability is the bottleneck in oversight, not policy capability.

### Group G — Domain-adjacent: peer-review-style debate for text quality

**15. Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis**
arXiv: 2502.14767 (2025). Kargupta et al. 19 GitHub stars.
LLM personas representing competing papers debate their novelty claims in a tree structure; designed for literature-review-style comparative analysis.
- **Pattern:** debate tree / role-playing / adversarial collaboration
- **Task evaluated on:** Scientific paper comparison across domains (no single-number reasoning benchmark).
- **Magnitude:** Reports superiority over single-pass and flat-debate baselines on human-rated comparative-analysis quality. Headline is qualitative ("structured analysis", "independent novelty arguments"). No "+X EM" number.
- **What didn't transfer:** Requires that the comparison axis be defined up front (the "personas"); doesn't auto-discover the dimensions on which to compete.
- **Mechanism stated by authors:** Forcing each paper-persona to argue its own novelty independently surfaces specific differences that get smoothed away in single-pass summarization.

## Datasets

Multi-agent / revision-loop work in this scout's window is almost entirely benchmark-reuse, not new dataset construction. The datasets cited above as evaluation surfaces:

- **GSM8K** — grade-school math (Cobbe et al.). HF: `openai/gsm8k`. Licence: MIT.
- **MATH / MATH-500** — competition math. HF: `hendrycks/competition_math`. Licence: MIT.
- **AIME 2024** — competition math; HF dataset under various mirrors (e.g., `HuggingFaceH4/aime_2024`), licence varies by mirror — flag and re-check before any use.
- **HotpotQA** — multi-hop Wikipedia QA. HF: `hotpot_qa`. Licence: CC BY-SA 4.0.
- **AlfWorld** — text-based household tasks; not on HF, MIT licensed via GitHub.
- **HumanEval / MBPP / LeetcodeHard** — code. HumanEval: `openai_humaneval` on HF, MIT. MBPP: `mbpp`, CC BY 4.0.
- **AlpacaEval 2.0** — instruction-following eval. HF: `tatsu-lab/alpaca_eval`. Apache 2.0.
- **MT-Bench / Arena-Hard** — judge-based eval. `lmsys/mt_bench_human_judgments`. Apache 2.0.
- **MMLU / MMLU-Pro / GPQA** — knowledge benchmarks. MMLU: `cais/mmlu`, MIT. GPQA: `Idavidrein/gpqa`, CC BY 4.0 (gated).
- **CommonsenseQA, HellaSwag, TriviaQA, PopQA, NQ** — standard reasoning/factuality suites, all on HF with permissive licences.
- **CriticBench / RealCritic** — critique-evaluation benchmarks. RealCritic: `tangzhy/realcritic` on GitHub; HF dataset mirror exists, MIT.
- **HH-RLHF** — Anthropic helpful-harmless preference data. HF: `Anthropic/hh-rlhf`. MIT.

No new dataset is required to *test* whether adding a debate/reflection/revision loop helps research-paper drafting — existing benchmarks cover reasoning, factuality, and instruction-following. A new dataset *would* be required to measure improvement on "ICLR-rubric-grade research-paper drafts" since no such benchmark exists in this scout's window — flagged as an open question below.

## Reference implementations

- **Self-Refine** — https://github.com/madaan/self-refine (referenced in paper; canonical reference implementation, mid-hundreds of stars in original).
- **Reflexion** — https://github.com/noahshinn/reflexion (very large star count; canonical agent-with-reflection reference).
- **Du et al. multi-agent debate** — https://github.com/composable-models/llm_multiagent_debate (528 stars). Drop-in three-agent two-round debate.
- **Tree of Thoughts** — https://github.com/ysymyth/tree-of-thought-llm (5,944 stars). Canonical ToT with BFS/DFS over thought tree, self-evaluation prompts.
- **Self-Rewarding LMs (community impl)** — https://github.com/gagan3012/self_rewarding_models (13 stars). Smaller-scale reimpl; the original Meta release is also referenced.
- **MARS (role-based debate)** — https://github.com/xwang97/MARS (6 stars).
- **M-MAD (multidimensional MAD for MT eval)** — https://github.com/su-jiayuan/m-mad (23 stars).
- **MAgICoRe (coarse-to-fine multi-agent refinement)** — https://github.com/dinobby/magicore (23 stars).
- **THOUGHTSCULPT (MCTS + revision)** — https://github.com/cyzus/thoughtsculpt (13 stars).
- **Debate-or-Vote (decomposition critique)** — https://github.com/deeplearning-wisc/debate-or-vote (73 stars). Useful as a *baseline harness* — implements decentralized/sparse/centralized MAD plus majority voting under one interface.
- **Tree-of-Debate** — https://github.com/pkargupta/tree-of-debate (19 stars). Scientific-comparison personas + debate tree.
- **CREAM** — https://github.com/raibows/cream (29 stars). Regularized self-rewarding.
- **Self-Correction Survey paper-list** — https://github.com/teacherpeterpan/self-correction-llm-papers (570 stars). Useful as an extended pointer index, not as an implementation.
- **Feedback Friction** — https://github.com/JHU-CLSP/Feedback-Friction (8 stars). Reference harness for "give the model good external feedback and measure how much it actually incorporates" — directly relevant to building MegaResearcher's red-team→revision loop.
- **CorrectBench** — https://github.com/HCR050806/CorrectBench. Benchmark harness for self-correction across methods.
- **RealCritic** — https://github.com/tangzhy/realcritic (14 stars). Closed-loop critique evaluation.

## Open questions you noticed

These are gaps the hypothesis-smith should consider; this scout does *not* propose hypotheses or pick winners.

1. **The debate-vs-vote question is unsettled for *long-form* quality.** Choi et al. (2508.17536) decisively show that for short-answer benchmarks (math, MCQ, factual QA), majority voting beats inter-agent debate, and debate alone adds little. But no paper in this scout's pull evaluates voting-vs-debate for *long-form research-text targets* (e.g., "is this related-work section publication-grade?"). The result may not transfer because long-form quality has no plurality to vote on.

2. **Self-correction is net-negative without an external signal.** Huang et al. (2310.01798), Feedback Friction (2506.11930), and the "self-bias in self-refinement" paper (2402.11436) all converge on this. None of the paper-generation systems cited in the MegaResearcher spec (AI Scientist, Agent Laboratory, etc.) appear to use external verifiers strong enough to break this floor for the *paper-text* path (the *code* path can use unit tests; the prose path cannot).

3. **Heterogeneous-model debate is the one MAD configuration that survives meta-evaluation.** Zhang et al. (2502.08788) call this out explicitly. None of the open MAD implementations linked above default to using *different* foundation models on the debate sides — they all run multiple instances of the same model. Whether MegaResearcher's worker pool is heterogeneous enough (literature-scout vs gap-finder vs hypothesis-smith vs red-team — same base model, different prompts) to capture this gain is an open question.

4. **The critic side is the bottleneck, not the actor side.** SCRIT (2501.05727) and Critique-RL (2510.24320) argue that critic capability lags policy capability in the same model. The implication for MegaResearcher's red-team worker: a vanilla LLM red-teamer may be *weaker* than the hypothesis-smith it is critiquing, which inverts the intended pressure. None of the cited work tests this on long-form research output.

5. **Length / verbosity exploit is the dominant degenerate mode in self-rewarding loops.** Self-Rewarding LMs go from 1092 → 2552 tokens across three iterations without proportional quality gain. Meta-Rewarding patches this only with length-control wrappers. Any MegaResearcher revision loop that uses LLM-as-judge for its stopping criterion must explicitly defend against length blow-up.

6. **No revision-loop paper in this pull evaluates on the "main-track conference paper" target.** The closest is Tree-of-Debate (2502.14767), which evaluates on scientific comparative analysis (a sub-task of related-work writing), not on whole-paper-grade output. The empirical evidence base for "does a debate/reflection loop materially raise the quality of a draft paper" is essentially absent — this is a forecasting gap, not a settled finding.

7. **The "feedback friction" floor is the most important constraint nobody is forecasting against.** Even with oracle-grade feedback and unlimited rounds, frontier models in 2025 cannot fully incorporate corrections. This caps any architectural fix that relies on "just add another revision pass."

## Sources

- arXiv:2303.17651 — Self-Refine
- arXiv:2303.11366 — Reflexion
- arXiv:2305.14325 — Du et al. Multi-Agent Debate
- arXiv:2305.10601 — Tree of Thoughts
- arXiv:2305.19118 — Liang et al. Multi-Agent Debate / Degeneration-of-Thought
- arXiv:2310.01798 — LLMs Cannot Self-Correct Reasoning Yet
- arXiv:2502.08788 — Stop Overvaluing Multi-Agent Debate
- arXiv:2508.17536 — Debate or Vote
- arXiv:2401.10020 — Self-Rewarding Language Models
- arXiv:2407.19594 — Meta-Rewarding Language Models
- arXiv:2212.08073 — Constitutional AI
- arXiv:2506.11930 — Feedback Friction
- arXiv:2505.24726 — Reflect, Retry, Reward
- arXiv:2501.05727 — SCRIT
- arXiv:2502.14767 — Tree-of-Debate
- arXiv:2305.11738 — CRITIC (referenced in Group F context)
- arXiv:2402.11436 — Pride and Prejudice (self-bias in self-refinement; cited in open question 2)
- arXiv:2510.24320 — Critique-RL (cited in open question 4)

GitHub repos linked in §Reference implementations above.
