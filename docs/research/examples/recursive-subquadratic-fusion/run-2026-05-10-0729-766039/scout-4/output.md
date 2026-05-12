# Scout 4 — Long-context reasoning benchmarks and documented failure modes

## 1. Scope

This bibliography maps the **evaluation surface** for long-context reasoning and the
**documented failure modes** of current long-context models, in service of the fusion
thesis (TRM-style recursion atop SubQ-style sparse-attention backbones for
long-context reasoning + math/proof/program-synthesis). The scout deliberately
distinguishes **needle retrieval** from **reasoning over needles** for every
benchmark cited, because the fusion's plausible value-add is in the reasoning regime
(multi-hop, latent-structure, depth-of-thought) rather than literal retrieval.

Narrowing decisions:
- Excluded synthetic single-hop NIAH benchmarks where no reasoning is required
  (e.g., the original NIAH probe). RULER and BABILong are **kept** because they
  contain explicit non-retrieval reasoning sub-tasks.
- Excluded vision-language and audio long-context benchmarks (out of scope vs the
  text/code/proof spec).
- Failure-mode papers selected only when the failure is **reproducible** and the
  paper documents an architectural correlate (position bias, attention dilution,
  multi-step composition, etc.).

Notation: **R** = needle retrieval, **R+** = reasoning over needles, **R++** =
multi-hop / latent-structure / agent-trace reasoning.

---

## 2. Benchmarks

### 2A. Multi-document text QA / mixed long-context

#### B1. RULER — *RULER: What's the Real Context Size of Your Long-Context Language Models?*
- arXiv: **2404.06654** (2024); Hsieh et al. (NVIDIA). 13 tasks across 4 categories: NIAH, multi-hop tracing (variable tracking), aggregation, and QA. Goes well beyond simple NIAH.
- HF dataset: `tonychenxyz/ruler-full` (mirror), `self-long/RULER-llama3-1M` (1M-token Llama-3 variant, MIT). Code at github.com/NVIDIA/RULER and github.com/hsiehjackson/ruler.
- License: **Apache-2.0** (NVIDIA repo) / MIT (mirror).
- Retrieval vs reasoning: **mixed and explicitly separable** (NIAH = R; variable-tracking, common-words aggregation = R+ to R++). The paper introduces "effective context length" as the gap between claimed and actual capability.
- Documented failure mode: most models claim 32k–1M but RULER shows they degrade sharply on multi-hop tracing and aggregation well before the claimed limit (e.g., GPT-4 holds at 64k, many open models collapse at 8k–16k effective length).
- Why it matters for the fusion: cleanly separates retrieval from reasoning, so the spec can claim "the fusion specifically lifts variable-tracking accuracy at >32k tokens" rather than NIAH.

#### B2. ∞Bench — *∞Bench: Extending Long Context Evaluation Beyond 100K Tokens*
- arXiv: **2402.13718** (2024); Zhang et al. (OpenBMB, Tsinghua). 12 tasks across synthetic + realistic, en + zh, all >100k tokens.
- HF dataset: `xinrongzhang2022/InfiniteBench`. License: **Apache-2.0**. Code: github.com/OpenBMB/InfiniteBench (MIT).
- Retrieval vs reasoning: **mixed**. Tasks include En.MC (multi-choice over books = R+), Math.Find/Math.Calc (R++), Code.Debug/Code.Run (R++), Retrieve.PassKey/KV/Number (R).
- Documented failure mode: all open-source models scored <30% on the realistic Math.Calc and Code.Run sub-tasks even with 128k+ context. Retrieval sub-tasks saturate; reasoning sub-tasks do not.
- Why it matters: the **Math.Calc** and **Code.Run** sub-tasks are exactly the math/proof/program-synthesis surface the fusion targets.

#### B3. LongBench — *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding*
- arXiv: **2308.14508** (2023); Bai et al. (THUDM). 21 tasks, en + zh, ~5k–15k token contexts.
- HF dataset: `zai-org/LongBench` (formerly THUDM/LongBench). Code: github.com/THUDM/LongBench (**MIT**). The dataset card itself does not pin a license tag, but the upstream repo is MIT — flag that the **dataset license inherits the MIT code license, not a CC tag**.
- Retrieval vs reasoning: **mostly R+** (single/multi-doc QA, summarization, few-shot, code completion, synthetic).
- Documented failure mode: across 8 commercial and 6 open models, scaled position embedding hits a wall around 16k effective tokens; pure context-window extension does not transfer to multi-doc reasoning gains.
- Why it matters: bilingual + 6-task taxonomy gives a quick health-check matrix; mostly mid-context (~10k), so used as a *control* against the >100k regimes.

#### B4. LongBench v2 — *LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks*
- arXiv: **2412.15204** (2024); Bai et al. (THUDM). 503 multiple-choice questions, 8k–2M words, six categories including code-repo and structured-data understanding. Designed so even experts achieve only 53.7% under time pressure.
- HF dataset: `zai-org/LongBench-v2`. License: **Apache-2.0** (per HF tag).
- Retrieval vs reasoning: **R++ throughout** — explicitly designed so retrieval alone is insufficient.
- Documented failure mode: Direct GPT-4o gets 50.1%; o1-preview using inference-time CoT reaches 57.7%, beating humans, but only because long-CoT compute compensates for context degradation. Vanilla long-context use without explicit reasoning fails.
- Why it matters for the fusion: this is the strongest evidence that **inference-time reasoning compute** matters more than raw context window — directly aligned with the recursion-on-top-of-SubQ thesis.

#### B5. HELMET — *HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly*
- arXiv: **2410.02694** (2024); Yen et al. (Princeton). 7 application-centric categories (RAG, citation, reranking, long-doc QA, summarization, ICL, synthetic recall) with controllable lengths up to 128k.
- HF dataset: `princeton-nlp/HELMET`, `chenzizhao/HELMET` (MIT). Code: github.com/princeton-nlp/HELMET.
- Retrieval vs reasoning: **explicitly separated** — synthetic recall sub-task is R; the others are R+/R++.
- Documented failure mode: synthetic NIAH is **not predictive** of downstream task performance — model rankings reorder. Provides a calibrated signal-to-noise diagnostic.
- Why it matters: HELMET is the cleanest "is your long-context benchmark actually measuring reasoning?" meta-evaluation.

#### B6. NoCha — *One Thousand and One Pairs: A "novel" challenge for long-context language models*
- arXiv: **2406.16264** (2024); Karpinska et al. (UMass). 1,001 minimally-different true/false claim pairs over 67 recently-published English novels.
- HF dataset: not on HF (data lives on github.com/marzenakrp/nocha; **MIT**). Many novels are copyrighted, so the repo distributes only annotations; **flag licensing**.
- Retrieval vs reasoning: **R++** — global narrative reasoning, often book-length (~100k–200k tokens). No literal-match shortcuts.
- Documented failure mode: best model (GPT-4o, Claude 3.5) scored ~55% pair accuracy; humans 95%+. Models that ace NIAH at the same length still fail NoCha.
- Why it matters: cleanest demonstration that **needle retrieval ≠ narrative reasoning**.

#### B7. BABILong — *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack*
- arXiv: **2406.10149** (2024); Kuratov et al. 20 reasoning sub-tasks (bAbI tasks: fact chaining, induction, deduction, counting, lists/sets) embedded in 0–10M-token PG-19 haystacks.
- HF dataset: `RMT-team/babilong`, `RMT-team/babilong-1k-samples`. License: **Apache-2.0**.
- Retrieval vs reasoning: **R++** — by construction, the bAbI tasks require k-step reasoning (k up to 5), and the haystack is irrelevant text.
- Documented failure mode: most LLMs use only 10–20% of effective context; recurrent memory transformers handle longer contexts than attention-based ones at equal parameter count.
- Why it matters: this is **the** canonical "reasoning-in-a-haystack" benchmark and the sub-quadratic / recurrent-memory community's reference point.

#### B8. Loong — *Leave No Document Behind: Benchmarking Long-Context LLMs with Extended Multi-Doc QA*
- arXiv: **2406.17419** (2024); Wang et al. Real-world multi-doc QA with 4 task types: Spotlight Locating, Comparison, Clustering, Chain of Reasoning. No filler noise — all docs are relevant.
- HF dataset: not directly on HF; data + code at github.com/mozerwang/loong (**Apache-2.0**).
- Retrieval vs reasoning: **R+/R++** explicitly graded by task. Chain-of-Reasoning is the hardest.
- Documented failure mode: models drop ~20–30 absolute points moving from Spotlight → Chain-of-Reasoning at the same context length, isolating reasoning from retrieval.
- Why it matters: only multi-doc benchmark where every document is relevant — removes the "ignore filler" confound that contaminates RULER-style synthetic tests.

#### B9. Michelangelo / Latent Structure Queries — *Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries*
- arXiv: **2409.12640** (2024); Vodrahalli et al. (Google DeepMind). Defines the LSQ framework where the model must infer a latent structure (graph, code state, list ops) from context, not retrieve.
- HF dataset: not released publicly (Google; **flag — closed eval**); evaluation methodology + scoring code published.
- Retrieval vs reasoning: **R++** by construction.
- Documented failure mode: Gemini 1.5 Pro / GPT-4 / Claude 3 all degrade super-linearly on LSQ tasks beyond 32k. Performance drop is structurally different from NIAH (which they ace).
- Why it matters: provides the strongest theoretical argument that long-context reasoning is **bottlenecked by latent-structure inference**, not retrieval — the exact regime architectural recursion targets.

#### B10. BrowseComp / BrowseComp-Plus — *BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents* / *BrowseComp-Plus*
- arXiv: **2504.12516** (2025); Wei et al. (OpenAI). 1,266 hard-to-find web questions requiring persistent multi-hop browsing. Plus follow-up: arXiv **2508.06600** (2025) BrowseComp-Plus, a curated transparent variant.
- HF dataset: distributed in github.com/openai/simple-evals (**MIT**); BrowseComp-Plus at github.com/texttron/BrowseComp-Plus.
- Retrieval vs reasoning: **R++** (multi-hop browsing) but heavily entangled with tool-use and policy.
- Documented failure mode: GPT-4o without browsing scored 0.6%; with browsing 1.9%. OpenAI's o1-with-browsing reached 51.5% only with extensive tool calls. Demonstrates that long-horizon search ≠ long-context reading.
- Why it matters: agent-trace tail of long-context — agent traces themselves grow into million-token contexts, and recursion + sparse attention may compress them.

### 2B. Code-repository benchmarks

#### B11. SWE-bench — *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*
- arXiv: **2310.06770** (2023); Jimenez et al. (Princeton). 2,294 issue-PR pairs from 12 popular Python repos. Variants: SWE-bench Lite (300), SWE-bench Verified (500, human-validated by OpenAI).
- HF dataset: `princeton-nlp/SWE-bench`, `princeton-nlp/SWE-bench_Lite`, `princeton-nlp/SWE-bench_Verified`. Code license: **MIT**. Dataset itself does not advertise a separate license tag — flag that the data follows the upstream code repo MIT, with each contained repo's original OSS licenses preserved.
- Retrieval vs reasoning: **R++**. Repository = long context (often >100k tokens for full-repo); requires localizing the bug, generating patch, and passing unit tests.
- Documented failure mode: original GPT-4 with retrieved oracle context resolved <2%; with full-repo context, near 0% on full split. SWE-bench Verified saturated faster (~70% by 2025 frontier agents) but Pro variant arXiv 2509.16941 stays <30%.
- Why it matters: program-synthesis surface where **long context + multi-step reasoning compose** — the fusion's argued sweet spot.

#### B12. SWE-bench Pro — *SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?*
- arXiv: **2509.16941** (2025); Deng et al. (Scale AI). 1,865 enterprise-grade problems across 41 actively-maintained repos, multi-language, designed to require substantial multi-file edits.
- License: **MIT** (github.com/scaleapi/SWE-bench_Pro-os).
- Retrieval vs reasoning: **R++**.
- Documented failure mode: Frontier coding agents drop from ~70% (Verified) to <30% on Pro at the same model. Long-horizon edits (>500 LOC across files) are where they fail.
- Why it matters: the closest published proxy for "really long-horizon reasoning over code repository" the spec cares about.

#### B13. RepoBench — *RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems*
- arXiv: **2306.03091** (2023, ICLR 2024); Liu et al. (UCSD). Three sub-tasks: RepoBench-R (retrieval), RepoBench-C (next-line completion), RepoBench-P (pipelined retrieval+completion).
- HF dataset: `tianyang/repobench_python_v1.1`, `tianyang/repobench-c`, `tianyang/repobench-r`, `tianyang/repobench-p`, `tianyang/repobench_java_v1.1`. License: **CC** (HF tag uses generic "cc"; repo card lists **CC-BY-4.0**).
- Retrieval vs reasoning: **explicitly separated** by sub-task. Excellent ablation surface.
- Documented failure mode: P sub-task (pipeline) shows strong compounding error vs the retrieval and completion sub-tasks done in isolation.
- Why it matters: the cleanest split between long-context retrieval and long-context generation in code.

#### B14. Long Code Arena — *Long Code Arena: a Set of Benchmarks for Long-Context Code Models*
- arXiv: **2406.11612** (2024); Bogomolov et al. (JetBrains). Six benchmarks: library-based codegen, CI builds repair, project-level code completion, commit-message gen, bug localization, module summarization.
- HF datasets: `JetBrains-Research/lca-library-based-code-generation`, `lca-bug-localization`, `lca-module-summarization`, `lca-commit-message-generation`, `lca-project-level-code-completion`, `lca-ci-builds-repair`. License: **Apache-2.0** across all six.
- Retrieval vs reasoning: **R+/R++** mix; bug localization + CI repair are R++.
- Documented failure mode: project-level completion accuracy drops sharply when relevant context is non-local (>5 files away in import graph).
- Why it matters: only suite that benchmarks **whole-repo reasoning** at the granularity of CI behavior.

#### B15. LiveCodeBench — *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code*
- arXiv: **2403.07974** (2024); Jain et al. (UC Berkeley). Continually-updated competitive programming problems from LeetCode/AtCoder/CodeForces; sub-tasks include self-repair, code execution, test-output prediction.
- HF dataset: `livecodebench/code_generation_lite`. License: **CC** (HF tag; specific variant not pinned on the card — flag).
- Retrieval vs reasoning: **R++** (algorithmic reasoning) but contexts are short (<2k usually). Used here as **short-context control** for the same reasoning surface.
- Documented failure mode: LiveCodeBench Pro (arXiv 2506.11928) shows LLMs handle implementation-heavy problems but break on nuanced algorithmic reasoning — orthogonal axis to long-context.
- Why it matters: provides a **short-context reasoning baseline** to disambiguate "did context length cause the failure or was it the reasoning itself?"

### 2C. Agent-trace benchmarks

#### B16. τ-bench — *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*
- arXiv: **2406.12045** (2024); Yao et al. (Sierra). Dynamic conversations between simulated user and tool-using agent; pass^k metric for consistency. Domains: retail, airline.
- HF dataset: not on HF; code at github.com/sierra-research/tau-bench (**MIT**). τ²-bench (arXiv 2506.07982) extends with Dec-POMDP user-agent dual-control telecom domain.
- Retrieval vs reasoning: **R++** — agent must hold long state, follow policy, and reason multi-turn.
- Documented failure mode: even GPT-4o resolves only ~50% of retail tasks; pass^8 (consistency) plummets to ~25%, showing agents are randomly correct rather than systematically.
- Why it matters: agent traces become long contexts; pass^k captures the specific kind of brittleness the fusion would target.

#### B17. ScienceAgentBench — *ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery*
- arXiv: **2410.05080** (2024); Chen et al. (Ohio State). 102 tasks from 44 peer-reviewed papers across 4 disciplines; agents must produce a runnable Python program.
- HF dataset: `osunlp/ScienceAgentBench`. License: **CC-BY-4.0** for benchmark data; **MIT** for code; **flag** subset of tasks under upstream rasterio/matminer licenses.
- Retrieval vs reasoning: **R++** — inputs are datasets + research questions; output is a verified Python pipeline.
- Documented failure mode: best agent (Claude-3.5 + self-debug) reaches 34.3%; direct prompting <15%. Failure clusters around long chains of data-prep + analysis steps.
- Why it matters: math/program-synthesis tail of the spec — and the failure pattern (chain-of-tool-calls breakdown) maps cleanly to recursion's claimed advantage.

#### B18. AgentBench — *AgentBench: Evaluating LLMs as Agents*
- arXiv: **2308.03688** (2023); Liu et al. 8 environments: OS, DB, KG, card game, lateral thinking, house-holding, web shopping, web browsing. Multi-turn, open-ended.
- License: **MIT** (github.com/thudm/agentbench).
- Retrieval vs reasoning: **R++**. Long horizons by construction.
- Documented failure mode: open-source models lag commercial ones by 2–3× pass rate; failure spikes on reasoning chains beyond ~10 actions.
- Why it matters: established cross-domain agent baseline; useful for breadth alongside τ-bench's depth.

### 2D. Targeted long-context reasoning probes (smaller, focused)

#### B19. NoLiMa — *NoLiMa: Long-Context Evaluation Beyond Literal Matching*
- arXiv: **2502.05167** (2025); Modarressi et al. (Adobe). NIAH-like but designed so the needle requires **inference**, not literal token match.
- HF dataset: `amodaresi/NoLiMa`. License: **Adobe Research License — non-commercial only**. **Flag**: not OSI-compliant; spec compliance for commercial use blocked.
- Retrieval vs reasoning: **R+** (one inference hop on top of retrieval).
- Documented failure mode: 10/12 frontier models drop >50% of their short-context score by 32k tokens; the gap with literal-NIAH is huge — i.e., models fake long-context capability via literal match.
- Why it matters: gives a **single-hop reasoning** floor. Combined with BABILong (multi-hop) and Michelangelo (latent structure) it forms a tidy reasoning-depth ladder.

#### B20. Needle Threading — *Needle Threading: Can LLMs Follow Threads through Near-Million-Scale Haystacks?*
- arXiv: **2411.05000** (2024); Roberts et al. (Cambridge). Multi-thread NIAH variants where the needle is a thread of references chained through the haystack.
- License: not pinned to a specific OSI license on the paper repo — **flag**.
- Retrieval vs reasoning: **R+** (multi-hop chained retrieval).
- Documented failure mode: 17 long-context LLMs evaluated; effective context for following >5 chained threads is roughly 10–20% of advertised window.
- Why it matters: cleanly probes "depth of needle chain" as a function of length.

#### B21. LongIns — *LongIns: A Challenging Long-context Instruction-based Exam for LLMs*
- arXiv: **2406.17588** (2024); Gavin et al. Three settings: Global Instr+Single Task, Local Instr+Single Task, Local Instr+Multiple Tasks.
- License: not explicitly tagged — **flag**.
- Retrieval vs reasoning: **R++** — multi-hop reasoning is the design centerpiece.
- Documented failure mode: GPT-4 with 128k window scores below GPT-3.5 with 16k on multi-hop sub-tasks. Effective reasoning window is ~16k regardless of advertised length.
- Why it matters: most direct evidence that "long context" claims by frontier vendors ≠ long reasoning ability.

---

## 3. Failure-mode literature

#### F1. Lost in the Middle — *Lost in the Middle: How Language Models Use Long Contexts*
- arXiv: **2307.03172** (2023); Liu et al. (Stanford / Allen).
- Failure: U-shaped accuracy curve as a function of needle position — models attend to start and end, miss the middle.
- Architectures: GPT-3.5/4, Claude, MPT, longchat (decoder-only with various positional schemes).
- Investigated under sparse / subquadratic? **No** — predates most subquadratic LLMs.
- Plausibly addressed by recursion? Partially — recursion lets the model re-read the middle on multiple passes, mechanically counteracting position bias. But sparse attention itself can also induce or mitigate the bias depending on stride pattern (see Gupta et al. 2505.14840 from the spec). Worth testing.

#### F2. NoLiMa (also F-paper) — see B19.
- Failure: models exploit literal token matches between query and needle as a shortcut; without literal match, retrieval collapses far below claimed window.
- Plausibly addressed by recursion? **Yes, plausibly** — recursive re-reads can implement implicit chained inference at lower depth per pass.

#### F3. Hyper-multi-step — *Hyper-multi-step: The Truth Behind Difficult Long-context Tasks*
- arXiv: **2410.04422** (2024); Yu.
- Failure: difficult long-context tasks decompose into "multi-matching retrieval" + "logic-based retrieval" — both are super-linear-in-step cost regardless of context window.
- Architectures: Llama-2/3, Mistral, GPT-4 long.
- Investigated under sparse/SubQ? **No**.
- Plausibly addressed by recursion? **Yes** — recursion provides exactly the iterative composition multi-matching needs.

#### F4. Needle Threading — see B20. Failure-mode angle: effective context shrinks with thread depth.

#### F5. Michelangelo / LSQ — see B9. Failure-mode angle: latent-structure inference scales worse than retrieval.

#### F6. Test-Time Scaling Is Not Effective for Knowledge-Intensive Tasks — *Test-Time Scaling in Reasoning Models Is Not Effective for Knowledge-Intensive Tasks Yet*
- arXiv: **2509.06861** (2025); Zhao et al.
- Failure: test-time CoT compute reduces hallucination on math but **increases overconfident errors** on knowledge-intensive QA. Long context aggravates this when distractors are present.
- Architectures: o1-style reasoning models, DeepSeek-R1.
- Investigated under sparse/SubQ? **No**.
- Plausibly addressed by recursion? **Mixed** — naive recursion that just reruns CoT could *worsen* this. Recursion combined with explicit retrieve-then-reason might help.

#### F7. Long Context vs RAG — *Long Context vs. RAG for LLMs: An Evaluation and Revisits*
- arXiv: **2501.01880** (2025); Li et al.
- Failure: long context outperforms RAG on Wikipedia QA but RAG wins on dialogue/general queries — long context is not strictly dominant.
- Plausibly addressed by recursion? **Indirectly** — recursion that selects re-attention regions is functionally close to a learned RAG.

#### F8. Long Context, Less Focus (Privacy/Personalization) — *Long Context, Less Focus: A Scaling Gap in LLMs Revealed through Privacy and Personalization*
- arXiv: **2602.15028** (placeholder — paper appears as Feb-26 in HF; treat as recent 2026 work). Documents **attention dilution** as the explanatory mechanism for long-context personalization failures, in fixed-capacity transformers.
- Architectures: Llama-3, GPT-4 class.
- Investigated under sparse/SubQ? **No**.
- Plausibly addressed by recursion? **Yes** — recursion can implement iterative re-focusing; attention dilution is precisely the failure recursive sharpening could combat.

#### F9. Context Denoising — *Revisiting Long-context Modeling from Context Denoising Perspective*
- arXiv: **2510.05862** (2025); Liu et al.
- Failure: contextual noise reduces attention mass on critical tokens; introduces Integrated Gradient score to quantify. Trains "Context Denoising Training" to mitigate.
- Architectures: Llama-3, Qwen2 long.
- Investigated under sparse/SubQ? **Partial** — touches sparse-attention sketch experiments.
- Plausibly addressed by recursion? **Yes** — IG-targeted denoising is essentially a learned recursion over re-weighting.

#### F10. Score Dilution at Test Time — *Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs*
- arXiv: **2512.13898** (2025).
- Failure: static self-attention score dilution worsens monotonically with length; test-time gradient updates outperform static long-context scaling.
- Architectures: long-context Llama derivatives.
- Plausibly addressed by recursion? **Yes** — test-time training and recursion are deeply related (both are length-amortized inference compute).

#### F11. Position Bias Emergence — *On the Emergence of Position Bias in Transformers*
- arXiv: **2502.01951** (2025).
- Failure: causal masking + positional encodings together create predictable position biases via graph-theoretic structure of attention.
- Architectures: standard transformer attention; framework agnostic.
- Investigated under sparse/SubQ? **Yes — the graph-theoretic argument extends to sparse masks**. This is the most directly load-bearing paper for whether SubQ + recursion compose.
- Plausibly addressed by recursion? **Theoretically yes** — recursion over a fixed-pattern sparse mask can recombine the biased attention graph at multiple effective scales, smoothing out the positional graph distance distortion.

#### F12. Pos2Distill — *Position Bias Mitigates Position Bias: Mitigate Position Bias Through Inter-Position Knowledge Distillation*
- arXiv: **2508.15709** (2025).
- Failure mode + remedy: KD from advantageous to disadvantageous positions to reduce position bias.
- Plausibly addressed by recursion? **Yes** — recursion plays a similar role to position-distillation but at inference time.

#### F13. CoT-with-Path-Supervision for Long Context — *Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision*
- arXiv: **2502.20790** (2025).
- Failure: vanilla CoT degrades as context grows; reasoning-path supervision (LongRePS) helps.
- Plausibly addressed by recursion? **Yes** — recursion = implicit reasoning-path supervision via repeated forward passes.

#### F14. ALR² Retrieve-then-Reason — *ALR^2: A Retrieve-then-Reason Framework for Long-context Question Answering*
- arXiv: **2410.03227** (2024); Li et al.
- Failure: monolithic long-context QA is inferior to explicit retrieve→reason decomposition.
- Plausibly addressed by recursion? **Yes** — the recursion can implement retrieve-then-reason internally.

#### F15. Distracting Effect in RAG — *The Distracting Effect: Understanding Irrelevant Passages in RAG*
- arXiv: **2505.06914** (2025); Levy et al.
- Failure: hard-distractor passages hurt RAG accuracy by up to 7.5%; failures concentrate on *semantically related but irrelevant* documents.
- Plausibly addressed by recursion? **Plausibly** — recursion gives multiple opportunities to discount distractors.

#### F16. Tunnel-Vision in Sequential Reasoning — *ParaThinker: Native Parallel Thinking as a New Paradigm to Scale LLM Test-time Compute*
- arXiv: **2509.04475** (2025).
- Failure: long sequential CoTs exhibit "Tunnel Vision" — once a wrong path is started, models can't escape.
- Architectures: o1-class reasoning LMs.
- Plausibly addressed by recursion? **Mixed** — naive recursion could reinforce tunnel vision; parallel-thinking-style recursion could break it. Important caveat for the fusion.

---

## 4. Datasets summary table

| # | Dataset | HF ID | License | R / R+ / R++ |
|---|---------|-------|---------|-------------|
| 1 | RULER (NVIDIA upstream) | self-long/RULER-llama3-1M (mirror) | Apache-2.0 | R + R+ + R++ |
| 2 | ∞Bench | xinrongzhang2022/InfiniteBench | Apache-2.0 | R + R+ + R++ |
| 3 | LongBench | zai-org/LongBench | MIT (code repo); HF tag unset — flag | R+ |
| 4 | LongBench-v2 | zai-org/LongBench-v2 | Apache-2.0 | R++ |
| 5 | HELMET | princeton-nlp/HELMET | MIT | R + R+ + R++ |
| 6 | NoCha | (no HF; github.com/marzenakrp/nocha) | MIT for code; novel texts copyrighted — **flag** | R++ |
| 7 | BABILong | RMT-team/babilong | Apache-2.0 (code); upstream PG-19 Apache-2.0; bAbI BSD | R++ |
| 8 | Loong | (no HF; github.com/mozerwang/loong) | Apache-2.0 | R+ + R++ |
| 9 | Michelangelo / LSQ | not released | **Closed eval — flag** | R++ |
| 10 | BrowseComp | github.com/openai/simple-evals | MIT | R++ |
| 11 | BrowseComp-Plus | github.com/texttron/BrowseComp-Plus | MIT (assumed; verify) — flag | R++ |
| 12 | SWE-bench | princeton-nlp/SWE-bench | MIT (code); dataset HF tag unset; upstream repo licenses preserved — flag | R++ |
| 13 | SWE-bench Verified | princeton-nlp/SWE-bench_Verified | MIT (inherited); HF tag unset — flag | R++ |
| 14 | SWE-bench Pro | github.com/scaleapi/SWE-bench_Pro-os | MIT | R++ |
| 15 | RepoBench | tianyang/repobench_python_v1.1 (+java +c +r +p) | CC-BY-4.0 | R / R+ / R++ separable |
| 16 | Long Code Arena (6 sub-datasets) | JetBrains-Research/lca-* | Apache-2.0 | R+ / R++ |
| 17 | LiveCodeBench | livecodebench/code_generation_lite | CC (variant unpinned — flag) | R++ (short context control) |
| 18 | τ-bench / τ²-bench | (no HF; github.com/sierra-research/tau-bench) | MIT | R++ |
| 19 | ScienceAgentBench | osunlp/ScienceAgentBench | CC-BY-4.0 (data) + MIT (code); some tasks under upstream rasterio/matminer — flag | R++ |
| 20 | AgentBench | github.com/thudm/agentbench | MIT | R++ |
| 21 | NoLiMa | amodaresi/NoLiMa | **Adobe Research License — non-commercial only — flag** | R+ |
| 22 | Needle Threading | (paper repo) | unspecified — flag | R+ |
| 23 | LongIns | (paper repo) | unspecified — flag | R++ |

---

## 5. Reference implementations (selected, with stars)

- **RULER** — github.com/NVIDIA/RULER (1.5k+ stars; canonical eval); github.com/hsiehjackson/ruler (1532 stars).
- **∞Bench** — github.com/openbmb/infinitebench (383 stars).
- **LongBench / LongBench-v2** — github.com/THUDM/LongBench (canonical).
- **HELMET** — github.com/princeton-nlp/HELMET.
- **BABILong** — github.com/booydar/babilong (248 stars).
- **NoCha** — github.com/marzenakrp/nocha.
- **Loong** — github.com/mozerwang/loong (152 stars).
- **SWE-bench** — github.com/SWE-bench/SWE-bench (>3k stars).
- **SWE-bench Pro** — github.com/scaleapi/SWE-bench_Pro-os (378 stars).
- **RepoBench** — github.com/Leolty/repobench (204 stars).
- **Long Code Arena** — github.com/jetbrains-research/lca-baselines (39 stars).
- **τ-bench** — github.com/sierra-research/tau-bench (1215 stars); τ²-bench (1144).
- **ScienceAgentBench** — github.com/osu-nlp-group/scienceagentbench (136 stars).
- **AgentBench** — github.com/thudm/agentbench (3403 stars).
- **NoLiMa** — github.com/adobe-research/NoLiMa (193 stars).
- **BrowseComp** — github.com/openai/simple-evals (4480 stars).
- **Lost-in-the-Middle** — github.com/nelson-liu/lost-in-the-middle (378 stars).
- **Hyper-multi-step / hard retrieval** — github.com/yuyijiong/hard_retrieval_for_llm.
- **Context Denoising Training** — github.com/LCM-Lab/context-denoising-training.

---

## 6. Open questions noticed (NO hypotheses — questions only)

1. **None of F1/F3/F11/F12/F13 have been evaluated on subquadratic-attention backbones at >100k tokens.** Whether sparse-attention regimes preserve the U-shape of "lost in the middle" or deform it (e.g., dilation-stride attention may instead create periodic dips) is empirically open.
2. **No benchmark currently varies recursion depth as an explicit axis** while holding context length fixed. RULER, ∞Bench, BABILong all vary length; none vary "number of internal reasoning passes."
3. **Michelangelo's LSQ tasks have not been re-implemented openly** — the strongest reasoning-over-needle separability comes from a closed Google eval. This is a tooling gap.
4. **NoCha's licensing forbids redistributing the novels.** Fusion experiments needing NoCha must use the official annotation set + sourced books locally; this is an operational constraint the spec should note.
5. **NoLiMa's non-commercial-only license** prevents inclusion in any commercial-track eval — flag for spec compliance ("open data only" intent).
6. **The retrieval-vs-reasoning split is not standardized across benchmarks.** RULER labels are clean; LongBench is muddled; ∞Bench partially separates; SWE-bench is treated as monolithic. A unified taxonomy would benefit any cross-benchmark fusion claim.
7. **τ-bench and ScienceAgentBench measure agent traces, not reading comprehension.** It is unclear whether agent-trace failure modes are isomorphic to long-context-reading failure modes, or governed by different mechanisms (state-tracking vs attention dilution).
8. **The "tunnel vision" failure (F16) cuts against naive recursion.** Whether TRM-style recursion inherits or escapes tunnel vision is empirically open.
9. **Hyper-multi-step (F3) shows multi-matching retrieval is super-linear in step count.** No paper has tested whether sparse attention's cost savings free up budget for more reasoning passes (the fusion's implicit claim).
10. **No paper triangulates SWE-bench Pro × long-context degradation × architectural recursion simultaneously.** Each axis has its own literature; none have been jointly studied.

---

## 7. Sources (flat list)

### arXiv IDs cited

- 2404.06654 — RULER
- 2402.13718 — ∞Bench
- 2308.14508 — LongBench
- 2412.15204 — LongBench-v2
- 2410.02694 — HELMET
- 2406.16264 — NoCha
- 2406.10149 — BABILong
- 2406.17419 — Loong (Leave No Document Behind)
- 2409.12640 — Michelangelo / LSQ
- 2504.12516 — BrowseComp
- 2508.06600 — BrowseComp-Plus
- 2310.06770 — SWE-bench
- 2509.16941 — SWE-bench Pro
- 2306.03091 — RepoBench
- 2406.11612 — Long Code Arena
- 2403.07974 — LiveCodeBench
- 2506.11928 — LiveCodeBench Pro (referenced)
- 2406.12045 — τ-bench
- 2506.07982 — τ²-bench (referenced)
- 2410.05080 — ScienceAgentBench
- 2308.03688 — AgentBench
- 2502.05167 — NoLiMa
- 2411.05000 — Needle Threading
- 2406.17588 — LongIns
- 2307.03172 — Lost in the Middle
- 2410.04422 — Hyper-multi-step
- 2509.06861 — Test-Time Scaling Not Effective for Knowledge
- 2501.01880 — Long Context vs RAG
- 2602.15028 — Long Context, Less Focus
- 2510.05862 — Context Denoising Training
- 2512.13898 — Test-Time Training for Long-Context LLMs
- 2502.01951 — On the Emergence of Position Bias
- 2508.15709 — Pos2Distill
- 2502.20790 — LongRePS / CoT-Path Supervision
- 2410.03227 — ALR^2
- 2505.06914 — Distracting Effect in RAG
- 2509.04475 — ParaThinker / Tunnel Vision

### URLs cited

- huggingface.co/datasets/zai-org/LongBench
- huggingface.co/datasets/zai-org/LongBench-v2
- huggingface.co/datasets/xinrongzhang2022/InfiniteBench
- huggingface.co/datasets/RMT-team/babilong
- huggingface.co/datasets/princeton-nlp/HELMET
- huggingface.co/datasets/chenzizhao/HELMET
- huggingface.co/datasets/princeton-nlp/SWE-bench
- huggingface.co/datasets/princeton-nlp/SWE-bench_Lite
- huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
- huggingface.co/datasets/tianyang/repobench_python_v1.1
- huggingface.co/datasets/tianyang/repobench_java_v1.1
- huggingface.co/datasets/tianyang/repobench-c
- huggingface.co/datasets/tianyang/repobench-r
- huggingface.co/datasets/tianyang/repobench-p
- huggingface.co/datasets/JetBrains-Research/lca-library-based-code-generation
- huggingface.co/datasets/JetBrains-Research/lca-bug-localization
- huggingface.co/datasets/JetBrains-Research/lca-module-summarization
- huggingface.co/datasets/JetBrains-Research/lca-commit-message-generation
- huggingface.co/datasets/JetBrains-Research/lca-project-level-code-completion
- huggingface.co/datasets/JetBrains-Research/lca-ci-builds-repair
- huggingface.co/datasets/livecodebench/code_generation_lite
- huggingface.co/datasets/osunlp/ScienceAgentBench
- huggingface.co/datasets/amodaresi/NoLiMa
- github.com/NVIDIA/RULER
- github.com/hsiehjackson/ruler
- github.com/openbmb/infinitebench
- github.com/THUDM/LongBench
- github.com/marzenakrp/nocha
- github.com/booydar/babilong
- github.com/mozerwang/loong
- github.com/SWE-bench/SWE-bench
- github.com/scaleapi/SWE-bench_Pro-os
- github.com/Leolty/repobench
- github.com/jetbrains-research/lca-baselines
- github.com/sierra-research/tau-bench
- github.com/sierra-research/tau2-bench
- github.com/osu-nlp-group/scienceagentbench
- github.com/thudm/agentbench
- github.com/adobe-research/NoLiMa
- github.com/openai/simple-evals
- github.com/texttron/BrowseComp-Plus
- github.com/nelson-liu/lost-in-the-middle
- github.com/yuyijiong/hard_retrieval_for_llm
- github.com/LCM-Lab/context-denoising-training
