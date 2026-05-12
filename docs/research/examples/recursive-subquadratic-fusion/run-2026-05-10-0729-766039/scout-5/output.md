# Scout-5 — Math / Formal Proof / Program-Synthesis Benchmarks With Long Reasoning Chains

## 1. Scope

Sub-topic (one sentence): I survey benchmarks where the *binding constraint* on success is the length of the reasoning chain (proof depth, scratch-pad steps, multi-file patch reasoning) and where solving an instance typically also requires consuming long context (proof state + library imports, repository code, long natural-language specifications), plus the recent (2024–2026) chain-length-scaling literature that documents how performance degrades or plateaus with chain length.

Narrowing decisions:
- I exclude benchmarks where chain depth is small even at the upper difficulty tail (e.g., GSM8K, plain MATH at <500-token solutions). I include MATH-500 only because it remains the canonical CoT-length-vs-difficulty diagnostic and is referenced by every chain-length-scaling paper below.
- I distinguish CoT-as-text-output (token sequence visible to the model itself) from architectural recursion (depth in the compute graph, e.g., TRM 2510.04871). All benchmarks below stress the *former*, but several (LeanDojo, PutnamBench, SWE-bench, BigCodeBench) also stress *long input context*, which is the axis the fusion thesis cares about.
- For "chain-length × context-length" interaction, only a small handful of papers explicitly cross those axes. I list every one I could verify and flag the rest as "single-axis."
- Recency bias: 2024–2026 primary; older work (miniF2F 2021, ProofNet 2023, CRUXEval 2024, SWE-bench 2023) included only as canonical references.

---

## 2. Key Papers / Benchmarks

### 2A. Math benchmarks (olympiad / competition / extended scratchpad)

**1. PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition** — arXiv:2407.11214 (2024). Tsoukalas et al.
- 1697 hand-built formalizations of 640 Putnam problems in Lean 4 + Isabelle (subset Coq).
- License: Apache-2.0 (Lean 4 / Isabelle), MIT (Coq); informal statements with MAA permission. HF dataset: `harrywsanders/putnambench` (mirror); canonical source GitHub `trishullab/PutnamBench`.
- Typical chain length: very high — Putnam-hard formal proofs commonly run hundreds of tactic steps when fully expanded (Goedel-Prover-V2 reports proofs >1k tokens with planning steps).
- Typical context length: medium-to-high — Lean 4 proof state plus imported Mathlib definitions can push 50–100k tokens when retrieval is wide.
- Degradation flag: Yes — every prover paper above (Goedel-Prover-V2, BFS-Prover-V2, DeepSeek-Prover-V2) reports declining success as ground-truth proof depth grows.
- Separable axes: Partial — chain length varies naturally per theorem; context length is bounded by the imports retrieval policy, so it can be held fixed by fixing the retrieval set.

**2. miniF2F: a cross-system benchmark for formal Olympiad-level mathematics** — arXiv:2109.00110 (2021/2022). Zheng, Han, Polu.
- 488 formalized problems (AIME, AMC, IMO + HS/UG material) in Metamath, Lean, Isabelle, HOL Light. Canonical Olympiad-level prover benchmark; nearly every theorem-proving paper in 2024–2026 reports miniF2F-test pass-rate.
- License: permissive — Apache-2.0 (Lean, Isabelle), MIT (Metamath), FreeBSD (HOL Light). HF mirrors: `AI4M/minif2f_dataset`, `wellecks/minif2f_isabelle`, `Tonic/MiniF2F`. GitHub `openai/miniF2F`, `yangky11/minif2f-lean4`.
- Typical chain length: moderate-to-long — solved proofs in DeepSeek-Prover-V2 average tens of tactic steps; failure is concentrated on theorems requiring >50 step proofs.
- Typical context length: low-moderate — statements are short, but proof state grows when many lemmas are loaded.
- Degradation flag: Yes — see 2502.07640, 2503.04697 reporting accuracy collapsing on long-proof-required theorems.
- Separable axes: Strong — the same statement set has been solved with very different chain lengths under different prompting / RL regimes.

**3. FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI** — arXiv:2411.04872 (2024). Glazer et al.
- Hundreds of original, expert-vetted problems across number theory, real analysis, algebraic geometry, category theory. Designed so that solving a typical problem requires multiple researcher-hours.
- License: held *closed* by Epoch AI (problems not openly redistributed); access is gated. **Flag: NOT an open dataset under the spec's open-data rule.** I include it because every chain-length-scaling claim above FrontierMath-tier difficulty pivots on it.
- Typical chain length: very high — Epoch reports thousands of tokens of scratch in successful o3 solutions.
- Typical context length: low input context (problem statement is short), but the *output* CoT is the dominant token cost.
- Degradation flag: Implicit — reasoning-model gains are concentrated on the *easier* tier (`FrontierMath Tier 1–3`) and saturate at the upper tier.
- Separable axes: Weak — context is small and fixed; only chain length varies.

**4. Omni-MATH: A Universal Olympiad Level Mathematic Benchmark For LLMs** — arXiv:2410.07985 (2024). Gao et al.
- 4,428 olympiad-level problems across 33 sub-domains, 10 difficulty levels (1.0–9.5).
- License: Apache-2.0. HF dataset: `KbsdJames/Omni-MATH`. GitHub `KbsdJames/Omni-MATH`.
- Typical chain length: high — difficulty 7.0+ problems consistently require >2k token CoT in evaluated reasoning models.
- Typical context length: low input; output-dominated.
- Degradation flag: Yes — sharp accuracy drop at difficulty 7.0+ for non-reasoning models; reasoning models compress the gap but still drop above 8.5.
- Separable axes: Difficulty stratification provides a good chain-length proxy; context length is uniform.

**5. OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems** — arXiv:2402.14008 (2024). He et al.
- 8,476 Olympiad math + physics problems (bilingual EN/ZH, multimodal).
- License: Apache-2.0. HF dataset: `Hothan/OlympiadBench`. GitHub `OpenBMB/OlympiadBench` (193 stars).
- Typical chain length: high (math-physics olympiad).
- Typical context length: low-moderate (problem text + diagram tokens).
- Degradation flag: Yes — paper documents "logical fallacies" and "knowledge omissions" that grow with required step count.
- Separable axes: Possible by problem subject (physics vs math) and difficulty.

**6. HARP: A challenging human-annotated math reasoning benchmark** — arXiv:2412.08819 (2024). Yue et al.
- 5,409 problems from US national math competitions (A(J)HSME, AMC, AIME, USA(J)MO); 4,780 with SymPy-checkable answers; multiple-choice options enable inference-time-compute studies.
- License: MIT (GitHub `aadityasingh/harp`).
- Typical chain length: medium-to-high; AIME and USA(J)MO problems require multi-step proofs.
- Typical context length: low.
- Degradation flag: Yes — the paper is explicitly built around inference-time-compute scaling and shows model accuracy declines on USA(J)MO tier.
- Separable axes: Difficulty-stratified along five competition tiers, providing a clean chain-length axis at fixed context.

**7. MathArena: Evaluating LLMs on Uncontaminated Math Competitions** — arXiv:2505.23281 (2025). Balunović et al.
- Continuously updated platform of *uncontaminated* recent contest problems (AIME, USAMO, IMO, etc.); evaluates proof-writing not just final answer.
- License: open evaluation framework (CC-BY-style on platform); paper does not standardize a dataset license — flag as **license: platform-specific, see matharena.ai**.
- Typical chain length: very high — proof-writing problems demand extended scratch.
- Typical context length: low.
- Degradation flag: Yes — the introduction motivates the benchmark by saturation/memorization on AIME 2024 and reports much lower scores on uncontaminated problems and proofs.
- Separable axes: Time-stratified (uncontaminated vs. contaminated splits) and proof-vs-answer split.

**8. MATH-500** — derived split from Hendrycks et al. MATH (subset of 500 problems used as the canonical CoT-length test by OpenAI PRM and most reasoning-LLM evaluations).
- License: HF dataset `HuggingFaceH4/MATH-500` does not state a license on its card. The underlying MATH dataset (Hendrycks et al., NeurIPS 2021, arXiv:2103.03874) is released under MIT. **Flag for spec compliance: MATH-500 derivative card lacks an explicit licence; the parent MIT licence likely applies but should be verified.**
- Typical chain length: moderate — 200–800 tokens for most problems, longer for level-5 hardest.
- Typical context length: low.
- Degradation flag: Yes — Wu et al. 2502.07266 use MATH-500 to demonstrate the inverted-U.
- Separable axes: 5-level difficulty stratification.

### 2B. Formal proof benchmarks (long proof terms; long library context)

**9. LeanDojo: Theorem Proving with Retrieval-Augmented Language Models** — arXiv:2306.15626 (2023). Yang et al.
- Toolkit + benchmark extracted from Mathlib. Proof states + premise pool define a long-context retrieval task. Distributed via Zenodo.
- License: MIT (LeanDojo code); dataset comes with a CC-BY-style license through Mathlib (Mathlib4 is **Apache-2.0**). HF mirror: `tasksource/leandojo`. **Flag: dataset card on HF mirror lacks explicit license; canonical source license is MIT (code) + Apache-2.0 (Mathlib content).**
- Typical chain length: tactic sequences are short individually but a Mathlib4 proof can chain 30–200 tactics.
- Typical context length: very high — full Mathlib4 has tens of thousands of premises; ReProver-style retrieval routinely returns context >50k tokens.
- Degradation flag: Yes — ReProver and successors document accuracy drops on theorems requiring uncommon premises.
- Separable axes: Strong — the retrieval cutoff K provides an explicit context-length knob holding the chain-length distribution roughly fixed.

**10. ProofNet: Autoformalizing and Formally Proving Undergraduate-Level Mathematics** — arXiv:2302.12433 (2023). Azerbayev et al.
- 371 examples (Lean 3 statement + NL statement + NL proof) from undergrad pure math (analysis, algebra, topology).
- License: GitHub `zhangir-azerbayev/proofnet` is permissively licensed (MIT). The paper is the canonical autoformalization eval. Translations to Lean 4 exist in community repos.
- Typical chain length: medium-to-high — undergraduate proofs.
- Typical context length: medium — Mathlib imports.
- Degradation flag: Yes — DeepSeek-Prover-V1.5 / Goedel-Prover both report substantially lower pass-rate on ProofNet than on miniF2F.
- Separable axes: Reasonable — same chain-length distribution evaluated under different premise-retrieval policies.

**11. CoqGym (Learning to Prove Theorems via Interacting with Proof Assistants)** — arXiv:1905.09381 (2019). Yang & Deng.
- 71k Coq proofs, 123 Coq projects. Older but the canonical Coq benchmark for tactic prediction with realistic long library contexts.
- License: MIT (GitHub `princeton-vl/CoqGym`, 417 stars).
- Typical chain length: tactic chains commonly 50–500 steps in real Coq projects.
- Typical context length: very high — full Coq project context plus library imports.
- Degradation flag: Yes — original ASTactic paper reports rapid accuracy decline with required tactic depth.
- Separable axes: Strong — different Coq projects give very different chain-length / context-length profiles.

**12. Goedel-Prover (V1 + V2): A Frontier Model for Open-Source Automated Theorem Proving** — arXiv:2502.07640 (V1, 2025), arXiv:2508.03613 (V2, 2025). Lin et al.
- Open-source SOTA prover that releases its training/evaluation infrastructure on miniF2F + PutnamBench + Lean Workbook. V2 introduces verifier-guided self-correction with explicit chain-of-thought trajectories in Lean 4.
- License: code MIT; dataset mostly Apache-2.0 inheriting from Lean Workbook. GitHub `Goedel-LM/Goedel-Prover` (233 stars), `Goedel-Prover-V2` (167 stars).
- Typical chain length: very long — V2 explicitly does multi-pass long-CoT proof generation.
- Typical context length: medium-to-high (Mathlib4 imports).
- Degradation flag: Yes (and quantified — V2 reports diminishing returns past N self-correction rounds).
- Separable axes: V2's RL framework explicitly varies CoT budget at fixed proof state.

**13. DeepSeek-Prover-V2** — arXiv:2504.21801 (2025). DeepSeek-Prover team.
- Lean 4 prover with recursive subgoal decomposition; reports new SOTA on miniF2F + PutnamBench. Crucially, V2 builds explicit informal-to-formal CoT *over* recursive subgoals — a structural analogue (in software) of the fusion thesis.
- License: code (GitHub `deepseek-ai/deepseek-prover-v2`, 1263 stars) — MIT-style permissive; weights released under DeepSeek License.
- Typical chain length: very high (subgoal CoT + tactic proofs).
- Typical context length: medium-to-high.
- Degradation flag: Implicit — failure modes documented around long subgoal chains.
- Separable axes: Subgoal decomposition depth is an explicit knob.

**14. BFS-Prover-V2: Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers** — arXiv:2509.06493 (2025). Xin et al.
- A multi-agent / planner-enhanced prover that explicitly studies the *training-time vs inference-time* compute scaling of long proof chains, on miniF2F and ProofNet.
- License: paper-only, no public weights at time of writing.
- Typical chain length: explicitly scaled in the paper (planner emits subgoals, step-prover emits tactic chains hundreds of steps deep).
- Typical context length: medium.
- Degradation flag: Yes — the paper's central observation is that RL gains plateau without architectural changes to handle long step sequences.
- Separable axes: Yes — number of MCTS rollouts and tactic depth are independently varied.

**15. FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models** — arXiv:2505.02735 (2025). Yu et al.
- 5,560 Lean 4 statements (HS Olympiad → undergrad), spanning algebra, calculus, number theory, etc., produced via a multi-LLM autoformalization pipeline.
- License: GitHub `Sphere-AI-Lab/FormalMATH-Bench` (77 stars); Apache-2.0 in repo.
- Typical chain length: medium-to-high.
- Typical context length: medium (Mathlib4).
- Degradation flag: Yes — the paper reports pass-rate cliffs on the calculus / number-theory subsets that require longer proof terms.
- Separable axes: Sub-domain stratification.

### 2C. Program-synthesis benchmarks (rich spec / repo context)

**16. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions** — arXiv:2406.15877 (2024). Zhuo et al.
- 1,140 Python tasks each requiring multi-library function-call composition + complex NL spec.
- License: Apache-2.0. HF dataset: `bigcode/bigcodebench`. GitHub `bigcode-project/bigcodebench` (498 stars).
- Typical chain length: medium — solutions require multi-step reasoning across library APIs.
- Typical context length: high — task descriptions are long, and BCB's `Hard` split adds extra spec text.
- Degradation flag: Yes — paper reports that frontier models drop substantially on the `Hard` split which has both longer specs and more required steps.
- Separable axes: Strong — `Hard` vs `Full` split varies spec length; `Instruct` vs `Complete` varies task framing.

**17. CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution** — arXiv:2401.03065 (2024). Gu et al.
- 800 Python functions (3–13 lines) with input/output pairs; tests *execution prediction* — the canonical code-reasoning chain benchmark.
- License: paper authors release under permissive license; HF dataset `cruxeval-org/cruxeval`.
- Typical chain length: short-to-medium — but CoT length scales with loop depth and recursion.
- Typical context length: low.
- Degradation flag: Yes — paper measures CoT-helpful vs CoT-harmful regimes per function.
- Separable axes: Function complexity (loop depth, recursion depth) is an explicit chain-length axis at fixed (low) context.

**18. LiveCodeBench: Holistic and Contamination-Free Evaluation of LLMs for Code** — arXiv:2403.07974 (2024). Jain et al.
- Continuously updated set of LeetCode/AtCoder/CodeForces problems with self-repair, execution, and test-output prediction sub-tasks.
- License: MIT (GitHub `livebench/livebench`, 1166 stars; LiveCodeBench follows the same license).
- Typical chain length: high for `Hard` competitive-programming problems.
- Typical context length: medium — long problem statements + sample I/O.
- Degradation flag: Yes — the paper's `Hard` split consistently breaks reasoning-LLMs at long-statement / long-CoT combination.
- Separable axes: Difficulty stratification across CodeForces ratings.

**19. LiveCodeBench Pro: How Do Olympiad Medalists Judge LLMs in Competitive Programming?** — arXiv:2506.11928 (2025). Zheng et al.
- Curated CodeForces / ICPC / IOI subset annotated by IOI/ICPC medalists; explicitly distinguishes *implementation* tasks from *algorithmic* tasks — exactly the chain-depth axis.
- License: GitHub `GavinZhengOI/LiveCodeBench-Pro` (172 stars), open.
- Typical chain length: high on algorithmic-reasoning split.
- Typical context length: medium.
- Degradation flag: Yes — paper explicitly shows accuracy collapse on the *algorithmic-reasoning* tag at fixed statement length.
- Separable axes: Strong — `algorithmic` vs `implementation` tags isolate chain depth from context length.

**20. SWE-bench (+ Verified) — Can Language Models Resolve Real-World GitHub Issues?** — arXiv:2310.06770 (2023, with Verified subset 2024). Jimenez et al.
- 2,294 GitHub issues from 12 Python repos. The Verified subset is 500 hand-validated. The canonical long-context program-synthesis benchmark.
- License: MIT. HF dataset: `princeton-nlp/SWE-bench`, `princeton-nlp/SWE-bench_Verified`.
- Typical chain length: high — multi-file edits with reasoning over diff and test suite.
- Typical context length: very high — repo context routinely 100k+ tokens; original paper uses an oracle retriever to reduce.
- Degradation flag: Yes — original paper documents performance collapse without retrieval; Verified leaderboard shows ceiling well below 100% for frontier reasoning models.
- Separable axes: Strong — oracle-retrieval vs full-repo input cleanly separates context length; difficulty by `difficulty` annotation in Verified separates chain length.

**21. Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving** — arXiv:2504.02605 (2025). Zan et al.
- 1,632 issue-resolving instances across Java, TS/JS, Go, Rust, C/C++; complements SWE-bench. Comes with Multi-SWE-RL training data.
- License: GitHub `multi-swe-bench/multi-swe-bench` (333 stars), open (Apache-2.0).
- Typical chain length: high — same shape as SWE-bench.
- Typical context length: very high — language-specific repos.
- Degradation flag: Yes — paper reports cross-language performance gap that widens with chain length.
- Separable axes: Strong — language axis + repo size axis.

**22. AlphaCode (CodeContests dataset)** — arXiv:2203.07814 (2022). DeepMind.
- The canonical CodeForces/CodeContests dataset; released open. Still used for chain-length scaling work because of long statements.
- License: permissive (Apache-2.0 on `google-deepmind/code_contests`).
- Typical chain length: high (algorithmic).
- Typical context length: medium-to-high (long statements + sample I/O).
- Degradation flag: Yes — implicit in original paper and revisited by all later competitive-coding benchmarks.
- Separable axes: Yes — difficulty rating + statement length.

**23. APPS: Measuring Coding Challenge Competence with APPS** — Hendrycks et al. 2021 (arXiv:2105.09938; also reported in Chen et al. 2021).
- 10,000 problems including a `competition-hard` split (~1,000 problems).
- License: MIT. HF dataset: `codeparrot/apps`.
- Typical chain length: high on `competition-hard`.
- Typical context length: low-to-medium.
- Degradation flag: Yes — well documented in the original paper.
- Separable axes: 3-tier difficulty.

### 2D. Chain-length-scaling literature (2024–2026)

**24. Inverse Scaling in Test-Time Compute** — arXiv:2507.14417 (2025). Gema et al.
- Constructs four task families (counting w/ distractors, regression w/ spurious features, deduction w/ constraint tracking, advanced AI risks) where extending CoT length *deteriorates* accuracy. Five distinct failure modes catalogued (Claude becomes distracted by irrelevant info; OpenAI o-series overfits problem framing; etc.).
- Architecture(s) studied: frontier reasoning LLMs (Claude, OpenAI o-series, DeepSeek, Qwen). Subquadratic-attention models: **NOT studied — explicitly absent.**
- Breakdown point: task-dependent; in the constraint-tracking deduction tasks, accuracy peaks at ~512–2k token CoT and falls afterwards.
- Recursive / iterative computation as remedy: not explored.
- GitHub `safety-research/inverse-scaling-ttc` (25 stars).

**25. When More is Less: Understanding Chain-of-Thought Length in LLMs** — arXiv:2502.07266 (2025). Wu et al.
- Theoretical + empirical inverted-U curve for accuracy vs CoT length on MATH and ARC. Optimal length grows with task difficulty and shrinks with model capability.
- Architecture(s) studied: dense Transformer LLMs.
- Breakdown point: model- and difficulty-dependent; e.g., a 7B model peaks around 1.5k tokens and collapses past 4k on hard MATH.
- Subquadratic-attention models: **NOT studied.**
- Recursive computation: not investigated; the paper attributes failure to "simplicity bias" and proposes length-aware filtering.

**26. What Characterizes Effective Reasoning? Revisiting Length, Review, and Structure of CoT** — arXiv:2509.19284 (2025). Feng, Kempe et al.
- Across 10 LRMs on math + science, *failed-step fraction* (FSF) predicts accuracy better than length or review behavior. Longer-is-not-better is empirically confirmed.
- Subquadratic-attention models: **NOT studied.**
- Recursive computation as remedy: not explicitly, but the paper's structural-quality-over-length finding is consistent with what architectural recursion might offer.

**27. Through the Valley: Path to Effective Long CoT Training for Small Language Models** — arXiv:2506.07712 (2025). Luo et al.
- Discovers "Long CoT Degradation": SLMs (≤3B) trained on limited long-CoT data systematically *lose* accuracy via error accumulation.
- Architecture(s): Qwen2.5, LLaMA3, Gemma3 families (all dense Transformer).
- Breakdown point: training-data-quantity dependent; degradation appears with <100k SFT examples and persists to RL.
- Subquadratic-attention models: **NOT studied.**

**28. Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning** — arXiv:2502.18080 (2025). Yang, Ma, Lin, Wei.
- Demonstrates excessively scaling CoT length harms math reasoning; proposes a thinking-optimal-scaling RL recipe on Qwen2.5-32B-Instruct that improves accuracy at lower CoT cost.
- Subquadratic-attention models: **NOT studied.**

**29. L1: Controlling How Long A Reasoning Model Thinks With RL** — arXiv:2503.04697 (2025). Aggarwal & Welleck.
- LCPO trains a model to obey a user-specified length budget, exposing the accuracy-vs-CoT-length frontier explicitly.
- Subquadratic-attention models: **NOT studied.**

**30. Don't Overthink it: Preferring Shorter Thinking Chains for Improved LLM Reasoning** — arXiv:2505.17813 (2025). 
- Empirical: shorter chains plus majority-voting matches or beats longer chains on competition math at fixed compute.
- Subquadratic-attention models: **NOT studied.**

**31. s1: Simple Test-Time Scaling** — arXiv:2501.19393 (2025). 
- Budget-forcing token mechanism on Qwen2.5-32B; the canonical reference implementation for "let the model think more, but bounded."
- Subquadratic-attention models: **NOT studied.**
- GitHub `simplescaling/s1` (6651 stars).

**32. InftyThink: Breaking the Length Limits of Long-Context Reasoning in LLMs** — arXiv:2503.06692 (2025). 
- Transforms monolithic CoT into iterative summarize-then-continue passes, which is *operationally close to* what architectural recursion would buy. Studies the chain-length × context-length plane explicitly: their central plot is accuracy vs total reasoning tokens × per-iteration context.
- Architecture(s): standard Transformer LLMs.
- Subquadratic-attention models: **NOT studied** (but the iterative scheme is compatible with them).
- Recursive / iterative computation: this paper *is* the iterative-computation analogue at the prompt level.
- GitHub `ZJU-REAL/InftyThink` (53 stars).

---

## 3. Datasets (open data; HF IDs + licenses)

| Benchmark | HF dataset ID | License | Notes |
|---|---|---|---|
| miniF2F (Lean) | `AI4M/minif2f_dataset`, `Tonic/MiniF2F`, `wellecks/minif2f_isabelle` | Apache-2.0 (Lean), MIT (Metamath), Apache-2.0 (Isabelle) | Multiple HF mirrors; canonical at `openai/miniF2F` |
| PutnamBench | `harrywsanders/putnambench` (community mirror); canonical `trishullab/PutnamBench` GitHub | Apache-2.0 (Lean 4 / Isabelle), MIT (Coq) | HF mirror is community, not official |
| Omni-MATH | `KbsdJames/Omni-MATH` | Apache-2.0 | Difficulty-stratified |
| OlympiadBench | `Hothan/OlympiadBench` | Apache-2.0 | Bilingual + multimodal |
| HARP | (no canonical HF dataset; data in GitHub `aadityasingh/harp`) | MIT | HF release pending per repo; flag |
| MathArena | platform at matharena.ai (no static HF dataset) | platform-specific | Continuously updated |
| MATH-500 | `HuggingFaceH4/MATH-500` | unspecified on card; underlying MATH is MIT (Hendrycks 2021) | **Flag: card lacks explicit license** |
| FrontierMath | NOT redistributed; gated access at Epoch AI | closed/gated | **Flag: not open data** |
| LeanDojo Benchmark | `tasksource/leandojo` (community mirror) | code MIT; Mathlib content Apache-2.0 | **Flag: HF mirror lacks explicit licence** |
| ProofNet | (no official HF release; GitHub `zhangir-azerbayev/proofnet`) | MIT | |
| CoqGym | (no HF release; GitHub `princeton-vl/CoqGym`) | MIT | |
| Goedel-Prover datasets | (training data on HF under `Goedel-LM`) | Apache-2.0 | |
| FormalMATH | GitHub `Sphere-AI-Lab/FormalMATH-Bench` | Apache-2.0 | |
| BigCodeBench | `bigcode/bigcodebench` | Apache-2.0 | |
| CRUXEval | `cruxeval-org/cruxeval` | permissive (per paper) | |
| LiveCodeBench | (latest snapshots on `livecodebench/code_generation_lite`) | MIT (per repo) | |
| LiveCodeBench Pro | GitHub `GavinZhengOI/LiveCodeBench-Pro` | open (per repo) | |
| SWE-bench / Verified | `princeton-nlp/SWE-bench`, `princeton-nlp/SWE-bench_Verified` | MIT (per upstream repo) | **Flag: HF card lacks explicit license field** |
| Multi-SWE-bench | (release in `multi-swe-bench/multi-swe-bench` GitHub) | Apache-2.0 | |
| APPS | `codeparrot/apps` | MIT | |
| CodeContests / AlphaCode | `deepmind/code_contests` (mirror) | Apache-2.0 (canonical repo) | |

---

## 4. Reference Implementations (public GitHub)

- LeanDojo + ReProver: `lean-dojo/leandojo` (796 stars).
- DeepSeek-Prover V1.5 / V2: `deepseek-ai/deepseek-prover-v1.5` (568), `deepseek-ai/deepseek-prover-v2` (1263).
- Goedel-Prover V1 / V2: `Goedel-LM/Goedel-Prover` (233), `Goedel-Prover-V2` (167).
- Leanabell-Prover-V2: `Leanabell-LM/Leanabell-Prover-V2` (12).
- BFS-Prover-V2: paper only (Sept 2025) — no public code yet.
- PutnamBench: `trishullab/PutnamBench`.
- miniF2F: `openai/miniF2F`, `yangky11/minif2f-lean4` (75).
- ProofNet: `zhangir-azerbayev/proofnet` (122).
- CoqGym (ASTactic): `princeton-vl/CoqGym` (417).
- Omni-MATH: `kbsdjames/omni-math` (93).
- OlympiadBench: `OpenBMB/OlympiadBench` (193).
- HARP: `aadityasingh/harp` (23).
- AMO-Bench (recent IMO-tier benchmark, useful sanity check): `meituan-longcat/AMO-Bench` (122).
- BigCodeBench: `bigcode-project/bigcodebench` (498).
- LiveCodeBench: `livebench/livebench` (1166).
- LiveCodeBench Pro: `GavinZhengOI/LiveCodeBench-Pro` (172).
- SWE-bench: `swe-bench/SWE-bench`.
- Multi-SWE-bench: `multi-swe-bench/multi-swe-bench` (333).
- AlphaCode / CodeContests: `google-deepmind/code_contests` (2193).
- s1 (test-time scaling reference): `simplescaling/s1` (6651).
- L1 (length-controlled RL): `cmu-l3/l1` (265).
- Inverse-scaling-TTC: `safety-research/inverse-scaling-ttc` (25).
- InftyThink (iterative long-CoT): `ZJU-REAL/InftyThink` (53).
- Frac-CoT (Fractured Chain-of-Thought): `BaohaoLiao/frac-cot` (16).
- DEER (dynamic early exit): `iie-ycx/deer` (235).

---

## 5. Open Questions Spotted While Reading (no hypotheses)

1. **No chain-length-scaling paper above evaluates a subquadratic-attention model.** Every entry in §2D used dense-Transformer LLMs (Qwen, LLaMA, Gemma, Claude, OpenAI o-series). Whether the inverted-U / inverse-scaling phenomena replicate on Mamba-/Hybrid-SSM-/SubQ-attention backbones is unmeasured.
2. **Prompt-level iterative compute (InftyThink, Fractured CoT) is the closest existing analogue to architectural recursion**; the question of whether *true* compute-graph recursion (TRM-style) avoids the failure modes that prompt-level iteration only partially mitigates is not addressed in the open literature.
3. **The chain-length × context-length plane is rarely measured jointly.** Most CoT-length papers (§2D) hold input context tiny; most long-context benchmarks hold reasoning depth small. The exceptions I found — InftyThink (2503.06692), SWE-bench-style oracle-vs-full-repo, LeanDojo retrieval-K — are partial. A clean factorial sweep of (chain length × context length) on a single benchmark is missing.
4. **PutnamBench-hard / FrontierMath-tier-4 chain-length statistics are not publicly tabulated.** Both papers report pass-rate per tier, but not the *empirical distribution of solved-proof or solved-CoT length* in solved instances. Without that we cannot calibrate "chain length >50" as a falsifiable threshold.
5. **License gaps.** FrontierMath is gated; MATH-500 HF card lacks an explicit license; HuggingFace mirror of LeanDojo and SWE-bench Verified lack explicit license fields on the HF dataset card (canonical upstream is MIT/Apache-2.0). For spec compliance these need a defensible upstream citation.
6. **Lean Workbook + ProverBench training data is what most modern Lean provers use, but the *distribution of proof-term length in Workbook* is not reported in any paper I read** — important for calibrating chain-length scaling experiments.
7. **No benchmark explicitly measures performance vs proof-term token-length holding context constant.** PutnamBench / miniF2F report per-theorem pass-rate; chain-length-controlled splits would need to be constructed.

---

## 6. Sources

ArXiv IDs cited:
- 2407.11214 (PutnamBench), 2109.00110 (miniF2F), 2411.04872 (FrontierMath), 2410.07985 (Omni-MATH), 2402.14008 (OlympiadBench), 2412.08819 (HARP), 2505.23281 (MathArena), 2306.15626 (LeanDojo), 2302.12433 (ProofNet), 1905.09381 (CoqGym), 2502.07640 (Goedel-Prover V1), 2508.03613 (Goedel-Prover V2), 2504.21801 (DeepSeek-Prover-V2), 2408.08152 (DeepSeek-Prover-V1.5), 2405.14333 (DeepSeek-Prover), 2509.06493 (BFS-Prover-V2), 2505.02735 (FormalMATH), 2505.04528 (FPS / formal problem solving), 2406.15877 (BigCodeBench), 2401.03065 (CRUXEval), 2403.07974 (LiveCodeBench), 2506.11928 (LiveCodeBench Pro), 2310.06770 (SWE-bench), 2504.02605 (Multi-SWE-bench), 2203.07814 (AlphaCode / CodeContests), 2105.09938 (APPS), 2103.03874 (MATH).
- Chain-length-scaling: 2507.14417 (Inverse Scaling TTC), 2502.07266 (When More Is Less), 2509.19284 (Effective Reasoning / FSF), 2506.07712 (Through the Valley), 2502.18080 (Thinking-Optimal Scaling), 2503.04697 (L1 / LCPO), 2505.17813 (Don't Overthink), 2501.19393 (s1), 2503.06692 (InftyThink), 2505.12992 (Fractured CoT), 2504.15895 (DEER / Dynamic Early Exit), 2503.21614 (Survey of Efficient Reasoning), 2503.09567 (Survey of Long CoT).

URLs (HF + GitHub):
- https://huggingface.co/papers/2407.11214 ; https://huggingface.co/papers/2109.00110 ; https://huggingface.co/papers/2411.04872 ; https://huggingface.co/papers/2410.07985 ; https://huggingface.co/papers/2402.14008 ; https://huggingface.co/papers/2412.08819 ; https://huggingface.co/papers/2505.23281 ; https://huggingface.co/papers/2306.15626 ; https://huggingface.co/papers/2302.12433 ; https://huggingface.co/papers/2502.07640 ; https://huggingface.co/papers/2504.21801 ; https://huggingface.co/papers/2509.06493 ; https://huggingface.co/papers/2505.02735 ; https://huggingface.co/papers/2406.15877 ; https://huggingface.co/papers/2401.03065 ; https://huggingface.co/papers/2403.07974 ; https://huggingface.co/papers/2506.11928 ; https://huggingface.co/papers/2310.06770 ; https://huggingface.co/papers/2504.02605 ; https://huggingface.co/papers/2507.14417 ; https://huggingface.co/papers/2502.07266 ; https://huggingface.co/papers/2509.19284 ; https://huggingface.co/papers/2506.07712 ; https://huggingface.co/papers/2502.18080 ; https://huggingface.co/papers/2503.04697 ; https://huggingface.co/papers/2505.17813 ; https://huggingface.co/papers/2501.19393 ; https://huggingface.co/papers/2503.06692.
- https://huggingface.co/datasets/bigcode/bigcodebench ; https://huggingface.co/datasets/Hothan/OlympiadBench ; https://huggingface.co/datasets/KbsdJames/Omni-MATH ; https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified ; https://huggingface.co/datasets/HuggingFaceH4/MATH-500 ; https://huggingface.co/datasets/AI4M/minif2f_dataset ; https://huggingface.co/datasets/harrywsanders/putnambench ; https://huggingface.co/datasets/tasksource/leandojo.
- https://github.com/trishullab/PutnamBench ; https://github.com/openai/miniF2F ; https://github.com/lean-dojo/LeanDojo ; https://github.com/zhangir-azerbayev/proofnet ; https://github.com/princeton-vl/CoqGym ; https://github.com/Goedel-LM/Goedel-Prover ; https://github.com/Goedel-LM/Goedel-Prover-V2 ; https://github.com/deepseek-ai/deepseek-prover-v2 ; https://github.com/Sphere-AI-Lab/FormalMATH-Bench ; https://github.com/bigcode-project/bigcodebench ; https://github.com/livebench/livebench ; https://github.com/GavinZhengOI/LiveCodeBench-Pro ; https://github.com/swe-bench/SWE-bench ; https://github.com/multi-swe-bench/multi-swe-bench ; https://github.com/google-deepmind/code_contests ; https://github.com/simplescaling/s1 ; https://github.com/cmu-l3/l1 ; https://github.com/safety-research/inverse-scaling-ttc ; https://github.com/ZJU-REAL/InftyThink ; https://github.com/aadityasingh/harp ; https://github.com/OpenBMB/OlympiadBench ; https://github.com/kbsdjames/omni-math.
- https://epoch.ai/frontiermath (FrontierMath access).
