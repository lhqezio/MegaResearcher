# red-team-S3 verification — REVISION-2 critique pass

Verification-before-completion checks per superpowers:receiving-code-review discipline. This file overwrites the prior pass.

## Independent literature queries (3 fresh queries this pass)

Fresh queries to confirm the gap claim still survives after the revision's substrate changes:

1. **`majority voting paper generation AI scientist multi-agent`** (limit=10).
   - Returned: OPTAGENT (debate, not draft-decision voting), AI Scientist v2 (no voting layer), Paper2Agent, Paper Circle (retrieval/KG), Hegelian Dialectic (MAMV on novelty in ideation), Beyond Brainstorming (multi-agent for ideation), AI Scientist v1, OpenCLAW-P2P, OmniScientist, Persona Inconstancy.
   - **Finding: no published AI-Scientist-family paper-gen system applies majority voting over enumerable structured paper-decisions. Gap claim survives.**

2. **`N=5 self-consistency same model effective sample size scaling`** (limit=5).
   - Returned: Optimal Self-Consistency (arXiv:2511.12309, Nov 2025, power-law scaling for SC sample efficiency), Compute-Optimal SC-vs-GenRM (arXiv:2504.01005), Make-Every-Penny-Count DSC (arXiv:2408.13457), Slim-SC (arXiv:2509.13990), Adaptive-Consistency (arXiv:2305.11860).
   - **Finding: none apply SC to paper-generation drafts. Optimal SC is the closest recent theoretical anchor and should be cited (Suggestion S-B).**

3. **`AI Scientist v2 majority voting self-consistency multiple drafts`** via DuckDuckGo web search.
   - Returned: no relevant results.
   - **Finding: gap claim survives. No published work documents AI Scientist v2 with a voting-over-drafts layer.**

## Citation spot-checks (3 primary-source checks this pass)

1. **Patel arXiv:2604.03809 §2.1 + §3.1 + §3.2 (CR1 anchor).**
   - `paper_details`: paper exists, abstract confirms "mean cosine similarity is 0.888 and effective rank is 2.17 out of 3.0."
   - `read_paper §2.1 Measuring Collapse`: definition `rank_eff(E) = exp(-Σ p_j log p_j)` where p_j = σ_j/Σσ_ℓ from SVD of stacked embeddings E∈ℝ^(N×768). Range [1, N]. Smith's "27.7% effective-N reduction at N=3" follows from 2.17/3.0 = 0.723 → 1 − 0.723 = 0.277.
   - `read_paper §3.1 Collapse Analysis`: Table 3 confirms GSM8K cosine 0.888, eff rank 2.17. MATH-500 cosine 0.904, eff rank 2.09. Smith's numbers are **exact**. Also confirmed: "Collapse is also more severe on MATH-500 (cosine 0.904 vs. 0.888 on GSM8K), consistent with agents converging more tightly on harder problems where the model has fewer confident alternative paths." This is the strongest published support for the smith's choice of the MATH-500-rate discount.
   - `read_paper §3.2 Ablation Studies`: **Table 5 reveals important run-to-run variance: on the SAME 100 GSM8K questions, single-agent shifted 82→79%, SC shifted 84→85%, with "Per-protocol swings range from 1 to 3 points, with total spread across methods reaching 6 points."** Single-vs-SC gap is +2 (Run 1) or +6 (Run 2). Paper explicitly warns: "individual point differences between protocols should not be over-interpreted." This is the basis for Important objection I-B (smith should pre-register replication).

2. **AbGen arXiv:2507.13300 §2.4 + §2.5 + §2.6 + §3.1 + §3.2 (CR3 substrate).**
   - `paper_details`: paper exists, 1,500 expert-annotated examples from 807 NLP papers, GitHub yale-nlp/AbGen (4 stars).
   - `read_paper §2.6 Data Statistics`: "We randomly split the dataset into two subsets: testmini and test. The testmini subset contains **500 examples** and is intended for both method validation and human analysis and evaluation. The test subset comprises the remaining 1,000 examples." Smith's "testmini-500" is exact.
   - `read_paper §2.4 Reference Ablation Study Annotation`: confirms three-section structure (Research Objective, Experiment Process, Result Discussion). Smith's plan to extract 6-axis binary vector by lexicon matching against Experiment Process is operationally plausible.
   - `read_paper §3.2 Human Evaluation Protocol`: "we sample 40 fixed LLM-generated outputs that are separately evaluated by all four expert annotators. They achieve inter-annotator agreement scores (i.e., Cohen's Kappa) of 0.735, 0.782, and 0.710 for the criteria of importance, faithfulness, and soundness, respectively." Smith's "0.71-0.78" is exact.
   - `read_paper §3.1 Evaluation Criteria`: "three external senior NLP researchers, all of whom serve as area chairs for the ACL Rolling Review" — smith's "ACL area chair" attribution is faithful.
   - **Important subtlety**: AbGen's Kappa was measured on **Likert-scoring of LLM-generated outputs**, not on per-axis-binary lexicon extraction from reference text. Smith's "Cohen's Kappa 0.71-0.78 noise floor on the reference annotations themselves" framing is one step removed. Basis for Important objection I-A (smith should run a pilot study to establish per-axis Kappa).
   - `find_all_resources`: confirmed AbGen has **no HF dataset card** (Collections only, no Datasets/Models). Smith's "frozen seed=42 sample of 100 from testmini-500" requires implementation, not canonical API call. Basis for Suggestion S-C.

3. **Choi arXiv:2508.17536 §3 Table 1 + §4 (magnitude anchor).**
   - `paper_details`: paper exists.
   - `read_paper §3 Is Debate Really Necessary?`: Table 1 confirms Qwen2.5-7B single-agent average **0.7205**, Majority Voting average **0.7691** = **+4.86 points** across 7 benchmarks (Arithmetics, GSM8K, MMLU Pro.Med., MMLU Formal Logics, HellaSwag, CommonSenseQA, HH-RLHF). Smith's anchor is exact.
   - **Important observation not in smith's submission**: Llama3.1-8B same-table single-agent **0.6203**, Majority Voting **0.7242** = **+10.39 points**. Weaker models show larger voting lift. Smith picked the conservative-for-frontier-model anchor (Qwen) without explanation. Basis for Suggestion S-A.
   - `read_paper §4 Theoretical Analysis`: Theorem 1 requires N > K/Δ². For per-binary decisions (K=2), N=5 needs Δ > √(2/5) ≈ 0.63. This is a tightness concern for the 6-axis ablation extraction (where each axis is plausibly 50/50) but acceptable for baseline-list/citation-attribution.

## Verdict-severity match check

- New verdict: APPROVE
- Critical count: 0 (all three revision-1 Critical defects CR1, CR2, CR3 addressed at primary-source level)
- Important count: 4 (I-A Kappa scope, I-B replication, I-C scoping multiplier, I-D 12-paper sample size)
- Suggestion count: 3 (S-A Llama anchor, S-B Optimal-SC citation, S-C AbGen dataset release)
- Gap claim survives: true (3 independent queries above)
- Prior Critical defects: CR1 addressed (Patel-grounded 30% discount), CR2 addressed (canonical leaderboards), CR3 addressed (AbGen substrate switch)

The APPROVE verdict matches severity:
- Zero Critical objections outstanding — required for APPROVE.
- All Important objections are revisable in eval-designer (pilot study, replication count, sensitivity analysis, paper-clustered CIs). None of them invalidate the hypothesis structure.
- Suggestions are nice-to-have, not gates.

## Discipline check (receiving-code-review)

- **Performative agreement avoided**: I did not auto-approve because the smith descoped honestly. I verified each CR claim against primary sources (Patel §3.1 Table 3, AbGen §3.2, Choi §3 Table 1) and surfaced two genuine subtleties: (a) AbGen's Kappa is on Likert-scoring not lexicon-extraction, (b) Patel's run-to-run variance puts +6 at the upper edge of empirical noise. Both are Important, not Critical.
- **Performative skepticism avoided**: I did not invent a new CR just to push to revision-3. The four Important objections (I-A through I-D) are concrete, fixable in eval-designer, and grounded in the same primary sources the smith cites.
- **Steelman constructed**: §7 of output.md presents the strongest case against APPROVE (5 sub-points a-e). I rebut (c) and (e); I concede (a), (b), (d) as Important.
- **APPROVE criteria**: would I defend this hypothesis publicly? Yes — at the descoped +6 with deterministic substrates, dual-threshold falsifiers, AbGen-grade ablation substrate, and pre-registered taxonomies. The result could land at +3 (failure-as-bound) and remain publishable as a measurement contribution. The smith has named this in §7 R6. APPROVE is correct.

## Banned-phrase audit

Final pass on output.md for banned emphatic phrases per global memory rules:
- emphatic-structural phrase (l-b) — initially used twice in draft; replaced with "single assumption the derivation rests on" and "critical assumption the entire magnitude derivation rests on" respectively. Now zero occurrences in output.md.
- emphatic-effort phrase (d-a-l-o-w) — zero occurrences.
- emphatic "real" adjective — zero occurrences (only "real-world" in cited paper titles, which is exempt).
- "honest" / "honestly" — three first-person usages caught in audit and rephrased ("descoped honestly" → "descoped the predicted floor", "the descope is honest" → "the descope is grounded in primary sources", "more honest derivation" → "more conservative derivation"). Smith's own §1 heading text "Honest framing" remains as direct quotation only.
