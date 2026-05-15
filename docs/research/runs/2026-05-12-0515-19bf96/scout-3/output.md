# scout-3 — Automated peer review and paper-quality evaluation

## 1. Scope

**Sub-topic restatement:** LLM-as-reviewer for scientific (primarily ML) papers — both end-to-end review-generation systems (OpenReviewer, SEA, DeepReviewer, AutoRev, ReviewerToo, AI Scientist's review module, AAAI-26 pilot, FactReview) and the benchmarks/datasets that interrogate them (PeerRead, DeepReview-13K, SEA_data, SPECS-Review-Benchmark, SPOT, RevUtil, ReviewMT, BadScientist), with **explicit surfacing of which proxy measures are demonstrably untrustworthy** — review-score prediction, helpfulness-style judgments unguarded against presentation manipulation, single-LLM-judge ensembles, and citation-faithfulness assumptions.

**Narrowing decisions:**
- Restricted to 2023–2026 LLM-era systems plus the canonical PeerRead (2018) as the foundational dataset still in use.
- Excluded medical / clinical / chemistry-only review systems unless they generalised (kept PWP-meta-prompting since it is methodological).
- Excluded purely sociological "is AI ruining peer review" position papers unless they ship a benchmark or measurable claim.
- Dropped one borderline paper (OpenCLAW-P2P v6.0, arxiv 2604.19792) because the abstract reads as a self-published artifact with no external evaluation rather than a peer-review system the eval-designer can stress-test.

## 2. Key papers

Grouped by sub-cluster.

### 2a. End-to-end review-generation systems (the things MegaResearcher would adopt or compete with)

**(1) The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery** — arXiv:2408.06292 (2024) — Lu, C. et al. (Sakana AI). The original closed-loop research-and-review system; ships a GPT-4-based "automated reviewer" module that scores its own generated papers against NeurIPS-style criteria. Self-evaluated agreement with human reviewers is the central credibility claim of the entire AI Scientist pipeline, so any MegaResearcher gap-finding around the reviewer-in-the-loop pattern starts here. Repo: github.com/sakanaai/ai-scientist (13.6k stars).

**(2) The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search** — arXiv:2504.08066 (2025) — Yamada, Y. et al. The system that produced the first fully AI-generated paper accepted at a peer-reviewed ICLR workshop; the reviewer module is iterated relative to v1 and is the closing oracle of the agentic tree-search. Critical for the spec because it is the existence proof that "main-track-grade" is plausible at workshop level and the place where the workshop-vs-main-track gap is sharpest. Repo: github.com/SakanaAI/AI-Scientist-v2 (6.2k stars).

**(3) OpenReviewer: A Specialized Large Language Model for Generating Critical Scientific Paper Reviews** — arXiv:2412.11948 (2024) — Idahl, M.; Ahmadi, Z. Llama-OpenReviewer-8B fine-tuned on 79k expert reviews from top ML conferences; produces structured reviews following conference-specific templates. Reports that human reviewers rated its outputs higher than GPT-4 and Claude-3.5 on critical-content axes — important because it claims fine-tuning on review data beats frontier in-context-only models specifically on the dimensions MegaResearcher cares about.

**(4) CycleResearcher: Improving Automated Research via Automated Review** — arXiv:2411.00816 (2024) — Weng, Y. et al. Paired CycleResearcher + CycleReviewer (open-source post-trained LLMs) with the Review-5k and Research-14k datasets; uses the reviewer's score as RL reward. The closed-loop researcher↔reviewer training pattern is exactly the architecture the spec is asking whether to adopt; the paper reports MAE on review scores below human inter-reviewer disagreement, which is the strongest published claim of human-agreement parity to date and the eval-designer should pressure-test it.

**(5) DeepReview: Improving LLM-based Paper Review with Human-like Deep Thinking Process** — arXiv:2503.08569 (2025) — Zhu, M. et al. Multi-stage framework (structured analysis → literature retrieval → evidence-based argumentation) with DeepReviewer-14B; ships the DeepReview-13K dataset. Beats CycleReviewer-70B and approaches GPT-o1 with fewer tokens. The staged-decomposition pattern is directly applicable to MegaResearcher's leaf-worker contract because each stage maps to a separate subagent. Repo: github.com/zhu-minjun/Researcher (379 stars).

**(6) ReviewerToo: Should AI Join The Program Committee? A Look At The Future of Peer Review** — arXiv:2510.08867 (2025) — Sahu, G.; Larochelle, H.; Charlin, L.; Pal, C. Modular framework with reviewer-persona slots and structured evaluation criteria; explicitly designed to be partially or fully integrated into a real conference workflow. Important because it draws an explicit boundary line — which review dimensions LLM personas are reliable on and which still require human expertise — exactly the question MegaResearcher needs answered to set the hand-off contract.

**(7) Automated Peer Reviewing in Paper SEA: Standardization, Evaluation, and Analysis** — arXiv:2407.12857 (2024) — Yu, J. et al. Three-module framework (SEA-S standardise / SEA-E evaluate / SEA-A analyse) with GPT-4 base and a novel "mismatch score" that flags when generated reviews disagree with each other on the same paper. The mismatch-score idea is a candidate cheap consistency proxy for MegaResearcher's audit trail. Repo: github.com/ecnu-sea/SEA (89 stars); dataset SEA_data on HF.

**(8) FactReview: Evidence-Grounded Reviews with Literature Positioning and Execution-Based Claim Verification** — arXiv:2604.04074 (2026) — Xu, H. et al. Explicitly motivated by the failure mode the spec calls out: "Most LLM-based reviewing systems read only the manuscript and generate comments from the paper's own narrative. This makes their outputs sensitive to presentation quality and leaves them weak when the evidence needed for review lies in related work or released code." Adds claim extraction + literature positioning + execution-based verification against released code. The architectural prescription this paper makes is one of the few that directly attacks presentation-overweighting.

**(9) AI-Assisted Peer Review at Scale: The AAAI-26 AI Review Pilot** — arXiv:2604.13940 (2026) — Biswas, J. et al. First large-scale field deployment: every main-track AAAI-26 submission received an AI-generated review next to human ones. Ships the SPECS-Review-Benchmark (Story / Presentation / Evaluations / Correctness / Significance — controlled flaw injection across five axes). For the spec, this is the most authoritative source on which review dimensions LLM reviewers actually catch and which they don't, at conference scale, with paired human ground truth.

**(10) AutoRev: Automatic Peer Review System for Academic Research Papers** — arXiv:2505.14376 (2025) — Chitale, M.P. et al. Graph-based representation of long-context papers that sidesteps token-window limits; reports SOTA on standard review-generation metrics. Mostly relevant as the long-context counterpoint to fine-tuned and tree-search approaches.

**(11) Reviewer2: Optimizing Review Generation Through Prompt Generation** — arXiv:2402.10886 (2024) — Gao, Z.; Brantley, K.; Joachims, T. Two-stage prompt-generation pipeline (aspect modelling → review writing); the aspect-modelling stage attempts to recover the diversity of opinions a real PC produces. Repo: github.com/zhaolingao/reviewer2 (16 stars).

**(12) ReviewRL: Towards Automated Scientific Review with RL** — arXiv:2508.10308 (2025) — Zeng, S. et al. SFT + RL with a composite reward (rating-accuracy + factuality + structure) and ArXiv-MCP literature retrieval; trained against ICLR 2025. The decomposition of the reward function is the most explicit treatment to date of *what dimensions of a review are even rewardable separately*.

**(13) AI-Driven Scholarly Peer Review via Persistent Workflow Prompting, Meta-Prompting, and Meta-Reasoning** — arXiv:2505.03332 (2025) — Markhasin, E. Zero-code prompt-engineering methodology (PWP) for critical manuscript analysis; demonstrates that the structured-workflow part of "review pipelines" is achievable without fine-tuning. Useful for MegaResearcher as a baseline rather than a target.

### 2b. Standalone benchmarks and meta-evaluation datasets (what the eval-designer can actually score against)

**(14) A Dataset of Peer Reviews (PeerRead): Collection, Insights and NLP Applications** — arXiv:1804.09635 (2018) — Kang, D. et al. (AI2). 14.7k paper drafts + 10.7k expert reviews from ACL/NIPS/ICLR with accept/reject decisions. The canonical pre-LLM-era dataset still embedded in nearly every modern review-generation benchmark. Repo: github.com/allenai/PeerRead (428 stars); dataset allenai/peer_read on HF.

**(15) Can large language models provide useful feedback on research papers? A large-scale empirical analysis** — arXiv:2310.01783 (2023) — Liang, W. et al. (Stanford). The most cited early result: GPT-4 feedback on Nature-family + ICLR papers gets reviewer-overlap rates comparable to human-vs-human, and authors rate it "helpful". The "helpful" claim is the one downstream eval-designer should pressure-test against the BadScientist / SPECS findings. Repo: github.com/weixin-liang/llm-scientific-feedback (531 stars).

**(16) Insights from the ICLR Peer Review and Rebuttal Process** — arXiv:2511.15462 (2025) — Kargaran, A.H. et al. Large-scale analysis of ICLR 2024–25 review-score dynamics before and after rebuttals, with LLM-categorised rebuttal strategies. Establishes the *human* score-stability and score-mobility baselines any automated reviewer claim has to clear. Repo: github.com/papercopilot/iclr-insights (15 stars).

**(17) Automatic Evaluation Metrics for Artificially Generated Scientific Research** — arXiv:2503.05712 (2025) — Höpner, N. et al. Direct head-to-head: citation-count prediction vs. review-score prediction as automatic proxies for AI-generated paper quality. **Finding: a simple title+abstract model beats LLM-based reviewers on review-score prediction, and citation prediction is more viable than review prediction overall.** This is the strongest published evidence that LLM-judge review scores are an *unreliable* automatic proxy — the eval-designer must treat single-shot LLM review scores as a known-bad metric.

**(18) When AI Co-Scientists Fail: SPOT — a Benchmark for Automated Verification of Scientific Research** — arXiv:2505.11855 (2025) — Son, G. et al. SPOT pairs 83 published papers with their actual verification failures (errata, retractions). LLMs as verifiers show poor recall, precision, and reliability. Repo: github.com/guijinSON/SPOT (9 stars). Important as the standalone verification benchmark — distinct from review-score prediction and not confounded by presentation.

**(19) BadScientist: Can a Research Agent Write Convincing but Unsound Papers that Fool LLM Reviewers?** — arXiv:2510.18003 (2025) — Jiang, F. et al. (UW). Adversarial framework that generates fabricated papers with *no real experiments* using presentation-manipulation strategies and feeds them to multi-model LLM review systems. Reports a "concern-acceptance conflict" — LLM reviewers flag integrity concerns yet still produce high acceptance scores. **This is the most direct empirical demonstration of presentation overweighting / reward-hacking that the spec asks to be surfaced.** Dataset: badscientist/BadScientist-Prompts on HF (MIT, research-only).

**(20) DeepReview-13K dataset** (sibling artifact to DeepReview, arXiv:2503.08569) — Zhu, M. et al. — 13k human-like-deep-thinking review traces designed for training and benchmarking staged reviewers. HF dataset: WestlakeNLP/DeepReview-13K.

**(21) The Good, the Bad and the Constructive: Automatically Measuring Peer Review's Utility for Authors (RevUtil)** — arXiv:2509.04484 (2025) — Sadallah, A. et al. Four-axis evaluation of individual review comments — Actionability / Grounding & Specificity / Verifiability / Helpfulness — with fine-tuned graders that match GPT-4 on utility judgments. Important because it decouples *review goodness for authors* from *review-score prediction* and operationalises a finer-grained alternative the eval-designer can use. Repo: github.com/bodasadallah/RevUtil (6 stars).

**(22) Peer Review as A Multi-Turn and Long-Context Dialogue with Role-Based Interactions (ReviewMT)** — arXiv:2406.05688 (2024) — Tan, C. et al. Reformulates peer review as a multi-turn author↔reviewer↔AC dialogue with a dedicated dataset and evaluation metrics. Useful for MegaResearcher because the rebuttal-loop dynamics are exactly what the synthesist's audit trail needs to mirror. Repo: github.com/chengtan9907/reviewmt (28 stars).

### 2c. LLM-as-judge failure modes and meta-critique (what *can't* be trusted)

**(23) Unveiling the Merits and Defects of LLMs in Automatic Review Generation for Scientific Papers** — arXiv:2509.19326 (2025) — Li, R. et al. Knowledge-graph-based comparison of generated vs. human reviews. Headline finding: **LLMs produce descriptive and affirmational content well but fail to identify weaknesses and fail to adjust feedback to paper quality.** Quality-insensitivity is a named known failure mode — directly relevant to BadScientist's findings.

**(24) Pre-review to Peer review: Pitfalls of Automating Reviews using Large Language Models** — arXiv:2512.22145 (2025) — Akella, A.P.; Siravuri, H.V.; Rohatgi, S. Frontier open-weight LLMs show limited alignment with human peer reviewers but moderate alignment with post-publication metrics; conclusion: pre-review screening yes, autonomous review no. Useful for setting the appropriate role MegaResearcher should give an LLM reviewer.

**(25) ReviewGuard: Enhancing Deficient Peer Review Detection via LLM-Driven Data Augmentation** — arXiv:2510.16549 (2025) — Zhang, H. et al. Detects deficient *human and AI* reviews; trained on ICLR/NeurIPS OpenReview data with GPT-4.1 synthetic augmentation. Critical because it provides a deficient-review detector — useful as a downstream gate on MegaResearcher's own reviewer outputs. Repo: github.com/haoxuan-unt2024/ReviewGuard.

**(26) Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review** — arXiv:2502.19614 (2025) — Yu, S. et al. (Intel). Distinguishes AI-written from human-written peer reviews; the context-aware detector beats stylometry. The threat-model framing — that humans use LLMs to ghostwrite reviews — is the dual of the spec's question (using LLMs to *do* the review) and should inform the audit-trail design.

**(27) CiteAudit: You Cited It, But Did You Read It? A Benchmark for Verifying Scientific References in the LLM Era** — arXiv:2602.23452 (2026) — Yuan, Z. et al. Multi-agent verification pipeline for fabricated citations. Reports that hallucinated citations have already appeared in *accepted* ML papers. The reference-faithfulness layer the spec needs underneath any LLM-generated related-work section.

**(28) Position: Machine Learning Conferences Should Establish a "Refutations and Critiques" Track** — arXiv:2506.19882 (2025) — Schaeffer, R. et al. (Stanford). Position paper that explicitly names the failure mode: "misleading, incorrect, flawed or perhaps even fraudulent studies being accepted." Cited not for hypotheses but as the venue-level frame the spec's "main-track-grade" target is implicitly arguing against.

## 3. Datasets

All HF dataset pages were resolved through paper-linked `find_datasets`. Licences flagged where present.

| Dataset | HF page | Linked paper | Licence |
|---|---|---|---|
| **allenai/peer_read** | huggingface.co/datasets/allenai/peer_read | PeerRead (1804.09635) | Original PeerRead T&Cs (CC-BY-compatible, expert-generated) — flag for reviewer-text redistribution |
| **WestlakeNLP/DeepReview-13K** | huggingface.co/datasets/WestlakeNLP/DeepReview-13K | DeepReview (2503.08569) | `license:other` — **paywall/redistribution flag**: derived from OpenReview content |
| **ECNU-SEA/SEA_data** | huggingface.co/datasets/ECNU-SEA/SEA_data | SEA (2407.12857) | Apache-2.0 |
| **ut-amrl/SPECS-Review-Benchmark** | huggingface.co/datasets/ut-amrl/SPECS-Review-Benchmark | AAAI-26 Pilot (2604.13940) | CC-BY-4.0 — controlled flaw injection across Story/Presentation/Evaluations/Correctness/Significance |
| **badscientist/BadScientist-Prompts** | huggingface.co/datasets/badscientist/BadScientist-Prompts | BadScientist (2510.18003) | MIT, **research-only**, gated — adversarial prompts that fool LLM reviewers |
| **CycleResearcher Review-5k / Research-14k** | Referenced in 2411.00816; not surfaced as separate HF dataset via `find_datasets` | CycleResearcher (2411.00816) | Project-published; check repo for licence |
| **DeepReviewer-14B model card** | github.com/zhu-minjun/Researcher | DeepReview (2503.08569) | Open release |

**Note on PeerRead reviewer text:** Reviewer comments in PeerRead come from real OpenReview submissions. Even with the dataset publicly redistributable, downstream eval-designer use should treat reviewer text as semi-identifiable. Flag for the synthesist.

## 4. Reference implementations

| Repo | Stars | Paper / Function |
|---|---|---|
| `sakanaai/ai-scientist` | 13560 | AI Scientist v1 — incl. automated reviewer module |
| `SakanaAI/AI-Scientist-v2` | 6157 | AI Scientist v2 — agentic tree search + reviewer |
| `weixin-liang/llm-scientific-feedback` | 531 | GPT-4 feedback comparison vs. human reviewers (2310.01783) |
| `allenai/PeerRead` | 428 | PeerRead corpus + baselines |
| `zhu-minjun/Researcher` | 379 | DeepReviewer-14B + DeepReview-13K |
| `ecnu-sea/SEA` | 89 | SEA reviewer framework |
| `chengtan9907/reviewmt` | 28 | Multi-turn review dialogue framework |
| `papercopilot/iclr-insights` | 15 | ICLR review-dynamics analysis tooling |
| `zhaolingao/reviewer2` | 16 | Reviewer2 aspect-then-review pipeline |
| `guijinSON/SPOT` | 9 | Verification benchmark + scaffolding |
| `bodasadallah/RevUtil` | 6 | Review-comment utility graders |
| `RichardLRC/Peer-Review` | 3 | Merits/defects KG benchmark code (2509.19326) |
| `haoxuan-unt2024/ReviewGuard` | 0 | Deficient-review detector |

The github_examples tool returned an internal error on the one query attempted; star counts above are from `hf_papers` paper records.

## 5. Open questions I noticed while reading (NOT hypotheses)

These are gaps the literature itself leaves open. The gap-finder / hypothesis-smith decide what to do with them.

1. **The CycleResearcher MAE-below-human-disagreement claim** — every other paper that has measured human-LLM agreement on review *scores* (notably 2503.05712, 2509.19326, 2512.22145) finds LLM reviewers worse than a trivial title+abstract baseline. The eval-designer needs to reconcile.
2. **No paper isolates the workshop-vs-main-track quality delta.** AI Scientist v2 cleared a workshop; nothing has even attempted main track on the published record. The gap-finder should ask whether the gap is reviewer-side (workshop reviewers cluster differently) or generator-side (workshop bar is lower).
3. **Presentation-overweighting is documented (BadScientist, FactReview, 2509.19326) but the magnitude is not quantified on a common axis.** SPECS-Review-Benchmark's Story-vs-Correctness split is the closest we have, but no paper has cross-evaluated all major reviewer systems on it.
4. **Rebuttal handling is essentially absent.** ReviewMT (2406.05688) is the only system that even attempts multi-turn rebuttal; the AAAI-26 pilot does not appear to evaluate whether the LLM reviewer's score moves with the same dynamics human scores do on rebuttal (which 2511.15462 *did* characterise for humans).
5. **Self-evaluation by an AI-Scientist-style closed loop has no external auditor.** The reviewer module in AI Scientist (v1, v2) is the same family of model that wrote the paper. None of the literature pressure-tests how badly this collapses on the BadScientist-style adversarial generator.
6. **Citation-faithfulness is treated as a separate problem** (CiteAudit 2602.23452, 17% Gap 2601.17431) but no end-to-end reviewer system *requires* citation verification as a precondition before scoring; FactReview is closest but still post-hoc.
7. **What dimensions are LLM-judge-reliable?** ReviewerToo claims "high accuracy in specific domains" but the published abstract does not name which dimensions are reliable vs. which need humans — the actual breakdown is needed for the spec's hand-off contract.
8. **Quality-insensitivity** (LLMs giving similar feedback regardless of paper quality, per 2509.19326) — no system in the corpus repairs this directly; ReviewRL's composite reward gestures at it but does not isolate quality-sensitivity as a metric.
9. **Cross-venue generalisation.** Nearly every system is trained on ICLR/NeurIPS/ACL OpenReview data; no published evaluation of how these reviewers behave on a venue with different review templates (TMLR, Nature ML, EMNLP industry track).

## Known-untrustworthy proxies (explicitly named, per spec requirement)

For the eval-designer's downstream use, the following measures are *demonstrably* unreliable as paper-quality proxies and should not be used in isolation:

- **Single-shot LLM-as-judge review score** — beaten by a title+abstract regression baseline (2503.05712). Untrustworthy.
- **GPT-4 helpfulness self-report from authors** (2310.01783) — does not distinguish presentation polish from soundness; falls to BadScientist-style attacks (2510.18003).
- **Self-evaluation by the same model family that generated the paper** — AI Scientist v1/v2 reviewer modules; no external audit in the published evaluations.
- **Reviewer–reviewer overlap rate alone** — used in 2310.01783; high overlap can be achieved by repeating generic observations (named in 2509.19326 as the "descriptive/affirmational" failure mode).
- **Acceptance-score prediction trained on PeerRead** — confounded by venue-and-year acceptance-rate drift and presentation-style shifts.

Trustworthy *enough to use as a triangulation channel*:

- **Citation-count prediction over a horizon ≥1 year** (2503.05712) — slow, but the only published metric that correlates non-trivially with eventual impact.
- **Controlled flaw-injection benchmarks** — SPECS (2604.13940), SPOT (2505.11855), BadScientist (2510.18003) — because the ground truth is constructed.
- **Multi-axis review-comment utility** (RevUtil, 2509.04484) — actionability/grounding/verifiability/helpfulness, scored per-comment rather than per-paper.
- **Deficient-review detection** (ReviewGuard, 2510.16549) — useful as a gate on the reviewer's own output, not as a paper-quality measure.

## 6. Sources

arXiv IDs cited above (all resolve via `hf_papers paper_details`):

- 2408.06292 — AI Scientist (Lu et al.)
- 2504.08066 — AI Scientist v2 (Yamada et al.)
- 2412.11948 — OpenReviewer (Idahl & Ahmadi)
- 2411.00816 — CycleResearcher (Weng et al.)
- 2503.08569 — DeepReview (Zhu et al.)
- 2510.08867 — ReviewerToo (Sahu et al.)
- 2407.12857 — SEA (Yu et al.)
- 2604.04074 — FactReview (Xu et al.)
- 2604.13940 — AAAI-26 AI Review Pilot (Biswas et al.)
- 2505.14376 — AutoRev (Chitale et al.)
- 2402.10886 — Reviewer2 (Gao et al.)
- 2508.10308 — ReviewRL (Zeng et al.)
- 2505.03332 — PWP (Markhasin)
- 1804.09635 — PeerRead (Kang et al.)
- 2310.01783 — Can LLMs provide useful feedback? (Liang et al.)
- 2511.15462 — ICLR Insights (Kargaran et al.)
- 2503.05712 — Automatic Evaluation Metrics for AI-Generated Research (Höpner et al.)
- 2505.11855 — SPOT (Son et al.)
- 2510.18003 — BadScientist (Jiang et al.)
- 2509.04484 — RevUtil (Sadallah et al.)
- 2406.05688 — ReviewMT (Tan et al.)
- 2509.19326 — Merits & Defects of LLMs in Review (Li et al.)
- 2512.22145 — Pre-review to Peer review (Akella et al.)
- 2510.16549 — ReviewGuard (Zhang et al.)
- 2502.19614 — AI Text Detection in Peer Review (Yu et al.)
- 2602.23452 — CiteAudit (Yuan et al.)
- 2506.19882 — Refutations & Critiques Track (Schaeffer et al.)
- 2401.15641 — PRE (Chu et al.)
- 2601.17431 — The 17% Gap (referenced for citation faithfulness)

HF dataset / repo URLs:
- https://huggingface.co/datasets/allenai/peer_read
- https://huggingface.co/datasets/WestlakeNLP/DeepReview-13K
- https://huggingface.co/datasets/ECNU-SEA/SEA_data
- https://huggingface.co/datasets/ut-amrl/SPECS-Review-Benchmark
- https://huggingface.co/datasets/badscientist/BadScientist-Prompts

GitHub URLs (subset, full list in §4):
- https://github.com/sakanaai/ai-scientist
- https://github.com/SakanaAI/AI-Scientist-v2
- https://github.com/weixin-liang/llm-scientific-feedback
- https://github.com/allenai/PeerRead
- https://github.com/zhu-minjun/Researcher
- https://github.com/ecnu-sea/SEA
- https://github.com/papercopilot/iclr-insights
- https://github.com/guijinSON/SPOT
- https://github.com/bodasadallah/RevUtil
- https://github.com/chengtan9907/reviewmt
- https://github.com/haoxuan-unt2024/ReviewGuard
- https://github.com/zhaolingao/reviewer2

**One paper considered and dropped:** arXiv:2604.19792 (OpenCLAW-P2P v6.0) — abstract reads as a self-published artifact with no external evaluation against the named benchmarks; dropped to keep the bibliography to citable, evaluated systems.
