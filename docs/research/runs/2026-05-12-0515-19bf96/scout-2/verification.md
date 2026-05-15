# Scout-2 Verification (superpowers:verification-before-completion)

Worker: literature-scout (scout-2)
Sub-topic: manuscript drafting and document-scale coherence
Run: 2026-05-12-0515-19bf96

## Checks performed

### 1. Every cited arXiv ID resolves via `hf_papers paper_details`

I ran `paper_details` directly against the following arXiv IDs and confirmed each returned a populated record (authors, abstract, keywords). Spot-checks below.

| arXiv ID | Title (as returned) | Verified? |
|---|---|---|
| 2402.14207 | Assisting in Writing Wikipedia-like Articles From Scratch (STORM) | YES — authors Shao, Jiang, Kanell, Xu, Khattab, Lam |
| 2408.06292 | The AI Scientist | YES — authors Lu, Lu, Lange, Foerster, Clune, Ha |
| 2504.08066 | The AI Scientist-v2 | YES — authors Yamada, Lange, Lu, Hu, Lu, Foerster, Clune, Ha |
| 2505.18705 | AI-Researcher: Autonomous Scientific Innovation | YES — authors Tang, Xia, Li, Huang |
| 2408.07055 | LongWriter | YES — authors Bai, Zhang, Lv, Zheng, Zhu, Hou, Dong, Tang, Li |
| 2406.10252 | AutoSurvey | YES — authors Wang, Guo, Yao, Zhang, Zhang, Wu, Zhang, Dai, Zhang, Wen |
| 2503.04629 | SurveyForge | YES — authors Yan, Feng, Yuan, Xia, Wang, Zhang, Bai |
| 2502.14776 | SurveyX | YES — authors Liang et al. |
| 2510.07733 | SurveyG | YES — authors Nguyen, Nguyen, N.T., Dang, Dong, Le |
| 2506.12689 | SciSage | YES — authors Shi, Kou, Li, Tang, Xie, Yu, Wang, Zhou |
| 2509.19370 | Meow | YES — authors Ma, Shan, Zhao, Xu, Wang |
| 2503.18102 | AgentRxiv | YES — authors Schmidgall, Moor |
| 2501.04227 | Agent Laboratory | YES — authors Schmidgall, Su, Wang, Sun, Wu, Yu, Liu, Liu, Barsoum |
| 2511.04583 | Jr. AI Scientist | YES — authors Miyai, Toyooka, Otonari, Zhao, Aizawa |
| 2503.23229 | Citegeist | YES — authors Beger, Henneking |
| 2407.12861 | CiteME | YES — authors Press, Hochlehnert, Prabhu, Udandarao, Press, Bethge |
| 2412.14860 | Think&Cite | YES — authors Li, Ng |
| 2305.06983 | FLARE | YES — authors Jiang, Xu, Gao, Sun, Liu, Dwivedi-Yu, Yang, Callan, Neubig |
| 2409.16191 | HelloBench | YES — authors Que, Duan, He, Mou, Wang, Liu, Rong, Wang, Yang, Zhang |
| 2502.19103 | LongEval | YES — authors Wu, Li, Qu, Ravikumar, Li, Loakman, Quan, Wei, Batista-Navarro, Lin |
| 2510.03120 | SurveyBench | YES — authors Sun, Zhu, Zhou, Tong, Wang, Fu, Li, Liu, Wu |
| 2509.08713 | Hidden Pitfalls of AI Scientist Systems | YES — authors Luo, Kasirzadeh, Shah |
| 2309.11495 | Chain-of-Verification (CoVe) | YES — authors Dhuliawala, Komeili, Xu, Raileanu, Li, Celikyilmaz, Weston |
| 2408.15232 | Co-STORM | YES — authors Jiang, Shao, Ma, Semnani, Lam |
| 2506.18841 | LongWriter-Zero | YES — authors Wu, Bai, Hu, Lee, Li |
| 2503.00751 | RAPID | YES — authors Gu, Li, Dong, Zhang, Lv, Wang, Lian, Liu, Chen |
| 2504.05732 | LLMxMapReduce-V2 | YES — authors Wang, Fu, Zhang, Wang, Ren, Wang, Li, He, An, Liu |

The remaining IDs (`2412.08268`, `2405.01930`, `2510.15624`, `2403.02270`, `2601.17431`, `2502.14297`) appeared in `search` results with their titles, abstracts, and arXiv IDs returned by the same `hf_papers` MCP tool. I did not run a dedicated `paper_details` on each because they appear in supporting roles (datasets table, implementations table, open-questions/related-evaluation framing) rather than as headline entries. The spot-check standard required by the role contract — "record one spot-check" — was exceeded with 27 explicit `paper_details` verifications above.

### 2. No invented citations

Every paper in §2 (Key papers), §3 (Datasets), §4 (Reference implementations), §5 (Open questions), and §6 (Sources) of `output.md` was returned by either `hf_papers search` or `hf_papers paper_details` during this scout's run. No paper was added from memory. No paper that failed to resolve was retained.

Papers I considered but did not include because they did not surface a clear manuscript-drafting / document-coherence contribution: ResearchAgent (`2404.07738`), Paper Copilot (`2409.04593`), SciPIP (`2410.23166`), LitLLM (`2402.01788`), LitLLMs (`2412.15249`) — all surfaced in searches but their drafting/coherence contribution was tangential to this scout's sub-topic and they belong more naturally in scouts focused on idea generation or literature review.

### 3. Bibliography count meets the "at least 8" floor

The role contract requires ≥8 entries; the assignment-specific contract requires ≥8. `output.md` §2 contains 26 numbered entries across five sub-clusters (2A outline-driven hierarchical: 6 entries; 2B scientific-manuscript pipelines: 5 entries; 2C survey-generation: 6 entries; 2D citation-anchored: 4 entries; 2E benchmarks/evaluations: 5 entries). Floor exceeded by a factor of ~3, justified by the sub-topic's density of prior work in 2024–2026.

### 4. Every dataset cited has a verifiable HF page or licence note

§3 of `output.md` lists 13 datasets, each tied to a paper whose arXiv ID resolves. For each dataset I noted access path (paper repo / HF page) and flagged any licence questions requiring verification before downstream adoption. The "open-access preferred" spec constraint is satisfied with explicit flags where verification is still required.

### 5. Required fields present in every bibliography entry

The assignment-specific contract names the required per-entry fields: arXiv ID/link, one-paragraph summary, drafting architecture, coherence mechanism, benchmark/dataset (if applicable), what fails. I cross-checked all 26 entries:

- arXiv ID present: all 26
- one-paragraph summary present: all 26
- drafting architecture noted: 24 (the two benchmark-only entries — HelloBench `2409.16191` and SurveyBench `2510.03120` — describe evaluation methodology rather than a drafting architecture; this is appropriate per their nature)
- coherence mechanism noted: 24 (same two benchmark-only exceptions)
- benchmark/dataset noted when applicable: all 26 (explicitly stated or marked N/A)
- what fails / why-it-matters: all 26

### 6. Lane discipline (worker contract)

I produced a bibliography and flagged open questions. I did NOT propose hypotheses, design experiments, or write claims that need their own citation. Open-questions section is phrased as questions, not as testable hypotheses.

## Paywall flags

None. All 26 headline papers are arXiv preprints (open access). All 17 reference implementations are public GitHub repos with star counts ≥1. All HF-hosted datasets in §3 are accessible without authentication based on the paper repos linked.

## Named gaps where evidence is thin

- **Cross-section coherence checking for *research papers specifically*** — the closest published evidence (SciSage `2506.12689`, SurveyX `2502.14776` re-polish) is on surveys, not on research papers. Reported in §5 open questions but worth flagging here too as a literature gap.
- **Manuscript-quality ablations within the AI-Scientist family** — no controlled comparison of writeup-module architectures exists. Reported in §5.
- **Document-scale RL with venue-grade rubrics** — LongWriter-Zero is the only document-scale RL paper in scope, and its reward model targets generic quality. No published reward model targets main-track-conference acceptance signals.
- **Unresolved-citation rate measurement on autonomous-research-system outputs** — anecdotal but not systematically measured at scale (the closest is the 17% Gap paper `2601.17431`, which targets AI-assisted survey papers, not fully autonomous research papers).

## Verification result

PASS. All required artifacts produced at:
- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/scout-2/output.md`
- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/scout-2/manifest.yaml`
- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/scout-2/verification.md`

Status: complete.
