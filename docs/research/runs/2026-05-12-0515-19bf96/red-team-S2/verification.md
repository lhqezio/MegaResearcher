# Verification — red-team-S2 (REVISION-1 pass)

## Verification-before-completion checks

### A. Independent literature queries (>=3 required)

Five independent queries executed via `mcp__plugin_megaresearcher_ml-intern__hf_papers` operation=search during this revision-1 pass:

1. **Q1** — `length bias LLM judge automated paper review post-hoc debias`
   - Hits: 10. Most relevant: Bias Fitting (2505.12843), Dubois (2404.04475), RBD (2505.17100), Justice or Prejudice (2410.02736), Position Bias study (2406.07791), CalibraEval (2410.15393), Verbosity Bias in Preference Labeling (2310.10076), Toward Robust LLM-Based Judges (2603.08091).
   - Result: none apply length-debiasing as a wrapper inside an AI-Scientist-family pipeline. Gap survives.

2. **Q2** — `AI Scientist v2 reviewer judge length debias scalar score wrapper`
   - Hits: 8. Most relevant: NAIPv2 (2509.25179, pairwise paper-quality estimation, not a wrapper), DeepReview (2503.08569, multi-stage paper-review framework, not a length-debiaser), J1 (2505.11875, test-time-scaling LLM-as-judge), Bias Fitting (2505.12843), Length-Controlled AlpacaEval (2404.04475).
   - Result: none wrap an off-the-shelf scalar-score length-debiaser around AI-Scientist-family judge calls. Gap survives.

3. **Q3** — `CycleResearcher AgentRxiv ResearchBench reviewer verbosity LLM judge bias`
   - Hits: 8. CycleResearcher (2411.00816), AgentReview (2406.12708), Mitigating the Bias of LLM Evaluation (2409.16788), Verbosity Bias in Preference Labeling (2310.10076), RBD (2505.17100), Justice or Prejudice (2410.02736), Position Bias study (2406.07791), Bias in the Loop / Software Engineering Judge (2604.16790).
   - Result: no AI-Scientist-family system applies a Bias-Fitting-style wrapper to its judge calls. Two relevant uncited papers (2310.10076, 2409.16788) flagged as Suggestion S-new-1 / S-new-2.

4. **Q4 (sanity)** — `AI Scientist length bias reviewer agent autonomous paper writing 2025`
   - Hits: 10. AI Scientist v1 (2408.06292), v2 (2504.08066), Jr. AI Scientist (2511.04583), AI-Researcher (2505.18705), AstaBench (2510.21652), CycleResearcher (2411.00816), AI-Assisted Peer Review at AAAI-26 (2604.13940), Why LLMs Aren't Scientists Yet (2601.03315), Sakana evaluation (2502.14297), Toward Autonomous Long-Horizon Engineering (2604.13018).
   - Result: none of the AI-Scientist-family systems document a length-debias wrapper on judge calls. Gap survives.

5. **Q5 (sanity)** — `Bias Fitting LLM-as-judge length debias scalar score post-hoc API`
   - Hits: 8. Top hit confirms Bias Fitting (2505.12843) is the only published scalar-score, non-pairwise, non-linear length-debiaser explicitly designed for "raw reward exhibits length bias; we want a debiased scalar reward" use case. The smith's pivot from Dubois to Bias Fitting is well-motivated.

**All five queries support the gap claim under the narrowed scope.**

### B. Citation spot-checks (>=3 required)

1. **arXiv:2505.12843 (Bias Fitting)** — verified via paper_details + read_paper section=3,4,5.
   - Architecture: length-encoding (sinusoidal positional-encoding analog, d=32) -> 2-layer ResNet -> linear regression head. Confirmed.
   - Loss: `-|Pearson| + MSE` against `r_detach`. Confirmed.
   - Input: scalar response length. No pairwise preference labels needed for fitting. Confirmed.
   - Debiased reward: `r(x,y) - model_f(len(y))`. Confirmed.
   - Empirical claim that non-linear fitting beats linear-debiasing baseline (ODIN): confirmed in section 4.2 (LC-WR improvements).
   - **Methodological caveat (I-new-1):** the paper's protocol assumes an internally-trained reward model with a warm-up phase (section 3.1 trains a Bradley-Terry RM from scratch). The smith's setup applies the fitting model directly to API-judge scalar outputs, bypassing the warm-up. Mathematically the fitting math doesn't require warm-up, but the paper does not test this direct-API-application setup. Defensible extrapolation, but should be flagged in M4.

2. **arXiv:2503.18102 (AgentRxiv) section 4** — verified via read_paper section=4.
   - Section 4 is "Limitations". Section 4.1 is "Agent hallucination & reward hacking" and documents fabrication of method scores in the code/results path due to mle-solver vs code-repair tension.
   - Direct quote: "papers that are reporting higher method scores [are] rated higher by the reward function." This is score-fabrication, not verbosity reward hacking.
   - The revision correctly characterizes this in section 1 (channel 3), M3 (explicit forecast flag), and section 8 sources. C1 fix verified.

3. **arXiv:2510.18003 (BadScientist)** — verified via paper_details.
   - Five fabrication strategies (TooGoodGains, BaselineSelect, StatTheater, CoherencePolish, ProofGap), all non-length-based. Up to 82% acceptance on o3/o4-mini/GPT-4.1 reviewers. Concern-acceptance conflict pattern documented.
   - The revision cites accurately in section 1 (sub-dominance framing), section 4 (upper-bound on contribution), section 5 F3 (3 BadScientist-inspired substantive proxies), section 7 R6 (BadScientist-dominance risk + future-work flag exit).

4. **arXiv:2407.19594 (Meta-Rewarding)** — re-verified from prior pass.
   - The smith now correctly disambiguates Meta-Rewarding's "length-control mechanism" (implemented at DPO-pair-selection stage, requires pairwise training data) from the scalar-score wrapper this hypothesis tests (Bias Fitting). The 22.92% -> 39.44% AlpacaEval LC magnitude is correctly cited as M2 grounding for "length-control has produced large gains in training loops," not as evidence of S2's specific mechanism. Honest disambiguation.

### C. Verdict-severity coherence

- Verdict: APPROVE.
- Critical count: 0.
- Important count: 4 (I-new-1 through I-new-4, all addressable by eval-designer in protocol design).
- Suggestion count: 3 (S-new-1, S-new-2, S-new-3, citation additions).

**Coherence check:** APPROVE with 0 Critical is consistent. The 4 Important objections are scoped to eval-designer's protocol-design surface, not the hypothesis's mechanism or falsification design. The hypothesis-smith has done the structural work; the remaining work is operational protocol specification.

### D. Discipline-rule audit (receiving-code-review applied to research artifact)

- **No performative agreement.** I argued explicitly through the "kill, don't run" steelman (graceful-no-op + BadScientist-dominance + R6 future-work-flag admission) before approving. The steelman has real force and was rebutted with concrete counter-arguments (F1 is missing empirical data on length-bias-in-paper-judges; F4 substrate is amortizable; wrapper is the cheapest reviewer-hardening intervention) rather than hand-waving.

- **No performative skepticism.** The three prior Critical objections are genuinely addressed by the revision via three structural moves: methodological pivot (Dubois -> Bias Fitting), scope narrowing (precondition -> co-defense), and explicit forecast-tagging in M3. Inventing new Critical objections to look rigorous would be performative.

- **Citations resolve.** All cited arXiv IDs in the revision (2505.12843, 2503.18102, 2510.18003, 2407.19594) verified via hf_papers paper_details / read_paper. No fabricated citations.

- **APPROVE only if I would defend publicly.** The narrowed hypothesis is a defensible transfer-test contribution. Graceful-no-op + future-work-flag + BadScientist-dominance exits are all explicit, so even the worst-case outcome is publishable as a survey finding. I would defend the narrowed S2 as workshop-grade calibration-study material.

### E. Output artifact completeness

- output.md: present (verdict line at end, all 9 required sections covered).
- manifest.yaml: present, verdict APPROVE, counts coherent with output.md.
- verification.md: this file, all 5 checks satisfied.
