# Verification — hypothesis-smith-S6 (revision 1)

Verification per the hypothesis-smith required checks (superpowers:verification-before-completion adapted) and revision-1 obligation that every red-team objection has an explicit response.

## Required-check audit

| Check | Status | Evidence |
|---|---|---|
| Hypothesis statement is in if/then form | PASS | §2 opens "If MegaResearcher's swarm is augmented with a leaf worker `ablation-coverage-checker` that … AND a follow-up eval-designer wave is dispatched with that artifact as input, then on a held-out 20-paper sample of AblationBench's AuthorAblation test set, the full two-wave configuration achieves AuthorAblation F1@5 ≥ 0.03 above the strongest of (B1, B2, B3)…". Antecedent and consequent both binary-checkable. |
| At least 3 falsification criteria, each genuinely falsifiable | PASS | §5 lists F1 (F1@5 lift threshold), F2 (composite deterministic file-diff with 3 sub-conditions all must pass — addresses C3 and O6), F3 (two-arm rubric sweep, narrowed per O7). F4 (cost) and F5 (AbGen non-transfer) listed as supplementary, not counting toward floor. |
| Every mechanism claim has a citation | PASS | §3 steps (a)/(b)/(c) all cited. Step (a) cites `arXiv:2507.08038` §4.2 (rubric structure) and §7.4 (p=0.009 magnitude). Step (b1) cites gap-finder-1 §2 Rank 1; (b2) cites `arXiv:2506.11930` §3.2 *qualitatively only* per O5 fix (magnitudes from AIME/GPQA NOT imported); (b3) cites gap-finder-1 §4 item 4. Step (c) cites `arXiv:2504.08066` (with O8 caveat about abstract-level claim), `arXiv:2505.18705`, `arXiv:2408.06292`, `arXiv:2503.18102`, `arXiv:2502.16069` (§3 with O10 caveat), `arXiv:2509.08713`. |
| All cited arxiv IDs resolve via `hf_papers paper_details` | PASS | Verified at revision-1 submission time. 10 IDs in §8 resolved. AbGen (`arXiv:2507.13300`) freshly verified via `paper_details` operation — 20 upvotes, Yale, 1500 expert-annotated examples from 807 NLP papers confirmed. AblationBench §4.2, §4.3, §7.4 re-read directly via `read_paper` operation. AbGen §2.6 (data stats) and §3.3 (automated eval) re-read directly. |
| Risks-to-the-hypothesis section is non-empty | PASS | §7 has 4 distinct risks (R1 model-specific lift, R2 rubric wrong, R3 file-handoff equivalent to in-context concat — i.e., C1 collapse re-materializing at architectural level, R4 AbGen/AblationBench disagreement). Each ends with "what the hypothesis still contributes if this materializes." |
| Differential-effect attack pre-emption | PASS | §4 names three baselines explicitly: B1 LM-Planner Claude 3.5 Sonnet, B2 AI Scientist v2, B3 AI-Researcher. Δ pre-registered at +0.03 absolute F1@5 with lower 95% CI > 0. |
| Non-judge / deterministic signal present | PASS (HONEST per C3) | F2 deterministic co-primary (file-diff over structured YAML; no LLM judge anywhere). AblationBench F1@5 acknowledged as LM-judge-mediated per §4.3 with cross-family majority-vote-of-3 and 0.74 F1 vs humans (§7.3 Table 3); not labeled "non-judge" anywhere in revision. AbGen human-eval column (§3.2) used as robustness signal, not GPT-4.1-mini automated column. |
| Pre-registration of decision thresholds | PASS | §6 — primary "Δ F1@5 ≥ +0.03 absolute, lower 95% CI > 0"; co-primary F2 composite (≥3 flags AND ≥60% incorporation AND ≥50% GT-hit precision); robustness F5 (≥+0.10 Likert on AbGen). All "locked at hypothesis-submission time." |
| Scope of "should NOT hold" pre-registered | PASS | §4 carves out ReviewerAblation (per AblationBench §7.4 grounding trade-off), modification-type ablations (per §7.4 domain-knowledge bottleneck), AbGen non-transfer (treated as substrate-portability limitation per F5, not F1-falsification). |
| Risks framed honestly (each risk includes "what the hypothesis still contributes") | PASS | R1–R4 each have follow-up. R3 in particular routes the worst case ("file-handoff substrate is functionally equivalent to in-context concatenation") to a measurement contribution about the architectural-equivalence claim. |

## Red-team revision-1 objections — explicit address per critical defect

The §0 section "Response to red-team revision-1 objections" of output.md addresses each objection by ID. Confirmation here:

| Objection | Severity | Where addressed | Concession? |
|---|---|---|---|
| C1 mechanism collapse on Step (a) | Critical | §0 C1, §2 (rewritten hypothesis), §3 step (a) reframed as input not contribution, §3 step (b) reframed as novel mechanism | Full concession; mechanism shifted to file-handoff substrate |
| C2 F1 → recall@5 metric substitution | Critical | §0 C2, §2, §4 magnitude reasoning, §5 F1 | Full concession; primary metric locked to F1@5, Δ reduced to +3pp |
| C3 non-judge claim false | Critical | §0 C3, §2 (cross-family majority-vote-of-3 explicit), §4 (both falsifiers), §5 F2 (elevated to co-primary), §6 falsifier breakdown | Full concession; deterministic F2 co-primary, AblationBench acknowledged LM-judge-mediated |
| C4 AbGen omission | Critical | §0 C4, §1 (added to gap source), §4 (added as robustness substrate), §5 F5, §6, §8 (added to sources) | Full concession; AbGen verified and added with human-eval column |
| O5 Feedback Friction domain transfer | Important | §0 O5, §3 step (b2) | Full concession; numeric magnitudes dropped, only qualitative F3>F1 direction retained |
| O6 F2 trivializability | Important | §0 O6, §5 F2 | Full concession; precision-of-flags floor added (≥3 flags AND ≥50% GT-hit precision) |
| O7 F3 third arm under-specified | Important | §0 O7, §5 F3 | Full concession; hand-curated arm dropped, F3 swept across two AblationBench rubrics only |
| O8 AI Scientist v2 coverage claim granularity | Important | §0 O8, §3 step (c) | Partial concession; language softened, eval-designer asked to treat as open measurement question |
| O9 ID space under-specified | Suggestion | §0 O9, §6 | Resolved; IDs are AblationBench GT ablation-name strings, fallback deterministic slug |
| O10 Curie rigor module overlap | Suggestion | §0 O10, §3 step (c) | Resolved; Curie's rigor module declared orthogonal to external coverage rubric |
| O11 field-weak overclaim | Suggestion | §0 O11, §1, §3 step (b1) | Conceded; language narrowed to "no AI-Scientist-family pipeline integrates an external ablation-coverage rubric as a worker with cross-wave file handoff" |

## Citation re-verification (revision-1)

Live re-checks performed at revision-1 submission:

- **`hf_papers paper_details arxiv_id=2507.13300`** — AbGen verified: 1500 expert-annotated examples from 807 NLP papers, 20 upvotes, Zhao et al. (Yale), GitHub yale-nlp/AbGen, headline finding "unreliability of current automated evaluation methods" matches red-team's description.
- **`hf_papers read_paper arxiv_id=2507.08038 section=4.2`** — AuthorAblation Planner section re-read directly. Confirmed: LM-Planner uses "single CoT style prompt to an LM for generating ablation plan" with "title, abstract, and aggregated TeX source," outputs "structured JSONL file containing up to k ablation entries" with "removal or modification of a component" typology. The red-team's C1 read is correct: the rubric structure IS in the LM-Planner baseline.
- **`hf_papers read_paper arxiv_id=2507.08038 section=4.3`** — AuthorAblation Judge section re-read. Confirmed: "we design an LM-based judge"; LMJudge (CoT) and AgentJudge (ReAct) scaffoldings; cross-family majority-vote-of-3 ensemble with positional and contextual debiasing. Red-team's C3 read is correct.
- **`hf_papers read_paper arxiv_id=2507.08038 section=7.4`** — Experimental Results re-read. Confirmed: +8 F1 percentage points (NOT recall@5) Agent-Planner → LM-Planner with Claude 3.5 Sonnet, p=0.009 paired t-test on F1 scores. Grounding trade-off across AuthorAblation/ReviewerAblation confirmed. Modification-type domain-knowledge bottleneck quote confirmed.
- **`hf_papers read_paper arxiv_id=2507.13300 section=3.3`** — AbGen Automated Evaluation section re-read. Confirmed: GPT-4.1-mini as base evaluator; four criteria (importance, faithfulness, soundness, overall) at 1–5 scale; AbGen §4.2 finding "current automated evaluation systems may not be fully reliable for our task" confirmed in conjunction with Table 2 disparity between LLM-based and human evaluations.
- **`hf_papers read_paper arxiv_id=2507.13300 section=2.6`** — AbGen data stats: 500 testmini + 1000 test confirmed; 1500 total expert-annotated examples from 807 NLP papers confirmed.

All other arxiv IDs (2506.11930, 2504.08066, 2505.18705, 2408.06292, 2503.18102, 2509.08713, 2502.16069, 2510.18003) were verified at revision-0 submission time and remain in the source list with no new claims added that would need fresh verification.

## Conformance to lane

- Hypothesis-smith forges hypotheses, does not design experiments. §6 remains a sketch; the file-handoff vs in-context arm (R3 diagnostic) is named but eval-designer formalizes the protocol.
- No code written, no experiments executed.
- Hypothesis is ONE hypothesis, scoped to S6 only.
- All claims cite prior art; no claim relies on hypothesis-smith's own reasoning beyond what the chain (a)+(b)+(c) supports.

## Honest disclosure (per revision-1 instructions)

The revised predicted lift (+3 F1pp) is **not a main-track-conference primary contribution by itself**. It is a measurement of whether MegaResearcher's stateless-leaf-dispatch + file-handoff architecture can integrate an external published rubric across waves with non-zero downstream value. The synthesist is asked in the manifest's `honest_disclosure` field to position S6 as **system-integration scaffolding rather than primary contribution**. This is the cost of fixing C1 honestly rather than papering over.

## Outstanding items

- None blocking submission. Every Critical objection (C1–C4) and every Important/Suggestion objection (O5–O11) is addressed with either a fix or an explicit concession.
- The file-handoff-vs-in-context diagnostic arm (§6) is the most informative experiment in the design and the most direct test of whether the C1 reframe is honest — if that arm shows no advantage for file-handoff, R3 materializes and the hypothesis collapses at the architectural level, which is the correct falsification.

Verification complete (revision 1).
