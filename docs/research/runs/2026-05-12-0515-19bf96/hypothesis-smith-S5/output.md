# hypothesis-smith-S5 — Deterministic block-on-unresolvable citation gate (revision 2)

## Response to red-team revision-2 objections

This revision engages the three NEW Critical defects and the three Important items from the red-team revision-2 critique. The narrow gap claim is further tightened against ARIS, the F4 falsifier is committed to a deterministic protocol identical to the gate's, the eval substrate is migrated to PaperRecon's PaperWrite-Bench, the F1 taxonomy is grounded against GSAP-NER with a hand-labeled completeness audit, and the §3 / §5 internal inconsistency about rewrite-as-prose is resolved (rewrite-to-prose counts as a drop; D ≤ 15% bound is now engineering judgment alone).

**CR1 — ARIS (arXiv:2605.03042) engaged head-on.** ARIS §3.2 describes `/citation-audit` with three axes: (i) existence (the cited paper resolves at claimed arXiv ID, DOI, or venue), (ii) metadata correctness, (iii) context appropriateness. Axis (i) overlaps the proposed gate's existence-check exactly. The architectural delta is therefore narrow but substantive, and engages three distinct dimensions:

1. **ARIS's citation-audit is advisory, not blocking.** ARIS §3.2 explicitly: "verdicts are recorded in a per-entry ledger and surfaced as KEEP/FIX/REPLACE/REMOVE recommendations *for human approval before submission*." ARIS §3.1 Stage 1 makes the pattern explicit: "The audit is advisory at the workflow level: it does not halt execution." The same human-in-loop pattern applies across the assurance stack. The proposed S5 gate, by contrast, blocks autonomously at emit time — H_treatment = 0 by construction, with no human gate required.
2. **ARIS's existence-check is LLM-judge based.** ARIS §3.2: "Verification uses fresh cross-family reviewers with web access." That is exactly the LLM-judge surface the smith expelled from F1. The proposed S5 gate is deterministic-resolver-based (`hf_papers paper_details` + Crossref). On the existence axis, deterministic resolution is operationally stronger than LLM-judge verification — it admits no false-positives from model hallucination of "I checked, looks fine."
3. **ARIS bundles three axes; the proposed gate scopes to one.** ARIS's context-appropriateness axis (axis iii) requires LLM judgment about whether a paper's content supports the surrounding claim. The proposed gate explicitly does NOT make this claim — it only enforces existence. The bundled-vs-unbundled question is itself a design choice; the gate's narrower scope makes the cost-of-enforcement bound (D ≤ 15%) measurable in a way ARIS's bundled audit cannot be measured.

**Updated narrow gap claim:** No AI-Scientist-family end-to-end system (now including ARIS in the population — see §1) implements a **block-on-unresolvable, deterministic-resolver, no-LLM-judge** citation gate that fires autonomously at emit time. ARIS is the closest peer; the three-dimensional architectural delta above is the contribution. If ARIS's citation-audit recommendations were converted to autonomous block-on-recommend behavior, the gap would collapse — but the architectural commitment in ARIS as published is advisory, not autonomous-blocking.

**Acknowledged contraction of magnitude:** the gap is materially smaller than this hypothesis claimed in revision-0 or revision-1. The contribution is now best framed as **(a) the architectural commitment delta against ARIS** (block vs advise; deterministic vs judge; existence-only vs bundled) and **(b) the cost-of-enforcement bound D ≤ 15% measured on PaperWrite-Bench**, which ARIS does not publish. The synthesist should downgrade the magnitude language in the final report accordingly.

**CR2 — F4 specified deterministically with identical resolver to the gate.** F4's off-corpus citation rate is now measured by extracting every arxiv-ID-shaped string from Ablation R's output via the same deterministic regex used by the gate, then firing `hf_papers paper_details` against each. A citation is "off-corpus" iff (a) it does not resolve OR (b) it resolves but its arxiv ID is not in the Citegeist retrieval-set returned for that prompt (Citegeist's retrieval-set is auditable via its open-source code). The threshold is pre-registered: if off-corpus rate < 2% across the held-out sample, retrieve-then-write subsumes the gate and the hypothesis collapses. The protocol is identical to the gate's resolver — no LLM-judge surface anywhere in F4.

**CR3 — PaperRecon (arXiv:2604.01128) cited and adopted as primary substrate.** The eval substrate migrates from the ad-hoc n=30 (10 each from v1/v2/AgentRxiv) to **PaperWrite-Bench (n=51 papers from top-tier venues post-2025)** as the primary substrate. PaperWrite-Bench's Hallucination dimension measures the same property as H_baseline / H_treatment. Same author cohort as Jr. AI Scientist (Miyai et al., already cited). MIT-licensed via the PaperRecon GitHub repo. The smaller AI-Scientist-family-only sample (n=30) is retained as a **secondary** substrate to support the AI-Scientist-family-specific narrow gap claim. F2 floors are recalibrated to PaperWrite-Bench's reported baseline of ">10 hallucinations per paper on average" for ClaudeCode, which translates to a per-citation rate substantially above the 5% total-phantom floor — F2 is therefore likely to pass cleanly on the primary substrate.

**Important items addressed:**

- **F1 / GSAP-NER (arXiv:2311.09860):** Class A detection is grounded in GSAP-NER's ML-scholarly NER pipeline (model + dataset entity types), not raw spaCy. Replication of GSAP-NER's published pipeline is the substrate; spaCy's `en_core_web_sm` is no longer the basis. An extraction-completeness audit on a 5-manuscript hand-labeled subset (drawn from the PaperWrite-Bench n=51) is pre-registered to bound measurement bias.
- **F1 / Class B regex coverage:** expanded to cover BLEU/ROUGE/F1/perplexity/loss/latency/FLOPs patterns with explicit named-metric anchors. Detection: `(?:BLEU|ROUGE-?[12L]?|F1|F-1|PPL|perplexity|loss|accuracy|MSE|MAE|latency|FLOPs|FLOP|parameters)\s*[:=]?\s*\d+\.?\d*` plus the original `(\d+\.?\d*\s*(%|pp|points|×|x|MB|GB))` plus a unit-bearing sample-count pattern `(\d+\.?\d*\s*[kKmM]?\s*(samples|examples|tokens))`. Extraction-completeness audited as above.
- **F1 / §3 Component C inconsistency RESOLVED:** rewrite-to-prose strips numerical values and IS counted as a drop under F1's matching rule. Component C's redundancy/rewrite mechanism story is therefore weakened — it bounds drop only via citation-level redundancy (multi-citation paragraphs), NOT via rewrite-to-prose. The D ≤ 15% prediction now rests on (i) redundancy alone and (ii) engineering judgment. If the protocol-step-1 measurement shows citation density < 2 per paragraph, the redundancy argument also collapses and D ≤ 15% becomes engineering judgment alone — this conditional is pre-registered as a measurement-contingent caveat (see §4).
- **n=10 per system sub-significant:** addressed by migrating to PaperWrite-Bench's n=51 primary substrate. Per-system reporting on the secondary n=30 sample is now explicitly pooled-only; per-system magnitudes are reported as point estimates with CIs but not as statistically distinct claims.
- **BibAgent (arXiv:2601.16993):** added to §8 sources as adjacent post-hoc miscitation-detection work. Position: BibAgent is post-hoc and downstream-consensus-based, not pre-flight; the gate addresses a different operational point in the pipeline.

**What the red-team's revised steelman (§8) buys vs does not buy:** the steelman concedes that the deterministic D measurement is genuinely novel even against ARIS. The smith accepts that the contribution is **workshop-paper magnitude on the architectural delta**, but **measurement-on-PaperWrite-Bench magnitude on the empirical contribution** — the first published D measurement on the canonical AI-paper-hallucination benchmark. The synthesist should report this with the contraction made explicit.

---

## 1. Targeted gap

Shortlist entry **S5** in gap-finder-3 — *"Citation pre-flight verification worker"*
([`/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/gap-finder-3/output.md`](../gap-finder-3/output.md) §(a) row S5, §(b) S5).

This closes the capability-matrix gap **Rank 3 — Citation Verification as a pre-flight gate** in gap-finder-1
([`/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/gap-finder-1/output.md`](../gap-finder-1/output.md) §2 Rank 3).

**Updated narrow gap claim (further tightened against ARIS):** Across the AI-Scientist-family end-to-end-experiment-and-paper population — the 14 systems in scout-1's matrix **plus ARIS (arXiv:2605.03042) added in this revision** — no system implements a **block-on-unresolvable, deterministic-resolver, no-LLM-judge** citation gate that fires autonomously at emit time. ARIS is the closest peer: its `/citation-audit` checks existence (axis i) but is advisory (human approval gate per §3.2), LLM-judge based ("fresh cross-family reviewers with web access" per §3.2), and bundles existence with metadata correctness and context appropriateness in one judge call. The three architectural dimensions where the proposed S5 gate is distinct from ARIS's citation-audit:

1. **Blocking vs advisory.** ARIS produces KEEP/FIX/REPLACE/REMOVE recommendations for human approval; S5 drops unresolvable citations autonomously.
2. **Deterministic resolver vs LLM-judge.** ARIS uses cross-family reviewers; S5 uses `hf_papers paper_details` + Crossref.
3. **Existence-only vs bundled.** ARIS audits existence + metadata + context-appropriateness; S5 scopes to existence only, which makes the cost-of-enforcement bound measurable as a clean axis.

**Broad gap claim explicitly NOT made:** Citegeist (arXiv:2503.23229), ScholarCopilot (arXiv:2504.00824), ARISE (arXiv:2511.17689), and ResearchPilot (arXiv:2603.14629) achieve a stronger guarantee via retrieve-then-write at generation time. The F4 falsifier tests subsumption directly.

Supporting prior-art citations (all resolved via `hf_papers paper_details`):

**AI-Scientist-family systems (the narrow gap target, now including ARIS):**

- **AI Scientist v1** (arXiv:2408.06292, Lu et al., 13.5k GitHub stars) — Semantic Scholar used as novelty check, not as a citation gate.
- **AI Scientist v2** (arXiv:2504.08066, Yamada et al., 6.1k stars) — single-pass writeup; no citation resolution gate before emit.
- **AgentRxiv** (arXiv:2503.18102, Schmidgall & Moor, 5.5k stars) — preprint server storing accepted reports; no pre-flight citation gate in the writer.
- **Jr. AI Scientist** (arXiv:2511.04583, Miyai et al., 30 stars) — baseline-paper-conditioned writer.
- **ARIS** (arXiv:2605.03042, Yang et al., 107 upvotes, 8890 stars, May 2026) — `/citation-audit` (existence + metadata + context-appropriateness axes), advisory KEEP/FIX/REPLACE/REMOVE recommendations for human approval, LLM-judge based. The closest peer to S5; the architectural delta is the three dimensions above.

**Retrieve-then-write systems (stronger constructive guarantee at generation time):**

- **Citegeist** (arXiv:2503.23229, Beger & Henneking) — dynamic RAG on the arXiv corpus for related-work generation. Citations retrieved from resolved arXiv IDs; unresolvable citations impossible at retrieval time.
- **ScholarCopilot** (arXiv:2504.00824, Wang et al., 43 upvotes, 250 stars) — Qwen-2.5-7B trained to emit a `[RET]` retrieval token; 40.1% top-1 citation accuracy vs 15.0% baseline.
- **ARISE** (arXiv:2511.17689, Wang et al.) — explicit Citation Preparation pipeline with Citation Retrieval Agent, Error List, Citation Completion Agent.
- **ResearchPilot** (arXiv:2603.14629, Zhang) — Semantic Scholar + arXiv multi-agent retrieval; retrieve-then-write architecture.

**Citation-verification systems (post-hoc / failure-mode documentation):**

- **CiteAudit** (arXiv:2602.23452, Yuan et al., 17 upvotes) — multi-agent post-hoc citation verification benchmark.
- **SemanticCite** (arXiv:2511.16198, Haan) — four-class classification via fine-tuned full-text analysis.
- **FactReview** (arXiv:2604.04074, Xu et al.) — evidence-grounded reviewer with claim extraction + execution-based verification.
- **CiteME** (arXiv:2407.12861, Press et al., 48 stars) — citation-attribution benchmark; deterministic binary scoring.
- **CiteGuard** (arXiv:2510.17853, Choi et al., 8 upvotes) — retrieval-aware citation attribution; 68.1% on CiteME with DeepSeek-R1.
- **LLM-Ref** (arXiv:2411.00294, Fuad & Chen) — iterative reference-handling writing assistant.
- **BibAgent** (arXiv:2601.16993, Li et al.) — agentic miscitation detection with downstream consensus, full-text-aware, MisciteBench (6,350 instances). Adjacent but post-hoc; the gate addresses a different operational point.

**Eval-substrate benchmarks:**

- **PaperRecon / PaperWrite-Bench** (arXiv:2604.01128, Miyai et al., 15 upvotes) — 51 papers from top-tier venues post-2025; orthogonal Presentation and Hallucination evaluation; baseline ">10 hallucinations per paper on average" for ClaudeCode. **Primary substrate** for this hypothesis's eval.

**Methodology baselines:**

- **GSAP-NER** (arXiv:2311.09860, Otto et al.) — published baseline for ML scholarly entity recognition (model + dataset entity types). Substrate for F1 Class A.

**Citation-failure-mode quantification (sanity check, not a derived prior):**

- **The 17% Gap** (arXiv:2601.17431, İlter) — forensic audit of 5,514 citations in 50 AI-assisted survey papers reports 5.1% Ghost + 78.5% Syntax Error + 16.4% Broken Link. Cited as a sanity-check on order of magnitude in a related population.

**Discipline and motivation:**

- **Hidden Pitfalls** (arXiv:2509.08713, Luo, Kasirzadeh & Shah) — four AI-Scientist-system failure modes.
- **Scout-4 §5 #3** — *"All AI-scientist systems I read first generate citations, then optionally verify… I did not find a paper that implements this pre-flight design."* (Note: scout-4 read pre-ARIS literature; ARIS is May 2026; the smith has now manually added ARIS to the population.)
- **CLAUDE.md rule #4** — *"Citations resolve or do not exist."*

## 2. Hypothesis statement

**If** MegaResearcher inserts a leaf `citation-verifier` worker between the synthesist and the synthesist's final emit, where the worker (a) extracts every cited arxiv ID and every cited DOI from the candidate output via deterministic regex, (b) fires `hf_papers paper_details` against each arxiv ID and Crossref against each DOI, (c) emits a `citations-resolution.yaml` artifact, and (d) **drops** any citation that does not resolve from the output before downstream emit — **then** on **PaperWrite-Bench (PaperRecon's n=51 papers, primary substrate)** plus a secondary held-out sample drawn from AI Scientist v1, v2, and AgentRxiv outputs (10 each, secondary substrate), the MegaResearcher-with-gate pipeline achieves **H_treatment = 0% hallucinated-citation rate** (deterministic: does not resolve via `hf_papers paper_details` AND no Crossref DOI hit), versus a measured baseline rate **H_baseline ≥ 5%** on total-unresolvable-phantom (the F2 floor; PaperWrite-Bench's reported >10 hallucinations/paper for ClaudeCode is expected to clear this floor by a wide margin), **AND** the gate causes the synthesist to drop **≤ 15%** of substantive claims as measured by the deterministic three-class taxonomy in §5 (named entity per GSAP-NER, numerical claim per the expanded regex, cited comparison), relative to a no-gate ablation on the same prompts.

The "AND" clause is critical: the hypothesis succeeds only if **both** the hallucination-elimination claim AND the bounded-quality-cost claim hold.

H_baseline is a **measurement, not a derived prior.** The PaperWrite-Bench published baseline is cited only as a sanity-check that the F2 floor is clearable.

## 3. Mechanism

This is a **gating function**, not a revision loop. Three components, each grounded in cited prior art:

**Component A — Deterministic resolution as a binary gate.** Every arxiv ID has a deterministic, idempotent resolution test: does `hf_papers paper_details` return a paper? Crossref provides the same for DOIs. These are binary, non-LLM-judge signals — exactly the external-signal requirement Huang et al. (arXiv:2310.01798) show is necessary for self-correction to be non-inert. ARIS (arXiv:2605.03042) §3.2 uses LLM-judge reviewers for the same existence axis; the proposed gate substitutes a deterministic resolver, which admits no false-positives from judge hallucination.

**Component B — The gate is constructive at emit time, autonomously blocking.** By DROPPING an unresolvable citation before downstream emit, the gate makes the failure mode (hallucinated citation in the output) **impossible by construction at emit time:** H_treatment = 0 as a structural property. The architectural commitment is autonomous-block, not human-approval-of-recommendations. ARIS's citation-audit (§3.2) explicitly emits KEEP/FIX/REPLACE/REMOVE recommendations "for human approval before submission" — autonomous block is not the ARIS pattern. Stage 1 of ARIS's audit cascade (§3.1) is explicit: "The audit is advisory at the workflow level: it does not halt execution." The block-on-unresolvable commitment is therefore the architectural delta against the strongest published peer.

**Positioning relative to retrieve-then-write:** Citegeist (arXiv:2503.23229) and ScholarCopilot (arXiv:2504.00824) achieve a stronger property: a citation cannot even be sampled unless it exists in the retrieved corpus. Post-emit autonomous-gate is constructive-after-the-fact. The system-integration contribution is that retrieve-then-write cannot be applied to AI-Scientist-family systems without rewriting the writer (a stateful multi-stage generator), whereas the post-emit gate is deployable inside a stateless-leaf architecture.

**Component C — Cost-of-enforcement is bounded by citation-level redundancy alone.** Multi-citation paragraphs (which Citegeist's per-paragraph RAG-grounding pattern produces at scale) survive the loss of one of N citations. **Important: rewrite-to-prose is NOT counted as survival.** Per the red-team's CR5-Important objection: if the synthesist rewrites "X achieves 73.2% on Y" as "X performs competitively on Y," the numerical-value-bearing Class B claim has been stripped and IS counted as a drop under F1's matching rule. Component C's predicted-D-bound rests on citation-level redundancy only; the rewrite path does not contribute to bounding D. The D ≤ 15% prediction is therefore weaker than in revision-1: if protocol-step-1 measures citation density < 2 per paragraph across the primary substrate, the redundancy argument collapses and D ≤ 15% reduces to engineering judgment alone (pre-registered conditional, §4).

**Why this is NOT a revision loop:** The gate fires once, deterministically, after the writer has emitted. There is no back-and-forth with the writer, no iterative self-critique, no LLM-judge involvement. Feedback Friction (arXiv:2506.11930) — the documented failure of LLMs to incorporate critic feedback — does not apply: the gate does not depend on the writer accepting feedback; it just removes the offending citations.

**Speculative element (explicitly flagged):** The 15% claim-drop threshold is the weakest empirical leg. ARIS's citation-audit, Citegeist, and ScholarCopilot do not report a substantive-claim drop rate against an ungrounded baseline. The redundancy argument is well-grounded conditional on citation density ≥ 2 per paragraph; if that condition fails, D ≤ 15% becomes engineering judgment without mechanism support, and F1 falsifies the hypothesis at D > 15% regardless.

## 4. Predicted outcome with magnitude

**Pre-flight measurement (protocol step 1, must run before the gate eval):**

Before the gate eval, measure across the primary (PaperWrite-Bench n=51) and secondary (n=30) substrates: (a) citation-source distribution (arxiv-resolvable, DOI-resolvable, off-substrate); (b) citation density per paragraph; (c) the off-corpus rate of the F4 retrieve-then-write comparator. These are reported with the manuscript-level results.

**Predicted distribution (engineering judgment, not derived prior):** ≥ 80% of cited identifiers are arxiv- or DOI-resolvable on PaperWrite-Bench (top-tier venues post-2025; most have arxiv mirrors). The actual measurement IS the finding.

**Pre-registered conditional (CR5 fix):** If citation density < 2 per paragraph across the primary substrate, the redundancy mechanism behind D ≤ 15% collapses and the bound reduces to engineering judgment alone. This is acknowledged in the pre-registration; the falsifier still fires at D > 15% but the explanatory narrative changes.

**Metric 1 — Hallucinated-citation rate H** (deterministic, binary, non-judge):

- **H_baseline (measurement, not derived prior):** Actual measured rate on PaperWrite-Bench. F2 floor: ≥ 5% total-unresolvable-phantom. PaperWrite-Bench's published baseline is ">10 hallucinations per paper on average" for ClaudeCode (PaperRecon §Experiments per arXiv:2604.01128 abstract), which translates well above the 5% floor on a per-citation basis for typical 30-50 citation manuscripts. F2 is expected to clear cleanly.
- **H_treatment:** 0% by construction. Structural property of the pipeline.

**Conditions under which H = 0 holds:**

- `hf_papers paper_details` + Crossref substrate online. If substrate is down, the gate defaults to *refuse* not *pass*.
- The citation-extraction regex correctly identifies all arxiv-ID-shaped and DOI-shaped tokens. Eval must measure extraction completeness against a hand-labeled subset (5 manuscripts).

**Conditions under which H = 0 would NOT hold (gate-integrity failures):**

- Resolver false-positives (extremely unlikely; arxiv IDs are deterministic).
- Citations to non-arxiv, non-DOI sources. Gate is **scoped** to arxiv-ID and DOI citations; off-substrate citations pass through ungated. Pre-flight distribution measurement bounds this gap.

**Metric 2 — Substantive-claim drop rate D** (the falsification surface):

- **Prediction:** D ≤ 15% under MegaResearcher-with-gate, where "substantive claim" is the deterministic three-class taxonomy in §5.
- **Reasoning:** Citation-level redundancy in multi-citation paragraphs alone (rewrite-to-prose is now counted as a drop). 15% is engineering judgment; the falsifier rejects at this threshold regardless of mechanism.
- **Acknowledgment of weakness:** If D ∈ [15%, 30%], that is a meaningful regression — the hypothesis formally fails but the synthesist can downgrade magnitude rather than killing the contribution.

**Conditions under which D ≤ 15% holds:**

- Citation density per paragraph ≥ 2 on average across the primary substrate.
- The deterministic three-class taxonomy (per GSAP-NER + expanded regex + cited-comparison detection) is implemented correctly.

**Conditions under which D ≤ 15% would NOT hold:**

- Manuscripts heavily reliant on single-citation key-claim sentences.
- Manuscripts where the writer cites the same fabricated ID across many sentences.
- Citation density < 2 per paragraph (the redundancy mechanism collapses).

## 5. Falsification criteria

The hypothesis is falsified if ANY of the following observed outcomes hold. Six criteria.

**F1 — Cost-of-enforcement exceeds the bound (D > 15%, deterministic taxonomy).** Three classes, no LLM-judge surface:

- **Class A — Named entity claim.** Detection grounded in **GSAP-NER (arXiv:2311.09860)** rather than raw spaCy NER. GSAP-NER's published pipeline provides ML-model and ML-dataset entity types as primary substrate. A Class A claim is a sentence containing a GSAP-NER-recognized ML entity bound to a comparison or property (verb-based detection via dependency parse on the surrounding sentence).
- **Class B — Numerical claim.** Expanded regex: `(?:BLEU|ROUGE-?[12L]?|F1|F-1|PPL|perplexity|loss|accuracy|MSE|MAE|latency|FLOPs|FLOP|parameters)\s*[:=]?\s*\d+\.?\d*` OR `(\d+\.?\d*\s*(%|pp|points|×|x|MB|GB))` OR `(\d+\.?\d*\s*[kKmM]?\s*(samples|examples|tokens))`. Binary; deterministic. Dependency parse verifies the number modifies a named entity from Class A.
- **Class C — Cited comparison.** A sentence containing an inline citation token (`[Author, Year]`, `(Author et al., 2024)`, `\cite{key}`, or arXiv-ID-shaped string) and a comparison verb (outperforms / improves / compared to / vs / against / matches / exceeds / lower than / higher than / better than). Binary.

**Excluded by design:** narrative sentences (motivational framing, discussion, future work, paper-organization). F1 measures the gate's cost on discipline-relevant substantive content.

**Drop definition (CR5 fix — explicit):** A substantive claim (Class A/B/C) survives only if the matching deterministic rule fires on a sentence in the with-gate output:
- Class A survives if the same GSAP-NER entity bound to the same property appears.
- Class B survives if the same named-metric entity + same numerical value (within ±0.5% tolerance) appears.
- Class C survives if a citation to the same anchor paper appears in a comparison sentence.
- **Rewrite-to-prose where the numerical value is stripped IS counted as a drop.** All matching is deterministic — string match + regex — no LLM-judge.

**Extraction-completeness audit (pre-registered eval-protocol check):** On a 5-manuscript hand-labeled subset drawn from PaperWrite-Bench, measure the precision and recall of the deterministic Class A/B/C detectors against human annotation. If recall < 80% on any class, the F1 measurement is biased and the threshold must be widened accordingly before the main eval runs.

If on the primary substrate D > 15% (substantive claims dropped by the gate, as defined above), the hypothesis FAILS even if H_treatment = 0.

**F2 — H_baseline below the gap-detection threshold (two-scale).** Two floors:

- **Total-unresolvable-phantom floor:** if H_baseline < 5% on the primary substrate, the gap is empirically too small for the differential (5% → 0%) to be publishable. The hypothesis fails. (Expected to clear cleanly given PaperWrite-Bench's published baseline.)
- **Pure-hallucination floor:** if the pure-Ghost rate (no resolver hit at all + no fuzzy match in arxiv corpus) is < 1%, the gate is catching mostly syntax errors and dead links, not hallucinations — the contribution narrative must be reframed. This is a finding to report, not a hypothesis kill.

If the 5% total-phantom floor is not crossed, F2 fires.

**F3 — H_treatment > 0 (gate-integrity failure).** If the gate is implemented but H_treatment > 0%, the gate is broken — either (a) the extractor is missing citation tokens, (b) the resolver is misreporting, or (c) the gate is letting unresolvable citations pass. Binary, deterministic; one example of an unresolvable citation in the emitted output is sufficient falsification.

**F4 — Retrieve-then-write subsumption test (the steelman falsifier, NOW DETERMINISTIC).** Two ablations on the same primary substrate prompts:

- **Ablation R (retrieve-then-write):** MegaResearcher synthesist replaced with a Citegeist-style retrieve-then-write generator using Citegeist's open-source code. **Off-corpus rate** is measured deterministically: extract every arxiv-ID-shaped string from Ablation R's output via the gate's identical regex; fire `hf_papers paper_details` against each; classify each as (i) resolved AND in Citegeist's retrieval-set for that prompt (Citegeist's retrieval-set is auditable via its open-source code, returning the abstract-conditioned retrieved arxiv-ID list), (ii) resolved but NOT in retrieval-set (off-corpus-but-existing), or (iii) unresolved. Off-corpus rate = (ii) + (iii) divided by total cited arxiv-ID-shaped strings.
- **Ablation G (gate-only, this hypothesis):** MegaResearcher-with-gate. Measure H_treatment.

**Pre-registered thresholds:** if Ablation R achieves H ≈ 0 AND off-corpus rate < 2% on the primary substrate, retrieve-then-write subsumes the gate and the hypothesis's contribution collapses to "post-emit gate as a defensive cross-check." If off-corpus rate ≥ 5%, the gate catches a failure mode retrieve-then-write leaves uncaught, and the contribution is preserved. The 2–5% band is reported as ambiguous. **All measurement uses `hf_papers paper_details` resolution — no LLM-judge surface anywhere in F4 (CR2 fix).**

**F5 — CiteME attribution-rate regression.** Use CiteME (arXiv:2407.12861) as a held-out attribution benchmark. CiteGuard achieves 68.1% on CiteME with DeepSeek-R1 (vs human 69.2%). If MegaResearcher-with-gate's attribution rate on CiteME is > 2 percentage points lower than MegaResearcher-without-gate on the same task, the gate is degrading other citation capabilities. Binary, deterministic, published-benchmark scoring.

**F6 — Synthesist failure-to-emit rate.** If on > 10% of the primary substrate manuscripts the synthesist fails to produce a valid output because the gate's enforcement removed too much content, the gate is too strict to deploy. F6 is the deployability falsifier; different failure mode from F1.

Six falsification criteria cover: cost-of-enforcement (F1), gap-existence (F2), gate-integrity (F3), retrieve-then-write subsumption (F4), cross-substrate generalization (F5), deployability (F6). All are deterministic; none admit an LLM-judge surface.

## 6. Required experiments (sketch only — eval-designer details these)

**Dataset substrate (updated for CR3):**

- **Primary:** **PaperWrite-Bench (arXiv:2604.01128, n=51 papers from top-tier venues post-2025).** Github: `Agent4Science-UTokyo/PaperRecon` (18 stars), MIT license. PaperRecon's PaperWrite protocol generates the AI-written papers; the gate eval consumes these.
- **Secondary:** 30-manuscript AI-Scientist-family-specific sample. Sources: AI Scientist v1 outputs (10), AI Scientist v2 outputs (10), AgentRxiv outputs (10). Repos: `sakanaai/ai-scientist`, `SakanaAI/AI-Scientist-v2`, `SamuelSchmidgall/AgentLaboratory`. Used for the AI-Scientist-family-specific narrow gap claim.
- **Tertiary (F5):** CiteME (arXiv:2407.12861, `bethgelab/CiteME`, MIT).
- **Quaternary (F4):** Citegeist (arXiv:2503.23229) open-source code as the retrieve-then-write comparator.

**Baselines:**

- **B1 (PaperWrite-Bench differential):** PaperWrite output unmodified. Measure H_baseline directly. Compare against PaperRecon's published Hallucination dimension.
- **B2 (gate ablation):** MegaResearcher without the citation-verifier gate. Isolates the gate's contribution.
- **B3 (ARIS-style advisory):** Post-hoc citation-check that flags but does not drop (LLM-judge + human-approval simulation). Isolates blocking-vs-advisory.
- **B4 (retrieve-then-write, F4 critical):** Citegeist on the same prompts.

**Ablations:**

- **A1 — Gate scope:** arxiv-only vs arxiv+DOI/Crossref.
- **A2 — Synthesist rewrite:** rewrite-permitted vs rewrite-forbidden after gate-removals. Tests Component C's residual contribution after CR5 fix.
- **A3 — Strict vs lenient gate:** strict (drop) vs lenient (flag-and-let-pass). Tests F1 vs F6 trade-off and isolates the blocking-vs-advisory architectural delta against ARIS.

**Pre-flight measurement (protocol step 1):**

Before the gate eval runs, across both primary and secondary substrates: (a) citation-source distribution; (b) citation density per paragraph (gates Component C's predicted-D-bound); (c) Ablation R's off-corpus citation rate (gates F4); (d) extraction-completeness audit on a 5-manuscript hand-labeled subset (gates F1 measurement bias).

**Non-judge signals:**

- Citation-resolution rate: `hf_papers paper_details` + Crossref.
- CiteME attribution rate: published benchmark.
- Substantive-claim drop rate D: GSAP-NER + expanded regex + cited-comparison detection; no LLM-judge.

**Pre-registration discipline (per CLAUDE.md rule #3):** All thresholds — 5% F2 floor, 1% pure-Ghost sub-floor, 15% F1 ceiling, 2pp F5, 10% F6, 2%/5% F4 off-corpus thresholds — locked BEFORE the eval runs. The deterministic three-class taxonomy must be implemented as code before eval starts. The extraction-completeness audit must clear ≥ 80% recall on each class before the main F1 eval is trusted.

**Cost estimate (sanity check):** ~$0.60/replication × 51 PaperWrite-Bench + 30 secondary + ablations B1–B4 + A1–A3 + F5 CiteME ≈ under $300/replication.

## 7. Risks to the hypothesis

**R1 — The contribution is fundamentally an engineering-integration claim, not an architectural-novelty claim.** Red-team's primary steelman. Architectural pattern (constructive citation guarantee) is established by Citegeist / ScholarCopilot; ARIS provides an LLM-judge-based advisory existence-check. **What the hypothesis still contributes:** the three-axis architectural delta against ARIS (blocking vs advisory; deterministic resolver vs LLM-judge; existence-only vs bundled) plus the published-first D measurement on PaperWrite-Bench.

**R2 — Citation distribution is heavily off-substrate.** If primary-substrate citations are predominantly to non-arxiv, non-DOI sources, the gate's effective coverage is limited. **Mitigation:** protocol-step-1 pre-flight measurement bounds this from below. PaperWrite-Bench top-tier-venue post-2025 papers should be ≥ 80% arxiv-resolvable.

**R3 — Synthesist rewrite degrades coherence in unmeasured ways.** Component C's residual rewrite affordance (now NOT counted as survival) may still produce cross-section coherence regressions. Scout-2 §5 #3 documents this as a known gap. **What the hypothesis still contributes:** the rewrite-induced coherence regression is itself a measurable phenomenon — if observed, it motivates a coherence-pass worker as future work.

**R4 — The gate is trivially implementable.** ~200 lines, ~$0.60/replication. **What the hypothesis still contributes:** the empirical D ≤ 15% bound on PaperWrite-Bench with a deterministic non-judge taxonomy IS the published-novel piece. ARIS publishes no D measurement.

**R5 — Retrieve-then-write subsumes the gate.** F4's Ablation R may achieve H ≈ 0 and off-corpus rate < 2%. **What the hypothesis still contributes:** the F4 measurement IS itself the contribution — empirically establishing that retrieve-then-write subsumes post-emit gating is publishable; the synthesist should report S5 with downgraded magnitude under this outcome.

**R6 — ARIS subsumes the gate (new this revision).** If ARIS's `/citation-audit` is deployed in autonomous-block mode (treating REMOVE recommendations as automatic drops without human approval), the three-axis delta collapses to "deterministic resolver vs LLM-judge" alone. **What the hypothesis still contributes:** the deterministic-resolver-vs-LLM-judge axis is itself a meaningful axis — it's the difference between a resolver that admits no false-positives from judge hallucination and one that does. The cost-of-enforcement bound on PaperWrite-Bench remains novel regardless. **If the architectural delta collapses entirely** (i.e., ARIS in some downstream paper is shown to deploy citation-audit autonomously with a deterministic resolver), the hypothesis is correctly killed and the lesson is "the gap was filled between revisions."

**R7 — Citation density < 2 per paragraph collapses Component C.** If protocol-step-1 measures citation density < 2 across PaperWrite-Bench, the redundancy mechanism behind D ≤ 15% collapses (pre-registered conditional, §4). **What the hypothesis still contributes:** the measurement of citation density on PaperWrite-Bench is itself a finding; the falsifier still fires at D > 15% regardless of mechanism.

## 8. Sources

All arxiv IDs verified via `hf_papers paper_details` in the course of this revision.

**AI-Scientist-family systems (narrow gap target):**

- **arXiv:2408.06292** — The AI Scientist v1 (Lu et al.)
- **arXiv:2504.08066** — AI Scientist v2 (Yamada et al.)
- **arXiv:2503.18102** — AgentRxiv (Schmidgall & Moor)
- **arXiv:2511.04583** — Jr. AI Scientist (Miyai et al.)
- **arXiv:2605.03042** — ARIS (Yang, Li, Li, May 2026, 107 upvotes, 8890 stars) — **NEW THIS REVISION.** `/citation-audit` (§3.2) with existence + metadata + context axes; advisory KEEP/FIX/REPLACE/REMOVE; LLM-judge based; closest peer to the proposed gate. The three-axis architectural delta is the contribution.

**Retrieve-then-write systems:**

- **arXiv:2503.23229** — Citegeist (Beger & Henneking)
- **arXiv:2504.00824** — ScholarCopilot (Wang et al.)
- **arXiv:2511.17689** — ARISE (Wang et al.)
- **arXiv:2603.14629** — ResearchPilot (Zhang)

**Citation-verification systems (post-hoc):**

- **arXiv:2602.23452** — CiteAudit (Yuan et al.)
- **arXiv:2511.16198** — SemanticCite (Haan)
- **arXiv:2604.04074** — FactReview (Xu et al.)
- **arXiv:2407.12861** — CiteME (Press et al.)
- **arXiv:2510.17853** — CiteGuard (Choi et al.)
- **arXiv:2411.00294** — LLM-Ref (Fuad & Chen)
- **arXiv:2601.16993** — BibAgent (Li et al., agentic post-hoc miscitation detection, MisciteBench) — **NEW THIS REVISION.**

**Eval-substrate benchmarks:**

- **arXiv:2604.01128** — PaperRecon / PaperWrite-Bench (Miyai et al., 51-paper benchmark, Presentation + Hallucination dimensions) — **NEW THIS REVISION; PRIMARY EVAL SUBSTRATE.**

**Methodology baselines:**

- **arXiv:2311.09860** — GSAP-NER (Otto et al., ML scholarly entity recognition) — **NEW THIS REVISION; F1 CLASS A SUBSTRATE.**

**Failure-mode quantification:**

- **arXiv:2601.17431** — The 17% Gap (İlter) — sanity-check on order of magnitude only

**Mechanism / discipline grounding:**

- **arXiv:2310.01798** — Huang et al. — external signal required for self-correction
- **arXiv:2506.11930** — Feedback Friction (Jiang et al.)
- **arXiv:2509.08713** — Hidden Pitfalls (Luo, Kasirzadeh & Shah)

**Swarm artifacts:**

- gap-finder-3 §(a) row S5 + §(b) S5
- gap-finder-1 §2 Rank 3
- scout-1 14-system capability matrix (does not include ARIS; the smith manually added ARIS in revision-2)
- scout-4 §5 #3
- scout-2 entry [19] CiteME + §5 #3
- scout-3 §line 124 item 6
- red-team-S5 rounds 1 and 2 — drove the system-integration repositioning, the deterministic taxonomy commitment, the F4 subsumption falsifier, the ARIS engagement, the F4 deterministic specification, the PaperWrite-Bench substrate migration, the GSAP-NER grounding, and the §3 / §5 rewrite-to-prose resolution
