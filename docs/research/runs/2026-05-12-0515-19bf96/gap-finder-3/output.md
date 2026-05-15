# gap-finder-3 — Feasibility-filtered shortlist

## 0. Slice scope and method

Sources read in full:

- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/gap-finder-1/output.md` — 8 ranked capability gaps + 5 contradictions + 4 discarded candidates
- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/gap-finder-2/output.md` — 12 architectural-pattern gaps (GAP-A1 through GAP-A12) + 5 KILL-rationales + 6 unresolved-literature notes
- `/Users/ggix/MegaResearcher/docs/research/specs/2026-05-12-megaresearcher-paper-pipeline-spec.md` — for constraint context

Spot-checked citations via `hf_papers paper_details`: 2502.08788, 2508.17536, 2407.19594, 2303.11366, 2507.08038, 2506.11930, 2310.01798, 2503.18102 — all resolve.

The four feasibility constraints applied as filters (each yes/no):

1. **C1 — Leaf-worker only, no nested dispatch.** Can the augmentation be expressed as one or more leaf workers fired by the main-session orchestrator? Patterns requiring an agent that itself spawns sub-agents fail.
2. **C2 — File-based artifact passing.** State that survives a worker invocation must be written to a file the next worker can read. Patterns requiring shared in-memory state, live IPC, or streaming between workers fail.
3. **C3 — Off-the-shelf model APIs.** No fine-tuning, no custom training, no preference-pair generation that requires DPO/RLHF on a local checkpoint. Inference-time prompting and routing only.
4. **C4 — ≤$200 per replication of the eval design.** A replication = the eval-designer's protocol re-run once. Hypotheses that intrinsically require >$200 must be flagged for user approval (per spec). Heuristic budgets used: a 20-paper held-out sample × ~3 model calls per paper × ~$2/call = ~$120 → leaves headroom for sweeps; LLM-judge panels add ~$0.50/manuscript-judge call.
5. **C5 — Testable without venue submission.** Falsification must use proxy measures + ≥1 non-judge signal. Scout-3 names untrustworthy signals (same-family LLM judge alone fails). Non-judge signals available: AblationBench solve rate (arXiv:2507.08038), citation-resolution rate via `hf_papers paper_details`, SPECS controlled-flaw injection (arXiv:2604.13940), CiteME attribution rate (arXiv:2407.12861), multi-seed CI presence (binary).

I rank shortlist entries by **expected impact ÷ implementation cost** where:

- impact magnitude comes from named-magnitude generic-task ceilings (Zhang et al. 2502.08788 heterogeneous-model survival, Choi et al. 2508.17536 voting deltas, Meta-Rewarding 2407.19594 AlpacaEval +16.5 LC pts, Reflexion 2303.11366 +22% AlfWorld) **discounted** by the negative-result ceilings (intrinsic self-correction 2310.01798, Feedback Friction 2506.11930)
- cost is measured in new workers + new artifact formats + expected orchestrator-loop additions

The shortlist below has 6 entries — within the spec's 5–8 target, with margin for red-team kills toward the spec's ≥3-surviving-hypothesis floor.

---

## (a) Shortlist table

| # | Source ID | Gap name | Expected impact | Implementation cost | Impact/cost rank |
|---|---|---|---|---|---|
| S1 | GAP-A1 (GF-2) + Contradiction 5 (GF-1) | Heterogeneous-model writer / red-team split | HIGH — the *only* MAD configuration surviving Zhang et al. 2502.08788 meta-eval; directly breaks the same-family-reviewer reward-hacking documented in AgentRxiv §4.1 (2503.18102) and Agent Lab §limitations (2501.04227); orthogonal to the Choi voting axis and partially bypasses Feedback Friction (different model = different intuitions). | 0 new workers (route existing workers to different model providers). 1 artifact addition (`provider` field on worker output). 0 new orchestrator loops. | **1 (highest)** |
| S2 | GAP-A10 (GF-2) | Meta-Rewarding length-control wrapper on red-team / synthesist judge calls | HIGH — directly patches the documented reward-hacking failure mode in two AI-Scientist systems (AgentRxiv §4.1, Agent Lab §limitations); Meta-Rewarding (2407.19594) reports 22.9 → 39.4 AlpacaEval LC win rate; Length-Controlled AlpacaEval (2404.04475) confirms the regression debiaser is necessary for valid auto-eval. Bypasses KILL-3 directly. | 0 new workers (wrap existing judge prompts with length-debiasing post-processor). 1 artifact addition (length-normalized score field). 0 new loops. | **2** |
| S3 | GAP-A2 (GF-2) + Rank 4 (GF-1) ICLR-rubric self-eval | Majority voting over N independent draft candidates on *structured decisions* (baseline list, ablation list, claim/no-claim, related-work cluster choice) | MEDIUM-HIGH — Choi et al. 2508.17536 voting > centralized debate on 7/7 benchmarks; on Arithmetic 0.81 → 0.99. Choi flag is voting needs plurality — restricting to structured decisions inside the manuscript gives that plurality. Also addresses ICLR-rubric self-eval gap (GF-1 Rank 4) since structured-rubric line items are exactly the plurality-bearing decisions. | 1 new worker role (structured-decision-voter) — but expressible as a hypothesis-smith / eval-designer prompt variant, not a full new agent. 1 artifact addition (decision-ballot JSON). 0 new loops (parallel fan-out → aggregate). | **3** |
| S4 | GAP-A5 (GF-2) + GAP-A12 (GF-2) + Rank 1-tie audit trail (GF-1) | Reflexion-style verbal reflection on **structured-decision rejections** keyed to a binary signal, written to a rejected-hypothesis ledger | MEDIUM-HIGH — Reflexion (2303.11366) +22% AlfWorld / +20% HotpotQA / +11% HumanEval require the binary external signal. The structured-decision sub-problem in paper-gen *has* a binary signal (citation resolves yes/no, ablation present yes/no, baseline-in-comparison yes/no). The free-prose sub-problem does not — explicitly limited to the structured surface. Closes GF-1 audit-trail gap (Rank 1-tie) and GF-2 pre-registration gap (GAP-A12). | 1 new artifact format (rejected-hypothesis ledger schema: `{hypothesis_id, killed_at_wave, binary_signal_that_killed_it, verbal_reflection, lesson}`). 0 new workers — the red-team already runs; the ledger is its write-side. 0 new loops. | **4** |
| S5 | Rank 3 (GF-1) citation pre-flight gate | Citation pre-flight verification worker — every claimed citation runs through `hf_papers paper_details` before any downstream worker reads the manuscript | MEDIUM — CiteAudit and 17% Gap (cited in GF-1 Rank 3) document the failure mode. Scout-4 §5 #3 names no system implements a pre-flight gate. Magnitude is bounded: this is a hallucination-prevention fix, not a quality-lift fix; reviewers rarely reject for citation hallucination alone. But the gap is genuinely empty in the published literature and is a *named MegaResearcher discipline rule* (rule #4), so the differentiator value is high even if the quality-lift magnitude is moderate. | 1 new leaf worker (citation-verifier) firing between hypothesis-smith and synthesist. 1 artifact format (citation-resolution table). 0 new loops (pre-flight gate, not iterative). | **5** |
| S6 | Rank 1-tie ablation discipline (GF-1) + AblationBench (2507.08038) | Ablation-coverage checklist worker firing after eval-designer, scored against AblationBench-style coverage | MEDIUM — AblationBench (2507.08038) exists *specifically* because the field is weak at this; no scout-1 system reports solving it. Direct route to scout-3 §5 #2 workshop→main-track delta since reviewers flag insufficient ablations as a recurring main-track rejection cause. Magnitude bounded by Feedback Friction (2506.11930) when the writer cannot incorporate the missing-ablation flag, but bypassable via the binary surface (ablation X is present yes/no — checkable). | 1 new leaf worker (ablation-coverage-checker) reading eval-designer's protocol and writing a coverage report. 1 artifact format (coverage matrix). 0 new loops. | **6** |

Total shortlist count: 6 entries. All 6 satisfy C1–C5 explicitly (see §(b) per-entry justification).

---

## (b) Per-shortlist-entry justification

### S1 — Heterogeneous-model writer / red-team split

**C1 (no nested dispatch):** Passes. Routing the existing writer worker to (e.g.) Anthropic Claude and the existing red-team worker to (e.g.) OpenAI GPT or Google Gemini is a top-level orchestrator decision. No nested dispatch — the orchestrator simply specifies the provider when it invokes each leaf worker.

**C2 (file-based artifacts):** Passes. The output artifact format gains a single new field — `provider` — and the orchestrator reads it when scheduling the cross-model check. No new IPC required; all worker-to-worker handoff stays through `output.md` / `manifest.yaml` writes.

**C3 (off-the-shelf APIs):** Passes trivially — the entire mechanism IS off-the-shelf API routing.

**C4 (≤$200/replication):** Passes. Per-paper cost: writer call (~$2) + red-team call (~$2) + synthesist (~$2) × 20-paper held-out sample = ~$120. Heterogeneous-model variant doubles the *cost-per-provider* not the call-count; same budget. Even with 3 provider combinations swept in an ablation: ~$60 × 3 = $180.

**C5 (testable without venue submission):** Passes — and is the most defensible falsification surface on the shortlist. Test design: SPECS-Review-Benchmark (2604.13940) controlled-flaw injection on a 20-paper held-out sample with same-family red-team vs heterogeneous-model red-team; non-judge signal = flaw-detection rate (binary per injected flaw, not a same-family judge score).

**Impact estimate grounding:** Zhang et al. 2502.08788 systematically eval 5 MAD methods × 9 benchmarks × 4 models and find heterogeneous-model is the only configuration to *survive* matched-compute CoT-self-consistency baselines. Liang et al. 2305.19118 shows GPT-3.5 + MAD with a different judge surpasses GPT-4 alone on Common MT. AgentRxiv §4.1 (2503.18102) and Agent Lab §limitations (2501.04227) directly document the same-family-reviewer reward-hacking pattern this would address. Feedback Friction (2506.11930) is the relevant ceiling, but as GF-2 notes the friction is *intra-model* intent-to-update — different-model critique provides external pressure that intrinsic self-correction (2310.01798) cannot. ARIS (2605.03042) advertises the pattern but is not in the AI-Scientist family.

### S2 — Meta-Rewarding length-control wrapper on judge calls

**C1:** Passes. The length-control wrapper is a deterministic post-processor on the red-team / synthesist judge's score output — runs in the orchestrator session, no nested dispatch.

**C2:** Passes. Adds one field (`length_normalized_score`) to red-team output `manifest.yaml`; downstream workers read the new field.

**C3:** Passes. The wrapper is the regression debiaser from Length-Controlled AlpacaEval (2404.04475) — a closed-form adjustment given a length distribution, no fine-tuning.

**C4:** Passes. Adds zero new API calls — it reweights the existing red-team score by length. Cost delta: ~$0.

**C5:** Passes. Falsification surface: inject deliberately-verbose paraphrases of a fixed-quality manuscript and verify the length-normalized score is stable while the raw score is not. Non-judge signal: token-count regression coefficient (statistical, not LLM-judge).

**Impact estimate grounding:** Meta-Rewarding (2407.19594) — Llama-3-8B-Instruct AlpacaEval 2 LC win rate 22.9 → 39.4 over 4 iterations *with length-control*; without length-control the model exploits verbosity to game the judge. Self-Rewarding LMs (2401.10020) tokens grow 1092 → 2552 across 3 iterations without proportional quality gain. AgentRxiv §4.1 documents the exact failure mode in AI-Scientist family on paper-quality reward. KILL-3 in GF-2 makes this a *precondition* for any other LLM-as-judge loop — meaning if this gets killed downstream, several other shortlist entries (S3, S4) need additional defense.

### S3 — Majority voting on structured decisions

**C1:** Passes. Fan-out N independent samples from the hypothesis-smith / eval-designer per structured decision (e.g., "which 3 baselines to compare against"); the orchestrator aggregates votes. The voter is a single deterministic aggregation step the orchestrator runs in-session, not a nested dispatch.

**C2:** Passes. Each candidate writes its own structured-decision ballot to a JSON artifact; the aggregator reads them all. Plurality is a deterministic file-fold.

**C3:** Passes. Voting is sampling from the same API at temperature > 0; no training.

**C4:** Passes. N=5 candidates × 20-paper held-out sample × ~$1/candidate-call = ~$100. Leaves headroom.

**C5:** Passes. Non-judge signal: agreement rate across N samples (a deterministic statistic, not a judge score). Falsification surface: held-out sample where the "true" baseline set is the one cited by an accepted version of the same paper — vote-quality is a hit-rate.

**Impact estimate grounding:** Choi et al. 2508.17536 — Qwen2.5-7B voting 0.7691 avg vs centralized MAD 0.6551 across 7 NLP benchmarks; on Arithmetic 0.81 → 0.99 single → voting. Choi flags voting requires plurality structure (the cap on transfer to free prose). Restricting the vote to *structured paper-decisions* (baseline list, ablation list, claim-vs-not-claim, related-work cluster) preserves plurality — these are the same decisions where ICLR-rubric self-evaluation (GF-1 Rank 4) is currently absent. Feedback Friction does not directly apply (voting bypasses revision-loop friction by skipping the friction surface entirely). KILL-1 (centralized debate worst configuration) is *bypassed* by this hypothesis — it replaces debate with voting.

### S4 — Rejected-hypothesis ledger keyed to structured binary signals

**C1:** Passes. The ledger is a write-only artifact format. The red-team worker already produces a kill/revise/accept verdict; the new schema requires it to attach a verbal reflection and the binary signal that triggered the kill. No new worker; no nested dispatch.

**C2:** Passes. The ledger IS the file-based artifact. Reflexion's "episodic memory" maps onto a versioned JSON file (`rejected-hypotheses.jsonl`) accessible to all later waves.

**C3:** Passes. Reflection is generation, not training.

**C4:** Passes. One extra red-team output per killed hypothesis — at most ~10 extra calls per replication × $1 = ~$10.

**C5:** Passes. Non-judge signal: rejected-hypothesis recoverability rate — given a known-flawed manuscript-claim, does the ledger surface the correct lesson? Measurable as a binary against a hand-labeled lesson key. AblationBench (2507.08038) AuthorAblation task is one viable substrate.

**Impact estimate grounding:** Reflexion (2303.11366) reports +22% AlfWorld / +20% HotpotQA / +11% HumanEval **only when the external binary signal is present** — intrinsic self-reflection without the signal is inert (Huang et al. 2310.01798). The structured-decision sub-problem in paper-gen has natural binary signals (citation-resolves, ablation-present, baseline-in-table); the free-prose path does not, and this hypothesis is *explicitly scoped* to the structured surface to avoid the 2310.01798 trap. Closes GF-1 Rank 1-tie audit-trail gap (only EvoScientist 2603.08127 has any analog, and it tracks code-execution failures not prose-decision rejections) and GF-2 GAP-A12 pre-registration gap. The MegaResearcher discipline rule #1 ("rejected/killed hypothesis appears in the synthesist's final document") becomes the externally-measurable behavior, not just an aspiration.

### S5 — Citation pre-flight verification worker

**C1:** Passes. A single new leaf worker that takes the hypothesis-smith / synthesist's draft, extracts citations, fires `hf_papers paper_details` per citation, and writes a resolution table. Pure leaf; no nested dispatch.

**C2:** Passes. Output is `citations-resolution.yaml` — every downstream worker reads it.

**C3:** Passes. `hf_papers paper_details` is the off-the-shelf substrate.

**C4:** Passes. ~30 citations/manuscript × 20 manuscripts × ~$0.001/lookup ≈ $0.60. Negligible.

**C5:** Passes. Non-judge signal: citation-resolution rate (binary, deterministic). CiteME (2407.12861) is a direct substrate for the eval.

**Impact estimate grounding:** Scout-4 §5 #3 names this as empty in the literature; no scout-1 system implements pre-flight resolution. AI Scientist v1 uses Semantic Scholar lookup as a *novelty* check, not a resolution gate; CiteAudit (cited in GF-1 Rank 3 as 2602.23452) reports hallucinated citations already in accepted ML papers. Magnitude estimate is bounded: this is a hallucination-prevention fix, not a quality lift — reviewers rarely reject *only* for citation hallucination, so the realized impact on main-track acceptance is moderate. The differentiator value is high because the gap is empty and the implementation is cheap; the absolute quality lift is moderate. Feedback Friction does not apply (this is a pre-flight gate, not a revision loop). I rank this 5 rather than 3 to reflect the moderate quality-lift even though the implementation cost is among the lowest.

### S6 — Ablation-coverage checklist worker

**C1:** Passes. Reads the eval-designer's protocol output, applies a coverage rubric drawn from AblationBench's AuthorAblation/ReviewerAblation task definitions, writes a coverage matrix. Leaf worker, single pass.

**C2:** Passes. Output is `ablation-coverage.yaml`; the synthesist reads it.

**C3:** Passes. The rubric is a prompt; AblationBench is benchmark data, not a model.

**C4:** Passes. ~3 calls/manuscript × 20 manuscripts × $2 = $120.

**C5:** Passes. AblationBench solve rate IS the non-judge signal — it is published with a binary scoring rubric and a held-out task set. Direct falsification: does the worker's coverage matrix lift the AblationBench score above the no-worker baseline by a pre-registered threshold?

**Impact estimate grounding:** AblationBench (2507.08038) exists *because* current AI co-scientist agents are weak at proposing and identifying ablations; the benchmark is the existence-proof of the gap (Abramovich & Chechik). Scout-3 §5 #2 names workshop→main-track delta; insufficient-ablations is one of the most-cited main-track rejection causes. Magnitude is partially capped by Feedback Friction (2506.11930) if the eval-designer cannot then add the missing ablations — but a checklist that names a missing ablation by ID converts the friction surface from "interpret feedback" (high friction) to "add this exact ablation" (low friction). KILL-1 doesn't apply (not a debate loop).

---

## (c) Failure roster — gaps that did NOT pass feasibility

Each entry names the specific constraint(s) violated and the one-line reason.

### GF-1 Rank 7 — Theoretical Reasoning (proofs, derivations)
**Violates C4 and partially C3.** Lean-based theorem proving (Prover Agent 2506.19923, Bolzano 2604.16989) requires either GPU-backed proof search (out of budget) or model-side fine-tuning on a corpus of derivations. A pure-prompting theorem prover is not at the SOTA published in this family. Future-work flag — re-evaluate when off-the-shelf proof-assistant APIs mature.

### GF-1 Rank 8 — Long-Horizon Coherence (persistent state)
**Violates C2.** Every published fix (EvoScientist's in-memory modules, freephdlabor's compaction-on-context, Idea2Paper's precomputed offline graph, ML-Master 2.0's hierarchical cognitive caching) requires either persistent in-agent state or a precomputed offline asset. The MegaResearcher file-handoff substrate is *exactly* the unmeasured substitute scout-4 flagged (§5 #4 in GF-1 §4) — making this a measurement gap rather than a hypothesis-smith target. Forward as a future-work flag for eval-designer to *measure* whether file-handoff is sufficient, not as a hypothesis to add an architectural fix.

### GF-2 GAP-A3 — Tree-of-thought search over revision states
**Violates C4 and C5.** ToT b=5 over revision states is N=5 branches per revision per section per manuscript × 20 manuscripts — call budget ~5 × 6 sections × 20 = 600 extra calls × $2 = $1,200 per replication. Even if scoped to a single section type, the partial-state-evaluator constraint (KILL-5 in GF-2) means C5 cannot be cleanly satisfied without a binary checker the field has not built. Future-work flag once a structured partial-state evaluator exists.

### GF-2 GAP-A4 — Constitutional / principle-guided critique
**Violates C5 (weakly).** The pattern is implementable as a leaf worker with a hand-authored principle list, and the cost is low (passes C1–C4). The C5 issue is that constitutional critique's empirical magnitude evidence (Bai et al. 2212.08073) is behavioral (harmlessness/helpfulness) — there is no published non-judge signal for *paper-quality* constitutional adherence. The falsification surface would have to be invented (which principles count, which manuscripts have them). Demoted to second-tier; could be a synthesist follow-on if a non-judge signal is later found.

### GF-2 GAP-A6 — Git Context Controller versioned context
**Violates C4 partially and C5.** The pattern is naturally compatible with stateless dispatch (passes C1–C3). C4 issue: Wu (2508.00031) reports qualitative gains on long-horizon coding only; transferring the COMMIT/BRANCH semantics to paper-gen requires an audit-trail-completeness eval that no published paper has built (GF-2 §(d) item 3). C5 issue: no non-judge signal exists — "audit trail is good" is judge-graded. The rejected-hypothesis ledger in S4 captures the core MegaResearcher discipline rule #1 value without the full GCC semantics.

### GF-2 GAP-A7 — A-MEM auto-linking + memory evolution
**Violates C2 and C4.** A-MEM's "memory evolution" rewrites linked notes on every new write — incompatible with append-only file-handoff (C2). Even if expressed as a periodic consolidation worker, KILL-4 in GF-2 documents that the evolution step itself recreates the AI-Researcher abstraction-drift failure under a new name. Budget: a memory-evolution pass over the run's accumulated state every wave is ~$50–100 *per wave* × 4–5 waves = approaches the $200 ceiling alone. Future-work flag.

### GF-2 GAP-A8 — AriGraph entity knowledge graph
**Violates C4 and C5.** Entity-resolution in research-paper space (is this baseline the same as that baseline; does this dataset match the cited version) is itself unsolved — the AriGraph failure modes (duplicate nodes, contradictory updates) hit the paper-gen target directly. Building the graph at run-time is cheap but querying-and-maintaining it across waves doesn't have a non-judge signal for fidelity. GF-2 transfer-plausibility was already LOW-MEDIUM.

### GF-2 GAP-A9 — Tree-of-Debate persona pattern
**Violates C4 and C2.** Persona-debate inside one related-work section requires N personas × M turns × manuscripts — likely ~$300+/replication. Worse, the persona state is conversational and benefits from streaming back-and-forth (C2 friction). Could be made to fit if collapsed to a single-turn-per-persona ballot, but at that point it converges to S3 (majority voting). Subsumed by S3.

### GF-2 GAP-A11 — MIRIX-typed memory taxonomy
**Violates C4 and C5.** GF-2 already scored MEDIUM-LOW. Six specialist memory agents × cross-store consistency management is expensive in tokens and lacks a published cross-store-consistency benchmark. No clean non-judge signal exists.

### GF-1 contradictions 1–5
Contradictions are *evidence about the literature*, not directly fileable as hypotheses. They feed S1 (cross-model split → addresses Contradiction 5) and inform the eval-designer's threats-to-validity section. They do not become shortlist entries themselves.

### GF-1 §4 unresolved-literature items 1–7
These are measurement gaps (cost-quality Pareto, unified failure taxonomy, cross-section coherence, stateless+memory interaction, tree-vs-wave-vs-linear architectures, sandbox latency floor, position-paper output). They are not hypothesis surfaces; flag as future-work or as eval-designer measurement targets. **Long-horizon coherence under file-handoff** (item 4 in GF-1 §4) is the most directly spec-relevant and should be carried to the synthesist as a measurement future-work flag.

---

## (d) Hand-off note to hypothesis-smith — order of attack

Hypothesis-smith should attack the shortlist in this order. The principle is **most-mature-prior-art first**: hypotheses with the cleanest generic-task magnitudes and clearest non-judge falsification signals get red-teamed first, because they have the highest survival probability and unblock the spec's ≥3-surviving floor earliest.

1. **S1 — Heterogeneous-model split.** Cleanest prior-art: Zhang 2502.08788 is a meta-eval (5 × 9 × 4 sweep) — strongest magnitude evidence on the shortlist. Falsification via SPECS controlled-flaw injection is well-defined. Highest-confidence shortlist entry.
2. **S2 — Length-control wrapper on judge calls.** Cleanest non-judge signal (token-count regression). Direct precondition for S3 and S4 — if S2 dies, S3 and S4 need extra defense for the LLM-as-judge components.
3. **S5 — Citation pre-flight gate.** Cheapest implementation; named MegaResearcher discipline rule; falsifiable via deterministic resolution rate. Survives red-team easily but absolute quality-lift magnitude is moderate (red-team may flag as low-impact).
4. **S6 — Ablation-coverage checklist.** AblationBench is a published, named-magnitude substrate. The friction-floor concern (will the eval-designer act on the flag?) is the most likely red-team objection, and the response is the "name the exact missing ablation by ID" framing.
5. **S3 — Voting on structured decisions.** Strong generic-task magnitude (Choi 0.81→0.99 on Arithmetic) but the long-form transfer is unmeasured (GF-2 §(d) item 4). Red-team will press on which decisions actually have plurality — the response is the structured-decision scoping. Strongly depends on S2 surviving.
6. **S4 — Rejected-hypothesis ledger.** The conceptually richest entry but also the one where Reflexion's binary-signal requirement (2310.01798 ceiling) bites hardest if the structured-decision scoping isn't held. The hypothesis-smith should hold the scoping discipline tight here — *only* structured decisions get reflected on, not free prose. Survival depends on S2 + S3 establishing the structured-decision substrate.

If red-team kills S1 or S2, the orchestrator should pause and surface to the user before continuing — S1 and S2 carry the heaviest fraction of expected impact on the shortlist, and losing both leaves the run below the impact level needed to justify a position-paper output.

---

## (e) What I deliberately did not produce

- Hypotheses. The hypothesis-smith owns those. Each shortlist entry is a *gap framing* with feasibility-pass evidence; the architectural augmentation is the hypothesis-smith's lane.
- Eval designs. The eval-designer owns those. C5 evidence above is sufficiency-check for *testability*, not a finished protocol.
- Cost projections beyond rough order-of-magnitude. Token cost is a feasibility filter here; pricing analysis is in the spec's YAGNI fence.
