# scout-1 — Verification

This worker ran the `superpowers:verification-before-completion` checks adapted for the literature-scout role. Below: the checks, the evidence, and the verdicts.

## Required check 1 — every cited arXiv ID resolves via `hf_papers paper_details`

I called either `paper_details` or `read_paper` against `mcp__ml-intern__hf_papers` for every arXiv ID cited in §6 of `output.md`. Each one returned a non-error result with author list and abstract matching the inline summary I wrote. None were fabricated.

**Spot-check recorded here (per role contract):**
- Called `hf_papers paper_details` with `arxiv_id=2505.18705`. Result returned title "AI-Researcher: Autonomous Scientific Innovation", authors "Jiabin Tang, Lianghao Xia, Zhonghang Li, Chao Huang", GitHub `hkuds/ai-researcher` at 5,312 stars. Matches the inline citation. Confirmed resolvable.
- Cross-spot-check `hf_papers paper_details` with `arxiv_id=2503.18102`. Result returned title "AgentRxiv: Towards Collaborative Autonomous Research", authors "Samuel Schmidgall, Michael Moor", GitHub `SamuelSchmidgall/AgentLaboratory` at 5,579 stars. Matches.
- I also `read_paper` the AI-Researcher § 6 and the AgentRxiv § 2.1 + § 4 directly to verify the architectural claims I quote (multi-turn fidelity, memory gap, agent roster, reward hacking) are accurate to the source text. They are.

Verdict: **pass.**

## Required check 2 — no invented citations

The discipline rule: if `hf_papers` doesn't return a paper, the paper does not exist for purposes of this output.

Coscientist (Boiko et al., Nature 2023) and Virtual Lab (Swanson/Zou et al., Nature 2025 / bioRxiv 2024) are named in the swarm spec but **did not return from `hf_papers`** in any search. I confirmed their non-existence in the index by running both `search` and verifying no matching arXiv ID was returned for: "Boiko Emergent autonomous scientific research large language models" and "Swanson Zou Virtual Lab nanobody design agent". Both are paywalled / Nature-only / bioRxiv-only.

Per the rule, I **excluded** both from the verified arXiv-ID bibliography and the count. I describe each from the public record (Nature URLs + biohub press) and explicitly mark them as flagged in:
- `output.md` § 2b ("Dolphin contemporaries already covered above. The following are flagged but **not arXiv-resolvable** and therefore excluded from the verified count:")
- `output.md` § 6 (under "Flagged paywalled / not on arXiv")
- `manifest.yaml` (`flagged_non_arxiv` block)

Verdict: **pass.**

## Required check 3 — bibliography count meets the ≥8 floor

The plan-level worker contract required "≥8 systems with this level of detail" (agent roster, loops, artifact format, evaluation, limitations).

I delivered **14 systems** with the full required-fields treatment (`detail_systems` in `manifest.yaml`):
1. AI Scientist v1
2. AI Scientist v2
3. Agent Laboratory
4. AgentRxiv
5. AI-Researcher
6. Dolphin
7. EvoScientist
8. freephdlabor
9. Idea2Paper
10. Jr. AI Scientist
11. CycleResearcher
12. PaperOrchestra
13. Aviary
14. Curie

Plus Baby-AIGS with the full treatment as a falsification-loop counterpoint, ResearchAgent and VirSci as ideation-focused near-neighbors (briefer), and the critical-commentary entries (Luo/Kasirzadeh/Shah, Trehan/Chopra). Total **27 arXiv-resolved citations** in §6.

The ≥8 floor is met with margin.

Verdict: **pass.**

## Required check 4 — every dataset cited has a verifiable HF page or licence note

Honest report: **partial.**

The end-to-end-pipelines sub-topic does not ship "training datasets" in the standard sense; what gets cited as datasets are evaluation suites (Scientist-Bench, MLAgentBench, AstaBench, ResearchGym, FML-bench, CORE-Bench, PRBench, FIRE-Bench, Review-5k, Research-14k). Most of these are distributed via the originating GitHub repos and *do not yet have a dedicated HF Datasets card*. I flagged this explicitly in §3 of `output.md`:

> "Licences and HF Dataset cards are not consistently posted for this family — flagging 'verifiable HF page or licence note' status as **incomplete for this sub-topic** (covered properly by gap-finders and the eval-designer downstream)."

This is a known gap and is handed off to the gap-finder workers + eval-designer rather than fabricated by this scout. Each dataset I listed has an arXiv ID (verified above) and a GitHub repo where licence can be inspected by a follow-up worker.

Verdict: **acknowledged gap; not invented.**

## Systems searched for but not found (full disclosure)

- **Coscientist (Boiko 2023)** — not in `hf_papers` index. Flagged.
- **Virtual Lab (Swanson/Zou 2024–2025)** — not in `hf_papers` index. Flagged.
- **Genesis-Flow** — named in the swarm spec. `hf_papers` search for "Genesis Flow autonomous research" returned PiFlow, FlowSearch, Genesis (driving-scene), LoongFlow, and AgentRxiv — none matched. I could not find a system called Genesis-Flow in the autonomous-research family in the index. Flagged as not-found.
- **Carl (Autoscience) / Zochi (Intology)** — named in some blog write-ups (not the swarm spec). `hf_papers` search "Zochi Carl AI scientist conference paper accepted" returned the AI Scientist papers + commentary, not Carl or Zochi as named systems. Flagged as not-found in arXiv-index — both are commercial systems with company blog posts but no arXiv entry I could resolve.

These four are noted here for the synthesist's benefit but **not cited** in `output.md`.

## Summary

| Check | Verdict |
|---|---|
| Every cited arXiv ID resolves | pass |
| No invented citations | pass |
| ≥8 systems with full required fields | pass (14 delivered) |
| Every dataset cited has HF page or licence note | partial — gap flagged, not papered over |

Worker reports complete and ready for the orchestrator's verification gate.
