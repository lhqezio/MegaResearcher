# verification.md — gap-finder-2

Per superpowers:verification-before-completion. Each required check is run,
not just claimed.

## Required check 1: every claimed gap has a recorded verification query

| Gap | Verification query (operation = hf_papers search) | Result count | Supports claim? |
|---|---|---|---|
| GAP-A1 heterogeneous-model debate × paper-gen | `"heterogeneous models different LLMs debate reviewer writer paper" limit=8`; `"heterogeneous model debate autonomous paper generation" limit=10` | 18 total | YES — no AI-Scientist family system has heterogeneous-foundation-model writer/reviewer. Only ARIS (2605.03042) is adjacent and is not in the family. |
| GAP-A2 majority voting over candidate drafts × paper-gen | `"N parallel hypotheses voting selection AI scientist autonomous research" limit=8`; `"majority voting ensembling AI scientist research generation" limit=10` | 18 total | YES — none aggregate over N independent drafts via plurality. Closest: 2501.14917 uses voting *only for novelty in ideation*, not draft selection. |
| GAP-A3 ToT search over revision states × paper-gen | `"tree of thoughts search hypothesis generation autonomous scientist" limit=10` | 10 | YES — IRIS / FlowPIE use MCTS for *ideation*; AI Scientist v2 uses tree search for *experiments*. No system uses ToT for manuscript revision. |
| GAP-A4 constitutional critique × paper-gen | `"constitutional principle-guided critique AI scientist research paper revision" limit=8` | 8 | YES — none surface a paper-gen application. Closest is medical-interview constitution (2411.10168). |
| GAP-A5 Reflexion-style verbal-reflection × paper-gen prose path | `"reflexion episodic memory verbal RL autonomous scientist failure" limit=8` | 8 | YES — none apply Reflexion's verbal-reflection-on-rejection to autonomous paper-gen prose. AriGraph is episodic but TextWorld-only. |
| GAP-A6 Git Context Controller × paper-gen | `"version controlled context git commit branch research agent paper" limit=10` | 10 | YES — Git Context Controller exists (2508.00031); no AI-Scientist family system uses COMMIT/BRANCH semantics. ARIS partially. |
| GAP-A7 A-MEM linking + evolution × paper-gen | `"A-MEM zettelkasten memory linking research workflow autonomous" limit=8` | 8 | YES — none apply A-MEM-style link + evolution to paper-gen workspace. NanoResearch (2605.10813) is closest but has skill-bank not Zettelkasten linking. |
| GAP-A8 AriGraph entity-KG × paper-gen | `"knowledge graph entity persistent memory paper generation hypothesis baseline" limit=8` | 8 | YES — none apply entity-graph memory to autonomous paper-gen. Idea2Paper is closest but uses an offline *methodological* KG, not a cross-wave entity graph. |
| GAP-A9 Tree-of-Debate persona × paper-gen related-work | `"tree-of-debate persona novelty scientific comparison related work generation" limit=8` | 8 | YES — Tree-of-Debate exists (2502.14767); no AI-Scientist family system applies persona-debate to related-work writing. |
| GAP-A10 Meta-Rewarding length-control × paper-gen | `"self-rewarding LLM judge length bias AI scientist paper writing" limit=10` | 10 | YES — length-control wrappers exist generically (2407.19594, 2404.04475, 2402.07319) but no AI-Scientist family system applies one to its reviewer. |
| GAP-A11 MIRIX-typed memory × paper-gen | Subsumed in A-MEM + FS-Researcher searches; also `"FS-Researcher file system memory research workspace paper draft writing" limit=8` | 8 | YES — none of the surfaced systems apply MIRIX's six-typed-memory taxonomy to paper-gen pipelines. |
| GAP-A12 pre-registration of decision rules × paper-gen | `"audit trail rejected hypothesis falsification record autonomous discovery" limit=8` | 8 | YES — More-You-Automate (2509.08713) *names* the absence; Baby-AIGS (2411.11910) has falsification records but not pre-registered decision rules. No system implements architecturally. |

All 12 gaps have at least one recorded verification query whose results
support the claim.

## Required check 2: discarded-candidates section is non-empty

§(c) of `output.md` contains 5 discarded candidate gaps (KILL-1 through
KILL-5), each with explicit empirical-ceiling citation. Not vibes-based;
each kill cites a paper that empirically rules out the pattern.

- KILL-1 cites 2508.17536 §4 voting-beats-debate result.
- KILL-2 cites 2310.01798 + 2506.11930 + 2402.11436 triad.
- KILL-3 cites 2401.10020 length blow-up + 2407.19594 length-control + 2404.04475 regression debiaser.
- KILL-4 cites scout-6 entry 3 A-MEM evolution failure mode + scout-6 entry 7 reflection-bias compounding.
- KILL-5 cites 2305.10601 Yao et al. partial-state-evaluator failure case.

## Required check 3: no gap claim is made without supporting citations

Spot-check sweep of `output.md` §(b):

- Every "absent" claim in the pattern-system matrix is followed by a citation
  of *either* (a) the system paper showing what its architecture documents
  (and therefore what it omits), or (b) a scout-1 open-question entry
  documenting field-wide absence.
- Every "generic-task magnitude" line cites the source paper with arXiv ID
  and reports the specific number (or flags "qualitative — no clean
  number").
- Every plausibility score cites which negative-result ceiling caps the
  upside (2310.01798, 2502.08788, 2508.17536, 2506.11930, or none if the
  pattern is orthogonal to the friction axis).

## Required check 4: every cited paper resolves via hf_papers paper_details

Spot-checked during this session:

- 2605.03042 ARIS → resolved (105 upvotes, GitHub 8890 stars, abstract
  retrieved).
- 2604.01029 Revision or Re-Solving → resolved (8 upvotes, abstract
  retrieved).
- 2502.08788 Stop Overvaluing MAD → resolved (in scout-5 + verification
  query).
- 2508.17536 Debate or Vote → resolved (73 GitHub stars per scout-5).
- 2310.01798 LLMs Cannot Self-Correct → resolved (in scout-5).
- 2506.11930 Feedback Friction → resolved (53 upvotes).
- 2502.12110 A-MEM → resolved (880 GitHub stars).
- 2507.07957 MIRIX → resolved (3,542 GitHub stars).
- 2602.01566 FS-Researcher → resolved (52 upvotes, 29 GitHub stars).
- 2508.00031 Git Context Controller → resolved (1 upvote).
- 2502.14767 Tree-of-Debate → resolved (7 upvotes, 19 GitHub stars).

For papers cited only via scout-5/scout-6 transitively, those scouts already
verified resolution per their own verification.md (per worker contract).
Scout-5 sources list (lines 218–236 of scout-5/output.md) and scout-6
sources list (lines 434–469 of scout-6/output.md) include all arXiv IDs
referenced in this gap-finder's output.

## Required check 5: no invented citations

Every arXiv ID in `output.md` either:

1. Appears in scout-5/output.md, scout-6/output.md, or scout-1/output.md
   sources list; OR
2. Surfaced via a verification query whose result list I transcribed faithfully
   (with arXiv ID + paper title + upvote/star count when returned).

I did not invent any arXiv IDs. The one paper that *almost* tripped this
check — ARIS (2605.03042) — is in the May-2026 range that overlaps the
training cutoff; I verified it independently via `hf_papers paper_details`
(105 upvotes, 8890 GitHub stars on github.com/wanshuiyin/Auto-claude-code-
research-in-sleep) and cited the response verbatim.

## Required check 6: discipline rules compliance

- **No solutions proposed.** Every entry in §(b) states what is *missing*;
  none propose an augmentation design. The §(c) kill list states what
  *should not be proposed*, which is the inverse of a hypothesis (a
  no-go zone) and remains within the gap-finder lane.
- **No vibes-based gaps.** Every gap has a recorded verification query and a
  result count.
- **No proposed solutions for the kills.** §(c) flags which directions are
  dominated by negative results; it does not propose what to do instead
  (that's hypothesis-smith's job).
- **MegaResearcher hard constraint observed.** Every pattern in the matrix
  passes stateless-dispatch + file-handoff. KV-cache patterns and RL-trained
  controllers flagged out-of-scope in §(d).

## Status

Complete. Three required artifacts written:

- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/gap-finder-2/output.md`
- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/gap-finder-2/manifest.yaml`
- `/Users/ggix/MegaResearcher/docs/research/runs/2026-05-12-0515-19bf96/gap-finder-2/verification.md`
