# Verification — hypothesis-smith-S5 (revision 2)

Wraps `superpowers:verification-before-completion`. Re-verifies the revised hypothesis after red-team-S5 round 2 issued REJECT (revision-2).

## Required checks

### 1. Hypothesis statement is in if/then form

PASS. §2 — "**If** MegaResearcher inserts a leaf `citation-verifier` worker … **then** … H_treatment = 0% … **AND** the gate causes the synthesist to drop ≤ 15% of substantive claims …" The "AND" clause is preserved; the substrate moves to PaperWrite-Bench primary + n=30 secondary; the deterministic-taxonomy and GSAP-NER substrate are explicit.

### 2. At least 3 falsification criteria, each genuinely falsifiable

PASS. Six criteria (F1–F6) in §5:

- **F1** — D > 15% on the deterministic three-class taxonomy. Class A grounded in **GSAP-NER (arXiv:2311.09860)** rather than raw spaCy. Class B regex expanded to cover BLEU/ROUGE/F1/perplexity/loss/latency/FLOPs patterns. Rewrite-to-prose explicitly counted as a drop. 5-manuscript extraction-completeness audit pre-registered as an eval-protocol check.
- **F2** — Two-scale calibration (5% total-phantom floor, 1% pure-Ghost sub-floor). PaperWrite-Bench's published baseline ">10 hallucinations/paper" expected to clear F2 cleanly.
- **F3** — Gate-integrity (H_treatment > 0). Binary; one counter-example sufficient.
- **F4** — Retrieve-then-write subsumption test, **now fully deterministic** (NEW-C2 fix). Off-corpus rate uses `hf_papers paper_details` resolution identical to the gate + Citegeist retrieval-set auditing. No LLM-judge surface.
- **F5** — CiteME attribution regression (> 2pp). Published benchmark.
- **F6** — Synthesist failure-to-emit > 10%. Binary deployability falsifier.

All criteria deterministic or threshold-on-measurement. No LLM-judge dependency anywhere in the falsification surface.

### 3. Every mechanism claim has a citation

PASS. §3:

- **Component A** — cites Huang et al. (arXiv:2310.01798) for the external-signal requirement; cites ARIS (arXiv:2605.03042) §3.2 for the LLM-judge contrast.
- **Component B** — cites Citegeist (arXiv:2503.23229) and ScholarCopilot (arXiv:2504.00824) for retrieve-then-write's stronger constructive guarantee; cites ARIS (arXiv:2605.03042) §3.1 ("audit is advisory at the workflow level: it does not halt execution") and §3.2 (KEEP/FIX/REPLACE/REMOVE recommendations for human approval) for the blocking-vs-advisory delta.
- **Component C** — cites Citegeist for the per-paragraph multi-citation pattern; explicitly drops the rewrite-to-prose contribution per red-team CR5; 15% threshold flagged as engineering judgment.
- Not-a-revision-loop framing — cites Feedback Friction (arXiv:2506.11930).
- Pre-registered conditional on citation density disclosed.

### 4. All cited arxiv IDs resolve via `hf_papers paper_details`

PASS. 22 distinct arxiv IDs cited. All verified in-thread (15 carried over from revision 1 + 4 new this revision + 3 retained mechanism). Verification of NEW citations performed in revision-2 thread:

| ID | Title | Status |
|---|---|---|
| 2408.06292 | AI Scientist v1 | resolves (13.5k stars; carried) |
| 2504.08066 | AI Scientist v2 | resolves (6.1k stars; carried) |
| 2503.18102 | AgentRxiv | resolves (5.5k stars; carried) |
| 2511.04583 | Jr. AI Scientist | resolves (30 stars; carried) |
| **2605.03042** | **ARIS** | **resolves (107 upvotes, 8890 stars; NEW; §3.1 + §3.2 + §4.4 + §6 read in-thread)** |
| 2503.23229 | Citegeist | resolves (carried) |
| 2504.00824 | ScholarCopilot | resolves (carried) |
| 2511.17689 | ARISE | resolves (carried) |
| 2603.14629 | ResearchPilot | resolves (carried) |
| 2602.23452 | CiteAudit | resolves (carried) |
| 2511.16198 | SemanticCite | resolves (carried) |
| 2604.04074 | FactReview | resolves (carried) |
| 2407.12861 | CiteME | resolves (carried) |
| 2510.17853 | CiteGuard | resolves (carried) |
| 2411.00294 | LLM-Ref | resolves (carried) |
| **2601.16993** | **BibAgent** | **resolves (NEW; abstract read in-thread)** |
| **2604.01128** | **PaperRecon / PaperWrite-Bench** | **resolves (15 upvotes; NEW; abstract read in-thread)** |
| **2311.09860** | **GSAP-NER** | **resolves (NEW; abstract read in-thread)** |
| 2601.17431 | The 17% Gap | resolves (carried) |
| 2310.01798 | Huang et al. | resolves (carried) |
| 2506.11930 | Feedback Friction | resolves (carried) |
| 2509.08713 | Hidden Pitfalls | resolves (carried) |

### 5. The "Risks to the hypothesis" section is non-empty

PASS. §7 lists seven risks (R1–R7). R6 (ARIS subsumption) and R7 (citation density < 2 collapses Component C) are new in revision 2. Each risk pairs with a what-still-contributes clause.

### 6. On revisions: every red-team objection has an explicit response

PASS. The opening "Response to red-team revision-2 objections" section addresses:

- **NEW-C1 (ARIS uncited)** — full engagement. ARIS §3.1 + §3.2 read; three-axis architectural delta documented (blocking vs advisory; deterministic vs judge; existence-only vs bundled); ARIS added to scout-1's matrix manually in §1; gap claim re-tightened; explicit contraction of magnitude acknowledged.
- **NEW-C2 (F4 not deterministic)** — F4 specification migrated to `hf_papers paper_details` + Citegeist retrieval-set auditing. No LLM-judge anywhere in F4. Thresholds (2% / 5%) pre-registered against this protocol.
- **NEW-C3 (PaperRecon uncited)** — PaperWrite-Bench adopted as primary substrate (n=51); AI-Scientist-family n=30 retained as secondary; PaperRecon added to S8 sources; same author cohort (Miyai) as Jr. AI Scientist disclosed.
- **Important #4 (GSAP-NER)** — F1 Class A grounded in GSAP-NER's published pipeline; Class B regex expanded; 5-manuscript hand-labeled extraction-completeness audit pre-registered.
- **Important #5 (§3 / §5 inconsistency)** — RESOLVED. Rewrite-to-prose explicitly counted as a drop under F1's matching rule. Component C narrowed to citation-level redundancy alone. D ≤ 15% explicitly downgraded to engineering judgment + conditional-on-density-≥-2.
- **Important #6 (n=10 per system)** — addressed via PaperWrite-Bench n=51 primary substrate.
- **Suggestion 7 (BibAgent)** — cited in §8.

## New-critical-defect confirmation matrix

| New critical defect (red-team §4) | Revision-2 change | Location |
|---|---|---|
| NEW-C1: ARIS (arXiv:2605.03042) uncited | ARIS engaged head-on; §3.1 + §3.2 + §4.4 + §6 read in-thread; three-axis architectural delta documented; ARIS added to AI-Scientist-family population; gap claim re-tightened; explicit contraction of magnitude | Response §CR1; §1 updated narrow gap; §3 Components A and B; §8 sources |
| NEW-C2: F4 off-corpus rate not deterministic | F4 specified deterministically using `hf_papers paper_details` + Citegeist retrieval-set auditing; thresholds pre-registered against this protocol; no LLM-judge surface | Response §CR2; §5 F4 full re-specification |
| NEW-C3: PaperRecon (arXiv:2604.01128) uncited | PaperWrite-Bench adopted as primary substrate (n=51); AI-Scientist-family n=30 retained as secondary; PaperRecon cited; same author cohort as Jr. AI Scientist disclosed | Response §CR3; §1; §6 dataset substrate; §8 sources |

## Important-defect confirmation matrix

| Important defect (red-team §9) | Revision-2 change | Location |
|---|---|---|
| F1 Class A under-specified vs GSAP-NER | Class A grounded in GSAP-NER pipeline | §5 F1 Class A |
| F1 Class B regex too narrow | Expanded to BLEU/ROUGE/F1/perplexity/loss/latency/FLOPs/samples | §5 F1 Class B |
| §3 Component C / §5 F1 contradiction | RESOLVED: rewrite-to-prose counted as drop; Component C narrowed to redundancy only | §3 Component C; §5 F1 drop definition |
| n=10 per system sub-significant | Migrated to PaperWrite-Bench n=51 primary | §6 dataset substrate |

## Carried-forward defect confirmation matrix (from round 1, must not regress)

| Round-1 critical defect | Round-1 fix | Round-2 status |
|---|---|---|
| C1: 17% prior overstated | H_baseline as measurement; F2 two-scale; population mismatch disclosed | PRESERVED |
| C2: Citegeist/ScholarCopilot subsumption | Narrow gap only; F4 steelman falsifier | PRESERVED + extended to include ARIS in the AI-Scientist-family population |
| C3: F1 admits LLM-judge | Deterministic three-class taxonomy | PRESERVED + strengthened (GSAP-NER substrate, expanded regex, extraction-completeness audit) |

## Discipline-rule compliance

- **Falsifiability non-negotiable.** Met — six falsifiers, all deterministic.
- **Cite every mechanism claim.** Met — §3 grounded in cited prior art including ARIS §3.1 / §3.2.
- **Specific magnitudes, not directions.** Met — H_treatment = 0% by construction; D ≤ 15%; F2 5%/1%; F4 2%/5%; F5 2pp; F6 10%.
- **Stay in lane.** Met — §6 sketches dataset/baseline/ablation; defers protocol details to eval-designer.
- **CLAUDE.md rule #3 (pre-registration).** Met — all thresholds locked; deterministic taxonomy specified; extraction-completeness audit pre-registered.
- **CLAUDE.md rule #4 (citations resolve).** Met — 22/22 verified.
- **CLAUDE.md rule #5 (workers stay in lanes).** Met.

## Banned-phrase scan

Per global user instructions: scanned output.md and manifest.yaml for banned phrases.

- The "load-..." banned phrase — 0 hits
- The "doing a lot of work" / "heavy lifting" patterns — 0 hits
- The "r-..." word used emphatically — 0 hits
- The "h-..." framing words — 0 hits

## Verification verdict

PASS. All six required checks pass. The three new critical defects from red-team revision-2 are addressed in the response section (CR1/CR2/CR3) and propagated through §1 (narrow gap re-tightened with ARIS), §3 (Components A and B engage ARIS directly), §5 (F1 grounded in GSAP-NER + expanded regex + rewrite-as-drop; F4 deterministic), §6 (PaperWrite-Bench primary substrate), §7 (R6 and R7 added), and §8 (NEW citations). Prior critical defects C1/C2/C3 from round 1 remain addressed and have not regressed.

The hypothesis is materially narrower than revision-1 in two places: (a) the gap claim now requires block-on-unresolvable + deterministic-resolver + existence-only-scope to survive ARIS as a peer; (b) Component C's predicted-D-bound now rests on citation-level redundancy alone (no rewrite-to-prose path). The contribution magnitude is explicitly contracted to (a) the three-axis architectural delta against ARIS and (b) the first published D measurement on PaperWrite-Bench.

Recommend forwarding to red-team round 3 with explicit acknowledgment that the contribution is workshop-paper magnitude on the architectural delta and measurement-novel on the PaperWrite-Bench D bound.
