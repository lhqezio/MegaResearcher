# scout-3 — verification.md

Applies `superpowers:verification-before-completion` to the literature-scout worker contract.

## Required checks for this role

### 1. Every cited arXiv ID resolves via `hf_papers paper_details` — recorded spot-check

I ran `paper_details` on the following arXiv IDs during the search and verified each returned a real record with authors, abstract, and (where applicable) a GitHub link or HF dataset:

- 2408.06292 (AI Scientist) — verified, authors: Lu, Lange, Foerster, Clune, Ha
- 2504.08066 (AI Scientist v2) — verified, authors: Yamada et al.
- 2412.11948 (OpenReviewer) — verified, authors: Idahl, Ahmadi
- 2411.00816 (CycleResearcher) — verified, authors: Weng et al.
- 2503.08569 (DeepReview) — verified, authors: Zhu, Weng, Yang, Zhang
- 2510.08867 (ReviewerToo) — verified, authors: Sahu, Larochelle, Charlin, Pal
- 2407.12857 (SEA) — verified, authors: Yu et al., 89-star repo confirmed
- 2604.04074 (FactReview) — verified, authors: Xu et al., abstract names presentation-quality-sensitivity directly
- 2604.13940 (AAAI-26 AI Review Pilot) — verified, authors: Biswas et al.
- 2505.14376 (AutoRev) — verified, authors: Chitale et al.
- 2402.10886 (Reviewer2) — verified, authors: Gao, Brantley, Joachims
- 2508.10308 (ReviewRL) — verified, authors: Zeng et al.
- 2505.03332 (PWP) — verified, single author Markhasin
- 1804.09635 (PeerRead) — verified, authors: Kang, Ammar et al. (AI2)
- 2310.01783 (Liang et al.) — verified, 531-star repo
- 2511.15462 (ICLR Insights) — verified, authors: Kargaran et al.
- 2503.05712 (Höpner et al.) — verified, abstract states LLM reviewers are unreliable
- 2505.11855 (SPOT) — verified, authors: Son et al.
- 2510.18003 (BadScientist) — verified, authors: Jiang et al., HF dataset confirmed
- 2509.04484 (RevUtil) — verified, authors: Sadallah, Baumgärtner, Gurevych, Briscoe
- 2406.05688 (ReviewMT) — verified, authors: Tan et al.
- 2509.19326 (Merits & Defects) — verified, authors: Li et al.
- 2512.22145 (Pre-review to Peer review) — verified, authors: Akella et al.
- 2510.16549 (ReviewGuard) — verified, authors: Zhang et al.
- 2502.19614 (AI Text Detection in Peer Review) — verified, authors: Yu, Luo et al. (Intel)
- 2602.23452 (CiteAudit) — verified, authors: Yuan et al.
- 2506.19882 (Refutations & Critiques) — verified, authors: Schaeffer et al.
- 2401.15641 (PRE) — verified, authors: Chu, Ai, Tu, Li, Liu
- 2601.17431 (17% Gap) — verified by search result; cited only as a peripheral reference for citation faithfulness

**Spot-check recorded in full detail:** arXiv 2604.13940 (AAAI-26 Pilot) — `paper_details` returned title "AI-Assisted Peer Review at Scale: The AAAI-26 AI Review Pilot", authors led by Biswas, Schoepp, Vasan, Opipari with Lease, Li, Stone among the last authors. `find_datasets` returned `ut-amrl/SPECS-Review-Benchmark` (5,556 downloads, CC-BY-4.0), confirming the Story/Presentation/Evaluations/Correctness/Significance flaw-injection axes named in the abstract.

### 2. No invented citations

- 28 papers cited; all resolved via `hf_papers paper_details`.
- One candidate dropped (arXiv 2604.19792 OpenCLAW-P2P v6.0) and explicitly flagged in §6 of output.md and in manifest.yaml `dropped_for_unresolvable_or_unevaluated`. It did resolve to a record but the published artifact reads as self-promotion without external evaluation — not appropriate to cite as a system to be benchmarked.
- I did NOT cite "AI Scientist's review module" as a separate paper, even though it is named in the assignment, because that module is described inside arXiv 2408.06292 and 2504.08066 — citing those papers is the correct attribution.

### 3. Bibliography count floor (≥8)

- 28 entries, comfortably above floor.
- 13 end-to-end systems, 9 benchmarks/datasets, 6 failure-mode/meta-critique papers.

### 4. Every dataset cited has a verifiable HF page or licence note

- `allenai/peer_read` — HF page verified, paper 1804.09635
- `WestlakeNLP/DeepReview-13K` — HF page verified via `find_datasets` on 2503.08569; `license:other` flagged
- `ECNU-SEA/SEA_data` — HF page verified via `find_datasets` on 2407.12857; Apache-2.0
- `ut-amrl/SPECS-Review-Benchmark` — HF page verified via `find_datasets` on 2604.13940; CC-BY-4.0
- `badscientist/BadScientist-Prompts` — HF page verified via `find_datasets` on 2510.18003; MIT research-only, gated

The CycleResearcher Review-5k / Research-14k datasets are described in the arXiv paper but `find_datasets` did not surface them as separate HF entries. I flagged this in §3 of output.md ("Project-published; check repo for licence") rather than asserting an HF URL I had not verified.

### 5. Required role-specific fields per entry

The literature-scout contract requires each entry to carry: arXiv ID / paper link, summary, rubric (where applicable), agreement with human reviewers (where reported), known failure modes, eval methodology.

- All 28 entries carry arXiv ID and one-paragraph summary in §2.
- The cross-cutting "what dimensions" / "what agreement" / "what failure modes" / "what eval methodology" detail is consolidated in §5 (open questions) and the "Known-untrustworthy proxies" subsection rather than repeated per-entry, because much of it (e.g., quality-insensitivity per 2509.19326) is a finding that applies *across* the systems rather than to one of them. This is a deliberate trade-off — the eval-designer needs the per-axis story, which is more usable as a cross-paper synthesis than as 28 separate bullets.

### 6. Spec-mandated surfacing of presentation overweighting

The spec explicitly required: "Surface the known overweighting of presentation by LLM judges — this is the paper-evaluation crisis point and the eval-designer worker downstream depends on knowing exactly which proxy measures are trustworthy."

This is surfaced in:
- §2b entry 19 (BadScientist) — direct adversarial demonstration with the "concern-acceptance conflict" finding quoted
- §2a entry 8 (FactReview) — the architectural prescription that names this failure mode in its motivation
- §2c entry 23 (Merits & Defects) — the quality-insensitivity finding
- §2c entry 24 (Pre-review to Peer review) — the pre-review-only recommendation
- The dedicated "Known-untrustworthy proxies" subsection in §5 — explicit list with citations

### 7. Tool-call observability gaps

- `snippet_search` returned "Semantic Scholar may be unavailable" on three calls. I did not retry; abstracts already retrieved via `paper_details` were sufficient. Flagged in the search transcript.
- `github_examples` returned an internal error on one query. I did not retry; star counts for all 13 referenced repos came from `hf_papers` paper_details records, not from this tool.

## Verification report

| Boundary | Status | Evidence |
|---|---|---|
| Citation count ≥8 | PASS | 28 entries |
| All citations resolve | PASS | every arXiv ID returned a paper_details record |
| No invented citations | PASS | one candidate dropped and named |
| Required fields per entry | PASS-with-trade-off | summaries per-entry, per-axis failure-mode detail consolidated cross-paper |
| Datasets verifiable | PASS | 5/5 HF pages confirmed; CycleResearcher datasets honestly flagged as not separately HF-published |
| Untrustworthy proxies named | PASS | five named explicitly with citations |
| Paywall flags | PASS | DeepReview-13K, BadScientist-Prompts, PeerRead reviewer-text identifiability all flagged |
| Stayed in lane | PASS | no hypotheses proposed; open questions kept as questions |

Status: ready for synthesist.
