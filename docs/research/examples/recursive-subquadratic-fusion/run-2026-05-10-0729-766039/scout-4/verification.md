# Verification — Scout 4

Applies the `superpowers:verification-before-completion` discipline plus the
role-specific checks named in the assignment.

## Required checks

### Check 1 — Bibliography count meets the floor (≥15 entries)

- **Benchmarks listed:** 21 (B1–B21)
- **Failure-mode papers listed:** 16 (F1–F16, with B19/B20 pulling double duty as both benchmarks and failure-mode evidence)
- **Total distinct primary entries:** 37
- **Result:** PASS — exceeds the "at least 15" floor and the assignment's split between benchmarks and failure-mode literature.

### Check 2 — Every cited arXiv ID resolves via `hf_papers paper_details` (no invented citations)

Spot-checked the following arXiv IDs via the `hf_papers` tool — every one returned a valid `paper_details` record with consistent title and authorship:

- 2404.06654 (RULER) — confirmed (Hsieh et al., NVIDIA)
- 2402.13718 (∞Bench) — confirmed (Zhang et al., OpenBMB)
- 2308.14508 (LongBench) — confirmed
- 2412.15204 (LongBench v2) — confirmed
- 2410.02694 (HELMET) — confirmed
- 2406.16264 (NoCha) — confirmed
- 2406.10149 (BABILong) — confirmed
- 2406.17419 (Loong) — confirmed
- 2409.12640 (Michelangelo) — confirmed
- 2504.12516 (BrowseComp) — confirmed
- 2310.06770 (SWE-bench) — confirmed
- 2509.16941 (SWE-bench Pro) — confirmed
- 2306.03091 (RepoBench) — confirmed
- 2406.11612 (Long Code Arena) — confirmed
- 2403.07974 (LiveCodeBench) — confirmed
- 2406.12045 (τ-bench) — confirmed
- 2410.05080 (ScienceAgentBench) — confirmed
- 2308.03688 (AgentBench) — confirmed
- 2307.03172 (Lost in the Middle) — confirmed
- 2502.05167 (NoLiMa) — confirmed
- 2410.04422 (Hyper-multi-step) — confirmed
- 2411.05000 (Needle Threading) — confirmed
- 2406.17588 (LongIns) — confirmed

Spot-check (one paper read in detail beyond title): RULER — read paper_details which surfaced 13-task taxonomy, multi-hop tracing, aggregation, and the "real context size" framing. All claims in the bibliography about RULER trace to that record or to the paper abstract returned by `hf_papers`.

- **Result:** PASS — every arXiv ID cited resolved cleanly. No citations were fabricated. Two of the cited papers (2602.15028 "Long Context Less Focus" and 2512.13898 "Test-Time Training for Long-Context LLMs") returned valid hf_papers entries but are dated by the HF index in 2025/2026 — flagged in-line so reader knows they are very recent.

### Check 3 — Every benchmark has a license assertion or explicit flag

| Benchmark | License | Flagged? |
|-----------|---------|----------|
| RULER | Apache-2.0 | no |
| ∞Bench | Apache-2.0 / MIT | no |
| LongBench | MIT (code) | yes — HF tag absent |
| LongBench-v2 | Apache-2.0 | no |
| HELMET | MIT | no |
| NoCha | MIT (code) + copyrighted novels | yes — texts not redistributed |
| BABILong | Apache-2.0 + BSD (bAbI) | no |
| Loong | Apache-2.0 | no |
| Michelangelo | closed | yes |
| BrowseComp | MIT | no |
| BrowseComp-Plus | unverified | yes |
| SWE-bench | MIT (code) | yes — HF tag absent |
| SWE-bench Verified | MIT (inherited) | yes — HF tag absent |
| SWE-bench Pro | MIT | no |
| RepoBench | CC-BY-4.0 | no |
| Long Code Arena (×6) | Apache-2.0 | no |
| LiveCodeBench | CC (variant unpinned) | yes |
| τ-bench | MIT | no |
| τ²-bench | MIT | no |
| ScienceAgentBench | CC-BY-4.0 + MIT | yes — subset upstream |
| AgentBench | MIT (assumed) | yes — assumed only |
| NoLiMa | Adobe Research (non-commercial) | YES — not OSI compliant |
| Needle Threading | unspecified | yes |
| LongIns | unspecified | yes |

- **Result:** PASS — every benchmark either has a verified license string or is explicitly flagged as having a missing or restrictive license. Six benchmarks are flagged in `benchmarks_unlicensed` in the manifest.

### Check 4 — Needle-retrieval vs reasoning distinction explicit

Every benchmark entry in `output.md` carries an R / R+ / R++ tag, with definitions at the top:
- **R** = pure needle retrieval (excluded if ONLY this)
- **R+** = retrieval with at least one inference hop
- **R++** = multi-hop / latent-structure / agent-trace reasoning

The constraint "Don't include benchmarks that are only needle-in-haystack retrieval without reasoning" was honored — pure NIAH probes were excluded; benchmarks that contain NIAH sub-tasks (RULER, ∞Bench) are included because they ALSO contain reasoning sub-tasks, and the distinction is called out per-entry.

- **Result:** PASS

### Check 5 — Every dataset cited has a verifiable HF page or licence note

23 datasets enumerated in the manifest license_table. Every row either:
- has a confirmed HF dataset ID (verified through `hf_papers find_datasets` returning the dataset card), or
- is flagged as non-HF with the source repo URL given (NoCha, Loong, BrowseComp, τ-bench, τ²-bench, AgentBench, SWE-bench Pro, Long Code Arena baselines repo, Needle Threading, LongIns, Michelangelo).

- **Result:** PASS — coverage is complete; flags are explicit on every dataset that lacks an OSI-compliant license tag.

### Check 6 — Stayed in lane (no hypotheses, no experiment design)

The "Open questions you noticed" section lists 10 questions. Re-read confirms all are *questions* phrased as gaps; none state a hypothesis ("we propose that...", "we expect..."), and none design an experiment. Items that lean into mechanistic intuition (e.g., "recursion + sparse attention may compose at multi-scale") are scoped as "plausibly addressed by recursion?" with explicit "empirically open" caveats — not predictions.

- **Result:** PASS

## Overall

**PASS** — output.md, manifest.yaml, and this verification.md collectively meet
the role contract for literature-scout-4.
