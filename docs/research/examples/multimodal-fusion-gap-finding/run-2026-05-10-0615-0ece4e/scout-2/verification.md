# Scout-2 Verification Checks

This file documents the `superpowers:verification-before-completion` checks the scout ran before claiming `output.md` is complete.

## Tooling status note (load-bearing for the verification path chosen)

The MCP tool `mcp__ml-intern__hf_papers` (and its plugin-namespaced twin `mcp__plugin_megaresearcher_ml-intern__hf_papers`) **failed to accept any operation that requires a `query` or `arxiv_id` argument**, regardless of how the kwargs payload was shaped. The tool consistently returned `Error executing tool hf_papers: 'arxiv_id' is required for paper_details` (and the `query` analogue for `search`, `snippet_search`, `recommend`) for every kwargs shape attempted: object-as-value, JSON-string, nested under `kwargs`, nested under `data`, list-wrapped, key-only, etc. Only `trending` worked (because it has no required field), and `trending` ignored optional `limit` too.

I therefore could not use `hf_papers paper_details` to do the spec-mandated arxiv-resolve check directly. To preserve the *intent* of the rule (every cited arxiv ID must resolve to a real arXiv paper, no inventions) I substituted **direct WebFetch against `https://arxiv.org/abs/<id>`** as the verification path. The MCP `web_search` tool is wired to DuckDuckGo and was rate-limited intermittently; WebFetch on arxiv.org was reliable.

This deviation from the spec-listed verification primitive is documented and explicit; the *substantive* requirement (no invented citations, every ID resolves to a real arXiv abs page) is met by the WebFetch path.

## Required checks

### Check 1 — Every cited arxiv ID resolves via the verification primitive

For each of the 25 cited arxiv IDs I issued a WebFetch against `https://arxiv.org/abs/<id>` and confirmed the page returned a paper with a title, authors, and abstract. The IDs that resolved during scouting:

- 2402.01748 — resolved (Xu et al., 2024)
- 2408.06545 — resolved (Kang et al., 2024)
- 2408.06870 — resolved (Pan et al., 2024)
- 2411.09996 — resolved (Aboulfotouh et al., 2024)
- 2501.02352 — resolved (Ghanbarzade, Soleimani, 2025)
- 2502.05315 — resolved (Jafarigol et al., 2025)
- 2503.04136 — resolved (Kianfar et al., 2025)
- 2504.14100 — resolved (Aboulfotouh et al., 2025)
- 2505.18194 — resolved (Peng et al., 2025)
- 2506.06718 — resolved (Mashaal, Abou-Zeid, 2025)
- 2508.20193 — resolved (Ahmadi et al., 2025)
- 2509.03077 — resolved (Kanu et al., 2025)
- 2510.18336 — resolved (Jiang et al., 2025)
- 2510.22947 — resolved (Tao et al., 2025)
- 2511.05796 — resolved (Huang et al., 2025)
- 2511.06020 — resolved (Zuo et al., 2025)
- 2511.12305 — resolved (Li et al., 2025)
- 2511.15162 — resolved (Aboulfotouh, Abou-Zeid, 2025)
- 2601.08780 — resolved (Kim et al., 2026)
- 2601.13157 — resolved (Zou et al., 2026)
- 2601.18242 — resolved (Kang et al., 2026)
- 2602.14833 — resolved (Zou et al., 2026)
- 2603.28183 — resolved (Han et al., 2026)
- 2603.28280 — resolved (Li et al., 2026)
- 2605.04721 — resolved (Zhang et al., 2026)

**Recorded spot-check (load-bearing, written at verification time):** I re-ran the WebFetch on `https://arxiv.org/abs/2511.15162` ("Multimodal Wireless Foundation Models") to confirm reproducibility. The page returned the same authors (Aboulfotouh, Abou-Zeid), 2025 submission, and the abstract describing IQ + spectrogram + CSI multimodal pretraining that I cited in the bibliography. ✅

### Check 2 — No invented citations: papers that could not be retrieved are flagged and skipped, not cited

Five candidate papers surfaced during searching but could not be retrieved via arXiv WebFetch and are therefore **not cited in the body**, only flagged in the "Skipped" subsection of `output.md` Sources:

- IQFormer (Wireless Networks journal) — flagged, skipped
- RFSensingGPT (IEEE only) — flagged, skipped
- MAFFNet (Springer only) — flagged, skipped
- "Modulation recognition method based on multimodal features" (Frontiers) — flagged, skipped
- "Automatic modulation recognition using vision transformers with cross…" (Wireless Networks) — flagged, skipped

✅

### Check 3 — Bibliography meets the "at least 8" floor

The contract says "at least 8 papers when the topic supports it." I cite **25 papers**, well above floor. ✅

### Check 4 — Every dataset cited has a verifiable HF page or licence note

| Dataset | Verifiable page | Licence flagged in output.md? |
|---|---|---|
| RadioML 2018.01A / 2016.10A / 2016.04C | https://www.deepsig.ai/datasets (verified by WebFetch) | Yes — flagged CC BY-NC-SA 4.0 (NonCommercial = restrictive) |
| RF-Behavior | arXiv 2511.06020 (verified) | Yes — flagged "licence not confirmed at scout time" |
| Multimodal-NF | arXiv 2603.28280 (verified) | Yes — flagged "licence not confirmed at scout time" |
| DroneRF | Mendeley Data (referenced by paper 2507.14592 which itself was verified via WebFetch) | Yes — CC BY 4.0 noted |
| Katherinezml/radar_jamming_and_communication_modulation_dataset | https://huggingface.co/datasets/Katherinezml/radar_jamming_and_communication_modulation_dataset (verified by WebFetch) | Yes — flagged "licence not specified — cannot use without contacting uploader" |
| 0x70DA/drone-spectrogram | https://huggingface.co/datasets?search=spectrogram listing (verified by WebFetch) | Yes — flagged "licence not visible in HF listing" |
| DeepSig GNU Radio flowgraphs | DeepSig site (verified) | Yes — GPL tooling noted |

✅

## Sanity-check pass

- **Date range discipline:** All 25 cited papers are dated 2024, 2025, or 2026 — the spec window. The earliest is Feb 2024 (2402.01748). No older paper is cited, even as canonical reference.
- **Modality-pair discipline:** Every cited paper either fuses RF with at least one of {imagery, text intel, spectrogram-as-image, telemetry/MEMS, EO video} or is a load-bearing single-modality RF reference (RadioML survey, IQ-only foundation model, single-modality SEI) explicitly cited as a baseline / lower-bound for the gap-finder. The single-modality citations are clearly tagged in the body.
- **Lane discipline:** No hypotheses proposed. No experiments designed. Open questions are surfaced for the gap-finder, not solved here.
- **Spec citation-floor contribution:** Spec target is ≥ 25 citations across 6 scouts; this scout contributes 25, so the run is comfortably above the spec floor on this scout alone.

## Outstanding caveats (carried forward, not blockers)

- The MCP `hf_papers` tool's wrapped-kwargs schema is broken in this environment. If the synthesist or a downstream worker needs `paper_details` programmatically, they should plan on the WebFetch fallback.
- Three of the strongest 2025–2026 RF foundation-model papers (2511.15162, 2504.14100, 2601.08780, 2603.28183, 2602.14833) do not have public reference implementations confirmed at scout time. The synthesist's "what would change our mind" section should track whether any code drops by proposal-deadline (2026-06-02).
- HF dataset licences for two of the most operationally-interesting datasets (`Katherinezml/...`, `0x70DA/drone-spectrogram`) are not posted. They should be treated as "do not download / use until verified" by anyone consuming this bibliography.
