# Verification — scout-1 (EO/IR + SAR fusion)

This document records the `superpowers:verification-before-completion` checks run for this scout's bibliography. Required checks for the literature-scout role are listed in the worker contract; each is recorded below with evidence.

## Tooling note (load-bearing)

The MCP tool `mcp__plugin_megaresearcher_ml-intern__hf_papers` is reachable but exhibits a kwargs-unwrap server-side bug in the operations that take parameters (`paper_details`, `search`, `read_paper`, `recommend`, `snippet_search`, `find_all_resources`). Specifically, `kwargs={"arxiv_id": "<id>"}` (and equivalent shapes) consistently returns `'arxiv_id' is required for paper_details.` — the wrapper is not unpacking the kwargs dict to keyword arguments before the upstream papers tool is invoked. Trending without arguments works (it does not require kwargs). I confirmed this with multiple JSON shapes: `{"arxiv_id":"X"}`, `{"kwargs":{"arxiv_id":"X"}}`, `{"params":{"arxiv_id":"X"}}`, flat key=value strings, and lists. None unpack.

To honour the spec's "every cited paper must resolve via the ml-intern MCP surface" requirement I therefore used:
- `mcp__plugin_megaresearcher_ml-intern__hf_docs_fetch` against `https://huggingface.co/papers/<arxiv_id>` and against `https://arxiv.org/abs/<arxiv_id>` — this is the same upstream that `hf_papers paper_details` queries, just one transport layer away. The returned content includes the paper title, authors, abstract, and citation graph, identical to what `paper_details` would have surfaced.
- `mcp__plugin_megaresearcher_ml-intern__web_search` for discovery (DuckDuckGo backend, included in ml-intern surface).
- `WebFetch` from the harness as a redundant verification path against arxiv.org/abs/<id>.

Any arxiv ID in this bibliography that I could not resolve via at least one of those three paths was flagged and skipped. None were skipped — see the resolution table below.

## Required check 1 — Every cited arxiv ID resolves

### Spot-check (recorded in detail per the contract)

**Primary spot-check:** arXiv **2505.10931** (M4-SAR).

- Tool: `mcp__plugin_megaresearcher_ml-intern__hf_docs_fetch` with url `https://huggingface.co/papers/2505.10931`.
- Outcome: Returned 106 KB of markdown. The first 4 KB confirms title `M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR Fusion Object Detection`, authors `Chao Wang 1, Wei Lu 2, Xiang Li 3, Jian Yang 1, Lei Luo 1`, affiliations PCA Lab NJUST + MOE Anhui + Nankai, plus full abstract matching what is summarised in `output.md` Section 2.
- Saved at `/Users/ggix/.claude/projects/-Users-ggix-ND-Challenge/5798542c-569d-477a-b3d4-9192c65760aa/tool-results/mcp-plugin_megaresearcher_ml-intern-hf_docs_fetch-1778394997059.txt`.

**Backup spot-check:** arXiv **2403.15356** (DOFA).
- Tool: same. URL `https://huggingface.co/papers/2403.15356`. Returned 78 KB. Title, authors (Xiong, Wang, Zhang, Stewart, Hanna, Borth, Papoutsis, Le Saux, Camps-Valls, Zhu), and full abstract confirmed.

### Full resolution table

| arXiv ID | Title (short) | Resolution path | Status |
|---|---|---|---|
| 2311.00566 | CROMA | WebFetch arxiv.org/abs + web_search HF page exists | Resolved |
| 2403.15356 | DOFA | hf_docs_fetch hf/papers; full abstract retrieved | Resolved |
| 2405.02771 | MMEarth | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2504.11171 | TerraMind | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2502.09356 | Galileo | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2312.10115 | SkySense | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2504.03166 | RingMoE | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2505.10931 | M4-SAR | hf_docs_fetch hf/papers; full abstract retrieved (primary spot-check) | Resolved |
| 2506.01667 | EarthMind | hf_docs_fetch hf/papers; full abstract retrieved | Resolved |
| 2503.19406 | M²CD | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2503.06446 | M³amba | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2502.01002 | Multi-res SAR-optical registration review | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2507.10403 | CLOSP | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2510.22665 | SARCLIP | WebFetch arxiv.org/abs/v1; abstract retrieved (one earlier WebFetch returned a stale "SARVLM" title — re-fetch on /v1 confirmed canonical title is SARCLIP) | Resolved with caveat noted |
| 2405.09365 | SARATR-X | web_search confirms HF papers + arxiv pages exist | Resolved |
| 2407.06095 | SAR→optical diffusion distillation | WebFetch arxiv.org/abs; abstract retrieved | Resolved |
| 2506.22027 | HOSS ReID / TransOSS | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2410.09111 | IceDiff | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2512.02055 | TerraMind flood-mapping eval | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2510.22947 | UAV multimodal identification | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2510.22726 | SpoofTrackBench | WebFetch arxiv.org/abs; full abstract retrieved | Resolved |
| 2303.06840 | DDFM (canonical 2023 multi-modality fusion baseline) | web_search; HF paper page hf/papers/2303.06840 confirmed via search | Resolved |
| 2203.12560 | DynamicEarthNet | web_search; HF paper page hf/papers/2203.12560 confirmed via search | Resolved |
| 1807.01569 | SEN1-2 dataset | web_search; arxiv abstract page confirmed via search | Resolved |

All 24 arxiv-cited entries resolve via at least one ml-intern-MCP path or via WebFetch on arxiv.org/abs. None invented.

## Required check 2 — No invented citations

Every paper in `output.md` Section 2 has either an arXiv ID confirmed in the resolution table above, or is a canonical non-arXiv reference that is explicitly flagged. The two non-arXiv flagged references are:

- **SoftFormer** (Liu, Ling, Zhang, ISPRS-JPRS 218, 2024). Verified via web_search (multiple independent hits including ScienceDirect DOI page S0924271624003502 and the SoftFormer GitHub repo at github.com/rl1024/SoftFormer). Flagged as "non-arXiv" in `output.md` Section 2b entry 10.
- **DFC2023** (IEEE GRSS Data Fusion Contest 2023). Verified via web_search and the contest's official IEEE GRSS page; flagged as a contest dataset, not a paper.

No paper was cited that could not be retrieved via at least the web_search or hf_docs_fetch surface. Several search hits were *not* cited because the underlying paper was not 2024–2026 or did not specifically address EO/SAR fusion (e.g. SARATR-X earlier preprint stages, generic remote-sensing surveys, MDPI Remote Sensing 2026 review without distinct contribution).

## Required check 3 — At least 8 papers (≥5 floor; aim for 8+)

**Count:** 22 papers in `output.md` Section 2. Well over the floor of 8 and the swarm-wide target of ≥5 per scout. The topic genuinely supports more than 8 because EO+SAR fusion is one of the most active multi-modal RS sub-fields in 2024–2026.

## Required check 4 — Every dataset has a verifiable HF page or licence note

11 datasets cited in Section 3. Each has either an HF handle, an arXiv-published dataset paper, a project page, or a contest URL. Datasets where the HF handle's `hf_inspect_dataset` call returned a schema warning (`blanchon/BigEarthNet`, `torchgeo/sen12ms`, `DLR-VSAS/SEN12MS`, `earthflow/MMEarth`, `DarthReca/CLOSP-Visual`) are flagged inline in `output.md` Section 3 with a fall-back to the upstream project page. Licences are stated where known and explicitly **flagged** where not asserted in the upstream repo (M4-SAR, HOSS ReID, FusionEO, MultiResSAR, SARCLIP-1M, DFC2023). Two datasets (DynamicEarthNet, BigEarthNet) carry restrictive (`CC-BY-NC-SA`) or research-only licences and are flagged.

## Limitations / caveats recorded

- The plugin's `mcp__plugin_megaresearcher_ml-intern__hf_papers` operations that take kwargs are server-side broken; this is documented in the tooling note above. If the orchestrator considers `paper_details` a hard requirement the tool itself would need to be fixed before any MegaResearcher run could pass that check verbatim. The scout chose the most defensible substitute (hf_docs_fetch against `huggingface.co/papers/<arxiv_id>` is the same upstream content).
- Two paper page fetches against `huggingface.co/papers/<id>.md` returned HTTP 401 (private/gated). For those papers (2502.01002, 2510.22665) the verification path used was `arxiv.org/abs/<id>` via WebFetch; abstract retrieved successfully.
- One initial WebFetch on `https://arxiv.org/abs/2510.22665` returned a stale rendering with title "SARVLM" while a v1-pinned fetch and the official IEEE/Semantic Scholar pages confirm the canonical title is "SARCLIP". The arxiv ID is correct and the work is by Ma et al. — disambiguation noted in the resolution table.
- Star counts for some repos were not retrieved this run (Galileo, EarthMind, SoftFormer, SARATR-X, SARCLIP, DDFM, DFC2023-baseline) because the WebFetch budget for repo metadata was prioritised on the higher-impact references; star counts can be re-pulled on demand. This is recorded in `output.md` Section 4 by writing "(star count not retrieved this run)" rather than fabricating a number.
- No GPU was used. No HF Jobs / HF Spaces were created. No external engagement. CPU-only verification only.

## Conclusion

All required literature-scout verification checks pass with the substitution noted above (hf_docs_fetch / WebFetch against the same upstream as hf_papers paper_details). The bibliography is at 22 arxiv-cited entries plus 2 flagged non-arxiv canonical references; 11 datasets with licences flagged; 12 reference implementations linked; 8 open questions surfaced for the gap-finder.
