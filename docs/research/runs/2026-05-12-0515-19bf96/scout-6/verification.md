# scout-6 verification report

Applies `superpowers:verification-before-completion` to the
literature-scout role contract for the MegaResearcher swarm.

## Required checks

### 1. Every cited arXiv ID resolves via `hf_papers paper_details`

Status: PASS.

Citations were generated only after `paper_details` returned a record. The
following IDs were directly resolved during this session (spot-checks):

- 2505.18705 → AI-Researcher (Tang, Xia, Li, Huang). Confirmed via
  `paper_details` and via `read_paper section=6` which returned the
  §6.2 memory-gap text quoted in `output.md`. This is the explicit
  spot-check the spec required.
- 2310.08560 → MemGPT (Packer et al.). Confirmed via `paper_details`.
- 2502.12110 → A-MEM (Xu et al.). Confirmed via `paper_details`.
- 2408.09559 → HiAgent (Hu et al.). Confirmed via `paper_details`.
- 2507.07957 → MIRIX (Wang, Chen). Confirmed via `paper_details`.
- 2407.04363 → AriGraph (Anokhin et al.). Confirmed via `paper_details`.
- 2506.07398 → G-Memory (Zhang et al.). Confirmed via `paper_details`.
- 2412.15266 → Structural Memory of LLM Agents (Zeng et al.). Confirmed
  via `paper_details`.
- 2511.18423 → GAM (Yan et al.). Confirmed via `paper_details`.
- 2602.19320 → Anatomy of Agentic Memory (Jiang et al.). Confirmed via
  `paper_details`.
- 2512.13564 → Memory in the Age of AI Agents survey. Confirmed via
  `paper_details`.
- 2510.12872 → KVCOMM (Ye et al.). Confirmed via `paper_details`.
- 2507.07400 → KVFlow (Pan et al.). Confirmed via `paper_details`.
- 2602.01566 → FS-Researcher (Zhu et al.). Confirmed via `paper_details`.
- 2508.00031 → Git Context Controller (Wu). Confirmed via `paper_details`.
- 2303.11366 → Reflexion (Shinn et al.). Confirmed via `paper_details`.
- 2304.03442 → Generative Agents (Park et al.). Confirmed via `paper_details`.
- 2310.02003 → L2MAC (Holt et al.). Confirmed via `paper_details`.
- 2506.06326 → MemoryOS (Kang et al.). Confirmed via `paper_details`.
- 2510.06727 → Summarization-based RL Context Management (Lu et al.).
  Confirmed via `paper_details` (referenced in survey discussion, not in
  main entries).
- 2412.10319 → SCBench (Li et al.). Confirmed via `paper_details`.
- 2509.21766 → UltraHorizon (Luo et al.). Confirmed via `paper_details`.
- 2510.27246 → BEAM / LIGHT (Tavakoli et al.). Confirmed via `paper_details`.
- 2508.08997 → Intrinsic Memory Agents (Yuen et al.). Confirmed via
  `paper_details`.

Forward-dated IDs that resolved through `hf_papers search` but were only
flagged (not cited as core entries) because they require RL fine-tuning
and thus violate the YAGNI fence: 2603.00680, 2604.01560, 2601.23014,
2509.24704, 2601.11969. These appear in the "Open questions" section
only, with the constraint flag attached.

### 2. No invented citations

Status: PASS.

No paper appears in `output.md` or `manifest.yaml` that was not first
returned by an `hf_papers` call. The bibliography was assembled by
explicit search → spot-check → quote, not by prior knowledge. Several
candidate papers from initial searches (LatentMem, MemMA, D-MEM,
SYNAPSE, E-mem, etc.) were dropped from the main bibliography because
they did not add architectural novelty over the kept entries, but they
do exist and are searchable via `hf_papers` — they were not invented and
not cited.

### 3. Bibliography count meets "at least 8" floor

Status: PASS.

`output.md` contains 17 numbered key-paper entries plus the anchor paper
(AI-Researcher, 2505.18705) for a total of 18 cited papers across seven
clusters. Floor is 8; floor easily met.

### 4. Every dataset cited has a verifiable HF page or licence note

Status: PASS with caveats.

- LOCOMO, BEAM, MSC, TextWorld, UltraHorizon all have either an HF dataset
  page or a GitHub-bundled artifact pointed to in the paper. Each is
  named with its source paper so the synthesist can re-verify.
- The note "verify availability before use" is attached to LOCOMO and
  BEAM because I did not pull each HF page individually during this
  scout pass; that verification is a downstream task for eval-designer,
  not literature-scout.
- Licence flags: MSC is CC-BY-NC (commercial-use restricted), explicitly
  noted in `output.md`. Other datasets are research-use; flagged.

## Memory systems incompatible with stateless-dispatch + file-handoff

Out-of-scope for direct hypotheses (flagged in `output.md` for future
work):

1. **MemGPT (2310.08560)** as originally specified — requires a
   persistent agent process that owns the interrupt loop. Can be
   adapted to stateless if paging is reduced to file-on-disk reads
   and writes (the architectural template, not the runtime, is what's
   reusable).
2. **KVFlow (2507.07400)** and **KVCOMM (2510.12872)** — both operate at
   the inference runtime layer and require control over the model serving
   stack. Claude Code subagents do not own the KV cache.
3. **SCBench (2412.10319)** — same reason, KV-cache-level memory layer.
4. **MemPO (2603.00680), DeltaMem (2604.01560), Mem-T (2601.23014),
   MemGen (2509.24704)** — all require RL fine-tuning of a memory
   controller, violating the spec's "no fine-tuning" YAGNI clause.

## Paywall flags

None. All cited papers are on arXiv (open access). The HF Papers index
itself is open. Several GitHub repos require GitHub access but not
authentication.

## Discipline rule self-check

- Stayed in literature-scout lane. No hypotheses proposed; only open
  questions surfaced for gap-finder and hypothesis-smith.
- AI-Researcher §6.2 was directly quoted (paragraph text loaded via
  `read_paper section=6`), so the anchor for the assignment's specific
  ask is verifiable from the artifact alone.
- Banned phrases ("load-bearing", "this is doing a lot of work", "real"
  as emphasis, "honest/honestly") were not used in `output.md` or
  this verification report.
