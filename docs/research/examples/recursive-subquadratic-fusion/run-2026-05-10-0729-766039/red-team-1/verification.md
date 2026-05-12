# Red-team-1 verification self-check

## Independent literature queries (≥3 required)

1. `hf_papers search "recursive transformer NSA native sparse attention BABILong"` (limit 10) — checked whether any paper combines architectural recursion + NSA on BABILong. Result: no positives. Gap survives.
2. `hf_papers search "looped depthwise iteration sparse attention compressed fallback"` (limit 10) — checked whether any paper studies loop iteration with compressed-fallback in attention. Result: no positives. Gap survives.
3. `hf_papers search "TRM tiny recursive model long context multi-hop reasoning"` (limit 10) — checked TRM-on-long-context. Result: only runtime/agentic recursive language models, which do not collapse the architectural-recursion gap.
4. `hf_papers search "tiny recursive model latent reasoning text long context language"` (limit 8) — checked text-domain recursive architectures. Result: only runtime recursion, surveys, and Michelangelo (eval, not architecture).
5. `hf_papers search "350M parameter long context BABILong small model accuracy"` (limit 8) — checked feasibility of 350M-scale BABILong. Result: no 350M-class published BABILong baseline; serves as evidence for noise-floor concern.
6. `hf_papers search "DeepSeek V3.2 DSA sparse attention long context BABILong evaluation"` (limit 8) — checked DSA architecture for cheap-path F3. Result: confirmed DSA architecturally distinct from NSA (lightning indexer, not three-branch compressed/selected/sliding).
7. `hf_papers search "Huginn recurrent depth 3.5B latent reasoning evaluation"` (limit 6) — checked depth-recurrent baselines. Found 2507.02199 (Huginn critique paper) showing latent CoT probing is inconsistent — relevant counter-evidence for F4.

7 independent queries, well above the ≥3 required.

## Citation spot-checks (≥3 required)

1. **NSA arXiv:2502.11089 — methodology + experiments sections read** via `read_paper`. Found the hypothesis misrepresents NSA Tab. 1 (NSA outperforms dense). CRITICAL citation failure.
2. **TRM arXiv:2510.04871 — §4.5 + §5 (Results) read** via `read_paper`. Confirmed TRM is a 5–19M-parameter, attention-free-or-self-attention puzzle architecture; magnitudes do not transfer cleanly to 350M text models.
3. **PLT arXiv:2510.24824 — §2 + §3.1 read** via `read_paper`. Found PLT's loop-recursion benefit on G-SWA (no compressed branch) directly contradicts the hypothesis's MoBA prediction.
4. **BABILong arXiv:2406.10149 — §3.1 read** via `read_paper`. Found that BABILong qa3 is at <80% even for GPT-4-class models; corroborates noise-floor concern at 350M.
5. **Sparse Frontier arXiv:2504.17768 — §4.2 read** via `read_paper`. Confirmed Sparse Frontier studies *training-free* sparse attention, NOT NSA/MoBA. Hypothesis cites it for NSA-vs-MoBA framing — citation mismatch.
6. **Huginn arXiv:2502.05171 — §5 read** via `read_paper`. Confirmed depth-recurrent K=1 evaluation of a model trained for K=32 is catastrophically degraded (GSM8K 0% at K=1 vs 42% at K=32) — supports critique of F3 cheap path.
7. **ParaThinker arXiv:2509.04475 — paper details checked** via `paper_details`. Confirmed Tunnel Vision is a sequential-CoT mechanism, not architectural-recursion. Hypothesis misapplies the concept.
8. **DeepSeek-V3.2 arXiv:2512.02556 — paper details + resources** via `paper_details` + `find_all_resources`. Confirmed DSA is architecturally distinct from NSA; only one community DSA llama3.2-1b-dsa checkpoint exists; only one community NSA-1B checkpoint exists. F3 cheap-path checkpoint claim does not hold.

8 citation spot-checks, well above the ≥3 required. **CRITICAL** misrepresentations found: 2 (NSA Tab. 1 reversal; DSA-NSA conflation). **SERIOUS** misrepresentations found: 3 (PLT dismissal, Sparse Frontier mismatch, Tunnel Vision misapplication).

## Verdict-severity match

- 5 Critical objections (must fix)
- 6 Serious objections (should fix)
- 3 Minor (suggestions)
- Verdict: REJECT (revision-1)

A REJECT with 5 Critical issues is severity-consistent. APPROVE would have been invalid. KILL would have been overkill — the gap survives, the design is recoverable with concrete revisions (R-1 through R-8 in output.md), and the hypothesis-smith can plausibly address them.

## Self-check: not agreeable, not performatively skeptical

I started with the assumption the hypothesis would survive (gap is real, mechanism is plausible). My final critique disagrees with the hypothesis at multiple specific points backed by literal citations from the cited papers (NSA Tab. 1 numbers, PLT §3.1 numbers, TRM §5 model sizes). The objections are concrete and citation-anchored, not "this seems weak." I steelmanned the strongest counter-argument (additive marginals explain everything) and identified it as the central threat the hypothesis must close.

I did not invent objections. I did not fail to find objections — I found 14 distinct issues, weighted by severity, with concrete revision guidance.

PASS.
