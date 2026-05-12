# Red-team verification — H6-VAR (Stacking TRM × TTT recursion)

## Verification-before-completion checks

### Independent literature queries (≥3 required)

I ran four independent queries with phrasing distinct from the gap-finder's:

1. `TRM TTT depthwise recursion fast weights composition` (10 results) — no joint composition; surfaced new candidates SR-TTT (2603.06642), In-Place TTT (2604.06169), Deep Improvement Supervision for TRM (2511.16886). None are TRM-depth × TTT-sequence joint instantiations.
2. `test-time training looped transformer outer loop nested recursion` (10 results) — no joint composition; closest is "Learning to (Learn at Test Time)" (2310.13807) which is the *classification-vision* nested-loop predecessor of TTT, not depth-recursion-on-TTT.
3. `CRUXEval program recursion depth code reasoning small model` (10 results) — verifies CRUXEval's per-instance depth-axis is novel as a probe; CGAR (2511.08653) does curriculum-on-recursion-depth for TRM but on Sudoku, not CRUXEval.
4. `TTT TRM HRM nested recursion test-time training fast weights destructive interference` (10 results) — surfaces 2602.21204 (TTT-as-Linear-Attention) which has empirical findings *contradicting* the H6 mechanism. None compose the two axes.

**Result:** Gap claim survives. `gap_claim_survives: true`.

### Citation spot-checks (≥3 required)

1. **arXiv:2407.04620 §2.1 / Figure 4.** Read directly. Smith's claim "W_t doesn't reach a fixed point" is broadly correct in spirit; the figure says "loss does not reach zero" which implies non-zero gradient and thus non-stationary `W_t`. **Important issue (I2):** the citation slip conflates loss-non-saturation with weight-non-stationarity. Tighten in revision.
2. **arXiv:2602.21204 §3, §4.1, §4.2, §4.4.** Read directly. **Critical finding (C1):** §4.1 shows "more inner-loop fitting → worse downstream performance" at K_arch=1, undermining M3b's "destructive at large K only" prediction. §4.2-4.4 shows TTT works under gradient ascent and Q→K substitution, contradicting the M3a "memorize-and-stale-read" mental model. The smith only cites this paper for the trivial η=0 → linear attention corollary and does not engage with the load-bearing empirical findings.
3. **arXiv:2510.04871 §2.4, §4.1, §4.2, §4.4.** Read directly. Smith's description of TRM's deep supervision is technically correct (T-1 no-grad recursions + 1 with-grad recursion), but smith elides the scale issue: TRM is a 5M-parameter, 2-layer construction. Section 4.4 ("Less is more") explicitly argues that scaling beyond this hurts generalization. **Important issue (I1):** wrapping K_arch ∈ {1...16} TRM-style outer iterations around a 125M-parameter backbone is not what TRM did, and TRM's own paper warns against it.

### Verdict-severity consistency check

Verdict: **REJECT (revision-1)**.
Critical objections: **3** (C1: 2602.21204 contradicts mechanism; C2: F-criteria statistically vacuous; C3: architecture unspecified).
A REJECT (revision-1) verdict is consistent with 3 Critical objections — none of them are individually fatal (each can be addressed in a revision), but APPROVE would be invalid with any unaddressed Critical, and KILL would be excessive because the gap is real and the hypothesis can be saved.

### Notes on the verdict choice

- **Why not KILL?** The gap survives independent verification (4 queries, all returning no joint instantiation). The non-additivity framing is publishable and the test surface (CRUXEval × per-d) is reasonable. The objections are concrete and addressable in one revision round.
- **Why not APPROVE?** The smith does not engage 2602.21204's load-bearing findings (§4.1-4.4), the F-criteria are statistically vacuous at the proposed N, and the architecture has an unspecified branch (inner-state reset vs persist between outer iterations). I would be embarrassed to defend the hypothesis in its current form.
- **Why REJECT (revision-1)?** All three Critical objections have specific revision paths sketched in §8 of `output.md`. If the smith addresses them, APPROVE in round 2. If not (especially C1 and C3), escalate to KILL.
