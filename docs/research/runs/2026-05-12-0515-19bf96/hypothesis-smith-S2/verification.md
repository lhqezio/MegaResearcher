# verification — hypothesis-smith-S2 (REVISION 1)

Verification against the spec discipline rules
(per `superpowers:verification-before-completion`) plus
revision-specific verification of each red-team objection.

## Required checks

### 1. Hypothesis statement is in if/then form

**Yes.** §2 of `output.md` opens "**If** the MegaResearcher
orchestrator wraps every red-team and synthesist LLM-as-judge
scalar-score call with a Bias Fitting length-debiased post-processor
… **then** on a held-out 20-manuscript fixed-quality test set with
controlled-verbosity paraphrase injection: (a) … (b) … (c) …"

Each of (a), (b), (c) is a statistical prediction with a specific
test (one-sided regression slope sign + significance; two-sided
CI brackets zero; difference-in-differences CI excludes zero).

### 2. At least 3 falsification criteria, each genuinely falsifiable

**Yes — 4 pre-registered.** §5:

- F1 — baseline shows no length bias → H1 KILLED for this judge/
  pipeline configuration. Treated as graceful no-op per I4.
- F2 — wrapper does not suppress the bias → H1 KILLED outright.
- F3 — hacking shifts to a different proxy (8 pre-registered:
  5 surface-textual + 3 substantive — substantive list inspired
  by BadScientist) → H1 KILLED.
- F4 — wrapper destroys substantive judge signal (AUROC drop > 0.05)
  → H1 KILLED on net-utility.

All four reference observable statistical thresholds, all four can
be checked from experimental data alone (no human judgment loop), and
all four are independent kill-criteria.

### 3. Every mechanism claim has a citation

**Yes.** §3 lists M1–M6, each closed with a "Grounded" or
"Forecast (flagged as such)" tag and at least one resolvable arXiv ID:

- M1 → arXiv:2404.04475, arXiv:2402.07319, arXiv:2410.21819 (Grounded)
- M2 → arXiv:2401.10020, arXiv:2407.19594 (Grounded for training loops)
- M3 → arXiv:2401.10020 + arXiv:2407.19594 + arXiv:2404.04475 +
  arXiv:2402.07319 + arXiv:2410.21819 — but EXPLICITLY FLAGGED AS A
  FORECAST (not direct AI-Scientist-family evidence). arXiv:2503.18102
  and arXiv:2501.04227 cited only as background failure-mode
  documentation, with the explicit caveat (per C1) that they
  document a DIFFERENT exploit mechanism (score-fabrication).
- M4 → arXiv:2505.12843 (Bias Fitting — the actual wrapper)
- M5 → arXiv:2505.12843, arXiv:2404.04475 (Partially Grounded —
  AlpacaEval / RLHF context; bounded by F4 in paper-judge domain)
- M6 → arXiv:2506.11930 (Grounded)

No mechanism claim is left ungrounded. The transfer claim (M3) is
*explicitly* flagged as forecast and tied to F1 as the direct test —
the C1 fix.

### 4. All cited arXiv IDs resolve via `hf_papers paper_details`

**Yes — verified in this revision pass.** Calls made:

| arxiv_id | resolved | title | revision-1 role |
|---|---|---|---|
| 2407.19594 | yes | Meta-Rewarding Language Models | M2 (training-loop magnitude) |
| 2404.04475 | yes | Length-Controlled AlpacaEval | M1, M5 cross-validation (NOT the debiaser, per C2) |
| 2401.10020 | yes | Self-Rewarding Language Models | M2 verbosity blowup |
| 2503.18102 | yes | AgentRxiv | background only — C1 caveat |
| 2501.04227 | yes | Agent Laboratory | background only — C1 caveat |
| 2506.11930 | yes | Feedback Friction | M6 / R1 |
| 2310.01798 | yes | LLMs Cannot Self-Correct Reasoning Yet | flagged not binding |
| 2402.07319 | yes | ODIN | M1 cross-validation |
| 2410.21819 | yes | Self-Preference Bias in LLM-as-a-Judge | M1 cross-validation |
| 2505.12843 | **yes (NEW)** | Bias Fitting | **M4 — the actual debiaser** |
| 2510.18003 | **yes (NEW)** | BadScientist | **§1 sub-dominance, §5 F3 proxies, §7 R6** |
| 2505.17100 | (cited, deferred) | RBD (Reviewer Bias Detection) | downstream-defense pointer in F3 / §7 |
| 2507.02694 | (cited, deferred) | LimitGen | optional sanity-check substrate in §6 |

**Active verification this pass:**
- `paper_details 2510.18003` → confirmed BadScientist (Jiang et al.,
  Oct 2025; 5 fabrication strategies, ICLR 2025 OpenReview
  calibration, o3/o4-mini/GPT-4.1 reviewers).
- `paper_details 2505.12843` → confirmed Bias Fitting (Zhao et al.,
  May 2025; FiMi-RM; non-linear length-encoding ResNet fit on
  (response, scalar reward) pairs).
- `read_paper 2407.19594 section=3` → confirmed Meta-Rewarding's
  length-control is implemented at the DPO-pair-selection stage
  in a training loop (NOT a post-hoc scalar-score wrapper); confirms
  the C2 pivot to Bias Fitting is the right move.
- `read_paper 2505.12843 section=3` → confirmed Bias Fitting's
  scalar-score / non-pairwise / non-linear architecture. M4 is
  grounded in the specific technique described in §3 (length-
  encoding sinusoidal projection → 2-layer ResNet → linear head;
  Pearson + MSE loss against raw reward; debiased reward = raw -
  fitted length-component).
- `read_paper 2510.18003 section=4` → confirmed BadScientist's
  Table 1 acceptance rates (s_1 = 82.0% at τ_0.5, all 5 strategies
  ≥ 49.0%, none length-based), confirmed concern-acceptance
  conflict pattern (Table 2), confirmed ICLR 2025 calibration set
  + GPT-5/o3/o4-mini/GPT-4.1 reviewer ensemble.

All 11 actively-cited arXiv IDs resolve via `paper_details`.
arXiv:2505.17100 and arXiv:2507.02694 are *cited* as downstream
pointers but not relied on for any mechanism claim — they were
flagged by red-team and gap-finder respectively; not re-verified in
this pass to conserve budget.

### 5. "Risks to the hypothesis" section is non-empty

**Yes — 6 risks listed in §7 (was 5 in revision 0; R5 revised + R6
added per I4 + C3).**

- R1 — judge competence ceiling.
- R2 — verbosity-injection contamination.
- R3 — wrapper shifts gaming target (F3 as risk).
- R4 — generalization off calibration set (F4 as risk).
- R5 (REVISED) — exploit already absent in modern judge → reframed
  as configuration-dependent survey result, not falsification.
- R6 (NEW) — BadScientist-dominance: even if S2 mechanism works
  cleanly, the field-impact magnitude is bounded by the fraction of
  LLM-judge variance attributable to length. Synthesist may move S2
  to future-work flag if BadScientist channels dominate.

Each risk lists a contribution-if-it-fails (per audit-trail
discipline rule #1).

### 6. On revisions: every red-team objection has an explicit response

**Yes — §0 of `output.md`** maps each objection from
`red-team-S2/output.md` to a specific resolution:

| Red-team item | Severity | §0 mapping | Resolution |
|---|---|---|---|
| C1 | Critical | §0 C1 | M3 rewritten — transfer claim is forecast, not empirical |
| C2 | Critical | §0 C2 | Pivoted debiaser from Dubois (pairwise) → Bias Fitting (scalar) |
| C3 | Critical | §0 C3 | Dropped "precondition for S3/S4"; cited BadScientist; scoped to sub-dominant channel |
| I1 | Important | §0 I1 | F3 proxy list 5 → 8 (added BadScientist-inspired substantive proxies) |
| I2 | Important | §0 I2 | Dropped entirely per C3 |
| I3 | Important | §0 I3 | Primary prediction = sign + significance, not numerical range |
| I4 | Important | §0 I4 | R5 reframed as graceful no-op / survey artifact |
| S1 | Suggestion | §0 S1 | B3 upgraded to required-if-S1-runs |
| S2 | Suggestion | §0 S2 | LimitGen added as optional sanity-check substrate |

All 3 critical objections addressed with specific structural changes.
All 4 important objections addressed. Both suggestions adopted.

## Spec compliance checks

### Falsifiable with ≥1 non-judge signal (spec requirement)

**Yes.** Primary signals are **regression coefficients β_raw, β_norm,
β_raw − β_norm on log(token-count)** — deterministic statistical
computations on the judge's output scores and token counts. Not an
LLM-judge re-evaluation. Cleanest non-judge falsification surface on
the shortlist (gap-finder-3 §(d) rationale).

Secondary non-judge signals: Spearman ρ between debiased score and
the 8 pre-registered proxies (F3 surface); Hedges' g on
manuscript-pair scores (F4 partial signal).

### File-based artifact discipline (CLAUDE.md rule)

**Yes.** Outputs are three files in
`docs/research/runs/2026-05-12-0515-19bf96/hypothesis-smith-S2/`:
`output.md`, `manifest.yaml`, `verification.md`. No nested dispatch,
no shared in-memory state.

Wrapper integration with the orchestrator: one new field on the
red-team / synthesist `manifest.yaml` (per gap-finder-3 §S2 C2) plus
the Bias Fitting calibration corpus + fitted `model_f` (lightweight
ResNet, CPU-feasible) as a one-time setup artifact. Calibration cost
~$45 (was incorrectly elided as $0 in revision 0; the C2 fix
includes the corrected cost in §6 Budget).

### Stay in lane (CLAUDE.md rule 5)

**Yes.** I produced ONE hypothesis (§2), grounded mechanism (§3),
predicted outcomes (§4), falsification criteria (§5), an experiment
*sketch* (§6 — eval-designer fills in protocol details), risks (§7),
and differential-effect attack pre-emption. I did not produce a full
experimental protocol, dataset construction, or write the
eval-designer's output.

## Revision-1-specific verification

### Three critical objections — final status

**C1 (AgentRxiv §4.1 misrepresentation) — ADDRESSED.**
The revised M3 explicitly:
- Acknowledges AgentRxiv §4.1 documents score-fabrication, not
  verbosity reward hacking.
- Restates M3 as a **transfer forecast** (instruction-following /
  RLHF length bias → AI-Scientist paper-judging) tested directly
  by F1.
- Cites only Self-Rewarding LMs + Meta-Rewarding as the direct
  verbosity-exploit evidence (both training-loop contexts), not
  the misread AgentRxiv §4.1.
- Surfaces F1 falsification as a "configuration-dependent no-op"
  survey result (acceptable contribution form, not a hypothesis
  failure).

**C2 (Dubois implementation gap) — ADDRESSED.**
The wrapper is pivoted from Dubois (pairwise logistic regression on
human preference labels) to **Bias Fitting arXiv:2505.12843**, a
published scalar-score non-linear debiaser. Verified by reading
§3 of the Bias Fitting paper:
- Input: scalar (response, raw reward) pairs.
- Architecture: sinusoidal length-encoding → 2-layer ResNet →
  linear regression head.
- Output: debiased scalar reward = raw - fitted length-component.
- No pairwise human-preference labels required.
This directly maps to MegaResearcher's scalar red-team judge.
Calibration cost is recomputed (~$45, was elided in revision 0).

**C3 (BadScientist sub-dominance) — ADDRESSED.**
- BadScientist arXiv:2510.18003 is cited in §1 (gap statement),
  §5 F3 (substantive proxies inspired by the 5 strategies), and
  §7 R6 (BadScientist-dominance risk).
- "Precondition for S3/S4" framing dropped from §1, §2, §7-Q3.
- S2 is reframed as "the cheapest hardening intervention among
  several published reviewer-exploit fixes, addressing one
  sub-dominant exploit channel."
- §4 explicitly states the field-impact-magnitude bound is the
  fraction of LLM-judge variance attributable to length, which may
  be small per BadScientist's evidence.
- §7 R6 contemplates the synthesist moving S2 to "future-work
  flag" if BadScientist-dominance is confirmed in the broader
  pipeline measurement.

### Important objections — final status

I1: F3 proxy list 5 → 8 with BadScientist-inspired substantive
proxies. **Addressed.**

I2: Precondition language dropped. **Addressed.**

I3: Primary prediction is sign + significance. **Addressed.**

I4: R5 reframed as configuration-dependent survey result.
**Addressed.**

### Suggestions — final status

S1: B3 heterogeneous-model judge baseline upgraded to "required if
S1 runs." **Addressed.**

S2: LimitGen arXiv:2507.02694 added as optional sanity-check
substrate. **Addressed.**

## Self-assessment (per discipline rule #4 + final discipline)

The hypothesis-smith's discipline rule from the task prompt: "If
after the revisions the magnitude drops below something interesting,
say so — that's a usable result and may move S2 to 'future-work
flag.'"

**Self-assessment.** Under revision 1's narrower scope:

- The hypothesis is now **technically defensible** (mechanism is
  grounded, falsification criteria are operationalizable, citations
  are accurate, calibration cost is bounded).
- The field-impact magnitude is **bounded by the BadScientist-
  dominance risk** (R6). If BadScientist's 49-82% non-length
  acceptance figures are representative of the dominant
  AI-Scientist-family failure mode, S2's practical contribution is
  small even when the mechanism works.
- S2 remains worth running because (a) the experimental cost is
  bounded (~$235), (b) the F1 outcome alone is a useful survey
  result (which judges still exhibit length-bias), (c) the F3
  8-proxy catalog produces a downstream-defense pointer regardless
  of which direction it trips, and (d) it tests a published
  technique in a new domain without claiming transformational
  contribution.
- The framing for the synthesist: **S2 is a small-but-clean
  hardening intervention worth running for its calibration value
  and its diagnostic value (F3 proxy catalog), not for its
  expected deployment impact.**

The hypothesis is NOT escalated to "cannot fix" — the three
critical objections are addressed with specific structural
changes, and the narrower scope is defensible.

## Notes on items I did not produce (per spec)

- Eval-designer protocol details: paraphrase generation algorithm,
  judge-model selection, exact venue/manuscript source for the
  20-manuscript test set and ~150-manuscript calibration corpus,
  randomization seed plan. §6 is a sketch only.
- Bias Fitting `model_f` training script. The hypothesis specifies
  the technique (arXiv:2505.12843 §3) but does not include code.
- Cost projection refinements beyond order-of-magnitude. The ~$235
  estimate is a feasibility-pass check; the eval-designer will
  produce a binding budget.
- A "joint defense" hypothesis pairing S2 with a content-fabrication
  detector. That would be a separate hypothesis (or a synthesis-
  phase recommendation). S2 stays in its lane: one published
  length-debiaser, applied in a new domain.
