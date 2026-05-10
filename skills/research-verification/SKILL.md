---
name: research-verification
description: Use when about to claim a MegaResearcher swarm run is complete, before reporting back to the user. Wraps superpowers:verification-before-completion with research-specific checks. Verifies every claim is cited, every hypothesis has falsification criteria, the audit trail is non-empty, and the synthesist's output reflects the spec's success criteria and YAGNI fence. Required before the executing-research-plan skill reports "done".
---

# Research Verification

You are about to claim a swarm run is complete. Verify it before saying so.

## Process

**Step 1 — Invoke `superpowers:verification-before-completion`.** Apply its core discipline: evidence before assertions, run the commands, confirm the output, do not declare success on hope.

**Step 2 — Run the research-specific checks below.** All must pass. If any fails, the run is NOT complete — surface the failure to the user and decide together whether to remediate or accept it.

## Required checks (research-specific)

For the run at `docs/research/runs/<run-id>/`:

**A. Run completeness**
- [ ] `output.md` exists at the run root
- [ ] `swarm-state.yaml` exists at the run root
- [ ] Every worker subdir has all three required artifacts: `output.md`, `manifest.yaml`, `verification.md`. List any missing.

**B. Synthesis quality**
- [ ] The final `output.md` has all 8 sections from the synthesist's contract: Executive summary, Surviving hypotheses, Rejected and killed hypotheses (audit trail), Escalations, What we did NOT explore, Recommended next actions, Run metadata, Sources.
- [ ] The "Rejected and killed hypotheses" section is consistent with `swarm-state.yaml` — every kill in state appears in the doc; no hidden rejections.
- [ ] The "What we did NOT explore" section reflects the spec's actual YAGNI fence (cross-reference against `docs/research/specs/<spec>.md`).
- [ ] The "Recommended next actions" section names a specific hypothesis or follow-on question, not "more research is needed."

**C. Hypothesis discipline (if novelty target was `hypothesis`)**
- [ ] Every surviving hypothesis has falsification criteria stated (≥3 each per the hypothesis-smith contract).
- [ ] Every surviving hypothesis has a red-team approval recorded (check the corresponding `red-team-N/manifest.yaml` for `verdict: APPROVE`).
- [ ] Every surviving hypothesis has an eval-designer experimental design.

**D. Citation discipline**
- [ ] Spot-check 3 random citations from the final `output.md` — each must resolve via `mcp__ml-intern__hf_papers paper_details`. Record the spot-checks in your verification report.
- [ ] No invented citations (every cited arxiv ID must resolve).

**E. Success-criteria check**
- [ ] Read the spec's "Success criteria" section. For each criterion, verify the run actually satisfies it. List any unmet criteria explicitly.

**F. Doom-loop check**
- [ ] `swarm-state.yaml` reports zero workers that hit the 3-retry cap without successful completion. If any did, they should be in the Escalations section of the final output.

## Verification report

Write `docs/research/runs/<run-id>/verification-report.md` with:

```markdown
# Verification Report — <run-id>

## Checks
- [x] / [ ] for each check above

## Failures (if any)
<details on each>

## Citation spot-checks
- <arxiv-id>: <claimed claim> — <verified by hf_papers paper_details: yes/no>
- ... (3 spot-checks)

## Verdict
PASS | FAIL | PASS-WITH-CAVEATS

## If FAIL or PASS-WITH-CAVEATS
<what the user should know>
```

## Terminal state

If all checks pass: report "done" to the user with the path to `output.md` and the verification report.

If any fail: surface the failures to the user before claiming completion. Do not paper over failures.

## Discipline rules

- **Evidence before assertions.** If a check is "passes" but you didn't actually run the verifying command, the check has failed.
- **Hidden rejections are a critical failure.** The audit trail discipline is the swarm's main epistemic value — if the synthesist hid rejections, that's a FAIL not a caveat.
- **Spot-checks must be random, not cherry-picked.** Pick the first, middle, and last cited paper in the final document.
