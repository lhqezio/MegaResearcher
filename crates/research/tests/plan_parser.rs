//! parse_plan tests: extract scout/gap-finder assignments + novelty target.

use megaresearcher_research::orchestrator::dispatch_plan::{parse_plan, NoveltyTarget};

const PLAN: &str = "\
---
novelty_target: gap-finding
---

## Phase 1 — literature-scout dispatches

### Cross-attention fusion
Survey 2024–2026 work on cross-attention fusion of EO and SAR imagery.

### Temporal coherence
Survey work on temporal coherence in SAR.

## Phase 2 — gap-finder dispatches

### Sub-topics 1–2
Read the consolidated bibliography for sub-topics 1–2 and identify gaps.
";

#[test]
fn parses_novelty_target_and_assignments() {
    let p = parse_plan(PLAN).unwrap();
    assert_eq!(p.novelty_target, NoveltyTarget::GapFinding);

    assert_eq!(p.scouts.len(), 2);
    assert_eq!(p.scouts[0].id, "literature-scout-1");
    assert_eq!(p.scouts[0].role, "literature-scout");
    assert_eq!(p.scouts[0].title, "Cross-attention fusion");
    assert_eq!(
        p.scouts[0].body,
        "Survey 2024–2026 work on cross-attention fusion of EO and SAR imagery."
    );
    assert_eq!(p.scouts[1].id, "literature-scout-2");
    assert_eq!(p.scouts[1].title, "Temporal coherence");

    assert_eq!(p.gap_finders.len(), 1);
    assert_eq!(p.gap_finders[0].id, "gap-finder-1");
    assert_eq!(p.gap_finders[0].role, "gap-finder");
    assert_eq!(p.gap_finders[0].title, "Sub-topics 1–2");
    assert_eq!(
        p.gap_finders[0].body,
        "Read the consolidated bibliography for sub-topics 1–2 and identify gaps."
    );
}

#[test]
fn hypothesis_target_round_trips() {
    let p = parse_plan("---\nnovelty_target: hypothesis\n---\n## Phase 1 — literature-scout dispatches\n\n### T\nbody\n").unwrap();
    assert_eq!(p.novelty_target, NoveltyTarget::Hypothesis);
    assert!(!p.novelty_target.skips_critique_phases());
    assert_eq!(p.scouts.len(), 1);
}

#[test]
fn empty_sections_are_ok() {
    let p = parse_plan("---\nnovelty_target: gap-finding\n---\n## Phase 1 — literature-scout dispatches\n\n## Phase 2 — gap-finder dispatches\n").unwrap();
    assert!(p.scouts.is_empty());
    assert!(p.gap_finders.is_empty());
}

#[test]
fn unknown_novelty_target_is_error() {
    let err = parse_plan("---\nnovelty_target: bogus\n---\n").unwrap_err();
    assert!(err.contains("novelty_target"), "err was: {err}");
}

#[test]
fn missing_frontmatter_is_error() {
    assert!(parse_plan("no frontmatter here").is_err());
}