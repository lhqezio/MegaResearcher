//! prompt_asset parse/load tests.

use std::path::PathBuf;

use megaresearcher_research::prompt_asset::{load, parse};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/agents")
}

#[test]
fn test_parse_literature_scout() {
    let asset = load(&fixtures().join("literature-scout.md")).unwrap();
    assert_eq!(asset.name, "literature-scout");
    assert_eq!(asset.model, "inherit");
    assert!(asset
        .description
        .contains("Survey prior art for a sub-topic"));
    assert!(asset.description.contains("annotated bibliography"));
    assert!(asset.description.contains("<example>"));
    assert!(
        asset
            .body
            .starts_with("You are a literature scout for MegaResearcher."),
        "body should start with the agent system-prompt opener; got: {:?}",
        asset.body.chars().take(80).collect::<String>()
    );
    assert!(asset.body.contains("## Inputs you receive"));
    assert!(asset.body.contains("## Tools you use"));
    assert!(asset.body.contains("## What to produce"));
    assert!(asset.body.contains("## Discipline rules"));
}

#[test]
fn test_parse_rejects_missing_frontmatter() {
    // No leading "---\n" delimiter.
    let result = parse("just a body, no frontmatter at all");
    assert!(result.is_err(), "must reject a file with no frontmatter");
}

#[test]
fn test_parse_rejects_missing_closing_delimiter() {
    // Opening delimiter but no closing "---".
    let result = parse("---\nname: x\ndescription: y\nmodel: inherit\n\nbody never closes");
    assert!(result.is_err(), "must reject unclosed frontmatter");
}
