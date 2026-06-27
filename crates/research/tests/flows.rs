use megaresearcher_research::flows::{load_embedded, parse, EMBEDDED_NAMES};

#[test]
fn parse_reads_frontmatter_and_body() {
    let text = "---\nname: brainstorm\ndescription: Clarify intent.\nargument-hint: \"[topic]\"\nmodel: inherit\nallowed-tools:\n  - Read\n  - Write\n---\n\nYou are guiding a brainstorm.\n";
    let a = parse(text).unwrap();
    assert_eq!(a.name, "brainstorm");
    assert_eq!(a.description, "Clarify intent.");
    assert_eq!(a.argument_hint.as_deref(), Some("[topic]"));
    assert_eq!(a.model.as_deref(), Some("inherit"));
    assert_eq!(
        a.allowed_tools.as_deref(),
        Some(&["Read".to_string(), "Write".to_string()][..])
    );
    assert!(a.body.contains("You are guiding a brainstorm."));
}

#[test]
fn parse_tolerates_missing_optional_fields() {
    let text = "---\nname: spec\ndescription: Write the spec.\n---\n\nBody.\n";
    let a = parse(text).unwrap();
    assert_eq!(a.name, "spec");
    assert!(a.argument_hint.is_none());
    assert!(a.model.is_none());
    assert!(a.allowed_tools.is_none());
}

#[test]
fn parse_rejects_missing_frontmatter_delimiter() {
    assert!(parse("no frontmatter here").is_err());
}

#[test]
fn load_embedded_returns_three_known_flows() {
    assert_eq!(EMBEDDED_NAMES, &["brainstorm", "spec", "plan"]);
    for name in EMBEDDED_NAMES {
        let a = load_embedded(name);
        assert_eq!(
            a.name, *name,
            "embedded flow {name} must have name frontmatter"
        );
        assert!(
            !a.body.is_empty(),
            "embedded flow {name} body must be non-empty"
        );
    }
}

#[test]
fn brainstorm_body_inlines_clarifications_and_no_skill_invocation() {
    let a = load_embedded("brainstorm");
    // The ported body must carry the 7 research clarifications and must NOT
    // reference the superpowers:brainstorming skill (it is inlined).
    assert!(a.body.contains("novelty target"));
    assert!(a.body.contains("YAGNI"));
    assert!(!a.body.contains("superpowers:"));
}

#[test]
fn spec_body_has_the_fixed_section_template() {
    let a = load_embedded("spec");
    assert!(a.body.contains("Novelty target"));
    assert!(a.body.contains("Out of scope"));
    assert!(a.body.contains("docs/research/specs/"));
}

#[test]
fn plan_body_has_swarm_decomposition_section() {
    let a = load_embedded("plan");
    assert!(a.body.contains("Swarm decomposition"));
    assert!(a.body.contains("MEGARESEARCHER_MAX_PARALLEL"));
    assert!(a.body.contains("docs/research/plans/"));
}
