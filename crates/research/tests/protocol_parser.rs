//! 1:1 port of tests/test_protocol_parser.py.

use std::path::PathBuf;

use megaresearcher_research::paper_chain::protocol_parser::parse_protocol;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/protocols")
}

#[test]
fn test_parse_specs_protocol() {
    let result = parse_protocol(&fixtures().join("specs_protocol.md")).unwrap();
    assert_eq!(result.substrate.as_deref(), Some("SPECS-Review-Benchmark"));
    assert_eq!(result.sample_size, Some(22));
    assert_eq!(result.seed, Some(42));
}

#[test]
fn test_parse_baselines_list() {
    let result = parse_protocol(&fixtures().join("specs_protocol.md")).unwrap();
    assert!(result.baselines[0].contains("stage-matched same-family 2-stage"));
}

#[test]
fn test_parse_metric() {
    let result = parse_protocol(&fixtures().join("specs_protocol.md")).unwrap();
    assert!(
        result.metrics.iter().any(|m| m.contains("flaw-detection")),
        "{:?}",
        result.metrics
    );
}

#[test]
fn test_parse_decision_rule() {
    let result = parse_protocol(&fixtures().join("specs_protocol.md")).unwrap();
    assert!(result.decision_rules.len() >= 1);
    assert!(result.decision_rules[0].raw.contains("0.05"));
}

#[test]
fn test_parse_malformed_returns_empty() {
    let result = parse_protocol(&fixtures().join("malformed.md")).unwrap();
    assert!(result.is_empty());
}
