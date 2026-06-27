//! 1:1 port of tests/test_verdict_parser.py.

use std::fs;
use std::path::PathBuf;

use megaresearcher_research::paper_chain::verdict::parse_verdict;

fn write_temp_md(text: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("review.md");
    fs::write(&path, text).unwrap();
    (dir, path)
}

#[test]
fn test_approve() {
    let (_d, p) = write_temp_md("# Review\n\nSummary...\n\nVERDICT: APPROVE\n");
    assert_eq!(parse_verdict(&p).unwrap(), Some("APPROVE".to_string()));
}

#[test]
fn test_revise() {
    let (_d, p) = write_temp_md("# Review\nVERDICT: REVISE\n");
    assert_eq!(parse_verdict(&p).unwrap(), Some("REVISE".to_string()));
}

#[test]
fn test_kill() {
    let (_d, p) = write_temp_md("# Review\nVERDICT: KILL\n");
    assert_eq!(parse_verdict(&p).unwrap(), Some("KILL".to_string()));
}

#[test]
fn test_verdict_must_be_last_nonblank_line() {
    // Verdict line not at end → still parsed (we scan, not strict-last)
    let (_d, p) = write_temp_md("# Review\nVERDICT: APPROVE\n\nSome trailing notes.\n");
    assert_eq!(parse_verdict(&p).unwrap(), Some("APPROVE".to_string()));
}

#[test]
fn test_no_verdict() {
    let (_d, p) = write_temp_md("# Review\n\nNo verdict here.\n");
    assert_eq!(parse_verdict(&p).unwrap(), None);
}

#[test]
fn test_malformed_verdict() {
    let (_d, p) = write_temp_md("# Review\nVERDICT: MAYBE\n");
    assert_eq!(parse_verdict(&p).unwrap(), None);
}

#[test]
fn test_case_sensitivity() {
    let (_d, p) = write_temp_md("# Review\nverdict: approve\n");
    assert_eq!(parse_verdict(&p).unwrap(), None);
}
