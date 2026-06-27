//! 1:1 port of tests/test_scaffold.py.

use std::fs;
use std::path::PathBuf;

use megaresearcher_research::paper_chain::scaffold::scaffold_paper_dir;

fn new_run() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

#[test]
fn test_creates_paper_subdir() {
    let run = new_run();
    let paper = scaffold_paper_dir(run.path()).unwrap();
    assert_eq!(paper, run.path().join("paper"));
    assert!(paper.is_dir());
}

#[test]
fn test_creates_revision_log_jsonl() {
    let run = new_run();
    let paper = scaffold_paper_dir(run.path()).unwrap();
    let log = paper.join("revision-log.jsonl");
    assert!(log.exists(), "revision-log.jsonl should exist");
    assert_eq!(
        fs::read_to_string(&log).unwrap(),
        "",
        "revision-log.jsonl should be empty"
    );
}

#[test]
fn test_idempotent() {
    let run = new_run();
    let p1 = scaffold_paper_dir(run.path()).unwrap();
    fs::write(p1.join("draft-v1.md"), "# draft").unwrap();
    let p2 = scaffold_paper_dir(run.path()).unwrap(); // safe to re-run
    assert_eq!(p1, p2);
    assert!(
        p1.join("draft-v1.md").exists(),
        "Idempotent scaffold must not destroy existing content"
    );
}
