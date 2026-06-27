//! 1:1 port of tests/test_finalize.py.

use std::fs;

use megaresearcher_research::paper_chain::finalize::finalize_paper;

/// Build a paper/ dir with draft-v1.md, draft-v2.md (if v2), review-v1.md, log.
fn setup_paper_dir(latest_draft: &str) -> tempfile::TempDir {
    let run = tempfile::tempdir().unwrap();
    let paper = run.path().join("paper");
    fs::create_dir(&paper).unwrap();
    fs::write(paper.join("draft-v1.md"), "# Draft v1\n\nContent.\n").unwrap();
    if latest_draft == "v2" {
        fs::write(paper.join("draft-v2.md"), "# Draft v2\n\nRevised.\n").unwrap();
    }
    fs::write(
        paper.join("review-v1.md"),
        "# Review v1\n\nVERDICT: REVISE\n",
    )
    .unwrap();
    fs::write(
        paper.join("revision-log.jsonl"),
        "{\"round\":1,\"review_point_index\":0,\"addressed\":true,\
         \"change_summary\":\"fixed W1\",\"line_range_modified\":[10,15]}\n",
    )
    .unwrap();
    run
}

#[test]
fn test_finalize_with_v1_only() {
    let run = setup_paper_dir("v1");
    let paper = run.path().join("paper");
    let out = finalize_paper(&paper, "APPROVE").unwrap();
    assert_eq!(out, paper.join("paper.md"));
    assert!(out.exists());
    assert!(fs::read_to_string(&out).unwrap().contains("Draft v1"));
    let history = paper.join("paper-history.md");
    assert!(history.exists());
    assert!(fs::read_to_string(&history).unwrap().contains("Review v1"));
}

#[test]
fn test_finalize_with_v2() {
    let run = setup_paper_dir("v2");
    let paper = run.path().join("paper");
    let out = finalize_paper(&paper, "APPROVE").unwrap();
    assert!(
        fs::read_to_string(&out).unwrap().contains("Draft v2"),
        "paper.md must point at latest draft"
    );
}

#[test]
fn test_finalize_includes_revision_log_in_history() {
    let run = setup_paper_dir("v2");
    let paper = run.path().join("paper");
    finalize_paper(&paper, "APPROVE").unwrap();
    let history = fs::read_to_string(paper.join("paper-history.md")).unwrap();
    assert!(
        history.contains("fixed W1"),
        "revision-log entries must appear in history"
    );
}

#[test]
fn test_finalize_records_final_verdict() {
    let run = setup_paper_dir("v1");
    let paper = run.path().join("paper");
    finalize_paper(&paper, "APPROVE").unwrap();
    let history = fs::read_to_string(paper.join("paper-history.md")).unwrap();
    assert!(history.contains("Final verdict: APPROVE"));
}
