//! 1:1 port of tests/test_regression.py.

use std::fs;
use std::path::PathBuf;

use megaresearcher_research::paper_chain::regression::{detect_regression, extract_weaknesses};

const REVIEW_V1: &str = "\
# Review v1
## Strengths
- Good idea.

## Weaknesses
- W1: Insufficient ablation coverage.
- W2: Citation for claim X does not resolve.
- W3: Method section unclear about step 3.

## Suggested Revisions
...

VERDICT: REVISE
";

const REVIEW_V2_ALL_CLOSED_NEW_PROBLEMS: &str = "\
# Review v2
## Strengths
- Improved.

## Weaknesses
- W4: New citation Y is also unresolved.
- W5: Ablation table has off-by-one error.
- W6: Discussion contradicts results.
- W7: New related-work section misattributes finding.

VERDICT: REVISE
";

const REVIEW_V2_PARTIAL_CLOSE: &str = "\
# Review v2
## Weaknesses
- W2: Citation still unresolved (carried over).
- W4: New small typo in abstract.

VERDICT: REVISE
";

fn write_temp_md(text: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("review.md");
    fs::write(&path, text).unwrap();
    (dir, path)
}

#[test]
fn test_extract_weaknesses_basic() {
    let (_d, p) = write_temp_md(REVIEW_V1);
    let ws = extract_weaknesses(&p).unwrap();
    assert_eq!(ws.len(), 3, "Expected 3 weaknesses, got {:?}", ws);
    assert!(ws[0].starts_with("W1:"), "{}", ws[0]);
}

#[test]
fn test_extract_weaknesses_empty_section() {
    let (_d, p) = write_temp_md("## Weaknesses\n\nVERDICT: APPROVE\n");
    let ws = extract_weaknesses(&p).unwrap();
    assert!(ws.is_empty());
}

#[test]
fn test_regression_fires_when_new_outnumber_closed() {
    let (_d1, v1) = write_temp_md(REVIEW_V1);
    let (_d2, v2) = write_temp_md(REVIEW_V2_ALL_CLOSED_NEW_PROBLEMS);
    let (flagged, closed_count, new_count) = detect_regression(&v1, &v2).unwrap();
    // v1 had 3, v2 has 4 NEW; closed = 3, new = 4 — regression
    assert!(
        flagged,
        "Expected regression flag: closed={}, new={}",
        closed_count, new_count
    );
    assert_eq!(new_count, 4);
    assert_eq!(closed_count, 3);
}

#[test]
fn test_regression_does_not_fire_on_partial_close() {
    let (_d1, v1) = write_temp_md(REVIEW_V1);
    let (_d2, v2) = write_temp_md(REVIEW_V2_PARTIAL_CLOSE);
    // v1 had 3; v2 has 2 (W2 carried, W4 new). closed=2 (W1, W3), new=1 (W4). No regression.
    let (flagged, closed_count, new_count) = detect_regression(&v1, &v2).unwrap();
    assert!(!flagged);
    assert_eq!(closed_count, 2);
    assert_eq!(new_count, 1);
}
