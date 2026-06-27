//! ScopedRead/ScopedWrite jail + check_artifacts tests.

use megaresearcher_research::worker_tools::{check_artifacts, ScopedRead, ScopedWrite, Tool};
use serde_json::json;

#[tokio::test]
async fn test_scoped_write_creates_file_in_dir() {
    let d = tempfile::tempdir().unwrap();
    let w = ScopedWrite::new(d.path());
    let r = w
        .call(json!({"file_path":"output.md","content":"# hi"}))
        .await;
    assert!(!r.is_error, "{}", r.content);
    assert_eq!(
        std::fs::read_to_string(d.path().join("output.md")).unwrap(),
        "# hi"
    );
}

#[tokio::test]
async fn test_scoped_write_rejects_parent_dir_escape() {
    let d = tempfile::tempdir().unwrap();
    let w = ScopedWrite::new(d.path());
    let r = w
        .call(json!({"file_path":"../escape.md","content":"x"}))
        .await;
    assert!(r.is_error);
    assert!(r.content.contains("tool_use_error"), "{}", r.content);
}

#[tokio::test]
async fn test_scoped_write_rejects_absolute_path() {
    let d = tempfile::tempdir().unwrap();
    let w = ScopedWrite::new(d.path());
    let r = w
        .call(json!({"file_path":"/etc/passwd","content":"x"}))
        .await;
    assert!(r.is_error);
    assert!(r.content.contains("tool_use_error"));
}

#[tokio::test]
async fn test_scoped_write_missing_fields_is_error() {
    let d = tempfile::tempdir().unwrap();
    let w = ScopedWrite::new(d.path());
    let r = w.call(json!({"file_path":"x.md"})).await; // missing content
    assert!(r.is_error);
    assert!(r.content.contains("tool_use_error"));
}

#[tokio::test]
async fn test_scoped_read_from_worker_dir() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("a.txt"), "alpha").unwrap();
    let r = ScopedRead::new(d.path())
        .call(json!({"file_path":"a.txt"}))
        .await;
    assert!(!r.is_error, "{}", r.content);
    assert_eq!(r.content, "alpha");
}

#[tokio::test]
async fn test_scoped_read_falls_through_to_shared_dir() {
    let d = tempfile::tempdir().unwrap();
    let shared = tempfile::tempdir().unwrap();
    std::fs::write(shared.path().join("shared.md"), "shared content").unwrap();
    let r = ScopedRead::with_shared(d.path(), shared.path())
        .call(json!({"file_path":"shared.md"}))
        .await;
    assert!(!r.is_error, "{}", r.content);
    assert_eq!(r.content, "shared content");
}

#[tokio::test]
async fn test_scoped_read_rejects_escape() {
    let d = tempfile::tempdir().unwrap();
    let r = ScopedRead::new(d.path())
        .call(json!({"file_path":"../secret"}))
        .await;
    assert!(r.is_error);
    assert!(r.content.contains("tool_use_error"));
}

#[tokio::test]
async fn test_scoped_read_missing_file_is_error() {
    let d = tempfile::tempdir().unwrap();
    let r = ScopedRead::new(d.path())
        .call(json!({"file_path":"nope.md"}))
        .await;
    assert!(r.is_error);
    assert!(r.content.contains("tool_use_error"));
}

#[test]
fn test_check_artifacts_flags_missing_manifest() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("output.md"), "").unwrap();
    std::fs::write(d.path().join("verification.md"), "").unwrap();
    let missing = check_artifacts(d.path(), &["output.md", "manifest.yaml", "verification.md"]);
    assert_eq!(missing, vec!["manifest.yaml".to_string()]);
}

#[test]
fn test_check_artifacts_all_present() {
    let d = tempfile::tempdir().unwrap();
    for name in &["output.md", "manifest.yaml", "verification.md"] {
        std::fs::write(d.path().join(name), "").unwrap();
    }
    assert!(
        check_artifacts(d.path(), &["output.md", "manifest.yaml", "verification.md"]).is_empty()
    );
}
