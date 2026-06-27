//! 1:1 port of tests/test_preflight.py.

use std::env;
use std::fs;

use megaresearcher_research::paper_chain::preflight::{
    preflight_check, preflight_check_with_paper,
};

/// Create a temporary run dir matching the swarm-state.yaml shape.
fn make_run(
    novelty_target: Option<&str>,
    with_output: bool,
    with_eval_designers: usize,
) -> tempfile::TempDir {
    let run = tempfile::tempdir().unwrap();
    if with_output {
        fs::write(run.path().join("output.md"), "# Research direction\n").unwrap();
    }
    if let Some(nt) = novelty_target {
        fs::write(
            run.path().join("swarm-state.yaml"),
            format!("novelty_target: {}\n", nt),
        )
        .unwrap();
    }
    for i in 0..with_eval_designers {
        let d = run.path().join(format!("eval-designer-S{}", i + 1));
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("output.md"), format!("# Eval design {}\n", i + 1)).unwrap();
    }
    run
}

#[test]
fn test_happy_path() {
    let run = make_run(Some("hypothesis"), true, 3);
    let (ok, reason) = preflight_check(run.path()).unwrap();
    assert!(ok, "Expected OK, got refusal: {}", reason);
}

#[test]
fn test_missing_output_md() {
    let run = make_run(Some("hypothesis"), false, 3);
    let (ok, reason) = preflight_check(run.path()).unwrap();
    assert!(!ok);
    assert!(
        reason.contains("output.md"),
        "Expected reason to name output.md; got: {}",
        reason
    );
}

#[test]
fn test_missing_swarm_state() {
    let run = make_run(None, true, 3);
    let (ok, reason) = preflight_check(run.path()).unwrap();
    assert!(!ok);
    assert!(reason.contains("swarm-state"));
}

#[test]
fn test_wrong_novelty_target_gap_finding() {
    let run = make_run(Some("gap-finding"), true, 0);
    let (ok, reason) = preflight_check(run.path()).unwrap();
    assert!(!ok);
    assert!(reason.contains("hypothesis") && reason.contains("gap-finding"));
}

#[test]
fn test_no_eval_designer_outputs() {
    let run = make_run(Some("hypothesis"), true, 0);
    let (ok, reason) = preflight_check(run.path()).unwrap();
    assert!(!ok);
    assert!(reason.contains("eval-designer"));
}

#[test]
fn test_preflight_warns_about_vercel_token_when_paper() {
    // When --paper is set and VERCEL_TOKEN absent, preflight returns ok=true
    // with a non-empty warnings list. The warning does not block.
    let run = make_run(Some("hypothesis"), true, 3);
    let saved = env::var_os("VERCEL_TOKEN");
    env::remove_var("VERCEL_TOKEN");
    let (ok, _reason, warnings) = preflight_check_with_paper(run.path(), true).unwrap();
    assert!(ok);
    assert!(warnings.iter().any(|w| w.contains("VERCEL_TOKEN")));
    if let Some(v) = saved {
        env::set_var("VERCEL_TOKEN", v);
    }
}
