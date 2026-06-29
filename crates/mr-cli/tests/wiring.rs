//! Verify the no-args → TUI decision and the lazy-resolution path without
//! launching a live terminal (test the dispatch decision via a seam).

use mr_cli::Command;

#[test]
fn parse_args_empty_is_list_not_tui() {
    // The TUI launch happens at run_cli, not parse_args. parse_args still
    // returns List for empty args (run_cli intercepts empty BEFORE parse_args).
    let cmd = mr_cli::parse_args(&["mr"]).unwrap();
    assert!(matches!(cmd, Command::List));
}

#[tokio::test]
async fn verify_list_watch_skip_provider_resolution() {
    // The lazy path: dispatch with Option::None provider for List/Watch/Verify
    // must not error. We test the dispatch signature, not a live run.
    // (Full e2e is T14.)
    let cwd = std::env::temp_dir();
    // List with no provider should succeed (it never needed one).
    let res = mr_cli::commands::dispatch(Command::List, cwd.clone(), None).await;
    // It may error if there are no runs dir, but it must NOT error with
    // "could not resolve a provider" — that's the regression we're fixing.
    if let Err(e) = &res {
        let msg = format!("{e}");
        assert!(
            !msg.contains("resolve a provider"),
            "lazy resolution leaked: {msg}"
        );
    }
}
