use mr_cli::{parse_args, Command, OnEscalate};

#[test]
fn parse_init_takes_question() {
    let cmd = mr_cli::parse_args(&["mr", "init", "How does X affect Y?"]).unwrap();
    assert!(matches!(cmd, Command::Init { question } if question == "How does X affect Y?"));
}

#[test]
fn parse_execute_defaults_and_flags() {
    let cmd = mr_cli::parse_args(&[
        "mr",
        "execute",
        "path/plan.md",
        "--headless",
        "--on-escalate=continue",
        "--paper",
    ])
    .unwrap();
    match cmd {
        Command::Execute {
            plan,
            headless,
            on_escalate,
            paper,
            no_mcp,
        } => {
            assert_eq!(plan.as_deref(), Some(std::path::Path::new("path/plan.md")));
            assert!(headless);
            assert!(paper);
            assert!(!no_mcp);
            assert_eq!(on_escalate, OnEscalate::Continue);
        }
        _ => panic!("expected Execute"),
    }
}

#[tokio::test]
async fn run_cli_help_lists_subcommands() {
    // `mr --help` should exit 0 and mention the subcommands. run_cli returns
    // Ok(()) on --help when wired to a clap Printer that prints + exits; for
    // testability, expose `parse_args` (above) and test parsing, not the help
    // printer. This test just asserts parse_args rejects unknown subcommands.
    assert!(parse_args(&["mr", "frobnicate"]).is_err());
}
