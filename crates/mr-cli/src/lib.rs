pub mod commands;
pub mod escalation;
pub mod io;
pub mod prelude;
pub mod render;

use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnEscalate {
    Continue,
    Pause,
    Fail,
}

#[derive(Debug, Clone)]
pub enum Command {
    Init {
        question: String,
    },
    Brainstorm {
        topic: String,
    },
    Spec {
        topic: String,
    },
    Plan {
        topic: String,
    },
    Execute {
        plan: Option<PathBuf>,
        paper: bool,
        headless: bool,
        no_mcp: bool,
        on_escalate: OnEscalate,
    },
    Verify {
        run_dir: PathBuf,
    },
    Watch {
        run_dir: Option<PathBuf>,
    },
    List,
}

/// Parse `mr` args (the program name is `args[0]`). Errors on unknown subcommands
/// or missing required arguments.
pub fn parse_args(args: &[&str]) -> anyhow::Result<Command> {
    let rest = &args[1..];
    if rest.is_empty() {
        return Ok(Command::List);
    }
    match rest[0] {
        "init" => Ok(Command::Init {
            question: rest
                .get(1)
                .map(|s| s.to_string())
                .ok_or_else(|| anyhow::anyhow!("init requires a question"))?,
        }),
        "brainstorm" => Ok(Command::Brainstorm {
            topic: rest.get(1).map(|s| s.to_string()).unwrap_or_default(),
        }),
        "spec" => Ok(Command::Spec {
            topic: rest.get(1).map(|s| s.to_string()).unwrap_or_default(),
        }),
        "plan" => Ok(Command::Plan {
            topic: rest.get(1).map(|s| s.to_string()).unwrap_or_default(),
        }),
        "execute" => parse_execute(&rest[1..]),
        "verify" => Ok(Command::Verify {
            run_dir: PathBuf::from(
                rest.get(1)
                    .ok_or_else(|| anyhow::anyhow!("verify requires a run dir"))?,
            ),
        }),
        "watch" => Ok(Command::Watch {
            run_dir: rest.get(1).map(PathBuf::from),
        }),
        "list" => Ok(Command::List),
        other => Err(anyhow::anyhow!("unknown subcommand: {other}")),
    }
}

fn parse_execute(args: &[&str]) -> anyhow::Result<Command> {
    let mut plan: Option<PathBuf> = None;
    let mut paper = false;
    let mut headless = false;
    let mut no_mcp = false;
    let mut on_escalate = OnEscalate::Fail;
    for a in args {
        match *a {
            "--paper" => paper = true,
            "--headless" => headless = true,
            "--no-mcp" => no_mcp = true,
            s if s.starts_with("--on-escalate=") => {
                on_escalate = match s.trim_start_matches("--on-escalate=") {
                    "continue" => OnEscalate::Continue,
                    "pause" => OnEscalate::Pause,
                    "fail" => OnEscalate::Fail,
                    other => anyhow::bail!("bad --on-escalate value: {other}"),
                };
            }
            s if s.starts_with("--") => anyhow::bail!("unknown flag: {s}"),
            other => plan = Some(PathBuf::from(other)),
        }
    }
    Ok(Command::Execute {
        plan,
        paper,
        headless,
        no_mcp,
        on_escalate,
    })
}

/// Top-level usage string. Lists subcommands with one-line descriptions.
/// Descriptions for `brainstorm`/`spec`/`plan` come from the embedded flow
/// assets' `description` frontmatter; the rest are static one-liners.
pub fn usage() -> String {
    use megaresearcher_research::flows::load_embedded;
    let brain = load_embedded("brainstorm").description;
    let spec = load_embedded("spec").description;
    let plan = load_embedded("plan").description;
    format!(
        "mr — MegaResearcher research-swarm CLI\n\n\
         Usage: mr <subcommand> [args]\n\n\
         Subcommands:\n  \
         init <question>       Brainstorm → spec → plan in one session\n  \
         brainstorm <topic>    {brain}\n  \
         spec <topic>          {spec}\n  \
         plan <topic>          {plan}\n  \
         execute [plan]        Run the swarm on a plan (flags: --paper --headless --no-mcp --on-escalate=continue|pause|fail)\n  \
         verify <run-dir>     Re-run the deterministic post-run checker on a completed run\n  \
         watch [run-dir]       (TUI arrives in Phase 6b)\n  \
         list                  List past runs under docs/research/runs/\n\n\
         Flags: --help / -h shows this message.\n"
    )
}

/// Entry point. Parses args and dispatches. Intercepts `--help`/`-h`/`help`
/// before `parse_args` so help works without an API key or provider.
pub async fn run_cli(args: Vec<String>) -> anyhow::Result<()> {
    use anyhow::Context as _;
    let args_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    // args[0] is the program name; the subcommand slot is args[1].
    if args_refs
        .get(1)
        .is_some_and(|s| matches!(*s, "--help" | "-h" | "help"))
    {
        println!("{}", usage());
        return Ok(());
    }
    let cmd = parse_args(&args_refs).context("bad args")?;
    let cwd = std::env::current_dir()?;
    let provider = prelude::resolve_provider(&cwd, None, None, None)
        .await
        .context("could not resolve a provider — set an API key (see claurst auth)")?;
    commands::dispatch(cmd, cwd, provider).await
}
