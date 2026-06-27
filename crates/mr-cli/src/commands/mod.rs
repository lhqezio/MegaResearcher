use std::path::PathBuf;
use std::sync::Arc;

use claurst_api::LlmProvider;

use crate::Command;

pub async fn dispatch(
    cmd: Command,
    cwd: PathBuf,
    provider: (Arc<dyn LlmProvider>, String),
) -> anyhow::Result<()> {
    match cmd {
        Command::List => crate::render::list_runs(&cwd).await,
        Command::Watch { run_dir } => crate::render::watch(&cwd, run_dir).await,
        Command::Init { question } => init::run(&cwd, provider, &question).await,
        Command::Brainstorm { topic } => {
            session::run_session(&cwd, provider, "brainstorm", &topic).await
        }
        Command::Spec { topic } => session::run_session(&cwd, provider, "spec", &topic).await,
        Command::Plan { topic } => session::run_session(&cwd, provider, "plan", &topic).await,
        Command::Execute {
            plan,
            paper,
            headless,
            no_mcp,
            on_escalate,
        } => execute::run(&cwd, provider, plan, paper, headless, no_mcp, on_escalate).await,
        Command::Verify { run_dir } => verify::run(&cwd, provider, run_dir).await,
    }
}

pub mod execute;
pub mod init;
pub mod session;
pub mod verify;
