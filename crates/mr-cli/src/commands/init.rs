//! `mr init "<question>"` — brainstorm → spec → plan as ONE continuous guided
//! session. The three flow bodies are concatenated with section dividers into a
//! single guiding prompt; the session gates on the spec artifact then the plan
//! artifact (both Rust-enforced: the file must exist before approval is
//! accepted).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;
use claurst_api::LlmProvider;
use megaresearcher_research::flows::load_embedded;
use megaresearcher_research::phases::{drive_session, DriveOutcome, Gate, GuidedSession, UserIo};
use megaresearcher_research::worker_tools::{ScopedRead, ScopedWrite, Tool};

use crate::io::StdinStdoutIo;

fn docs_root(cwd: &Path) -> PathBuf {
    cwd.join("docs").join("research")
}

fn slug(topic: &str) -> String {
    topic
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn date_prefix() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}

/// Drive the init chain (brainstorm → spec → plan) with an injected `UserIo`
/// (test seam). Gates: spec then plan.
pub async fn run_with(
    cwd: &Path,
    provider: (Arc<dyn LlmProvider>, String),
    question: &str,
    io: &dyn UserIo,
) -> anyhow::Result<()> {
    let docs = docs_root(cwd);
    std::fs::create_dir_all(docs.join("specs")).ok();
    std::fs::create_dir_all(docs.join("plans")).ok();

    // Concatenate the three flow bodies into one guiding prompt.
    let body = format!(
        "# Phase 1 — Brainstorm\n\n{}\n\n# Phase 2 — Spec\n\n{}\n\n# Phase 3 — Plan\n\n{}\n\n\
         Drive all three phases in order. Pause for user approval after the spec and again after the plan.",
        load_embedded("brainstorm").body,
        load_embedded("spec").body,
        load_embedded("plan").body,
    );

    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ScopedRead::with_shared(docs.clone(), docs.clone())),
        Arc::new(ScopedWrite::new(docs.clone())),
    ];
    let (p, model) = provider;
    let mut session = GuidedSession::new(body, tools, p, model, 4096, 60);
    session.inject_user(&format!("Topic: {question}"));

    let spec_path =
        docs.join("specs")
            .join(format!("{}-{}-spec.md", date_prefix(), slug(question)));
    let plan_path =
        docs.join("plans")
            .join(format!("{}-{}-plan.md", date_prefix(), slug(question)));
    let gates = vec![
        Gate {
            artifact: spec_path.clone(),
            label: "spec".into(),
        },
        Gate {
            artifact: plan_path.clone(),
            label: "plan".into(),
        },
    ];
    let outcome = drive_session(&mut session, io, gates, &["approve", "yes", "y", "done"])
        .await
        .context("guided session failed")?;
    match outcome {
        DriveOutcome::Approved { .. } => {
            io.print(&format!(
                "\nSpec: {}\nPlan: {}\nRun: `mr execute {}`\n",
                spec_path.display(),
                plan_path.display(),
                plan_path.display()
            ))
            .await?;
        }
        DriveOutcome::MaxTurns => {
            io.print("\nHit the turn ceiling before both approvals.\n")
                .await?;
        }
    }
    Ok(())
}

/// Production entry: drive the init chain over stdin/stdout.
pub async fn run(
    cwd: &Path,
    provider: (Arc<dyn LlmProvider>, String),
    question: &str,
) -> anyhow::Result<()> {
    run_with(cwd, provider, question, &StdinStdoutIo).await
}
