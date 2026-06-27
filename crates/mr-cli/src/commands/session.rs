//! `mr brainstorm`/`mr spec`/`mr plan` — drive a single embedded flow body as a
//! guided session with approval gates enforced by `drive_session`.
//!
//! The flow body is seeded as the first user message (byte-identical to the
//! asset), then `"Topic: <topic>"` is injected as the second user message so
//! the topic reaches the model without mutating the asset body. `ScopedWrite`
//! is jailed to `cwd/docs/research/`; the gate artifact path is computed under
//! that same dir, and the scripted/model-emitted `file_path` is relative to
//! the jail (e.g. `specs/<date>-<slug>-spec.md`).

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
    let s: String = topic
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    s.trim_matches('-').to_string()
}

fn date_prefix() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}

/// Drive a single flow-body session with an injected `UserIo` (test seam).
/// `name` is one of `"brainstorm"`, `"spec"`, `"plan"`.
pub async fn run_session_with(
    cwd: &Path,
    provider: (Arc<dyn LlmProvider>, String),
    name: &str,
    topic: &str,
    io: &dyn UserIo,
) -> anyhow::Result<()> {
    let asset = load_embedded(name);
    let docs = docs_root(cwd);
    std::fs::create_dir_all(docs.join("specs")).ok();
    std::fs::create_dir_all(docs.join("plans")).ok();

    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ScopedRead::with_shared(docs.clone(), docs.clone())),
        Arc::new(ScopedWrite::new(docs.clone())),
    ];
    let (p, model) = provider;
    let mut session = GuidedSession::new(&asset.body, tools, p, model, 4096, 30);
    session.inject_user(&format!("Topic: {topic}"));

    let gates: Vec<Gate> = match name {
        "brainstorm" => vec![],
        "spec" => vec![Gate {
            artifact: docs
                .join("specs")
                .join(format!("{}-{}-spec.md", date_prefix(), slug(topic))),
            label: "spec".into(),
        }],
        "plan" => vec![Gate {
            artifact: docs
                .join("plans")
                .join(format!("{}-{}-plan.md", date_prefix(), slug(topic))),
            label: "plan".into(),
        }],
        other => anyhow::bail!("unknown session: {other}"),
    };
    let outcome = drive_session(&mut session, io, gates, &["approve", "yes", "y", "done"])
        .await
        .context("guided session failed")?;
    match outcome {
        DriveOutcome::Approved { .. } => {
            io.print("\nSession approved.\n").await?;
            if name == "plan" {
                io.print(&format!(
                    "Run: `mr execute {}`\n",
                    docs.join("plans").display()
                ))
                .await?;
            }
        }
        DriveOutcome::MaxTurns => {
            io.print("\nSession hit the turn ceiling before approval.\n")
                .await?;
        }
    }
    Ok(())
}

/// Production entry: drive a single flow-body session over stdin/stdout.
pub async fn run_session(
    cwd: &Path,
    provider: (Arc<dyn LlmProvider>, String),
    flow: &str,
    topic: &str,
) -> anyhow::Result<()> {
    run_session_with(cwd, provider, flow, topic, &StdinStdoutIo).await
}
