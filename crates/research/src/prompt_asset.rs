//! Worker prompt-asset loader: parse v0 `agents/<name>.md` into (name, description, model, body).
//!
//! The v0 worker agent files are YAML-frontmatter + body markdown:
//!
//! ```text
//! ---
//! name: literature-scout
//! description: |
//!   Survey prior art ...
//! model: inherit
//! ---
//!
//! You are a literature scout for MegaResearcher. ...
//! ```
//!
//! The body (after the closing `---`) is the worker system prompt. The
//! `description` frontmatter field is the orchestrator's routing key.

use std::fs;
use std::io;
use std::path::Path;

use serde::Deserialize;

#[derive(Deserialize)]
struct FrontMatter {
    name: String,
    description: String,
    model: String,
}

/// A parsed worker agent file.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptAsset {
    /// The agent's short id, e.g. `literature-scout`.
    pub name: String,
    /// The full `description:` block scalar — used by the orchestrator to route.
    pub description: String,
    /// The declared model (v0 uses `inherit`).
    pub model: String,
    /// The body after the closing `---` — the worker system prompt.
    pub body: String,
}

/// Parse the raw text of an agent file into a [`PromptAsset`].
///
/// The text must start with a `---` delimiter line and contain a closing
/// `---` delimiter line; the text between them is YAML frontmatter, and the
/// text after the closing delimiter is the body.
pub fn parse(text: &str) -> Result<PromptAsset, String> {
    let after_open = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
        .ok_or_else(|| "file must start with a frontmatter delimiter '---'".to_string())?;

    // The closing delimiter is the first line after the opener that is exactly "---".
    let close_rel = after_open
        .lines()
        .position(|line| line.trim() == "---")
        .ok_or_else(|| "missing closing frontmatter delimiter '---'".to_string())?;

    let frontmatter_text: String = after_open
        .lines()
        .take(close_rel)
        .collect::<Vec<_>>()
        .join("\n");
    let body: String = after_open
        .lines()
        .skip(close_rel + 1)
        .collect::<Vec<_>>()
        .join("\n");

    let front: FrontMatter = serde_yml::from_str(&frontmatter_text)
        .map_err(|e| format!("invalid frontmatter YAML: {e}"))?;

    Ok(PromptAsset {
        name: front.name,
        description: front.description,
        model: front.model,
        body: body.trim().to_string(),
    })
}

/// Load and parse an agent file from disk.
pub fn load(path: &Path) -> io::Result<PromptAsset> {
    let text = fs::read_to_string(path)?;
    parse(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
