//! Front-half guided-session flow bodies: frontmatter + body markdown assets,
//! embedded into the binary via `include_str!`. Mirrors `prompt_asset.rs`'s
//! parser shape but with flow-specific frontmatter (description, argument-hint,
//! model, allowed-tools). The body is the guiding prompt for a `GuidedSession`
//! (the first user message). See design §8/§180/§182.

use std::fs;
use std::io;
use std::path::Path;

use serde::Deserialize;

/// The names of the flow bodies embedded into the binary.
pub const EMBEDDED_NAMES: &[&str] = &["brainstorm", "spec", "plan"];

const BRAINSTORM_MD: &str = include_str!("flows/brainstorm.md");
const SPEC_MD: &str = include_str!("flows/spec.md");
const PLAN_MD: &str = include_str!("flows/plan.md");

#[derive(Deserialize)]
struct FrontMatter {
    name: String,
    description: String,
    #[serde(default, rename = "argument-hint")]
    argument_hint: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default, rename = "allowed-tools")]
    allowed_tools: Option<Vec<String>>,
}

/// A parsed flow-body asset.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowAsset {
    pub name: String,
    pub description: String,
    pub argument_hint: Option<String>,
    pub model: Option<String>,
    pub allowed_tools: Option<Vec<String>>,
    pub body: String,
}

/// Parse raw text into a [`FlowAsset`]. Text must start with a `---` line and
/// contain a closing `---` line; between them is YAML frontmatter, after is the
/// body.
pub fn parse(text: &str) -> Result<FlowAsset, String> {
    let after_open = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
        .ok_or_else(|| "file must start with a frontmatter delimiter '---'".to_string())?;
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
    Ok(FlowAsset {
        name: front.name,
        description: front.description,
        argument_hint: front.argument_hint,
        model: front.model,
        allowed_tools: front.allowed_tools,
        body: body.trim().to_string(),
    })
}

/// Load and parse a flow body from disk.
pub fn load(path: &Path) -> io::Result<FlowAsset> {
    let text = fs::read_to_string(path)?;
    parse(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Load an embedded flow body by name (`"brainstorm"`, `"spec"`, `"plan"`).
/// Panics on an unknown name — only call with literals.
pub fn load_embedded(name: &str) -> FlowAsset {
    let text = match name {
        "brainstorm" => BRAINSTORM_MD,
        "spec" => SPEC_MD,
        "plan" => PLAN_MD,
        other => panic!("unknown embedded flow: {other}"),
    };
    parse(text).unwrap_or_else(|e| panic!("embedded flow {name} failed to parse: {e}"))
}
