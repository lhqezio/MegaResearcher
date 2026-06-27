//! Scoped worker tools: jailed Read/Write + the three-artifact presence check.
//!
//! The hard path jail (spec §7) rejects absolute paths and `..` parent-dir
//! components before any I/O. Write is jailed to the worker output dir; Read
//! may also read a shared research dir. Tool errors are wrapped in
//! `<tool_use_error>...</tool_use_error>`.

use std::fs;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::Value;

/// The outcome of a tool call: the text content to place in the `tool_result`
/// block, and whether it is an error.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }
    pub fn err(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// Wrap a message in the spec §7 `<tool_use_error>` envelope.
fn tool_use_error(message: impl Into<String>) -> String {
    format!("<tool_use_error>{}</tool_use_error>", message.into())
}

/// A worker tool. Phase 3 ships the scoped file-I/O tools; Phase 5 adds async
/// MCP tools on the same trait.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> Value;
    async fn call(&self, input: Value) -> ToolResult;
    /// Safe to run concurrently with other tools. Read-only tools return true.
    fn is_concurrency_safe(&self) -> bool {
        false
    }
    fn is_read_only(&self) -> bool {
        false
    }
}

/// Hard path jail: resolve `file_path` under `root`, rejecting absolute paths
/// and any `..` parent-dir component. Returns the joined path on success.
pub fn jail_under(root: &Path, file_path: &str) -> Result<PathBuf, String> {
    let p = Path::new(file_path);
    if p.is_absolute() {
        return Err(format!("absolute paths are not allowed: {file_path}"));
    }
    if p.components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(format!(
            "parent-dir (..) components are not allowed: {file_path}"
        ));
    }
    Ok(root.join(file_path))
}

/// Try to read `file_path` under `root`. Returns:
/// - `Some(Ok(text))` if the file exists and reads cleanly,
/// - `Some(Err(msg))` if the jail rejects the path or the read fails,
/// - `None` if the file does not exist under `root` (so the caller can fall
///   through to a shared dir).
fn read_under(root: &Path, file_path: &str) -> Option<Result<String, String>> {
    match jail_under(root, file_path) {
        Ok(p) => {
            if !p.exists() {
                return None;
            }
            Some(fs::read_to_string(&p).map_err(|e| format!("read failed: {e}")))
        }
        Err(msg) => Some(Err(msg)),
    }
}

/// A `Read` tool scoped to the worker output dir, optionally with a shared
/// research dir for read-only access to shared artifacts.
pub struct ScopedRead {
    dir: PathBuf,
    shared: Option<PathBuf>,
}

impl ScopedRead {
    /// Read only from the worker output dir.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            shared: None,
        }
    }
    /// Read from the worker dir first, then fall through to `shared`.
    pub fn with_shared(dir: impl Into<PathBuf>, shared: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            shared: Some(shared.into()),
        }
    }
}

#[async_trait]
impl Tool for ScopedRead {
    fn name(&self) -> &str {
        "Read"
    }
    fn description(&self) -> &str {
        "Read a file under the worker output dir (or the shared research dir)."
    }
    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string" }
            },
            "required": ["file_path"]
        })
    }
    async fn call(&self, input: Value) -> ToolResult {
        let file_path = match input.get("file_path").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::err(tool_use_error("missing 'file_path' string field")),
        };
        if let Some(result) = read_under(&self.dir, file_path) {
            return match result {
                Ok(text) => ToolResult::ok(text),
                Err(msg) => ToolResult::err(tool_use_error(msg)),
            };
        }
        if let Some(shared) = &self.shared {
            if let Some(result) = read_under(shared, file_path) {
                return match result {
                    Ok(text) => ToolResult::ok(text),
                    Err(msg) => ToolResult::err(tool_use_error(msg)),
                };
            }
        }
        ToolResult::err(tool_use_error(format!("file not found: {file_path}")))
    }
    fn is_concurrency_safe(&self) -> bool {
        true
    }
    fn is_read_only(&self) -> bool {
        true
    }
}

/// A `Write` tool jailed to the worker output dir.
pub struct ScopedWrite {
    dir: PathBuf,
}

impl ScopedWrite {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }
}

#[async_trait]
impl Tool for ScopedWrite {
    fn name(&self) -> &str {
        "Write"
    }
    fn description(&self) -> &str {
        "Write a file under the worker output dir."
    }
    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "Path relative to the worker output dir." },
                "content": { "type": "string" }
            },
            "required": ["file_path", "content"]
        })
    }
    async fn call(&self, input: Value) -> ToolResult {
        let file_path = match input.get("file_path").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::err(tool_use_error("missing 'file_path' string field")),
        };
        let content = match input.get("content").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::err(tool_use_error("missing 'content' string field")),
        };
        match jail_under(&self.dir, file_path) {
            Ok(p) => {
                let parent = p.parent().unwrap_or(Path::new(""));
                if let Err(e) = fs::create_dir_all(parent).and_then(|_| fs::write(&p, content)) {
                    ToolResult::err(tool_use_error(format!("write failed: {e}")))
                } else {
                    ToolResult::ok(format!("wrote {file_path} ({} bytes)", content.len()))
                }
            }
            Err(msg) => ToolResult::err(tool_use_error(msg)),
        }
    }
    fn is_read_only(&self) -> bool {
        false
    }
}

/// Return the names from `expected` whose files do not exist under `dir`.
/// The orchestrator's verification gate (spec §11): a non-empty result means the
/// worker's three-artifact contract is unmet.
pub fn check_artifacts(dir: &Path, expected: &[&str]) -> Vec<String> {
    expected
        .iter()
        .filter(|name| !dir.join(name).exists())
        .map(|s| s.to_string())
        .collect()
}
