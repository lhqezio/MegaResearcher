# MegaResearcher-rs Phase 5 — MCP Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the research workers to the Python `ml-intern` MCP server subprocess so workers can actually call `hf_papers` and its 8 siblings, via claurst's `McpClient`, with a uv pre-warm and a 9-tool smoke test.

**Architecture:** A new `crates/research/src/mcp.rs` module exposes each server tool as a research `worker_tools::Tool` under the `mcp__<server>__<tool>` name (the Claude-Code convention the byte-identical agent fixtures already use — `mcp__ml-intern__hf_papers`). A research-owned `McpCaller` trait is the seam: the production impl wraps a connected `claurst_mcp::McpClient`; tests supply a `FakeMcpCaller`. (We do NOT use claurst's `McpManager`/`McpToolWrapper`, which produce `<server>_<tool>` single-underscore names that clash with the fixtures, and we do NOT touch claurst's private `McpClient::from_backend`.) `McpToolSet::connect` pre-warms the uv venv (`uv sync --project <mcp>`) then connects the stdio subprocess. The orchestrator connects in pre-flight (per design §202) and threads `extra_tools: &[Arc<dyn Tool>]` through every wave/phase function; `&[]` is the no-MCP default so the existing fake-provider integration tests stay byte-identical.

**Tech Stack:** Rust, `claurst-mcp` (rmcp-backed `McpClient`), `claurst-core::config::McpServerConfig`, `tokio` (process + time, workspace `full`), `async-trait`, `serde_json`. Python `ml-intern` subprocess driven via `uv run`.

## Global Constraints

- **Edit scope:** ONLY `crates/research/` (+ workspace `Cargo.toml` to add `claurst-mcp` if not already a research dep — it is a workspace dep at `Cargo.toml:113`). The v0 port-reference (`lib/`, repo-root `tests/test_*.py`, `skills/`, repo-root `agents/`, `.claude-plugin/`, `commands/`, `hooks/`, `mcp/`, `tools/ml-intern`, `.mcp.json`) MUST NOT be modified. Repo-root `agents/*.md` and `mcp/server.py` are read-only port-reference; the `#[ignore]` smoke test *reads* the real `mcp/` subprocess, it does not edit it.
- **No crate-root `pub use` re-exports.** `lib.rs` uses `pub mod` only — add `pub mod mcp;`. Consumers use full paths `megaresearcher_research::mcp::...`.
- **No `api`-crate changes** (the `Arc<dyn LlmProvider>` seam is consumed, not modified). No `claurst-mcp` crate changes (we use its public API only).
- **Naming convention is `mcp__<server>__<tool>`** (double underscore) — to match the byte-identical agent fixtures in `tests/fixtures/agents/*.md` which tell workers to call `mcp__ml-intern__hf_papers`. This is intentionally distinct from claurst's own `<server>_<tool>` wrapper, which we do not use.
- **Per-task hygiene:** `cargo fmt -p megaresearcher-research` before commit; `cargo clippy -p megaresearcher-research --all-targets -- -D warnings` MUST be clean (the `--all-targets` flag is required — bare `-D warnings` skips test binaries); `cargo test -p megaresearcher-research` green. Final sweep (T9) runs `cargo clippy --workspace --all-targets -- -D warnings` + `cargo test --workspace`.
- **GPL-3.0.** Commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>`.
- **No git worktrees.** Work on branch `main` directly. Confirm `git branch --show-current` is `main` before each task; include `main` in every subagent prompt so they never `git switch`.
- **Banned phrases/words** ("load-bearing", "this is doing a lot of work", emphatic "real", "honest/honestly/to be honest") never appear in implementer-produced text. ("real" in the sense of "the actual subprocess" is fine; "real" as an emphatic adjective is not. The `#[ignore]` smoke-test name uses "real" as a factual modifier of the subprocess — acceptable; avoid it elsewhere.)
- **Determinism preserved:** passing `&[]` as `extra_tools` leaves `run_worker` building `vec![read, write]` exactly as today. The `FakeProvider` ignores `req.tools` and replays scripted `StreamEvent`s, so the 4a/4b gap-finding + hypothesis integration tests (44-turn revision loop, 16-turn kill, etc.) stay byte-identical and green without re-scripting.

## File Structure

- **Create:** `crates/research/src/mcp.rs` — the MCP wiring module (built across T1–T3).
- **Create:** `crates/research/tests/mcp_smoke.rs` — the `#[ignore]` real-subprocess 9-tool smoke test (T4).
- **Modify:** `crates/research/src/lib.rs` — add `pub mod mcp;` (T1).
- **Modify:** `crates/research/Cargo.toml` — add `claurst-mcp = { workspace = true }` (T1).
- **Modify:** `crates/research/src/orchestrator/dispatch.rs` — `run_worker` + `dispatch_wave` gain `extra_tools` (T5).
- **Modify:** `crates/research/src/orchestrator/gate.rs` — `verify_wave` gains `extra_tools` (T5).
- **Modify:** `crates/research/src/orchestrator/{hypothesis,redteam,evaldesign,synthesize}.rs` — the 5 phase entry functions gain `extra_tools` (T6).
- **Modify:** `crates/research/src/orchestrator/mod.rs` — `OrchestratorConfig.mcp`, `execute()` connect + thread `extra_tools` (T7).
- **Modify:** `crates/research/tests/orchestrator.rs` — update ~18 direct call sites + 6 `OrchestratorConfig` literals + add the MCP-in-loop test (T5/T6/T7/T8).

---

### Task 1: McpCaller seam + McpClientCaller + result conversion

**Files:**
- Create: `crates/research/src/mcp.rs`
- Modify: `crates/research/src/lib.rs` (add `pub mod mcp;`)
- Modify: `crates/research/Cargo.toml` (add `claurst-mcp` dep)

**Interfaces:**
- Produces: `pub struct McpError(pub String)` (Debug/Clone/Display/Error); `#[async_trait] pub trait McpCaller: Send + Sync { async fn call_tool(&self, name: &str, arguments: Option<Value>) -> Result<CallToolResult, McpError>; }`; `pub struct McpClientCaller(pub Arc<McpClient>);` impl `McpCaller`; `pub fn mcp_result_to_tool_result(result: &CallToolResult) -> ToolResult`.
- Consumes: `claurst_mcp::{McpClient, CallToolResult, McpContent}` (all public), `crate::worker_tools::ToolResult`.

- [ ] **Step 1: Add the dependency + module declaration**

Add to `crates/research/Cargo.toml` `[dependencies]` (alphabetical order — insert after `claurst-core`):

```toml
claurst-mcp = { workspace = true }
```

Add to `crates/research/src/lib.rs` (after `pub mod orchestrator;`, before `pub mod paper_chain;` — keep alphabetical-ish; the existing order is `orchestrator, paper_chain, prompt_asset, state, worker, worker_tools`, so insert `mcp` after `orchestrator`):

```rust
pub mod mcp;
```

- [ ] **Step 2: Write the failing tests (create `crates/research/src/mcp.rs` with tests first)**

Create `crates/research/src/mcp.rs`:

```rust
//! MCP tool wiring (Phase 5). The research crate drives the Python ml-intern
//! MCP server subprocess through claurst's `McpClient`, exposing each server
//! tool as a research `worker_tools::Tool` under the `mcp__<server>__<tool>`
//! name. The orchestrator connects the server in pre-flight (design §202) and
//! passes the tool set to every worker as `extra_tools`.
//!
//! This file is built across Phase 5 tasks:
//! - Task 1: the `McpCaller` seam, the production `McpClientCaller`, and
//!   `CallToolResult` → `ToolResult` conversion.
//! - Task 2: the `McpTool` wrapper + `McpToolSet`.
//! - Task 3: `ml_intern_config`, `mcp_project_dir`, `prewarm`.

use std::sync::Arc;

use async_trait::async_trait;
use claurst_mcp::{CallToolResult, McpClient, McpContent};
use serde_json::Value;

use crate::worker_tools::ToolResult;

/// An MCP-client error surfaced through the research layer.
#[derive(Debug, Clone)]
pub struct McpError(pub String);

impl std::fmt::Display for McpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mcp error: {}", self.0)
    }
}

impl std::error::Error for McpError {}

/// The seam a research `McpTool` calls through. The production impl wraps a
/// connected `claurst_mcp::McpClient`; tests supply a `FakeMcpCaller`. We use
/// this trait rather than claurst's `McpClientBackend` because
/// `McpClient::from_backend` is private to the claurst-mcp crate, so a fake
/// backend cannot be injected from outside it. Keeping our own seam lets the
/// research crate be unit-tested without spawning a subprocess.
#[async_trait]
pub trait McpCaller: Send + Sync {
    async fn call_tool(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<CallToolResult, McpError>;
}

/// Production `McpCaller` over a connected `claurst_mcp::McpClient`.
pub struct McpClientCaller(pub Arc<McpClient>);

#[async_trait]
impl McpCaller for McpClientCaller {
    async fn call_tool(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<CallToolResult, McpError> {
        self.0
            .call_tool(name, arguments)
            .await
            .map_err(|e| McpError(e.to_string()))
    }
}

/// Convert an MCP `CallToolResult` to a research `ToolResult`. The content is
/// the concatenated text of every content block (via claurst-mcp's
/// `mcp_result_to_string`); `is_error` selects ok/err.
pub fn mcp_result_to_tool_result(result: &CallToolResult) -> ToolResult {
    let text = claurst_mcp::mcp_result_to_string(result);
    if result.is_error {
        ToolResult::err(text)
    } else {
        ToolResult::ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker_tools::Tool;
    use std::sync::Mutex;

    /// Test `McpCaller` that records every call and returns a canned result.
    struct FakeMcpCaller {
        calls: Mutex<Vec<(String, Option<Value>)>>,
        result: CallToolResult,
    }
    impl FakeMcpCaller {
        fn new(result: CallToolResult) -> Self {
            Self {
                calls: Mutex::new(vec![]),
                result,
            }
        }
        fn calls(&self) -> Vec<(String, Option<Value>)> {
            self.calls.lock().unwrap().clone()
        }
    }
    #[async_trait]
    impl McpCaller for FakeMcpCaller {
        async fn call_tool(
            &self,
            name: &str,
            arguments: Option<Value>,
        ) -> Result<CallToolResult, McpError> {
            self.calls.lock().unwrap().push((name.to_string(), arguments));
            Ok(self.result.clone())
        }
    }

    fn text_result(text: &str, is_error: bool) -> CallToolResult {
        CallToolResult {
            content: vec![McpContent::Text { text: text.into() }],
            is_error,
        }
    }

    #[test]
    fn mcp_result_to_tool_result_ok_concatenates_text() {
        let result = text_result("hello", false);
        let tr = mcp_result_to_tool_result(&result);
        assert!(!tr.is_error);
        assert_eq!(tr.content, "hello");
    }

    #[test]
    fn mcp_result_to_tool_result_error_keeps_content() {
        let result = text_result("boom: bad query", true);
        let tr = mcp_result_to_tool_result(&result);
        assert!(tr.is_error);
        assert_eq!(tr.content, "boom: bad query");
    }

    #[tokio::test]
    async fn fake_mcp_caller_returns_canned_result_and_records_call() {
        let caller = FakeMcpCaller::new(text_result("# 1 result", false));
        let out = caller
            .call_tool("hf_papers", Some(json!({"operation": "trending"})))
            .await
            .unwrap();
        assert!(!out.is_error);
        let calls = caller.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "hf_papers");
        assert_eq!(calls[0].1, Some(json!({"operation": "trending"})));
    }

    // Silence unused imports until Task 2/3 use them.
    #[allow(dead_code)]
    fn _silence(_t: &dyn Tool, _v: Value) {}
}
```

- [ ] **Step 3: Run tests to verify they fail (RED)**

Run: `cargo test -p megaresearcher-research --lib mcp`
Expected: FAIL — `error[E0433]: failed to resolve: could not find mcp` (until `pub mod mcp;` is added) — after Step 1 it should compile and the 3 tests PASS (Task 1 defines exactly what the tests use). If `claurst-mcp` is not yet resolvable, confirm `Cargo.toml` edit landed. The RED check here is: before Step 1's edits, `cargo test` cannot find the module. After Step 1+2 together, tests pass — that is the intended TDD flow for a brand-new module (the tests and the code land together because the module is new; Task 2 onward is strict RED-then-GREEN on the unit tests within the file).

- [ ] **Step 4: Verify GREEN + clippy clean**

Run: `cargo test -p megaresearcher-research --lib mcp` → 3 passed.
Run: `cargo clippy -p megaresearcher-research --all-targets -- -D warnings` → exit 0.
Run: `cargo fmt -p megaresearcher-research`.

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/mcp.rs crates/research/src/lib.rs crates/research/Cargo.toml
git commit -m "feat(rs): Phase 5 Task 1 — McpCaller seam + result conversion

Add crates/research/src/mcp.rs with the McpCaller trait (the seam a
research McpTool calls through), the production McpClientCaller over a
connected claurst_mcp::McpClient, and CallToolResult → ToolResult
conversion. Uses a research-owned trait rather than claurst's
McpClientBackend because McpClient::from_backend is private to the
claurst-mcp crate, so a fake backend cannot be injected from outside it.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: McpTool wrapper + McpToolSet

**Files:**
- Modify: `crates/research/src/mcp.rs` (append McpTool + McpToolSet + tests)

**Interfaces:**
- Produces: `pub struct McpTool { … }` impl `crate::worker_tools::Tool` (name = `mcp__<server>__<bare>`, read-only, concurrency-safe); `pub struct McpToolSet { tools: Vec<Arc<dyn Tool>> }` with `pub fn tools(&self) -> &[Arc<dyn Tool>]`, `pub fn from_caller(server_name, caller: Arc<dyn McpCaller>, server_tools: Vec<claurst_mcp::McpTool>) -> Self`, `pub async fn connect(config: &McpServerConfig) -> Result<Self, McpError>`.
- Consumes: `claurst_mcp::{McpClient, McpTool as ServerMcpTool, expand_server_config}`, `claurst_core::config::McpServerConfig`, `crate::worker_tools::{Tool, ToolResult}`.

- [ ] **Step 1: Write the failing tests (append to the `tests` module in `mcp.rs`)**

Append inside `#[cfg(test)] mod tests` (the `FakeMcpCaller`/`text_result` helpers from Task 1 are reused):

```rust
    use claurst_mcp::McpTool as ServerMcpTool;

    fn server_tool(name: &str, desc: &str) -> ServerMcpTool {
        ServerMcpTool {
            name: name.into(),
            description: Some(desc.into()),
            input_schema: json!({"type": "object"}),
        }
    }

    #[test]
    fn mcp_tool_name_uses_double_underscore_convention() {
        let caller = Arc::new(FakeMcpCaller::new(text_result("ok", false))) as Arc<dyn McpCaller>;
        let set = McpToolSet::from_caller("ml-intern", caller, vec![server_tool("hf_papers", "papers")]);
        assert_eq!(set.tools().len(), 1);
        assert_eq!(set.tools()[0].name(), "mcp__ml-intern__hf_papers");
        assert_eq!(set.tools()[0].description(), "papers");
    }

    #[tokio::test]
    async fn mcp_tool_call_routes_to_caller_with_bare_name_and_input() {
        let caller = Arc::new(FakeMcpCaller::new(text_result("# 1 result", false))) as Arc<dyn McpCaller>;
        let set = McpToolSet::from_caller("ml-intern", caller.clone(), vec![server_tool("hf_papers", "papers")]);
        let tool = set.tools()[0].clone();
        let tr = tool.call(json!({"operation": "trending", "limit": 1})).await;
        assert!(!tr.is_error);
        assert_eq!(tr.content, "# 1 result");
        let calls = caller.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "hf_papers"); // bare name, not the prefixed name
        assert_eq!(calls[0].1, Some(json!({"operation": "trending", "limit": 1})));
    }

    #[tokio::test]
    async fn mcp_tool_call_passes_none_for_null_input() {
        let caller = Arc::new(FakeMcpCaller::new(text_result("ok", false))) as Arc<dyn McpCaller>;
        let set = McpToolSet::from_caller("ml-intern", caller.clone(), vec![server_tool("web_search", "search")]);
        let tool = set.tools()[0].clone();
        tool.call(Value::Null).await;
        let calls = caller.calls();
        assert_eq!(calls[0].1, None);
    }

    #[tokio::test]
    async fn mcp_tool_call_maps_caller_error_to_toolresult_err() {
        struct ErrCaller;
        #[async_trait]
        impl McpCaller for ErrCaller {
            async fn call_tool(&self, _name: &str, _arguments: Option<Value>) -> Result<CallToolResult, McpError> {
                Err(McpError("server crashed".into()))
            }
        }
        let caller = Arc::new(ErrCaller) as Arc<dyn McpCaller>;
        let set = McpToolSet::from_caller("ml-intern", caller, vec![server_tool("hf_papers", "papers")]);
        let tr = set.tools()[0].clone().call(json!({})).await;
        assert!(tr.is_error);
        assert!(tr.content.contains("mcp__ml-intern__hf_papers"), "content was: {}", tr.content);
        assert!(tr.content.contains("server crashed"), "content was: {}", tr.content);
    }

    #[test]
    fn mcp_tool_set_preserves_server_order_and_marks_read_only_concurrency_safe() {
        let caller = Arc::new(FakeMcpCaller::new(text_result("ok", false))) as Arc<dyn McpCaller>;
        let set = McpToolSet::from_caller(
            "ml-intern",
            caller,
            vec![server_tool("hf_papers", "a"), server_tool("web_search", "b")],
        );
        assert_eq!(set.tools().len(), 2);
        assert_eq!(set.tools()[0].name(), "mcp__ml-intern__hf_papers");
        assert_eq!(set.tools()[1].name(), "mcp__ml-intern__web_search");
        assert!(set.tools()[0].is_read_only());
        assert!(set.tools()[0].is_concurrency_safe());
    }
```

- [ ] **Step 2: Run tests to verify they fail (RED)**

Run: `cargo test -p megaresearcher-research --lib mcp`
Expected: FAIL — `error[E0425]: cannot find function/value McpToolSet` / `McpTool` not defined.

- [ ] **Step 3: Implement McpTool + McpToolSet (append to `mcp.rs`, above the `#[cfg(test)]` block)**

Add the needed imports near the top of `mcp.rs` (merge into the existing `use` block — the file already has `use std::sync::Arc; use async_trait::async_trait; use claurst_mcp::{CallToolResult, McpClient, McpContent}; use serde_json::Value; use crate::worker_tools::ToolResult;`). Add:

```rust
use std::path::Path;

use claurst_core::config::McpServerConfig;
use claurst_mcp::{expand_server_config, McpTool as ServerMcpTool};

use crate::worker_tools::Tool;
```

Then append the types (above the `#[cfg(test)] mod tests` block):

```rust
/// A research `Tool` backed by one MCP tool exposed by a server. The exposed
/// name follows the `mcp__<server>__<tool>` convention so the byte-identical
/// agent prompt fixtures (which tell workers to call `mcp__ml-intern__hf_papers`
/// etc.) resolve through the worker's tool dispatch without edits. This is
/// intentionally distinct from claurst's own `McpToolWrapper`, which uses
/// `<server>_<tool>` (single underscore).
pub struct McpTool {
    full_name: String, // mcp__<server>__<bare>
    bare_name: String, // hf_papers
    description: String,
    input_schema: Value,
    caller: Arc<dyn McpCaller>,
}

impl McpTool {
    /// Build one wrapper from a server-exposed tool definition + a shared caller.
    fn from_server_tool(server_name: &str, tool: ServerMcpTool, caller: Arc<dyn McpCaller>) -> Arc<dyn Tool> {
        let bare_name = tool.name.clone();
        let full_name = format!("mcp__{server_name}__{bare_name}");
        Arc::new(McpTool {
            full_name,
            bare_name,
            description: tool.description.unwrap_or_default(),
            input_schema: tool.input_schema,
            caller,
        }) as Arc<dyn Tool>
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.full_name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn input_schema(&self) -> Value {
        self.input_schema.clone()
    }
    async fn call(&self, input: Value) -> ToolResult {
        let arguments = if input.is_null() { None } else { Some(input) };
        match self.caller.call_tool(&self.bare_name, arguments).await {
            Ok(result) => mcp_result_to_tool_result(&result),
            Err(e) => ToolResult::err(format!("MCP tool '{}' failed: {}", self.full_name, e)),
        }
    }
    // All ml-intern tools are read-only searches/inspects/reads.
    fn is_read_only(&self) -> bool {
        true
    }
    fn is_concurrency_safe(&self) -> bool {
        true
    }
}

/// A connected MCP server plus the research `Tool` wrappers for every tool it
/// exposes. Owns the `McpCaller` (and thus the subprocess lifetime for the
/// production path). Pass `set.tools()` as the orchestrator's `extra_tools`.
pub struct McpToolSet {
    tools: Vec<Arc<dyn Tool>>,
}

impl McpToolSet {
    /// The tool wrappers, in server-listed order. Pass this slice to the
    /// orchestrator's `extra_tools` argument.
    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }

    /// Test/non-subprocess constructor: build a set from a provided caller +
    /// server-exposed tool list. No subprocess, no prewarm.
    pub fn from_caller(
        server_name: &str,
        caller: Arc<dyn McpCaller>,
        server_tools: Vec<ServerMcpTool>,
    ) -> Self {
        let tools = server_tools
            .into_iter()
            .map(|t| McpTool::from_server_tool(server_name, t, caller.clone()))
            .collect();
        Self { tools }
    }

    /// Production constructor: prewarm the venv (Task 3), connect the stdio
    /// subprocess, and build tool wrappers from the server's tool list.
    /// `config` may use `${VAR}` tokens (resolved from the process env at
    /// connect time via `expand_server_config`).
    pub async fn connect(config: &McpServerConfig) -> Result<Self, McpError> {
        prewarm(config).await?;
        let expanded = expand_server_config(config);
        let client = McpClient::connect_stdio(&expanded)
            .await
            .map_err(|e| McpError(e.to_string()))?;
        let server_name = client.server_name.clone();
        let server_tools = client.tools.clone();
        let caller = Arc::new(McpClientCaller(Arc::new(client))) as Arc<dyn McpCaller>;
        Ok(Self::from_caller(&server_name, caller, server_tools))
    }
}
```

- [ ] **Step 4: Run tests to verify they pass (GREEN)**

Run: `cargo test -p megaresearcher-research --lib mcp` → Task 1's 3 + Task 2's 5 = 8 passed.
Run: `cargo clippy -p megaresearcher-research --all-targets -- -D warnings` → exit 0.
Run: `cargo fmt -p megaresearcher-research`.

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/mcp.rs
git commit -m "feat(rs): Phase 5 Task 2 — McpTool wrapper + McpToolSet

McpTool implements the research worker_tools::Tool trait, exposing each
server tool as mcp__<server>__<tool> (the double-underscore convention the
byte-identical agent fixtures already use). McpToolSet.connect prewarms +
connects the stdio subprocess and builds the tool vec from client.tools;
from_caller is the no-subprocess constructor for tests.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: ml_intern_config + mcp_project_dir + prewarm

**Files:**
- Modify: `crates/research/src/mcp.rs` (append config builder + prewarm + tests)

**Interfaces:**
- Produces: `pub fn ml_intern_config(repo_root: &Path) -> McpServerConfig`; `pub async fn prewarm(config: &McpServerConfig) -> Result<(), McpError>`; (private) `fn mcp_project_dir(config: &McpServerConfig) -> Option<&str>`.

- [ ] **Step 1: Write the failing tests (append to the `tests` module)**

```rust
    use claurst_core::config::McpServerConfig;
    use std::collections::HashMap;
    use std::path::Path;

    fn no_project_config() -> McpServerConfig {
        McpServerConfig {
            name: "x".into(),
            command: Some("uv".into()),
            args: vec!["run".into(), "python".into()],
            env: HashMap::new(),
            url: None,
            server_type: "stdio".into(),
        }
    }

    #[test]
    fn ml_intern_config_resolves_absolute_paths_and_env_tokens() {
        let root = Path::new("/repo");
        let cfg = ml_intern_config(root);
        assert_eq!(cfg.name, "ml-intern");
        assert_eq!(cfg.command.as_deref(), Some("uv"));
        assert_eq!(
            cfg.args,
            vec![
                "run".into(),
                "--project".into(),
                "/repo/mcp".into(),
                "python".into(),
                "/repo/mcp/server.py".into(),
            ]
        );
        assert_eq!(cfg.server_type, "stdio");
        assert_eq!(cfg.url, None);
        assert_eq!(cfg.env.get("HF_TOKEN").unwrap(), "${HF_TOKEN}");
        assert_eq!(cfg.env.get("GITHUB_TOKEN").unwrap(), "${GITHUB_TOKEN}");
        assert_eq!(cfg.env.get("ML_INTERN_PATH").unwrap(), "/repo/tools/ml-intern");
    }

    #[test]
    fn mcp_project_dir_finds_project_arg() {
        let cfg = ml_intern_config(Path::new("/repo"));
        assert_eq!(mcp_project_dir(&cfg), Some("/repo/mcp"));
    }

    #[test]
    fn mcp_project_dir_returns_none_without_project_arg() {
        assert_eq!(mcp_project_dir(&no_project_config()), None);
    }

    #[tokio::test]
    async fn prewarm_errors_when_no_project_arg() {
        let err = prewarm(&no_project_config()).await.unwrap_err();
        assert!(err.0.contains("no --project"), "err was: {}", err.0);
    }
```

- [ ] **Step 2: Run tests to verify they fail (RED)**

Run: `cargo test -p megaresearcher-research --lib mcp`
Expected: FAIL — `cannot find function ml_intern_config` / `mcp_project_dir` / `prewarm`.

- [ ] **Step 3: Implement the config builder + prewarm (append to `mcp.rs`, above the `#[cfg(test)]` block)**

Add imports near the top of `mcp.rs` (merge into the existing `use` block):

```rust
use std::collections::HashMap;
use std::process::Stdio;

use tokio::process::Command;
use tokio::time::{timeout, Duration};
```

(If `use std::path::Path;` was added in Task 2, keep it. If not, add it now.)

Then append:

```rust
/// The uv project dir for an MCP server, derived from `--project` in its
/// args. Used by `prewarm` to run `uv sync` before connect.
fn mcp_project_dir(config: &McpServerConfig) -> Option<&str> {
    let mut iter = config.args.iter();
    while let Some(a) = iter.next() {
        if a == "--project" {
            return iter.next().map(|s| s.as_str());
        }
    }
    None
}

/// The default ml-intern server config: spawn `mcp/server.py` via
/// `uv run --project <repo>/mcp`, forwarding `HF_TOKEN`/`GITHUB_TOKEN` from the
/// process env (resolved at connect), and pointing `ML_INTERN_PATH` at the
/// vendored library. Mirrors the repo-root `.mcp.json` the v0 plugin uses,
/// but with absolute paths resolved from `repo_root` (no `${CLAUDE_PLUGIN_ROOT}`).
pub fn ml_intern_config(repo_root: &Path) -> McpServerConfig {
    let mcp_dir = repo_root.join("mcp");
    let server = mcp_dir.join("server.py");
    let ml_intern_path = repo_root.join("tools").join("ml-intern");
    let mut env = HashMap::new();
    env.insert("HF_TOKEN".to_string(), "${HF_TOKEN}".to_string());
    env.insert("GITHUB_TOKEN".to_string(), "${GITHUB_TOKEN}".to_string());
    env.insert(
        "ML_INTERN_PATH".to_string(),
        ml_intern_path.to_string_lossy().to_string(),
    );
    McpServerConfig {
        name: "ml-intern".to_string(),
        command: Some("uv".to_string()),
        args: vec![
            "run".to_string(),
            "--project".to_string(),
            mcp_dir.to_string_lossy().to_string(),
            "python".to_string(),
            server.to_string_lossy().to_string(),
        ],
        env,
        url: None,
        server_type: "stdio".to_string(),
    }
}

/// Pre-warm the server's uv venv (`uv sync --project <dir>`) before connecting,
/// so the first `client.connect` does not pay the (potentially >30s) venv
/// creation cost and miss the handshake timeout. Idempotent: `uv sync` is a
/// no-op on an already-synced venv. Times out after 5 minutes.
pub async fn prewarm(config: &McpServerConfig) -> Result<(), McpError> {
    let dir = mcp_project_dir(config)
        .ok_or_else(|| McpError("no --project <dir> in MCP server args".to_string()))?;
    let output = timeout(
        Duration::from_secs(300),
        Command::new("uv")
            .arg("sync")
            .arg("--project")
            .arg(dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output(),
    )
    .await
    .map_err(|_| McpError("uv sync timed out after 300s".to_string()))?
    .map_err(|e| McpError(format!("failed to spawn uv: {e}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(McpError(format!("uv sync failed: {stderr}")));
    }
    Ok(())
}
```

- [ ] **Step 4: Run tests to verify they pass (GREEN)**

Run: `cargo test -p megaresearcher-research --lib mcp` → 8 + 4 = 12 passed.
Run: `cargo clippy -p megaresearcher-research --all-targets -- -D warnings` → exit 0.
Run: `cargo fmt -p megaresearcher-research`.

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/mcp.rs
git commit -m "feat(rs): Phase 5 Task 3 — ml_intern_config + uv prewarm

ml_intern_config builds the McpServerConfig that mirrors the v0 .mcp.json
(uv run --project <repo>/mcp, HF_TOKEN/GITHUB_TOKEN forwarded from the
process env, ML_INTERN_PATH at the vendored library). prewarm runs
'uv sync --project <dir>' before connect so the first handshake doesn't
pay the venv-creation cost — the deliberate improvement over Claude Code.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Real ml-intern 9-tool smoke test (`#[ignore]`)

**Files:**
- Create: `crates/research/tests/mcp_smoke.rs`

**Interfaces:**
- Consumes: `megaresearcher_research::mcp::{ml_intern_config, McpToolSet}`, `megaresearcher_research::worker_tools::Tool`, `serde_json::json`.

- [ ] **Step 1: Write the smoke test**

Create `crates/research/tests/mcp_smoke.rs`:

```rust
//! Real ml-intern subprocess smoke test (Phase 5). Connects to the actual
//! Python MCP server via uv, asserts the 9 expected tools surface, and calls
//! one read-only tool that needs no API key. Ignored by default — run with:
//!   cargo test -p megaresearcher-research --test mcp_smoke -- --ignored
//! when uv, the ml-intern venv, and (for HF tools) HF_TOKEN are available.

use std::env;
use std::path::PathBuf;

use serde_json::json;

use megaresearcher_research::mcp::{ml_intern_config, McpToolSet};
use megaresearcher_research::worker_tools::Tool;

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/research; ../.. = repo root.
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("..")
        .join("..")
        .canonicalize()
        .expect("repo root")
}

const EXPECTED_TOOLS: &[&str] = &[
    "mcp__ml-intern__hf_papers",
    "mcp__ml-intern__hf_inspect_dataset",
    "mcp__ml-intern__hf_docs_explore",
    "mcp__ml-intern__hf_docs_fetch",
    "mcp__ml-intern__hf_repo_files",
    "mcp__ml-intern__github_examples",
    "mcp__ml-intern__github_list_repos",
    "mcp__ml-intern__github_read_file",
    "mcp__ml-intern__web_search",
];

#[tokio::test]
#[ignore = "requires uv + ml-intern venv (+ HF_TOKEN for HF tools); run with --ignored"]
async fn real_ml_intern_exposes_9_tools_and_web_search_works() {
    let config = ml_intern_config(&repo_root());
    let set = McpToolSet::connect(&config).await.expect("connect ml-intern");

    let names: Vec<&str> = set.tools().iter().map(|t| t.name()).collect();
    assert_eq!(names.len(), 9, "expected 9 tools, got {names:?}");
    for expected in EXPECTED_TOOLS {
        assert!(
            names.contains(expected),
            "missing tool {expected}; have {names:?}"
        );
    }

    // web_search (DuckDuckGo) needs no API key, so it works without HF_TOKEN.
    let web = set
        .tools()
        .iter()
        .find(|t| t.name() == "mcp__ml-intern__web_search")
        .expect("web_search present");
    let result = web.call(json!({"query": "hello world", "max_results": 1})).await;
    assert!(
        !result.is_error,
        "web_search returned an error: {}",
        result.content
    );
    assert!(!result.content.is_empty(), "web_search returned empty content");
}
```

- [ ] **Step 2: Verify the test compiles + is collected (ignored)**

Run: `cargo test -p megaresearcher-research --test mcp_smoke --no-run` → compiles.
Run: `cargo test -p megaresearcher-research --test mcp_smoke` → `1 filtered out, 0 passed` (ignored, as expected). The non-ignored run must not fail the suite.
Run: `cargo clippy -p megaresearcher-research --all-targets -- -D warnings` → exit 0.
Run: `cargo fmt -p megaresearcher-research`.

- [ ] **Step 3: (Manual, when env allows) confirm the ignored test passes**

Run: `cargo test -p megaresearcher-research --test mcp_smoke -- --ignored`
Expected: 1 passed (requires uv + a synced `mcp/.venv`; HF_TOKEN only needed if you change the called tool to an HF one). Record the result in the task report. If the env is not available locally, leave it ignored and note that in the report — the compile + collect check in Step 2 is the gating verification for this task.

- [ ] **Step 4: Commit**

```bash
git add crates/research/tests/mcp_smoke.rs
git commit -m "test(rs): Phase 5 Task 4 — real ml-intern 9-tool smoke test

An #[ignore] integration test that connects to the actual ml-intern
subprocess via McpToolSet::connect, asserts all 9 tools surface under
the mcp__ml-intern__<tool> names, and calls web_search (no API key
needed). Compile + collect is gating; the run is manual under --ignored.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Thread `extra_tools` through run_worker / dispatch_wave / verify_wave

**Files:**
- Modify: `crates/research/src/orchestrator/dispatch.rs` (`run_worker`, `dispatch_wave`)
- Modify: `crates/research/src/orchestrator/gate.rs` (`verify_wave`)
- Modify: `crates/research/tests/orchestrator.rs` (direct call sites: lines 280, 309, 342, 347, 380, 384, 417, 421)

**Interfaces:**
- `run_worker` signature becomes:
  ```rust
  pub async fn run_worker(
      spec: &WorkerSpec,
      agents_dir: &Path,
      provider: Arc<dyn LlmProvider>,
      default_model: &str,
      extra_tools: &[Arc<dyn Tool>],
  ) -> Result<WorkerOutcome, OrchestratorError>
  ```
  Body: build `vec![read, write]`, then `tools.extend(extra_tools.iter().cloned())`.
- `dispatch_wave` gains a final `extra_tools: &[Arc<dyn Tool>]` param, forwarded into the `run_worker` call inside its `async move` closure.
- `verify_wave` gains a final `extra_tools: &[Arc<dyn Tool>]` param, forwarded into its `run_worker(&retry, ...)` call.

- [ ] **Step 1: Write the failing test first is N/A here (signature change) — instead, update the existing direct call sites to `&[]` and verify the suite compiles green with the change**

This task is a mechanical signature threading, not new behavior. The TDD guard is: (a) the existing 4a/4b tests must stay green after the change (proving `&[]` is a no-op), and (b) the new behavior (extra tools appended) is tested in Task 8. So: make the code change + update call sites, then run the full orchestrator suite.

Edit `crates/research/src/orchestrator/dispatch.rs` `run_worker`:

```rust
/// Run a single worker: load its agent prompt asset, resolve the model,
/// wire jailed Read/Write tools plus any `extra_tools` (MCP tools), and drive
/// the worker.
pub async fn run_worker(
    spec: &WorkerSpec,
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    extra_tools: &[Arc<dyn Tool>],
) -> Result<WorkerOutcome, OrchestratorError> {
    let asset_path = agents_dir.join(format!("{}.md", spec.role));
    let asset = load_asset(&asset_path).map_err(OrchestratorError::Io)?;
    let model = if asset.model == "inherit" {
        default_model.to_string()
    } else {
        asset.model.clone()
    };
    let read =
        Arc::new(ScopedRead::with_shared(&spec.output_dir, &spec.shared_dir)) as Arc<dyn Tool>;
    let write = Arc::new(ScopedWrite::new(&spec.output_dir)) as Arc<dyn Tool>;
    let mut tools: Vec<Arc<dyn Tool>> = vec![read, write];
    tools.extend(extra_tools.iter().cloned());
    let worker = Worker::new(
        asset.body.clone(),
        tools,
        provider,
        WorkerConfig {
            max_turns: 50,
            max_tokens: 4096,
            model,
        },
        spec.output_dir.clone(),
    );
    worker
        .run(&spec.prompt)
        .await
        .map_err(OrchestratorError::Worker)
}
```

Edit `dispatch_wave` signature + closure:

```rust
pub async fn dispatch_wave(
    specs: Vec<WorkerSpec>,
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
    extra_tools: &[Arc<dyn Tool>],
) -> Result<Vec<(String, WorkerOutcome)>, OrchestratorError> {
    let n = specs.len();
    let indexed: Vec<(usize, WorkerSpec)> = specs.into_iter().enumerate().collect();
    let results: Vec<Result<(usize, String, WorkerOutcome), OrchestratorError>> =
        stream::iter(indexed)
            .map(|(i, spec)| {
                let provider = provider.clone();
                async move {
                    run_worker(&spec, agents_dir, provider, default_model, extra_tools)
                        .await
                        .map(|o| (i, spec.name.clone(), o))
                }
            })
            .buffer_unordered(max_parallel.max(1) as usize)
            .collect()
            .await;

    let mut collected: Vec<(usize, String, WorkerOutcome)> = Vec::with_capacity(n);
    for r in results {
        collected.push(r?);
    }
    collected.sort_by_key(|(i, _, _)| *i);
    Ok(collected
        .into_iter()
        .map(|(_, name, o)| (name, o))
        .collect())
}
```

(`extra_tools` is `&[Arc<dyn Tool>]`, which is `Copy`-by-reference; the `async move` block captures the reference cheaply. `agents_dir` and `default_model` are also `Copy` references. `provider` is cloned per iteration as before.)

Edit `crates/research/src/orchestrator/gate.rs` `verify_wave`:

```rust
pub async fn verify_wave(
    outcomes: Vec<(String, WorkerOutcome)>,
    specs: &[WorkerSpec],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    extra_tools: &[Arc<dyn Tool>],
) -> Result<Vec<GateOutcome>, OrchestratorError> {
    let mut results = Vec::with_capacity(outcomes.len());
    for (name, _first) in outcomes {
        let spec = specs
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| OrchestratorError::Finalize(format!("no spec for worker {name}")))?;
        let missing = check_artifacts(&spec.output_dir, REQUIRED_ARTIFACTS);
        if missing.is_empty() {
            results.push(GateOutcome {
                name,
                status: GateStatus::Passed,
                retries: 0,
            });
            continue;
        }
        // One retry with the missing list appended.
        let retry = retry_spec(spec, &missing);
        run_worker(&retry, agents_dir, provider.clone(), default_model, extra_tools).await?;
        let still_missing = check_artifacts(&spec.output_dir, REQUIRED_ARTIFACTS);
        let status = if still_missing.is_empty() {
            GateStatus::Passed
        } else {
            GateStatus::Escalated
        };
        results.push(GateOutcome {
            name,
            status,
            retries: 1,
        });
    }
    Ok(results)
}
```

`gate.rs` needs `use crate::worker_tools::Tool;` added to its imports (it currently imports `check_artifacts` from worker_tools; add `Tool` to that line: `use crate::worker_tools::{check_artifacts, Tool};`).

- [ ] **Step 2: Update the direct call sites in `crates/research/tests/orchestrator.rs`**

These 8 sites now need a trailing `, &[]` (the parameter type is inferred from the function signature). Add the trailing argument to each (the implementer finds the exact lines via `grep -nE "run_worker\(|dispatch_wave\(|verify_wave\(" crates/research/tests/orchestrator.rs`):

- line ~280: `dispatch_wave(specs, &fixture_agents_dir(), provider, "fake-model", 1)` → add `, &[]`
- line ~309: `run_worker(&spec, &fixture_agents_dir(), provider, "resolved-model")` → add `, &[]`
- line ~342: `run_worker(&spec, &fixture_agents_dir(), provider.clone(), "fake-model")` → add `, &[]`
- line ~347: `verify_wave(...)` → add `, &[]`
- line ~380: `run_worker(&spec, &fixture_agents_dir(), provider.clone(), "fake-model")` → add `, &[]`
- line ~384: `verify_wave(...)` → add `, &[]`
- line ~417: `run_worker(&spec, &fixture_agents_dir(), provider.clone(), "fake-model")` → add `, &[]`
- line ~421: `verify_wave(...)` → add `, &[]`

(Exact line numbers may drift if earlier tasks shifted lines; the implementer uses grep to locate the call sites and appends `&[]` to each `run_worker`/`dispatch_wave`/`verify_wave` call that is missing the final argument. Do NOT touch the 6 `Orchestrator::execute`-style call sites in the integration tests — those go through `execute()`, which is wired in Task 7.)

- [ ] **Step 3: Verify the suite is green (the no-op `&[]` guard)**

Run: `cargo test -p megaresearcher-research` → all green (the 4a/4b tests pass unchanged because `&[]` keeps `run_worker` building `vec![read, write]`).
Run: `cargo clippy -p megaresearcher-research --all-targets -- -D warnings` → exit 0.
Run: `cargo fmt -p megaresearcher-research`.

- [ ] **Step 4: Commit**

```bash
git add crates/research/src/orchestrator/dispatch.rs crates/research/src/orchestrator/gate.rs crates/research/tests/orchestrator.rs
git commit -m "feat(rs): Phase 5 Task 5 — thread extra_tools through dispatch + gate

run_worker, dispatch_wave, and verify_wave gain a final extra_tools:
&[Arc<dyn Tool>] parameter (default &[]), appended to the jailed
Read/Write tools inside run_worker. The &[] default keeps every 4a/4b
fake-provider test byte-identical and green.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Thread `extra_tools` through the 5 phase entry functions

**Files:**
- Modify: `crates/research/src/orchestrator/hypothesis.rs` (`dispatch_hypothesis_smiths`, `redispatch_smith_revision`)
- Modify: `crates/research/src/orchestrator/redteam.rs` (`run_redteam_loop`)
- Modify: `crates/research/src/orchestrator/evaldesign.rs` (`run_eval_designers`)
- Modify: `crates/research/src/orchestrator/synthesize.rs` (`run_synthesist`)
- Modify: `crates/research/tests/orchestrator.rs` (direct call sites: lines 1012, 1049, 1086, 1148, 1184, 1214, 1253, 1362, 1394, 1424, 636)

**Interfaces:** Each function gains a final `extra_tools: &[Arc<dyn Tool>]` parameter, which it forwards as the final argument to every `dispatch_wave` / `verify_wave` / `run_worker` / `redispatch_smith_revision` call inside it. Each file needs `use crate::worker_tools::Tool;` (or `{..., Tool}` merged into an existing worker_tools use) in scope.

- [ ] **Step 1: `dispatch_hypothesis_smiths` + `redispatch_smith_revision` (`hypothesis.rs`)**

New signature (now 8 args → add `#[allow(clippy::too_many_arguments)]`):

```rust
#[allow(clippy::too_many_arguments)]
pub async fn dispatch_hypothesis_smiths(
    run_dir: &Path,
    spec_text: &str,
    gaps: &[Gap],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
    extra_tools: &[Arc<dyn Tool>],
) -> Result<Vec<Hypothesis>, OrchestratorError> {
    // ... unchanged body, but every dispatch_wave(...)/verify_wave(...) call
    //     inside gets a trailing `, extra_tools`.
}
```

New signature for `redispatch_smith_revision` (now 7 args — no allow needed):

```rust
pub async fn redispatch_smith_revision(
    hyp: &Hypothesis,
    spec_text: &str,
    redteam_output: &str,
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    extra_tools: &[Arc<dyn Tool>],
) -> Result<(), OrchestratorError> {
    // ... unchanged body, but its run_worker(...) (or dispatch_wave) call
    //     gets a trailing `, extra_tools`.
}
```

The implementer opens `hypothesis.rs`, applies the two new signatures, and appends `extra_tools` as the final argument to each `dispatch_wave` / `verify_wave` / `run_worker` call within both functions (located via `grep -nE "dispatch_wave\(|verify_wave\(|run_worker\(" crates/research/src/orchestrator/hypothesis.rs`). Add `use crate::worker_tools::Tool;` (merge into the existing `worker_tools` import if one is present).

- [ ] **Step 2: `run_redteam_loop` (`redteam.rs`)**

New signature (now 9 args — keep its existing `#[allow(clippy::too_many_arguments)]`):

```rust
#[allow(clippy::too_many_arguments)]
pub async fn run_redteam_loop(
    run_dir: &Path,
    spec_text: &str,
    hypotheses: Vec<Hypothesis>,
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
    swarm: &mut SwarmState,
    extra_tools: &[Arc<dyn Tool>],
) -> Result<RedTeamResult, OrchestratorError> {
    // ... unchanged body, but the dispatch_wave(...) and verify_wave(...) calls
    //     inside the inner loop AND the redispatch_smith_revision(...) call
    //     each get a trailing `, extra_tools`.
}
```

The implementer appends `extra_tools` to each `dispatch_wave` / `verify_wave` / `redispatch_smith_revision` call inside `run_redteam_loop` (grep locates them at ~lines 108, 116, 159). Add/merge `use crate::worker_tools::Tool;`.

- [ ] **Step 3: `run_eval_designers` (`evaldesign.rs`)**

New signature (now 9 args — keep its existing `#[allow(clippy::too_many_arguments)]`):

```rust
#[allow(clippy::too_many_arguments)]
pub async fn run_eval_designers(
    run_dir: &Path,
    spec_text: &str,
    survivors: &[Hypothesis],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
    swarm: &mut SwarmState,
    extra_tools: &[Arc<dyn Tool>],
) -> Result<EvalDesignResult, OrchestratorError> {
    // ... unchanged body, but its dispatch_wave(...)/verify_wave(...) calls
    //     get a trailing `, extra_tools`.
}
```

Append `extra_tools` to the `dispatch_wave` (~line 93) and `verify_wave` (~line 101) calls. Add/merge `use crate::worker_tools::Tool;`.

- [ ] **Step 4: `run_synthesist` (`synthesize.rs`)**

New signature (now 13 args — keep its existing `#[allow(clippy::too_many_arguments)]`):

```rust
#[allow(clippy::too_many_arguments)]
pub async fn run_synthesist(
    run_dir: &Path,
    spec_text: &str,
    plan_text: &str,
    scout_dirs: &[PathBuf],
    gap_dirs: &[PathBuf],
    smith_dirs: &[PathBuf],
    redteam_dirs: &[PathBuf],
    eval_dirs: &[PathBuf],
    agents_dir: &Path,
    provider: Arc<dyn LlmProvider>,
    default_model: &str,
    max_parallel: u32,
    extra_tools: &[Arc<dyn Tool>],
) -> Result<(WorkerSpec, WorkerOutcome), OrchestratorError> {
    // ... unchanged body, but its dispatch_wave(...) call (~line 77) and
    //     any verify_wave/run_worker call get a trailing `, extra_tools`.
}
```

Append `extra_tools` to the `dispatch_wave` call inside `run_synthesist` (~line 77). Add/merge `use crate::worker_tools::Tool;`.

- [ ] **Step 5: Update the direct call sites in `crates/research/tests/orchestrator.rs`**

Append `, &[]` as the final argument to each of these calls (grep to locate; do NOT touch `execute()`-based tests — those are wired in Task 7):

- ~1012, ~1049: `dispatch_hypothesis_smiths(...)` → add `, &[]`
- ~1086: `redispatch_smith_revision(...)` → add `, &[]`
- ~1148, ~1184, ~1214, ~1253: `run_redteam_loop(...)` → add `, &[]` (after the `&mut swarm` argument)
- ~1362, ~1394, ~1424: `run_eval_designers(...)` → add `, &[]` (after `&mut swarm`)
- ~636: `run_synthesist(...)` → add `, &[]` (after `max_parallel`)

- [ ] **Step 6: Verify GREEN**

Run: `cargo test -p megaresearcher-research` → all green (`&[]` default preserves behavior).
Run: `cargo clippy -p megaresearcher-research --all-targets -- -D warnings` → exit 0.
Run: `cargo fmt -p megaresearcher-research`.

- [ ] **Step 7: Commit**

```bash
git add crates/research/src/orchestrator/hypothesis.rs crates/research/src/orchestrator/redteam.rs crates/research/src/orchestrator/evaldesign.rs crates/research/src/orchestrator/synthesize.rs crates/research/tests/orchestrator.rs
git commit -m "feat(rs): Phase 5 Task 6 — thread extra_tools through phase entry fns

dispatch_hypothesis_smiths, redispatch_smith_revision, run_redteam_loop,
run_eval_designers, and run_synthesist gain a final extra_tools parameter
forwarded to their dispatch_wave/verify_wave/run_worker calls. The &[]
default keeps the 4b hypothesis integration tests byte-identical.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: OrchestratorConfig.mcp + execute() connect + thread through execute's 9 call sites

**Files:**
- Modify: `crates/research/src/orchestrator/mod.rs` (`OrchestratorConfig`, `execute()`, imports)
- Modify: `crates/research/tests/orchestrator.rs` (6 `OrchestratorConfig { ... }` literals at lines 531, 597, 699, 1492, 1576, 1647)

**Interfaces:**
- `OrchestratorConfig` gains `pub mcp: Option<McpServerConfig>`.
- `execute()` connects `McpToolSet::connect(&cfg)` in pre-flight when `config.mcp` is `Some`, builds `extra_tools: &[Arc<dyn Tool>]`, and passes it to all 9 wave/phase call sites.

- [ ] **Step 1: Add imports + the config field + the connect block in `mod.rs`**

Add to `mod.rs` imports (near the existing `use claurst_api::LlmProvider;`):

```rust
use claurst_core::config::McpServerConfig;
```

Add to `mod.rs` imports (near the other `use crate::...` lines):

```rust
use crate::mcp::McpToolSet;
use crate::worker_tools::Tool;
```

Extend `OrchestratorConfig`:

```rust
/// Orchestrator configuration: where runs live, where agent prompt assets
/// are, the model to resolve "inherit" to, the wave concurrency bound, and an
/// optional MCP server (ml-intern) to connect in pre-flight. When `mcp` is
/// `None`, workers get only the jailed Read/Write tools (fake-provider tests).
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub research_base: PathBuf,
    pub agents_dir: PathBuf,
    pub default_model: String,
    pub max_parallel: u32,
    pub mcp: Option<McpServerConfig>,
}
```

Insert the connect block in `execute()`, immediately after the `preflight_check(...).map_err(OrchestratorError::Preflight)?;` line and before `let run_dir = create_run_tree(...)`:

```rust
        preflight_check(spec_path, plan_path, &self.config.agents_dir, target)
            .map_err(OrchestratorError::Preflight)?;

        // Connect the MCP server (ml-intern) in pre-flight: pre-warm the uv
        // venv, then connect the stdio subprocess. When `config.mcp` is `None`
        // (no MCP server configured, e.g. fake-provider tests), `extra_tools`
        // is empty and workers get only the jailed Read/Write tools — the
        // existing gap-finding / hypothesis integration tests stay
        // byte-identical.
        let mcp_set: Option<McpToolSet> = if let Some(mcp_cfg) = self.config.mcp.as_ref() {
            Some(
                McpToolSet::connect(mcp_cfg)
                    .await
                    .map_err(|e| OrchestratorError::Preflight(format!("ml-intern unreachable: {e}")))?,
            )
        } else {
            None
        };
        let empty: Vec<Arc<dyn Tool>> = Vec::new();
        let extra_tools: &[Arc<dyn Tool>] = match &mcp_set {
            Some(s) => s.tools(),
            None => &empty,
        };

        let run_dir = create_run_tree(&self.config.research_base, run_id)?;
```

- [ ] **Step 2: Thread `extra_tools` through the 9 wave/phase call sites in `execute()`**

Append `, extra_tools` as the final argument to each of these calls in `execute()` (the implementer locates them via `grep -nE "dispatch_wave\(|verify_wave\(|dispatch_hypothesis_smiths\(|run_redteam_loop\(|run_eval_designers\(|run_synthesist\(" crates/research/src/orchestrator/mod.rs`):

- ~143 `dispatch_wave(...)` (scouts) → add `, extra_tools`
- ~151 `verify_wave(...)` (scouts) → add `, extra_tools`
- ~209 `dispatch_wave(...)` (gaps) → add `, extra_tools`
- ~217 `verify_wave(...)` (gaps) → add `, extra_tools`
- ~252 `dispatch_hypothesis_smiths(...)` → add `, extra_tools`
- ~273 `run_redteam_loop(...)` → add `, extra_tools` (after `&mut swarm`)
- ~301 `run_eval_designers(...)` → add `, extra_tools` (after `&mut swarm`)
- ~323 `run_synthesist(...)` → add `, extra_tools` (after `max_parallel`)
- ~338 `verify_wave(...)` (synth) → add `, extra_tools`

- [ ] **Step 3: Update the 6 `OrchestratorConfig { ... }` literals in `crates/research/tests/orchestrator.rs`**

Each literal (lines 531, 597, 699, 1492, 1576, 1647) gains a `mcp: None,` field. Example (apply the same one-line addition inside each literal's closing brace):

```rust
        OrchestratorConfig {
            research_base: tmp.path().to_path_buf(),
            agents_dir: agents.clone(),
            default_model: "fake-model".into(),
            max_parallel: 1,
            mcp: None,
        }
```

The implementer finds all 6 via `grep -n "OrchestratorConfig {" crates/research/tests/orchestrator.rs` and adds `mcp: None,` to each (matching the field order in the struct: after `max_parallel`).

- [ ] **Step 4: Verify GREEN (the `mcp: None` no-op guard)**

Run: `cargo test -p megaresearcher-research` → all green (every `execute()`-based test now passes `mcp: None`, so `extra_tools = &[]` and behavior is unchanged).
Run: `cargo clippy -p megaresearcher-research --all-targets -- -D warnings` → exit 0.
Run: `cargo fmt -p megaresearcher-research`.

- [ ] **Step 5: Commit**

```bash
git add crates/research/src/orchestrator/mod.rs crates/research/tests/orchestrator.rs
git commit -m "feat(rs): Phase 5 Task 7 — execute() connects ml-intern in pre-flight

OrchestratorConfig gains mcp: Option<McpServerConfig>. execute() connects
McpToolSet in pre-flight when set, building extra_tools threaded through
all 9 wave/phase call sites. mcp: None yields extra_tools = &[], so the
existing execute()-based tests stay byte-identical.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: MCP-in-the-loop deterministic integration test

**Files:**
- Modify: `crates/research/tests/orchestrator.rs` (append one test + one turn-builder helper)

**Goal:** Prove end-to-end that `run_worker` threads `extra_tools` into the `Worker`, the worker dispatches an `mcp__ml-intern__hf_papers` tool-use to the `McpTool`, the `McpTool` calls through `McpCaller` (a `FakeMcpCaller` records it), converts the result, and the worker continues to write its artifacts — all with no subprocess and no network.

- [ ] **Step 1: Write the failing test**

Append to `crates/research/tests/orchestrator.rs`. Add the imports needed at the top of the file's second import block (the `use std::sync::Arc; use claurst_api::...; use claurst_core::types::...; use serde_json::json; use common::fake_provider::FakeProvider; ...` block starting ~line 134):

```rust
use async_trait::async_trait;
use claurst_mcp::{CallToolResult, McpContent};
use megaresearcher_research::mcp::{McpCaller, McpError, McpToolSet};
use megaresearcher_research::worker_tools::Tool;
use std::sync::Mutex;
```

Then append the helper + test (after the existing tests, near the bottom of the file):

```rust
/// An MCP tool-use turn: same stream shape as `write_turn` but the ToolUse
/// block carries the MCP tool's `mcp__<server>__<tool>` name + the tool input,
/// and `stop_reason` is `ToolUse` so the worker dispatches and loops.
fn mcp_turn(name: &str, input: serde_json::Value) -> Vec<StreamEvent> {
    vec![
        StreamEvent::MessageStart {
            id: "m".into(),
            model: "fake".into(),
            usage: UsageInfo::default(),
        },
        StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::Text {
                text: String::new(),
            },
        },
        StreamEvent::TextDelta {
            index: 0,
            text: format!("calling {name}"),
        },
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::ContentBlockStart {
            index: 1,
            content_block: ContentBlock::ToolUse {
                id: "tu_mcp".into(),
                name: name.into(),
                input,
            },
        },
        StreamEvent::ContentBlockStop { index: 1 },
        StreamEvent::MessageDelta {
            stop_reason: Some(StopReason::ToolUse),
            usage: Some(UsageInfo::default()),
        },
        StreamEvent::MessageStop,
    ]
}

/// A `McpCaller` that records every call and returns a canned result.
struct FakeMcpCaller {
    calls: Mutex<Vec<(String, Option<serde_json::Value>)>>,
    result: CallToolResult,
}
#[async_trait]
impl McpCaller for FakeMcpCaller {
    async fn call_tool(
        &self,
        name: &str,
        arguments: Option<serde_json::Value>,
    ) -> Result<CallToolResult, McpError> {
        self.calls.lock().unwrap().push((name.to_string(), arguments));
        Ok(self.result.clone())
    }
}

#[tokio::test]
async fn run_worker_threads_mcp_tool_and_dispatches_call() {
    let tmp = tempdir().unwrap();
    let agents = tmp.path().join("agents");
    fs::create_dir_all(&agents).unwrap();
    fs::write(agents.join("literature-scout.md"), "# literature-scout\n\nbody\n").unwrap();
    let out_dir = tmp.path().join("literature-scout-1");
    fs::create_dir_all(&out_dir).unwrap();

    // The fake MCP caller records the bare-name call and returns one result.
    let caller = Arc::new(FakeMcpCaller {
        calls: Mutex::new(vec![]),
        result: CallToolResult {
            content: vec![McpContent::Text {
                text: "# trending papers\n...".into(),
            }],
            is_error: false,
        },
    }) as Arc<dyn McpCaller>;
    let set = McpToolSet::from_caller(
        "ml-intern",
        caller.clone(),
        vec![claurst_mcp::McpTool {
            name: "hf_papers".into(),
            description: Some("papers".into()),
            input_schema: json!({"type": "object"}),
        }],
    );
    assert_eq!(set.tools()[0].name(), "mcp__ml-intern__hf_papers");

    // Provider scripts: call the MCP tool, then write the three artifacts, end.
    let provider = FakeProvider::new(
        "fake",
        vec![
            mcp_turn(
                "mcp__ml-intern__hf_papers",
                json!({"operation": "trending", "limit": 1}),
            ),
            write_turn("output.md", "# output\n"),
            write_turn("manifest.yaml", "kind: scout\n"),
            write_turn("verification.md", "ok\n"),
            final_turn("done"),
        ],
    );

    let spec = WorkerSpec {
        name: "literature-scout-1".into(),
        role: "literature-scout".into(),
        output_dir: out_dir.clone(),
        shared_dir: tmp.path().to_path_buf(),
        prompt: build_prompt("spec", &[], None, &out_dir),
    };
    let outcome = run_worker(&spec, &agents, Arc::new(provider) as Arc<dyn LlmProvider>, "fake-model", set.tools())
        .await
        .expect("worker runs");
    assert_eq!(outcome.stop, WorkerStop::EndTurn);

    // The MCP tool was dispatched with the bare name + the scripted input.
    let calls = caller.calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 1, "expected one MCP call, got {calls:?}");
    assert_eq!(calls[0].0, "hf_papers");
    assert_eq!(calls[0].1, Some(json!({"operation": "trending", "limit": 1})));

    // The file-I/O tools still wrote the three artifacts.
    for f in ["output.md", "manifest.yaml", "verification.md"] {
        assert!(out_dir.join(f).exists(), "missing artifact {f}");
    }
}
```

- [ ] **Step 2: Run the test to verify it fails (RED)**

Run: `cargo test -p megaresearcher-research --test orchestrator run_worker_threads_mcp_tool_and_dispatches_call`
Expected: FAIL — before the threading (Tasks 5–7) it would not compile; after Tasks 5–7 the test should PASS. Since Tasks 5–7 already landed, this test is the GREEN confirmation of the threading. (If the worker does not dispatch the MCP tool, the scripted `mcp_turn`'s tool-use would be an "unknown tool" error result and the artifacts would still be written, but `calls.len()` would be 0 → assertion fails. That is the real failure mode this test guards against.)

- [ ] **Step 3: Verify GREEN**

Run: `cargo test -p megaresearcher-research --test orchestrator run_worker_threads_mcp_tool_and_dispatches_call` → 1 passed.
Run: `cargo test -p megaresearcher-research` → all green.
Run: `cargo clippy -p megaresearcher-research --all-targets -- -D warnings` → exit 0.
Run: `cargo fmt -p megaresearcher-research`.

- [ ] **Step 4: Commit**

```bash
git add crates/research/tests/orchestrator.rs
git commit -m "test(rs): Phase 5 Task 8 — MCP-in-the-loop deterministic integration test

run_worker_threads_mcp_tool_and_dispatches_call scripts a worker that
calls mcp__ml-intern__hf_papers (FakeMcpCaller records the bare-name call
and returns a canned result), then writes the three artifacts. Proves
the extra_tools threading + McpTool dispatch end-to-end with no
subprocess and no network.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Green sweep + final whole-branch review + push

**Files:** none (verification + review + push)

- [ ] **Step 1: Full workspace green sweep**

From the repo root:
```bash
cargo fmt -p megaresearcher-research --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
Expected: fmt clean; clippy exit 0; all tests pass. Record the test count (Phase 4b baseline 1474 + Phase 5 additions: 12 mcp unit tests + 1 ignored smoke test + 1 MCP-in-loop integration test = +14 collected, +13 run).

- [ ] **Step 2: Edit-scope verification**

```bash
git diff --stat 1619738..HEAD -- lib/ 'tests/test_*.py' skills/ agents/ .claude-plugin/ commands/ hooks/ mcp/ tools/ml-intern .mcp.json
```
Expected: EMPTY (the v0 port-reference is untouched). The only files changed are under `crates/research/` (and `Cargo.lock` may shift from the new `claurst-mcp` research dep — that is expected and in-scope). Confirm `mcp/server.py` and `tools/ml-intern/` are unchanged.

- [ ] **Step 3: Structure + discipline check**

- `crates/research/src/lib.rs` has `pub mod mcp;` and NO crate-root `pub use` re-exports.
- `crates/research/src/mcp.rs` uses `pub` on `McpError`, `McpCaller`, `McpClientCaller`, `mcp_result_to_tool_result`, `McpTool`, `McpToolSet`, `ml_intern_config`, `prewarm`; `McpTool::from_server_tool` and `mcp_project_dir` are private (internal).
- No `api`-crate or `claurst-mcp`-crate changes (only the research crate consumed their public APIs).
- Naming is `mcp__<server>__<tool>` everywhere (double underscore).
- Banned-phrase scan: `grep -rnE "load-bearing|this is doing a lot of work|honestly|to be honest" crates/research/src/mcp.rs crates/research/tests/mcp_smoke.rs crates/research/tests/orchestrator.rs` → no hits (emphatic "real" only as a factual modifier in the smoke-test name/ignore reason).
- All Phase 5 commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>`.

- [ ] **Step 4: Final whole-branch review**

Run `scripts/review-package 1619738 HEAD` (from the subagent-driven-development skill directory) to produce the branch diff package, then dispatch the final code reviewer on the most capable available model using the requesting-code-review `code-reviewer.md` template, pointing it at the package path + this plan + the `.superpowers/sdd/progress.md` Phase 5 section. Have it verify specifically:
- The determinism model still holds (`&[]` keeps fake-provider tests byte-identical; the MCP-in-loop test is deterministic via `FakeMcpCaller`).
- The `McpCaller` seam is sound (no dependency on claurst's private `from_backend`).
- The `mcp__<server>__<tool>` naming matches the byte-locked fixtures and resolves through worker dispatch.
- `execute()` holds `mcp_set` across the run without borrow/Send issues; the subprocess lifetime is correct.
- Edit scope is `crates/research/` + `Cargo.lock` only; v0 untouched.
- Triage any findings; fix Critical/Important (one fix subagent, re-review); record deferred Minors in the ledger.

- [ ] **Step 5: Push + record**

On `main` (confirm `git branch --show-current` is `main`):
```bash
git push origin main
```
Append to `.superpowers/sdd/progress.md`:
```
=== Phase 5 PUSHED to origin/main (<4b-tip>..<5-tip>) ===
```
with the per-task completion lines recorded as each task was reviewed.

---

## Self-Review (run before handing off)

**1. Spec coverage.** Design §13 item 5 ("MCP wiring. Rust mcp client drives the ml-intern subprocess; uv pre-warm; 9-tool smoke test. Workers can now actually call hf_papers.") maps to: T1–T3 (client + seam + prewarm), T4 (9-tool smoke), T5–T7 (workers get the tools), T8 (deterministic proof). Design §9 (MCP client contract — config shape, env interpolation, stdio spawn, routing `mcp__server__tool`) is honored: `ml_intern_config` mirrors `.mcp.json`, `expand_server_config` resolves `${VAR}`, `McpClient::connect_stdio` spawns, `McpTool` routes by name. Design §202 (preflight connects ml-intern, pre-warmed) → T7 `execute()` connect in pre-flight + T3 prewarm. Design §216 (MCP transport failures → clean escalation) → T7 `OrchestratorError::Preflight("ml-intern unreachable: ...")`.

**2. Placeholder scan.** Every code step shows complete code. No "TBD"/"add error handling". The phase-function edits in T6 show exact new signatures + the precise "append `extra_tools` to each internal `dispatch_wave`/`verify_wave`/`run_worker` call" rule with grep locators — this is mechanical and unambiguous, not a placeholder.

**3. Type consistency.** `extra_tools: &[Arc<dyn Tool>]` is the same type at every signature (run_worker, dispatch_wave, verify_wave, the 5 phase fns, execute's local). `McpCaller::call_tool` returns `Result<CallToolResult, McpError>` everywhere. `McpToolSet::tools() -> &[Arc<dyn Tool>]` matches the `extra_tools` type at the call sites. `McpServerConfig` is `claurst_core::config::McpServerConfig` everywhere (the type claurst-mcp's `McpClient::connect_stdio` expects).

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-27-megaresearcher-rs-phase5-mcp-wiring.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**