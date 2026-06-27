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
use claurst_core::config::McpServerConfig;
use claurst_mcp::{expand_server_config, CallToolResult, McpClient, McpTool as ServerMcpTool};
use serde_json::Value;

use crate::worker_tools::{Tool, ToolResult};

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
    fn from_server_tool(
        server_name: &str,
        tool: ServerMcpTool,
        caller: Arc<dyn McpCaller>,
    ) -> Arc<dyn Tool> {
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

/// Task 3 placeholder. The production `prewarm` runs `uv sync --project <mcp>`
/// before `McpClient::connect_stdio`; this no-op stub is overwritten by Task 3
/// (which lands `ml_intern_config`, `mcp_project_dir`, and the `uv sync` body).
/// Required so `McpToolSet::connect` compiles in the T2-only state without
/// breaking the green-bar.
async fn prewarm(_config: &McpServerConfig) -> Result<(), McpError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker_tools::Tool;
    use claurst_mcp::McpContent;
    use serde_json::json;
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
            self.calls
                .lock()
                .unwrap()
                .push((name.to_string(), arguments));
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
        let set = McpToolSet::from_caller(
            "ml-intern",
            caller,
            vec![server_tool("hf_papers", "papers")],
        );
        assert_eq!(set.tools().len(), 1);
        assert_eq!(set.tools()[0].name(), "mcp__ml-intern__hf_papers");
        assert_eq!(set.tools()[0].description(), "papers");
    }

    #[tokio::test]
    async fn mcp_tool_call_routes_to_caller_with_bare_name_and_input() {
        let caller = Arc::new(FakeMcpCaller::new(text_result("# 1 result", false)));
        let set = McpToolSet::from_caller(
            "ml-intern",
            caller.clone() as Arc<dyn McpCaller>,
            vec![server_tool("hf_papers", "papers")],
        );
        let tool = set.tools()[0].clone();
        let tr = tool
            .call(json!({"operation": "trending", "limit": 1}))
            .await;
        assert!(!tr.is_error);
        assert_eq!(tr.content, "# 1 result");
        let calls = caller.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "hf_papers"); // bare name, not the prefixed name
        assert_eq!(
            calls[0].1,
            Some(json!({"operation": "trending", "limit": 1}))
        );
    }

    #[tokio::test]
    async fn mcp_tool_call_passes_none_for_null_input() {
        let caller = Arc::new(FakeMcpCaller::new(text_result("ok", false)));
        let set = McpToolSet::from_caller(
            "ml-intern",
            caller.clone() as Arc<dyn McpCaller>,
            vec![server_tool("web_search", "search")],
        );
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
            async fn call_tool(
                &self,
                _name: &str,
                _arguments: Option<Value>,
            ) -> Result<CallToolResult, McpError> {
                Err(McpError("server crashed".into()))
            }
        }
        let caller = Arc::new(ErrCaller) as Arc<dyn McpCaller>;
        let set = McpToolSet::from_caller(
            "ml-intern",
            caller,
            vec![server_tool("hf_papers", "papers")],
        );
        let tr = set.tools()[0].clone().call(json!({})).await;
        assert!(tr.is_error);
        assert!(
            tr.content.contains("mcp__ml-intern__hf_papers"),
            "content was: {}",
            tr.content
        );
        assert!(
            tr.content.contains("server crashed"),
            "content was: {}",
            tr.content
        );
    }

    #[test]
    fn mcp_tool_set_preserves_server_order_and_marks_read_only_concurrency_safe() {
        let caller = Arc::new(FakeMcpCaller::new(text_result("ok", false))) as Arc<dyn McpCaller>;
        let set = McpToolSet::from_caller(
            "ml-intern",
            caller,
            vec![
                server_tool("hf_papers", "a"),
                server_tool("web_search", "b"),
            ],
        );
        assert_eq!(set.tools().len(), 2);
        assert_eq!(set.tools()[0].name(), "mcp__ml-intern__hf_papers");
        assert_eq!(set.tools()[1].name(), "mcp__ml-intern__web_search");
        assert!(set.tools()[0].is_read_only());
        assert!(set.tools()[0].is_concurrency_safe());
    }
}
