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
use claurst_mcp::{CallToolResult, McpClient};
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
}
