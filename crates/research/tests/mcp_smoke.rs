//! Real ml-intern subprocess smoke test (Phase 5). Connects to the actual
//! Python MCP server via uv, asserts the 9 expected tools surface, and calls
//! one read-only tool that needs no API key. Ignored by default — run with:
//!   cargo test -p megaresearcher-research --test mcp_smoke -- --ignored
//! when uv, the ml-intern venv, and (for HF tools) HF_TOKEN are available.

use std::env;
use std::path::PathBuf;

use serde_json::json;

use megaresearcher_research::mcp::{ml_intern_config, McpToolSet};

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
    let set = McpToolSet::connect(&config)
        .await
        .expect("connect ml-intern");

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
    let result = web
        .call(json!({"query": "hello world", "max_results": 1}))
        .await;
    assert!(
        !result.is_error,
        "web_search returned an error: {}",
        result.content
    );
    assert!(
        !result.content.is_empty(),
        "web_search returned empty content"
    );
}
