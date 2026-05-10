"""MCP server exposing ml-intern's research tools to Claude Code.

We import ml-intern's tool modules directly rather than vendoring code, so
upstream fixes flow in via `git pull` in tools/ml-intern/.

v1 surface (research-focused):
  - hf_papers           : arxiv/HF papers — search, read, recs, related artifacts
  - hf_inspect_dataset  : inspect a specific HF dataset (rows, splits, schema)
  - hf_docs_explore     : search/list within an HF docs endpoint
  - hf_docs_fetch       : fetch full markdown of a docs page
  - hf_repo_files       : list files in an HF repo (model/dataset/space)
  - github_examples     : GitHub code-search for usage examples
  - github_list_repos   : list/search GitHub repos
  - github_read_file    : read a file from a GitHub repo
  - web_search          : general web search (DuckDuckGo backend)

What we deliberately skip in v1:
  - sandbox_tool        : ml-intern uses HF Spaces remote sandboxes;
                          CC's Bash already covers local sandboxed execution.
  - research_tool       : runs its own LLM internally; CC's Agent/subagents
                          replace this without spawning a second agent loop.
  - jobs_tool           : HF training-job submission; reintroduce in v2 once
                          we want to push training runs (needs approval gates).
  - local_tools         : Bash/Read/Write/Edit — CC has these natively.
  - hf_repo_git_tool    : clones/writes; defer.
  - private_hf_repo     : writes; defer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Make the cloned ml-intern repo importable. Standard layout assumed:
#   <project>/tools/ml-intern/    (the clone)
#   <project>/ml_intern_cc/       (this server)
ML_INTERN_ROOT = (Path(__file__).resolve().parent.parent / "tools" / "ml-intern")
if not ML_INTERN_ROOT.exists():
    sys.exit(f"ml-intern clone not found at {ML_INTERN_ROOT}")
sys.path.insert(0, str(ML_INTERN_ROOT))

# Load HF_TOKEN / GITHUB_TOKEN from a .env beside this file (or process env).
load_dotenv(Path(__file__).resolve().parent / ".env")

# Imports must come after the sys.path tweak above.
from agent.tools.papers_tool import hf_papers_handler  # noqa: E402
from agent.tools.dataset_tools import hf_inspect_dataset_handler  # noqa: E402
from agent.tools.docs_tools import (  # noqa: E402
    explore_hf_docs_handler,
    hf_docs_fetch_handler,
)
from agent.tools.web_search_tool import web_search_handler  # noqa: E402
from agent.tools.github_find_examples import github_find_examples_handler  # noqa: E402
from agent.tools.github_list_repos import github_list_repos_handler  # noqa: E402
from agent.tools.github_read_file import github_read_file_handler  # noqa: E402
from agent.tools.hf_repo_files_tool import hf_repo_files_handler  # noqa: E402


class _Session:
    """Minimal duck-typed session for ml-intern handlers that read `session.hf_token`.

    The real ml-intern Session object carries much more state (cost tracking,
    conversation history, etc.), but the tool handlers only touch `hf_token`.
    """

    def __init__(self) -> None:
        self.hf_token = os.environ.get("HF_TOKEN")


_SESSION = _Session()


def _unwrap(result: Any) -> str:
    """ml-intern handlers return (text, ok) tuples. Surface text either way;
    raise on error so MCP propagates a tool error to the model."""
    if isinstance(result, tuple) and len(result) == 2:
        text, ok = result
        if not ok:
            raise RuntimeError(text)
        return text
    if isinstance(result, dict):
        if result.get("isError"):
            raise RuntimeError(result.get("formatted", "tool error"))
        return result.get("formatted", "")
    return str(result)


mcp = FastMCP("ml-intern")


@mcp.tool()
async def hf_papers(
    operation: str,
    arxiv_id: str | None = None,
    query: str | None = None,
    limit: int | None = None,
    section: str | None = None,
    sort: str | None = None,
    sort_by: str | None = None,
    direction: str | None = None,
    date: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    categories: list[str] | None = None,
    min_citations: int | None = None,
    positive_ids: list[str] | None = None,
    negative_ids: list[str] | None = None,
) -> str:
    """Search, read, and explore papers on HF Papers + arXiv + Semantic Scholar.

    Operations (from ml-intern's papers_tool):
      trending, search, paper_details, read_paper, citation_graph,
      find_datasets, find_models, find_collections, find_all_resources,
      snippet_search, recommend.

    Per-operation kwargs (only relevant ones need to be set):
      - search:           query, limit, date_from, date_to, categories, min_citations, sort_by
      - paper_details:    arxiv_id
      - read_paper:       arxiv_id, section
      - citation_graph:   arxiv_id, direction, limit
      - find_datasets:    arxiv_id, sort, limit
      - find_models:      arxiv_id, sort, limit
      - find_collections: arxiv_id, limit
      - find_all_resources: arxiv_id, limit
      - snippet_search:   arxiv_id, query, limit
      - recommend:        positive_ids, negative_ids, limit
      - trending:         date, limit
    """
    args: dict[str, Any] = {"operation": operation}
    for k, v in (
        ("arxiv_id", arxiv_id),
        ("query", query),
        ("limit", limit),
        ("section", section),
        ("sort", sort),
        ("sort_by", sort_by),
        ("direction", direction),
        ("date", date),
        ("date_from", date_from),
        ("date_to", date_to),
        ("categories", categories),
        ("min_citations", min_citations),
        ("positive_ids", positive_ids),
        ("negative_ids", negative_ids),
    ):
        if v is not None:
            args[k] = v
    return _unwrap(await hf_papers_handler(args))


@mcp.tool()
async def hf_inspect_dataset(
    dataset: str,
    config: str | None = None,
    split: str | None = None,
    sample_rows: int = 3,
) -> str:
    """Inspect a Hugging Face dataset: schema, splits, and sample rows."""
    args: dict[str, Any] = {"dataset": dataset, "sample_rows": sample_rows}
    if config:
        args["config"] = config
    if split:
        args["split"] = split
    return _unwrap(await hf_inspect_dataset_handler(args, session=_SESSION))


@mcp.tool()
async def hf_docs_explore(
    endpoint: str,
    query: str | None = None,
    max_results: int | None = None,
) -> str:
    """Explore HF docs: search/list within an endpoint (e.g. transformers, hub, gradio)."""
    args: dict[str, Any] = {"endpoint": endpoint}
    if query:
        args["query"] = query
    if max_results is not None:
        args["max_results"] = max_results
    return _unwrap(await explore_hf_docs_handler(args, session=_SESSION))


@mcp.tool()
async def hf_docs_fetch(url: str) -> str:
    """Fetch full markdown of an HF docs page (the .md is appended if absent)."""
    return _unwrap(await hf_docs_fetch_handler({"url": url}, session=_SESSION))


@mcp.tool()
async def hf_repo_files(repo_id: str, repo_type: str = "model", revision: str | None = None) -> str:
    """List files in an HF repo. repo_type: model | dataset | space."""
    args: dict[str, Any] = {"repo_id": repo_id, "repo_type": repo_type}
    if revision:
        args["revision"] = revision
    return _unwrap(await hf_repo_files_handler(args, session=_SESSION))


@mcp.tool()
async def github_examples(query: str, language: str | None = None, max_results: int = 10) -> str:
    """Search GitHub code for usage examples. `language` optional filter."""
    args: dict[str, Any] = {"query": query, "max_results": max_results}
    if language:
        args["language"] = language
    return _unwrap(await github_find_examples_handler(args))


@mcp.tool()
async def github_list_repos(query: str, max_results: int = 10) -> str:
    """Search GitHub repositories matching a query."""
    return _unwrap(
        await github_list_repos_handler({"query": query, "max_results": max_results})
    )


@mcp.tool()
async def github_read_file(repo: str, path: str, ref: str | None = None) -> str:
    """Read a file from a GitHub repo. `repo` is "owner/name", `ref` defaults to default branch."""
    args: dict[str, Any] = {"repo": repo, "path": path}
    if ref:
        args["ref"] = ref
    return _unwrap(await github_read_file_handler(args))


@mcp.tool()
async def web_search(query: str, max_results: int = 10) -> str:
    """General web search via ml-intern's web_search_tool (DuckDuckGo backend)."""
    return _unwrap(
        await web_search_handler({"query": query, "max_results": max_results}, session=_SESSION)
    )


if __name__ == "__main__":
    mcp.run()
