"""Smoke test: every wrapped ml-intern handler imports cleanly, and the MCP
server registers exactly the expected tool set.

Run from the plugin root:

    cd mcp && uv sync   # one-time
    cd .. && python tests/test_mcp_imports.py
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_TOOLS = {
    "hf_papers",
    "hf_inspect_dataset",
    "hf_docs_explore",
    "hf_docs_fetch",
    "hf_repo_files",
    "github_examples",
    "github_list_repos",
    "github_read_file",
    "web_search",
}


def main() -> int:
    # Run the import + introspection inside the mcp uv project so we get its venv.
    code = f"""
import sys
sys.path.insert(0, {str(PLUGIN_ROOT / "tools" / "ml-intern")!r})

# Import every handler the server wraps. Failure here means upstream ml-intern
# renamed a handler or removed a module.
from agent.tools.papers_tool import hf_papers_handler
from agent.tools.dataset_tools import hf_inspect_dataset_handler
from agent.tools.docs_tools import explore_hf_docs_handler, hf_docs_fetch_handler
from agent.tools.web_search_tool import web_search_handler
from agent.tools.github_find_examples import github_find_examples_handler
from agent.tools.github_list_repos import github_list_repos_handler
from agent.tools.github_read_file import github_read_file_handler
from agent.tools.hf_repo_files_tool import hf_repo_files_handler

# Now load server.py and confirm tool registration.
sys.path.insert(0, {str(PLUGIN_ROOT / "mcp")!r})
import server
import asyncio
tools = asyncio.run(server.mcp.list_tools())
names = sorted(t.name for t in tools)
print("REGISTERED:", " ".join(names))
"""
    result = subprocess.run(
        ["uv", "run", "--project", str(PLUGIN_ROOT / "mcp"), "python", "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print("FAIL: subprocess errored", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 1

    output = result.stdout.strip().splitlines()
    registered_line = next((l for l in output if l.startswith("REGISTERED:")), None)
    if not registered_line:
        print("FAIL: no REGISTERED line in output", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        return 1

    registered = set(registered_line.replace("REGISTERED:", "").split())
    if registered != EXPECTED_TOOLS:
        missing = EXPECTED_TOOLS - registered
        extra = registered - EXPECTED_TOOLS
        print(f"FAIL: tool set mismatch. Missing: {missing}. Extra: {extra}", file=sys.stderr)
        return 1

    print(f"PASS: all {len(EXPECTED_TOOLS)} tools registered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
