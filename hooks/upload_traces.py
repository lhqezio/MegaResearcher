#!/usr/bin/env python3
"""Claude Code SessionEnd hook: upload session transcript to a private HF dataset.

Replicates ml-intern's session-trace-upload behavior using ml-intern's own
`session_uploader.py` script (which supports `--format claude_code`). The
script is invoked through MegaResearcher's `mcp/` uv project, which has
ml-intern installed editable as a path dependency, so we reuse that venv
rather than syncing a separate one against `tools/ml-intern/`.

Configuration (env vars; set in shell or load via your usual mechanism):
  ML_INTERN_TRACES_REPO     HF dataset repo id, e.g. "you/ml-intern-sessions".
                            REQUIRED — if unset, this hook is a silent no-op.
  HF_TOKEN                  HF write token. REQUIRED.
  ML_INTERN_TRACES_PRIVATE  "true" (default) or "false".

Plugin-root resolution — first match wins:
  1. CLAUDE_PLUGIN_ROOT env var (set by Claude Code when running plugin hooks)
  2. Walk up from this script looking for the MegaResearcher repo root
     (sibling `mcp/`, `tools/ml-intern/`, `agents/`, etc.)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _plugin_root() -> Path | None:
    explicit = os.environ.get("CLAUDE_PLUGIN_ROOT")
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "mcp" / "pyproject.toml").exists() and (
            ancestor / "tools" / "ml-intern"
        ).exists():
            return ancestor
    return None


def main() -> None:
    repo_id = os.environ.get("ML_INTERN_TRACES_REPO")
    if not repo_id:
        return  # Trace upload is opt-in; silent no-op when unconfigured.

    if not os.environ.get("HF_TOKEN"):
        print("upload_traces: HF_TOKEN not set; skipping", file=sys.stderr)
        return

    raw = sys.stdin.read()
    if not raw.strip():
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return

    transcript_path = payload.get("transcript_path")
    if not transcript_path or not Path(transcript_path).exists():
        return

    plugin_root = _plugin_root()
    if plugin_root is None:
        print("upload_traces: MegaResearcher plugin root not found; skipping", file=sys.stderr)
        return

    mcp_project = plugin_root / "mcp"
    private = os.environ.get("ML_INTERN_TRACES_PRIVATE", "true")

    cmd = [
        "uv", "run", "--project", str(mcp_project),
        "python", "-m", "agent.core.session_uploader",
        "upload", transcript_path, repo_id,
        "--format", "claude_code",
        "--private", private,
    ]

    try:
        subprocess.run(cmd, check=False, timeout=120, capture_output=True)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"upload_traces: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
