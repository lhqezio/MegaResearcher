"""Manual SP2a dispatch test using FakeSandboxBackend.

Run from plugin root:
    python3 tests/manual_dispatch.py

Exercises the Phase 6.5 orchestrator path end-to-end on the SP1 fixture
without spinning up real Vercel Sandbox VMs. With the 5 skeleton runners,
every experiment is expected to return status=failed with one of:
- runner_not_implemented (substrate matches a skeleton)
- unsupported_substrate (substrate not registered)
- failed_parse (protocol can't be parsed)
"""
from __future__ import annotations
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.sandbox import set_backend, FakeSandboxBackend
from lib.paper_chain.experiment import dispatch_experiment

set_backend(FakeSandboxBackend(canned_id="sb_manual"))

FIXTURE = PLUGIN_ROOT / "tests" / "fixtures" / "paper-chain"

for h in ("S1", "S2", "S3"):
    out = FIXTURE / "paper" / "experiments" / h
    out.mkdir(parents=True, exist_ok=True)
    path = dispatch_experiment(
        hypothesis_id=h,
        protocol_path=FIXTURE / f"eval-designer-{h}" / "output.md",
        output_dir=out,
        sandbox_budget_usd=5.0,
        api_budget_usd=5.0,
    )
    print(f"{h}: wrote {path}")
