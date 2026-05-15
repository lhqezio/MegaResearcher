"""SPECS-Review-Benchmark runner — SKELETON.

Real implementation is deferred to follow-up plan SP2a.1.
This skeleton satisfies the Runner schema contract by returning a
schema-valid `failed_runner_not_implemented` result.
"""
from __future__ import annotations
from lib.runners._base import make_failed_result

SUBSTRATE = "SPECS-Review-Benchmark"


def run(params: dict) -> dict:
    return make_failed_result(
        hypothesis_id=params.get("hypothesis_id", "unknown"),
        substrate=SUBSTRATE,
        failure_code="runner_not_implemented",
        failure_message=(
            "SPECS runner skeleton — real implementation pending in SP2a.1 "
            "follow-up plan. See "
            "docs/superpowers/specs/2026-05-15-experimentalist-sandbox-design.md "
            "for the substrate's expected behavior."
        ),
        seed=params.get("seed"),
    )


if __name__ == "__main__":
    import json
    print(json.dumps(run({"hypothesis_id": "cli"}), indent=2))
