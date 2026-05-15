"""CiteME runner — SKELETON.

Real implementation is deferred to follow-up plan SP2a.3.
"""
from __future__ import annotations
from lib.runners._base import make_failed_result

SUBSTRATE = "CiteME"


def run(params: dict) -> dict:
    return make_failed_result(
        hypothesis_id=params.get("hypothesis_id", "unknown"),
        substrate=SUBSTRATE,
        failure_code="runner_not_implemented",
        failure_message=(
            "CiteME runner skeleton — real implementation pending in SP2a.3 "
            "follow-up plan."
        ),
        seed=params.get("seed"),
    )


if __name__ == "__main__":
    import json
    print(json.dumps(run({"hypothesis_id": "cli"}), indent=2))
