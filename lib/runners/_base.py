"""Runner base class and result-schema validator.

Each lib/runners/<substrate>/runner.py module exports a `run(params: dict) -> dict`
function that returns a dict matching REQUIRED_FIELDS. Skeleton runners return
a failed-result dict (status='failed', failure_code='runner_not_implemented').

CLI:
    python3 -m lib.runners._base validate <result-json-path>
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REQUIRED_FIELDS = [
    "hypothesis_id",
    "substrate",
    "metric_name",
    "baseline_value",
    "treatment_value",
    "p_value",
    "ci_low",
    "ci_high",
    "n",
    "seed",
    "runtime_seconds",
    "cost_usd",
    "status",
    "failure_code",
    "failure_message",
]

VALID_STATUSES = {"completed", "failed", "escalated", "skipped"}


def validate_result(result: dict) -> tuple[bool, list[str]]:
    """Return (ok, errors) — schema validation on a runner result dict."""
    errors: list[str] = []
    for f in REQUIRED_FIELDS:
        if f not in result:
            errors.append(f"missing required field: {f}")
    if "status" in result and result["status"] not in VALID_STATUSES:
        errors.append(f"invalid status: {result['status']!r} (expected one of {VALID_STATUSES})")
    return (len(errors) == 0, errors)


def make_failed_result(
    hypothesis_id: str,
    substrate: str,
    failure_code: str,
    failure_message: str,
    seed: int | None = None,
) -> dict:
    """Construct a schema-valid failed result dict."""
    return {
        "hypothesis_id": hypothesis_id,
        "substrate": substrate,
        "metric_name": None,
        "baseline_value": None,
        "treatment_value": None,
        "p_value": None,
        "ci_low": None,
        "ci_high": None,
        "n": None,
        "seed": seed,
        "runtime_seconds": 0.0,
        "cost_usd": 0.0,
        "status": "failed",
        "failure_code": failure_code,
        "failure_message": failure_message,
    }


class Runner:
    """Base class. Subclasses set `substrate` and implement `run`."""

    substrate: str = ""

    def run(self, params: dict) -> dict:
        raise NotImplementedError("Subclasses must implement run().")


def _main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] != "validate":
        print("usage: _base.py validate <result-json-path>", file=sys.stderr)
        return 2
    result = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    ok, errs = validate_result(result)
    if not ok:
        for e in errs:
            print(e, file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
