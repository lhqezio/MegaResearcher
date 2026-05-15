"""Parse an eval-designer protocol markdown into a structured dict.

Returns {} when no recognizable structure is found.

CLI:
    python3 -m lib.paper_chain.protocol_parser <protocol-path>
        → prints JSON; exit 0 on parse success, exit 1 on empty result
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

_SUBSTRATE_RE = re.compile(r"^\s*[-*]\s*Substrate:\s*(.+?)\s*$", re.MULTILINE)
_SAMPLE_RE = re.compile(r"^\s*[-*]\s*Sample size:\s*(\d+)", re.MULTILINE)
_SEED_RE = re.compile(r"^\s*[-*]\s*Seed:\s*(\d+)", re.MULTILINE)
_BASELINES_RE = re.compile(r"^\s*[-*]\s*Baselines?:\s*(.+?)\s*$", re.MULTILINE)
_METRIC_RE = re.compile(r"^\s*[-*]\s*Metric[s]?:\s*(.+?)\s*$", re.MULTILINE)
_DECISION_RE = re.compile(r"^\s*[-*]\s*Decision rule[s]?:\s*(.+?)\s*$", re.MULTILINE)


def parse_protocol(protocol_path: Path) -> dict:
    """Parse the protocol file; return structured dict or {} if unrecognized."""
    text = protocol_path.read_text(encoding="utf-8")
    substrate_m = _SUBSTRATE_RE.search(text)
    if substrate_m is None:
        return {}
    out: dict = {"substrate": substrate_m.group(1).strip()}

    sample_m = _SAMPLE_RE.search(text)
    out["sample_size"] = int(sample_m.group(1)) if sample_m else None

    seed_m = _SEED_RE.search(text)
    out["seed"] = int(seed_m.group(1)) if seed_m else None

    baselines_m = _BASELINES_RE.search(text)
    out["baselines"] = (
        [b.strip() for b in baselines_m.group(1).split(",")]
        if baselines_m
        else []
    )

    out["metrics"] = [m.group(1).strip() for m in _METRIC_RE.finditer(text)]
    out["decision_rules"] = [
        {"raw": d.group(1).strip()} for d in _DECISION_RE.finditer(text)
    ]
    return out


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: protocol_parser.py <protocol-path>", file=sys.stderr)
        return 2
    result = parse_protocol(Path(argv[1]))
    print(json.dumps(result, indent=2))
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
