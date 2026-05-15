"""Runaway-revision regression detector.

Compares two consecutive review files; flags regression when the count of
NEW weaknesses in v2 (not seen in v1) is greater than or equal to the count
of CLOSED weaknesses (v1 items absent from v2).

Weaknesses are identified by their leading tag (e.g., 'W1:', 'W2:'). The
tag is the substring before the first ':'. This is a simple convention,
not a perfect identity check — peer-reviewer agents are instructed to use
W<int>: tags so identity tracking works.

CLI:
    python3 -m lib.paper_chain.regression <review-v1> <review-v2>
        → prints 'REGRESSION' (exit 1) or 'OK' (exit 0) plus counts on stderr
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

_WEAKNESS_LINE_RE = re.compile(r"^- (W\d+):", re.MULTILINE)
_SECTION_RE = re.compile(
    r"^## Weaknesses\s*$\n(.*?)(?=^## |\Z)",
    re.MULTILINE | re.DOTALL,
)


def extract_weaknesses(review_path: Path) -> list[str]:
    """Return list of full weakness bullet lines from a review file.

    Looks for a '## Weaknesses' section and returns each '- W<int>: ...' line.
    """
    text = review_path.read_text(encoding="utf-8")
    section_match = _SECTION_RE.search(text)
    if not section_match:
        return []
    section = section_match.group(1)
    out: list[str] = []
    for raw in section.splitlines():
        stripped = raw.strip()
        if _WEAKNESS_LINE_RE.match(stripped):
            # Drop leading "- " so callers see 'W1: ...' directly.
            out.append(stripped[2:].strip())
    return out


def _tag(line: str) -> str:
    # 'W1: Insufficient...' -> 'W1'; line passed in is '- W1: ...'
    body = line.lstrip("- ").strip()
    return body.split(":", 1)[0]


def detect_regression(v1_path: Path, v2_path: Path) -> tuple[bool, int, int]:
    """Return (flagged, closed_count, new_count).

    closed_count = items in v1 with tag NOT in v2
    new_count = items in v2 with tag NOT in v1
    flagged when new_count >= closed_count (and at least one of each, to avoid
    flagging trivial pass-throughs).
    """
    v1 = extract_weaknesses(v1_path)
    v2 = extract_weaknesses(v2_path)
    tags_v1 = {_tag(w) for w in v1}
    tags_v2 = {_tag(w) for w in v2}
    closed = tags_v1 - tags_v2
    new = tags_v2 - tags_v1
    flagged = len(new) >= len(closed) and len(new) > 0
    return (flagged, len(closed), len(new))


def _main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: regression.py <review-v1> <review-v2>", file=sys.stderr)
        return 2
    flagged, closed, new = detect_regression(Path(argv[1]), Path(argv[2]))
    print(f"closed={closed} new={new}", file=sys.stderr)
    if flagged:
        print("REGRESSION")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
