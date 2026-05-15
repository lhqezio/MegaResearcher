"""Parse the VERDICT line from a review-vN.md file.

A valid verdict line matches exactly `^VERDICT: (APPROVE|REVISE|KILL)$`.
Returns the verdict word as a string, or None if no valid verdict found.

CLI:
    python3 -m lib.paper_chain.verdict path/to/review-v1.md
        → prints verdict word or 'NONE'; exit 0 on parse success, 1 if NONE
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

VALID = {"APPROVE", "REVISE", "KILL"}
_VERDICT_RE = re.compile(r"^VERDICT: (APPROVE|REVISE|KILL)$", re.MULTILINE)


def parse_verdict(review_path: Path) -> str | None:
    """Return verdict word from the given review file, or None if not found.

    Scans the whole file; takes the first match. The exact-line regex
    (anchored ^ and $ via MULTILINE) rejects lowercase, embedded, or
    unknown verdict words.
    """
    text = review_path.read_text(encoding="utf-8")
    m = _VERDICT_RE.search(text)
    if m is None:
        return None
    return m.group(1)


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verdict.py <review-path>", file=sys.stderr)
        return 2
    v = parse_verdict(Path(argv[1]))
    print(v if v else "NONE")
    return 0 if v else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
