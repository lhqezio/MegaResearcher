"""Scaffold the paper/ subdirectory under a swarm run dir.

CLI:
    python3 -m lib.paper_chain.scaffold <run-dir>
        → prints the created path; exit 0
"""
from __future__ import annotations
import sys
from pathlib import Path


def scaffold_paper_dir(run_dir: Path) -> Path:
    """Create <run-dir>/paper/ with an empty revision-log.jsonl.

    Idempotent: safe to call multiple times; preserves existing files.
    """
    paper = run_dir / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    log = paper / "revision-log.jsonl"
    if not log.exists():
        log.touch()
    return paper


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: scaffold.py <run-dir>", file=sys.stderr)
        return 2
    p = scaffold_paper_dir(Path(argv[1]))
    print(p)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
