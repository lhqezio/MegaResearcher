"""Phase 9 finalize: produce paper.md (latest draft) and paper-history.md.

Strategy:
- Find the highest-numbered draft-vN.md in paper/
- Copy its content to paper.md (no symlink — cross-platform safer)
- Concatenate all review-vN.md files + revision-log.jsonl + final verdict
  marker into paper-history.md

CLI:
    python3 -m lib.paper_chain.finalize <paper-dir> <final-verdict>
        → exit 0; prints path to paper.md
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

_DRAFT_RE = re.compile(r"^draft-v(\d+)\.md$")
_REVIEW_RE = re.compile(r"^review-v(\d+)\.md$")


def _latest_draft(paper_dir: Path) -> Path:
    drafts = []
    for p in paper_dir.iterdir():
        m = _DRAFT_RE.match(p.name)
        if m:
            drafts.append((int(m.group(1)), p))
    if not drafts:
        raise FileNotFoundError(f"No draft-vN.md in {paper_dir}")
    drafts.sort()
    return drafts[-1][1]


def _ordered_reviews(paper_dir: Path) -> list[Path]:
    reviews = []
    for p in paper_dir.iterdir():
        m = _REVIEW_RE.match(p.name)
        if m:
            reviews.append((int(m.group(1)), p))
    reviews.sort()
    return [p for _, p in reviews]


def finalize_paper(paper_dir: Path, final_verdict: str) -> Path:
    """Produce paper.md (latest draft) and paper-history.md.

    Returns path to paper.md.
    """
    latest = _latest_draft(paper_dir)
    paper_md = paper_dir / "paper.md"
    paper_md.write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")

    history_parts = [f"# Paper history\n\nFinal verdict: {final_verdict}\n"]
    for r in _ordered_reviews(paper_dir):
        history_parts.append(f"\n---\n\n## {r.name}\n\n{r.read_text(encoding='utf-8')}")
    log = paper_dir / "revision-log.jsonl"
    if log.exists() and log.stat().st_size > 0:
        history_parts.append(f"\n---\n\n## revision-log.jsonl\n\n```jsonl\n{log.read_text(encoding='utf-8')}```\n")

    (paper_dir / "paper-history.md").write_text("".join(history_parts), encoding="utf-8")
    return paper_md


def _main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: finalize.py <paper-dir> <final-verdict>", file=sys.stderr)
        return 2
    out = finalize_paper(Path(argv[1]), argv[2])
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
