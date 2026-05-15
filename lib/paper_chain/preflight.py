"""Pre-flight checks for the paper-drafting chain.

The chain runs ONLY when:
  1. `output.md` exists at the run root (synthesist produced it)
  2. `swarm-state.yaml` exists at the run root
  3. The run's novelty_target is `hypothesis` (not `gap-finding` or `synthesis`)
  4. At least one `eval-designer-*` subdir exists with its own `output.md`

CLI:
    python3 -m lib.paper_chain.preflight <run-dir>
        → exit 0 if OK, exit 1 with refusal message on stderr otherwise
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

_NOVELTY_RE = re.compile(r"^novelty_target:\s*(\S+)\s*$", re.MULTILINE)


def preflight_check(run_dir: Path) -> tuple[bool, str]:
    """Return (ok, reason). reason is empty when ok=True."""
    output_md = run_dir / "output.md"
    if not output_md.exists():
        return (
            False,
            f"Pre-flight refusal: output.md not found at {output_md}. "
            "Re-run /research-execute first to produce the synthesist's output.",
        )

    state = run_dir / "swarm-state.yaml"
    if not state.exists():
        return (
            False,
            f"Pre-flight refusal: swarm-state.yaml not found at {state}.",
        )

    text = state.read_text(encoding="utf-8")
    m = _NOVELTY_RE.search(text)
    if m is None:
        return (
            False,
            f"Pre-flight refusal: novelty_target not found in {state}.",
        )
    target = m.group(1)
    if target != "hypothesis":
        return (
            False,
            f"Pre-flight refusal: paper chain only runs on hypothesis-target outputs. "
            f"This run's novelty_target is {target} (expected hypothesis); "
            f"gap-finding runs lack the eval-designer protocols the paper chain consumes.",
        )

    eval_dirs = list(run_dir.glob("eval-designer-*"))
    if not eval_dirs:
        return (
            False,
            f"Pre-flight refusal: no eval-designer-* subdirs in {run_dir}. "
            "Paper chain requires Phase 5 protocols as input.",
        )
    for d in eval_dirs:
        if not (d / "output.md").exists():
            return (
                False,
                f"Pre-flight refusal: eval-designer subdir {d} missing output.md.",
            )

    return (True, "")


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: preflight.py <run-dir>", file=sys.stderr)
        return 2
    ok, reason = preflight_check(Path(argv[1]))
    if not ok:
        print(reason, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
