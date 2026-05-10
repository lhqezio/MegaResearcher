"""Smoke test: the doom_loop hook detects identical-consecutive and repeating-
sequence patterns from synthetic CC transcript JSONL and emits the right
additionalContext JSON. Pure stdlib; no external deps.

Run from the plugin root:

    python3 tests/test_doom_loop.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
HOOK = PLUGIN_ROOT / "hooks" / "doom_loop.py"


def _make_transcript(tool_calls: list[tuple[str, dict]]) -> Path:
    """Write a synthetic CC transcript JSONL given a list of (tool_name, args) tuples."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w") as fh:
        for i, (name, args) in enumerate(tool_calls):
            fh.write(
                json.dumps(
                    {"message": {"content": [{"type": "tool_use", "id": f"tu{i}", "name": name, "input": args}]}}
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {"message": {"content": [{"type": "tool_result", "tool_use_id": f"tu{i}", "content": "result"}]}}
                )
                + "\n"
            )
    return Path(path)


def _run_hook(transcript_path: Path) -> tuple[int, str]:
    payload = json.dumps({"transcript_path": str(transcript_path), "tool_name": "x"})
    proc = subprocess.run(
        ["python3", str(HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.returncode, proc.stdout


def _expect_correction(stdout: str, marker: str) -> bool:
    if not stdout.strip():
        return False
    try:
        out = json.loads(stdout)
    except json.JSONDecodeError:
        return False
    ctx = out.get("hookSpecificOutput", {}).get("additionalContext", "")
    return marker in ctx


def test_no_input_silent() -> None:
    proc = subprocess.run(["python3", str(HOOK)], input="", capture_output=True, text=True, timeout=5)
    assert proc.returncode == 0, "hook must exit 0 on empty stdin"
    assert proc.stdout.strip() == "", "hook must be silent on empty stdin"
    print("PASS: no-input silent no-op")


def test_no_transcript_silent() -> None:
    proc = subprocess.run(["python3", str(HOOK)], input="{}", capture_output=True, text=True, timeout=5)
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""
    print("PASS: missing transcript_path silent no-op")


def test_three_identical_calls_fires_repetition_guard() -> None:
    transcript = _make_transcript([("web_search", {"query": "foo"})] * 3)
    code, out = _run_hook(transcript)
    transcript.unlink()
    assert code == 0
    assert _expect_correction(out, "REPETITION GUARD"), f"expected REPETITION GUARD, got: {out!r}"
    assert _expect_correction(out, "web_search"), "expected the tool name in the correction"
    print("PASS: 3 identical consecutive calls → REPETITION GUARD")


def test_two_distinct_calls_silent() -> None:
    transcript = _make_transcript(
        [("web_search", {"query": "foo"}), ("hf_papers", {"operation": "search", "query": "bar"})]
    )
    code, out = _run_hook(transcript)
    transcript.unlink()
    assert code == 0
    assert out.strip() == "", "should not fire on 2 distinct calls"
    print("PASS: 2 distinct calls → silent")


def test_repeating_AB_sequence_fires() -> None:
    transcript = _make_transcript(
        [
            ("web_search", {"q": "x"}),
            ("hf_papers", {"op": "y"}),
            ("web_search", {"q": "x"}),
            ("hf_papers", {"op": "y"}),
        ]
    )
    code, out = _run_hook(transcript)
    transcript.unlink()
    assert code == 0
    assert _expect_correction(out, "repeating cycle"), f"expected repeating-cycle correction, got: {out!r}"
    print("PASS: A→B→A→B → repeating-cycle guard")


def main() -> int:
    failures = 0
    for fn in [
        test_no_input_silent,
        test_no_transcript_silent,
        test_three_identical_calls_fires_repetition_guard,
        test_two_distinct_calls_silent,
        test_repeating_AB_sequence_fires,
    ]:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL: {fn.__name__}: {e}", file=sys.stderr)
            failures += 1
    if failures:
        print(f"\n{failures} test(s) failed", file=sys.stderr)
        return 1
    print("\nAll doom_loop tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
