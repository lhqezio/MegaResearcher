#!/usr/bin/env python3
"""Claude Code PostToolUse hook: doom-loop detection.

Ported from huggingface/ml-intern's `agent/core/doom_loop.py`. The original
operates on litellm Message objects; here we read CC's transcript JSONL
directly and reconstruct tool-call signatures from it. Pure stdlib — no
external deps so the hook stays cheap to invoke.

CC delivers PostToolUse hook input on stdin as JSON with:
  - session_id        : current session id
  - transcript_path   : absolute path to the session JSONL
  - tool_name         : name of the tool that just ran
  - tool_input        : tool arguments
  - tool_response     : tool result

To inject a corrective system message back into the conversation we emit on
stdout:
  {"hookSpecificOutput": {"hookEventName": "PostToolUse",
                           "additionalContext": "<correction>"}}

Exit 0 always — we never want loop detection itself to break the session.

Detection thresholds:
  - 3 identical consecutive calls to the same tool with the same args
  - A repeating sequence of length 2..5 with 2+ consecutive repetitions
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

LOOKBACK = 30  # number of recent transcript entries to consider
IDENTICAL_THRESHOLD = 3
SEQ_MIN_LEN = 2
SEQ_MAX_LEN = 5
SEQ_MIN_REPS = 2


@dataclass(frozen=True)
class ToolCallSignature:
    name: str
    args_hash: str
    result_hash: str | None = None


def _normalize_args(args: object) -> str:
    """Canonicalize args before hashing so re-orderings don't dodge detection."""
    if args is None or args == "":
        return ""
    try:
        if isinstance(args, str):
            obj = json.loads(args)
        else:
            obj = args
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return str(args)


def _hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _signatures_from_transcript(path: Path) -> list[ToolCallSignature]:
    """Walk the CC transcript JSONL and build tool-call signatures.

    CC's transcript format: one JSON object per line. Assistant turns include
    a `message.content` list whose entries may be `{"type":"tool_use","name":...,
    "input":{...}, "id":...}`. The matching tool result appears in a later
    user turn as `{"type":"tool_result","tool_use_id":...,"content":...}`.
    """
    if not path.exists():
        return []

    # First pass: collect all tool_use events and indexed tool_results by id.
    tool_uses: list[tuple[str, str, str]] = []  # (id, name, args_hash)
    tool_results: dict[str, str] = {}  # id -> result_hash

    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = entry.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t == "tool_use":
                    tool_uses.append((
                        item.get("id", ""),
                        item.get("name", ""),
                        _hash(_normalize_args(item.get("input"))),
                    ))
                elif t == "tool_result":
                    tid = item.get("tool_use_id", "")
                    if tid:
                        # content can be a string or a list of content blocks
                        result_repr = item.get("content", "")
                        if isinstance(result_repr, list):
                            result_repr = json.dumps(result_repr, sort_keys=True)
                        tool_results[tid] = _hash(str(result_repr))

    # Take the last LOOKBACK tool uses; attach result hashes when available.
    recent = tool_uses[-LOOKBACK:] if len(tool_uses) > LOOKBACK else tool_uses
    return [
        ToolCallSignature(name=name, args_hash=ahash, result_hash=tool_results.get(uid))
        for uid, name, ahash in recent
    ]


def _detect_identical_consecutive(
    sigs: list[ToolCallSignature], threshold: int = IDENTICAL_THRESHOLD
) -> str | None:
    if len(sigs) < threshold:
        return None
    count = 1
    for i in range(1, len(sigs)):
        if sigs[i] == sigs[i - 1]:
            count += 1
            if count >= threshold:
                return sigs[i].name
        else:
            count = 1
    return None


def _detect_repeating_sequence(
    sigs: list[ToolCallSignature],
) -> list[ToolCallSignature] | None:
    n = len(sigs)
    for seq_len in range(SEQ_MIN_LEN, SEQ_MAX_LEN + 1):
        min_required = seq_len * SEQ_MIN_REPS
        if n < min_required:
            continue
        pattern = sigs[-min_required:][:seq_len]
        reps = 0
        for start in range(n - seq_len, -1, -seq_len):
            chunk = sigs[start : start + seq_len]
            if chunk == pattern:
                reps += 1
            else:
                break
        if reps >= SEQ_MIN_REPS:
            return pattern
    return None


def _check(sigs: list[ToolCallSignature]) -> str | None:
    if len(sigs) < 3:
        return None
    name = _detect_identical_consecutive(sigs)
    if name:
        return (
            f"[REPETITION GUARD] You have called '{name}' with the same arguments "
            f"{IDENTICAL_THRESHOLD}+ times in a row, getting the same result each time. "
            "STOP repeating this approach — it is not working. Step back and try a "
            "fundamentally different strategy: a different tool, materially different "
            "arguments, or surface the blocker to the user and ask for guidance."
        )
    pattern = _detect_repeating_sequence(sigs)
    if pattern:
        desc = " → ".join(s.name for s in pattern)
        return (
            f"[REPETITION GUARD] You are stuck in a repeating cycle of tool calls: "
            f"[{desc}]. This pattern has repeated {SEQ_MIN_REPS}+ times without progress. "
            "STOP this cycle and try a fundamentally different approach: break the "
            "problem down differently, use alternative tools, or ask the user for guidance."
        )
    return None


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return

    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        return

    try:
        sigs = _signatures_from_transcript(Path(transcript_path))
        correction = _check(sigs)
    except Exception:
        # Hook must never break the session. Swallow and exit clean.
        return

    if correction:
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": correction,
            }
        }
        print(json.dumps(out))


if __name__ == "__main__":
    main()
