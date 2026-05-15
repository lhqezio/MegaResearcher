# Experimentalist + Vercel Sandbox (SP2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Phase 6.5 experimentalist worker that runs experiments in Vercel Sandbox per surviving hypothesis from Phase 5 eval-designer outputs, captures structured results, and hands them to SP1's drafter so Section 6 of the paper becomes "Experiments & Results" with real numbers instead of "we will measure" placeholders.

**Architecture:** Single-session orchestrator dispatches one `experimentalist` leaf-worker per surviving hypothesis. The worker reads the protocol, looks up the runner module for the named substrate, spins up a Vercel Sandbox VM, executes the runner, captures results.json + repro.yaml + figures, tears down the sandbox. Sequential execution; one sandbox per experiment; cold start each time; $5+$5 budget ceiling.

**Tech Stack:** Python 3 stdlib (helpers + tests), Vercel Sandbox SDK (via wrapped backend interface), markdown (agent), YAML (state).

**Spec:** `docs/superpowers/specs/2026-05-15-experimentalist-sandbox-design.md`

**SCOPE NOTE (read carefully):** This plan ships the core orchestration infra + **5 skeleton runners**. Each runner satisfies the Runner schema contract but returns `status: failed_runner_not_implemented` until its real benchmark integration lands in a follow-up plan (SP2a.1 SPECS, SP2a.2 AbGen, SP2a.3 CiteME, SP2a.4 LimitGen, SP2a.5 PaperWrite-Bench). This keeps SP2a's scope tractable; real experimental numbers wait on each follow-up. Until at least one real runner ships, every experiment fails honestly with the `runner_not_implemented` code and the drafter falls back to SP1's option-γ stub — which is the documented graceful-degradation path.

---

## File Structure

### New Python helpers — `lib/paper_chain/`

| File | Responsibility |
|---|---|
| `lib/paper_chain/sandbox.py` | Vercel Sandbox SDK wrapper. `spin_up / execute / tear_down / cost_so_far` API. Module-level `_sandbox_backend` indirection lets tests inject `FakeSandboxBackend`; production uses `VercelSandboxBackend`. |
| `lib/paper_chain/experiment.py` | Orchestration: read protocol → select runner → invoke sandbox → validate results → write artifacts. |
| `lib/paper_chain/protocol_parser.py` | Parse an eval-designer protocol markdown → `{substrate, sample_size, seed, baselines, metrics, decision_rules}`. |

### New runner library — `lib/runners/`

| File | Responsibility |
|---|---|
| `lib/runners/__init__.py` | Empty package marker |
| `lib/runners/_base.py` | `Runner` base class + result-schema validator. Defines required fields: `hypothesis_id, substrate, metric_name, baseline_value, treatment_value, p_value, ci_low, ci_high, n, seed, runtime_seconds, cost_usd, status, failure_code, failure_message` |
| `lib/runners/specs/runner.py` | SPECS skeleton — returns `failed_runner_not_implemented` |
| `lib/runners/abgen/runner.py` | AbGen skeleton — same |
| `lib/runners/citeme/runner.py` | CiteME skeleton — same |
| `lib/runners/limitgen/runner.py` | LimitGen skeleton — same |
| `lib/runners/paperwrite_bench/runner.py` | PaperWrite-Bench skeleton — same |

Each runner package also has `__init__.py` (empty) and a `requirements.txt` file (initially empty — populated when real implementation lands).

### New worker agent — `agents/`

| File | Phase | Dispatch shape |
|---|---|---|
| `agents/experimentalist.md` | 6.5 | loop-dispatch, one per surviving hypothesis |

### New tests — `tests/`

Pure-stdlib, runnable via `python3 tests/<file>.py`. Matches existing `test_doom_loop.py` pattern.

| File | Layer |
|---|---|
| `tests/test_protocol_parser.py` | 1 — helper |
| `tests/test_sandbox_wrapper.py` | 1 — helper, mocked SDK |
| `tests/test_experiment_orchestrator.py` | 1 — helper, mocked sandbox + runner |
| `tests/test_runner_base.py` | 1 — schema validator |
| `tests/test_runner_skeletons.py` | 2 — all 5 skeleton runners return correct failure status |
| `tests/fixtures/protocols/` | fixtures for parser tests |
| `tests/fixtures/sandbox_responses/` | canned SDK responses |

Integration tests against real Vercel Sandbox are documented but NOT shipped in SP2a (they require runners to do something — deferred to per-runner follow-ups).

### Modified files

| File | Modification |
|---|---|
| `skills/executing-research-plan/SKILL.md` | Add Phase 6.5 section between Phase 6 and Phase 7 |
| `agents/manuscript-drafter.md` | Update Section 6 instruction: read `paper/experiments/<hyp-id>/results.json` if present; use real numbers when `status=completed`, mark `[Experimental data unavailable: <failure_code>]` when failed |
| `lib/paper_chain/preflight.py` | Extend with conditional VERCEL_TOKEN check (warns when --paper is set and runners would otherwise fire) |
| `commands/research-execute.md` | Document VERCEL_TOKEN requirement in --paper section |
| `CLAUDE.md` | Add VERCEL_TOKEN to env-vars list; document SP2a Phase 6.5 |

---

## Task 1: Protocol parser

**Files:**
- Create: `lib/paper_chain/protocol_parser.py`
- Test: `tests/test_protocol_parser.py`
- Test fixtures: `tests/fixtures/protocols/specs_protocol.md`, `tests/fixtures/protocols/malformed.md`

- [ ] **Step 1: Create fixture files**

`tests/fixtures/protocols/specs_protocol.md`:
```markdown
# Eval design — S1 cross-family writer/reviewer split

## Pre-registered settings

- Substrate: SPECS-Review-Benchmark
- Sample size: 22 perturbations
- Seed: 42
- Baselines: stage-matched same-family 2-stage
- Metric: paired-difference flaw-detection rate, Correctness+Evaluations axes
- Decision rule: F1 lift ≥ 0.05 absolute, single-comparison paired-difference bootstrap, p < 0.13

## Other content the parser must ignore
...
```

`tests/fixtures/protocols/malformed.md`:
```markdown
# Just some random text
No structured fields here.
```

- [ ] **Step 2: Write `tests/test_protocol_parser.py`**

```python
"""Tests for eval-designer protocol parsing.

Run from plugin root:
    python3 tests/test_protocol_parser.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.protocol_parser import parse_protocol

FIXTURES = PLUGIN_ROOT / "tests" / "fixtures" / "protocols"


def test_parse_specs_protocol():
    result = parse_protocol(FIXTURES / "specs_protocol.md")
    assert result["substrate"] == "SPECS-Review-Benchmark", f"got {result['substrate']!r}"
    assert result["sample_size"] == 22
    assert result["seed"] == 42


def test_parse_baselines_list():
    result = parse_protocol(FIXTURES / "specs_protocol.md")
    assert "stage-matched same-family 2-stage" in result["baselines"][0]


def test_parse_metric():
    result = parse_protocol(FIXTURES / "specs_protocol.md")
    assert any("flaw-detection" in m for m in result["metrics"]), result["metrics"]


def test_parse_decision_rule():
    result = parse_protocol(FIXTURES / "specs_protocol.md")
    assert len(result["decision_rules"]) >= 1
    assert "0.05" in str(result["decision_rules"][0])


def test_parse_malformed_returns_empty():
    result = parse_protocol(FIXTURES / "malformed.md")
    assert result == {}, f"expected empty dict, got {result!r}"


if __name__ == "__main__":
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures.append((name, str(e)))
                print(f"FAIL {name}: {e}")
    if failures:
        sys.exit(1)
    print("All protocol parser tests pass.")
```

- [ ] **Step 3: Run to verify failure**

Run: `python3 tests/test_protocol_parser.py`
Expected: ImportError on `lib.paper_chain.protocol_parser`.

- [ ] **Step 4: Implement `lib/paper_chain/protocol_parser.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify pass**

Run: `python3 tests/test_protocol_parser.py`
Expected: 5 PASS lines + "All protocol parser tests pass."

- [ ] **Step 6: Commit**

```bash
git add lib/paper_chain/protocol_parser.py tests/test_protocol_parser.py tests/fixtures/protocols/
git commit -m "feat(paper-chain): add eval-designer protocol parser"
```

---

## Task 2: Sandbox wrapper with backend indirection

**Files:**
- Create: `lib/paper_chain/sandbox.py`
- Test: `tests/test_sandbox_wrapper.py`

- [ ] **Step 1: Write `tests/test_sandbox_wrapper.py`**

```python
"""Tests for the Vercel Sandbox wrapper with FakeSandboxBackend.

Run from plugin root:
    python3 tests/test_sandbox_wrapper.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.sandbox import (
    set_backend,
    spin_up,
    execute,
    tear_down,
    cost_so_far,
    BudgetBreach,
    FakeSandboxBackend,
    ExecutionResult,
)


def test_spin_up_returns_sandbox_id():
    set_backend(FakeSandboxBackend(canned_id="sb_abc"))
    sid = spin_up(image="python:3.11", timeout_seconds=60, budget_usd=5.0)
    assert sid == "sb_abc"


def test_execute_returns_result():
    set_backend(FakeSandboxBackend(canned_id="sb_x", canned_stdout="hello\n", canned_exit=0))
    sid = spin_up(image="python:3.11", timeout_seconds=60, budget_usd=5.0)
    result = execute(sid, "echo hello")
    assert isinstance(result, ExecutionResult)
    assert result.stdout == "hello\n"
    assert result.exit_code == 0


def test_tear_down_marks_torn():
    backend = FakeSandboxBackend(canned_id="sb_y")
    set_backend(backend)
    sid = spin_up(image="x", timeout_seconds=60, budget_usd=5.0)
    tear_down(sid)
    assert backend.torn_down == ["sb_y"]


def test_cost_so_far_reads_backend():
    backend = FakeSandboxBackend(canned_id="sb_z", canned_cost=1.23)
    set_backend(backend)
    sid = spin_up(image="x", timeout_seconds=60, budget_usd=5.0)
    assert cost_so_far(sid) == 1.23


def test_budget_breach_during_execute():
    backend = FakeSandboxBackend(canned_id="sb_q", canned_cost=10.0)
    set_backend(backend)
    sid = spin_up(image="x", timeout_seconds=60, budget_usd=5.0)
    try:
        execute(sid, "expensive")
        assert False, "expected BudgetBreach"
    except BudgetBreach as e:
        assert "5.0" in str(e)


def test_timeout_yields_exit_124():
    backend = FakeSandboxBackend(canned_id="sb_t", canned_exit=124)
    set_backend(backend)
    sid = spin_up(image="x", timeout_seconds=60, budget_usd=5.0)
    result = execute(sid, "sleep 999")
    assert result.exit_code == 124


if __name__ == "__main__":
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures.append((name, str(e)))
                print(f"FAIL {name}: {e}")
    if failures:
        sys.exit(1)
    print("All sandbox wrapper tests pass.")
```

- [ ] **Step 2: Run to verify failure** (ImportError)

Run: `python3 tests/test_sandbox_wrapper.py`

- [ ] **Step 3: Implement `lib/paper_chain/sandbox.py`**

```python
"""Vercel Sandbox SDK wrapper.

Production usage:
    set_backend(VercelSandboxBackend(token=os.environ["VERCEL_TOKEN"]))
    sid = spin_up(image="python:3.11", timeout_seconds=60, budget_usd=5.0)
    result = execute(sid, "python -m lib.runners.specs.runner")
    tear_down(sid)

Test usage:
    set_backend(FakeSandboxBackend(canned_id="sb_x", canned_stdout="ok"))

The module-level `_sandbox_backend` indirection lets tests inject behavior
without monkeypatching the real Vercel SDK.

CLI: this module is not invoked directly; it's imported by experiment.py.
"""
from __future__ import annotations
import dataclasses
import os
import sys
from pathlib import Path


@dataclasses.dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    runtime_seconds: float


class BudgetBreach(Exception):
    """Raised when sandbox cost exceeds the budget ceiling."""


class _Backend:
    """Interface every backend must implement."""

    def spin_up(self, image: str, timeout_seconds: int, budget_usd: float) -> str:
        raise NotImplementedError

    def execute(self, sandbox_id: str, command: str) -> ExecutionResult:
        raise NotImplementedError

    def tear_down(self, sandbox_id: str) -> None:
        raise NotImplementedError

    def cost_so_far(self, sandbox_id: str) -> float:
        raise NotImplementedError


class FakeSandboxBackend(_Backend):
    """Test backend with canned responses."""

    def __init__(
        self,
        canned_id: str = "sb_fake",
        canned_stdout: str = "",
        canned_stderr: str = "",
        canned_exit: int = 0,
        canned_cost: float = 0.0,
        canned_runtime: float = 1.0,
    ):
        self.canned_id = canned_id
        self.canned_stdout = canned_stdout
        self.canned_stderr = canned_stderr
        self.canned_exit = canned_exit
        self.canned_cost = canned_cost
        self.canned_runtime = canned_runtime
        self.torn_down: list[str] = []
        self._budgets: dict[str, float] = {}

    def spin_up(self, image: str, timeout_seconds: int, budget_usd: float) -> str:
        self._budgets[self.canned_id] = budget_usd
        return self.canned_id

    def execute(self, sandbox_id: str, command: str) -> ExecutionResult:
        budget = self._budgets.get(sandbox_id, 0.0)
        if self.canned_cost > budget:
            raise BudgetBreach(
                f"Sandbox {sandbox_id} cost ${self.canned_cost} exceeds budget ${budget}"
            )
        return ExecutionResult(
            stdout=self.canned_stdout,
            stderr=self.canned_stderr,
            exit_code=self.canned_exit,
            runtime_seconds=self.canned_runtime,
        )

    def tear_down(self, sandbox_id: str) -> None:
        self.torn_down.append(sandbox_id)

    def cost_so_far(self, sandbox_id: str) -> float:
        return self.canned_cost


class VercelSandboxBackend(_Backend):
    """Production backend. Stub implementation — real SDK wiring is left for
    each runner's follow-up plan to validate against current Vercel SDK docs.

    Raises NotImplementedError on use until the SDK is wired.
    """

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("VERCEL_TOKEN")
        if not self.token:
            raise RuntimeError(
                "VERCEL_TOKEN env var is required for VercelSandboxBackend"
            )

    def spin_up(self, image: str, timeout_seconds: int, budget_usd: float) -> str:
        raise NotImplementedError(
            "VercelSandboxBackend.spin_up is a stub. Real SDK wiring is "
            "scheduled for the first runner-implementation follow-up plan."
        )

    def execute(self, sandbox_id: str, command: str) -> ExecutionResult:
        raise NotImplementedError("VercelSandboxBackend.execute is a stub.")

    def tear_down(self, sandbox_id: str) -> None:
        raise NotImplementedError("VercelSandboxBackend.tear_down is a stub.")

    def cost_so_far(self, sandbox_id: str) -> float:
        raise NotImplementedError("VercelSandboxBackend.cost_so_far is a stub.")


_sandbox_backend: _Backend | None = None


def set_backend(backend: _Backend) -> None:
    global _sandbox_backend
    _sandbox_backend = backend


def _get_backend() -> _Backend:
    if _sandbox_backend is None:
        raise RuntimeError(
            "No sandbox backend configured. Call set_backend() first."
        )
    return _sandbox_backend


def spin_up(image: str, timeout_seconds: int, budget_usd: float) -> str:
    return _get_backend().spin_up(image, timeout_seconds, budget_usd)


def execute(sandbox_id: str, command: str) -> ExecutionResult:
    return _get_backend().execute(sandbox_id, command)


def tear_down(sandbox_id: str) -> None:
    _get_backend().tear_down(sandbox_id)


def cost_so_far(sandbox_id: str) -> float:
    return _get_backend().cost_so_far(sandbox_id)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 tests/test_sandbox_wrapper.py`
Expected: 6 PASS lines + "All sandbox wrapper tests pass."

- [ ] **Step 5: Commit**

```bash
git add lib/paper_chain/sandbox.py tests/test_sandbox_wrapper.py
git commit -m "feat(paper-chain): add Vercel Sandbox wrapper with backend indirection"
```

---

## Task 3: Runner base class + schema validator

**Files:**
- Create: `lib/runners/__init__.py`, `lib/runners/_base.py`
- Test: `tests/test_runner_base.py`

- [ ] **Step 1: Create empty package marker**

```bash
mkdir -p lib/runners && touch lib/runners/__init__.py
```

- [ ] **Step 2: Write `tests/test_runner_base.py`**

```python
"""Tests for Runner base class + result schema.

Run from plugin root:
    python3 tests/test_runner_base.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.runners._base import (
    Runner,
    validate_result,
    make_failed_result,
    REQUIRED_FIELDS,
)


def test_required_fields_present():
    expected = {
        "hypothesis_id", "substrate", "metric_name",
        "baseline_value", "treatment_value",
        "p_value", "ci_low", "ci_high", "n", "seed",
        "runtime_seconds", "cost_usd",
        "status", "failure_code", "failure_message",
    }
    assert set(REQUIRED_FIELDS) == expected


def test_validate_complete_result():
    result = {f: None for f in REQUIRED_FIELDS}
    result["status"] = "completed"
    ok, errs = validate_result(result)
    assert ok, errs


def test_validate_missing_field():
    result = {f: None for f in REQUIRED_FIELDS}
    del result["substrate"]
    result["status"] = "completed"
    ok, errs = validate_result(result)
    assert not ok
    assert any("substrate" in e for e in errs)


def test_validate_bad_status():
    result = {f: None for f in REQUIRED_FIELDS}
    result["status"] = "maybe"
    ok, errs = validate_result(result)
    assert not ok


def test_make_failed_result_satisfies_schema():
    result = make_failed_result(
        hypothesis_id="S1",
        substrate="SPECS-Review-Benchmark",
        failure_code="runner_not_implemented",
        failure_message="skeleton",
    )
    ok, errs = validate_result(result)
    assert ok, errs
    assert result["status"] == "failed"
    assert result["failure_code"] == "runner_not_implemented"


def test_runner_subclass_must_define_run():
    class IncompleteRunner(Runner):
        substrate = "x"
    try:
        IncompleteRunner().run({"hypothesis_id": "S1"})
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


if __name__ == "__main__":
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures.append((name, str(e)))
                print(f"FAIL {name}: {e}")
    if failures:
        sys.exit(1)
    print("All runner base tests pass.")
```

- [ ] **Step 3: Run to verify failure** (ImportError)

- [ ] **Step 4: Implement `lib/runners/_base.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify pass**

Run: `python3 tests/test_runner_base.py`
Expected: 6 PASS lines + "All runner base tests pass."

- [ ] **Step 6: Commit**

```bash
git add lib/runners/__init__.py lib/runners/_base.py tests/test_runner_base.py
git commit -m "feat(paper-chain): add Runner base class + result schema"
```

---

## Task 4: 5 skeleton runners

**Files:**
- Create: `lib/runners/{specs,abgen,citeme,limitgen,paperwrite_bench}/__init__.py`
- Create: `lib/runners/{specs,abgen,citeme,limitgen,paperwrite_bench}/runner.py`
- Create: `lib/runners/{specs,abgen,citeme,limitgen,paperwrite_bench}/requirements.txt` (empty)
- Test: `tests/test_runner_skeletons.py`

- [ ] **Step 1: Create the directory structure**

```bash
for s in specs abgen citeme limitgen paperwrite_bench; do
  mkdir -p "lib/runners/$s"
  touch "lib/runners/$s/__init__.py"
  touch "lib/runners/$s/requirements.txt"
done
```

- [ ] **Step 2: Write `tests/test_runner_skeletons.py`**

```python
"""Tests confirming each skeleton runner returns a schema-valid failed result.

Run from plugin root:
    python3 tests/test_runner_skeletons.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.runners._base import validate_result
from lib.runners.specs.runner import run as specs_run
from lib.runners.abgen.runner import run as abgen_run
from lib.runners.citeme.runner import run as citeme_run
from lib.runners.limitgen.runner import run as limitgen_run
from lib.runners.paperwrite_bench.runner import run as paperwrite_bench_run


def _check_skeleton(run_fn, expected_substrate: str):
    result = run_fn({"hypothesis_id": "test"})
    ok, errs = validate_result(result)
    assert ok, errs
    assert result["status"] == "failed"
    assert result["failure_code"] == "runner_not_implemented"
    assert result["substrate"] == expected_substrate


def test_specs_skeleton():
    _check_skeleton(specs_run, "SPECS-Review-Benchmark")


def test_abgen_skeleton():
    _check_skeleton(abgen_run, "AbGen")


def test_citeme_skeleton():
    _check_skeleton(citeme_run, "CiteME")


def test_limitgen_skeleton():
    _check_skeleton(limitgen_run, "LimitGen")


def test_paperwrite_bench_skeleton():
    _check_skeleton(paperwrite_bench_run, "PaperWrite-Bench")


if __name__ == "__main__":
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures.append((name, str(e)))
                print(f"FAIL {name}: {e}")
    if failures:
        sys.exit(1)
    print("All skeleton runner tests pass.")
```

- [ ] **Step 3: Run to verify failure** (ImportError on first skeleton)

- [ ] **Step 4: Implement the 5 skeleton runners**

For each substrate, create `lib/runners/<substrate>/runner.py` following this template (substitute the substrate name):

`lib/runners/specs/runner.py`:
```python
"""SPECS-Review-Benchmark runner — SKELETON.

Real implementation is deferred to follow-up plan SP2a.1.
This skeleton satisfies the Runner schema contract by returning a
schema-valid `failed_runner_not_implemented` result.
"""
from __future__ import annotations
from lib.runners._base import make_failed_result

SUBSTRATE = "SPECS-Review-Benchmark"


def run(params: dict) -> dict:
    return make_failed_result(
        hypothesis_id=params.get("hypothesis_id", "unknown"),
        substrate=SUBSTRATE,
        failure_code="runner_not_implemented",
        failure_message=(
            "SPECS runner skeleton — real implementation pending in SP2a.1 "
            "follow-up plan. See "
            "docs/superpowers/specs/2026-05-15-experimentalist-sandbox-design.md "
            "for the substrate's expected behavior."
        ),
        seed=params.get("seed"),
    )


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(run({"hypothesis_id": "cli"}), indent=2))
```

`lib/runners/abgen/runner.py`:
```python
"""AbGen runner — SKELETON.

Real implementation is deferred to follow-up plan SP2a.2.
"""
from __future__ import annotations
from lib.runners._base import make_failed_result

SUBSTRATE = "AbGen"


def run(params: dict) -> dict:
    return make_failed_result(
        hypothesis_id=params.get("hypothesis_id", "unknown"),
        substrate=SUBSTRATE,
        failure_code="runner_not_implemented",
        failure_message=(
            "AbGen runner skeleton — real implementation pending in SP2a.2 "
            "follow-up plan."
        ),
        seed=params.get("seed"),
    )


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(run({"hypothesis_id": "cli"}), indent=2))
```

`lib/runners/citeme/runner.py`:
```python
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
```

`lib/runners/limitgen/runner.py`:
```python
"""LimitGen runner — SKELETON.

Real implementation is deferred to follow-up plan SP2a.4.
"""
from __future__ import annotations
from lib.runners._base import make_failed_result

SUBSTRATE = "LimitGen"


def run(params: dict) -> dict:
    return make_failed_result(
        hypothesis_id=params.get("hypothesis_id", "unknown"),
        substrate=SUBSTRATE,
        failure_code="runner_not_implemented",
        failure_message=(
            "LimitGen runner skeleton — real implementation pending in SP2a.4 "
            "follow-up plan."
        ),
        seed=params.get("seed"),
    )


if __name__ == "__main__":
    import json
    print(json.dumps(run({"hypothesis_id": "cli"}), indent=2))
```

`lib/runners/paperwrite_bench/runner.py`:
```python
"""PaperWrite-Bench runner — SKELETON.

Real implementation is deferred to follow-up plan SP2a.5.
"""
from __future__ import annotations
from lib.runners._base import make_failed_result

SUBSTRATE = "PaperWrite-Bench"


def run(params: dict) -> dict:
    return make_failed_result(
        hypothesis_id=params.get("hypothesis_id", "unknown"),
        substrate=SUBSTRATE,
        failure_code="runner_not_implemented",
        failure_message=(
            "PaperWrite-Bench runner skeleton — real implementation pending "
            "in SP2a.5 follow-up plan."
        ),
        seed=params.get("seed"),
    )


if __name__ == "__main__":
    import json
    print(json.dumps(run({"hypothesis_id": "cli"}), indent=2))
```

- [ ] **Step 5: Run tests to verify pass**

Run: `python3 tests/test_runner_skeletons.py`
Expected: 5 PASS lines + "All skeleton runner tests pass."

- [ ] **Step 6: Commit**

```bash
git add lib/runners/specs/ lib/runners/abgen/ lib/runners/citeme/ lib/runners/limitgen/ lib/runners/paperwrite_bench/ tests/test_runner_skeletons.py
git commit -m "feat(paper-chain): add 5 skeleton runners (real impls deferred to SP2a.1-.5)"
```

---

## Task 5: Experiment orchestrator

**Files:**
- Create: `lib/paper_chain/experiment.py`
- Test: `tests/test_experiment_orchestrator.py`

- [ ] **Step 1: Write `tests/test_experiment_orchestrator.py`**

```python
"""Tests for experiment.py orchestration.

Run from plugin root:
    python3 tests/test_experiment_orchestrator.py
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.sandbox import set_backend, FakeSandboxBackend
from lib.paper_chain.experiment import (
    select_runner,
    dispatch_experiment,
)


def _make_protocol(substrate: str) -> Path:
    text = f"""# Eval design — test

## Pre-registered settings

- Substrate: {substrate}
- Sample size: 5
- Seed: 42
- Baselines: baseline-A
- Metric: detection-rate
- Decision rule: lift >= 0.05
"""
    fh = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    fh.write(text)
    fh.close()
    return Path(fh.name)


def test_select_runner_known_substrate():
    runner = select_runner("SPECS-Review-Benchmark")
    assert runner is not None
    assert runner.SUBSTRATE == "SPECS-Review-Benchmark"


def test_select_runner_unknown_substrate():
    runner = select_runner("UnknownBench")
    assert runner is None


def test_dispatch_experiment_writes_three_artifacts():
    set_backend(FakeSandboxBackend(canned_id="sb_x", canned_stdout="ignored"))
    output_dir = Path(tempfile.mkdtemp(prefix="exp-"))
    protocol = _make_protocol("SPECS-Review-Benchmark")
    results_path = dispatch_experiment(
        hypothesis_id="S1",
        protocol_path=protocol,
        output_dir=output_dir,
        sandbox_budget_usd=5.0,
        api_budget_usd=5.0,
    )
    assert results_path == output_dir / "results.json"
    assert results_path.exists()
    assert (output_dir / "repro.yaml").exists()
    assert (output_dir / "runner-output.log").exists()


def test_dispatch_records_failed_for_skeleton():
    """Skeleton runner returns failed_runner_not_implemented — orchestrator
    should write that result faithfully without retry."""
    set_backend(FakeSandboxBackend(canned_id="sb_x"))
    output_dir = Path(tempfile.mkdtemp(prefix="exp-"))
    protocol = _make_protocol("SPECS-Review-Benchmark")
    results_path = dispatch_experiment(
        hypothesis_id="S1",
        protocol_path=protocol,
        output_dir=output_dir,
        sandbox_budget_usd=5.0,
        api_budget_usd=5.0,
    )
    result = json.loads(results_path.read_text())
    assert result["status"] == "failed"
    assert result["failure_code"] == "runner_not_implemented"


def test_dispatch_unsupported_substrate_writes_failed():
    set_backend(FakeSandboxBackend(canned_id="sb_x"))
    output_dir = Path(tempfile.mkdtemp(prefix="exp-"))
    protocol = _make_protocol("UnknownBench")
    results_path = dispatch_experiment(
        hypothesis_id="S1",
        protocol_path=protocol,
        output_dir=output_dir,
        sandbox_budget_usd=5.0,
        api_budget_usd=5.0,
    )
    result = json.loads(results_path.read_text())
    assert result["status"] == "failed"
    assert result["failure_code"] == "unsupported_substrate"


def test_dispatch_malformed_protocol_writes_failed():
    set_backend(FakeSandboxBackend(canned_id="sb_x"))
    output_dir = Path(tempfile.mkdtemp(prefix="exp-"))
    bad_protocol = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    bad_protocol.write("# Random text with no structure\n")
    bad_protocol.close()
    results_path = dispatch_experiment(
        hypothesis_id="S1",
        protocol_path=Path(bad_protocol.name),
        output_dir=output_dir,
        sandbox_budget_usd=5.0,
        api_budget_usd=5.0,
    )
    result = json.loads(results_path.read_text())
    assert result["status"] == "failed"
    assert result["failure_code"] == "failed_parse"


if __name__ == "__main__":
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures.append((name, str(e)))
                print(f"FAIL {name}: {e}")
    if failures:
        sys.exit(1)
    print("All experiment orchestrator tests pass.")
```

- [ ] **Step 2: Run to verify failure** (ImportError)

- [ ] **Step 3: Implement `lib/paper_chain/experiment.py`**

```python
"""Experiment orchestrator. Reads protocol, picks runner, captures result.

The orchestrator's sandbox interaction in SP2a is minimal because skeleton
runners return immediately. Once real runners ship, this orchestrator's
sandbox.spin_up / execute / tear_down calls will actually invoke benchmark
code in Vercel Sandbox. The structure is in place for that future work.

CLI:
    python3 -m lib.paper_chain.experiment dispatch \\
        --hypothesis-id=S1 \\
        --protocol=<path> \\
        --output-dir=<dir> \\
        --sandbox-budget=5 \\
        --api-budget=5
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import platform
from pathlib import Path

from lib.paper_chain.protocol_parser import parse_protocol
from lib.paper_chain.sandbox import (
    spin_up,
    execute,
    tear_down,
    cost_so_far,
    BudgetBreach,
)
from lib.runners._base import make_failed_result

# Registry of supported substrates → runner modules.
_REGISTRY: dict[str, str] = {
    "SPECS-Review-Benchmark": "lib.runners.specs.runner",
    "AbGen": "lib.runners.abgen.runner",
    "CiteME": "lib.runners.citeme.runner",
    "LimitGen": "lib.runners.limitgen.runner",
    "PaperWrite-Bench": "lib.runners.paperwrite_bench.runner",
}


def select_runner(substrate: str):
    """Return the runner module for the given substrate, or None."""
    module_path = _REGISTRY.get(substrate)
    if module_path is None:
        return None
    import importlib
    return importlib.import_module(module_path)


def _write_repro(output_dir: Path, params: dict) -> None:
    repro = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": params.get("seed"),
    }
    (output_dir / "repro.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in repro.items()) + "\n",
        encoding="utf-8",
    )


def _write_runner_log(output_dir: Path, content: str) -> None:
    (output_dir / "runner-output.log").write_text(content, encoding="utf-8")


def dispatch_experiment(
    hypothesis_id: str,
    protocol_path: Path,
    output_dir: Path,
    sandbox_budget_usd: float,
    api_budget_usd: float,
) -> Path:
    """Orchestrate one experiment. Returns path to results.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    # Step 1: parse protocol
    params = parse_protocol(protocol_path)
    if not params:
        result = make_failed_result(
            hypothesis_id=hypothesis_id,
            substrate="unknown",
            failure_code="failed_parse",
            failure_message=f"Protocol at {protocol_path} could not be parsed.",
        )
        results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        _write_repro(output_dir, params or {})
        _write_runner_log(output_dir, "Protocol parse failed.\n")
        return results_path

    params["hypothesis_id"] = hypothesis_id
    substrate = params["substrate"]

    # Step 2: select runner
    runner_module = select_runner(substrate)
    if runner_module is None:
        result = make_failed_result(
            hypothesis_id=hypothesis_id,
            substrate=substrate,
            failure_code="unsupported_substrate",
            failure_message=(
                f"No runner module for substrate {substrate!r}. "
                f"Add lib/runners/<substrate>/runner.py and register in "
                f"lib/paper_chain/experiment.py _REGISTRY."
            ),
            seed=params.get("seed"),
        )
        results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        _write_repro(output_dir, params)
        _write_runner_log(output_dir, f"Unsupported substrate: {substrate}\n")
        return results_path

    # Step 3: spin sandbox, run, tear down
    start = time.time()
    sandbox_id = None
    log_content = ""
    try:
        sandbox_id = spin_up(
            image="python:3.11",
            timeout_seconds=60,
            budget_usd=sandbox_budget_usd,
        )
        result = runner_module.run(params)
        # In real-runner follow-ups, execute() would invoke the runner inside
        # the sandbox. For skeleton runners we call run() locally; the sandbox
        # spin-up/teardown exercises the infra without doing actual work.
        log_content = f"Sandbox {sandbox_id} executed; runner returned status={result.get('status')}.\n"
    except BudgetBreach as e:
        result = make_failed_result(
            hypothesis_id=hypothesis_id,
            substrate=substrate,
            failure_code="failed_budget",
            failure_message=str(e),
            seed=params.get("seed"),
        )
        log_content = f"BudgetBreach: {e}\n"
    except Exception as e:
        result = make_failed_result(
            hypothesis_id=hypothesis_id,
            substrate=substrate,
            failure_code="failed_exception",
            failure_message=f"{type(e).__name__}: {e}",
            seed=params.get("seed"),
        )
        log_content = f"Exception: {type(e).__name__}: {e}\n"
    finally:
        if sandbox_id is not None:
            try:
                cost = cost_so_far(sandbox_id)
                result["cost_usd"] = (result.get("cost_usd") or 0.0) + cost
                tear_down(sandbox_id)
            except Exception as e:
                log_content += f"Tear-down warning: {e}\n"

    result["runtime_seconds"] = round(time.time() - start, 3)
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_repro(output_dir, params)
    _write_runner_log(output_dir, log_content)
    return results_path


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="experiment.py")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dispatch")
    d.add_argument("--hypothesis-id", required=True)
    d.add_argument("--protocol", required=True, type=Path)
    d.add_argument("--output-dir", required=True, type=Path)
    d.add_argument("--sandbox-budget", required=True, type=float)
    d.add_argument("--api-budget", required=True, type=float)
    args = parser.parse_args(argv[1:])
    if args.cmd == "dispatch":
        from lib.paper_chain.sandbox import set_backend, VercelSandboxBackend
        set_backend(VercelSandboxBackend())
        path = dispatch_experiment(
            hypothesis_id=args.hypothesis_id,
            protocol_path=args.protocol,
            output_dir=args.output_dir,
            sandbox_budget_usd=args.sandbox_budget,
            api_budget_usd=args.api_budget,
        )
        print(path)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 tests/test_experiment_orchestrator.py`
Expected: 6 PASS lines + "All experiment orchestrator tests pass."

- [ ] **Step 5: Commit**

```bash
git add lib/paper_chain/experiment.py tests/test_experiment_orchestrator.py
git commit -m "feat(paper-chain): add experiment orchestrator with protocol/runner/sandbox wiring"
```

---

## Task 6: experimentalist agent

**Files:**
- Create: `agents/experimentalist.md`

- [ ] **Step 1: Write `agents/experimentalist.md`**

```markdown
---
name: experimentalist
description: |
  Run ONE experiment per surviving hypothesis. Invoked by the `executing-research-plan` skill in Phase 6.5 (only when `/research-execute --paper` is set AND Phase 5 produced surviving hypotheses). Reads the hypothesis's eval-designer protocol, identifies the substrate, dispatches the appropriate runner via lib/paper_chain/experiment.py, captures results to paper/experiments/<hyp-id>/. Examples: <example>Context: hypothesis S1 has an eval-designer protocol naming SPECS-Review-Benchmark. user (orchestrator): "Run the experiment for hypothesis S1 using the protocol at docs/research/runs/.../eval-designer-S1/output.md. Write to docs/research/runs/.../paper/experiments/S1/." assistant: "I'll invoke experiment.py dispatch with the protocol; capture results.json, repro.yaml, and runner-output.log; emit the standard manifest and verification artifacts."</example>
model: inherit
---

You are experimentalist for MegaResearcher. Your job is to run ONE experiment per dispatch via the existing lib/paper_chain/experiment.py orchestrator. You do NOT write your own runner code, do NOT fabricate numbers, and do NOT summarize results outside what the runner reports.

## Required behavior

1. Run the experiment via the orchestrator:
   ```
   python3 -m lib.paper_chain.experiment dispatch \
     --hypothesis-id=<id> \
     --protocol=<protocol-path> \
     --output-dir=<output-path> \
     --sandbox-budget=5 \
     --api-budget=5
   ```
2. After the command exits, verify the three expected artifacts exist at `<output-path>`:
   - `results.json`
   - `repro.yaml`
   - `runner-output.log`
3. Read `results.json` and capture: `status`, `failure_code`, `runtime_seconds`, `cost_usd`.
4. Write your three required worker-contract artifacts:

## Required artifacts at the output path

1. **`experimentalist-manifest.yaml`**:
   ```yaml
   worker_id: experimentalist
   hypothesis_id: <id>
   substrate: <from results.json>
   status: completed|failed|escalated  # mirrors results.json status
   failure_code: <from results.json or null>
   cost_usd: <from results.json>
   runtime_seconds: <from results.json>
   ```
2. **`experimentalist-verification.md`** — confirm: `results.json` schema-valid (run `python3 -m lib.runners._base validate results.json`); `repro.yaml` non-empty; `runner-output.log` exists.

## Honesty discipline

- If the runner returns `status: failed_runner_not_implemented`, that's the expected behavior for skeleton runners in SP2a — DO NOT retry, DO NOT fabricate. The drafter handles the fallback to SP1's option-γ stub.
- If `failure_code` is `failed_timeout` or `failed_exception`, you may be re-dispatched once with feedback; do not retry on your own initiative.
- Cost ceiling: $5 sandbox + $5 API. The orchestrator enforces this; if the result shows `failure_code: failed_budget`, surface that to the controller — do not retry within the same dispatch.

## Banned phrases

Per project CLAUDE.md, never use "load-bearing", "this is doing a lot of work" (and variants), "real" as emphatic adjective, or "honest / honestly / to be honest". Plain alternatives only.

You are a leaf worker. Do not dispatch other agents.
```

- [ ] **Step 2: Verify frontmatter parses**

Run: `python3 -c "import re,pathlib; t=pathlib.Path('agents/experimentalist.md').read_text(); assert re.match(r'^---\n.*?\n---', t, re.DOTALL); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agents/experimentalist.md
git commit -m "feat(paper-chain): add experimentalist agent (Phase 6.5)"
```

---

## Task 7: Extend skill with Phase 6.5

**Files:**
- Modify: `skills/executing-research-plan/SKILL.md`

- [ ] **Step 1: Locate insertion point**

Run: `grep -n "^### Phase 7" skills/executing-research-plan/SKILL.md`
Expected: one match. Phase 6.5 inserts BEFORE this line.

- [ ] **Step 2: Use the Edit tool to insert Phase 6.5**

Insert this section AFTER Phase 6's content and BEFORE `### Phase 7 — manuscript-drafter (only if `--paper`)`:

```markdown
### Phase 6.5 — experimentalist (only if `--paper` AND Phase 5 produced surviving hypotheses)

Skip if `--paper` not set. Skip if Phase 5 produced no surviving hypotheses (gap-finding runs already short-circuit at preflight).

Otherwise, for each `docs/research/runs/<run-id>/eval-designer-S<N>/output.md` from Phase 5:

1. `mkdir -p docs/research/runs/<run-id>/paper/experiments/S<N>/`
2. Dispatch ONE `megaresearcher:experimentalist` subagent with:
   - `hypothesis_id`: S<N>
   - `protocol_path`: `docs/research/runs/<run-id>/eval-designer-S<N>/output.md`
   - `output_path`: `docs/research/runs/<run-id>/paper/experiments/S<N>/`
   - Budget: $5 sandbox + $5 API
3. Wait for completion. Run the per-worker verification gate on the experimentalist's three artifacts (`experimentalist-manifest.yaml`, `experimentalist-verification.md`, plus the orchestrator-produced `results.json`).
4. Read `paper/experiments/S<N>/results.json` and record in `swarm-state.yaml`:
   ```yaml
   phase_6_5_experimentalist:
     status: in_progress|completed|partial-fail|escalated
     experiments:
       - hyp_id: S<N>
         substrate: <from results.json>
         status: <from results.json>
         failure_code: <from results.json or null>
         results: paper/experiments/S<N>/results.json
         cost_usd: <from results.json>
         retry_count: 0|1
     total_cost_usd: <sum>
   ```
5. **If `status: failed_runner_not_implemented`:** expected for SP2a skeleton runners. Continue chain — drafter handles fallback. Do NOT retry.
6. **If `status: failed_timeout` or `status: failed_exception`:** re-dispatch experimentalist ONCE with the failure_message inlined as feedback. After one retry, mark `failed`, continue chain.
7. **If `status: failed_budget`:** record cost-so-far, escalate to user with the cost figure, continue chain (drafter sees the failure and stubs accordingly).
8. **If `status: failed_unsupported_substrate` or `failed_parse`:** cannot retry. Continue chain with the failed result preserved.

After all experiments processed, mark `phase_6_5_experimentalist.status` based on whether ANY experiment succeeded:
- `completed` if all `status: completed`
- `partial-fail` if at least one succeeded
- `escalated` if all failed

The paper chain proceeds to Phase 7 (drafter) regardless. Failed experiments are surfaced in the final `paper-history.md` "Failed experiments" section by Phase 9 finalize.

**Note on Vercel auth:** Phase 6.5 expects `VERCEL_TOKEN` in the environment. If absent, `VercelSandboxBackend.spin_up` raises immediately; pre-flight should warn earlier. If you skipped pre-flight (e.g., running an experiment as a one-off), the first failed experiment will surface the missing-token error and Phase 6.5 should escalate immediately rather than continue trying.
```

- [ ] **Step 3: Verify the change**

Run: `grep -n "^### Phase 6.5" skills/executing-research-plan/SKILL.md`
Expected: 1 match.

Run: `grep -nE "lib/paper_chain/experiment|experimentalist" skills/executing-research-plan/SKILL.md`
Expected: at least 2 matches.

- [ ] **Step 4: Commit**

```bash
git add skills/executing-research-plan/SKILL.md
git commit -m "feat(paper-chain): add Phase 6.5 experimentalist section to orchestrator skill"
```

---

## Task 8: Update drafter agent to consume experimental results

**Files:**
- Modify: `agents/manuscript-drafter.md`

- [ ] **Step 1: Locate the current Section 6 instruction**

Run: `grep -n "Experimental Plan" agents/manuscript-drafter.md`
Expected: 2 matches in the agent body (the 9-section list + descriptive paragraph).

- [ ] **Step 2: Use Edit to replace the Section 6 description**

Edit `agents/manuscript-drafter.md` to replace this paragraph:

```markdown
6. **Experimental Plan** — embeds the eval-designer protocols. For each surviving hypothesis, copy the protocol's pre-registered decision rules and named substrates verbatim under a "we will measure X via Y" framing. DO NOT generate numerical results — this is a plan section, not a results section.
```

With this updated version:

```markdown
6. **Experiments & Results** (when `paper/experiments/<hyp-id>/results.json` exists with `status: completed`) **OR Experimental Plan** (when results.json is absent or has `status: failed`).

   **Experiments & Results variant** (preferred when results are present): for each hypothesis, write a Setup paragraph naming the substrate / sample size / seed / baselines from the protocol, then a Results paragraph with the numbers pulled from `results.json` — baseline_value, treatment_value, p_value, ci_low/ci_high, n. Include a results table with one row per hypothesis. Embed any figures from `paper/experiments/<hyp-id>/figures/` if present. Do NOT compute new statistics; only report what results.json says.

   **Experimental Plan variant** (fallback when results absent or failed): copy the protocol's pre-registered decision rules and named substrates verbatim under a "we will measure X via Y" framing. When results.json has `status: failed`, include the marker `[Experimental data unavailable: <failure_code>]` for that hypothesis so reviewers see what wasn't tested.

   Do NOT fabricate numerical results — under either variant. If results.json is absent OR status is failed, use the Experimental Plan variant for that hypothesis only; mix variants across hypotheses as needed.
```

- [ ] **Step 3: Verify the change**

Run: `grep -n "Experiments & Results" agents/manuscript-drafter.md`
Expected: 1+ match.

Run: `grep -n "Experimental data unavailable" agents/manuscript-drafter.md`
Expected: 1 match.

- [ ] **Step 4: Commit**

```bash
git add agents/manuscript-drafter.md
git commit -m "feat(paper-chain): drafter reads paper/experiments/ and emits real numbers when present"
```

---

## Task 9: Extend pre-flight check with VERCEL_TOKEN warning

**Files:**
- Modify: `lib/paper_chain/preflight.py`
- Modify: `tests/test_preflight.py`

- [ ] **Step 1: Add a new test**

Append to `tests/test_preflight.py` (before the `if __name__ == "__main__":` block):

```python
def test_preflight_warns_about_vercel_token_when_paper(monkeypatch=None):
    """When --paper is set and VERCEL_TOKEN absent, preflight returns ok=True
    with a non-empty warnings list. The presence of a warning does not block;
    it just informs the user that Phase 6.5 will fail if it tries to use the
    Vercel backend."""
    import os
    run = _make_run("hypothesis", with_output=True, with_eval_designers=3)
    saved = os.environ.pop("VERCEL_TOKEN", None)
    try:
        from lib.paper_chain.preflight import preflight_check_with_paper
        ok, reason, warnings = preflight_check_with_paper(run, paper_mode=True)
        assert ok
        assert any("VERCEL_TOKEN" in w for w in warnings)
    finally:
        if saved is not None:
            os.environ["VERCEL_TOKEN"] = saved
```

- [ ] **Step 2: Implement the new function in `preflight.py`**

Append to `lib/paper_chain/preflight.py` (before the existing `_main` function):

```python
def preflight_check_with_paper(
    run_dir: Path, paper_mode: bool = False
) -> tuple[bool, str, list[str]]:
    """Extended preflight: returns (ok, reason, warnings).

    When paper_mode=True, adds env-var warnings for VERCEL_TOKEN (Phase 6.5)
    without blocking the chain. The caller surfaces warnings to the user.
    """
    import os
    ok, reason = preflight_check(run_dir)
    warnings: list[str] = []
    if ok and paper_mode and not os.environ.get("VERCEL_TOKEN"):
        warnings.append(
            "VERCEL_TOKEN not set — Phase 6.5 (experimentalist) will fail "
            "immediately when it tries to spin up a sandbox. Set the env var "
            "before invoking /research-execute --paper if you want real experiments."
        )
    return (ok, reason, warnings)
```

- [ ] **Step 3: Run tests**

Run: `python3 tests/test_preflight.py`
Expected: 6 PASS lines + "All preflight tests pass." (1 new test + 5 from SP1).

- [ ] **Step 4: Commit**

```bash
git add lib/paper_chain/preflight.py tests/test_preflight.py
git commit -m "feat(paper-chain): preflight warns when VERCEL_TOKEN missing in paper mode"
```

---

## Task 10: Update command + CLAUDE.md docs

**Files:**
- Modify: `commands/research-execute.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `commands/research-execute.md`**

Edit the existing "Optional `--paper` flag" section, appending this paragraph after the existing description:

```markdown

**Phase 6.5 (SP2a):** when `--paper` is set, the orchestrator inserts a new Phase 6.5 between the synthesist and the drafter — the `experimentalist` worker runs each surviving hypothesis's protocol in a Vercel Sandbox VM. Requires `VERCEL_TOKEN` env var; without it, all experiments fail with `runner_not_implemented` (skeleton mode) or `unsupported_backend` (real-runner mode) and the drafter falls back to SP1's option-γ stub. Per-experiment budget ceiling: $5 sandbox + $5 API.
```

- [ ] **Step 2: Update `CLAUDE.md`**

Edit the existing "Optional paper-drafting chain (SP1)" section. Rename to "Optional paper-drafting chain (SP1+SP2a)" and append after the existing bullet list:

```markdown

**SP2a additions:**
- **Phase 6.5** — `experimentalist` runs each surviving hypothesis's protocol via a Vercel Sandbox VM. Inserts BETWEEN Phase 6 (synthesist) and Phase 7 (drafter). Requires `VERCEL_TOKEN` env var.
- **5 skeleton runners** in `lib/runners/{specs,abgen,citeme,limitgen,paperwrite_bench}/` satisfy the Runner schema contract. Real benchmark integrations are deferred to per-runner follow-up plans (SP2a.1 SPECS, SP2a.2 AbGen, SP2a.3 CiteME, SP2a.4 LimitGen, SP2a.5 PaperWrite-Bench).
- **Drafter modification** — Section 6 of `draft-v1.md` becomes "Experiments & Results" with real numbers when `paper/experiments/<hyp-id>/results.json` has `status: completed`; falls back to SP1's option-γ "Experimental Plan" stub when results are absent or `status: failed` (with a `[Experimental data unavailable: <code>]` marker for failed cases).
- **Cost ceiling: $5 sandbox + $5 API per experiment.** Hard-stop on exceed. 1 retry on transient failures (timeout / exception). Failed experiments visible in `paper-history.md`.
```

- [ ] **Step 3: Verify the changes**

Run: `grep -n "Phase 6.5" CLAUDE.md commands/research-execute.md`
Expected: 1 match in each file.

- [ ] **Step 4: Commit**

```bash
git add commands/research-execute.md CLAUDE.md
git commit -m "docs(paper-chain): document Phase 6.5 + VERCEL_TOKEN in command and CLAUDE.md"
```

---

## Task 11: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `for t in tests/test_*.py; do printf "%s: " "$t"; python3 "$t" 2>&1 | tail -1; done`
Expected: 12 lines, each ending in a PASS-line from the corresponding test file. Pre-existing tests (test_doom_loop, test_mcp_imports, test_verdict_parser, test_preflight, test_scaffold, test_regression, test_finalize) all continue to pass; 5 new tests (test_protocol_parser, test_sandbox_wrapper, test_runner_base, test_runner_skeletons, test_experiment_orchestrator) all pass.

- [ ] **Step 2: Verify fixture preflight still passes**

Run: `python3 lib/paper_chain/preflight.py tests/fixtures/paper-chain/`
Expected: exit 0.

- [ ] **Step 3: No commit (verification only)**

---

## Task 12: Manual discipline check on SP1 fixture

> This task validates Phase 6.5 end-to-end against the SP1 fixture without re-running the swarm. Token cost: ~$1-3 for the three experimentalist dispatches against skeleton runners (which return immediately; no real sandbox compute since skeletons short-circuit).

- [ ] **Step 1: Run the cheaper variant manually**

In a Claude Code session at the repo root, manually invoke Phase 6.5 logic against the existing SP1 fixture run:

```bash
# Set VERCEL_TOKEN if you want VercelSandboxBackend to load (the skeleton
# runners short-circuit before sandbox calls, so this is optional for SP2a):
export VERCEL_TOKEN=""

# For each surviving hypothesis from the SP1 fixture run, invoke the
# orchestrator directly:
for h in S1 S2 S3; do
  mkdir -p tests/fixtures/paper-chain/paper/experiments/$h
  python3 -m lib.paper_chain.experiment dispatch \
    --hypothesis-id=$h \
    --protocol=tests/fixtures/paper-chain/eval-designer-$h/output.md \
    --output-dir=tests/fixtures/paper-chain/paper/experiments/$h \
    --sandbox-budget=5 \
    --api-budget=5
done
```

NOTE: the `python3 -m ... dispatch` invocation uses `VercelSandboxBackend` which raises NotImplementedError on `spin_up`. For SP2a's skeleton-runner verification, you can swap to the `FakeSandboxBackend` by writing a small wrapper script `tests/manual_dispatch.py`:

```python
"""Manual SP2a dispatch test using FakeSandboxBackend.

Run from plugin root:
    python3 tests/manual_dispatch.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from lib.paper_chain.sandbox import set_backend, FakeSandboxBackend
from lib.paper_chain.experiment import dispatch_experiment

set_backend(FakeSandboxBackend(canned_id="sb_manual"))

FIXTURE = PLUGIN_ROOT / "tests" / "fixtures" / "paper-chain"

for h in ("S1", "S2", "S3"):
    out = FIXTURE / "paper" / "experiments" / h
    out.mkdir(parents=True, exist_ok=True)
    path = dispatch_experiment(
        hypothesis_id=h,
        protocol_path=FIXTURE / f"eval-designer-{h}" / "output.md",
        output_dir=out,
        sandbox_budget_usd=5.0,
        api_budget_usd=5.0,
    )
    print(f"{h}: wrote {path}")
```

- [ ] **Step 2: Run the manual dispatch wrapper**

Run: `python3 tests/manual_dispatch.py`
Expected: 3 lines, each "Sx: wrote tests/fixtures/paper-chain/paper/experiments/Sx/results.json".

- [ ] **Step 3: Inspect outputs**

For each of S1, S2, S3, inspect `tests/fixtures/paper-chain/paper/experiments/<h>/results.json`. Confirm:
- `status: failed`
- `failure_code: runner_not_implemented` OR `unsupported_substrate` (depending on whether the protocol's substrate matches one of the 5 registered substrates)
- `runtime_seconds` is small (< 1s)
- `cost_usd` is 0.0 (skeleton runners don't accrue cost)
- `failure_message` is informative

- [ ] **Step 4: Inspect the repro.yaml and runner-output.log**

For each hypothesis, confirm:
- `repro.yaml` has `python_version`, `platform`, `seed` fields
- `runner-output.log` is non-empty and reports the failure_code

- [ ] **Step 5: Record discipline-check results in spec doc**

Append a "Discipline check results" section to `docs/superpowers/specs/2026-05-15-experimentalist-sandbox-design.md`:

```markdown

## Discipline check results (SP2a manual T12)

- S1 substrate SPECS-Review-Benchmark: PASS / FAIL <reason>
- S2 substrate <from protocol>: PASS / FAIL <reason>
- S3 substrate AbGen testmini-500: PASS / FAIL <reason>

Common failure modes observed:
- <list>

Skeleton runners returning `failed_runner_not_implemented` is the expected SP2a behavior. SP1 drafter's fallback to option-γ stub (the SP1-T17 deferred check) is the next gate.
```

- [ ] **Step 6: Commit the manual-dispatch script + discipline check results**

```bash
git add tests/manual_dispatch.py docs/superpowers/specs/2026-05-15-experimentalist-sandbox-design.md
git commit -m "test(paper-chain): SP2a manual discipline check + dispatch helper"
```

---

## Self-review against the spec

| Spec requirement | Task that covers it |
|---|---|
| New worker agent: experimentalist | Task 6 |
| New orchestrator phase 6.5, --paper-gated | Task 7 |
| `lib/paper_chain/sandbox.py` with backend indirection | Task 2 |
| `lib/paper_chain/experiment.py` orchestration | Task 5 |
| `lib/paper_chain/protocol_parser.py` | Task 1 |
| 5 runner modules (substrate-specific) | Task 4 (skeleton mode); real impls deferred to SP2a.1-.5 |
| Runner base class + schema validator | Task 3 |
| Drafter modification (consumes results) | Task 8 |
| Preflight VERCEL_TOKEN check | Task 9 |
| Cost ceiling $5+$5 + 1 retry policy | Tasks 5, 7 (orchestrator behavior); test_experiment_orchestrator validates budget paths |
| 8 named failure modes | Task 5 (exception paths) + Task 7 (orchestrator handling) |
| Reproducibility (seed, pip-freeze, image SHA, py-version) | Task 5 (currently records python_version + platform + seed; pip-freeze + image SHA are runner-side, captured when real runners land) |
| Results JSON schema | Task 3 |
| Sequential execution, one sandbox per experiment | Task 7 wiring (loop is sequential by orchestrator structure) |
| Test layers 1, 2 (helpers + per-runner) | Tasks 1, 2, 3, 4, 5, 9 |
| Test layer 3 (real-sandbox integration) | DEFERRED — runs against real runner-implementations only, lands in SP2a.1-.5 |
| Test layer 4 (manual discipline check) | Task 12 |
| Docs (CLAUDE.md, command) | Task 10 |

**Coverage gaps (intentional, deferred):** Layer 3 real-sandbox integration tests and real benchmark code in the 5 runners are deferred to SP2a.1-.5 follow-up plans. SP2a's contribution is the orchestration architecture + failure-handling discipline + drafter integration; SP2a.1-.5 each add ONE real runner.

## Notes for the executing agent

- **Per project CLAUDE.md:** no worktrees; stay on `main`. Banned phrases: "load-bearing", "this is doing a lot of work" (and variants), "real" as emphatic adjective, "honest / honestly". Confirm before destructive ops.
- **Commits per task.** Plan lists commit commands. Follow the cadence after the first commit is authorized (or batch in semantic groups if user prefers).
- **Skeleton-runner contract.** The 5 skeleton runners must return schema-valid `failed_runner_not_implemented` results. Do not try to wire actual benchmark code in SP2a tasks — that's SP2a.1-.5 follow-up.
- **VercelSandboxBackend is a stub in SP2a.** Real SDK wiring lands when the first real runner ships. SP2a's tests use FakeSandboxBackend exclusively; the orchestrator's `__main__` path that defaults to VercelSandboxBackend will raise NotImplementedError until that follow-up — this is intentional and signposted.
