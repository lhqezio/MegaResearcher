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
