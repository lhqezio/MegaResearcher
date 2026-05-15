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
