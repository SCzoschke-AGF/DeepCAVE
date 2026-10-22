"""Measure the trial cap using paired runs of the unchanged coverage planner.

Run from the repository root, for example::

    PYTHONPATH=. UV_CACHE_DIR=/tmp/codex-uv-cache OMP_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked python \
        examples/api/benchmark_coverage_planning.py \
        --output docs/validation/coverage_planning/results.json

The uncapped variant changes only MAX_ADDITIONAL_TRIALS in this process. Both
variants retain the same search algorithm, timeout, input data and planner seed.
Elapsed time excludes imports, input preparation and the final payload-size check.
"""

from typing import Any, Dict

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time
from unittest.mock import patch

import numpy as np
from ConfigSpace import (
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from deepcave.evaluators import coverage_planning as planner
from deepcave.evaluators.coverage import normalize_configurations


@dataclass(frozen=True)
class Case:
    """Describe a synthetic independent mixed continuous/integer search space."""

    dimensions: int
    levels: int
    strength: int
    design: str = "uniform"
    observations: int = 299

    @property
    def name(self) -> str:
        """Return a stable identifier for selecting and comparing cases."""
        return f"d{self.dimensions}_l{self.levels}_t{self.strength}_{self.design}"


CASES = (
    Case(6, 4, 3),
    Case(6, 4, 4),
    Case(6, 4, 5),
    Case(6, 4, 6),
    Case(6, 8, 3),
    Case(6, 8, 4),
    Case(8, 4, 4),
    Case(12, 8, 3),
    Case(20, 8, 3),
    Case(8, 4, 4, "clustered"),
    Case(12, 8, 4),
)
DATA_SEED = 20261021


def dataset(case: Case) -> Any:
    """Prepare fixed observations, retaining repeated configurations in history."""
    names = [f"p{i:02}" for i in range(case.dimensions)]
    space = ConfigurationSpace()
    space.add(
        [
            (
                UniformFloatHyperparameter(name, 0, 1)
                if i < case.dimensions - 2
                else UniformIntegerHyperparameter(name, 0, case.levels - 1)
            )
            for i, name in enumerate(names)
        ]
    )
    rng = np.random.default_rng(DATA_SEED)
    shape = (case.observations, case.dimensions)
    if case.design == "clustered":
        probabilities = [0.7] + [0.3 / (case.levels - 1)] * (case.levels - 1)
        points = rng.choice(case.levels, size=shape, p=probabilities)
    else:
        points = rng.integers(case.levels, size=shape)
    values = [
        {
            name: (
                float((point[i] + 0.5) / case.levels) if i < case.dimensions - 2 else int(point[i])
            )
            for i, name in enumerate(names)
        }
        for point in points
    ]
    return normalize_configurations(space, values)


def measure(data: Any, case: Case, seed: int, capped: bool, timeout: float) -> Dict[str, Any]:
    """Time planning and CPU usage, including validation and result construction."""
    state_type = planner.CoverageState
    candidates = planner._candidates
    phases: Dict[str, float] = {}
    state_calls = 0
    iterations = 0

    def state(*args: Any, **kwargs: Any) -> Any:
        nonlocal state_calls
        state_calls += 1
        if state_calls == 2:
            phases["validation_start"] = perf_counter()
        value = state_type(*args, **kwargs)
        if state_calls == 1:
            phases["initialization_end"] = perf_counter()
        return value

    def count_candidates(*args: Any, **kwargs: Any) -> Any:
        nonlocal iterations
        iterations += 1
        return candidates(*args, **kwargs)

    limit = planner.MAX_ADDITIONAL_TRIALS if capped else math.inf
    with (
        patch.object(planner, "MAX_ADDITIONAL_TRIALS", limit),
        patch.object(planner, "CoverageState", state),
        patch.object(planner, "_candidates", count_candidates),
    ):
        start, cpu_start = perf_counter(), process_time()
        plan = planner.generate_coverage_trial_plan(
            data,
            {
                "strategy": "minimize_trials",
                "resolution": case.levels,
                "strength_mode": "custom",
                "target_strength": case.strength,
                "max_additional_trials": None,
                "solver_timeout": timeout,
                "seed": seed,
            },
        )
        cpu_seconds = process_time() - cpu_start
        end = perf_counter()
    assert state_calls == 2
    assert len(plan["trials"]) <= iterations
    assert not capped or iterations <= planner.MAX_ADDITIONAL_TRIALS
    assert plan["target_complete"] == (plan["missing_tuples_after"] == 0)
    assert plan["projected_coverage"] >= plan["initial_coverage"]
    payload_bytes = len(json.dumps(plan, separators=(",", ":"), allow_nan=False).encode())
    return {
        "case": case.name,
        "seed": seed,
        "variant": "cap_1000" if capped else "uncapped",
        "wall_seconds": end - start,
        "cpu_seconds": cpu_seconds,
        "initialization_seconds": phases["initialization_end"] - start,
        "search_and_pruning_seconds": phases["validation_start"] - phases["initialization_end"],
        "validation_and_output_seconds": end - phases["validation_start"],
        "iterations": iterations,
        "proposed_trials": len(plan["trials"]),
        "initial_coverage": plan["initial_coverage"],
        "projected_coverage": plan["projected_coverage"],
        "missing_before": plan["missing_tuples_before"],
        "missing_after": plan["missing_tuples_after"],
        "lower_bound_trials": plan["lower_bound_trials"],
        "complete": plan["target_complete"],
        "termination_reason": plan["termination_reason"],
        "plan_id": plan["plan_id"],
        "payload_bytes": payload_bytes,
    }


def environment() -> Dict[str, Any]:
    """Record enough implementation and machine information to reproduce the run."""
    root = Path(__file__).resolve().parents[2]
    modules = ("coverage_planning", "coverage_state", "combinatorial", "coverage")
    cpu_info = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    cpu_model = next(
        (
            line.split(":", 1)[1].strip()
            for line in cpu_info.splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "cpu_affinity_count": (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "ConfigSpace")},
        "source_sha256": {
            name: hashlib.sha256(
                (root / "deepcave/evaluators" / f"{name}.py").read_bytes()
            ).hexdigest()
            for name in modules
        },
    }


def main() -> None:
    """Run sequential paired experiments and persist each completed measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--cases", nargs="+", choices=[case.name for case in CASES])
    args = parser.parse_args()
    cases = [case for case in CASES if not args.cases or case.name in args.cases]
    output = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
        "data_seed": DATA_SEED,
        "solver_timeout_seconds": args.timeout,
        "production_trial_cap": planner.MAX_ADDITIONAL_TRIALS,
        "planner_seeds": args.seeds,
        "cases": [{"name": case.name, **asdict(case)} for case in cases],
        "runs": [],
    }
    # Warm imports and common code paths before recording the first pair.
    warmup = Case(6, 4, 3)
    measure(dataset(warmup), warmup, 0, True, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        data = dataset(case)
        for index, seed in enumerate(args.seeds):
            pair = []
            # Alternate order to reduce bias from temperature and system load.
            for capped in ((True, False) if index % 2 == 0 else (False, True)):
                record = measure(data, case, seed, capped, args.timeout)
                output["runs"].append(record)
                pair.append(record)
                temporary = args.output.with_suffix(".tmp")
                temporary.write_text(json.dumps(output, indent=2) + "\n")
                temporary.replace(args.output)
                print(
                    f"{case.name} seed={seed} {record['variant']}: "
                    f"wall={record['wall_seconds']:.3f}s cpu={record['cpu_seconds']:.3f}s "
                    f"search={record['search_and_pruning_seconds']:.3f}s "
                    f"trials={record['proposed_trials']} coverage={record['projected_coverage']:.3%} "
                    f"stop={record['termination_reason']}",
                    flush=True,
                )
            if all(item["termination_reason"] == "target_reached" for item in pair):
                assert pair[0]["plan_id"] == pair[1]["plan_id"], "Non-binding cap changed the plan"
    output["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
