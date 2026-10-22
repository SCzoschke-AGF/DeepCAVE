"""Bounded, reproducible coverage planning for independent search spaces."""

from typing import Any, Dict, List, Optional

import hashlib
import json
from itertools import product
from math import prod
from time import monotonic

import numpy as np

from deepcave.evaluators.combinatorial import RESOLUTIONS, CoverageSpec
from deepcave.evaluators.coverage import CoverageData, _python
from deepcave.evaluators.coverage_state import CoverageState as _CoverageState

PLANNING_VERSION = 1
TRIAL_ESTIMATE_VERSION = 1
MAX_GROUPS = 10_000
MAX_TUPLES = 5_000_000
MAX_PROJECTIONS = 20_000_000
MAX_ADDITIONAL_TRIALS = 1000
CANDIDATE_PROJECTIONS = 250_000


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    try:
        valid = not isinstance(value, bool) and int(value) == value and lower <= value <= upper
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer between {lower} and {upper}.")
    return int(value)


def planning_options(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate strategy-specific controls without measuring coverage or starting work."""
    strategy = inputs.get("strategy", "maximize_coverage")
    if strategy not in ("maximize_coverage", "minimize_trials"):
        raise ValueError("Select Maximize Coverage or Minimize Trials.")
    resolution = _integer(inputs.get("resolution", 2), "Resolution", 2, 8)
    if resolution not in RESOLUTIONS:
        raise ValueError("Select a resolution of 2, 3, 4 or 8 levels.")
    mode = inputs.get("strength_mode", "next")
    if mode not in ("next", "custom", "maximum"):
        raise ValueError("Select Next, Custom or Maximum strength.")
    strength = (
        _integer(inputs.get("target_strength"), "Target strength", 1, 1000)
        if mode == "custom"
        else None
    )
    budget = inputs.get("max_additional_trials")
    if budget is not None or strategy == "maximize_coverage":
        budget = _integer(budget, "Additional trial budget", 1, MAX_ADDITIONAL_TRIALS)
    timeout = inputs.get("solver_timeout", 30)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (float, int))
        or not 0.1 <= timeout <= 120
    ):
        raise ValueError("Search time must be between 0.1 and 120 seconds.")
    return {
        "strategy": strategy,
        "resolution": resolution,
        "strength_mode": mode,
        "target_strength": strength,
        "max_additional_trials": budget,
        "solver_timeout": float(timeout),
        "seed": _integer(inputs.get("seed", 0), "Random seed", 0, 2**32 - 1),
    }


def planning_source_error(run: Any) -> Optional[str]:
    """Reject unsupported search spaces independently of recorded sampler metadata."""
    space = run.configspace
    if space.conditions or space.forbidden_clauses:
        return "Planning supports independent search spaces without conditions or constraints."
    return None


def estimate_coverage_trials(
    data: CoverageData, observed: Dict[str, Any], seed: int = 0
) -> Dict[str, Any]:
    """Attach cached, bounded next-strength estimates inside an explicit worker job."""
    if observed.get("trial_estimate_version") == TRIAL_ESTIMATE_VERSION:
        return observed
    dimensions = len(observed["parameters"])
    rows = []
    for row in observed["rows"]:
        current = row["observed_strength"]
        target = current + 1 if current < dimensions else None
        estimate: Dict[str, Any] = {
            "target_strength": target,
            "trials": None,
            "optimality_proven": False,
            "status": "Unavailable",
        }
        if row["strength_is_lower_bound"]:
            estimate["target_strength"] = None
            estimate["reason"] = "Current strength is only a lower bound; next target is unknown."
        elif target is None:
            estimate.update(trials=0, optimality_proven=True, status="Completed")
        else:
            try:
                if target == dimensions:
                    # Full-strength gaps each require exactly one trial; no search is needed.
                    spec = CoverageSpec(data, row["resolution"])
                    count = prod(spec.sizes) - len({tuple(point) for point in spec.points})
                    estimate.update(trials=count, optimality_proven=True, status="Completed")
                else:
                    plan = generate_coverage_trial_plan(
                        data,
                        {
                            "strategy": "minimize_trials",
                            "resolution": row["resolution"],
                            "strength_mode": "custom",
                            "target_strength": target,
                            "solver_timeout": 1.0,
                            "seed": seed,
                        },
                    )
                    estimate.update(
                        trials=len(plan["trials"]) if plan["target_complete"] else None,
                        optimality_proven=plan["optimality_proven"],
                        status="Completed" if plan["target_complete"] else "Incomplete",
                        lower_bound_trials=plan["lower_bound_trials"],
                        proposed_trials=len(plan["trials"]),
                        reason=plan["termination_reason"],
                    )
            except ValueError as error:
                estimate["reason"] = str(error)
        rows.append({**row, "trial_estimate": estimate})
    return {**observed, "rows": rows, "trial_estimate_version": TRIAL_ESTIMATE_VERSION}


class CoverageState(_CoverageState):
    """Apply the planner's configurable limits to shared missing-tuple accounting."""

    def __init__(self, spec: CoverageSpec, strength: int) -> None:
        super().__init__(spec, strength, MAX_GROUPS, MAX_TUPLES, MAX_PROJECTIONS)


def _candidates(spec: CoverageSpec, state: CoverageState, rng: np.random.Generator) -> np.ndarray:
    if prod(spec.sizes) <= 4096:
        return np.asarray(list(product(*(range(size) for size in spec.sizes))), dtype=np.int64)
    count = max(8, min(128, CANDIDATE_PROJECTIONS // len(state.groups)))
    points = np.column_stack([rng.integers(size, size=count) for size in spec.sizes])
    if len(spec.points):
        for index in range(count // 4):
            points[index] = spec.points[rng.integers(len(spec.points))]
            column = rng.integers(len(spec.axes))
            points[index, column] = rng.integers(spec.sizes[column])
    missing = np.flatnonzero(state.missing)
    for index, tuple_id in enumerate(rng.choice(missing, size=count // 2)):
        state.complete_tuple(points[index], int(tuple_id))
    return np.unique(points, axis=0)


def _improve(
    point: np.ndarray, state: CoverageState, rng: np.random.Generator, deadline: float
) -> np.ndarray:
    for column in rng.permutation(len(point)):
        if monotonic() >= deadline:
            break
        size = int(state.sizes[column])
        values = np.arange(size) if size <= 32 else rng.choice(size, 32, replace=False)
        variants = np.repeat(point[None, :], len(values) + 1, axis=0)
        variants[1:, column] = values
        point = variants[int(np.argmax(state.gains(variants)))].copy()
    return point


def generate_coverage_trial_plan(data: CoverageData, options: Dict[str, Any]) -> Dict[str, Any]:
    """Return a validated heuristic plan; never execute trials or modify observed coverage."""
    options = planning_options(options)
    spec = CoverageSpec(data, options["resolution"])
    if options["strength_mode"] == "next":
        for strength in range(1, len(spec.axes) + 1):
            state = CoverageState(spec, strength)
            if state.missing.any():
                break
    else:
        strength = (
            len(spec.axes) if options["strength_mode"] == "maximum" else options["target_strength"]
        )
        state = CoverageState(spec, strength)
    initial_missing = state.missing.copy()
    missing_before = int(initial_missing.sum())
    lower_bound = state.lower_bound()
    rng = np.random.default_rng(options["seed"])
    deadline = monotonic() + options["solver_timeout"]
    limit = options["max_additional_trials"] or MAX_ADDITIONAL_TRIALS
    selected: List[np.ndarray] = []
    remaining = missing_before
    reason = "target_reached"
    while remaining:
        if len(selected) >= limit:
            reason = "budget_exhausted" if options["max_additional_trials"] else "trial_limit"
            break
        if monotonic() >= deadline:
            reason = "timeout"
            break
        candidates = _candidates(spec, state, rng)
        gains = state.gains(candidates)
        best = int(gains.max())
        if best == 0:
            reason = "no_gain_found"
            break
        point = candidates[rng.choice(np.flatnonzero(gains == best))].copy()
        point = _improve(point, state, rng, deadline)
        remaining -= len(state.add(point))
        selected.append(point)

    if options["strategy"] == "minimize_trials" and not remaining and selected:
        counts = np.zeros(state.total, dtype=np.uint16)
        for point in selected:
            counts[state.tuple_ids(point)[0]] += 1
        for index in range(len(selected) - 1, -1, -1):
            ids = state.tuple_ids(selected[index])[0]
            needed = ids[initial_missing[ids]]
            if np.all(counts[needed] > 1):
                counts[ids] -= 1
                selected.pop(index)

    # Independently validate materialized values, then rebuild the complete marginal-gain trajectory.
    validated = CoverageState(spec, strength)
    materializer = np.random.default_rng(options["seed"])
    trials = []
    remaining = missing_before
    for point in selected:
        parameters, encoded = spec.materialize(point, materializer)
        new = validated.add(encoded)
        if not len(new):
            continue
        remaining -= len(new)
        closed = []
        for tuple_id in new[:20]:
            group = int(np.searchsorted(validated.offsets, tuple_id, side="right") - 1)
            closed.append(
                {
                    spec.names[column]: spec.definitions[spec.names[column]][int(encoded[column])][
                        "range"
                    ]
                    for column in validated.groups[group]
                }
            )
        trials.append(
            {
                "position": len(trials) + 1,
                "parameters": parameters,
                "levels": dict(zip(spec.names, map(int, encoded))),
                "new_combinations": len(new),
                "cumulative_coverage": (validated.total - remaining) / validated.total,
                "remaining_missing": remaining,
                "closed_combinations": closed,
            }
        )
    definition = spec.serialize()
    space_version = hashlib.sha256(
        json.dumps(
            [spec.configspace.to_serialized_dict(), definition], sort_keys=True, default=_python
        ).encode()
    ).hexdigest()
    plan_id = hashlib.sha256(
        json.dumps(
            [data.fingerprint, space_version, options, trials], sort_keys=True, default=_python
        ).encode()
    ).hexdigest()[:16]
    return {
        "version": PLANNING_VERSION,
        "plan_id": plan_id,
        **options,
        "target_strength": strength,
        "source_fingerprint": data.fingerprint,
        "search_space_version": space_version,
        "search_space": definition,
        "observed_trials": data.count,
        "initial_coverage": (state.total - missing_before) / state.total,
        "projected_coverage": (validated.total - remaining) / validated.total,
        "missing_tuples_before": missing_before,
        "missing_tuples_after": remaining,
        "lower_bound_trials": lower_bound,
        "target_complete": remaining == 0,
        "optimality_proven": remaining == 0
        and (options["strategy"] == "maximize_coverage" or len(trials) == lower_bound),
        "termination_reason": reason,
        "budget_target_impossible": options["max_additional_trials"] is not None
        and options["max_additional_trials"] < lower_bound,
        "trials": trials,
    }
