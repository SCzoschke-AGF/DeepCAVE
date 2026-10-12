"""Computations for cost and objective analysis, independent of Dash and plotting."""

from typing import Any, Dict, List, Mapping, Sequence

from types import SimpleNamespace

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace, Constant

from deepcave.constants import COMBINED_COST_NAME
from deepcave.evaluators.fanova import fANOVA
from deepcave.evaluators.rashomon import select_near_optimal_indices
from deepcave.runs import AbstractRun

# The only cost target in v1. Keep its field and label separate from objective handling.
COST_METRIC = {"field": "duration", "label": "Trial duration", "unit": "s"}


def duration_trend(
    rows: Sequence[Dict[str, Any]], parameter: str, target: str = "duration"
) -> Dict[str, List[float]]:
    """Return robust binned target medians and IQRs over observed numeric parameter values.

    Keep equal parameter values in one bin. Use up to 20 contiguous bins over distinct values,
    with roughly sqrt(n) bins; require at least three distinct values for a diagnostic trend.
    This summarizes observed associations, without fitting a causal or extrapolating model.
    """
    observed = [row for row in rows if row["configuration"].get(parameter) is not None]
    values = np.asarray([row["configuration"][parameter] for row in observed], dtype=float)
    durations = np.asarray([row[target] for row in observed], dtype=float)
    unique = np.unique(values)
    result: Dict[str, List[float]] = {"x": [], "median": [], "lower": [], "upper": []}
    if len(unique) < 3:
        return result
    bins = min(len(unique), 20, max(3, int(np.sqrt(len(observed)))))
    for group in np.array_split(unique, bins):
        mask = (values >= group[0]) & (values <= group[-1])
        lower, median, upper = np.quantile(durations[mask], [0.25, 0.5, 0.75])
        result["x"].append(float(np.median(values[mask])))
        for key, value in (("lower", lower), ("median", median), ("upper", upper)):
            result[key].append(float(value))
    return result


def select_elite_indices(
    rows: Sequence[Dict[str, Any]], fraction: float = 0.05, maximize: bool = False
) -> List[int]:
    """Select ceil(n * fraction) trials, at least one, using the existing elite definition.

    Equal objective values are ordered by their stable extraction index (source then history).
    Ties at the boundary do not expand the set, consistently with Near-Optimal Configurations.
    """
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Elite fraction must be in (0, 1].")
    return select_near_optimal_indices(
        [row["objective"] for row in rows],
        list(range(len(rows))),
        "upper" if maximize else "lower",
        "top_percent",
        100 * fraction,
    ).tolist()


def importance_comparison(
    cost: Mapping[str, float], objective: Mapping[str, float]
) -> Dict[str, Any]:
    """Compare finite main effects with strict shared median thresholds; avoid ratios."""
    names = sorted(
        name
        for name in cost.keys() & objective.keys()
        if np.isfinite(cost[name]) and np.isfinite(objective[name])
    )
    if not names:
        return {"parameters": [], "cost_median": None, "objective_median": None, "candidates": []}
    cost_median = float(np.median([cost[name] for name in names]))
    objective_median = float(np.median([objective[name] for name in names]))
    candidates: List[Dict[str, Any]] = [
        {
            "id": name,
            "parameter": name,
            "objective_importance": objective[name],
            "cost_importance": cost[name],
            "classification": "Candidate",
        }
        for name in names
        if cost[name] > cost_median and objective[name] < objective_median
    ]
    candidates.sort(key=lambda row: (-row["cost_importance"], row["parameter"]))
    return {
        "parameters": names,
        "cost_median": cost_median,
        "objective_median": objective_median,
        "candidates": candidates,
    }


def elite_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize elite cost; a zero minimum has no finite max/min ratio."""
    durations = [row["duration"] for row in rows]
    if not durations:
        return {"count": 0, "minimum": None, "median": None, "maximum": None, "ratio": None}
    minimum, maximum = min(durations), max(durations)
    return {
        "count": len(rows),
        "minimum": minimum,
        "median": float(np.median(durations)),
        "maximum": maximum,
        "ratio": maximum / minimum if minimum > 0 else None,
    }


def calculate_importances(
    run: AbstractRun, rows: Sequence[Dict[str, Any]], n_trees: int = 16
) -> Dict[str, Any]:
    """Reuse fANOVA on identical valid trial samples for duration and objective targets.

    A small read-only data adapter supplies the existing evaluator with encoded trials rather
    than configuration averages. Constants are represented as one-choice categoricals because
    pyrfr cannot marginalize numerical constants. This preserves conditions and forbiddens,
    changes neither the source run nor its configurations, and assigns constants zero importance.
    Undefined main effects (constant targets / zero forest variance) remain unavailable.
    """
    names = list(run.configspace.keys())
    result: Dict[str, Any] = {"cost": {}, "objective": {}, "errors": {}}
    if len(rows) < 2 or not names:
        result["errors"] = {
            "cost": "Insufficient valid trials.",
            "objective": "Insufficient valid trials.",
        }
        return result

    serialized = run.configspace.to_serialized_dict()
    constants = {name for name, hp in run.configspace.items() if isinstance(hp, Constant)}
    for index, hp in enumerate(serialized["hyperparameters"]):
        if hp["name"] in constants:
            serialized["hyperparameters"][index] = {
                "type": "categorical",
                "name": hp["name"],
                "choices": [hp["value"]],
                "weights": None,
                "default_value": hp["value"],
                "meta": hp.get("meta"),
            }
    cs = ConfigurationSpace.from_serialized_dict(serialized)
    matrix = np.asarray([Configuration(cs, row["configuration"]).get_array() for row in rows])

    for target, field in (("cost", COST_METRIC["field"]), ("objective", "objective")):
        values = np.asarray([row[field] for row in rows], dtype=float)
        if np.ptp(values) == 0:
            result["errors"][target] = "Importance is undefined: the target has no variation."
            continue
        frame = pd.DataFrame(matrix, columns=list(cs.keys()))
        frame[COMBINED_COST_NAME] = values
        adapter = SimpleNamespace(
            configspace=cs,
            get_encoded_data=lambda *args, **kwargs: frame,
            get_highest_budget=lambda: 0,
        )
        try:
            # fANOVA reads only configspace and get_encoded_data from this read-only adapter.
            evaluator = fANOVA(adapter)  # type: ignore[arg-type]
            evaluator.calculate(objectives=[], budget=0, n_trees=n_trees, seed=0)
            raw = evaluator.get_importances([name for name in names if name not in constants])
            result[target] = {
                name: float(effect[0]) for name, effect in raw.items() if np.isfinite(effect[0])
            }
            result[target].update({name: 0.0 for name in constants})
            if not raw or len(result[target]) != len(names):
                result[target] = {}
                result["errors"][
                    target
                ] = "Insufficient variation to estimate finite fANOVA main effects."
        except (RuntimeError, ValueError, TypeError) as error:
            result["errors"][target] = f"fANOVA unavailable: {error}"
    return result
