# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared browser-session state and value helpers for search-space planning."""

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    Hyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)
from dash import dcc
from dash.development.base_component import Component

PLANNED_SEARCH_SPACE_STORE_ID = "planned-search-space-state"
PLANNED_SEARCH_SPACE_INTENT_TYPE = "planned-search-space-intent"
PLANNED_SEARCH_SPACE_STATE_VERSION = 1
DEFAULT_SLIDER_EXTENSION = 0.25
NUMERIC_PARAMETER_TYPES = (FloatHyperparameter, IntegerHyperparameter)


def as_python_value(value: Any) -> Any:
    """Convert NumPy scalar values to JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def transform_value(value: float, log: bool) -> float:
    """Transform a value into the parameter's sampling space."""
    value = float(value)
    if log:
        if value <= 0:
            raise ValueError("Log-scaled planning values must be positive.")
        return float(np.log(value))
    return value


def normalize_value(value: float, parameter: Mapping[str, Any]) -> float:
    """Normalize an original value relative to the current ConfigSpace bounds."""
    lower = transform_value(parameter["current_min"], bool(parameter["log"]))
    upper = transform_value(parameter["current_max"], bool(parameter["log"]))
    transformed = transform_value(value, bool(parameter["log"]))
    width = upper - lower
    if np.isclose(width, 0):
        return 0.5
    return float((transformed - lower) / width)


def denormalize_value(position: float, parameter: Mapping[str, Any]) -> float:
    """Map a normalized planner coordinate into the original parameter space."""
    lower = transform_value(parameter["current_min"], bool(parameter["log"]))
    upper = transform_value(parameter["current_max"], bool(parameter["log"]))
    transformed = lower + float(position) * (upper - lower)
    return float(np.exp(transformed)) if parameter["log"] else float(transformed)


def snap_value(value: float, parameter: Mapping[str, Any]) -> Any:
    """Snap a value to its integer/step lattice without clipping extensions."""
    step = parameter.get("step")
    if step is not None and float(step) > 0:
        lower = float(parameter["current_min"])
        value = lower + round((float(value) - lower) / float(step)) * float(step)
    if parameter["parameter_type"] == "integer":
        return int(round(value))
    return float(f"{float(value):.12g}")


def coerce_selection(
    minimum: float,
    maximum: float,
    parameter: Mapping[str, Any],
) -> Dict[str, Any]:
    """Snap, validate and order a numerical planning selection."""
    minimum = snap_value(float(minimum), parameter)
    maximum = snap_value(float(maximum), parameter)
    transform_value(minimum, bool(parameter["log"]))
    transform_value(maximum, bool(parameter["log"]))
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    return {"min": minimum, "max": maximum}


def parameter_type(hyperparameter: Hyperparameter) -> str:
    """Return the stable planner type for a ConfigSpace hyperparameter."""
    if isinstance(hyperparameter, IntegerHyperparameter):
        return "integer"
    if isinstance(hyperparameter, FloatHyperparameter):
        return "float"
    if isinstance(hyperparameter, OrdinalHyperparameter):
        return "ordinal"
    if isinstance(hyperparameter, Constant):
        return "constant"
    return "categorical"


def categorical_choices(hyperparameter: Hyperparameter) -> List[Any]:
    """Return JSON-compatible selectable values in ConfigSpace order."""
    if isinstance(hyperparameter, CategoricalHyperparameter):
        return [as_python_value(value) for value in hyperparameter.choices]
    if isinstance(hyperparameter, OrdinalHyperparameter):
        return [as_python_value(value) for value in hyperparameter.sequence]
    if isinstance(hyperparameter, Constant):
        return [as_python_value(hyperparameter.value)]
    return []


def parameter_metadata(hyperparameter: Hyperparameter, name: str, index: int) -> Dict[str, Any]:
    """Build the immutable planning metadata shared by both plugins."""
    kind = parameter_type(hyperparameter)
    metadata: Dict[str, Any] = {"index": index, "name": name, "parameter_type": kind}
    if isinstance(hyperparameter, NUMERIC_PARAMETER_TYPES):
        step = getattr(hyperparameter, "step", None)
        if step is None and hyperparameter.meta is not None:
            step = hyperparameter.meta.get("step")
        metadata.update(
            current_min=as_python_value(hyperparameter.lower),
            current_max=as_python_value(hyperparameter.upper),
            log=bool(getattr(hyperparameter, "log", False)),
            step=as_python_value(step),
        )
        if kind == "integer" and metadata["step"] is None:
            metadata["step"] = 1
    else:
        metadata["choices"] = categorical_choices(hyperparameter)
    return metadata


def configspace_metadata(configspace: Any, names: Sequence[str]) -> List[Dict[str, Any]]:
    """Build planning metadata for named ConfigSpace parameters."""
    return [parameter_metadata(configspace[name], name, index) for index, name in enumerate(names)]


def initial_selection(metadata: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Select the full current space represented by metadata."""
    selection: Dict[str, Dict[str, Any]] = {}
    for parameter in metadata:
        if parameter["parameter_type"] in ("integer", "float"):
            selection[str(parameter["name"])] = {
                "min": parameter["current_min"],
                "max": parameter["current_max"],
            }
        else:
            selection[str(parameter["name"])] = {"choices": list(parameter["choices"])}
    return selection


def reconcile_selection(
    metadata: Sequence[Mapping[str, Any]],
    stored: Any,
) -> Dict[str, Dict[str, Any]]:
    """Validate stored values against current metadata and fill missing parameters."""
    reconciled = initial_selection(metadata)
    if not isinstance(stored, dict):
        return reconciled
    for parameter in metadata:
        name = str(parameter["name"])
        candidate = stored.get(name)
        if not isinstance(candidate, dict):
            continue
        try:
            if parameter["parameter_type"] in ("integer", "float"):
                reconciled[name] = coerce_selection(candidate["min"], candidate["max"], parameter)
            else:
                if "choices" not in candidate:
                    continue
                selected = candidate["choices"]
                if not isinstance(selected, list):
                    continue
                reconciled[name] = {
                    "choices": [choice for choice in parameter["choices"] if choice in selected]
                }
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return reconciled


def empty_planning_state() -> Dict[str, Any]:
    """Return a new versioned browser-memory planning state."""
    return {"version": PLANNED_SEARCH_SPACE_STATE_VERSION, "revision": 0, "runs": {}}


def planning_state_store() -> Component:
    """Build the globally mounted, reload-scoped browser-memory store."""
    return dcc.Store(
        id=PLANNED_SEARCH_SPACE_STORE_ID,
        data=empty_planning_state(),
        storage_type="memory",
    )


def selection_from_state(
    state: Any,
    run_id: str,
    metadata: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Resolve and reconcile one run's selection from the global state."""
    valid_state = (
        state
        if isinstance(state, dict)
        and state.get("version") == PLANNED_SEARCH_SPACE_STATE_VERSION
        else empty_planning_state()
    )
    runs = valid_state.get("runs", {})
    stored = runs.get(run_id, {}) if isinstance(runs, dict) else {}
    return reconcile_selection(metadata, stored)


def apply_planning_intent(state: Any, intent: Any) -> Dict[str, Any]:
    """Apply one idempotent run-local patch or replacement to browser state."""
    current = (
        state
        if isinstance(state, dict)
        and state.get("version") == PLANNED_SEARCH_SPACE_STATE_VERSION
        else empty_planning_state()
    )
    runs = current.get("runs", {})
    runs = dict(runs) if isinstance(runs, dict) else {}
    if not isinstance(intent, dict) or not isinstance(intent.get("run_id"), str):
        return current
    run_id = intent["run_id"]
    selection = intent.get("selection")
    if not isinstance(selection, dict):
        return current
    if intent.get("mode") == "replace":
        runs[run_id] = selection
    else:
        previous = runs.get(run_id, {})
        merged = dict(previous) if isinstance(previous, dict) else {}
        merged.update(selection)
        runs[run_id] = merged
    return {
        "version": PLANNED_SEARCH_SPACE_STATE_VERSION,
        "revision": int(current.get("revision", 0)) + 1,
        "runs": runs,
    }


def selection_metrics(parameter: Mapping[str, Any], selection: Mapping[str, Any]) -> Dict[str, Any]:
    """Describe a numerical range in normalized planner coordinates."""
    minimum = normalize_value(selection["min"], parameter)
    maximum = normalize_value(selection["max"], parameter)
    span_ratio = maximum - minimum
    comparison = "same"
    if span_ratio > 1.01:
        comparison = "wider"
    elif span_ratio < 0.99:
        comparison = "narrower"
    return {
        "min_normalized": minimum,
        "max_normalized": maximum,
        "span_ratio": span_ratio,
        "left_extension": max(0.0, -minimum),
        "right_extension": max(0.0, maximum - 1.0),
        "extends": minimum < -1e-9 or maximum > 1.0 + 1e-9,
        "comparison": comparison,
    }


def slider_domain(metrics: Mapping[str, Any]) -> Tuple[float, float]:
    """Keep the reference visible while containing direct range extensions."""
    return (
        min(-DEFAULT_SLIDER_EXTENSION, float(metrics["min_normalized"]) - DEFAULT_SLIDER_EXTENSION),
        max(
            1.0 + DEFAULT_SLIDER_EXTENSION,
            float(metrics["max_normalized"]) + DEFAULT_SLIDER_EXTENSION,
        ),
    )


def position_percent(value: float, domain: Tuple[float, float]) -> float:
    """Return a normalized value's percentage position in a slider domain."""
    return 100 * (float(value) - domain[0]) / (domain[1] - domain[0])
