# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared browser-session state and value helpers for search-space planning."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import hashlib
import json
from threading import RLock

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
PLANNED_SEARCH_SPACE_HYDRATION_ID = "planned-search-space-hydration"
PLANNED_SEARCH_SPACE_INTENT_TYPE = "planned-search-space-intent"
PLANNED_SEARCH_SPACE_CONTEXT_TYPE = "planned-search-space-context"
PLANNED_SEARCH_SPACE_CACHE_KEY = "planned_search_space_state"
LEGACY_SAVED_SEARCH_SPACES_KEY = "hpo_planning_saved_search_spaces"
PLANNED_SEARCH_SPACE_STATE_VERSION = 2
DEFAULT_SLIDER_EXTENSION = 0.25
NUMERIC_PARAMETER_TYPES = (FloatHyperparameter, IntegerHyperparameter)
_PERSISTENCE_LOCK = RLock()


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
    if not np.isfinite(value):
        raise ValueError("Planning values must be finite.")
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


def planning_metadata_fingerprint(metadata: Sequence[Mapping[str, Any]]) -> str:
    """Return a stable fingerprint of definitions which constrain stored selections."""
    definition_keys = (
        "name",
        "parameter_type",
        "current_min",
        "current_max",
        "log",
        "step",
        "choices",
    )
    definitions = [
        {key: parameter[key] for key in definition_keys if key in parameter}
        for parameter in metadata
    ]
    serialized = json.dumps(
        definitions,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def objective_persistence_key(objective: Any) -> str:
    """Return a stable JSON object key for a native or derived objective."""
    if isinstance(objective, bool) or objective is None:
        raise ValueError("A valid objective is required for the planned search space.")
    if isinstance(objective, (int, np.integer)):
        return f"objective:{int(objective)}"
    if isinstance(objective, str) and objective:
        if objective.startswith("objective:"):
            return objective
        try:
            return f"objective:{int(objective)}"
        except ValueError:
            return objective
    raise ValueError("A valid objective is required for the planned search space.")


def planning_context_fingerprint(metadata_version: str, objective: Any) -> str:
    """Bind mounted control identities to metadata and the selected objective."""
    serialized = json.dumps(
        [metadata_version, objective_persistence_key(objective)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _same_typed_value(left: Any, right: Any) -> bool:
    """Compare category values without conflating booleans and integers."""
    return type(left) is type(right) and left == right


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
        if isinstance(candidate, list):
            candidate = (
                {"min": candidate[0], "max": candidate[1]}
                if parameter["parameter_type"] in ("integer", "float") and len(candidate) == 2
                else {"choices": candidate}
            )
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
                    "choices": [
                        choice
                        for choice in parameter["choices"]
                        if any(_same_typed_value(choice, item) for item in selected)
                    ]
                }
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return reconciled


def empty_planning_state() -> Dict[str, Any]:
    """Return a new versioned objective-scoped browser planning state."""
    return {
        "version": PLANNED_SEARCH_SPACE_STATE_VERSION,
        "revision": 0,
        "runs": {},
        "event_sequences": {},
        "last_change": None,
    }


def planning_state_from_cache(cache: Any) -> Dict[str, Any]:
    """Hydrate the browser state from persisted run/objective selections."""
    state = empty_planning_state()
    cached = cache.get(PLANNED_SEARCH_SPACE_CACHE_KEY)
    if not isinstance(cached, dict):
        cached = cache.get(LEGACY_SAVED_SEARCH_SPACES_KEY)
    if isinstance(cached, dict):
        state["runs"] = cached
    return state


def persist_planning_state_change(cache: Any, state: Any) -> int:
    """Persist only the objective scope changed by the latest browser commit."""
    if not isinstance(state, dict) or state.get("version") != PLANNED_SEARCH_SPACE_STATE_VERSION:
        raise ValueError("The planned search-space state has an unsupported version.")
    change = state.get("last_change")
    if not isinstance(change, dict):
        raise ValueError("The planned search-space state has no changed scope.")
    run_id = change.get("run_id")
    objective_key = change.get("objective_key")
    if not isinstance(run_id, str) or not run_id or not isinstance(objective_key, str):
        raise ValueError("The planned search-space state has an invalid changed scope.")
    runs = state.get("runs")
    run_scopes = runs.get(run_id) if isinstance(runs, dict) else None
    selection = run_scopes.get(objective_key) if isinstance(run_scopes, dict) else None
    if not isinstance(selection, dict):
        raise ValueError("The changed planned search-space selection is invalid.")
    with _PERSISTENCE_LOCK:
        cache.set(PLANNED_SEARCH_SPACE_CACHE_KEY, run_id, objective_key, value=selection)
    return int(state.get("revision", 0))


def planning_state_store(data: Optional[Dict[str, Any]] = None) -> Component:
    """Build the globally mounted browser store, optionally hydrated from disk."""
    return dcc.Store(
        id=PLANNED_SEARCH_SPACE_STORE_ID,
        data=data if data is not None else empty_planning_state(),
        storage_type="memory",
    )


def selection_from_state(
    state: Any,
    run_id: str,
    objective: Any,
    metadata: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Resolve one run/objective selection from the global state."""
    valid_state = (
        state
        if isinstance(state, dict) and state.get("version") == PLANNED_SEARCH_SPACE_STATE_VERSION
        else empty_planning_state()
    )
    runs = valid_state.get("runs", {})
    run_scopes = runs.get(run_id, {}) if isinstance(runs, dict) else {}
    try:
        objective_key = objective_persistence_key(objective)
    except ValueError:
        objective_key = ""
    stored = run_scopes.get(objective_key, {}) if isinstance(run_scopes, dict) else {}
    return reconcile_selection(metadata, stored)


def _intent_sequence(intent: Mapping[str, Any]) -> Optional[int]:
    """Read an optional non-negative event sequence."""
    value = intent.get("sequence")
    if isinstance(value, bool):
        return None
    try:
        sequence = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return sequence if sequence >= 0 else None


def _valid_intent_selection(value: Any) -> bool:
    """Reject malformed and non-finite values before they can damage browser state."""
    if not isinstance(value, dict):
        return False
    if "min" in value or "max" in value:
        if set(value) != {"min", "max"}:
            return False
        try:
            return bool(np.isfinite(float(value["min"]))) and bool(np.isfinite(float(value["max"])))
        except (TypeError, ValueError, OverflowError):
            return False
    choices = value.get("choices")
    if set(value) != {"choices"} or not isinstance(choices, list):
        return False
    return all(
        not isinstance(choice, (float, np.floating)) or bool(np.isfinite(choice))
        for choice in choices
    )


def apply_planning_intent(state: Any, intent: Any) -> Dict[str, Any]:
    """Apply one idempotent, ordered run/objective-local patch or reset."""
    current = (
        state
        if isinstance(state, dict) and state.get("version") == PLANNED_SEARCH_SPACE_STATE_VERSION
        else empty_planning_state()
    )
    if not isinstance(intent, dict) or not isinstance(intent.get("run_id"), str):
        return current
    run_id = intent["run_id"]
    try:
        objective_key = objective_persistence_key(intent.get("objective"))
    except ValueError:
        return current
    selection = intent.get("selection")
    if not isinstance(selection, dict):
        return current

    runs_value = current.get("runs", {})
    runs = dict(runs_value) if isinstance(runs_value, dict) else {}
    run_scopes_value = runs.get(run_id, {})
    run_scopes = dict(run_scopes_value) if isinstance(run_scopes_value, dict) else {}
    previous_value = run_scopes.get(objective_key, {})
    previous = dict(previous_value) if isinstance(previous_value, dict) else {}

    sequence = _intent_sequence(intent)
    source = str(intent.get("source", "unknown"))
    sequences_value = current.get("event_sequences", {})
    sequences = dict(sequences_value) if isinstance(sequences_value, dict) else {}
    sequence_scope = f"{run_id}:{objective_key}"
    run_sequences_value = sequences.get(sequence_scope, {})
    run_sequences = dict(run_sequences_value) if isinstance(run_sequences_value, dict) else {}

    accepted: Dict[str, Any] = {}
    for name, value in selection.items():
        if not isinstance(name, str) or not _valid_intent_selection(value):
            continue
        event_key = f"{source}:{name}"
        previous_sequence = run_sequences.get(event_key)
        if (
            sequence is not None
            and isinstance(previous_sequence, int)
            and sequence <= previous_sequence
        ):
            continue
        accepted[name] = value
        if sequence is not None:
            run_sequences[event_key] = sequence

    if intent.get("mode") == "replace" and len(accepted) != len(selection):
        return current
    if intent.get("mode") == "replace":
        updated_run = accepted
    else:
        updated_run = dict(previous)
        updated_run.update(accepted)

    # Invalid or stale patches and semantically identical events are true no-ops. In particular,
    # they must not cause downstream figures to render again merely by increasing a revision.
    if updated_run == previous:
        return current

    run_scopes[objective_key] = updated_run
    runs[run_id] = run_scopes
    if sequence is not None:
        sequences[sequence_scope] = run_sequences
    return {
        "version": PLANNED_SEARCH_SPACE_STATE_VERSION,
        "revision": int(current.get("revision", 0)) + 1,
        "runs": runs,
        "event_sequences": sequences,
        "last_change": {"run_id": run_id, "objective_key": objective_key},
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
