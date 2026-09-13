# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# noqa: D400
"""Empirical single-hyperparameter ranges for planning subsequent HPO runs."""

from typing import Any, Callable, Dict, List, Sequence, Tuple

import math
import pprint

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    Hyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import config, interactive
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.plugins.hyperparameter.analysis_utils import (
    _get_objectives_without_time,
    _is_optuna_run_or_group,
)
from deepcave.runs import AbstractRun
from deepcave.runs.status import Status
from deepcave.utils.analysis_layout import (
    NONBREAKING_RANGE_SEPARATOR,
    analysis_toolbar_item,
    bin_label,
    bin_scale_exponent,
    style_analysis_panel,
    superscript_exponent,
)
from deepcave.utils.layout import get_select_options, help_button

ELITE_PROPORTION_DEFAULTS = (0.10, 0.05, 0.01)
ELITE_PROPORTION_MIN = 0.001
ELITE_PROPORTION_MAX = 0.99
NUMERIC_PARAMETER_TYPES = (FloatHyperparameter, IntegerHyperparameter)
ELITE_MARKER_COLORS = ("#440154", "#21918C", "#D95F02")
DEFAULT_SLIDER_EXTENSION = 0.25
SPAN_COMPARISON_TOLERANCE = 0.01
DEFAULT_TOP_TRIAL_COUNT = 10
TOP_TRIAL_LIMIT = 100
INCLUDED_TRIAL_COLOR = "#52B788"
EXCLUDED_TRIAL_COLOR = "#DC3545"
INSIGHT_BIN_COUNT = 10
INSIGHT_Q90_COLOR = "#FF7F0E"
INSIGHT_MEDIAN_COLOR = "#43A047"
INSIGHT_TOP_TRIAL_COLOR = "#27364A"
INSIGHT_ELITE_BAR_OPACITY = 0.65
INSIGHT_TOP_TRIAL_RUG_Y = -0.12
INSIGHT_TOP_TRIAL_STEM_Y = -0.07
INSIGHT_X_TITLE_Y = -0.22
CACHE_SCHEMA_VERSION = 10

USER_ATTR_PREFIX = "user_attr:"

METADATA_STORE_ID = "hpo-planning-metadata"
SELECTION_STORE_ID = "hpo-planning-selection"
TRIALS_STORE_ID = "hpo-planning-trials"
SIDEBAR_BODY_ID = "hpo-planning-sidebar-body"
ROWS_ID = "hpo-planning-rows"
DICT_PREVIEW_ID = "hpo-planning-dict-preview"
COPY_ID = "hpo-planning-copy"
RESET_ID = "hpo-planning-reset"
COVERAGE_GRAPH_ID = "hpo-planning-coverage-graph"
COVERAGE_BADGE_ID = "hpo-planning-coverage-badge"
COVERAGE_TITLE_ID = "hpo-planning-coverage-title"
COVERAGE_SUBTITLE_ID = "hpo-planning-coverage-subtitle"
SPREAD_ATTRIBUTE_ID = "hpo-planning-spread-attribute"
SPREAD_TOGGLE_ID = "hpo-planning-spread-toggle"
SPREAD_MODE_ID = "hpo-planning-spread-mode"
SPREAD_SCATTER_ID = "hpo-planning-spread-scatter"
TOP_TRIAL_COUNT_ID = "hpo-planning-top-trial-count"
INSIGHTS_STORE_ID = "hpo-planning-insights"
INSIGHT_SELECTION_STORE_ID = "hpo-planning-insight-selection"
INSIGHT_MODAL_ID = "hpo-planning-insight-modal"
INSIGHT_MODAL_TITLE_ID = "hpo-planning-insight-modal-title"
INSIGHT_GRAPH_ID = "hpo-planning-insight-graph"
INSIGHT_ELITE_CONTROL_ID = "hpo-planning-insight-elite-group"
INSIGHT_CLOSE_ID = "hpo-planning-insight-close"
INSIGHT_BUTTON_TYPE = "hpo-planner-insight"


def _validate_elite_proportions(values: Sequence[Any]) -> List[float]:
    """Validate and order the three configured elite proportions."""
    proportions = [float(value) for value in values]
    if len(proportions) != 3:
        raise ValueError("Exactly three elite proportions are required.")
    if any(
        not ELITE_PROPORTION_MIN <= proportion <= ELITE_PROPORTION_MAX for proportion in proportions
    ):
        raise ValueError(
            "Elite proportions must be between "
            f"{ELITE_PROPORTION_MIN} and {ELITE_PROPORTION_MAX}."
        )
    if len(set(proportions)) != len(proportions):
        raise ValueError("Elite proportions must be different.")
    return sorted(proportions, reverse=True)


def _group_label(proportion: float) -> str:
    """Return a concise percentage label for an elite group."""
    return f"Top {100 * proportion:g}%"


def _as_python_value(value: Any) -> Any:
    """Convert NumPy scalar values to JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_finite_number(value: Any) -> bool:
    """Return whether a value is a finite real number, excluding booleans."""
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and bool(np.isfinite(value))
    )


def _is_numeric_list(value: Any) -> bool:
    """Return whether a value is a non-empty list of finite real numbers."""
    return (
        isinstance(value, list) and bool(value) and all(_is_finite_number(item) for item in value)
    )


def _trial_user_attrs(trial: Any) -> Dict[str, Any]:
    """Return one trial's preserved Optuna user attributes."""
    additional = trial.additional if isinstance(trial.additional, dict) else {}
    user_attrs = additional.get("user_attrs", {})
    return user_attrs if isinstance(user_attrs, dict) else {}


def _complete_user_attr_names(trials: Sequence[Any], list_values: bool = False) -> List[str]:
    """Return sorted attributes with valid values on every successful trial."""
    successful_trials = [trial for trial in trials if trial.status == Status.SUCCESS]
    if not successful_trials:
        return []
    candidate_names = set(_trial_user_attrs(successful_trials[0]))
    for trial in successful_trials[1:]:
        candidate_names.intersection_update(_trial_user_attrs(trial))
    validator = _is_numeric_list if list_values else _is_finite_number
    return sorted(
        name
        for name in candidate_names
        if all(validator(_trial_user_attrs(trial).get(name)) for trial in successful_trials)
    )


def _nearest_quantile(values: pd.Series, quantile: float) -> float:
    """Return an empirical quantile without inventing intermediate categories."""
    return float(values.quantile(quantile, interpolation="nearest"))


def _feature_statistics(
    all_values: pd.Series,
    elite_values: pd.Series,
    hyperparameter: Hyperparameter,
) -> Dict[str, Any]:
    """Calculate raw and normalized range statistics for one elite group."""
    categorical_values: List[Any] = []
    if isinstance(hyperparameter, CategoricalHyperparameter):
        categorical_values = list(hyperparameter.choices)
    elif isinstance(hyperparameter, OrdinalHyperparameter):
        categorical_values = list(hyperparameter.sequence)
    elif isinstance(hyperparameter, Constant):
        categorical_values = [hyperparameter.value]

    if categorical_values:
        value_to_code = {value: index for index, value in enumerate(categorical_values)}
        code_to_value = {index: value for index, value in enumerate(categorical_values)}
        all_transformed = all_values.map(value_to_code).astype(float)
        elite_transformed = elite_values.map(value_to_code).astype(float)
        raw_codes = {
            "minimum": float(elite_transformed.min()),
            "q10": _nearest_quantile(elite_transformed, 0.10),
            "median": _nearest_quantile(elite_transformed, 0.50),
            "q90": _nearest_quantile(elite_transformed, 0.90),
            "maximum": float(elite_transformed.max()),
        }
        raw_values = {
            statistic: _as_python_value(code_to_value[int(code)])
            for statistic, code in raw_codes.items()
        }
        transformed_statistics = raw_codes
    else:
        all_numeric = all_values.astype(float)
        elite_numeric = elite_values.astype(float)
        raw_values = {
            "minimum": float(elite_numeric.min()),
            "q10": float(elite_numeric.quantile(0.10)),
            "median": float(elite_numeric.median()),
            "q90": float(elite_numeric.quantile(0.90)),
            "maximum": float(elite_numeric.max()),
        }
        if getattr(hyperparameter, "log", False):
            all_transformed = np.log(all_numeric)
            transformed_statistics = {
                statistic: float(np.log(value)) for statistic, value in raw_values.items()
            }
        else:
            all_transformed = all_numeric
            transformed_statistics = raw_values

    observed_minimum = float(all_transformed.min())
    observed_maximum = float(all_transformed.max())
    observed_range = observed_maximum - observed_minimum
    if np.isclose(observed_range, 0):
        normalized = {statistic: 0.5 for statistic in transformed_statistics}
    else:
        normalized = {
            statistic: float((value - observed_minimum) / observed_range)
            for statistic, value in transformed_statistics.items()
        }

    return {
        "n": int(len(elite_values)),
        **raw_values,
        **{f"{statistic}_normalized": value for statistic, value in normalized.items()},
    }


def _summarize_features(
    trials: List[Dict[str, Any]],
    configspace: Any,
    optimize: str,
    elite_proportions: Sequence[float],
) -> List[Dict[str, Any]]:
    """Summarize every ConfigSpace feature for each run-wide elite group."""
    objective_values = pd.Series([trial["objective"] for trial in trials], dtype=float)
    rows = []
    for hyperparameter_name in configspace.keys():
        hyperparameter = configspace[hyperparameter_name]
        active_values = pd.Series(
            [trial["configuration"].get(hyperparameter_name) for trial in trials],
            dtype=object,
        )
        active_mask = active_values.notna()
        all_values = active_values.loc[active_mask]

        for proportion in elite_proportions:
            threshold_quantile = 1 - proportion if optimize == "upper" else proportion
            threshold = float(objective_values.quantile(threshold_quantile))
            if optimize == "upper":
                elite_mask = objective_values >= threshold
            else:
                elite_mask = objective_values <= threshold
            elite_values = active_values.loc[active_mask & elite_mask]

            row = {
                "feature": hyperparameter_name,
                "proportion": float(proportion),
                "group": _group_label(proportion),
                "threshold": threshold,
                "n": int(len(elite_values)),
            }
            if not elite_values.empty and not all_values.empty:
                row.update(_feature_statistics(all_values, elite_values, hyperparameter))
            else:
                row.update(
                    {
                        statistic: None
                        for statistic in (
                            "minimum",
                            "q10",
                            "median",
                            "q90",
                            "maximum",
                            "minimum_normalized",
                            "q10_normalized",
                            "median_normalized",
                            "q90_normalized",
                            "maximum_normalized",
                        )
                    }
                )
            rows.append(row)
    return rows


def _elite_mask(objectives: pd.Series, optimize: str, proportion: float) -> pd.Series:
    """Return the run-wide elite mask for one objective and proportion."""
    threshold_quantile = 1 - proportion if optimize == "upper" else proportion
    threshold = float(objectives.quantile(threshold_quantile))
    if optimize == "upper":
        return objectives >= threshold
    return objectives <= threshold


def _numeric_bin_index(value: Any, edges: np.ndarray, log: bool) -> int:
    """Map a numerical ConfigSpace value to one of the fixed insight bins."""
    transformed = _transform_value(float(value), log)
    return int(np.clip(np.searchsorted(edges, transformed, side="right") - 1, 0, len(edges) - 2))


def _center_rug_positions(bin_indices: Sequence[int], bin_count: int) -> List[float]:
    """Center ranked trial markers in their bins and separate overlapping markers."""
    positions = [0.0] * len(bin_indices)
    bin_width = 1 / bin_count
    minimum_gap = min(0.025, 0.2 * bin_width)
    for bin_index in set(bin_indices):
        members = [index for index, value in enumerate(bin_indices) if value == bin_index]
        center = (bin_index + 0.5) * bin_width
        start = center - minimum_gap * (len(members) - 1) / 2
        for offset, member in enumerate(members):
            positions[member] = start + offset * minimum_gap
    return positions


def _compact_insight_bin_label(value: str) -> str:
    """Format a regular numeric interval and omit redundant integer decimals."""
    label = bin_label(value, multiline=False)
    bounds = label.split(NONBREAKING_RANGE_SEPARATOR)
    if len(bounds) != 2:
        return label
    compact_bounds = [bound[:-2] if bound.endswith(".0") else bound for bound in bounds]
    return NONBREAKING_RANGE_SEPARATOR.join(compact_bounds)


def _scaled_insight_bin_label(left: float, right: float, exponent: int) -> str:
    """Format a small numeric interval using one shared engineering exponent."""
    scale = 10**exponent
    lower = left / scale
    upper = right / scale
    width = upper - lower
    decimals = 2
    if width > 0 and math.isfinite(width):
        decimals = max(decimals, min(6, math.ceil(round(-math.log10(width), 6))))

    def format_bound(bound: float) -> str:
        text = f"{bound:.{decimals}f}"
        if decimals and text.endswith(f".{('0' * decimals)}"):
            return text[: -(decimals + 1)]
        return text

    return NONBREAKING_RANGE_SEPARATOR.join((format_bound(lower), format_bound(upper)))


def _insight_numeric_axis(
    name: str,
    display_edges: np.ndarray,
) -> Tuple[List[str], List[str], str]:
    """Build stable numeric bin keys plus compact tick labels and an axis title."""
    edge_pairs = list(zip(display_edges[:-1], display_edges[1:]))
    labels = [f"{_format_value(left)}–{_format_value(right)}" for left, right in edge_pairs]
    interval_values = [f"({float(left)!r}, {float(right)!r}]" for left, right in edge_pairs]
    scale_exponent = bin_scale_exponent(interval_values)
    if scale_exponent is None:
        tick_labels = [_compact_insight_bin_label(value) for value in interval_values]
        return labels, tick_labels, name

    tick_labels = [
        _scaled_insight_bin_label(float(left), float(right), scale_exponent)
        for left, right in edge_pairs
    ]
    axis_title = f"{name} (×10{superscript_exponent(scale_exponent)})"
    return labels, tick_labels, axis_title


def _insight_integer_bins(
    hyperparameter: IntegerHyperparameter,
) -> Tuple[List[Tuple[int, int]], List[str], List[str]]:
    """Split an integer domain into at most ten inclusive, discrete ranges."""
    lower = int(hyperparameter.lower)
    upper = int(hyperparameter.upper)
    value_count = upper - lower + 1
    bin_count = min(INSIGHT_BIN_COUNT, value_count)
    base_size, larger_bins = divmod(value_count, bin_count)
    ranges = []
    start = lower
    for index in range(bin_count):
        size = base_size + (1 if index < larger_bins else 0)
        end = start + size - 1
        ranges.append((start, end))
        start = end + 1

    labels = [str(start) if start == end else f"{start}–{end}" for start, end in ranges]
    tick_labels = [
        str(start) if start == end else NONBREAKING_RANGE_SEPARATOR.join((str(start), str(end)))
        for start, end in ranges
    ]
    return ranges, labels, tick_labels


def _integer_bin_index(value: Any, ranges: Sequence[Tuple[int, int]]) -> int:
    """Map an integer value to its inclusive discrete insight range."""
    integer_value = int(value)
    for index, (lower, upper) in enumerate(ranges):
        if lower <= integer_value <= upper:
            return index
    return 0 if integer_value < ranges[0][0] else len(ranges) - 1


def _parameter_insights(
    trials: Sequence[Dict[str, Any]],
    configspace: Any,
    optimize: str,
    elite_proportions: Sequence[float],
) -> List[Dict[str, Any]]:
    """Aggregate performance and elite enrichment over shared parameter ranges."""
    objectives = pd.Series([trial["objective"] for trial in trials], dtype=float)
    elite_masks = {
        float(proportion): _elite_mask(objectives, optimize, float(proportion))
        for proportion in elite_proportions
    }
    insights = []

    for index, name in enumerate(configspace.keys()):
        hyperparameter = configspace[name]
        parameter_type = _parameter_type(hyperparameter)
        active_rows = [
            (trial["configuration"].get(name), trial["objective"], row_index)
            for row_index, trial in enumerate(trials)
            if trial["configuration"].get(name) is not None
        ]
        row_bin_indices: Dict[int, int] = {}

        if isinstance(hyperparameter, IntegerHyperparameter):
            integer_ranges, labels, tick_labels = _insight_integer_bins(hyperparameter)
            axis_title = name
            grouped_rows = [[] for _ in integer_ranges]
            for row in active_rows:
                bin_index = _integer_bin_index(row[0], integer_ranges)
                grouped_rows[bin_index].append(row)
                row_bin_indices[row[2]] = bin_index
        elif isinstance(hyperparameter, FloatHyperparameter):
            log = bool(getattr(hyperparameter, "log", False))
            transformed_edges = np.linspace(
                _transform_value(float(hyperparameter.lower), log),
                _transform_value(float(hyperparameter.upper), log),
                INSIGHT_BIN_COUNT + 1,
            )
            display_edges = np.exp(transformed_edges) if log else transformed_edges
            labels, tick_labels, axis_title = _insight_numeric_axis(
                name,
                display_edges,
            )
            grouped_rows: List[List[Tuple[Any, float, int]]] = [
                [] for _ in range(INSIGHT_BIN_COUNT)
            ]
            for row in active_rows:
                bin_index = _numeric_bin_index(row[0], transformed_edges, log)
                grouped_rows[bin_index].append(row)
                row_bin_indices[row[2]] = bin_index
        else:
            choices = _categorical_choices(hyperparameter)
            labels = [_format_value(choice) for choice in choices]
            tick_labels = labels
            axis_title = name
            grouped_rows = [[] for _ in choices]
            for row in active_rows:
                bin_index = next(
                    (
                        choice_index
                        for choice_index, choice in enumerate(choices)
                        if row[0] == choice
                    ),
                    None,
                )
                if bin_index is not None:
                    grouped_rows[bin_index].append(row)
                    row_bin_indices[row[2]] = bin_index

        active_count = len(active_rows)
        medians = []
        q90_values = []
        trial_counts = []
        for rows in grouped_rows:
            values = pd.Series([row[1] for row in rows], dtype=float)
            trial_count = len(rows)
            medians.append(None if values.empty else float(values.median()))
            q90_values.append(None if values.empty else float(values.quantile(0.90)))
            trial_counts.append(trial_count)

        elite_groups = []
        for color, proportion in zip(ELITE_MARKER_COLORS, elite_proportions):
            proportion = float(proportion)
            elite_mask = elite_masks[proportion]
            active_elite_count = sum(bool(elite_mask.iloc[row[2]]) for row in active_rows)
            enrichments = []
            elite_counts = []
            for rows, trial_count in zip(grouped_rows, trial_counts):
                elite_count = sum(bool(elite_mask.iloc[row[2]]) for row in rows)
                overall_share = trial_count / active_count if active_count else 0.0
                elite_share = elite_count / active_elite_count if active_elite_count else 0.0
                enrichment = elite_share / overall_share if overall_share else 0.0
                enrichments.append(float(enrichment))
                elite_counts.append(elite_count)
            elite_groups.append(
                {
                    "proportion": proportion,
                    "label": _group_label(proportion),
                    "color": color,
                    "enrichment": enrichments,
                    "elite_counts": elite_counts,
                }
            )

        ranked_active_rows = sorted(
            active_rows,
            key=lambda row: (
                row[1] if optimize == "lower" else -row[1],
                int(trials[row[2]].get("trial_number", row[2])),
            ),
        )[:5]
        top_trials = [
            {
                "rank": rank,
                "bin_index": row_bin_indices[row[2]],
                "trial_number": trials[row[2]].get("trial_number", row[2]),
                "objective": row[1],
                "value": _as_python_value(row[0]),
            }
            for rank, row in enumerate(ranked_active_rows, start=1)
            if row[2] in row_bin_indices
        ]

        insights.append(
            {
                "index": index,
                "name": name,
                "parameter_type": parameter_type,
                "labels": labels,
                "tick_labels": tick_labels,
                "axis_title": axis_title,
                "median": medians,
                "q90": q90_values,
                "trial_counts": trial_counts,
                "elite_groups": elite_groups,
                "top_trials": top_trials,
            }
        )
    return insights


def _format_value(value: Any) -> str:
    """Format a raw value compactly without changing its type."""
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def _transform_value(value: float, log: bool) -> float:
    """Transform a value into the parameter's actual sampling space."""
    value = float(value)
    if log:
        if value <= 0:
            raise ValueError("Log-scaled planning values must be positive.")
        return float(np.log(value))
    return value


def _normalize_value(value: float, parameter: Dict[str, Any]) -> float:
    """Normalize an original value relative to the current ConfigSpace bounds."""
    transformed_lower = _transform_value(parameter["current_min"], parameter["log"])
    transformed_upper = _transform_value(parameter["current_max"], parameter["log"])
    transformed_value = _transform_value(value, parameter["log"])
    width = transformed_upper - transformed_lower
    if np.isclose(width, 0):
        return 0.5
    return float((transformed_value - transformed_lower) / width)


def _denormalize_value(position: float, parameter: Dict[str, Any]) -> float:
    """Map a planner coordinate back into the original parameter space."""
    transformed_lower = _transform_value(parameter["current_min"], parameter["log"])
    transformed_upper = _transform_value(parameter["current_max"], parameter["log"])
    transformed_value = transformed_lower + float(position) * (
        transformed_upper - transformed_lower
    )
    if parameter["log"]:
        return float(np.exp(transformed_value))
    return float(transformed_value)


def _snap_value(value: float, parameter: Dict[str, Any]) -> Any:
    """Snap a numerical value to its ConfigSpace integer/step lattice without clipping it."""
    step = parameter.get("step")
    if step is not None and float(step) > 0:
        lower = float(parameter["current_min"])
        value = lower + round((float(value) - lower) / float(step)) * float(step)

    if parameter["parameter_type"] == "integer":
        return int(round(value))
    return float(f"{float(value):.12g}")


def _coerce_selection(
    minimum: float,
    maximum: float,
    parameter: Dict[str, Any],
) -> Dict[str, Any]:
    """Transform, snap and order a numerical selection while permitting extensions."""
    minimum = _snap_value(float(minimum), parameter)
    maximum = _snap_value(float(maximum), parameter)
    # This also performs the mathematical domain check required for log transforms.
    _transform_value(minimum, parameter["log"])
    _transform_value(maximum, parameter["log"])
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    return {"min": minimum, "max": maximum}


def _parameter_type(hyperparameter: Hyperparameter) -> str:
    """Return the planner row type from ConfigSpace metadata, never from the name."""
    if isinstance(hyperparameter, IntegerHyperparameter):
        return "integer"
    if isinstance(hyperparameter, FloatHyperparameter):
        return "float"
    if isinstance(hyperparameter, OrdinalHyperparameter):
        return "ordinal"
    if isinstance(hyperparameter, Constant):
        return "constant"
    return "categorical"


def _categorical_choices(hyperparameter: Hyperparameter) -> List[Any]:
    """Return selectable values for categorical-like ConfigSpace parameters."""
    if isinstance(hyperparameter, CategoricalHyperparameter):
        return [_as_python_value(value) for value in hyperparameter.choices]
    if isinstance(hyperparameter, OrdinalHyperparameter):
        return [_as_python_value(value) for value in hyperparameter.sequence]
    if isinstance(hyperparameter, Constant):
        return [_as_python_value(hyperparameter.value)]
    return []


def _build_planning_metadata(
    configspace: Any,
    features: Sequence[str],
    elite_proportions: Sequence[float],
    summary: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build immutable row metadata separately from the mutable planned selection."""
    summary_lookup = {(row["feature"], row["proportion"]): row for row in summary}
    metadata = []
    for index, feature in enumerate(features):
        hyperparameter = configspace[feature]
        parameter_type = _parameter_type(hyperparameter)
        parameter: Dict[str, Any] = {
            "index": index,
            "name": feature,
            "parameter_type": parameter_type,
        }
        if isinstance(hyperparameter, NUMERIC_PARAMETER_TYPES):
            step = getattr(hyperparameter, "step", None)
            if step is None and hyperparameter.meta is not None:
                step = hyperparameter.meta.get("step")
            parameter.update(
                {
                    "current_min": _as_python_value(hyperparameter.lower),
                    "current_max": _as_python_value(hyperparameter.upper),
                    "log": bool(getattr(hyperparameter, "log", False)),
                    "step": _as_python_value(step),
                }
            )
            if parameter_type == "integer" and parameter["step"] is None:
                parameter["step"] = 1

            elite_positions = []
            for color, proportion in zip(ELITE_MARKER_COLORS, elite_proportions):
                row = summary_lookup[(feature, proportion)]
                statistics = {
                    statistic: row.get(statistic)
                    for statistic in ("minimum", "q10", "median", "q90", "maximum")
                }
                elite_positions.append(
                    {
                        "group": _group_label(proportion),
                        "proportion": proportion,
                        "position": (
                            _normalize_value(float(statistics["median"]), parameter)
                            if statistics["median"] is not None
                            else None
                        ),
                        "value": statistics["median"],
                        **statistics,
                        **{
                            f"{statistic}_position": (
                                _normalize_value(float(value), parameter)
                                if value is not None
                                else None
                            )
                            for statistic, value in statistics.items()
                        },
                        "n": row["n"],
                        "color": color,
                    }
                )
            parameter["elite_positions"] = elite_positions
        else:
            choices = _categorical_choices(hyperparameter)
            parameter.update(
                {
                    "choices": choices,
                    "elite_values": [
                        {
                            "group": _group_label(proportion),
                            "value": summary_lookup[(feature, proportion)].get("median"),
                            "n": summary_lookup[(feature, proportion)]["n"],
                            "color": color,
                        }
                        for color, proportion in zip(ELITE_MARKER_COLORS, elite_proportions)
                    ],
                }
            )
        metadata.append(parameter)
    return metadata


def _initial_selection(metadata: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Select the complete current search space for every parameter."""
    selection = {}
    for parameter in metadata:
        if parameter["parameter_type"] in ("integer", "float"):
            selection[parameter["name"]] = {
                "min": parameter["current_min"],
                "max": parameter["current_max"],
            }
        else:
            selection[parameter["name"]] = {"choices": list(parameter["choices"])}
    return selection


def _selection_metrics(
    parameter: Dict[str, Any],
    selection: Dict[str, Any],
) -> Dict[str, Any]:
    """Describe a numerical planned range in normalized planner coordinates."""
    minimum = _normalize_value(selection["min"], parameter)
    maximum = _normalize_value(selection["max"], parameter)
    span_ratio = maximum - minimum
    left_extension = max(0.0, -minimum)
    right_extension = max(0.0, maximum - 1.0)
    if span_ratio > 1.0 + SPAN_COMPARISON_TOLERANCE:
        comparison = "wider"
    elif span_ratio < 1.0 - SPAN_COMPARISON_TOLERANCE:
        comparison = "narrower"
    else:
        comparison = "same"

    return {
        "min_normalized": minimum,
        "max_normalized": maximum,
        "span_ratio": span_ratio,
        "left_extension": left_extension,
        "right_extension": right_extension,
        "extends": left_extension > 1e-9 or right_extension > 1e-9,
        "comparison": comparison,
    }


def _slider_domain(metrics: Dict[str, Any]) -> Tuple[float, float]:
    """Keep current space visible while expanding to contain direct numerical edits."""
    minimum = min(
        -DEFAULT_SLIDER_EXTENSION,
        float(metrics["min_normalized"]) - DEFAULT_SLIDER_EXTENSION,
    )
    maximum = max(
        1.0 + DEFAULT_SLIDER_EXTENSION,
        float(metrics["max_normalized"]) + DEFAULT_SLIDER_EXTENSION,
    )
    return minimum, maximum


def _position_percent(value: float, domain: Tuple[float, float]) -> float:
    """Return a planner coordinate's percentage position in a slider domain."""
    return 100 * (float(value) - domain[0]) / (domain[1] - domain[0])


def _python_space(
    metadata: Sequence[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the selected space with stable order and exact integer values."""
    result = {}
    for parameter in metadata:
        selected = selection[parameter["name"]]
        if parameter["parameter_type"] in ("integer", "float"):
            result[parameter["name"]] = [selected["min"], selected["max"]]
        else:
            result[parameter["name"]] = list(selected["choices"])
    return result


def _python_space_preview(
    metadata: Sequence[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
) -> str:
    """Format the complete selected search space as copyable Python code."""
    return pprint.pformat(_python_space(metadata, selection), sort_dicts=False, width=56)


def _insight_button(parameter: Dict[str, Any]) -> Component:
    """Return the button opening the combined parameter-insight plot."""
    return dbc.Button(
        html.I(className="fas fa-chart-bar"),
        id={"type": INSIGHT_BUTTON_TYPE, "index": parameter["index"]},
        n_clicks=0,
        color="link",
        size="sm",
        title=f"Show performance and elite enrichment for {parameter['name']}",
        className="planner-insight-button",
    )


def _numeric_row(parameter: Dict[str, Any], selection: Dict[str, Any]) -> Component:
    """Render one compact numerical planning row."""
    index = parameter["index"]
    metrics = _selection_metrics(parameter, selection)
    slider_minimum, slider_maximum = _slider_domain(metrics)
    step = parameter.get("step")
    input_step = step if step is not None else "any"
    return html.Div(
        [
            html.Div(
                [
                    html.Span(parameter["name"], className="planner-parameter-name"),
                    _insight_button(parameter),
                ],
                className="planner-parameter-heading",
            ),
            html.Div(
                [
                    html.Div(
                        html.Span(
                            className="planner-current-reference-segment",
                            style=_reference_style((slider_minimum, slider_maximum)),
                        ),
                        id={"type": "hpo-planner-reference", "index": index},
                        className="planner-current-reference",
                    ),
                    dcc.RangeSlider(
                        id={"type": "hpo-planner-range", "index": index},
                        min=slider_minimum,
                        max=slider_maximum,
                        step=0.001,
                        value=[metrics["min_normalized"], metrics["max_normalized"]],
                        marks={
                            0: _format_value(parameter["current_min"]),
                            1: _format_value(parameter["current_max"]),
                        },
                        allowCross=False,
                        updatemode="mouseup",
                        tooltip={
                            "placement": "bottom",
                            "always_visible": False,
                            "transform": f"hpoPlanning.parameter{index}",
                        },
                        className="planner-range-slider",
                    ),
                    html.Div(
                        id={"type": "hpo-planner-extensions", "index": index},
                        className="planner-extension-layer",
                        children=_extension_components(metrics, (slider_minimum, slider_maximum)),
                    ),
                    html.Div(
                        id={"type": "hpo-planner-elites", "index": index},
                        className="planner-elite-layer",
                        children=_elite_distribution_components(
                            parameter, (slider_minimum, slider_maximum)
                        ),
                    ),
                    html.Div(
                        id={"type": "hpo-planner-unselected", "index": index},
                        className="planner-unselected-layer",
                        children=_unselected_components(metrics, (slider_minimum, slider_maximum)),
                    ),
                ],
                className="planner-range-control",
                **{
                    "data-current-min": str(parameter["current_min"]),
                    "data-current-max": str(parameter["current_max"]),
                    "data-log": str(parameter["log"]).lower(),
                    "data-step": "" if step is None else str(step),
                    "data-parameter-type": parameter["parameter_type"],
                },
            ),
            dbc.Input(
                id={"type": "hpo-planner-min", "index": index},
                type="number",
                value=selection["min"],
                step=input_step,
                debounce=True,
                size="sm",
                className="planner-bound-input",
            ),
            dbc.Input(
                id={"type": "hpo-planner-max", "index": index},
                type="number",
                value=selection["max"],
                step=input_step,
                debounce=True,
                size="sm",
                className="planner-bound-input",
            ),
        ],
        className="planner-parameter-row planner-parameter-row--numeric",
    )


def _categorical_row(parameter: Dict[str, Any], selection: Dict[str, Any]) -> Component:
    """Render a categorical, ordinal, or constant parameter as a compact multiselect."""
    index = parameter["index"]
    elite_badges = []
    for elite in parameter["elite_values"]:
        label = "no active trials" if elite["value"] is None else _format_value(elite["value"])
        elite_badges.append(
            html.Span(
                f"{elite['group']}: {label}",
                className="planner-category-elite",
                style={"borderColor": elite["color"]},
            )
        )
    return html.Div(
        [
            html.Div(
                [
                    html.Span(parameter["name"], className="planner-parameter-name"),
                    _insight_button(parameter),
                ],
                className="planner-parameter-heading",
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id={"type": "hpo-planner-categories", "index": index},
                        options=get_select_options(parameter["choices"]),
                        value=selection["choices"],
                        multi=True,
                        clearable=True,
                        className="planner-category-select",
                    ),
                    html.Div(elite_badges, className="planner-category-elites"),
                ],
                className="planner-category-control",
            ),
            html.Div(),
            html.Div(),
        ],
        className="planner-parameter-row planner-parameter-row--categorical",
    )


def _planner_rows(
    metadata: Sequence[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
) -> List[Component]:
    """Render all planner rows while keeping numerical and categorical rows distinct."""
    rows = []
    for parameter in metadata:
        selected = selection[parameter["name"]]
        if parameter["parameter_type"] in ("integer", "float"):
            rows.append(_numeric_row(parameter, selected))
        else:
            rows.append(_categorical_row(parameter, selected))
    return rows


def _planner_legend() -> Component:
    """Return the compact legend for current space, elite ranges, and planned range."""
    elite_items = [
        html.Span(
            [
                html.I(
                    [
                        html.I(
                            className="planner-legend-elite-whisker",
                            style={"backgroundColor": color},
                        ),
                        html.I(
                            className="planner-legend-elite-band",
                            style={"backgroundColor": color},
                        ),
                        html.I(
                            className="planner-legend-elite-median",
                            style={"backgroundColor": color},
                        ),
                    ],
                    className="planner-legend-elite-glyph",
                ),
                f"Top {percent}% (min–max · Q10–Q90 · median)",
            ]
        )
        for color, percent in zip(ELITE_MARKER_COLORS, (10, 5, 1))
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        [html.I(className="planner-legend-line"), "Current search space"],
                    ),
                    html.Span(
                        [html.I(className="planner-legend-selection"), "Selected range"],
                    ),
                    html.Span(
                        [
                            html.I(className="planner-legend-unselected"),
                            "Outside selected range",
                        ],
                    ),
                    html.Span(
                        [html.I(className="planner-legend-extension"), "Beyond current space"],
                    ),
                ],
                className="planner-legend-row",
            ),
            html.Div(
                elite_items,
                className="planner-legend-row planner-legend-elites",
            ),
        ],
        className="planner-legend",
    )


def _trial_violations(
    trial: Dict[str, Any],
    metadata: Sequence[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Return selected-space constraints violated by one trial configuration."""
    configuration = trial["configuration"]
    violations = []
    for parameter in metadata:
        name = parameter["name"]
        # Inactive conditional parameters do not constrain the configuration.
        if name not in configuration or configuration[name] is None:
            continue
        value = configuration[name]
        selected = selection[name]
        if parameter["parameter_type"] in ("integer", "float"):
            try:
                included = float(selected["min"]) <= float(value) <= float(selected["max"])
            except (TypeError, ValueError):
                included = False
        else:
            included = value in selected["choices"]
        if not included:
            violations.append(name)
    return violations


def _coverage_state(
    trials: Sequence[Dict[str, Any]],
    metadata: Sequence[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach live inclusion and violation information to ranked top trials."""
    state = []
    for trial in trials:
        violations = _trial_violations(trial, metadata, selection)
        state.append({**trial, "included": not violations, "violations": violations})
    return state


def _coverage_badge(coverage: Sequence[Dict[str, Any]]) -> str:
    """Return the compact top-trial coverage summary."""
    included = sum(trial["included"] for trial in coverage)
    return f"Included top trials: {included} / {len(coverage)}"


def _spread_control_state(spread_attribute: Any) -> Tuple[bool, bool]:
    """Enable and initially select spread display when an attribute is selected."""
    selected = bool(spread_attribute)
    return not selected, selected


def _spread_mode_options(disabled: bool) -> List[Dict[str, Any]]:
    """Return SD and range options with a shared disabled state."""
    return [
        {"label": "SD", "value": "sd", "disabled": disabled},
        {"label": "Range", "value": "range", "disabled": disabled},
    ]


def _coverage_trial_count(
    total_trials: int,
    top_trial_count: Any = DEFAULT_TOP_TRIAL_COUNT,
) -> int:
    """Return the requested number of trials, bounded by availability and 100."""
    try:
        requested_count = min(TOP_TRIAL_LIMIT, max(1, int(float(top_trial_count))))
    except (OverflowError, TypeError, ValueError):
        requested_count = DEFAULT_TOP_TRIAL_COUNT
    return min(max(0, int(total_trials)), requested_count)


def _visible_coverage_trials(
    coverage_data: Dict[str, Any],
    top_trial_count: Any,
) -> List[Dict[str, Any]]:
    """Slice the pre-ranked coverage candidates to the selected trial count."""
    trial_count = _coverage_trial_count(coverage_data["total_trials"], top_trial_count)
    return list(coverage_data["trials"][:trial_count])


def _coverage_title(top_trial_count: Any) -> str:
    """Return the live coverage card title."""
    trial_count = _coverage_trial_count(TOP_TRIAL_LIMIT, top_trial_count)
    return f"Top-{trial_count} trial coverage (max. {TOP_TRIAL_LIMIT})"


def _coverage_subtitle(top_trial_count: Any) -> str:
    """Return the live coverage card explanation."""
    trial_count = _coverage_trial_count(TOP_TRIAL_LIMIT, top_trial_count)
    noun = "trial" if trial_count == 1 else "trials"
    return (
        f"Best {trial_count} {noun} ranked by the main objective; green points are "
        "still inside the current selection."
    )


def _coverage_figure(
    trials: Sequence[Dict[str, Any]],
    metadata: Sequence[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
    objective_name: str,
    optimize: str,
    spread_attribute: Any = None,
    show_spread: bool = False,
    spread_mode: str = "sd",
    show_scatter: bool = False,
) -> go.Figure:
    """Plot direction-aware trial rank against objective and current range coverage."""
    coverage = _coverage_state(trials, metadata, selection)
    figure = go.Figure()
    spread_enabled = bool(
        show_spread
        and spread_attribute
        and all(spread_attribute in trial.get("spread_values", {}) for trial in coverage)
    )
    range_traces = []
    scatter_traces = []
    for included, name, color, opacity in (
        (True, "Included in current range", INCLUDED_TRIAL_COLOR, 0.95),
        (False, "Excluded from current range", EXCLUDED_TRIAL_COLOR, 0.62),
    ):
        matching = [trial for trial in coverage if trial["included"] is included]
        customdata = [
            [
                trial["trial_number"],
                len(trial["violations"]),
                ", ".join(trial["violations"]) or "None",
            ]
            for trial in matching
        ]
        error_y = None
        spread_hover = ""
        if spread_enabled:
            spread_values = [trial["spread_values"][spread_attribute] for trial in matching]
            if spread_mode == "range":
                spread_minima = [float(min(values)) for values in spread_values]
                spread_maxima = [float(max(values)) for values in spread_values]
                for row, values, minimum, maximum in zip(
                    customdata, spread_values, spread_minima, spread_maxima
                ):
                    row.extend(
                        [
                            minimum,
                            maximum,
                            "[" + ", ".join(_format_value(value) for value in values) + "]",
                        ]
                    )
                if matching:
                    range_centers = [
                        (minimum + maximum) / 2
                        for minimum, maximum in zip(spread_minima, spread_maxima)
                    ]
                    range_half_widths = [
                        (maximum - minimum) / 2
                        for minimum, maximum in zip(spread_minima, spread_maxima)
                    ]
                    range_traces.append(
                        go.Scatter(
                            x=[trial["rank"] for trial in matching],
                            y=range_centers,
                            mode="markers",
                            showlegend=False,
                            marker={"size": 1, "opacity": 0},
                            error_y={
                                "type": "data",
                                "array": range_half_widths,
                                "visible": True,
                                "color": color,
                                "thickness": 1.2,
                                "width": 3,
                            },
                            hoverinfo="skip",
                        )
                    )
                spread_hover = (
                    f"{spread_attribute} range: %{{customdata[3]:.6g}}–"
                    "%{customdata[4]:.6g}<br>Values: %{customdata[5]}<br>"
                )
            else:
                spread_deviations = [float(np.std(values, ddof=0)) for values in spread_values]
                for row, values, deviation in zip(customdata, spread_values, spread_deviations):
                    row.extend(
                        [
                            deviation,
                            "[" + ", ".join(_format_value(value) for value in values) + "]",
                        ]
                    )
                error_y = {
                    "type": "data",
                    "array": spread_deviations,
                    "visible": True,
                    "color": color,
                    "thickness": 1.2,
                    "width": 3,
                }
                spread_hover = (
                    f"{spread_attribute} SD: %{{customdata[3]:.6g}}<br>"
                    "Values: %{customdata[4]}<br>"
                )

            if show_scatter and matching:
                scatter_x = []
                scatter_y = []
                scatter_customdata = []
                for trial, values in zip(matching, spread_values):
                    offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0.0]
                    for offset, value in zip(offsets, values):
                        scatter_x.append(float(trial["rank"]) + float(offset))
                        scatter_y.append(float(value))
                        scatter_customdata.append([trial["trial_number"], trial["rank"]])
                scatter_traces.append(
                    go.Scatter(
                        x=scatter_x,
                        y=scatter_y,
                        mode="markers",
                        name=f"{spread_attribute} values",
                        showlegend=False,
                        marker={"color": color, "size": 5, "opacity": 0.55},
                        customdata=scatter_customdata,
                        hovertemplate=(
                            "<b>Trial %{customdata[0]}</b><br>"
                            "Rank: %{customdata[1]}<br>"
                            f"{spread_attribute}: %{{y:.6g}}<extra></extra>"
                        ),
                    )
                )
        figure.add_trace(
            go.Scatter(
                x=[trial["rank"] for trial in matching],
                y=[trial["objective"] for trial in matching],
                mode="markers",
                name=name,
                showlegend=not included,
                marker={"color": color, "size": 8, "opacity": opacity},
                error_y=error_y,
                customdata=customdata,
                hovertemplate=(
                    "<b>Trial %{customdata[0]}</b><br>"
                    "Rank: %{x}<br>"
                    f"{objective_name}: %{{y:.6g}}<br>"
                    f"{spread_hover}"
                    f"Included: {'Yes' if included else 'No'}<br>"
                    "Violated parameters: %{customdata[1]}<br>"
                    "%{customdata[2]}<extra></extra>"
                ),
            )
        )

    for range_trace in range_traces:
        figure.add_trace(range_trace)
    for scatter_trace in scatter_traces:
        figure.add_trace(scatter_trace)

    trial_count = len(coverage)
    tick_values = [1]
    tick_values.extend(range(5, trial_count + 1, 5))
    if trial_count > 1 and trial_count not in tick_values:
        tick_values.append(trial_count)
    tick_values = sorted(set(value for value in tick_values if value <= trial_count))
    direction = "higher is better" if optimize == "upper" else "lower is better"
    figure.update_xaxes(
        title_text="Trial rank",
        tickmode="array",
        tickvals=tick_values,
        range=[0.3, max(1.7, trial_count + 0.7)],
    )
    figure.update_yaxes(title_text=f"{objective_name} ({direction})")

    positive_objectives = [trial["objective"] for trial in coverage if trial["objective"] > 0]
    if (
        len(positive_objectives) == trial_count
        and trial_count > 1
        and max(positive_objectives) / min(positive_objectives) >= 100
    ):
        figure.update_yaxes(type="log")

    style_analysis_panel(
        figure,
        height=300,
        bottom=54,
        top=36,
        left=70,
        right=22,
        show_legend=any(not trial["included"] for trial in coverage),
    )
    figure.update_layout(
        legend={
            "orientation": "h",
            "x": 1,
            "xanchor": "right",
            "y": 1.03,
            "yanchor": "bottom",
            "font": {"size": 11},
        },
        hovermode="closest",
        uirevision="hpo-planning-coverage",
    )
    return figure


def _add_insight_top_trials_rug(figure: go.Figure, insight: Dict[str, Any]) -> None:
    """Mark the five best active trials below their corresponding panel bins."""
    top_trials = insight.get("top_trials") or []
    labels = insight["labels"]
    if not top_trials or not labels:
        return
    positions = _center_rug_positions(
        [trial["bin_index"] for trial in top_trials],
        len(labels),
    )
    for x_value, trial in zip(positions, top_trials):
        figure.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=x_value,
            x1=x_value,
            y0=INSIGHT_TOP_TRIAL_RUG_Y,
            y1=INSIGHT_TOP_TRIAL_STEM_Y,
            line={"color": "#D9DEE7", "width": 1},
        )
        annotation = {
            "x": x_value,
            "xref": "paper",
            "y": INSIGHT_TOP_TRIAL_RUG_Y,
            "yref": "paper",
            "xanchor": "center",
            "yanchor": "middle",
            "showarrow": False,
            "captureevents": False,
        }
        figure.add_annotation(
            **annotation,
            text="◆",
            font={"color": INSIGHT_TOP_TRIAL_COLOR, "size": 21},
            name=f"top_trial_marker_{trial['rank']}",
        )
        figure.add_annotation(
            **annotation,
            text=str(trial["rank"]),
            font={"color": "#FFFFFF", "size": 9},
            name=f"top_trial_rank_{trial['rank']}",
        )
    figure.add_annotation(
        text="Top 5 trials",
        xref="paper",
        yref="paper",
        x=0,
        xshift=-5,
        xanchor="right",
        y=INSIGHT_TOP_TRIAL_RUG_Y,
        yanchor="middle",
        showarrow=False,
        font={"size": 9, "color": INSIGHT_TOP_TRIAL_COLOR},
    )
    figure.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        x1=1,
        y0=INSIGHT_TOP_TRIAL_RUG_Y,
        y1=INSIGHT_TOP_TRIAL_RUG_Y,
        line={"color": "#D9DEE7", "width": 1},
    )


def _parameter_insight_figure(
    insight: Dict[str, Any],
    objective_name: str,
    elite_proportion: Any,
) -> go.Figure:
    """Combine elite enrichment and objective summaries on two vertical axes."""
    labels = insight["labels"]
    elite_group = next(
        (
            group
            for group in insight["elite_groups"]
            if elite_proportion is not None and np.isclose(group["proportion"], elite_proportion)
        ),
        insight["elite_groups"][0],
    )
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=labels,
            y=elite_group["enrichment"],
            name=f"Elite enrichment ({elite_group['label']})",
            marker={
                "color": elite_group["color"],
                "opacity": INSIGHT_ELITE_BAR_OPACITY,
            },
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=insight["q90"],
            name="q90",
            mode="lines+markers",
            yaxis="y2",
            line={"color": INSIGHT_Q90_COLOR, "width": 2},
            marker={"size": 6},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=insight["median"],
            name="median",
            mode="lines+markers",
            yaxis="y2",
            line={"color": INSIGHT_MEDIAN_COLOR, "width": 2},
            marker={"size": 6},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=[1.0] * len(labels),
            name="1.0× baseline",
            mode="lines",
            line={"color": "#DC3545", "width": 1.5, "dash": "dash"},
            hoverinfo="skip",
        )
    )

    style_analysis_panel(
        figure,
        height=520,
        bottom=110,
        top=62,
        left=72,
        right=82,
        show_legend=True,
    )
    figure.update_layout(
        barmode="overlay",
        hovermode=False,
        legend={
            "orientation": "h",
            "x": 1,
            "xanchor": "right",
            "y": 1.04,
            "yanchor": "bottom",
            "font": {"size": 11},
        },
        yaxis={"title": "Elite enrichment", "rangemode": "tozero"},
        yaxis2={
            "title": objective_name,
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        uirevision=(f"hpo-planning-insight-{insight['name']}-{elite_group['proportion']}"),
    )
    figure.update_xaxes(
        title_text=None,
        type="category",
        tickmode="array",
        tickvals=labels,
        ticktext=insight.get("tick_labels", labels),
        tickangle=0,
        categoryorder="array",
        categoryarray=labels,
    )
    _add_insight_top_trials_rug(figure, insight)
    figure.add_annotation(
        name="insight_x_title",
        text=insight.get("axis_title", insight["name"]),
        xref="paper",
        yref="paper",
        x=0.5,
        y=INSIGHT_X_TITLE_Y,
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font={"size": 13},
    )
    return figure


def _planner_layout(
    metadata: List[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
    coverage_data: Dict[str, Any],
    insights: List[Dict[str, Any]],
) -> Component:
    """Return the planner, sidebar, and live top-trial coverage card."""
    visible_trials = _visible_coverage_trials(coverage_data, DEFAULT_TOP_TRIAL_COUNT)
    coverage = _coverage_state(visible_trials, metadata, selection)
    elite_groups = insights[0]["elite_groups"] if insights else []
    elite_options = [
        {"label": group["label"], "value": group["proportion"]} for group in elite_groups
    ]
    default_elite_proportion = elite_groups[0]["proportion"] if elite_groups else None
    return html.Div(
        [
            dcc.Store(id=METADATA_STORE_ID, data=metadata),
            dcc.Store(id=SELECTION_STORE_ID, data=selection),
            dcc.Store(id=TRIALS_STORE_ID, data=coverage_data),
            dcc.Store(id=INSIGHTS_STORE_ID, data=insights),
            dcc.Store(id=INSIGHT_SELECTION_STORE_ID, data=None),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "Parameter range planner",
                                        className="planner-card-title",
                                    ),
                                ]
                            ),
                            help_button(
                                "The 0–1 reference is the current ConfigSpace range. "
                                "Selections may extend beyond it; warnings are informational."
                            ),
                        ],
                        className="planner-card-heading",
                    ),
                    _planner_legend(),
                    html.Div(
                        [
                            html.Div("Parameter"),
                            html.Div("Position in current search space"),
                            html.Div("Min (original)"),
                            html.Div("Max (original)"),
                        ],
                        className="planner-table-header",
                    ),
                    html.Div(
                        _planner_rows(metadata, selection),
                        id=ROWS_ID,
                        className="planner-rows",
                    ),
                ],
                className="analysis-card planner-main-card",
            ),
            html.Div(
                [
                    html.Div("Selected search space", className="planner-card-title"),
                    html.Div(
                        "Ranges from all parameters are collected below.",
                        className="planner-card-subtitle",
                    ),
                    html.Div(
                        _sidebar_summary(metadata, selection),
                        id=SIDEBAR_BODY_ID,
                    ),
                    html.Div("Copy as dict", className="planner-section-title"),
                    html.Pre(
                        _python_space_preview(metadata, selection),
                        id=DICT_PREVIEW_ID,
                        className="planner-dict-preview",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.I(className="far fa-copy me-2"),
                                    "Copy search space dict",
                                    dcc.Clipboard(
                                        id=COPY_ID,
                                        content=_python_space_preview(metadata, selection),
                                        title="Copy search space dict",
                                        className="planner-copy-overlay",
                                    ),
                                ],
                                className="btn btn-primary btn-sm planner-copy-button",
                            ),
                            html.Button(
                                [
                                    html.I(className="fas fa-undo me-2"),
                                    "Reset selections",
                                ],
                                id=RESET_ID,
                                className="btn btn-outline-primary btn-sm",
                                n_clicks=0,
                            ),
                        ],
                        className="planner-sidebar-actions",
                    ),
                ],
                className="analysis-card planner-sidebar-card",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        _coverage_title(DEFAULT_TOP_TRIAL_COUNT),
                                        id=COVERAGE_TITLE_ID,
                                        className="planner-card-title",
                                    ),
                                    html.Div(
                                        _coverage_subtitle(DEFAULT_TOP_TRIAL_COUNT),
                                        id=COVERAGE_SUBTITLE_ID,
                                        className="planner-card-subtitle",
                                    ),
                                ]
                            ),
                            html.Span(
                                _coverage_badge(coverage),
                                id=COVERAGE_BADGE_ID,
                                className="planner-coverage-badge",
                            ),
                        ],
                        className="planner-coverage-heading",
                    ),
                    dcc.Graph(
                        id=COVERAGE_GRAPH_ID,
                        figure=_coverage_figure(
                            visible_trials,
                            metadata,
                            selection,
                            coverage_data["objective_name"],
                            coverage_data["optimize"],
                        ),
                        responsive=True,
                        style={"width": "100%", "height": "300px"},
                        config={
                            "displaylogo": False,
                            "toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE},
                        },
                    ),
                    html.Div(
                        [
                            dbc.Label(
                                "Top trials",
                                html_for=TOP_TRIAL_COUNT_ID,
                                className="small text-muted mb-1",
                            ),
                            dcc.Slider(
                                id=TOP_TRIAL_COUNT_ID,
                                min=1,
                                max=100,
                                step=1,
                                value=DEFAULT_TOP_TRIAL_COUNT,
                                marks={1: "1", 10: "10", 25: "25", 50: "50", 100: "100"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        className="planner-coverage-trial-count",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dbc.Label(
                                        "Spread attribute",
                                        html_for=SPREAD_ATTRIBUTE_ID,
                                        className="small text-muted mb-1",
                                    ),
                                    dcc.Dropdown(
                                        id=SPREAD_ATTRIBUTE_ID,
                                        options=get_select_options(
                                            coverage_data["spread_attributes"]
                                        ),
                                        value=None,
                                        clearable=True,
                                        disabled=not bool(coverage_data["spread_attributes"]),
                                        placeholder="Select numeric list user_attr ...",
                                    ),
                                ],
                                className="planner-coverage-spread-select",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dbc.Label(
                                                "Error bars",
                                                html_for=SPREAD_MODE_ID,
                                                className="small text-muted mb-1",
                                            ),
                                            dbc.RadioItems(
                                                id=SPREAD_MODE_ID,
                                                options=_spread_mode_options(disabled=True),
                                                value="sd",
                                                inline=True,
                                                className="planner-coverage-spread-mode",
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            dbc.Label(
                                                "Options",
                                                className="small text-muted mb-1",
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Checkbox(
                                                        id=SPREAD_TOGGLE_ID,
                                                        label="Show spread",
                                                        value=False,
                                                        disabled=True,
                                                        className=(
                                                            "planner-coverage-spread-toggle"
                                                        ),
                                                    ),
                                                    dbc.Checkbox(
                                                        id=SPREAD_SCATTER_ID,
                                                        label="Scatter",
                                                        value=False,
                                                        disabled=True,
                                                        className=(
                                                            "planner-coverage-spread-toggle"
                                                        ),
                                                    ),
                                                ],
                                                className="planner-coverage-option-toggles",
                                            ),
                                        ],
                                        className="planner-coverage-options-group",
                                    ),
                                ],
                                className="planner-coverage-spread-options",
                            ),
                        ],
                        className="planner-coverage-controls",
                    ),
                ],
                className="analysis-card planner-coverage-card",
            ),
            dbc.Offcanvas(
                [
                    html.Div(
                        id=INSIGHT_MODAL_TITLE_ID,
                        className="planner-insight-panel-subtitle",
                    ),
                    html.Div(
                        [
                            html.Span(
                                "Elite group",
                                className="planner-insight-elite-label",
                            ),
                            dbc.RadioItems(
                                id=INSIGHT_ELITE_CONTROL_ID,
                                options=elite_options,
                                value=default_elite_proportion,
                                className="btn-group planner-insight-elite-control",
                                inputClassName="btn-check",
                                labelClassName="btn btn-outline-primary btn-sm",
                                labelCheckedClassName="active",
                            ),
                        ],
                        className="planner-insight-elite-selector",
                    ),
                    dcc.Graph(
                        id=INSIGHT_GRAPH_ID,
                        responsive=True,
                        style={"width": "100%", "height": "520px"},
                        config={
                            "displaylogo": False,
                            "toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE},
                        },
                    ),
                    dbc.Button(
                        "Close",
                        id=INSIGHT_CLOSE_ID,
                        color="secondary",
                        className="planner-insight-close",
                        n_clicks=0,
                    ),
                ],
                id=INSIGHT_MODAL_ID,
                title="Parameter detail",
                is_open=False,
                placement="end",
                backdrop=False,
                scrollable=True,
                autofocus=False,
                className="planner-insight-panel",
            ),
        ],
        className="hpo-planning-grid",
    )


def _updated_selection(
    metadata: Sequence[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
    triggered_id: Any,
    value: Any,
) -> Dict[str, Dict[str, Any]]:
    """Apply one UI edit to a copy of the central selection state."""
    if triggered_id == RESET_ID:
        return _initial_selection(metadata)

    updated = {
        name: {
            key: list(item) if isinstance(item, list) else item for key, item in selected.items()
        }
        for name, selected in selection.items()
    }
    if not isinstance(triggered_id, dict) or value is None:
        return updated

    index = triggered_id.get("index")
    parameter = next((item for item in metadata if item["index"] == index), None)
    if parameter is None:
        return updated
    name = parameter["name"]
    control_type = triggered_id.get("type")
    try:
        if control_type == "hpo-planner-range":
            minimum = _denormalize_value(value[0], parameter)
            maximum = _denormalize_value(value[1], parameter)
            updated[name] = _coerce_selection(minimum, maximum, parameter)
        elif control_type == "hpo-planner-min":
            updated[name] = _coerce_selection(value, updated[name]["max"], parameter)
        elif control_type == "hpo-planner-max":
            updated[name] = _coerce_selection(updated[name]["min"], value, parameter)
        elif control_type == "hpo-planner-categories":
            choices = value or []
            updated[name] = {
                "choices": [choice for choice in parameter["choices"] if choice in choices]
            }
    except (TypeError, ValueError, OverflowError):
        # A temporarily incomplete number input or an undefined logarithm keeps the last state.
        return updated
    return updated


def _control_matches_selection(
    metadata: Sequence[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
    triggered_id: Any,
    value: Any,
) -> bool:
    """Return whether a triggering control already represents the central state."""
    if not isinstance(triggered_id, dict):
        return True
    parameter = next(
        (item for item in metadata if item["index"] == triggered_id.get("index")),
        None,
    )
    if parameter is None:
        return True
    selected = selection[parameter["name"]]
    control_type = triggered_id.get("type")
    if control_type == "hpo-planner-min":
        return value is not None and np.isclose(float(value), float(selected["min"]))
    if control_type == "hpo-planner-max":
        return value is not None and np.isclose(float(value), float(selected["max"]))
    if control_type == "hpo-planner-range":
        if value is None or len(value) != 2:
            return False
        metrics = _selection_metrics(parameter, selected)
        return bool(np.allclose(value, [metrics["min_normalized"], metrics["max_normalized"]]))
    if control_type == "hpo-planner-categories":
        return list(value or []) == list(selected["choices"])
    return True


def _reference_style(domain: Tuple[float, float]) -> Dict[str, str]:
    """Position the immutable 0–1 reference band within an expandable slider domain."""
    left = _position_percent(0, domain)
    right = _position_percent(1, domain)
    return {"left": f"{left:.6f}%", "width": f"{right - left:.6f}%"}


def _elite_distribution_components(
    parameter: Dict[str, Any],
    domain: Tuple[float, float],
) -> List[Component]:
    """Place Q10–Q90 bands and medians on three aligned, non-interactive subtracks."""
    tracks = []
    for elite in parameter["elite_positions"]:
        minimum = elite["minimum_position"]
        q10 = elite["q10_position"]
        median = elite["median_position"]
        q90 = elite["q90_position"]
        maximum = elite["maximum_position"]
        children: List[Component] = []
        title = f"{elite['group']}: no active trials"
        if q10 is not None and median is not None and q90 is not None:
            q10_percent = _position_percent(q10, domain)
            median_percent = _position_percent(median, domain)
            q90_percent = _position_percent(q90, domain)
            if minimum is not None and maximum is not None:
                minimum_percent = _position_percent(minimum, domain)
                maximum_percent = _position_percent(maximum, domain)
                children.append(
                    html.Span(
                        className="planner-elite-whisker",
                        style={
                            "left": f"{minimum_percent:.6f}%",
                            "width": f"{max(0.0, maximum_percent - minimum_percent):.6f}%",
                            "backgroundColor": elite["color"],
                        },
                    )
                )
            children.extend(
                [
                    html.Span(
                        className="planner-elite-band",
                        style={
                            "left": f"{q10_percent:.6f}%",
                            "width": f"{max(0.0, q90_percent - q10_percent):.6f}%",
                            "backgroundColor": elite["color"],
                        },
                    ),
                    html.Span(
                        className="planner-elite-marker",
                        style={
                            "left": f"{median_percent:.6f}%",
                            "backgroundColor": elite["color"],
                        },
                    ),
                ]
            )
            title = (
                f"{elite['group']} min–max: {_format_value(elite['minimum'])}–"
                f"{_format_value(elite['maximum'])}; Q10–Q90: {_format_value(elite['q10'])}–"
                f"{_format_value(elite['q90'])}; median: "
                f"{_format_value(elite['median'])} (n={elite['n']})"
            )
        tracks.append(
            html.Div(
                children,
                title=title,
                className="planner-elite-track",
            )
        )
    return tracks


def _extension_components(
    metrics: Dict[str, Any],
    domain: Tuple[float, float],
) -> List[Component]:
    """Render striped portions of a selection outside the current 0–1 reference."""
    segments = []
    for start, end in (
        (metrics["min_normalized"], min(metrics["max_normalized"], 0.0)),
        (max(metrics["min_normalized"], 1.0), metrics["max_normalized"]),
    ):
        if end <= start:
            continue
        left = _position_percent(start, domain)
        right = _position_percent(end, domain)
        segments.append(
            html.Span(
                className="planner-extension-segment",
                style={"left": f"{left:.6f}%", "width": f"{right - left:.6f}%"},
            )
        )
    return segments


def _unselected_components(
    metrics: Dict[str, Any],
    domain: Tuple[float, float],
) -> List[Component]:
    """Shade the visible planner regions outside the selected range."""
    minimum = min(100.0, max(0.0, _position_percent(metrics["min_normalized"], domain)))
    maximum = min(100.0, max(0.0, _position_percent(metrics["max_normalized"], domain)))
    return [
        html.Span(
            className="planner-unselected-segment",
            style={"left": "0", "width": f"{minimum:.6f}%"},
        ),
        html.Span(
            className="planner-unselected-segment",
            style={"left": f"{maximum:.6f}%", "width": f"{100 - maximum:.6f}%"},
        ),
    ]


def _sidebar_summary(
    metadata: Sequence[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
) -> Component:
    """Build the live selected-space table and relative-to-current summary."""
    table_rows = []
    counts = {"wider": 0, "narrower": 0, "same": 0}
    for parameter in metadata:
        selected = selection[parameter["name"]]
        if parameter["parameter_type"] in ("integer", "float"):
            metrics = _selection_metrics(parameter, selected)
            counts[metrics["comparison"]] += 1
            value_cells = [
                html.Td(_format_value(selected["min"])),
                html.Td(_format_value(selected["max"])),
            ]
        else:
            selected_choices = selected["choices"]
            value_cells = [
                html.Td(
                    ", ".join(str(value) for value in selected_choices) or "—",
                    colSpan=2,
                )
            ]
        table_rows.append(html.Tr([html.Th(parameter["name"]), *value_cells]))

    numeric_count = sum(counts.values())
    comparison_rows = []
    for comparison in ("wider", "narrower", "same"):
        count = counts[comparison]
        percentage = 0 if numeric_count == 0 else 100 * count / numeric_count
        comparison_rows.append(
            html.Div(
                [
                    html.Span(comparison.title()),
                    html.Div(
                        html.Div(
                            className=f"planner-summary-fill planner-summary-fill--{comparison}",
                            style={"width": f"{percentage:.6f}%"},
                        ),
                        className="planner-summary-bar",
                    ),
                    html.Span(f"{count} ({percentage:.0f}%)"),
                ],
                className="planner-summary-row",
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Span("Selection summary", className="planner-section-title"),
                    html.Span(f"{len(metadata)} parameters", className="planner-count-badge"),
                ],
                className="planner-selection-heading",
            ),
            html.Div(
                html.Table(
                    [
                        html.Thead(html.Tr([html.Th("Parameter"), html.Th("Min"), html.Th("Max")])),
                        html.Tbody(table_rows),
                    ],
                    className="planner-selection-table",
                ),
                className="planner-selection-table-wrap",
            ),
            html.Div("Relative to current space", className="planner-section-title"),
            html.Div(comparison_rows, className="planner-summary"),
        ]
    )


class HPOPlanning(DynamicPlugin):
    """Show empirical single-feature ranges for configurable elite groups."""

    id = "hpo_planning"
    name = "HPO Planner"
    icon = "fas fa-drafting-compass"
    help = "plugins/hpo_planning.html"
    activate_run_selection = True
    main_objective_input = "objective_id"
    main_objective_input_type = None

    def _dict_as_key(self, data: Dict[str, Any], remove_filters: bool = False) -> str:
        """Version cached process outputs when their serialized schema changes."""
        key = super()._dict_as_key(data, remove_filters=remove_filters)
        return f"{key}-v{CACHE_SCHEMA_VERSION}"

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """Accept non-empty Optuna studies and Optuna-only groups."""
        return _is_optuna_run_or_group(run) and not run.empty() and len(run.configspace) > 0

    @staticmethod
    def get_input_layout(register: Callable) -> List[Component]:
        """Return planning controls; the active run lives in the sidebar."""
        return [
            html.Div(
                [
                    dcc.Dropdown(
                        id=register("objective_direction", ["value", "options"]),
                        style={"display": "none"},
                    ),
                    html.Div(
                        [
                            html.Span("Elite groups", className="analysis-toolbar__label"),
                            help_button(
                                "Three distinct fractions of the run-wide best trials. "
                                "Their Q10–Q90 ranges and medians are shown from broadest "
                                "to strictest."
                            ),
                        ],
                        className="analysis-toolbar__group-label",
                    ),
                    *[
                        analysis_toolbar_item(
                            f"Top {percent}%",
                            dbc.Input(
                                id=register(f"elite_proportion_{index}", "value", type=float),
                                type="number",
                                min=ELITE_PROPORTION_MIN,
                                max=ELITE_PROPORTION_MAX,
                                step=0.001,
                            ),
                            width="7rem",
                        )
                        for index, percent in enumerate((10, 5, 1), start=1)
                    ],
                ],
                className="analysis-toolbar d-flex flex-wrap align-items-center gap-2",
            )
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Any]:
        """Return no additional display filters."""
        return []

    @interactive
    def register_callbacks(self) -> None:
        """Synchronize every planner control and derived view through one selection store."""
        super().register_callbacks()
        from deepcave import app

        @app.callback(  # type: ignore
            [
                Output(SELECTION_STORE_ID, "data"),
                Output(ROWS_ID, "children"),
                Output(SIDEBAR_BODY_ID, "children"),
                Output(DICT_PREVIEW_ID, "children"),
                Output(COPY_ID, "content"),
                Output(COVERAGE_GRAPH_ID, "figure"),
                Output(COVERAGE_BADGE_ID, "children"),
                Output(COVERAGE_TITLE_ID, "children"),
                Output(COVERAGE_SUBTITLE_ID, "children"),
            ],
            [
                Input({"type": "hpo-planner-range", "index": ALL}, "value"),
                Input({"type": "hpo-planner-min", "index": ALL}, "value"),
                Input({"type": "hpo-planner-max", "index": ALL}, "value"),
                Input({"type": "hpo-planner-categories", "index": ALL}, "value"),
                Input(RESET_ID, "n_clicks"),
                Input(SPREAD_ATTRIBUTE_ID, "value"),
                Input(SPREAD_TOGGLE_ID, "value"),
                Input(SPREAD_MODE_ID, "value"),
                Input(SPREAD_SCATTER_ID, "value"),
                Input(TOP_TRIAL_COUNT_ID, "value"),
            ],
            [
                State(METADATA_STORE_ID, "data"),
                State(SELECTION_STORE_ID, "data"),
                State(TRIALS_STORE_ID, "data"),
            ],
            prevent_initial_call=True,
        )
        def synchronize_planner(
            _ranges: List[Any],
            _minima: List[Any],
            _maxima: List[Any],
            _categories: List[Any],
            _reset_clicks: int,
            spread_attribute: Any,
            show_spread: bool,
            spread_mode: str,
            show_scatter: bool,
            top_trial_count: Any,
            metadata: List[Dict[str, Any]],
            selection: Dict[str, Dict[str, Any]],
            coverage_data: Dict[str, Any],
        ) -> Any:
            if not metadata or not selection or not coverage_data:
                return [no_update] * 9
            visible_trials = _visible_coverage_trials(coverage_data, top_trial_count)
            coverage_control_triggered = isinstance(ctx.triggered_id, str) and ctx.triggered_id in {
                SPREAD_ATTRIBUTE_ID,
                SPREAD_TOGGLE_ID,
                SPREAD_MODE_ID,
                SPREAD_SCATTER_ID,
                TOP_TRIAL_COUNT_ID,
            }
            if coverage_control_triggered:
                coverage = _coverage_state(visible_trials, metadata, selection)
                return [
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    _coverage_figure(
                        visible_trials,
                        metadata,
                        selection,
                        coverage_data["objective_name"],
                        coverage_data["optimize"],
                        spread_attribute,
                        show_spread,
                        spread_mode,
                        show_scatter,
                    ),
                    _coverage_badge(coverage),
                    _coverage_title(top_trial_count),
                    _coverage_subtitle(top_trial_count),
                ]
            triggered_value = ctx.triggered[0].get("value") if ctx.triggered else None
            updated = _updated_selection(metadata, selection, ctx.triggered_id, triggered_value)
            if updated == selection:
                if not _control_matches_selection(
                    metadata,
                    selection,
                    ctx.triggered_id,
                    triggered_value,
                ):
                    return [no_update, _planner_rows(metadata, selection)] + [no_update] * 7
                return [no_update] * 9
            preview = _python_space_preview(metadata, updated)
            coverage = _coverage_state(visible_trials, metadata, updated)
            return [
                updated,
                _planner_rows(metadata, updated),
                _sidebar_summary(metadata, updated),
                preview,
                preview,
                _coverage_figure(
                    visible_trials,
                    metadata,
                    updated,
                    coverage_data["objective_name"],
                    coverage_data["optimize"],
                    spread_attribute,
                    show_spread,
                    spread_mode,
                    show_scatter,
                ),
                _coverage_badge(coverage),
                _coverage_title(top_trial_count),
                _coverage_subtitle(top_trial_count),
            ]

        @app.callback(  # type: ignore
            [
                Output(SPREAD_TOGGLE_ID, "disabled"),
                Output(SPREAD_TOGGLE_ID, "value"),
                Output(SPREAD_MODE_ID, "options"),
                Output(SPREAD_SCATTER_ID, "disabled"),
            ],
            Input(SPREAD_ATTRIBUTE_ID, "value"),
        )
        def toggle_spread_control(spread_attribute: Any) -> Tuple[bool, bool, Any, bool]:
            """Enable and select the spread checkbox when an attribute is selected."""
            disabled, show_spread = _spread_control_state(spread_attribute)
            return disabled, show_spread, _spread_mode_options(disabled), disabled

        @app.callback(  # type: ignore
            [
                Output(INSIGHT_MODAL_ID, "is_open"),
                Output(INSIGHT_MODAL_TITLE_ID, "children"),
                Output(INSIGHT_GRAPH_ID, "figure"),
                Output(INSIGHT_SELECTION_STORE_ID, "data"),
            ],
            [
                Input({"type": INSIGHT_BUTTON_TYPE, "index": ALL}, "n_clicks"),
                Input(INSIGHT_CLOSE_ID, "n_clicks"),
                Input(INSIGHT_ELITE_CONTROL_ID, "value"),
            ],
            [
                State(INSIGHTS_STORE_ID, "data"),
                State(TRIALS_STORE_ID, "data"),
                State(INSIGHT_MODAL_ID, "is_open"),
                State(INSIGHT_SELECTION_STORE_ID, "data"),
            ],
            prevent_initial_call=True,
        )
        def show_parameter_insight(
            _open_clicks: List[int],
            _close_clicks: int,
            elite_proportion: Any,
            insights: List[Dict[str, Any]],
            coverage_data: Dict[str, Any],
            is_open: bool,
            selected_index: Any,
        ) -> Any:
            """Open the combined plot for the clicked parameter or close its modal."""
            if ctx.triggered_id == INSIGHT_CLOSE_ID:
                return False, no_update, no_update, None
            if ctx.triggered_id == INSIGHT_ELITE_CONTROL_ID:
                if not is_open or selected_index is None:
                    raise PreventUpdate
                insight = next(
                    (item for item in insights or [] if item.get("index") == selected_index),
                    None,
                )
                if insight is None or not coverage_data:
                    raise PreventUpdate
                return (
                    no_update,
                    no_update,
                    _parameter_insight_figure(
                        insight,
                        coverage_data["objective_name"],
                        elite_proportion,
                    ),
                    no_update,
                )
            if not isinstance(ctx.triggered_id, dict):
                raise PreventUpdate
            if not ctx.triggered or not ctx.triggered[0].get("value"):
                raise PreventUpdate

            index = ctx.triggered_id.get("index")
            if is_open and selected_index == index:
                return False, no_update, no_update, None
            insight = next(
                (item for item in insights or [] if item.get("index") == index),
                None,
            )
            if insight is None or not coverage_data:
                raise PreventUpdate
            title = f"Performance and elite enrichment — {insight['name']}"
            return (
                True,
                title,
                _parameter_insight_figure(
                    insight,
                    coverage_data["objective_name"],
                    elite_proportion,
                ),
                index,
            )

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Return defaults before an Optuna study is selected."""
        return {
            "objective_id": {"options": [], "value": None},
            "objective_direction": {
                "options": get_select_options(
                    ["Maximize", "Minimize"],
                    ["upper", "lower"],
                ),
                "value": "upper",
            },
            **{
                f"elite_proportion_{index}": {"value": proportion}
                for index, proportion in enumerate(ELITE_PROPORTION_DEFAULTS, start=1)
            },
        }

    def load_dependency_inputs(
        self,
        run: AbstractRun,
        previous_inputs: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Populate native and complete scalar-user-attribute objective choices."""
        objectives = _get_objectives_without_time(run)
        objective_ids = [run.get_objective_id(objective) for objective in objectives]
        user_attr_names = _complete_user_attr_names(getattr(run, "history", []))
        user_attr_values = [f"{USER_ATTR_PREFIX}{name}" for name in user_attr_names]
        objective_value = inputs["objective_id"]["value"]
        if isinstance(objective_value, str) and not objective_value.startswith(USER_ATTR_PREFIX):
            try:
                numeric_objective_value = int(objective_value)
            except ValueError:
                numeric_objective_value = None
            if numeric_objective_value in objective_ids:
                objective_value = numeric_objective_value
        if objective_value not in [*objective_ids, *user_attr_values]:
            objective_value = 0 if 0 in objective_ids else objective_ids[0]

        previous_objective = previous_inputs.get("objective_id", {}).get("value")
        if isinstance(objective_value, str) and objective_value.startswith(USER_ATTR_PREFIX):
            current_direction = inputs.get("objective_direction", {}).get("value")
            direction = (
                current_direction
                if previous_objective == objective_value and current_direction in ("upper", "lower")
                else "upper"
            )
        else:
            selected_objective = objectives[objective_ids.index(objective_value)]
            direction = selected_objective.optimize

        return {
            "objective_id": {
                "options": get_select_options(
                    [
                        *[f"Objective · {objective.name}" for objective in objectives],
                        *[f"User attr · {name}" for name in user_attr_names],
                    ],
                    [*objective_ids, *user_attr_values],
                ),
                "value": objective_value,
            },
            "objective_direction": {
                "options": get_select_options(
                    ["Maximize", "Minimize"],
                    ["upper", "lower"],
                ),
                "value": direction,
            },
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Collect successful trials and calculate all single-feature ranges."""
        objective_selection = inputs["objective_id"]
        user_attr_name = None
        objective_id = None
        if isinstance(objective_selection, str) and objective_selection.startswith(
            USER_ATTR_PREFIX
        ):
            user_attr_name = objective_selection[len(USER_ATTR_PREFIX) :]
            objective_name = user_attr_name
            optimize = inputs.get("objective_direction", "upper")
            if optimize not in ("upper", "lower"):
                optimize = "upper"
        else:
            try:
                objective_id = int(objective_selection)
            except (TypeError, ValueError):
                raise PreventUpdate
            objective = run.get_objective(objective_id)
            if objective is None:
                raise PreventUpdate
            objective_name = objective.name
            optimize = objective.optimize

        elite_proportions = _validate_elite_proportions(
            [inputs[f"elite_proportion_{index}"] for index in range(1, 4)]
        )
        spread_attributes = _complete_user_attr_names(run.history, list_values=True)
        trials: List[Dict[str, Any]] = []
        for history_index, history_trial in enumerate(run.history):
            objective_value: Any
            if user_attr_name is None:
                objective_value = history_trial.costs[objective_id]  # type: ignore[index]
            else:
                objective_value = _trial_user_attrs(history_trial).get(user_attr_name)
            if history_trial.status != Status.SUCCESS or not _is_finite_number(objective_value):
                continue
            user_attrs = _trial_user_attrs(history_trial)
            trials.append(
                {
                    "trial_number": _as_python_value(
                        history_trial.additional.get("optuna_trial_number", history_index)
                    ),
                    "objective": float(objective_value),
                    "configuration": {
                        name: _as_python_value(value)
                        for name, value in dict(run.get_config(history_trial.config_id)).items()
                    },
                    "spread_values": {
                        name: [float(value) for value in user_attrs[name]]
                        for name in spread_attributes
                    },
                }
            )
        if not trials:
            raise ValueError("No successful trials are available for the HPO Planner.")

        ranked_trials = sorted(
            trials,
            key=lambda trial: trial["objective"],
            reverse=optimize == "upper",
        )[:TOP_TRIAL_LIMIT]
        for rank, ranked_trial in enumerate(ranked_trials, start=1):
            ranked_trial["rank"] = rank

        return {
            "elite_proportions": elite_proportions,
            "features": list(run.configspace.keys()),
            "objective_name": objective_name,
            "optimize": optimize,
            "trials": ranked_trials,
            "total_trials": len(trials),
            "spread_attributes": spread_attributes,
            "insights": _parameter_insights(
                trials,
                run.configspace,
                optimize,
                elite_proportions,
            ),
            "summary": _summarize_features(
                trials,
                run.configspace,
                optimize,
                elite_proportions,
            ),
        }

    @staticmethod
    def get_output_layout(register: Callable) -> List[Any]:
        """Return the planner mount point; its children share one central selection state."""
        return [html.Div(id=register("planner", "children"))]

    @staticmethod
    def load_outputs(
        run: AbstractRun,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> List[Any]:
        """Build immutable parameter metadata and the reset planning state."""
        metadata = _build_planning_metadata(
            run.configspace,
            outputs["features"],
            outputs["elite_proportions"],
            outputs["summary"],
        )
        selection = _initial_selection(metadata)
        coverage_data = {
            "trials": outputs["trials"],
            "total_trials": outputs.get("total_trials", len(outputs["trials"])),
            "spread_attributes": outputs.get("spread_attributes", []),
            "objective_name": outputs["objective_name"],
            "optimize": outputs["optimize"],
        }
        return [_planner_layout(metadata, selection, coverage_data, outputs["insights"])]
