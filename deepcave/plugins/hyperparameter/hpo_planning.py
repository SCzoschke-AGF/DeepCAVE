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

import pprint

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
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

from deepcave import interactive
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.plugins.hyperparameter.analysis_utils import _get_objectives_without_time
from deepcave.runs import AbstractRun
from deepcave.runs.status import Status
from deepcave.utils.analysis_layout import analysis_toolbar_item
from deepcave.utils.layout import get_select_options, help_button

ELITE_PROPORTION_DEFAULTS = (0.10, 0.05, 0.01)
ELITE_PROPORTION_MIN = 0.001
ELITE_PROPORTION_MAX = 0.99
NUMERIC_PARAMETER_TYPES = (FloatHyperparameter, IntegerHyperparameter)
ELITE_MARKER_COLORS = ("#440154", "#21918C", "#FDE725")
DEFAULT_SLIDER_EXTENSION = 0.25
SPAN_COMPARISON_TOLERANCE = 0.01

METADATA_STORE_ID = "hpo-planning-metadata"
SELECTION_STORE_ID = "hpo-planning-selection"
SIDEBAR_BODY_ID = "hpo-planning-sidebar-body"
ROWS_ID = "hpo-planning-rows"
DICT_PREVIEW_ID = "hpo-planning-dict-preview"
COPY_ID = "hpo-planning-copy"
RESET_ID = "hpo-planning-reset"


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


def _numeric_row(parameter: Dict[str, Any], selection: Dict[str, Any]) -> Component:
    """Render one compact numerical planning row."""
    index = parameter["index"]
    metrics = _selection_metrics(parameter, selection)
    slider_minimum, slider_maximum = _slider_domain(metrics)
    step = parameter.get("step")
    input_step = step if step is not None else "any"
    return html.Div(
        [
            html.Div(parameter["name"], className="planner-parameter-name"),
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
            html.Div(parameter["name"], className="planner-parameter-name"),
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


def _planner_layout(
    metadata: List[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
) -> Component:
    """Return the two-column planner with immutable metadata and mutable selection stores."""
    return html.Div(
        [
            dcc.Store(id=METADATA_STORE_ID, data=metadata),
            dcc.Store(id=SELECTION_STORE_ID, data=selection),
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
                                    html.Div(
                                        "Adjust ranges graphically or edit original values.",
                                        className="planner-card-subtitle",
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
                            html.Div("Normalized position in current search space"),
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
                    html.Div(
                        [
                            html.I(className="far fa-lightbulb"),
                            " Values are normalized internally and shown in original units.",
                        ],
                        className="planner-tip",
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
    name = "HPO Planning"
    icon = "fas fa-drafting-compass"
    help = "plugins/hpo_planning.html"
    activate_run_selection = True

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """Accept non-empty Optuna studies with at least one hyperparameter."""
        return run.prefix == "Optuna" and not run.empty() and len(run.configspace) > 0

    @staticmethod
    def get_run_input_layout(register: Callable) -> Component:
        """Return all HPO planning controls in the shared compact toolbar."""
        return html.Div(
            [
                analysis_toolbar_item(
                    "Run",
                    dbc.Select(
                        id=register("run", ["value", "options"]),
                        placeholder="Select run ...",
                    ),
                    width="17rem",
                ),
                analysis_toolbar_item(
                    "Objective",
                    dbc.Select(
                        id=register("objective_id", ["value", "options"], type=int),
                        placeholder="Select objective ...",
                    ),
                    width="11rem",
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
                dcc.Loading(
                    html.Button(
                        "Apply",
                        id=register("process_analysis", "n_clicks", type=int),
                        className="btn btn-primary btn-sm",
                    ),
                    parent_className="ms-auto",
                ),
            ],
            className="analysis-toolbar d-flex flex-wrap align-items-center gap-2",
        )

    @staticmethod
    def get_input_layout(register: Callable) -> List[Any]:
        """Return no further inputs; every control lives in the toolbar above."""
        return []

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
            ],
            [
                Input({"type": "hpo-planner-range", "index": ALL}, "value"),
                Input({"type": "hpo-planner-min", "index": ALL}, "value"),
                Input({"type": "hpo-planner-max", "index": ALL}, "value"),
                Input({"type": "hpo-planner-categories", "index": ALL}, "value"),
                Input(RESET_ID, "n_clicks"),
            ],
            [
                State(METADATA_STORE_ID, "data"),
                State(SELECTION_STORE_ID, "data"),
            ],
            prevent_initial_call=True,
        )
        def synchronize_planner(
            _ranges: List[Any],
            _minima: List[Any],
            _maxima: List[Any],
            _categories: List[Any],
            _reset_clicks: int,
            metadata: List[Dict[str, Any]],
            selection: Dict[str, Dict[str, Any]],
        ) -> Any:
            if not metadata or not selection:
                return [no_update] * 5
            triggered_value = ctx.triggered[0].get("value") if ctx.triggered else None
            updated = _updated_selection(metadata, selection, ctx.triggered_id, triggered_value)
            if updated == selection:
                if not _control_matches_selection(
                    metadata,
                    selection,
                    ctx.triggered_id,
                    triggered_value,
                ):
                    return [
                        no_update,
                        _planner_rows(metadata, selection),
                        no_update,
                        no_update,
                        no_update,
                    ]
                return [no_update] * 5
            preview = _python_space_preview(metadata, updated)
            return [
                updated,
                _planner_rows(metadata, updated),
                _sidebar_summary(metadata, updated),
                preview,
                preview,
            ]

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Return defaults before an Optuna study is selected."""
        return {
            "objective_id": {"options": [], "value": None},
            **{
                f"elite_proportion_{index}": {"value": proportion}
                for index, proportion in enumerate(ELITE_PROPORTION_DEFAULTS, start=1)
            },
            "process_analysis": {"n_clicks": 0},
        }

    def load_dependency_inputs(
        self,
        run: AbstractRun,
        previous_inputs: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Populate objective choices for the selected study."""
        objectives = _get_objectives_without_time(run)
        objective_ids = [run.get_objective_id(objective) for objective in objectives]
        objective_value = inputs["objective_id"]["value"]
        if objective_value not in objective_ids:
            objective_value = objective_ids[0]

        return {
            "objective_id": {
                "options": get_select_options(
                    [objective.name for objective in objectives],
                    objective_ids,
                ),
                "value": objective_value,
            }
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Collect successful trials and calculate all single-feature ranges."""
        objective = run.get_objective(inputs["objective_id"])
        if objective is None:
            raise PreventUpdate

        elite_proportions = _validate_elite_proportions(
            [inputs[f"elite_proportion_{index}"] for index in range(1, 4)]
        )
        trials = []
        for trial in run.history:
            objective_value = trial.costs[inputs["objective_id"]]
            if (
                trial.status != Status.SUCCESS
                or objective_value is None
                or not np.isfinite(objective_value)
            ):
                continue
            trials.append(
                {
                    "objective": float(objective_value),
                    "configuration": dict(run.get_config(trial.config_id)),
                }
            )
        if not trials:
            raise ValueError("No successful trials are available for HPO planning.")

        return {
            "elite_proportions": elite_proportions,
            "features": list(run.configspace.keys()),
            "summary": _summarize_features(
                trials,
                run.configspace,
                objective.optimize,
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
        return [_planner_layout(metadata, selection)]
