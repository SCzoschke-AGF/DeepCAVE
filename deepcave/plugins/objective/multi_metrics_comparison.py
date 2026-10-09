# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare two metrics within one Optuna study."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import math

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter
from dash import dcc, html
from dash.development.base_component import Component
from scipy import stats

from deepcave import config
from deepcave.constants import VALUE_RANGE
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun, Status
from deepcave.utils.analysis_layout import (
    analysis_card,
    analysis_graph,
)
from deepcave.utils.global_objective import (
    is_user_attr_objective,
    main_objective_options,
    resolve_main_objective_id,
)
from deepcave.utils.styled_plotty import get_hyperparameter_ticks, save_image

BLUE = "#2563EB"
VIOLET = "#7C3AED"
LIGHT_GRAY = "#CBD5E1"
PARETO_COLOR = "#0F172A"
DEFAULT_TOP_N = 50
PLOT_HEIGHT = 620
PARALLEL_PLOT_HEIGHT = 420
SELECTION_OPTIONS = [
    {"label": "Metric 1", "value": "x"},
    {"label": "Metric 2", "value": "y"},
    {"label": "Compare both", "value": "compare"},
]
DISPLAY_OPTIONS = [
    {"label": "All trials", "value": "all"},
    {"label": "Top N only", "value": "selected"},
]


def _options_with_disabled(options: List[Dict[str, Any]], disabled: bool) -> List[Dict[str, Any]]:
    """Return fresh radio options with a shared disabled state."""
    return [{**option, "disabled": disabled} for option in options]


def _is_finite_number(value: Any) -> bool:
    """Return whether a value is a usable scalar metric value."""
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric_direction(run: AbstractRun, metric: Any) -> str:
    """Return the user-facing default direction for a metric."""
    objective_id = resolve_main_objective_id(run, metric)
    objective = run.get_objective(objective_id) if objective_id is not None else None
    return "maximize" if objective is not None and objective.optimize == "upper" else "minimize"


def _metric_name(run: AbstractRun, metric: Any) -> str:
    """Return a readable metric name without converter-specific prefixes."""
    if is_user_attr_objective(metric):
        return str(metric).split(":", 1)[1]
    objective_id = resolve_main_objective_id(run, metric)
    objective = run.get_objective(objective_id) if objective_id is not None else None
    return objective.name if objective is not None else "Metric"


def _trial_rows(run: AbstractRun, x_metric: Any, y_metric: Any) -> List[Dict[str, Any]]:
    """Collect successful Optuna trials with finite values on both selected metrics."""
    x_id = resolve_main_objective_id(run, x_metric)
    y_id = resolve_main_objective_id(run, y_metric)
    if x_id is None or y_id is None:
        return []

    rows = []
    for fallback_number, trial in enumerate(run.history):
        if trial.status != Status.SUCCESS:
            continue
        if x_id >= len(trial.costs) or y_id >= len(trial.costs):
            continue
        x_value = trial.costs[x_id]
        y_value = trial.costs[y_id]
        if not _is_finite_number(x_value) or not _is_finite_number(y_value):
            continue
        rows.append(
            {
                "trial_number": int(trial.additional.get("optuna_trial_number", fallback_number)),
                "config_id": int(trial.config_id),
                "x": float(x_value),
                "y": float(y_value),
            }
        )
    return rows


def _competition_ranks(rows: Sequence[Dict[str, Any]], key: str, direction: str) -> Dict[int, int]:
    """Return competition ranks, retaining a shared rank for equal values."""
    ordered = sorted(
        rows,
        key=lambda row: (
            -row[key] if direction == "maximize" else row[key],
            row["trial_number"],
        ),
    )
    ranks: Dict[int, int] = {}
    previous: Optional[float] = None
    current_rank = 0
    for position, row in enumerate(ordered, start=1):
        value = float(row[key])
        if previous is None or value != previous:
            current_rank = position
            previous = value
        ranks[int(row["trial_number"])] = current_rank
    return ranks


def _top_n(rows: Sequence[Dict[str, Any]], key: str, direction: str, n: int) -> Set[int]:
    """Return exactly n best trial numbers with a deterministic tie break."""
    ordered = sorted(
        rows,
        key=lambda row: (
            -row[key] if direction == "maximize" else row[key],
            row["trial_number"],
        ),
    )
    return {int(row["trial_number"]) for row in ordered[:n]}


def _best_trial(
    rows: Sequence[Dict[str, Any]], key: str, direction: str
) -> Optional[Dict[str, Any]]:
    """Return the best trial for one metric with a deterministic tie break."""
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            -row[key] if direction == "maximize" else row[key],
            row["trial_number"],
        ),
    )


def _pareto_trials(rows: Sequence[Dict[str, Any]], x_direction: str, y_direction: str) -> Set[int]:
    """Return non-dominated trials for two independently directed metrics."""
    efficient: Set[int] = set()
    directions = (x_direction, y_direction)
    for candidate in rows:
        dominated = False
        for other in rows:
            if other["trial_number"] == candidate["trial_number"]:
                continue
            weakly_better = []
            strictly_better = []
            for key, direction in zip(("x", "y"), directions):
                if direction == "maximize":
                    weakly_better.append(other[key] >= candidate[key])
                    strictly_better.append(other[key] > candidate[key])
                else:
                    weakly_better.append(other[key] <= candidate[key])
                    strictly_better.append(other[key] < candidate[key])
            if all(weakly_better) and any(strictly_better):
                dominated = True
                break
        if not dominated:
            efficient.add(int(candidate["trial_number"]))
    return efficient


def _correlations(rows: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Calculate correlations, returning None when either statistic is undefined."""
    if len(rows) < 2:
        return {"pearson": None, "spearman": None}
    x = [float(row["x"]) for row in rows]
    y = [float(row["y"]) for row in rows]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {"pearson": None, "spearman": None}

    pearson = float(stats.pearsonr(x, y)[0])
    spearman = float(stats.spearmanr(x, y)[0])
    return {
        "pearson": pearson if math.isfinite(pearson) else None,
        "spearman": spearman if math.isfinite(spearman) else None,
    }


def _format_statistic(value: Optional[float]) -> str:
    """Format a statistic consistently for the compact summary cards."""
    return "—" if value is None else f"{value:.2f}"


def _stat_tile(label: str, value: str) -> Component:
    """Return a compact label/value statistic."""
    return html.Div(
        [
            html.Div(label, className="multi-metrics-stat__label"),
            html.Div(value, className="multi-metrics-stat__value"),
        ],
        className="multi-metrics-stat",
    )


def _venn_figure(overlap: Dict[str, Any]) -> go.Figure:
    """Build a compact, deliberately non-area-proportional Venn diagram."""
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        height=175,
        margin={"l": 5, "r": 5, "t": 12, "b": 5},
        xaxis={"visible": False, "range": [0, 3]},
        yaxis={"visible": False, "range": [0, 2], "scaleanchor": "x", "scaleratio": 1},
        showlegend=False,
        shapes=[
            {
                "type": "circle",
                "x0": 0.35,
                "x1": 1.9,
                "y0": 0.2,
                "y1": 1.75,
                "fillcolor": "rgba(37, 99, 235, 0.22)",
                "line": {"color": BLUE, "width": 2},
            },
            {
                "type": "circle",
                "x0": 1.1,
                "x1": 2.65,
                "y0": 0.2,
                "y1": 1.75,
                "fillcolor": "rgba(124, 58, 237, 0.20)",
                "line": {"color": VIOLET, "width": 2},
            },
        ],
        annotations=[
            {"x": 0.82, "y": 0.98, "text": str(overlap["only_x"]), "showarrow": False},
            {
                "x": 1.5,
                "y": 0.98,
                "text": f"<b>{overlap['intersection']}</b>",
                "showarrow": False,
            },
            {"x": 2.18, "y": 0.98, "text": str(overlap["only_y"]), "showarrow": False},
            {"x": 0.65, "y": 1.95, "text": "Metric 1", "showarrow": False},
            {"x": 2.35, "y": 1.95, "text": "Metric 2", "showarrow": False},
        ],
    )
    return figure


def _parameter_label(parameter: str) -> str:
    """Wrap long underscored parameter names for the coordinate axes."""
    parts = parameter.split("_")
    if len(parameter) <= 14 or len(parts) == 1:
        return parameter
    split = (len(parts) + 1) // 2
    return f"{'_'.join(parts[:split])}<br>{'_'.join(parts[split:])}"


def _empty_parallel_figure(message: str) -> go.Figure:
    """Return an informative empty state for studies without plottable configurations."""
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, x=0.5, y=0.5)
    figure.update_layout(
        template="plotly_white",
        height=PARALLEL_PLOT_HEIGHT,
        margin={"l": 40, "r": 30, "t": 35, "b": 35},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def _parallel_ticks(hyperparameter: Any, additional_values: List[float]) -> Any:
    """Return complete search-space ticks in the normalized display coordinates."""
    if isinstance(hyperparameter, OrdinalHyperparameter):
        denominator = max(1, len(hyperparameter.sequence) - 1)
        return (
            [index / denominator for index in range(len(hyperparameter.sequence))]
            + [VALUE_RANGE[0]],
            [str(value) for value in hyperparameter.sequence] + ["NaN"],
        )
    return get_hyperparameter_ticks(
        hyperparameter,
        additional_values=additional_values,
        ticks=5,
        include_nan=True,
    )


def _trial_config_id(run: AbstractRun, row: Dict[str, Any]) -> Optional[int]:
    """Resolve a trial's configuration, including outputs cached before it was embedded."""
    if row.get("config_id") is not None:
        return int(row["config_id"])
    trial_number = int(row["trial_number"])
    for fallback_number, trial in enumerate(run.history):
        number = int(trial.additional.get("optuna_trial_number", fallback_number))
        if number == trial_number:
            return int(trial.config_id)
    return None


def _parameter_difference(hyperparameter: Any, left: float, right: float) -> float:
    """Measure a parameter difference relative to its complete search-space range."""
    left_inactive = left == VALUE_RANGE[0]
    right_inactive = right == VALUE_RANGE[0]
    if left_inactive or right_inactive:
        return 0.0 if left_inactive and right_inactive else 1.0
    if isinstance(hyperparameter, CategoricalHyperparameter):
        return 0.0 if left == right else 1.0
    return abs(float(left) - float(right))


def _parameter_sort_key(
    hyperparameter: Any, left: float, right: float, original_position: int
) -> Any:
    """Group parameters by visual direction, then order each group by difference."""
    difference = _parameter_difference(hyperparameter, left, right)
    if difference == 0:
        direction_group = 0
    elif left < right:
        direction_group = 1
    else:
        direction_group = 2
    return direction_group, difference, original_position


def _best_configurations_figure(run: AbstractRun, outputs: Dict[str, Any]) -> go.Figure:
    """Compare the best Metric 1 and Metric 2 configurations across the search space."""
    rows = outputs.get("trials", [])
    best_x = _best_trial(rows, "x", outputs["x_direction"])
    best_y = _best_trial(rows, "y", outputs["y_direction"])
    configspace = getattr(run, "configspace", None)
    if not best_x or not best_y or configspace is None or len(configspace) == 0:
        return _empty_parallel_figure("No parameter configurations available.")

    parameters = list(configspace.keys())
    selected = [
        (best_x, "Metric 1", outputs["x_name"], BLUE),
        (best_y, "Metric 2", outputs["y_name"], VIOLET),
    ]
    encoded_configurations = []
    raw_configurations = []
    for trial, _axis, _metric_name_value, _color in selected:
        config_id = _trial_config_id(run, trial)
        if config_id is None:
            return _empty_parallel_figure("No parameter configurations available.")
        configuration = run.get_config(config_id)
        raw_configurations.append(configuration)
        encoded = run.encode_config(configuration)
        for column, hyperparameter in enumerate(configspace.values()):
            if (
                isinstance(hyperparameter, OrdinalHyperparameter)
                and encoded[column] != VALUE_RANGE[0]
            ):
                encoded[column] /= max(1, len(hyperparameter.sequence) - 1)
        encoded_configurations.append(encoded)

    parameter_order = sorted(
        range(len(parameters)),
        key=lambda column: _parameter_sort_key(
            configspace[parameters[column]],
            encoded_configurations[0][column],
            encoded_configurations[1][column],
            column,
        ),
    )
    parameters = [parameters[column] for column in parameter_order]
    encoded_configurations = [
        [values[column] for column in parameter_order] for values in encoded_configurations
    ]

    x_values = list(range(len(parameters)))
    figure = go.Figure()
    for index, (trial, axis, metric_name, color) in enumerate(selected):
        raw_configuration = raw_configurations[index]
        encoded = encoded_configurations[index]
        hover_values = []
        for parameter in parameters:
            value = raw_configuration.get(parameter)
            hover_values.append(
                [parameter, "inactive" if value is None else str(value), trial["trial_number"]]
            )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=encoded,
                customdata=hover_values,
                mode="lines+markers",
                name=f"Best {axis} ({metric_name}) · Trial {trial['trial_number']}",
                line={"color": color, "width": 3},
                marker={
                    "color": color,
                    "size": 7,
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate=(
                    "%{customdata[0]}<br>Value: %{customdata[1]}"
                    "<extra>Trial %{customdata[2]}</extra>"
                ),
            )
        )

    shapes = []
    annotations = []
    for column, (x_value, parameter) in enumerate(zip(x_values, parameters)):
        hyperparameter = configspace[parameter]
        additional_values = [values[column] for values in encoded_configurations]
        tick_values, tick_labels = _parallel_ticks(hyperparameter, additional_values)
        shapes.append(
            {
                "type": "line",
                "x0": x_value,
                "x1": x_value,
                "y0": VALUE_RANGE[0],
                "y1": VALUE_RANGE[1],
                "line": {"color": "#CBD5E1", "width": 1},
                "layer": "below",
            }
        )
        annotations.append(
            {
                "x": x_value,
                "y": 1.08,
                "text": _parameter_label(parameter),
                "showarrow": False,
                "yref": "paper",
                "font": {"size": 11, "color": "#334155"},
            }
        )
        for tick_value, tick_label in zip(tick_values, tick_labels):
            annotations.append(
                {
                    "x": x_value,
                    "y": tick_value,
                    "text": tick_label,
                    "showarrow": False,
                    "xshift": -7,
                    "xanchor": "right",
                    "font": {"size": 9, "color": "#64748B"},
                }
            )

    figure.update_layout(
        template="plotly_white",
        height=PARALLEL_PLOT_HEIGHT,
        margin={"l": 70, "r": 35, "t": 75, "b": 70},
        shapes=shapes,
        annotations=annotations,
        font={"family": "Arial, sans-serif", "size": 12, "color": "#334155"},
        hovermode="closest",
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": -0.16,
            "yanchor": "top",
        },
    )
    figure.update_xaxes(
        visible=False,
        range=[-0.08, max(0, len(parameters) - 1) + 0.08],
        fixedrange=True,
    )
    figure.update_yaxes(visible=False, range=VALUE_RANGE, fixedrange=True)
    return figure


class MultiMetricsComparison(DynamicPlugin):
    """Compare values, rankings, top sets, and Pareto efficiency within one study."""

    id = "multi_metrics_comparison"
    name = "Multi-Metrics Comparison"
    icon = "fas fa-project-diagram"
    help = "plugins/multi_metrics_comparison.html"
    activate_run_selection = True

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """Accept only individual Optuna studies."""
        return run.prefix == "Optuna"

    @staticmethod
    def get_input_layout(register: Callable) -> List[Component]:
        """Return the compact comparison control bar."""
        direction_options = [
            {"label": "Maximize", "value": "maximize"},
            {"label": "Minimize", "value": "minimize"},
        ]
        scale_options = [
            {"label": "Metric values", "value": "values"},
            {"label": "Rank", "value": "rank"},
        ]

        def field(label: str, control: Component, modifier: str = "") -> Component:
            class_name = "multi-metrics-field"
            if modifier:
                class_name += f" {modifier}"
            return html.Div(
                [html.Label(label, className="multi-metrics-field__label"), control],
                className=class_name,
            )

        return [
            html.Div(
                [
                    html.P(
                        "Compare two metrics, analyze top trials and their agreement.",
                        className="multi-metrics-subtitle",
                    ),
                    html.Div(
                        [
                            field(
                                "Metric 1",
                                dbc.Select(
                                    id=register("x_metric", ["value", "options"]),
                                    placeholder="Select metric ...",
                                ),
                                "multi-metrics-field--metric",
                            ),
                            field(
                                "Metric 2",
                                dbc.Select(
                                    id=register("y_metric", ["value", "options"]),
                                    placeholder="Select metric ...",
                                ),
                                "multi-metrics-field--metric",
                            ),
                            field(
                                "Trials",
                                dbc.RadioItems(
                                    id=register("trial_mode", "value"),
                                    options=[
                                        {"label": "All", "value": "all"},
                                        {"label": "Top N", "value": "top_n"},
                                    ],
                                    className="multi-metrics-segmented",
                                ),
                                "multi-metrics-field--trials",
                            ),
                            html.Div(
                                field(
                                    "N",
                                    dbc.Input(
                                        id=register("n", ["value", "max", "disabled"], type=int),
                                        type="number",
                                        min=1,
                                        step=1,
                                    ),
                                ),
                                className="multi-metrics-field-wrapper--n",
                            ),
                            html.Div(
                                field(
                                    "Top N trials by",
                                    dbc.RadioItems(
                                        id=register("selection", ["value", "options"]),
                                        options=_options_with_disabled(
                                            SELECTION_OPTIONS, disabled=True
                                        ),
                                        className="multi-metrics-segmented",
                                    ),
                                ),
                                className="multi-metrics-field-wrapper--selection",
                            ),
                        ],
                        className="multi-metrics-toolbar__primary",
                    ),
                    html.Div(
                        [
                            field(
                                "Scale",
                                dbc.RadioItems(
                                    id=register("scale", "value", filter=True),
                                    options=scale_options,
                                    className="multi-metrics-segmented",
                                ),
                                "multi-metrics-field--scale",
                            ),
                            html.Div(
                                field(
                                    "Display",
                                    dbc.RadioItems(
                                        id=register("display", ["value", "options"], filter=True),
                                        options=_options_with_disabled(
                                            DISPLAY_OPTIONS, disabled=True
                                        ),
                                        className="multi-metrics-segmented",
                                    ),
                                    "multi-metrics-field--display",
                                ),
                            ),
                            dbc.Checkbox(
                                id=register("show_pareto", "value", filter=True),
                                label="Show Pareto front",
                                value=True,
                                className="multi-metrics-pareto-check",
                            ),
                            html.Details(
                                [
                                    html.Summary("Advanced (metric direction)"),
                                    html.Div(
                                        [
                                            field(
                                                "Metric 1 direction",
                                                dbc.Select(
                                                    id=register("x_direction", "value"),
                                                    options=direction_options,
                                                ),
                                            ),
                                            field(
                                                "Metric 2 direction",
                                                dbc.Select(
                                                    id=register("y_direction", "value"),
                                                    options=direction_options,
                                                ),
                                            ),
                                        ],
                                        className="multi-metrics-advanced__fields",
                                    ),
                                ],
                                className="multi-metrics-advanced",
                            ),
                        ],
                        className="multi-metrics-toolbar__secondary",
                    ),
                ],
                className="multi-metrics-controls",
            )
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Any]:
        """Keep display-only controls in the shared toolbar."""
        return []

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Return defaults which are completed once the selected run is available."""
        return {
            "x_metric": {"value": None, "options": []},
            "x_direction": {"value": "maximize"},
            "y_metric": {"value": None, "options": []},
            "y_direction": {"value": "maximize"},
            "trial_mode": {"value": "all"},
            "n": {"value": DEFAULT_TOP_N, "max": DEFAULT_TOP_N, "disabled": True},
            "selection": {
                "value": "compare",
                "options": _options_with_disabled(SELECTION_OPTIONS, disabled=True),
            },
            "scale": {"value": "values"},
            "display": {
                "value": "selected",
                "options": _options_with_disabled(DISPLAY_OPTIONS, disabled=True),
            },
            "show_pareto": {"value": True},
        }

    def load_dependency_inputs(
        self,
        run: Optional[AbstractRun],
        previous_inputs: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Populate metrics, directions, N bounds, and conditional control availability."""
        if run is None:
            return {}

        options = main_objective_options(run)
        values = [option["value"] for option in options]
        if not values:
            return {
                "x_metric": {"options": [], "value": None},
                "y_metric": {"options": [], "value": None},
                "n": {"value": 1, "max": 1, "disabled": True},
                "selection": {"options": _options_with_disabled(SELECTION_OPTIONS, disabled=True)},
                "display": {"options": _options_with_disabled(DISPLAY_OPTIONS, disabled=True)},
            }

        x_metric = inputs.get("x_metric", {}).get("value")
        if x_metric not in values:
            x_metric = values[0]
        y_metric = inputs.get("y_metric", {}).get("value")
        if y_metric not in values:
            y_metric = values[1] if len(values) > 1 else values[0]

        previous_x = previous_inputs.get("x_metric", {}).get("value")
        previous_y = previous_inputs.get("y_metric", {}).get("value")
        previous_run = previous_inputs.get("run", {}).get("value")
        current_run = inputs.get("run", {}).get("value")
        run_changed = previous_run is not None and previous_run != current_run
        x_direction = inputs.get("x_direction", {}).get("value")
        y_direction = inputs.get("y_direction", {}).get("value")
        if run_changed or x_metric != previous_x or x_direction not in ("maximize", "minimize"):
            x_direction = _metric_direction(run, x_metric)
        if run_changed or y_metric != previous_y or y_direction not in ("maximize", "minimize"):
            y_direction = _metric_direction(run, y_metric)

        eligible_count = len(_trial_rows(run, x_metric, y_metric))
        current_n = inputs.get("n", {}).get("value")
        top_n = inputs.get("trial_mode", {}).get("value") == "top_n"
        if eligible_count:
            if current_n is None:
                # Keep the transient empty state while the user replaces the input value.
                effective_n = None
            else:
                try:
                    requested_n = int(current_n)
                except (TypeError, ValueError):
                    requested_n = DEFAULT_TOP_N
                effective_n = min(max(requested_n, 1), eligible_count)
            maximum_n = eligible_count
            n_disabled = not top_n
        else:
            effective_n = 1
            maximum_n = 1
            n_disabled = True

        return {
            "x_metric": {"options": options, "value": x_metric},
            "x_direction": {"value": x_direction},
            "y_metric": {"options": options, "value": y_metric},
            "y_direction": {"value": y_direction},
            "n": {"value": effective_n, "max": maximum_n, "disabled": n_disabled},
            "selection": {
                "options": _options_with_disabled(
                    SELECTION_OPTIONS, disabled=not top_n or not eligible_count
                )
            },
            "display": {
                "options": _options_with_disabled(
                    DISPLAY_OPTIONS, disabled=not top_n or not eligible_count
                )
            },
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Build the valid universe, selection sets, ranks, and descriptive statistics."""
        rows = _trial_rows(run, inputs["x_metric"], inputs["y_metric"])
        universe = {int(row["trial_number"]) for row in rows}
        effective_n = min(max(int(inputs["n"]), 1), len(rows)) if rows else 0

        x_ranks = _competition_ranks(rows, "x", inputs["x_direction"])
        y_ranks = _competition_ranks(rows, "y", inputs["y_direction"])
        for row in rows:
            trial_number = int(row["trial_number"])
            row["x_rank"] = x_ranks[trial_number]
            row["y_rank"] = y_ranks[trial_number]

        set_x = _top_n(rows, "x", inputs["x_direction"], effective_n)
        set_y = _top_n(rows, "y", inputs["y_direction"], effective_n)
        if inputs["trial_mode"] == "all":
            analysis_set = set(universe)
        elif inputs["selection"] == "x":
            analysis_set = set(set_x)
        elif inputs["selection"] == "y":
            analysis_set = set(set_y)
        else:
            analysis_set = set_x | set_y

        analysis_rows = [row for row in rows if row["trial_number"] in analysis_set]
        pareto = _pareto_trials(
            analysis_rows,
            inputs["x_direction"],
            inputs["y_direction"],
        )
        intersection = set_x & set_y
        union = set_x | set_y
        overlap = {
            "intersection": len(intersection),
            "only_x": len(set_x - set_y),
            "only_y": len(set_y - set_x),
            "union": len(union),
            "jaccard": len(intersection) / len(union) if union else None,
            "overlap_rate": len(intersection) / effective_n if effective_n else None,
        }
        correlations = _correlations(analysis_rows)
        return {
            "trials": rows,
            "universe": sorted(universe),
            "set_x": sorted(set_x),
            "set_y": sorted(set_y),
            "analysis_set": sorted(analysis_set),
            "display_context": sorted(universe - analysis_set),
            "pareto_set": sorted(pareto),
            "overlap": overlap,
            "correlations": correlations,
            "effective_n": effective_n,
            "x_name": _metric_name(run, inputs["x_metric"]),
            "y_name": _metric_name(run, inputs["y_metric"]),
            "x_direction": inputs["x_direction"],
            "y_direction": inputs["y_direction"],
        }

    @staticmethod
    def get_output_layout(register: Callable) -> List[Any]:
        """Return the comparison plot and compact descriptive sidebar."""
        graph = analysis_graph(
            register("graph", "figure"),
            PLOT_HEIGHT,
            config.FIGURE_DOWNLOAD_SCALE,
        )
        agreement_container_id = register("agreement_container", "style")
        agreement_id = register("agreement", "children")
        insights_id = register("insights", "children")
        parallel_graph = analysis_graph(
            register("parallel_graph", "figure"),
            PARALLEL_PLOT_HEIGHT,
            config.FIGURE_DOWNLOAD_SCALE,
        )
        return [
            html.Div(
                [
                    html.Div(
                        [
                            analysis_card("Metric comparison", graph, "multi-metrics-main-card"),
                            analysis_card(
                                "Best-trial configurations",
                                parallel_graph,
                                "multi-metrics-main-card",
                            ),
                        ],
                        className="multi-metrics-plots",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "Analysis & Insights",
                                        className="multi-metrics-card__header",
                                    ),
                                    html.Div(id=insights_id),
                                ],
                                className="multi-metrics-card",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Agreement & overlap",
                                        className="multi-metrics-card__header",
                                    ),
                                    html.Div(id=agreement_id),
                                ],
                                id=agreement_container_id,
                                className="multi-metrics-card",
                            ),
                        ],
                        className="multi-metrics-sidebar",
                    ),
                ],
                className="multi-metrics-grid",
            )
        ]

    @staticmethod
    def load_outputs(
        run: AbstractRun, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> List[Any]:
        """Render the selected analysis set and optional display context."""
        rows = {int(row["trial_number"]): row for row in outputs["trials"]}
        analysis_set = set(outputs["analysis_set"])
        set_x = set(outputs["set_x"])
        set_y = set(outputs["set_y"])
        compare = inputs["trial_mode"] == "top_n" and inputs["selection"] == "compare"
        show_context = inputs["trial_mode"] == "top_n" and inputs["display"] == "all"
        scale = inputs["scale"]

        x_key = "x_rank" if scale == "rank" else "x"
        y_key = "y_rank" if scale == "rank" else "y"
        figure = go.Figure()

        def add_points(
            trial_numbers: Set[int],
            name: str,
            color: str,
            size: int = 9,
            opacity: float = 0.9,
            line: Optional[Dict[str, Any]] = None,
            hover: bool = True,
        ) -> None:
            selected = [rows[number] for number in sorted(trial_numbers) if number in rows]
            if not selected:
                return
            customdata = [
                [row["trial_number"], row["x"], row["y"], row["x_rank"], row["y_rank"]]
                for row in selected
            ]
            marker: Dict[str, Any] = {
                "color": color,
                "size": size,
                "opacity": opacity,
            }
            if line is not None:
                marker["line"] = line
            figure.add_trace(
                go.Scatter(
                    x=[row[x_key] for row in selected],
                    y=[row[y_key] for row in selected],
                    customdata=customdata,
                    mode="markers",
                    name=name,
                    marker=marker,
                    hovertemplate=(
                        "Trial %{customdata[0]}<br>"
                        + f"{outputs['x_name']}: %{{customdata[1]:.6g}} "
                        "(rank %{customdata[3]})<br>"
                        + f"{outputs['y_name']}: %{{customdata[2]:.6g}} "
                        "(rank %{customdata[4]})<extra></extra>"
                        if hover
                        else None
                    ),
                    hoverinfo=None if hover else "skip",
                )
            )

        if show_context:
            add_points(
                set(outputs["display_context"]),
                "Other trials",
                LIGHT_GRAY,
                size=7,
                opacity=0.55,
                hover=False,
            )

        if compare:
            add_points((set_x - set_y) & analysis_set, "Top N by Metric 1", BLUE)
            add_points((set_y - set_x) & analysis_set, "Top N by Metric 2", VIOLET)
            add_points(
                (set_x & set_y) & analysis_set,
                "Top N by Metric 1 and Metric 2",
                BLUE,
                size=11,
                line={"color": VIOLET, "width": 3},
            )
        elif inputs["trial_mode"] == "top_n" and inputs["selection"] == "y":
            add_points(analysis_set, "Top N by Metric 2", VIOLET)
        elif inputs["trial_mode"] == "top_n":
            add_points(analysis_set, "Top N by Metric 1", BLUE)
        else:
            add_points(analysis_set, "All valid trials", BLUE)

        if inputs["show_pareto"]:
            pareto_rows = [
                rows[number]
                for number in outputs["pareto_set"]
                if number in rows and number in analysis_set
            ]
            pareto_rows.sort(key=lambda row: (row[x_key], row[y_key], row["trial_number"]))
            if pareto_rows:
                figure.add_trace(
                    go.Scatter(
                        x=[row[x_key] for row in pareto_rows],
                        y=[row[y_key] for row in pareto_rows],
                        mode="lines+markers",
                        name="Pareto front",
                        showlegend=False,
                        line={"color": PARETO_COLOR, "width": 2},
                        marker={
                            "color": "white",
                            "line": {"color": PARETO_COLOR, "width": 2},
                            "size": 8,
                        },
                        customdata=[row["trial_number"] for row in pareto_rows],
                        hovertemplate="Pareto-optimal trial %{customdata}<extra></extra>",
                    )
                )

        x_title = f"Rank — {outputs['x_name']}" if scale == "rank" else outputs["x_name"]
        y_title = f"Rank — {outputs['y_name']}" if scale == "rank" else outputs["y_name"]
        figure.update_xaxes(title_text=f"<b>{x_title}</b>", autorange=True)
        figure.update_yaxes(title_text=f"<b>{y_title}</b>", autorange=True)
        figure.update_layout(
            template="plotly_white",
            showlegend=False,
            height=PLOT_HEIGHT,
            margin={"l": 70, "r": 25, "t": 48, "b": 115},
            font={"family": "Arial, sans-serif", "size": 13, "color": "#334155"},
            legend={
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": -0.24,
                "yanchor": "top",
            },
            hovermode="closest",
        )
        save_image(figure, "multi_metrics_comparison.pdf")
        parallel_figure = _best_configurations_figure(run, outputs)
        save_image(parallel_figure, "multi_metrics_best_configurations.pdf")

        agreement_children: List[Component] = []
        if compare:
            overlap = outputs["overlap"]
            agreement_children = [
                dcc.Graph(
                    figure=_venn_figure(overlap),
                    config={"displayModeBar": False, "staticPlot": True},
                    className="multi-metrics-venn",
                ),
                html.Div(
                    [
                        _stat_tile("Jaccard", _format_statistic(overlap["jaccard"])),
                        _stat_tile(
                            "Overlap rate",
                            (
                                "—"
                                if overlap["overlap_rate"] is None
                                else (
                                    f"{100 * overlap['overlap_rate']:.0f} % "
                                    f"({overlap['intersection']}/{outputs['effective_n']})"
                                )
                            ),
                        ),
                    ],
                    className="multi-metrics-stats",
                ),
            ]

        correlations = outputs["correlations"]
        insights_children = html.Div(
            [
                _stat_tile("Pearson r", _format_statistic(correlations["pearson"])),
                _stat_tile("Spearman ρ", _format_statistic(correlations["spearman"])),
                _stat_tile(
                    "Pareto",
                    f"{len(outputs['pareto_set'])} / {len(outputs['analysis_set'])} trials",
                ),
            ],
            className="multi-metrics-stats",
        )
        comparison_style = {} if compare else {"display": "none"}
        return [
            figure,
            comparison_style,
            agreement_children,
            insights_children,
            parallel_figure,
        ]
