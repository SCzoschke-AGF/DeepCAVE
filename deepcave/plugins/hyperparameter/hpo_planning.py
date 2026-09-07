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

from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    OrdinalHyperparameter,
)
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave import config
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.plugins.hyperparameter.analysis_utils import _get_objectives_without_time
from deepcave.runs import AbstractRun
from deepcave.runs.status import Status
from deepcave.utils.layout import get_select_options, help_button
from deepcave.utils.styled_plotty import save_image

ELITE_PROPORTION_DEFAULTS = (0.10, 0.05, 0.01)
ELITE_PROPORTION_MIN = 0.001
ELITE_PROPORTION_MAX = 0.99
SECTION_HEIGHT = 4
ROWS_PER_FEATURE = 3


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
    """Format raw values compactly for Plotly tooltips."""
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


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
    def get_input_layout(register: Callable) -> List[Any]:
        """Return objective and elite-group controls."""
        elite_inputs = []
        for index in range(1, 4):
            elite_inputs.append(
                html.Div(
                    [
                        html.Label(f"Elite Proportion #{index}"),
                        dcc.Input(
                            id=register(f"elite_proportion_{index}", "value", type=float),
                            type="number",
                            min=ELITE_PROPORTION_MIN,
                            max=ELITE_PROPORTION_MAX,
                            step=0.001,
                        ),
                    ],
                    className="col-md-4",
                )
            )

        return [
            html.Div(
                [
                    html.Label("Objective"),
                    dcc.Dropdown(
                        id=register("objective_id", ["value", "options"], type=int),
                        placeholder="Select objective ...",
                        clearable=False,
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Elite Groups"),
                            help_button(
                                "Three distinct fractions of the run-wide best trials. "
                                "Values are displayed from broadest to strictest."
                            ),
                        ],
                        className="col-12",
                    ),
                    *elite_inputs,
                ],
                className="row",
            ),
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Any]:
        """Return no additional display filters."""
        return []

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Return defaults before an Optuna study is selected."""
        return {
            "objective_id": {"options": [], "value": None},
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
        """Return the dynamically sized elite-range graph."""
        return [
            dcc.Graph(
                id=register("graph", ["figure", "style"]),
                style={"height": config.FIGURE_HEIGHT},
                config={
                    "toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE},
                },
            )
        ]

    @staticmethod
    def load_outputs(
        run: AbstractRun,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> List[Any]:
        """Build the normalized elite-group range figure."""
        summary_lookup = {(row["feature"], row["proportion"]): row for row in outputs["summary"]}
        features = outputs["features"]
        elite_proportions = outputs["elite_proportions"]
        row_positions = []
        row_labels = []
        shapes = []
        annotations = []
        empty_annotations = []

        range_x: List[Any] = []
        range_y: List[Any] = []
        range_customdata: List[Any] = []
        quantile_x: List[Any] = []
        quantile_y: List[Any] = []
        quantile_customdata: List[Any] = []
        median_x: List[Any] = []
        median_y: List[Any] = []
        median_customdata: List[Any] = []

        for feature_index, feature in enumerate(features):
            section_start = SECTION_HEIGHT * feature_index
            positions = [section_start + offset for offset in range(1, ROWS_PER_FEATURE + 1)]
            row_positions.extend(positions)
            row_labels.extend([_group_label(proportion) for proportion in elite_proportions])
            annotations.append(
                {
                    "x": 0.5,
                    "y": section_start,
                    "xref": "x",
                    "yref": "y",
                    "text": f"<b>{feature}</b>",
                    "showarrow": False,
                }
            )
            if feature_index % 2 == 0:
                shapes.append(
                    {
                        "type": "rect",
                        "xref": "paper",
                        "yref": "y",
                        "x0": 0,
                        "x1": 1,
                        "y0": section_start - 0.5,
                        "y1": section_start + 3.5,
                        "fillcolor": "#F2F2F2",
                        "line": {"width": 0},
                        "layer": "below",
                    }
                )

            for position, proportion in zip(positions, elite_proportions):
                row = summary_lookup[(feature, proportion)]
                if row["n"] == 0:
                    empty_annotations.append(
                        {
                            "x": 0.5,
                            "y": position,
                            "xref": "x",
                            "yref": "y",
                            "text": "No active trials",
                            "font": {"color": "#888888", "size": 10},
                            "showarrow": False,
                        }
                    )
                    continue

                customdata = [
                    feature,
                    row["group"],
                    row["n"],
                    _format_value(row["minimum"]),
                    _format_value(row["q10"]),
                    _format_value(row["median"]),
                    _format_value(row["q90"]),
                    _format_value(row["maximum"]),
                    f"{row['minimum_normalized']:.3f}",
                    f"{row['q10_normalized']:.3f}",
                    f"{row['median_normalized']:.3f}",
                    f"{row['q90_normalized']:.3f}",
                    f"{row['maximum_normalized']:.3f}",
                ]
                range_x.extend([row["minimum_normalized"], row["maximum_normalized"], None])
                range_y.extend([position, position, None])
                range_customdata.extend([customdata, customdata, None])
                quantile_x.extend([row["q10_normalized"], row["q90_normalized"]])
                quantile_y.extend([position, position])
                quantile_customdata.extend([customdata, customdata])
                median_x.append(row["median_normalized"])
                median_y.append(position)
                median_customdata.append(customdata)

        hovertemplate = (
            "Feature: %{customdata[0]}<br>Elite group: %{customdata[1]}<br>"
            "Active trials: %{customdata[2]}<br>Minimum: %{customdata[3]} "
            "(%{customdata[8]})<br>Q10: %{customdata[4]} (%{customdata[9]})<br>"
            "Median: %{customdata[5]} (%{customdata[10]})<br>Q90: %{customdata[6]} "
            "(%{customdata[11]})<br>Maximum: %{customdata[7]} "
            "(%{customdata[12]})<extra></extra>"
        )
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=range_x,
                y=range_y,
                mode="lines+markers",
                name="Minimum–Maximum",
                line={"color": "#4C78A8", "width": 4},
                marker={
                    "color": "#4C78A8",
                    "size": 7,
                    "line": {"color": "white", "width": 0.6},
                },
                customdata=range_customdata,
                hovertemplate=hovertemplate,
                connectgaps=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=quantile_x,
                y=quantile_y,
                mode="markers",
                name="Q10 / Q90",
                marker={"color": "#222222", "size": 18, "symbol": "line-ns"},
                customdata=quantile_customdata,
                hovertemplate=hovertemplate,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=median_x,
                y=median_y,
                mode="markers",
                name="Median",
                marker={
                    "color": "#F58518",
                    "size": 10,
                    "line": {"color": "white", "width": 0.8},
                },
                customdata=median_customdata,
                hovertemplate=hovertemplate,
            )
        )

        figure_height = max(500, 100 + 150 * len(features))
        figure.update_layout(
            title="Elite-Group Hyperparameter Ranges",
            xaxis={
                "title": "Normalized Position in Observed Search Space (0 = Min, 1 = Max)",
                "range": [-0.02, 1.02],
                "tickvals": [0, 0.25, 0.5, 0.75, 1],
            },
            yaxis={
                "tickvals": row_positions,
                "ticktext": row_labels,
                "range": [SECTION_HEIGHT * len(features) - 0.25, -0.75],
                "showgrid": False,
            },
            shapes=shapes,
            annotations=annotations + empty_annotations,
            legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.02},
            margin=config.FIGURE_MARGIN,
            font={"size": config.FIGURE_FONT_SIZE},
            height=figure_height,
            hovermode="closest",
        )
        save_image(figure, "hpo_planning.pdf")
        return [figure, {"height": f"{figure_height}px"}]
