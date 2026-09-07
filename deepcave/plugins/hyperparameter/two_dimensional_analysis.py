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
"""Two-dimensional empirical trial and elite analysis for Optuna studies."""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import (
    FloatHyperparameter,
    Hyperparameter,
    IntegerHyperparameter,
)
from dash import dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from deepcave import config
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.plugins.hyperparameter.analysis_utils import (
    _add_trial_statistics,
    _collect_trials,
    _get_objectives_without_time,
    _group_parameter_values,
)
from deepcave.runs import AbstractRun
from deepcave.utils.layout import get_select_options, help_button
from deepcave.utils.styled_plotty import save_image

N_BINS = 4
ELITE_PROPORTION_DEFAULT = 0.10


def _observed_categories(groups: pd.Series) -> List[Any]:
    """Trim unused edge groups while retaining gaps inside the observed range."""
    categories = list(groups.cat.categories)
    used_categories = set(groups.dropna())
    used_indices = [
        index for index, category in enumerate(categories) if category in used_categories
    ]
    if not used_indices:
        return []
    return categories[min(used_indices) : max(used_indices) + 1]


def _group_coordinate(group: Any, hyperparameter: Hyperparameter) -> Any:
    """Return an axis coordinate for an empirical group."""
    if isinstance(group, pd.Interval):
        if getattr(hyperparameter, "log", False):
            return float(np.sqrt(max(group.left, 0) * group.right))
        return float(group.mid)
    if isinstance(hyperparameter, (FloatHyperparameter, IntegerHyperparameter)):
        return float(group)
    return str(group)


def _supports_contour(hyperparameter: Hyperparameter) -> bool:
    """Return whether an axis has meaningful numeric interpolation geometry."""
    return isinstance(hyperparameter, (FloatHyperparameter, IntegerHyperparameter))


def _summarize_pair(
    trials: pd.DataFrame,
    hyperparameter_1: Hyperparameter,
    hyperparameter_2: Hyperparameter,
    optimize: str,
    elite_proportion: float,
) -> Tuple[pd.DataFrame, List[Any], List[Any], float, int, List[Dict[str, Any]]]:
    """Calculate direction-aware trial and joint-bin statistics."""
    trials, elite_threshold, elite_count = _add_trial_statistics(
        trials,
        optimize,
        elite_proportion,
    )
    groups_1 = _group_parameter_values(
        trials["hyperparameter_1"],
        hyperparameter_1,
        n_bins=N_BINS,
        discrete_numeric=True,
    )
    groups_2 = _group_parameter_values(
        trials["hyperparameter_2"],
        hyperparameter_2,
        n_bins=N_BINS,
        discrete_numeric=True,
    )
    categories_1 = _observed_categories(groups_1)
    categories_2 = _observed_categories(groups_2)
    grouped_trials = trials.assign(_group_1=groups_1, _group_2=groups_2)

    rows = []
    grouped = grouped_trials.groupby(
        ["_group_1", "_group_2"],
        observed=True,
        sort=True,
    )
    for (group_1, group_2), cell_trials in grouped:
        cell_elite_count = int(cell_trials["is_elite"].sum())
        elite_ranks = cell_trials["elite_rank"].dropna()
        rows.append(
            {
                "group_1": str(group_1),
                "group_2": str(group_2),
                "n": int(len(cell_trials)),
                "elite": cell_elite_count,
                "elite_rate_pct": float(100 * cell_elite_count / len(cell_trials)),
                "elite_median_rank": (
                    float(elite_ranks.median()) if not elite_ranks.empty else None
                ),
                "objective_median": float(cell_trials["objective"].median()),
            }
        )

    return trials, categories_1, categories_2, elite_threshold, elite_count, rows


def _cell_matrix(
    cells: pd.DataFrame,
    x_labels: List[str],
    y_labels: List[str],
    value: str,
) -> List[List[Any]]:
    """Build a rectangular matrix without filling unobserved joint bins."""
    lookup = {(row["group_1"], row["group_2"]): row[value] for _, row in cells.iterrows()}
    return [[lookup.get((x_label, y_label)) for x_label in x_labels] for y_label in y_labels]


class TwoDimensionalAnalysis(DynamicPlugin):
    """Visualize empirical two-hyperparameter behavior for an Optuna study."""

    id = "two_dimensional_analysis"
    name = "2-D Analyses"
    icon = "fas fa-th"
    help = "plugins/two_dimensional_analysis.html"
    activate_run_selection = True

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """Accept non-empty Optuna studies with at least two hyperparameters."""
        return run.prefix == "Optuna" and not run.empty() and len(run.configspace) >= 2

    @staticmethod
    def get_input_layout(register: Callable) -> List[Any]:
        """Return objective, hyperparameter-pair, and elite controls."""
        return [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Objective"),
                            dcc.Dropdown(
                                id=register("objective_id", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                                clearable=False,
                            ),
                        ],
                        className="col-md-6",
                    ),
                    html.Div(
                        [
                            html.Label("Elite Trial Proportion"),
                            help_button("Fraction of the best plottable trials considered elite."),
                            dcc.Input(
                                id=register("elite_proportion", "value", type=float),
                                type="number",
                                min=0.01,
                                max=0.99,
                                step=0.01,
                            ),
                        ],
                        className="col-md-6",
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Hyperparameter #1"),
                            dcc.Dropdown(
                                id=register(
                                    "hyperparameter_name_1",
                                    ["value", "options"],
                                ),
                                placeholder="Select hyperparameter ...",
                                clearable=False,
                            ),
                        ],
                        className="col-md-6",
                    ),
                    html.Div(
                        [
                            html.Label("Hyperparameter #2"),
                            dcc.Dropdown(
                                id=register(
                                    "hyperparameter_name_2",
                                    ["value", "options"],
                                ),
                                placeholder="Select hyperparameter ...",
                                clearable=False,
                            ),
                        ],
                        className="col-md-6",
                    ),
                ],
                className="row mt-3",
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
            "hyperparameter_name_1": {"options": [], "value": None},
            "hyperparameter_name_2": {"options": [], "value": None},
            "elite_proportion": {"value": ELITE_PROPORTION_DEFAULT},
        }

    def load_dependency_inputs(
        self,
        run: AbstractRun,
        previous_inputs: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Populate study-dependent choices and enforce distinct parameters."""
        objectives = _get_objectives_without_time(run)
        objective_ids = [run.get_objective_id(objective) for objective in objectives]
        objective_value = inputs["objective_id"]["value"]
        if objective_value not in objective_ids:
            objective_value = objective_ids[0]

        hyperparameter_names = list(run.configspace.keys())
        hyperparameter_1 = inputs["hyperparameter_name_1"]["value"]
        if hyperparameter_1 not in hyperparameter_names:
            hyperparameter_1 = hyperparameter_names[0]

        hyperparameter_2_names = [name for name in hyperparameter_names if name != hyperparameter_1]
        hyperparameter_2 = inputs["hyperparameter_name_2"]["value"]
        if hyperparameter_2 not in hyperparameter_2_names:
            hyperparameter_2 = hyperparameter_2_names[0]

        return {
            "objective_id": {
                "options": get_select_options(
                    [objective.name for objective in objectives],
                    objective_ids,
                ),
                "value": objective_value,
            },
            "hyperparameter_name_1": {
                "options": get_select_options(hyperparameter_names),
                "value": hyperparameter_1,
            },
            "hyperparameter_name_2": {
                "options": get_select_options(hyperparameter_2_names),
                "value": hyperparameter_2,
            },
            "elite_proportion": {"value": inputs["elite_proportion"]["value"]},
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Collect trials and calculate empirical joint-bin summaries."""
        objective = run.get_objective(inputs["objective_id"])
        if objective is None:
            raise PreventUpdate

        hyperparameter_name_1 = inputs["hyperparameter_name_1"]
        hyperparameter_name_2 = inputs["hyperparameter_name_2"]
        if hyperparameter_name_1 == hyperparameter_name_2:
            raise ValueError("Select two different hyperparameters.")

        elite_proportion = float(inputs["elite_proportion"])
        if not 0.01 <= elite_proportion <= 0.99:
            raise ValueError("Elite trial proportion must be between 0.01 and 0.99.")

        trials = _collect_trials(
            run,
            inputs["objective_id"],
            [hyperparameter_name_1, hyperparameter_name_2],
        )
        if trials.empty:
            raise ValueError("No successful trials are available for this analysis.")
        trials = trials.rename(
            columns={
                hyperparameter_name_1: "hyperparameter_1",
                hyperparameter_name_2: "hyperparameter_2",
            }
        ).sort_values("trial_number")

        (
            trials,
            categories_1,
            categories_2,
            elite_threshold,
            elite_count,
            cells,
        ) = _summarize_pair(
            trials,
            run.configspace[hyperparameter_name_1],
            run.configspace[hyperparameter_name_2],
            objective.optimize,
            elite_proportion,
        )
        trial_columns = [
            "trial_number",
            "hyperparameter_1",
            "hyperparameter_2",
            "objective",
            "objective_rank",
            "rank_quantile",
        ]

        return {
            "trials": trials[trial_columns].to_dict(orient="records"),
            "cells": cells,
            "axis_1": [
                {
                    "label": str(group),
                    "coordinate": _group_coordinate(
                        group,
                        run.configspace[hyperparameter_name_1],
                    ),
                }
                for group in categories_1
            ],
            "axis_2": [
                {
                    "label": str(group),
                    "coordinate": _group_coordinate(
                        group,
                        run.configspace[hyperparameter_name_2],
                    ),
                }
                for group in categories_2
            ],
            "elite_threshold": elite_threshold,
            "elite_count": elite_count,
        }

    @staticmethod
    def get_output_layout(register: Callable) -> List[Any]:
        """Return the paired overview and full-width empirical objective graphs."""
        graph_config = {
            "toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE},
        }
        return [
            dcc.Graph(
                id=register("overview_graph", "figure"),
                style={"height": config.FIGURE_HEIGHT},
                config=graph_config,
            ),
            dcc.Graph(
                id=register("objective_graph", "figure"),
                style={"height": config.FIGURE_HEIGHT},
                config=graph_config,
            ),
        ]

    @staticmethod
    def load_outputs(
        run: AbstractRun,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> List[go.Figure]:
        """Build the empirical scatter, elite-rate matrix, and objective surface."""
        objective = run.get_objective(inputs["objective_id"])
        if objective is None:
            raise PreventUpdate

        hyperparameter_name_1 = inputs["hyperparameter_name_1"]
        hyperparameter_name_2 = inputs["hyperparameter_name_2"]
        hyperparameter_1 = run.configspace[hyperparameter_name_1]
        hyperparameter_2 = run.configspace[hyperparameter_name_2]
        trials = pd.DataFrame(outputs["trials"]).sort_values("rank_quantile")
        cells = pd.DataFrame(outputs["cells"])
        x_labels = [axis["label"] for axis in outputs["axis_1"]]
        y_labels = [axis["label"] for axis in outputs["axis_2"]]

        overview = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Individual Trials (Unbinned)", "Elite Rate by Joint Bin"),
            horizontal_spacing=0.16,
        )
        overview.add_trace(
            go.Scatter(
                x=trials["hyperparameter_1"],
                y=trials["hyperparameter_2"],
                mode="markers",
                name="Trials",
                marker={
                    "color": trials["rank_quantile"],
                    "colorscale": "RdYlBu",
                    "cmin": 0,
                    "cmax": 1,
                    "showscale": True,
                    "colorbar": {"title": "Rank quantile<br>(1 = best)", "x": 0.43},
                    "size": 7,
                    "opacity": 0.7,
                },
                customdata=trials[["objective", "trial_number", "objective_rank", "rank_quantile"]],
                hovertemplate=(
                    f"{hyperparameter_name_1}: %{{x}}<br>"
                    f"{hyperparameter_name_2}: %{{y}}<br>"
                    f"{objective.name}: %{{customdata[0]}}<br>"
                    "Trial: %{customdata[1]}<br>Global rank: %{customdata[2]:.2f}<br>"
                    "Rank quantile: %{customdata[3]:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        elite_rates = _cell_matrix(cells, x_labels, y_labels, "elite_rate_pct")
        elite_text = []
        elite_customdata = []
        cell_lookup = {(row["group_1"], row["group_2"]): row for _, row in cells.iterrows()}
        for y_label in y_labels:
            text_row = []
            customdata_row = []
            for x_label in x_labels:
                cell = cell_lookup.get((x_label, y_label))
                if cell is None:
                    text_row.append("")
                    customdata_row.append([x_label, y_label, None, None, None])
                    continue
                text_row.append(f"{cell['elite_rate_pct']:.1f}%<br>(n={int(cell['n'])})")
                customdata_row.append(
                    [
                        x_label,
                        y_label,
                        int(cell["n"]),
                        int(cell["elite"]),
                        (
                            f"{cell['elite_median_rank']:.2f}"
                            if pd.notna(cell["elite_median_rank"])
                            else "n/a"
                        ),
                    ]
                )
            elite_text.append(text_row)
            elite_customdata.append(customdata_row)

        overview.add_trace(
            go.Heatmap(
                x=x_labels,
                y=y_labels,
                z=elite_rates,
                colorscale="YlOrRd",
                zmin=0,
                colorbar={"title": "Elite rate [%]", "x": 1.01},
                text=elite_text,
                texttemplate="%{text}",
                customdata=elite_customdata,
                hoverongaps=False,
                hovertemplate=(
                    f"{hyperparameter_name_1} bin: %{{customdata[0]}}<br>"
                    f"{hyperparameter_name_2} bin: %{{customdata[1]}}<br>"
                    "Trials: %{customdata[2]}<br>Elite trials: %{customdata[3]}<br>"
                    "Elite rate: %{z:.2f}%<br>Median elite rank: "
                    "%{customdata[4]}<br>Elite threshold: "
                    f"{outputs['elite_threshold']:.6g}<extra></extra>"
                ),
                showscale=True,
            ),
            row=1,
            col=2,
        )
        overview.update_xaxes(title_text=hyperparameter_name_1, row=1, col=1)
        overview.update_yaxes(title_text=hyperparameter_name_2, row=1, col=1)
        if getattr(hyperparameter_1, "log", False):
            overview.update_xaxes(type="log", row=1, col=1)
        if getattr(hyperparameter_2, "log", False):
            overview.update_yaxes(type="log", row=1, col=1)
        overview.update_xaxes(
            title_text=hyperparameter_name_1,
            categoryorder="array",
            categoryarray=x_labels,
            row=1,
            col=2,
        )
        overview.update_yaxes(
            title_text=hyperparameter_name_2,
            categoryorder="array",
            categoryarray=y_labels,
            row=1,
            col=2,
        )
        overview.update_layout(
            margin=config.FIGURE_MARGIN,
            font={"size": config.FIGURE_FONT_SIZE},
        )
        save_image(overview, "two_dimensional_overview.pdf")

        objective_medians = _cell_matrix(cells, x_labels, y_labels, "objective_median")
        objective_counts = _cell_matrix(cells, x_labels, y_labels, "n")
        objective_customdata = [
            [
                [x_label, y_label, objective_counts[y_index][x_index]]
                for x_index, x_label in enumerate(x_labels)
            ]
            for y_index, y_label in enumerate(y_labels)
        ]
        use_contour = (
            _supports_contour(hyperparameter_1)
            and _supports_contour(hyperparameter_2)
            and len(x_labels) > 1
            and len(y_labels) > 1
        )
        objective_figure = go.Figure()
        if use_contour:
            x_coordinates = [axis["coordinate"] for axis in outputs["axis_1"]]
            y_coordinates = [axis["coordinate"] for axis in outputs["axis_2"]]
            objective_figure.add_trace(
                go.Contour(
                    x=x_coordinates,
                    y=y_coordinates,
                    z=objective_medians,
                    connectgaps=False,
                    colorscale="Viridis",
                    colorbar={"title": objective.name},
                    customdata=objective_customdata,
                    hovertemplate=(
                        f"{hyperparameter_name_1} bin: %{{customdata[0]}}<br>"
                        f"{hyperparameter_name_2} bin: %{{customdata[1]}}<br>"
                        f"Median {objective.name}: %{{z}}<br>"
                        "Trials: %{customdata[2]}<extra></extra>"
                    ),
                )
            )
            objective_figure.update_xaxes(
                tickvals=x_coordinates,
                ticktext=x_labels,
                type="log" if getattr(hyperparameter_1, "log", False) else "linear",
            )
            objective_figure.update_yaxes(
                tickvals=y_coordinates,
                ticktext=y_labels,
                type="log" if getattr(hyperparameter_2, "log", False) else "linear",
            )
            title = "Empirical Objective Median Contour"
        else:
            objective_figure.add_trace(
                go.Heatmap(
                    x=x_labels,
                    y=y_labels,
                    z=objective_medians,
                    colorscale="Viridis",
                    colorbar={"title": objective.name},
                    customdata=objective_customdata,
                    hoverongaps=False,
                    hovertemplate=(
                        f"{hyperparameter_name_1} bin: %{{customdata[0]}}<br>"
                        f"{hyperparameter_name_2} bin: %{{customdata[1]}}<br>"
                        f"Median {objective.name}: %{{z}}<br>"
                        "Trials: %{customdata[2]}<extra></extra>"
                    ),
                )
            )
            title = "Empirical Objective Median by Joint Bin"

        objective_figure.update_layout(
            title=title,
            xaxis={"title": hyperparameter_name_1},
            yaxis={"title": hyperparameter_name_2},
            margin=config.FIGURE_MARGIN,
            font={"size": config.FIGURE_FONT_SIZE},
        )
        save_image(objective_figure, "two_dimensional_objective.pdf")
        return [overview, objective_figure]
