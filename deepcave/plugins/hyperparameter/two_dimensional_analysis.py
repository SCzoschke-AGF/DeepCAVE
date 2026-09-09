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

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import (
    FloatHyperparameter,
    Hyperparameter,
    IntegerHyperparameter,
)
from dash import dcc, html
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import config
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.plugins.hyperparameter.analysis_utils import (
    _add_trial_statistics,
    _collect_trials,
    _get_objectives_without_time,
    _group_parameter_values,
)
from deepcave.runs import AbstractRun
from deepcave.utils.analysis_layout import (
    add_colorbar_label,
    analysis_card,
    analysis_graph,
    analysis_toolbar_item,
    bin_label,
    horizontal_colorbar,
    style_analysis_panel,
)
from deepcave.utils.layout import get_select_options
from deepcave.utils.styled_plotty import save_image

N_BINS = 4
ELITE_PROPORTION_DEFAULT = 0.10
COLOR_SCALE = "Viridis"
PANEL_HEIGHT = 270
PANEL_BOTTOM_MARGIN = 94
ENRICHMENT_TOP_MARGIN = 30
ENRICHMENT_BOTTOM_MARGIN = 60


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
                "elite_enrichment": float(cell_elite_count / len(cell_trials) / elite_proportion),
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


def _style_two_dimensional_panel(figure: go.Figure) -> None:
    """Apply the same compact in-card figure styling to every 2-D panel."""
    style_analysis_panel(
        figure,
        height=PANEL_HEIGHT,
        bottom=PANEL_BOTTOM_MARGIN,
        show_legend=False,
    )


def _observed_trials_figure(
    trials: pd.DataFrame,
    hyperparameter_1: Hyperparameter,
    hyperparameter_2: Hyperparameter,
    hyperparameter_name_1: str,
    hyperparameter_name_2: str,
    objective_name: str,
) -> go.Figure:
    """Build the unbinned trial scatter with rank one on Viridis' bright end."""
    figure = go.Figure(
        go.Scatter(
            x=trials["hyperparameter_1"],
            y=trials["hyperparameter_2"],
            mode="markers",
            marker={
                "color": trials["objective_rank"],
                "colorscale": COLOR_SCALE,
                "reversescale": True,
                "cmin": 1,
                "cmax": float(trials["objective_rank"].max()),
                "showscale": True,
                "colorbar": horizontal_colorbar(),
                "size": 6,
                "opacity": 0.7,
            },
            customdata=trials[["objective", "trial_number", "objective_rank", "rank_quantile"]],
            hovertemplate=(
                f"{hyperparameter_name_1}: %{{x}}<br>"
                f"{hyperparameter_name_2}: %{{y}}<br>"
                f"{objective_name}: %{{customdata[0]}}<br>"
                "Trial: %{customdata[1]}<br>Global rank: %{customdata[2]:.0f}<br>"
                "Rank quantile: %{customdata[3]:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    _style_two_dimensional_panel(figure)
    figure.update_xaxes(
        title_text=hyperparameter_name_1,
        type="log" if getattr(hyperparameter_1, "log", False) else None,
    )
    figure.update_yaxes(
        title_text=hyperparameter_name_2,
        type="log" if getattr(hyperparameter_2, "log", False) else None,
    )
    add_colorbar_label(
        figure,
        "Trial rank (1 = best)",
        PANEL_HEIGHT,
        PANEL_BOTTOM_MARGIN,
    )
    return figure


def _format_enrichment(value: float) -> str:
    """Format an enrichment value compactly while making a true zero unambiguous."""
    if np.isclose(value, 0.0):
        return "0×"
    return f"{value:.2f}×"


def _elite_enrichment_figure(
    cells: pd.DataFrame,
    x_labels: List[str],
    y_labels: List[str],
    hyperparameter_name_1: str,
    hyperparameter_name_2: str,
    elite_threshold: float,
    elite_proportion: float,
) -> go.Figure:
    """Build the joint-bin elite-enrichment heatmap using the existing bins."""
    observed_enrichments = [
        float(cell["elite_rate_pct"]) / 100 / elite_proportion
        for _, cell in cells.iterrows()
        if int(cell["n"]) > 0
    ]
    max_enrichment = max(observed_enrichments, default=1.0)
    color_upper_bound = max(1.0, max_enrichment)
    enrichments: List[List[Any]] = []
    elite_customdata: List[List[List[Any]]] = []
    annotations: List[Dict[str, Any]] = []
    cell_lookup = {(row["group_1"], row["group_2"]): row for _, row in cells.iterrows()}
    for y_label in y_labels:
        enrichment_row: List[Any] = []
        customdata_row: List[List[Any]] = []
        for x_label in x_labels:
            cell = cell_lookup.get((x_label, y_label))
            if cell is None or int(cell["n"]) == 0:
                enrichment_row.append(None)
                customdata_row.append(
                    [x_label, y_label, None, None, 0, elite_proportion * 100, "n/a"]
                )
                annotations.extend(
                    [
                        {
                            "x": x_label,
                            "y": y_label,
                            "text": "No data",
                            "showarrow": False,
                            "yshift": 6,
                            "font": {"size": 11, "color": "#64748b", "weight": 600},
                        },
                        {
                            "x": x_label,
                            "y": y_label,
                            "text": "n = 0",
                            "showarrow": False,
                            "yshift": -8,
                            "font": {"size": 9, "color": "#64748b"},
                        },
                    ]
                )
                continue

            elite_rate_pct = float(cell["elite_rate_pct"])
            enrichment = elite_rate_pct / 100 / elite_proportion
            text_color = "#f8fafc" if enrichment / color_upper_bound < 0.62 else "#172033"
            enrichment_row.append(enrichment)
            customdata_row.append(
                [
                    x_label,
                    y_label,
                    elite_rate_pct,
                    int(cell["elite"]),
                    int(cell["n"]),
                    elite_proportion * 100,
                    (
                        f"{cell['elite_median_rank']:.2f}"
                        if pd.notna(cell["elite_median_rank"])
                        else "n/a"
                    ),
                ]
            )
            annotations.extend(
                [
                    {
                        "x": x_label,
                        "y": y_label,
                        "text": _format_enrichment(enrichment),
                        "showarrow": False,
                        "yshift": 6,
                        "font": {"size": 13, "color": text_color, "weight": 600},
                    },
                    {
                        "x": x_label,
                        "y": y_label,
                        "text": f"n = {int(cell['n'])}",
                        "showarrow": False,
                        "yshift": -8,
                        "font": {"size": 9, "color": text_color},
                    },
                ]
            )
        enrichments.append(enrichment_row)
        elite_customdata.append(customdata_row)

    figure = go.Figure(
        go.Heatmap(
            x=x_labels,
            y=y_labels,
            z=enrichments,
            colorscale=COLOR_SCALE,
            zmin=0,
            zmax=color_upper_bound,
            customdata=elite_customdata,
            hoverongaps=False,
            hovertemplate=(
                f"{hyperparameter_name_1}: %{{customdata[0]}}<br>"
                f"{hyperparameter_name_2}: %{{customdata[1]}}<br>"
                "Elite enrichment: %{z:.2f}×<br>"
                "Elite rate: %{customdata[2]:.1f}%<br>"
                "Elite trials: %{customdata[3]}<br>"
                "Trials in bin: %{customdata[4]}<br>"
                "Expected elite rate: %{customdata[5]:.1f}%<br>"
                "Median elite rank: %{customdata[6]}<br>"
                f"Elite threshold: {elite_threshold:.6g}<extra></extra>"
            ),
            showscale=False,
            xgap=2,
            ygap=2,
        )
    )
    style_analysis_panel(
        figure,
        height=PANEL_HEIGHT,
        bottom=ENRICHMENT_BOTTOM_MARGIN,
        show_legend=False,
        top=ENRICHMENT_TOP_MARGIN,
        left=70,
        right=28,
    )
    figure.update_layout(annotations=annotations, plot_bgcolor="#eef1f5")
    figure.update_xaxes(
        title_text=hyperparameter_name_1,
        tickmode="array",
        tickvals=x_labels,
        ticktext=[bin_label(label, multiline=False) for label in x_labels],
        categoryorder="array",
        categoryarray=x_labels,
        domain=[0.06, 0.94],
        title_standoff=4,
    )
    figure.update_yaxes(
        title_text=hyperparameter_name_2,
        tickmode="array",
        tickvals=y_labels,
        ticktext=[bin_label(label) for label in y_labels],
        categoryorder="array",
        categoryarray=y_labels,
    )
    return figure


def _objective_median_figure(
    cells: pd.DataFrame,
    axes_1: List[Dict[str, Any]],
    axes_2: List[Dict[str, Any]],
    hyperparameter_1: Hyperparameter,
    hyperparameter_2: Hyperparameter,
    hyperparameter_name_1: str,
    hyperparameter_name_2: str,
    objective_name: str,
    optimize: str,
) -> go.Figure:
    """Build the empirical median contour, retaining the categorical heatmap fallback."""
    x_labels = [axis["label"] for axis in axes_1]
    y_labels = [axis["label"] for axis in axes_2]
    objective_medians = _cell_matrix(cells, x_labels, y_labels, "objective_median")
    objective_counts = _cell_matrix(cells, x_labels, y_labels, "n")
    objective_customdata = [
        [
            [x_label, y_label, objective_counts[y_index][x_index]]
            for x_index, x_label in enumerate(x_labels)
        ]
        for y_index, y_label in enumerate(y_labels)
    ]
    trace_arguments = {
        "z": objective_medians,
        "colorscale": COLOR_SCALE,
        "reversescale": optimize == "lower",
        "colorbar": horizontal_colorbar(),
        "customdata": objective_customdata,
        "hovertemplate": (
            f"{hyperparameter_name_1} bin: %{{customdata[0]}}<br>"
            f"{hyperparameter_name_2} bin: %{{customdata[1]}}<br>"
            f"Median {objective_name}: %{{z}}<br>"
            "Trials: %{customdata[2]}<extra></extra>"
        ),
    }
    use_contour = (
        _supports_contour(hyperparameter_1)
        and _supports_contour(hyperparameter_2)
        and len(x_labels) > 1
        and len(y_labels) > 1
    )
    figure = go.Figure()
    if use_contour:
        x_coordinates = [axis["coordinate"] for axis in axes_1]
        y_coordinates = [axis["coordinate"] for axis in axes_2]
        figure.add_trace(
            go.Contour(
                x=x_coordinates,
                y=y_coordinates,
                connectgaps=False,
                **trace_arguments,
            )
        )
        figure.update_xaxes(
            tickmode="array",
            tickvals=x_coordinates,
            ticktext=[bin_label(label) for label in x_labels],
            type="log" if getattr(hyperparameter_1, "log", False) else "linear",
        )
        figure.update_yaxes(
            tickmode="array",
            tickvals=y_coordinates,
            ticktext=[bin_label(label) for label in y_labels],
            type="log" if getattr(hyperparameter_2, "log", False) else "linear",
        )
    else:
        figure.add_trace(
            go.Heatmap(
                x=x_labels,
                y=y_labels,
                hoverongaps=False,
                **trace_arguments,
            )
        )
        figure.update_xaxes(
            tickmode="array",
            tickvals=x_labels,
            ticktext=[bin_label(label) for label in x_labels],
            categoryorder="array",
            categoryarray=x_labels,
        )
        figure.update_yaxes(
            tickmode="array",
            tickvals=y_labels,
            ticktext=[bin_label(label) for label in y_labels],
            categoryorder="array",
            categoryarray=y_labels,
        )

    _style_two_dimensional_panel(figure)
    figure.update_xaxes(title_text=hyperparameter_name_1)
    figure.update_yaxes(title_text=hyperparameter_name_2)
    add_colorbar_label(figure, objective_name, PANEL_HEIGHT, PANEL_BOTTOM_MARGIN)
    return figure


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
    def get_run_input_layout(register: Callable) -> Component:
        """Return the run and analysis controls in the shared compact toolbar."""
        return html.Div(
            [
                analysis_toolbar_item(
                    "Run",
                    dbc.Select(
                        id=register("run", ["value", "options"]),
                        placeholder="Select run ...",
                    ),
                    width="14rem",
                ),
                analysis_toolbar_item(
                    "Objective",
                    dbc.Select(
                        id=register("objective_id", ["value", "options"], type=int),
                        placeholder="Select objective ...",
                    ),
                    width="9rem",
                ),
                analysis_toolbar_item(
                    "Hyperparameter #1",
                    dbc.Select(
                        id=register("hyperparameter_name_1", ["value", "options"]),
                        placeholder="Select hyperparameter ...",
                    ),
                    width="12rem",
                ),
                analysis_toolbar_item(
                    "Hyperparameter #2",
                    dbc.Select(
                        id=register("hyperparameter_name_2", ["value", "options"]),
                        placeholder="Select hyperparameter ...",
                    ),
                    width="12rem",
                ),
                analysis_toolbar_item(
                    "Elite",
                    dbc.Input(
                        id=register("elite_proportion", "value", type=float),
                        type="number",
                        min=0.01,
                        max=0.99,
                        step=0.01,
                    ),
                    width="6rem",
                    help_text=(
                        "Fraction of the best trials considered elite. For example, "
                        "0.10 selects the best 10%."
                    ),
                ),
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

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Return defaults before an Optuna study is selected."""
        return {
            "objective_id": {"options": [], "value": None},
            "hyperparameter_name_1": {"options": [], "value": None},
            "hyperparameter_name_2": {"options": [], "value": None},
            "elite_proportion": {"value": ELITE_PROPORTION_DEFAULT},
            "process_analysis": {"n_clicks": 0},
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
            "process_analysis": {"n_clicks": inputs.get("process_analysis", {}).get("n_clicks", 0)},
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
        """Return the three analysis cards in a responsive two-column grid."""
        return [
            html.Div(
                [
                    analysis_card(
                        "Observed trials",
                        analysis_graph(
                            register("trials_graph", "figure"),
                            PANEL_HEIGHT,
                            config.FIGURE_DOWNLOAD_SCALE,
                        ),
                    ),
                    analysis_card(
                        "Elite enrichement (1x = Baseline)",
                        analysis_graph(
                            register("elite_rate_graph", "figure"),
                            PANEL_HEIGHT,
                            config.FIGURE_DOWNLOAD_SCALE,
                        ),
                    ),
                    analysis_card(
                        "Empirical objective median contour",
                        analysis_graph(
                            register("objective_graph", "figure"),
                            PANEL_HEIGHT,
                            config.FIGURE_DOWNLOAD_SCALE,
                        ),
                    ),
                    html.Div(
                        className="analysis-card analysis-card--empty",
                        **{"aria-hidden": "true"},
                    ),
                ],
                className="analysis-grid analysis-grid--two-dimensional",
            )
        ]

    @staticmethod
    def load_outputs(
        run: AbstractRun,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> List[go.Figure]:
        """Build separate empirical scatter, elite-enrichment, and objective figures."""
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

        trials_figure = _observed_trials_figure(
            trials,
            hyperparameter_1,
            hyperparameter_2,
            hyperparameter_name_1,
            hyperparameter_name_2,
            objective.name,
        )
        elite_enrichment_figure = _elite_enrichment_figure(
            cells,
            x_labels,
            y_labels,
            hyperparameter_name_1,
            hyperparameter_name_2,
            outputs["elite_threshold"],
            float(inputs["elite_proportion"]),
        )
        objective_figure = _objective_median_figure(
            cells,
            outputs["axis_1"],
            outputs["axis_2"],
            hyperparameter_1,
            hyperparameter_2,
            hyperparameter_name_1,
            hyperparameter_name_2,
            objective.name,
            objective.optimize,
        )

        save_image(trials_figure, "two_dimensional_trials.pdf")
        save_image(elite_enrichment_figure, "two_dimensional_elite_enrichment.pdf")
        save_image(objective_figure, "two_dimensional_objective.pdf")
        return [trials_figure, elite_enrichment_figure, objective_figure]
