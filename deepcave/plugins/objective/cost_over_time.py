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

#  noqa: D400
"""
# CostOverTime

This module provides utilities for visualizing the cost over time.

Visualized changes can be regarding to number of configurations or time.
It includes a corresponding plugin class.

## Classes
    - CostOverTime: A plugin to provide a visualization for the cost over time.
"""

from typing import Any, Callable, Dict, List

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from deepcave import config, notification
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun, check_equality
from deepcave.runs.exceptions import NotMergeableError, RunInequality
from deepcave.runs.status import Status
from deepcave.utils.analysis_layout import style_analysis
from deepcave.utils.layout import get_select_options, help_button
from deepcave.utils.styled_plotty import (
    get_color,
    get_hovertext_from_config,
    save_image,
)


class CostOverTime(DynamicPlugin):
    """
    A plugin to provide a visualization for the cost over time.

    Properties
    ----------
    objective_options : List[Dict[str, Any]]
        A list of dictionaries of the objective options.
    budget_options : List[Dict[str, Any]]
        A list of dictionaries of the budget options.
    """

    id = "cost_over_time"
    name = "Cost Over Time"
    icon = "fas fa-chart-line"
    help = "plugins/cost_over_time.html"
    allow_additional_runs = True

    @property
    def optuna_runs(self) -> List[AbstractRun]:
        """Return the loaded Optuna studies available to this plugin."""
        return [run for run in self.runs if run.prefix == "Optuna"]

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """Accept individual Optuna studies as the global basis."""
        return run.prefix == "Optuna"

    def get_additional_run_candidates(self, active_run: AbstractRun) -> List[AbstractRun]:
        """Return compatible individual Optuna studies except for the active study."""
        return [
            run
            for run in self.optuna_runs
            if run.id != active_run.id and self.check_additional_run_compatibility(active_run, run)
        ]

    def check_runs_compatibility(self, runs: List[AbstractRun]) -> None:
        """
        Check if the runs are compatible.

        This function is needed if all selected runs need something in common
        (e.g. budget or objective).
        Since this function is called before the layout is created,
        it can be also used to set common values for the plugin.

        If the runs are not mergeable, they still should be displayed
        but with a corresponding warning message.

        Parameters
        ----------
        runs : List[AbstractRun]
            A list containing the selected runs.

        Raises
        ------
        NotMergeableError
            If the meta data of the runs are not equal.
            If the configuration spaces of the runs are not equal.
            If the budgets of the runs are not equal.
            If the objective of the runs are not equal.
        """
        optuna_runs = [run for run in runs if run.prefix == "Optuna"]
        if not optuna_runs:
            raise NotMergeableError("No Optuna studies are loaded.")

        try:
            check_equality(optuna_runs, objectives=True, budgets=True)
        except NotMergeableError as e:
            run_inequality = e.args[1]
            if run_inequality == RunInequality.INEQ_BUDGET:
                notification.update("The budgets of the runs are not equal.", color="warning")
            elif run_inequality == RunInequality.INEQ_CONFIGSPACE:
                notification.update(
                    "The configuration spaces of the runs are not equal.", color="warning"
                )
            elif run_inequality == RunInequality.INEQ_META:
                notification.update("The meta data of the runs is not equal.", color="warning")
            elif run_inequality == RunInequality.INEQ_OBJECTIVE:
                raise NotMergeableError("The objectives of the selected runs cannot be merged.")

        # Set some attributes here
        # It is necessary to get the run with the smallest budget and objective options
        # as first comparative value, else there is gonna be an index problem
        objective_options = []
        budget_options = []
        for run in optuna_runs:
            objective_names = run.get_objective_names()
            objective_ids = run.get_objective_ids()
            objective_options.append(get_select_options(objective_names, objective_ids))

            budgets = run.get_budgets(human=True)
            budget_ids = run.get_budget_ids()
            budget_options.append(get_select_options(budgets, budget_ids))
        self.objective_options = min(objective_options, key=len)
        self.budget_options = min(budget_options, key=len)

    @staticmethod
    def get_input_layout(register: Callable) -> List[dbc.Row]:
        """
        Get the layout for the input block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[dbc.Row]
            Layouts for the input block.
        """
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Objective"),
                            dbc.Select(
                                id=register("objective_id", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Budget"),
                            help_button(
                                "Budget refers to the multi-fidelity budget. "
                                "Combined budget means that the trial on the highest evaluated "
                                "budget is used."
                            ),
                            dbc.Select(
                                id=register("budget_id", ["value", "options"], type=int),
                                placeholder="Select budget ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
            ),
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Any]:
        """
        Get the layout for the filter block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            Layouts for the filter block.
        """
        return [
            html.Div(
                [
                    dbc.Label("X-Axis"),
                    dbc.Select(
                        id=register("xaxis", ["value", "options"]),
                        placeholder="Select ...",
                    ),
                ],
                className="mb-3",
            ),
        ]

    def load_inputs(self) -> Dict[str, Any]:
        """
        Load the content for the defined inputs in 'get_input_layout' and 'get_filter_layout'.

        This method is necessary to pre-load contents for the inputs.
        So, if the plugin is called for the first time or there are no results in the cache,
        the plugin gets its content from this method.

        Returns
        -------
        Dict[str, Any]
            The content to be filled.
        """
        return {
            "objective_id": {
                "options": self.objective_options,
                "value": self.objective_options[0]["value"],
            },
            "budget_id": {
                "options": self.budget_options,
                "value": (
                    self.budget_options[0]["value"]
                    if len(self.budget_options) == 1
                    else self.budget_options[-2]["value"]
                ),
            },
            "xaxis": {
                "options": [
                    {"label": "Time", "value": "times"},
                    {"label": "Logarithmic Time", "value": "times_log"},
                    {"label": "Evaluated trials", "value": "trials"},
                ],
                "value": "times",
            },
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, Any]:  # type: ignore
        """
        Return raw data based on a run and input data.

        Warning
        -------
        The returned data must be JSON serializable.

        Note
        ----
        The passed inputs are cleaned and therefore differs compared to 'load_inputs'
        or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run : AbstractRun
            The selected run to process.
        inputs : Dict[str, Any]
            The input data.

        Returns
        -------
        Dict[str, Any]
            A serialized dictionary.
        """
        budget = run.get_budget(inputs["budget_id"])
        objective = run.get_objective(inputs["objective_id"])

        times, costs_mean, costs_std, ids, config_ids = run.get_trajectory(
            objective=objective, budget=budget
        )
        objective_id = run.get_objective_id(objective)
        objective_values = [
            float(trial.costs[objective_id])
            for trial in run.history
            if trial.status == Status.SUCCESS
            and trial.budget == budget
            and trial.costs[objective_id] is not None
            and np.isfinite(trial.costs[objective_id])
        ]

        return {
            "times": times,
            "costs_mean": costs_mean,
            "costs_std": costs_std,
            "ids": ids,
            "config_ids": config_ids,
            "objective_values": objective_values,
        }

    @staticmethod
    def get_output_layout(register: Callable) -> dcc.Graph:
        """
        Get the layout for the output block.

        Parameters
        ----------
        register : Callable
            Method to register outputs.
            The register_output function is located in the Plugin superclass.

        Returns
        -------
        dcc.Graph
            The layouts for the output block.
        """
        return dcc.Graph(
            register("graph", "figure"),
            responsive=True,
            style={"width": "100%", "minHeight": "500px"},
            config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
        )

    @staticmethod
    def load_outputs(runs, inputs, outputs) -> go.Figure:  # type: ignore
        """
        Read in the raw data and prepare them for the layout.

        Note
        ----
        The passed inputs are cleaned and therefore differs compared to 'load_inputs'
        or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        runs :
            The selected runs.
        inputs :
            The input and filter values from the user.
        outputs :
            The raw outputs from the runs.

        Returns
        -------
        go.Figure
            The output figure.
        """
        objective = None

        if not runs:
            return go.Figure()

        figure = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.45, 0.55],
            subplot_titles=("Objective Distribution", "Running Best"),
            horizontal_spacing=0.1,
        )
        for idx, run in enumerate(runs):
            objective = run.get_objective(inputs["objective_id"])
            ids = outputs[run.id]["ids"]
            x = outputs[run.id]["times"]
            if inputs["xaxis"] == "trials":
                x = outputs[run.id]["ids"]

            y = np.array(outputs[run.id]["costs_mean"])
            y_err = np.array(outputs[run.id]["costs_std"])
            y_upper = list(y + y_err)
            y_lower = list(y - y_err)
            y_list = list(y)

            hovertext = None
            hoverinfo = "skip"
            symbol = None
            mode = "lines"
            if len(run.history) > 0:
                hovertext = [
                    get_hovertext_from_config(run, trial.config_id, trial.budget)
                    for id, trial in enumerate(run.history)
                    if id in ids
                ]
                hoverinfo = "text"
                symbol = "circle"
                mode = "lines+markers"

            color = get_color(idx)
            objective_values = outputs[run.id]["objective_values"]
            figure.add_trace(
                go.Histogram(
                    x=objective_values,
                    name=run.name,
                    marker_color=color,
                    opacity=0.55,
                    nbinsx=40,
                    bingroup="objective-distribution",
                    legendgroup=run.id,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            if objective_values:
                best_value = (
                    max(objective_values)
                    if objective.optimize == "upper"
                    else min(objective_values)
                )
                figure.add_vline(
                    x=best_value,
                    line_dash="dash",
                    line_color=color,
                    row=1,
                    col=1,
                )

            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=y_list,
                    name=run.name,
                    line_shape="hv",
                    line=dict(color=color),
                    hovertext=hovertext,
                    hoverinfo=hoverinfo,
                    marker=dict(symbol=symbol),
                    mode=mode,
                    legendgroup=run.id,
                    showlegend=True,
                ),
                row=1,
                col=2,
            )

            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=y_upper,
                    line=dict(color=get_color(idx, 0)),
                    line_shape="hv",
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=run.id,
                    marker=dict(symbol=None),
                ),
                row=1,
                col=2,
            )

            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=y_lower,
                    fill="tonexty",
                    fillcolor=get_color(idx, 0.2),
                    line=dict(color=get_color(idx, 0)),
                    line_shape="hv",
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=run.id,
                    marker=dict(symbol=None),
                ),
                row=1,
                col=2,
            )

        if objective is None:
            raise PreventUpdate

        type = None
        if inputs["xaxis"] == "times_log":
            type = "log"

        xaxis_label = "Wallclock time [s]"
        if inputs["xaxis"] == "trials":
            xaxis_label = "Number of evaluated trials"

        figure.update_xaxes(title_text=objective.name, row=1, col=1)
        figure.update_yaxes(title_text="Trials", row=1, col=1)
        figure.update_xaxes(title_text=xaxis_label, type=type, row=1, col=2)
        figure.update_yaxes(title_text=objective.name, row=1, col=2)
        figure.update_layout(
            barmode="overlay",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.3,
                xanchor="center",
                x=0.5,
                groupclick="togglegroup",
            ),
            margin={**config.FIGURE_MARGIN, "b": 120},
            font=dict(size=config.FIGURE_FONT_SIZE),
        )
        style_analysis(figure, height=520 + 24 * max(0, len(runs) - 2))
        save_image(figure, "cost_over_time.pdf")

        return figure
