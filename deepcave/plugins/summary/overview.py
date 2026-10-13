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
"""
# Overview

This module provides utilities for visualizing an overview of the selected runs.

It holds the most important information, e.g. meta data, objectives and statistics.

The module includes a dynamic plugin for the overview.

## Classes
    - Overview: Visualize an overall overview of the selected run.
"""

from typing import Any, Callable, Dict, List

import dash_bootstrap_components as dbc
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from dash import html

from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs.group import Group
from deepcave.runs.status import Status
from deepcave.utils.layout import create_table, help_button
from deepcave.utils.util import custom_round, get_latest_change


class Overview(DynamicPlugin):
    """Visualize an overall overview of the selected run."""

    id = "overview"
    name = "Run Overview"
    icon = "fas fa-search"
    help = "plugins/overview.html"
    use_cache = False
    activate_run_selection = True

    @staticmethod
    def get_output_layout(register: Callable) -> List[Any]:
        """
        Get the layout for the output block.

        Parameters
        ----------
        register : Callable
            Method to register the outputs.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            The layouts for the output block.
        """
        return [
            html.Div(
                id=register("card", "children"),
                className="mb-3",
            ),
            html.Hr(),
            html.H3("Meta"),
            html.Div(id=register("meta", "children")),
            html.Hr(),
            html.H3("Objectives"),
            html.Div(id=register("objectives", "children")),
            html.Hr(),
            html.H3("Statuses"),
            html.Div(id=register("status_text", "children"), className="mb-3"),
            html.H4("Details"),
            html.Div(id=register("status_details", "children")),
            html.Hr(),
            html.H3("Configuration Space"),
            html.Div(id=register("configspace", "children")),
        ]

    @staticmethod
    def load_outputs(run, *_: Any) -> List[Any]:  # type: ignore
        """
        Read in the raw data and prepare them for the layout.

        Note
        ----
        The passed inputs are cleaned and therefore differs compared to 'load_inputs'
        or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run
            The selected run.

        Returns
        -------
        List[Any]
            A list of the created tables of the overview.
        """
        optimizer = run.prefix
        if isinstance(run, Group):
            optimizer = run.get_runs()[0].prefix

        performance_outputs = []
        for idx, obj in enumerate(run.get_objectives()):
            # Get best cost for the objective, highest budget
            incumbent, _ = run.get_incumbent(objectives=obj, statuses=[Status.SUCCESS])
            config_id = run.get_config_id(incumbent)
            avg_costs, std_costs = run.get_avg_costs(config_id)

            if len(run.get_seeds(include_combined=False)) > 1:
                best_performance = (
                    f"{custom_round(avg_costs[idx])} " f"± {custom_round(std_costs[idx])}"
                )
            else:
                best_performance = f"{custom_round(avg_costs[idx])}"

            performance_outputs.append(
                html.Div(
                    html.Span(f"Best {obj.name}: {best_performance}"),
                    className="card-text",
                ),
            )

        if isinstance(run, Group):
            runtime_str = "Maximum runtime"
        else:
            runtime_str = "Total runtime"

        # Design card for quick information here
        card = dbc.Card(
            [
                dbc.CardHeader(html.H3("Quick Information", className="mb-0")),
                dbc.CardBody(
                    [
                        html.Div(
                            f"Optimizer: {optimizer}",
                            className="card-text",
                        ),
                        html.Div(
                            f"Latest change: {get_latest_change(run.latest_change)}",
                            className="card-text",
                        ),
                        *performance_outputs,
                        html.Div(
                            [
                                html.Span(
                                    f"{runtime_str} [s]: "
                                    f"{max(trial.end_time for trial in run.history)}"
                                ),
                            ],
                            className="card-text",
                        ),
                        html.Div(
                            [
                                html.Span(f"Total configurations: {run.get_num_configs()}"),
                            ],
                            className="card-text",
                        ),
                        html.Div(
                            [
                                html.Span(f"Total trials: {len(run.history)}"),
                            ],
                            className="card-text",
                        ),
                    ]
                ),
            ],
            color="secondary",
            inverse=True,
        )

        # Meta
        meta: Dict[str, List[str]] = {"Attribute": [], "Value": []}
        for k, v in run.get_meta().items():
            if k == "objectives":
                continue

            if isinstance(v, list):
                v = ", ".join(str(_v) for _v in v)

            meta["Attribute"].append(k)
            meta["Value"].append(str(v))

        # Objectives
        objectives: Dict[str, List[str]] = {"Name": [], "Bounds": []}
        for objective in run.get_objectives():
            objectives["Name"].append(objective.name)
            objectives["Bounds"].append(f"[{objective.lower}, {objective.upper}]")

        # Setup details dict for to collect information on failed trials
        status_details: Dict[str, List[Any]] = {
            "Configuration ID": [],
            "Budget": [],
            "Seed": [],
            "Status": [],
            "Error": [],
        }

        status_count = {}
        len_trials = 0
        for trial in run.get_trials():
            budget = round(trial.budget, 2)
            seed = trial.seed

            len_trials += 1

            # Total status count for text information
            if trial.status not in status_count:
                status_count[trial.status] = 1
            else:
                status_count[trial.status] += 1

            # Add to table data
            if trial.status != Status.SUCCESS:
                status_details["Configuration ID"] += [trial.config_id]
                status_details["Budget"] += [budget]
                status_details["Seed"] += [seed]
                status_details["Status"] += [trial.status.to_text()]

                if "traceback" in trial.additional:
                    traceback = trial.additional["traceback"]
                    status_details["Error"] += [help_button(traceback)]
                else:
                    status_details["Error"] += ["No traceback available."]

        successful_trials = status_count.get(Status.SUCCESS, 0)
        failed_trials = len_trials - successful_trials
        status_text = (
            f"Successful trials: {successful_trials} · Failed trials: {failed_trials}"
        )

        # Prepare configspace table
        configspace: Dict[str, List] = {
            "Hyperparameter": [],
            "Possible Values": [],
            "Default": [],
            "Log": [],
        }

        for hp_name, hp in run.configspace.get_hyperparameters_dict().items():
            log = False
            value = None
            if (
                isinstance(hp, UniformIntegerHyperparameter)
                or isinstance(hp, NormalIntegerHyperparameter)
                or isinstance(hp, UniformFloatHyperparameter)
                or isinstance(hp, NormalFloatHyperparameter)
            ):
                value = str([hp.lower, hp.upper])
                log = hp.log
            elif isinstance(hp, CategoricalHyperparameter):
                value = ", ".join([str(i) for i in hp.choices])
            elif isinstance(hp, OrdinalHyperparameter):
                value = ", ".join([str(i) for i in hp.sequence])
            elif isinstance(hp, Constant):
                value = str(hp.value)

            default = str(hp.default_value)
            log_str = str(log)

            configspace["Hyperparameter"].append(hp_name)
            configspace["Possible Values"].append(value)
            configspace["Default"].append(default)
            configspace["Log"].append(log_str)

        return [
            card,
            create_table(meta, fixed=True),
            create_table(objectives, fixed=True),
            status_text,
            create_table(status_details),
            create_table(configspace, mb=False),
        ]
