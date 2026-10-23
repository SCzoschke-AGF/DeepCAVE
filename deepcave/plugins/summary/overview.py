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

import math
import re
from numbers import Integral, Real

import dash_bootstrap_components as dbc
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    NumericalHyperparameter,
    OrdinalHyperparameter,
)
from dash import html

from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs.group import Group
from deepcave.runs.status import Status
from deepcave.utils.layout import create_table
from deepcave.utils.util import get_latest_change


def _format_value(value: Any) -> str:
    """Round display values only; retain the stored precision in cell titles."""
    if value is None:
        return "—"
    if isinstance(value, Integral):
        return str(value)
    if isinstance(value, Real):
        return f"{value:.5g}" if math.isfinite(value) else "—"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def _value(value: Any, label: Any = None) -> html.Span:
    """Keep full text available on hover and to assistive technology."""
    return html.Span(
        _format_value(value) if label is None else label,
        title=str(value),
        className="run-metadata-value",
    )


def _objective_name(name: str) -> str:
    """Expand recognizable precision metric names without guessing other meanings."""
    match = re.fullmatch(r"(?:user_attr[:/])?precision_at_recall(\d+)(?:_(mean|std))?", name)
    if match:
        suffix = f" ({match[2]})" if match[2] else ""
        return f"Precision @ Recall {match[1]}%{suffix}"
    return name


def _table(data: Dict[str, List[Any]], modifier: str = "") -> html.Div:
    """Use compact tables inside a locally scrollable container."""
    table = create_table(data, striped=False, mb=False)
    table.className += f" run-metadata-table {modifier}"
    table.bordered = False
    return html.Div(table, className="run-metadata-table-wrap")


def _section(title: str, icon: str, content: Any) -> dbc.Card:
    """Share the General page's card headings, spacing, and icon treatment."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H3(
                    [
                        html.I(
                            className=f"fas fa-{icon} general-card-icon", **{"aria-hidden": "true"}
                        ),
                        title,
                    ],
                    className="general-card-title run-metadata-heading",
                ),
                content,
            ]
        ),
        className="general-management-card run-metadata-card",
    )


def _kpi(label: str, value: Any, icon: str, tone: str = "", title: str = "") -> html.Div:
    """Render one responsive KPI with a restrained status accent."""
    return html.Div(
        [
            html.I(className=f"fas fa-{icon} general-card-icon", **{"aria-hidden": "true"}),
            html.Div(
                [html.Div(label, className="run-metadata-kpi-label"), html.Strong(value)],
                className="run-metadata-kpi-content",
            ),
        ],
        className=f"run-metadata-kpi {tone}",
        title=title,
    )


class Overview(DynamicPlugin):
    """Visualize an overall overview of the selected run."""

    id = "overview"
    name = "Run Metadata"
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
                [
                    html.Div(id=register("card", "children"), className="run-metadata-kpis"),
                    html.Div(
                        [
                            _section("Meta", "file-alt", html.Div(id=register("meta", "children"))),
                            _section(
                                "Objectives",
                                "bullseye",
                                html.Div(id=register("objectives", "children")),
                            ),
                        ],
                        className="run-metadata-grid",
                    ),
                    html.Div(
                        [
                            _section(
                                "Failed Trials",
                                "exclamation-triangle",
                                html.Div(id=register("status_details", "children")),
                            ),
                            _section(
                                "Configuration Space",
                                "sliders-h",
                                html.Div(id=register("configspace", "children")),
                            ),
                        ],
                        className="run-metadata-grid",
                    ),
                ],
                className="run-metadata",
            )
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

        # Meta
        meta: Dict[str, List[Any]] = {
            "Attribute": ["Optimizer", "Latest Change"],
            "Value": [
                _value(optimizer),
                _value(run.latest_change, get_latest_change(run.latest_change)),
            ],
        }
        labels = {
            "model_family": "Model Family",
            "optuna_sampler": "Sampler",
            "optuna_study_name": "Study Name",
        }
        for k, v in run.get_meta().items():
            if k in {"objectives", "budgets", "seeds"}:
                continue
            meta["Attribute"].append(_value(k, labels.get(k, k.replace("_", " ").title())))
            meta["Value"].append(_value(v))

        # Objectives
        objectives: Dict[str, List[Any]] = {"Objective": [], "Min": [], "Max": [], "Best": []}
        best_values = []
        for idx, objective in enumerate(run.get_objectives()):
            # Preserve the existing incumbent and highest-budget averaging semantics.
            try:
                incumbent, _ = run.get_incumbent(objectives=objective, statuses=[Status.SUCCESS])
            except RuntimeError:
                best = _value(None, "Unavailable")
            else:
                config_id = run.get_config_id(incumbent)
                avg_costs, std_costs = run.get_avg_costs(config_id)
                best = _value(avg_costs[idx])
                if len(run.get_seeds(include_combined=False)) > 1:
                    best = _value(
                        f"{avg_costs[idx]} ± {std_costs[idx]}",
                        f"{_format_value(avg_costs[idx])} ± {_format_value(std_costs[idx])}",
                    )
            best_values.append(best)
            objectives["Objective"].append(_value(objective.name, _objective_name(objective.name)))
            objectives["Min"].append(_value(objective.lower))
            objectives["Max"].append(_value(objective.upper))
            objectives["Best"].append(best)

        # Setup details dict for to collect information on failed trials
        status_details: Dict[str, List[Any]] = {
            "Configuration ID": [],
            "Status": [],
            "Error": [],
        }

        status_count = {}
        len_trials = 0
        for trial in run.get_trials():
            len_trials += 1

            # Total status count for text information
            if trial.status not in status_count:
                status_count[trial.status] = 1
            else:
                status_count[trial.status] += 1

            # Add to table data
            if trial.status != Status.SUCCESS:
                status_details["Configuration ID"] += [trial.config_id]
                status_details["Status"] += [
                    html.Span(trial.status.to_text(), className="run-metadata-status")
                ]

                if "traceback" in trial.additional:
                    traceback = trial.additional["traceback"]
                    status_details["Error"] += [
                        html.Details(
                            [html.Summary("Show traceback"), html.Pre(str(traceback))],
                            className="run-metadata-error",
                        )
                    ]
                else:
                    status_details["Error"] += ["No traceback available."]

        successful_trials = status_count.get(Status.SUCCESS, 0)
        failed_trials = len_trials - successful_trials
        runtime = max((trial.end_time for trial in run.history), default=0)
        if runtime >= 3600:
            runtime_label = f"{runtime / 3600:,.1f} h"
        elif runtime >= 60:
            runtime_label = f"{runtime / 60:,.1f} min"
        else:
            runtime_label = f"{runtime:,.1f} s"
        card = [
            _kpi(
                "Best Objective",
                best_values[0] if best_values else "Unavailable",
                "trophy",
                title=run.get_objectives()[0].name if best_values else "No objectives available",
            ),
            _kpi("Total Trials", f"{len_trials:,}", "clipboard-list"),
            _kpi(
                "Successful Trials",
                f"{successful_trials:,}",
                "check-circle",
                "run-metadata-kpi--success",
            ),
            _kpi("Failed Trials", f"{failed_trials:,}", "times-circle", "run-metadata-kpi--failed"),
            _kpi(
                "Maximum Runtime" if isinstance(run, Group) else "Total Runtime",
                runtime_label,
                "clock",
                title=f"{runtime} seconds",
            ),
        ]
        if run.get_num_configs() != len_trials:
            card.append(_kpi("Total Configurations", f"{run.get_num_configs():,}", "database"))

        # Prepare configspace table
        configspace: Dict[str, List] = {
            "Hyperparameter": [],
            "Possible Values": [],
            "Default": [],
            "Log": [],
        }

        for hp_name, hp in run.configspace.items():
            log = False
            value = _value(str(hp))
            if isinstance(hp, NumericalHyperparameter):
                value = _value(
                    [hp.lower, hp.upper], f"[{_format_value(hp.lower)}, {_format_value(hp.upper)}]"
                )
                log = hp.log
            elif isinstance(hp, CategoricalHyperparameter):
                value = _value(hp.choices)
            elif isinstance(hp, OrdinalHyperparameter):
                value = _value(hp.sequence)
            elif isinstance(hp, Constant):
                value = _value(hp.value)

            configspace["Hyperparameter"].append(_value(hp_name))
            configspace["Possible Values"].append(value)
            configspace["Default"].append(_value(hp.default_value))
            configspace["Log"].append(html.Span("Yes" if log else "No", className="text-muted"))

        space_details = [_table(configspace, "run-metadata-space")]
        for label, clauses in (
            ("Conditions", run.configspace.conditions),
            ("Forbidden Clauses", run.configspace.forbidden_clauses),
        ):
            if clauses:
                space_details.append(
                    html.Details(
                        [
                            html.Summary(label),
                            html.Ul([html.Li(str(clause)) for clause in clauses]),
                        ],
                        className="run-metadata-space-details",
                    )
                )

        return [
            card,
            _table(meta, "run-metadata-meta"),
            _table(objectives, "run-metadata-objectives"),
            (
                _table(status_details, "run-metadata-failures")
                if failed_trials
                else html.Div(
                    [
                        html.I(className="fas fa-check-circle me-2", **{"aria-hidden": "true"}),
                        "No failed trials.",
                    ],
                    className="run-metadata-empty",
                )
            ),
            space_details,
        ]
