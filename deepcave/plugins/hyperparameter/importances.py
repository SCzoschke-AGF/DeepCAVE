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
# Importances

This module provides a plugin for the visualization of the importances.

Provided utilities include getting input and output layout (filtered or non-filtered),
processing the data and loading the outputs. Also provides a matplotlib version.

## Classes
    - Importances: This class provides a plugin for the visualization of the importances.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import itertools
import json
from copy import copy
from io import StringIO
from textwrap import dedent

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from ConfigSpace import ConfigurationSpace, Constant
from dash import Input, Output, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate

from deepcave import config, interactive
from deepcave.evaluators.fanova import fANOVA as GlobalEvaluator
from deepcave.evaluators.lpi import LPI as LocalEvaluator
from deepcave.evaluators.mo_fanova import MOfANOVA
from deepcave.evaluators.mo_lpi import MOLPI
from deepcave.evaluators.ped_anova import PEDANOVA
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status
from deepcave.utils.analysis_layout import style_analysis
from deepcave.utils.cast import optional_int
from deepcave.utils.interaction_screening import get_cached_fanova, save_cached_fanova
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_color, save_image

logger = get_logger(__name__)

MAX_INTERACTION_HYPERPARAMETERS = 4


def _importance_method_guide() -> html.Details:
    """Explain the methods and interaction screening below the analysis results."""
    return html.Details(
        [
            html.Summary(
                "How are importances calculated?",
                style={"cursor": "pointer", "fontWeight": "600"},
            ),
            dcc.Markdown(
                dedent(r"""
                    #### fANOVA — global importance

                    A random forest learns to predict the selected objective across the search
                    space. fANOVA decomposes each tree's prediction variance into individual
                    parameter effects and interactions, then averages the variance fractions
                    across trees. A main-effect score of 0.20 means that this parameter alone
                    accounts for about 20% of the modeled variance under the integration measure.
                    Main effects need not sum to one: interactions can explain the remainder.
                    These are effects in the fitted model, not evidence of causality.

                    #### LPI — local parameter importance

                    LPI uses a random forest to vary one parameter at a time around the best
                    observed configuration (Incumbent) while holding the other settings fixed.
                    Parameters whose changes produce more prediction variance receive higher
                    importance. The variances are normalized across parameters within each tree
                    and then averaged. This explains sensitivity near the incumbent, so its ranking
                    can differ from global fANOVA. LPI does not separately estimate
                    pair interactions.

                    #### PED-ANOVA — importance in the top-performing region

                    PED-ANOVA compares estimated parameter distributions in the best trials with
                    their distributions across the observed search. The target quantile selects
                    that top-performing region: 0.10 corresponds to approximately the best 10%.
                    A parameter receives a high score when its distribution differs strongly
                    between these regions. Scores are normalized to sum to one. This method uses
                    density estimates rather than a random forest and does not calculate pair
                    interactions here. Its scores have a different meaning from fANOVA fractions.

                    #### How fANOVA pair interactions are determined

                    An interaction is the additional joint effect of two parameters after their
                    individual effects have been removed. Two parameters can have weak main
                    effects and still interact strongly. With **Interaction order 2**, choose:

                    - **Selected parameters:** calculate exact pair interactions among the
                      selected two to four parameters.
                    - **Screen all pairs:** rank every pair cheaply, refine promising or
                      numerically uncertain candidates, then calculate selected pairs exactly.

                    The screening follows these steps:

                    1. **Reuse one fitted forest.** Screening and exact calculations use the same
                       trees, parameter encoding, bounds and integration measure. Logarithmic
                       parameters are integrated in their encoded space; categories use counting
                       weights. Parameters are never excluded because of small main effects.
                    2. **Separate the pure interaction in each tree.** For tree $t$, marginalize
                       over all other parameters to obtain the joint surface $a_{jk,t}$, the
                       individual surfaces $a_{j,t}$ and $a_{k,t}$, and the overall mean
                       $f_{\emptyset,t}$:

                       $$f_{jk,t}(a,b)=a_{jk,t}(a,b)-a_{j,t}(a)-a_{k,t}(b)+f_{\emptyset,t}.$$

                       The constant and one-parameter marginal predictions are exact for the
                       fitted tree. Approximation enters through the two-dimensional integration.
                    3. **Screen every pair on an 8 × 8 grid.** Estimate the weighted mean square
                       of this residual under the integration measure $Q_{jk}$ and divide by the
                       tree's exact total variance $V_t$:

                       $$I_{jk,t}\approx
                       \frac{\mathbb{E}_{Q_{jk}}[f_{jk,t}(a,b)^2]}{V_t}.$$

                       Average these fractions over trees with finite positive variance.
                       Squaring happens before averaging across trees. The residual mean is not
                       subtracted: mean square preserves the fANOVA definition even when a coarse
                       grid leaves a numerical offset.
                    4. **Refine and check the ranking.** Leading pairs, pairs near the selection
                       boundary and pairs with a large offset are evaluated on a 16 × 16 grid.
                       Up to two lower-ranked pairs are also checked with a shifted coarse grid;
                       a substantial change promotes them to refinement. Newly promoted leaders
                       and pairs near the updated boundary are refined as well.
                    5. **Calculate selected pairs exactly.** **Exact top pairs** selects how many
                       leading screening pairs to calculate exactly (default 5; 0 means screening
                       only). Click a pair in the table or use **Additional exact pairs**, then
                       click **Process**. Completed pairs disappear from the picker; their exact
                       results remain in the table and plot.
                       The forest and computed marginals are cached and reused. Exact means
                       integration over the fitted trees' partitions, not exact knowledge of the
                       underlying objective. Only main effects and exact pair results enter the
                       bar plot; approximate scores remain in the screening table.

                    ##### Reading the screening diagnostics

                    - **Approx. importance:** the 16 × 16 estimate when available, otherwise the
                      8 × 8 estimate. **Exact importance** is blank until that pair is calculated
                      exactly; a blank cell does not mean zero importance.
                    - **Offset:** the mean across trees of
                      $(\mathbb{E}_{Q_{jk}}[f_{jk,t}])^2/V_t$. The residual mean should be zero
                      under exact integration; a larger value indicates integration error.
                    - **Resolution Δ:** the absolute difference between the 8 × 8 and 16 × 16
                      scores. **Shifted-grid Δ** is the absolute difference from the coarse-grid
                      audit. Blank diagnostics indicate that the corresponding check was not run.
                    - **Exact:** an exact result for the fitted forest is available. This status
                      takes precedence over the screening status, including **Refine recommended**.
                      The grid estimates and their diagnostics remain visible for comparison.
                    - **Stable:** the resolution difference is at most the larger of 0.002 and
                      20% of the larger score, the normalized offset is at most 0.001, and the
                      score is at most 1.002. Otherwise a refined pair is marked
                      **Refine recommended**. **Coarse** and **Audited coarse** have not received
                      the 16 × 16 check.

                    Stability describes numerical sensitivity, not statistical confidence.
                    Narrow interactions can fall between grid points, and closely ranked pairs
                    can change order after exact calculation. Increasing **Exact top pairs** or
                    checking additional pairs reduces reliance on the approximate shortlist.
                    """),
                mathjax=True,
                className="mt-3",
                style={"maxWidth": "1000px", "lineHeight": "1.6"},
            ),
        ],
        id="importances-method-guide",
        open=False,
        className="mt-4 p-3 border rounded",
    )


def _serialize_global_importances(
    importances: Dict[Union[str, Tuple[str, ...]], Tuple[float, float, float, float]],
    order: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convert fANOVA's tuple-keyed result into JSON-safe effect records."""
    records = []
    for hp_names, values in importances.items():
        members = [hp_names] if isinstance(hp_names, str) else list(hp_names)
        if order is not None and len(members) != order:
            continue

        records.append(
            {
                "members": members,
                "order": len(members),
                "importance": float(values[0]),
                "variance": float(values[1]),
                "total_importance": float(values[2]),
                "total_variance": float(values[3]),
            }
        )

    return records


class Importances(StaticPlugin):
    """
    Provide a plugin for the visualization of the importances.

    Evaluators are fANOVA, LPI (local parameter importance), and PED-ANOVA.

    Provided utilities include getting input/output layout, data processing
    and loading outputs. Also provides a matplotlib version.
    """

    id = "importances"
    processing_version = 3
    name = "Importances"
    icon = "far fa-star"
    help = "plugins/importances.html"
    activate_run_selection = True
    main_objective_input = "objective_id1"
    input_update_inputs = (
        ("screening_cell", "importances-screening-output", "active_cell"),
        ("screening_ready", "importances-screening_ready-output", "data"),
    )

    @interactive
    def register_callbacks(self) -> None:
        """Show method-specific fields immediately when the method or order changes."""
        super().register_callbacks()
        from deepcave import app

        @app.callback(
            [
                Output("importances-trees-field", "style"),
                Output("importances-quantile-field", "style"),
                Output("importances-interactions-field", "style"),
                Output("importances-interaction-hps-field", "style"),
                Output("importances-interaction-mode-field", "style"),
                Output("importances-screening-fields", "style"),
            ],
            [
                Input(self.get_internal_input_id("method"), "value"),
                Input(self.get_internal_input_id("interaction_order"), "value"),
                Input(self.get_internal_input_id("interaction_mode"), "value"),
            ],
        )
        def update_method_fields(method: str, order: Any, mode: str) -> List[Dict[str, str]]:
            pairwise = method == "global" and order in (2, "2")
            return [
                {"display": "none"} if method == "ped_anova" else {},
                {} if method == "ped_anova" else {"display": "none"},
                {} if method == "global" else {"display": "none"},
                {} if pairwise and mode == "selected" else {"display": "none"},
                {} if pairwise else {"display": "none"},
                {} if pairwise and mode == "screening" else {"display": "none"},
            ]

    @staticmethod
    def get_input_layout(register: Callable) -> List[Any]:
        """
        Get the layout for the input block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            Layout for the input block.
        """
        return [
            dcc.Dropdown(
                id=register("exact_pairs", ["value", "options"]),
                multi=True,
                style={"display": "none"},
            ),
            dbc.Select(
                id=register("objective_id2", ["value", "options", "disabled"], type=int),
                style={"display": "none"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Method"),
                            help_button(
                                "Local Parameter Importance: Quantify importance by changing the "
                                "neighborhood of a configuration. Uses default and incumbent "
                                "configuration as reference. \n\n"
                                "fANOVA: Quantify importance globally. \n\n"
                                "PED-ANOVA: Quantify which parameters distinguish the selected "
                                "top-performing region."
                            ),
                            dbc.RadioItems(
                                id=register("method", ["value", "options"]),
                                className="multi-metrics-segmented",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Trees"),
                            help_button(
                                "The number of trees of the internal random forest to estimate the "
                                "hyperparameter importance. "
                                "The more trees are used the more accurate the results. "
                                "However, also it takes longer to compute."
                            ),
                            dbc.Input(
                                id=register("n_trees", ["value", "disabled"], type=optional_int),
                                type="number",
                            ),
                        ],
                        id="importances-trees-field",
                        md=6,
                    ),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Target quantile"),
                            help_button(
                                "Fraction of best trials analyzed by PED-ANOVA. "
                                "For example, 0.10 analyzes the top 10%."
                            ),
                            dbc.Input(
                                id=register("target_quantile", ["value", "disabled"], type=float),
                                type="number",
                                min=0.01,
                                max=0.99,
                                step=0.01,
                            ),
                        ],
                        md=6,
                    ),
                ],
                id="importances-quantile-field",
                className="mt-3",
                style={"display": "none"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Interaction order"),
                            help_button(
                                "Order 1 calculates main effects only. Order 2 additionally "
                                "screens all parameter pairs or calculates exact interactions "
                                "for selected hyperparameters. Exact marginalization can take "
                                "several minutes per budget."
                            ),
                            dbc.Select(
                                id=register(
                                    "interaction_order",
                                    ["value", "options", "disabled"],
                                    type=int,
                                ),
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Interaction analysis"),
                            dbc.RadioItems(
                                id=register("interaction_mode", "value"),
                                options=[
                                    {"label": "Screen all pairs", "value": "screening"},
                                    {"label": "Selected parameters", "value": "selected"},
                                ],
                                value="screening",
                                className="multi-metrics-segmented",
                            ),
                        ],
                        id="importances-interaction-mode-field",
                        style={"display": "none"},
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Interaction hyperparameters"),
                            help_button(
                                "Select between two and four hyperparameters. All main effects "
                                "remain visible; only pairwise interactions are limited to this "
                                "selection."
                            ),
                            dcc.Dropdown(
                                id=register(
                                    "interaction_hp_names", ["value", "options", "disabled"]
                                ),
                                multi=True,
                                placeholder="Select 2–4 hyperparameters ...",
                            ),
                        ],
                        id="importances-interaction-hps-field",
                        style={"display": "none"},
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Exact top pairs"),
                                        help_button(
                                            "Screen all pairs and refine promising or uncertain "
                                            "pairs. Calculate the top pairs exactly; set 0 for "
                                            "screening only."
                                        ),
                                        dbc.Input(
                                            id=register("screen_top_k", "value", type=optional_int),
                                            type="number",
                                            min=0,
                                            step=1,
                                            value=5,
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Additional exact pairs"),
                                        dcc.Dropdown(
                                            id=register(
                                                "pending_exact_pairs",
                                                ["value", "options"],
                                                filter=True,
                                            ),
                                            multi=True,
                                            placeholder="Select pairs, then click Process ...",
                                        ),
                                    ],
                                    md=8,
                                ),
                            ]
                        ),
                        id="importances-screening-fields",
                        style={"display": "none"},
                        className="mt-3",
                        md=12,
                    ),
                ],
                id="importances-interactions-field",
                className="mt-3",
                style={"display": "none"},
            ),
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[html.Div]:
        """
        Get the layout for the filter block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[html.Div]
            Layout for the filter block.
        """
        return [
            dbc.Checklist(
                id=register("hyperparameter_names", ["value", "options"]),
                style={"display": "none"},
            ),
            html.Div(
                [
                    dbc.Label("Plot top N Importances"),
                    help_button(
                        "Shows the n most important main or interaction effects, "
                        "considering all hyperparameters."
                    ),
                    dbc.Input(id=register("n_hps", "value"), type="number"),
                ],
                className="mb-3",
            ),
            dbc.Checklist(
                id=register("budget_ids", ["value", "options"]),
                style={"display": "none"},
            ),
        ]

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the content for the defined inputs in 'get_input_layout' and 'get_filter_layout'.

        This method is necessary to pre-load contents for the inputs.
        If the plugin is called for the first time, or there are no results in the cache,
        the plugin gets its content from this method.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Content to be filled.
        """
        method_labels = [
            "Local Parameter Importance (local)",
            "fANOVA (global)",
            "PED-ANOVA (top region)",
        ]
        method_values = ["local", "global", "ped_anova"]

        return {
            "objective_id1": {"options": [], "value": None},
            "objective_id2": {"options": [], "value": -1, "disabled": False},
            "method": {
                "options": get_select_options(method_labels, method_values),
                "value": "local",
            },
            "n_trees": {"value": 10, "disabled": False},
            "target_quantile": {"value": 0.1, "disabled": True},
            "interaction_order": {
                "options": get_select_options(
                    ["1 — Main effects", "2 — Main effects + pairwise interactions"], [1, 2]
                ),
                "value": 1,
                "disabled": True,
            },
            "interaction_hp_names": {"options": [], "value": [], "disabled": True},
            "interaction_mode": {"value": "screening"},
            "screen_top_k": {"value": 5},
            "exact_pairs": {"value": [], "options": []},
            "pending_exact_pairs": {"value": [], "options": []},
            "hyperparameter_names": {"options": get_checklist_options(), "value": []},
            "n_hps": {"value": 0},
            "budget_ids": {"options": get_checklist_options(), "value": []},
        }

    def _completed_pairs(self, run_id: Optional[str], inputs: Dict[str, Any]) -> set[str]:
        """Find pairs calculated exactly at every budget for the current forest settings."""
        output = (self.raw_outputs or {}).get(run_id) or {}
        context = output.get("_screening_context", {})
        if not context or any(
            context.get(key) != inputs.get(key, {}).get("value")
            for key in ("objective_id1", "n_trees")
        ):
            return set()
        budgets = list(output.get("_screening", {}).values())
        if not budgets:
            return set()
        exact_by_budget = [
            {
                json.dumps(row["members"])
                for row in rows
                if row.get("exact") is not None and np.isfinite(row["exact"])
            }
            for rows in budgets
        ]
        return set.intersection(*exact_by_budget)

    def load_input_updates(self, events: Dict[str, Any]) -> Dict[str, Any]:
        """Queue table selections without changing already completed calculation requests."""
        requests = list(self.previous_inputs.get("exact_pairs", {}).get("value") or [])
        cell = events.get("screening_cell")
        if cell and cell.get("column_id") == "pair":
            try:
                _budget, first, second = json.loads(cell["row_id"])
            except (KeyError, TypeError, ValueError):
                return {}
            pair = json.dumps([first, second])
            run_id = self.previous_inputs.get("run", {}).get("value")
            completed = self._completed_pairs(run_id, self.previous_inputs)
            if pair not in completed and pair not in requests:
                requests.append(pair)
            return {"exact_pairs": {"value": requests}}
        if events.get("screening_ready"):
            # Consume the ready event in the native callback, which then refreshes the picker.
            return {"exact_pairs": {"value": requests}}
        return {}

    def load_dependency_inputs(self, run, previous_inputs, inputs) -> Dict[str, Any]:  # type: ignore # noqa: E501
        """
        Works like 'load_inputs' but called after inputs have changed.

        Note
        ----
        Only the changes have to be returned.
        The returned dictionary will be merged with the inputs.

        Parameters
        ----------
        run
            The selected run.
        inputs : Dict[str, Any]
            Current content of the inputs.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the changes.
        """
        # Prepare objectives
        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        objective_value1 = inputs["objective_id1"]["value"]
        objective_value2 = -1  # Only the main objective selected in the sidebar is used.

        # Pre-set values
        if objective_value1 is None:
            objective_value1 = objective_ids[0]

        objective_options = get_select_options(objective_names, objective_ids)
        objective_options2 = [
            dict for dict in objective_options if dict["value"] != objective_value1
        ]  # make sure the same objective cannot be chosen twice
        objective_options2 += [
            {"label": "Select objective ...", "value": -1}
        ]  # add the option to deselect the second objective

        # Prepare budgets
        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        budget_options = get_checklist_options(budgets, budget_ids)
        budget_value = inputs["budget_ids"]["value"]

        hp_names = list(run.configspace.keys())
        hp_options = get_checklist_options(hp_names)
        n_hps = inputs["n_hps"]["value"]

        # Pairwise interactions are deliberately restricted to global, single-objective fANOVA.
        method = inputs["method"]["value"]
        ped_anova_selected = method == "ped_anova"
        if ped_anova_selected:
            objective_value2 = -1
        interactions_available = method == "global" and objective_value2 in (None, -1, "-1")
        try:
            interaction_order = int(inputs.get("interaction_order", {}).get("value") or 1)
        except (TypeError, ValueError):
            interaction_order = 1

        eligible_interaction_hp_names = [
            name for name, hp in dict(run.configspace).items() if not isinstance(hp, Constant)
        ]
        selected_interaction_hp_names = inputs.get("interaction_hp_names", {}).get("value") or []
        if not isinstance(selected_interaction_hp_names, list):
            selected_interaction_hp_names = []

        # Canonical configspace order keeps equivalent selections stable in URLs and cache keys.
        selected_set = set(selected_interaction_hp_names)
        selected_interaction_hp_names = [
            name for name in eligible_interaction_hp_names if name in selected_set
        ][:MAX_INTERACTION_HYPERPARAMETERS]

        if not interactions_available:
            interaction_order = 1
            selected_interaction_hp_names = []
        elif interaction_order != 2:
            interaction_order = 1
            selected_interaction_hp_names = []

        interaction_limit_reached = (
            len(selected_interaction_hp_names) >= MAX_INTERACTION_HYPERPARAMETERS
        )
        interaction_options = get_select_options(
            eligible_interaction_hp_names,
            eligible_interaction_hp_names,
            [
                interaction_limit_reached and name not in selected_interaction_hp_names
                for name in eligible_interaction_hp_names
            ],
        )
        pair_options = [
            {"label": " × ".join(pair), "value": json.dumps(pair)}
            for pair in itertools.combinations(eligible_interaction_hp_names, 2)
        ]
        valid_pairs = {option["value"] for option in pair_options}
        selected_pairs = inputs.get("exact_pairs", {}).get("value") or []
        completed = self._completed_pairs(run.id, inputs)
        pending = inputs.get("pending_exact_pairs", {}).get("value") or []
        previous_pending = previous_inputs.get("pending_exact_pairs", {}).get("value") or []
        if pending != previous_pending:
            selected_pairs = [pair for pair in selected_pairs if pair in completed] + pending
        selected_pairs = list(dict.fromkeys(pair for pair in selected_pairs if pair in valid_pairs))

        if n_hps == 0:
            n_hps = len(hp_names)

        # Pre-selection of the hyperparameters
        if run is not None:
            if len(budget_value) == 0:
                budget_value = [budget_ids[-1]]

        return {
            "objective_id1": {
                "options": objective_options,
                "value": objective_value1,
            },
            "objective_id2": {
                "options": objective_options2,
                "value": objective_value2,
                "disabled": ped_anova_selected,
            },
            "method": {
                "value": inputs["method"]["value"],
            },
            "hyperparameter_names": {
                "options": hp_options,
                "value": hp_names,
            },
            "budget_ids": {
                "options": budget_options,
                "value": budget_value,
            },
            "n_hps": {"value": n_hps},
            "n_trees": {
                "value": inputs["n_trees"]["value"],
                "disabled": ped_anova_selected,
            },
            "target_quantile": {
                "value": inputs.get("target_quantile", {}).get("value", 0.1),
                "disabled": not ped_anova_selected,
            },
            "interaction_order": {
                "options": get_select_options(
                    ["1 — Main effects", "2 — Main effects + pairwise interactions"], [1, 2]
                ),
                "value": interaction_order,
                "disabled": not interactions_available,
            },
            "interaction_hp_names": {
                "options": interaction_options,
                "value": selected_interaction_hp_names,
                "disabled": not interactions_available or interaction_order != 2,
            },
            "exact_pairs": {
                "options": pair_options,
                "value": selected_pairs,
            },
            "pending_exact_pairs": {
                "options": [option for option in pair_options if option["value"] not in completed],
                "value": [pair for pair in selected_pairs if pair not in completed],
            },
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return raw data based on the run and input data.

        Warning
        -------
        The returned data must be JSON serializable.

        Note
        ----
        The passed inputs are cleaned and therefore differ
        compared to 'load_inputs' or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run : AbstractRun
            The run to process.
        inputs :  Dict[str, Any]
            The input data.

        Returns
        -------
        Dict[str, Any]
            A serialized dictionary.

        Raises
        ------
        RuntimeError
            If required method inputs are missing or invalid.
            If the method is not found.
        """
        objective: Any = None
        objective = run.get_objective(inputs["objective_id1"])
        if inputs["objective_id2"] not in (None, -1):
            objective = [objective, run.get_objective(inputs["objective_id2"])]
        method = inputs["method"]
        n_trees = inputs["n_trees"]
        target_quantile = inputs.get("target_quantile", 0.1)
        interaction_order = inputs.get("interaction_order", 1)
        interaction_hp_names = inputs.get("interaction_hp_names") or []
        interaction_mode = inputs.get("interaction_mode", "selected")
        screening = interaction_order == 2 and interaction_mode == "screening"
        top_k = inputs.get("screen_top_k", 5)
        top_k = 5 if top_k is None else top_k

        if method in ("local", "global") and n_trees is None:
            raise RuntimeError("Please specify the number of trees.")

        if method == "ped_anova":
            if isinstance(objective, list):
                raise RuntimeError(
                    "PED-ANOVA currently supports single-objective analysis only. "
                    "Please deselect Objective 2."
                )
            if target_quantile is None or not 0.0 < float(target_quantile) < 1.0:
                raise RuntimeError("Target quantile must be between 0 and 1.")

        if interaction_order not in (1, 2):
            raise RuntimeError("Interaction order must be either 1 or 2.")
        if interaction_order == 2:
            if method != "global" or isinstance(objective, list):
                raise RuntimeError(
                    "Pairwise interactions are only available for global single-objective fANOVA."
                )
            if interaction_mode not in ("selected", "screening"):
                raise RuntimeError("Unknown interaction analysis mode.")
            if screening and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0):
                raise RuntimeError("Exact top pairs must be a non-negative integer.")
            if not isinstance(interaction_hp_names, list):
                raise RuntimeError("Interaction hyperparameters must be provided as a list.")
            if len(interaction_hp_names) != len(set(interaction_hp_names)):
                raise RuntimeError("Interaction hyperparameters must be unique.")
            if (
                not screening
                and not 2 <= len(interaction_hp_names) <= MAX_INTERACTION_HYPERPARAMETERS
            ):
                raise RuntimeError("Select between two and four interaction hyperparameters.")

        # Filter before encoding can replace missing objectives with worst-value penalties.
        selected_objectives = objective if isinstance(objective, list) else [objective]
        objective_ids = [run.get_objective_id(selected) for selected in selected_objectives]
        run = copy(run)
        run.history = [
            trial
            for trial in run.history
            if trial.status == Status.SUCCESS
            and all(
                trial.costs[index] is not None and np.isfinite(trial.costs[index])
                for index in objective_ids
            )
        ]

        # Handle constant values in fANOVA: As the fANOVA implementation relies on pyrfr
        # and pyrfr cannot be applied to constant hyperparameters
        # (see https://github.com/automl/fanova/issues/81), as a workaround constant
        # hyperparameters are removed before calculation.
        # Note: This will break if there are conditions or forbiddens including
        # constant hyperparameters.
        hp_dict = dict(run.configspace)
        eligible_interaction_hp_names = {
            name for name, hp in hp_dict.items() if not isinstance(hp, Constant)
        }
        additional_pairs = []
        if screening:
            if len(eligible_interaction_hp_names) < 2:
                raise RuntimeError(
                    "Interaction screening requires at least two non-constant parameters."
                )
            for encoded in inputs.get("exact_pairs") or []:
                try:
                    pair = json.loads(encoded)
                except (TypeError, ValueError):
                    raise RuntimeError("Invalid exact pair selection.")
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or pair[0] == pair[1]
                    or not all(
                        isinstance(name, str) and name in eligible_interaction_hp_names
                        for name in pair
                    )
                ):
                    raise RuntimeError(
                        "Exact pairs must contain two distinct non-constant parameters."
                    )
                additional_pairs.append(
                    tuple(sorted(pair, key=lambda name: run.configspace.index_of[name]))
                )
        if interaction_order == 2 and not set(interaction_hp_names).issubset(
            eligible_interaction_hp_names
        ):
            raise RuntimeError("Interaction hyperparameters must exist and must not be constant.")

        if method == "global" and any(isinstance(v, Constant) for v in hp_dict.values()):
            hp_dict_wo_const = {k: v for k, v in hp_dict.items() if not isinstance(v, Constant)}
            configspace_wo_const = ConfigurationSpace()
            for k in hp_dict_wo_const.keys():
                configspace_wo_const.add_hyperparameter(hp_dict_wo_const[k])
            configspace_wo_const.add(run.configspace.conditions)
            configspace_wo_const.add(run.configspace.forbidden_clauses)
            run.configspace = configspace_wo_const

            configs_wo_const = []
            for n in range(len(run.configs)):
                configs_wo_const.append(
                    {k: v for k, v in run.configs[n].items() if k in hp_dict_wo_const.keys()}
                )
            run.configs = dict(enumerate(configs_wo_const))

        hp_names = list(run.configspace.keys())
        interaction_hp_names = [name for name in hp_names if name in interaction_hp_names]
        budgets = run.get_budgets(include_combined=True)

        # Initialize the evaluator
        evaluator: Optional[Union[LocalEvaluator, GlobalEvaluator, PEDANOVA]] = None
        if method == "local" and isinstance(objective, list):
            evaluator = MOLPI(run)
        elif method == "local":
            evaluator = LocalEvaluator(run)
        elif method == "global" and isinstance(objective, list):
            evaluator = MOfANOVA(run)
        elif method == "global":
            evaluator = GlobalEvaluator(run)
        elif method == "ped_anova":
            evaluator = PEDANOVA(run)
        else:
            raise RuntimeError("Method was not found.")

        # Collect data
        data = {}
        screening_results = {}
        importances: Union[
            str,
            list[dict[str, Any]],
            dict[str, tuple[float, float]],
            dict[Union[str, tuple[str, ...]], tuple[float, float, float, float]],
        ]
        for budget_id, budget in enumerate(budgets):
            assert isinstance(budget, (int, float))
            model_path = None
            if screening:
                assert isinstance(objective, Objective)
                evaluator, model_path = get_cached_fanova(
                    run, objective, budget, n_trees, config.CACHE_DIR / "interaction_models"
                )
            elif method == "ped_anova":
                assert isinstance(evaluator, PEDANOVA)
                assert isinstance(objective, Objective)
                evaluator.calculate(
                    objective,
                    budget,
                    target_quantile=float(target_quantile),
                    region_quantile=1.0,
                )
            else:
                assert not isinstance(evaluator, PEDANOVA)
                evaluator.calculate(objective, budget, n_trees=n_trees, seed=0)
            if isinstance(objective, list):
                assert isinstance(evaluator, (MOLPI, MOfANOVA))
                importances = evaluator.get_importances_(hp_names)
                if any(pd.read_json(StringIO(importances))["importance"].isna()):
                    logger.warning(f"Nan encountered in importance values for budget {budget}.")
            elif method == "global":
                assert isinstance(evaluator, GlobalEvaluator)
                global_importances = evaluator.get_importances(hp_names)
                if any(np.isnan(val) for value in global_importances.values() for val in value):
                    logger.warning(f"Nan encountered in importance values for budget {budget}.")

                records = _serialize_global_importances(global_importances, order=1)
                if interaction_order == 2:
                    if screening:
                        ranked = evaluator.screen_interactions(hp_names, top_k=top_k)
                        pairs = list(
                            dict.fromkeys(
                                [tuple(row["members"]) for row in ranked[:top_k]] + additional_pairs
                            )
                        )
                        pairwise_importances = evaluator.get_pair_importances(
                            pairs, include_cached=True
                        )
                        for row in ranked:
                            exact = pairwise_importances.get(tuple(row["members"]))
                            row["exact"] = float(exact[0]) if exact is not None else None
                        screening_results[budget_id] = ranked
                    else:
                        pairwise_importances = evaluator.get_importances(
                            interaction_hp_names, depth=2
                        )
                    if any(
                        np.isnan(val) for value in pairwise_importances.values() for val in value
                    ):
                        logger.warning(
                            f"Nan encountered in interaction values for budget {budget}."
                        )
                    records.extend(_serialize_global_importances(pairwise_importances, order=2))
                importances = records
            elif method == "ped_anova":
                assert isinstance(evaluator, PEDANOVA)
                importances = evaluator.get_importances(hp_names)
            else:
                assert isinstance(evaluator, LocalEvaluator)
                importances = evaluator.get_importances(hp_names)
                if any(np.isnan(val) for value in importances.values() for val in value):
                    logger.warning(f"Nan encountered in importance values for budget {budget}.")
            data[budget_id] = importances
            if model_path is not None:
                assert isinstance(evaluator, GlobalEvaluator)
                save_cached_fanova(evaluator, model_path)

        if screening:
            data["_screening"] = screening_results
            data["_screening_context"] = {
                "objective_id1": inputs["objective_id1"],
                "n_trees": n_trees,
            }
        return data  # type: ignore

    @staticmethod
    def get_output_layout(register: Callable) -> html.Div:
        """
        Get the layout for the output block.

        Parameters
        ----------
        register : Callable
            Method to register outputs.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        html.Div
            Layout for the output block.
        """
        number_format = Format(precision=5, scheme=Scheme.decimal_or_exponent)
        return html.Div(
            [
                dcc.Store(id=register("screening_ready", "data")),
                dcc.Graph(
                    register("graph", "figure"),
                    responsive="auto",
                    style={"width": "100%"},
                    config={
                        "responsive": True,
                        "toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE},
                    },
                ),
                html.Div(
                    [
                        html.H5("Interaction screening"),
                        html.P(
                            "All parameter pairs are screened. Promising and numerically uncertain "
                            "pairs are refined from 8 × 8 to 16 × 16. Stability describes grid "
                            "sensitivity, not statistical confidence. Click a pair to add it to "
                            "Additional exact pairs, then click Process. Completed pairs disappear "
                            "from the selection; their exact results remain available.",
                            className="text-muted small",
                        ),
                        dash_table.DataTable(
                            id=register("screening", "data"),
                            columns=[
                                {"name": "Budget", "id": "budget"},
                                {"name": "Pair", "id": "pair"},
                                *[
                                    {
                                        "name": label,
                                        "id": key,
                                        "type": "numeric",
                                        "format": number_format,
                                    }
                                    for key, label in (
                                        ("score", "Approx. importance"),
                                        ("exact", "Exact importance"),
                                        ("coarse", "8 × 8"),
                                        ("medium", "16 × 16"),
                                        ("offset", "Offset"),
                                        ("stability", "Resolution Δ"),
                                        ("audit_difference", "Shifted-grid Δ"),
                                    )
                                ],
                                {"name": "Calculation status", "id": "status"},
                            ],
                            data=[],
                            sort_action="native",
                            page_size=15,
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "12px"},
                            style_header={"fontWeight": "600", "backgroundColor": "#f5f6fa"},
                            style_data_conditional=[
                                {
                                    "if": {"column_id": "pair"},
                                    "cursor": "pointer",
                                    "color": "#4f46e5",
                                }
                            ],
                        ),
                    ],
                    id=register("screening_panel", "style"),
                    style={"display": "none"},
                ),
                _importance_method_guide(),
            ]
        )

    @staticmethod
    def load_outputs(run, inputs, outputs) -> Dict[str, Any]:  # type: ignore
        """Keep exact effects in the bar plot and approximate diagnostics in a separate table."""
        selected_budgets = inputs["budget_ids"]
        table = []
        for budget_id, rows in outputs.get("_screening", {}).items():
            if int(budget_id) not in selected_budgets:
                continue
            for row in rows:
                exact = row.get("exact")
                table.append(
                    {
                        **row,
                        "id": json.dumps([int(budget_id), *row["members"]]),
                        "status": (
                            "Exact" if exact is not None and np.isfinite(exact) else row["status"]
                        ),
                        "budget": run.get_budget(int(budget_id), human=True),
                        "pair": " × ".join(row["members"]),
                    }
                )
        exact_outputs = {
            key: value for key, value in outputs.items() if not str(key).startswith("_")
        }
        return {
            "graph": {"figure": Importances._importance_figure(run, inputs, exact_outputs)},
            "screening": {"data": table},
            "screening_panel": {"style": {} if table else {"display": "none"}},
            "screening_ready": {"data": {"run": run.id} if outputs.get("_screening") else None},
        }

    @staticmethod
    def _importance_figure(run, inputs, outputs) -> go.Figure:  # type: ignore
        """
        Read in raw data and prepare for layout.

        Note
        ----
        The passed inputs are cleaned and therefore differ
        compared to 'load_inputs' or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run
            The selected run.
        inputs
            Input and filter values from the user.
        outputs
            Raw output from the run.

        Returns
        -------
        go.figure
            The figure of the importances.
        """
        if inputs["objective_id2"] not in (None, -1):
            # MO case: other plot
            return Importances.load_ouputs_mo_fanova(run, inputs, outputs)

        # First selected, should always be shown first
        selected_budget_ids = inputs["budget_ids"]
        n_hps = inputs["n_hps"]

        if n_hps == "" or n_hps is None:
            raise PreventUpdate
        else:
            n_hps = int(n_hps)

        if len(selected_budget_ids) == 0:
            raise PreventUpdate()

        # Collect data. Effects are keyed by their members so values remain correctly aligned
        # across budgets even when evaluators return a different importance-based order.
        data: Dict[int, Dict[Tuple[str, ...], Tuple[str, float, float]]] = {}
        for budget_id, importances in outputs.items():
            # Important to cast budget_id here because of json serialization
            budget_id = int(budget_id)
            if budget_id not in selected_budget_ids:
                continue

            effects: Dict[Tuple[str, ...], Tuple[str, float, float]] = {}
            if inputs["method"] == "global":
                for record in importances:
                    members = tuple(record["members"])
                    effects[members] = (
                        " × ".join(members),
                        record["importance"],
                        record["variance"],
                    )
            else:
                for hp_name, results in importances.items():
                    effects[(hp_name,)] = (hp_name, results[0], results[1])

            data[budget_id] = effects

        # Sort by last fidelity
        selected_budget_id = max(selected_budget_ids)
        if selected_budget_id not in data:
            raise PreventUpdate
        selected_effect_ids = sorted(
            data[selected_budget_id],
            key=lambda effect_id: data[selected_budget_id][effect_id][1],
            reverse=True,
        )[:n_hps]

        bar_data = []
        for budget_id, effects in data.items():
            budget = run.get_budget(budget_id, human=True)
            displayed_effects = [
                effects[effect_id] for effect_id in selected_effect_ids if effect_id in effects
            ]

            bar_data += [
                go.Bar(
                    name=budget,
                    y=[effect[0] for effect in displayed_effects],
                    x=[effect[1] for effect in displayed_effects],
                    error_x_array=[effect[2] for effect in displayed_effects],
                    orientation="h",
                    marker_color=get_color(budget_id),
                )
            ]

        figure = go.Figure(data=bar_data)
        figure.update_layout(
            barmode="group",
            xaxis_title="Importance",
            legend={"title": "Budget"},
            margin=config.FIGURE_MARGIN,
            yaxis=dict(autorange="reversed"),
            font=dict(size=config.FIGURE_FONT_SIZE),
        )
        show_legend = len(bar_data) > 1
        bottom = 130 if show_legend else 56
        style_analysis(
            figure,
            height=max(280, 28 + bottom + 22 * len(selected_effect_ids)),
            show_legend=show_legend,
        )
        figure.update_layout(margin=dict(t=28, b=bottom))
        save_image(figure, "importances.pdf")

        return figure

    @staticmethod
    def load_ouputs_mo_fanova(run, inputs, outputs) -> go.Figure:  # type: ignore
        """
        Multi-objective case for read in raw data and prepare for layout.

        Note
        ----
        The passed inputs are cleaned and therefore differ
        compared to 'load_inputs' or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run
            The selected run.
        inputs
            Input and filter values from the user.
        outputs
            Raw output from the run.

        Returns
        -------
        go.figure
            The figure of the importances.
        """
        # First selected, should always be shown first
        objective1 = run.get_objective(inputs["objective_id1"]).name
        selected_budget_ids = inputs["budget_ids"]
        n_hps = inputs["n_hps"]

        if n_hps == "" or n_hps is None:
            raise PreventUpdate
        else:
            n_hps = int(n_hps)

        if len(selected_budget_ids) == 0:
            raise PreventUpdate()

        # Collect data
        data = {}
        for budget_id, importances_json in outputs.items():
            df_importances = pd.read_json(StringIO(importances_json))
            # Important to cast budget_id here because of json serialization
            budget_id = int(budget_id)
            if budget_id not in selected_budget_ids:
                continue

            data[budget_id] = df_importances

        # Keep only n_hps most important hyperparameters according to max importance
        selected_budget_id = max(selected_budget_ids)
        idx = (
            data[selected_budget_id]
            .groupby("hp_name")["importance"]
            .max()
            .sort_values(ascending=False)
            .index
        )
        idx = idx[:n_hps]

        color_palette = px.colors.qualitative.Plotly  # Choose a color palette
        colors = {
            hp: color_palette[i % len(color_palette)]
            for i, hp in enumerate(list(run.configspace.keys()))
        }

        # Create the figure
        figure = go.Figure()
        df = data[selected_budget_id][
            data[selected_budget_id]["hp_name"].isin(idx)
        ]  # only keep top hps

        # Group by 'hp_name' and plot each group
        for group_id, group_data in df.groupby("hp_name"):
            # Sort data by the weight column
            group_data = group_data.sort_values(by="weight")

            figure.add_trace(
                go.Scatter(
                    x=group_data["weight"],
                    y=group_data["importance"],
                    mode="lines",
                    name=group_id,
                    line=dict(color=colors[group_id]),
                )
            )

            # Add the shaded area representing the variance
            x = group_data["weight"]
            y1 = (group_data["importance"] - group_data["variance"]).to_list()
            y2 = group_data["importance"] + group_data["variance"]

            figure.add_trace(
                go.Scatter(
                    x=x.tolist() + x[::-1].tolist(),
                    y=y1 + y2[::-1].tolist(),
                    fill="toself",
                    hoverinfo="skip",
                    showlegend=False,
                    opacity=0.2,
                    fillcolor=colors[group_id],
                    line=dict(color=colors[group_id]),
                    mode="lines",
                )
            )

        # Update the layout for labels, title, and axis limits
        figure.update_layout(
            xaxis_title="Weight for " + objective1,
            yaxis_title="Importance",
            xaxis=dict(range=[0, 1], tickangle=-45),
            yaxis=dict(range=[0, df["importance"].max()]),
            margin=config.FIGURE_MARGIN,
            font=dict(size=config.FIGURE_FONT_SIZE),
        )

        style_analysis(figure)
        figure.update_xaxes(tickangle=0)
        save_image(figure, "importances.pdf")

        return figure
