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
"""Interactive neighborhood analysis for a selected Optuna trial."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import copy
import math

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import Constant
from dash import Input, Output, Patch, State, ctx, dash_table, dcc, html, no_update
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import config, interactive
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.plugins.hyperparameter.analysis_utils import (
    _get_objectives_without_time,
    _get_trial_table,
    clear_normalized_trial_cache,
    get_normalized_trials,
)
from deepcave.runs import AbstractRun
from deepcave.utils.analysis_layout import analysis_graph, style_analysis_panel
from deepcave.utils.configspace import (
    configuration_euclidean_distances,
    normalized_configuration_distance_components,
    normalized_configuration_matrix,
)
from deepcave.utils.layout import get_select_options

SELECTED_TRIAL_STORE_ID = "configuration-neighborhood-selected-trial"
VIEW_MODE_STORE_ID = "configuration-neighborhood-view-mode"
DISTANCE_STORE_ID = "configuration-neighborhood-distances"
TOP_FIVE_STORE_ID = "configuration-neighborhood-top-five"
CONTEXT_STORE_ID = "configuration-neighborhood-context"
ANALYSIS_DATA_ID = "configuration-neighborhood-analysis-data"
SELECTION_VIEW_ID = "configuration-neighborhood-selection-view"
SIMILARITY_VIEW_ID = "configuration-neighborhood-similarity-view"
HISTORY_GRAPH_ID = "configuration-neighborhood-history-graph"
TRIAL_TABLE_ID = "configuration-neighborhood-trial-table"
SIMILARITY_GRAPH_ID = "configuration-neighborhood-similarity-graph"
NEIGHBOR_TABLE_ID = "configuration-neighborhood-neighbor-table"
DIFFERENCE_GRAPH_ID = "configuration-neighborhood-difference-graph"
REFERENCE_LABEL_ID = "configuration-neighborhood-reference-label"
NEIGHBOR_TITLE_ID = "configuration-neighborhood-neighbor-title"
NEIGHBOR_COUNT_ID = "configuration-neighborhood-neighbor-count"
SELECT_NEW_TRIAL_ID = "configuration-neighborhood-select-new"
TOP_FIVE_STATUS_ID = "configuration-neighborhood-top-five-status"
QUICK_BEST_ID = "configuration-neighborhood-quick-best"
QUICK_MEDIAN_ID = "configuration-neighborhood-quick-median"
QUICK_WORST_ID = "configuration-neighborhood-quick-worst"
QUICK_LATEST_ID = "configuration-neighborhood-quick-latest"
QUICK_TOP_FIVE_ID = "configuration-neighborhood-quick-top-five"

DEFAULT_NEIGHBORS = 10


def clear_configuration_neighborhood_cache() -> None:
    """Clear cached normalized trial configurations."""
    clear_normalized_trial_cache()


def _trial_records(run: AbstractRun, objective_id: int) -> Dict[str, Any]:
    """Combine the shared objective table with cached configuration encodings."""
    trial_table = _get_trial_table(run, objective_id)
    normalized = get_normalized_trials(run, normalized_configuration_matrix)
    objective_rows = {
        int(row["trial_number"]): row for _, row in trial_table.iterrows()
    }
    records = []
    matrix = []
    for metadata, vector in zip(normalized["records"], normalized["matrix"]):
        trial_id = metadata["trial_id"]
        objective_row = objective_rows.get(trial_id)
        if objective_row is None:
            continue
        record = copy.deepcopy(metadata)
        record.update(
            {
                "objective": float(objective_row["objective"]),
                "objective_rank": float(objective_row["objective_rank"]),
            }
        )
        records.append(record)
        matrix.append(vector)

    order = np.argsort([record["trial_id"] for record in records])
    return {
        "records": [records[index] for index in order],
        "matrix": [matrix[index] for index in order],
        "hyperparameters": normalized["hyperparameters"],
    }


def _objective_order(records: List[Dict[str, Any]], optimize: str) -> List[Dict[str, Any]]:
    """Return deterministic best-to-worst trial ordering."""
    direction = -1 if optimize == "upper" else 1
    return sorted(
        records,
        key=lambda record: (direction * record["objective"], record["trial_id"]),
    )


def quick_selection_ids(records: List[Dict[str, Any]], optimize: str) -> Dict[str, int]:
    """Resolve every unambiguous quick-selection action to a trial ID."""
    if not records:
        return {}
    ordered = _objective_order(records, optimize)
    median = float(np.median([record["objective"] for record in records]))
    median_record = min(
        records,
        key=lambda record: (abs(record["objective"] - median), record["trial_id"]),
    )
    latest = max(records, key=lambda record: (record["end_time"], -record["trial_id"]))
    return {
        "best": int(ordered[0]["trial_id"]),
        "median": int(median_record["trial_id"]),
        "worst": int(ordered[-1]["trial_id"]),
        "latest": int(latest["trial_id"]),
    }


def top_five_percent_records(
    records: List[Dict[str, Any]], optimize: str
) -> List[Dict[str, Any]]:
    """Return the direction-aware best five percent, rounded up to one trial."""
    count = max(1, math.ceil(0.05 * len(records))) if records else 0
    selected_ids = {
        record["trial_id"] for record in _objective_order(records, optimize)[:count]
    }
    return [record for record in records if record["trial_id"] in selected_ids]


def view_styles(view_mode: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return visibility styles without unmounting either page container."""
    if view_mode == "similarity":
        return {"display": "block"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


def reduce_selection_state(
    triggered: Any,
    analysis_data: Dict[str, Any],
    selected_trial_id: Any,
    view_mode: str,
    top_five: bool,
    context: Any,
    *,
    history_click: Any = None,
    active_cell: Any = None,
    similarity_click: Any = None,
) -> Tuple[Any, str, bool, Dict[str, Any], Any]:
    """Reduce every selection mechanism into one trial and view state transition."""
    current_context = {
        "run_id": analysis_data["run_id"],
        "objective_id": analysis_data["objective_id"],
    }
    valid_ids = {record["trial_id"] for record in analysis_data["records"]}
    if triggered == ANALYSIS_DATA_ID:
        run_changed = (
            isinstance(context, dict) and context.get("run_id") != current_context["run_id"]
        )
        if run_changed or selected_trial_id not in valid_ids:
            return None, "select", False, current_context, None
        return selected_trial_id, view_mode or "select", top_five, current_context, no_update
    if triggered == SELECT_NEW_TRIAL_ID:
        return selected_trial_id, "select", top_five, current_context, no_update
    if triggered == QUICK_TOP_FIVE_ID:
        return selected_trial_id, "select", not bool(top_five), current_context, no_update

    candidate = None
    if triggered == HISTORY_GRAPH_ID:
        candidate = _trial_id_from_click(history_click)
    elif triggered == SIMILARITY_GRAPH_ID:
        candidate = _trial_id_from_click(similarity_click)
    elif triggered == TRIAL_TABLE_ID and active_cell:
        candidate = active_cell.get("row_id")
    elif triggered in {QUICK_BEST_ID, QUICK_MEDIAN_ID, QUICK_WORST_ID, QUICK_LATEST_ID}:
        action = {
            QUICK_BEST_ID: "best",
            QUICK_MEDIAN_ID: "median",
            QUICK_WORST_ID: "worst",
            QUICK_LATEST_ID: "latest",
        }[triggered]
        candidate = quick_selection_ids(
            analysis_data["records"], analysis_data["optimize"]
        ).get(action)
    if candidate is None or int(candidate) not in valid_ids:
        raise PreventUpdate
    clear_active_cell = None if triggered == TRIAL_TABLE_ID else no_update
    return int(candidate), "similarity", top_five, current_context, clear_active_cell


def _display_record(record: Dict[str, Any], hyperparameters: List[str]) -> Dict[str, Any]:
    """Flatten a trial record for a Dash table while retaining a stable row ID."""
    row = {
        "id": record["trial_id"],
        "trial_id": record["trial_id"],
        "objective": record["objective"],
        "start_time": record["start_time"],
        "end_time": record["end_time"],
    }
    row.update(
        {f"hp::{name}": record["configuration"].get(name) for name in hyperparameters}
    )
    return row


def _trial_columns(hyperparameters: List[str]) -> List[Dict[str, Any]]:
    """Return dynamic selection-table columns."""
    columns = [
        {"name": "Trial", "id": "trial_id", "type": "numeric"},
        {"name": "Objective", "id": "objective", "type": "numeric"},
        {"name": "Start time [s]", "id": "start_time", "type": "numeric"},
        {"name": "Completion time [s]", "id": "end_time", "type": "numeric"},
    ]
    columns.extend({"name": name, "id": f"hp::{name}"} for name in hyperparameters)
    return columns


def history_figure(data: Dict[str, Any]) -> go.Figure:
    """Build the selectable objective history with a persistent highlight trace."""
    records = data["records"]
    trial_ids = [record["trial_id"] for record in records]
    objectives = [record["objective"] for record in records]
    incumbent = []
    current = -np.inf if data["optimize"] == "upper" else np.inf
    for objective in objectives:
        current = (
            max(current, objective)
            if data["optimize"] == "upper"
            else min(current, objective)
        )
        incumbent.append(current)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=trial_ids,
            y=objectives,
            mode="markers",
            name="Trials",
            customdata=trial_ids,
            marker={"color": "#18A999", "size": 7, "opacity": 0.8},
            hovertemplate=(
                "Trial %{customdata}<br>"
                f"{data['objective_name']}: %{{y:.6g}}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trial_ids,
            y=incumbent,
            mode="lines",
            name="Best so far",
            line={"color": "#8B5CF6", "shape": "hv", "width": 2},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers+text",
            name="Reference",
            marker={"color": "#EF4444", "size": 13, "symbol": "diamond"},
            text=[],
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.update_xaxes(title_text="Trial")
    figure.update_yaxes(title_text=data["objective_name"])
    figure.update_layout(uirevision=f"{data['run_id']}:{data['objective_id']}")
    style_analysis_panel(figure, height=500, bottom=75, show_legend=True)
    return figure


def _empty_figure(message: str = "Select a trial to begin.") -> go.Figure:
    """Return a styled empty analysis figure."""
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, x=0.5, y=0.5)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    style_analysis_panel(figure, height=390, bottom=50)
    return figure


def distance_payload(
    data: Dict[str, Any], selected_trial_id: int, configspace: Any
) -> Dict[str, Any]:
    """Calculate all reference-dependent distances from the cached matrix."""
    records = data["records"]
    selected_index = next(
        (index for index, record in enumerate(records) if record["trial_id"] == selected_trial_id),
        None,
    )
    if selected_index is None:
        return {}
    matrix = np.asarray(data["matrix"], dtype=float)
    components = normalized_configuration_distance_components(
        matrix[selected_index], matrix, configspace
    )
    distances = configuration_euclidean_distances(components, configspace)
    distance_records = []
    for record, distance, component_row in zip(records, distances, components):
        distance_record = copy.deepcopy(record)
        distance_record["distance"] = float(distance)
        distance_record["components"] = component_row.tolist()
        distance_records.append(distance_record)
    return {
        "selected_trial_id": selected_trial_id,
        "records": distance_records,
        "hyperparameters": data["hyperparameters"],
        "constant_indices": data["constant_indices"],
        "objective_name": data["objective_name"],
    }


def similarity_figure(payload: Dict[str, Any]) -> go.Figure:
    """Build the distance-versus-objective scatter plot."""
    selected_id = payload["selected_trial_id"]
    selected = next(record for record in payload["records"] if record["trial_id"] == selected_id)
    others = [record for record in payload["records"] if record["trial_id"] != selected_id]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[record["distance"] for record in others],
            y=[record["objective"] for record in others],
            mode="markers",
            name="Trials",
            customdata=[record["trial_id"] for record in others],
            marker={"color": "#60A5FA", "size": 7, "opacity": 0.75},
            hovertemplate=(
                "Trial %{customdata}<br>Distance: %{x:.6g}<br>"
                f"{payload['objective_name']}: %{{y:.6g}}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0.0],
            y=[selected["objective"]],
            mode="markers+text",
            name="Reference",
            customdata=[selected_id],
            marker={"color": "#EF4444", "size": 13, "symbol": "diamond"},
            text=[f"Trial {selected_id}"],
            textposition="top center",
            hovertemplate=(
                "Trial %{customdata}<br>Distance: %{x:.6g}<br>"
                f"{payload['objective_name']}: %{{y:.6g}}<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title_text="Distance to selected configuration")
    figure.update_yaxes(title_text=payload["objective_name"])
    style_analysis_panel(figure, height=430, bottom=75, show_legend=True)
    return figure


def nearest_records(payload: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    """Return the reference followed by the nearest ``k`` other trials."""
    selected_id = payload["selected_trial_id"]
    reference = next(record for record in payload["records"] if record["trial_id"] == selected_id)
    others = sorted(
        (record for record in payload["records"] if record["trial_id"] != selected_id),
        key=lambda record: (record["distance"], record["trial_id"]),
    )
    return [reference, *others[: max(0, int(k))]]


def _neighbor_rows(
    payload: Dict[str, Any], k: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return flattened nearest-neighbor rows and their dynamic columns."""
    nearest = nearest_records(payload, k)
    rows = []
    for rank, record in enumerate(nearest, start=1):
        row = _display_record(record, payload["hyperparameters"])
        row.update({"rank": rank, "distance": record["distance"]})
        rows.append(row)
    columns = [
        {"name": "Rank", "id": "rank", "type": "numeric"},
        {"name": "Trial", "id": "trial_id", "type": "numeric"},
        {"name": "Distance", "id": "distance", "type": "numeric"},
        {"name": "Objective", "id": "objective", "type": "numeric"},
    ]
    columns.extend(
        {"name": name, "id": f"hp::{name}"} for name in payload["hyperparameters"]
    )
    return rows, columns


def parameter_difference_figure(payload: Dict[str, Any], k: int) -> go.Figure:
    """Plot mean normalized component differences for the nearest neighbors."""
    nearest = nearest_records(payload, k)[1:]
    constant_indices = set(payload["constant_indices"])
    names = [
        name
        for index, name in enumerate(payload["hyperparameters"])
        if index not in constant_indices
    ]
    if not nearest or not names:
        return _empty_figure("No neighboring parameter differences are available.")

    values = []
    for index, _name in enumerate(payload["hyperparameters"]):
        if index in constant_indices:
            continue
        values.append(float(np.mean([record["components"][index] for record in nearest])))
    order = np.argsort(values)
    figure = go.Figure(
        go.Bar(
            x=[values[index] for index in order],
            y=[names[index] for index in order],
            orientation="h",
            marker_color="#6366F1",
            hovertemplate="%{y}: %{x:.4g}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Mean normalized absolute difference", range=[0, 1])
    style_analysis_panel(figure, height=390, bottom=75)
    return figure


def _table_style_data_conditional(selected_trial_id: Optional[int]) -> List[Dict[str, Any]]:
    """Highlight the selected trial by its stable DataTable row ID."""
    if selected_trial_id is None:
        return []
    return [
        {
            "if": {"filter_query": f"{{trial_id}} = {selected_trial_id}"},
            "backgroundColor": "#DBEAFE",
            "fontWeight": "600",
        }
    ]


def _data_table(
    table_id: str,
    records: List[Dict[str, Any]],
    columns: List[Dict[str, Any]],
    *,
    filterable: bool,
) -> dash_table.DataTable:
    """Create a compact horizontally scrollable trial table."""
    return dash_table.DataTable(
        id=table_id,
        data=records,
        columns=columns,
        sort_action="native",
        sort_mode="multi",
        filter_action="native" if filterable else "none",
        page_action="native" if filterable else "none",
        page_size=12,
        cell_selectable=filterable,
        style_table={"overflowX": "auto", "maxHeight": "430px", "overflowY": "auto"},
        style_header={"fontWeight": "600", "backgroundColor": "#F8FAFC"},
        style_cell={
            "fontFamily": "Arial, sans-serif",
            "fontSize": 12,
            "padding": "0.45rem",
            "minWidth": "90px",
            "maxWidth": "220px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
    )


def _page_layout(data: Dict[str, Any]) -> Component:
    """Create both permanently mounted view containers for one objective."""
    displayed = [_display_record(record, data["hyperparameters"]) for record in data["records"]]
    maximum_neighbors = max(1, len(data["records"]) - 1)
    default_neighbors = min(DEFAULT_NEIGHBORS, maximum_neighbors)
    quick_button = {"size": "sm", "outline": True, "color": "primary"}

    selection_view = html.Div(
        [
            html.P(
                "Select a trial in the history, choose a table row, or use a quick selection.",
                className="text-muted",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("History", className="analysis-card__header"),
                            analysis_graph(HISTORY_GRAPH_ID, 500, config.FIGURE_DOWNLOAD_SCALE),
                        ],
                        className="analysis-card configuration-neighborhood-history-card",
                    ),
                    html.Div(
                        [
                            html.Div("Trials", className="configuration-neighborhood-card-title"),
                            _data_table(
                                TRIAL_TABLE_ID,
                                displayed,
                                _trial_columns(data["hyperparameters"]),
                                filterable=True,
                            ),
                            html.Div(
                                [
                                    dbc.Button("Best objective", id=QUICK_BEST_ID, **quick_button),
                                    dbc.Button(
                                        "Median objective", id=QUICK_MEDIAN_ID, **quick_button
                                    ),
                                    dbc.Button(
                                        "Worst objective", id=QUICK_WORST_ID, **quick_button
                                    ),
                                    dbc.Button(
                                        "Latest completed", id=QUICK_LATEST_ID, **quick_button
                                    ),
                                    dbc.Button("Top 5%", id=QUICK_TOP_FIVE_ID, **quick_button),
                                    html.Span(id=TOP_FIVE_STATUS_ID, className="small text-muted"),
                                ],
                                className="configuration-neighborhood-quick-actions",
                            ),
                        ],
                        className="analysis-card configuration-neighborhood-trials-card",
                    ),
                ],
                className="configuration-neighborhood-selection-grid",
            ),
        ],
        id=SELECTION_VIEW_ID,
        style={"display": "block"},
    )

    similarity_view = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        id=REFERENCE_LABEL_ID,
                        className="configuration-neighborhood-reference",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-arrow-left me-2"), "Select new trial"],
                        id=SELECT_NEW_TRIAL_ID,
                        color="primary",
                        outline=True,
                        size="sm",
                    ),
                ],
                className="configuration-neighborhood-reference-bar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Settings", className="configuration-neighborhood-card-title"),
                            dbc.Label("Objective", className="small text-muted"),
                            dbc.Input(value=data["objective_name"], disabled=True, size="sm"),
                            dbc.FormText("Controlled by the global sidebar selector."),
                            dbc.Label("Distance metric", className="small text-muted mt-3"),
                            dcc.Dropdown(
                                options=[
                                    {
                                        "label": "Normalized Euclidean",
                                        "value": "normalized_euclidean",
                                    }
                                ],
                                value="normalized_euclidean",
                                clearable=False,
                                disabled=True,
                            ),
                            dbc.Label("Number of neighbors (k)", className="small text-muted mt-3"),
                            dbc.Input(
                                id=NEIGHBOR_COUNT_ID,
                                type="number",
                                min=1,
                                max=maximum_neighbors,
                                step=1,
                                value=default_neighbors,
                                disabled=len(data["records"]) <= 1,
                                size="sm",
                            ),
                        ],
                        className="analysis-card configuration-neighborhood-settings-card",
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Distance vs Objective", className="analysis-card__header"
                            ),
                            analysis_graph(
                                SIMILARITY_GRAPH_ID, 430, config.FIGURE_DOWNLOAD_SCALE
                            ),
                        ],
                        className="analysis-card configuration-neighborhood-distance-card",
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Parameter Differences — Neighbors vs. Selected",
                                className="analysis-card__header",
                            ),
                            analysis_graph(
                                DIFFERENCE_GRAPH_ID, 390, config.FIGURE_DOWNLOAD_SCALE
                            ),
                        ],
                        className="analysis-card configuration-neighborhood-difference-card",
                    ),
                ],
                className="configuration-neighborhood-similarity-grid",
            ),
            html.Div(
                [
                    html.Div(
                        id=NEIGHBOR_TITLE_ID,
                        className="configuration-neighborhood-card-title",
                    ),
                    _data_table(NEIGHBOR_TABLE_ID, [], [], filterable=False),
                ],
                className="analysis-card configuration-neighborhood-neighbor-card",
            ),
        ],
        id=SIMILARITY_VIEW_ID,
        style={"display": "none"},
    )
    return html.Div(
        [
            dcc.Store(id=ANALYSIS_DATA_ID, data=data),
            selection_view,
            similarity_view,
        ]
    )


class ConfigurationNeighborhood(DynamicPlugin):
    """Inspect normalized configuration distances around one Optuna trial."""

    id = "configuration_neighborhood"
    name = "Configuration Neighborhood"
    icon = "fas fa-project-diagram"
    help = "plugins/configuration_neighborhood.html"
    activate_run_selection = True
    main_objective_input = "objective_id"

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """Accept non-empty individual Optuna studies with a search space."""
        return run.prefix == "Optuna" and not run.empty() and len(run.configspace) > 0

    @staticmethod
    def get_input_layout(_register: Callable) -> List[Component]:
        """Use the global run and objective controls without local duplicates."""
        return []

    @staticmethod
    def get_filter_layout(_register: Callable) -> List[Any]:
        """Return no base-plugin filters; local controls use dedicated callbacks."""
        return []

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Return defaults before an Optuna study is selected."""
        return {"objective_id": {"options": [], "value": None}}

    def load_dependency_inputs(
        self,
        run: AbstractRun,
        _previous_inputs: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Populate objective choices for the selected study."""
        objectives = _get_objectives_without_time(run)
        objective_ids = [run.get_objective_id(objective) for objective in objectives]
        value = inputs["objective_id"]["value"]
        if value not in objective_ids:
            value = objective_ids[0]
        return {
            "objective_id": {
                "options": get_select_options(
                    [objective.name for objective in objectives], objective_ids
                ),
                "value": value,
            }
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Collect objective data and reuse the run-wide normalized configuration cache."""
        objective_id = int(inputs["objective_id"])
        objective = run.get_objective(objective_id)
        if objective is None:
            raise PreventUpdate
        trials = _trial_records(run, objective_id)
        if not trials["records"]:
            raise ValueError("No successful trials are available for this analysis.")
        trials.update(
            {
                "run_id": run.id,
                "objective_id": objective_id,
                "objective_name": objective.name,
                "optimize": objective.optimize,
                "constant_indices": [
                    index
                    for index, hyperparameter in enumerate(run.configspace.values())
                    if isinstance(hyperparameter, Constant)
                ],
            }
        )
        return trials

    @staticmethod
    def get_output_layout(register: Callable) -> Component:
        """Return stable state stores around the replaceable objective content."""
        return html.Div(
            [
                dcc.Store(id=SELECTED_TRIAL_STORE_ID, data=None),
                dcc.Store(id=VIEW_MODE_STORE_ID, data="select"),
                dcc.Store(id=DISTANCE_STORE_ID, data=None),
                dcc.Store(id=TOP_FIVE_STORE_ID, data=False),
                dcc.Store(id=CONTEXT_STORE_ID, data=None),
                html.Div(id=register("content", "children")),
            ]
        )

    @staticmethod
    def load_outputs(
        _run: AbstractRun, _inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Component:
        """Build both mounted views for the current run and objective."""
        page = _page_layout(outputs)
        history = next(
            component
            for component in _walk_components(page)
            if getattr(component, "id", None) == HISTORY_GRAPH_ID
        )
        history.figure = history_figure(outputs)
        similarity = next(
            component
            for component in _walk_components(page)
            if getattr(component, "id", None) == SIMILARITY_GRAPH_ID
        )
        similarity.figure = _empty_figure()
        differences = next(
            component
            for component in _walk_components(page)
            if getattr(component, "id", None) == DIFFERENCE_GRAPH_ID
        )
        differences.figure = _empty_figure()
        return page

    @interactive
    def register_callbacks(self) -> None:
        """Register the central selection reducer and dependent view callbacks."""
        super().register_callbacks()
        from deepcave import app, run_handler

        @app.callback(  # type: ignore
            [
                Output(SELECTED_TRIAL_STORE_ID, "data"),
                Output(VIEW_MODE_STORE_ID, "data"),
                Output(TOP_FIVE_STORE_ID, "data"),
                Output(CONTEXT_STORE_ID, "data"),
                Output(TRIAL_TABLE_ID, "active_cell"),
            ],
            [
                Input(ANALYSIS_DATA_ID, "data"),
                Input(HISTORY_GRAPH_ID, "clickData"),
                Input(TRIAL_TABLE_ID, "active_cell"),
                Input(QUICK_BEST_ID, "n_clicks"),
                Input(QUICK_MEDIAN_ID, "n_clicks"),
                Input(QUICK_WORST_ID, "n_clicks"),
                Input(QUICK_LATEST_ID, "n_clicks"),
                Input(QUICK_TOP_FIVE_ID, "n_clicks"),
                Input(SIMILARITY_GRAPH_ID, "clickData"),
                Input(SELECT_NEW_TRIAL_ID, "n_clicks"),
            ],
            [
                State(SELECTED_TRIAL_STORE_ID, "data"),
                State(VIEW_MODE_STORE_ID, "data"),
                State(TOP_FIVE_STORE_ID, "data"),
                State(CONTEXT_STORE_ID, "data"),
            ],
        )
        def reduce_selection(
            analysis_data: Dict[str, Any],
            history_click: Any,
            active_cell: Any,
            _best_clicks: Any,
            _median_clicks: Any,
            _worst_clicks: Any,
            _latest_clicks: Any,
            _top_clicks: Any,
            similarity_click: Any,
            _select_new_clicks: Any,
            selected_trial_id: Any,
            view_mode: str,
            top_five: bool,
            context: Any,
        ) -> Any:
            if not analysis_data:
                raise PreventUpdate
            return reduce_selection_state(
                ctx.triggered_id,
                analysis_data,
                selected_trial_id,
                view_mode,
                top_five,
                context,
                history_click=history_click,
                active_cell=active_cell,
                similarity_click=similarity_click,
            )

        app.clientside_callback(
            """
            function(viewMode, analysisData) {
                window.setTimeout(function() {
                    window.dispatchEvent(new Event("resize"));
                }, 0);
                if (viewMode === "similarity") {
                    return [{display: "block"}, {display: "block"}];
                }
                return [{display: "block"}, {display: "none"}];
            }
            """,
            [
                Output(SELECTION_VIEW_ID, "style"),
                Output(SIMILARITY_VIEW_ID, "style"),
            ],
            [Input(VIEW_MODE_STORE_ID, "data"), Input(ANALYSIS_DATA_ID, "data")],
        )

        @app.callback(  # type: ignore
            [
                Output(TRIAL_TABLE_ID, "data"),
                Output(TRIAL_TABLE_ID, "style_data_conditional"),
                Output(TOP_FIVE_STATUS_ID, "children"),
                Output(QUICK_TOP_FIVE_ID, "active"),
            ],
            [
                Input(ANALYSIS_DATA_ID, "data"),
                Input(TOP_FIVE_STORE_ID, "data"),
                Input(SELECTED_TRIAL_STORE_ID, "data"),
            ],
        )
        def update_trial_table(
            analysis_data: Dict[str, Any], top_five: bool, selected_trial_id: Any
        ) -> Any:
            if not analysis_data:
                raise PreventUpdate
            records = analysis_data["records"]
            if top_five:
                records = top_five_percent_records(records, analysis_data["optimize"])
            displayed = [
                _display_record(record, analysis_data["hyperparameters"]) for record in records
            ]
            status = f"Showing {len(records)} candidate(s)" if top_five else ""
            return (
                displayed,
                _table_style_data_conditional(selected_trial_id),
                status,
                bool(top_five),
            )

        @app.callback(  # type: ignore
            [
                Output(DISTANCE_STORE_ID, "data"),
                Output(SIMILARITY_GRAPH_ID, "figure"),
                Output(REFERENCE_LABEL_ID, "children"),
                Output(HISTORY_GRAPH_ID, "figure"),
            ],
            [
                Input(SELECTED_TRIAL_STORE_ID, "data"),
                Input(ANALYSIS_DATA_ID, "data"),
            ],
        )
        def update_reference(selected_trial_id: Any, analysis_data: Dict[str, Any]) -> Any:
            history_patch = Patch()
            if selected_trial_id is None or not analysis_data:
                history_patch["data"][2]["x"] = []
                history_patch["data"][2]["y"] = []
                history_patch["data"][2]["text"] = []
                return None, _empty_figure(), "", history_patch
            try:
                run = run_handler.get_run(analysis_data["run_id"])
            except RuntimeError:
                raise PreventUpdate
            payload = distance_payload(analysis_data, int(selected_trial_id), run.configspace)
            if not payload:
                return None, _empty_figure(), "", history_patch
            selected = next(
                record
                for record in payload["records"]
                if record["trial_id"] == int(selected_trial_id)
            )
            history_patch["data"][2]["x"] = [selected_trial_id]
            history_patch["data"][2]["y"] = [selected["objective"]]
            history_patch["data"][2]["text"] = [f"Trial {selected_trial_id}"]
            label = html.Div(
                [
                    html.Strong(f"Reference configuration: Trial {selected_trial_id}"),
                    html.Span(f" | Objective: {selected['objective']:.6g}"),
                ]
            )
            return payload, similarity_figure(payload), label, history_patch

        @app.callback(  # type: ignore
            [
                Output(NEIGHBOR_TABLE_ID, "data"),
                Output(NEIGHBOR_TABLE_ID, "columns"),
                Output(NEIGHBOR_TABLE_ID, "style_data_conditional"),
                Output(DIFFERENCE_GRAPH_ID, "figure"),
                Output(NEIGHBOR_TITLE_ID, "children"),
            ],
            [Input(DISTANCE_STORE_ID, "data"), Input(NEIGHBOR_COUNT_ID, "value")],
        )
        def update_neighbors(payload: Dict[str, Any], k: Any) -> Any:
            if not payload:
                return [], [], [], _empty_figure(), "Nearest Configurations"
            neighbor_count = max(0, int(k or DEFAULT_NEIGHBORS))
            rows, columns = _neighbor_rows(payload, neighbor_count)
            selected_id = payload["selected_trial_id"]
            return (
                rows,
                columns,
                _table_style_data_conditional(selected_id),
                parameter_difference_figure(payload, neighbor_count),
                f"Nearest Configurations (k = {neighbor_count})",
            )


def _trial_id_from_click(click_data: Any) -> Optional[int]:
    """Extract the real trial ID carried by a Plotly point's customdata."""
    try:
        customdata = click_data["points"][0]["customdata"]
    except (KeyError, IndexError, TypeError):
        return None
    if isinstance(customdata, (list, tuple)):
        customdata = customdata[0]
    try:
        return int(customdata)
    except (TypeError, ValueError):
        return None


def _walk_components(component: Component) -> Any:
    """Yield one Dash component and all descendants."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "to_plotly_json"):
            yield from _walk_components(child)
