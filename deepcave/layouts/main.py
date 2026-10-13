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
# MainLayout

This module provides and defines the visualization of the main layout.

Also registers and handles the callbacks.

## Classes
    - MainLayout: This class defines and provides the main layout.
"""

from typing import Any, Dict, List, Optional

from urllib.parse import urlparse

from dash import ALL, ClientsideFunction, Patch, dcc, html, no_update
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component

from deepcave import app, c, config, queue, run_handler
from deepcave.layouts import Layout
from deepcave.layouts.general import GeneralLayout
from deepcave.layouts.header import HeaderLayout
from deepcave.layouts.not_found import NotFoundLayout
from deepcave.layouts.notification import NotificationLayout
from deepcave.layouts.sidebar import SidebarLayout
from deepcave.plugins import Plugin
from deepcave.plugins.hyperparameter.search_space_state import (
    PLANNED_SEARCH_SPACE_CONTEXT_TYPE,
    PLANNED_SEARCH_SPACE_HYDRATION_ID,
    PLANNED_SEARCH_SPACE_STORE_ID,
    persist_planning_state_change,
    planning_state_from_cache,
    planning_state_store,
)
from deepcave.utils.dash import alert

PERSISTENT_ANALYSIS_IDS = (
    "hpo_planning",
    "one_dimensional_analysis",
    "two_dimensional_analysis",
    "search_space_coverage",
)
PERSISTENT_ANALYSIS_CACHE_ID = "persistent-analysis-cache"
PERSISTENT_ANALYSIS_STATE_ID = "persistent-analysis-state"
PERSISTENT_ANALYSIS_PAGE_TYPE = "persistent-analysis-page"
PERSISTENT_ANALYSIS_LOADING_ID = "persistent-analysis-loading"
GENERAL_CONTAINER_ID = "general-page-container"
PLANNED_SEARCH_SPACE_PERSISTENCE_ID = "planned-search-space-persistence-revision"


def _plugin_id_from_path(pathname: str) -> Optional[str]:
    """Return the plugin id encoded in a plugin route, if present."""
    parts = urlparse(pathname or "").path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "plugins":
        return parts[1]
    return None


class MainLayout(Layout):
    """
    Define and provide the main layout.

    Handle different callbacks of the layout.

    Properties
    ----------
    plugins : Dict[str, Plugin]
        A dictionary containing the different plugins.
    sidebar_layout : SidebarLayout
        A sidebar layout with the categorized plugins.
    head_layout : HeaderLayout
        The header layout.
    general_layout : GeneralLayout
        The general layout.
    notification_layout : NotificationLayout
        The notification layout.
    not_found_layout : NotFoundLayout
        The Not Found layout.
    """

    def __init__(self, categorized_plugins: Dict[str, List[Plugin]]):
        super().__init__()
        self.plugins = {}
        self.sidebar_layout = SidebarLayout(categorized_plugins)
        self.header_layout = HeaderLayout()
        self.general_layout = GeneralLayout()
        self.notification_layout = NotificationLayout()
        self.not_found_layout = NotFoundLayout  # Needs to be class here to add url
        for plugins in categorized_plugins.values():
            for plugin in plugins:
                self.plugins[plugin.id] = plugin

    def register_callbacks(self) -> None:
        """Register and handle the callbacks."""

        # A single synchronous browser reducer owns the canonical planning store and the value
        # properties which can feed back into it. Keeping both sides in one callback is Dash's
        # supported circular-synchronization pattern and prevents stale server responses from
        # overwriting a newer edit.
        app.clientside_callback(
            ClientsideFunction(
                namespace="searchSpaceState",
                function_name="reducePlanningEvents",
            ),
            [
                Output(PLANNED_SEARCH_SPACE_STORE_ID, "data"),
                Output({"type": "hpo-planner-range", "index": ALL, "context": ALL}, "value"),
                Output({"type": "hpo-planner-range", "index": ALL, "context": ALL}, "min"),
                Output({"type": "hpo-planner-range", "index": ALL, "context": ALL}, "max"),
                Output(
                    {"type": "hpo-planner-categories", "index": ALL, "context": ALL},
                    "value",
                ),
                Output(
                    {
                        "type": "two-dimensional-planning-range",
                        "index": ALL,
                        "context": ALL,
                    },
                    "value",
                ),
                Output(
                    {
                        "type": "two-dimensional-planning-range",
                        "index": ALL,
                        "context": ALL,
                    },
                    "min",
                ),
                Output(
                    {
                        "type": "two-dimensional-planning-range",
                        "index": ALL,
                        "context": ALL,
                    },
                    "max",
                ),
                Output(
                    {
                        "type": "two-dimensional-planning-categories",
                        "index": ALL,
                        "context": ALL,
                    },
                    "value",
                ),
            ],
            [
                Input({"type": "hpo-planner-range", "index": ALL, "context": ALL}, "value"),
                Input(
                    {"type": "hpo-planner-categories", "index": ALL, "context": ALL},
                    "value",
                ),
                Input({"type": "hpo-planning-reset", "context": ALL}, "n_clicks"),
                Input(
                    {
                        "type": "two-dimensional-planning-range",
                        "index": ALL,
                        "context": ALL,
                    },
                    "value",
                ),
                Input(
                    {
                        "type": "two-dimensional-planning-categories",
                        "index": ALL,
                        "context": ALL,
                    },
                    "value",
                ),
                Input(
                    {"type": "two-dimensional-planning-reset", "context": ALL},
                    "n_clicks",
                ),
                Input({"type": PLANNED_SEARCH_SPACE_CONTEXT_TYPE, "source": ALL}, "data"),
                Input(PLANNED_SEARCH_SPACE_HYDRATION_ID, "data"),
            ],
            [
                State({"type": "hpo-planner-range", "index": ALL, "context": ALL}, "id"),
                State({"type": "hpo-planner-categories", "index": ALL, "context": ALL}, "id"),
                State(
                    {
                        "type": "two-dimensional-planning-range",
                        "index": ALL,
                        "context": ALL,
                    },
                    "id",
                ),
                State(
                    {
                        "type": "two-dimensional-planning-categories",
                        "index": ALL,
                        "context": ALL,
                    },
                    "id",
                ),
                State(PLANNED_SEARCH_SPACE_STORE_ID, "data"),
            ],
            prevent_initial_call=True,
        )

        @app.callback(  # type: ignore
            Output(PLANNED_SEARCH_SPACE_HYDRATION_ID, "data"),
            Input("planned-search-space-hydration-on-load", "n_intervals"),
            prevent_initial_call=False,
        )
        def hydrate_planned_search_space(_: int) -> Dict[str, Any]:
            """Read the latest metacache once for every full browser page load."""
            return planning_state_from_cache(c)

        @app.callback(  # type: ignore
            Output(PLANNED_SEARCH_SPACE_PERSISTENCE_ID, "data"),
            Input(PLANNED_SEARCH_SPACE_STORE_ID, "data"),
            prevent_initial_call=True,
        )
        def persist_planned_search_space(state: Any) -> Any:
            """Persist a committed browser edit without delaying its visual update."""
            try:
                return persist_planning_state_change(c, state)
            except ValueError:
                return no_update

        @app.callback(  # type: ignore
            [
                Output(PERSISTENT_ANALYSIS_CACHE_ID, "children"),
                Output(PERSISTENT_ANALYSIS_STATE_ID, "data"),
            ],
            [
                Input("on-page-load", "pathname"),
                Input("global-run-selection-trigger", "data"),
                Input("general-optuna-import-trigger", "data"),
                Input("general-selected-runs", "data"),
                Input("general-group-trigger", "data"),
                Input("general-cache-cleared-trigger", "data"),
            ],
            [
                State("global-run-selector", "value"),
                State(PERSISTENT_ANALYSIS_STATE_ID, "data"),
            ],
        )
        def mount_persistent_analysis(
            pathname: str,
            _: Optional[int],
            optuna_revision: Optional[int],
            selected_runs: Any,
            group_revision: Optional[int],
            cache_revision: Optional[int],
            active_run_id: Optional[str],
            state: Any,
        ) -> Any:
            """Mount only the requested analysis and retain warm pages for one data context."""
            active_plugin_id = _plugin_id_from_path(pathname)
            current = dict(state) if isinstance(state, dict) else {}
            invalidation = {
                "optuna": optuna_revision or 0,
                "selected_runs": selected_runs or [],
                "groups": group_revision or 0,
                "cache": cache_revision or 0,
            }

            if active_plugin_id not in PERSISTENT_ANALYSIS_IDS:
                if current.get("invalidation") == invalidation:
                    return no_update, no_update
                current["invalidation"] = invalidation
                current["dirty"] = True
                return no_update, current

            if active_plugin_id not in self.plugins:
                return no_update, no_update

            if active_run_id is None:
                page = html.Div(
                    alert("Please select a run in the sidebar first."),
                    id={"type": PERSISTENT_ANALYSIS_PAGE_TYPE, "plugin": active_plugin_id},
                    className="persistent-analysis-page",
                )
                return [page], {
                    "run_id": None,
                    "data_revision": None,
                    "mounted": [active_plugin_id],
                    "statuses": {active_plugin_id: "error"},
                    "invalidation": invalidation,
                    "dirty": False,
                }

            try:
                run_handler.set_active_run_id(active_run_id)
                active_run = run_handler.get_run(active_run_id)
            except RuntimeError:
                page = html.Div(
                    alert("The selected run is no longer available."),
                    id={"type": PERSISTENT_ANALYSIS_PAGE_TYPE, "plugin": active_plugin_id},
                    className="persistent-analysis-page",
                )
                return [page], {
                    "run_id": active_run_id,
                    "data_revision": None,
                    "mounted": [active_plugin_id],
                    "statuses": {active_plugin_id: "error"},
                    "invalidation": invalidation,
                    "dirty": False,
                }

            # This is intentionally cheap and evaluated only on explicit run/data events, not on
            # every navigation callback of every plugin. Groups include their members through
            # ``latest_change`` and their merged history length.
            data_revision = [
                getattr(active_run, "latest_change", None),
                len(getattr(active_run, "history", [])),
                [[objective.name, objective.optimize] for objective in active_run.get_objectives()],
            ]
            context_changed = (
                current.get("run_id") != active_run_id
                or current.get("data_revision") != data_revision
                or current.get("invalidation") != invalidation
                or bool(current.get("dirty"))
            )
            mounted = [] if context_changed else list(current.get("mounted", []))
            if active_plugin_id in mounted:
                return no_update, no_update

            plugin = self.plugins[active_plugin_id]
            status = "mounted"
            if plugin.check_run_compatibility(active_run):
                page_content = plugin()
            else:
                status = "error"
                page_content = [
                    html.H1(plugin.name),
                    alert(
                        "The selected run is not compatible with this analysis. "
                        "Select another run in the sidebar."
                    ),
                ]
            page = html.Div(
                page_content,
                id={"type": PERSISTENT_ANALYSIS_PAGE_TYPE, "plugin": active_plugin_id},
                className="persistent-analysis-page",
                style={"display": "block"},
            )
            mounted.append(active_plugin_id)
            statuses = {} if context_changed else dict(current.get("statuses", {}))
            statuses[active_plugin_id] = status
            next_state = {
                "run_id": active_run_id,
                "data_revision": data_revision,
                "mounted": mounted,
                "statuses": statuses,
                "invalidation": invalidation,
                "dirty": False,
            }
            if context_changed:
                return [page], next_state
            children = Patch()
            children.append(page)
            return children, next_state

        app.clientside_callback(
            """
            function(pathname, pageIds) {
                const parts = String(pathname || "").split("?")[0]
                    .split("/").filter(Boolean);
                const pluginId = parts.length >= 2 && parts[0] === "plugins"
                    ? parts[1]
                    : null;
                const persistentIds = new Set([
                    "hpo_planning",
                    "one_dimensional_analysis",
                    "two_dimensional_analysis",
                ]);
                const styles = (pageIds || []).map(function(pageId) {
                    return {display: pageId.plugin === pluginId ? "block" : "none"};
                });
                const isPersistentRoute = persistentIds.has(pluginId);
                const hasActivePage = isPersistentRoute && (pageIds || []).some(function(pageId) {
                    return pageId.plugin === pluginId;
                });
                if (hasActivePage) {
                    window.setTimeout(function() {
                        window.dispatchEvent(new Event("resize"));
                    }, 0);
                }
                const isGeneral = parts.length === 0;
                return [
                    styles,
                    {display: isPersistentRoute && !hasActivePage ? "block" : "none"},
                    {display: isGeneral ? "block" : "none"},
                    {display: !isPersistentRoute && !isGeneral ? "block" : "none"},
                ];
            }
            """,
            [
                Output({"type": PERSISTENT_ANALYSIS_PAGE_TYPE, "plugin": ALL}, "style"),
                Output(PERSISTENT_ANALYSIS_LOADING_ID, "style"),
                Output(GENERAL_CONTAINER_ID, "style"),
                Output("content", "style"),
            ],
            Input("on-page-load", "pathname"),
            Input({"type": PERSISTENT_ANALYSIS_PAGE_TYPE, "plugin": ALL}, "id"),
        )

        output = Output("content", "children")
        inputs = [
            Input("on-page-load", "pathname"),
            Input("global-run-selection-trigger", "data"),
        ]

        @app.callback(output, inputs, State("global-run-selector", "value"))  # type: ignore
        def display_page(pathname: str, _: Optional[int], active_run_id: Optional[str]) -> Any:
            """
            Display the page with the given path url, check for plugins.

            Parameters
            ----------
            pathname : str
                Pathname.

            Returns
            -------
            Any
                Either a General Layout, Not Found Layout or an Alert.
            """
            persistent_plugin_id = _plugin_id_from_path(pathname)
            if (
                persistent_plugin_id in PERSISTENT_ANALYSIS_IDS
                and persistent_plugin_id in self.plugins
            ):
                return no_update

            pathname = urlparse(pathname).path
            paths = pathname.split("/")[1:]

            if paths[0] == "":
                return no_update
            else:
                if not queue.ready():
                    return alert("At least one worker has to be enabled.")
                if paths[0] == "plugins":
                    plugin = self.plugins.get(paths[1], None)
                    if plugin is not None:
                        if active_run_id is None:
                            return alert("Please select a run in the sidebar first.")
                        try:
                            run_handler.set_active_run_id(active_run_id)
                            active_run = run_handler.get_run(active_run_id)
                        except RuntimeError:
                            return alert("The selected run is no longer available.")

                        if not plugin.check_run_compatibility(active_run):
                            return [
                                html.H1(plugin.name),
                                alert(
                                    "The selected run is not compatible with this "
                                    "analysis. Select another run in the sidebar."
                                ),
                            ]
                        return plugin()

            return self.not_found_layout(pathname)()

    def __call__(self) -> Component:  # noqa: D102
        return html.Div(
            children=[
                dcc.Interval(id="global-update", interval=config.REFRESH_RATE),
                planning_state_store(),
                dcc.Store(id=PLANNED_SEARCH_SPACE_HYDRATION_ID, data=None),
                dcc.Interval(
                    id="planned-search-space-hydration-on-load",
                    interval=1,
                    max_intervals=1,
                ),
                dcc.Store(id=PLANNED_SEARCH_SPACE_PERSISTENCE_ID, data=0),
                dcc.Store(id="two-dimensional-planning-mask-revision", data=0),
                dcc.Store(
                    id=PERSISTENT_ANALYSIS_STATE_ID,
                    data={
                        "run_id": None,
                        "data_revision": None,
                        "mounted": [],
                        "statuses": {},
                        "invalidation": {},
                        "dirty": False,
                    },
                    storage_type="memory",
                ),
                self.header_layout(),
                html.Div(
                    id="main-container",
                    className="container-fluid",
                    children=[
                        html.Div(
                            className="row",
                            children=[
                                self.sidebar_layout(),
                                html.Main(
                                    id="main-content",
                                    className="col-md-9 ms-sm-auto col-lg-10 px-md-4",
                                    children=[
                                        html.Div(
                                            className="",
                                            children=[
                                                dcc.Location(id="on-page-load", refresh=False),
                                                self.notification_layout(),
                                                html.Div(
                                                    "Loading analysis …",
                                                    id=PERSISTENT_ANALYSIS_LOADING_ID,
                                                    className="text-muted py-4",
                                                    style={"display": "none"},
                                                ),
                                                html.Div(
                                                    id=PERSISTENT_ANALYSIS_CACHE_ID,
                                                    children=[],
                                                ),
                                                html.Div(
                                                    self.general_layout(),
                                                    id=GENERAL_CONTAINER_ID,
                                                ),
                                                html.Div(id="content", style={"display": "none"}),
                                            ],
                                        )
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ]
        )
