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
# SidebarLayout

This module defines layout for the sidebar.

Callbacks are registered and handled.

## Classes
    - SidebarLayout: Customize the Sidebar Layout.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import dash_bootstrap_components as dbc
from dash import ALL, dcc, html, no_update
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Trigger

from deepcave import app, queue
from deepcave.layouts import Layout
from deepcave.plugins import Plugin
from deepcave.runs.group import Group
from deepcave.utils.global_objective import (
    DEFAULT_MAIN_OBJECTIVE,
    is_user_attr_objective,
    main_objective_options,
)


class SidebarLayout(Layout):
    """
    Customize the Sidebar Layout.

    Callbacks are registered and handled.

    Properties
    ----------
    plugins : Dict[str, List[Plugin]]
        A dictionary of all categorized plugins.
    nav_points : Dict[str, List[Tuple[str, str, str]]]
        A dictionary with plugins attributes corresponding to their category.
    """

    def __init__(self, categorized_plugins: Dict[str, List[Plugin]]) -> None:
        super().__init__()
        self.plugins = categorized_plugins

        nav_points: Dict[str, List[Tuple[str, str, str]]] = {
            category: [] for category in categorized_plugins
        }
        for category, plugins in categorized_plugins.items():
            for plugin in plugins:
                nav_points[category].append((plugin.id, plugin.name, plugin.icon))

        self.nav_points = nav_points

    def register_callbacks(self) -> None:
        """Register the callbacks for the sidebar layout."""
        # Collapse/expand the sidebar so wide analysis views can use the full width
        app.clientside_callback(
            """
            function(n_clicks) {
                return (n_clicks % 2 === 1)
                    ? "container-fluid sidebar-collapsed"
                    : "container-fluid";
            }
            """,
            Output("main-container", "className"),
            Input("sidebar-collapse-toggle", "n_clicks"),
        )

        # Update navigation items
        outputs = [
            Output("navigation-items", "children"),
            Output("general-navigation-link", "className"),
        ]
        input = Input("on-page-load", "pathname")

        @app.callback(outputs, input)  # type: ignore
        def update_navigation_items(pathname: str) -> Tuple[List[Component], str]:
            """
            Update the navigation items.

            Parameters
            ----------
            pathname : str
                The pathname.

            Returns
            -------
            List[Component]
                The navigation items.
            """
            layouts = []
            for category, points in self.nav_points.items():
                layouts += [
                    html.H6(
                        className="sidebar-heading d-flex justify-content-between "
                        "align-items-center px-3 mt-4 mb-1 text-muted",
                        children=[html.Span(category)],
                    )
                ]

                point_layouts = []
                for id, name, icon in points:
                    href = f"/plugins/{id}"
                    point_layouts += [
                        html.Li(
                            className="nav-item",
                            children=[
                                dcc.Link(
                                    [html.I(className=icon), name],
                                    className=f"nav-link {'active' if href in pathname else ''}",
                                    href=href,
                                )
                            ],
                        )
                    ]

                layouts += [html.Ul(className="nav flex-column", children=point_layouts)]

            general_class = f"nav-link {'active' if pathname == '/' else ''}"
            return layouts, general_class

        @app.callback(  # type: ignore
            [
                Output("global-run-selector", "options"),
                Output("global-run-selector", "disabled"),
                Output("global-run-selector", "value"),
            ],
            Input("global-update", "n_intervals"),
            State("global-run-selector", "options"),
            State("global-run-selector", "value"),
            State("global-run-selector", "disabled"),
        )
        def refresh_global_run_selector(
            _: int,
            current_options: List[Dict[str, Any]],
            current_run_id: Optional[str],
            current_disabled: bool,
        ) -> Any:
            """Refresh loaded runs while preserving a still-valid selection."""
            options, active_run_id = self._get_global_run_state()
            valid_ids = {option["value"] for option in options if option.get("value")}
            disabled = len(valid_ids) == 0

            # Re-emitting either identical options or an already selected value makes
            # dcc.Dropdown briefly clear its value.  The transient None used to trigger the page
            # callback and caused the content to oscillate between the plugin and its empty-state
            # alert.
            options_changed = options != (current_options or []) or disabled != current_disabled
            if not options_changed:
                return no_update, no_update, no_update

            # The browser owns an explicit dropdown selection.  In particular, refreshing the
            # option list must never re-emit or replace a valid choice.
            if current_run_id is None and active_run_id in valid_ids:
                value = active_run_id
            else:
                value = no_update

            return (
                options if options_changed else no_update,
                disabled if options_changed else no_update,
                value,
            )

        @app.callback(  # type: ignore
            Output("global-run-selection-trigger", "data"),
            Input("global-run-selector", "value"),
            State("global-run-selection-trigger", "data"),
        )
        def select_global_run(run_id: Optional[str], trigger: Optional[int]) -> int:
            """Persist an explicit selection made in the sidebar."""
            from deepcave import run_handler

            # dcc.Dropdown can emit a short-lived None while its options are reconciled.  It is
            # not a user selection and must neither clear persisted state nor rerender the page.
            if run_id is None:
                raise PreventUpdate()

            try:
                run_handler.set_active_run_id(run_id)
            except RuntimeError:
                raise PreventUpdate()

            return (trigger or 0) + 1

        @app.callback(  # type: ignore
            [
                Output("global-main-objective-selector", "options"),
                Output("global-main-objective-selector", "disabled"),
                Output("global-main-objective-selector", "value"),
            ],
            Input("global-run-selector", "value"),
            [
                State("global-main-objective-selector", "options"),
                State("global-main-objective-selector", "value"),
            ],
        )
        def refresh_global_main_objective(
            run_id: Optional[str],
            current_options: List[Dict[str, Any]],
            current_value: Any,
        ) -> Any:
            """Refresh objective choices when the selected run changes."""
            options = self._get_global_objective_options(run_id)
            valid_values = {option["value"] for option in options}
            disabled = not options
            value = current_value if current_value in valid_values else None
            if value is None and options:
                value = (
                    DEFAULT_MAIN_OBJECTIVE
                    if DEFAULT_MAIN_OBJECTIVE in valid_values
                    else options[0]["value"]
                )
            if options == (current_options or []) and current_value == value:
                return no_update, no_update, no_update
            return options, disabled, value

        @app.callback(  # type: ignore
            Output("global-objective-direction-container", "hidden"),
            Input("global-main-objective-selector", "value"),
        )
        def toggle_global_objective_direction(objective_value: Any) -> bool:
            """Show direction only for a derived user-attribute objective."""
            return not is_user_attr_objective(objective_value)

        @app.callback(  # type: ignore
            Output("global-main-objective-state", "data"),
            [
                Input("global-main-objective-selector", "value"),
                Input("global-objective-direction", "value"),
            ],
        )
        def update_global_main_objective_state(
            objective_value: Any, direction: Any
        ) -> Dict[str, Any]:
            """Publish the objective selection to every objective-driven plugin."""
            return {"value": objective_value, "direction": direction}

        # Callback to cancel jobs
        @app.callback(
            # Output('dropdown-container-output', 'children'),
            Input({"type": "cancel-job", "index": ALL}, "n_clicks"),
            State({"type": "cancel-job", "index": ALL}, "name"),
        )
        def delete_job(n_clicks, job_ids):  # type: ignore
            """Delete the job from the queue."""
            for n_click, job_id in zip(n_clicks, job_ids):
                if n_click is not None:
                    queue.delete_job(job_id)

        # Update queue information panel
        output = Output("queue-info", "children")

        @app.callback(output, Trigger("global-update", "n_intervals"))  # type: ignore
        def update_queue_info() -> List[Component]:
            """Update the information of the queue."""
            try:
                all_jobs = [
                    queue.get_finished_jobs(),
                    queue.get_running_jobs(),
                    queue.get_pending_jobs(),
                ]

                job_stati = ["[FINISHED]", "[RUNNING]", "[PENDING]"]

                collect = []
                for jobs, status in zip(all_jobs, job_stati):
                    for job in jobs:
                        name = job.meta["display_name"]
                        job_id = job.id
                        link = job.meta["link"]

                        collect += [(name, job_id, status, link)]

                items = []
                for name, job_id, status, link in collect:
                    items += [
                        html.Li(
                            # className="nav-item",
                            children=[
                                html.A(
                                    f"{status} {name}",
                                    href=link,
                                    style={
                                        "width": "100%",
                                        "display": "inline-block",
                                        "vertical-align": "middle",
                                    },
                                ),
                                html.Span(
                                    dbc.Button(
                                        "-",
                                        id={"type": "cancel-job", "index": name},
                                        name=job_id,
                                        color="danger",
                                        size="sm",
                                        disabled=True if status == job_stati[0] else False,
                                        style={
                                            "display": "inline-block",
                                            "padding": "0 0.4rem",
                                        },
                                        className="ms-1",
                                    ),
                                    style={
                                        "display": "table-cell",
                                        "vertical-align": "middle",
                                    },
                                ),
                            ],
                            className="nav-link",
                            style={"display": "flex", "flex-direction": "row"},
                        )
                    ]

                if len(collect) > 0:
                    return [
                        html.Hr(),
                        html.H6(
                            className="sidebar-heading d-flex justify-content-between "
                            "align-items-center px-3 mt-4 mb-1 text-muted",
                            children=[html.Span("Queue Information")],
                        ),
                        html.Ul(className="nav flex-column", children=items),
                    ]

                return []
            except Exception:
                return []

    def __call__(self) -> Union[List[Component], Component]:  # noqa: D102
        options, active_run_id = self._get_global_run_state()
        objective_options = self._get_global_objective_options(active_run_id)
        objective_values = {option["value"] for option in objective_options}
        objective_value = (
            DEFAULT_MAIN_OBJECTIVE
            if DEFAULT_MAIN_OBJECTIVE in objective_values
            else objective_options[0]["value"] if objective_options else None
        )
        return html.Nav(
            className="col-md-3 col-lg-2 d-md-block sidebar collapse",
            id="sidebarMenu",
            children=[
                html.Div(
                    className="position-sticky pt-3",
                    children=[
                        html.Ul(
                            className="nav flex-column",
                            children=[
                                dcc.Link(
                                    "General",
                                    id="general-navigation-link",
                                    className="nav-link",
                                    href="/",
                                )
                            ],
                        ),
                        html.Div(
                            [
                                dbc.Label("Run", className="small text-muted mb-1"),
                                dcc.Dropdown(
                                    id="global-run-selector",
                                    options=options,
                                    value=active_run_id,
                                    clearable=False,
                                    disabled=len(options) == 0,
                                    placeholder="Select run ...",
                                    # The application layout is created once when the server
                                    # starts. Persist the user's value in the browser so a page
                                    # reload cannot restore that original layout value.
                                    persistence=True,
                                    persistence_type="local",
                                ),
                                dcc.Store(id="global-run-selection-trigger", data=0),
                                html.Div(
                                    [
                                        dbc.Label(
                                            "Main Objective",
                                            className="small text-muted mb-1",
                                        ),
                                        dcc.Dropdown(
                                            id="global-main-objective-selector",
                                            options=objective_options,
                                            value=objective_value,
                                            clearable=False,
                                            disabled=not objective_options,
                                            placeholder="Select objective ...",
                                            persistence=True,
                                            persistence_type="local",
                                        ),
                                        html.Div(
                                            [
                                                dbc.Label(
                                                    "Direction",
                                                    className="small text-muted mb-1",
                                                ),
                                                dbc.Select(
                                                    id="global-objective-direction",
                                                    options=[
                                                        {"label": "Maximize", "value": "upper"},
                                                        {"label": "Minimize", "value": "lower"},
                                                    ],
                                                    value="upper",
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                            ],
                                            id="global-objective-direction-container",
                                            hidden=True,
                                            className="mt-2",
                                        ),
                                        dcc.Store(
                                            id="global-main-objective-state",
                                            data={
                                                "value": objective_value,
                                                "direction": "upper",
                                            },
                                        ),
                                    ],
                                    className="global-main-objective pt-3",
                                ),
                            ],
                            className="global-run-selector px-3 pt-2 pb-1",
                        ),
                        html.Div(id="navigation-items"),
                    ],
                ),
                html.Div(id="queue-info"),
            ],
        )

    @staticmethod
    def _get_global_run_state() -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Build global run options and reconcile the persisted selection."""
        from deepcave import run_handler

        loaded_runs = run_handler.get_runs(include_groups=True)
        runs = [run for run in loaded_runs if not isinstance(run, Group)]
        groups = [run for run in loaded_runs if isinstance(run, Group)]
        inputs = Plugin.load_run_inputs(runs, groups, lambda _: True)
        return inputs["run"]["options"], run_handler.get_active_run_id()

    @staticmethod
    def _get_global_objective_options(run_id: Optional[str]) -> List[Dict[str, Any]]:
        """Return global objective options for the selected run."""
        if run_id is None:
            return []
        try:
            from deepcave import run_handler
        except ImportError:
            # Runtime-only globals are deliberately unavailable in API/test mode.
            return []

        try:
            run = run_handler.get_run(run_id)
        except RuntimeError:
            return []
        return main_objective_options(run)
