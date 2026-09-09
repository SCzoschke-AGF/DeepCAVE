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

from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component

from deepcave import app, config, queue, run_handler
from deepcave.layouts import Layout
from deepcave.layouts.general import GeneralLayout
from deepcave.layouts.header import HeaderLayout
from deepcave.layouts.not_found import NotFoundLayout
from deepcave.layouts.notification import NotificationLayout
from deepcave.layouts.sidebar import SidebarLayout
from deepcave.plugins import Plugin
from deepcave.utils.dash import alert


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
        output = Output("content", "children")
        inputs = [
            Input("on-page-load", "pathname"),
            Input("global-run-selection-trigger", "data"),
        ]

        @app.callback(output, inputs, State("global-run-selector", "value"))  # type: ignore
        def display_page(
            pathname: str, _: Optional[int], active_run_id: Optional[str]
        ) -> Any:
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
            pathname = urlparse(pathname).path
            paths = pathname.split("/")[1:]

            if paths[0] == "":
                return self.general_layout()
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
                                                html.Div(id="content"),
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
