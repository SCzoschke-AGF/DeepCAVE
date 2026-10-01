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
# GeneralLayout

This module provides the General Layout.

It handles different callbacks of the layout.

## Classes
    - GeneralLayout: Provide different utilities to handle callbacks for the layout.
"""

from typing import Any, Dict, List, Optional, Type

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import ALL, Input, Output, State
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import app, c, notification, queue, rc, run_handler
from deepcave.layouts import Layout
from deepcave.runs import NotMergeableError
from deepcave.runs.converters.optuna import OptunaRun
from deepcave.runs.run import Run
from deepcave.utils.hash import string_to_hash
from deepcave.utils.layout import help_button
from deepcave.utils.optuna import (
    SNAPSHOT_FILENAME,
    create_optuna_snapshot,
    get_optuna_snapshot_path,
    list_optuna_studies,
    resolve_optuna_database,
)
from deepcave.utils.util import short_string

OPTUNA_DISCOVERY_STORE_ID = "general-optuna-discovery"
GROUP_HYDRATION_STORE_ID = "general-group-hydration"
CACHE_CLEARED_TRIGGER_ID = "general-cache-cleared-trigger"


class GeneralLayout(Layout):
    """
    Provide different utilities to handle callbacks for the general layout.

    Properties
    ----------
    logger : Logger
        A logger used for debug information of the groups.
    """

    def register_callbacks(self) -> None:
        """Register if there were any callbacks."""
        self._callback_working_directory_changed()
        self._callback_discover_optuna_studies()
        self._callback_update_optuna_button()
        self._callback_display_available_runs()
        self._callback_display_selected_runs()
        self._callback_manage_run()
        self._callback_display_groups()
        self._callback_manage_groups()
        self._callback_clear_cache()

    def _callback_working_directory_changed(self) -> None:
        """Check for change in working directory."""
        inputs = [
            Input("general-working-directory-input", "value"),
            # Those inputs are for changing working directory quickly
            # (if someone is pressing on a directory)
            # Only works with "ALL"
            Input({"type": "general-dynamic-change-directory", "index": ALL}, "n_clicks"),
            Input("general-optuna-import-trigger", "data"),
            State({"type": "general-dynamic-available-run-path", "index": ALL}, "data"),
        ]

        outputs = [
            # Working directory input
            Output("general-working-directory-input", "value"),
            # Converter text
            Output("general-converter-label", "children"),
            Output("general-available-runs", "data"),
        ]

        # Register updates from inputs
        @app.callback(outputs, inputs)  # type: ignore
        def callback(
            working_dir: str,
            dynamic_n_clicks: List[Optional[int]],
            _optuna_import_trigger: int,
            dynamic_working_dirs: List[str],
        ):
            """
            Register updates from inputs.

            Parameters
            ----------
            working_dir : str
                The working directory.
            dynamic_n_clicks : List[Optional[int]]
                The number of clicks.
            dynamic_working_dirs : List[str]
                The working directories.
            """
            # `working_dir` is only none on page load
            if working_dir is None:
                # Handler working directory
                working_dir = run_handler.get_working_directory()
                converter = run_handler.available_run_classes
            else:
                for dir, n_clicks in zip(dynamic_working_dirs, dynamic_n_clicks):
                    if n_clicks is not None:
                        working_dir = dir
                        break

                run_handler.set_working_directory(working_dir)
                converter = run_handler.available_run_classes

            return (
                str(working_dir),
                self.get_converter_text(converter),
                run_handler.get_available_run_paths(),
            )

    def _callback_discover_optuna_studies(self) -> None:
        """Discover studies whenever the SQLite database path changes."""
        outputs = [
            Output("general-optuna-study-dropdown", "options"),
            Output("general-optuna-study-dropdown", "value"),
            Output("general-optuna-database-status", "children"),
            Output("general-optuna-database-status", "className"),
            Output(OPTUNA_DISCOVERY_STORE_ID, "data"),
        ]
        inputs = [
            Input("general-optuna-database-input", "value"),
            Input("general-working-directory-input", "value"),
            State("general-optuna-study-dropdown", "value"),
        ]

        @app.callback(outputs, inputs)  # type: ignore
        def callback(database_value: str, working_dir: str, selected_study: Optional[str]):
            """Populate the study dropdown from a local Optuna database."""
            if database_value is None or working_dir is None:
                return [], None, "Studies", "mt-3", None

            try:
                database = resolve_optuna_database(database_value, Path(working_dir))
                study_names = list_optuna_studies(database)
            except Exception as exception:
                discovery = {
                    "database": None,
                    "studies": [],
                    "error": str(exception),
                }
                return [], None, str(exception), "mt-3 text-danger", discovery

            if len(study_names) == 0:
                message = "Studies (0 studies included)"
                discovery = {"database": str(database), "studies": [], "error": None}
                return [], None, message, "mt-3 text-warning", discovery

            options = [
                {
                    "label": study_name,
                    "value": study_name,
                }
                for study_name in study_names
            ]
            if selected_study not in study_names:
                selected_study = study_names[0] if len(study_names) == 1 else None

            noun = "study" if len(study_names) == 1 else "studies"
            message = f"Studies ({len(study_names)} {noun} included)"
            discovery = {
                "database": str(database),
                "studies": study_names,
                "error": None,
            }
            return options, selected_study, message, "mt-3", discovery

    def _callback_update_optuna_button(self) -> None:
        """Update the Optuna import button based on the selected snapshot."""
        outputs = [
            Output("general-optuna-load-button", "children"),
            Output("general-optuna-load-button", "disabled"),
        ]
        inputs = [
            Input("general-optuna-database-input", "value"),
            Input("general-optuna-study-dropdown", "value"),
            Input("general-working-directory-input", "value"),
            Input("general-optuna-import-trigger", "data"),
            Input(OPTUNA_DISCOVERY_STORE_ID, "data"),
        ]

        @app.callback(outputs, inputs)  # type: ignore
        def callback(
            database_value: str,
            study_name: Optional[str],
            working_dir: str,
            _optuna_import_trigger: int,
            discovery: Any,
        ):
            """Choose the load or refresh state for the import button."""
            if database_value is None or study_name is None or working_dir is None:
                return "Load Optuna Study", True

            try:
                database = resolve_optuna_database(database_value, Path(working_dir))
                discovered_database = (
                    discovery.get("database") if isinstance(discovery, dict) else None
                )
                study_names = discovery.get("studies", []) if isinstance(discovery, dict) else []
                if str(database) != discovered_database or study_name not in study_names:
                    return "Load Optuna Study", True
                run_path = get_optuna_snapshot_path(database, study_name, Path(working_dir))
            except Exception:
                return "Load Optuna Study", True

            if (run_path / SNAPSHOT_FILENAME).is_file():
                return "Refresh from Optuna", False

            return "Load Optuna Study", False

    def _callback_display_available_runs(self) -> None:
        """Display available runs."""
        output = Output("general-available-runs-container", "children")
        input = Input("general-available-runs", "data")

        @app.callback(output, input)  # type: ignore
        def callback(run_paths: List[str]):
            """
            Display the available runs.

            Parameters
            ----------
            run_paths : List[str]
                The run paths.
            """
            children = []

            # Add text to go to parent directory
            new_element = html.Div(
                [
                    html.I(
                        className="fas fa-folder-open fa-lg",
                        id={"type": "general-dynamic-add-run", "index": -1},
                        style={"pointer-events": "none"},
                    ),
                    dbc.Button(
                        "..",
                        id={"type": "general-dynamic-change-directory", "index": -1},
                        color="link",
                    ),
                    dcc.Store(
                        id={"type": "general-dynamic-available-run-path", "index": -1},
                        data=str(Path(run_handler.get_working_directory()).parent.absolute()),
                    ),
                ],
                className="mb-1",
            )
            children.append(new_element)

            for i, run_path in enumerate(run_paths):
                run_name = run_handler.get_run_name(run_path)

                is_run = run_handler.is_run(run_path)
                # Differenciate between run and directory for visibility and usability reasons
                if is_run:
                    new_element = html.Div(
                        [
                            dbc.Button(
                                "+", id={"type": "general-dynamic-add-run", "index": i}, size="sm"
                            ),
                            dbc.Button(
                                run_name,
                                id={"type": "general-dynamic-change-directory", "index": i},
                                color="light",
                                disabled=True,
                            ),
                            dcc.Store(
                                id={"type": "general-dynamic-available-run-path", "index": i},
                                data=run_path,
                            ),
                        ],
                        className="mb-1",
                    )
                else:
                    new_element = html.Div(
                        [
                            html.I(className="fas fa-folder fa-lg"),
                            dbc.Button(
                                run_name,
                                id={"type": "general-dynamic-change-directory", "index": i},
                                color="link",
                            ),
                            dcc.Store(
                                id={"type": "general-dynamic-available-run-path", "index": i},
                                data=run_path,
                            ),
                        ],
                        className="mb-1",
                    )
                children.append(new_element)
            if len(children) == 0:
                return html.Div("No runs found.")

            return children

    def _callback_display_selected_runs(self) -> None:
        """Display the selected runs."""
        output = Output("general-selected-runs-container", "children")
        input = Input("general-selected-runs", "data")

        @app.callback(output, input)  # type: ignore
        def callback(run_paths: List[str]):
            """
            Display the selected runs.

            Parameters
            ----------
            run_paths : List[str]
                The run paths.
            """
            children = []

            for i, run_path in enumerate(run_paths):
                run_name = run_handler.get_run_name(run_path)

                shortened_run_path = short_string(run_path, 30, mode="prefix")

                new_element = html.Div(
                    [
                        dbc.Button(
                            "-",
                            id={"type": "general-dynamic-remove-run", "index": i},
                            color="danger",
                        ),
                        html.Span(
                            [
                                run_name,
                                html.I(
                                    f"({shortened_run_path})",
                                    style={"font-size": "10px", "margin-left": "6px"},
                                ),
                            ],
                            # color="light",
                            style={"margin-left": "12px"},
                        ),
                        dcc.Store(
                            id={"type": "general-dynamic-selected-run-path", "index": i},
                            data=run_path,
                        ),
                    ],
                    className="mb-1",
                )
                children.append(new_element)

            if len(children) == 0:
                return html.Div("No runs selected.")

            return children

    def _callback_manage_run(self) -> None:
        """Manage runs, including adding and removing them."""
        outputs = [
            Output({"type": "general-dynamic-add-run", "index": ALL}, "n_clicks"),
            Output({"type": "general-dynamic-remove-run", "index": ALL}, "n_clicks"),
            Output("general-selected-runs", "data"),
            Output("general-optuna-import-trigger", "data"),
            Output("general-optuna-load-button", "n_clicks"),
        ]

        inputs = [
            Input({"type": "general-dynamic-add-run", "index": ALL}, "n_clicks"),
            Input({"type": "general-dynamic-remove-run", "index": ALL}, "n_clicks"),
            Input("general-optuna-load-button", "n_clicks"),
        ]

        states = [
            State({"type": "general-dynamic-available-run-path", "index": ALL}, "data"),
            State({"type": "general-dynamic-selected-run-path", "index": ALL}, "data"),
            State("general-optuna-database-input", "value"),
            State("general-optuna-study-dropdown", "value"),
            State("general-optuna-import-trigger", "data"),
        ]

        @app.callback(outputs, inputs, states)  # type: ignore
        def callback(
            add_n_clicks,
            remove_n_clicks,
            optuna_n_clicks,
            available_run_paths,
            selected_run_paths,
            database_value,
            study_name,
            optuna_import_trigger,
        ):
            """Add and remove runs."""
            # Add run path
            for n_click, run_path in zip(add_n_clicks, available_run_paths):
                if n_click is not None:
                    success = run_handler.add_run(run_path)
                    if not success:
                        notification.update("The run could not be added.")

            # Remove run path
            for n_click, run_path in zip(remove_n_clicks, selected_run_paths):
                if n_click is not None:
                    run_handler.remove_run(run_path)

            if optuna_n_clicks is not None:
                try:
                    if database_value is None or study_name is None:
                        raise ValueError("Select a SQLite database and an Optuna study first.")

                    working_dir = run_handler.get_working_directory()
                    database = resolve_optuna_database(database_value, working_dir)
                    # Discovery is a UI snapshot. Validate against the source immediately before
                    # the mutating import so a changed SQLite/WAL cannot make it stale.
                    if study_name not in list_optuna_studies(database):
                        raise ValueError(f'Optuna study "{study_name}" no longer exists.')
                    target_path = get_optuna_snapshot_path(database, study_name, working_dir)
                    target_path_str = str(target_path)
                    run_id = string_to_hash(f"{OptunaRun.prefix}:{target_path}")

                    if queue.has_active_jobs_for_run(run_id):
                        notification.update(
                            "The Optuna run still has pending or running jobs. "
                            "Wait for them or cancel them before refreshing.",
                            "warning",
                        )
                    else:
                        result = create_optuna_snapshot(database, study_name, working_dir)
                        queue.delete_inactive_jobs_for_run(run_id)

                        if target_path_str in run_handler.get_selected_run_paths():
                            if target_path_str in run_handler.runs:
                                run_handler.update()
                                success = target_path_str in run_handler.runs
                            else:
                                success = run_handler.update_runs()
                        else:
                            success = run_handler.add_run(target_path_str)

                        if success:
                            run_handler.update_groups()
                            c.set("last_inputs", value={})
                            optuna_import_trigger = (optuna_import_trigger or 0) + 1
                            action = "refreshed" if result.refreshed else "loaded"
                            notification.update(
                                f'Optuna study "{study_name}" {action}: '
                                f"{result.imported_trials} trials imported, "
                                f"{result.skipped_trials} unfinished trials skipped.",
                                "success",
                            )
                        else:
                            notification.update("The imported Optuna run could not be selected.")
                except Exception as exception:
                    self.logger.exception("Could not import Optuna study.")
                    notification.update(str(exception))

            # Reset clicks
            add_n_clicks = [None for _ in add_n_clicks]
            remove_n_clicks = [None for _ in remove_n_clicks]

            # Remove last inputs
            c.set("last_inputs", value={})

            return (
                add_n_clicks,
                remove_n_clicks,
                run_handler.get_selected_run_paths(),
                optuna_import_trigger or 0,
                None,
            )

    def _callback_display_groups(self) -> None:
        """Responsible of displaying the groups."""
        outputs = [
            Output("general-group-container", "children"),
            Output("general-add-group", "n_clicks"),
            Output(GROUP_HYDRATION_STORE_ID, "data"),
        ]
        inputs = [
            Input("general-add-group", "n_clicks"),
            Input("general-selected-runs-container", "children"),
            Input("general-group-trigger", "data"),
            State("general-group-container", "children"),
        ]

        # Let's take care of the groups here
        @app.callback(outputs, inputs)  # type: ignore
        def callback(n_clicks: int, _trigger1, _trigger2, children):
            """Display groups."""

            def get_layout(
                index: int,
                options: Dict[str, str],
                input_value: str = "",
                dropdown_value: Optional[List[str]] = None,
                require_configspace_congruence: bool = True,
            ) -> Component:
                """
                Get the layout for the groups.

                Parameters
                ----------
                index : int
                    The index.
                options : Dict[str, str]
                    The options.
                input_value : str, optional
                    The input value, by default "".
                dropdown_value : Optional[List[str]], optional
                    The dropdown value, by default None.
                require_configspace_congruence : bool, optional
                    Whether identical search spaces are required, by default True.

                Returns
                -------
                Component
                    The layout for the groups.
                """
                if dropdown_value is None:
                    dropdown_value = []

                return html.Div(
                    [
                        dbc.Input(
                            id={"type": "group-name", "index": index},
                            placeholder="Name",
                            type="text",
                            value=input_value,
                            style={"margin-bottom": "-1px"},
                        ),
                        dcc.Dropdown(
                            id={"type": "group-dropdown", "index": index},
                            options=[{"label": v, "value": k} for k, v in options.items()],
                            value=dropdown_value,
                            multi=True,
                        ),
                        html.Div(
                            [
                                dbc.Checkbox(
                                    id={
                                        "type": "group-require-configspace-congruence",
                                        "index": index,
                                    },
                                    label="Require identical search spaces",
                                    value=require_configspace_congruence,
                                ),
                                help_button(
                                    "Disable this only for runs of the same model. "
                                    "Hyperparameter names, types and scales must still match; "
                                    "numeric ranges will be combined."
                                ),
                            ],
                            className="d-flex align-items-center mt-1",
                        ),
                    ],
                    className="mb-2",
                )

            groups = run_handler.get_selected_groups()
            configspace_congruence = run_handler.get_groups_configspace_congruence()
            selected_run_paths = run_handler.get_selected_run_paths()
            selected_runs = {p: run_handler.get_run_name(p) for p in selected_run_paths}
            index = 0

            # Load from cache if page is loaded
            children = []

            for name, paths in groups.items():
                if name is None:
                    continue

                children.append(
                    get_layout(
                        index,
                        selected_runs,
                        name,
                        paths,
                        configspace_congruence.get(name, True),
                    )
                )

                index += 1

            if n_clicks is not None and len(selected_runs) > 0:
                children.append(get_layout(index, selected_runs))

            return (
                children,
                None,
                {
                    "groups": groups,
                    "congruence": configspace_congruence,
                },
            )

    def _callback_manage_groups(self) -> None:
        """Manage the groups."""
        outputs = Output("general-group-trigger", "data")
        inputs = [
            Input({"type": "group-name", "index": ALL}, "value"),
            Input({"type": "group-dropdown", "index": ALL}, "value"),
            Input(
                {"type": "group-require-configspace-congruence", "index": ALL},
                "value",
            ),
            State("general-group-trigger", "data"),
            State(GROUP_HYDRATION_STORE_ID, "data"),
        ]

        @app.callback(outputs, inputs, prevent_initial_call=True)  # type: ignore
        def callback(group_names: List[str], all_run_paths, congruence_values, i, hydration):
            """Manage the groups."""
            # For the default group names, if no name was entered
            group_counter = 0

            groups = {}
            configspace_congruence = {}
            congruence_values = list(congruence_values or [])
            congruence_values.extend([True] * (len(group_names) - len(congruence_values)))
            for group_name, run_paths, require_congruence in zip(
                group_names, all_run_paths, congruence_values
            ):
                if group_name is None or group_name == "":
                    # Set the default group name with a counter,
                    # so the groups dont overwrite themselves
                    group_name_unavailable = True

                    groups_cache = c.get("groups")
                    if groups_cache is None:
                        continue

                    # Check to see that no group name that already exists
                    # gets picked
                    while group_name_unavailable:
                        group_name = f"Group {group_counter}"
                        assert groups_cache is not None
                        if group_name not in groups_cache.keys():
                            group_name_unavailable = False
                        else:
                            group_counter += 1

                if run_paths is None or len(run_paths) == 0:
                    continue

                valid_run_paths = []
                for run_path in run_paths:
                    if run_path in run_handler.get_selected_run_paths():
                        valid_run_paths.append(run_path)

                if len(valid_run_paths) == 0:
                    continue

                groups[group_name] = valid_run_paths
                configspace_congruence[group_name] = bool(require_congruence)

            # Sort the groups alphabetically, so when added
            # they appear ordered
            sorted_groups = dict(sorted(groups.items()))
            stored_groups = hydration.get("groups", {}) if isinstance(hydration, dict) else {}
            stored_congruence = (
                hydration.get("congruence", {}) if isinstance(hydration, dict) else {}
            )
            effective_stored_congruence = {
                name: bool(stored_congruence.get(name, True)) for name in stored_groups
            }
            if (
                sorted_groups == stored_groups
                and configspace_congruence == effective_stored_congruence
            ):
                raise PreventUpdate()

            try:
                # Now save it
                run_handler.update_groups(sorted_groups, configspace_congruence)
            except NotMergeableError as exception:
                notification.update(str(exception))

                # This will automatically trigger the group display s.t. the selection is redo.
                return i + 1

            self.logger.debug(f"Groups: {sorted_groups}")
            return (i or 0) + 1

    def _callback_clear_cache(self) -> None:
        """Responsible for clearing the cache."""
        output = [
            Output("general-clear-cache-button", "n_clicks"),
            Output(CACHE_CLEARED_TRIGGER_ID, "data"),
        ]
        input = Input("general-clear-cache-button", "n_clicks")
        state = State(CACHE_CLEARED_TRIGGER_ID, "data")

        @app.callback(output, input, state)  # type: ignore
        def callback(n_clicks, revision):
            """Clear the cache, stop all running jobs, create new run caches."""
            if n_clicks is not None:
                rc.clear()
                from deepcave.plugins.hyperparameter.analysis_utils import (
                    clear_trial_table_cache,
                )
                from deepcave.plugins.hyperparameter.one_dimensional_analysis import (
                    clear_one_dimensional_figure_cache,
                )

                clear_trial_table_cache()
                clear_one_dimensional_figure_cache()

                # Also remove last inputs
                c.set("last_inputs", value={})

                # Also: All the running jobs have to be stopped
                queue.delete_jobs()

                # `update_runs` has to be called now to create the run caches again
                run_handler.update_runs()
                notification.update("Plugin caches successfully cleared.", "success")

                revision = (revision or 0) + 1

            return None, revision or 0

    @staticmethod
    def get_converter_text(converters: List[Type[Run]]) -> html.Div:
        """
        Get the text for the available run converters.

        Parameters
        ----------
        converters : List[Type[Run]]
            A list of the avaialble run converters.

        Returns
        -------
        html.Div
            The text with all the available converters.
        """
        converter_text = []
        for converter in converters:
            converter_text += [converter.prefix]

        return html.Div(
            html.Span(f"Available run converters: {', '.join(converter_text)}"),
            className="mt-2",
        )

    def __call__(self) -> List[Component]:  # noqa: D102
        return [
            html.H1("General"),
            # Working dir
            dbc.Label("Working Directory"),
            help_button("Relative and absolute paths are supported."),
            dbc.InputGroup(
                [
                    dbc.Input(
                        id="general-working-directory-input",
                        placeholder="",
                        type="text",
                        debounce=True,
                    ),
                    dbc.DropdownMenu(
                        html.Div(id="general-available-runs-container", className="px-2 py-1"),
                        label="Browse",
                        addon_type="append",
                        align_end=True,
                        color="secondary",
                    ),
                ]
            ),
            dbc.FormText(id="general-converter-label"),
            dcc.Store(id="general-available-runs", data={}),
            dcc.Store(id="general-optuna-import-trigger", data=0),
            dcc.Store(id=OPTUNA_DISCOVERY_STORE_ID, data=None),
            dcc.Store(id=GROUP_HYDRATION_STORE_ID, data=None),
            dcc.Store(id=CACHE_CLEARED_TRIGGER_ID, data=0),
            dbc.Card(
                [
                    dbc.CardHeader("Load Optuna Study"),
                    dbc.CardBody(
                        [
                            dbc.Label("SQLite database"),
                            dbc.Input(
                                id="general-optuna-database-input",
                                type="text",
                                value="optuna.db",
                                debounce=True,
                            ),
                            dbc.Label(
                                "Studies",
                                id="general-optuna-database-status",
                                className="mt-3",
                            ),
                            dcc.Dropdown(id="general-optuna-study-dropdown", options=[]),
                            dbc.Button(
                                "Load Optuna Study",
                                id="general-optuna-load-button",
                                color="primary",
                                className="mt-3",
                                disabled=True,
                            ),
                        ]
                    ),
                ],
                className="mt-3 mb-3",
            ),
            html.Hr(),
            # Selected Runs
            html.H2("Selected Runs"),
            dcc.Store(id="general-selected-runs", data={}),
            html.Div(id="general-selected-runs-container", className="mt-2"),
            html.Hr(),
            # Groups
            html.H2("Groups"),
            html.Div(id="general-group-container", children=[]),
            dbc.Button("Add Group", id="general-add-group"),
            dcc.Store(id="general-group-trigger", data=0),
            html.Hr(),
            # Cache
            html.H2("Caches"),
            dbc.Button("Clear Plugin Caches", id="general-clear-cache-button", color="primary"),
        ]
