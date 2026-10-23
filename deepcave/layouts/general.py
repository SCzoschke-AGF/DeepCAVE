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

from typing import Any, Dict, List, Optional

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import ctx, dcc, html, no_update
from dash.dependencies import ALL, Input, Output, State
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import app, c, notification, queue, rc, run_handler
from deepcave.layouts import Layout
from deepcave.runs import NotMergeableError, Status
from deepcave.runs.converters.optuna import OptunaRun
from deepcave.utils.hash import string_to_hash
from deepcave.utils.layout import help_button
from deepcave.utils.optuna import (
    SNAPSHOT_FILENAME,
    create_optuna_snapshot,
    get_optuna_snapshot_path,
    list_optuna_studies,
    resolve_optuna_database,
)

OPTUNA_DISCOVERY_STORE_ID = "general-optuna-discovery"
GROUP_HYDRATION_STORE_ID = "general-group-hydration"
CACHE_CLEARED_TRIGGER_ID = "general-cache-cleared-trigger"
SQLITE_EXTENSIONS = {".db", ".sqlite", ".sqlite3"}


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
        self._callback_browse_optuna_database()
        self._callback_discover_optuna_studies()
        self._callback_update_optuna_button()
        self._callback_display_selected_runs()
        self._callback_manage_run()
        self._callback_display_groups()
        self._callback_manage_groups()
        self._callback_clear_cache()

    def _callback_browse_optuna_database(self) -> None:
        """Browse server-side directories without changing the run working directory."""
        outputs = [
            Output("general-optuna-browser", "is_open"),
            Output("general-optuna-browser-directory", "value"),
            Output("general-optuna-browser-entries", "children"),
            Output("general-optuna-database-input", "value"),
            Output("general-optuna-browser-error", "children"),
        ]
        inputs = [
            Input("general-optuna-browse-button", "n_clicks"),
            Input("general-optuna-browser-parent", "n_clicks"),
            Input("general-optuna-browser-close", "n_clicks"),
            Input("general-optuna-browser-directory", "value"),
            Input({"type": "general-optuna-browser-entry", "path": ALL}, "n_clicks"),
        ]
        states = [State("general-optuna-database-input", "value")]

        @app.callback(outputs, inputs, states, prevent_initial_call=True)  # type: ignore
        def callback(_browse, _parent, _close, directory, entry_clicks, database_value):
            """Navigate directories or select a validated local SQLite file."""
            trigger = ctx.triggered_id
            if trigger == "general-optuna-browser-close":
                return False, no_update, no_update, no_update, ""

            try:
                if trigger == "general-optuna-browse-button":
                    working_dir = run_handler.get_working_directory()
                    try:
                        directory = resolve_optuna_database(
                            database_value or "", working_dir
                        ).parent
                    except (OSError, ValueError):
                        directory = working_dir
                elif trigger == "general-optuna-browser-parent":
                    directory = Path(directory).expanduser().parent
                elif isinstance(trigger, dict):
                    # Replacing the listing also changes the wildcard inputs without a click.
                    if not any(entry_clicks):
                        raise PreventUpdate
                    path = Path(trigger["path"])
                    if not path.is_dir():
                        if path.suffix.lower() not in SQLITE_EXTENSIONS:
                            raise ValueError("Select a .db, .sqlite or .sqlite3 file.")
                        database = resolve_optuna_database(
                            str(path), run_handler.get_working_directory()
                        )
                        return False, no_update, no_update, str(database), ""
                    directory = path
                elif trigger != "general-optuna-browser-directory" or not directory:
                    raise PreventUpdate

                path = Path(directory).expanduser().resolve(strict=True)
                if not path.is_dir():
                    raise ValueError(f"Not a directory: {path}")
                entries = []
                for entry in path.iterdir():
                    if entry.is_dir():
                        entries.append((True, entry))
                    elif entry.is_file() and entry.suffix.lower() in SQLITE_EXTENSIONS:
                        entries.append((False, entry))
                entries.sort(key=lambda item: (not item[0], item[1].name.casefold()))
                children = [
                    dbc.Button(
                        [
                            html.I(className=f"fas fa-{'folder' if is_dir else 'database'} me-2"),
                            entry.name,
                        ],
                        id={"type": "general-optuna-browser-entry", "path": str(entry)},
                        n_clicks=0,
                        color="link",
                        className="text-start text-break w-100",
                        title=str(entry),
                    )
                    for is_dir, entry in entries
                ]
                if not children:
                    children = [html.Div("No folders or SQLite files found.", className="p-2")]
                return True, str(path), children, no_update, ""
            except (OSError, ValueError) as exception:
                return True, no_update, [], no_update, str(exception)

    def _callback_discover_optuna_studies(self) -> None:
        """Discover studies whenever the SQLite database path changes."""
        outputs = [
            Output("general-optuna-study-dropdown", "options"),
            Output("general-optuna-study-dropdown", "value"),
            Output("general-optuna-database-status", "children"),
            Output("general-optuna-database-status", "className"),
            Output(OPTUNA_DISCOVERY_STORE_ID, "data"),
            Output("general-optuna-study-dropdown", "placeholder"),
            Output("general-optuna-study-dropdown", "disabled"),
        ]
        inputs = [
            Input("general-optuna-database-input", "value"),
            Input("general-working-directory", "data"),
            State("general-optuna-study-dropdown", "value"),
            State(OPTUNA_DISCOVERY_STORE_ID, "data"),
        ]

        @app.callback(outputs, inputs)  # type: ignore
        def callback(
            database_value: str,
            working_dir: str,
            selected_study: Optional[str],
            previous_discovery: Any,
        ):
            """Populate the study dropdown from a local Optuna database."""
            placeholder = "Select a study"
            if not database_value or not database_value.strip() or working_dir is None:
                return [], None, "", "d-none", None, placeholder, True

            try:
                database = resolve_optuna_database(database_value, Path(working_dir))
                study_names = list_optuna_studies(database)
            except Exception as exception:
                discovery = {
                    "database": None,
                    "studies": [],
                    "error": str(exception),
                }
                return (
                    [],
                    None,
                    str(exception),
                    "small text-danger mt-1",
                    discovery,
                    placeholder,
                    True,
                )

            if len(study_names) == 0:
                message = "No Optuna studies found in this database."
                discovery = {"database": str(database), "studies": [], "error": message}
                return [], None, message, "small text-warning mt-1", discovery, placeholder, True

            options = [
                {
                    "label": study_name,
                    "value": study_name,
                }
                for study_name in study_names
            ]
            if (
                not isinstance(previous_discovery, dict)
                or previous_discovery.get("database") != str(database)
                or previous_discovery.get("input") != database_value
                or selected_study not in study_names
            ):
                selected_study = None

            discovery = {
                "database": str(database),
                "input": database_value,
                "studies": study_names,
                "error": None,
            }
            placeholder = f"Select a study ({len(study_names)} available)"
            return options, selected_study, "", "d-none", discovery, placeholder, False

    def _callback_update_optuna_button(self) -> None:
        """Update the Optuna import button based on the selected snapshot."""
        outputs = [
            Output("general-optuna-load-button", "children"),
            Output("general-optuna-load-button", "disabled"),
        ]
        inputs = [
            Input("general-optuna-database-input", "value"),
            Input("general-optuna-study-dropdown", "value"),
            Input("general-working-directory", "data"),
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
                return "Load Study", True

            try:
                database = resolve_optuna_database(database_value, Path(working_dir))
                discovered_database = (
                    discovery.get("database") if isinstance(discovery, dict) else None
                )
                study_names = discovery.get("studies", []) if isinstance(discovery, dict) else []
                if (
                    str(database) != discovered_database
                    or discovery.get("input") != database_value
                    or study_name not in study_names
                ):
                    return "Load Study", True
                run_path = get_optuna_snapshot_path(database, study_name, Path(working_dir))
            except Exception:
                return "Load Study", True

            if (run_path / SNAPSHOT_FILENAME).is_file():
                return "Refresh Study", False

            return "Load Study", False

    def _callback_display_selected_runs(self) -> None:
        """Display the selected runs."""
        outputs = [
            Output("general-selected-runs-container", "children"),
            Output("general-selected-runs-count", "children"),
        ]
        input = Input("general-selected-runs", "data")

        @app.callback(outputs, input)  # type: ignore
        def callback(run_paths: List[str]):
            """
            Display the selected runs.

            Parameters
            ----------
            run_paths : List[str]
                The run paths.
            """
            rows = []

            for i, run_path in enumerate(run_paths or []):
                run_name = run_handler.get_run_name(run_path)
                run = run_handler.runs.get(run_path)
                successful_trials = (
                    sum(trial.status == Status.SUCCESS for trial in run.history)
                    if run is not None
                    else "—"
                )
                hyperparameters = len(run.configspace) if run is not None else "—"
                sampler = (
                    (run.get_meta().get("optuna_sampler") or "Unknown") if run is not None else "—"
                )

                rows.append(
                    html.Tr(
                        [
                            html.Td(
                                [
                                    html.Div(run_name, className="general-run-name"),
                                ],
                            ),
                            html.Td(successful_trials),
                            html.Td(hyperparameters),
                            html.Td(sampler),
                            html.Td(
                                [
                                    dbc.Button(
                                        [
                                            html.I(
                                                className="far fa-trash-alt",
                                                **{"aria-hidden": "true"},
                                            ),
                                            html.Span(
                                                f"Remove {run_name}", className="visually-hidden"
                                            ),
                                        ],
                                        id={"type": "general-dynamic-remove-run", "index": i},
                                        color="danger",
                                        outline=True,
                                        size="sm",
                                        title=f"Remove {run_name}",
                                        className="general-run-remove",
                                    ),
                                    dcc.Store(
                                        id={
                                            "type": "general-dynamic-selected-run-path",
                                            "index": i,
                                        },
                                        data=run_path,
                                    ),
                                ],
                                className="general-run-actions",
                            ),
                        ],
                    )
                )

            count = len(rows)
            if not rows:
                rows = [
                    html.Tr(html.Td("No runs selected.", colSpan=5, className="general-runs-empty"))
                ]

            table = html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Run Name", scope="col"),
                                html.Th(
                                    "Successful Trials", scope="col", className="general-run-trials"
                                ),
                                html.Th(
                                    "Hyperparameters",
                                    scope="col",
                                    className="general-run-hyperparameters",
                                ),
                                html.Th("Sampler", scope="col", className="general-run-sampler"),
                                html.Th("Actions", scope="col", className="general-run-actions"),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                className="general-runs-table",
                **{"aria-label": "Selected runs"},
            )
            return table, f"{count} {'run' if count == 1 else 'runs'}"

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
            runs_changed = False

            # Add run path
            for n_click, run_path in zip(add_n_clicks, available_run_paths):
                if n_click is not None:
                    success = run_handler.add_run(run_path)
                    if not success:
                        notification.update("The run could not be added.")
                    else:
                        runs_changed = True

            # Remove run path
            for n_click, run_path in zip(remove_n_clicks, selected_run_paths):
                if n_click is not None:
                    run_handler.remove_run(run_path)
                    runs_changed = True

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
                            runs_changed = True
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

            if runs_changed:
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
                            size="sm",
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
                    className="general-group-editor",
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
                from deepcave.plugins.hyperparameter.configuration_neighborhood import (
                    clear_configuration_neighborhood_cache,
                )
                from deepcave.plugins.hyperparameter.one_dimensional_analysis import (
                    clear_one_dimensional_figure_cache,
                )
                from deepcave.plugins.hyperparameter.near_optimal_configurations import (
                    clear_near_optimal_analysis_cache,
                )
                from deepcave.plugins.hyperparameter.search_space_coverage import (
                    clear_coverage_cache,
                )

                clear_trial_table_cache()
                clear_configuration_neighborhood_cache()
                clear_one_dimensional_figure_cache()
                clear_near_optimal_analysis_cache()
                clear_coverage_cache()

                # Also remove last inputs
                c.set("last_inputs", value={})

                # Also: All the running jobs have to be stopped
                queue.delete_jobs()

                # `update_runs` has to be called now to create the run caches again
                run_handler.update_runs()
                notification.update("Plugin caches successfully cleared.", "success")

                revision = (revision or 0) + 1

            return None, revision or 0

    def __call__(self) -> List[Component]:  # noqa: D102
        return [
            html.H1("General"),
            dcc.Store(
                id="general-working-directory", data=str(run_handler.get_working_directory())
            ),
            dcc.Store(id="general-optuna-import-trigger", data=0),
            dcc.Store(id=OPTUNA_DISCOVERY_STORE_ID, data=None),
            dcc.Store(id=GROUP_HYDRATION_STORE_ID, data=None),
            dcc.Store(id=CACHE_CLEARED_TRIGGER_ID, data=0),
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    html.Span("Load Optuna Study"),
                                    html.I(className="fas fa-database text-muted"),
                                ],
                                className="d-flex justify-content-between mb-2",
                            ),
                            dbc.Label("SQLite database", html_for="general-optuna-database-input"),
                            html.Div(
                                [
                                    dbc.Input(
                                        id="general-optuna-database-input",
                                        type="text",
                                        value="",
                                        placeholder="/app/data/optuna.db",
                                        debounce=True,
                                    ),
                                    dbc.Button(
                                        [html.I(className="far fa-folder-open me-1"), "Browse"],
                                        id="general-optuna-browse-button",
                                        color="secondary",
                                        outline=True,
                                    ),
                                ],
                                className="general-optuna-row",
                            ),
                            html.Div(
                                id="general-optuna-database-status",
                                className="d-none",
                                role="status",
                            ),
                            dbc.Label(
                                "Study", html_for="general-optuna-study-dropdown", className="mt-2"
                            ),
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="general-optuna-study-dropdown",
                                        options=[],
                                        placeholder="Select a study",
                                        disabled=True,
                                        style={"flex": "1", "minWidth": 0},
                                    ),
                                    dbc.Button(
                                        "Load Study",
                                        id="general-optuna-load-button",
                                        color="primary",
                                        disabled=True,
                                    ),
                                ],
                                className="general-optuna-row",
                            ),
                        ]
                    ),
                ],
                className="general-optuna-card mb-2",
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Select SQLite database"), close_button=False),
                    dbc.ModalBody(
                        [
                            dbc.Label("Directory", html_for="general-optuna-browser-directory"),
                            dbc.InputGroup(
                                [
                                    dbc.Input(id="general-optuna-browser-directory", debounce=True),
                                    dbc.Button(
                                        "Up", id="general-optuna-browser-parent", outline=True
                                    ),
                                ]
                            ),
                            html.Div(
                                id="general-optuna-browser-error",
                                className="small text-danger mt-1",
                                role="status",
                            ),
                            html.Div(
                                id="general-optuna-browser-entries",
                                className="mt-2",
                                style={"maxHeight": "50vh", "overflowY": "auto"},
                            ),
                        ]
                    ),
                    dbc.ModalFooter(dbc.Button("Cancel", id="general-optuna-browser-close")),
                ],
                id="general-optuna-browser",
                is_open=False,
                size="lg",
            ),
            dcc.Store(id="general-selected-runs", data={}),
            dcc.Store(id="general-group-trigger", data=0),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H2(
                                    [
                                        html.I(
                                            className="fas fa-play general-card-icon",
                                            **{"aria-hidden": "true"},
                                        ),
                                        "Selected Runs",
                                    ],
                                    className="general-card-title",
                                ),
                                html.Span(
                                    "0 runs",
                                    id="general-selected-runs-count",
                                    className="general-run-count",
                                ),
                            ],
                            className="general-card-heading",
                        ),
                        html.Div(
                            id="general-selected-runs-container",
                            className="general-runs-table-wrapper",
                        ),
                    ]
                ),
                className="general-management-card general-selected-runs-card",
            ),
            html.Div(
                [
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H2(
                                    [
                                        html.I(
                                            className="fas fa-users general-card-icon",
                                            **{"aria-hidden": "true"},
                                        ),
                                        "Groups",
                                    ],
                                    className="general-card-title",
                                ),
                                html.P(
                                    "Organize runs into groups.",
                                    className="general-card-description",
                                ),
                                dbc.Button(
                                    [
                                        html.I(
                                            className="fas fa-plus me-1", **{"aria-hidden": "true"}
                                        ),
                                        "Add Group",
                                    ],
                                    id="general-add-group",
                                    color="primary",
                                    size="sm",
                                ),
                                html.Div(id="general-group-container", children=[]),
                            ]
                        ),
                        className="general-management-card",
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H2(
                                    [
                                        html.I(
                                            className="fas fa-database general-card-icon",
                                            **{"aria-hidden": "true"},
                                        ),
                                        "Caches",
                                    ],
                                    className="general-card-title",
                                ),
                                html.P(
                                    "Manage cached analysis data.",
                                    className="general-card-description",
                                ),
                                dbc.Button(
                                    "Clear Plugin Caches",
                                    id="general-clear-cache-button",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                ),
                            ]
                        ),
                        className="general-management-card",
                    ),
                ],
                className="general-management-grid",
            ),
        ]
