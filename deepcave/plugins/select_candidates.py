"""Interactive, study-spanning selection of concrete Optuna trials."""

from typing import Any, Callable, Dict, List

from urllib.parse import quote

import dash_bootstrap_components as dbc
from dash import ALL, ClientsideFunction, Input, Output, State, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate

from deepcave import config, interactive
from deepcave.plugins import Plugin
from deepcave.runs import AbstractRun
from deepcave.runs.group import Group
from deepcave.utils.analysis_layout import analysis_card, analysis_graph
from deepcave.utils.candidates import (
    CANDIDATE_COLORS,
    candidate_data,
    central_objective,
    parallel_figure,
    scatter_figure,
    selected_trials_toml,
)

PREFIX = "select-candidates-"


def cid(name: str) -> str:
    """Namespace the workspace's components."""
    return PREFIX + name


def compact_selector(name: str, label: str, placeholder: str) -> Any:
    """Keep the existing options/value contract in a compact checkbox dropdown."""
    return html.Div(
        [
            dbc.Label(label, id=cid(name + "-label")),
            html.Details(
                [
                    html.Summary(
                        html.Span(placeholder, id=cid(name + "-summary")),
                        **{"aria-labelledby": cid(name + "-label") + " " + cid(name + "-summary")},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Search " + label.lower(),
                                htmlFor=cid(name + "-search"),
                                className="candidate-selector-search-label",
                            ),
                            dcc.Input(
                                id=cid(name + "-search"),
                                type="search",
                                placeholder="Search " + name + " …",
                                className="candidate-selector-search",
                            ),
                            html.Div(
                                [
                                    html.Button("Select all", type="button", **{"data-action": "all"}),
                                    html.Button("Clear all", type="button", **{"data-action": "clear"}),
                                    html.Span(id=cid(name + "-count"), **{"aria-live": "polite"}),
                                ],
                                className="candidate-selector-actions",
                            ),
                            dcc.Checklist(
                                id=cid(name),
                                options=[],
                                value=None if name == "objectives" else [],
                                className="candidate-selector-options",
                                labelStyle={"display": "flex"},
                                persistence=name == "studies",
                                persistence_type="session",
                            ),
                            html.Div("No matching options", className="candidate-selector-empty", hidden=True),
                        ],
                        className="candidate-selector-menu",
                    ),
                ],
                className="candidate-selector",
                **{"data-selector": name},
            ),
        ],
        className="candidate-" + name,
    )


def table_data(data: Dict[str, Any], metrics: List[str]) -> tuple:
    """Flatten only the table view; the independent selection retains full trial records."""
    parameters = sorted({name for row in data.get("rows", []) for name in row["parameters"]})
    columns = [
        {"name": "Study", "id": "study_name", "type": "text"},
        {"name": "Trial", "id": "trial_number", "type": "numeric"},
        {
            "name": "Duration (s)",
            "id": "duration",
            "type": "numeric",
            "format": Format(precision=6, scheme=Scheme.decimal_or_exponent),
        },
    ]
    for key in metrics or []:
        if key in data.get("objectives", {}):
            columns.append(
                {
                    "name": data["objectives"][key]["label"],
                    "id": key,
                    "type": "numeric",
                    "format": Format(precision=6, scheme=Scheme.decimal_or_exponent),
                }
            )
    for name in parameters:
        values = [row["parameters"][name] for row in data["rows"] if name in row["parameters"]]
        numeric = all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
        )
        columns.append(
            {"name": name, "id": "param:" + name, "type": "numeric" if numeric else "text"}
        )
    numeric_parameters = {column["id"] for column in columns if column["type"] == "numeric"}
    rows = []
    for row in data.get("rows", []):
        if row["state"] != "COMPLETE":
            continue
        flat = {
            key: row[key]
            for key in ("id", "study_name", "trial_number", "trial_id", "state", "duration")
        }
        flat.update(row["metrics"])
        flat.update(
            {
                "param:" + name: value if "param:" + name in numeric_parameters else str(value)
                for name, value in row["parameters"].items()
            }
        )
        rows.append(flat)
    return rows, columns


class SelectCandidates(Plugin):
    """Keep trial data, temporary highlighting and session selection separate."""

    id = "select_candidates"
    name = "Select Candidates"
    icon = "fas fa-check-square"
    help = "plugins/select_candidates.html"
    activate_run_selection = True

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """Accept individual loaded Optuna studies, including heterogeneous search spaces."""
        return run.prefix == "Optuna" and not isinstance(run, Group)

    @staticmethod
    def get_output_layout(register: Callable) -> Any:
        """Render a compact explorer above the collection and full-width comparison."""
        return html.Div(
            [
                dcc.Store(id=cid("mounted"), data=True),
                dcc.Store(id=cid("data")),
                dcc.Store(id=cid("objective-context")),
                dcc.Store(id=cid("visible-ids")),
                dcc.Store(
                    id=cid("presentation"),
                    data={
                        "scatter": scatter_figure([], None).to_plotly_json(),
                        "numberFormat": Format(
                            precision=6, scheme=Scheme.decimal_or_exponent
                        ).to_plotly_json(),
                    },
                ),
                dcc.Store(id=cid("table-source")),
                dcc.Store(id=cid("selection"), storage_type="session", data={}),
                dcc.Store(id=cid("event")),
                dcc.Store(id=cid("highlighted")),
                dcc.Store(id=cid("viewport")),
                html.Div(
                    [
                        compact_selector("objectives", "Table objectives", "Select objectives …"),
                        compact_selector("studies", "Additional studies", "Add loaded studies …"),
                    ],
                    className="analysis-toolbar candidate-toolbar",
                ),
                html.Div(
                    [
                        analysis_card(
                            "Trial Explorer",
                            dash_table.DataTable(
                                id=cid("table"),
                                data=[],
                                columns=[],
                                row_selectable="multi",
                                selected_rows=[],
                                selected_row_ids=[],
                                sort_action="custom",
                                sort_by=[],
                                sort_mode="multi",
                                filter_action="native",
                                page_action="native",
                                page_size=20,
                                style_table={"overflowX": "auto", "minHeight": "430px"},
                                style_cell={
                                    "fontFamily": "Arial, sans-serif",
                                    "fontSize": "12px",
                                    "padding": "5px 8px",
                                    "minWidth": "105px",
                                    "width": "120px",
                                    "maxWidth": "220px",
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                    "textAlign": "right",
                                    "fontVariantNumeric": "tabular-nums",
                                },
                                style_header={
                                    "backgroundColor": "#f8f9fb",
                                    "fontWeight": "500",
                                    "color": "#505766",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"state": "selected"},
                                        "backgroundColor": "#eef2ff",
                                        "border": "1px solid #dce2f2",
                                    },
                                    {"if": {"column_id": "study_name"}, "textAlign": "left"},
                                ],
                                tooltip_duration=None,
                            ),
                        ),
                        analysis_card(
                            "Trial Duration × Objective",
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dbc.Switch(
                                                id=cid("pareto"),
                                                label="Show Pareto Frontier",
                                                value=False,
                                                className="candidate-plot-switch",
                                            ),
                                            dbc.Switch(
                                                id=cid("elites"),
                                                label="Show elites",
                                                value=False,
                                                className="candidate-plot-switch",
                                            ),
                                            dbc.Select(
                                                id=cid("elite-fraction"),
                                                value="0.05",
                                                options=[
                                                    {
                                                        "label": f"Top {percent}%",
                                                        "value": str(percent / 100),
                                                    }
                                                    for percent in (1, 5, 10, 20)
                                                ],
                                                style={"width": "110px"},
                                            ),
                                        ],
                                        className="candidate-plot-controls",
                                    ),
                                    analysis_graph(
                                        cid("scatter"), 430, config.FIGURE_DOWNLOAD_SCALE
                                    ),
                                    html.Div(
                                        "Click to toggle a trial. "
                                        "Box or lasso adds trials to the selection.",
                                        className="candidate-hint",
                                    ),
                                ]
                            ),
                        ),
                    ],
                    className="candidate-explorer",
                ),
                html.Section(
                    [
                        html.Div(
                            [
                                html.Strong(id=cid("count"), children="Selected Trials · 0"),
                                dbc.Button(
                                    "Clear selection",
                                    id=cid("clear"),
                                    n_clicks=0,
                                    size="sm",
                                    color="secondary",
                                    outline=True,
                                ),
                                html.A(
                                    "Export TOML",
                                    id=cid("export"),
                                    download="selected-trials.toml",
                                    className="btn btn-outline-secondary btn-sm disabled",
                                    **{"aria-disabled": "true"},
                                ),
                            ],
                            className="candidate-selection-header",
                        ),
                        html.Div(id=cid("chips"), className="candidate-chips"),
                    ],
                    className="candidate-selection",
                    **{"aria-label": "Selected Trials"},
                ),
                analysis_card(
                    "Parallel Coordinates",
                    html.Div(
                        [
                            html.Div(
                                [
                                    dbc.Label("Hyperparameter axes", html_for=cid("axes")),
                                    dcc.Dropdown(
                                        id=cid("axes"),
                                        multi=True,
                                        value=[],
                                        placeholder="Choose axes …",
                                    ),
                                    html.Small(
                                        "Remove axes to hide them; "
                                        "drag axes in the plot to reorder.",
                                        className="candidate-hint",
                                    ),
                                ],
                                className="candidate-axis-controls",
                            ),
                            html.Div(
                                analysis_graph(cid("parallel"), 440, config.FIGURE_DOWNLOAD_SCALE),
                                id=cid("parallel-scroll"),
                            ),
                            html.Div(id=cid("parallel-note"), className="candidate-hint"),
                        ]
                    ),
                ),
            ],
            id=cid("workspace"),
            className="candidate-workspace",
        )

    @interactive
    def __call__(self) -> List[Any]:
        """Mount a workspace without extra configuration or global action cards."""
        return [html.H1(self.name), self.get_output_layout(self.register_output)]

    @interactive
    def register_callbacks(self) -> None:
        """Download trial data once; prepare table and scatter views locally in the browser."""
        from deepcave import app, rc, run_handler

        for name in ("objectives", "studies"):
            app.clientside_callback(
                ClientsideFunction(namespace="candidateSelectors", function_name="summarize"),
                Output(cid(name + "-summary"), "children"),
                Output(cid(name + "-summary"), "title"),
                Output(cid(name + "-summary"), "className"),
                Output(cid(name + "-count"), "children"),
                Input(cid(name), "options"),
                Input(cid(name), "value"),
                State(cid(name), "id"),
            )

        @app.callback(
            Output(cid("data"), "data"),
            Output(cid("studies"), "options"),
            Output(cid("objective-context"), "data"),
            Input(cid("mounted"), "data"),
            Input("global-run-selector", "value"),
            Input(cid("studies"), "value"),
        )
        def load_context(_mounted: Any, run_id: str, additional: List[str]) -> tuple:
            if not run_id:
                raise PreventUpdate
            runs = [run for run in run_handler.get_runs() if self.check_run_compatibility(run)]
            available = {run.id: run for run in runs}
            if run_id not in available:
                raise PreventUpdate
            chosen = list(dict.fromkeys([run_id] + (additional or [])))
            data: Dict[str, Any] = {
                "rows": [],
                "objectives": {},
                "keymap": {},
                "palette": CANDIDATE_COLORS,
            }
            for source_id in chosen:
                if source_id not in available:
                    continue
                run = available[source_id]
                prepared = rc.get(run, self.id, "trials-v2")
                if prepared is None:
                    prepared = candidate_data(run)
                    rc.set(run, self.id, "trials-v2", value=prepared)
                data["rows"].extend(prepared["rows"])
                data["objectives"].update(prepared["objectives"])
                if source_id == run_id:
                    data["keymap"] = prepared["keymap"]
            # Loading the same source through two snapshot paths must not duplicate trials.
            data["rows"] = list({row["id"]: row for row in data["rows"]}.values())
            options = [{"label": run.name, "value": run.id} for run in runs if run.id != run_id]
            return data, options, {"keymap": data["keymap"]}

        app.clientside_callback(
            ClientsideFunction(namespace="candidateExplorer", function_name="objectives"),
            Output(cid("objectives"), "options"),
            Output(cid("objectives"), "value"),
            Input(cid("data"), "data"),
            State(cid("objectives"), "value"),
        )

        app.clientside_callback(
            ClientsideFunction(namespace="candidateExplorer", function_name="table"),
            Output(cid("table-source"), "data"),
            Output(cid("table"), "columns"),
            Input(cid("data"), "data"),
            Input(cid("objectives"), "value"),
            State(cid("presentation"), "data"),
        )
        app.clientside_callback(
            ClientsideFunction(namespace="candidateExplorer", function_name="visibleIds"),
            Output(cid("visible-ids"), "data"),
            Input(cid("table"), "derived_virtual_row_ids"),
            State(cid("visible-ids"), "data"),
        )
        app.clientside_callback(
            ClientsideFunction(namespace="candidateExplorer", function_name="scatter"),
            Output(cid("scatter"), "figure"),
            Input(cid("visible-ids"), "data"),
            Input(cid("data"), "data"),
            Input("global-main-objective-state", "data"),
            Input(cid("pareto"), "value"),
            Input(cid("elites"), "value"),
            Input(cid("elite-fraction"), "value"),
            State(cid("presentation"), "data"),
        )

        # The only writer of permanent selection also owns table checkbox projection.
        # Dash supports circular synchronization within one callback.
        app.clientside_callback(
            ClientsideFunction(namespace="candidateSelection", function_name="reduce"),
            [
                Output(cid("selection"), "data"),
                Output(cid("table"), "selected_row_ids"),
                Output(cid("table"), "selected_rows"),
                Output(cid("table"), "active_cell"),
                Output(cid("table"), "style_data_conditional"),
                Output(cid("table"), "data"),
                Output(cid("table"), "tooltip_data"),
                Output(cid("table"), "page_current"),
            ],
            [
                Input(cid("table"), "selected_row_ids"),
                Input(cid("table"), "active_cell"),
                Input(cid("event"), "data"),
                Input({"type": "candidate-remove", "key": ALL}, "n_clicks"),
                Input(cid("clear"), "n_clicks"),
                Input(cid("table-source"), "data"),
                Input(cid("data"), "data"),
                Input(cid("table"), "sort_by"),
            ],
            State(cid("selection"), "data"),
        )
        app.clientside_callback(
            ClientsideFunction(namespace="candidateSelection", function_name="visibleRows"),
            Output(cid("viewport"), "data"),
            Input(cid("table"), "derived_viewport_data"),
            State(cid("table"), "data"),
        )

        @app.callback(
            Output(cid("chips"), "children"),
            Output(cid("count"), "children"),
            Output(cid("export"), "href"),
            Output(cid("export"), "className"),
            Output(cid("export"), "title"),
            Output(cid("export"), "aria-disabled"),
            Input(cid("selection"), "data"),
        )
        def update_collection(selection: Any) -> tuple:
            rows = list((selection or {}).values())
            chips = [
                html.Div(
                    [
                        html.Span(
                            style={"backgroundColor": row["color"]}, className="candidate-swatch"
                        ),
                        html.Span(f"{row['study_name']} · Trial {row['trial_number']}"),
                        dcc.Link(
                            "Sensitivity",
                            href=f"/plugins/sensitivity_analysis?trial={row['id']}",
                            title=(
                                "Inspect this trial's configuration and seed repetitions "
                                "in the current study/group context"
                            ),
                        ),
                        html.Button(
                            "×",
                            id={"type": "candidate-remove", "key": row["id"]},
                            n_clicks=0,
                            title="Remove trial",
                            **{
                                "aria-label": (
                                    f"Remove {row['study_name']} trial {row['trial_number']}"
                                )
                            },
                        ),
                    ],
                    className="candidate-chip",
                    title=f"Trial ID: {row['trial_id']} · {row['storage']}",
                    tabIndex=0,
                    **{"data-candidate-key": row["id"]},
                )
                for row in rows
            ]
            href, title = None, "Select trials to export."
            if rows:
                try:
                    content = selected_trials_toml(rows)
                    href = "data:application/toml;charset=utf-8," + quote(content, safe="")
                    title = f"Download {len(rows):,} selected trials as TOML"
                except ValueError as error:
                    title = str(error)
            classes = "btn btn-outline-secondary btn-sm" + (" disabled" if href is None else "")
            return (
                chips,
                f"Selected Trials · {len(rows):,}",
                href,
                classes,
                title,
                str(href is None).lower(),
            )

        @app.callback(
            Output(cid("axes"), "options"),
            Output(cid("axes"), "value"),
            Input(cid("selection"), "data"),
            State(cid("axes"), "options"),
            State(cid("axes"), "value"),
        )
        def update_axes(selection: Any, previous: Any, current: Any) -> tuple:
            names = sorted(
                {name for row in (selection or {}).values() for name in row["parameters"]}
            )
            old_names = {option["value"] for option in previous or []}
            chosen = [name for name in current or [] if name in names]
            chosen.extend(name for name in names if name not in old_names)
            return [{"label": name, "value": name} for name in names], chosen

        @app.callback(
            Output(cid("parallel"), "figure"),
            Output(cid("parallel-note"), "children"),
            Output(cid("parallel"), "style"),
            Input(cid("selection"), "data"),
            Input(cid("axes"), "value"),
            Input("global-main-objective-state", "data"),
            Input(cid("objective-context"), "data"),
        )
        def update_parallel(selection: Any, axes: Any, state: Any, data: Any) -> tuple:
            figure, note = parallel_figure(
                list((selection or {}).values()), central_objective(data or {}, state), axes
            )
            width = max(700, (len(axes or []) + 2) * 145)
            return figure, note, {"width": "100%", "minWidth": f"{width}px", "height": "440px"}
