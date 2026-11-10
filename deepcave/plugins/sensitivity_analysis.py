"""Native, linked configuration exploration and seed robustness diagnostics."""

from typing import Any, Callable, Dict, List, Sequence

import hashlib
import sqlite3

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import (
    ALL,
    ClientsideFunction,
    Input,
    Output,
    Patch,
    State,
    ctx,
    dash_table,
    dcc,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate

from deepcave import config, interactive
from deepcave.evaluators.sensitivity import (
    available_modes,
    build_groups,
    canonical,
    elite_subset,
    prepare_study,
    ranking,
    seed_design,
    summary,
    variance_decomposition,
)
from deepcave.plugins import Plugin
from deepcave.runs.group import Group
from deepcave.runs.handler import (
    DATA_SEED_ATTRIBUTE_CACHE_KEY,
    MODEL_SEED_ATTRIBUTE_CACHE_KEY,
)
from deepcave.utils.analysis_layout import (
    analysis_card,
    analysis_graph,
    analysis_toolbar_item,
)
from deepcave.utils.candidates import central_objective
from deepcave.utils.optuna import resolve_optuna_database
from deepcave.utils.sensitivity import (
    distribution_figure,
    distribution_reference_trace,
    explorer_figure,
    global_sensitivity_figure,
    randomness_range_figure,
    seed_matrix_figure,
    styled,
    table_projection,
    trials_by_seed_figure,
)
from deepcave.utils.sensitivity_references import (
    REFERENCE_CACHE_KEY,
    find_reference_trials,
    reference_key,
    reference_label,
    reference_objective,
    refresh_reference_objectives,
)

PREFIX = "sensitivity-"


def cid(name: str) -> str:
    """Namespace analysis components."""
    return PREFIX + name


def note(message: str) -> Any:
    """Render readable methodology and empty states without empty plots."""
    return html.P(message, className="sensitivity-note", role="status")


def graph(figure: go.Figure, name: str = "detail-graph") -> Any:
    """Reuse the native responsive graph and export settings."""
    component = analysis_graph(cid(name), figure.layout.height or 390, config.FIGURE_DOWNLOAD_SCALE)
    component.figure = figure
    return component


def table(name: str, rows: List[Dict], columns: List[Dict], **kwargs: Any) -> Any:
    """Use the compact candidate explorer's native sort/filter table pattern."""
    return dash_table.DataTable(
        id=cid(name),
        data=rows,
        columns=columns,
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
        page_action="native",
        page_size=12,
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "Arial, sans-serif",
            "fontSize": "12px",
            "padding": "5px 8px",
            "minWidth": "95px",
            "maxWidth": "240px",
            "textAlign": "right",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "fontVariantNumeric": "tabular-nums",
        },
        style_header={"backgroundColor": "#f8f9fb", "fontWeight": "normal"},
        tooltip_data=[
            {key: {"value": str(value), "type": "text"} for key, value in row.items()}
            for row in rows
        ],
        tooltip_duration=None,
        **kwargs,
    )


def statistics_table(name: str, rows: List[Dict]) -> Any:
    """Project a short diagnostic result with parallel numeric columns."""
    if not rows:
        return note("No comparable observations available.")
    columns = [
        {
            "id": key,
            "name": key.replace("_", " ").title(),
            "type": (
                "numeric" if any(isinstance(row.get(key), (int, float)) for row in rows) else "text"
            ),
        }
        for key in rows[0]
    ]
    return table(name, rows, columns)


def context_label(group: Dict) -> str:
    """Expose why configurations are kept apart, without assuming config IDs are unique."""
    context = group["context"]
    attrs = context["metadata"]
    label = " · ".join(
        str(attrs[key])
        for key in ("experiment_id", "model", "model_type", "dataset", "dataset_id")
        if key in attrs
    )
    if context["scope"]:
        label = (label + " · " if label else "") + context["scope"][1]
    digest = hashlib.sha256(canonical(context).encode()).hexdigest()[:8]
    return (label or "Experiment") + " · " + digest


def eligible_groups(data: Dict, context: Any) -> List[Dict]:
    """Apply context and distinct-seed controls consistently across the suite."""
    return [
        g
        for g in data.get("groups", [])
        if g["seeds"] >= 2 and (not context or context == "all" or context == g["context_label"])
    ]


def distribution_description(groups: Sequence[Dict], references: Any = None) -> str:
    """Explain the first distribution plot and unavailable reference values."""
    description = ""
    if references is not None:
        description += (
            " Red diamonds mark assigned reference trials with a value for the selected objective."
        )
        unavailable = sum(
            1
            for group in groups
            if (trial := references.get(reference_key(group)))
            and reference_objective(trial, group) is None
        )
        if unavailable:
            description += f" Reference values for the selected objective are unavailable for {unavailable} displayed configurations."
    return description


def render_detail(
    tab: str,
    active: Any,
    selected: Sequence[str],
    data: Dict,
    elite: Any = 1.0,
    trial_selection: Sequence[str] = (),
) -> Any:
    """Run advanced calculations only for the requested tab, on cached configuration groups."""
    groups = data.get("groups", [])
    lookup = {g["id"]: g for g in groups}
    group = lookup.get(active)
    chosen = [lookup[key] for key in selected or [] if key in lookup]
    overview = []
    if tab == "distribution":
        if not groups:
            return note("No configurations available for the current filters.")
        overview = [
            analysis_card(
                "Trials by Seed per Configuration",
                graph(trials_by_seed_figure(groups), "trials-by-seed"),
            ),
            analysis_card(
                "Potential performance impact by randomness source",
                graph(randomness_range_figure(groups), "randomness-range"),
            ),
        ]
    if tab in ("distribution", "effects"):
        if group is None:
            return html.Div(
                overview
                + [
                    note(
                        "Select a configuration by clicking a trial in the boxplot to inspect its seeds."
                    )
                ]
            )
        if tab == "distribution":
            rows = []
            for row in group["rows"]:
                rows.append(
                    {
                        "id": row["id"],
                        "study": row["study_name"],
                        "trial": row["trial_number"],
                        "trial_id": row["trial_id"],
                        "model_seed": row["model_seed"],
                        "data_seed": row["data_seed"],
                        "objective": row["objective"],
                        "duration_s": row["duration"],
                        "source_config_id": str(row["user_attrs"].get("config_id", "")),
                        **row["metrics"],
                    }
                )
            columns = [
                {"id": key, "name": key.replace("_", " ").title()} for key in rows[0] if key != "id"
            ]
            selected_ids = [row["id"] for row in rows if row["id"] in (trial_selection or [])]
            return html.Div(
                [
                    *overview,
                    note(
                        "Trials for the active configuration. Select trial checkboxes to compare "
                        "observations or add them to Select Candidates."
                    ),
                    table(
                        "trials",
                        rows,
                        columns,
                        row_selectable="multi",
                        selected_row_ids=selected_ids,
                        selected_rows=[
                            i for i, row in enumerate(rows) if row["id"] in selected_ids
                        ],
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "Add configuration trials to Select Candidates",
                                id=cid("add-config"),
                                size="sm",
                                outline=True,
                            ),
                            dbc.Button(
                                "Add checked trials", id=cid("add-trials"), size="sm", outline=True
                            ),
                            dcc.Link(
                                "Open Select Candidates",
                                href="/plugins/select_candidates",
                                className="btn btn-link btn-sm",
                            ),
                        ],
                        className="d-flex flex-wrap gap-2 mt-2",
                    ),
                ]
            )
        design = seed_design(group)
        # Conditional explorer rows stay isolated; inspect the full configuration only
        # in Combined mode to avoid silently pooling the held-constant seed.
        if design["model_levels"] and design["data_levels"]:
            children = [graph(seed_matrix_figure(group))]
            if not design["crossed"]:
                children.append(
                    note(
                        "Model and data seeds are coupled or only one factor varies; "
                        "separate seed effects are not identifiable."
                    )
                )
            return html.Div(children)
        return html.Div(
            [
                note(
                    "Seeds are categorical identifiers. Their numerical order is not "
                    "a performance trend."
                ),
                graph(distribution_figure([group])),
            ]
        )
    if tab == "ranking":
        result = ranking(chosen)
        if "rows" not in result:
            return note(result["reason"])
        figure = go.Figure()
        for row in result["rows"]:
            figure.add_trace(
                go.Box(
                    name=row["label"], y=row["ranks"], boxpoints=False, line={"color": row["color"]}
                )
            )
        rows = [
            {key: value for key, value in row.items() if key not in ("ranks", "color")}
            for row in result["rows"]
        ]
        return html.Div(
            [
                note(
                    f"{result['method']} · {result['draws']:,} deterministic resamples · "
                    "95% percentile intervals. Top-1 / Top-3 frequencies describe bootstrap "
                    "stability, not calibrated probabilities of being truly best. "
                    "Small seed samples remain uncertain."
                ),
                statistics_table("rank-statistics", rows),
                graph(styled(figure, "Configuration", "Bootstrap rank")),
            ]
        )
    if tab == "global":
        if not any(g["parameters"] and g["sd"] is not None for g in groups):
            return note("No hyperparameters with comparable seed observations available.")
        return html.Div(
            [
                note(
                    "Descriptive association in the sampled HPO search space. Uneven "
                    "configuration sampling and unequal seed counts prevent causal "
                    "interpretation. Color encodes mean performance; marker size encodes "
                    "distinct seed count. All panels share the same scales."
                ),
                graph(global_sensitivity_figure(groups), "global-scatter"),
            ]
        )
    if tab == "variance":
        result = variance_decomposition(elite_subset(groups, float(elite or 1.0)))
        if "rows" not in result:
            return note(
                result["reason"]
                + " Missing cells, coupled seeds and unbalanced repeats are not filled in."
            )
        rows = result["rows"]
        figure = go.Figure(
            go.Bar(
                x=[r["variance"] for r in rows],
                y=[r["component"] for r in rows],
                orientation="h",
                marker_color="#64748b",
            )
        )
        return html.Div(
            [
                note(
                    result["method"]
                    + f" · {result['repeats']} evaluations per cell. All configuration × seed "
                    "interactions are included. Interpretation assumes independent repeat "
                    "errors and exchangeable random-factor levels. Negative estimates remain "
                    "visible; only displayed variance shares truncate them to zero. "
                    "Elite selection conditions these descriptive estimates."
                ),
                statistics_table("variance-statistics", rows),
                graph(styled(figure, "Estimated variance (nonnegative display)", "Component", 470)),
            ]
        )
    return note("Select an analysis tab.")


class SensitivityAnalysis(Plugin):
    """Explore seed robustness using the sidebar context and shared candidate selection."""

    id = "sensitivity_analysis"
    name = "Sensitivity Analysis"
    icon = "fas fa-random"
    help = "plugins/sensitivity_analysis.html"
    activate_run_selection = True

    @staticmethod
    def check_run_compatibility(run: Any) -> bool:
        """Accept Optuna studies and native groups of Optuna studies."""
        sources = run.runs if isinstance(run, Group) else [run]
        return bool(sources) and all(source.prefix == "Optuna" for source in sources)

    @staticmethod
    def get_output_layout(register: Callable) -> Any:
        """Keep overview and explorer visible above one compact tabbed detail section."""
        return html.Div(
            [
                dcc.Store(id=cid("mounted"), data=True),
                dcc.Store(id=cid("source")),
                dcc.Store(id=cid("plot-event")),
                dcc.Store(id=cid("viewport")),
                dcc.Store(id=cid("trial-view")),
                dcc.Store(id=cid("groups")),
                dcc.Store(id=cid("active")),
                dcc.Store(id=cid("comparison"), data=[]),
                dcc.Store(id=cid("reference-results"), data={}),
                dcc.Store(id=cid("trial-selection"), data=[]),
                # Same ID, storage type and row schema as Select Candidates.
                dcc.Store(id="select-candidates-selection", storage_type="session", data={}),
                html.Div(id=cid("summary"), className="sensitivity-summary"),
                html.Div(
                    [
                        analysis_toolbar_item(
                            "Experiment context",
                            dcc.Dropdown(id=cid("context"), clearable=False),
                            "min(26rem, 100%)",
                        ),
                        dbc.Checkbox(id=cid("pareto"), value=True, label="Show Pareto frontier"),
                    ],
                    className="analysis-toolbar d-flex flex-wrap align-items-center gap-2",
                ),
                html.Div(id=cid("status"), className="sensitivity-note", role="status"),
                html.Div(
                    table(
                        "table",
                        [],
                        [],
                        row_selectable="multi",
                        selected_row_ids=[],
                        selected_rows=[],
                    ),
                    style={"display": "none"},
                ),
                html.Div(
                    [
                        analysis_card(
                            "Objective Distribution",
                            html.Div(
                                [
                                    html.Div(
                                        dbc.Checkbox(
                                            id=cid("show-reference-objectives"),
                                            value=False,
                                            label="Show reference trial objectives",
                                        ),
                                        id=cid("reference-objectives-control"),
                                        style={"padding": ".5rem .75rem 0"},
                                    ),
                                    graph(distribution_figure([]), "distribution"),
                                    html.Div(
                                        id=cid("distribution-description"),
                                        className="sensitivity-note",
                                    ),
                                ]
                            ),
                        ),
                        analysis_card(
                            "Mean Performance × Seed Variability",
                            html.Div(id=cid("scatter-container")),
                        ),
                    ],
                    className="sensitivity-explorer",
                ),
                html.Div(
                    [
                        html.Strong(id=cid("active-label")),
                        dbc.Button(
                            "Locate selected candidate", id=cid("locate"), size="sm", outline=True
                        ),
                    ],
                    id=cid("detail-heading"),
                    className="sensitivity-detail-heading",
                ),
                dcc.Tabs(
                    id=cid("tab"),
                    value="distribution",
                    children=[
                        dcc.Tab(label=label, value=value)
                        for value, label in (
                            ("distribution", "Distribution"),
                            ("effects", "Seed Effects"),
                            ("global", "Parameter sensitivity"),
                            ("reference-trials", "Reference Trials"),
                        )
                    ],
                    className="sensitivity-tabs",
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Find reference trials",
                            id=cid("reference-search"),
                            n_clicks=0,
                            size="sm",
                            outline=True,
                        ),
                        note(
                            "Search other studies in the Optuna database selected under General "
                            "for the visible configurations and the same dataset. "
                            "The search runs only when you click the button."
                        ),
                        dcc.Loading(html.Div(id=cid("reference-search-status")), type="dot"),
                        html.Div(
                            [html.Strong("Configuration"), html.Strong("Reference trial")],
                            className="sensitivity-reference-row",
                        ),
                        html.Div(id=cid("reference-controls")),
                        html.Div(id=cid("reference-save-status"), role="status"),
                    ],
                    id=cid("reference-panel"),
                    className="sensitivity-reference-controls",
                    style={"display": "none"},
                ),
                html.Div(
                    [
                        html.Div(
                            analysis_toolbar_item(
                                "Elite subset",
                                dbc.Select(
                                    id=cid("elite"),
                                    options=[
                                        {"label": label, "value": value}
                                        for label, value in (
                                            ("All configurations", "1"),
                                            ("Top 50%", ".5"),
                                            ("Top 20%", ".2"),
                                            ("Top 10%", ".1"),
                                        )
                                    ],
                                    value="1",
                                ),
                                "17rem",
                            ),
                            id=cid("elite-control"),
                        ),
                    ],
                    className="sensitivity-controls",
                ),
                dcc.Loading(
                    html.Div(id=cid("content")),
                    type="dot",
                    target_components={cid("content"): "children"},
                ),
                html.Div(id=cid("candidate-status"), className="sensitivity-note", role="status"),
            ],
            id=cid("workspace"),
            className="sensitivity-workspace",
        )

    @interactive
    def __call__(self) -> List[Any]:
        """Mount the analysis without an additional actions card."""
        return [html.H1(self.name), self.get_output_layout(self.register_output)]

    @interactive
    def register_callbacks(self) -> None:
        """Cache source loading; keep group computation independent of active selection."""
        from deepcave import app, c, rc, run_handler

        def reference_values(groups: Sequence[Dict], enabled: Any) -> tuple:
            references = {} if enabled else None
            errors = []
            if enabled:
                for key in {reference_key(group) for group in groups}:
                    trial = c.get(REFERENCE_CACHE_KEY, key)
                    if not trial:
                        continue
                    if "metrics" not in trial or "user_attrs" not in trial:
                        try:
                            trial = refresh_reference_objectives(trial)
                            c.set(REFERENCE_CACHE_KEY, key, value=trial)
                        except (ValueError, OSError, sqlite3.Error) as error:
                            errors.append(str(error))
                    references[key] = trial
            return references, errors

        @app.callback(
            Output(cid("distribution"), "figure"),
            Output(cid("distribution-description"), "children"),
            Input(cid("groups"), "data"),
            Input(cid("table"), "derived_virtual_row_ids"),
            Input(cid("table"), "data"),
            Input(cid("trial-selection"), "data"),
            Input(cid("reference-save-status"), "children"),
            State(cid("show-reference-objectives"), "value"),
        )
        def boxplot(
            data: Any, visible: Any, rows: Any, trials: Any, _status: Any, enabled: Any
        ) -> tuple:
            ids = set(visible if visible is not None else [row["id"] for row in rows or []])
            groups = [g for g in (data or {}).get("groups", []) if g["id"] in ids]
            references, errors = reference_values(groups, enabled)
            description = distribution_description(groups, references)
            if errors:
                description += " Reference objectives could not be loaded: " + "; ".join(errors)
            return distribution_figure(groups, trials or [], references), description

        @app.callback(
            Output(cid("distribution"), "figure", allow_duplicate=True),
            Output(cid("distribution-description"), "children", allow_duplicate=True),
            Input(cid("show-reference-objectives"), "value"),
            State(cid("groups"), "data"),
            State(cid("table"), "derived_virtual_row_ids"),
            State(cid("table"), "data"),
            prevent_initial_call=True,
        )
        def toggle_reference_objectives(enabled: Any, data: Any, visible: Any, rows: Any) -> tuple:
            ids = set(visible if visible is not None else [row["id"] for row in rows or []])
            groups = [g for g in (data or {}).get("groups", []) if g["id"] in ids]
            if not groups:
                raise PreventUpdate
            references, errors = reference_values(groups, enabled)
            trace = distribution_reference_trace(groups, references)
            patch = Patch()
            del patch["data"][2 * len(groups)]
            if trace is not None:
                patch["data"].append(trace.to_plotly_json())
            patch["layout"]["showlegend"] = trace is not None
            patch["layout"]["margin"]["t"] = 50 if trace is not None else 25
            patch["layout"]["legend"] = {"orientation": "h", "x": 0, "y": 1.12}
            description = distribution_description(groups, references)
            if errors:
                description += " Reference objectives could not be loaded: " + "; ".join(errors)
            return patch, description

        @app.callback(
            Output(cid("reference-panel"), "style"),
            Output(cid("content"), "style"),
            Output(cid("detail-heading"), "style"),
            Input(cid("tab"), "value"),
        )
        def reference_tab(tab: Any) -> tuple:
            hidden = {"display": "none"}
            return ({}, hidden, hidden) if tab == "reference-trials" else (hidden, {}, {})

        @app.callback(
            Output(cid("reference-results"), "data"),
            Output(cid("reference-search-status"), "children"),
            Input(cid("reference-search"), "n_clicks"),
            State(cid("groups"), "data"),
            State(cid("table"), "derived_virtual_row_ids"),
            State(cid("table"), "data"),
            prevent_initial_call=True,
        )
        def search_references(clicks: Any, data: Any, visible: Any, rows: Any) -> tuple:
            if not clicks:
                raise PreventUpdate
            ids = set(visible if visible is not None else [row["id"] for row in rows or []])
            groups = [g for g in (data or {}).get("groups", []) if g["id"] in ids]
            if not groups:
                return {"matches": {}}, note("No visible configurations to search for.")
            try:
                database = resolve_optuna_database(
                    c.get("optuna_database") or "", run_handler.get_working_directory()
                )
                result = find_reference_trials(database, groups, (data or {}).get("mappings", {}))
                for key, trials in result["matches"].items():
                    saved = c.get(REFERENCE_CACHE_KEY, key)
                    refreshed = next(
                        (trial for trial in trials if saved and trial["id"] == saved["id"]), None
                    )
                    if refreshed and refreshed != saved:
                        c.set(REFERENCE_CACHE_KEY, key, value=refreshed)
            except (ValueError, OSError, sqlite3.Error, ImportError) as error:
                return {"matches": {}}, note(f"Reference search failed: {error}")
            total = sum(len(trials) for trials in result["matches"].values())
            message = (
                f"Found {total} matching trials for {result['configurations']} configurations."
            )
            if result["missing_dataset"]:
                message += (
                    f" {result['missing_dataset']} configurations have no dataset metadata; "
                    "no dataset match can be established for them."
                )
            return result, note(message)

        @app.callback(
            Output(cid("reference-controls"), "children"),
            Input(cid("groups"), "data"),
            Input(cid("reference-results"), "data"),
            Input(cid("table"), "derived_virtual_row_ids"),
            Input(cid("table"), "data"),
        )
        def reference_controls(data: Any, results: Any, visible: Any, rows: Any) -> Any:
            ids = visible if visible is not None else [row["id"] for row in rows or []]
            lookup = {g["id"]: g for g in (data or {}).get("groups", [])}
            configurations = {}
            for group_id in ids:
                if group_id in lookup:
                    group = lookup[group_id]
                    configurations.setdefault(reference_key(group), group)
            controls = []
            for key, group in configurations.items():
                saved = c.get(REFERENCE_CACHE_KEY, key)
                matches = {
                    trial["id"]: trial for trial in (results or {}).get("matches", {}).get(key, [])
                }
                if saved:
                    matches.setdefault(saved["id"], saved)
                controls.append(
                    html.Div(
                        [
                            html.Strong(group["label"].split(" · ", 1)[0]),
                            dcc.Dropdown(
                                id={"type": cid("reference-trial"), "configuration": key},
                                options=[
                                    {"label": reference_label(trial), "value": trial["id"]}
                                    for trial in matches.values()
                                ],
                                value=saved["id"] if saved else None,
                                placeholder="Find reference trials, then select the original trial…",
                                clearable=True,
                                optionHeight=64,
                                maxHeight=320,
                            ),
                        ],
                        className="sensitivity-reference-row",
                    )
                )
            return controls or note("No configurations available for the current filters.")

        @app.callback(
            Output(cid("reference-save-status"), "children"),
            Input({"type": cid("reference-trial"), "configuration": ALL}, "value"),
            State({"type": cid("reference-trial"), "configuration": ALL}, "id"),
            State(cid("reference-results"), "data"),
            prevent_initial_call=True,
        )
        def save_reference(values: Any, identifiers: Any, results: Any) -> Any:
            triggered = ctx.triggered_id
            if not isinstance(triggered, dict):
                return ""
            if triggered not in (identifiers or []):
                return no_update
            key = triggered["configuration"]
            value = next(
                (value for component, value in zip(identifiers, values) if component == triggered),
                None,
            )
            saved = c.get(REFERENCE_CACHE_KEY, key)
            if (saved["id"] if saved else None) == value:
                return no_update
            trial = next(
                (
                    trial
                    for trial in (results or {}).get("matches", {}).get(key, [])
                    if trial["id"] == value
                ),
                None,
            )
            if value is not None and trial is None:
                return no_update
            c.set(REFERENCE_CACHE_KEY, key, value=trial)
            return note("Reference trial saved." if trial else "Reference trial cleared.")

        @app.callback(
            Output(cid("source"), "data"),
            Input(cid("mounted"), "data"),
            Input("global-run-selector", "value"),
        )
        def load_source(_mounted: Any, run_id: str) -> Dict:
            data: Dict[str, Any] = {
                "rows": [],
                "objectives": {},
                "keymap": {},
                "mappings": {
                    "model": c.get(MODEL_SEED_ATTRIBUTE_CACHE_KEY),
                    "data": c.get(DATA_SEED_ATTRIBUTE_CACHE_KEY),
                },
            }
            runs = {run.id: run for run in run_handler.get_runs(include_groups=True)}
            run = runs.get(run_id)
            if run is None or not self.check_run_compatibility(run):
                return data
            for source in (run.runs if isinstance(run, Group) else [run]):
                prepared = rc.get(source, self.id, "trials-v2")
                if prepared is None:
                    prepared = prepare_study(source)
                    rc.set(source, self.id, "trials-v2", value=prepared)
                data["rows"].extend(prepared["rows"])
                for key, info in prepared["objectives"].items():
                    data["objectives"].setdefault(key, info)
                if not data["keymap"]:
                    data["keymap"] = prepared["keymap"]
            return data

        @app.callback(
            Output(cid("groups"), "data"),
            Input(cid("source"), "data"),
            Input("global-main-objective-state", "data"),
        )
        def aggregate(data: Any, state: Any) -> Dict:
            data = data or {}
            modes = available_modes(data.get("mappings", {}))
            mode = "combined" if "combined" in modes else next(iter(modes), None)
            objective = central_objective(data, state)
            if not objective:
                return {"groups": [], "mappings": data.get("mappings", {})}
            direction = (
                (state or {}).get("direction") if objective.startswith("user_attr:") else None
            )
            if direction in ("lower", "upper"):
                data = {
                    **data,
                    "rows": [
                        (
                            {
                                **row,
                                "metric_info": {
                                    **row["metric_info"],
                                    objective: {
                                        **row["metric_info"][objective],
                                        "optimize": direction,
                                    },
                                },
                            }
                            if objective in row["metric_info"]
                            else row
                        )
                        for row in data["rows"]
                    ],
                }
            # A study can omit the chosen metric; exclude it without substituting another metric.
            target_direction = direction or data.get("objectives", {}).get(objective, {}).get(
                "optimize", "upper"
            )
            compatible = [
                row
                for row in data.get("rows", [])
                if objective in row["metric_info"]
                and row["metric_info"][objective]["optimize"] == target_direction
            ]
            incompatible = len(data.get("rows", [])) - len(compatible)
            data = {**data, "rows": compatible}
            result = build_groups(data, data.get("mappings", {}), mode, objective)
            if incompatible:
                result["excluded"][
                    "unavailable objectives or incompatible directions"
                ] = incompatible
            for group in result["groups"]:
                group["context_label"] = context_label(group)
            result["mappings"] = data.get("mappings", {})
            result["direction"] = target_direction
            return result

        @app.callback(
            Output(cid("context"), "options"),
            Output(cid("context"), "value"),
            Input(cid("groups"), "data"),
            State(cid("context"), "value"),
        )
        def contexts(data: Any, current: Any) -> tuple:
            labels = sorted({g["context_label"] for g in (data or {}).get("groups", [])})
            options = [{"label": "All contexts (kept separate)", "value": "all"}] + [
                {"label": label, "value": label} for label in labels
            ]
            return options, current if current in labels else "all"

        @app.callback(
            Output(cid("table"), "data"),
            Output(cid("table"), "columns"),
            Output(cid("table"), "sort_by"),
            Output(cid("table"), "tooltip_data"),
            Output(cid("summary"), "children"),
            Output(cid("status"), "children"),
            Input(cid("groups"), "data"),
            Input(cid("context"), "value"),
        )
        def explorer(data: Any, context: Any) -> tuple:
            data = data or {}
            all_groups = [
                g for g in data.get("groups", []) if context in (None, "all", g["context_label"])
            ]
            groups = eligible_groups(data, context)
            groups.sort(key=lambda g: g["mean"], reverse=data.get("direction") == "upper")
            rows, columns = table_projection(groups)
            stats = summary(all_groups)
            summary_text = (
                f"Configurations: {stats['configurations']} · "
                f"Seed evaluations: {stats['evaluations']} · "
                f"Median seeds / conditional config: {stats['median_seeds']:g} · "
                f"Comparable configurations: {stats['comparable']} · "
                f"{stats['model_seeds']} model seeds · {stats['data_seeds']} data seeds"
            )
            mappings = data.get("mappings", {})
            status = []
            if not available_modes(mappings):
                status.append(
                    dcc.Link(
                        "Map Model Seed and/or Data Seed under General → Optuna metadata.", href="/"
                    )
                )
            elif len(available_modes(mappings)) == 1:
                status.append(
                    "Only one seed type is mapped. Observed variability may include unknown "
                    "random sources; it is not a causally isolated seed effect. "
                )
            if not groups and available_modes(mappings):
                status.append("No comparable seed repetitions found for the current selection. ")
            if data.get("excluded"):
                status.append(
                    "Excluded: "
                    + ", ".join(f"{count} {name}" for name, count in data["excluded"].items())
                    + ". "
                )
            if any(g["duplicates"] for g in all_groups):
                status.append(
                    "Repeated identical seed pairs are retained as raw trials and averaged "
                    "per pair for seed statistics. "
                )
            return (
                rows,
                columns,
                [
                    {
                        "column_id": "mean",
                        "direction": "desc" if data.get("direction") == "upper" else "asc",
                    }
                ],
                [
                    {key: {"value": str(value), "type": "text"} for key, value in row.items()}
                    for row in rows
                ],
                summary_text,
                status,
            )

        @app.callback(
            Output(cid("scatter-container"), "children"),
            Input(cid("table"), "derived_virtual_row_ids"),
            Input(cid("table"), "data"),
            Input(cid("groups"), "data"),
            Input(cid("pareto"), "value"),
            Input(cid("active"), "data"),
            Input(cid("comparison"), "data"),
        )
        def scatter(
            visible: Any, rows: Any, data: Any, pareto: Any, active: Any, selected: Any
        ) -> Any:
            ids = set(visible if visible is not None else [r["id"] for r in rows or []])
            groups = [
                g for g in (data or {}).get("groups", []) if g["id"] in ids and g["sd"] is not None
            ]
            if not groups:
                return note("No comparable seed repetitions found for the current selection.")
            return graph(
                explorer_figure(
                    groups,
                    (data or {}).get("direction") == "upper",
                    bool(pareto),
                    active,
                    selected or [],
                ),
                "scatter",
            )

        app.clientside_callback(
            ClientsideFunction(namespace="sensitivity", function_name="visibleRows"),
            Output(cid("viewport"), "data"),
            Input(cid("table"), "derived_viewport_data"),
        )
        app.clientside_callback(
            ClientsideFunction(namespace="sensitivity", function_name="selection"),
            [
                Output(cid("active"), "data"),
                Output(cid("comparison"), "data"),
                Output(cid("table"), "selected_row_ids"),
                Output(cid("table"), "selected_rows"),
                Output(cid("table"), "active_cell"),
                Output(cid("table"), "style_data_conditional"),
            ],
            [
                Input(cid("table"), "active_cell"),
                Input(cid("table"), "selected_row_ids"),
                Input(cid("table"), "data"),
                Input(cid("table"), "derived_virtual_row_ids"),
                Input(cid("plot-event"), "data"),
                Input(cid("locate"), "n_clicks"),
            ],
            [
                State(cid("active"), "data"),
                State(cid("comparison"), "data"),
                State(cid("groups"), "data"),
                State("select-candidates-selection", "data"),
            ],
        )

        @app.callback(
            Output(cid("active-label"), "children"),
            Input(cid("active"), "data"),
            Input(cid("groups"), "data"),
        )
        def controls(active: Any, data: Any) -> str:
            data = data or {}
            lookup = {g["id"]: g for g in data.get("groups", [])}
            group = lookup.get(active)
            return group["label"] if group else "Configuration Details"

        @app.callback(
            Output(cid("content"), "children"),
            Output(cid("elite-control"), "style"),
            Input(cid("tab"), "value"),
            Input(cid("active"), "data"),
            Input(cid("comparison"), "data"),
            Input(cid("groups"), "data"),
            Input(cid("elite"), "value"),
            Input(cid("trial-selection"), "data"),
            Input(cid("table"), "derived_virtual_row_ids"),
            Input(cid("table"), "data"),
        )
        def details(
            tab: Any,
            active: Any,
            selected: Any,
            data: Any,
            elite: Any,
            trials: Any,
            visible: Any,
            rows: Any,
        ) -> tuple:
            ids = set(visible if visible is not None else [r["id"] for r in rows or []])
            data = {
                **(data or {}),
                "groups": [g for g in (data or {}).get("groups", []) if g["id"] in ids],
            }
            content = render_detail(
                tab,
                active,
                selected or [],
                data,
                elite,
                trials or [],
            )
            return content, {} if tab == "variance" else {"display": "none"}

        app.clientside_callback(
            ClientsideFunction(namespace="sensitivity", function_name="trialRows"),
            Output(cid("trial-view"), "data"),
            Input(cid("trials"), "derived_virtual_data"),
        )
        app.clientside_callback(
            "function(ids, current) { "
            "return JSON.stringify(ids || []) === JSON.stringify(current || []) "
            "? window.dash_clientside.no_update : (ids || []); }",
            Output(cid("trial-selection"), "data"),
            Input(cid("trials"), "selected_row_ids"),
            State(cid("trial-selection"), "data"),
            prevent_initial_call=True,
        )
        app.clientside_callback(
            ClientsideFunction(namespace="sensitivity", function_name="addCandidates"),
            [
                Output("select-candidates-selection", "data", allow_duplicate=True),
                Output(cid("candidate-status"), "children"),
            ],
            [Input(cid("add-config"), "n_clicks"), Input(cid("add-trials"), "n_clicks")],
            [
                State(cid("active"), "data"),
                State(cid("groups"), "data"),
                State(cid("trial-selection"), "data"),
                State("select-candidates-selection", "data"),
            ],
            prevent_initial_call=True,
        )
