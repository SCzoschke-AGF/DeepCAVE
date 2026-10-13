"""Native parameter and configuration analyses connecting performance to trial duration."""

from typing import Any, Callable, Dict, List, Optional, Union

from html import escape

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from ConfigSpace.hyperparameters import NumericalHyperparameter
from dash import ctx, dash_table, html
from dash.dash_table.Format import Format, Scheme
from dash.development.base_component import Component

from deepcave import config
from deepcave.plugins.objective.performance_cost import PerformanceCost
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import AbstractRun
from deepcave.runs.group import Group
from deepcave.utils.analysis_layout import (
    analysis_card,
    analysis_graph,
    analysis_toolbar_item,
    style_analysis,
)
from deepcave.utils.cost_efficiency import (
    calculate_importances,
    duration_trend,
    elite_summary,
    importance_comparison,
    select_elite_indices,
)
from deepcave.utils.layout import get_select_options
from deepcave.utils.performance_cost import (
    extract_trial_data,
    filter_trial_data,
    prepare_plot_data,
)
from deepcave.utils.styled_plotty import get_color


def _empty_figure(message: str) -> go.Figure:
    """Use the standard annotated empty plot for unavailable estimates or trial data."""
    figure = go.Figure()
    figure.add_annotation(
        text=escape(message), x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
    )
    style_analysis(figure, show_legend=False)
    return figure


def importance_bar_figure(
    run: AbstractRun, values: Dict[str, float], label: str, error: Optional[str] = None
) -> go.Figure:
    """Render reusable main-effect bars with selectable parameter labels."""
    colors = {name: get_color(index) for index, name in enumerate(run.configspace.keys())}
    if values:
        names = sorted(values, key=lambda name: (-values[name], name))
        figure = go.Figure(
            go.Bar(
                x=[values[name] for name in names],
                y=names,
                orientation="h",
                customdata=names,
                marker_color=[colors[name] for name in names],
                hovertemplate=f"%{{y}}<br>{label}: %{{x:.4f}}<extra></extra>",
            )
        )
        style_analysis(figure, show_legend=False, height=max(500, 140 + 28 * len(names)))
        figure.update_layout(
            xaxis_title=label,
            yaxis=dict(
                autorange="reversed",
                showticklabels=False,
                categoryorder="array",
                categoryarray=names,
            ),
            margin_l=max(75, 7 * max(map(len, names)) + 16),
        )
        # Native annotation click events make even zero-length bars selectable by name.
        for name in names:
            figure.add_annotation(
                name=name,
                text=escape(name),
                x=0,
                xref="paper",
                xanchor="right",
                xshift=-5,
                y=name,
                yref="y",
                yanchor="middle",
                showarrow=False,
                captureevents=True,
                font=dict(size=12),
            )
    else:
        figure = _empty_figure(error or f"{label} unavailable.")

    return figure


def importance_figures(
    run: AbstractRun, outputs: Dict[str, Any], parameter: Optional[str] = None
) -> List[go.Figure]:
    """Prepare horizontal cost main effects and the objective-versus-cost comparison."""
    importance = outputs["importances"]
    colors = {name: get_color(index) for index, name in enumerate(run.configspace.keys())}
    cost, objective = importance["cost"], importance["objective"]
    cost_figure = importance_bar_figure(
        run, cost, "Cost Importance", importance["errors"].get("cost")
    )

    comparison = outputs["comparison"]
    names = comparison["parameters"]
    if not names:
        return [
            cost_figure,
            _empty_figure(
                importance["errors"].get("objective")
                or importance["errors"].get("cost")
                or "No comparable importance estimates available."
            ),
        ]
    scatter = go.Figure(
        go.Scatter(
            x=[cost[name] for name in names],
            y=[objective[name] for name in names],
            customdata=names,
            text=names,
            mode="markers",
            marker_size=12,
            marker_color=[colors[name] for name in names],
            marker_line={
                "color": "black",
                "width": [2 if name == parameter else 0 for name in names],
            },
            hovertemplate="%{text}<br>Cost Importance: %{x:.4f}"
            "<br>Objective Importance: %{y:.4f}<extra></extra>",
        )
    )
    style_analysis(scatter, show_legend=False)
    scatter.update_layout(xaxis_title="Cost Importance", yaxis_title="Objective Importance")
    return [cost_figure, scatter]


def parameter_figure(
    run: AbstractRun,
    rows: List[Dict[str, Any]],
    parameter: Optional[str],
    show_trend: bool = False,
    elite_rows: Optional[List[Dict[str, Any]]] = None,
    target: str = "duration",
    target_label: Optional[str] = None,
) -> go.Figure:
    """Show observations for numeric parameters or distributions in declared category order."""
    label = target_label or ("Trial duration [s]" if target == "duration" else "Objective")
    unit = " s" if target == "duration" else ""
    if parameter is None or parameter not in run.configspace:
        return _empty_figure("Select a hyperparameter to inspect observed trials.")
    observed = [row for row in rows if row["configuration"].get(parameter) is not None]
    if not observed:
        return _empty_figure("No valid completed trials with this parameter active.")
    hp = run.configspace[parameter]
    color = get_color(list(run.configspace.keys()).index(parameter))
    values = [row["configuration"][parameter] for row in observed]
    durations = [row[target] for row in observed]
    hover = [
        f"Trial {row['trial_id']}<br>{escape(parameter)}: {escape(str(value))}"
        f"<br>{escape(label)}: {row[target]:.4g}"
        for row, value in zip(observed, values)
    ]
    if isinstance(hp, NumericalHyperparameter):
        trace = go.Scatter(
            x=values,
            y=durations,
            mode="markers",
            marker_color=color,
            marker={"size": 7, "opacity": 0.3},
            hovertext=hover,
            hoverinfo="text",
        )
    else:
        trace = go.Box(
            x=[str(value) for value in values],
            y=durations,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            marker_color=color,
            hovertext=hover,
            hoverinfo="text",
        )
    figure = go.Figure(trace)
    if show_trend and isinstance(hp, NumericalHyperparameter):
        trend = duration_trend(observed, parameter, target=target)
        if trend["x"]:
            # Blend the parameter's RGB color with black to retain its hue.
            red, green, blue = (
                round(int(channel) * 0.65) for channel in str(color)[5:-1].split(",")[:3]
            )
            figure.add_trace(
                go.Scatter(
                    x=trend["x"] + trend["x"][::-1],
                    y=trend["upper"] + trend["lower"][::-1],
                    fill="toself",
                    fillcolor="rgba(148,163,184,0.2)",
                    line_width=0,
                    hoverinfo="skip",
                    name="25th–75th percentile",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=trend["x"],
                    y=trend["median"],
                    mode="lines",
                    line={"color": f"rgb({red}, {green}, {blue})", "width": 3},
                    name="Binned median",
                    hovertemplate=f"Binned median: %{{y:.4g}}{unit}<extra></extra>",
                )
            )
    if elite_rows:
        elite = [row for row in elite_rows if row["configuration"].get(parameter) is not None]
        figure.add_trace(
            go.Scatter(
                x=[
                    (
                        row["configuration"][parameter]
                        if isinstance(hp, NumericalHyperparameter)
                        else str(row["configuration"][parameter])
                    )
                    for row in elite
                ],
                y=[row[target] for row in elite],
                mode="markers",
                name="Elite trials",
                marker={
                    "color": color,
                    "size": 10,
                    "symbol": "diamond",
                    "opacity": 1,
                    "line": {"color": "#334155", "width": 1.5},
                },
                hovertext=[
                    f"Elite trial {row['trial_id']}<br>Objective: {row['objective']:.4g}"
                    f"<br>{escape(parameter)}: "
                    f"{escape(str(row['configuration'][parameter]))}"
                    f"<br>Trial duration: {row['duration']:.4g} s"
                    for row in elite
                ],
                hoverinfo="text",
            )
        )
    style_analysis(figure, show_legend=False)
    figure.update_layout(xaxis_title=parameter, yaxis_title=label)
    figure.update_yaxes(
        type=prepare_plot_data(observed)["xaxis_type"] if target == "duration" else "linear"
    )
    if isinstance(hp, NumericalHyperparameter):
        figure.update_xaxes(type="log" if getattr(hp, "log", False) else "linear")
    else:
        categories = getattr(hp, "choices", getattr(hp, "sequence", values))
        figure.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=[str(value) for value in categories],
        )
    return figure


def elite_figure(
    run: AbstractRun, inputs: Dict[str, Any], rows: List[Dict[str, Any]], maximize: bool
) -> List[Any]:
    """Render only the elite set and its Pareto front with native drilldown."""
    order = select_elite_indices(rows, fraction=1.0, maximize=maximize)
    ranks = {index: rank for rank, index in enumerate(order, 1)}
    indices = select_elite_indices(rows, float(inputs.get("elite_fraction") or 0.05), maximize)
    elite = [rows[index] for index in indices]
    plot_data = prepare_plot_data(elite, maximize)
    figure = PerformanceCost.load_outputs(run, inputs, plot_data)
    if elite:
        trace = figure.data[0]
        trace.name = "Elite trials"
        trace.marker.color = get_color(0)
        trace.marker.opacity = 0.7
        trace.hovertext = [
            f"{hover}<br>Objective rank: {ranks[index]} / {len(rows)}"
            f" (top {100 * ranks[index] / len(rows):.1f}%)"
            for hover, index in zip(trace.hovertext, indices)
        ]
    stats = elite_summary(elite)
    summary = "No valid elite trials available."
    if elite:
        ratio = (
            f"{stats['ratio']:.2f}×" if stats["ratio"] is not None else "undefined (zero minimum)"
        )
        summary = (
            f"Elite trials: {stats['count']} · Min: {stats['minimum']:.4g} s · "
            f"Median: {stats['median']:.4g} s · Max: {stats['maximum']:.4g} s · "
            f"Cost ratio (max/min): {ratio}"
        )
    return [figure, summary]


class CostEfficiency(StaticPlugin):
    """Connect fANOVA main effects to observed duration and configuration efficiency."""

    id = "cost_efficiency"
    name = "Compuational Cost"
    description = (
        "Understand which hyperparameters drive computational cost, where cost can be reduced, "
        "and which configurations are compute-efficient."
    )
    icon = "fas fa-balance-scale"
    help = "plugins/cost_efficiency.html"
    activate_run_selection = True
    main_objective_input = "objective_id"
    configuration_output_ids = ("performance", "elite")
    input_update_inputs = (
        ("cost_click", "cost_efficiency-cost_importance-output", "clickData"),
        ("comparison_click", "cost_efficiency-comparison-output", "clickData"),
        ("candidate_cell", "cost_efficiency-candidates-output", "active_cell"),
        ("cost_label_click", "cost_efficiency-cost_importance-output", "clickAnnotationData"),
        ("comparison_label_click", "cost_efficiency-comparison-output", "clickAnnotationData"),
    )

    @staticmethod
    def get_input_layout(register: Callable) -> List[Component]:  # type: ignore[override]
        """Reuse native trial/budget selection without an additional objective selector."""
        return [
            analysis_toolbar_item(
                "Trials", dbc.Select(id=register("trials", ["value", "options"])), "14rem"
            )
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Component]:
        """Register inexpensive view controls, excluded from the fANOVA processing cache."""
        # The view control is rendered below the lower parameter-analysis row.
        register("detail_view", "value")
        register("importance_view", "value")
        return [
            html.Div(
                [
                    analysis_toolbar_item(
                        "Parameter",
                        dbc.Select(id=register("parameter", ["value", "options"])),
                        "16rem",
                    ),
                    dbc.Checkbox(
                        id=register("show_pareto", "value"), label="Pareto front", value=True
                    ),
                    dbc.Checkbox(
                        id=register("show_trend", "value"), label="Duration trend", value=True
                    ),
                    dbc.Checkbox(
                        id=register("mark_elite", "value"), label="Mark elite trials", value=False
                    ),
                    analysis_toolbar_item(
                        "Elite trials",
                        dbc.Select(
                            id=register("elite_fraction", "value"),
                            value="0.05",
                            options=[
                                {"label": f"Top {percent}%", "value": str(percent / 100)}
                                for percent in (1, 5, 10, 20)
                            ],
                        ),
                        "12rem",
                    ),
                ],
                className="analysis-toolbar d-flex flex-wrap align-items-center gap-2",
            )
        ]

    def load_inputs(self) -> Dict[str, Any]:
        """Default to all valid trials, enabled trend/Pareto front and the top five percent."""
        return {
            "objective_id": {"value": None, "options": []},
            "trials": {"value": "all", "options": [{"label": "All trials", "value": "all"}]},
            "show_pareto": {"value": True},
            "parameter": {"value": None, "options": []},
            "elite_fraction": {"value": "0.05"},
            "show_trend": {"value": True},
            "mark_elite": {"value": False},
            "detail_view": {"value": 0},
            "importance_view": {"value": 0},
        }

    def load_dependency_inputs(
        self,
        run: Optional[Union[AbstractRun, List[AbstractRun]]],
        previous_inputs: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Reuse native objective and budget options; retain an available selected parameter."""
        # The shared method is stateless: it reads only the supplied run and controls.
        dependencies = PerformanceCost.load_dependency_inputs(
            self, run, previous_inputs, inputs  # type: ignore[arg-type]
        )
        # Keep previously saved dropdown choices compatible with numeric control values.
        dependencies["detail_view"] = {
            "value": int(inputs.get("detail_view", {}).get("value") in (1, "objective"))
        }
        if run is None or isinstance(run, list):
            return dependencies
        names = list(run.configspace.keys())
        selected = inputs.get("parameter", {}).get("value")
        dependencies["parameter"] = {
            "options": get_select_options(names),
            "value": selected if selected in names else (names[0] if names else None),
        }
        return dependencies

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract valid trials once, then calculate both targets on precisely those samples."""
        objective_id = inputs["objective_id"]
        objective = run.get_objective(objective_id)
        assert objective is not None
        subset = inputs.get("trials") or "all"
        budget = run.get_budget(int(subset.split(":", 1)[1])) if subset != "all" else None
        rows = filter_trial_data(extract_trial_data(run, objective_id), budget)
        sources = run.runs if isinstance(run, Group) else [run]
        for row in rows:
            row["configuration"] = dict(sources[row["source_index"]].get_config(row["config_id"]))
        importance = calculate_importances(run, rows)
        return {
            **prepare_plot_data(rows, maximize=objective.optimize == "upper"),
            "maximize": objective.optimize == "upper",
            "importances": importance,
            "comparison": importance_comparison(importance["cost"], importance["objective"]),
        }

    @staticmethod
    def get_output_layout(register: Callable) -> Component:
        """Keep parameter and complete-configuration workflows visibly grouped."""

        def graph(name: str) -> Component:
            return analysis_graph(register(name, "figure"), 500, config.FIGURE_DOWNLOAD_SCALE)

        candidate_filter = "{cost_candidate} = 1"
        return html.Div(
            [
                html.H4("Parameter-level analysis", className="mt-3"),
                html.Small(
                    "fANOVA main effects on valid completed trials; trial duration in seconds."
                ),
                dbc.Row(
                    [
                        dbc.Col(analysis_card("Cost Importance", graph("cost_importance")), lg=6),
                        dbc.Col(
                            analysis_card(
                                "Parameter vs. Trial Duration",
                                html.Div(
                                    [
                                        graph("parameter_duration"),
                                    ]
                                ),
                            ),
                            lg=6,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            analysis_card("Cost vs. Objective Importance", graph("comparison")),
                            lg=6,
                        ),
                        dbc.Col(
                            analysis_card(
                                "Parameter Importances",
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dash_table.DataTable(
                                                    id=register(
                                                        "candidates",
                                                        ["data", "style_data_conditional"],
                                                    ),
                                                    columns=[
                                                        {
                                                            "name": label,
                                                            "id": key,
                                                            **(
                                                                {
                                                                    "type": "numeric",
                                                                    "format": Format(
                                                                        precision=4,
                                                                        scheme=Scheme.fixed,
                                                                    ),
                                                                }
                                                                if key.endswith("importance")
                                                                else {}
                                                            ),
                                                        }
                                                        for label, key in (
                                                            ("Parameter", "parameter"),
                                                            ("Cost Importance", "cost_importance"),
                                                            (
                                                                "Objective Importance",
                                                                "objective_importance",
                                                            ),
                                                        )
                                                    ],
                                                    data=[],
                                                    cell_selectable=True,
                                                    page_action="none",
                                                    style_table={"overflowX": "auto"},
                                                    style_cell={"textAlign": "left"},
                                                    style_data_conditional=[
                                                        {
                                                            "if": {
                                                                "filter_query": candidate_filter
                                                            },
                                                            "backgroundColor": "#FEF3C7",
                                                            "color": "#78350F",
                                                        }
                                                    ],
                                                ),
                                                html.Div(
                                                    id=register("candidate_note", "children"),
                                                    className="small text-muted",
                                                ),
                                            ],
                                            id=register("importance_table", "style"),
                                            style={"paddingTop": "65px"},
                                        ),
                                        html.Div(
                                            graph("parameter_objective"),
                                            id=register("objective_panel", "style"),
                                            style={"display": "none"},
                                        ),
                                    ],
                                ),
                            ),
                            lg=6,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("View", className="multi-metrics-field__label"),
                                dbc.RadioItems(
                                    id="cost_efficiency-importance_view-input",
                                    value=0,
                                    options=[
                                        {"label": "Importance Relation", "value": 0},
                                        {"label": "Objective Importance", "value": 1},
                                    ],
                                    className="multi-metrics-segmented",
                                ),
                            ],
                            className="multi-metrics-field",
                            style={"width": "23rem", "maxWidth": "100%"},
                        ),
                        html.Div(
                            [
                                html.Div("View", className="multi-metrics-field__label"),
                                dbc.RadioItems(
                                    id="cost_efficiency-detail_view-input",
                                    value=0,
                                    options=[
                                        {"label": "Table", "value": 0},
                                        {"label": "Objective", "value": 1},
                                    ],
                                    className="multi-metrics-segmented",
                                ),
                            ],
                            className="multi-metrics-field",
                            style={"width": "14rem", "maxWidth": "100%"},
                        ),
                    ],
                    className="analysis-toolbar d-flex justify-content-between flex-wrap mt-2",
                ),
                html.H4("Configuration-level analysis", className="mt-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            analysis_card(
                                "Performance vs. Computational Cost", graph("performance")
                            ),
                            lg=6,
                        ),
                        dbc.Col(
                            analysis_card(
                                "Near-Optimal Cost Efficiency",
                                html.Div(
                                    [
                                        graph("elite"),
                                        html.Div(
                                            id=register("elite_summary", "children"),
                                            className="small py-2",
                                        ),
                                    ]
                                ),
                            ),
                            lg=6,
                        ),
                    ]
                ),
            ]
        )

    @staticmethod
    def load_outputs(
        run: Union[AbstractRun, Dict[str, AbstractRun]],
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Redraw selected parameter, Pareto visibility and elite fraction from cached data."""
        assert isinstance(run, AbstractRun)
        cost, comparison = importance_figures(run, outputs, parameter=inputs.get("parameter"))
        if inputs.get("importance_view", 0) == 1:
            comparison = importance_bar_figure(
                run,
                outputs["importances"]["objective"],
                "Objective Importance",
                outputs["importances"]["errors"].get("objective"),
            )
        elite, summary = elite_figure(run, inputs, outputs["trials"], outputs["maximize"])
        elite_rows = (
            [
                outputs["trials"][index]
                for index in select_elite_indices(
                    outputs["trials"],
                    fraction=float(inputs.get("elite_fraction", "0.05")),
                    maximize=outputs["maximize"],
                )
            ]
            if inputs.get("mark_elite", False)
            else None
        )
        objective = run.get_objective(inputs["objective_id"])
        assert objective is not None
        objective_view = inputs.get("detail_view", 0) in (1, "objective")
        candidates = outputs["comparison"]["candidates"]
        candidate_names = {row["parameter"] for row in candidates}
        importances = outputs["importances"]
        table_rows = [
            {
                "id": name,
                "parameter": name,
                "objective_importance": importances["objective"].get(name),
                "cost_importance": importances["cost"].get(name),
                "cost_candidate": int(name in candidate_names),
            }
            for name in sorted(
                run.configspace.keys(),
                key=lambda name: (-importances["cost"].get(name, -1), name),
            )
        ]
        color_indices = {name: index for index, name in enumerate(run.configspace.keys())}
        return {
            "cost_importance": {"figure": cost},
            "comparison": {"figure": comparison},
            "candidates": {
                "data": table_rows,
                "style_data_conditional": [
                    {
                        "if": {"filter_query": "{cost_candidate} = 1"},
                        "backgroundColor": "#FEF3C7",
                        "color": "#78350F",
                    },
                    *[
                        {
                            "if": {"row_index": row_index, "state": state},
                            "backgroundColor": get_color(color_indices[row["parameter"]], 0.25),
                            "border": f"1px solid {get_color(color_indices[row['parameter']])}",
                            "color": "#212529",
                        }
                        for row_index, row in enumerate(table_rows)
                        for state in ("active", "selected")
                    ],
                ],
            },
            "candidate_note": {
                "children": (
                    ""
                    if candidates
                    else (
                        "No candidates meet the strict median thresholds."
                        if outputs["comparison"]["parameters"]
                        else "Candidate screening requires finite cost and "
                        "objective importance estimates."
                    )
                )
            },
            "parameter_duration": {
                "figure": parameter_figure(
                    run,
                    outputs["trials"],
                    inputs.get("parameter"),
                    inputs.get("show_trend", True),
                    elite_rows=elite_rows,
                )
            },
            "parameter_objective": {
                "figure": parameter_figure(
                    run,
                    outputs["trials"],
                    inputs.get("parameter"),
                    inputs.get("show_trend", True),
                    elite_rows=elite_rows,
                    target="objective",
                    target_label=objective.name,
                )
            },
            "importance_table": {
                "style": {"paddingTop": "65px", "display": "none" if objective_view else "block"}
            },
            "objective_panel": {"style": {"display": "block" if objective_view else "none"}},
            "performance": {"figure": PerformanceCost.load_outputs(run, inputs, outputs)},
            "elite": {"figure": elite},
            "elite_summary": {"children": summary},
        }

    def load_input_updates(self, events: Dict[str, Any]) -> Dict[str, Any]:
        """Select parameters through the native control callback without duplicate writers."""
        for name, component_id, property_name in self.input_update_inputs:
            if f"{component_id}.{property_name}" not in ctx.triggered_prop_ids:
                continue
            event = events.get(name)
            if not event:
                return {}
            if name == "candidate_cell":
                parameter = event.get("row_id")
            elif name in ("cost_label_click", "comparison_label_click"):
                parameter = event.get("annotation", {}).get("name")
            else:
                points = event.get("points") or []
                parameter = points[0].get("customdata") if points else None
            return {"parameter": {"value": str(parameter)}} if parameter is not None else {}
        return {}
