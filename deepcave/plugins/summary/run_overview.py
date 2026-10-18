"""Single-run objective summary with trial duration and fANOVA main effects."""

from typing import Any, Callable, Dict, List

import math

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html
from dash.development.base_component import Component
from plotly.subplots import make_subplots

from deepcave import config
from deepcave.plugins.cost_efficiency import importance_bar_figure, importance_figures
from deepcave.plugins.objective.cost_over_time import CostOverTime, performance_cost_data
from deepcave.plugins.objective.performance_cost import PerformanceCost
from deepcave.runs import AbstractRun
from deepcave.runs.group import Group
from deepcave.utils.analysis_layout import analysis_card, analysis_graph
from deepcave.utils.cost_efficiency import (
    calculate_importances,
    importance_comparison,
    select_elite_indices,
)
from deepcave.utils.layout import get_select_options
from deepcave.utils.styled_plotty import get_color, save_image


IMPORTANCE_OPTIONS = [
    {"label": "Objective Importance", "value": "objective"},
    {"label": "Cost Importance", "value": "cost"},
    {"label": "Cost vs. Objective Importance", "value": "comparison"},
]


class RunOverview(CostOverTime):
    """Summarize the globally selected run without additional comparison runs."""

    id = "run_overview"
    processing_version = 1
    name = "Run Overview"
    activate_run_selection = True
    allow_additional_runs = False
    help = "plugins/run_overview.html"

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """Accept individual runs supported by the shared trajectory analysis."""
        return not isinstance(run, Group)

    def check_runs_compatibility(self, runs: List[AbstractRun]) -> None:
        """Initialize objective and budget controls from the active run."""
        if not runs:
            return
        run = runs[0]
        self.objective_options = get_select_options(
            run.get_objective_names(), run.get_objective_ids()
        )
        self.budget_options = get_select_options(run.get_budgets(human=True), run.get_budget_ids())

    @staticmethod
    def get_input_layout(register: Callable) -> List[Component]:
        """Select trial highlighting and its top-percent threshold in a compact toolbar."""
        return [
            html.Div(
                [
                    dbc.Select(
                        id=register("budget_id", ["value", "options"], type=int),
                        style={"display": "none"},
                    ),
                    html.Div(
                        "Highlight",
                        className="multi-metrics-field__label mb-0",
                        style={"flex": "0 0 100%"},
                    ),
                    dbc.RadioItems(
                        id=register("highlight_mode", "value", filter=True),
                        options=[
                            {"label": "None", "value": "none"},
                            {"label": "Pareto front", "value": "pareto"},
                            {"label": "Elite trials", "value": "top"},
                        ],
                        value="pareto",
                        className="multi-metrics-segmented",
                        style={"flex": "0 0 50%"},
                    ),
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Elite trials (top N %)"),
                            dbc.Input(
                                id=register("top_percent", "value", filter=True),
                                type="number",
                                min=0.1,
                                max=100,
                                step=0.1,
                                value=5,
                            ),
                        ],
                        size="sm",
                        className="flex-nowrap",
                        style={"flex": "0 0 17rem", "width": "17rem"},
                    ),
                    dbc.Switch(
                        id=register("deactivate_tooltips", "value", filter=True),
                        label="deactivate tool tips",
                        value=False,
                        className="mb-0 ms-2",
                    ),
                ],
                className=(
                    "analysis-toolbar cost-over-time-toolbar "
                    "d-flex flex-wrap align-items-center gap-2"
                ),
            )
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Component]:
        """Register the cached-data view selector rendered below the importance plot."""
        register("importance_view", ["value", "options"])
        return []

    def load_inputs(self) -> Dict[str, Any]:
        """Reuse the objective and budget defaults with objective importance selected."""
        inputs = super().load_inputs()
        inputs.pop("elite_fraction")
        inputs.pop("xaxis")
        inputs.pop("show_pareto")
        inputs["highlight_mode"] = {"value": "pareto"}
        inputs["top_percent"] = {"value": 5}
        inputs["deactivate_tooltips"] = {"value": False}
        inputs["importance_view"] = {
            "value": "objective",
            "options": IMPORTANCE_OPTIONS,
        }
        return inputs

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reuse computational-cost data and objective importance samples."""
        outputs = {"performance_cost": performance_cost_data(run, inputs)}
        rows = [dict(row) for row in outputs["performance_cost"]["trials"]]
        for row in rows:
            row["configuration"] = dict(run.get_config(row["config_id"]))
        outputs["importances"] = calculate_importances(run, rows, objective_id=inputs["objective_id"])
        return outputs

    @staticmethod
    def get_output_layout(register: Callable) -> Component:  # type: ignore[override]
        """Embed complete analysis figures so their native layout is preserved."""
        return html.Div(
            [
                analysis_card(
                    "Performance vs. Computational Cost",
                    analysis_graph(register("graph", "figure"), 500, config.FIGURE_DOWNLOAD_SCALE),
                ),
                analysis_card(
                    "fANOVA Feature Importance",
                    html.Div(
                        [
                            analysis_graph(
                                register("objective_importance", "figure"),
                                500,
                                config.FIGURE_DOWNLOAD_SCALE,
                            ),
                            html.Div(
                                [
                                    html.Div("Importance", className="multi-metrics-field__label"),
                                    dbc.RadioItems(
                                        id="run_overview-importance_view-input",
                                        value="objective",
                                        options=IMPORTANCE_OPTIONS,
                                        className="multi-metrics-segmented",
                                    ),
                                ],
                                className="multi-metrics-field mx-3 mb-3",
                            ),
                        ]
                    ),
                    modifier="w-50",
                ),
            ]
        )

    @staticmethod
    def load_outputs(
        run: AbstractRun, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Dict[str, Any]:  # type: ignore[override]
        """Combine trial duration and objective distribution above feature importance."""
        importance = outputs["importances"]
        importance_view = inputs.get("importance_view", "objective")
        if importance_view == "comparison":
            _, importance_figure = importance_figures(
                run,
                {
                    "importances": importance,
                    "comparison": importance_comparison(
                        importance["cost"], importance["objective"]
                    ),
                },
            )
        else:
            target = "cost" if importance_view == "cost" else "objective"
            importance_figure = importance_bar_figure(
                run,
                importance[target],
                "Cost Importance" if target == "cost" else "Objective Importance",
                importance["errors"].get(target),
            )
        highlight_mode = inputs.get("highlight_mode", "pareto")
        performance = PerformanceCost.load_outputs(
            run, {**inputs, "show_pareto": highlight_mode == "pareto"}, outputs["performance_cost"]
        )
        rows = outputs["performance_cost"]["trials"]
        top_threshold = None
        if highlight_mode == "top" and rows:
            percent = min(100, max(0.1, float(inputs.get("top_percent") or 5)))
            objective = run.get_objective(inputs["objective_id"])
            indices = select_elite_indices(
                rows, fraction=percent / 100, maximize=objective.optimize == "upper"
            )
            elite_values = [rows[index]["objective"] for index in indices]
            top_threshold = (
                min(elite_values) if objective.optimize == "upper" else max(elite_values)
            )
            trials = performance.data[0]
            performance.add_trace(
                go.Scatter(
                    x=[trials.x[index] for index in indices],
                    y=[trials.y[index] for index in indices],
                    customdata=[trials.customdata[index] for index in indices],
                    hovertext=[trials.hovertext[index] for index in indices],
                    hoverinfo="text",
                    mode="markers",
                    marker={"color": get_color(0), "size": 9},
                    name=f"Top {percent:g}%",
                )
            )
        figure = make_subplots(
            rows=1,
            cols=2,
            shared_yaxes=True,
            column_widths=[0.8, 0.18],
            horizontal_spacing=0.02,
        )
        figure.update_layout(performance.layout)
        figure.update_layout(height=500)
        for trace in performance.data:
            figure.add_trace(trace, row=1, col=1)
        values = [row["objective"] for row in outputs["performance_cost"]["trials"]]
        color = get_color(0)
        figure.add_trace(
            go.Histogram(
                y=values,
                orientation="h",
                nbinsy=40,
                name="Trials",
                showlegend=False,
                marker={"color": color},
                hovertemplate="Objective: %{y}<br>Trials: %{x}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        figure.update_xaxes(
            title_text="Trials",
            showticklabels=False,
            ticks="",
            showgrid=False,
            rangemode="tozero",
            row=1,
            col=2,
        )
        figure.update_yaxes(
            title_text=None, showticklabels=False, ticks="", row=1, col=2
        )
        if values:
            objective = run.get_objective(inputs["objective_id"])
            best_value = max(values) if objective.optimize == "upper" else min(values)
            markers = [(best_value, f"{best_value:.6g}")]
            if top_threshold is not None and top_threshold != best_value:
                markers.append((top_threshold, f"{top_threshold:.6g}"))
            lower_marker = min(value for value, _ in markers) if len(markers) > 1 else None
            for value, label in markers:
                if highlight_mode in ("none", "pareto"):
                    best_duration = min(
                        row["duration"] for row in rows if row["objective"] == value
                    )
                    durations = [row["duration"] for row in rows]
                    logarithmic = figure.layout.xaxis.type == "log"
                    axis_values = (
                        [math.log10(duration) for duration in durations]
                        if logarithmic else durations
                    )
                    lower, upper = min(axis_values), max(axis_values)
                    padding = (upper - lower) * 0.05 if upper > lower else 0.5
                    # Fix the right edge so the data-anchored line joins the histogram exactly.
                    figure.update_xaxes(range=[lower - padding, upper + padding], row=1, col=1)
                    right_edge = 10 ** (upper + padding) if logarithmic else upper + padding
                    figure.add_shape(
                        type="line",
                        x0=best_duration,
                        x1=right_edge,
                        xref="x",
                        y0=value,
                        y1=value,
                        yref="y",
                        line={"dash": "dash", "color": color},
                    )
                figure.add_shape(
                    type="line",
                    x0=figure.layout.xaxis.domain[1] if highlight_mode in ("none", "pareto") else 0,
                    x1=1,
                    xref="paper",
                    y0=value,
                    y1=value,
                    yref="y",
                    line={"dash": "dash", "color": color},
                )
                figure.add_annotation(
                    text=label,
                    x=figure.layout.xaxis2.domain[0],
                    xref="paper",
                    xanchor="left",
                    y=value,
                    yref="y",
                    yanchor="top" if value == lower_marker else "bottom",
                    yshift=-6 if value == lower_marker else 6,
                    showarrow=False,
                    font={"color": color, "size": 12},
                    bgcolor="rgba(255,255,255,0.85)",
                    borderpad=2,
                )
            if top_threshold is not None and top_threshold != best_value:
                figure.add_annotation(
                    text=f"Δ {abs(best_value - top_threshold):.6g}",
                    x=1,
                    xref="paper",
                    xanchor="right",
                    y=(best_value + top_threshold) / 2,
                    yref="y",
                    yanchor="middle",
                    showarrow=False,
                    font={"color": color, "size": 12},
                    bgcolor="rgba(255,255,255,0.85)",
                    borderpad=2,
                )
        if inputs.get("deactivate_tooltips", False):
            figure.update_layout(hovermode=False)
            importance_figure.update_layout(hovermode=False)
        save_image(figure, "run_overview.pdf")
        return {
            "graph": {"figure": figure},
            "objective_importance": {"figure": importance_figure},
        }
