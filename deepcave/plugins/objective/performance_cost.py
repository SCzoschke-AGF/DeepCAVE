"""Native trial-level analysis of performance gained for additional computational cost."""

from typing import Any, Callable, Dict, List, Optional, Union

from html import escape

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html
from dash.development.base_component import Component

from deepcave import config
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.plugins.summary.configurations import Configurations
from deepcave.runs import AbstractRun
from deepcave.runs.group import Group
from deepcave.utils.analysis_layout import (
    analysis_card,
    analysis_graph,
    analysis_toolbar_item,
    style_analysis,
)
from deepcave.utils.layout import get_select_options
from deepcave.utils.performance_cost import (
    extract_trial_data,
    filter_trial_data,
    prepare_plot_data,
)
from deepcave.utils.styled_plotty import (
    get_color,
    get_hovertext_from_config,
    save_image,
)

PLOT_HEIGHT = 620


class PerformanceCost(DynamicPlugin):
    """Show individual trial performance versus duration for the sidebar run and objective."""

    id = "performance_cost"
    name = "Performance ↔ Computational Cost"
    icon = "fas fa-balance-scale"
    main_objective_input = "objective_id"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register: Callable) -> List[Component]:  # type: ignore[override]
        """Return one compact control row."""
        return [
            html.Div(
                [
                    analysis_toolbar_item(
                        "Cost metric",
                        dbc.Select(
                            id=register("cost_metric", "value"),
                            options=[{"label": "Trial duration", "value": "duration"}],
                            value="duration",
                        ),
                        width="16rem",
                    ),
                    analysis_toolbar_item(
                        "Trials",
                        dbc.Select(id=register("trials", ["value", "options"])),
                        width="14rem",
                    ),
                    dbc.Checkbox(
                        id=register("show_pareto", "value", filter=True),
                        label="Pareto front",
                        value=True,
                        className="mb-0",
                    ),
                ],
                className="analysis-toolbar d-flex flex-wrap align-items-center gap-2",
            ),
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Component]:
        """Keep the display filter in the compact toolbar."""
        return []

    def load_inputs(self) -> Dict[str, Any]:
        """Initialize the duration metric, full trial set, and enabled frontier."""
        return {
            "objective_id": {"value": None, "options": []},
            "cost_metric": {"value": "duration"},
            "trials": {"value": "all", "options": [{"label": "All trials", "value": "all"}]},
            "show_pareto": {"value": True},
        }

    def load_dependency_inputs(
        self,
        run: Optional[Union[AbstractRun, List[AbstractRun]]],
        previous_inputs: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Populate the hidden objective index and reuse native multi-fidelity budgets."""
        if run is None or isinstance(run, list):
            return {}
        options = [{"label": "All trials", "value": "all"}]
        budgets = run.get_budgets(include_combined=False)
        if len(budgets) > 1:
            options.extend(
                {"label": f"Budget {budget}", "value": f"budget:{index}"}
                for index, budget in enumerate(budgets)
            )
        subset = inputs.get("trials", {}).get("value")
        if subset not in {option["value"] for option in options}:
            subset = "all"
        objective_id = inputs.get("objective_id", {}).get("value")
        ids = run.get_objective_ids()
        if objective_id not in ids:
            objective_id = ids[0] if ids else None
        return {
            "objective_id": {
                "value": objective_id,
                "options": get_select_options(run.get_objective_names(), ids),
            },
            "trials": {"value": subset, "options": options},
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and filter trials before preparing their Pareto frontier."""
        objective_id = inputs["objective_id"]
        objective = run.get_objective(objective_id)
        assert objective is not None
        subset = inputs.get("trials") or "all"
        budget = run.get_budget(int(subset.split(":", 1)[1])) if subset != "all" else None
        rows = filter_trial_data(extract_trial_data(run, objective_id), budget)
        return prepare_plot_data(rows, maximize=objective.optimize == "upper")

    @staticmethod
    def get_output_layout(register: Callable) -> Component:
        """Return one full-width plot with an understated card header."""
        return analysis_card(
            "Performance vs. Computational Cost",
            analysis_graph(register("graph", "figure"), PLOT_HEIGHT, config.FIGURE_DOWNLOAD_SCALE),
        )

    @staticmethod
    def load_outputs(
        run: Union[AbstractRun, Dict[str, AbstractRun]],
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> go.Figure:
        """Render muted trials and optionally their accent-colored frontier."""
        assert isinstance(run, AbstractRun)
        objective = run.get_objective(inputs["objective_id"])
        assert objective is not None
        rows = outputs["trials"]
        figure = go.Figure()
        sources = run.runs if isinstance(run, Group) else [run]
        hovertexts = []
        for row in rows:
            source = sources[row["source_index"]]
            hover = (
                f"<b>Trial {row['trial_id']}</b><br>"
                f"Run: {escape(source.name)}<br>"
                f"{escape(objective.name)}: {row['objective']:.4g}<br>"
                f"Trial duration: {row['duration']:.4g} s<br><br>"
            )
            hover += get_hovertext_from_config(source, row["config_id"], row["budget"])
            link = Configurations.get_link(source, row["config_id"])
            hover += f"<br><a href='{link}'>Open configuration</a>"
            hovertexts.append(hover)

        if rows:
            figure.add_trace(
                go.Scatter(
                    x=[row["duration"] for row in rows],
                    y=[row["objective"] for row in rows],
                    customdata=[[row["source_index"], row["history_id"]] for row in rows],
                    mode="markers",
                    marker={"color": "#94A3B8", "opacity": 0.5, "size": 6},
                    hovertext=hovertexts,
                    hoverinfo="text",
                    name="Trials",
                )
            )
            if inputs.get("show_pareto", True):
                indices = outputs["pareto_indices"]
                figure.add_trace(
                    go.Scatter(
                        x=[rows[index]["duration"] for index in indices],
                        y=[rows[index]["objective"] for index in indices],
                        customdata=[
                            [rows[index]["source_index"], rows[index]["history_id"]]
                            for index in indices
                        ],
                        mode="lines+markers",
                        marker={"color": get_color(0), "size": 9},
                        line={"color": get_color(0), "width": 1.5},
                        hovertext=[
                            "<b>Pareto-efficient</b><br>" + hovertexts[index] for index in indices
                        ],
                        hoverinfo="text",
                        name="Pareto front",
                    )
                )
        else:
            figure.add_annotation(
                text="No valid completed trials with duration data available.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
        style_analysis(figure, height=PLOT_HEIGHT, show_legend=False)
        figure.update_layout(hovermode="closest", margin={"t": 25, "b": 60, "l": 75, "r": 25})
        figure.update_xaxes(title_text="Trial duration [s]", type=outputs["xaxis_type"])
        figure.update_yaxes(title_text=objective.name)
        save_image(figure, "performance_cost.pdf")
        return figure
