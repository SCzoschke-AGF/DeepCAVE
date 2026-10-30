"""Compact run selection and shared presentation state for the comparison view."""

from typing import Any, Callable, Dict, List

from colorsys import hls_to_rgb

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import ALL, ctx, dcc, html, no_update
from dash.dependencies import ClientsideFunction, Input, Output, State
from dash.exceptions import PreventUpdate

from deepcave.utils.styled_plotty import get_color

PANEL_ID = "cost-over-time-comparison"
STATE_ID = f"{PANEL_ID}-state"
RUN_COLORS = [str(get_color(index)) for index in [2, 1, 3, 0, *range(4, 36)]]
DISTRIBUTION_MODE_ID = "cost-over-time-distribution-mode"


def prepare_distribution(figure: go.Figure) -> None:
    """Share histogram bins and provide both frequencies without changing trial data."""
    histograms = [trace for trace in figure.data if trace.type == "histogram"]
    values = [value for trace in histograms for value in trace.x if np.isfinite(value)]
    # Pad the shared range so extrema lie inside bins, including identical-valued runs.
    lower, upper = (min(values), max(values)) if values else (0.0, 0.0)
    padding = (upper - lower) / 80 if upper > lower else 0.5
    edges = np.histogram_bin_edges(values, bins=40, range=(lower - padding, upper + padding))
    for trace in histograms:
        counts, _ = np.histogram(trace.x, bins=edges)
        total = int(counts.sum())
        occupied = np.flatnonzero(counts)
        # Plotly drops leading/trailing empty bins from calcdata and indexes customdata
        # against the remaining bins. Keep internal empty bins to preserve that alignment.
        bins = range(int(occupied[0]), int(occupied[-1]) + 1) if total else []
        trace.customdata = [
            [edges[index], edges[index + 1], int(counts[index]), counts[index] / total]
            for index in bins
        ]
        trace.xbins = {"start": edges[0], "end": edges[-1], "size": edges[1] - edges[0]}
        trace.histnorm = "probability"
        trace.meta = {**(trace.meta or {}), "distribution_total": total}
        trace.hovertemplate = (
            "<b>%{fullData.name}</b><br>Bin: %{customdata[0]:.4g} – %{customdata[1]:.4g}"
            "<br>Trials: %{customdata[2]}<br>Relative frequency: %{customdata[3]:.1%}"
            "<br>Valid trials: %{meta.distribution_total}<extra></extra>"
        )
    figure.update_yaxes(title_text="Probability", tickformat=".0%")


def comparison_state(selected: List[str], previous: Dict[str, Any]) -> Dict[str, Any]:
    """Keep color assignments even after removal, separately from selected/hidden runs."""
    colors = dict(previous.get("colors", {}))
    for run_id in selected:
        if run_id not in colors:
            index = len(colors)
            if index < len(RUN_COLORS):
                colors[run_id] = RUN_COLORS[index]
            else:
                red, green, blue = hls_to_rgb((index * 0.61803398875) % 1, 0.45, 0.65)
                colors[run_id] = (
                    f"rgba({round(red * 255)}, {round(green * 255)}, {round(blue * 255)}, 1)"
                )
    return {
        "selected": selected,
        "colors": colors,
        "hidden": [run_id for run_id in previous.get("hidden", []) if run_id in selected],
    }


def comparison_layout(register: Callable) -> html.Div:
    """Reuse the native additional-run field as the canonical selection behind the panel."""
    return html.Div(
        [
            dcc.Dropdown(
                id=register("additional_runs", ["value", "options"]),
                multi=True,
                style={"display": "none"},
            ),
            dcc.Store(id=STATE_ID, data={}, storage_type="session"),
            html.Div(
                [
                    html.Strong("Run Comparison"),
                    html.Span(
                        "0 selected", id=f"{PANEL_ID}-count", className="run-comparison__count"
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-plus", **{"aria-hidden": "true"}), " Add run"],
                        id=f"{PANEL_ID}-add",
                        color="light",
                        size="sm",
                        n_clicks=0,
                    ),
                ],
                className="run-comparison__header",
            ),
            dbc.Collapse(
                dcc.Dropdown(
                    id=f"{PANEL_ID}-picker",
                    options=[],
                    placeholder="Select a run …",
                    clearable=False,
                    optionHeight=64,
                ),
                id=f"{PANEL_ID}-popover",
                is_open=False,
                className="run-comparison__popover",
            ),
            html.Div(id=f"{PANEL_ID}-rows", className="run-comparison__rows"),
        ],
        id=PANEL_ID,
        className="run-comparison",
    )


def comparison_rows(runs: List[Any], state: Dict[str, Any]) -> List[html.Div]:
    """Show full names and accessible actions, keeping the required primary run in place."""
    rows = []
    for index, run in enumerate(runs):
        hidden = run.id in state["hidden"]
        actions = [
            html.Button(
                html.I(
                    className="far fa-eye-slash" if hidden else "far fa-eye",
                    **{"aria-hidden": "true"},
                ),
                id={"type": "run-comparison-eye", "run": run.id},
                n_clicks=0,
                title=f"{'Show' if hidden else 'Hide'} {run.name}",
                **{
                    "aria-label": f"{'Show' if hidden else 'Hide'} {run.name}",
                    "aria-pressed": str(not hidden).lower(),
                },
                className="run-comparison__action",
            ),
        ]
        if index > 0:
            actions.append(
                html.Button(
                    html.I(className="fas fa-times", **{"aria-hidden": "true"}),
                    id={"type": "run-comparison-remove", "run": run.id},
                    n_clicks=0,
                    title=f"Remove {run.name}",
                    **{"aria-label": f"Remove {run.name}"},
                    className="run-comparison__action",
                )
            )
        rows.append(
            html.Div(
                [
                    html.Span(
                        className="run-comparison__color",
                        style={"backgroundColor": state["colors"][run.id]},
                    ),
                    html.Span(run.name, className="run-comparison__name"),
                    html.Div(actions, className="run-comparison__actions"),
                ],
                className="run-comparison__row"
                + (" run-comparison__row--hidden" if hidden else ""),
                **{"data-run-id": run.id},
            )
        )
    return rows


def register_comparison_callbacks(plugin: Any) -> None:
    """Connect panel actions to the existing run selection and browser-local view state."""
    from deepcave import app, run_handler

    selection_id = plugin.get_internal_input_id("additional_runs")

    distribution_id = plugin.get_internal_output_id("distribution")
    app.clientside_callback(
        ClientsideFunction(namespace="runComparison", function_name="distributionMode"),
        Output(distribution_id + "-graph", "figure"),
        Input(DISTRIBUTION_MODE_ID, "value"),
        Input(distribution_id, "data"),
    )

    @app.callback(
        Output(f"{PANEL_ID}-rows", "children"),
        Output(f"{PANEL_ID}-count", "children"),
        Output(STATE_ID, "data"),
        Output(f"{PANEL_ID}-picker", "options"),
        Input("global-run-selector", "value"),
        Input(selection_id, "value"),
        Input(selection_id, "options"),
        Input({"type": "run-comparison-eye", "run": ALL}, "n_clicks"),
        State(STATE_ID, "data"),
    )
    def update_panel(
        primary: str,
        additional: List[str],
        options: List[Any],
        _clicks: List[int],
        previous: Dict[str, Any],
    ) -> Any:
        previous = previous or {}
        selected = list(dict.fromkeys(([primary] if primary else []) + (additional or [])))
        state = comparison_state(selected, previous)
        trigger = ctx.triggered_id
        if isinstance(trigger, dict):
            if not any(item.get("value") for item in ctx.triggered):
                raise PreventUpdate
            run_id = trigger["run"]
            if run_id in selected:
                if run_id in state["hidden"]:
                    state["hidden"].remove(run_id)
                else:
                    state["hidden"].append(run_id)
        runs = [run_handler.get_run(run_id) for run_id in selected]
        available = [
            option
            for option in (options or [])
            if option.get("value") not in selected and not option.get("disabled")
        ]
        return (
            comparison_rows(runs, state),
            f"{len(selected)} selected",
            state if state != previous else no_update,
            available,
        )

    app.clientside_callback(
        ClientsideFunction(namespace="runComparison", function_name="selectRun"),
        Output(selection_id, "value", allow_duplicate=True),
        Output(f"{PANEL_ID}-picker", "value"),
        Output(f"{PANEL_ID}-popover", "is_open"),
        Input(f"{PANEL_ID}-add", "n_clicks"),
        Input(f"{PANEL_ID}-picker", "value"),
        Input({"type": "run-comparison-remove", "run": ALL}, "n_clicks"),
        State(selection_id, "value"),
        State("global-run-selector", "value"),
        State(f"{PANEL_ID}-popover", "is_open"),
        prevent_initial_call=True,
    )
