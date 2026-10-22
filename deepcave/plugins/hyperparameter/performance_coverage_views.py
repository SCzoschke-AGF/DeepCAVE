"""Compact Performance-aware Coverage panel and independently queued analysis."""

from typing import Any, Dict

import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from deepcave.evaluators.performance_coverage import (
    PERFORMANCE_VERSION,
    _digest,
    analyze_performance_distances,
    performance_source_key,
    prepare_performance_trials,
    threshold_statistics,
)
from deepcave.evaluators.rashomon import cached_pairwise_configuration_distances
from deepcave.plugins.hyperparameter.coverage_views import _empty_figure, _id
from deepcave.utils.global_objective import resolve_main_objective_id

_JOB_LOCK = RLock()


def run_performance_analysis(run: Any, objective_id: int, directory: str) -> Dict[str, Any]:
    """Reuse Rashomon's global fANOVA weights and exact distance implementation in a worker."""
    from deepcave.plugins.hyperparameter.near_optimal_configurations import (
        _fanova_importances,
        _run_version,
    )

    objective = run.get_objective(objective_id)
    if objective is None:
        raise ValueError("The selected sidebar objective is unavailable for this study.")
    trials = prepare_performance_trials(run, objective_id)
    if len(trials["trial_ids"]) < 2:
        raise ValueError("At least two successful trials with a finite objective are required.")
    importance = _fanova_importances(run, objective_id)
    weights = importance["values"]
    if not weights or sum(weights.values()) <= 0:
        raise ValueError(
            "Importance-weighted distance is unavailable: "
            + str(importance["error"] or "No positive global fANOVA main effects.")
        )

    source_key = performance_source_key(run, objective_id)
    key = _digest([source_key, weights, "rashomon-rms-v1", PERFORMANCE_VERSION])
    folder = Path(directory) / key
    summary_path = folder / "summary.json"
    differences_path = folder / "differences.npy"
    if summary_path.exists() and differences_path.exists():
        return json.loads(summary_path.read_text())

    names = list(run.configspace.keys())
    weight_signature = tuple(sorted((name, round(float(v), 12)) for name, v in weights.items()))
    # Keep the caller key identical to Rashomon's when the trial cohort is compatible.
    distance_key = (
        run.id,
        _run_version(run),
        tuple(trials["trial_ids"]),
        "importance_weighted",
        tuple(names),
        weight_signature,
    )
    disk_key = _digest(
        [
            run.configspace.to_serialized_dict(),
            trials["matrix"].tolist(),
            weights,
            "rashomon-rms-v1",
        ]
    )
    distance_path = Path(directory) / "distances" / f"{disk_key}.npy"
    if distance_path.exists():
        distances = np.load(distance_path, mmap_mode="r", allow_pickle=False)
    else:
        distances = cached_pairwise_configuration_distances(
            distance_key, trials["matrix"], run.configspace, names, weights
        )
        distance_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = distance_path.with_name(f"{uuid4().hex}.npy")
        np.save(temporary, distances, allow_pickle=False)
        temporary.replace(distance_path)

    folder.mkdir(parents=True, exist_ok=True)
    temporary_differences = folder / f"{uuid4().hex}.npy"
    result = analyze_performance_distances(
        distances, trials["objectives"], trials["trial_ids"], temporary_differences
    )
    temporary_differences.replace(differences_path)
    result.update(
        cache_key=key,
        source_key=source_key,
        objective_id=objective_id,
        objective=objective.name,
        excluded=trials["excluded"],
        history_indices=trials["history_indices"],
        weights=weights,
        weight_source=f"Global fANOVA main effects · successful trials · highest budget "
        f"{run.get_highest_budget():g} · 16 trees · seed 0",
        performance_version=PERFORMANCE_VERSION,
    )
    temporary_summary = folder / f"{uuid4().hex}.json"
    temporary_summary.write_text(json.dumps(result, allow_nan=False))
    temporary_summary.replace(summary_path)
    return result


def performance_panel() -> Any:
    """Build only the new right-hand analysis panel."""
    return html.Section(
        [
            dcc.Store(id=_id("performance-data")),
            dcc.Store(id=_id("performance-view")),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                html.I(className="fas fa-chart-bar"),
                                className="coverage-performance-icon",
                                **{"aria-hidden": "true"},
                            ),
                            html.H3("Performance-aware Coverage"),
                        ],
                        className="coverage-performance-title",
                    ),
                    dbc.Button(
                        "Refresh",
                        id=_id("performance-run"),
                        n_clicks=0,
                        color="secondary",
                        outline=True,
                        size="sm",
                        title="Analyze all valid trial pairs for the sidebar objective",
                    ),
                ],
                className="coverage-performance-header",
            ),
            html.P(
                "Relate configuration distance to observed performance differences.",
                className="coverage-performance-description",
            ),
            html.Div(id=_id("performance-status"), role="status"),
            html.Div(
                [
                    html.Div([html.Span(label), html.Strong("—", id=_id(f"performance-{name}"))])
                    for name, label in (
                        ("trials", "Trials analyzed"),
                        ("pairs", "Pairwise comparisons"),
                        ("objective", "Selected objective"),
                    )
                ],
                className="coverage-performance-metrics",
            ),
            html.Div(
                [
                    html.H4("Performance sensitivity by configuration distance"),
                    dcc.Graph(
                        id=_id("performance-chart"),
                        figure=performance_figure(None),
                        config={"displaylogo": False},
                        responsive=True,
                        style={"height": "330px"},
                    ),
                    html.P(
                        id=_id("performance-interpretation"),
                        className="coverage-performance-interpretation",
                    ),
                    html.H4("Nearest-neighbor sensitivity"),
                    html.Div(
                        [
                            dbc.Label(
                                "Relevant difference: Δ >",
                                html_for=_id("performance-threshold"),
                            ),
                            dbc.Input(
                                id=_id("performance-threshold"),
                                type="number",
                                min=0,
                                step="any",
                                value=0.01,
                                debounce=True,
                                size="sm",
                            ),
                            html.Span(id=_id("performance-unit")),
                        ],
                        className="coverage-performance-threshold",
                    ),
                    html.Div(id=_id("performance-threshold-error"), role="status"),
                    html.Div(id=_id("performance-neighbors")),
                ],
                className="coverage-performance-body",
            ),
            html.Details(
                [html.Summary("Method & distance weights"), html.Div(id=_id("performance-method"))],
                className="coverage-performance-method",
            ),
            html.Small(
                "Observed trial pairs and leave-one-out nearest neighbors; "
                "not calibrated predictions for unseen configurations.",
                className="coverage-performance-footnote",
            ),
        ],
        className="coverage-card coverage-performance",
    )


def performance_figure(result: Any) -> go.Figure:
    """Show all small-study pairs or full-count density, with unsmoothed empirical curves."""
    if result is None:
        figure = _empty_figure("Refresh to analyze the selected objective.")
        figure.update_layout(height=330)
        return figure
    bins = result["bins"]
    plot = result["plot"]
    figure = go.Figure()
    group_hover = (
        "<br>Distance group: %{customdata[0]:.4g} – %{customdata[1]:.4g}"
        "<br>Group pairs: %{customdata[2]:,}"
        "<br>Group median: %{customdata[3]:.4g}<br>Group P90: %{customdata[4]:.4g}"
    )

    def bin_data(stats: Dict) -> list:
        return [stats[key] for key in ("lower", "upper", "count", "median", "p90")]

    if plot["kind"] == "scatter":
        lookup = dict(zip(plot["labels"], bins))
        figure.add_trace(
            go.Scattergl(
                x=plot["distances"],
                y=plot["differences"],
                mode="markers",
                marker={"size": 4, "color": "#9675e8", "opacity": 0.22},
                name="Observed pairs",
                customdata=[bin_data(lookup[label]) for label in plot["bins"]],
                hovertemplate="Distance: %{x:.4g}<br>Absolute difference: %{y:.4g}"
                "<br>Pairs at this point: 1" + group_hover + "<extra></extra>",
            )
        )
    else:
        x_edges = np.asarray(plot["x_edges"])
        y_edges = np.asarray(plot["y_edges"])
        x = (x_edges[:-1] + x_edges[1:]) / 2
        y = (y_edges[:-1] + y_edges[1:]) / 2
        counts = np.asarray(plot["counts"], dtype=float)
        custom = []
        for row in range(len(y)):
            cells = []
            for column, center in enumerate(x):
                stats = min(bins, key=lambda stats: abs(stats["distance_median"] - center))
                cells.append(
                    bin_data(stats)
                    + [x_edges[column], x_edges[column + 1], y_edges[row], y_edges[row + 1]]
                )
            custom.append(cells)
        figure.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=np.where(counts > 0, counts, np.nan).tolist(),
                customdata=custom,
                colorscale=[[0, "#f8f5ff"], [0.3, "#c1a7f2"], [1, "#7752bf"]],
                colorbar={"title": "Pairs", "thickness": 9, "len": 0.8},
                hoverongaps=False,
                hovertemplate="Distance: %{customdata[5]:.4g} – %{customdata[6]:.4g}"
                "<br>Absolute difference: %{customdata[7]:.4g} – %{customdata[8]:.4g}"
                "<br>Cell pairs: %{z:,}" + group_hover + "<extra></extra>",
            )
        )
    for statistic, label, color, dash in (
        ("median", "Median", "#553291", "solid"),
        ("p90", "P90", "#9580b8", "dot"),
    ):
        figure.add_trace(
            go.Scatter(
                x=[stats["distance_median"] for stats in bins],
                y=[stats[statistic] for stats in bins],
                mode="lines+markers",
                line={"color": color, "width": 2, "dash": dash, "shape": "linear"},
                marker={"size": 5},
                name=label,
                customdata=[bin_data(stats) for stats in bins],
                hovertemplate=label + ": %{y:.4g}" + group_hover + "<extra></extra>",
            )
        )
    figure.update_layout(
        template="plotly_white",
        height=330,
        margin={"l": 55, "r": 25, "t": 5, "b": 65},
        font={"family": "inherit", "size": 11, "color": "#505a78"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        xaxis_title="Importance-weighted configuration distance",
        yaxis_title="Absolute performance difference",
        yaxis={"type": "linear", "rangemode": "tozero"},
    )
    return figure


def performance_interpretation(result: Dict) -> str:
    """Describe only the observed bin medians, without an inferential trend test."""
    medians = np.asarray([stats["median"] for stats in result["bins"]])
    if len(medians) >= 3 and np.all(np.diff(medians) > 0):
        return "Median performance differences increase across the observed distance groups."
    if len(medians) >= 3 and np.all(np.diff(medians) < 0):
        return "Median performance differences decrease across the observed distance groups."
    return "No consistent trend in median performance differences across observed distance groups."


def neighbor_table(result: Dict) -> Any:
    """Render statistics computed exclusively from one nearest neighbor per trial."""
    rows = []
    for stats in result["nearest_bins"]:
        fraction = stats["relevant_fraction"]
        rows.append(
            html.Tr(
                [
                    html.Td(f"{stats['lower']:.4g} – {stats['upper']:.4g}"),
                    html.Td(f"{stats['count']:,}"),
                    html.Td(f"{stats['mean']:.4g}"),
                    html.Td(f"{stats['p90']:.4g}"),
                    html.Td(
                        [
                            html.Span(f"{fraction:.1%}"),
                            html.Span(
                                html.Span(style={"width": f"{fraction * 100:.3f}%"}),
                                className="coverage-performance-bar",
                                **{"aria-hidden": "true"},
                            ),
                        ]
                    ),
                ]
            )
        )
    return html.Div(
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th(label)
                            for label in (
                                "Distance range",
                                "Trials",
                                "Mean Δ perf.",
                                "P90 Δ perf.",
                                "Δ > threshold",
                            )
                        ]
                    )
                ),
                html.Tbody(rows),
            ]
        ),
        className="coverage-performance-table",
    )


def register_performance_callbacks(app: Any, jobs: Any, run_handler: Any) -> None:
    """Keep computation explicit and threshold changes independent from distance jobs."""
    directory = jobs.directory / "_performance_data"

    @app.callback(
        Output(_id("performance-data"), "data"),
        Output(_id("performance-status"), "children"),
        Output(_id("performance-run"), "disabled"),
        Output(_id("performance-view"), "data"),
        Input(_id("performance-run"), "n_clicks"),
        Input(_id("context"), "data"),
        Input(_id("tick"), "n_intervals"),
        Input("global-main-objective-state", "data"),
        State("global-run-selector", "value"),
        State(_id("performance-view"), "data"),
    )
    def performance_job(
        clicks: Any, context: Any, tick: Any, objective_state: Any, active_run: str, previous: Any
    ) -> Any:
        if not context or context["run_id"] != active_run:
            raise PreventUpdate
        run = run_handler.get_run(active_run)
        objective_id = resolve_main_objective_id(run, (objective_state or {}).get("value"))
        if objective_id is None:
            view = {"run_id": run.id, "objective_id": None}
            if view == previous:
                raise PreventUpdate
            return view, "Select a Main Objective in the sidebar.", True, view
        source_key = performance_source_key(run, objective_id)
        options = {"objective_id": objective_id, "performance_version": PERFORMANCE_VERSION}
        with _JOB_LOCK:
            entry = jobs.poll(run.id, "performance")
            running = entry.get("status") == "running"
            if clicks and _id("performance-run") in ctx.triggered_prop_ids.values() and not running:
                try:
                    if not jobs.queue.ready():
                        raise RuntimeError(
                            "No analysis worker is available. Start a worker and retry."
                        )
                    job_id = f"coverage-performance-{uuid4().hex}"
                    jobs.queue.enqueue(
                        run_performance_analysis,
                        args=[run, objective_id, str(directory)],
                        job_id=job_id,
                        meta={
                            "display_name": "Performance-aware Coverage",
                            "run_name": run.name,
                            "run_id": run.id,
                            "plugin_id": "search_space_coverage",
                            "analysis": "performance",
                            "link": "/plugins/search_space_coverage",
                        },
                    )
                    entry.update(
                        status="running",
                        error=None,
                        job={
                            "id": job_id,
                            "fingerprint": source_key,
                            "trial_count": len(run.history),
                            "options": options,
                        },
                    )
                except Exception as error:
                    entry.update(status="failed", error=str(error))
                jobs._write(run.id, "performance", entry)
            running = entry.get("status") == "running"
            result = entry.get("result")
            if result and result.get("objective_id") != objective_id:
                result = None
            stale = bool(result) and (
                result.get("source_key") != source_key
                or result.get("performance_version") != PERFORMANCE_VERSION
            )
            if result and not (directory / result["cache_key"] / "differences.npy").exists():
                result, stale = None, True
            status = (
                "Computing…"
                if running
                else (
                    str(entry.get("error"))
                    if entry.get("status") == "failed"
                    else "Study changed · Refresh to update." if stale else ""
                )
            )
            view = {
                "run_id": run.id,
                "objective_id": objective_id,
                "source_key": source_key,
                "result_key": result.get("cache_key") if result else None,
                "status": status,
                "computed_at": entry.get("computed_at"),
                "revision": context.get("revision"),
            }
            if previous == view:
                raise PreventUpdate
            payload = {
                "run_id": run.id,
                "objective_id": objective_id,
                "objective": run.get_objective(objective_id).name,
                "cache_key": result.get("cache_key") if result else None,
            }
            previous_payload = (
                {key: previous.get(key) for key in ("run_id", "objective_id")} if previous else {}
            )
            unchanged = (
                previous_payload == {"run_id": run.id, "objective_id": objective_id}
                and previous.get("result_key") == payload["cache_key"]
            )
            return no_update if unchanged else payload, status, running, view

    @app.callback(
        Output(_id("performance-trials"), "children"),
        Output(_id("performance-pairs"), "children"),
        Output(_id("performance-objective"), "children"),
        Output(_id("performance-chart"), "figure"),
        Output(_id("performance-interpretation"), "children"),
        Output(_id("performance-neighbors"), "children"),
        Output(_id("performance-unit"), "children"),
        Output(_id("performance-method"), "children"),
        Output(_id("performance-threshold-error"), "children"),
        Input(_id("performance-data"), "data"),
        Input(_id("performance-threshold"), "value"),
        State(_id("context"), "data"),
    )
    def performance_results(payload: Any, threshold: Any, context: Any) -> Any:
        if not payload or not context or payload.get("run_id") != context.get("run_id"):
            return "—", "—", "—", performance_figure(None), "", "", "", "", ""
        objective = payload.get("objective", "—")
        entry = jobs.read(context["run_id"], "performance")
        result = entry.get("result")
        if not result or result.get("cache_key") != payload.get("cache_key"):
            return "—", "—", objective, performance_figure(None), "", "", objective, "", ""
        differences_path = directory / result["cache_key"] / "differences.npy"
        threshold_error = ""
        try:
            stats = threshold_statistics(result, threshold, differences_path)
        except (ValueError, TypeError) as error:
            stats = result
            threshold_error = str(error)
        method = [
            html.P(result["weight_source"]),
            html.P(
                "Existing Rashomon weighted RMS distance; ConfigSpace normalization, "
                "categorical mismatches and inactive-parameter handling are unchanged. "
                "All valid successful trials at every budget enter the pair analysis. "
                "Equal nearest-neighbor distances select the lowest trial ID, then history row. "
                "Quantile groups merge identical distances. Lines join empirical group statistics."
            ),
            html.P(
                f"Excluded unsuccessful, missing-objective or incompatible trials: {result['excluded']:,}."
            ),
            html.P(
                "Weights are derived from this study's performance. Pairs are dependent; "
                "differences may include evaluation noise. No causal or significance claims."
            ),
            html.Ul(
                [html.Li(f"{name}: {weight:.4g}") for name, weight in result["weights"].items()]
            ),
        ]
        return (
            f"{result['trials']:,}",
            f"{result['pairs']:,}",
            objective,
            performance_figure(stats),
            performance_interpretation(stats),
            (
                "Enter a valid relevant-difference threshold."
                if threshold_error
                else neighbor_table(stats)
            ),
            objective,
            method,
            threshold_error,
        )
