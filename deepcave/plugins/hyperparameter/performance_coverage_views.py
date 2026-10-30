"""Queued CHR and full-space candidate rankings with shared KDE interpretation."""

from typing import Any, Dict

import json
from threading import RLock
from urllib.parse import quote
from uuid import uuid4

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme, Sign
from dash.exceptions import PreventUpdate
from scipy.special import expit

from deepcave.evaluators.coverage_candidates import (
    candidate_options,
    score_coverage_candidates,
    scoring_source_key,
)
from deepcave.plugins.hyperparameter.coverage_planning_views import (
    optuna_export_study,
    plan_toml,
)
from deepcave.plugins.hyperparameter.coverage_views import _empty_figure, _id
from deepcave.utils.global_objective import resolve_main_objective_id

_JOB_LOCK = RLock()
_SETTINGS = (
    ("pool_size", "CHR candidate pool", 2000, 1, 10000),
    ("holes", "Coverage holes", 15, 1, 100),
    ("reference_samples", "CHR reference samples", 2048, 32, 16384),
    ("seed", "CHR seed", 42, 0, 2**32 - 2),
    ("global_pool_size", "Full-space candidate pool", 10000, 1, 100000),
    ("global_seed", "Full-space seed", 42, 0, 2**32 - 2),
    ("number_of_candidates", "Number of candidates", 20, 1, 100000),
)


def run_performance_analysis(
    run: Any, objective_id: int, directory: str, options: Any = None
) -> Dict[str, Any]:
    """Worker entry point; the dashboard never imports a benchmark script."""
    try:
        return score_coverage_candidates(run, objective_id, directory, options)
    except (ValueError, RuntimeError, FloatingPointError, ImportError, TimeoutError) as error:
        # RQ's generic failure state otherwise loses the actionable fit error in the UI.
        return {"error": str(error), "objective_id": objective_id}


def _export_label() -> list:
    return [html.I(className="fas fa-download", **{"aria-hidden": "true"}), "Export TOML"]


def ranking_columns(tab: str = "overall") -> list:
    """Share score columns while distinguishing CHR geometry and hybrid selection ranks."""
    columns = [
        {"id": "rank", "name": "Rank", "type": "numeric"},
        {
            "id": "good_probability",
            "name": "Top-20% probability",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.percentage),
        },
        {
            "id": "kde_score",
            "name": "KDE score",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "id": "relative_score",
            "name": "KDE vs. reference",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed, sign=Sign.positive),
        },
    ]
    if tab != "full_space":
        columns.extend(
            [
                {"id": "hole_id", "name": "Coverage hole"},
                {
                    "id": "hole_radius",
                    "name": "Hole radius",
                    "type": "numeric",
                    "format": Format(precision=4, scheme=Scheme.fixed),
                },
            ]
        )
    else:
        columns[1:1] = [
            {"id": "selection", "name": "Selection"},
            {"id": "kde_rank", "name": "KDE rank", "type": "numeric"},
        ]
        columns.append({"id": "bin", "name": "Distance bin", "type": "numeric"})
    # RTL table layout places Dash's leading selection column on the right.
    return columns[::-1]


def performance_panel() -> Any:
    """Build the right column without changing the covering-array controls."""
    advanced = html.Details(
        [
            html.Summary([html.Span("☷", **{"aria-hidden": "true"}), " Advanced"]),
            html.Div(
                [
                    html.H5("Candidate search settings"),
                    html.Div(
                        [
                            html.Label(
                                [
                                    label,
                                    dbc.Input(
                                        id=_id(f"performance-{name}-setting"),
                                        type="number",
                                        value=default,
                                        min=low,
                                        max=high,
                                        step=1,
                                        debounce=True,
                                        size="sm",
                                    ),
                                ]
                            )
                            for name, label, default, low, high in _SETTINGS
                        ],
                        className="coverage-candidate-settings",
                    ),
                ],
                className="coverage-planning-advanced-content",
            ),
        ],
        className="coverage-planning-advanced coverage-performance-advanced",
    )
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
                            html.H3("Performance scoring of coverage candidates"),
                        ],
                        className="coverage-performance-title",
                    ),
                    dbc.Button(
                        "Refresh",
                        id=_id("performance-run"),
                        n_clicks=0,
                        color="primary",
                        size="sm",
                    ),
                    advanced,
                ],
                className="coverage-performance-header",
            ),
            html.Div(id=_id("performance-status"), role="status", **{"aria-live": "polite"}),
            html.Div(
                [
                    html.Div([html.Span(label), html.Strong("—", id=_id(f"performance-{name}"))])
                    for name, label in (
                        ("holes", "Coverage holes analyzed"),
                        ("candidates", "Generated candidates"),
                        ("reference", "Reference"),
                        ("top", "Best candidate vs. reference"),
                    )
                ],
                className="coverage-performance-metrics",
            ),
            html.Div(
                [
                    html.H5("KDE score (Log Density Ratio) distribution"),
                    dcc.Loading(
                        dcc.Graph(
                            id=_id("performance-chart"),
                            figure=performance_figure(None),
                            config={"displayModeBar": False},
                            responsive=True,
                            style={"height": "165px"},
                        ),
                        color="#9470d5",
                    ),
                    html.Div(
                        [
                            html.H4("Candidate rankings"),
                            html.Div(
                                dbc.Button(
                                    _export_label(),
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                    disabled=True,
                                    title=(
                                        "Select trials using the checkboxes "
                                        "to export their parameters."
                                    ),
                                ),
                                id=_id("performance-export"),
                                className="coverage-candidate-export",
                            ),
                        ],
                        className="coverage-candidate-rankings-header",
                    ),
                    dcc.Tabs(
                        id=_id("performance-tab"),
                        value="overall",
                        children=[
                            dcc.Tab(
                                label=label,
                                value=value,
                                className="coverage-candidate-tab",
                                selected_className="coverage-candidate-tab-selected",
                            )
                            for value, label in (
                                ("overall", "Coverage Holes: Best candidates overall"),
                                ("holes", "Coverage Holes: Best candidate per hole"),
                                ("full_space", "Best candidates: Full search space"),
                            )
                        ],
                        className="coverage-candidate-tabs",
                    ),
                    html.Div(
                        [
                            dcc.RadioItems(
                                id=_id("performance-mode"),
                                options=[
                                    {"label": "Recommended", "value": "recommended"},
                                    {"label": "All candidates", "value": "all"},
                                ],
                                value="recommended",
                                inline=True,
                                labelStyle={
                                    "display": "inline-flex",
                                    "alignItems": "center",
                                    "gap": "5px",
                                    "marginRight": "16px",
                                },
                                inputStyle={"width": "auto", "margin": "0"},
                                style={"fontSize": "12px"},
                            ),
                        ],
                        id=_id("performance-mode-controls"),
                        style={"display": "none"},
                    ),
                    dash_table.DataTable(
                        id=_id("performance-ranking"),
                        data=[],
                        columns=ranking_columns(),
                        cell_selectable=True,
                        row_selectable="multi",
                        selected_rows=[],
                        selected_row_ids=[],
                        page_size=8,
                        page_action="native",
                        sort_action="native",
                        sort_by=[],
                        tooltip_header={
                            "good_probability": (
                                "Modelled membership in the best trial group, using the KDE "
                                "densities and actual good/bad group sizes. "
                                "Uncalibrated model estimate."
                            ),
                        },
                        style_table={"overflowX": "auto"},
                        style_as_list_view=True,
                        style_cell={
                            "fontSize": "11px",
                            "padding": "6px 7px",
                            "textAlign": "left",
                            "borderBottom": "1px solid #eeedf6",
                            "fontFamily": "inherit",
                            "color": "#393853",
                            "height": "29px",
                        },
                        style_header={
                            "backgroundColor": "#f8f7fd",
                            "fontWeight": "600",
                            "color": "#696287",
                            "fontSize": "10px",
                        },
                        style_cell_conditional=[
                            {
                                "if": {"column_id": c},
                                "textAlign": "right",
                                "fontVariantNumeric": "tabular-nums",
                            }
                            for c in (
                                "kde_score",
                                "relative_score",
                                "good_probability",
                                "hole_radius",
                            )
                        ],
                        style_data_conditional=[
                            {
                                "if": {"state": "active"},
                                "backgroundColor": "#f0e9ff",
                                "border": "1px solid #d8c8f5",
                            },
                            {
                                "if": {
                                    "filter_query": "{good_probability} > 0.5",
                                    "column_id": ["good_probability", "kde_score"],
                                },
                                "color": "#13944c",
                            },
                            {
                                "if": {
                                    "filter_query": "{relative_score} > 0",
                                    "column_id": "relative_score",
                                },
                                "color": "#13944c",
                            },
                            {
                                "if": {
                                    "filter_query": "{relative_score} < 0",
                                    "column_id": "relative_score",
                                },
                                "color": "#a14b78",
                            },
                        ],
                    ),
                    html.P(
                        "Half of the candidates are selected by global KDE ranking. "
                        "The other half explore different distances from the incumbent. "
                        "Each distance bin is supplied with at least 1,000 candidates "
                        "before selection.",
                        id=_id("performance-selection-description"),
                        className="coverage-performance-interpretation",
                        style={"display": "none"},
                    ),
                    html.P(
                        id=_id("performance-pool-summary"),
                        className="coverage-performance-interpretation",
                    ),
                ],
                className="coverage-performance-body",
            ),
            html.Div(id=_id("performance-details"), className="coverage-candidate-details"),
            html.Small(
                "Score = log p(x | good) − log p(x | bad); good = best 20% of completed trials; "
                "bad = remaining 80% of completed trials. "
                "The best observed trial is an in-sample reference, not independent validation. "
                "Scores are neither predicted objective values nor success probabilities. "
                "Top-20% probability uses the actual good/bad group sizes and is an "
                "uncalibrated KDE model estimate.",
                className="coverage-performance-footnote",
            ),
            html.Section(
                [
                    html.H4("How candidates are selected"),
                    html.P(
                        "A candidate is a parameter configuration that has not been tried yet. "
                        "Both procedures use the same KDE model (kernel density estimation). "
                        "It learns which parameter combinations are typical of the best 20% "
                        "of completed trials and which are typical of the remaining trials. "
                        "A higher score means a candidate looks more like the better group."
                    ),
                    html.H5("Coverage Holes: Explore gaps in previous trials"),
                    html.Ol(
                        [
                            html.Li(
                                "Find gaps: Sample points across the search space and measure "
                                "their distance to the nearest tried configuration. Points "
                                "with large distances mark coverage holes: regions with "
                                "little previous exploration. Objective values do not "
                                "influence this step."
                            ),
                            html.Li(
                                "Generate candidates: Keep untried configurations inside "
                                "these holes, giving larger holes more candidates. Then "
                                "score the candidates with the KDE model."
                            ),
                            html.Li(
                                "Choose a view: 'Best candidates overall' ranks all hole "
                                "candidates by score, so several may come from the same hole. "
                                "'Best candidate per hole' keeps only the highest-scoring "
                                "candidate from each hole that has candidates."
                            ),
                        ]
                    ),
                    html.H5("Full search space: KDE + exploration"),
                    html.Ol(
                        [
                            html.Li(
                                "Start broadly: Sample untried configurations across the "
                                "full search space, including regions outside coverage holes."
                            ),
                            html.Li(
                                "Group by distance: The incumbent is the best configuration "
                                "observed so far. Put candidates into five distance bins "
                                "around it: bin 1 is closest and bin 5 is farthest. Parameter "
                                "ranges are normalized so different units are comparable. "
                                "The spread of the best 10% of trials sets the distance scale; "
                                "bin 5 also includes candidates beyond that spread. Add "
                                "candidates where needed to supply at least 1,000 per bin "
                                "before selection."
                            ),
                            html.Li(
                                "Select global winners: Fill half of the requested places "
                                "with the highest KDE scores across the entire candidate "
                                "pool. For an odd number of places, this half gets one extra."
                            ),
                            html.Li(
                                "Add exploration: Fill the remaining places by cycling "
                                "through bins 1 to 5, taking the highest-scoring candidate "
                                "still available in each bin without selecting duplicates. "
                                "For 20 recommendations, this gives 10 global winners and "
                                "10 exploration candidates, two from each bin. Exploration "
                                "therefore still uses KDE scores while covering different "
                                "distances from the incumbent."
                            ),
                        ]
                    ),
                    html.P(
                        "'Recommended' shows this combined selection. 'All candidates' "
                        "shows the full pool sorted by KDE score. The procedures propose "
                        "configurations; their actual performance is only known after "
                        "running new trials."
                    ),
                ],
                className="coverage-candidate-procedures",
            ),
        ],
        className="coverage-card coverage-performance",
    )


def performance_figure(result: Any, tab: str = "overall") -> go.Figure:
    """Histogram of the entire pool's relative scores, with a zero reference."""
    candidates = candidate_pool(result, tab) if result else []
    if not candidates:
        figure = _empty_figure(
            "No unseen candidates in this pool."
            if result
            else "Refresh to generate and score candidates."
        )
        figure.update_layout(height=165)
        return figure
    values = np.asarray([c["relative_score"] for c in candidates])
    counts, edges = np.histogram(values, bins=min(36, max(1, int(np.sqrt(len(values))))))
    centers = (edges[:-1] + edges[1:]) / 2
    figure = go.Figure(
        go.Bar(
            x=centers.tolist(),
            y=counts.tolist(),
            width=(np.diff(edges) * 0.92).tolist(),
            marker_color=[
                f"rgba(139, 94, 207, {0.3 + 0.6 * i / max(1, len(centers) - 1)})"
                for i in range(len(centers))
            ],
            hovertemplate="Relative KDE score: %{x:.4g}<br>Candidates: %{y}<extra></extra>",
        )
    )
    figure.add_vline(
        x=0,
        line_dash="dash",
        line_color="#995deb",
        line_width=1.4,
        annotation_text="Best observed trial<br>(Reference)",
        annotation_position="top left" if np.median(values) < 0 else "top right",
    )
    lower, upper = min(0, edges[0]), max(0, edges[-1])
    padding = 0.04 * (upper - lower)
    figure.update_layout(
        template="plotly_white",
        height=165,
        showlegend=False,
        margin={"l": 22, "r": 15, "t": 24, "b": 40},
        bargap=0.06,
        font={"family": "inherit", "size": 10, "color": "#70678e"},
        xaxis={
            "title": "KDE score relative to reference",
            "range": [lower - padding, upper + padding],
            "showgrid": True,
            "gridcolor": "#f4f1fa",
            "zeroline": False,
        },
        yaxis={"showticklabels": False, "showgrid": False, "zeroline": False},
    )
    return figure


def candidate_pool(result: Dict, tab: str) -> list:
    """Keep the augmented full-space pool separate from CHR candidates."""
    return result.get("full_space_candidates", []) if tab == "full_space" else result["candidates"]


def ranking_candidates(result: Dict, tab: str, mode: str = "recommended") -> list:
    """Rank each pool by KDE score, optionally keeping the best per hole."""
    if tab == "full_space" and mode == "recommended":
        return result.get("recommendations", [])
    ordered = sorted(candidate_pool(result, tab), key=lambda c: (-c["kde_score"], c["id"]))
    if tab == "holes":
        best: Dict[str, Dict] = {}
        for candidate in ordered:
            best.setdefault(candidate["hole_id"], candidate)
        return list(best.values())
    return ordered


def ranking_rows(result: Dict, tab: str, mode: str = "recommended") -> list:
    """Build numeric rankings with stable candidate IDs for detail selection."""
    labels = {h["id"]: f"Hole {i + 1}" for i, h in enumerate(result["holes"])}
    model = result["model"]
    log_prior_odds = np.log(len(model["good_trials"]) / len(model["bad_trials"]))
    selections = {c["id"]: c["selection"] for c in result.get("recommendations", [])}
    rows = [
        {
            "id": c["id"],
            "rank": i + 1,
            "selection": selections.get(c["id"], c.get("selection", "—")),
            "kde_rank": c.get("kde_rank", i + 1),
            "distance": c.get("distance"),
            "bin": c.get("bin"),
            "kde_score": c["kde_score"],
            "good_probability": float(expit(c["kde_score"] + log_prior_odds)),
            "relative_score": c["relative_score"],
            "hole_id": None if tab == "full_space" else labels[c["hole_id"]],
            "hole_radius": None if tab == "full_space" else c["hole_radius"],
            "configuration": ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in c["configuration"].items()
            ),
        }
        for i, c in enumerate(ranking_candidates(result, tab, mode))
    ]
    if tab == "holes":
        represented = {c["hole_id"] for c in result["candidates"]}
        rows.extend(
            {
                "id": f"empty-{hole['id']}",
                "rank": None,
                "kde_score": None,
                "good_probability": None,
                "relative_score": None,
                "hole_id": labels[hole["id"]],
                "hole_radius": hole["radius"],
                "configuration": "No candidate generated",
            }
            for hole in result["holes"]
            if hole["id"] not in represented
        )
    return rows


def candidate_details(result: Dict, identifier: Any) -> Any:
    """Show actual raw/relative scores, CHR geometry, and a full configuration."""
    candidate = next(
        (
            c
            for c in result["candidates"] + result.get("full_space_candidates", [])
            if c["id"] == identifier
        ),
        None,
    )
    if candidate is None:
        if isinstance(identifier, str) and identifier.startswith("empty-"):
            return html.P(
                "No valid candidate was generated for this hole. "
                "Increase the candidate pool size or refresh."
            )
        return html.P("Select a candidate to inspect its score and configuration.")
    c = candidate

    def group(title: str, rows: list) -> Any:
        return html.Div(
            [
                html.H5(title),
                html.Dl([html.Div([html.Dt(label), html.Dd(value)]) for label, value in rows]),
            ]
        )

    if "hole_id" in c:
        hole_number = next(i + 1 for i, h in enumerate(result["holes"]) if h["id"] == c["hole_id"])
        search_title = "Coverage contribution"
        search_rows = [
            ("Coverage hole", f"Hole {hole_number} · {c['hole_id']}"),
            ("Original hole radius", f"{c['hole_radius']:.5g}"),
            ("Distance to hole center", f"{c['center_distance']:.5g}"),
            ("Radius reduction at center", f"{c['radius_reduction']:.5g}"),
        ]
    else:
        sampling = result["global_sampling"]
        search_title = "Full-space search"
        search_rows = [
            (
                "Evaluated unseen configurations",
                f"{result.get('supply', {}).get('final_count', sampling['count']):,}",
            ),
            ("Requested initial pool size", f"{sampling['requested']:,}"),
            ("Seed", str(sampling["seed"])),
        ]
        if "distance" in c:
            recommendation = next(
                (r for r in result.get("recommendations", []) if r["id"] == c["id"]), {}
            )
            search_rows.extend(
                [
                    ("Selection", recommendation.get("selection", "Not recommended")),
                    ("KDE rank", str(c["kde_rank"])),
                    ("Distance to incumbent (RMS)", f"{c['distance']:.5g}"),
                    ("Distance bin", str(c["bin"])),
                ]
            )
    config_group = group("Configuration", [(k, str(v)) for k, v in c["configuration"].items()][:5])
    config_group.children.append(
        html.Details(
            [
                html.Summary("Show full config"),
                html.Pre(json.dumps(c["configuration"], indent=2)),
            ],
            className="coverage-candidate-full-config",
        )
    )
    return [
        html.Div(
            html.H4("Selected candidate details"),
            className="coverage-candidate-details-header",
        ),
        html.Div(
            [
                group(
                    "Score interpretation",
                    [
                        ("Original KDE score", f"{c['kde_score']:.5g}"),
                        ("Reference KDE score", f"{c['reference_score']:.5g}"),
                        ("Relative KDE score", f"{c['relative_score']:+.5g}"),
                        ("Percentile in current pool", f"{c['percentile']:.1f}%"),
                        (
                            f"Trial #{c['reference_trial_number']} · "
                            f"{result['model']['objective']}",
                            f"{c['reference_objective']:.6g}",
                        ),
                    ],
                ),
                group(search_title, search_rows),
                config_group,
            ],
            className="coverage-candidate-detail-columns",
        ),
    ]


def register_performance_callbacks(app: Any, jobs: Any, run_handler: Any) -> None:
    """Explicit refresh queues work; tab and row interactions use cached results."""
    directory = jobs.directory / "_coverage_candidates"

    @app.callback(  # type: ignore[untyped-decorator]
        Output(_id("performance-data"), "data"),
        Output(_id("performance-status"), "children"),
        Output(_id("performance-run"), "disabled"),
        Output(_id("performance-view"), "data"),
        Input(_id("performance-run"), "n_clicks"),
        Input(_id("context"), "data"),
        Input(_id("tick"), "n_intervals"),
        Input("global-main-objective-state", "data"),
        *[Input(_id(f"performance-{name}-setting"), "value") for name, *_ in _SETTINGS],
        State("global-run-selector", "value"),
        State(_id("performance-view"), "data"),
    )
    def performance_job(
        clicks: Any,
        context: Any,
        tick: Any,
        objective_state: Any,
        pool_size: Any,
        holes: Any,
        reference_samples: Any,
        seed: Any,
        global_pool_size: Any,
        global_seed: Any,
        number_of_candidates: Any,
        active_run: str,
        previous: Any,
    ) -> Any:
        if not context or context["run_id"] != active_run:
            raise PreventUpdate
        run = run_handler.get_run(active_run)
        objective_id = resolve_main_objective_id(run, (objective_state or {}).get("value"))
        error = None
        try:
            options = candidate_options(
                dict(
                    pool_size=pool_size,
                    holes=holes,
                    reference_samples=reference_samples,
                    seed=seed,
                    global_pool_size=global_pool_size,
                    global_seed=global_seed,
                    number_of_candidates=number_of_candidates,
                )
            )
        except ValueError as exception:
            options, error = {}, str(exception)
        if objective_id is None:
            view = {"run_id": run.id, "objective_id": None, "error": error}
            if view == previous:
                raise PreventUpdate
            return view, "Select a Main Objective in the sidebar.", True, view
        source_key = scoring_source_key(run, objective_id)
        with _JOB_LOCK:
            entry = jobs.poll(run.id, "candidate_scoring")
            running = entry.get("status") == "running"
            if (
                not error
                and clicks
                and _id("performance-run") in ctx.triggered_prop_ids.values()
                and not running
            ):
                try:
                    if not jobs.queue.ready():
                        raise RuntimeError(
                            "No analysis worker is available. Start a worker and retry."
                        )
                    job_id = f"coverage-candidates-{uuid4().hex}"
                    jobs.queue.enqueue(
                        run_performance_analysis,
                        args=[run, objective_id, str(directory), options],
                        job_id=job_id,
                        meta={
                            "display_name": "Coverage candidate KDE scoring",
                            "run_name": run.name,
                            "run_id": run.id,
                            "plugin_id": "search_space_coverage",
                            "analysis": "candidate_scoring",
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
                except Exception as exception:
                    entry.update(status="failed", error=str(exception))
                jobs._write(run.id, "candidate_scoring", entry)
            running = entry.get("status") == "running"
            result = entry.get("result")
            if result and result.get("objective_id") != objective_id:
                result = None
            if result and result.get("error"):
                error, result = result["error"], None
            if result and (
                result.get("source_key") != source_key or result.get("options") != options
            ):
                result = None
            status = error or (
                "Detecting coverage holes, sampling and scoring candidates…"
                if running
                else (str(entry.get("error")) if entry.get("status") == "failed" else "")
            )
            view = {
                "run_id": run.id,
                "objective_id": objective_id,
                "source_key": source_key,
                "options": options,
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
                "cache_key": result.get("cache_key") if result else None,
            }
            unchanged = previous and all(
                previous.get(key) == view.get(key)
                for key in ("run_id", "objective_id", "result_key")
            )
            return no_update if unchanged else payload, status, running or not bool(options), view

    def current_result(payload: Any, context: Any) -> Any:
        if not payload or not context or payload.get("run_id") != context.get("run_id"):
            return None
        result = jobs.read(context["run_id"], "candidate_scoring").get("result")
        if (
            result
            and "candidates" in result
            and result.get("cache_key") == payload.get("cache_key")
            and result.get("source_key")
            == scoring_source_key(run_handler.get_run(context["run_id"]), result["objective_id"])
        ):
            return result
        return None

    @app.callback(  # type: ignore[untyped-decorator]
        Output(_id("performance-mode-controls"), "style"),
        Output(_id("performance-selection-description"), "style"),
        Input(_id("performance-tab"), "value"),
    )
    def performance_mode(tab: str) -> tuple:
        style = {} if tab == "full_space" else {"display": "none"}
        return style, style

    @app.callback(  # type: ignore[untyped-decorator]
        Output(_id("performance-chart"), "figure"),
        Input(_id("performance-data"), "data"),
        Input(_id("performance-tab"), "value"),
        State(_id("context"), "data"),
    )
    def performance_chart(payload: Any, tab: str, context: Any) -> go.Figure:
        return performance_figure(current_result(payload, context), tab)

    @app.callback(  # type: ignore[untyped-decorator]
        Output(_id("performance-holes"), "children"),
        Output(_id("performance-candidates"), "children"),
        Output(_id("performance-reference"), "children"),
        Output(_id("performance-top"), "children"),
        Output(_id("performance-ranking"), "data"),
        Output(_id("performance-ranking"), "active_cell"),
        Output(_id("performance-ranking"), "page_current"),
        Output(_id("performance-ranking"), "selected_rows"),
        Output(_id("performance-ranking"), "selected_row_ids"),
        Output(_id("performance-ranking"), "columns"),
        Output(_id("performance-pool-summary"), "children"),
        Output(_id("performance-ranking"), "sort_by"),
        Input(_id("performance-data"), "data"),
        Input(_id("performance-tab"), "value"),
        Input(_id("performance-mode"), "value"),
        State(_id("context"), "data"),
    )
    def performance_results(payload: Any, tab: Any, mode: Any, context: Any) -> Any:
        result = current_result(payload, context)
        columns = ranking_columns(tab)
        if not result:
            return "—", "—", "—", "—", [], None, 0, [], [], columns, "", []
        rows = ranking_rows(result, tab, mode)
        pool = candidate_pool(result, tab)
        top = max((c["relative_score"] for c in pool), default=None)
        if tab == "full_space" and "global_sampling" not in result:
            summary = "Refresh to generate full-space candidates."
        elif tab == "full_space":
            sampling = result["global_sampling"]
            summary = ""
            if sampling["count"] < sampling["requested"]:
                summary += (
                    "No unseen configurations remain in the finite search space."
                    if sampling["stop_reason"] == "finite_space_exhausted"
                    else f"Requested {sampling['requested']:,}; sampling limit reached after "
                    f"{sampling['draws']:,} valid draws."
                )
            if "supply" in result:
                supply, hybrid = result["supply"], result["hybrid"]
                occupancy = " · ".join(f"B{b['bin']}: {b['count']:,}" for b in supply["bins"])
                summary += (
                    f" Initial: {supply['initial_count']:,} · Added: {supply['added_count']:,}"
                    f" · Pool: {supply['final_count']:,} · Global KDE: {hybrid['global_count']}"
                    f" · Exploration: {hybrid['exploration_count']} · {occupancy}"
                )
                if hybrid.get("error"):
                    summary += f". {hybrid['error']}"
            else:
                summary += " Refresh to prepare KDE + exploration recommendations."
        else:
            summary = ""
        cell = (
            {
                "row": 0,
                "column": next(i for i, c in enumerate(columns) if c["id"] == "relative_score"),
                "column_id": "relative_score",
                "row_id": rows[0]["id"],
            }
            if rows
            else None
        )
        return (
            f"{len(result['holes']):,}",
            f"{len(pool):,}",
            f"Best observed trial #{result['reference_trial_number']}",
            f"{top:+.3f}" if top is not None else "—",
            rows,
            cell,
            0,
            [],
            [],
            columns,
            summary,
            [],
        )

    @app.callback(  # type: ignore[untyped-decorator]
        Output(_id("performance-export"), "children"),
        Input(_id("performance-ranking"), "selected_row_ids"),
        Input(_id("performance-data"), "data"),
        Input(_id("performance-tab"), "value"),
        Input(_id("performance-mode"), "value"),
        State(_id("context"), "data"),
    )
    def performance_export(
        selected_ids: Any, payload: Any, tab: Any, mode: Any, context: Any
    ) -> Any:
        result = current_result(payload, context)
        selected = set(selected_ids or [])
        trials = (
            [
                {"parameters": c["configuration"]}
                for c in ranking_candidates(result, tab, mode)
                if c["id"] in selected
            ]
            if result
            else []
        )
        title = "Select trials using the checkboxes to export their parameters."
        if trials:
            try:
                study = optuna_export_study(run_handler.get_run(context["run_id"]))
                return html.A(
                    _export_label(),
                    href="data:application/toml;charset=utf-8,"
                    + quote(
                        plan_toml({"trials": trials}, study, sampling_method="objective_guided"),
                        safe="",
                    ),
                    download=(
                        "full-space-candidates.toml"
                        if tab == "full_space"
                        else "coverage-candidates.toml"
                    ),
                    className="btn btn-outline-secondary btn-sm",
                    title=f"Export {len(trials):,} selected configurations as TOML",
                )
            except ValueError as error:
                title = str(error)
        return dbc.Button(
            _export_label(), color="secondary", outline=True, size="sm", disabled=True, title=title
        )

    @app.callback(  # type: ignore[untyped-decorator]
        Output(_id("performance-details"), "children"),
        Input(_id("performance-ranking"), "active_cell"),
        Input(_id("performance-data"), "data"),
        State(_id("context"), "data"),
    )
    def performance_detail(cell: Any, payload: Any, context: Any) -> Any:
        result = current_result(payload, context)
        if not result:
            return ""
        identifier = cell.get("row_id") if cell else None
        return candidate_details(result, identifier)
