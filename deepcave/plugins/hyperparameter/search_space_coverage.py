"""Coverage suite with immediate marginals and independently queued analyses."""

from typing import Any, Dict, List

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from time import time
from uuid import uuid4

import dash_bootstrap_components as dbc
from dash import ctx, dcc, html, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from deepcave import interactive
from deepcave.evaluators.coverage import (
    DEFAULT_REFERENCES,
    VERSION,
    CoverageData,
    compute_univariate_coverage,
    prepare_coverage,
    run_coverage_analysis,
)
from deepcave.plugins import Plugin
from deepcave.plugins.hyperparameter.coverage_views import (
    GROWTH_METRICS,
    PAIR_METRICS,
    _baseline_details,
    _card,
    _compatible_growth,
    _discrepancy_details,
    _empty_figure,
    _growth_figure,
    _id,
    _module_summary,
    _overview,
    _pair_matrix,
    _radius_details,
    _region_detail,
    _sampler,
    _scatter,
    _table,
    _univariate,
)

PREFIX = "search_space_coverage"
ANALYSES = {
    "pairwise": ("Pairwise Coverage", "Analyse all parameter pairs for gaps and uneven coverage."),
    "discrepancy": (
        "Global Discrepancy",
        "Evaluate discrepancy in the continuous sampling-space projection.",
    ),
    "radius": (
        "Coverage Radius & Holes",
        "Measure nearest-trial distances and locate uncovered regions.",
    ),
    "baseline": (
        "Random Baseline",
        "Compare coverage against random designs of identical trial count.",
    ),
    "growth": ("Coverage Growth", "Evaluate coverage as additional trials were sampled."),
    "growth_baseline": (
        "Random Coverage Growth",
        "Compare the realised trajectory against random expectation.",
    ),
}
_PREPARED: OrderedDict = OrderedDict()
_LOCK = RLock()


def clear_coverage_cache() -> None:
    """Clear memory preprocessing; disk results follow the shared cache reset."""
    with _LOCK:
        _PREPARED.clear()


def _prepared(run: Any) -> tuple:
    key = (run.id, id(run), id(run.configspace), run.latest_change, len(run.history))
    with _LOCK:
        if key not in _PREPARED:
            data = prepare_coverage(run)
            _PREPARED[key] = (data, compute_univariate_coverage(data))
            while len(_PREPARED) > 3:
                _PREPARED.popitem(last=False)
        _PREPARED.move_to_end(key)
        return _PREPARED[key]


def analysis_options(analysis: str, designs: Any, references: Any, seed: Any) -> Dict[str, int]:
    """Validate only the controls consumed by the selected analysis."""
    if analysis in ("pairwise", "discrepancy"):
        return {}
    options = {}
    for name, value, lower, upper in (
        ("reference_samples", references, 32, 20000),
        ("seed", seed, 0, 2**32 - 2001),
    ):
        if (
            value is None
            or isinstance(value, bool)
            or int(value) != value
            or not lower <= value <= upper
        ):
            raise ValueError(
                f"{name.replace('_', ' ').capitalize()} must be an integer "
                f"between {lower} and {upper}."
            )
        options[name] = int(value)
    if analysis in ("baseline", "growth_baseline"):
        if (
            designs is None
            or isinstance(designs, bool)
            or int(designs) != designs
            or not 1 <= designs <= 2000
        ):
            raise ValueError("Reference designs must be an integer between 1 and 2000.")
        options["designs"] = int(designs)
    return options


def analysis_status(entry: Dict, data: CoverageData, options: Dict) -> str:
    """Derive stale status without recomputing or discarding the previous result."""
    if entry.get("status") in ("running", "failed"):
        return entry["status"]
    if entry.get("result") is None:
        return "not_run"
    if (
        entry.get("fingerprint") != data.fingerprint
        or entry.get("options") != options
        or entry.get("version") != VERSION
    ):
        return "stale"
    return "completed"


class CoverageJobs:
    """Persist each module separately, surviving ordinary run-cache invalidation.

    The reserved directory lives alongside run IDs inside run_cache: clearing
    all plugin caches removes it, while refreshing one run preserves stale results.
    Only the server writes these manifests; workers return their snapshot results.
    """

    def __init__(self, cache_dir: Path, queue: Any) -> None:
        self.directory = Path(cache_dir) / "_coverage_suite"
        self.queue = queue

    def _path(self, run_id: str, analysis: str) -> Path:
        key = hashlib.sha256(run_id.encode()).hexdigest()
        return self.directory / key / f"{analysis}.json"

    def read(self, run_id: str, analysis: str) -> Dict:
        """Read one persisted module, or return an empty initial state."""
        path = self._path(run_id, analysis)
        return json.loads(path.read_text()) if path.exists() else {}

    def _write(self, run_id: str, analysis: str, entry: Dict) -> None:
        path = self._path(run_id, analysis)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(entry, allow_nan=False))
        temporary.replace(path)

    def poll(self, run_id: str, analysis: str) -> Dict:
        """Collect only an existing job; this method never starts work."""
        with _LOCK:
            entry = self.read(run_id, analysis)
            submitted = entry.get("job")
            if not submitted:
                return entry
            job_id = submitted["id"]
            try:
                job = self.queue.get_job(job_id)
                status = job.get_status(refresh=True)
                status = getattr(status, "value", status)
                if status in ("queued", "started", "scheduled", "deferred"):
                    return entry
                if status != "finished" or job.result is None:
                    raise RuntimeError("Analysis failed or was cancelled. Use Recompute to retry.")
                entry.update(
                    status="completed",
                    result=job.result,
                    fingerprint=submitted["fingerprint"],
                    trial_count=submitted["trial_count"],
                    options=submitted["options"],
                    version=VERSION,
                    computed_at=time(),
                    error=None,
                )
            except Exception as error:
                entry.update(status="failed", error=str(error))
            entry.pop("job", None)
            self._write(run_id, analysis, entry)
            # Results are persisted before the queue record is removed.
            try:
                self.queue.delete_job(job_id)
            except Exception:
                pass
            return entry

    def start(self, run: Any, analysis: str, data: CoverageData, options: Dict) -> Dict:
        """Submit exactly one module on an explicit user action."""
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown coverage analysis: {analysis}")
        with _LOCK:
            entry = self.poll(run.id, analysis)
            if entry.get("status") == "running":
                return entry
            job_id = f"coverage-{uuid4().hex}"
            submitted = {
                "id": job_id,
                "fingerprint": data.fingerprint,
                "trial_count": data.count,
                "options": options,
            }
            try:
                if not self.queue.ready():
                    raise RuntimeError(
                        "No analysis worker is available. Start a DeepCAVE worker, then retry."
                    )
                if not data.count:
                    raise ValueError("This analysis requires sampled trials.")
                self.queue.enqueue(
                    run_coverage_analysis,
                    args=[analysis, data, run.configspace, options],
                    job_id=job_id,
                    meta={
                        "display_name": ANALYSES[analysis][0],
                        "run_name": run.name,
                        "run_id": run.id,
                        "plugin_id": PREFIX,
                        "analysis": analysis,
                        "link": f"/plugins/{PREFIX}",
                    },
                )
                entry.update(status="running", job=submitted, error=None, cancelled=False)
            except Exception as error:
                entry.update(status="failed", error=str(error))
            self._write(run.id, analysis, entry)
            return entry

    def cancel(self, run_id: str, analysis: str) -> Dict:
        """Cancel only this module's job and retain its previous result."""
        with _LOCK:
            entry = self.poll(run_id, analysis)
            submitted = entry.get("job")
            if not submitted:
                return entry
            try:
                self.queue.delete_job(submitted["id"])
            except Exception as error:
                entry["error"] = f"Could not cancel analysis: {error}"
            else:
                entry.pop("job", None)
                entry.update(
                    status="completed" if entry.get("result") is not None else "not_run",
                    error=None,
                    cancelled=True,
                )
            self._write(run_id, analysis, entry)
            return entry


def _analysis_panel(analysis: str, summary: bool = False) -> Any:
    title, description = ANALYSES[analysis]
    cost = (
        "Very high"
        if analysis == "growth_baseline"
        else "High" if analysis in ("baseline", "growth", "radius", "discrepancy") else "Medium"
    )
    return html.Section(
        [
            (
                html.H4(
                    "Realised coverage"
                    if analysis == "growth"
                    else "Random expectation" if analysis == "growth_baseline" else title
                )
                if analysis != "pairwise"
                else None
            ),
            html.Small(description, className="coverage-job-description"),
            html.Div("Not computed", id=_id(f"{analysis}-status"), className="coverage-job-status"),
            html.Div(
                f"Estimated compute: {cost}",
                id=_id(f"{analysis}-metadata"),
                className="coverage-job-metadata",
            ),
            html.Div(id=_id(f"{analysis}-summary")) if summary else None,
            html.Div(
                [
                    dbc.Button(
                        "Run analysis",
                        id=_id(f"{analysis}-run"),
                        n_clicks=0,
                        title=f"Run or recompute {title}",
                        color="primary",
                        size="sm",
                    ),
                    dbc.Button(
                        "Cancel",
                        id=_id(f"{analysis}-cancel"),
                        n_clicks=0,
                        color="secondary",
                        outline=True,
                        size="sm",
                        style={"display": "none"},
                    ),
                    (
                        dbc.Button(
                            "Details",
                            id=_id(f"{analysis}-details"),
                            n_clicks=0,
                            color="link",
                            size="sm",
                        )
                        if summary
                        else None
                    ),
                ],
                className="coverage-job-actions",
            ),
        ],
        className="coverage-job coverage-job--inline" if analysis == "pairwise" else "coverage-job",
    )


def _current(entry: Any, context: Any) -> Dict:
    return (
        entry
        if isinstance(entry, dict) and context and entry.get("run_id") == context.get("run_id")
        else {}
    )


def _status_label(status: str, entry: Dict, count: int) -> str:
    computed = entry.get("trial_count", 0)
    if status == "running":
        snapshot = entry.get("job", {})
        return f"● Computing… · {snapshot.get('trial_count', count):,} trials"
    if status == "failed":
        return "Failed: " + (entry.get("error") or "Analysis failed")
    if status == "stale":
        return f"⚠ Computed for {computed:,} of {count:,} trials · Results remain visible"
    if status == "completed":
        return f"✓ Computed · {computed:,} trials"
    return "Cancelled" if entry.get("cancelled") else "Not computed"


def _metadata(analysis: str, status: str, entry: Dict) -> str:
    options = (
        entry.get("job", {}).get("options", {}) if status == "running" else entry.get("options", {})
    )
    parts = []
    if "designs" in options:
        parts.append(f"{options['designs']:,} reference designs")
    if "reference_samples" in options:
        parts.append(f"{options['reference_samples']:,} reference samples")
    if entry.get("computed_at") and status != "running":
        elapsed = max(0, int(time() - entry["computed_at"]))
        age = (
            "just now"
            if elapsed < 60
            else f"{elapsed // 60} min ago" if elapsed < 3600 else f"{elapsed // 3600} h ago"
        )
        parts.append(f"Last computed {age}")
    if entry.get("cancelled") and entry.get("result") is not None:
        parts.append("Recomputation cancelled · Previous results retained")
    if entry.get("error") and status == "running":
        parts.append(entry["error"])
    if not parts:
        parts.append(
            "Estimated compute: "
            + (
                "Very high"
                if analysis == "growth_baseline"
                else "Medium" if analysis == "pairwise" else "High"
            )
        )
    return " · ".join(parts)


class SearchSpaceCoverage(Plugin):
    """An objective-independent coverage page with no global process action."""

    id = PREFIX
    name = "Search Space Coverage"
    icon = "fas fa-border-all"
    help = "plugins/search_space_coverage.html"
    activate_run_selection = True

    @staticmethod
    def check_run_compatibility(run: Any) -> bool:
        """Accept runs and groups with a defined nonempty search space."""
        return run.configspace is not None and len(run.configspace) > 0

    @staticmethod
    def process(run: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return automatic content, or run one explicitly named analysis in API mode."""
        data = prepare_coverage(run)
        analysis = inputs.get("analysis")
        if analysis:
            options = analysis_options(
                analysis,
                inputs.get("designs", 500),
                inputs.get("reference_samples", DEFAULT_REFERENCES),
                inputs.get("seed", 0),
            )
            return run_coverage_analysis(analysis, data, run.configspace, options)
        return {"overview": data.overview, "parameters": compute_univariate_coverage(data)}

    @interactive
    def __call__(self, render_button: bool = False) -> List[Any]:
        """Build five question-focused tabs, retaining every module's explicit controls."""
        overview = [
            html.Div(id=_id("overview")),
            _card("Coverage status", [html.Div(id=_id("coverage-status"))]),
            html.Div(id=_id("definition")),
        ]
        univariate = _card(
            "Univariate Coverage",
            [
                html.Div(
                    [
                        html.Div(
                            [
                                dbc.Label("Parameter"),
                                dcc.Dropdown(id=_id("parameter"), clearable=False),
                            ],
                            className="coverage-selector",
                        ),
                        dbc.RadioItems(
                            id=_id("univariate-mode"),
                            value="distribution",
                            inline=True,
                            options=[
                                {"label": "Distribution", "value": "distribution"},
                                {"label": "ECDF", "value": "ecdf"},
                            ],
                        ),
                    ],
                    className="coverage-toolbar",
                ),
                html.Div(id=_id("univariate")),
            ],
        )
        pairwise = _card(
            "Pairwise Coverage",
            [
                _analysis_panel("pairwise"),
                html.Div(
                    [
                        html.Div(
                            [
                                dbc.Label("Matrix metric"),
                                dcc.Dropdown(
                                    id=_id("pair-metric"),
                                    value="occupancy",
                                    clearable=False,
                                    options=[
                                        {"label": label, "value": key}
                                        for key, label in PAIR_METRICS.items()
                                    ],
                                ),
                                dcc.Graph(
                                    id=_id("pair-matrix"),
                                    responsive=True,
                                    style={"height": "360px"},
                                    figure=_empty_figure(
                                        "Run pairwise analysis to populate the matrix."
                                    ),
                                    config={"displaylogo": False},
                                ),
                            ],
                            className="coverage-plot",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dbc.Label("X"),
                                                dcc.Dropdown(id=_id("x"), clearable=False),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                dbc.Label("Y"),
                                                dcc.Dropdown(id=_id("y"), clearable=False),
                                            ]
                                        ),
                                    ],
                                    className="coverage-pair-selectors",
                                ),
                                html.Div(id=_id("scatter")),
                            ],
                            className="coverage-plot",
                        ),
                    ],
                    className="coverage-pair-diagnostic",
                ),
                html.Small(
                    "Click a matrix cell to inspect that pair. "
                    "Empty diagonal and unavailable metrics are not selectable."
                ),
                html.Details(
                    [
                        html.Summary("Show detailed table"),
                        html.Div(id=_id("pairwise-note")),
                        _table(
                            [],
                            [
                                "x",
                                "y",
                                "active_trials",
                                "occupancy",
                                "empty_cell_rate",
                                "hole_score",
                                "discrepancy",
                            ],
                            id=_id("pair-ranking"),
                            cell_selectable=True,
                        ),
                    ],
                    className="coverage-details",
                ),
            ],
        )
        global_view = [
            html.Div(
                [
                    _analysis_panel(analysis, summary=True)
                    for analysis in ("discrepancy", "radius", "baseline")
                ],
                className="coverage-global-modules",
            ),
            dcc.Store(id=_id("global-detail")),
            html.Section(
                id=_id("discrepancy-section"),
                style={"display": "none"},
                children=html.Div(id=_id("discrepancy-result")),
            ),
            html.Section(
                id=_id("radius-section"),
                style={"display": "none"},
                children=[
                    html.Div(id=_id("radius-result")),
                    html.H4("Largest uncovered regions"),
                    html.Div(
                        [
                            _table(
                                [],
                                ["region", "distance"],
                                id=_id("regions"),
                                cell_selectable=True,
                                page_size=5,
                            ),
                            html.Div(id=_id("region-detail")),
                        ],
                        className="coverage-regions",
                    ),
                ],
            ),
            html.Section(
                id=_id("baseline-section"),
                style={"display": "none"},
                children=html.Div(id=_id("baseline-result")),
            ),
        ]
        growth = _card(
            "Coverage Growth",
            [
                html.Div(
                    [_analysis_panel("growth"), _analysis_panel("growth_baseline")],
                    className="coverage-growth-modules",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dbc.Label("Metric"),
                                dcc.Dropdown(
                                    id=_id("growth-metric"),
                                    value="p95_radius",
                                    clearable=False,
                                    options=[
                                        {"label": label, "value": key}
                                        for key, label in GROWTH_METRICS.items()
                                    ],
                                ),
                            ],
                            className="coverage-selector",
                        ),
                        dbc.Checklist(
                            id=_id("growth-overlay"),
                            value=[],
                            inline=True,
                            options=[
                                {
                                    "label": "+ Random expectation",
                                    "value": "random",
                                    "disabled": True,
                                }
                            ],
                        ),
                    ],
                    className="coverage-toolbar",
                ),
                dcc.Graph(
                    id=_id("growth-graph"),
                    responsive=True,
                    style={"height": "330px"},
                    figure=_empty_figure("Run coverage growth to see its trajectory."),
                    config={"displaylogo": False},
                ),
                html.Small(id=_id("growth-note")),
            ],
        )
        settings = dbc.Offcanvas(
            [
                html.P(
                    "Settings apply on the next explicit start. Existing results remain visible."
                ),
                dbc.Label("Reference designs (random baselines)"),
                dbc.Input(id=_id("designs"), type="number", value=500, min=1, max=2000, step=1),
                dbc.Label("Reference samples (radius and growth)", className="mt-3"),
                dbc.Input(
                    id=_id("references"),
                    type="number",
                    value=DEFAULT_REFERENCES,
                    min=32,
                    max=20000,
                    step=1,
                ),
                dbc.Label("Random seed", className="mt-3"),
                dbc.Input(id=_id("seed"), type="number", value=0, min=0, max=2**32 - 2001, step=1),
            ],
            id=_id("settings"),
            title="Analysis Settings",
            placement="end",
            is_open=False,
        )
        return [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div([html.H1(self.name), html.Div(id=_id("header-summary"))]),
                            dbc.Button(
                                "⚙ Analysis settings",
                                id=_id("settings-open"),
                                n_clicks=0,
                                color="secondary",
                                outline=True,
                                size="sm",
                            ),
                        ],
                        className="coverage-header",
                    ),
                    dcc.Interval(id=_id("tick"), interval=2000),
                    dcc.Store(id=_id("context")),
                    *[
                        dcc.Store(id=_id(f"{analysis}-{kind}"))
                        for analysis in ANALYSES
                        for kind in ("data", "view")
                    ],
                    settings,
                    dbc.Tabs(
                        [
                            dbc.Tab(overview, label="Overview", tab_id="overview"),
                            dbc.Tab(univariate, label="Univariate", tab_id="univariate"),
                            dbc.Tab(pairwise, label="Pairwise", tab_id="pairwise"),
                            dbc.Tab(global_view, label="Global", tab_id="global"),
                            dbc.Tab(growth, label="Growth", tab_id="growth"),
                        ],
                        id=_id("tabs"),
                        active_tab="overview",
                    ),
                ],
                className="coverage-suite",
            )
        ]

    @interactive
    def register_callbacks(self) -> None:
        """Register view-only interactions and six independently started worker jobs."""
        from deepcave import app, queue, rc, run_handler

        jobs = CoverageJobs(rc.cache_dir, queue)

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("overview"), "children"),
            Output(_id("definition"), "children"),
            Output(_id("context"), "data"),
            Output(_id("header-summary"), "children"),
            Input("global-run-selector", "value"),
            Input(_id("tick"), "n_intervals"),
            Input("general-cache-cleared-trigger", "data"),
            State(_id("context"), "data"),
        )
        def automatic(run_id: str, _: Any, revision: int, previous: Any) -> Any:
            if not run_id:
                return html.P("Select a run to inspect its search-space coverage."), [], None, ""
            if ctx.triggered_id == "general-cache-cleared-trigger":
                clear_coverage_cache()
            run = run_handler.get_run(run_id)
            data, marginals = _prepared(run)
            sampler = _sampler(run)
            context = {
                "run_id": run_id,
                "fingerprint": data.fingerprint,
                "trials": data.count,
                "revision": revision,
                "sampler": sampler,
            }
            if previous == context:
                raise PreventUpdate
            view = _overview(data, marginals, sampler)
            return (
                view.children[0],
                view.children[1],
                context,
                f"{data.count:,} trials · {len(data.axes)} parameters · "
                + (f"{sampler} sampler" if sampler != "Not recorded" else "Sampler not recorded"),
            )

        @app.callback(  # type: ignore[untyped-decorator]
            [
                Output(_id(name), prop)
                for name in ("parameter", "x", "y")
                for prop in ("options", "value")
            ]
            + [Output(_id("pair-ranking"), "active_cell"), Output(_id("pair-matrix"), "clickData")],
            Input(_id("context"), "data"),
            Input(_id("pair-ranking"), "active_cell"),
            Input(_id("pair-matrix"), "clickData"),
            State(_id("pair-ranking"), "data"),
            *[State(_id(name), "value") for name in ("parameter", "x", "y")],
        )
        def selectors(
            context: Any, active_cell: Any, click: Any, rows: Any, parameter: str, x: str, y: str
        ) -> Any:
            if not context:
                return [[], None, [], None, [], None, None, None]
            data, _ = _prepared(run_handler.get_run(context["run_id"]))
            names = [a.name for a in data.axes]
            options = [{"label": name, "value": name} for name in names]
            if ctx.triggered_id == _id("pair-ranking") and active_cell:
                row_id = active_cell.get("row_id")
                selected = next((row for row in rows or [] if row.get("id") == row_id), None)
                if selected:
                    x, y = selected["x"], selected["y"]
            if ctx.triggered_id == _id("pair-matrix") and click and click.get("points"):
                point = click["points"][0]
                if (
                    point.get("x") in names
                    and point.get("y") in names
                    and point["x"] != point["y"]
                    and point.get("z") is not None
                ):
                    x, y = point["x"], point["y"]
            return [
                options,
                parameter if parameter in names else names[0],
                options,
                x if x in names else names[0],
                options,
                y if y in names else names[min(1, len(names) - 1)],
                None,
                None,
            ]

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("univariate"), "children"),
            Output(_id("univariate-mode"), "options"),
            Output(_id("univariate-mode"), "value"),
            Input(_id("context"), "data"),
            Input(_id("parameter"), "value"),
            Input(_id("univariate-mode"), "value"),
        )
        def marginal(context: Any, parameter: str, mode: str) -> Any:
            if not context:
                raise PreventUpdate
            _, rows = _prepared(run_handler.get_run(context["run_id"]))
            row = next((r for r in rows if r["name"] == parameter), None)
            continuous = row is not None and row["cardinality"] is None
            options = [
                {"label": "Distribution", "value": "distribution"},
                {"label": "ECDF", "value": "ecdf", "disabled": not continuous},
            ]
            mode = mode if continuous and mode in ("distribution", "ecdf") else "distribution"
            return _univariate(row, mode) if row else [], options, mode

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("scatter"), "children"),
            Input(_id("context"), "data"),
            Input(_id("x"), "value"),
            Input(_id("y"), "value"),
            Input(_id("pairwise-data"), "data"),
        )
        def scatter(context: Any, x: str, y: str, entry: Any) -> Any:
            if not context:
                return []
            data, _ = _prepared(run_handler.get_run(context["run_id"]))
            return _scatter(data, x, y, _current(entry, context).get("result"))

        for analysis in ANALYSES:
            self._register_analysis_callback(app, jobs, run_handler, analysis)

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("settings"), "is_open"),
            Input(_id("settings-open"), "n_clicks"),
            State(_id("settings"), "is_open"),
            prevent_initial_call=True,
        )
        def settings(n_clicks: int, is_open: bool) -> bool:
            return not is_open if n_clicks else is_open

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("coverage-status"), "children"),
            Input(_id("context"), "data"),
            *[Input(_id(f"{analysis}-data"), "data") for analysis in ANALYSES],
        )
        def coverage_status(context: Any, *entries: Any) -> Any:
            if not context:
                return []
            rows = [html.Div([html.Dt("Marginal coverage"), html.Dd("Available automatically")])]
            for analysis, raw in zip(ANALYSES, entries):
                entry = _current(raw, context)
                status = entry.get("status", "not_run")
                if status == "completed" and entry.get("fingerprint") != context["fingerprint"]:
                    status = "stale"
                labels = {
                    "not_run": "Not computed",
                    "running": "● Computing…",
                    "completed": "✓ Computed",
                    "stale": "⚠ Outdated",
                    "failed": "Failed",
                }
                rows.append(html.Div([html.Dt(ANALYSES[analysis][0]), html.Dd(labels[status])]))
            return html.Dl(rows, className="coverage-definition-list coverage-status-list")

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("pair-matrix"), "figure"),
            Output(_id("pair-ranking"), "data"),
            Output(_id("pairwise-note"), "children"),
            Input(_id("context"), "data"),
            Input(_id("pairwise-data"), "data"),
            Input(_id("pair-metric"), "value"),
        )
        def matrix(context: Any, raw: Any, metric: str) -> Any:
            if not context:
                raise PreventUpdate
            data, _ = _prepared(run_handler.get_run(context["run_id"]))
            result = _current(raw, context).get("result")
            names = [a.name for a in data.axes if a.kind != "constant"]
            metric = metric if metric in PAIR_METRICS else "occupancy"
            rows = [{**r, "id": f"{r['x']} × {r['y']}"} for r in result["pairs"]] if result else []
            return _pair_matrix(result, metric, names), rows, result["note"] if result else ""

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("global-detail"), "data"),
            *[
                Input(_id(f"{analysis}-details"), "n_clicks")
                for analysis in ("discrepancy", "radius", "baseline")
            ],
            Input(_id("context"), "data"),
            State(_id("global-detail"), "data"),
        )
        def global_detail(_: Any, __: Any, ___: Any, context: Any, previous: Any) -> Any:
            if not context:
                return None
            for analysis in ("discrepancy", "radius", "baseline"):
                if ctx.triggered_id == _id(f"{analysis}-details"):
                    return {"run_id": context["run_id"], "analysis": analysis}
            return previous if previous and previous.get("run_id") == context["run_id"] else None

        @app.callback(  # type: ignore[untyped-decorator]
            *[
                Output(_id(f"{analysis}-section"), "style")
                for analysis in ("discrepancy", "radius", "baseline")
            ],
            Input(_id("global-detail"), "data"),
        )
        def show_detail(selected: Any) -> Any:
            return [
                {} if selected and selected.get("analysis") == analysis else {"display": "none"}
                for analysis in ("discrepancy", "radius", "baseline")
            ]

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("discrepancy-result"), "children"),
            Output(_id("radius-result"), "children"),
            Output(_id("baseline-result"), "children"),
            Output(_id("regions"), "data"),
            Input(_id("context"), "data"),
            *[
                Input(_id(f"{analysis}-data"), "data")
                for analysis in ("discrepancy", "radius", "baseline")
            ],
        )
        def global_results(context: Any, discrepancy: Any, radius: Any, baseline: Any) -> Any:
            results = [
                _current(entry, context).get("result") for entry in (discrepancy, radius, baseline)
            ]
            regions = [
                {"id": i, "region": i + 1, "distance": r["distance"]}
                for i, r in enumerate((results[1] or {}).get("regions", []))
            ]
            return (
                _discrepancy_details(results[0]),
                _radius_details(results[1]),
                _baseline_details(results[2]),
                regions,
            )

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("region-detail"), "children"),
            Output(_id("regions"), "active_cell"),
            Input(_id("context"), "data"),
            Input(_id("radius-data"), "data"),
            Input(_id("regions"), "active_cell"),
        )
        def region_detail(context: Any, entry: Any, cell: Any) -> Any:
            regions = (_current(entry, context).get("result") or {}).get("regions", [])
            if not regions:
                return html.P("No uncovered regions computed."), None
            index = cell.get("row_id", 0) if cell and ctx.triggered_id == _id("regions") else 0
            index = index if isinstance(index, int) and 0 <= index < len(regions) else 0
            data, _ = _prepared(run_handler.get_run(context["run_id"]))
            return _region_detail(regions[index], index, [a.name for a in data.axes]), no_update

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("growth-graph"), "figure"),
            Output(_id("growth-overlay"), "options"),
            Output(_id("growth-overlay"), "value"),
            Output(_id("growth-note"), "children"),
            Input(_id("context"), "data"),
            Input(_id("growth-data"), "data"),
            Input(_id("growth_baseline-data"), "data"),
            Input(_id("growth-metric"), "value"),
            Input(_id("growth-overlay"), "value"),
        )
        def growth(context: Any, raw: Any, raw_random: Any, metric: str, overlay: Any) -> Any:
            realised, random = _current(raw, context), _current(raw_random, context)
            compatible = _compatible_growth(realised, random)
            value = ["random"] if compatible and "random" in (overlay or []) else []
            metric = metric if metric in GROWTH_METRICS else "p95_radius"
            options = [
                {"label": "+ Random expectation", "value": "random", "disabled": not compatible}
            ]
            result = realised.get("result") or random.get("result")
            note = (
                "Random expectation requires its own explicit start."
                if not random.get("result")
                else (
                    "Random expectation uses the same trial snapshot and reference settings."
                    if compatible
                    else "Random expectation uses a different trial snapshot "
                    "or reference settings. Recompute to compare."
                )
            )
            if result:
                note += " Discrepancy projection: " + (
                    ", ".join(result["discrepancy_parameters"]) or "unavailable"
                )
            return _growth_figure(realised, random, metric, bool(value)), options, value, note

    def _register_analysis_callback(
        self, app: Any, jobs: CoverageJobs, run_handler: Any, analysis: str
    ) -> None:
        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id(f"{analysis}-data"), "data"),
            Output(_id(f"{analysis}-status"), "children"),
            Output(_id(f"{analysis}-run"), "children"),
            Output(_id(f"{analysis}-run"), "disabled"),
            Output(_id(f"{analysis}-view"), "data"),
            Output(_id(f"{analysis}-cancel"), "style"),
            Output(_id(f"{analysis}-metadata"), "children"),
            *(
                [Output(_id(f"{analysis}-summary"), "children")]
                if analysis in ("discrepancy", "radius", "baseline")
                else []
            ),
            Input(_id(f"{analysis}-run"), "n_clicks"),
            Input(_id(f"{analysis}-cancel"), "n_clicks"),
            Input(_id("tick"), "n_intervals"),
            Input(_id("context"), "data"),
            Input(_id("designs"), "value"),
            Input(_id("references"), "value"),
            Input(_id("seed"), "value"),
            State(_id(f"{analysis}-view"), "data"),
            State("global-run-selector", "value"),
        )
        def update(
            n_clicks: int,
            cancel_clicks: int,
            _: Any,
            context: Any,
            designs: Any,
            references: Any,
            seed: Any,
            previous: Any,
            active_run: str,
        ) -> Any:
            if not context or context["run_id"] != active_run:
                raise PreventUpdate
            run = run_handler.get_run(context["run_id"])
            data, _ = _prepared(run)
            entry = jobs.poll(run.id, analysis)
            if cancel_clicks and ctx.triggered_id == _id(f"{analysis}-cancel"):
                entry = jobs.cancel(run.id, analysis)
            error = None
            try:
                options = analysis_options(analysis, designs, references, seed)
            except (ValueError, TypeError, OverflowError) as exception:
                error, options = str(exception), {}
            if not error and n_clicks and ctx.triggered_id == _id(f"{analysis}-run"):
                entry = jobs.start(run, analysis, data, options)
            status = analysis_status(entry, data, options)
            view = {
                "run": run.id,
                "fingerprint": data.fingerprint,
                "status": status,
                "options": options,
                "result_fingerprint": entry.get("fingerprint"),
                "result_options": entry.get("options"),
                "error": error or entry.get("error"),
                "job": entry.get("job"),
                "revision": context.get("revision"),
                "computed_at": entry.get("computed_at"),
                "cancelled": entry.get("cancelled"),
                "age_minute": int(time() // 60) if entry.get("computed_at") else None,
            }
            if previous == view:
                raise PreventUpdate
            payload = {
                "run_id": run.id,
                "fingerprint": entry.get("fingerprint"),
                "version": entry.get("version"),
                "trial_count": entry.get("trial_count"),
                "options": entry.get("options", {}),
                "status": status,
                "result": entry.get("result"),
                "computed_at": entry.get("computed_at"),
            }
            label = _status_label(status, entry, data.count)
            if error:
                label += " · " + error
            button = (
                "Computing…"
                if status == "running"
                else (
                    "↻"
                    if status == "completed"
                    else (
                        "Recompute"
                        if entry.get("result") is not None or status == "failed"
                        else "Run analysis"
                    )
                )
            )
            outputs = [
                payload,
                label,
                button,
                status == "running" or bool(error) or data.count == 0,
                view,
                {} if status == "running" else {"display": "none"},
                _metadata(analysis, status, entry),
            ]
            if analysis in ("discrepancy", "radius", "baseline"):
                outputs.append(_module_summary(analysis, entry.get("result")))
            return outputs
