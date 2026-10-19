"""Coverage suite with immediate marginals and independently queued analyses."""

from typing import Any, Dict, List, Tuple

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from time import time
from uuid import uuid4

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import ctx, dcc, html, no_update
from dash.dependencies import ALL, Input, Output, State
from dash.exceptions import PreventUpdate

from deepcave import interactive
from deepcave.evaluators.combinatorial import BASELINE_VERSION
from deepcave.evaluators.coverage import (
    DEFAULT_REFERENCES,
    MIXED_MMD_VERSION,
    VERSION,
    CoverageData,
    compute_univariate_coverage,
    prepare_coverage,
    run_coverage_analysis,
)
from deepcave.evaluators.coverage_forecast import FORECAST_VERSION, forecast_options
from deepcave.evaluators.coverage_planning import (
    PLANNING_VERSION,
    planning_options,
    planning_source_error,
)
from deepcave.plugins import Plugin
from deepcave.plugins.hyperparameter.analysis_utils import (
    _add_trial_statistics,
    _get_objectives_without_time,
)
from deepcave.plugins.hyperparameter.combinatorial_groups import (
    diagnostic_row,
    group_drilldown,
    group_overview,
    group_records,
    missing_tuple_page,
)
from deepcave.plugins.hyperparameter.combinatorial_views import (
    combinatorial_resolution_detail,
    combinatorial_view,
)
from deepcave.plugins.hyperparameter.coverage_planning_views import (
    plan_trial_detail,
    plan_view,
    planning_coverage,
    planning_panel,
)
from deepcave.plugins.hyperparameter.coverage_views import (
    DISCREPANCY_METRICS,
    GROWTH_METRICS,
    PAIR_METRICS,
    PAIR_PLOT_HEIGHT,
    _card,
    _compatible_growth,
    _empty_figure,
    _growth_figure,
    _id,
    _module_summary,
    _overview,
    _pair_legend,
    _pair_matrix,
    _pair_panel,
    _radius_details,
    _region_detail,
    _sampler,
    _scatter,
    _table,
    _univariate_rows,
)
from deepcave.runs.status import Status

PREFIX = "search_space_coverage"
ANALYSES = {
    "forecast": ("Trial forecast", "Forecast additional random trials to complete coverage."),
    "planning": ("Improve Coverage", "Plan additional configurations to close coverage gaps."),
    "combinatorial": (
        "Global Combinatorial Coverage",
        "Measure coverage across resolutions and interaction strengths for independent parameters.",
    ),
    "combinatorial_baseline": (
        "Coverage Array",
        "Measure coverage and calibrate it using complete random studies of the same size.",
    ),
    "pairwise": ("Pairwise Coverage", "Analyse all parameter pairs for gaps and uneven coverage."),
    "discrepancy": (
        "Sampling distribution",
        "Compare sampled configurations with random designs from the complete search space.",
    ),
    "radius": (
        "Coverage Radius & Holes",
        "Measure nearest-trial distances and locate uncovered regions.",
    ),
    "baseline": (
        "Random comparisons",
        "Add a separate random reference to each discrepancy and distance metric.",
    ),
    "growth": ("Coverage Growth", "Evaluate coverage as additional trials were sampled."),
    "growth_baseline": (
        "Random Coverage Growth",
        "Compare the realised trajectory against random expectation.",
    ),
}
UI_ANALYSES = {
    key: value
    for key, value in ANALYSES.items()
    if key not in ("discrepancy", "baseline", "combinatorial", "planning", "forecast")
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
    options = {}
    if analysis == "combinatorial":
        return options
    for name, value, lower, upper in (
        ("reference_samples", references, 32, 20000),
        ("seed", seed, 0, 2**32 - 2001),
    ):
        if analysis == "combinatorial_baseline" and name == "reference_samples":
            continue
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
    if analysis in ("baseline", "growth_baseline", "combinatorial_baseline"):
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
                if not data.count and analysis not in ("planning", "forecast"):
                    raise ValueError("This analysis requires successful trials.")
                worker_args = [analysis, data, run.configspace, options]
                if analysis == "forecast":
                    snapshot = None
                    for source in ("combinatorial_baseline", "combinatorial"):
                        cached = self.read(run.id, source)
                        if (
                            cached.get("fingerprint") == data.fingerprint
                            and cached.get("version") == VERSION
                            and cached.get("result")
                        ):
                            snapshot = (
                                cached["result"]["observed"]
                                if source == "combinatorial_baseline"
                                else cached["result"]
                            )
                            break
                    worker_args.append(snapshot)
                self.queue.enqueue(
                    run_coverage_analysis,
                    args=worker_args,
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


def _elite_trials(run: Any, proportion: float) -> Tuple[List[int], str]:
    """Identify elite history rows with the same quantile and tie rules as 1-D analysis."""
    objectives = _get_objectives_without_time(run)
    if not objectives:
        return [], "Elite unavailable: no optimization objective"
    objective = objectives[0]
    objective_id = run.get_objective_id(objective)
    rows = []
    for index, trial in enumerate(t for t in run.history if t.status == Status.SUCCESS):
        value = trial.costs[objective_id]
        if trial.status == Status.SUCCESS and value is not None and np.isfinite(value):
            rows.append({"history_index": index, "objective": float(value)})
    if not rows:
        return [], f"Elite unavailable: no successful trials with {objective.name}"
    trials, _, _ = _add_trial_statistics(pd.DataFrame(rows), objective.optimize, proportion)
    indices = trials.loc[trials["is_elite"], "history_index"].astype(int).tolist()
    return indices, f"Elite: best {100 * proportion:g}% ({objective.name})"


def _analysis_panel(analysis: str, summary: bool = False) -> Any:
    title, description = ANALYSES[analysis]
    cost = (
        "Very high"
        if analysis in ("growth_baseline", "combinatorial_baseline")
        else "High" if analysis in ("baseline", "growth", "radius", "discrepancy") else "Medium"
    )
    heading: Any = (
        html.H4(
            "Realised coverage"
            if analysis == "growth"
            else "Random expectation" if analysis == "growth_baseline" else title
        )
        if analysis not in ("pairwise", "combinatorial", "combinatorial_baseline", "forecast")
        else None
    )
    if analysis == "discrepancy":
        heading = html.Div(
            [
                heading,
                html.Div(
                    [
                        dbc.Label(
                            "Comparison method",
                            html_for=_id("discrepancy-metric"),
                            className="visually-hidden",
                        ),
                        dcc.Dropdown(
                            id=_id("discrepancy-metric"),
                            value="mixed_mmd",
                            clearable=False,
                            searchable=False,
                            options=[
                                {"label": label, "value": key}
                                for key, label in DISCREPANCY_METRICS.items()
                            ],
                        ),
                    ],
                    className="coverage-uniformity-method",
                ),
            ],
            className="coverage-uniformity-header",
        )
    return html.Section(
        [
            heading,
            (
                html.Small(description, className="coverage-job-description")
                if analysis
                not in (
                    "pairwise",
                    "discrepancy",
                    "combinatorial",
                    "combinatorial_baseline",
                    "forecast",
                )
                else None
            ),
            html.Div(
                "" if analysis == "pairwise" else "Not computed",
                id=_id(f"{analysis}-status"),
                className="coverage-job-status",
            ),
            html.Div(
                (
                    ""
                    if analysis in ("pairwise", "combinatorial_baseline", "forecast")
                    else f"Estimated compute: {cost}"
                ),
                id=_id(f"{analysis}-metadata"),
                className="coverage-job-metadata",
            ),
            html.Div(id=_id(f"{analysis}-summary")) if summary else None,
            html.Div(
                [
                    dbc.Button(
                        "Compute forecast" if analysis == "forecast" else "Run analysis",
                        id=_id(f"{analysis}-run"),
                        n_clicks=0,
                        title=(
                            "Refresh coverage and random baseline"
                            if analysis == "combinatorial_baseline"
                            else f"Run or recompute {title}"
                        ),
                        color="secondary" if analysis == "discrepancy" else "primary",
                        outline=analysis == "discrepancy",
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
                            "Method & raw values" if analysis == "discrepancy" else "Details",
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
        className=(
            "coverage-ccm-baseline-controls"
            if analysis in ("combinatorial_baseline", "forecast")
            else (
                "coverage-job coverage-ccm-controls"
                if analysis == "combinatorial"
                else (
                    "coverage-job coverage-job--inline"
                    if analysis == "pairwise"
                    else (
                        "coverage-job coverage-uniformity"
                        if analysis == "discrepancy"
                        else "coverage-job"
                    )
                )
            )
        ),
    )


def _current(entry: Any, context: Any) -> Dict:
    return (
        entry
        if isinstance(entry, dict) and context and entry.get("run_id") == context.get("run_id")
        else {}
    )


def _combinatorial_snapshot(context: Any, raw: Any, random: Any) -> tuple:
    """Reuse the same measured snapshot in the profile and resolution diagnosis."""
    observed, reference = _current(raw, context), _current(random, context)
    result, baseline = observed.get("result"), reference.get("result")
    if not result and baseline:
        result = baseline["observed"]
    elif baseline and (
        not observed.get("fingerprint")
        or any(
            observed.get(key) != reference.get(key)
            for key in ("fingerprint", "version", "trial_count")
        )
    ):
        baseline = None
    return result, baseline


def _pair_result(entry: Any, context: Any) -> Any:
    """Require the current metric definitions before displaying pairwise results."""
    current = _current(entry, context)
    return current.get("result") if current.get("version") == VERSION else None


def _matching_global_baseline(
    entry: Dict, baseline: Dict, radius: bool = False, mixed: bool = False
) -> Any:
    """Compare only identical snapshots, and identical radius reference settings."""
    result = baseline.get("result") or {}
    if not entry.get("result") or "discrepancy_bands" not in result:
        return None
    if not entry.get("fingerprint") or any(
        entry.get(key) != baseline.get(key)
        for key in ("run_id", "fingerprint", "version", "trial_count")
    ):
        return None
    if mixed and (
        entry["result"].get("mixed_version") != MIXED_MMD_VERSION
        or result.get("mixed_version") != MIXED_MMD_VERSION
    ):
        return None
    if (radius or mixed) and any(
        entry.get("options", {}).get(key) != baseline.get("options", {}).get(key)
        for key in ("reference_samples", "seed")
    ):
        return None
    return result


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
    global_analysis = analysis in ("discrepancy", "radius", "baseline")
    options = (
        entry.get("job", {}).get("options", {}) if status == "running" else entry.get("options", {})
    )
    parts = []
    if "designs" in options and analysis != "combinatorial_baseline":
        parts.append(f"{options['designs']:,} reference designs")
    if "reference_samples" in options:
        parts.append(f"{options['reference_samples']:,} reference samples")
    if entry.get("computed_at") and status != "running" and not global_analysis:
        elapsed = max(0, int(time() - entry["computed_at"]))
        age = (
            "just now"
            if elapsed < 60
            else f"{elapsed // 60} min ago" if elapsed < 3600 else f"{elapsed // 3600} h ago"
        )
        parts.append(
            f"Updated {age}" if analysis == "combinatorial_baseline" else f"Last computed {age}"
        )
    if entry.get("cancelled") and entry.get("result") is not None:
        parts.append("Recomputation cancelled · Previous results retained")
    if entry.get("error") and status == "running":
        parts.append(entry["error"])
    if (
        not parts
        and analysis != "combinatorial_baseline"
        and not (global_analysis and entry.get("result") is not None)
    ):
        parts.append(
            "Estimated compute: "
            + (
                "Very high"
                if analysis in ("growth_baseline", "combinatorial_baseline")
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
        if analysis == "forecast":
            return run_coverage_analysis(analysis, data, run.configspace, forecast_options(inputs))
        if analysis == "planning":
            error = planning_source_error(run)
            if error:
                raise ValueError(error)
            return run_coverage_analysis(analysis, data, run.configspace, planning_options(inputs))
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
            html.Div(id=_id("definition")),
        ]
        univariate = html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Univariate Coverage"),
                                html.P(
                                    "Distribution of observed trials across the search space "
                                    "for each parameter."
                                ),
                            ],
                            className="coverage-univariate-heading",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dbc.Label("View", html_for=_id("univariate-mode")),
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
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Group by", html_for=_id("univariate-group")),
                                        dcc.Dropdown(
                                            id=_id("univariate-group"),
                                            value="type",
                                            clearable=False,
                                            options=[
                                                {"label": "Type", "value": "type"},
                                                {"label": "None", "value": "none"},
                                            ],
                                        ),
                                    ],
                                    className="coverage-univariate-select",
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Sort by", html_for=_id("univariate-sort")),
                                        dcc.Dropdown(
                                            id=_id("univariate-sort"),
                                            value="max_gap",
                                            clearable=False,
                                            options=[
                                                {"label": "Largest gap", "value": "max_gap"},
                                                {"label": "Coverage", "value": "coverage"},
                                                {"label": "KS", "value": "ks"},
                                                {"label": "Name", "value": "name"},
                                            ],
                                        ),
                                    ],
                                    className="coverage-univariate-select",
                                ),
                                html.Div(
                                    dbc.Switch(
                                        id=_id("univariate-problematic"),
                                        value=False,
                                        label="Only problematic",
                                    ),
                                    title="Coverage below 100%, KS distance above 0.1, or a "
                                    "continuous sampling-space gap above 0.1. ECDF applies to "
                                    "continuous and integer parameters; categories retain their "
                                    "frequency distribution. Integer KS is descriptive, with no p-value.",
                                    className="coverage-univariate-filter",
                                ),
                            ],
                            className="coverage-univariate-controls",
                        ),
                    ],
                    className="coverage-univariate-header",
                ),
                html.Div(
                    [
                        html.Span("Observed", className="coverage-legend-observed"),
                        html.Span("Expected", className="coverage-legend-expected"),
                        html.Span(
                            "Largest gap",
                            className="coverage-legend-gap",
                            title=(
                                "Largest interval without observed trials in transformed "
                                "search space, including edge gaps. Orange rug ticks mark "
                                "observed boundaries; labels identify search-space edges."
                            ),
                        ),
                    ],
                    className="coverage-univariate-legend",
                ),
                html.Div(id=_id("univariate")),
            ],
            className="coverage-card coverage-univariate",
        )
        pairwise = _card(
            "Pairwise Coverage",
            [
                _analysis_panel("pairwise"),
                html.Div(
                    [
                        dbc.Label("Elite trials", html_for=_id("elite-proportion")),
                        dcc.Dropdown(
                            id=_id("elite-proportion"),
                            options=[
                                {"label": f"{percent}%", "value": percent / 100}
                                for percent in (20, 10, 5, 2, 1)
                            ],
                            value=0.1,
                            clearable=False,
                            searchable=False,
                        ),
                    ],
                    className="coverage-pair-elite-controls",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dbc.Label(
                                                    "Matrix metric", html_for=_id("pair-metric")
                                                ),
                                                dbc.RadioItems(
                                                    id=_id("pair-metric"),
                                                    value="occupancy",
                                                    className=(
                                                        "multi-metrics-segmented "
                                                        "coverage-pair-metric-buttons"
                                                    ),
                                                    options=[
                                                        {"label": label, "value": key}
                                                        for key, label in PAIR_METRICS.items()
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                    className="coverage-pair-controls",
                                ),
                                _pair_panel(
                                    "Pairwise Parameter Matrix",
                                    _pair_legend(),
                                    dcc.Graph(
                                        id=_id("pair-matrix"),
                                        responsive=True,
                                        style={"height": f"{PAIR_PLOT_HEIGHT}px"},
                                        figure=_pair_matrix(None, "occupancy", []),
                                        className="coverage-pair-plot-row",
                                        config={"displaylogo": False},
                                    ),
                                ),
                            ],
                            className="coverage-pair-column",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dbc.Label(name.upper()),
                                                dcc.Dropdown(id=_id(name), clearable=False),
                                            ]
                                        )
                                        for name in ("x", "y")
                                    ],
                                    className="coverage-pair-selectors",
                                ),
                                html.Div(
                                    [
                                        html.Div(id=_id("scatter")),
                                        html.Div(
                                            [
                                                dbc.Switch(
                                                    id=_id("highlight-empty-cells"),
                                                    label="Highlight empty cells",
                                                    value=True,
                                                    className="coverage-pair-highlight",
                                                ),
                                                dbc.Switch(
                                                    id=_id("highlight-elite-trials"),
                                                    label="Highlight elite trials",
                                                    value=False,
                                                    className="coverage-pair-highlight",
                                                ),
                                            ],
                                            className="coverage-pair-scatter-footer",
                                        ),
                                    ],
                                    className="coverage-pair-slot",
                                ),
                            ],
                            className="coverage-pair-column",
                        ),
                    ],
                    className="coverage-pair-diagnostic",
                ),
                html.Dl(
                    [
                        html.Div([html.Dt(PAIR_METRICS[key]), html.Dd(description)])
                        for key, description in (
                            (
                                "occupancy",
                                "Share of grid cells with at least one trial. "
                                "Higher means broader grid coverage.",
                            ),
                            (
                                "empty_cell_rate",
                                "Share of grid cells without trials (100% − occupancy). "
                                "Lower means fewer empty cells.",
                            ),
                            (
                                "mmd",
                                "Difference between the observed joint distribution and a "
                                "synthetic reference distribution randomly sampled from the "
                                "configured search space (Similar to Mutual Information, but "
                                "here the reference distribution is sampled from the defined "
                                "search space instead of from the observed marginals of the "
                                "two features).",
                            ),
                            (
                                "max_radius",
                                "Largest normalized distance from a randomly drawn reference "
                                "point to its nearest observed feature configuration. "
                                "Lower means smaller sampled gaps.",
                            ),
                        )
                    ],
                    className="coverage-pair-glossary",
                    **{"aria-label": "Metric glossary"},
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
                                "max_radius",
                                "mmd",
                            ],
                            id=_id("pair-ranking"),
                            cell_selectable=True,
                        ),
                    ],
                    className="coverage-details",
                ),
            ],
        )
        pairwise.className = "coverage-card coverage-pairwise"
        global_view = [
            html.Section(
                [
                    dcc.Store(id=_id("combinatorial-resolution-selection")),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        html.I(className="fas fa-chart-bar"),
                                        className="coverage-profile-icon",
                                        **{"aria-hidden": "true"},
                                    ),
                                    html.Div(
                                        [
                                            html.H3("Coverage Array"),
                                            html.Div(id=_id("combinatorial-profile-counts")),
                                        ]
                                    ),
                                ],
                                className="coverage-profile-title",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        "Random baseline · awaiting values",
                                        id=_id("combinatorial-baseline-label"),
                                        className="coverage-profile-baseline-badge",
                                    ),
                                    _analysis_panel("combinatorial_baseline"),
                                    _analysis_panel("forecast"),
                                ],
                                className="coverage-profile-benchmark-controls",
                            ),
                        ],
                        className="coverage-profile-header",
                    ),
                    html.Div(id=_id("combinatorial-result")),
                ],
                className="coverage-card coverage-profile-card",
            ),
            planning_panel(),
        ]
        growth = _card(
            "Coverage Growth",
            [
                dcc.Store(id=_id("global-detail")),
                _analysis_panel("radius", summary=True),
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
                dbc.Label("Reference samples (pairwise, radius and growth)", className="mt-3"),
                dbc.Input(
                    id=_id("references"),
                    type="number",
                    value=DEFAULT_REFERENCES,
                    min=32,
                    max=20000,
                    step=1,
                ),
                html.Hr(),
                dbc.Label("Forecast probability"),
                dcc.Dropdown(
                    id=_id("forecast-probability"),
                    value=0.8,
                    clearable=False,
                    options=[{"label": f"{p:.0%}", "value": p} for p in (0.5, 0.8, 0.95)],
                ),
                dbc.Label("Forecast simulations", className="mt-3"),
                dbc.Input(
                    id=_id("forecast-simulations"),
                    type="number",
                    value=2000,
                    min=20,
                    max=10000,
                    step=1,
                ),
                dbc.Label("Maximum trials per continuation", className="mt-3"),
                dbc.Input(
                    id=_id("forecast-max-trials"),
                    type="number",
                    value=100000,
                    min=1,
                    max=10000000,
                    step=1,
                ),
                dbc.Label("Forecast time limit (seconds)", className="mt-3"),
                dbc.Input(id=_id("forecast-timeout"), type="number", value=30, min=0.1, max=120),
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
                        for analysis in (*UI_ANALYSES, "forecast")
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
        """Register view-only interactions and independently started worker jobs."""
        from deepcave import app, queue, rc, run_handler

        jobs = CoverageJobs(rc.cache_dir, queue)
        self._register_planning_callbacks(app, jobs, run_handler)
        self._register_forecast_callback(app, jobs, run_handler)

        def coverage_array_snapshot(context: Any, random: Any) -> tuple:
            reference = _current(random, context)
            measured = None
            if context and not reference.get("result"):
                measured = {
                    "run_id": context["run_id"],
                    **jobs.read(context["run_id"], "combinatorial"),
                }
            result, baseline = _combinatorial_snapshot(context, measured, reference)
            source = reference if baseline else measured or {}
            if result is None and context:
                saved = jobs.read(context["run_id"], "forecast")
                result = (saved.get("result") or {}).get("observed")
                source = saved
            if result:
                result = {**result, "source_fingerprint": source.get("fingerprint")}
            return result, baseline

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("planning-budget-field"), "style"),
            Output(_id("planning-strength-field"), "style"),
            Output(_id("planning-strength"), "disabled"),
            Output(_id("planning-coverage"), "children"),
            Output(_id("planning-strength-mode"), "options"),
            Output(_id("planning-strength"), "max"),
            Input(_id("planning-strategy"), "value"),
            Input(_id("planning-strength-mode"), "value"),
            Input(_id("planning-resolution"), "value"),
            Input(_id("context"), "data"),
            Input(_id("combinatorial_baseline-data"), "data"),
            Input(_id("forecast-data"), "data"),
        )
        def planning_controls(
            strategy: Any, mode: Any, resolution: Any, context: Any, random: Any, forecast: Any
        ) -> Any:
            result, _ = coverage_array_snapshot(context, random)
            metrics, options, maximum = planning_coverage(result, resolution)
            return (
                {} if strategy == "maximize_coverage" else {"display": "none"},
                {} if mode == "custom" else {"display": "none"},
                mode != "custom",
                metrics,
                options,
                maximum,
            )

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("combinatorial-result"), "children"),
            Output(_id("combinatorial-profile-counts"), "children"),
            Output(_id("combinatorial-baseline-label"), "children"),
            Input(_id("context"), "data"),
            Input(_id("combinatorial_baseline-data"), "data"),
            Input(_id("forecast-data"), "data"),
        )
        def combinatorial_results(context: Any, random: Any, raw_forecast: Any = None) -> Any:
            result, baseline = coverage_array_snapshot(context, random)
            entry = _current(raw_forecast, context)
            forecast = entry.get("result")
            if (
                forecast
                and result
                and forecast.get("source_fingerprint") != result.get("source_fingerprint")
            ):
                forecast = None
            counts = (
                f"{len(result['parameters'])} Parameters · {result['trials']:,} Trials"
                if result
                else ""
            )
            label = (
                f"Random baseline · {baseline['designs']:,} samples"
                if baseline
                else "Random baseline · awaiting values"
            )
            return combinatorial_view(result, baseline, forecast), counts, label

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("combinatorial-resolution-detail"), "children"),
            Output(_id("combinatorial-resolution"), "value"),
            Output(_id("combinatorial-resolution-selection"), "data"),
            Input(_id("combinatorial-resolution"), "value"),
            Input(_id("context"), "data"),
            Input(_id("combinatorial_baseline-data"), "data"),
            State(_id("combinatorial-resolution-selection"), "data"),
        )
        def combinatorial_detail(
            selected: Any, context: Any, random: Any, previous: Any = None
        ) -> Any:
            result, _ = coverage_array_snapshot(context, random)
            values = selected or []
            run_id = (context or {}).get("run_id")
            previous = previous or {}
            if previous.get("run_id") not in (None, run_id):
                resolution = None
            else:
                added = [value for value in values if value != previous.get("resolution")]
                resolution = (
                    added[-1]
                    if added
                    else (
                        previous.get("resolution") if previous.get("resolution") in values else None
                    )
                )
            if resolution not in [row["resolution"] for row in (result or {}).get("rows", [])]:
                resolution = None
            return (
                combinatorial_resolution_detail(result, resolution),
                [resolution] if resolution is not None else [],
                {"run_id": run_id, "resolution": resolution},
            )

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("group-overview"), "children"),
            Input(_id("group-filter"), "value"),
            Input(_id("group-limit"), "value"),
            State(_id("combinatorial-resolution-selection"), "data"),
            State(_id("context"), "data"),
            State(_id("combinatorial_baseline-data"), "data"),
        )
        def combinatorial_groups(
            parameters: Any, limit: Any, selection: Any, context: Any, random: Any
        ) -> Any:
            result, _ = coverage_array_snapshot(context, random)
            row = diagnostic_row(result, (selection or {}).get("resolution"))
            if not row:
                raise PreventUpdate
            return group_overview(result, row, parameters, limit)

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("group-drilldown"), "children"),
            Output(_id("group-selection"), "data"),
            Input({"type": _id("group-open"), "resolution": ALL, "group": ALL}, "n_clicks"),
            Input(_id("group-filter"), "value"),
            Input(_id("group-limit"), "value"),
            State(_id("group-selection"), "data"),
            State(_id("combinatorial-resolution-selection"), "data"),
            State(_id("context"), "data"),
            State(_id("combinatorial_baseline-data"), "data"),
            prevent_initial_call=True,
        )
        def combinatorial_group_detail(
            clicks: Any,
            parameters: Any,
            limit: Any,
            previous: Any,
            selection: Any,
            context: Any,
            random: Any,
        ) -> Any:
            trigger = ctx.triggered_id
            if trigger in (_id("group-filter"), _id("group-limit")):
                return [], None
            if not isinstance(trigger, dict) or not any(clicks or []):
                raise PreventUpdate
            result, _ = coverage_array_snapshot(context, random)
            resolution = (selection or {}).get("resolution")
            row = diagnostic_row(result, resolution)
            if not row or trigger.get("resolution") != resolution:
                return [], None
            selected = {
                "group": trigger["group"],
                "resolution": resolution,
                "run_id": (context or {}).get("run_id"),
            }
            if selected == previous:
                return [], None
            groups = group_records(result, row, parameters)
            shown = groups if limit == "all" else groups[: int(limit or 5)]
            group = next((group for group in shown if group["index"] == trigger["group"]), None)
            if group is None:
                return [], None
            return group_drilldown(result, row, group), selected

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("group-tuples"), "data"),
            Output(_id("group-tuples"), "tooltip_data"),
            Input(_id("group-tuples"), "page_current"),
            State(_id("group-selection"), "data"),
            State(_id("context"), "data"),
            State(_id("combinatorial_baseline-data"), "data"),
            prevent_initial_call=True,
        )
        def combinatorial_group_page(page: Any, selected: Any, context: Any, random: Any) -> Any:
            if not selected or selected.get("run_id") != (context or {}).get("run_id"):
                raise PreventUpdate
            result, _ = coverage_array_snapshot(context, random)
            row = diagnostic_row(result, selected["resolution"])
            if not row:
                raise PreventUpdate
            group = next(
                (
                    group
                    for group in group_records(result, row)
                    if group["index"] == selected["group"]
                ),
                None,
            )
            if group is None:
                raise PreventUpdate
            return missing_tuple_page(result, row, group, page)

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
            [Output(_id(name), prop) for name in ("x", "y") for prop in ("options", "value")]
            + [Output(_id("pair-ranking"), "active_cell"), Output(_id("pair-matrix"), "clickData")],
            Input(_id("context"), "data"),
            Input(_id("pair-ranking"), "active_cell"),
            Input(_id("pair-matrix"), "clickData"),
            State(_id("pair-ranking"), "data"),
            *[State(_id(name), "value") for name in ("x", "y")],
        )
        def selectors(context: Any, active_cell: Any, click: Any, rows: Any, x: str, y: str) -> Any:
            if not context:
                return [[], None, [], None, None, None]
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
            Input(_id("univariate-mode"), "value"),
            Input(_id("univariate-group"), "value"),
            Input(_id("univariate-sort"), "value"),
            Input(_id("univariate-problematic"), "value"),
        )
        def marginal(
            context: Any, mode: str, group_by: str, sort_by: str, only_problematic: bool
        ) -> Any:
            if not context:
                raise PreventUpdate
            _, rows = _prepared(run_handler.get_run(context["run_id"]))
            supports_ecdf = any(
                row["cardinality"] is None or row["type"] == "integer" for row in rows
            )
            options = [
                {"label": "Distribution", "value": "distribution"},
                {"label": "ECDF", "value": "ecdf", "disabled": not supports_ecdf},
            ]
            mode = mode if supports_ecdf and mode in ("distribution", "ecdf") else "distribution"
            return _univariate_rows(rows, mode, group_by, sort_by, only_problematic), options, mode

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("scatter"), "children"),
            Input(_id("context"), "data"),
            Input(_id("x"), "value"),
            Input(_id("y"), "value"),
            Input(_id("pairwise-data"), "data"),
            Input(_id("highlight-empty-cells"), "value"),
            Input(_id("highlight-elite-trials"), "value"),
            Input(_id("elite-proportion"), "value"),
        )
        def scatter(
            context: Any,
            x: str,
            y: str,
            entry: Any,
            highlight_empty: bool,
            highlight_elite: bool = False,
            elite_proportion: float = 0.1,
        ) -> Any:
            if not context:
                return []
            run = run_handler.get_run(context["run_id"])
            data, _ = _prepared(run)
            proportion = (
                elite_proportion if elite_proportion in (0.2, 0.1, 0.05, 0.02, 0.01) else 0.1
            )
            elite_indices, elite_label = (
                _elite_trials(run, proportion) if highlight_elite else (None, None)
            )
            return _scatter(
                data,
                x,
                y,
                _pair_result(entry, context),
                highlight_empty,
                elite_indices,
                elite_label,
            )

        for analysis in UI_ANALYSES:
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
            result = _pair_result(raw, context)
            names = [a.name for a in data.axes if a.kind != "constant"]
            metric = metric if metric in PAIR_METRICS else "occupancy"
            rows = [{**r, "id": f"{r['x']} × {r['y']}"} for r in result["pairs"]] if result else []
            return _pair_matrix(result, metric, names), rows, result["note"] if result else ""

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("global-detail"), "data"),
            Input(_id("radius-details"), "n_clicks"),
            Input(_id("context"), "data"),
            State(_id("global-detail"), "data"),
        )
        def global_detail(_: Any, context: Any, previous: Any) -> Any:
            if not context:
                return None
            if ctx.triggered_id == _id("radius-details"):
                return {"run_id": context["run_id"], "analysis": "radius"}
            return previous if previous and previous.get("run_id") == context["run_id"] else None

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("radius-section"), "style"),
            Input(_id("global-detail"), "data"),
        )
        def show_detail(selected: Any) -> Any:
            return {} if selected and selected.get("analysis") == "radius" else {"display": "none"}

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("radius-result"), "children"),
            Output(_id("radius-summary"), "children"),
            Output(_id("regions"), "data"),
            Input(_id("context"), "data"),
            Input(_id("radius-data"), "data"),
        )
        def global_results(context: Any, radius: Any) -> Any:
            result = _current(radius, context).get("result")
            regions = [
                {"id": i, "region": i + 1, "distance": region["distance"]}
                for i, region in enumerate((result or {}).get("regions", []))
            ]
            return _radius_details(result), _module_summary("radius", result), regions

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

    def _register_forecast_callback(self, app: Any, jobs: CoverageJobs, run_handler: Any) -> None:
        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("forecast-data"), "data"),
            Output(_id("forecast-status"), "children"),
            Output(_id("forecast-run"), "children"),
            Output(_id("forecast-run"), "disabled"),
            Output(_id("forecast-view"), "data"),
            Output(_id("forecast-cancel"), "style"),
            Output(_id("forecast-metadata"), "children"),
            Input(_id("forecast-run"), "n_clicks"),
            Input(_id("forecast-cancel"), "n_clicks"),
            Input(_id("tick"), "n_intervals"),
            Input(_id("context"), "data"),
            Input(_id("forecast-probability"), "value"),
            Input(_id("forecast-simulations"), "value"),
            Input(_id("forecast-max-trials"), "value"),
            Input(_id("forecast-timeout"), "value"),
            Input(_id("seed"), "value"),
            State("global-run-selector", "value"),
            State(_id("forecast-view"), "data"),
            State(_id("forecast-data"), "data"),
        )
        def forecast_job(
            clicks: Any,
            cancel: Any,
            tick: Any,
            context: Any,
            probability: Any,
            simulations: Any,
            max_trials: Any,
            timeout: Any,
            seed: Any,
            active_run: Any,
            previous: Any,
            current: Any = None,
        ) -> Any:
            if not context or context["run_id"] != active_run:
                raise PreventUpdate
            run = run_handler.get_run(active_run)
            data, _ = _prepared(run)
            entry = jobs.poll(run.id, "forecast")
            if cancel and ctx.triggered_id == _id("forecast-cancel"):
                entry = jobs.cancel(run.id, "forecast")
            error, options = None, {}
            try:
                options = forecast_options(
                    dict(
                        probability=probability,
                        simulations=simulations,
                        max_trials=max_trials,
                        timeout=timeout,
                        seed=seed,
                    )
                )
            except ValueError as exception:
                error = str(exception)
            if not error and clicks and ctx.triggered_id == _id("forecast-run"):
                entry = jobs.start(run, "forecast", data, options)
            status = analysis_status(entry, data, options)
            if (
                status == "completed"
                and entry["result"].get("forecast_version") != FORECAST_VERSION
            ):
                status = "stale"
            message = {
                "not_run": "Forecast additional trials under independent random sampling.",
                "running": "Computing trial forecast…",
                "completed": "Forecast saved. Approximate values and conservative bounds are labelled.",
                "stale": "Forecast uses earlier data, settings or methods. Recompute to update it.",
                "failed": entry.get("error") or "Forecast failed. Retry to recompute.",
            }[status]
            if error:
                message += " " + error
            if status == "completed":
                measured = jobs.read(run.id, "combinatorial_baseline")
                if measured.get("result") and measured.get("fingerprint") != entry.get(
                    "fingerprint"
                ):
                    message += " Refresh coverage to display matching forecasts."
            view = dict(
                run_id=run.id,
                fingerprint=data.fingerprint,
                options=options,
                status=status,
                error=error,
                computed_at=entry.get("computed_at"),
                job=entry.get("job"),
                revision=context.get("revision"),
            )
            if view == previous:
                raise PreventUpdate
            payload = dict(
                run_id=run.id,
                fingerprint=entry.get("fingerprint"),
                status=status,
                result=entry.get("result"),
                options=entry.get("options"),
            )
            return (
                no_update if payload == current else payload,
                message,
                (
                    "Computing…"
                    if status == "running"
                    else "Recompute forecast" if entry.get("result") else "Compute forecast"
                ),
                status == "running" or bool(error),
                view,
                {} if status == "running" else {"display": "none"},
                "",
            )

    def _register_planning_callbacks(self, app: Any, jobs: CoverageJobs, run_handler: Any) -> None:
        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("planning-resolution"), "value"),
            Input(_id("combinatorial-resolution-selection"), "data"),
            prevent_initial_call=True,
        )
        def planning_resolution(selection: Any) -> Any:
            return (selection or {}).get("resolution") or no_update

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("planning-data"), "data"),
            Output(_id("planning-status"), "children"),
            Output(_id("planning-run"), "children"),
            Output(_id("planning-run"), "disabled"),
            Output(_id("planning-cancel"), "style"),
            Output(_id("planning-result"), "children"),
            Output(_id("planning-view"), "data"),
            Input(_id("planning-run"), "n_clicks"),
            Input(_id("planning-cancel"), "n_clicks"),
            Input(_id("tick"), "n_intervals"),
            Input(_id("context"), "data"),
            Input(_id("planning-strategy"), "value"),
            Input(_id("planning-resolution"), "value"),
            Input(_id("planning-strength-mode"), "value"),
            Input(_id("planning-strength"), "value"),
            Input(_id("planning-budget"), "value"),
            Input(_id("planning-timeout"), "value"),
            Input(_id("planning-seed"), "value"),
            State("global-run-selector", "value"),
            State(_id("planning-view"), "data"),
        )
        def planning_job(
            clicks: Any,
            cancel: Any,
            tick: Any,
            context: Any,
            strategy: Any,
            resolution: Any,
            mode: Any,
            strength: Any,
            budget: Any,
            timeout: Any,
            seed: Any,
            active_run: Any,
            previous: Any,
        ) -> Any:
            if not context or context["run_id"] != active_run:
                raise PreventUpdate
            run = run_handler.get_run(active_run)
            data, _ = _prepared(run)
            entry = jobs.poll(run.id, "planning")
            # A click may arrive together with a timer tick or a control update.
            triggered = ctx.triggered_prop_ids.values()
            cancel_requested = bool(cancel) and _id("planning-cancel") in triggered
            if cancel_requested:
                entry = jobs.cancel(run.id, "planning")
            error = planning_source_error(run)
            options = {}
            try:
                options = planning_options(
                    {
                        "strategy": strategy,
                        "resolution": resolution,
                        "strength_mode": mode,
                        "target_strength": strength,
                        "max_additional_trials": budget if strategy == "maximize_coverage" else None,
                        "solver_timeout": timeout,
                        "seed": seed,
                    }
                )
            except ValueError as exception:
                error = error or str(exception)
            if not cancel_requested and not error and clicks and _id("planning-run") in triggered:
                entry = jobs.start(run, "planning", data, options)
            status = analysis_status(entry, data, options)
            if (
                entry.get("result")
                and entry["result"].get("version") != PLANNING_VERSION
                and status == "completed"
            ):
                status = "stale"
            message = {
                "not_run": "Generate a plan to improve coverage of the selected run or group.",
                "running": "Generating coverage plan…",
                "completed": "Plan saved in DeepCAVE. Observed coverage is unchanged.",
                "stale": "This plan uses an earlier study state or different settings. Recompute to update it.",
                "failed": entry.get("error") or "Planning failed. Retry to generate a new plan.",
            }[status]
            if error:
                message += " " + error
            view = {
                "run_id": run.id,
                "fingerprint": data.fingerprint,
                "status": status,
                "options": options,
                "error": error or entry.get("error"),
                "computed_at": entry.get("computed_at"),
                "job": entry.get("job"),
                "revision": context.get("revision"),
            }
            if view == previous:
                raise PreventUpdate
            payload = {"run_id": run.id, "status": status, "result": entry.get("result")}
            same_result = previous and all(
                previous.get(key) == view.get(key) for key in ("run_id", "computed_at", "revision")
            )
            return (
                payload,
                message,
                (
                    "Generating…"
                    if status == "running"
                    else "Recompute plan" if entry.get("result") else "Generate plan"
                ),
                status == "running" or bool(error),
                {} if status == "running" else {"display": "none"},
                no_update if same_result else plan_view(entry.get("result")),
                view,
            )

        @app.callback(  # type: ignore[untyped-decorator]
            Output(_id("planning-trial-detail"), "children"),
            Input(_id("planning-trials"), "active_cell"),
            State(_id("planning-trials"), "derived_virtual_data"),
            State(_id("planning-data"), "data"),
            prevent_initial_call=True,
        )
        def planning_detail(cell: Any, rows: Any, entry: Any) -> Any:
            if not cell or not entry:
                return []
            position = cell.get("row_id")
            if position is None and rows and 0 <= cell.get("row", -1) < len(rows):
                position = rows[cell["row"]]["id"]
            return plan_trial_detail(entry.get("result") or {}, position)

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
            Input(_id(f"{analysis}-run"), "n_clicks"),
            Input(_id(f"{analysis}-cancel"), "n_clicks"),
            Input(_id("tick"), "n_intervals"),
            Input(_id("context"), "data"),
            Input(_id("designs"), "value"),
            Input(_id("references"), "value"),
            Input(_id("seed"), "value"),
            State(_id(f"{analysis}-view"), "data"),
            State("global-run-selector", "value"),
            State(_id(f"{analysis}-data"), "data"),
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
            current: Any = None,
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
            missing_global_details = (
                analysis in ("discrepancy", "baseline")
                and entry.get("result") is not None
                and (
                    ("checkpoints" if analysis == "discrepancy" else "discrepancy_bands")
                    not in entry["result"]
                    or entry["result"].get("mixed_version") != MIXED_MMD_VERSION
                )
            )
            if status == "completed" and missing_global_details:
                status = "stale"
            missing_ccm_reference = (
                analysis == "combinatorial_baseline"
                and entry.get("result") is not None
                and entry["result"].get("baseline_version") != BASELINE_VERSION
            )
            if status == "completed" and missing_ccm_reference:
                status = "stale"
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
            if missing_global_details and status == "stale":
                label = "⚠ Recalculate to compare the complete search space"
            if missing_ccm_reference and status == "stale":
                label = "Refresh to use the updated strength calculation"
            if analysis == "pairwise" and status in ("not_run", "completed") and not error:
                label = ""
            if analysis == "combinatorial_baseline" and status == "not_run" and not error:
                label = ""
            if (
                analysis
                in ("discrepancy", "radius", "baseline", "combinatorial", "combinatorial_baseline")
                and status == "completed"
                and not error
            ):
                label = ""
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
            if analysis == "discrepancy" and button in ("↻", "Recompute"):
                button = "Recalculate"
            if analysis == "combinatorial_baseline":
                if status == "not_run":
                    button = "Compute coverage"
                elif status == "completed":
                    button = "Refresh"
            outputs = [
                # Updating the age label must not rebuild result views and reset
                # expanded details, filters or pagination when their data is unchanged.
                no_update if payload == current else payload,
                label,
                button,
                status == "running" or bool(error) or data.count == 0,
                view,
                {} if status == "running" else {"display": "none"},
                "" if analysis == "pairwise" else _metadata(analysis, status, entry),
            ]
            return outputs
