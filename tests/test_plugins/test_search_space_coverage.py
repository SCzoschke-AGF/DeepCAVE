"""The coverage page never starts expensive work without a module button."""

from types import SimpleNamespace

import pytest
from ConfigSpace import ConfigurationSpace
from dash import Dash, dcc, html
from dash.exceptions import PreventUpdate

import deepcave
from deepcave.evaluators.coverage import prepare_coverage
from deepcave.plugins.hyperparameter import coverage_views as views
from deepcave.plugins.hyperparameter import search_space_coverage as suite
from deepcave.runs.status import Status
from deepcave.utils.run_caches import RunCaches


class FakeQueue:
    def __init__(self):
        self.jobs = {}
        self.calls = []
        self.available = True

    def ready(self):
        return self.available

    def enqueue(self, function, args, job_id, meta):
        job = SimpleNamespace(id=job_id, status="queued", result=None, function=function, args=args)
        job.get_status = lambda refresh=True: job.status
        self.jobs[job_id] = job
        self.calls.append(meta)

    def get_job(self, job_id):
        return self.jobs[job_id]

    def delete_job(self, job_id):
        self.jobs.pop(job_id, None)

    def finish(self, job_id):
        job = self.jobs[job_id]
        job.result = job.function(*job.args)
        job.status = "finished"


@pytest.fixture
def run():
    configs = [{"x": 0.25, "y": 0.75}, {"x": 0.75, "y": 0.25}]
    return SimpleNamespace(
        id="run",
        name="Example run",
        latest_change=1,
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)}),
        history=[
            SimpleNamespace(config_id=i, budget=1, seed=0, status=Status.SUCCESS) for i in range(2)
        ],
        get_config=lambda index: configs[index],
    )


def test_api_defaults_only_compute_cheap_analyses(monkeypatch, run):
    def unexpected(*args, **kwargs):
        pytest.fail("Opening coverage must not dispatch expensive analyses")

    monkeypatch.setattr(suite, "run_coverage_analysis", unexpected)
    result = suite.SearchSpaceCoverage.process(run, {})
    assert result["overview"]["trials"] == 2
    assert len(result["parameters"]) == 2


def test_normalized_data_and_marginals_are_reused(monkeypatch, run):
    suite.clear_coverage_cache()
    original = suite.prepare_coverage
    calls = []

    def tracked(run):
        calls.append(run.id)
        return original(run)

    monkeypatch.setattr(suite, "prepare_coverage", tracked)
    assert suite._prepared(run) is suite._prepared(run)
    assert calls == ["run"]
    run.latest_change = 2
    suite._prepared(run)
    assert calls == ["run", "run"]


def test_jobs_have_independent_caches_and_survive_run_invalidation(tmp_path, run):
    queue = FakeQueue()
    jobs = suite.CoverageJobs(tmp_path / "run_cache", queue)
    data = prepare_coverage(run)
    assert jobs.poll(run.id, "pairwise") == {}
    assert not queue.calls
    started = jobs.start(run, "pairwise", data, {})
    assert suite.analysis_status(started, data, {}) == "running"
    # Repeated clicks and another page session do not duplicate an active job.
    jobs.start(run, "pairwise", data, {})
    assert len(queue.calls) == 1
    queue.finish(started["job"]["id"])
    finished = jobs.poll(run.id, "pairwise")
    assert suite.analysis_status(finished, data, {}) == "completed"
    assert jobs.poll(run.id, "radius") == {}
    assert not queue.jobs

    config = SimpleNamespace(CACHE_DIR=tmp_path, DEBUG=False)
    caches = RunCaches(config)
    (caches.cache_dir / run.id).mkdir()
    caches.clear_run(run)
    reopened = suite.CoverageJobs(caches.cache_dir, queue)
    assert reopened.read(run.id, "pairwise")["result"] == finished["result"]
    changed = prepare_coverage(run)
    changed.fingerprint = "new-trials"
    changed.values.append({"x": 0.5, "y": 0.5})
    assert suite.analysis_status(finished, changed, {}) == "stale"
    assert len(queue.calls) == 1
    caches.clear()
    assert reopened.read(run.id, "pairwise") == {}


def test_inflight_snapshot_becomes_stale_and_does_not_restart(tmp_path, run):
    queue = FakeQueue()
    jobs = suite.CoverageJobs(tmp_path, queue)
    data = prepare_coverage(run)
    started = jobs.start(run, "discrepancy", data, {})
    current = prepare_coverage(run)
    current.fingerprint = "updated"
    queue.finish(started["job"]["id"])
    result = jobs.poll(run.id, "discrepancy")
    assert result["fingerprint"] == data.fingerprint
    assert suite.analysis_status(result, current, {}) == "stale"
    assert len(queue.calls) == 1
    assert suite.CoverageJobs(tmp_path, queue).read(run.id, "discrepancy") == result


def test_failures_keep_prior_results_and_allow_explicit_retry(tmp_path, run):
    queue = FakeQueue()
    jobs = suite.CoverageJobs(tmp_path, queue)
    data = prepare_coverage(run)
    first = jobs.start(run, "discrepancy", data, {})
    queue.finish(first["job"]["id"])
    result = jobs.poll(run.id, "discrepancy")["result"]
    retry = jobs.start(run, "discrepancy", data, {})
    queue.jobs[retry["job"]["id"]].status = "failed"
    failed = jobs.poll(run.id, "discrepancy")
    assert failed["status"] == "failed" and failed["result"] == result
    assert suite.analysis_status(failed, data, {}) == "failed"
    retry = jobs.start(run, "discrepancy", data, {})
    assert retry["status"] == "running"
    assert len(queue.calls) == 3


def test_missing_worker_does_not_hide_existing_results(tmp_path, run):
    queue = FakeQueue()
    queue.available = False
    jobs = suite.CoverageJobs(tmp_path, queue)
    result = jobs.start(run, "radius", prepare_coverage(run), {"reference_samples": 32, "seed": 0})
    assert result["status"] == "failed"
    assert "worker" in result["error"]
    assert not queue.calls


def test_controls_invalidate_only_the_modules_that_use_them():
    assert suite.analysis_options("pairwise", None, None, None) == {}
    assert suite.analysis_options("discrepancy", None, None, None) == {}
    assert suite.analysis_options("radius", None, 32, 7) == {"reference_samples": 32, "seed": 7}
    assert suite.analysis_options("growth", 500, 32, 7) == suite.analysis_options(
        "growth", 1000, 32, 7
    )
    for invalid in (None, 0, 2.5, True, 2001):
        with pytest.raises(ValueError):
            suite.analysis_options("baseline", invalid, 32, 0)


def test_cancel_stops_only_the_selected_job_and_preserves_previous_results(tmp_path, run):
    queue = FakeQueue()
    jobs = suite.CoverageJobs(tmp_path, queue)
    data = prepare_coverage(run)
    initial = jobs.start(run, "discrepancy", data, {})
    queue.finish(initial["job"]["id"])
    previous = jobs.poll(run.id, "discrepancy")
    repeated = jobs.start(run, "discrepancy", data, {})
    other = jobs.start(run, "pairwise", data, {})
    cancelled = jobs.cancel(run.id, "discrepancy")
    assert cancelled["result"] == previous["result"]
    assert cancelled["computed_at"] == previous["computed_at"]
    assert cancelled["cancelled"] is True
    assert suite.analysis_status(cancelled, data, {}) == "completed"
    assert repeated["job"]["id"] not in queue.jobs
    assert other["job"]["id"] in queue.jobs
    fresh = jobs.cancel(run.id, "pairwise")
    assert suite.analysis_status(fresh, data, {}) == "not_run"
    assert not queue.jobs


def test_region_table_keeps_parameters_inactive_in_the_first_region():
    table = suite._table(
        [{"parent": "off", "distance": 1}, {"parent": "on", "child": 0.25, "distance": 0.5}]
    )
    assert "child" in [column["id"] for column in table.columns]


@pytest.fixture
def callbacks(monkeypatch, tmp_path, run):
    registered = []

    def register(*args, **kwargs):
        def decorate(function):
            registered.append(function)
            return function

        return decorate

    queue = FakeQueue()
    context = SimpleNamespace(triggered_id=None)
    monkeypatch.setattr(suite, "ctx", context)
    monkeypatch.setattr(deepcave, "app", SimpleNamespace(callback=register), raising=False)
    monkeypatch.setattr(deepcave, "queue", queue, raising=False)
    monkeypatch.setattr(deepcave, "rc", SimpleNamespace(cache_dir=tmp_path), raising=False)
    monkeypatch.setattr(
        deepcave, "run_handler", SimpleNamespace(get_run=lambda _: run), raising=False
    )
    suite.clear_coverage_cache()
    plugin = suite.SearchSpaceCoverage()
    suite.SearchSpaceCoverage.register_callbacks.__wrapped__(plugin)
    return registered, queue, context


def test_each_button_is_the_only_start_trigger(callbacks, run):
    registered, queue, event = callbacks
    context = {"run_id": run.id, "fingerprint": prepare_coverage(run).fingerprint, "revision": 0}
    analysis_callbacks = [callback for callback in registered if callback.__name__ == "update"]
    assert len(analysis_callbacks) == len(suite.ANALYSES)
    for analysis, callback in zip(suite.ANALYSES, analysis_callbacks):
        for trigger in (
            None,
            suite._id("tick"),
            suite._id("context"),
            suite._id("designs"),
            suite._id("references"),
            suite._id("tabs"),
            suite._id("growth-overlay"),
            suite._id("pair-metric"),
        ):
            event.triggered_id = trigger
            output = callback(12, 0, 1, context, 2, 32, 0, None, run.id)
            assert output[1] == "Not computed"
            assert not queue.calls
    for index, (analysis, callback) in enumerate(zip(suite.ANALYSES, analysis_callbacks)):
        event.triggered_id = suite._id(f"{analysis}-run")
        output = callback(1, 0, 1, context, 2, 32, 0, None, run.id)
        assert output[3] is True
        assert queue.calls[-1]["analysis"] == analysis
        assert len(queue.calls) == index + 1


def test_callback_rejects_stale_run_context(callbacks, run):
    registered, queue, event = callbacks
    event.triggered_id = suite._id("pairwise-run")
    with pytest.raises(PreventUpdate):
        registered[4](1, 0, 1, {"run_id": run.id}, 2, 32, 0, None, "other-run")
    assert not queue.calls


def test_ranking_selection_uses_row_identity_after_sorting(callbacks, run):
    registered, _, event = callbacks
    event.triggered_id = suite._id("pair-ranking")
    rows = [{"id": "x × y", "x": "x", "y": "y"}, {"id": "y × x", "x": "y", "y": "x"}]
    outputs = registered[1](
        {"run_id": run.id}, {"row": 0, "row_id": "y × x"}, None, rows, "x", "x", "y"
    )
    assert outputs[3] == "y" and outputs[5] == "x"
    assert outputs[6] is None


def test_matrix_selection_updates_pair_but_ignores_diagonal_and_missing_metrics(callbacks, run):
    registered, queue, event = callbacks
    event.triggered_id = suite._id("pair-matrix")
    selectors = registered[1]
    context = {"run_id": run.id}
    changed = selectors(
        context, None, {"points": [{"x": "y", "y": "x", "z": 0.5}]}, [], "x", "x", "y"
    )
    assert changed[3] == "y" and changed[5] == "x"
    assert changed[7] is None
    for point in ({"x": "x", "y": "x", "z": 1}, {"x": "y", "y": "x", "z": None}):
        same = selectors(context, None, {"points": [point]}, [], "x", "x", "y")
        assert same[3] == "x" and same[5] == "y"
    assert not queue.calls


def test_region_drilldown_uses_sorted_row_identity_and_shows_inactive_values(callbacks, run):
    registered, queue, event = callbacks
    callback = next(c for c in registered if c.__name__ == "region_detail")
    event.triggered_id = suite._id("regions")
    entry = {
        "run_id": run.id,
        "result": {
            "regions": [
                {"x": 0.2, "distance": 0.8},
                {"x": 0.9, "y": 0.1, "distance": 0.7},
            ]
        },
    }
    detail, _ = callback({"run_id": run.id}, entry, {"row": 0, "row_id": 1})
    assert detail.children[0].children == "Region 2"
    first, _ = callback({"run_id": run.id}, entry, {"row": 1, "row_id": 0})
    definition = first.children[2]
    assert definition.children[1].children[1].children == "Inactive"
    assert not queue.calls


def test_univariate_switches_one_plot_and_disables_ecdf_for_categories(callbacks, run):
    registered, queue, _ = callbacks
    callback = next(c for c in registered if c.__name__ == "marginal")
    content, options, mode = callback({"run_id": run.id}, "x", "ecdf")
    assert mode == "ecdf" and options[1]["disabled"] is False
    diagnostic = content.children[0]
    assert diagnostic.children[0].className == "coverage-metric-stack"
    graph = diagnostic.children[1].children
    assert graph.figure.data[0].name == "Observed ECDF"
    assert len(graph.figure.data) == 2
    assert not queue.calls


def test_global_details_and_settings_are_view_only(callbacks, run):
    registered, queue, event = callbacks
    by_name = {c.__name__: c for c in registered}
    event.triggered_id = suite._id("radius-details")
    selected = by_name["global_detail"](0, 1, 0, {"run_id": run.id}, None)
    assert selected["analysis"] == "radius"
    assert by_name["show_detail"](selected) == [{"display": "none"}, {}, {"display": "none"}]
    assert by_name["settings"](1, False) is True
    assert by_name["settings"](2, True) is False
    assert not queue.calls


def test_cancel_still_works_when_settings_are_invalid(callbacks, run):
    registered, queue, event = callbacks
    callback = registered[6]  # Coverage radius job.
    context = {"run_id": run.id, "fingerprint": prepare_coverage(run).fingerprint}
    event.triggered_id = suite._id("radius-run")
    started = callback(1, 0, 1, context, 2, 32, 0, None, run.id)
    assert started[3] is True
    event.triggered_id = suite._id("radius-cancel")
    cancelled = callback(1, 1, 2, context, 2, None, 0, started[4], run.id)
    assert cancelled[0]["status"] == "not_run"
    assert cancelled[5] == {"display": "none"}
    assert not queue.jobs


def test_growth_overlay_requires_matching_trial_snapshot_and_reference_settings():
    rows = [{"trials": 1, "p95_radius": 0.5}, {"trials": 2, "p95_radius": 0.25}]
    realised = {
        "run_id": "run",
        "fingerprint": "a",
        "version": 2,
        "options": {"reference_samples": 32, "seed": 0},
        "result": {"checkpoints": rows},
    }
    random = {
        **realised,
        "options": {**realised["options"], "designs": 5},
        "result": {
            "checkpoints": rows,
            "bands": [
                {"trials": r["trials"], "p95_radius": {"lower": 0.1, "upper": 0.9, "mean": 0.4}}
                for r in rows
            ],
        },
    }
    assert views._compatible_growth(realised, random)
    assert len(views._growth_figure(realised, random, "p95_radius", True).data) == 4
    assert len(views._growth_figure(realised, random, "p95_radius", False).data) == 1
    for changed in (
        {"fingerprint": "b"},
        {"run_id": "other"},
        {"options": {"reference_samples": 64, "seed": 0}},
        {"version": 1},
    ):
        mismatched = {**random, **changed}
        assert not views._compatible_growth(realised, mismatched)
        assert len(views._growth_figure(realised, mismatched, "p95_radius", True).data) == 1
    assert views._compatible_growth(None, random)


def test_matrix_is_symmetric_and_baseline_reverses_occupancy_direction():
    matrix = views._pair_matrix(
        {"pairs": [{"x": "x", "y": "y", "occupancy": 0.75}]}, "occupancy", ["x", "y"]
    )
    assert matrix.data[0].z[0][1] == matrix.data[0].z[1][0] == 0.75
    assert matrix.data[0].hoverongaps is False
    result = {
        "designs": 2,
        "seed": 0,
        "discrepancy_parameters": [],
        "metrics": {
            "pairwise_occupancy": {
                "observed": 0.8,
                "mean": 0.6,
                "lower": 0.5,
                "upper": 0.7,
                "percentile": 99,
                "standardized_deviation": 2,
            },
        },
    }
    detail = views._baseline_details(result)
    card = detail.children[2].children[0]
    assert card.children[2].figure.layout.xaxis.autorange == "reversed"


def test_sampler_metadata_is_displayed_without_guessing(run):
    assert views._sampler(run) == "Not recorded"
    run.meta = {"optuna_sampler": "RandomSampler"}
    assert views._sampler(run) == "Random"


def test_dash_registers_callbacks_and_layout_contains_no_global_process(monkeypatch, tmp_path, run):
    app = Dash(__name__, suppress_callback_exceptions=True)
    monkeypatch.setattr(deepcave, "app", app, raising=False)
    monkeypatch.setattr(deepcave, "queue", FakeQueue(), raising=False)
    monkeypatch.setattr(deepcave, "rc", SimpleNamespace(cache_dir=tmp_path), raising=False)
    monkeypatch.setattr(
        deepcave, "run_handler", SimpleNamespace(get_run=lambda _: run), raising=False
    )
    plugin = suite.SearchSpaceCoverage()
    suite.SearchSpaceCoverage.register_callbacks.__wrapped__(plugin)
    layout = suite.SearchSpaceCoverage.__call__.__wrapped__(plugin)
    ids = []
    labels = []
    components = []

    def visit(component):
        if isinstance(component, (list, tuple)):
            for child in component:
                visit(child)
        elif hasattr(component, "to_plotly_json"):
            components.append(component)
            if getattr(component, "id", None):
                ids.append(component.id)
            visit(getattr(component, "children", None))
        elif isinstance(component, str):
            labels.append(component)

    visit(layout)
    assert len(ids) == len(set(ids))
    assert sum(label == "Run analysis" for label in labels) == 6
    assert not any("update-button" in name for name in ids)
    tabs = next(c for c in components if getattr(c, "id", None) == suite._id("tabs"))
    assert [tab.label for tab in tabs.children] == [
        "Overview",
        "Univariate",
        "Pairwise",
        "Global",
        "Growth",
    ]
    assert tabs.active_tab == "overview"
    settings = next(c for c in components if getattr(c, "id", None) == suite._id("settings"))
    assert settings.is_open is False
    assert all(suite._id(f"{analysis}-cancel") in ids for analysis in suite.ANALYSES)
    assert len(app.callback_map) == 18
    assert {callback["callback"].__name__ for callback in app.callback_map.values()} == {
        "automatic",
        "selectors",
        "marginal",
        "scatter",
        "update",
        "settings",
        "coverage_status",
        "matrix",
        "global_detail",
        "show_detail",
        "global_results",
        "region_detail",
        "growth",
    }


def test_dash_http_updates_render_automatic_content_and_one_queued_analysis(
    monkeypatch, tmp_path, run
):
    app = Dash(__name__, suppress_callback_exceptions=True)
    queue = FakeQueue()
    monkeypatch.setattr(deepcave, "app", app, raising=False)
    monkeypatch.setattr(deepcave, "queue", queue, raising=False)
    monkeypatch.setattr(deepcave, "rc", SimpleNamespace(cache_dir=tmp_path), raising=False)
    monkeypatch.setattr(
        deepcave, "run_handler", SimpleNamespace(get_run=lambda _: run), raising=False
    )
    plugin = suite.SearchSpaceCoverage()
    suite.SearchSpaceCoverage.register_callbacks.__wrapped__(plugin)
    app.layout = html.Div(
        [
            dcc.Dropdown(id="global-run-selector", value=run.id),
            dcc.Store(id="general-cache-cleared-trigger", data=0),
            *suite.SearchSpaceCoverage.__call__.__wrapped__(plugin),
        ]
    )
    client = app.server.test_client()
    assert client.get("/_dash-layout").status_code == 200

    def post(first_output, values, changed):
        key, registered = next(
            (key, registered)
            for key, registered in app.callback_map.items()
            if isinstance(registered["output"], list)
            and registered["output"][0].component_id == first_output
        )
        outputs = [
            {"id": o.component_id, "property": o.component_property} for o in registered["output"]
        ]
        inputs = [
            {**i, "value": values.get((i["id"], i["property"]))} for i in registered["inputs"]
        ]
        states = [{**i, "value": values.get((i["id"], i["property"]))} for i in registered["state"]]
        response = client.post(
            "/_dash-update-component",
            json={
                "output": key,
                "outputs": outputs,
                "inputs": inputs,
                "state": states,
                "changedPropIds": [changed],
            },
        )
        assert response.status_code == 200, response.data.decode()
        return response.get_json()["response"]

    automatic = post(
        suite._id("overview"),
        {
            ("global-run-selector", "value"): run.id,
            (suite._id("tick"), "n_intervals"): 0,
            ("general-cache-cleared-trigger", "data"): 0,
        },
        "global-run-selector.value",
    )
    context = automatic[suite._id("context")]["data"]
    assert context["trials"] == 2
    assert not queue.calls
    controls = {
        (suite._id("context"), "data"): context,
        (suite._id("tick"), "n_intervals"): 0,
        (suite._id("designs"), "value"): 2,
        (suite._id("references"), "value"): 32,
        (suite._id("seed"), "value"): 0,
        ("global-run-selector", "value"): run.id,
    }
    selectors = post(suite._id("parameter"), controls, suite._id("context") + ".data")
    assert selectors[suite._id("x")]["value"] == "x"
    controls[(suite._id("pairwise-run"), "n_clicks")] = 1
    started = post(suite._id("pairwise-data"), controls, suite._id("pairwise-run") + ".n_clicks")
    assert started[suite._id("pairwise-run")]["disabled"] is True
    assert len(queue.calls) == 1
    queue.finish(next(iter(queue.jobs)))
    controls[(suite._id("pairwise-view"), "data")] = started[suite._id("pairwise-view")]["data"]
    completed = post(suite._id("pairwise-data"), controls, suite._id("tick") + ".n_intervals")
    assert completed[suite._id("pairwise-status")]["children"].startswith("✓ Computed")
    assert completed[suite._id("pairwise-run")]["disabled"] is False
    assert len(queue.calls) == 1
    controls[(suite._id("pairwise-data"), "data")] = completed[suite._id("pairwise-data")]["data"]
    controls[(suite._id("pair-metric"), "value")] = "occupancy"
    matrix = post(suite._id("pair-matrix"), controls, suite._id("pairwise-data") + ".data")
    assert matrix[suite._id("pair-matrix")]["figure"]["data"][0]["type"] == "heatmap"
