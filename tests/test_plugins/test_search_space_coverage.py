"""The coverage page never starts expensive work without a module button."""

from types import SimpleNamespace

import numpy as np
import pytest
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    EqualsCondition,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from dash import Dash, dcc, html
from dash.exceptions import PreventUpdate

import deepcave
from deepcave.evaluators.coverage import (
    compute_pairwise_coverage,
    compute_univariate_coverage,
    normalize_configurations,
    prepare_coverage,
)
from deepcave.plugins.hyperparameter import coverage_views as views
from deepcave.plugins.hyperparameter import search_space_coverage as suite
from deepcave.runs.status import Status
from deepcave.utils.run_caches import RunCaches


def visible_text(component):
    if isinstance(component, str):
        return component
    if isinstance(component, (list, tuple)):
        return " ".join(visible_text(child) for child in component)
    return visible_text(getattr(component, "children", []))


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


def test_combinatorial_settings_only_validate_consumed_controls():
    assert suite.analysis_options("combinatorial", None, None, None) == {}
    assert suite.analysis_options("combinatorial_baseline", 2, None, 7) == {"designs": 2, "seed": 7}
    with pytest.raises(ValueError, match="Reference designs"):
        suite.analysis_options("combinatorial_baseline", 0, None, 7)


def test_saved_baselines_without_medians_request_refresh_without_starting_jobs(
    callbacks, tmp_path, run
):
    registered, queue, event = callbacks
    data = prepare_coverage(run)
    result = suite.SearchSpaceCoverage.process(
        run, {"analysis": "combinatorial_baseline", "designs": 2}
    )
    result.pop("baseline_version")
    suite.CoverageJobs(tmp_path, queue)._write(
        run.id,
        "combinatorial_baseline",
        {
            "status": "completed",
            "result": result,
            "fingerprint": data.fingerprint,
            "version": suite.VERSION,
            "trial_count": data.count,
            "options": {"designs": 2, "seed": 0},
        },
    )
    updates = [callback for callback in registered if callback.__name__ == "update"]
    callback = dict(zip(suite.UI_ANALYSES, updates))["combinatorial_baseline"]
    event.triggered_id = suite._id("tick")
    output = callback(0, 0, 1, {"run_id": run.id}, 2, 32, 0, None, run.id)
    assert output[0]["status"] == "stale"
    assert output[1] == "Refresh to use the updated strength calculation"
    assert output[2] == "Recompute"
    assert not queue.calls


def test_combinatorial_profile_and_random_reference_are_view_only(callbacks, run):
    registered, queue, _ = callbacks
    callback = next(c for c in registered if c.__name__ == "combinatorial_results")
    data = prepare_coverage(run)
    context = {"run_id": run.id}
    observed = {
        "run_id": run.id,
        "fingerprint": data.fingerprint,
        "version": suite.VERSION,
        "trial_count": data.count,
        "result": suite.SearchSpaceCoverage.process(run, {"analysis": "combinatorial"}),
    }
    baseline = {
        **observed,
        "result": suite.SearchSpaceCoverage.process(
            run, {"analysis": "combinatorial_baseline", "designs": 2}
        ),
    }
    view = callback(context, baseline)
    text = visible_text(view)
    selector = view[0].children[0].children[1]
    assert "t = 1" in visible_text(selector.options[0]["label"])
    assert "Coverage at next strength" in text
    assert "Trial-count" not in text
    assert "Measured trial data" in text
    assert "Random comparison" not in text
    assert view[1] == "2 Parameters · 2 Trials"
    assert view[2] == "Random baseline · 2 samples"
    strength = selector.options[0]["label"].children[1].children[0]
    assert "Random strength:" in strength.title
    assert "Random strength:" not in visible_text(selector.options[0]["label"])
    assert "Selected parameter group" not in text
    assert "Random strength distribution" not in text
    _, mismatch = suite._combinatorial_snapshot(
        context, observed, {**baseline, "fingerprint": "older"}
    )
    assert mismatch is None
    other = callback({"run_id": "other"}, baseline)
    assert "Run combinatorial coverage" in visible_text(other)
    detail = next(c for c in registered if c.__name__ == "combinatorial_detail")
    assert "Select a resolution" in visible_text(detail([], context, baseline)[0])
    assert "Trial-count limit: t ≤ 1" in visible_text(detail([2], context, baseline)[0])
    assert not queue.calls


def test_strength_profile_explains_next_strength_and_ranks_missing_groups(callbacks, run):
    from deepcave.plugins.hyperparameter.combinatorial_views import combinatorial_view

    cell = {
        "strength": 2,
        "status": "computed",
        "coverage": 238 / 240,
        "covered": 238,
        "total": 240,
        "missing": 2,
        "complete_groups": 13,
        "groups": 15,
        "deficits": [
            {
                "parameters": ["a", "b"],
                "missing": 1,
                "covered": 15,
                "total": 16,
                "coverage": 15 / 16,
            },
            {
                "parameters": ["c", "d"],
                "missing": 2,
                "covered": 14,
                "total": 16,
                "coverage": 14 / 16,
            },
        ],
    }
    result = {
        "parameters": list("abcdef"),
        "trials": 80,
        "max_strength": 2,
        "rows": [
            {
                "resolution": 4,
                "observed_strength": 1,
                "strength_is_lower_bound": False,
                "trial_bound": 3,
                "levels": dict.fromkeys("abcdef", 4),
                "cells": [cell],
            }
        ],
    }
    profile = combinatorial_view(result)
    selector = profile.children[0].children[1]
    assert isinstance(selector, dcc.Checklist)
    assert selector.value == []
    label = visible_text(selector.options[0]["label"])
    assert all(
        text in label
        for text in ["4 levels", "t = 1 /6", "max. feasible strength = 3", "t = 2", "99.2%"]
    )
    value = selector.options[0]["label"].children[1].children[0]
    assert value.children[0].children == "t = 1"
    assert value.children[0].className == "coverage-strength-badge"
    assert value.children[1].children == "/6"
    assert value.children[1].className == "coverage-strength-total"
    assert "necessary, not sufficient" in selector.options[0]["label"].children[1].children[1].title
    assert "Random reference:" not in label
    advanced = profile.children[1]
    assert isinstance(advanced, html.Details)
    assert advanced.children[0].children == "Advanced Details"
    assert not getattr(advanced, "open", False)
    registered, queue, _ = callbacks
    detail = next(c for c in registered if c.__name__ == "combinatorial_detail")
    entry = {"run_id": run.id, "result": {"observed": result}}
    diagnosis = detail([4], {"run_id": run.id}, entry)[0]
    text = visible_text(diagnosis)
    assert "All combinations up to strength 1 are covered." in text
    assert "99.2% of combinations are covered." in text
    assert "are missing." not in text
    assert "Trial-count limit: t ≤ 3" in text
    table = diagnosis.children[-1].children[-1]
    assert [row["parameters"] for row in table.data] == ["c × d", "a × b"]
    assert "Select a resolution" in visible_text(detail([8], {"run_id": run.id}, entry)[0])
    assert not queue.calls


def test_coverage_array_rows_toggle_close_and_switch_without_starting_jobs(callbacks, run):
    registered, queue, _ = callbacks
    callback = next(c for c in registered if c.__name__ == "combinatorial_detail")
    context = {"run_id": run.id}
    entry = {
        "run_id": run.id,
        "result": {
            "observed": suite.SearchSpaceCoverage.process(run, {"analysis": "combinatorial"})
        },
    }
    opened, values, previous = callback([2], context, entry)
    assert "2 Levels — Detail" in visible_text(opened)
    assert values == [2]
    switched, values, previous = callback([2, 4], context, entry, previous)
    assert "4 Levels — Detail" in visible_text(switched)
    assert values == [4]
    # Normalizing the selected row and refreshing results keep the same detail open.
    refreshed, values, previous = callback(values, context, entry, previous)
    assert "4 Levels — Detail" in visible_text(refreshed)
    assert values == [4]
    closed, values, previous = callback([], context, entry, previous)
    assert "Select a resolution" in visible_text(closed)
    assert values == []
    reopened, values, previous = callback([4], context, entry, previous)
    assert "4 Levels — Detail" in visible_text(reopened)
    other_run, values, _ = callback(
        values, {"run_id": "other"}, {**entry, "run_id": "other"}, previous
    )
    assert "Select a resolution" in visible_text(other_run)
    assert values == []
    assert not queue.calls


def test_shared_coverage_refresh_updates_measurement_and_baseline_in_one_job(
    callbacks, tmp_path, run
):
    registered, queue, event = callbacks
    jobs = suite.CoverageJobs(tmp_path, queue)
    history = run.history
    run.history = history[:1]
    old_result = suite.SearchSpaceCoverage.process(run, {"analysis": "combinatorial"})
    run.history = history
    jobs._write(run.id, "combinatorial", {"result": old_result})
    context = {"run_id": run.id}
    render = next(c for c in registered if c.__name__ == "combinatorial_results")
    assert render(context, None)[1] == "2 Parameters · 1 Trials"
    updates = [c for c in registered if c.__name__ == "update"]
    refresh = dict(zip(suite.UI_ANALYSES, updates))["combinatorial_baseline"]
    event.triggered_id = suite._id("combinatorial_baseline-run")
    started = refresh(1, 0, 1, context, 2, 32, 0, None, run.id)
    assert started[3] is True
    assert len(queue.calls) == 1
    assert queue.calls[0]["analysis"] == "combinatorial_baseline"
    assert render(context, started[0])[1] == "2 Parameters · 1 Trials"
    queue.finish(started[4]["job"]["id"])
    event.triggered_id = suite._id("tick")
    completed = refresh(1, 0, 2, context, 2, 32, 0, started[4], run.id)
    assert completed[3] is False
    assert completed[2] == "Refresh"
    assert completed[0]["result"]["observed"]["trials"] == 2
    assert completed[0]["result"]["designs"] == 2
    result, counts, label = render(context, completed[0])
    assert counts == "2 Parameters · 2 Trials"
    assert label == "Random baseline · 2 samples"
    assert "Measured trial data" in visible_text(result)
    assert len(queue.calls) == 1


def test_strength_profile_distinguishes_complete_uncomputed_and_trial_limits(run):
    from deepcave.plugins.hyperparameter.combinatorial_views import (
        combinatorial_resolution_detail,
        combinatorial_view,
    )

    result = suite.SearchSpaceCoverage.process(run, {"analysis": "combinatorial"})
    result["rows"] = [result["rows"][0]]
    text = visible_text(combinatorial_resolution_detail(result, 2))
    assert "Full coverage at strength 2 requires more trials." in text
    row = result["rows"][0]
    row["cells"][1]["status"] = "work_limit"
    row["strength_is_lower_bound"] = True
    view = combinatorial_view(result)
    label = visible_text(view.children[0].children[1].options[0]["label"])
    assert "t ≥ 1" in label and "…" in label
    assert "Coverage at strength 2 has not been computed." in visible_text(
        combinatorial_resolution_detail(result, 2)
    )
    row["observed_strength"] = 2
    row["strength_is_lower_bound"] = False
    label = visible_text(combinatorial_view(result).children[0].children[1].options[0]["label"])
    assert "All strengths covered" in label


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
    assert suite.analysis_options("pairwise", None, 32, 7) == {"reference_samples": 32, "seed": 7}
    assert suite.analysis_options("discrepancy", None, 32, 7) == {
        "reference_samples": 32,
        "seed": 7,
    }
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
    assert len(analysis_callbacks) == len(suite.UI_ANALYSES)
    for analysis, callback in zip(suite.UI_ANALYSES, analysis_callbacks):
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
            assert output[1] == (
                "" if analysis in ("pairwise", "combinatorial_baseline") else "Not computed"
            )
            assert not queue.calls
    for index, (analysis, callback) in enumerate(zip(suite.UI_ANALYSES, analysis_callbacks)):
        event.triggered_id = suite._id(f"{analysis}-run")
        output = callback(1, 0, 1, context, 2, 32, 0, None, run.id)
        assert output[3] is True
        assert queue.calls[-1]["analysis"] == analysis
        assert len(queue.calls) == index + 1


def test_callback_rejects_stale_run_context(callbacks, run):
    registered, queue, event = callbacks
    event.triggered_id = suite._id("pairwise-run")
    with pytest.raises(PreventUpdate):
        next(c for c in registered if c.__name__ == "update")(
            1, 0, 1, {"run_id": run.id}, 2, 32, 0, None, "other-run"
        )
    assert not queue.calls


def test_ranking_selection_uses_row_identity_after_sorting(callbacks, run):
    registered, _, event = callbacks
    event.triggered_id = suite._id("pair-ranking")
    rows = [{"id": "x × y", "x": "x", "y": "y"}, {"id": "y × x", "x": "y", "y": "x"}]
    selectors = next(c for c in registered if c.__name__ == "selectors")
    outputs = selectors({"run_id": run.id}, {"row": 0, "row_id": "y × x"}, None, rows, "x", "y")
    assert outputs[1] == "y" and outputs[3] == "x"
    assert outputs[4] is None


def test_matrix_selection_updates_pair_but_ignores_diagonal_and_missing_metrics(callbacks, run):
    registered, queue, event = callbacks
    event.triggered_id = suite._id("pair-matrix")
    selectors = next(c for c in registered if c.__name__ == "selectors")
    context = {"run_id": run.id}
    changed = selectors(context, None, {"points": [{"x": "y", "y": "x", "z": 0.5}]}, [], "x", "y")
    assert changed[1] == "y" and changed[3] == "x"
    assert changed[5] is None
    for point in ({"x": "x", "y": "x", "z": 1}, {"x": "y", "y": "x", "z": None}):
        same = selectors(context, None, {"points": [point]}, [], "x", "y")
        assert same[1] == "x" and same[3] == "y"
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


def test_univariate_switches_all_continuous_plots_without_queuing(callbacks, run):
    registered, queue, _ = callbacks
    callback = next(c for c in registered if c.__name__ == "marginal")
    content, options, mode = callback({"run_id": run.id}, "ecdf", "type", "max_gap", False)
    assert options[1]["disabled"] is False and mode == "ecdf"
    group = content.children[0]
    assert isinstance(group, html.Section)
    assert group.children[0].children == "Continuous (2)"
    for row in group.children[1].children:
        graph = row.children[0].children[1]
        assert graph.figure.data[0].line.shape == "hv"
        assert [trace.name for trace in graph.figure.data] == [
            "Observed ECDF",
            "Expected CDF",
            "Observed trials",
            "Largest gap",
        ]
    assert not queue.calls


def test_univariate_resets_ecdf_when_switching_to_a_categorical_run(callbacks, run):
    registered, queue, _ = callbacks
    run.configspace = ConfigurationSpace()
    run.configspace.add(CategoricalHyperparameter("model", ["a", "b"]))
    run.get_config = lambda index: {"model": "a" if index == 0 else "b"}
    callback = next(c for c in registered if c.__name__ == "marginal")
    content, options, mode = callback({"run_id": run.id}, "ecdf", "type", "name", False)
    assert mode == "distribution" and options[1]["disabled"] is True
    assert (
        content.children[0].children[1].children[0].children[0].children[1].figure.data[0].type
        == "bar"
    )
    assert not queue.calls


def test_univariate_integer_only_run_keeps_ecdf_enabled(callbacks, run):
    registered, queue, _ = callbacks
    run.configspace = ConfigurationSpace()
    run.configspace.add(UniformIntegerHyperparameter("depth", 13, 17))
    run.get_config = lambda index: {"depth": 13 if index == 0 else 17}
    callback = next(c for c in registered if c.__name__ == "marginal")
    content, options, mode = callback({"run_id": run.id}, "ecdf", "type", "ks", False)
    assert mode == "ecdf" and options[1]["disabled"] is False
    row = content.children[0].children[1].children[0]
    preview = row.children[0].children[1]
    assert preview.figure.layout.xaxis.type == "linear"
    observed, expected = preview.figure.data
    assert observed.line.shape == expected.line.shape == "hv"
    assert list(expected.x) == [13, 13, 14, 15, 16, 17]
    assert list(expected.y) == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1])
    assert list(observed.y) == pytest.approx([0, 0.5, 0.5, 0.5, 0.5, 1])
    summary_metrics = row.children[0].children[2].children
    assert [metric.children[0].children for metric in summary_metrics] == ["Values", "KS"]
    assert summary_metrics[1].children[1].children == "0.3"
    detail = row.children[1].children
    graph = detail.children[0].children[1].children
    assert graph.figure.layout.xaxis.title.text == "Value"
    assert graph.figure.data[0].line.shape == graph.figure.data[1].line.shape == "hv"
    assert list(graph.figure.data[1].x) == list(expected.x)
    metrics = detail.children[0].children[0].children
    visited = next(metric for metric in metrics if metric.children[0].children == "Values visited")
    assert visited.children[1].children == "2 / 5"
    assert all(metric.children[0].children != "Bins occupied" for metric in metrics)
    assert all(metric.children[0].children != "Largest gap" for metric in metrics)
    assert all(metric.children[0].children != "Edge gaps" for metric in metrics)
    ks = next(metric for metric in metrics if metric.children[0].children == "KS distance")
    assert ks.children[1].children == "0.3"
    assert ks.children[2].children == "Descriptive; no p-value"
    assert not queue.calls


def test_univariate_mixed_types_keep_groups_visible_and_discrete_distributions(run):
    run.configspace = ConfigurationSpace()
    run.configspace.add(
        [
            UniformFloatHyperparameter("rate", 0.001, 0.1, log=True),
            UniformIntegerHyperparameter("depth", 1, 2),
            CategoricalHyperparameter("model", ["a", "b"]),
        ]
    )
    configs = [
        {"rate": 0.001, "depth": 1, "model": "a"},
        {"rate": 0.1, "depth": 2, "model": "b"},
    ]
    run.get_config = lambda index: configs[index]
    rows = compute_univariate_coverage(prepare_coverage(run))
    content = views._univariate_rows(rows, "ecdf", "type", "name", False)
    assert [group.children[0].children for group in content.children] == [
        "Continuous (1)",
        "Integer (1)",
        "Categorical (1)",
    ]
    assert all(isinstance(group, html.Section) for group in content.children)
    for group in content.children[1:]:
        row = group.children[1].children[0]
        assert isinstance(row, html.Details)
        summary = row.children[0]
        integer = group.children[0].children.startswith("Integer")
        assert summary.children[1].figure.data[0].type == ("scatter" if integer else "bar")
        metrics = summary.children[2].children
        assert metrics[-1].children[1].children == ("0" if integer else "—")
        assert [metric.children[0].children for metric in metrics] == (
            ["Values", "KS"] if integer else ["Values", "Largest gap", "KS"]
        )
        assert metrics[0].children[1].children == "2 / 2"
    continuous = content.children[0].children[1].children[0].children[0].children[1]
    metrics = content.children[0].children[1].children[0].children[0].children[2].children
    assert [metric.children[0].children for metric in metrics] == ["Bins", "Largest gap", "KS"]
    assert continuous.figure.layout.xaxis.range == (0, 1)
    assert continuous.figure.layout.xaxis.ticktext is None
    distribution = views._univariate_preview(
        next(row for row in rows if row["name"] == "rate"), "distribution"
    )
    assert distribution.figure.layout.xaxis.ticktext[0] == "0.001"
    assert distribution.figure.layout.xaxis.ticktext[-1] == "0.1"
    filtered = views._univariate_rows(rows, "distribution", "type", "name", True)
    assert [group.children[0].children for group in filtered.children] == ["Continuous (1)"]


@pytest.mark.parametrize("mode", ["distribution", "ecdf"])
@pytest.mark.parametrize("detail", [False, True])
def test_gap_plot_marks_trials_and_preserves_empty_bins(mode, detail):
    space = ConfigurationSpace(space={"x": (0.0, 1.0)})
    values = [0.1, 0.2, 0.2, 0.8, 0.9]
    (row,) = compute_univariate_coverage(
        normalize_configurations(space, [{"x": value} for value in values]), bins=10
    )
    graph = (
        views._univariate(row, mode).children[0].children[1].children
        if detail
        else views._univariate_preview(row, mode)
    )
    figure = graph.figure
    rug = next(trace for trace in figure.data if trace.name == "Observed trials")
    assert list(rug.x) == values
    assert list(rug.marker.color) == ["#a39bc4", "#f4511e", "#f4511e", "#f4511e", "#a39bc4"]
    assert rug.cliponaxis is False
    assert set(rug.y) == {0.5}
    assert figure.layout.yaxis2.overlaying is None
    assert figure.layout.yaxis2.domain[1] < figure.layout.yaxis.domain[0]
    verticals = [s for s in figure.layout.shapes if s.type == "line" and s.x0 == s.x1]
    assert [s.x0 for s in verticals] == [0.2, 0.8]
    assert all(s.line.dash == "dash" for s in verticals)
    bracket = next(s for s in figure.layout.shapes if s.type == "line" and s.x0 != s.x1)
    assert [bracket.x0, bracket.x1] == [0.2, 0.8]
    assert bracket.y0 == bracket.y1 > 1
    assert figure.layout.annotations[0].text == "<b>Largest gap</b>"
    expected = figure.data[1]
    assert expected.mode == "lines" and expected.line.dash == "dash"
    assert expected.line.color == "#94a3b8"
    if mode == "distribution":
        assert list(figure.data[0].y) == row["frequencies"]
        assert sum(figure.data[0].y) == len(values)
        assert all(n == 0 for x, n in zip(figure.data[0].x, figure.data[0].y) if 0.3 < x < 0.8)
    hover = next(trace for trace in figure.data if trace.name == "Largest gap")
    assert "Lower boundary: 0.2 (observed trial)" in hover.text
    assert "60.00% of transformed search space" in hover.text


def test_gap_inside_occupied_bin_keeps_bar_and_highlights_exact_trials():
    space = ConfigurationSpace(space={"x": (0.0, 1.0)})
    values = [x for x in np.linspace(0, 1, 51) if not 0.4 < x < 0.46] + [0.411, 0.449]
    (row,) = compute_univariate_coverage(
        normalize_configurations(space, [{"x": value} for value in values]), bins=20
    )
    figure = views._univariate_preview(row, "distribution").figure
    gap = row["largest_gap"]
    assert [gap["lower"], gap["upper"]] == pytest.approx([0.411, 0.449])
    assert int(gap["lower"] * 20) == int(gap["upper"] * 20) == 8
    assert figure.data[0].y[8] == 3
    rug = figure.data[2]
    assert [x for x, color in zip(rug.x, rug.marker.color) if color == "#f4511e"] == [0.411, 0.449]


@pytest.mark.parametrize("values, edge", [([0.7, 0.8], 0), ([0.2, 0.3], 1), ([0.5], 0)])
def test_edge_gap_identifies_search_space_boundary_without_fabricating_trial(values, edge):
    space = ConfigurationSpace(space={"x": (0.0, 1.0)})
    (row,) = compute_univariate_coverage(
        normalize_configurations(space, [{"x": value} for value in values])
    )
    figure = views._univariate_preview(row, "distribution").figure
    rug = figure.data[2]
    assert edge not in rug.x
    assert list(rug.marker.color).count("#f4511e") == 1
    edge_marker = next(s for s in figure.layout.shapes if s.type == "line" and s.x0 == s.x1 == edge)
    assert edge_marker.line.dash == "dash"
    assert any(a.text == "Search-space boundary" and a.x == edge for a in figure.layout.annotations)
    assert "search-space boundary" in figure.data[3].text


def test_unobserved_parameter_has_no_gap_bracket_or_invented_rug_ticks():
    space = ConfigurationSpace(space={"x": (0.0, 1.0)})
    (row,) = compute_univariate_coverage(normalize_configurations(space, []))
    figure = views._univariate_preview(row, "distribution").figure
    assert not figure.layout.shapes and not figure.layout.annotations
    assert list(figure.data[2].x) == []


def test_univariate_sorting_filtering_and_empty_results(run):
    rows = compute_univariate_coverage(prepare_coverage(run))
    rows[0].update(max_gap=0.02, coverage=1.0, ks=0.01)
    rows[1].update(max_gap=0.2, coverage=0.5, ks=0.2)
    for sort_by in ("max_gap", "coverage", "ks"):
        content = views._univariate_rows(rows, "distribution", "none", sort_by, False)
        items = content.children[0].children[1].children
        assert [item.children[0].children[0].children[0].children for item in items] == ["y", "x"]
    filtered = views._univariate_rows(rows, "distribution", "none", "name", True)
    assert len(filtered.children[0].children[1].children) == 1
    empty = views._univariate_rows(rows[:1], "distribution", "type", "name", True)
    assert isinstance(empty, html.P)
    assert empty.children == "No parameters match the selected filter."


def test_global_details_and_settings_are_view_only(callbacks, run):
    registered, queue, event = callbacks
    by_name = {c.__name__: c for c in registered}
    event.triggered_id = suite._id("radius-details")
    selected = by_name["global_detail"](1, {"run_id": run.id}, None)
    assert selected["analysis"] == "radius"
    assert by_name["show_detail"](selected) == {}
    assert by_name["settings"](1, False) is True
    assert by_name["settings"](2, True) is False
    assert not queue.calls


@pytest.mark.parametrize(
    "metric,percentile", [("mixed_mmd", 31), ("centered", 31), ("l2_star", 70), ("wrap_around", 4)]
)
def test_legacy_distribution_views_render_comparisons_and_bands(callbacks, run, metric, percentile):
    registered, queue, _ = callbacks
    data = prepare_coverage(run)
    context = {"run_id": run.id, "fingerprint": data.fingerprint}
    options = {"designs": 3, "reference_samples": 32, "seed": 7}
    snapshot = {
        "run_id": run.id,
        "fingerprint": data.fingerprint,
        "version": suite.VERSION,
        "trial_count": data.count,
    }
    discrepancy = {
        **snapshot,
        "options": {"reference_samples": 32, "seed": 7},
        "result": suite.run_coverage_analysis("discrepancy", data, run.configspace, options),
    }
    radius = {
        **snapshot,
        "options": {"reference_samples": 32, "seed": 7},
        "result": suite.run_coverage_analysis("radius", data, run.configspace, options),
    }
    baseline = {
        **snapshot,
        "options": options,
        "result": suite.run_coverage_analysis("baseline", data, run.configspace, options),
    }
    key = "centered_discrepancy" if metric == "centered" else metric
    baseline["result"]["metrics"][key]["percentile"] = percentile
    baseline["result"]["discrepancy_bands"][-1][metric]["percentile"] = percentile
    baseline["result"]["metrics"]["p95_radius"]["percentile"] = 20
    baseline["result"]["metrics"]["max_radius"]["percentile"] = 90
    callback = next(c for c in registered if c.__name__ == "global_results")
    detail = views._discrepancy_details(discrepancy["result"], baseline["result"], metric)
    summary = views._module_summary(
        "discrepancy", discrepancy["result"], baseline["result"], metric
    )
    radius_summary = views._module_summary("radius", radius["result"], baseline["result"])
    radius_detail, observed_radius_summary, regions = callback(context, radius)
    assert visible_text(radius_detail) == visible_text(views._radius_details(radius["result"]))
    assert "percentile vs random" not in visible_text(observed_radius_summary)
    assert summary.children[0].children[0].children[0] == (
        "2 parameters" if metric == "mixed_mmd" else "2 continuous parameters"
    )
    assert "x" in summary.children[0].children[1].children
    assert "y" in summary.children[0].children[1].children
    assert summary.children[1].children[1].children == (
        ("The observed distribution" if metric == "mixed_mmd" else "The observed uniformity")
        + " is within the usual range for 2 random trials."
        if 5 <= percentile <= 95
        else "More uniform than typical random designs of 2 trials."
    )
    text = visible_text(summary)
    assert "better" not in text.lower()
    assert views._number(discrepancy["result"][metric]) not in text
    assert "percentile" not in text.lower()
    assert "More uniform than about" not in text
    assert (
        "Closer to the reference distribution than about "
        if metric == "mixed_mmd"
        else "More uniform than about "
    ) + f"{100 - percentile}%" in visible_text(detail)
    assert summary.children[3].children == "2 trials · 3 random reference designs"
    assert radius_summary.children[0].children[1].children[0].children == (
        "80th percentile vs random"
    )
    assert radius_summary.children[1].children[1].children[0].children == (
        "10th percentile vs random"
    )
    relative = summary.children[2].figure
    assert relative.layout.height == 170
    assert list(relative.layout.yaxis.range) == [0, 100]
    assert relative.data[0].y[-1] == 100 - percentile
    assert list(relative.data[0].y) == [
        100 - row[metric]["percentile"] for row in baseline["result"]["discrepancy_bands"]
    ]
    assert any(shape.y0 == shape.y1 == 50 for shape in relative.layout.shapes)
    assert {shape.y0 for shape in relative.layout.shapes} == {5, 50, 95}
    assert all(shape.type == "line" for shape in relative.layout.shapes)
    assert not relative.layout.annotations
    assert list(relative.layout.yaxis.ticktext) == (
        ["Further from reference", "Random median", "Closer to reference"]
        if metric == "mixed_mmd"
        else ["Less uniform", "Random median", "More uniform"]
    )
    assert "better" not in relative.layout.title.text
    raw = detail.children[1].children
    assert raw[0].children[0].children == (
        "MMD distance"
        if metric == "mixed_mmd"
        else views.DISCREPANCY_METRICS[metric] + " discrepancy"
    )
    assert raw[0].children[1].children == views._number(discrepancy["result"][metric])
    assert raw[1].children[1].children == views._number(
        baseline["result"]["metrics"][key]["median"]
    )
    figure = next(child.figure for child in detail.children if isinstance(child, dcc.Graph))
    assert len(figure.data) == 4
    assert list(figure.data[2].y) == [
        row[metric]["median"] for row in baseline["result"]["discrepancy_bands"]
    ]
    assert figure.data[2].name == "Random median"
    assert figure.data[2].line.dash == "dash"
    assert figure.data[-1].y[-1] == discrepancy["result"][metric]
    assert figure.layout.yaxis.type == ("linear" if metric == "mixed_mmd" else "log")
    assert regions and not queue.calls
    other_run = callback({"run_id": "other"}, radius)
    assert other_run[-1] == []


@pytest.mark.parametrize(
    "percentile,status",
    [
        (0, "Outside the usual random range"),
        (4, "Outside the usual random range"),
        (5, "Consistent with random sampling"),
        (31, "Consistent with random sampling"),
        (50, "Consistent with random sampling"),
        (95, "Consistent with random sampling"),
        (96, "Outside the usual random range"),
        (100, "Outside the usual random range"),
    ],
)
def test_uniformity_comparison_leads_with_status_and_avoids_quality_judgements(percentile, status):
    comparison = views._global_comparison(
        {
            "designs": 500,
            "metrics": {
                "centered_discrepancy": {
                    "percentile": percentile,
                    "median": 0.0121,
                    "lower": 0.005,
                    "upper": 0.02,
                }
            },
        },
        "centered_discrepancy",
        uniformity=True,
        trials=80,
    )
    assert [child.children for child in comparison.children] == [
        status,
        (
            "The observed uniformity is within the usual range for 80 random trials."
            if 5 <= percentile <= 95
            else ("More" if percentile < 5 else "Less")
            + " uniform than typical random designs of 80 trials."
        ),
    ]
    assert "better" not in comparison.title
    assert "Ties count half" in comparison.title
    assert "Metric percentile" not in comparison.title


@pytest.mark.parametrize(
    "parameters,label",
    [
        (["learning_rate"], "1 continuous parameter"),
        ([], "No eligible continuous parameters"),
        (None, "Continuous parameters"),
    ],
)
def test_uniformity_scope_is_accessible_without_inventing_legacy_parameter_counts(
    parameters, label
):
    result = {"parameters": parameters} if parameters is not None else {}
    scope = views._uniformity_scope(result)
    assert isinstance(scope, html.Details)
    assert isinstance(scope.children[0], html.Summary)
    assert scope.children[0].children[0] == label
    if parameters:
        assert parameters[0] in scope.children[1].children
    details = views._discrepancy_details({**result, "centered": 0.01})
    assert "0.01" in visible_text(details)
    unavailable = views._module_summary(
        "discrepancy", {**result, "reason": "No continuous projection."}
    )
    assert unavailable.children[1].children == "No continuous projection."
    assert unavailable.children[2] is None


def test_relative_discrepancy_preserves_missing_ranks_and_rejects_mismatched_prefixes():
    result = {"checkpoints": [{"trials": 1}, {"trials": 2}]}
    baseline = {
        "discrepancy_bands": [
            {"trials": 1, "centered": {"percentile": None}},
            {"trials": 2, "centered": {"percentile": 31}},
        ]
    }
    figure = views._relative_discrepancy_figure(result, baseline, "centered")
    assert list(figure.data[0].y) == [None, 69]
    assert list(figure.data[0].customdata) == [
        "Reference unavailable",
        "Within the usual random range",
    ]
    baseline["discrepancy_bands"][1]["trials"] = 3
    mismatch = views._relative_discrepancy_figure(result, baseline, "centered")
    assert not mismatch.data
    assert "Recompute" in mismatch.layout.annotations[0].text


@pytest.mark.parametrize(
    "changed",
    [
        {"run_id": "other"},
        {"fingerprint": "new"},
        {"version": -1},
        {"trial_count": 9},
        {"options": {"reference_samples": 64, "seed": 7}},
        {"options": {"reference_samples": 32, "seed": 8}},
    ],
)
def test_global_comparisons_require_matching_snapshots_and_radius_references(changed):
    entry = {
        "run_id": "run",
        "fingerprint": "snapshot",
        "version": suite.VERSION,
        "trial_count": 2,
        "options": {"reference_samples": 32, "seed": 7},
        "result": {"p95": 0.2},
    }
    baseline = {**entry, "result": {"discrepancy_bands": []}, **changed}
    assert suite._matching_global_baseline(entry, baseline, radius=True) is None
    if "options" in changed:
        assert suite._matching_global_baseline(entry, baseline) == baseline["result"]
    else:
        assert suite._matching_global_baseline(entry, baseline) is None


def test_full_space_comparison_requires_identical_reference_settings_and_current_metric():
    entry = {
        "run_id": "run",
        "fingerprint": "same",
        "version": suite.VERSION,
        "trial_count": 80,
        "options": {"reference_samples": 32, "seed": 7},
        "result": {"mixed_mmd": 0.1, "mixed_version": suite.MIXED_MMD_VERSION},
    }
    baseline = {
        **entry,
        "result": {
            "discrepancy_bands": [],
            "mixed_version": suite.MIXED_MMD_VERSION,
        },
    }
    assert suite._matching_global_baseline(entry, baseline, mixed=True) == baseline["result"]
    for changed in (
        {"options": {"reference_samples": 64, "seed": 7}},
        {"options": {"reference_samples": 32, "seed": 8}},
        {"result": {"discrepancy_bands": []}},
    ):
        assert suite._matching_global_baseline(entry, {**baseline, **changed}, mixed=True) is None


def test_full_space_card_and_details_work_without_continuous_parameters():
    space = ConfigurationSpace(space={"choice": ["a", "b"], "integer": (0, 1)})
    data = normalize_configurations(space, [{"choice": "a", "integer": 0}] * 4)
    options = {"reference_samples": 32, "seed": 7, "designs": 3}
    result = suite.run_coverage_analysis("discrepancy", data, space, options)
    baseline = suite.run_coverage_analysis("baseline", data, space, options)
    baseline["metrics"]["mixed_mmd"]["percentile"] = 50
    summary = views._module_summary("discrepancy", result, baseline, "mixed_mmd")
    assert summary.children[0].children[0].children[0] == "2 parameters"
    assert "Consistent with random sampling" in visible_text(summary)
    assert "No unconditional continuous parameters" not in visible_text(summary)
    assert "categorical" in visible_text(summary.children[0])
    assert "integer" in visible_text(summary.children[0])
    detail = views._discrepancy_details(result, baseline, "mixed_mmd")
    assert "MMD distance" in visible_text(detail)
    assert "Included parameters: choice, integer" in visible_text(detail)


def test_global_discrepancy_without_references_shows_only_observed_trajectory():
    result = {"checkpoints": [{"trials": 1, "centered": 0.1}, {"trials": 2, "centered": 0.2}]}
    summary = views._module_summary("discrepancy", {**result, "centered": 0.2})
    assert summary.children[2] is None
    assert "Run or recompute random comparisons" in summary.children[1].children
    figure = views._discrepancy_figure(result, None, "centered")
    assert len(figure.data) == 1
    assert list(figure.data[0].y) == [
        0.1,
        0.2,
    ]  # Preserve increases rather than smoothing them away.
    legacy = views._discrepancy_figure({}, None, "centered")
    assert "Recompute" in legacy.layout.annotations[0].text


def test_cancel_still_works_when_settings_are_invalid(callbacks, run):
    registered, queue, event = callbacks
    updates = [c for c in registered if c.__name__ == "update"]
    callback = dict(zip(suite.UI_ANALYSES, updates))["radius"]
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


def test_every_pairwise_metric_includes_continuous_integer_and_categorical_axes(callbacks, run):
    registered, queue, _ = callbacks
    run.configspace = ConfigurationSpace()
    run.configspace.add(
        [
            UniformFloatHyperparameter("rate", 0, 1),
            UniformIntegerHyperparameter("depth", 13, 17),
            CategoricalHyperparameter("kind", ["a", "b"]),
        ]
    )
    configs = [{"rate": 0.25, "depth": 13, "kind": "a"}, {"rate": 0.75, "depth": 17, "kind": "b"}]
    run.get_config = lambda index: configs[index]
    data = prepare_coverage(run)
    entry = {"run_id": run.id, "version": suite.VERSION, "result": compute_pairwise_coverage(data)}
    callback = next(callback for callback in registered if callback.__name__ == "matrix")
    for metric in views.PAIR_METRICS:
        figure, rows, _ = callback({"run_id": run.id}, entry, metric)
        assert set(figure.data[0].x) == {"rate", "depth", "kind"}
        assert all(
            value >= 0
            for i, row in enumerate(figure.data[0].z)
            for j, value in enumerate(row)
            if i != j
        )
        assert len(rows) == 3
    assert not queue.calls


def test_old_pairwise_metrics_are_not_relabelled_as_mmd(run):
    old = {"run_id": run.id, "version": suite.VERSION - 1, "result": {"pairs": []}}
    assert suite._pair_result(old, {"run_id": run.id}) is None
    current = {**old, "version": suite.VERSION}
    assert suite._pair_result(current, {"run_id": run.id}) == current["result"]
    assert suite._pair_result(current, {"run_id": "another-run"}) is None


def test_scatter_shades_the_actual_empty_coverage_cells(run):
    data = prepare_coverage(run)
    result = compute_pairwise_coverage(data, bins=2, references=data)
    figure = views._scatter(data, "x", "y", result).children[1].children[0].figure
    empty = [shape for shape in figure.layout.shapes if shape.type == "rect"]
    assert (
        len(empty) == result["pairs"][0]["total_cells"] - result["pairs"][0]["occupied_cells"] == 2
    )
    assert {(shape.x0, shape.x1, shape.y0, shape.y1) for shape in empty} == {
        (0, 0.5, 0, 0.5),
        (0.5, 1, 0.5, 1),
    }
    assert all(shape.layer == "below" for shape in figure.layout.shapes)
    assert figure.layout.xaxis.range == figure.layout.yaxis.range == (0, 1)
    assert figure.layout.xaxis.showgrid is False
    assert figure.layout.yaxis.showgrid is False
    assert list(figure.data[0].x) == [0.25, 0.75]
    assert figure.data[0].hoverinfo == "none"
    assert figure.data[0].hovertemplate is None


def test_highlight_empty_cells_toggle_changes_only_shading_without_starting_jobs(callbacks, run):
    registered, queue, _ = callbacks
    callback = next(callback for callback in registered if callback.__name__ == "scatter")
    enabled = callback({"run_id": run.id}, "x", "y", None, True).children[1].children[0].figure
    disabled = callback({"run_id": run.id}, "x", "y", None, False).children[1].children[0].figure
    assert any(shape.type == "rect" for shape in enabled.layout.shapes)
    assert all(shape.type == "line" for shape in disabled.layout.shapes)
    assert [shape.to_plotly_json() for shape in enabled.layout.shapes if shape.type == "line"] == [
        shape.to_plotly_json() for shape in disabled.layout.shapes
    ]
    assert enabled.data[0].to_plotly_json() == disabled.data[0].to_plotly_json()
    assert enabled.layout.height == disabled.layout.height
    assert not queue.calls


@pytest.mark.parametrize("optimize", ["lower", "upper"])
@pytest.mark.parametrize("proportion", [0.2, 0.1, 0.05, 0.02, 0.01])
def test_elite_trials_rank_successful_history_rows_by_objective(run, optimize, proportion):
    objective = SimpleNamespace(name="Score", optimize=optimize)
    run.get_objectives = lambda: [objective, SimpleNamespace(name="Time")]
    run.get_objective_id = lambda _: 0
    run.history = [
        SimpleNamespace(status=Status.SUCCESS, costs=[float(index)]) for index in range(100)
    ]
    run.history.extend(
        [
            SimpleNamespace(
                status=Status.FAILED, costs=[-1000.0 if optimize == "lower" else 1000.0]
            ),
            SimpleNamespace(status=Status.SUCCESS, costs=[None]),
            SimpleNamespace(status=Status.SUCCESS, costs=[np.nan]),
        ]
    )
    indices, label = suite._elite_trials(run, proportion)
    count = int(100 * proportion)
    expected = list(range(count)) if optimize == "lower" else list(range(100 - count, 100))
    assert indices == expected
    assert label == f"Elite: best {100 * proportion:g}% (Score)"


def test_elite_trial_quantile_includes_ties_and_handles_missing_objectives(run):
    objective = SimpleNamespace(name="Loss", optimize="lower")
    run.get_objectives = lambda: [objective]
    run.get_objective_id = lambda _: 0
    run.history = [SimpleNamespace(status=Status.SUCCESS, costs=[value]) for value in (1, 1, 2, 3)]
    assert suite._elite_trials(run, 0.1)[0] == [0, 1]
    run.history = [SimpleNamespace(status=Status.FAILED, costs=[0.0])]
    assert suite._elite_trials(run, 0.1)[0] == []
    run.get_objectives = lambda: [SimpleNamespace(name="Time")]
    assert suite._elite_trials(run, 0.1)[0] == []


def test_elite_highlight_changes_only_scatter_styling_without_starting_jobs(callbacks, run):
    registered, queue, _ = callbacks
    objective = SimpleNamespace(name="Loss", optimize="lower")
    run.get_objectives = lambda: [objective]
    run.get_objective_id = lambda _: 0
    for trial, cost in zip(run.history, (0.5, 0.2)):
        trial.costs = [cost]
    callback = next(callback for callback in registered if callback.__name__ == "scatter")
    disabled = callback({"run_id": run.id}, "x", "y", None, True, False, 0.1)
    enabled = callback({"run_id": run.id}, "x", "y", None, True, True, 0.1)
    base = disabled.children[1].children[0].figure
    highlighted = enabled.children[1].children[0].figure
    assert len(base.data) == 1 and len(highlighted.data) == 2
    assert list(highlighted.data[1].x) == [0.75]
    assert list(highlighted.data[1].y) == [0.25]
    assert highlighted.data[1].marker.symbol == "diamond"
    assert all(trace.hoverinfo == "none" for trace in highlighted.data)
    assert list(highlighted.data[0].x) == list(base.data[0].x)
    assert [shape.to_plotly_json() for shape in highlighted.layout.shapes] == [
        shape.to_plotly_json() for shape in base.layout.shapes
    ]
    assert "best 10% (Loss)" in enabled.children[0].children
    assert not queue.calls


def test_elite_overlay_keeps_only_joint_active_trials(run):
    run.configspace = ConfigurationSpace()
    kind = CategoricalHyperparameter("kind", ["a", "b"])
    x = UniformFloatHyperparameter("x", 0, 1)
    y = UniformFloatHyperparameter("y", 0, 1)
    run.configspace.add([kind, x, y, EqualsCondition(x, kind, "a")])
    configs = [{"kind": "a", "x": 0.25, "y": 0.75}, {"kind": "b", "y": 0.25}]
    run.get_config = lambda index: configs[index]
    data = prepare_coverage(run)
    figure = views._scatter(data, "x", "y", elite_indices=[0, 1]).children[1].children[0].figure
    assert list(figure.data[1].x) == [0.25]
    assert list(figure.data[1].y) == [0.75]


@pytest.mark.parametrize("categorical", [False, True])
def test_scatter_discrete_cell_boundaries_match_support_grouping(run, categorical):
    run.configspace = ConfigurationSpace()
    parameter = (
        CategoricalHyperparameter("kind", ["a", "b", "c"])
        if categorical
        else UniformIntegerHyperparameter("kind", 13, 17, meta={"step": 2})
    )
    run.configspace.add([parameter, UniformFloatHyperparameter("rate", 0, 1)])
    values = ["a", "c"] if categorical else [13, 17]
    configs = [{"kind": values[0], "rate": 0.25}, {"kind": values[1], "rate": 0.75}]
    run.get_config = lambda index: configs[index]
    data = prepare_coverage(run)
    result = compute_pairwise_coverage(data, bins=2, references=data)
    figure = views._scatter(data, "kind", "rate", result).children[1].children[0].figure
    edges = [-0.5, 1.5, 2.5] if categorical else [12, 16, 18]
    empty = [shape for shape in figure.layout.shapes if shape.type == "rect"]
    assert {(shape.x0, shape.x1, shape.y0, shape.y1) for shape in empty} == {
        (edges[0], edges[1], 0.5, 1),
        (edges[1], edges[2], 0, 0.5),
    }
    assert list(figure.data[0].x) == values
    assert figure.layout.xaxis.range == (edges[0], edges[-1])
    if categorical:
        assert figure.layout.xaxis.type == "category"
        assert figure.layout.xaxis.categoryarray == ("a", "b", "c")
    else:
        assert figure.layout.xaxis.tickvals == (13, 15, 17)


@pytest.mark.parametrize("metric", list(views.PAIR_METRICS))
@pytest.mark.parametrize("computed", [False, True])
def test_pairwise_figures_have_identical_vertical_domains_and_fixed_padding(run, metric, computed):
    data = prepare_coverage(run)
    result = (
        {
            "pairs": [
                {
                    "x": "x",
                    "y": "y",
                    "occupancy": 0.75,
                    "empty_cell_rate": 0.25,
                    "mmd": 0.01,
                    "max_radius": 0.1,
                }
            ],
        }
        if computed
        else None
    )
    matrix = views._pair_matrix(result, metric, ["x", "y"])
    scatter = views._scatter(data, "x", "y", result)
    assert scatter.className == "coverage-pair-panel"
    plot_row = scatter.children[1]
    graph, metrics = plot_row.children
    assert ("coverage-pair-scatter--with-metrics" in plot_row.className) is computed
    if computed:
        assert isinstance(metrics, html.Aside)
        assert len(metrics.children) == 4
    else:
        assert metrics is None
    assert graph.className == "coverage-pair-plot-row"
    assert graph.style["height"] == f"{matrix.layout.height}px"
    for layout in (matrix.layout, graph.figure.layout):
        assert layout.height == views.PAIR_PLOT_HEIGHT
        assert layout.margin.t == 10 and layout.margin.b == 80
        assert layout.margin.autoexpand is False
        assert layout.xaxis.automargin is False and layout.yaxis.automargin is False
        assert layout.yaxis.domain == (0, 1)
    assert matrix.layout.height - matrix.layout.margin.t - matrix.layout.margin.b == 420
    assert isinstance(scatter.children[2], html.Footer)


def test_matrix_is_symmetric_and_baseline_reverses_occupancy_direction():
    matrix = views._pair_matrix(
        {"pairs": [{"x": "x", "y": "y", "occupancy": 0.75}]}, "occupancy", ["x", "y"]
    )
    assert matrix.data[0].z[0][1] == matrix.data[0].z[1][0] == 0.25
    assert matrix.data[0].customdata[0][1] == matrix.data[0].customdata[1][0] == 0.75
    assert matrix.data[0].hoverongaps is False
    assert matrix.data[0].showscale is False
    assert matrix.layout.margin.r == 15
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


@pytest.mark.parametrize(
    "metric,good,bad,expected",
    [
        ("occupancy", 0.95, 0.6, [0.05, 0.4]),
        ("empty_cell_rate", 0.05, 0.4, [0.05, 0.4]),
        ("max_radius", 0.04, 0.09, [0.2, 0.3]),
        ("mmd", 0.04, 0.09, np.sqrt(np.array([0.04, 0.09]) / np.sqrt(2))),
    ],
)
def test_pairwise_concern_uses_metric_direction_and_fixed_domains(metric, good, bad, expected):
    first = {"x": "x", "y": "y", metric: good}
    matrix = views._pair_matrix(
        {"pairs": [first, {"x": "x", "y": "z", metric: bad}]}, metric, ["x", "y", "z"]
    )
    heatmap = matrix.data[0]
    np.testing.assert_allclose([heatmap.z[0][1], heatmap.z[0][2]], expected)
    assert heatmap.z[0][1] < heatmap.z[0][2]
    assert heatmap.zmin == 0 and heatmap.zmax == 1
    assert heatmap.colorscale == tuple(tuple(stop) for stop in views.PAIR_CONCERN_COLORS)
    assert heatmap.hoverinfo == "none"
    assert heatmap.hovertemplate is None
    assert heatmap.customdata[0][1] == good
    labels = {(a.x, a.y): a.text for a in matrix.layout.annotations}
    assert labels[("y", "x")] == (
        f"{good:.0%}" if metric in ("occupancy", "empty_cell_rate") else f"{good:.3g}"
    )
    assert np.isnan(heatmap.z[0][0]) and np.isnan(heatmap.z[1][2])
    assert ("x", "x") not in labels and ("z", "y") not in labels

    # Different companion pairs and a different visible ordering cannot rescale a cell.
    other = views._pair_matrix(
        {"pairs": [first, {"x": "x", "y": "z", metric: 1.0}]}, metric, ["z", "y", "x"]
    ).data[0]
    assert other.z[2][1] == heatmap.z[0][1]


def test_pairwise_concern_endpoints_contrast_and_unavailable_pairs():
    for metric, scale in views.PAIR_METRIC_SCALES.items():
        values = np.array([0.0, scale["upper"]])
        expected = [1.0, 0.0] if scale["higher_is_better"] else [0.0, 1.0]
        np.testing.assert_allclose(views._pair_severity(values, metric), expected)
    matrix = views._pair_matrix(
        {"pairs": [{"x": "x", "y": "y", "occupancy": None}]}, "occupancy", ["x", "y"]
    )
    assert np.isnan(matrix.data[0].z).all()
    assert not matrix.layout.annotations
    for occupancy, color in [(1.0, "#000000"), (0.0, "#ffffff")]:
        matrix = views._pair_matrix(
            {"pairs": [{"x": "x", "y": "y", "occupancy": occupancy}]},
            "occupancy",
            ["x", "y"],
        )
        assert all(a.font.color == color for a in matrix.layout.annotations)
    legend = views._pair_legend()
    assert [legend.children[i].children for i in (0, 1, 3)] == ["Coverage concern", "Low", "High"]


def test_overview_compares_defined_and_observed_ranges_with_compact_counts(run):
    run.configspace = ConfigurationSpace()
    run.configspace.add(
        [
            UniformFloatHyperparameter("learning_rate", 0.00001, 0.1, log=True),
            UniformIntegerHyperparameter("depth", 13, 17),
        ]
    )
    configs = [{"learning_rate": 0.00001, "depth": 13}, {"learning_rate": 0.01, "depth": 17}]
    run.get_config = lambda index: configs[index]
    data = prepare_coverage(run)
    overview = views._overview(data, compute_univariate_coverage(data), "Random")
    assert [metric.children[0].children for metric in overview.children[0].children] == [
        "Trials",
        "Parameters",
        "Sampler",
    ]
    details = overview.children[1]
    assert details.children[0].children[1].children == "2 parameters · 1 continuous · 1 integer"
    assert details.children[1].children == "No conditional parameters"
    table = details.children[2]
    assert [column["id"] for column in table.columns] == [
        "parameter",
        "type",
        "search_space",
        "observed_range",
        "distribution",
        "step",
    ]
    rows = {row["parameter"]: row for row in table.data}
    assert rows["depth"]["search_space"] == rows["depth"]["observed_range"] == "13 – 17"
    assert rows["learning_rate"]["search_space"] == "1e-05 – 0.1"
    assert rows["learning_rate"]["observed_range"] == "1e-05 – 0.01"
    assert "coverage-chip--log" in rows["learning_rate"]["distribution"]
    assert "Log-uniform" in rows["learning_rate"]["distribution"]
    assert "Uniform" in rows["depth"]["distribution"]


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("categorical", [False, True])
def test_overview_only_shows_needed_optional_columns(run, conditional, categorical):
    space = ConfigurationSpace()
    parent = (
        CategoricalHyperparameter("parent", ["off", "on"], weights=[1, 3])
        if categorical
        else UniformIntegerHyperparameter("parent", 0, 1)
    )
    child = UniformFloatHyperparameter("child", 0, 1)
    space.add([parent, child])
    if conditional:
        space.add(EqualsCondition(child, parent, "on" if categorical else 1))
    configs = [
        {"parent": "off" if categorical else 0, **({} if conditional else {"child": 0.25})},
        {"parent": "on" if categorical else 1, "child": 0.75},
    ]
    run.configspace = space
    run.get_config = lambda index: configs[index]
    data = prepare_coverage(run)
    details = views._overview(data, compute_univariate_coverage(data), "Random").children[1]
    table = details.children[2]
    columns = [column["id"] for column in table.columns]
    assert ("conditional" in columns) == conditional
    assert ("probabilities" in columns) == categorical
    assert (details.children[1] is None) == conditional
    rows = {row["parameter"]: row for row in table.data}
    if conditional:
        assert rows["child"]["conditional"] is True
        assert rows["parent"]["conditional"] is False
        assert rows["child"]["observed_range"] == "0.75 – 0.75"
    if categorical:
        assert rows["parent"]["probabilities"] == "0.25, 0.75"
        assert rows["child"]["probabilities"] == ""
        assert rows["parent"]["search_space"] == "off, on"
        assert rows["parent"]["observed_range"] == "—"


def test_overview_handles_parameters_without_observations(run):
    run.history = []
    data = prepare_coverage(run)
    table = views._definition_table(compute_univariate_coverage(data))
    assert all(row["observed_range"] == "—" for row in table.data)


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
    tab_ids = {}

    def visit(component, tab_id=None):
        if isinstance(component, (list, tuple)):
            for child in component:
                visit(child, tab_id)
        elif hasattr(component, "to_plotly_json"):
            tab_id = getattr(component, "tab_id", tab_id)
            components.append(component)
            if getattr(component, "id", None):
                ids.append(component.id)
                if tab_id:
                    tab_ids.setdefault(tab_id, set()).add(component.id)
            visit(getattr(component, "children", None), tab_id)
        elif isinstance(component, str):
            labels.append(component)

    visit(layout)
    assert len(ids) == len(set(ids))
    assert sum(label == "Run analysis" for label in labels) == len(suite.UI_ANALYSES)
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
    assert suite._id("combinatorial-run") not in ids
    assert suite._id("combinatorial_baseline-run") in tab_ids["global"]
    refresh = next(
        c for c in components if getattr(c, "id", None) == suite._id("combinatorial_baseline-run")
    )
    assert refresh.title == "Refresh coverage and random baseline"
    assert ids.index(suite._id("combinatorial_baseline-run")) < ids.index(
        suite._id("combinatorial-result")
    )
    assert suite._id("combinatorial-baseline-result") not in ids
    assert "CCM random comparison" not in labels
    assert "Coverage Array" in labels
    assert "Strength Profile" not in labels
    assert "Selected parameter group" not in labels
    radius_ids = {
        suite._id(name)
        for name in ("radius-run", "radius-summary", "radius-section", "regions", "region-detail")
    }
    assert radius_ids.isdisjoint(tab_ids["global"])
    assert radius_ids <= tab_ids["growth"]
    removed_ids = {
        suite._id(name)
        for name in (
            "discrepancy-run",
            "baseline-run",
            "discrepancy-metric",
            "discrepancy-section",
            "discrepancy-result",
            "baseline-data",
            "discrepancy-data",
        )
    }
    assert removed_ids.isdisjoint(ids)
    assert "Sampling distribution" not in labels
    assert "Random comparisons" not in labels
    assert all(
        item["id"] not in removed_ids
        for callback in app.callback_map.values()
        for item in callback["inputs"]
    )
    pairwise = next(
        c for c in components if getattr(c, "className", None) == "coverage-pair-diagnostic"
    )
    matrix_column, scatter_column = pairwise.children
    assert matrix_column.children[0].className == "coverage-pair-controls"
    metric_control = next(
        c for c in components if getattr(c, "id", None) == suite._id("pair-metric")
    )
    assert isinstance(metric_control, suite.dbc.RadioItems)
    assert metric_control.value == "occupancy"
    assert metric_control.options == [
        {"label": label, "value": key} for key, label in views.PAIR_METRICS.items()
    ]
    selectors, scatter_slot = scatter_column.children
    assert selectors.className == "coverage-pair-selectors"
    assert [field.children[1].id for field in selectors.children] == [
        suite._id("x"),
        suite._id("y"),
    ]
    assert scatter_slot.children[0].id == suite._id("scatter")
    highlight_footer = scatter_slot.children[1]
    assert highlight_footer.className == "coverage-pair-scatter-footer"
    assert [control.id for control in highlight_footer.children] == [
        suite._id("highlight-empty-cells"),
        suite._id("highlight-elite-trials"),
    ]
    elite_proportion = next(
        c for c in components if getattr(c, "id", None) == suite._id("elite-proportion")
    )
    assert elite_proportion.value == 0.1
    assert [option["value"] for option in elite_proportion.options] == [0.2, 0.1, 0.05, 0.02, 0.01]
    settings = next(c for c in components if getattr(c, "id", None) == suite._id("settings"))
    assert settings.is_open is False
    assert all(suite._id(f"{analysis}-cancel") in ids for analysis in suite.UI_ANALYSES)
    assert len(app.callback_map) == 19
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
        "combinatorial_results",
        "combinatorial_detail",
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
    selectors = post(suite._id("x"), controls, suite._id("context") + ".data")
    assert selectors[suite._id("x")]["value"] == "x"
    controls[(suite._id("pairwise-run"), "n_clicks")] = 1
    started = post(suite._id("pairwise-data"), controls, suite._id("pairwise-run") + ".n_clicks")
    assert started[suite._id("pairwise-run")]["disabled"] is True
    assert len(queue.calls) == 1
    queue.finish(next(iter(queue.jobs)))
    controls[(suite._id("pairwise-view"), "data")] = started[suite._id("pairwise-view")]["data"]
    completed = post(suite._id("pairwise-data"), controls, suite._id("tick") + ".n_intervals")
    assert completed[suite._id("pairwise-status")]["children"] == ""
    assert completed[suite._id("pairwise-metadata")]["children"] == ""
    assert completed[suite._id("pairwise-run")]["disabled"] is False
    assert len(queue.calls) == 1
    controls[(suite._id("pairwise-data"), "data")] = completed[suite._id("pairwise-data")]["data"]
    controls[(suite._id("pair-metric"), "value")] = "occupancy"
    matrix = post(suite._id("pair-matrix"), controls, suite._id("pairwise-data") + ".data")
    assert matrix[suite._id("pair-matrix")]["figure"]["data"][0]["type"] == "heatmap"
