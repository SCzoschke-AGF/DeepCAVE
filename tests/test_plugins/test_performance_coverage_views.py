"""Performance panel uses the sidebar objective and leaves coverage jobs independent."""

from types import SimpleNamespace

import numpy as np
import pytest
from dash.exceptions import PreventUpdate

from deepcave.plugins.hyperparameter import near_optimal_configurations as rashomon
from deepcave.plugins.hyperparameter import performance_coverage_views as views
from deepcave.plugins.hyperparameter import search_space_coverage as suite

from tests.test_evaluators.test_performance_coverage import make_run
from tests.test_plugins.test_search_space_coverage import FakeQueue, visible_text


@pytest.fixture
def weights(monkeypatch):
    monkeypatch.setattr(
        rashomon, "_fanova_importances", lambda *args: {"values": {"x": 1.0}, "error": None}
    )


def test_worker_reuses_rashomon_metric_and_cached_distances(tmp_path, monkeypatch, weights):
    run = make_run()
    original = views.cached_pairwise_configuration_distances
    calls = []

    def tracked(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(views, "cached_pairwise_configuration_distances", tracked)
    result = views.run_performance_analysis(run, 0, str(tmp_path))
    assert result["pairs"] == 6 and result["weights"] == {"x": 1.0}
    assert "highest budget" in result["weight_source"]
    assert len(calls) == 1
    assert calls[0][0][3] == "importance_weighted"
    assert views.run_performance_analysis(run, 0, str(tmp_path)) == result
    changed = views.run_performance_analysis(run, 1, str(tmp_path))
    assert len(calls) == 1  # Objective changed; the same configuration distances remain valid.
    assert changed["cache_key"] != result["cache_key"]
    monkeypatch.setattr(
        rashomon, "_fanova_importances", lambda *args: {"values": {"x": 0.5}, "error": None}
    )
    reweighted = views.run_performance_analysis(run, 0, str(tmp_path))
    assert reweighted["cache_key"] != result["cache_key"]
    assert len(calls) == 2


def test_unavailable_importances_are_reported_without_unweighted_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rashomon, "_fanova_importances", lambda *args: {"values": {}, "error": "not enough data"}
    )
    with pytest.raises(ValueError, match="not enough data"):
        views.run_performance_analysis(make_run(), 0, str(tmp_path))


@pytest.fixture
def callbacks(tmp_path, monkeypatch):
    functions = {}

    def callback(*args, **kwargs):
        def register(function):
            functions[function.__name__] = function
            return function

        return register

    run = make_run()
    queue = FakeQueue()
    jobs = suite.CoverageJobs(tmp_path, queue)
    event = SimpleNamespace(triggered_prop_ids={})
    monkeypatch.setattr(views, "ctx", event)
    views.register_performance_callbacks(
        SimpleNamespace(callback=callback), jobs, SimpleNamespace(get_run=lambda _: run)
    )
    return functions, jobs, run, event


def job_call(callbacks, trigger="tick", objective="objective:1", previous=None, clicks=0):
    functions, jobs, run, event = callbacks
    event.triggered_prop_ids = {"trigger": suite._id(trigger)}
    return functions["performance_job"](
        clicks, {"run_id": run.id}, 0, {"value": objective}, run.id, previous
    )


def test_only_refresh_queues_sidebar_objective_and_threshold_never_queues(callbacks, weights):
    functions, jobs, run, event = callbacks
    idle = job_call(callbacks)
    assert idle[0]["objective"] == "Loss" and not jobs.queue.calls
    started = job_call(callbacks, "performance-run", clicks=1)
    assert started[2] is True and len(jobs.queue.calls) == 1
    queued = next(iter(jobs.queue.jobs.values()))
    assert queued.args[1] == 1
    job_call(callbacks, "performance-run", clicks=2)
    assert len(jobs.queue.calls) == 1
    jobs.queue.finish(queued.id)
    complete = job_call(callbacks, previous=started[3])
    assert complete[2] is False and complete[1] == ""
    rendered = functions["performance_results"](complete[0], 0.01, {"run_id": run.id})
    assert rendered[:3] == ("4", "6", "Loss")
    assert "Distance range" in visible_text(rendered[5])
    assert "highest budget" in visible_text(rendered[7])
    changed = functions["performance_results"](complete[0], 0.5, {"run_id": run.id})
    assert changed[-1] == "" and len(jobs.queue.calls) == 1
    invalid = functions["performance_results"](complete[0], None, {"run_id": run.id})
    assert invalid[:3] == ("4", "6", "Loss")
    assert invalid[-1] and len(jobs.queue.calls) == 1
    switched = job_call(callbacks, objective="objective:0", previous=complete[3])
    assert switched[0]["objective"] == "AP" and switched[0]["cache_key"] is None
    assert len(jobs.queue.calls) == 1
    hidden = functions["performance_results"](switched[0], 0.01, {"run_id": run.id})
    assert hidden[:3] == ("—", "—", "AP")


def test_stale_results_preserved_without_automatic_recomputation(callbacks, weights):
    functions, jobs, run, event = callbacks
    started = job_call(callbacks, "performance-run", clicks=1)
    jobs.queue.finish(next(iter(jobs.queue.jobs)))
    complete = job_call(callbacks)
    run.history[0].costs[1] = 10.0
    stale = job_call(callbacks, previous=complete[3])
    assert stale[1] == "Study changed · Refresh to update."
    assert len(jobs.queue.calls) == 1
    assert stale[0] is views.no_update
    with pytest.raises(PreventUpdate):
        job_call(callbacks, previous=stale[3])


def test_missing_objective_and_worker_are_explained(callbacks):
    idle = job_call(callbacks, objective=None)
    assert "sidebar" in idle[1] and idle[2]
    callbacks[1].queue.available = False
    failed = job_call(callbacks, "performance-run", clicks=1)
    assert "No analysis worker" in failed[1]


def test_figures_use_empirical_curves_and_linear_axis(tmp_path, weights):
    result = views.run_performance_analysis(make_run(), 0, str(tmp_path))
    figure = views.performance_figure(result)
    assert figure.data[0].type == "scattergl"
    assert figure.layout.yaxis.type == "linear"
    assert [trace.name for trace in figure.data[1:]] == ["Median", "P90"]
    assert all(trace.line.shape == "linear" for trace in figure.data[1:])
    assert "Group pairs" in figure.data[0].hovertemplate
    result["plot"] = {
        "kind": "density",
        "x_edges": [0, 0.5, 1],
        "y_edges": [0, 0.5, 1],
        "counts": [[1, 2], [3, 0]],
    }
    density = views.performance_figure(result)
    assert density.data[0].type == "heatmap"
    assert np.nansum(np.asarray(density.data[0].z, dtype=float)) == 6


def test_interpretation_requires_consistent_observed_bin_medians():
    for values, expected in (
        ([1, 2, 3], "increase"),
        ([3, 2, 1], "decrease"),
        ([1, 3, 2], "No consistent"),
        ([1], "No consistent"),
    ):
        result = {"bins": [{"median": value} for value in values]}
        assert expected in views.performance_interpretation(result)
