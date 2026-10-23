"""Real callback behavior for CHR candidate scoring, rankings and detail selection."""

from types import SimpleNamespace

import pytest
from dash.exceptions import PreventUpdate

from deepcave.plugins.hyperparameter import performance_coverage_views as views
from deepcave.plugins.hyperparameter import search_space_coverage as suite

from tests.test_evaluators.test_coverage_candidates import OPTIONS, make_run
from tests.test_plugins.test_search_space_coverage import FakeQueue, visible_text


@pytest.fixture
def callbacks(tmp_path, monkeypatch):
    functions = {}

    def callback(*args, **kwargs):
        def register(function):
            functions[function.__name__] = function
            return function

        return register

    run = make_run()
    jobs = suite.CoverageJobs(tmp_path, FakeQueue())
    event = SimpleNamespace(triggered_prop_ids={})
    monkeypatch.setattr(views, "ctx", event)
    views.register_performance_callbacks(
        SimpleNamespace(callback=callback), jobs, SimpleNamespace(get_run=lambda _: run)
    )
    return functions, jobs, run, event


def job_call(
    callbacks, trigger="tick", objective="objective:0", previous=None, clicks=0, **settings
):
    functions, jobs, run, event = callbacks
    event.triggered_prop_ids = {"trigger": suite._id(trigger)}
    options = {**OPTIONS, **settings}
    return functions["performance_job"](
        clicks,
        {"run_id": run.id},
        0,
        {"value": objective},
        *[options[name] for name, *_ in views._SETTINGS],
        run.id,
        previous,
    )


def finish(callbacks):
    started = job_call(callbacks, "performance-run", clicks=1)
    jobs = callbacks[1]
    jobs.queue.finish(next(iter(jobs.queue.jobs)))
    complete = job_call(callbacks, previous=started[3])
    return complete


def test_refresh_queues_once_tabs_and_details_never_queue(callbacks):
    functions, jobs, run, event = callbacks
    idle = job_call(callbacks)
    assert idle[0]["cache_key"] is None and not jobs.queue.calls
    started = job_call(callbacks, "performance-run", clicks=1)
    assert started[2] and "scoring" in started[1]
    queued = next(iter(jobs.queue.jobs.values()))
    assert queued.args[1] == 0 and queued.args[3] == OPTIONS
    job_call(callbacks, "performance-run", clicks=2)
    assert len(jobs.queue.calls) == 1
    jobs.queue.finish(queued.id)
    complete = job_call(callbacks, previous=started[3])
    assert not complete[2] and complete[1] == ""
    context = {"run_id": run.id}
    result = jobs.read(run.id, "candidate_scoring")["result"]
    overall = functions["performance_results"](complete[0], "overall", context)
    assert overall[:4] == (
        str(len(result["holes"])),
        str(len(result["candidates"])),
        f"Best observed trial #{result['reference_trial_number']}",
        f"{result['top_relative_score']:+.3f}",
    )
    scores = [r["relative_score"] for r in overall[5]]
    assert scores == sorted(scores, reverse=True)
    per_hole = functions["performance_results"](complete[0], "holes", context)
    assert len({r["hole_id"] for r in per_hole[5]}) == len(per_hole[5])
    for row in per_hole[5]:
        assert row["relative_score"] == max(
            r["relative_score"] for r in overall[5] if r["hole_id"] == row["hole_id"]
        )
    plan = functions["performance_results"](complete[0], "plan", context)
    assert [r["id"] for r in plan[5]] == result["plan"]
    assert [r["selected"] for r in plan[5]] == [f"✓ {i + 1}" for i in range(len(result["plan"]))]
    selected = functions["performance_detail"](plan[6], complete[0], context)
    assert "Included in CHR plan" in visible_text(selected)
    other = overall[5][-1]
    changed = functions["performance_detail"]({"row_id": other["id"]}, complete[0], context)
    assert changed != selected
    assert "Show full config" in visible_text(changed)
    assert len(jobs.queue.calls) == 1


def test_study_settings_and_objective_invalidation_preserve_explicit_refresh(callbacks):
    functions, jobs, run, event = callbacks
    complete = finish(callbacks)
    run.history[0].costs[0] += 5
    stale = job_call(callbacks, previous=complete[3])
    assert stale[0] is views.no_update
    assert "Study or settings changed" in stale[1]
    with pytest.raises(PreventUpdate):
        job_call(callbacks, previous=stale[3])
    settings = job_call(callbacks, previous=stale[3], pool_size=80)
    assert "Refresh" in settings[1]
    assert len(jobs.queue.calls) == 1
    changed = job_call(callbacks, objective="objective:1", previous=settings[3])
    assert changed[0]["objective_id"] == 1 and changed[0]["cache_key"] is None
    hidden = functions["performance_results"](changed[0], "overall", {"run_id": run.id})
    assert hidden[:4] == ("—",) * 4 and hidden[5] == []
    assert functions["performance_detail"](None, changed[0], {"run_id": run.id}) == ""
    assert len(jobs.queue.calls) == 1


def test_missing_objective_settings_worker_and_fit_failure_are_explained(callbacks):
    idle = job_call(callbacks, objective=None)
    assert "sidebar" in idle[1] and idle[2]
    invalid = job_call(callbacks, "performance-run", clicks=1, pool_size=None)
    assert "pool_size" in invalid[1] and invalid[2] and not callbacks[1].queue.calls
    callbacks[1].queue.available = False
    failed = job_call(callbacks, "performance-run", clicks=1)
    assert "No analysis worker" in failed[1]
    callbacks[1].queue.available = True
    started = job_call(callbacks, "performance-run", clicks=2)
    queued = next(iter(callbacks[1].queue.jobs.values()))
    queued.status = "failed"
    result = job_call(callbacks, previous=started[3])
    assert "failed" in result[1]
    payload = started[0] if result[0] is views.no_update else result[0]
    rendered = callbacks[0]["performance_results"](payload, "overall", {"run_id": callbacks[2].id})
    assert rendered[5] == []


def test_error_results_and_run_switch_cannot_show_other_studies(callbacks):
    complete = finish(callbacks)
    functions, jobs, run, event = callbacks
    assert (
        functions["performance_results"](complete[0], "overall", {"run_id": "other"})[:4]
        == ("—",) * 4
    )
    assert functions["performance_detail"]({"row_id": "x"}, complete[0], {"run_id": "other"}) == ""
    with pytest.raises(PreventUpdate):
        functions["performance_job"](
            1, {"run_id": "other"}, 0, None, 60, 4, 4, 128, 42, run.id, None
        )


def test_worker_fit_error_reaches_ui_without_scores_and_allows_retry(callbacks):
    functions, jobs, run, event = callbacks
    run.history = run.history[:5]
    complete = finish(callbacks)
    assert "Insufficient completed finite trials" in complete[1]
    assert not complete[2]
    payload = complete[0]
    if payload is views.no_update:
        payload = {"run_id": run.id, "cache_key": None}
    assert (
        functions["performance_results"](payload, "overall", {"run_id": run.id})[:4] == ("—",) * 4
    )


def test_small_pool_explains_holes_without_generated_candidates(callbacks):
    functions, jobs, run, event = callbacks
    started = job_call(callbacks, "performance-run", clicks=1, pool_size=1, plan_size=1)
    jobs.queue.finish(next(iter(jobs.queue.jobs)))
    complete = job_call(callbacks, previous=started[3], pool_size=1, plan_size=1)
    context = {"run_id": run.id}
    view = functions["performance_results"](complete[0], "holes", context)
    result = jobs.read(run.id, "candidate_scoring")["result"]
    assert len(view[5]) == len(result["holes"])
    missing = next(row for row in view[5] if row["relative_score"] is None)
    detail = functions["performance_detail"]({"row_id": missing["id"]}, complete[0], context)
    assert "No valid candidate" in visible_text(detail)


@pytest.mark.parametrize("values", [[-12, -5, -2], [1, 3, 8], [2, 2, 2], [-1e5, 2e5]])
def test_histogram_counts_all_relative_scores_with_reference_at_zero(values):
    figure = views.performance_figure({"candidates": [{"relative_score": v} for v in values]})
    assert sum(figure.data[0].y) == len(values)
    assert figure.layout.shapes[0].x0 == figure.layout.shapes[0].x1 == 0
    assert figure.layout.shapes[0].line.dash == "dash"
    assert figure.layout.xaxis.range[0] < min(0, *values)
    assert figure.layout.xaxis.range[1] > max(0, *values)
    assert figure.layout.annotations[0].text == "Best observed trial"
    assert figure.layout.xaxis.title.text == "KDE score relative to reference"


def test_panel_titles_loading_settings_and_readonly_membership():
    panel = views.performance_panel()
    text = visible_text(panel)
    assert "Performance scoring of coverage candidates" in text
    assert "Best candidate per hole" not in text  # Tab labels are props, not children.
    assert "Scores are neither predicted objective values nor success probabilities." in text

    def walk(component):
        if isinstance(component, (list, tuple)):
            for child in component:
                yield from walk(child)
        elif hasattr(component, "children"):
            yield component
            yield from walk(component.children)
        else:
            yield component

    components = list(walk(panel))
    table = next(
        c for c in components if getattr(c, "id", None) == suite._id("performance-ranking")
    )
    assert not getattr(table, "editable", False)
    assert not getattr(table, "row_selectable", False)
    assert any(c.__class__.__name__ == "Loading" for c in components)
