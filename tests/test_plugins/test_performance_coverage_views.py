"""Real callback behavior for CHR candidate scoring, rankings and detail selection."""

from types import SimpleNamespace
from urllib.parse import unquote

import pytest
from dash.exceptions import PreventUpdate

from deepcave.plugins.hyperparameter import performance_coverage_views as views
from deepcave.plugins.hyperparameter import search_space_coverage as suite

from tests.test_evaluators.test_coverage_candidates import OPTIONS, make_run
from tests.test_plugins.test_coverage_planning_views import tomllib
from tests.test_plugins.test_search_space_coverage import FakeQueue, visible_text


@pytest.fixture
def callbacks(tmp_path, monkeypatch):
    functions = {}

    def callback(*args, **kwargs):
        def register(function):
            functions[function.__name__] = function
            function.callback_dependencies = args
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
    overall = functions["performance_results"](complete[0], "overall", "recommended", context)
    assert overall[:4] == (
        str(len(result["holes"])),
        str(len(result["candidates"])),
        f"Best observed trial #{result['reference_trial_number']}",
        f"{result['top_relative_score']:+.3f}",
    )
    chart = functions["performance_chart"]
    assert [d.component_id for d in chart.callback_dependencies if isinstance(d, views.Input)] == [
        suite._id("performance-data"),
        suite._id("performance-tab"),
    ]
    assert sum(chart(complete[0], "overall", context).data[0].y) == len(result["candidates"])
    scores = [r["relative_score"] for r in overall[4]]
    assert scores == sorted(scores, reverse=True)
    per_hole = functions["performance_results"](complete[0], "holes", "recommended", context)
    assert len({r["hole_id"] for r in per_hole[4]}) == len(per_hole[4])
    for row in per_hole[4]:
        assert row["relative_score"] == max(
            r["relative_score"] for r in overall[4] if r["hole_id"] == row["hole_id"]
        )
    assert overall[7:9] == per_hole[7:9] == ([], [])
    assert overall[11] == per_hole[11] == []
    assert overall[10] == per_hole[10] == ""
    selected = functions["performance_detail"](overall[5], complete[0], context)
    assert "Coverage contribution" in visible_text(selected)
    other = overall[4][-1]
    changed = functions["performance_detail"]({"row_id": other["id"]}, complete[0], context)
    assert changed != selected
    assert "Show full config" in visible_text(changed)
    assert len(jobs.queue.calls) == 1


def test_study_settings_and_objective_invalidation_preserve_explicit_refresh(callbacks):
    functions, jobs, run, event = callbacks
    complete = finish(callbacks)
    run.history[0].costs[0] += 5
    stale = job_call(callbacks, previous=complete[3])
    assert stale[0]["cache_key"] is None
    assert stale[1] == ""
    with pytest.raises(PreventUpdate):
        job_call(callbacks, previous=stale[3])
    settings = job_call(callbacks, previous=stale[3], pool_size=80)
    assert settings[1] == ""
    assert len(jobs.queue.calls) == 1
    changed = job_call(callbacks, objective="objective:1", previous=settings[3])
    assert changed[0]["objective_id"] == 1 and changed[0]["cache_key"] is None
    hidden = functions["performance_results"](
        changed[0], "overall", "recommended", {"run_id": run.id}
    )
    assert hidden[:4] == ("—",) * 4 and hidden[4] == []
    assert not functions["performance_chart"](changed[0], "overall", {"run_id": run.id}).data
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
    rendered = callbacks[0]["performance_results"](
        payload, "overall", "recommended", {"run_id": callbacks[2].id}
    )
    assert rendered[4] == []


def test_error_results_and_run_switch_cannot_show_other_studies(callbacks):
    complete = finish(callbacks)
    functions, jobs, run, event = callbacks
    assert (
        functions["performance_results"](
            complete[0], "overall", "recommended", {"run_id": "other"}
        )[:4]
        == ("—",) * 4
    )
    assert functions["performance_detail"]({"row_id": "x"}, complete[0], {"run_id": "other"}) == ""
    with pytest.raises(PreventUpdate):
        functions["performance_job"](
            1, {"run_id": "other"}, 0, None, 60, 4, 128, 42, 80, 17, 20, run.id, None
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
        functions["performance_results"](payload, "overall", "recommended", {"run_id": run.id})[:4]
        == ("—",) * 4
    )


def test_small_pool_explains_holes_without_generated_candidates(callbacks):
    functions, jobs, run, event = callbacks
    started = job_call(callbacks, "performance-run", clicks=1, pool_size=1)
    jobs.queue.finish(next(iter(jobs.queue.jobs)))
    complete = job_call(callbacks, previous=started[3], pool_size=1)
    context = {"run_id": run.id}
    view = functions["performance_results"](complete[0], "holes", "recommended", context)
    result = jobs.read(run.id, "candidate_scoring")["result"]
    assert len(view[4]) == len(result["holes"])
    missing = next(row for row in view[4] if row["relative_score"] is None)
    assert missing["kde_score"] is None
    assert missing["good_probability"] is None
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
    assert figure.layout.annotations[0].text == "Best observed trial<br>(Reference)"
    assert figure.layout.xaxis.title.text == "KDE score relative to reference"


def test_panel_titles_loading_settings_and_candidate_selection():
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
    assert table.row_selectable == "multi"
    assert table.selected_rows == table.selected_row_ids == []
    assert table.sort_action == "native" and table.sort_by == []
    assert table.columns[2]["id"] == "relative_score"
    score_column = next(c for c in table.columns if c["id"] == "kde_score")
    assert score_column["name"] == "KDE score"
    assert score_column["type"] == "numeric"
    assert score_column["format"].to_plotly_json()["specifier"] == ".2f"
    relative_column = next(c for c in table.columns if c["id"] == "relative_score")
    assert relative_column["format"].to_plotly_json()["specifier"] == "+.2f"
    probability_column = next(c for c in table.columns if c["id"] == "good_probability")
    assert probability_column["name"] == "Top-20% probability"
    assert probability_column["type"] == "numeric"
    assert probability_column["format"].to_plotly_json()["specifier"] == ".2%"
    assert "Uncalibrated" in table.tooltip_header["good_probability"]
    tabs = next(c for c in components if getattr(c, "id", None) == suite._id("performance-tab"))
    assert tabs.children[-1].label == "Best candidates: KDE + exploration"
    assert tabs.children[-1].value == "full_space"
    assert [tab.value for tab in tabs.children] == ["overall", "holes", "full_space"]
    assert not any(
        getattr(c, "id", None) == suite._id("performance-plan_size-setting") for c in components
    )
    assert any(c.__class__.__name__ == "Loading" for c in components)


@pytest.mark.parametrize("mode", ["recommended", "all"])
@pytest.mark.parametrize("tab", ["overall", "holes", "full_space"])
@pytest.mark.parametrize("study", [None, {"name": 'catboost_hpo "test"', "id": 7}])
def test_selected_trials_export_full_parameters_in_ranking_order(callbacks, tab, study, mode):
    complete = finish(callbacks)
    functions, jobs, run, _ = callbacks
    if study:
        run.meta.update(optuna_study_name=study["name"], optuna_study_id=study["id"])
    context = {"run_id": run.id}
    result = jobs.read(run.id, "candidate_scoring")["result"]
    ranked = views.ranking_candidates(result, tab, mode)
    chosen = [ranked[0], ranked[-1]]
    if tab == "overall":
        assert len(ranked) > 8  # Selection can span table pages.
    link = functions["performance_export"](
        [c["id"] for c in reversed(chosen)] + ["unknown", "empty-hole"],
        complete[0],
        tab,
        mode,
        context,
    )
    assert link.download == (
        "full-space-candidates.toml" if tab == "full_space" else "coverage-candidates.toml"
    )
    prefix, content = link.href.split(",", 1)
    assert prefix == "data:application/toml;charset=utf-8"
    exported = tomllib.loads(unquote(content))
    expected = {"trials": [c["configuration"] for c in chosen]}
    if study:
        expected["study"] = study
    assert exported == expected
    assert type(exported["trials"][0]["x"]) is float
    assert type(exported["trials"][0]["n"]) is int
    assert len(jobs.queue.calls) == 1


def test_export_requires_valid_selection_and_current_results(callbacks):
    complete = finish(callbacks)
    functions, jobs, run, _ = callbacks
    context = {"run_id": run.id}
    entry = jobs.read(run.id, "candidate_scoring")
    candidate = entry["result"]["candidates"][0]
    export = functions["performance_export"]
    for selected, payload, current_context in (
        ([], complete[0], context),
        (["unknown", "empty-hole"], complete[0], context),
        ([candidate["id"]], complete[0], {"run_id": "other"}),
        ([candidate["id"]], {"run_id": run.id, "cache_key": "stale"}, context),
    ):
        assert export(selected, payload, "overall", "recommended", current_context).disabled
    candidate["configuration"]["unsupported"] = None
    jobs._write(run.id, "candidate_scoring", entry)
    disabled = export([candidate["id"]], complete[0], "overall", "recommended", context)
    assert disabled.disabled
    assert "cannot represent parameter" in disabled.title


def test_full_space_tab_ranks_its_pool_and_shows_sampling_details(callbacks):
    complete = finish(callbacks)
    functions, jobs, run, _ = callbacks
    context = {"run_id": run.id}
    result = jobs.read(run.id, "candidate_scoring")["result"]
    rendered = functions["performance_results"](complete[0], "full_space", "all", context)
    ranked = views.ranking_candidates(result, "full_space", "all")
    assert len(rendered[4]) == len(ranked) == result["supply"]["final_count"]
    assert [r["id"] for r in rendered[4]] == [c["id"] for c in ranked]
    assert [c["kde_score"] for c in ranked] == sorted(
        [c["kde_score"] for c in ranked], reverse=True
    )
    assert rendered[1] == f"{len(ranked):,}"
    assert rendered[3] == f"{ranked[0]['relative_score']:+.3f}"
    assert rendered[7:9] == ([], [])
    assert [column["id"] for column in rendered[9]] == [
        "bin",
        "distance",
        "relative_score",
        "kde_score",
        "good_probability",
        "kde_rank",
        "selection",
        "rank",
    ]
    assert [row["kde_score"] for row in rendered[4]] == [c["kde_score"] for c in ranked]
    probabilities = [r["good_probability"] for r in rendered[4]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert rendered[11] == []
    assert "Initial: 80" in rendered[10] and "Global KDE: 10" in rendered[10]
    assert all(row["hole_id"] is None and row["hole_radius"] is None for row in rendered[4])
    chart = functions["performance_chart"](complete[0], "full_space", context)
    assert sum(chart.data[0].y) == result["supply"]["final_count"]
    details = visible_text(functions["performance_detail"](rendered[5], complete[0], context))
    assert "Full-space search" in details and "Show full config" in details
    assert "Coverage contribution" not in details
    assert len(jobs.queue.calls) == 1
    changed = job_call(callbacks, previous=complete[3], global_pool_size=90)
    assert changed[1] == "" and len(jobs.queue.calls) == 1


def test_recommended_view_keeps_selection_order_kde_rank_and_geometry(callbacks):
    complete = finish(callbacks)
    functions, jobs, run, _ = callbacks
    context = {"run_id": run.id}
    result = jobs.read(run.id, "candidate_scoring")["result"]
    rendered = functions["performance_results"](complete[0], "full_space", "recommended", context)
    rows = rendered[4]
    assert len(rows) == 20
    assert [r["rank"] for r in rows] == list(range(1, 21))
    assert [r["selection"] for r in rows] == ["Global KDE"] * 10 + ["Exploration"] * 10
    assert [r["bin"] for r in rows[10:]] == [1, 2, 3, 4, 5] * 2
    assert any(r["rank"] != r["kde_rank"] for r in rows[10:])
    all_rows = {r["id"]: r for r in views.ranking_rows(result, "full_space", "all")}
    for row in rows:
        assert row["kde_score"] == all_rows[row["id"]]["kde_score"]
        assert row["good_probability"] == all_rows[row["id"]]["good_probability"]
        assert row["distance"] == all_rows[row["id"]]["distance"]
    detail = visible_text(
        functions["performance_detail"]({"row_id": rows[-1]["id"]}, complete[0], context)
    )
    assert "Exploration" in detail and "Distance to incumbent (RMS)" in detail
    assert functions["performance_mode"]("overall") == {"display": "none"}
    assert functions["performance_mode"]("full_space") == {}


def test_failed_supply_never_falls_back_and_keeps_full_ranking_available(callbacks):
    complete = finish(callbacks)
    functions, jobs, run, _ = callbacks
    context = {"run_id": run.id}
    entry = jobs.read(run.id, "candidate_scoring")
    entry["result"].update(recommendations=[])
    entry["result"]["hybrid"].update(
        available=False,
        error="Hybrid selection unavailable. Bin 1: 12/1000",
        global_count=0,
        exploration_count=0,
    )
    jobs._write(run.id, "candidate_scoring", entry)
    recommended = functions["performance_results"](
        complete[0], "full_space", "recommended", context
    )
    assert recommended[4] == [] and "Bin 1: 12/1000" in recommended[10]
    assert functions["performance_results"](complete[0], "full_space", "all", context)[4]
    assert functions["performance_results"](complete[0], "overall", "recommended", context)[4]


@pytest.mark.parametrize("reason", ["finite_space_exhausted", "sampling_budget_reached"])
def test_empty_global_pool_keeps_hole_results_available_and_explains_limit(callbacks, reason):
    complete = finish(callbacks)
    functions, jobs, run, _ = callbacks
    entry = jobs.read(run.id, "candidate_scoring")
    entry["result"].update(full_space_candidates=[], recommendations=[])
    entry["result"]["global_sampling"].update(count=0, stop_reason=reason)
    jobs._write(run.id, "candidate_scoring", entry)
    context = {"run_id": run.id}
    rendered = functions["performance_results"](complete[0], "full_space", "recommended", context)
    assert rendered[1] == "0" and rendered[3] == "—" and rendered[4] == []
    assert (
        "No unseen configurations remain"
        if reason == "finite_space_exhausted"
        else "sampling limit reached"
    ) in rendered[10]
    assert functions["performance_results"](complete[0], "overall", "recommended", context)[4]


def test_global_only_results_and_older_cached_results_render_without_errors(callbacks):
    complete = finish(callbacks)
    functions, jobs, run, _ = callbacks
    entry = jobs.read(run.id, "candidate_scoring")
    entry["result"].update(candidates=[], holes=[])
    jobs._write(run.id, "candidate_scoring", entry)
    context = {"run_id": run.id}
    rendered = functions["performance_results"](complete[0], "full_space", "recommended", context)
    assert len(rendered[4]) == OPTIONS["number_of_candidates"]
    assert functions["performance_chart"](complete[0], "full_space", context).data
    assert "Full-space search" in visible_text(
        functions["performance_detail"](rendered[5], complete[0], context)
    )
    entry["result"].pop("full_space_candidates")
    entry["result"].pop("global_sampling")
    entry["result"].pop("recommendations")
    jobs._write(run.id, "candidate_scoring", entry)
    rendered = functions["performance_results"](complete[0], "full_space", "recommended", context)
    assert rendered[4] == [] and "Refresh" in rendered[10]


@pytest.mark.parametrize(
    "score, good_count, bad_count, expected",
    [
        (0, 2, 5, 2 / 7),  # Rounding the split can change the prior from exactly 20%.
        (4.9415, 1, 4, 0.9722183757),
        (8.7992, 1, 4, 0.999397),
        (-10000, 1, 4, 0),
        (10000, 1, 4, 1),
    ],
)
def test_probability_uses_raw_score_and_actual_group_sizes(score, good_count, bad_count, expected):
    result = {
        "holes": [],
        "model": {"good_trials": list(range(good_count)), "bad_trials": list(range(bad_count))},
        "full_space_candidates": [
            {
                "id": "candidate",
                "kde_score": score,
                "relative_score": score - 8.7992,
                "configuration": {"x": 0.5},
            }
        ],
    }
    row = views.ranking_rows(result, "full_space", "all")[0]
    assert row["kde_score"] == score
    assert row["good_probability"] == pytest.approx(expected, abs=1e-6)
