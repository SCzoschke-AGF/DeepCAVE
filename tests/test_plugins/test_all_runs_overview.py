"""All-run statistics, loading, and column selection."""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import optuna
import pytest
from ConfigSpace import ConfigurationSpace
from dash import Patch, dash_table, dcc, no_update
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

import deepcave
from deepcave.plugins.summary import all_runs_overview
from deepcave.plugins.summary.all_runs_overview import (
    AllRunsOverview,
    load_overview_runs,
    load_selected_statistics,
    summarize_runs,
    table_view,
)
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status
from deepcave.utils.optuna import get_optuna_snapshot_path


def column(source, name, statistic):
    return json.dumps([json.dumps([source, name]), statistic])


def make_run(tmp_path, name="run"):
    run = DeepCAVERun(
        name,
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0)}),
        objectives=[Objective("score", optimize="upper")],
    )
    run.path = tmp_path / name
    return run


def add(run, index, score, attrs=None, status=Status.SUCCESS, budget=1):
    run.add(
        config={"x": index / 10},
        costs=[score],
        seed=index,
        budget=budget,
        status=status,
        start_time=index,
        end_time=index + 2,
        additional={"user_attrs": attrs or {}},
    )


def test_statistics_include_only_successful_trials_across_budgets(tmp_path):
    run = make_run(tmp_path)
    add(run, 0, 1, {"accuracy": 2, "partial": 7, "boolean": True, "list": [1]})
    add(run, 1, 3, {"accuracy": 4, "partial": None, "nan": float("nan")}, budget=2)
    add(run, 2, 8, {"accuracy": 9, "partial": "text", "infinite": float("inf")})
    add(run, 3, 1000, {"accuracy": 1000, "failed_only": 1}, status=Status.FAILED)
    add(run, 4, 2000, {"accuracy": 2000}, status=Status.PRUNED)
    summary = summarize_runs([run])
    row = summary["rows"][0]
    assert row["successful"] == 3
    assert row["duration"] == 6
    for source, name, values in [
        ("objective", "score", [1, 3, 8]),
        ("user_attr", "accuracy", [2, 4, 9]),
    ]:
        assert row[column(source, name, "min")] == min(values)
        assert row[column(source, name, "max")] == max(values)
        assert row[column(source, name, "avg")] == pytest.approx(sum(values) / 3)
        assert row[column(source, name, "median")] == values[1]
        assert row[column(source, name, "std")] == pytest.approx((26 / 3) ** 0.5)
        assert row[column(source, name, "range")] == 7
    assert row[column("user_attr", "partial", "std")] == 0
    assert row[column("user_attr", "partial", "avg")] == 7
    labels = " ".join(item["name"] for item in summary["columns"])
    for excluded in ("boolean", "list", "nan", "infinite", "failed_only"):
        assert excluded not in labels


def test_heterogeneous_and_empty_runs_keep_missing_values_empty(tmp_path):
    first = make_run(tmp_path, "same-name")
    add(first, 0, 2, {"extra": 42})
    second = make_run(tmp_path / "other", "same-name")
    add(second, 0, 5)
    empty = make_run(tmp_path, "empty")
    add(empty, 0, 100, status=Status.FAILED)
    summary = summarize_runs([first, second, empty])
    assert len(summary["rows"]) == 3
    assert summary["rows"][0]["path"] != summary["rows"][1]["path"]
    extra = column("user_attr", "extra", "median")
    columns, rows = table_view(summary, [extra, "successful", "removed-column"])
    assert [item["id"] for item in columns] == ["run", extra, "successful"]
    assert rows[0][extra] == 42
    assert rows[1].get(extra) is None
    assert rows[2]["successful"] == 0
    assert rows[2]["duration"] is None
    assert rows[2][column("objective", "score", "min")] is None
    assert table_view(None, None)[1] == []


def test_promoted_user_attributes_appear_once_and_objective_collisions_stay_distinct(tmp_path):
    run = make_run(tmp_path)
    add(run, 0, 2, {"score": 9})
    row = summarize_runs([run])["rows"][0]
    assert row[column("objective", "score", "avg")] == 2
    assert row[column("user_attr", "score", "avg")] == 9
    run.meta["user_attr_objectives"] = {"score": 0}
    summary = summarize_runs([run])
    assert len(summary["columns"]) == 5 + 6
    assert summary["rows"][0][column("user_attr", "score", "avg")] == 2


def test_local_run_statistics_are_loaded_on_demand_with_missing_values(tmp_path):
    first = make_run(tmp_path, "first")
    second = make_run(tmp_path, "second")
    add(first, 0, 2, {"precision": 8})
    add(second, 0, 6)
    runs = {str(run.path): run for run in [first, second]}
    summary = summarize_runs(list(runs.values()), selected=[])
    key = column("user_attr", "precision", "avg")
    assert key not in summary["rows"][0]
    updated = load_selected_statistics(SimpleNamespace(update_run=runs.get), summary, [key])
    assert [row[key] for row in updated["rows"]] == [8, None]
    assert [row["successful"] for row in updated["rows"]] == [1, 1]


def test_loading_includes_available_directories_and_keeps_loaded_runs(tmp_path):
    existing = make_run(tmp_path, "loaded")
    new = make_run(tmp_path, "new")
    handler = SimpleNamespace(
        get_runs=Mock(return_value=[existing]),
        get_available_run_paths=lambda: {
            str(existing.path): "loaded",
            str(new.path): "new",
            "bad": "bad",
            "folder": "folder",
        },
        is_run=lambda path: path != "folder",
        update_run=Mock(side_effect=lambda path: new if path == str(new.path) else None),
    )
    runs, errors = load_overview_runs(handler)
    assert runs == [existing, new]
    handler.get_runs.assert_called_once_with(apply_filter=False)
    assert handler.update_run.call_count == 2
    assert errors == ["Could not load bad."]


def test_loads_all_database_studies_including_empty_studies_without_snapshots(tmp_path):
    database = tmp_path / "studies.db"
    storage = f"sqlite:///{database}"
    for name, score in [("first", 1), ("second", 3)]:
        study = optuna.create_study(storage=storage, study_name=name)
        study.set_user_attr("dataset", name)
        trial = optuna.trial.create_trial(
            value=score,
            params={"x": 0.5},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
            user_attrs={"precision": score * 2},
        )
        study.add_trial(trial)
    empty_study = optuna.create_study(storage=storage, study_name="empty")
    empty_study.set_user_attr("dataset", "empty-cohort")
    existing = make_run(tmp_path, "existing")
    # A previously imported study must be replaced, not added as a second row.
    existing.path = get_optuna_snapshot_path(database, "first", tmp_path)
    handler = SimpleNamespace(
        get_runs=lambda **_: [existing],
        get_available_run_paths=lambda: {},
        get_working_directory=lambda: tmp_path,
    )
    runs, errors = load_overview_runs(handler, str(database))
    assert len(runs) == 3
    summary = summarize_runs(runs)
    assert sorted(
        row[column("user_attr", "precision", "max")] for row in summary["rows"] if row["successful"]
    ) == [2, 6]
    assert not errors
    empty = next(row for row in summary["rows"] if row["run"] == "empty")
    assert empty["successful"] == 0 and empty["duration"] is None
    study_key = json.dumps(["study_attr", "dataset"])
    assert empty[study_key] == "empty-cohort"
    assert {row[study_key] for row in summary["rows"]} == {"first", "second", "empty-cohort"}
    assert not list(tmp_path.rglob("*.pkl"))
    assert all(row["metadata_url"].startswith("/runs/optuna/") for row in summary["rows"])


@pytest.mark.parametrize("changing_ranges", [False, True])
def test_database_overview_handles_conditional_and_changing_search_spaces(
    tmp_path, changing_ranges
):
    database = tmp_path / "dynamic.db"
    study = optuna.create_study(storage=f"sqlite:///{database}", study_name="dynamic")
    for index in range(3):
        params = {"x": 0.5}
        distributions = {
            "x": optuna.distributions.FloatDistribution(0, 1 + index if changing_ranges else 1)
        }
        if index:
            params["average_before_softmax"] = False
            distributions["average_before_softmax"] = optuna.distributions.CategoricalDistribution(
                [False]
            )
        study.add_trial(
            optuna.trial.create_trial(
                value=index + 1,
                params=params,
                distributions=distributions,
                user_attrs={"precision": 2 * index, "invalid": [1, 2]},
            )
        )
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.FAIL,
            user_attrs={"precision": 1000},
        )
    )
    handler = SimpleNamespace(
        get_runs=lambda **_: [],
        get_available_run_paths=lambda: {},
        get_working_directory=lambda: tmp_path,
    )
    runs, errors = load_overview_runs(handler, str(database))
    assert not errors and len(runs) == 1
    row = summarize_runs(runs)["rows"][0]
    assert row["successful"] == 3
    assert row[column("objective", "Objective0", "avg")] == 2
    assert row[column("user_attr", "precision", "avg")] == 2
    assert row[column("user_attr", "precision", "max")] == 4
    assert not list(tmp_path.rglob("*.pkl"))
    assert len(study.trials) == 4


def test_default_order_uses_first_trial_start_descending_including_failed_trials(tmp_path):
    database = tmp_path / "studies.db"
    for name, observations in [
        ("a-old", [(1, optuna.trial.TrialState.FAIL), (5, optuna.trial.TrialState.COMPLETE)]),
        ("z-new", [(3, optuna.trial.TrialState.COMPLETE)]),
        ("m-unstarted", []),
    ]:
        study = optuna.create_study(storage=f"sqlite:///{database}", study_name=name)
        for day, state in observations:
            trial = optuna.trial.create_trial(
                state=state, value=1 if state == optuna.trial.TrialState.COMPLETE else None
            )
            trial.datetime_start = datetime(2026, 1, day, 12)
            trial.datetime_complete = trial.datetime_start + timedelta(seconds=5)
            study.add_trial(trial)
    handler = SimpleNamespace(
        get_runs=lambda **_: [], get_available_run_paths=lambda: {},
        get_working_directory=lambda: tmp_path,
    )
    runs, errors = load_overview_runs(handler, str(database))
    assert not errors
    summary = summarize_runs(runs, selected=[])
    assert [row["run"] for row in summary["rows"]] == ["z-new", "a-old", "m-unstarted"]
    assert [row["successful"] for row in summary["rows"]] == [1, 1, 0]
    assert [row["duration"] for row in summary["rows"]] == [5, 5, None]
    columns, rows = table_view(summary, ["successful"])
    assert "start_datetime" not in {column["id"] for column in columns}
    assert "start_datetime" not in {column["id"] for column in summary["columns"]}
    assert rows[-1]["start_datetime"] is None


def components(node):
    if isinstance(node, Component):
        yield node
        yield from components(getattr(node, "children", []))
    elif isinstance(node, (list, tuple)):
        for child in node:
            yield from components(child)


def test_study_attributes_are_searchable_selectable_and_keep_run_values(tmp_path):
    first = make_run(tmp_path, "first")
    first.meta["optuna_user_attrs"] = {
        "dataset": "cohort-a", "seed": 42, "settings": {"enabled": True}, "missing": None,
    }
    add(first, 0, 2, {"dataset": 9})
    second = make_run(tmp_path, "second")
    summary = summarize_runs([first, second])
    key = json.dumps(["study_attr", "dataset"])
    columns, rows = table_view(summary, [key])
    assert columns[1]["name"] == "dataset"
    assert rows[0][key] == "cohort-a"
    assert rows[1].get(key) is None
    assert rows[0][json.dumps(["study_attr", "seed"])] == 42
    assert rows[0][json.dumps(["study_attr", "settings"])] == '{"enabled": true}'
    assert rows[0][json.dumps(["study_attr", "missing"])] is None
    assert rows[0][column("user_attr", "dataset", "avg")] == 9
    tree = all_runs_overview.column_choices(summary, "DATASET", [key])
    checklist = next(
        item for item in components(tree)
        if isinstance(item, dcc.Checklist) and item.id["metric"] == "study_attr"
    )
    assert checklist.options == [{"label": "dataset", "value": key}]
    assert checklist.value == [key]


def test_layout_and_callbacks_load_only_on_button_press(monkeypatch, tmp_path):
    callbacks = {}

    def callback(*args, **kwargs):
        def register(function):
            callbacks[function.__name__] = function
            return function

        return register

    handler = SimpleNamespace(
        get_runs=Mock(return_value=[make_run(tmp_path)]),
        get_available_run_paths=lambda: {},
    )
    monkeypatch.setattr(deepcave, "app", SimpleNamespace(callback=callback), raising=False)
    monkeypatch.setattr(deepcave, "c", SimpleNamespace(get=lambda _: None), raising=False)
    monkeypatch.setattr(deepcave, "queue", SimpleNamespace(), raising=False)
    monkeypatch.setattr(deepcave, "run_handler", handler, raising=False)
    plugin = object.__new__(AllRunsOverview)
    AllRunsOverview.register_callbacks.__wrapped__(plugin)
    layout = list(components(plugin()))
    assert not plugin.requires_run
    handler.get_runs.assert_not_called()
    table = next(item for item in layout if isinstance(item, dash_table.DataTable))
    assert table.sort_action == table.filter_action == "native"
    assert table.sort_mode == "multi"
    assert table.data == []
    assert table.row_selectable == "multi"
    assert table.selected_rows == []
    selection = next(item for item in layout if getattr(item, "id", None) == "all-runs-columns")
    assert selection.data == ["successful", column("objective", "Objective0", "max")]
    assert callbacks["show_studies"]("selected-studies") == (
        {"display": "none"}, {"display": "block"}
    )
    assert callbacks["show_studies"]("all-studies") == (
        {"display": "block"}, {"display": "none"}
    )
    assert callbacks["toggle_group_filter"](False) == (True, True, True)
    assert callbacks["toggle_group_filter"](True) == (False, False, False)
    with pytest.raises(PreventUpdate):
        callbacks["reset_grouping"](0)
    assert callbacks["reset_grouping"](1) == []
    assert callbacks["count_visible_runs"](None) == "0 runs"
    assert any(getattr(item, "id", None) == "all-runs-column-search" for item in layout)
    assert {item.id for item in layout if isinstance(item, dcc.Dropdown)} == {
        "all-runs-group-by", "all-runs-group-target",
    }
    run_style = table.style_cell_conditional[0]
    assert run_style["maxWidth"] == "260px"
    assert run_style["textOverflow"] == "ellipsis"
    with pytest.raises(PreventUpdate):
        callbacks["load_runs"](0, None)
    summary, message = callbacks["load_runs"](1, None)
    assert message == "1 runs loaded"
    assert callbacks["count_visible_runs"](summary["rows"]) == "1 run"
    style, chips = callbacks["show_group_chips"](
        summary, True, ["optimizer"], column("objective", "score", "max"), "max"
    )
    assert style == {"display": "flex"}
    assert [chip.children for chip in chips] == ["Optimizer", "score · max"]
    assert callbacks["show_group_chips"](
        summary, False, ["optimizer"], column("objective", "score", "max"), "max"
    ) == ({"display": "none"}, [])
    assert callbacks["show_group_chips"](
        summary, True, [], column("objective", "score", "max"), "max"
    ) == ({"display": "none"}, [])
    assert [item["label"] for item in summary["metrics"]].count("score (objective)") == 1
    groups, targets = callbacks["group_filter_options"](summary)
    assert {item["value"] for item in targets} < {item["value"] for item in groups}
    assert all(item["value"] != "run" for item in targets)
    columns, rows, tooltips = callbacks["select_columns"](summary, ["successful"])
    assert [item["id"] for item in columns] == ["run", "successful"]
    assert len(rows) == 1
    assert tooltips == [{"run": {"value": rows[0]["run"], "type": "text"}}]
    _, unchanged_rows, unchanged_tooltips = callbacks["select_columns"](
        summary, ["successful", "path"], current_rows=rows
    )
    assert unchanged_rows is unchanged_tooltips is no_update
    metric_key = column("objective", "score", "avg")
    enriched = {**summary, "rows": [{**rows[0], metric_key: 3.5}]}
    _, patched_rows, unchanged_tooltips = callbacks["select_columns"](
        enriched, ["successful", metric_key], current_rows=rows
    )
    assert isinstance(patched_rows, Patch)
    assert patched_rows.to_plotly_json()["operations"] == [
        {"operation": "Assign", "location": [0, metric_key], "params": {"value": 3.5}}
    ]
    assert unchanged_tooltips is no_update
    assert columns[1]["format"].to_plotly_json()["specifier"] == ".0f"
    assert not columns[0]["deletable"] and columns[1]["deletable"]
    open_metadata = callbacks["open_run_metadata"]
    visible_rows = [
        {"run": "second", "metadata_url": "/runs/optuna/second"},
        {"run": "first", "metadata_url": "/runs/optuna/first"},
    ]
    assert open_metadata({"row": 0, "column_id": "run"}, visible_rows) == "/runs/optuna/second"
    assert open_metadata({"row": 1, "column_id": "run"}, visible_rows) == "/runs/optuna/first"
    for cell, data in [
        (None, visible_rows),
        ({"row": 0, "column_id": "successful"}, visible_rows),
        ({"row": 0, "column_id": "run"}, [{"metadata_url": None}]),
    ]:
        with pytest.raises(PreventUpdate):
            open_metadata(cell, data)
    tree = callbacks["render_column_choices"](summary, "score", 1, columns)
    checklists = [item for item in components(tree) if isinstance(item, dcc.Checklist)]
    assert len(checklists) == 1
    checklist = checklists[0]
    assert [item["label"] for item in checklist.options] == [
        "min",
        "max",
        "avg",
        "median",
        "std",
        "range",
    ]
    monkeypatch.setattr(all_runs_overview, "ctx", SimpleNamespace(triggered_id=checklist.id))
    update = callbacks["update_column_choices"]
    median_key = column("objective", "score", "median")
    max_key = column("objective", "score", "max")
    selected = update([[median_key, max_key]], [checklist.id], [checklist.options], columns)
    assert selected == ["successful", median_key, max_key]
    updated_columns, _, _ = callbacks["select_columns"](summary, selected)
    assert (
        update([[median_key, max_key]], [checklist.id], [checklist.options], updated_columns)
        is no_update
    )
    assert update([[max_key]], [checklist.id], [checklist.options], updated_columns) == [
        "successful",
        max_key,
    ]
    # Search refreshes preserve checked statistics and unrelated columns.
    tree = callbacks["render_column_choices"](summary, "SCORE", 2, updated_columns)
    checklist = next(item for item in components(tree) if isinstance(item, dcc.Checklist))
    assert checklist.value == [max_key, median_key]
    empty = callbacks["render_column_choices"](summary, "not-found", 2, updated_columns)
    assert empty[0].children == "No matching columns."


def test_database_selection_loads_its_studies_without_the_load_button(monkeypatch, tmp_path):
    callbacks = {}

    def callback(*args, **kwargs):
        def register(function):
            callbacks[function.__name__] = function
            return function
        return register

    monkeypatch.setattr(deepcave, "app", SimpleNamespace(callback=callback), raising=False)
    monkeypatch.setattr(deepcave, "run_handler", SimpleNamespace(), raising=False)
    loader = Mock(return_value=([make_run(tmp_path)], []))
    monkeypatch.setattr(all_runs_overview, "load_overview_runs", loader)
    monkeypatch.setattr(all_runs_overview, "_database_revision", lambda *_: ["chosen.db", 1])
    AllRunsOverview.register_callbacks.__wrapped__(object.__new__(AllRunsOverview))
    summary, message = callbacks["load_runs"](0, {"database": "chosen.db", "error": None})
    assert message == "1 runs loaded"
    loader.assert_called_once_with(deepcave.run_handler, "chosen.db")
    assert summary["rows"][0]["id"] == str(tmp_path / "run")
    loader.reset_mock()
    summary, message = callbacks["load_runs"](1, {"error": "Invalid database"})
    assert summary["rows"] == []
    loader.assert_not_called()


def test_database_base_load_skips_trial_objects_statistics_and_duplicate_snapshots(monkeypatch, tmp_path):
    database = tmp_path / "studies.db"
    study = optuna.create_study(storage=f"sqlite:///{database}", study_name="study")
    study.set_user_attr("dataset", "cohort")
    study.set_metric_names(["score"])
    start = datetime(2026, 1, 1)
    for score, seconds in [(2, 1), (6, 2)]:
        trial = optuna.trial.create_trial(
            value=score,
            user_attrs={"precision": score * 2, "invalid": [1], "boolean": True},
        )
        trial.datetime_start = start
        trial.datetime_complete = start + timedelta(seconds=seconds, microseconds=123456)
        study.add_trial(trial)
    study.add_trial(
        optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL, user_attrs={"failed_only": 5})
    )
    snapshot = str(get_optuna_snapshot_path(database, "study", tmp_path))
    handler = SimpleNamespace(
        get_runs=lambda **_: [],
        get_working_directory=lambda: tmp_path,
        get_available_run_paths=lambda: {snapshot: "study"},
        update_run=Mock(),
        is_run=Mock(),
    )
    monkeypatch.setattr(optuna, "load_study", Mock(side_effect=AssertionError("No Optuna import")))
    samples = Mock(side_effect=AssertionError("No raw values during base load"))
    monkeypatch.setattr(all_runs_overview.OptunaStudyOverview, "metric_samples", samples)
    runs, errors = load_overview_runs(handler, str(database))
    summary = summarize_runs(runs, selected=[])
    assert not errors
    row = summary["rows"][0]
    assert row["successful"] == 2
    assert row["duration"] == pytest.approx(3.246912)
    assert row[json.dumps(["study_attr", "dataset"])] == "cohort"
    assert column("objective", "score", "avg") not in row
    assert summary["computed_columns"] == []
    assert {item["label"] for item in summary["metrics"]} == {
        "score (objective)",
        "Time (objective)",
        "precision (user attribute)",
    }
    assert runs[0].history == []
    samples.assert_not_called()
    handler.update_run.assert_not_called()
    handler.is_run.assert_not_called()


def test_selected_statistics_only_read_requested_metrics_and_reuse_computed_columns(
    monkeypatch, tmp_path
):
    database = tmp_path / "studies.db"
    study = optuna.create_study(storage=f"sqlite:///{database}", study_name="study")
    for score in [2, 6]:
        study.add_trial(optuna.trial.create_trial(value=score, user_attrs={"precision": score * 2}))
    handler = SimpleNamespace(
        get_runs=lambda **_: [],
        get_working_directory=lambda: tmp_path,
        get_available_run_paths=lambda: {},
        update_run=Mock(),
    )
    runs, _ = load_overview_runs(handler, str(database))
    summary = summarize_runs(runs, selected=[])
    original = all_runs_overview.OptunaStudyOverview.metric_samples
    reads = []

    def samples(run, metrics):
        reads.extend(metrics)
        return original(run, metrics)

    monkeypatch.setattr(all_runs_overview.OptunaStudyOverview, "metric_samples", samples)
    objective_avg = column("objective", "Objective0", "avg")
    precision_max = column("user_attr", "precision", "max")
    monkeypatch.setitem(
        all_runs_overview.STATISTICS, "std", Mock(side_effect=AssertionError("Not selected"))
    )
    first = load_selected_statistics(handler, summary, [objective_avg])
    assert first["rows"][0][objective_avg] == 4
    assert precision_max not in first["rows"][0]
    assert reads == [json.dumps(["objective", "Objective0"])]
    assert load_selected_statistics(handler, first, []) is first
    assert load_selected_statistics(handler, first, [objective_avg]) is first
    second = load_selected_statistics(handler, first, [objective_avg, precision_max])
    assert second["rows"][0][precision_max] == 12
    assert second["rows"][0][objective_avg] == 4
    assert reads == [
        json.dumps(["objective", "Objective0"]),
        json.dumps(["user_attr", "precision"]),
    ]
    assert second["rows"][0]["successful"] == 2
    handler.update_run.assert_not_called()


def test_overview_cache_is_reused_until_database_changes(monkeypatch, tmp_path):
    database = tmp_path / "studies.db"
    study = optuna.create_study(storage=f"sqlite:///{database}", study_name="study")
    study.add_trial(optuna.trial.create_trial(value=2))
    callbacks = {}

    def callback(*args, **kwargs):
        def register(function):
            callbacks[function.__name__] = function
            return function

        return register

    handler = SimpleNamespace(
        get_runs=lambda **_: [],
        get_working_directory=lambda: tmp_path,
        get_available_run_paths=lambda: {},
    )
    monkeypatch.setattr(deepcave, "app", SimpleNamespace(callback=callback), raising=False)
    monkeypatch.setattr(deepcave, "run_handler", handler, raising=False)
    loader = Mock(wraps=load_overview_runs)
    monkeypatch.setattr(all_runs_overview, "load_overview_runs", loader)
    AllRunsOverview.register_callbacks.__wrapped__(object.__new__(AllRunsOverview))
    discovery = {"database": str(database)}
    load = callbacks["load_runs"]
    base_columns = ["successful", "duration"]
    first, message = load(1, discovery, base_columns, None)
    assert message == "1 runs loaded"
    assert first["rows"][0]["successful"] == 1
    assert load(2, discovery, base_columns, first)[0] is no_update
    assert loader.call_count == 1
    key = column("objective", "Objective0", "avg")
    assert callbacks["load_statistics"](base_columns, False, None, key, first) is no_update
    hidden_target = callbacks["load_statistics"](base_columns, True, ["optimizer"], key, first)
    assert hidden_target["rows"][0][key] == 2
    assert loader.call_count == 1
    enriched = callbacks["load_statistics"]([key], data=first)
    assert enriched["rows"][0][key] == 2
    assert callbacks["load_statistics"]([key], data=enriched) is no_update
    assert loader.call_count == 1
    study.add_trial(optuna.trial.create_trial(value=6))
    second, _ = load(3, discovery, base_columns, enriched, True, ["optimizer"], key)
    assert second["rows"][0]["successful"] == 2
    assert second["rows"][0][key] == 4
    assert loader.call_count == 2


@pytest.mark.parametrize("statistic, expected", [
    ("max", ["a-high", "b-high"]), ("min", ["b-low", "a-low"]),
])
def test_group_filter_keeps_extreme_rows_with_stable_ties_and_missing_values(statistic, expected):
    group = json.dumps(["study_attr", "ModelFamily"])
    target = column("objective", "Objective0", "max")
    rows = [
        {"id": name, group: family, target: value}
        for name, family, value in [
            ("a-high", "a", 9),
            ("b-low", "b", 2),
            ("a-low", "a", 1),
            ("b-high", "b", 5),
            ("a-tie", "a", 9),
            ("missing-group", None, 100),
            ("missing-value", "c", None),
            ("nan", "c", float("nan")),
            ("infinite", "c", float("inf")),
            ("boolean", "c", True),
        ]
    ]
    data = {
        "rows": rows,
        "columns": [{"id": target, "type": "numeric"}],
        "computed_columns": [target],
    }
    filtered = all_runs_overview.group_filter_rows(data, True, [group], target, statistic)
    assert [row["id"] for row in filtered] == expected
    assert len(data["rows"]) == 10
    assert all_runs_overview.group_filter_rows(data, False, [group], target, statistic) is rows
    assert all_runs_overview.group_filter_rows(data, True, None, target, statistic) is rows
    pending = {**data, "computed_columns": []}
    assert all_runs_overview.group_filter_rows(pending, True, [group], target, statistic) is rows


@pytest.mark.parametrize("statistic, expected", [
    ("max", ["a-x-high", "a-y-high", "b-x-high"]),
    ("min", ["a-x-low", "a-y-low", "b-x-low"]),
])
def test_group_filter_groups_by_column_combinations(statistic, expected):
    rows = [
        {"id": name, "family": family, "dataset": dataset, "successful": value}
        for name, family, dataset, value in [
            ("a-x-high", "a", "x", 9), ("a-x-low", "a", "x", 1),
            ("a-y-high", "a", "y", 7), ("a-y-low", "a", "y", 3),
            ("b-x-high", "b", "x", 8), ("b-x-low", "b", "x", 2),
            ("missing", "a", None, 99),
        ]
    ]
    data = {"rows": rows, "columns": [{"id": "successful", "type": "numeric"}]}
    filtered = all_runs_overview.group_filter_rows(
        data, True, ["family", "dataset"], "successful", statistic
    )
    assert [row["id"] for row in filtered] == expected
    assert all_runs_overview.group_filter_rows(data, True, [], "successful", statistic) is rows
