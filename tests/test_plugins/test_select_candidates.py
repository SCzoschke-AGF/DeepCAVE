"""Candidate identity, typed comparison axes, filtering and native Optuna references."""

import pickle
from datetime import datetime, timedelta

import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState, create_trial

from deepcave.config import Config
from deepcave.plugins.select_candidates import SelectCandidates, table_data
from deepcave.runs.converters.optuna import OptunaRun
from deepcave.utils.candidates import (
    candidate_data,
    central_objective,
    parallel_figure,
    scatter_figure,
    selected_trials_toml,
)
from deepcave.utils.optuna import create_optuna_snapshot

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@pytest.fixture
def studies(tmp_path):
    runs = []
    for index in range(2):
        study = optuna.create_study(study_name=f"study-{index}", direction="maximize")
        params = {
            "rate": FloatDistribution(0.001, 1, log=True),
            "depth": IntDistribution(1, 10),
            "model": CategoricalDistribution(["a", "b", None]),
        }
        if index:
            params["extra"] = FloatDistribution(0, 1)
        for number in range(3):
            values = {
                "rate": 10 ** (-3 + number),
                "depth": number + 1,
                "model": ["a", "b", None][number],
            }
            if index:
                values["extra"] = 0.5
            trial = create_trial(
                params=values,
                distributions=params,
                value=number + index,
                user_attrs={"score": 10 - number},
            )
            trial.datetime_start = datetime(2025, 1, 1)
            trial.datetime_complete = trial.datetime_start + timedelta(seconds=number + 1)
            study.add_trial(trial)
        study.add_trial(create_trial(state=TrialState.RUNNING))
        path = tmp_path / f"study-{index}"
        path.mkdir()
        with (path / "study.pkl").open("wb") as file:
            pickle.dump(study, file)
        runs.append(OptunaRun.from_path(path))
    return runs


def test_registration_and_complete_table(studies):
    assert "select_candidates" in [p.id for p in Config().PLUGINS["Configuration Analyses"]]
    assert SelectCandidates.check_run_compatibility(studies[0])
    data = candidate_data(studies[0])
    rows, columns = table_data(data, ["metric:Objective0", "user_attr:score"])
    assert len(data["rows"]) == 4 and len(rows) == 3
    assert {c["id"] for c in columns} >= {
        "trial_number",
        "duration",
        "study_name",
        "param:rate",
        "param:depth",
        "param:model",
        "user_attr:score",
    }
    assert central_objective(data, {"value": "user_attr:score"}) == "user_attr:score"
    assert data["objectives"]["user_attr:score"]["label"] == "score"
    assert next(column["name"] for column in columns if column["id"] == "user_attr:score") == "score"
    assert scatter_figure(data["rows"], "user_attr:score").layout.yaxis.title.text == "score"
    parallel, _ = parallel_figure(data["rows"], "user_attr:score")
    assert parallel.data[0].dimensions[-2].label == "Objective · score"
    assert {"trial_id", "state"}.isdisjoint(c["id"] for c in columns)
    assert all(row["state"] == "COMPLETE" for row in rows)
    assert len(table_data(data, [])[0]) == 3


def test_candidate_labels_use_global_names_for_legacy_objectives(studies):
    run = studies[0]
    index = run.meta["user_attr_objectives"]["score"]
    run.meta["objectives"][index]["name"] = "user_attr:score"
    data = candidate_data(run)
    assert data["objectives"]["user_attr:score"]["label"] == "score"
    _, columns = table_data(data, ["user_attr:score"])
    assert next(column["name"] for column in columns if column["id"] == "user_attr:score") == "score"
    assert scatter_figure(data["rows"], "user_attr:score").layout.yaxis.title.text == "score"
    figure, _ = parallel_figure(data["rows"], "user_attr:score")
    assert figure.data[0].dimensions[-2].label == "Objective · score"


def test_identity_is_study_and_storage_scoped(studies):
    first, second = map(candidate_data, studies)
    assert first["rows"][0]["trial_number"] == second["rows"][0]["trial_number"] == 0
    assert first["rows"][0]["id"] != second["rows"][0]["id"]
    studies[0].meta["optuna_storage_identity"] = "another-database"
    assert candidate_data(studies[0])["rows"][0]["id"] != first["rows"][0]["id"]


def test_native_metrics_match_by_name_and_not_by_study_local_index(studies):
    studies[1].meta["objectives"][0]["name"] = "Different metric"
    first, second = map(candidate_data, studies)
    key = central_objective(first, {"value": "objective:0"})
    assert key not in second["rows"][0]["metrics"]
    assert central_objective(second, {"value": "objective:0"}) == "metric:Different metric"
    assert len(scatter_figure(first["rows"] + second["rows"], key).data[0].x) == 3


def test_scatter_excludes_invalid_values_and_respects_pareto_direction(studies):
    data = candidate_data(studies[0])
    figure = scatter_figure(data["rows"], "metric:Objective0", True)
    assert list(figure.data[0].x) == [1, 2, 3]
    assert figure.data[0].hoverinfo == "none"
    assert figure.data[0].hovertemplate is None
    assert list(figure.data[1].x) == [1, 2, 3]
    figure = scatter_figure(data["rows"], "metric:Objective0", True, "lower")
    assert list(figure.data[1].x) == [1]
    data["rows"][0]["duration"] = None
    data["rows"][1]["metrics"]["metric:Objective0"] = float("nan")
    figure = scatter_figure(data["rows"], "metric:Objective0")
    assert list(figure.data[0].x) == [3]


@pytest.mark.parametrize("direction, expected", [("upper", 2), ("lower", 0)])
def test_scatter_elites_follow_direction_and_visible_valid_rows(studies, direction, expected):
    rows = candidate_data(studies[0])["rows"]
    figure = scatter_figure(rows, "metric:Objective0", True, direction, True, 0.05)
    trace = figure.data[0]
    assert [item[2] for item in trace.customdata if item[4]] == [expected]
    assert list(trace.marker.color) == ["#737373"] * len(trace.x)
    assert list(trace.marker.size) == [6] * len(trace.x)
    assert list(trace.marker.opacity) == [1] * len(trace.x)
    assert figure.layout.xaxis.range[0] < expected + 1 < figure.layout.xaxis.range[1]
    assert figure.layout.yaxis.range[0] < expected < figure.layout.yaxis.range[1]
    assert not (figure.layout.yaxis.range[0] <= 2 - expected <= figure.layout.yaxis.range[1])
    assert figure.layout.xaxis.autorange is False
    assert figure.layout.yaxis.autorange is False
    assert len(figure.data) == 2  # Pareto remains independently available.
    filtered = scatter_figure(
        rows[1:3], "metric:Objective0", direction=direction, show_elites=True, elite_fraction=0.5
    )
    assert [item[2] for item in filtered.data[0].customdata if item[4]] == [
        2 if direction == "upper" else 1
    ]
    disabled = scatter_figure(rows, "metric:Objective0")
    assert not any(item[4] for item in disabled.data[0].customdata)
    assert disabled.layout.xaxis.autorange is True
    assert disabled.layout.yaxis.autorange is True
    assert disabled.layout.uirevision != figure.layout.uirevision
    wider = scatter_figure(
        rows, "metric:Objective0", direction=direction, show_elites=True, elite_fraction=0.5
    )
    assert wider.layout.uirevision != figure.layout.uirevision
    elites = [item[2] for item in wider.data[0].customdata if item[4]]
    assert all(wider.layout.xaxis.range[0] < i + 1 < wider.layout.xaxis.range[1] for i in elites)
    assert all(wider.layout.yaxis.range[0] < i < wider.layout.yaxis.range[1] for i in elites)
    assert not scatter_figure([], "metric:Objective0", show_elites=True).data


def test_scatter_elites_require_consistent_native_directions(studies):
    rows = candidate_data(studies[0])["rows"][:3]
    rows[0]["metric_info"] = {"metric:Objective0": {"optimize": "lower", "label": "Objective0"}}
    figure = scatter_figure(rows, "metric:Objective0", show_elites=True)
    assert not any(item[4] for item in figure.data[0].customdata)
    assert "different directions" in figure.layout.annotations[0].text


def test_parallel_axes_log_categories_missing_and_color_consistency(studies):
    first, second = map(candidate_data, studies)
    rows = first["rows"][:3] + second["rows"][:1]
    figure, note = parallel_figure(rows, "metric:Objective0", ["extra", "rate", "model", "depth"])
    trace = figure.data[0]
    extra, rate, model, depth, objective, duration = trace.dimensions
    assert extra.label == "extra"
    assert "Missing" in extra.ticktext and list(extra.values[:3]) != [0, 0, 0]
    assert "Missing" in note
    assert list(rate.values[:3]) == [-3, -2, -1]
    assert "log" in rate.label
    assert set(model.ticktext) == {'"a"', '"b"', "null"}
    assert list(depth.values[:3]) == [1, 2, 3]
    assert "Objective" in objective.label and duration.label == "Duration (s)"
    assert trace.line.colorscale[0][1] == rows[0]["color"]
    assert figure.layout.meta["rows"][0]["id"] == rows[0]["id"]
    hidden, _ = parallel_figure(rows, "metric:Objective0", [])
    assert len(hidden.data[0].dimensions) == 2


def test_parallel_marks_incompatible_scales_as_categories(studies):
    rows = [candidate_data(run)["rows"][0] for run in studies]
    rows[1]["distributions"]["rate"]["log"] = False
    figure, note = parallel_figure(rows, "metric:Objective0", ["rate"])
    assert "category" in figure.data[0].dimensions[0].label
    assert "different parameter types/scales" in note


def test_parallel_empty_state():
    figure, note = parallel_figure([], "metric:loss")
    assert not figure.data
    assert "Select trials" in figure.layout.annotations[0].text


def test_toml_exports_parameters_across_detached_snapshots(tmp_path):
    database = tmp_path / "studies.db"
    storage = f"sqlite:///{database}"
    for name in ("first", 'second "study"'):
        study = optuna.create_study(storage=storage, study_name=name)
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=2)
    source = optuna.load_study(storage=storage, study_name='second "study"')
    snapshot = create_optuna_snapshot(database, source.study_name, tmp_path)
    run = OptunaRun.from_path(snapshot.run_path)
    data = candidate_data(run)
    assert data["rows"][0]["trial_id"] == source.trials[0]._trial_id
    assert data["rows"][0]["trial_id"] != data["rows"][0]["trial_number"]
    direct = tmp_path / "direct"
    direct.mkdir()
    with (direct / "study.pkl").open("wb") as file:
        pickle.dump(source, file)
    direct_data = candidate_data(OptunaRun.from_path(direct))
    assert direct_data["rows"][0]["id"] == data["rows"][0]["id"]
    exported = tomllib.loads(selected_trials_toml(data["rows"] + data["rows"]))
    assert exported["source_optuna_study_id"] == source._study_id
    assert exported["source_study_name"] == source.study_name
    assert len(exported["trials"]) == 2
    assert exported["trials"] == [trial.params for trial in source.trials]
    # Offline analysis/export never queries the source database again.
    database.rename(tmp_path / "away.db")
    assert selected_trials_toml(data["rows"])
    data["rows"][0]["trial_id"] = None
    assert tomllib.loads(selected_trials_toml(data["rows"]))["trials"] == exported["trials"]
    data["rows"][0]["study_id"] = None
    with pytest.raises(ValueError, match="Refresh"):
        selected_trials_toml(data["rows"])


def test_toml_includes_trials_and_source_metadata_without_selection():
    row = {
        "id": "candidate-1",
        "study_id": 41,
        "study_name": 'source "study"',
        "trial_id": 123,
        "trial_number": 0,
        "storage": "optuna.db",
        "parameters": {"max_depth": 9, "subsample": 0.632, "criterion": "gini", "bootstrap": True},
    }
    exported = tomllib.loads(selected_trials_toml([row]))
    assert exported["source_optuna_study_id"] == 41
    assert exported["source_study_name"] == row["study_name"]
    assert "selection" not in exported
    assert exported["trials"] == [row["parameters"]]

    other = {**row, "id": "candidate-2", "study_id": 42, "study_name": "other"}
    mixed = tomllib.loads(selected_trials_toml([row, other]))
    assert "selection" not in mixed
    assert "source_optuna_study_id" not in mixed
    assert "source_study_name" not in mixed
    assert mixed["trials"] == [row["parameters"], other["parameters"]]


def test_legacy_detached_snapshot_exports_parameters_without_trial_ids(tmp_path):
    study = optuna.create_study(study_name="legacy")
    study.add_trial(create_trial(value=1))
    study._deepcave_study_id = 15
    study._deepcave_sampler_name = "Unknown sampler"
    path = tmp_path / "legacy"
    path.mkdir()
    with (path / "study.pkl").open("wb") as file:
        pickle.dump(study, file)
    row = candidate_data(OptunaRun.from_path(path))["rows"][0]
    assert row["trial_id"] is None and row["trial_number"] == 0
    exported = tomllib.loads(selected_trials_toml([row]))
    assert exported["source_optuna_study_id"] == 15
    assert exported["trials"] == [row["parameters"]]
