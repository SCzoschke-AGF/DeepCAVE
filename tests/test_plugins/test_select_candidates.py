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
    rows, columns = table_data(data, ["metric:Objective0", "user_attr:score"], ["COMPLETE"])
    assert len(data["rows"]) == 4 and len(rows) == 3
    assert {c["id"] for c in columns} >= {
        "trial_number",
        "trial_id",
        "duration",
        "state",
        "study_name",
        "param:rate",
        "param:depth",
        "param:model",
        "user_attr:score",
    }
    assert central_objective(data, {"value": "user_attr:score"}) == "user_attr:score"
    assert len(table_data(data, [], ["RUNNING"])[0]) == 1


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
    assert list(figure.data[1].x) == [1, 2, 3]
    figure = scatter_figure(data["rows"], "metric:Objective0", True, "lower")
    assert list(figure.data[1].x) == [1]
    data["rows"][0]["duration"] = None
    data["rows"][1]["metrics"]["metric:Objective0"] = float("nan")
    figure = scatter_figure(data["rows"], "metric:Objective0")
    assert list(figure.data[0].x) == [3]


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


def test_toml_preserves_native_ids_across_detached_snapshots(tmp_path):
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
    assert len(exported["selected_trials"]) == 2
    assert exported["selected_trials"][0] == {
        "study_name": source.study_name,
        "study_id": source._study_id,
        "trial_id": source.trials[0]._trial_id,
        "trial_number": 0,
        "storage": str(database),
    }
    # Offline analysis/export never queries the source database again.
    database.rename(tmp_path / "away.db")
    assert selected_trials_toml(data["rows"])
    data["rows"][0]["trial_id"] = None
    with pytest.raises(ValueError, match="Refresh"):
        selected_trials_toml(data["rows"])


def test_legacy_detached_snapshot_never_exports_remapped_trial_ids(tmp_path):
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
    with pytest.raises(ValueError, match="Original Optuna IDs"):
        selected_trials_toml([row])
