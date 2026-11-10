"""Reference searches use real Optuna storage while leaving it unchanged."""

import copy

import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState, create_trial

from deepcave.utils.sensitivity_references import (
    dataset_identity,
    find_reference_trials,
    reference_key,
    reference_label,
    reference_objective,
    refresh_reference_objectives,
)


@pytest.fixture
def reference_database(tmp_path):
    database = tmp_path / "optuna.db"
    storage = f"sqlite:///{database}"

    def study(name, dataset="dataset-a", rate=0.25, state=TrialState.COMPLETE, trial_attrs=None):
        source = optuna.create_study(study_name=name, storage=storage, direction="maximize")
        if dataset is not None:
            source.set_user_attr("dataset_sha256", dataset)
        source.set_user_attr("model_family", "xgb")
        source.set_user_attr("model_seed", 42)
        source.add_trial(
            create_trial(
                params={"rate": rate, "depth": 3, "booster": "tree", "model_seed": 42},
                distributions={
                    "rate": FloatDistribution(0.01, 1),
                    "depth": IntDistribution(1, 10),
                    "booster": CategoricalDistribution(["tree", "linear"]),
                    "model_seed": IntDistribution(0, 100),
                },
                value=0.85 if state == TrialState.COMPLETE else None,
                state=state,
                user_attrs=trial_attrs or {},
            )
        )
        return source

    study("sensitivity")
    study("original-hpo")
    study("other-dataset", dataset="dataset-b")
    study("other-parameters", rate=0.5)
    study("unknown-dataset", dataset=None)
    study("failed", state=TrialState.FAIL)
    return database, study


@pytest.fixture
def reference_group():
    return {
        "id": "conditional-group",
        "config_id": "configuration",
        "parameters": {"rate": 0.25, "depth": 3, "booster": "tree"},
        "rows": [
            {
                "storage": "/source/optuna.db",
                "study_name": "sensitivity",
                "study_attrs": {"dataset_sha256": "dataset-a", "model_family": "xgb"},
                "user_attrs": {"model_seed": 1},
            }
        ],
    }


def test_reference_search_matches_dataset_and_typed_parameters_read_only(
    reference_database, reference_group
):
    database, _ = reference_database
    before = database.read_bytes()
    result = find_reference_trials(database, [reference_group], {"model": "trial:model_seed"})
    matches = result["matches"][reference_key(reference_group)]
    assert [trial["study_name"] for trial in matches] == ["original-hpo"]
    assert matches[0]["trial_number"] == 0
    assert matches[0]["model_seed"] == 42
    assert matches[0]["values"] == [0.85]
    assert "original-hpo · Trial 0" in reference_label(matches[0])
    assert result["missing_dataset"] == 0
    legacy = {
        key: value for key, value in matches[0].items() if key not in ("metrics", "user_attrs")
    }
    assert refresh_reference_objectives(legacy) == matches[0]
    with pytest.raises(ValueError, match="not found"):
        refresh_reference_objectives({**legacy, "trial_number": 999})
    assert database.read_bytes() == before


def test_reference_search_requires_dataset_and_keeps_configuration_identity(
    reference_database, reference_group
):
    database, _ = reference_database
    original = reference_key(reference_group)
    changed_mode = copy.deepcopy(reference_group)
    changed_mode.update(id="another-condition", config_id="another-objective", mode="combined")
    changed_mode["rows"][0]["user_attrs"]["model_seed"] = 7
    assert reference_key(changed_mode) == original
    unknown = copy.deepcopy(reference_group)
    unknown["rows"][0]["study_attrs"].pop("dataset_sha256")
    result = find_reference_trials(database, [unknown], {"model": "trial:model_seed"})
    assert result["missing_dataset"] == 1
    assert not any(result["matches"].values())
    reference_group["parameters"]["depth"] = 3.0
    result = find_reference_trials(database, [reference_group], {"model": "trial:model_seed"})
    assert not any(result["matches"].values())
    assert dataset_identity({"dataset_sha256": "a"}) == dataset_identity({"dataset_hash": "a"})


def test_reference_search_respects_model_and_trial_metadata(reference_database, reference_group):
    database, make_study = reference_database
    conflicting = make_study("other-model")
    conflicting.set_user_attr("model_family", "catboost")
    make_study(
        "trial-dataset",
        dataset=None,
        trial_attrs={"dataset_sha256": "dataset-a", "score": 0.7, "fold_scores": [0.6, 0.8]},
    )
    result = find_reference_trials(database, [reference_group], {"model": "trial:model_seed"})
    assert [trial["study_name"] for trial in result["matches"][reference_key(reference_group)]] == [
        "original-hpo",
        "trial-dataset",
    ]
    trial = result["matches"][reference_key(reference_group)][1]
    assert trial["metrics"] == {"metric:Objective0": 0.85}
    assert trial["user_attrs"]["score"] == 0.7
    assert "fold_scores" not in trial["user_attrs"]


def test_reference_objective_resolves_native_and_derived_metrics_without_substitution():
    group = {
        "context": {"objective": ["metric:score", "upper"]},
        "rows": [{"metric_info": {"metric:score": {"index": 1}}}],
    }
    assert reference_objective({"values": [0.1, 0.8]}, group) == 0.8
    trial = {"values": [0.9, 0.1], "metrics": {"metric:score": 0.9}, "user_attrs": {"score": 0.7}}
    assert reference_objective(trial, group) == 0.9
    group["context"]["objective"][0] = "user_attr:score"
    assert reference_objective(trial, group) == 0.7
    assert reference_objective({"values": [0.9]}, group) is None
    trial["user_attrs"]["score"] = float("nan")
    assert reference_objective(trial, group) is None
    group["context"]["objective"][0] = "metric:missing"
    assert reference_objective(trial, group) is None
