"""Metadata pages read source studies without converting their search spaces."""

from types import SimpleNamespace

import dash_bootstrap_components as dbc
import optuna
import pytest
from dash import html
from dash.development.base_component import Component

from deepcave.layouts.optuna_metadata import (
    optuna_metadata_page,
    read_study_metadata,
    study_metadata_url,
)


@pytest.fixture
def source(tmp_path):
    database = tmp_path / "metadata.db"
    study = optuna.create_study(storage=f"sqlite:///{database}", study_name="study / <name>")
    study.set_user_attr("model", {"family": "tree", "nested": [1, 2]})
    study._storage.set_study_system_attr(study._study_id, "custom", "original")
    study.add_trial(
        optuna.trial.create_trial(
            value=0.7,
            params={"depth": 3},
            distributions={"depth": optuna.distributions.IntDistribution(1, 5)},
            user_attrs={"precision": 0.8, "text": "<script>example</script>"},
            system_attrs={"reason": "source"},
            intermediate_values={1: 0.2, 2: 0.7},
        )
    )
    # A changing search space and non-successful trial must both remain inspectable.
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.FAIL,
            params={"depth": 7, "optional": False},
            distributions={
                "depth": optuna.distributions.IntDistribution(1, 10),
                "optional": optuna.distributions.CategoricalDistribution([False]),
            },
            user_attrs={"failure": "details"},
        )
    )
    study.ask()
    return database, study


def test_metadata_keeps_study_and_all_trial_information(source):
    database, study = source
    metadata = read_study_metadata(database, study.study_name)
    assert metadata["study"]["name"] == study.study_name
    assert metadata["study"]["directions"] == ["MINIMIZE"]
    assert metadata["study"]["trial_states"] == {"COMPLETE": 1, "FAIL": 1, "RUNNING": 1}
    assert metadata["user_attrs"] == {"model": {"family": "tree", "nested": [1, 2]}}
    assert metadata["system_attrs"] == {"custom": "original"}
    complete, failed, running = metadata["trials"]
    assert complete["values"] == [0.7]
    assert complete["params"] == {"depth": 3}
    assert complete["distributions"]["depth"]["attributes"]["high"] == 5
    assert complete["user_attrs"]["precision"] == 0.8
    assert complete["system_attrs"] == {"reason": "source"}
    assert complete["intermediate_values"] == {1: 0.2, 2: 0.7}
    assert complete["datetime_start"] is not None
    assert complete["datetime_complete"] is not None
    assert complete["duration_seconds"] >= 0
    assert failed["params"] == {"depth": 7, "optional": False}
    assert failed["user_attrs"] == {"failure": "details"}
    assert running["datetime_complete"] is None
    assert running["duration_seconds"] is None


def test_metadata_page_resolves_database_identity_and_displays_full_values(source, tmp_path):
    database, study = source
    url = study_metadata_url(database, study.study_name)
    assert url != study_metadata_url(tmp_path / "other.db", study.study_name)
    handler = SimpleNamespace(get_working_directory=lambda: tmp_path)
    page = optuna_metadata_page(url.rsplit("/", 1)[-1], handler, str(database))
    assert page[1].children == study.study_name
    attrs = page[5]
    assert isinstance(attrs, dbc.Table)
    model_row = attrs.children[1].children[0]
    assert model_row.children[0].children == "model"
    nested = model_row.children[1].children
    assert isinstance(nested, dbc.Table)
    assert nested.children[0].children[0].children[0].children == "family"
    assert nested.children[0].children[0].children[1].children == "tree"
    assert nested.children[0].children[1].children[1].children == "1, 2"
    trials = page[-1].children
    assert len(trials) == 3
    assert isinstance(trials[0].children[1], dbc.Table)

    def text(node):
        if isinstance(node, Component):
            assert not isinstance(node, html.Script)
            return text(getattr(node, "children", []))
        if isinstance(node, list):
            return " ".join(text(child) for child in node)
        return str(node)

    assert "<script>example</script>" in text(trials[0])
    missing = optuna_metadata_page("unknown", handler, str(database))
    assert "not available" in missing[-1].children
    no_database = optuna_metadata_page(url.rsplit("/", 1)[-1], handler, None)
    assert "Select an Optuna database" in no_database[-1].children
