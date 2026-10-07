from pathlib import Path
from types import SimpleNamespace

import deepcave
from deepcave.plugins import Plugin, _should_initialize_inputs


class MultiRunPlugin(Plugin):
    id = "multi_run_test"
    name = "Multi Run Test"
    allow_additional_runs = True

    def __init__(self, candidates):
        self.candidates = candidates
        super().__init__()

    def get_additional_run_candidates(self, active_run):
        return self.candidates


def test_remounted_non_persistent_plugin_hydrates_cached_inputs() -> None:
    previous_inputs = {"cluster_count": {"value": "1"}}

    assert _should_initialize_inputs(False, previous_inputs, "on-page-load") is True
    assert _should_initialize_inputs(False, previous_inputs, None) is True
    assert _should_initialize_inputs(False, previous_inputs, "cluster-count") is False
    assert _should_initialize_inputs(True, previous_inputs, "on-page-load") is False


def test_multi_run_selection_keeps_global_run_first_and_sanitizes_additions(monkeypatch):
    active = SimpleNamespace(id="active")
    additional = SimpleNamespace(id="additional")
    runs = {run.id: run for run in [active, additional]}
    monkeypatch.setattr(
        deepcave,
        "run_handler",
        SimpleNamespace(get_run=lambda run_id: runs[run_id]),
        raising=False,
    )
    plugin = MultiRunPlugin([additional])

    selected = Plugin.get_selected_runs.__wrapped__(
        plugin,
        {
            "run": {"value": "active"},
            "additional_runs": {"value": ["additional", "additional", "removed", "active"]},
        },
    )

    assert [run.id for run in selected] == ["active", "additional"]


def test_run_option_omits_parent_directory_and_keeps_full_name_searchable():
    run_name = "catboost_data-ccffacbaefb5_phase-b69dc7f3132e"
    run = SimpleNamespace(
        id="catboost",
        name=run_name,
        path=Path("/experiments/deepcave_optuna") / run_name,
    )

    inputs = Plugin.load_run_inputs.__wrapped__([run], [], lambda _: True)

    assert inputs["run"]["options"] == [
        {
            "label": run_name,
            "value": "catboost",
            "disabled": False,
            "title": run_name,
            "search": run_name,
        }
    ]
