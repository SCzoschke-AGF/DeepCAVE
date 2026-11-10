"""Global sampler subsets preserve source data, stable identities and cache isolation."""

from copy import deepcopy
from threading import RLock
from types import SimpleNamespace

import pytest
from optuna.trial import TrialState

import deepcave
from deepcave.plugins import Plugin
from deepcave.plugins.static import _processing_inputs_key
from deepcave.runs.group import Group
from deepcave.runs.handler import OPTUNA_SAMPLER_ATTRIBUTE_CACHE_KEY, RunHandler
from deepcave.utils.cache import Cache
from deepcave.utils.candidates import candidate_data
from deepcave.utils.run_caches import RunCaches
from deepcave.utils.sampling import (
    MISSING_METHOD,
    SAMPLING_METHODS_CACHE_KEY,
    filtered_run,
    method_options,
    method_token,
    sampling_signature,
)

from tests.test_evaluators.test_sensitivity import make_run  # noqa: F401

MAPPING = "trial:sampling_method"


@pytest.fixture
def study(make_run):  # noqa: F811
    return make_run(
        [
            {"x": 0.25, "y": 1, "attrs": {"sampling_method": "random"}},
            {"x": 0.25, "y": 100, "attrs": {"sampling_method": "bo"}},
            {"x": 0.75, "y": 3, "attrs": {"sampling_method": "grid"}},
            {"x": 0.5, "y": 4},
            {"x": 0.5, "state": TrialState.RUNNING, "attrs": {"sampling_method": "bo"}},
        ]
    )


def test_mapping_options_do_not_assume_attribute_name(study):
    options = method_options([study], MAPPING)
    assert {option["value"] for option in options} == {
        method_token("random"),
        method_token("bo"),
        method_token("grid"),
        MISSING_METHOD,
    }
    assert method_options([study], None) == []
    assert method_token("missing") != MISSING_METHOD
    assert method_token(1) != method_token("1")
    for trial in study.history:
        trial.additional["user_attrs"]["custom_strategy"] = "targeted"
    assert {option["label"] for option in method_options([study], "trial:custom_strategy")} == {
        "targeted",
        "Missing metadata",
    }


def test_filtered_history_configs_bounds_lookup_and_candidates(study):
    original = deepcopy(study.meta)
    view = filtered_run(study, MAPPING, [method_token("random")])
    assert [trial.costs[0] for trial in view.history] == [1]
    assert view.get_trial((0, 0.0, -1)).costs[0] == 1
    assert set(view.configs) == {0} and view.get_highest_budget(0) == 0
    assert view.get_objective(0).upper == 1
    assert [row["trial_number"] for row in candidate_data(view)["rows"]] == [0]
    assert view.id == study.id and view.hash != study.hash
    assert study.meta == original and len(study.history) == 4
    assert study.get_trial((0, 0.0, -1)).costs[0] == 100
    assert filtered_run(study, MAPPING, []) is study
    assert filtered_run(study, None, [method_token("random")]) is study


def test_multiple_values_missing_metadata_and_no_matches(study):
    selected = [method_token("random"), method_token("grid")]
    assert len(filtered_run(study, MAPPING, selected).history) == 2
    missing = filtered_run(study, MAPPING, [MISSING_METHOD])
    assert [r.additional["optuna_trial_number"] for r in missing.history] == [3]
    empty = filtered_run(study, MAPPING, [method_token("unknown")])
    assert empty.history == [] and empty.configs == {} and empty.meta["optuna_candidates"] == []
    assert len(filtered_run(study, MAPPING, [method_token("bo")]).meta["optuna_candidates"]) == 2
    assert sampling_signature(MAPPING, selected) == sampling_signature(MAPPING, selected[::-1])


def test_study_mapping_and_group_rebuild_filter_before_duplicate_replacement(
    study, make_run  # noqa: F811
):  # noqa: F811
    study.meta["optuna_user_attrs"]["strategy"] = "random"
    assert len(filtered_run(study, "study:strategy", [method_token("random")]).history) == 4
    other = make_run([{"x": 0.25, "y": 200, "attrs": {"sampling_method": "bo"}}], name="other")
    group = Group("both", [study, other])
    view = filtered_run(group, MAPPING, [method_token("random")])
    assert isinstance(view, Group) and len(view.history) == 1
    assert view.history[0].costs[0] == 1
    assert view.get_original_run(view.history[0].config_id) is view.runs[0]
    assert len(view.runs[1].history) == 0
    assert view.id == group.id and view.hash != group.hash
    assert len(group.runs[0].history) == 4
    assert (
        filtered_run(SimpleNamespace(prefix="SMAC"), MAPPING, [method_token("random")]).prefix
        == "SMAC"
    )


def test_handler_supplies_one_filtered_view_to_all_analysis_access_paths(study):
    handler = object.__new__(RunHandler)
    handler.c = Cache(defaults={"groups": {}})
    handler._update_lock = RLock()
    handler.runs, handler.groups = {"study": study}, {"group": Group("group", [study])}
    handler.rc = SimpleNamespace(update=lambda run: False)
    handler.c.set(OPTUNA_SAMPLER_ATTRIBUTE_CACHE_KEY, value=MAPPING)
    handler.set_sampling_methods([method_token("random")])
    assert len(handler.get_run(study.id).history) == 1
    assert handler.get_run(study.id) is handler.get_run(study.id)
    assert len(handler.get_runs()[0].history) == 1
    assert len(handler.get_groups()[0].history) == 1
    assert handler.get_runs(apply_filter=False)[0] is study
    assert len(handler.get_sampling_method_options()) == 4
    handler.set_optuna_sampler_attribute("study:strategy")
    assert handler.c.get(SAMPLING_METHODS_CACHE_KEY) == []
    assert handler.get_run(study.id) is study


def test_caches_and_static_job_signatures_are_isolated(study, tmp_path, monkeypatch):
    cache = RunCaches(SimpleNamespace(CACHE_DIR=tmp_path / "cache", DEBUG=False))
    random = filtered_run(study, MAPPING, [method_token("random")])
    bo = filtered_run(study, MAPPING, [method_token("bo")])
    cache.set(study, "analysis", "same", value={"value": "all"})
    cache.set(random, "analysis", "same", value={"value": "random"})
    assert cache.get(bo, "analysis", "same") is None
    assert cache.get(study, "analysis", "same") == {"value": "all"}
    assert cache.get(random, "analysis", "same") == {"value": "random"}
    monkeypatch.setattr(
        deepcave, "run_handler", SimpleNamespace(get_run=lambda run_id: random), raising=False
    )
    plugin = SimpleNamespace(activate_run_selection=True, allow_additional_runs=False, inputs=[])
    inputs = {"run": {"value": study.id}}
    Plugin.get_selected_runs.__wrapped__(plugin, inputs)
    assert inputs["_sampling_filter"][study.id] == random._sampling_filter_key
    all_key = _processing_inputs_key(plugin, {})
    assert _processing_inputs_key(plugin, inputs) != all_key
