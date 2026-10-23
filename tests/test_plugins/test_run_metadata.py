"""Run metadata keeps the existing metrics and full source values accessible."""

from copy import deepcopy

import pytest
from ConfigSpace import ConfigurationSpace, EqualsCondition, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
)
from dash.development.base_component import Component

from deepcave.plugins.summary.overview import Overview
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.group import Group
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status


def text(node):
    if isinstance(node, Component):
        return text(getattr(node, "children", []))
    if isinstance(node, (list, tuple)):
        return " ".join(text(child) for child in node)
    return str(node)


def rows(wrapper):
    return [row.children for row in wrapper.children.children[1].children]


@pytest.fixture
def run(tmp_path):
    study = DeepCAVERun(
        "metadata",
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0)}),
        objectives=[Objective("score", optimize="upper"), Objective("loss")],
    )
    study.path = tmp_path
    (tmp_path / "history.jsonl").touch()
    study.meta.update({"optuna_sampler": "Unknown sampler", "custom": "full-value-" * 30})
    return study


def add(run, x=0.1, costs=None, seed=0, budget=1, status=Status.SUCCESS, additional=None):
    run.add(
        config={"x": x},
        costs=costs or [0.5, 0.8],
        seed=seed,
        budget=budget,
        end_time=1124225,
        status=status,
        additional=additional,
    )


def test_layout_registers_only_compact_sections():
    registered = []
    layout = Overview.get_output_layout(lambda name, *_: registered.append(name) or name)
    assert registered == ["card", "meta", "objectives", "status_details", "configspace"]
    assert "Quick Information" not in text(layout)
    assert "Statuses" not in text(layout)
    assert "Failed Trials" in text(layout)


def test_metrics_preserve_incumbents_bounds_and_source_values(run):
    add(run, costs=[0.123456789, 0.9], seed=0)
    add(run, costs=[0.523456789, 0.7], seed=1)
    add(run, x=0.2, costs=[0.95, 0.2], budget=0.5)
    before = deepcopy(run.meta)
    kpis, meta, objectives, failures, _ = Overview.load_outputs(run)
    assert "312.3 h" in text(kpis)
    assert "Total Configurations" in text(kpis)
    assert "0.32346 ± 0.2" in text(kpis[0])
    objective_rows = rows(objectives)
    assert text(objective_rows[0][1]) == "0.12346"
    assert text(objective_rows[0][2]) == "0.95"
    assert text(objective_rows[0][3]) == "0.32346 ± 0.2"
    assert text(objective_rows[1][3]) == "0.8 ± 0.1"
    assert objective_rows[0][1].children.title == "0.123456789"
    assert any(cell.children.title == run.meta["custom"] for _, cell in rows(meta))
    assert "Unknown sampler" in text(meta)
    assert "Budgets" not in text(meta)
    assert "Seeds" not in text(meta)
    assert "No failed trials." in text(failures)
    assert run.meta == before


def test_configuration_count_is_omitted_when_equal_to_trials(run):
    add(run)
    assert "Total Configurations" not in text(Overview.load_outputs(run)[0])


@pytest.mark.parametrize("status", [Status.FAILED, Status.PRUNED, Status.TIMEOUT, Status.UNKNOWN])
def test_non_success_statuses_and_tracebacks_remain_available(run, status):
    add(run)
    add(run, x=0.2, status=status, additional={"traceback": "full traceback\nlast line"})
    add(run, x=0.3, status=Status.CRASHED)
    kpis, _, _, failures, _ = Overview.load_outputs(run)
    assert text(kpis[2]).endswith("1")
    assert text(kpis[3]).endswith("2")
    assert status.to_text() in text(failures)
    assert "full traceback\nlast line" in text(failures)
    assert "No traceback available." in text(failures)
    assert "Budget" not in text(failures)
    assert "Seed" not in text(failures)


@pytest.mark.parametrize("failed_only", [False, True])
def test_missing_incumbent_does_not_break_page(run, failed_only):
    if failed_only:
        add(run, status=Status.FAILED)
    rendered = Overview.load_outputs(run)
    assert "Unavailable" in text(rendered[0])
    assert "Unavailable" in text(rendered[2])
    if not failed_only:
        assert "0.0 s" in text(rendered[0])
        assert "—" in text(rendered[2])


def test_objective_names_and_non_numeric_search_spaces(run):
    run.meta["objectives"][0]["name"] = "user_attr:precision_at_recall5_mean"
    space = ConfigurationSpace()
    parent = CategoricalHyperparameter("model", ["a", "b"])
    child = UniformFloatHyperparameter("rate", 0.000123456789, 0.1, log=True)
    space.add(
        [parent, child, OrdinalHyperparameter("size", [1, 4, 16]), Constant("constant", "fixed")]
    )
    space.add(EqualsCondition(child, parent, "a"))
    space.add(ForbiddenEqualsClause(parent, "b"))
    run.configspace = space
    _, _, objectives, _, configspace = Overview.load_outputs(run)
    assert "Precision @ Recall 5% (mean)" in text(objectives)
    assert rows(objectives)[0][0].children.title == "user_attr:precision_at_recall5_mean"
    assert "[0.00012346, 0.1]" in text(configspace)
    assert "a, b" in text(configspace)
    assert "1, 4, 16" in text(configspace)
    assert "fixed" in text(configspace)
    assert "Yes" in text(configspace)
    assert "Conditions" in text(configspace)
    assert "Forbidden Clauses" in text(configspace)
    assert space["rate"].lower == 0.000123456789


def test_group_retains_maximum_runtime_semantics(run):
    add(run)
    group = Group("group", [run])
    rendered = Overview.load_outputs(group)
    assert "Maximum Runtime" in text(rendered[0])
    assert "312.3 h" in text(rendered[0])
    assert "DeepCAVE" in text(rendered[1])
