"""Coverage plan summaries must distinguish proposals, guarantees and executed trials."""

from itertools import product
from urllib.parse import unquote

import pytest
from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace
from dash import dash_table, dcc, html

from deepcave.evaluators.coverage import normalize_configurations
from deepcave.evaluators.coverage_planning import generate_coverage_trial_plan
from deepcave.plugins.hyperparameter.coverage_planning_views import (
    plan_toml,
    plan_view,
    planning_panel,
)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def text(component):
    if isinstance(component, str):
        return component
    if isinstance(component, (list, tuple)):
        return " ".join(" ".join(text(child) for child in component).split())
    return text(getattr(component, "children", []))


def components(component):
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from components(child)
    elif hasattr(component, "to_plotly_json"):
        yield component
        yield from components(getattr(component, "children", []))


def test_target_settings_collapse_keeps_strategy_and_actions_outside():
    panel = planning_panel()
    controls = next(c for c in components(panel) if isinstance(c, html.Aside))
    settings, strategy, actions = controls.children
    assert isinstance(settings, html.Details) and settings.open
    assert "Parameter resolution" in text(settings) and "Target strength" in text(settings)
    assert not any(isinstance(c, dcc.RadioItems) for c in components(settings))
    assert any(isinstance(c, dcc.RadioItems) for c in components(strategy))
    assert "Compute Plan" in text(actions)
    assert "Actions" not in text(controls) and "⚙" not in text(controls)
    assert not any(isinstance(c, html.Details) for c in components([strategy, actions]))


def make_plan(dimensions=3, strength=3, strategy="minimize_trials", budget=None, rows=None):
    space = ConfigurationSpace()
    names = [f"p{i}" for i in range(dimensions)]
    space.add([CategoricalHyperparameter(name, [0, 1]) for name in names])
    if rows is None:
        rows = [(0,) * dimensions, (1,) * dimensions]
    data = normalize_configurations(space, [dict(zip(names, row)) for row in rows])
    return generate_coverage_trial_plan(
        data,
        {
            "strategy": strategy,
            "resolution": 2,
            "strength_mode": "custom",
            "target_strength": strength,
            "max_additional_trials": budget,
        },
    )


def test_full_strength_summary_shows_coverage_metrics_and_configuration_action():
    plan = make_plan()
    view = plan_view(plan)
    summary = text(view)
    assert "Coverage Improvement Plan" in summary
    assert "levels · Target strength" not in summary
    assert "Additional trials required" not in summary
    assert "Proposed configurations" not in summary
    assert "Known lower bound" not in summary
    assert "Proven minimum" not in summary
    assert "Current coverage 25.0% +6 trials → Planned coverage 100%" in summary
    assert "2 already covered" in summary and "+6 newly covered" in summary
    assert "0 remaining" not in summary
    assert "possible configurations at strength" not in summary
    assert "Solution quality" not in summary
    children = list(components(view))
    assert not any(getattr(c, "className", None) == "coverage-plan-metrics" for c in children)
    details = next(c for c in children if isinstance(c, html.Details))
    assert "Show 6 configurations →" in text(details.children[0])
    assert not getattr(details, "open", False)
    table = next(c for c in components(details) if isinstance(c, dash_table.DataTable))
    assert [row["id"] for row in table.data] == [trial["position"] for trial in plan["trials"]]
    assert not any(isinstance(c, dcc.Graph) for c in children)
    bar = next(c for c in children if getattr(c, "className", None) == "coverage-plan-bar")
    assert [segment.style["width"] for segment in bar.children] == ["25%", "75%"]


@pytest.mark.parametrize("study", [None, {"name": 'catboost_hpo "test"', "id": 7}])
def test_toml_download_exports_all_trials_and_preserves_parameter_types(study):
    plan = make_plan(dimensions=4, strength=4)
    constants = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adam",
        "enabled": True,
        "model.name": 'Größe 🚀 "test"\\path\n\t\x7f',
    }
    plan["search_space"]["constants"].update(constants)
    for trial in plan["trials"]:
        trial["parameters"].update(constants)
    view = plan_view(plan, study)
    children = list(components(view))
    link = next(c for c in children if isinstance(c, html.A))
    assert text(link) == "Export TOML"
    actions = next(c for c in children if getattr(c, "className", None) == "coverage-plan-actions")
    details, download = actions.children
    assert isinstance(details, html.Details)
    assert download is link
    assert link not in list(components(details))
    assert link.download == "coverage-plan.toml"
    prefix, content = link.href.split(",", 1)
    assert prefix == "data:application/toml;charset=utf-8"
    parsed = tomllib.loads(unquote(content))
    expected = {"trials": [trial["parameters"] for trial in plan["trials"]]}
    if study:
        expected["study"] = study
        assert study["name"] not in text(view)
    assert parsed == expected
    assert type(parsed["trials"][0]["enabled"]) is bool
    assert type(parsed["trials"][0]["batch_size"]) is int
    assert type(parsed["trials"][0]["learning_rate"]) is float
    table = next(c for c in children if isinstance(c, dash_table.DataTable))
    assert len(parsed["trials"]) > table.page_size


def test_toml_download_is_disabled_for_values_toml_cannot_represent():
    plan = make_plan()
    plan["trials"][0]["parameters"]["p0"] = None
    with pytest.raises(ValueError, match="cannot represent parameter 'p0'"):
        plan_toml(plan)
    children = list(components(plan_view(plan)))
    assert not any(isinstance(c, html.A) for c in children)
    disabled = next(c for c in children if text(c) == "Export TOML")
    assert disabled.to_plotly_json()["props"]["aria-disabled"] == "true"


def test_coverage_breakdown_counts_match_the_reference_example():
    plan = make_plan(dimensions=4, strength=4, strategy="maximize_coverage", budget=1)
    for parameter, size in zip(plan["search_space"]["parameters"], [4, 4, 4, 60]):
        parameter["levels"] = list(range(size))
    plan.update(
        initial_coverage=2595 / 3840,
        projected_coverage=3469 / 3840,
        missing_tuples_before=1245,
        missing_tuples_after=371,
    )
    view = plan_view(plan)
    summary = text(view)
    assert "Current coverage 67.6% +1 / 1 trial → Planned coverage 90.3%" in summary
    assert "2,595 already covered" in summary
    assert "+874 newly covered" in summary
    assert "371 remaining" in summary
    assert "possible configurations at strength" not in summary
    children = list(components(view))
    bar = next(c for c in children if getattr(c, "className", None) == "coverage-plan-bar")
    assert [float(segment.style["width"][:-1]) for segment in bar.children] == pytest.approx(
        [100 * count / 3840 for count in [2595, 874, 371]]
    )
    assert "Total: 3,840" in bar.to_plotly_json()["props"]["aria-label"]


@pytest.mark.parametrize(
    "levels,added",
    [(16, 10), (64, 1), (16, 0)],
)
def test_small_gains_use_one_stacked_bar_on_the_full_scale(levels, added):
    plan = make_plan(strategy="maximize_coverage", budget=10)
    total = levels**3
    for parameter in plan["search_space"]["parameters"]:
        parameter["levels"] = list(range(levels))
    plan.update(
        initial_coverage=285 / total,
        projected_coverage=(285 + added) / total,
        missing_tuples_before=total - 285,
        missing_tuples_after=total - 285 - added,
        target_complete=False,
        optimality_proven=False,
    )
    view = plan_view(plan)
    children = list(components(view))
    assert not any(getattr(c, "className", None) == "coverage-plan-current-bar" for c in children)
    planned = next(c for c in children if getattr(c, "className", None) == "coverage-plan-bar")
    widths = [float(segment.style["width"][:-1]) for segment in planned.children]
    assert widths[0] == pytest.approx(100 * 285 / total)
    assert sum(widths) == pytest.approx(100)
    assert sum(widths[:2] if added else widths[:1]) == pytest.approx(100 * (285 + added) / total)
    assert "percentage points" not in text(view)
    if levels == 16 and added == 10:
        count = len(plan["trials"])
        assert f"Current coverage 7.0% +{count} / 10 trials → Planned coverage 7.2%" in text(view)


def test_almost_full_coverage_keeps_small_gains_visible_and_omits_zero_missing():
    plan = make_plan(dimensions=6, strength=6, rows=list(product([0, 1], repeat=6))[:-1])
    assert plan["missing_tuples_before"] == 1 and plan["missing_tuples_after"] == 0
    labels = [
        c
        for c in components(plan_view(plan))
        if "coverage-plan-label " in getattr(c, "className", "")
    ]
    assert [label.className for label in labels] == [
        "coverage-plan-label coverage-plan-key-observed",
        "coverage-plan-label coverage-plan-key-added",
    ]
    assert [text(label) for label in labels] == ["63 already covered", "+1 newly covered"]
    assert "0 remaining" not in text(plan_view(plan))


def test_tiny_segments_keep_all_counts_visible():
    plan = make_plan(dimensions=6, strength=6, strategy="maximize_coverage", budget=1)
    for parameter, size in zip(plan["search_space"]["parameters"], [8, 8, 8, 8, 8, 6]):
        parameter["levels"] = list(range(size))
    plan.update(
        initial_coverage=297 / 196608,
        projected_coverage=1297 / 196608,
        missing_tuples_before=196311,
        missing_tuples_after=195311,
    )
    children = list(components(plan_view(plan)))
    labels = [c for c in children if "coverage-plan-label " in getattr(c, "className", "")]
    assert [text(label) for label in labels] == [
        "297 already covered",
        "+1,000 newly covered",
        "195,311 remaining",
    ]
    assert [label.className.rsplit("-", 1)[-1] for label in labels] == [
        "observed",
        "added",
        "remaining",
    ]
    assert not any("coverage-plan-connectors" in getattr(c, "className", "") for c in children)


def test_lower_strength_heuristic_omits_lower_bound_and_proof_hints():
    plan = make_plan(dimensions=4, strength=2, rows=[])
    assert plan["target_complete"] and not plan["optimality_proven"]
    view = plan_view(plan)
    summary = text(view)
    assert "Heuristic solution" not in summary
    assert "Known lower bound" not in summary
    assert "Minimum not proven" not in summary
    assert "A smaller plan may exist" not in summary
    assert "Proven minimum" not in summary and "at most one new combination" not in summary
    assert "possible configurations at strength" not in summary
    assert "Show coverage gain by trial" not in summary
    assert not any(isinstance(c, dcc.Graph) for c in components(view))


def test_lower_strength_optimality_omits_proof_hints():
    plan = make_plan(strength=2, rows=[(0, 0, 0), (0, 1, 1), (1, 0, 1)])
    assert plan["optimality_proven"] and len(plan["trials"]) == 1
    summary = text(plan_view(plan))
    assert "Optimal solution" not in summary
    assert "Why is this optimal?" not in summary
    assert "no smaller plan is possible" not in summary
    assert "Proven minimum" not in summary
    assert "+1 trial → Planned coverage" in summary
    assert "Known lower bound" not in summary


def test_budget_strategy_emphasizes_coverage_without_claiming_minimum_trials():
    plan = make_plan(dimensions=4, strength=2, strategy="maximize_coverage", budget=20, rows=[])
    assert plan["target_complete"] and len(plan["trials"]) > plan["lower_bound_trials"]
    view = plan_view(plan)
    comparison = next(
        c for c in components(view) if getattr(c, "className", None) == "coverage-plan-comparison"
    )
    assert (
        f"Current coverage 0.0% +{len(plan['trials'])} / 20 trials → Planned coverage 100%"
        in text(comparison)
    )
    assert "Known lower bound" not in text(view)
    assert "Minimum trials for full coverage" not in text(view)
    assert "trial budget used" not in text(view)
    assert "Optimal solution" not in text(view)
    assert "Why is this optimal?" not in text(view)
    assert "No plan can achieve higher coverage" not in text(view)
    assert "Proven minimum" not in text(view)


@pytest.mark.parametrize(
    "reason",
    [
        "budget_exhausted",
        "timeout",
        "trial_limit",
        "no_gain_found",
    ],
)
def test_incomplete_plans_show_remaining_gaps_without_claiming_full_coverage(reason):
    plan = make_plan(strategy="maximize_coverage", budget=1)
    plan["termination_reason"] = reason
    if reason != "budget_exhausted":
        plan.update(
            strategy="minimize_trials", max_additional_trials=None, budget_target_impossible=False
        )
    view = plan_view(plan)
    summary = text(view)
    assert "Target not reached" not in summary
    assert "5 remaining" in summary
    assert "Solution quality" not in summary and "gaps remain" not in summary
    assert "achieves full target coverage" not in summary
    assert "Why was the target not reached?" not in summary
    assert "combinations would still be missing" not in summary
    assert "The additional trial budget has been reached" not in summary
    assert "Optimal solution" not in summary and "Proven minimum" not in summary
    if reason != "budget_exhausted":
        assert "+1 trial → Planned coverage" in summary
    else:
        assert "+1 / 1 trial → Planned coverage" in summary
        assert "budgeted trials" not in summary


def test_already_covered_target_has_no_execution_action():
    view = plan_view(make_plan(rows=list(product([0, 1], repeat=3))))
    assert "No additional configurations required" in text(view)
    assert "Target already covered · Nothing to execute" in text(view)
    assert "Current coverage 100% +0 trials → Planned coverage 100%" in text(view)
    assert "Show configurations" not in text(view)
    assert "Export TOML" not in text(view)
    assert not any(isinstance(c, (dcc.Graph, dash_table.DataTable)) for c in components(view))


def test_zero_progress_does_not_claim_the_target_is_already_covered():
    plan = make_plan(strategy="maximize_coverage", budget=1)
    plan.update(
        trials=[],
        projected_coverage=plan["initial_coverage"],
        missing_tuples_after=plan["missing_tuples_before"],
        termination_reason="timeout",
        budget_target_impossible=False,
    )
    summary = text(plan_view(plan))
    assert "No proposed configurations" in summary and "Target not reached" in summary
    assert "Target already covered" not in summary and "Ready for execution" not in summary


def test_incomplete_coverage_never_rounds_up_to_100_percent():
    plan = make_plan(strategy="maximize_coverage", budget=1)
    for parameter in plan["search_space"]["parameters"]:
        parameter["levels"] = list(range(16))
    plan.update(
        initial_coverage=4094 / 4096,
        projected_coverage=4095 / 4096,
        missing_tuples_before=2,
        missing_tuples_after=1,
    )
    summary = text(plan_view(plan))
    assert "Current coverage >99.9% +1 / 1 trial → Planned coverage >99.9%" in summary
    comparison = next(
        c
        for c in components(plan_view(plan))
        if getattr(c, "className", None) == "coverage-plan-comparison"
    )
    assert "100%" not in text(comparison)
