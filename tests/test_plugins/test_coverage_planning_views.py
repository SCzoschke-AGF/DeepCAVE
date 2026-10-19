"""Coverage plan summaries must distinguish proposals, guarantees and executed trials."""

from itertools import product

import pytest
from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace
from dash import dash_table, dcc, html

from deepcave.evaluators.coverage import normalize_configurations
from deepcave.evaluators.coverage_planning import generate_coverage_trial_plan
from deepcave.plugins.hyperparameter.coverage_planning_views import plan_view


def text(component):
    if isinstance(component, str):
        return component
    if isinstance(component, (list, tuple)):
        return " ".join(text(child) for child in component)
    return text(getattr(component, "children", []))


def components(component):
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from components(child)
    elif hasattr(component, "to_plotly_json"):
        yield component
        yield from components(getattr(component, "children", []))


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


def test_full_strength_summary_keeps_metrics_without_explanatory_text():
    plan = make_plan()
    view = plan_view(plan)
    summary = text(view)
    assert "Coverage Improvement Plan" in summary and "Optimal solution" in summary
    assert "Target: Strength 3 / 3 · 2 levels · Minimize additional trials" in summary
    assert "Additional trials required 6 Proven minimum" in summary
    assert "Projected coverage 25.0% → 100% At target Strength 3" in summary
    assert "Already covered: 2" in summary and "Covered by plan: 6" in summary
    assert "Total possible combinations 8" in summary
    assert "Remaining after successful execution 0" in summary
    assert "Why is this optimal?" not in summary
    assert "Exactly 6 combinations are missing." not in summary
    assert "Observed coverage at target Strength" not in summary
    assert "Proposed improvement:" not in summary
    assert "Observed coverage is unchanged" not in summary
    assert "Ready for execution · Not yet applied" in summary
    children = list(components(view))
    details = next(c for c in children if isinstance(c, html.Details))
    assert "6 proposed configurations" in text(details.children[0])
    assert not getattr(details, "open", False)
    table = next(c for c in components(details) if isinstance(c, dash_table.DataTable))
    assert [row["id"] for row in table.data] == [trial["position"] for trial in plan["trials"]]
    assert not any(isinstance(c, dcc.Graph) for c in children)
    bar = next(c for c in children if getattr(c, "className", None) == "coverage-plan-bar")
    assert [segment.style["width"] for segment in bar.children] == ["25%", "75%", "0%"]


def test_lower_strength_heuristic_shows_bound_and_optional_trajectory():
    plan = make_plan(dimensions=4, strength=2, rows=[])
    assert plan["target_complete"] and not plan["optimality_proven"]
    view = plan_view(plan)
    summary = text(view)
    assert "Heuristic solution" in summary
    assert "Known lower bound: 4 trials" in summary
    assert "A smaller plan may exist" in summary
    assert "Proven minimum" not in summary and "at most one new combination" not in summary
    assert "Total possible combinations 24" in summary
    trajectory = next(
        c for c in components(view) if getattr(c, "className", None) == "coverage-plan-trajectory"
    )
    assert not getattr(trajectory, "open", False)
    assert any(isinstance(c, dcc.Graph) for c in components(trajectory))


def test_lower_strength_optimality_keeps_badge_without_explanation():
    plan = make_plan(strength=2, rows=[(0, 0, 0), (0, 1, 1), (1, 0, 1)])
    assert plan["optimality_proven"] and len(plan["trials"]) == 1
    summary = text(plan_view(plan))
    assert "Optimal solution" in summary
    assert "Why is this optimal?" not in summary
    assert "no smaller plan is possible" not in summary
    assert "Proven minimum" in summary


def test_budget_strategy_emphasizes_coverage_without_claiming_minimum_trials():
    plan = make_plan(dimensions=4, strength=2, strategy="maximize_coverage", budget=20, rows=[])
    assert plan["target_complete"] and len(plan["trials"]) > plan["lower_bound_trials"]
    view = plan_view(plan)
    metrics = next(
        c for c in components(view) if getattr(c, "className", None) == "coverage-plan-metrics"
    )
    assert text(metrics.children[0]).startswith("Projected coverage 0.0% → 100%")
    assert f"{len(plan['trials'])} of 20 trial budget used" in text(metrics.children[1])
    assert "Optimal solution" in text(view)
    assert "Why is this optimal?" not in text(view)
    assert "No plan can achieve higher coverage" not in text(view)
    assert "Proven minimum" not in text(view)


@pytest.mark.parametrize(
    "reason,explanation",
    [
        ("budget_exhausted", "exceeding this budget"),
        ("timeout", "Search time elapsed"),
        ("trial_limit", "planning limit of 1,000 additional trials"),
        ("no_gain_found", "not an impossibility proof"),
    ],
)
def test_incomplete_plans_explain_the_limit_and_projected_remaining_gaps(reason, explanation):
    plan = make_plan(strategy="maximize_coverage", budget=1)
    plan["termination_reason"] = reason
    if reason != "budget_exhausted":
        plan.update(
            strategy="minimize_trials", max_additional_trials=None, budget_target_impossible=False
        )
    view = plan_view(plan)
    summary = text(view)
    assert "Target not reached" in summary
    assert explanation in summary
    assert "Remaining after successful execution 5" in summary
    assert "5 combinations would still be missing" in summary
    assert "Optimal solution" not in summary and "Proven minimum" not in summary
    if reason != "budget_exhausted":
        assert "Additional trials proposed" in summary


def test_already_covered_target_has_no_execution_action():
    view = plan_view(make_plan(rows=list(product([0, 1], repeat=3))))
    assert "No additional configurations required" in text(view)
    assert "Target already covered · Nothing to execute" in text(view)
    assert "Show configurations" not in text(view)
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
    assert "Projected coverage >99.9% → >99.9%" in summary
    assert "Projected coverage 100" not in summary
