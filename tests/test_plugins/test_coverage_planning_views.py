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
    assert "Coverage Improvement Plan" not in summary and "Optimal solution" not in summary
    assert "Target: Strength" not in summary
    assert view.children[0].className == "coverage-plan-metrics"
    assert "6 Additional trials required Proven minimum" in summary
    assert "25.0% → 100% Projected coverage change" in summary
    assert "At target Strength" not in summary
    assert "Already covered configurations:  2" in summary
    assert "Newly covered configurations:  6" in summary
    assert "Missing configurations" not in summary
    assert "8  possible configurations at strength 3" in summary
    assert "Why is this optimal?" not in summary
    assert "Exactly 6 combinations are missing." not in summary
    assert "Observed coverage at target Strength" not in summary
    assert "Proposed improvement:" not in summary
    assert "Observed coverage is unchanged" not in summary
    assert "Ready for execution · Not yet applied" not in summary
    children = list(components(view))
    details = next(c for c in children if isinstance(c, html.Details))
    assert "6 proposed configurations" in text(details.children[0])
    assert not getattr(details, "open", False)
    table = next(c for c in components(details) if isinstance(c, dash_table.DataTable))
    assert [row["id"] for row in table.data] == [trial["position"] for trial in plan["trials"]]
    assert not any(isinstance(c, dcc.Graph) for c in children)
    bar = next(c for c in children if getattr(c, "className", None) == "coverage-plan-bar")
    assert [segment.style["width"] for segment in bar.children] == ["25%", "75%", "0%"]


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
    assert "67.6% → 90.3% Projected coverage change" in summary
    assert "Already covered configurations:  2,595" in summary
    assert "Newly covered configurations:  874" in summary
    assert "Missing configurations:  371" in summary
    assert "3,840  possible configurations at strength 4" in summary
    children = list(components(view))
    bar = next(c for c in children if getattr(c, "className", None) == "coverage-plan-bar")
    assert [float(segment.style["width"][:-1]) for segment in bar.children] == pytest.approx(
        [100 * count / 3840 for count in [2595, 874, 371]]
    )
    assert "Total: 3,840" in bar.to_plotly_json()["props"]["aria-label"]
    labels = [
        c for c in children if "coverage-plan-label " in getattr(c, "className", "")
    ]
    assert [float(label.to_plotly_json()["props"]["data-anchor"]) for label in labels] == pytest.approx(
        [100 * center / 3840 for center in [2595 / 2, 2595 + 874 / 2, 2595 + 874 + 371 / 2]]
    )


def test_almost_full_coverage_anchors_labels_to_segments_and_omits_zero_missing():
    plan = make_plan(dimensions=6, strength=6, rows=list(product([0, 1], repeat=6))[:-1])
    assert plan["missing_tuples_before"] == 1 and plan["missing_tuples_after"] == 0
    labels = [
        c for c in components(plan_view(plan))
        if "coverage-plan-label " in getattr(c, "className", "")
    ]
    assert [label.className for label in labels] == [
        "coverage-plan-label coverage-plan-key-observed",
        "coverage-plan-label coverage-plan-key-added",
    ]
    assert [float(label.to_plotly_json()["props"]["data-anchor"]) for label in labels] == pytest.approx(
        [49.21875, 99.21875]
    )
    assert "Missing configurations" not in text(plan_view(plan))


def test_tiny_segments_keep_labels_attached_to_actual_segments():
    plan = make_plan(dimensions=6, strength=6, strategy="maximize_coverage", budget=1)
    for parameter, size in zip(plan["search_space"]["parameters"], [8, 8, 8, 8, 8, 6]):
        parameter["levels"] = list(range(size))
    plan.update(
        initial_coverage=297 / 196608, projected_coverage=1297 / 196608,
        missing_tuples_before=196311, missing_tuples_after=195311,
    )
    children = list(components(plan_view(plan)))
    labels = [c for c in children if "coverage-plan-label " in getattr(c, "className", "")]
    assert [float(label.to_plotly_json()["props"]["data-anchor"]) for label in labels] == pytest.approx(
        [100 * center / 196608 for center in [297 / 2, 297 + 1000 / 2, 1297 + 195311 / 2]]
    )
    assert [label.className.rsplit("-", 1)[-1] for label in labels] == ["observed", "added", "remaining"]
    assert not any("coverage-plan-connectors" in getattr(c, "className", "") for c in children)


def test_lower_strength_heuristic_shows_bound_without_trajectory():
    plan = make_plan(dimensions=4, strength=2, rows=[])
    assert plan["target_complete"] and not plan["optimality_proven"]
    view = plan_view(plan)
    summary = text(view)
    assert "Heuristic solution" not in summary
    assert "Known lower bound: 4 trials" in summary
    assert "A smaller plan may exist" in summary
    assert "Proven minimum" not in summary and "at most one new combination" not in summary
    assert "24  possible configurations at strength 2" in summary
    assert "Show coverage gain by trial" not in summary
    assert not any(isinstance(c, dcc.Graph) for c in components(view))


def test_lower_strength_optimality_keeps_proven_minimum_without_header():
    plan = make_plan(strength=2, rows=[(0, 0, 0), (0, 1, 1), (1, 0, 1)])
    assert plan["optimality_proven"] and len(plan["trials"]) == 1
    summary = text(plan_view(plan))
    assert "Optimal solution" not in summary
    assert "Why is this optimal?" not in summary
    assert "no smaller plan is possible" not in summary
    assert "Proven minimum" in summary
    assert "1 Additional trial required Proven minimum" in summary


def test_budget_strategy_emphasizes_coverage_without_claiming_minimum_trials():
    plan = make_plan(dimensions=4, strength=2, strategy="maximize_coverage", budget=20, rows=[])
    assert plan["target_complete"] and len(plan["trials"]) > plan["lower_bound_trials"]
    view = plan_view(plan)
    metrics = next(
        c for c in components(view) if getattr(c, "className", None) == "coverage-plan-metrics"
    )
    assert text(metrics.children[1]).startswith("0.0% → 100% Projected coverage change")
    assert f"{len(plan['trials'])} / 20 Additional trials used" in text(metrics.children[0])
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
def test_incomplete_plans_show_remaining_gaps_without_explanation(reason):
    plan = make_plan(strategy="maximize_coverage", budget=1)
    plan["termination_reason"] = reason
    if reason != "budget_exhausted":
        plan.update(
            strategy="minimize_trials", max_additional_trials=None, budget_target_impossible=False
        )
    view = plan_view(plan)
    summary = text(view)
    assert "Target not reached" not in summary
    assert "Missing configurations:  5" in summary
    assert "Why was the target not reached?" not in summary
    assert "combinations would still be missing" not in summary
    assert "The additional trial budget has been reached" not in summary
    assert "Optimal solution" not in summary and "Proven minimum" not in summary
    if reason != "budget_exhausted":
        assert "Additional trial proposed" in summary
    else:
        assert "1 / 1 Additional trial used" in summary


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
    assert ">99.9% → >99.9% Projected coverage change" in summary
    assert "100%" not in summary
