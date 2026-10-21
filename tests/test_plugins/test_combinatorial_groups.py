"""All-group diagnostics, AND filters and lazy drilldowns use saved CCM measurements."""

import json
from copy import deepcopy
from itertools import combinations, product

import numpy as np
import pytest
from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace, UniformFloatHyperparameter
from dash import dcc, html

from deepcave.evaluators import combinatorial as ccm
from deepcave.evaluators.coverage import normalize_configurations
from deepcave.plugins.hyperparameter import combinatorial_groups as views


def text(component):
    if isinstance(component, (str, int)):
        return str(component)
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


@pytest.fixture
def measured():
    names = [f"feature_{index}" for index in range(9)]
    space = ConfigurationSpace()
    space.add([UniformFloatHyperparameter(name, 0, 1) for name in names])
    configs = [dict(zip(names, [x] * 7 + [y, x ^ y])) for x, y in product(range(2), repeat=2)] * 4
    result = ccm.compute_combinatorial_coverage(normalize_configurations(space, configs))
    return json.loads(json.dumps(result))


def test_all_group_counts_survive_top_twenty_limit_and_only_incomplete_groups_are_listed(measured):
    row = measured["rows"][0]
    cell = row["cells"][1]
    assert len(cell["deficits"]) == 20
    groups = views.group_records(measured, row)
    assert len(groups) == 36
    assert sum(group["missing"] > 0 for group in groups) == 21
    assert sum(group["covered"] for group in groups) == cell["covered"]
    assert [group["parameters"] for group in groups[:21]] == sorted(
        group["parameters"] for group in groups[:21]
    )
    assert "21 of 36 parameter groups incomplete" in text(
        views.group_diagnostics(measured, row, cell)
    )
    initial = views.group_overview(measured, row)
    assert len(initial.children[2].children) == 5
    assert all(isinstance(entry, html.Details) and not entry.open for entry in initial.children[2].children)
    assert not any(isinstance(component, dcc.Graph) for component in components(initial))
    assert "Parameter group Missing Coverage" == text(initial.children[1])
    assert len(views.group_overview(measured, row, limit="all").children[2].children) == 21
    assert (
        len(
            views.group_overview(measured, row, limit="all", include_complete=True)
            .children[2]
            .children
        )
        == 36
    )
    filtered = views.group_overview(measured, row, ["feature_0"], limit=10)
    assert len(filtered.children[2].children) == 6
    and_groups = views.group_records(measured, row, ["feature_7", "feature_8"])
    assert len(and_groups) == 1 and and_groups[0]["missing"] == 0
    assert "No incomplete parameter groups" in text(
        views.group_overview(measured, row, ["feature_7", "feature_8"])
    )
    empty = views.group_overview(measured, row, ["feature_0", "feature_1", "feature_2"])
    assert "No parameter groups contain all selected parameters." in text(empty)
    assert "Most missing" not in text(initial)
    assert "combinations analyzed" not in text(initial)


def test_missing_tuples_use_persisted_encoding_and_definitions_without_recalculation(
    measured, monkeypatch
):
    def unexpected(*args, **kwargs):
        pytest.fail("UI interactions must never recalculate coverage, binning or random references")

    monkeypatch.setattr(ccm, "compute_combinatorial_coverage", unexpected)
    monkeypatch.setattr(ccm, "compute_combinatorial_baseline", unexpected)
    monkeypatch.setattr(ccm, "_discrete_cuts", unexpected)
    row = measured["rows"][0]
    group = views.group_records(measured, row)[0]
    values, tooltips = views.missing_tuple_page(measured, row, group)
    assert values == [
        {"feature_0": "[0, 0.5)", "feature_1": "[0.5, 1]"},
        {"feature_0": "[0.5, 1]", "feature_1": "[0, 0.5)"},
    ]
    assert tooltips[0]["feature_0"]["value"] == "[0, 0.5)"
    assert tooltips[0]["feature_1"]["value"] == "[0.5, 1]"
    assert views.group_drilldown(measured, row, group).children[1].data == values
    fully_covered = views.group_records(measured, row, ["feature_7", "feature_8"])[0]
    assert views.missing_tuple_page(measured, row, fully_covered) == ([], [])


def test_missing_tuple_pages_are_disjoint_and_cover_exact_cartesian_complement():
    names = list("abc")
    row = {
        "levels": dict.fromkeys(names, 4),
        "discretization": {
            "points": [[0, 0, 0]] * 64,
            "definitions": {
                name: [{"label": f"L{i}", "range": f"range {i}"} for i in range(4)]
                for name in names
            },
        },
    }
    group = {"parameters": names, "missing": 63}
    pages = [
        views.missing_tuple_page({"parameters": names}, row, group, page)[0] for page in range(4)
    ]
    assert [len(page) for page in pages] == [20, 20, 20, 3]
    tuples = [tuple(record[name] for name in names) for page in pages for record in page]
    assert len(set(tuples)) == 63
    assert set(tuples) == set(product([f"range {i}" for i in range(4)], repeat=3)) - {
        ("range 0",) * 3
    }


def test_old_results_remain_visible_and_request_manual_refresh_for_new_diagnostics(measured):
    row = measured["rows"][0]
    del row["cells"][1]["group_covered"]
    view = views.group_diagnostics(measured, row, row["cells"][1])
    assert "Refresh coverage" in text(view)
    assert "parameter groups incomplete" not in text(view)
    assert "Random percentile" not in text(view)


def test_eight_levels_example_reports_eight_of_fifteen_actual_incomplete_groups():
    names = list("abcdef")
    space = ConfigurationSpace(space={name: (0.0, 1.0) for name in names})
    points = np.random.default_rng(8).integers(8, size=(299, 6))
    configs = [dict(zip(names, (point + 0.5) / 8)) for point in points]
    result = ccm.compute_combinatorial_coverage(normalize_configurations(space, configs))
    original = deepcopy(result)
    row = next(row for row in result["rows"] if row["resolution"] == 8)
    cell = views.diagnostic_cell(result, row)
    assert cell["strength"] == 2
    expected = [len(set(map(tuple, points[:, group]))) for group in combinations(range(6), 2)]
    assert sum(count < 64 for count in expected) == 8
    groups = views.group_records(result, row)
    assert sorted(group["covered"] for group in groups) == sorted(expected)
    diagnosis = views.group_diagnostics(result, row, cell)
    assert "Strength 2 · Coverage gaps" in text(diagnosis)
    assert "8 of 15 parameter groups incomplete" in text(diagnosis)
    assert sum(group["missing"] for group in groups) == cell["missing"]
    assert result == original


def test_four_parameter_groups_use_actual_counts_and_actual_missing_combinations():
    names = list("abcdef")
    space = ConfigurationSpace()
    space.add([CategoricalHyperparameter(name, [0, 1]) for name in names])
    points = [(a, b, c, d, a ^ b ^ c, a ^ b ^ d) for a, b, c, d in product([0, 1], repeat=4)]
    result = ccm.compute_combinatorial_coverage(
        normalize_configurations(space, [dict(zip(names, point)) for point in points])
    )
    row = result["rows"][0]
    assert row["observed_strength"] == 3
    cell = views.diagnostic_cell(result, row)
    groups = views.group_records(result, row)
    assert cell["strength"] == 4 and len(groups) == 15
    assert all(len(group["parameters"]) == 4 for group in groups)
    assert sum(group["missing"] for group in groups) == cell["missing"]
    incomplete = sum(group["missing"] > 0 for group in groups)
    assert f"{incomplete} of 15 parameter groups incomplete" in text(
        views.group_diagnostics(result, row, cell)
    )
    group = groups[0]
    indices = [names.index(name) for name in group["parameters"]]
    observed = {tuple(str(point[index]) for index in indices) for point in points}
    expected = set(product(["0", "1"], repeat=4)) - observed
    data, _ = views.missing_tuple_page(result, row, group)
    assert {tuple(record[name] for name in group["parameters"]) for record in data} == expected
    assert len(data) == group["missing"]


@pytest.mark.parametrize("complete", [False, True])
def test_full_strength_handles_trial_limit_and_singular_grammar(complete):
    names = list("abcdef")
    space = ConfigurationSpace()
    space.add([CategoricalHyperparameter(name, [0, 1]) for name in names])
    points = list(product([0, 1], repeat=6))
    if not complete:
        points.pop()
    result = ccm.compute_combinatorial_coverage(
        normalize_configurations(space, [dict(zip(names, point)) for point in points])
    )
    row = result["rows"][0]
    cell = views.diagnostic_cell(result, row)
    assert cell["strength"] == 6
    if not complete:
        assert cell["status"] == "trial_limit"
    summary = text(views.group_diagnostics(result, row, cell))
    assert (
        "All 1 parameter group fully covered" if complete else "1 of 1 parameter group incomplete"
    ) in summary
    assert views.group_records(result, row)[0]["missing"] == (0 if complete else 1)


def test_different_parameter_cardinalities_use_group_specific_totals():
    space = ConfigurationSpace()
    space.add(
        [
            CategoricalHyperparameter("a", [0, 1]),
            CategoricalHyperparameter("b", [0, 1, 2]),
            UniformFloatHyperparameter("c", 0, 1),
        ]
    )
    configs = [{"a": i % 2, "b": i % 3, "c": (i % 4 + 0.5) / 4} for i in range(12)]
    result = ccm.compute_combinatorial_coverage(normalize_configurations(space, configs))
    row = next(row for row in result["rows"] if row["resolution"] == 4)
    groups = views.group_records(result, row)
    assert sorted(group["total"] for group in groups) == [6, 8, 12]
    assert [(group["parameters"], group["missing"]) for group in groups if group["missing"]] == [
        (["a", "c"], 4)
    ]


def test_missing_bin_descriptions_fall_back_to_one_based_bin_numbers():
    result = {"parameters": ["a", "b"]}
    row = {"levels": {"a": 2, "b": 1}, "discretization": {"points": [[0, 0]]}}
    group = {"parameters": ["a", "b"], "missing": 1}
    assert views.missing_tuple_page(result, row, group)[0] == [{"a": "Bin 2", "b": "Bin 1"}]


def test_partial_saved_group_counts_are_not_reported_as_a_complete_summary(measured):
    row = measured["rows"][0]
    cell = views.diagnostic_cell(measured, row)
    cell["group_covered"] = cell["group_covered"][:20]
    assert views.group_records(measured, row) == []
    assert "Refresh coverage" in text(views.group_diagnostics(measured, row, cell))
