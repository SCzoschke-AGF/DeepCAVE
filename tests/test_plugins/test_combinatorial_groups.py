"""All-group diagnostics, AND filters and lazy drilldowns use saved CCM measurements."""

import json
from itertools import product

import pytest
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from deepcave.evaluators import combinatorial as ccm
from deepcave.evaluators.coverage import normalize_configurations
from deepcave.plugins.hyperparameter import combinatorial_groups as views


def text(component):
    if isinstance(component, (str, int)):
        return str(component)
    if isinstance(component, (list, tuple)):
        return " ".join(text(child) for child in component)
    return text(getattr(component, "children", []))


@pytest.fixture
def measured():
    names = [f"feature_{index}" for index in range(9)]
    space = ConfigurationSpace()
    space.add([UniformFloatHyperparameter(name, 0, 1) for name in names])
    configs = [dict(zip(names, [x] * 7 + [y, x ^ y])) for x, y in product(range(2), repeat=2)] * 4
    result = ccm.compute_combinatorial_coverage(normalize_configurations(space, configs))
    return json.loads(json.dumps(result))


def test_all_groups_survive_top_twenty_limit_and_filter_counts_match_histogram(measured):
    row = measured["rows"][0]
    cell = row["cells"][1]
    assert len(cell["deficits"]) == 20
    groups = views.group_records(measured, row)
    assert len(groups) == 36
    assert sum(group["missing"] > 0 for group in groups) == 21
    assert [group["parameters"] for group in groups[:21]] == sorted(
        group["parameters"] for group in groups[:21]
    )
    initial = views.group_overview(measured, row)
    assert len(initial.children[3].children[1].children) == 5
    assert "Groups analyzed 36 Fully covered 15 Most missing 2 / 4" == text(initial.children[0])
    trace = initial.children[1].figure.data[0]
    assert list(trace.x) == [0, 2]
    assert list(trace.y) == [15, 21]
    assert (
        len(views.group_overview(measured, row, limit="all").children[3].children[1].children) == 36
    )
    filtered = views.group_overview(measured, row, ["feature_0"], limit=10)
    assert "Groups analyzed 8 Fully covered 2 Most missing 2 / 4" == text(filtered.children[0])
    assert sum(filtered.children[1].figure.data[0].y) == 8
    and_groups = views.group_records(measured, row, ["feature_7", "feature_8"])
    assert len(and_groups) == 1
    assert and_groups[0]["parameters"] == ["feature_7", "feature_8"]
    assert and_groups[0]["missing"] == 0
    empty = views.group_overview(measured, row, ["feature_0", "feature_1", "feature_2"])
    assert "No groups contain all selected parameters." in text(empty)
    assert not empty.children[1].figure.data[0].x


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
    assert "4 possible · 2 observed · 2 missing" in text(
        views.group_drilldown(measured, row, group)
    )
    fully_covered = views.group_records(measured, row, ["feature_7", "feature_8"])[0]
    assert "This group covers every combination." in text(
        views.group_drilldown(measured, row, fully_covered)
    )


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
    assert set(tuples) == set(product([f"range {i}" for i in range(4)], repeat=3)) - {("range 0",) * 3}


def test_old_results_remain_visible_and_request_manual_refresh_for_new_diagnostics(measured):
    row = measured["rows"][0]
    del row["cells"][1]["group_covered"]
    view = views.group_diagnostics(measured, row, row["cells"][1])
    assert "Refresh coverage" in text(view)
    assert "existing coverage results remain available" in text(view)
    assert "Random percentile" not in text(view)
