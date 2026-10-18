"""Random reference graphics use measured values and explicitly handle missing ranks."""

from copy import deepcopy

import pytest

from deepcave.plugins.hyperparameter.combinatorial_views import (
    _coverage_benchmark,
    _strength_reference,
    combinatorial_view,
)


def visible_text(component):
    if isinstance(component, str):
        return component
    if isinstance(component, (list, tuple)):
        return " ".join(visible_text(child) for child in component)
    return visible_text(getattr(component, "children", []))


def test_benchmark_zooms_band_median_and_observation_with_explicit_scale():
    cell = {
        "strength": 2,
        "status": "computed",
        "coverage": 0.8,
        "missing": 2,
        "covered": 8,
        "total": 10,
        "complete_groups": 0,
        "groups": 1,
    }
    reference = {
        "lower": 0.6,
        "upper": 0.9,
        "median": 0.83,
        "mean": 0.77,
        "percentile": 40,
        "degenerate": False,
    }
    view = _coverage_benchmark(cell, reference, 80)
    band, median, point = view.children[1].children
    assert band.style["left"] == "12.5%"
    assert float(band.style["width"].rstrip("%")) == pytest.approx(75)
    assert median.style["left"] == "70%"
    assert point.style["left"] == "62.5%"
    assert visible_text(view.children[2]) == "55.0% 95.0%"
    text = visible_text(view)
    tooltip = view.children[1].title
    assert "median: 83.0%" in tooltip
    assert "Within the typical random range" in tooltip
    assert "Random percentile: 40" in tooltip
    assert "Random reference" not in text
    assert "Random percentile" not in text
    assert view.children[1].role == "img"
    assert "Observed coverage: 80.0%" in view.children[1].to_plotly_json()["props"]["aria-label"]
    assert "awaiting values" in _coverage_benchmark(cell, None, 80).children[1].title
    assert len(_coverage_benchmark(cell, None, 80).children[1].children) == 1
    old = deepcopy(reference)
    del old["median"]
    del old["degenerate"]
    tooltip = _coverage_benchmark(cell, old, 80).children[1].title
    assert "refresh baseline" in tooltip
    assert "Random percentile" not in tooltip


@pytest.mark.parametrize(
    "observed,lower,upper,expected_axis",
    [
        (0.992, 0.975, 1, "97.0% 100%"),
        (1, 1, 1, "99.0% 100%"),
        (0, 0, 0, "0.0% 1.0%"),
        (0.4, 0.7, 0.9, "32.0% 98.0%"),
    ],
)
def test_zoom_keeps_outlying_observations_and_constant_references_visible(
    observed, lower, upper, expected_axis
):
    cell = {
        "strength": 2,
        "status": "computed",
        "coverage": observed,
        "missing": 1,
        "covered": 9,
        "total": 10,
        "complete_groups": 0,
        "groups": 1,
    }
    reference = {"lower": lower, "upper": upper, "median": lower}
    view = _coverage_benchmark(cell, reference, 80)
    assert visible_text(view.children[2]) == expected_axis
    for marker in view.children[1].children:
        assert 0 <= float(marker.style["left"].rstrip("%")) <= 100


def test_degenerate_reference_avoids_rank_and_censored_strength_avoids_exact_comparison():
    cell = {
        "strength": 2,
        "status": "computed",
        "coverage": 0.9,
        "missing": 1,
        "covered": 9,
        "total": 10,
        "complete_groups": 0,
        "groups": 1,
    }
    reference = {"lower": 1, "upper": 1, "median": 1, "percentile": None, "degenerate": True}
    text = _coverage_benchmark(cell, reference, 80).children[1].title
    assert "Below the typical random range" in text
    assert "All random designs have the same value; no percentile." in text
    assert "Random percentile:" not in text
    assert "all designs t = 1" in visible_text(_strength_reference(reference))
    reference["censored"] = True
    assert "only lower bounds available" in visible_text(_strength_reference(reference))


@pytest.mark.parametrize("with_baseline", [True, False])
def test_coverage_array_uses_the_same_extreme_endpoints_for_every_row(with_baseline):
    rows, comparisons = [], []
    for resolution, strength, coverage, lower, upper, median in [
        (2, 4, 0.938, 0.870, 0.964, 0.922),
        (3, 2, 0.963, 0.924, 0.970, 0.952),
        (4, 1, 0.992, 0.975, 1, 0.992),
        (8, 1, 0.750, 0.711, 0.758, 0.735),
    ]:
        cell = {
            "strength": strength + 1,
            "status": "computed",
            "coverage": coverage,
            "covered": round(coverage * 1000),
            "total": 1000,
            "missing": 1000 - round(coverage * 1000),
            "complete_groups": 0,
            "groups": 1,
        }
        rows.append(
            {
                "resolution": resolution,
                "observed_strength": strength,
                "strength_is_lower_bound": False,
                "trial_bound": {2: 6, 3: 3, 4: 3, 8: 2}[resolution],
                "levels": {},
                "cells": [cell],
            }
        )
        comparisons.append(
            {
                "resolution": resolution,
                "strength": strength + 1,
                "lower": lower,
                "upper": upper,
                "median": median,
                    "observed": coverage,
                    "mean": median,
                    "deficit": coverage - median,
            }
        )
    result = {"trials": 80, "parameters": list("abcdef"), "max_strength": 6, "rows": rows}
    # An unavailable row must not expand the common axis towards zero.
    result["rows"].append(
        {
            "resolution": 16,
            "observed_strength": 0,
            "strength_is_lower_bound": True,
            "trial_bound": 1,
            "levels": {},
            "cells": [{"strength": 1, "status": "work_limit"}],
        }
    )
    baseline = {"cells": comparisons, "designs": 500, "seed": 0} if with_baseline else None
    view = combinatorial_view(result, baseline)
    options = view.children[0].children[1].options
    points = []
    for option in options[:4]:
        benchmark = option["label"].children[2]
        assert visible_text(benchmark.children[2]) == "70.0% 100%"
        assert "shared across resolutions" in benchmark.children[1].title
        points.append(float(benchmark.children[1].children[-1].style["left"].rstrip("%")))
    assert points == pytest.approx([79.3333, 87.6667, 97.3333, 16.6667], abs=1e-4)
