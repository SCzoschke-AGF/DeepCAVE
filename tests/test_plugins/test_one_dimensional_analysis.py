from types import SimpleNamespace

import pandas as pd
import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from dash import html

from deepcave import config
from deepcave.plugins.hyperparameter import one_dimensional_analysis as analysis
from deepcave.runs.objective import Objective


def test_plugin_is_registered_directly_after_ablation_paths():
    plugin_ids = [plugin.id for plugin in config.PLUGINS["Hyperparameter Analysis"]]

    ablation_index = plugin_ids.index("ablation_paths")
    assert plugin_ids[ablation_index + 1] == "one_dimensional_analysis"


def test_toolbar_does_not_require_an_apply_button():
    (toolbar,) = analysis.OneDimensionalAnalysis.get_input_layout(
        lambda input_id, *_args, **_kwargs: input_id
    )

    assert not any(isinstance(component, html.Button) for component in _walk_components(toolbar))


def test_every_control_shares_the_single_toolbar_row():
    registered = []
    (toolbar,) = analysis.OneDimensionalAnalysis.get_input_layout(
        lambda input_id, *_args, **_kwargs: registered.append(input_id) or input_id
    )

    assert registered == [
        "objective_id",
        "hyperparameter_name",
        "elite_proportion",
    ]
    assert "analysis-toolbar" in toolbar.className


def _walk_components(component):
    """Yield a Dash component and all of its descendants."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children"):
            yield from _walk_components(child)


@pytest.mark.parametrize(
    ("hyperparameter", "values", "expected_groups"),
    [
        (
            UniformIntegerHyperparameter("integer", lower=1, upper=3),
            [1, 2, 3],
            ["1", "2", "3"],
        ),
        (
            CategoricalHyperparameter("category", choices=["a", "b"]),
            ["b", "a"],
            ["b", "a"],
        ),
    ],
)
def test_discrete_hyperparameters_keep_observed_values(
    hyperparameter,
    values,
    expected_groups,
):
    groups = analysis._group_parameter_values(pd.Series(values), hyperparameter)

    assert groups.astype(str).tolist() == expected_groups


def test_numeric_hyperparameters_use_linear_or_logarithmic_bins():
    linear = UniformFloatHyperparameter("linear", lower=1.0, upper=101.0)
    logarithmic = UniformFloatHyperparameter("logarithmic", lower=1.0, upper=100.0, log=True)

    linear_groups = analysis._group_parameter_values(
        pd.Series([1.0, 11.0, 101.0]),
        linear,
    )
    logarithmic_groups = analysis._group_parameter_values(
        pd.Series([1.0, 10.0, 100.0]),
        logarithmic,
    )

    assert len(linear_groups.cat.categories) == 10
    assert len(logarithmic_groups.cat.categories) == 10
    assert 11.0 in linear_groups.cat.categories[0]
    assert 10.0 in logarithmic_groups.cat.categories[4]


def test_summary_uses_median_elite_rank_instead_of_mean():
    hyperparameter = CategoricalHyperparameter("group", choices=["a", "b"])
    trials = pd.DataFrame(
        {
            "trial_number": range(8),
            "hyperparameter_value": ["a", "a", "b", "a", "b", "b", "b", "b"],
            "objective": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        }
    )

    summary, tail_quantile, threshold, elite_count = analysis._summarize_trials(
        trials,
        hyperparameter,
        optimize="upper",
        elite_proportion=0.5,
    )

    group_a = next(row for row in summary if row["range"] == "a")
    assert group_a["elite_median_rank"] == pytest.approx(2.0)
    assert group_a["elite_median_rank"] != pytest.approx((1 + 2 + 4) / 3)
    assert group_a["elite_rate_pct"] == pytest.approx(100.0)
    # 100% elite in the bin against a 50% global elite share.
    assert group_a["elite_enrichment"] == pytest.approx(2.0)
    assert tail_quantile == pytest.approx(0.9)
    assert threshold == pytest.approx(6.5)
    assert elite_count == 4


def test_minimization_uses_lower_tail_and_lower_values_as_elite():
    hyperparameter = CategoricalHyperparameter("group", choices=["a", "b"])
    trials = pd.DataFrame(
        {
            "trial_number": range(5),
            "hyperparameter_value": ["a", "a", "b", "b", "b"],
            "objective": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    summary, tail_quantile, threshold, elite_count = analysis._summarize_trials(
        trials,
        hyperparameter,
        optimize="lower",
        elite_proportion=0.2,
    )

    assert tail_quantile == pytest.approx(0.1)
    assert threshold == pytest.approx(1.8)
    assert elite_count == 1
    assert summary[0]["elite_median_rank"] == pytest.approx(1.0)
    assert summary[1]["elite_median_rank"] is None


def test_dependencies_exclude_time_and_preserve_analysis_controls():
    objectives = [
        Objective("Loss", optimize="lower"),
        Objective("Accuracy", optimize="upper"),
        Objective("Time"),
    ]
    configspace = ConfigurationSpace(
        space={
            "x": (0.0, 1.0),
            "kind": ["a", "b"],
        }
    )

    class DependencyRun:
        def get_objectives(self):
            return objectives

        def get_objective_id(self, objective):
            return objectives.index(objective)

    run = DependencyRun()
    run.configspace = configspace
    inputs = {
        "objective_id": {"value": 1},
        "hyperparameter_name": {"value": "x"},
        "elite_proportion": {"value": 0.2},
    }
    previous_inputs = {
        **inputs,
        "elite_proportion": {"value": 0.1},
    }

    result = analysis.OneDimensionalAnalysis.load_dependency_inputs(
        object.__new__(analysis.OneDimensionalAnalysis),
        run,
        previous_inputs,
        inputs,
    )

    assert [option["label"] for option in result["objective_id"]["options"]] == [
        "Loss",
        "Accuracy",
    ]
    assert result["objective_id"]["value"] == 1
    assert result["elite_proportion"]["value"] == pytest.approx(0.2)


def test_all_plots_are_immediate(monkeypatch):
    monkeypatch.setattr(analysis, "save_image", lambda *_args: None)
    objective = Objective("Accuracy", optimize="upper")
    hyperparameter = CategoricalHyperparameter("kind", choices=["a", "b"])
    run = SimpleNamespace(
        configspace={"kind": hyperparameter},
        get_objective=lambda _objective_id: objective,
    )
    outputs = {
        "trials": [
            {"trial_number": 2, "hyperparameter_value": "a", "objective": 0.7},
            {"trial_number": 5, "hyperparameter_value": "b", "objective": 0.9},
        ],
        "summary": [
            {
                "range": "a",
                "n": 1,
                "elite": 0,
                "elite_rate_pct": 0.0,
                "elite_enrichment": 0.0,
                "elite_median_rank": None,
                "objective_median": 0.7,
                "objective_tail": 0.7,
            },
            {
                "range": "b",
                "n": 1,
                "elite": 1,
                "elite_rate_pct": 100.0,
                "elite_enrichment": 2.0,
                "elite_median_rank": 1.0,
                "objective_median": 0.9,
                "objective_tail": 0.9,
            },
        ],
        "tail_quantile": 0.9,
        "elite_threshold": 0.88,
        "elite_count": 1,
    }
    base_inputs = {
        "objective_id": 0,
        "hyperparameter_name": "kind",
        "elite_proportion": 0.1,
    }

    trials_figure, performance_figure, enrichment_figure = (
        analysis.OneDimensionalAnalysis.load_outputs(
            run,
            base_inputs,
            outputs,
        )
    )
    assert [trace.type for trace in trials_figure.data] == ["scatter"]
    assert [trace.type for trace in performance_figure.data] == ["scatter", "scatter"]
    ranked, without_rank, baseline = enrichment_figure.data
    assert list(ranked.y) == pytest.approx([2.0])
    assert list(ranked.marker.color) == pytest.approx([1.0])
    assert ranked.marker.reversescale is True
    assert "Median elite rank" in ranked.hovertemplate
    assert without_rank.marker.color == analysis.DEPLETED_COLOR
    assert without_rank.customdata[0][-1] == "n/a"
    assert list(baseline.y) == pytest.approx([1.0, 1.0])
    assert baseline.name == "1.0× baseline"


def test_range_panels_stay_aligned_and_fit_next_to_the_trials_panel(monkeypatch):
    monkeypatch.setattr(analysis, "save_image", lambda *_args: None)
    objective = Objective("Accuracy", optimize="upper")
    hyperparameter = CategoricalHyperparameter("kind", choices=["a", "b"])
    run = SimpleNamespace(
        configspace={"kind": hyperparameter},
        get_objective=lambda _objective_id: objective,
    )
    outputs = {
        "trials": [{"trial_number": 2, "hyperparameter_value": "a", "objective": 0.7}],
        "summary": [
            {
                "range": "a",
                "n": 1,
                "elite": 1,
                "elite_rate_pct": 100.0,
                "elite_enrichment": 1.0,
                "elite_median_rank": 1.0,
                "objective_median": 0.7,
                "objective_tail": 0.7,
            }
        ],
        "tail_quantile": 0.9,
        "elite_threshold": 0.5,
        "elite_count": 1,
    }

    trials_figure, performance_figure, enrichment_figure = (
        analysis.OneDimensionalAnalysis.load_outputs(
            run,
            {
                "objective_id": 0,
                "hyperparameter_name": "kind",
                "elite_proportion": 0.1,
            },
            outputs,
        )
    )

    # Identical bins, order and left margin are what couples the two range panels.
    for figure in (performance_figure, enrichment_figure):
        assert list(figure.layout.xaxis.categoryarray) == ["a"]
        assert figure.layout.xaxis.categoryorder == "array"
        assert figure.layout.margin.l == analysis.PANEL_LEFT_MARGIN
        assert figure.layout.xaxis.automargin is False
        assert figure.layout.yaxis.automargin is False

    # The stacked range panels plus the grid gap add up to the trials panel height.
    assert (
        analysis.PERFORMANCE_HEIGHT + analysis.ENRICHMENT_HEIGHT + analysis.GRID_GAP
        == analysis.TRIALS_HEIGHT
    )
    # No panel reserves space for a bottom legend block.
    assert trials_figure.layout.showlegend is False
    assert performance_figure.layout.legend.y > 1


def test_output_layout_is_one_grid_of_three_cards():
    registered = []
    layout = analysis.OneDimensionalAnalysis.get_output_layout(
        lambda name, *_args, **_kwargs: registered.append(name) or name
    )

    (grid,) = layout
    assert grid.className == "analysis-grid"
    assert registered == ["trials_graph", "performance_graph", "enrichment_graph"]
    assert [card.className for card in grid.children] == [
        "analysis-card analysis-card--tall",
        "analysis-card",
        "analysis-card",
    ]


def test_only_optuna_studies_are_compatible():
    optuna_run = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": 1})
    group = SimpleNamespace(prefix="group", empty=lambda: False, configspace={"x": 1})

    assert analysis.OneDimensionalAnalysis.check_run_compatibility(optuna_run)
    assert not analysis.OneDimensionalAnalysis.check_run_compatibility(group)
