from types import SimpleNamespace

from deepcave.plugins import Plugin
from deepcave.plugins.budget.budget_correlation import BudgetCorrelation
from deepcave.plugins.hyperparameter.ablation_paths import AblationPaths
from deepcave.plugins.hyperparameter.configuration_cube import ConfigurationCube
from deepcave.plugins.hyperparameter.hpo_planning import HPOPlanning
from deepcave.plugins.hyperparameter.importances import Importances
from deepcave.plugins.hyperparameter.one_dimensional_analysis import (
    OneDimensionalAnalysis,
)
from deepcave.plugins.hyperparameter.parallel_coordinates import ParallelCoordinates
from deepcave.plugins.hyperparameter.pdp import PartialDependencies
from deepcave.plugins.hyperparameter.symbolic_explanations import SymbolicExplanations
from deepcave.plugins.hyperparameter.two_dimensional_analysis import (
    TwoDimensionalAnalysis,
)
from deepcave.plugins.objective.cost_over_time import CostOverTime
from deepcave.plugins.objective.pareto_front import ParetoFront
from deepcave.plugins.summary.footprint import FootPrint
from deepcave.runs.objective import Objective


def test_every_objective_selector_plugin_declares_its_primary_input() -> None:
    plugin_classes = (
        BudgetCorrelation,
        AblationPaths,
        ConfigurationCube,
        HPOPlanning,
        Importances,
        OneDimensionalAnalysis,
        ParallelCoordinates,
        PartialDependencies,
        SymbolicExplanations,
        TwoDimensionalAnalysis,
        CostOverTime,
        ParetoFront,
        FootPrint,
    )
    expected_inputs = {
        "budget_correlation": "objective_id",
        "ablation_paths": "objective_id1",
        "ccube": "objective_id",
        "hpo_planning": "objective_id",
        "importances": "objective_id1",
        "one_dimensional_analysis": "objective_id",
        "parallel_coordinates": "objective_id",
        "pdp": "objective_id",
        "symbolic_explanations": "objective_id",
        "two_dimensional_analysis": "objective_id",
        "cost_over_time": "objective_id",
        "pareto_front": "objective_id_1",
        "footprint": "objective_id",
    }
    assert {plugin.id: plugin.main_objective_input for plugin in plugin_classes} == expected_inputs

    for plugin_class in plugin_classes:
        locally_registered = []
        plugin_class.get_input_layout(
            lambda name, *_args, **_kwargs: locally_registered.append(name) or name
        )
        assert expected_inputs[plugin_class.id] not in locally_registered

        plugin = plugin_class()
        registered = {(id, attribute) for id, attribute, *_ in plugin.inputs}
        main_input = expected_inputs[plugin.id]
        assert (main_input, "value") in registered
        assert (main_input, "options") in registered


def test_global_selection_updates_local_input_direction_and_cache_signature() -> None:
    plugin = SimpleNamespace(main_objective_input="objective_id")
    run = SimpleNamespace(
        meta={
            "objectives": [
                Objective("Loss").to_json(),
                Objective("user_attr:accuracy", optimize="upper").to_json(),
            ],
            "user_attr_objectives": {"accuracy": 1},
        },
        runs=[],
        get_objective_ids=lambda: [0, 1],
    )
    inputs = {"objective_id": {"value": 0}}

    Plugin.apply_global_main_objective(
        plugin,
        inputs,
        run,
        {"value": "user_attr:accuracy", "direction": "lower"},
        include_cache_signature=True,
    )

    assert inputs["objective_id"]["value"] == 1
    assert inputs["_global_main_objective"] == {
        "value": "user_attr:accuracy",
        "direction": "lower",
    }
    assert run.meta["objectives"][1]["optimize"] == "lower"
