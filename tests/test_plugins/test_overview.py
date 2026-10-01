from types import SimpleNamespace

import dash_bootstrap_components as dbc
from dash import dcc

from deepcave.plugins.summary.overview import Overview
from deepcave.runs.status import Status


def _walk_components(component):
    """Yield a Dash component and all descendants."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "to_plotly_json"):
            yield from _walk_components(child)


def test_status_layout_only_contains_summary_and_details():
    registered = []
    layout = Overview.get_output_layout(
        lambda output_id, _property: registered.append(output_id) or output_id
    )
    components = [component for item in layout for component in _walk_components(item)]

    assert "status_statistics" not in registered
    assert "config_statistics" not in registered
    assert "status_text" in registered
    assert "status_details" in registered
    assert not any(isinstance(component, (dbc.Tabs, dbc.Tab, dcc.Graph)) for component in components)


def test_status_summary_counts_successful_and_failed_trials():
    trials = [
        SimpleNamespace(
            status=Status.SUCCESS,
            budget=1.0,
            seed=0,
            config_id=0,
            end_time=1.0,
            additional={},
        ),
        SimpleNamespace(
            status=Status.FAILED,
            budget=1.0,
            seed=0,
            config_id=1,
            end_time=2.0,
            additional={},
        ),
        SimpleNamespace(
            status=Status.PRUNED,
            budget=1.0,
            seed=0,
            config_id=2,
            end_time=3.0,
            additional={},
        ),
    ]
    run = SimpleNamespace(
        prefix="Optuna",
        latest_change=0,
        history=trials,
        configspace=SimpleNamespace(get_hyperparameters_dict=lambda: {}),
        get_objectives=lambda: [],
        get_num_configs=lambda: 3,
        get_meta=lambda: {},
        get_trials=lambda: trials,
    )

    outputs = Overview.load_outputs(run)

    assert outputs[3] == "Successful trials: 1 · Failed trials: 2"
