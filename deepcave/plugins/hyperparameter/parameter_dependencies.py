"""Sampling dependency and conditional performance views over realized trials."""

from typing import Any, Callable, Dict, List

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, html
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import config
from deepcave.evaluators.parameter_dependencies import (
    conditional_effect,
    encode_parameter,
    orient_pair,
    pair_summary,
)
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.plugins.hyperparameter.analysis_utils import (
    _configuration_values,
    _get_objectives_without_time,
    _is_optuna_run_or_group,
)
from deepcave.runs import AbstractRun
from deepcave.runs.status import Status
from deepcave.utils.analysis_layout import (
    analysis_card,
    analysis_graph,
    analysis_toolbar_item,
    bin_label,
    horizontal_colorbar,
    style_analysis_panel,
)
from deepcave.utils.layout import get_select_options

PLUGIN_ID = "parameter_dependencies"
MATRIX_ID = f"{PLUGIN_ID}-matrix-output"
RANKING_ID = f"{PLUGIN_ID}-ranking-output"
DEFAULT_BINS = 4
DEFAULT_SUPPORT = 5
PANEL_HEIGHT = 380


def collect_sampled_trials(run: AbstractRun, objective_id: int) -> pd.DataFrame:
    """Retain every realized trial for dependency, finite successes for performance.

    Failed or missing-objective trials still describe the realized sampling design.
    Conditional/inactive parameters are excluded per pair rather than listwise.
    Repeated configurations count once per trial, matching the realized trial set.
    """
    names = list(run.configspace.keys())
    configurations: Dict[int, Any] = {}
    rows = []
    for trial in run.history:
        if trial.config_id not in configurations:
            configurations[trial.config_id] = _configuration_values(run, trial.config_id)
        values = configurations[trial.config_id]
        objective = trial.costs[objective_id]
        usable = trial.status == Status.SUCCESS and objective is not None and np.isfinite(objective)
        rows.append([*(values.get(name) for name in names), float(objective) if usable else np.nan])
    return pd.DataFrame(rows, columns=[*names, "__performance_objective__"])


def selected_pair(outputs: Dict[str, Any], target: str, conditioning: str) -> Dict[str, Any]:
    """Retrieve and orient the unique aggregate for a parameter pair."""
    names = outputs["names"]
    a, b = names.index(target), names.index(conditioning)
    return orient_pair(outputs["pairs"][f"{min(a, b)}:{max(a, b)}"], reverse=a > b)


def selection_updates(events: Dict[str, Any]) -> Dict[str, Any]:
    """Translate matrix clicks and stable ranking row IDs into the same controls."""
    click = events.get("matrix_click")
    points = click.get("points", []) if isinstance(click, dict) else []
    if points:
        target, conditioning = points[0].get("y"), points[0].get("x")
        if isinstance(target, str) and isinstance(conditioning, str) and target != conditioning:
            return {"target": {"value": target}, "conditioning": {"value": conditioning}}
    cell = events.get("ranking_cell")
    if isinstance(cell, dict) and isinstance(cell.get("row_id"), str):
        return {"conditioning": {"value": cell["row_id"]}}
    return {}


def _empty_figure(message: str) -> go.Figure:
    """Display an explicit unavailable result without inventing a score."""
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    style_analysis_panel(figure, height=PANEL_HEIGHT, bottom=100)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def matrix_figure(outputs: Dict[str, Any]) -> go.Figure:
    """Draw the symmetric objective-independent MI matrix with selectable pairs."""
    names = outputs["names"]
    scores = np.full((len(names), len(names)), np.nan)
    counts = np.zeros_like(scores, dtype=int)
    for key, pair in outputs["pairs"].items():
        a, b = map(int, key.split(":"))
        score = pair["dependency"]["score"]
        scores[a, b] = scores[b, a] = np.nan if score is None else score
        counts[a, b] = counts[b, a] = pair["dependency"]["n"]
    labels = [
        [
            "—" if i == j else "N/A" if not np.isfinite(value) else f"{value:.2f}"
            for j, value in enumerate(row)
        ]
        for i, row in enumerate(scores)
    ]
    # None keeps the diagonal and undefined pairs out of the continuous color scale.
    figure = go.Figure(
        go.Heatmap(
            x=names,
            y=names,
            z=scores,
            text=labels,
            texttemplate="%{text}",
            customdata=counts,
            zmin=0,
            zmax=1,
            colorscale="Blues",
            hoverongaps=False,
            colorbar=horizontal_colorbar("Normalized mutual information"),
            hovertemplate=(
                "%{y} × %{x}<br>NMI: %{z:.3f}<br>"
                "Jointly active trials: %{customdata}<extra></extra>"
            ),
        )
    )
    # Plotly omits heatmap text on null cells; annotate these explicitly.
    for i, row in enumerate(scores):
        for j, value in enumerate(row):
            if not np.isfinite(value):
                figure.add_annotation(x=names[j], y=names[i], text=labels[i][j], showarrow=False)
    style_analysis_panel(
        figure, height=max(PANEL_HEIGHT, 45 * len(names) + 170), bottom=140, left=150
    )
    figure.update_xaxes(type="category", tickangle=-35, automargin=True)
    figure.update_yaxes(type="category", autorange="reversed", automargin=True)
    return figure


def joint_figure(pair: Dict[str, Any], target: str, conditioning: str, kind: str) -> go.Figure:
    """Explain MI using joint counts, relative enrichment, or performance support."""
    if not pair["labels_a"] or not pair["labels_b"]:
        return _empty_figure("No jointly observed values available.")
    deviation = kind == "deviation"
    support = kind == "support"
    values = (
        pair["dependency"]["deviation"]
        if deviation
        else pair["performance_counts"] if support else pair["counts"]
    )
    observed = np.asarray(pair["counts"], dtype=float).T
    expected = np.asarray(pair["dependency"]["expected"], dtype=float).T
    figure = go.Figure(
        go.Heatmap(
            x=pair["labels_a"],
            y=pair["labels_b"],
            z=np.asarray(values, dtype=float).T,
            customdata=np.stack([observed, expected], axis=-1),
            colorscale="RdBu_r" if deviation else "Blues",
            zmid=0 if deviation else None,
            texttemplate="%{z:+.0%}" if deviation else "%{z:.0f}",
            hoverongaps=False,
            colorbar=horizontal_colorbar(
                "(O − E) / E" if deviation else "Successful trials" if support else "Sampled trials"
            ),
            hovertemplate=(
                "%{x} × %{y}<br>"
                + (
                    "Deviation: %{z:+.1%}<br>O: %{customdata[0]:.0f}<br>E: %{customdata[1]:.2f}"
                    if deviation
                    else "Count: %{z:.0f}"
                )
                + "<extra></extra>"
            ),
        )
    )
    style_analysis_panel(figure, height=PANEL_HEIGHT, bottom=120, left=100)
    figure.update_xaxes(
        type="category",
        title_text=target,
        ticktext=[bin_label(label) for label in pair["labels_a"]],
        tickvals=pair["labels_a"],
        automargin=True,
    )
    figure.update_yaxes(
        type="category",
        title_text=conditioning,
        ticktext=[bin_label(label) for label in pair["labels_b"]],
        tickvals=pair["labels_b"],
        automargin=True,
    )
    return figure


def conditional_figure(
    pair: Dict[str, Any], effect: Dict[str, Any], target: str, conditioning: str, objective: str
) -> go.Figure:
    """Show supported conditional medians, with gaps wherever support is too low."""
    if not effect["performance_n"]:
        return _empty_figure("No successful trials with a finite objective for this pair.")
    counts = np.asarray(pair["performance_counts"], dtype=int)
    medians = np.asarray(pair["medians"], dtype=float)
    supported = np.asarray(effect["supported"], dtype=bool)
    figure = go.Figure()
    for b, label in enumerate(pair["labels_b"]):
        if counts[:, b].sum() == 0:
            continue
        figure.add_trace(
            go.Scatter(
                x=pair["labels_a"],
                y=[
                    float(value) if supported[a, b] else None
                    for a, value in enumerate(medians[:, b])
                ],
                mode="lines+markers",
                connectgaps=False,
                name=f"{conditioning}: {bin_label(label, multiline=False)}",
                customdata=counts[:, b],
                hovertemplate=(
                    "%{x}<br>Median: %{y:.6g}<br>Successful trials: %{customdata}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    style_analysis_panel(figure, height=PANEL_HEIGHT, bottom=100, show_legend=True, top=90)
    figure.update_layout(legend={"font": {"size": 10}})
    figure.update_xaxes(
        type="category",
        title_text=target,
        ticktext=[bin_label(label) for label in pair["labels_a"]],
        tickvals=pair["labels_a"],
        automargin=True,
    )
    figure.update_yaxes(title_text=f"Median {objective}", automargin=True)
    return figure


class ParameterDependencies(DynamicPlugin):
    """Separate sampling dependencies, performance interactions, and interpretability."""

    id = PLUGIN_ID
    name = "Parameter Dependencies & Interactions"
    icon = "fas fa-th"
    help = "plugins/parameter_dependencies.html"
    activate_run_selection = True
    main_objective_input = "objective_id"
    input_update_inputs = (
        ("matrix_click", MATRIX_ID, "clickData"),
        ("ranking_cell", RANKING_ID, "active_cell"),
    )

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """Accept Optuna studies and Optuna-only groups with at least two parameters."""
        return _is_optuna_run_or_group(run) and not run.empty() and len(run.configspace) >= 2

    @staticmethod
    def get_input_layout(register: Callable) -> List[Component]:  # type: ignore[override]
        """Register local controls and expose only binning/support in the input toolbar."""
        register("target", ["value", "options"], filter=True)
        register("conditioning", ["value", "options"], filter=True)
        return [
            html.Div(
                [
                    analysis_toolbar_item(
                        "Adaptive bins",
                        dbc.Input(
                            id=register("n_bins", "value", type=int),
                            type="number",
                            min=2,
                            max=10,
                            step=1,
                        ),
                        "9rem",
                        "Quantile bins for numeric parameters; discrete levels are preserved "
                        "where possible. NMI estimates depend on binning and sample size.",
                    ),
                    analysis_toolbar_item(
                        "Minimum trials / cell",
                        dbc.Input(
                            id=register("min_support", "value", filter=True, type=int),
                            type="number",
                            min=2,
                            max=100,
                            step=1,
                        ),
                        "12rem",
                        "Only cells with this many successful, finite-objective trials "
                        "enter conditional curves and overlap diagnostics.",
                    ),
                ],
                className="analysis-toolbar d-flex flex-wrap align-items-center gap-2",
            )
        ]

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Provide defaults before a study is selected."""
        return {
            "objective_id": {"value": None, "options": []},
            "target": {"value": None, "options": []},
            "conditioning": {"value": None, "options": []},
            "n_bins": {"value": DEFAULT_BINS},
            "min_support": {"value": DEFAULT_SUPPORT},
        }

    def load_input_updates(self, events: Dict[str, Any]) -> Dict[str, Any]:
        """Consume pair-selection events using the shared input callback."""
        return selection_updates(events)

    def load_dependency_inputs(
        self,
        run: AbstractRun,  # type: ignore[override]  # activate_run_selection passes one run.
        _previous_inputs: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Keep both parameter selections valid when a study or target changes."""
        names = list(run.configspace.keys())
        target = inputs["target"]["value"]
        target = target if target in names else names[0]
        others = [name for name in names if name != target]
        conditioning = inputs["conditioning"]["value"]
        conditioning = conditioning if conditioning in others else others[0]
        objectives = _get_objectives_without_time(run)
        objective_ids = [run.get_objective_id(objective) for objective in objectives]
        objective_id = inputs["objective_id"]["value"]
        return {
            "target": {"options": get_select_options(names), "value": target},
            "conditioning": {"options": get_select_options(others), "value": conditioning},
            "objective_id": {
                "options": get_select_options(
                    [objective.name for objective in objectives], objective_ids
                ),
                "value": objective_id if objective_id in objective_ids else objective_ids[0],
            },
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate all unordered pairs once; target/support changes only redraw."""
        n_bins = int(inputs["n_bins"])
        if not 2 <= n_bins <= 10:
            raise ValueError("Adaptive bins must be between 2 and 10.")
        names = list(run.configspace.keys())
        table = collect_sampled_trials(run, int(inputs["objective_id"]))
        encoded = {
            name: encode_parameter(table[name], run.configspace[name], n_bins) for name in names
        }
        objective = table["__performance_objective__"].to_numpy(dtype=float)
        pairs = {}
        for a, name_a in enumerate(names):
            for b in range(a + 1, len(names)):
                codes_a, labels_a = encoded[name_a]
                codes_b, labels_b = encoded[names[b]]
                pairs[f"{a}:{b}"] = pair_summary(codes_a, codes_b, labels_a, labels_b, objective)
        return {"names": names, "pairs": pairs, "n_trials": len(table)}

    @staticmethod
    def get_output_layout(register: Callable) -> Component:
        """Arrange the dependency explanation separately from conditional effects."""

        def graph(name: str) -> Component:
            return analysis_graph(
                register(name, "figure"), PANEL_HEIGHT, config.FIGURE_DOWNLOAD_SCALE
            )

        def selector(name: str, label: str) -> Component:
            return analysis_toolbar_item(label, dbc.Select(id=f"{PLUGIN_ID}-{name}-input"), "12rem")

        return html.Div(
            [
                html.H5("Parameter Dependencies"),
                html.P(
                    "Which parameters were sampled dependently? NMI describes the realized "
                    "sampling design; it makes no performance or causal claim.",
                    className="text-muted",
                ),
                analysis_card("Pairwise dependency matrix · click a pair", graph("matrix")),
                html.Div(id=register("pair_title", "children"), className="mt-3 mb-2 fw-bold"),
                dbc.Row(
                    [
                        dbc.Col(
                            analysis_card("Joint distribution · sampled trials", graph("joint")),
                            md=6,
                        ),
                        dbc.Col(
                            analysis_card(
                                "Observed vs expected under independence", graph("deviation")
                            ),
                            md=6,
                        ),
                    ],
                    className="g-3",
                ),
                html.H5("Conditional Performance", className="mt-4"),
                html.P(
                    "Does another parameter change the target's observed performance effect? "
                    "Curves use successful trials only. Results are descriptive associations.",
                    className="text-muted",
                ),
                html.Div(
                    [
                        selector("target", "Target parameter"),
                        selector("conditioning", "Conditioning parameter"),
                    ],
                    className="analysis-toolbar d-flex flex-wrap gap-2 mb-3",
                ),
                dash_table.DataTable(
                    id=register("ranking", "data"),
                    columns=[
                        {"name": "Parameter", "id": "parameter"},
                        {"name": "Sampling dependency", "id": "sampling"},
                        {"name": "NMI", "id": "nmi", "type": "numeric"},
                        {"name": "Performance interaction", "id": "interaction"},
                        {"name": "Sweet-spot stability", "id": "stability"},
                        {"name": "Shared target bins", "id": "overlap"},
                        {"name": "Diagnosis", "id": "diagnosis"},
                    ],
                    sort_action="native",
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "8px", "whiteSpace": "normal"},
                    style_header={"fontWeight": "600"},
                ),
                html.Div(id=register("diagnostics", "children"), className="my-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            analysis_card(
                                "Conditional performance · supported medians", graph("conditional")
                            ),
                            md=8,
                        ),
                        dbc.Col(
                            analysis_card(
                                "Support / coverage · successful trials per cell", graph("support")
                            ),
                            md=4,
                        ),
                    ],
                    className="g-3",
                ),
                html.P(
                    "NMI bands: Low < 0.10, Medium < 0.30, High ≥ 0.30. Interaction bands use "
                    "the spread of centered conditional profiles relative to objective "
                    "variation: Low < 0.10, Medium < 0.30, Strong ≥ 0.30. These are descriptive "
                    "thresholds, not significance tests. Sparse samples and binning can inflate "
                    "NMI; zero does not prove independence.",
                    className="small text-muted mt-3",
                ),
            ]
        )

    @staticmethod
    def load_outputs(
        run: AbstractRun,  # type: ignore[override]  # activate_run_selection passes one run.
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> List[Any]:
        """Rank conditioning parameters and render the currently selected pair."""
        target, conditioning = inputs["target"], inputs["conditioning"]
        objective = run.get_objective(int(inputs["objective_id"]))
        if objective is None:
            raise PreventUpdate
        min_support = int(inputs["min_support"])
        if not 2 <= min_support <= 100:
            raise ValueError("Minimum trials per cell must be between 2 and 100.")
        ranking = []
        for other in outputs["names"]:
            if other == target:
                continue
            pair = selected_pair(outputs, target, other)
            effect = conditional_effect(pair, objective.optimize, min_support)
            ranking.append(
                {
                    "id": other,
                    "parameter": other,
                    "sampling": effect["sampling"],
                    "nmi": (
                        None
                        if pair["dependency"]["score"] is None
                        else round(pair["dependency"]["score"], 3)
                    ),
                    "interaction": effect["interaction"],
                    "stability": effect["stability"],
                    "overlap": f"{effect['overlap']:.0%}",
                    "diagnosis": effect["diagnosis"],
                    "interaction_score": effect["interaction_score"],
                }
            )
        ranking.sort(
            key=lambda row: (
                row["interaction_score"] is None,
                -(row["interaction_score"] or 0),
                row["parameter"],
            )
        )
        pair = selected_pair(outputs, target, conditioning)
        effect = conditional_effect(pair, objective.optimize, min_support)
        diagnostics = html.Div(
            [
                html.Strong(f"{effect['diagnosis']} · Sweet-spot stability: {effect['stability']}"),
                html.P(effect["explanation"], className="mb-1"),
                html.Small(
                    f"{pair['dependency']['n']} / {outputs['n_trials']} jointly active sampled "
                    f"trials; {pair['excluded']} excluded for inactive/missing values. "
                    f"{effect['performance_n']} successful finite-objective trials. "
                    f"{sum(effect['common'])} shared target bins ({effect['overlap']:.0%}) "
                    f"across {effect['regions']} observed conditioning regions, "
                    f"with ≥ {min_support} trials per cell. Comparability requires ≥ 2 shared "
                    "bins, ≥ 60% overlap and coverage of marginal and supported regional optima. "
                    "Unsupported cells remain gaps in the curves."
                ),
            ],
            className="analysis-card p-3",
        )
        # Return in registration order: matrix, title, joint, deviation, ranking, diagnostics,
        # conditional, support. Pair selection is a display filter and reuses all aggregates.
        return [
            matrix_figure(outputs),
            f"{target} × {conditioning}",
            joint_figure(pair, target, conditioning, "counts"),
            joint_figure(pair, target, conditioning, "deviation"),
            ranking,
            diagnostics,
            conditional_figure(pair, effect, target, conditioning, objective.name),
            joint_figure(pair, target, conditioning, "support"),
        ]
