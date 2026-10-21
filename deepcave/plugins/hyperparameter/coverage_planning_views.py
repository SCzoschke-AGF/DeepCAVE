"""Controls and explanations for coverage plans in the Global coverage workspace."""

from typing import Any, Dict

import json
from itertools import combinations
from math import prod

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from deepcave.evaluators.combinatorial import RESOLUTIONS
from deepcave.plugins.hyperparameter.coverage_views import _id

_PERSISTENCE = {"persistence": True, "persistence_type": "session"}


def strength_options(next_strength: Any = None, maximum: Any = None) -> list:
    """Describe coverage goals using the selected resolution's measured strengths."""
    return [
        {
            "label": html.Span(
                [html.Strong(label)],
                className="coverage-planning-choice",
            ),
            "value": value,
        }
        for value, label in (
            (
                "next",
                "Reach next strength" + (f" (t = {next_strength})" if next_strength else ""),
            ),
            ("custom", "Reach custom strength"),
            (
                "maximum",
                "Reach maximum strength" + (f" (t = {maximum})" if maximum else ""),
            ),
        )
    ]


def planning_coverage(result: Any, resolution: Any) -> tuple:
    """Reuse observed coverage without triggering a new calculation."""
    row = next(
        (row for row in (result or {}).get("rows", []) if row["resolution"] == resolution),
        None,
    )
    maximum = len(result["parameters"]) if result else None
    next_strength = None
    achieved, coverage = "—", "—"
    next_label = "Next strength"
    if row:
        strength = row["observed_strength"]
        sign = "≥" if row["strength_is_lower_bound"] else "="
        achieved = f"t {sign} {strength} / {maximum}"
        if strength < maximum:
            next_strength = strength + 1
            next_label = f"Next strength (t = {next_strength})"
            cell = next((cell for cell in row["cells"] if cell["strength"] == next_strength), None)
            if cell and cell["status"] == "computed":
                coverage = f"{cell['coverage']:.1%}"
        else:
            coverage = "All strengths covered"
    return (
        html.Div(
            [
                html.Div([html.Small("Achieved strength"), html.Strong(achieved)]),
                html.Div([html.Small(next_label), html.Strong(coverage)]),
                (
                    html.Span(
                        "Awaiting coverage measurements", className="coverage-planning-caption"
                    )
                    if not row
                    else None
                ),
            ],
            className="coverage-planning-metrics",
        ),
        strength_options(next_strength, maximum),
        maximum,
    )


def planning_panel() -> Any:
    """Keep all planning controls mounted without submitting a job."""
    controls = html.Div(
        [
            html.H4("Improve Coverage"),
            html.Div(
                [
                    html.Section(
                        [
                            html.H5("1. Choose parameter resolution"),
                            html.P("Choose how parameter values are discretized."),
                            dbc.RadioItems(
                                id=_id("planning-resolution"),
                                options=[{"label": f"{r} levels", "value": r} for r in RESOLUTIONS],
                                value=2,
                                inline=True,
                                className="coverage-planning-resolution",
                                **_PERSISTENCE,
                            ),
                            html.Div(
                                id=_id("planning-coverage"), children=planning_coverage(None, 2)[0]
                            ),
                        ],
                        className="coverage-planning-section coverage-planning-resolution-section",
                    ),
                    html.Section(
                        [
                            html.H5("2. Choose coverage goal"),
                            dcc.RadioItems(
                                id=_id("planning-strength-mode"),
                                options=strength_options(),
                                value="next",
                                labelStyle={"display": "flex"},
                                className="coverage-planning-options",
                                **_PERSISTENCE,
                            ),
                            html.Div(
                                [
                                    dbc.Label("Custom strength", html_for=_id("planning-strength")),
                                    dbc.Input(
                                        id=_id("planning-strength"),
                                        type="number",
                                        value=2,
                                        min=1,
                                        step=1,
                                        disabled=True,
                                        **_PERSISTENCE,
                                    ),
                                ],
                                id=_id("planning-strength-field"),
                                style={"display": "none"},
                                className="coverage-planning-field",
                            ),
                        ],
                        className="coverage-planning-section coverage-planning-goal",
                    ),
                ],
                className="coverage-planning-top",
            ),
            html.Section(
                [
                    html.H5("3. Determine concrete missing configurations to optimize coverage"),
                    dcc.RadioItems(
                        id=_id("planning-strategy"),
                        options=[
                            {
                                "label": html.Span(
                                    [
                                        html.Strong(
                                            "Find minimum trials required to reach the coverage goal"
                                        ),
                                        html.Small(
                                            "Find a small set of configurations to reach the coverage goal."
                                        ),
                                    ],
                                    className="coverage-planning-choice",
                                ),
                                "value": "minimize_trials",
                            },
                            {
                                "label": html.Span(
                                    [
                                        html.Strong("Optimize coverage with a fix trial budget"),
                                        html.Small(
                                            "Get the largest coverage improvement for a fixed budget."
                                        ),
                                    ],
                                    className="coverage-planning-choice",
                                ),
                                "value": "maximize_coverage",
                            },
                        ],
                        value="maximize_coverage",
                        labelStyle={"display": "flex"},
                        className="coverage-planning-options",
                        **_PERSISTENCE,
                    ),
                    html.Div(
                        [
                            dbc.Label(
                                "Additional trial budget (required)",
                                html_for=_id("planning-budget"),
                            ),
                            dbc.Input(
                                id=_id("planning-budget"),
                                type="number",
                                value=10,
                                min=1,
                                max=1000,
                                step=1,
                                **_PERSISTENCE,
                            ),
                        ],
                        id=_id("planning-budget-field"),
                        className="coverage-planning-field",
                    ),
                ],
                className="coverage-planning-section coverage-planning-strategy",
            ),
            html.Details(
                [
                    html.Summary("Advanced settings"),
                    dbc.Label("Search time (seconds)", html_for=_id("planning-timeout")),
                    dbc.Input(
                        id=_id("planning-timeout"),
                        type="number",
                        value=30,
                        min=0.1,
                        max=120,
                        **_PERSISTENCE,
                    ),
                    dbc.Label("Random seed", html_for=_id("planning-seed"), className="mt-2"),
                    dbc.Input(
                        id=_id("planning-seed"),
                        type="number",
                        value=0,
                        min=0,
                        max=2**32 - 1,
                        step=1,
                        **_PERSISTENCE,
                    ),
                    html.Small(
                        "Large strengths may exceed the planning limits. Search results are heuristic."
                    ),
                ],
                className="coverage-planning-advanced",
            ),
            html.Div(id=_id("planning-status"), role="status", className="mb-2"),
            dbc.Button("Generate plan", id=_id("planning-run"), n_clicks=0, size="sm"),
            dbc.Button(
                "Cancel",
                id=_id("planning-cancel"),
                n_clicks=0,
                size="sm",
                color="secondary",
                outline=True,
                className="ms-2",
                style={"display": "none"},
            ),
            dcc.Store(id=_id("planning-data")),
            dcc.Store(id=_id("planning-view")),
        ],
        className="coverage-card coverage-planning-controls",
    )
    return html.Div(
        [controls, html.Div(id=_id("planning-result"))],
        className="coverage-planning",
    )


def plan_view(plan: Dict[str, Any]) -> Any:
    """Explain the saved plan's outcome, proof and next action before technical details."""
    if not plan:
        return html.P(
            "Generate a plan to see concrete configurations and their expected coverage gain."
        )
    trials = plan["trials"]
    count = len(trials)
    strength = plan["target_strength"]
    parameters = plan["search_space"]["parameters"]
    total = sum(
        prod(len(parameter["levels"]) for parameter in group)
        for group in combinations(parameters, strength)
    )
    missing = plan["missing_tuples_before"]
    remaining = plan["missing_tuples_after"]
    observed = total - missing
    added = missing - remaining
    segments = (
        ("observed", "Already covered configurations", observed, 0),
        ("added", "Newly covered configurations", added, observed),
        ("remaining", "Missing configurations", remaining, observed + added),
    )
    complete = plan["target_complete"]
    minimize = plan["strategy"] == "minimize_trials"
    minimum_proven = complete and plan["optimality_proven"] and count == plan["lower_bound_trials"]

    def percent(value: float) -> str:
        if value == 1:
            return "100%"
        if 0 < value < 0.001:
            return "<0.1%"
        return ">99.9%" if value > 0.999 else f"{value:.1%}"

    if not complete:
        outcome = "incomplete"
    elif plan["optimality_proven"]:
        outcome = "optimal"
    else:
        outcome = "heuristic"
    trial_noun = "trial" if count == 1 else "trials"
    trial_metric = (
        f"Additional {trial_noun} required" if minimum_proven else f"Additional {trial_noun} proposed",
        f"{count:,}",
        (
            "Proven minimum"
            if minimum_proven
            else f"Known lower bound: {plan['lower_bound_trials']:,} trials"
        ),
    )
    if not minimize:
        budget = plan["max_additional_trials"]
        trial_metric = (f"Additional {trial_noun} used", f"{count:,} / {budget:,}", None)
    trial_label, trial_value, trial_hint = trial_metric
    coverage_value = f"{percent(plan['initial_coverage'])} → {percent(plan['projected_coverage'])}"
    if not missing:
        explanation_title = "Target already covered"
        explanation = (
            f"Observed coverage at Strength {strength} is already 100%. "
            "No additional trials are needed."
        )
    elif not complete:
        explanation_title = explanation = None
    elif plan["optimality_proven"]:
        explanation_title = explanation = None
    else:
        explanation_title = "What does this solution guarantee?"
        explanation = (
            f"The plan reaches full target coverage with {count:,} additional trials. "
            f"The known lower bound is {plan['lower_bound_trials']:,} trials. "
            "A smaller plan may exist; minimum trial count has not been proven."
        )

    names = [parameter["name"] for parameter in parameters]
    names += list(plan["search_space"]["constants"])
    rows = [
        {
            "id": trial["position"],
            "position": trial["position"],
            "gain": trial["new_combinations"],
            "coverage": percent(trial["cumulative_coverage"]),
            "missing": trial["remaining_missing"],
            **{f"parameter:{name}": trial["parameters"][name] for name in names},
        }
        for trial in trials
    ]
    configuration_details = html.Details(
        [
            html.Summary(
                [
                    html.Span(
                        [
                            html.Strong(f"{count:,} proposed configurations"),
                        ]
                    ),
                    html.Span("Show configurations →", className="coverage-plan-action"),
                ],
                className="coverage-plan-footer",
            ),
            html.P(
                "All proposals are validated. Run these configurations in your optimizer to apply "
                "the plan. Select a row to inspect its parameters and the gaps it would close.",
                className="coverage-plan-note",
            ),
            dash_table.DataTable(
                id=_id("planning-trials"),
                columns=[
                    {"name": "Order", "id": "position", "type": "numeric"},
                    {"name": "New combinations", "id": "gain", "type": "numeric"},
                    {"name": "Cumulative coverage", "id": "coverage"},
                    {"name": "Missing combinations", "id": "missing", "type": "numeric"},
                    *[{"name": name, "id": f"parameter:{name}"} for name in names],
                ],
                data=rows,
                page_size=10,
                sort_action="native",
                cell_selectable=True,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "8px"},
            ),
            html.Div(id=_id("planning-trial-detail"), className="mt-3"),
        ],
        className="coverage-plan-configurations",
    )
    return html.Article(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(trial_value),
                            html.Div(
                                [
                                    html.Small(trial_label),
                                    html.Span(
                                        trial_hint,
                                        className="coverage-plan-proof" if minimum_proven else "",
                                    ) if trial_hint else None,
                                ],
                                className="coverage-plan-metric-copy",
                            ),
                        ],
                        className="coverage-plan-metric coverage-plan-metric-trials",
                    ),
                    html.Div(
                        [html.Strong(coverage_value), html.Small("Projected coverage change")],
                        className="coverage-plan-metric coverage-plan-metric-coverage",
                    ),
                ],
                className="coverage-plan-metrics",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.Span(
                                [label + ": ", html.Strong(f"{amount:,}")],
                                className=f"coverage-plan-label coverage-plan-key-{kind}",
                                **{"data-anchor": f"{100 * (offset + amount / 2) / total:.8g}"},
                            )
                            for kind, label, amount, offset in segments
                            if amount > 0
                        ],
                        className="coverage-plan-legend",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        className=f"coverage-plan-segment coverage-plan-segment-{kind}",
                                        style={"width": f"{100 * amount / total:.8g}%"},
                                    )
                                    for kind, _, amount, _ in segments
                                ],
                                className="coverage-plan-bar",
                                role="img",
                                **{
                                    "aria-label": f"{observed:,} already covered; {added:,} covered by plan; "
                                    f"{remaining:,} remaining after successful execution. Total: {total:,}."
                                },
                            ),
                        ],
                        className="coverage-plan-track",
                    ),
                    html.Div(className="coverage-plan-brace", **{"aria-hidden": "true"}),
                    html.Div(
                        [html.Strong(f"{total:,}"), f" possible configurations at strength {strength}"],
                        className="coverage-plan-total",
                    ),
                ],
                className="coverage-plan-breakdown",
            ),
            (
                html.Section(
                    [html.H5(explanation_title), html.P(explanation)],
                    className=f"coverage-plan-explanation coverage-plan-{outcome}",
                )
                if explanation
                else None
            ),
            (
                configuration_details
                if trials
                else html.Div(
                    [
                        html.Strong(
                            "No additional configurations required"
                            if complete
                            else "No proposed configurations"
                        ),
                        html.Small(
                            "Target already covered · Nothing to execute"
                            if complete
                            else "Target not reached · Nothing to execute"
                        ),
                    ],
                    className="coverage-plan-footer",
                )
            ),
        ],
        className="coverage-card coverage-plan-result",
    )


def plan_trial_detail(plan: Dict[str, Any], position: Any) -> Any:
    """Explain one proposed configuration using the saved plan's original levels."""
    trial = next((trial for trial in plan.get("trials", []) if trial["position"] == position), None)
    if trial is None:
        return []
    return html.Div(
        [
            html.H5(f"Proposed trial {position} · validated"),
            html.Pre(json.dumps(trial["parameters"], indent=2, ensure_ascii=False)),
            html.P(f"{trial['new_combinations']:,} newly covered combinations:"),
            html.Ul(
                [
                    html.Li(" · ".join(f"{name}: {value}" for name, value in combination.items()))
                    for combination in trial["closed_combinations"]
                ]
            ),
            (
                html.Small("Showing the first 20 closed combinations.")
                if trial["new_combinations"] > 20
                else None
            ),
        ]
    )
