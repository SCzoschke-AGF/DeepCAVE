"""Controls and explanations for coverage plans in the Global coverage workspace."""

from typing import Any, Dict

import json
from itertools import combinations
from math import prod

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dash_table, dcc, html

from deepcave.evaluators.combinatorial import RESOLUTIONS
from deepcave.plugins.hyperparameter.coverage_views import _id

_PERSISTENCE = {"persistence": True, "persistence_type": "session"}


def strength_options(next_strength: Any = None, maximum: Any = None) -> list:
    """Describe coverage goals using the selected resolution's measured strengths."""
    return [
        {
            "label": html.Span(
                [html.Strong(label), html.Small(description)],
                className="coverage-planning-choice",
            ),
            "value": value,
        }
        for value, label, description in (
            (
                "next",
                "Reach next strength" + (f" (t = {next_strength})" if next_strength else ""),
                (
                    f"Close remaining gaps to reach strength {next_strength}."
                    if next_strength
                    else "Close remaining gaps at the next interaction order."
                ),
            ),
            ("custom", "Reach custom strength", "Choose which interaction order to cover."),
            (
                "maximum",
                "Reach maximum strength" + (f" (t = {maximum})" if maximum else ""),
                "Full discretized search-space coverage.",
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
                            html.H5("1. Coverage resolution"),
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
                            html.H5("2. Coverage goal"),
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
                    html.H5("3. Optimization Goal"),
                    dcc.RadioItems(
                        id=_id("planning-strategy"),
                        options=[
                            {
                                "label": html.Span(
                                    [
                                        html.Strong("Minimum budget (trials) required to reach the coverage goal"),
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
                                        html.Strong("Optimal coverage increase with a fixed budget (trials)"),
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


def _plan_trajectory(plan: Dict[str, Any]) -> Any:
    """Keep the marginal-gain curve available for lower interaction strengths."""
    trials = plan["trials"]
    figure = go.Figure()
    figure.add_scatter(
        x=[0, len(trials)],
        y=[plan["initial_coverage"]] * 2,
        mode="lines",
        name="Observed coverage",
        line=dict(color="#64748b", dash="dash"),
    )
    figure.add_scatter(
        x=list(range(len(trials) + 1)),
        y=[plan["initial_coverage"]] + [trial["cumulative_coverage"] for trial in trials],
        mode="lines+markers",
        name="Projected coverage",
        line=dict(color="#8661bd"),
        hovertemplate="%{x} additional trials<br>%{y:.1%} coverage<extra></extra>",
    )
    figure.add_hline(y=1, line_dash="dot", line_color="#94a3b8")
    figure.update_layout(
        height=290,
        margin=dict(l=55, r=20, t=25, b=45),
        xaxis=dict(title="Additional trials", dtick=1 if len(trials) < 20 else None),
        yaxis=dict(title="Coverage", tickformat=".0%", range=[0, 1.03]),
        legend=dict(orientation="h", y=1.15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return html.Details(
        [
            html.Summary("Show coverage gain by trial"),
            dcc.Graph(figure=figure, config={"displaylogo": False}, responsive=True),
        ],
        className="coverage-plan-trajectory",
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
    dimensions = len(parameters)
    total = sum(
        prod(len(parameter["levels"]) for parameter in group)
        for group in combinations(parameters, strength)
    )
    missing = plan["missing_tuples_before"]
    remaining = plan["missing_tuples_after"]
    observed = total - missing
    added = missing - remaining
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
        badge, outcome = "Target not reached", "incomplete"
    elif plan["optimality_proven"]:
        badge, outcome = "Optimal solution", "optimal"
    else:
        badge, outcome = "Heuristic solution", "heuristic"
    trial_metric = (
        "Additional trials required" if minimum_proven else "Additional trials proposed",
        f"{count:,}",
        (
            "Proven minimum"
            if minimum_proven
            else f"Known lower bound: {plan['lower_bound_trials']:,} trials"
        ),
        "trials",
    )
    coverage_metric = (
        "Projected coverage",
        f"{percent(plan['initial_coverage'])} → {percent(plan['projected_coverage'])}",
        f"At target Strength {strength}",
        "coverage",
    )
    if minimize:
        metrics = [trial_metric, coverage_metric]
    else:
        budget = plan["max_additional_trials"]
        metrics = [
            coverage_metric,
            (
                "Additional trials used",
                f"{count:,} / {budget:,}",
                f"{count:,} of {budget:,} trial budget used",
                "trials",
            ),
        ]
    reasons = {
        "target_reached": "The plan covers all combinations at the target strength.",
        "budget_exhausted": "The additional trial budget has been reached.",
        "timeout": "Search time elapsed; the validated proposals remain available.",
        "trial_limit": "The planning limit of 1,000 additional trials has been reached.",
        "no_gain_found": "The search found no further improvement; this is not an impossibility proof.",
    }
    if not missing:
        explanation_title = "Target already covered"
        explanation = (
            f"Observed coverage at Strength {strength} is already 100%. "
            "No additional trials are needed."
        )
    elif not complete:
        explanation_title = "Why was the target not reached?"
        explanation = reasons[plan["termination_reason"]]
        if plan["budget_target_impossible"]:
            explanation += (
                f" Full target coverage requires at least {plan['lower_bound_trials']:,} "
                "additional trials, exceeding this budget."
            )
        else:
            explanation += f" Known lower bound for full coverage: {plan['lower_bound_trials']:,} additional trials."
        explanation += (
            f" After successful execution, {remaining:,} combinations would still be missing."
        )
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
                            html.Small("Ready for execution · Not yet applied"),
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
            html.Header(
                [
                    html.H4("Coverage Improvement Plan"),
                    html.Span(
                        ("✓ " if outcome == "optimal" else "") + badge,
                        className=f"coverage-plan-badge coverage-plan-{outcome}",
                    ),
                ],
                className="coverage-plan-heading",
            ),
            html.P(
                f"Target: Strength {strength} / {dimensions} · {plan['resolution']} levels · "
                + (
                    "Minimum budget (trials) required to reach the coverage goal"
                    if minimize
                    else "Optimal coverage increase with a fixed budget (trials)"
                ),
                className="coverage-plan-target",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Small(label), html.Strong(value), html.Span(hint)],
                        className=f"coverage-plan-metric coverage-plan-metric-{kind}",
                    )
                    for label, value, hint, kind in metrics
                ],
                className="coverage-plan-metrics",
            ),
            html.Section(
                [
                    html.H5("Coverage breakdown"),
                    html.Div(
                        [
                            html.Span(
                                className=f"coverage-plan-segment coverage-plan-segment-{kind}",
                                style={"width": f"{100 * amount / total:.8g}%"},
                            )
                            for kind, amount in (
                                ("observed", observed),
                                ("added", added),
                                ("remaining", remaining),
                            )
                        ],
                        className="coverage-plan-bar",
                        role="img",
                        **{
                            "aria-label": f"{observed:,} already covered; {added:,} covered by plan; "
                            f"{remaining:,} remaining after successful execution. Total: {total:,}."
                        },
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"Already covered: {observed:,}",
                                className="coverage-plan-key-observed",
                            ),
                            html.Span(
                                f"Covered by plan: {added:,}", className="coverage-plan-key-added"
                            ),
                        ],
                        className="coverage-plan-legend",
                    ),
                    html.Dl(
                        [
                            html.Div(
                                [html.Dt("Total possible combinations"), html.Dd(f"{total:,}")]
                            ),
                            html.Div(
                                [
                                    html.Dt("Remaining after successful execution"),
                                    html.Dd(
                                        f"{remaining:,}",
                                        className=(
                                            "coverage-plan-remaining"
                                            if remaining
                                            else "coverage-plan-complete"
                                        ),
                                    ),
                                ]
                            ),
                        ],
                        className="coverage-plan-counts",
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
            _plan_trajectory(plan) if strength < dimensions and trials else None,
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
