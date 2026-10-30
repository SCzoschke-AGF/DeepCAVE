"""Controls and explanations for coverage plans in the Global coverage workspace."""

from typing import Any, Dict, Optional

import json
from itertools import combinations
from math import prod
from numbers import Integral, Real
from urllib.parse import quote

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from deepcave.evaluators.combinatorial import RESOLUTIONS
from deepcave.plugins.hyperparameter.coverage_views import _id

_PERSISTENCE = {"persistence": True, "persistence_type": "session"}


def strength_options(next_strength: Any = None, maximum: Any = None) -> list:
    """Describe coverage goals using the selected resolution's measured strengths."""
    return [
        {
            "label": label,
            "value": value,
        }
        for value, label in (
            (
                "next",
                "Next" + (f" · t = {next_strength}" if next_strength else " strength"),
            ),
            ("custom", "Custom · t ="),
            (
                "maximum",
                "Maximum" + (f" · t = {maximum}" if maximum else " strength"),
            ),
        )
    ]


def planning_coverage(result: Any, resolution: Any) -> tuple:
    """Derive target strength options without triggering a new calculation."""
    row = next(
        (row for row in (result or {}).get("rows", []) if row["resolution"] == resolution),
        None,
    )
    maximum = len(result["parameters"]) if result else None
    next_strength = None
    if row:
        strength = row["observed_strength"]
        if strength < maximum:
            next_strength = strength + 1
    return strength_options(next_strength, maximum), maximum


def planning_panel() -> Any:
    """Keep all planning controls mounted without submitting a job."""
    target_settings = html.Details(
        [
            html.Summary(
                [
                    html.Span("1", className="coverage-planning-step"),
                    html.H5("Target Settings"),
                ]
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dbc.Label("Parameter resolution", html_for=_id("planning-resolution")),
                            dcc.Dropdown(
                                id=_id("planning-resolution"),
                                options=[{"label": f"{r} levels", "value": r} for r in RESOLUTIONS],
                                value=2,
                                clearable=False,
                                searchable=False,
                                className="coverage-planning-select",
                                **_PERSISTENCE,
                            ),
                        ],
                        className="coverage-planning-field",
                    ),
                    html.Div(
                        [
                            dbc.Label("Target strength", html_for=_id("planning-strength-mode")),
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id=_id("planning-strength-mode"),
                                        options=strength_options(),
                                        value="next",
                                        clearable=False,
                                        searchable=False,
                                        className="coverage-planning-select",
                                        **_PERSISTENCE,
                                    ),
                                    html.Div(
                                        [
                                            dbc.Label(
                                                "Custom strength",
                                                html_for=_id("planning-strength"),
                                                className="visually-hidden",
                                            ),
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
                                        className="coverage-planning-custom-strength",
                                    ),
                                ],
                                className="coverage-planning-target",
                            ),
                        ],
                        className="coverage-planning-field",
                    ),
                ],
                className="coverage-planning-top",
            ),
        ],
        open=True,
        className="coverage-planning-section coverage-planning-settings",
    )
    strategy = html.Section(
        [
            html.H5("Optimization goal"),
            dcc.RadioItems(
                id=_id("planning-strategy"),
                options=[
                    {
                        "label": "Minimize trials to reach target",
                        "value": "minimize_trials",
                    },
                    {
                        "label": "Maximize coverage within trial budget",
                        "value": "maximize_coverage",
                    },
                ],
                value="minimize_trials",
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
                style={"display": "none"},
                className="coverage-planning-field",
            ),
        ],
        className="coverage-planning-section coverage-planning-strategy",
    )
    advanced = html.Details(
        [
            html.Summary([html.Span("☷", **{"aria-hidden": "true"}), " Advanced"]),
            html.Div(
                [
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
                        "Large strengths may exceed the planning limits. "
                        "Search results are heuristic."
                    ),
                ],
                className="coverage-planning-advanced-content",
            ),
        ],
        className="coverage-planning-advanced",
    )
    actions = html.Section(
        [
            html.Div(
                [
                    dbc.Button(
                        "Compute Plan",
                        id=_id("planning-run"),
                        n_clicks=0,
                        size="sm",
                        className="coverage-planning-run",
                    ),
                    dbc.Button(
                        "Cancel",
                        id=_id("planning-cancel"),
                        n_clicks=0,
                        size="sm",
                        color="secondary",
                        outline=True,
                        style={"display": "none"},
                    ),
                ],
                className="coverage-planning-actions",
            ),
            html.Div(
                id=_id("planning-status"), role="status", className="coverage-planning-status"
            ),
        ],
        className="coverage-planning-section coverage-planning-action-section",
    )
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Improve Coverage"),
                        ],
                        className="coverage-planning-heading",
                    ),
                    advanced,
                ],
                className="coverage-planning-header",
            ),
            html.Div(
                [
                    html.Aside(
                        [target_settings, strategy, actions],
                        className="coverage-planning-controls",
                        **{"aria-label": "Coverage planning settings"},
                    ),
                    html.Div(
                        id=_id("planning-result"),
                        className="coverage-planning-result",
                    ),
                ],
                className="coverage-planning-workspace",
            ),
            dcc.Store(id=_id("planning-data")),
            dcc.Store(id=_id("planning-view")),
        ],
        className="coverage-planning",
    )


def optuna_export_study(run: Any) -> Dict[str, Any]:
    """Return available source study metadata for downloads only."""
    meta = getattr(run, "meta", {})
    return {
        key: meta[source]
        for key, source in (("name", "optuna_study_name"), ("id", "optuna_study_id"))
        if meta.get(source) is not None
    }


def plan_toml(
    plan: Dict[str, Any],
    study: Optional[Dict[str, Any]] = None,
    *,
    sampling_method: str = "coverage_guided",
) -> str:
    """Export configurations and the user attributes to attach when enqueuing them."""
    blocks = []
    if study:
        lines = ["[study]"]
        for key, value in study.items():
            encoded = json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
            lines.append(f"{key} = {encoded}")
        blocks.append("\n".join(lines))
    for trial in plan["trials"]:
        lines = ["[[trials]]"]
        for name, value in trial["parameters"].items():
            key = json.dumps(name, ensure_ascii=False).replace("\x7f", "\\u007f")
            if isinstance(value, (str, bool)):
                encoded = json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
            elif isinstance(value, (Integral, Real)):
                encoded = str(value)
            else:
                raise ValueError(f"TOML export cannot represent parameter {name!r}: {value!r}.")
            lines.append(f"{key} = {encoded}")
        blocks.append("\n".join(lines))
    blocks.append(
        "[trial_user_attrs]\nsampling_method = " + json.dumps(sampling_method, ensure_ascii=False)
    )
    return "\n\n".join(blocks) + "\n"


def plan_view(plan: Dict[str, Any], study: Optional[Dict[str, Any]] = None) -> Any:
    """Explain the saved plan's outcome, proof and next action before technical details."""
    if not plan:
        return html.P(
            "Generate a plan to see concrete configurations and their expected coverage gain."
        )
    trials = plan["trials"]
    count = len(trials)
    download = None
    if trials:
        try:
            download = html.A(
                "Export TOML",
                href="data:application/toml;charset=utf-8," + quote(plan_toml(plan, study), safe=""),
                download="coverage-plan.toml",
                className="btn btn-outline-secondary btn-sm coverage-plan-download",
                title=f"Download all {count:,} proposed configurations as TOML",
            )
        except ValueError as error:
            download = html.Span(
                "Export TOML",
                className="btn btn-outline-secondary btn-sm disabled coverage-plan-download",
                title=str(error),
                **{"aria-disabled": "true"},
            )
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
        ("observed", "already covered", observed),
        ("added", "newly covered", added),
        ("remaining", "remaining", remaining),
    )
    complete = plan["target_complete"]
    minimize = plan["strategy"] == "minimize_trials"

    def percent(value: float) -> str:
        if value == 1:
            return "100%"
        if 0 < value < 0.001:
            return "<0.1%"
        return ">99.9%" if value > 0.999 else f"{value:.1%}"

    trial_value = f"{count:,}" if minimize else f"{count:,} / {plan['max_additional_trials']:,}"

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
                f"Show {count:,} configurations →",
                className="coverage-plan-action",
            ),
            dash_table.DataTable(
                id=_id("planning-trials"),
                columns=[
                    {"name": "Order", "id": "position", "type": "numeric"},
                    {"name": "Covered combinations", "id": "gain", "type": "numeric"},
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
    planned_bar = html.Div(
        [
            html.Span(
                className=f"coverage-plan-segment coverage-plan-segment-{kind}",
                style={"width": f"{100 * amount / total:.8g}%"},
            )
            for kind, _, amount in segments
            if amount > 0
        ],
        className="coverage-plan-bar",
        role="img",
        **{
            "aria-label": f"{observed:,} already covered; {added:,} covered by plan; "
            f"{remaining:,} remaining after successful execution. Total: {total:,}."
        },
    )
    return html.Article(
        [
            html.Header(
                [
                    html.H4("Coverage Improvement Plan"),
                ],
                className="coverage-plan-heading",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Small("Current coverage"),
                                    html.Strong(percent(plan["initial_coverage"])),
                                ],
                                className="coverage-plan-current",
                            ),
                            html.Div(
                                [
                                    html.Small(
                                        f"+{trial_value} {'trial' if count == 1 else 'trials'}"
                                    ),
                                    html.Span(
                                        "→",
                                        className="coverage-plan-arrow",
                                        **{"aria-hidden": "true"},
                                    ),
                                ],
                                className="coverage-plan-transition",
                            ),
                            html.Div(
                                [
                                    html.Small("Planned coverage"),
                                    html.Strong(percent(plan["projected_coverage"])),
                                ],
                                className="coverage-plan-projected",
                            ),
                        ],
                        className="coverage-plan-comparison",
                    ),
                    planned_bar,
                    html.Div(
                        [
                            html.Span(
                                f"{'+' if kind == 'added' else ''}{amount:,} {label}",
                                className=f"coverage-plan-label coverage-plan-key-{kind}",
                            )
                            for kind, label, amount in segments
                            if amount > 0
                        ],
                        className="coverage-plan-counts",
                    ),
                ],
                className="coverage-plan-breakdown",
            ),
            (
                html.Div(
                    [configuration_details, download],
                    className="coverage-plan-actions",
                )
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
        className="coverage-plan-result",
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
