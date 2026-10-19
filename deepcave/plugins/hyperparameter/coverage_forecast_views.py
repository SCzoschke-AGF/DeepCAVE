"""Compact forecast values and explanations for the existing coverage array."""

from typing import Any, Dict, Optional

from dash import html


def trial_count(value: int) -> str:
    """Format full trial counts with thousands separators."""
    return f"{value:,}"


def forecast_cell(row: Optional[Dict], include_full: bool = True) -> Any:
    """Show compact trial budgets and retain method details in tooltips."""
    if row is None:
        return html.Span("Not computed", className="coverage-forecast-cell")
    lines = []
    for name, label in (("next", "Next"), ("full", "Full")):
        if name == "full" and not include_full:
            continue
        target = row[f"{name}_strength_target"]
        result = row[name]
        value = result.get("trials_p80")
        if name == "next" and target is None and value == 0:
            text = "All strengths covered"
        elif value is not None:
            text = ("≈ " if result["status"] == "Approximate" else "") + f"+{trial_count(value)}"
        elif result.get("upper_bound_trials") is not None and result.get("kind") == "upper_bound":
            text = f"~{trial_count(result['upper_bound_trials'])}"
        else:
            text = "Censored" if result["status"] == "Censored" else "Unavailable"
        explanation = f"{result['status']}. Method: {result.get('method') or 'unavailable'}. {result.get('reason', '')}"
        if value is not None:
            explanation += f" Additional trials: {value:,}."
        elif result.get("upper_bound_trials") is not None:
            explanation += f" Conservative sufficient budget: {result['upper_bound_trials']:,} additional trials."
        uncertainty = result.get("uncertainty")
        if uncertainty:
            prefix = (
                "95% confidence interval for the quantile"
                if uncertainty["kind"] == "quantile_confidence_interval"
                else "Rigorous bounds on the exact quantile"
            )
            explanation += f" {prefix}: {uncertainty['lower']:,} to {uncertainty['upper'] if uncertainty['upper'] is not None else 'unresolved'}."
        lines.append(
            html.Span(
                (
                    [html.Small(f"{label} · t={target}" if target is not None else label)]
                    if include_full
                    else []
                )
                + [html.Strong(text)],
                title=f"{label} · t={target}. {explanation}",
            )
        )
    return html.Span(lines, className="coverage-forecast-cell")


def forecast_details(forecast: Optional[Dict]) -> Any:
    """Explain the probability event, assumptions, methods and actual work performed."""
    if not forecast:
        return []
    records = []
    for row in forecast["rows"]:
        for name in ("next", "full"):
            result = row[name]
            uncertainty = result.get("uncertainty")
            interval = "—"
            if uncertainty:
                interval = (
                    f"{uncertainty['lower']:,} – "
                    + (
                        f"{uncertainty['upper']:,}"
                        if uncertainty["upper"] is not None
                        else "unresolved"
                    )
                    + (
                        " (95% CI)"
                        if uncertainty["kind"] == "quantile_confidence_interval"
                        else " (bounds)"
                    )
                )
            explanation = result.get("reason") or "Completed from the observed coverage state."
            if "simulations_completed" in result:
                explanation += (
                    f" Completed continuations: {result['simulations_completed']:,}; "
                    f"censored: {result['simulations_censored']:,}; "
                    f"not started: {result['simulations_not_started']:,}."
                )
            records.append(
                html.Tr(
                    [
                        html.Td(
                            f"{row['resolution']} levels · {name} · t={row[f'{name}_strength_target']}"
                        ),
                        html.Td(f"{result['status']} · {result.get('method') or '—'}"),
                        html.Td(interval),
                        html.Td(explanation),
                    ]
                )
            )
    return html.Details(
        [
            html.Summary("Trial forecast · method & uncertainty"),
            html.P(
                f"{forecast['probability_threshold']:.0%} is the probability of reaching 100% coverage "
                "at the stated strength. Every budget counts additional trials from the observed "
                "configurations. Full strength includes every varying parameter."
            ),
            html.P(forecast["sampling_model"] + ". " + forecast["assumption"]),
            html.P(
                "Continuous levels follow the sampling CDF; discrete levels use their configured "
                "probability masses. Poisson approximations apply to full configurations only. "
                "Upper bounds are conservative sufficient budgets, not estimated quantiles. "
                "Confidence intervals describe Monte Carlo uncertainty; bounds bracket the exact quantile."
            ),
            html.P(
                f"Requested simulations: {forecast['simulations']:,}; seed: {forecast['seed']}; "
                f"time limit: {forecast['options']['timeout']:g} s; "
                f"trial limit per continuation: {forecast['options']['max_trials']:,}."
            ),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th(label)
                                    for label in (
                                        "Target",
                                        "Status / method",
                                        "Uncertainty",
                                        "Explanation",
                                    )
                                ]
                            )
                        ),
                        html.Tbody(records),
                    ],
                    className="table table-sm",
                ),
                style={"overflowX": "auto"},
            ),
        ],
        className="coverage-details",
    )
