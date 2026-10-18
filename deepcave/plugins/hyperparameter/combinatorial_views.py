"""Compact strength profiles and on-demand combinatorial coverage diagnoses."""

from typing import Any, Dict, Optional

from math import ceil, floor

from dash import dcc, html

from deepcave.plugins.hyperparameter.combinatorial_groups import group_diagnostics
from deepcave.plugins.hyperparameter.coverage_views import _id, _table


def _percent(value: float) -> str:
    if value == 1:
        return "100%"
    if 0 < value < 0.001:
        return "<0.1%"
    return ">99.9%" if value > 0.999 else f"{100 * value:.1f}%"


def _next_cell(row: Dict) -> Optional[Dict]:
    return next(
        (cell for cell in row["cells"] if cell["strength"] == row["observed_strength"] + 1),
        None,
    )


def _cell_label(cell: Dict, trials: int) -> tuple:
    if cell["status"] == "computed":
        return (
            _percent(cell["coverage"]),
            "complete" if cell["missing"] == 0 else "partial",
            f"{cell['covered']:,} / {cell['total']:,} combinations · "
            f"{cell['complete_groups']:,} / {cell['groups']:,} fully covered groups",
        )
    if cell["status"] == "trial_limit":
        return (
            "—",
            "unavailable",
            f"Full coverage needs at least {cell['required_trials']:,} trials; "
            f"{trials:,} sampled trials are available.",
        )
    if cell["status"] == "not_computed":
        return "…", "unavailable", "Not evaluated after stopping this resolution's calculation."
    return "…", "unavailable", "Not computed: exact enumeration exceeds the work limit."


def _legend() -> Any:
    return html.Span(
        [
            html.Span([html.Span(className="coverage-benchmark-key-band"), "Random 5–95%"]),
            html.Span([html.Span(className="coverage-benchmark-key-median"), "Random median"]),
            html.Span([html.Span(className="coverage-benchmark-key-point"), "Your result"]),
        ],
        className="coverage-benchmark-legend",
        **{"aria-hidden": "true"},
    )


def _rank(summary: Dict) -> str:
    if summary.get("degenerate"):
        return "All random designs have the same value; no percentile."
    percentile = summary.get("percentile")
    return (
        f"Random percentile: {percentile:.0f}"
        if percentile is not None
        else "Percentile unavailable"
    )


def _strength_reference(summary: Optional[Dict]) -> str:
    if not summary or "median" not in summary:
        return "Random strength: awaiting values"
    if summary.get("censored"):
        return "Random strength: only lower bounds available"
    if summary.get("degenerate"):
        text = f"Random strength: all designs t = {summary['median']:g}"
    else:
        text = (
            f"Random strength: {summary['lower']:g}–{summary['upper']:g} "
            f"(median {summary['median']:g}) · {_rank(summary)}"
        )
    return text


def _coverage_range(value: float, summary: Optional[Dict]) -> tuple:
    """Zoom to the observation and reference band, with readable percentage endpoints."""
    values = [value]
    if summary:
        values.extend([summary["lower"], summary["upper"]])
        if summary.get("median") is not None:
            values.append(summary["median"])
    lower, upper = min(values), max(values)
    padding = max((upper - lower) * 0.15, 0.005 if summary else 0.05)
    return max(0, floor((lower - padding) * 100) / 100), min(1, ceil((upper + padding) * 100) / 100)


def _coverage_benchmark(
    cell: Dict, summary: Optional[Dict], trials: int, axis: Optional[tuple] = None
) -> Any:
    label, _, title = _cell_label(cell, trials)
    heading = html.Span(
        [
            html.Small(f"t = {cell['strength']}"),
            html.Strong(label),
        ],
        className="coverage-benchmark-heading",
    )
    if cell["status"] != "computed":
        return html.Span([heading, html.Small(title, className="coverage-benchmark-caption")])
    value = cell["coverage"]
    axis_lower, axis_upper = axis or _coverage_range(value, summary)

    def position(value: float) -> float:
        return 100 * (value - axis_lower) / (axis_upper - axis_lower)

    markers = []
    caption = "Random reference: awaiting values"
    assessment = ""
    if summary:
        lower, upper = summary["lower"], summary["upper"]
        markers.append(
            html.Span(
                className="coverage-benchmark-band",
                style={
                    "left": f"{position(lower):.6g}%",
                    "width": f"{position(upper) - position(lower):.6g}%",
                },
            )
        )
        median = summary.get("median")
        if median is not None:
            markers.append(
                html.Span(
                    className="coverage-benchmark-median",
                    style={"left": f"{position(median):.6g}%"},
                )
            )
        median_text = _percent(median) if median is not None else "refresh baseline"
        caption = f"Random reference: {_percent(lower)} – {_percent(upper)} (median: {median_text})"
        if "degenerate" in summary:
            comparison = (
                "Below the typical random range"
                if value < lower
                else (
                    "Above the typical random range"
                    if value > upper
                    else "Within the typical random range"
                )
            )
            assessment = f"{comparison} · {_rank(summary)}"
    markers.append(
        html.Span(className="coverage-benchmark-point", style={"left": f"{position(value):.6g}%"})
    )
    return html.Span(
        [
            heading,
            html.Span(
                markers,
                className="coverage-benchmark-track",
                role="img",
                title=f"{caption}. {assessment} Coverage scale: {_percent(axis_lower)}–{_percent(axis_upper)} (shared across resolutions)",
                **{
                    "aria-label": f"Observed coverage: {label}. {caption}. {assessment} Scale: {_percent(axis_lower)}–{_percent(axis_upper)}."
                },
            ),
            html.Span(
                [html.Small(_percent(axis_lower)), html.Small(_percent(axis_upper))],
                className="coverage-benchmark-axis",
            ),
        ],
        className="coverage-benchmark",
    )


def combinatorial_view(result: Optional[Dict], baseline: Optional[Dict] = None) -> Any:
    """Compare achieved strength and remaining coverage at each resolution."""
    if not result:
        return html.P(
            "Run combinatorial coverage to measure the coverage profile of sampled trials."
        )
    comparisons = {
        (cell["resolution"], cell["strength"]): cell for cell in (baseline or {}).get("cells", [])
    }
    ranges = []
    for row in result["rows"]:
        cell = _next_cell(row)
        if cell and cell["status"] == "computed":
            reference = comparisons.get((row["resolution"], cell["strength"]))
            ranges.append(_coverage_range(cell["coverage"], reference))
    shared_axis = (
        (min(lower for lower, _ in ranges), max(upper for _, upper in ranges)) if ranges else None
    )
    options = []
    for row in result["rows"]:
        strength = row["observed_strength"]
        cell = _next_cell(row)
        comparison = comparisons.get((row["resolution"], cell["strength"])) if cell else None
        strength_reference = next(
            (
                item
                for item in (baseline or {}).get("strengths", [])
                if item["resolution"] == row["resolution"]
            ),
            None,
        )
        if cell:
            next_coverage = _coverage_benchmark(cell, comparison, result["trials"], shared_axis)
        else:
            next_coverage = html.Small(
                "All strengths covered"
                if strength == len(result["parameters"])
                else f"t = {strength + 1}: not computed"
            )
        options.append(
            {
                "value": row["resolution"],
                "label": html.Span(
                    [
                        html.Strong(
                            f"{row['resolution']} levels", className="coverage-strength-resolution"
                        ),
                        html.Span(
                            [
                                html.Span(
                                    [
                                        html.Strong(
                                            f"t {'≥' if row['strength_is_lower_bound'] else '='} {strength}",
                                            className="coverage-strength-badge",
                                        ),
                                        html.Span(
                                            f"/{len(result['parameters'])}",
                                            className="coverage-strength-total",
                                        ),
                                    ],
                                    className="coverage-strength-value",
                                    title=_strength_reference(strength_reference),
                                ),
                                html.Small(
                                    f"max. feasible strength = {row['trial_bound']}",
                                    className="coverage-strength-reference",
                                    title="Trial-count upper bound for full coverage; necessary, not sufficient.",
                                ),
                            ],
                            className="coverage-strength-achieved",
                        ),
                        next_coverage,
                    ],
                    className="coverage-strength-row",
                ),
            }
        )
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Feature Binning"),
                            html.Span("Strength"),
                            html.Span(["Coverage at next strength", _legend()]),
                        ],
                        className="coverage-strength-row coverage-strength-head",
                    ),
                    dcc.Checklist(
                        id=_id("combinatorial-resolution"),
                        options=options,
                        value=[],
                        persistence=True,
                        persistence_type="memory",
                        labelStyle={"display": "flex", "alignItems": "center"},
                        className="coverage-strength-options",
                    ),
                    html.Div(
                        combinatorial_resolution_detail(None, None),
                        id=_id("combinatorial-resolution-detail"),
                    ),
                ],
                className="coverage-strength-profile",
            ),
            html.Details(
                [
                    html.Summary("Method & level definitions"),
                    html.P(
                        "Coverage is the sum of observed combinations divided by all combinations "
                        "across every group of t parameters. Strength is the highest verified t "
                        "with every group complete; all lower strengths are then complete as well. "
                        "An exact strength of 0 means single-parameter coverage is incomplete. "
                        "Strengths are examined up to the first incomplete interaction or "
                        "the trial-count or computation limit. The trial-count bound is necessary, "
                        "not sufficient: a covering array is not guaranteed to exist."
                    ),
                    html.P(
                        "Continuous levels divide the theoretical sampling CDF, including log scales. "
                        "Ordered discrete values are grouped near theoretical quantiles without "
                        "splitting a value. Categories and ordinals retain their original values. "
                        "Constants are excluded. All recorded trials, including failures and "
                        "repetitions, contribute. Conditions and forbidden combinations are unsupported."
                    ),
                    _table(
                        [
                            {"levels": row["resolution"], **row["levels"]}
                            for row in result["rows"]
                        ]
                    ),
                ],
                className="coverage-details",
            ),
            html.Details(
                [
                    html.Summary("Was bedeutet Strength?"),
                    html.P(
                        html.Strong(
                            "Strength beschreibt, bis zu wie vielen Parametern gleichzeitig alle "
                            "möglichen Wertekombinationen durch die Trials abgedeckt sind."
                        )
                    ),
                    html.P(
                        [
                            "Dafür wird jeder Parameter in eine bestimmte Anzahl von Wertebereichen (",
                            html.Em("Levels"),
                            ") aufgeteilt.",
                        ]
                    ),
                    html.P(
                        [
                            html.Strong("Beispiel:"),
                            " Wir haben 6 Parameter mit jeweils 2 Levels.",
                        ]
                    ),
                    html.Ul(
                        [
                            html.Li(
                                [
                                    html.Strong("Strength 1:"),
                                    " Bei jedem einzelnen Parameter wurden beide Wertebereiche besucht.",
                                ]
                            ),
                            html.Li(
                                [
                                    html.Strong("Strength 2:"),
                                    " Für jedes mögliche Parameterpaar wurden alle 4 Wertekombinationen beobachtet.",
                                ]
                            ),
                            html.Li(
                                [
                                    html.Strong("Strength 3:"),
                                    " Für jede mögliche Dreiergruppe wurden alle 8 Wertekombinationen beobachtet.",
                                ]
                            ),
                            html.Li(
                                [
                                    html.Strong("Strength 4:"),
                                    " Für jede mögliche Vierergruppe wurden alle 16 Wertekombinationen beobachtet.",
                                ]
                            ),
                        ]
                    ),
                    html.P(
                        "Das setzt sich fort, bis bei Strength 6 alle 64 Kombinationen des gesamten "
                        "Suchraums abgedeckt sind."
                    ),
                    html.P(
                        html.Strong(
                            "Je höher die Strength, desto mehr Parameter können wir gleichzeitig "
                            "betrachten, ohne dass Kombinationen fehlen."
                        )
                    ),
                    html.P(
                        [
                            "Eine Strength von 4 bedeutet also nicht, dass nur 4 von 6 Parametern "
                            "abgedeckt sind. Sie bedeutet, dass bei ",
                            html.Strong("jeder beliebigen Auswahl von 4 Parametern"),
                            " alle möglichen Wertekombinationen beobachtet wurden.",
                        ]
                    ),
                    html.P(
                        "Die Strength misst damit die Vollständigkeit der kombinatorischen Abdeckung."
                    ),
                ],
                className="coverage-strength-explanation",
            ),
        ]
    )


def combinatorial_resolution_detail(result: Optional[Dict], resolution: Any) -> Any:
    """Explain the selected resolution and rank its next-strength coverage deficits."""
    row = next(
        (row for row in (result or {}).get("rows", []) if row["resolution"] == resolution), None
    )
    if not row:
        return html.P(
            "Select a resolution above to see coverage by strength.",
            className="coverage-strength-prompt",
        )
    strength = row["observed_strength"]
    cell = _next_cell(row)
    if strength:
        explanation = f"All combinations up to strength {strength} are covered. "
    elif row["strength_is_lower_bound"]:
        explanation = "Single-parameter coverage has not been fully evaluated. "
    else:
        explanation = "Single-parameter coverage is incomplete. "
    if cell and cell["status"] == "computed":
        explanation += (
            f"At strength {cell['strength']}, {_percent(cell['coverage'])} of combinations "
            "are covered."
        )
    elif cell and cell["status"] == "trial_limit":
        explanation += f"Full coverage at strength {cell['strength']} requires more trials."
    elif strength < len(result["parameters"]):
        explanation += f"Coverage at strength {strength + 1} has not been computed."
    strip = []
    for current in row["cells"]:
        label, color, title = _cell_label(current, result["trials"])
        strip.append(
            html.Div(
                [
                    html.Small(f"t={current['strength']}"),
                    html.Strong("✓" if color == "complete" else label),
                ],
                className=f"coverage-strength-step coverage-ccm-{color}",
                title=title,
            )
        )
    diagnosis = None
    if cell and cell["status"] == "computed" and cell["missing"]:
        diagnosis = group_diagnostics(result, row, cell)
    return html.Section(
        [
            html.H4(f"{row['resolution']} Levels — Detail"),
            html.Small(
                f"Trial-count limit: t ≤ {row['trial_bound']}",
                title="Necessary, not sufficient, for full coverage.",
            ),
            html.P(explanation),
            html.Div(strip, className="coverage-strength-strip"),
            diagnosis,
        ],
        className="coverage-strength-detail",
    )
