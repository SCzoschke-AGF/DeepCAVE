"""Compact strength profiles and on-demand combinatorial coverage diagnoses."""

from typing import Any, Dict, Optional

from math import prod

from dash import dcc, html

from deepcave.plugins.hyperparameter.combinatorial_groups import group_diagnostics
from deepcave.plugins.hyperparameter.coverage_forecast_views import (
    forecast_cell,
    forecast_details,
)
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
            f"{cell['complete_groups']:,} / {cell['groups']:,} fully covered {cell['strength']}-parameter combinations",
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


def _configuration_cells(row: Dict, parameters: int, trials: int) -> list:
    """Count full binned configurations independently of next-strength coverage."""
    cell = _next_cell(row)
    label = "—"
    if cell:
        label, _, title = _cell_label(cell, trials)
    else:
        title = (
            "All strengths covered"
            if row["observed_strength"] == parameters
            else f"t = {row['observed_strength'] + 1}: not computed"
        )
    points = row.get("discretization", {}).get("points")
    unique = f"{len({tuple(point) for point in points}):,}" if points is not None else "—"
    return [
        html.Span(
            unique,
            className="coverage-strength-count",
            title=(
                "Unique full configurations after feature binning; repetitions count once."
                if points is not None
                else "Unique full configurations unavailable in this saved result."
            ),
        ),
        html.Span(
            f"{prod(int(size) for size in row['levels'].values()):,}",
            className="coverage-strength-count",
            title=f"All possible configurations at maximum strength t = {parameters}.",
        ),
        html.Span(
            [
                html.Small(f"t = {cell['strength']}") if cell else html.Small(title),
                html.Strong(label),
            ],
            className="coverage-strength-coverage",
            title=title,
        )
    ]


def combinatorial_view(
    result: Optional[Dict], baseline: Optional[Dict] = None, forecast: Optional[Dict] = None
) -> Any:
    """Compare achieved strength and remaining coverage at each resolution."""
    if not result:
        if forecast:
            return html.Div(
                [
                    html.P("Coverage array unavailable for this search space."),
                    forecast_details(forecast),
                ]
            )
        return html.P(
            "Run combinatorial coverage to measure the coverage profile of sampled trials."
        )
    options = []
    for row in result["rows"]:
        strength = row["observed_strength"]
        strength_reference = next(
            (
                item
                for item in (baseline or {}).get("strengths", [])
                if item["resolution"] == row["resolution"]
            ),
            None,
        )
        configuration_cells = _configuration_cells(row, len(result["parameters"]), result["trials"])
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
                        configuration_cells[2],
                        forecast_cell(
                            next(
                                (
                                    item
                                    for item in (forecast or {}).get("rows", [])
                                    if item["resolution"] == row["resolution"]
                                ),
                                None,
                            ),
                            include_full=False,
                        ),
                        *configuration_cells[:2],
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
                            html.Span("Coverage at next strength"),
                            html.Span(
                                "Needed trials for next strength "
                                f"({(forecast or {}).get('probability_threshold', 0.8):.0%} prob.)"
                            ),
                            html.Span("Nr. observed configurations"),
                            html.Span("Nr. possible configurations (total)"),
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
                        "across every combination of t parameters. Strength is the highest verified t "
                        "with every t-parameter combination complete; all lower strengths are then complete as well. "
                        "An exact strength of 0 means single-parameter coverage is incomplete. "
                        "Strengths are examined up to the first incomplete interaction or "
                        "the trial-count or computation limit. The trial-count bound is necessary, "
                        "not sufficient: a covering array is not guaranteed to exist."
                    ),
                    html.P(
                        "Continuous levels divide the theoretical sampling CDF, including log scales. "
                        "Ordered discrete values are grouped near theoretical quantiles without "
                        "splitting a value. Categories and ordinals retain their original values. "
                        "Constants are excluded. Successful trials contribute; repetitions add no "
                        "new combinations. Conditions and forbidden combinations are unsupported."
                    ),
                    _table(
                        [{"levels": row["resolution"], **row["levels"]} for row in result["rows"]]
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
                        "Vollständige Abdeckung für jede Parametergruppe bedeutet dabei nicht, "
                        "dass wir schon den gesamten Suchraum durchprobiert haben müssen. "
                        "Wie das funktioniert, lässt sich an einem kleineren Beispiel zeigen: "
                        "drei Parameter A, B und C mit jeweils zwei Levels, 0 und 1."
                    ),
                    html.P(
                        "Insgesamt gibt es acht mögliche Konfigurationen. Um Strength 2 zu "
                        "erreichen, genügen jedoch diese vier Trials:"
                    ),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th(label, scope="col")
                                        for label in ("Trial", "A", "B", "C")
                                    ]
                                )
                            ),
                            html.Tbody(
                                [
                                    html.Tr(
                                        [html.Th(trial, scope="row")]
                                        + [html.Td(value) for value in values]
                                    )
                                    for trial, values in enumerate(
                                        [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], start=1
                                    )
                                ]
                            ),
                        ],
                        className="table table-sm table-bordered",
                    ),
                    html.P(
                        "Betrachten wir jeweils nur zwei Spalten, finden wir für jedes "
                        "Parameterpaar – A × B, A × C und B × C – alle vier Kombinationen: "
                        "00, 01, 10 und 11."
                    ),
                    html.P(
                        html.Strong(
                            "Mit nur 4 statt 8 Trials haben wir bereits alle paarweisen "
                            "Kombinationen vollständig abgedeckt."
                        )
                    ),
                    html.P(
                        "Genau das bedeutet Strength 2: Jedes Paar ist vollständig abgedeckt. "
                        "Erst für Strength 3 müssten auch alle acht Kombinationen der drei "
                        "Parameter gemeinsam vorkommen."
                    ),
                ],
                className="coverage-strength-explanation",
            ),
            forecast_details(forecast),
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
