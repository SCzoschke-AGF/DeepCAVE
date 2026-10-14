"""Focused, read-only diagnostic views for the search-space coverage suite."""

from typing import Any, Dict, List, Optional

from html import escape

import numpy as np
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Scheme
from plotly.colors import sample_colorscale, unlabel_rgb

from deepcave.evaluators.coverage import DEFAULT_BINS, CoverageData, _pair_grid

PREFIX = "search_space_coverage"
GROWTH_METRICS = {
    "p95_radius": "p95 coverage radius",
    "max_radius": "Max coverage radius",
    "centered_discrepancy": "Centered discrepancy",
    "pairwise_occupancy": "Pairwise occupancy",
    "marginal_max_gap": "Marginal max gap",
}
PAIR_METRICS = {
    "occupancy": "Occupancy",
    "empty_cell_rate": "Empty cells",
    "mmd": "MMD distance",
    "max_radius": "Coverage radius",
}
PAIR_PLOT_HEIGHT = 510
PAIR_METRIC_SCALES = {
    "occupancy": {"higher_is_better": True, "upper": 1.0, "power": 1.0},
    "empty_cell_rate": {"higher_is_better": False, "upper": 1.0, "power": 1.0},
    "mmd": {"higher_is_better": False, "upper": np.sqrt(2.0), "power": 0.5},
    "max_radius": {"higher_is_better": False, "upper": 1.0, "power": 0.5},
}
PAIR_CONCERN_COLORS = [
    [0.0, "#faf7f0"],
    [0.25, "#f3d9bd"],
    [0.5, "#dfa16b"],
    [0.75, "#ae593e"],
    [1.0, "#682c35"],
]


def _pair_severity(values: np.ndarray, metric: str) -> np.ndarray:
    """Map fixed mathematical metric domains to concern, independently of the run.

    Occupancy and empty cells are complementary. RMS radius is bounded by one;
    MMD with the nonnegative, unit-bounded pair kernel is bounded by sqrt(2).
    A fixed square root makes smaller distances visible without fitted thresholds.
    """
    scale = PAIR_METRIC_SCALES[metric]
    severity = np.clip(values / scale["upper"], 0, 1)
    if scale["higher_is_better"]:
        severity = 1 - severity
    return severity ** scale["power"]


def _pair_text_color(severity: float) -> str:
    """Choose the higher-contrast text color against the actual warm cell color."""
    color = sample_colorscale(PAIR_CONCERN_COLORS, [severity])[0]
    channels = np.array(unlabel_rgb(color)) / 255
    linear = np.where(channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)
    luminance = float(linear @ [0.2126, 0.7152, 0.0722])
    return "#ffffff" if 1.05 / (luminance + 0.05) > (luminance + 0.05) / 0.05 else "#000000"


def _pair_legend() -> Any:
    gradient = ", ".join(f"{color} {position * 100:g}%" for position, color in PAIR_CONCERN_COLORS)
    return html.Div(
        [
            html.Span("Coverage concern"),
            html.Span("Low"),
            html.Span(
                className="coverage-pair-concern-ramp",
                style={"background": f"linear-gradient(to right, {gradient})"},
                **{"aria-hidden": "true"},
            ),
            html.Span("High"),
        ],
        className="coverage-pair-concern-legend",
        title=(
            "Darker cells need more attention. Fixed scales across runs: "
            "1 − occupancy; empty-cell fraction; sqrt(radius); sqrt(MMD / sqrt(2)). "
            "Colors are descriptive, not statistical significance thresholds."
        ),
    )


def _id(name: str) -> str:
    return f"{PREFIX}-{name}"


def _number(value: Any, percent: bool = False) -> str:
    if value is None:
        return "Unavailable"
    if percent:
        return f"{100 * value:.2f}%"
    return f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)


def _metric(label: str, value: Any, hint: Optional[str] = None) -> Any:
    return html.Div(
        [
            html.Span(label, className="coverage-metric-label"),
            html.Strong(value, className="coverage-metric-value"),
            html.Small(hint) if hint else None,
        ],
        className="coverage-metric" + (" coverage-metric-gap" if label == "Largest gap" else ""),
    )


def _card(title: str, children: List[Any]) -> Any:
    return html.Section([html.H3(title), *children], className="coverage-card")


def _table(rows: List[Dict], columns: Optional[List[str]] = None, **kwargs: Any) -> Any:
    columns = (
        columns if columns is not None else list(dict.fromkeys(k for row in rows for k in row))
    )
    return dash_table.DataTable(
        data=rows,
        columns=[
            {
                "name": name.replace("_", " ").title(),
                "id": name,
                **(
                    {"type": "numeric", "format": FormatTemplate.percentage(2)}
                    if name
                    in (
                        "range_fraction",
                        "occupancy",
                        "empty_cell_rate",
                        "relative_deviation",
                    )
                    else (
                        {"type": "numeric", "format": Format(precision=4, scheme=Scheme.fixed)}
                        if name == "distance"
                        else {}
                    )
                ),
            }
            for name in columns
        ],
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "whiteSpace": "normal"},
        page_size=kwargs.pop("page_size", 10),
        **kwargs,
    )


def _figure(figure: go.Figure, x: str, y: str) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        xaxis_title=x,
        yaxis_title=y,
        height=330,
        margin=dict(l=55, r=25, t=25, b=55),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return figure


def _graph(figure: go.Figure, x: str, y: str) -> Any:
    return dcc.Graph(
        figure=_figure(figure, x, y),
        responsive=True,
        style={"height": "330px"},
        config={"displaylogo": False},
    )


def _sampler(run: Any) -> str:
    meta = getattr(run, "meta", {}) or {}
    sampler = meta.get("optuna_sampler") or meta.get("sampler") or meta.get("sampler_name")
    return str(sampler).removesuffix("Sampler") if sampler else "Not recorded"


def _definition_range(lower: Any, upper: Any) -> str:
    if lower is None or upper is None:
        return "—"
    return f"{lower:.12g} – {upper:.12g}"


def _definition_table(marginals: List[Dict]) -> Any:
    columns = ["parameter", "type", "search_space", "observed_range", "distribution", "step"]
    if any(m["conditional"] for m in marginals):
        columns.append("conditional")
    if any(m["probabilities"] for m in marginals):
        columns.append("probabilities")
    rows = []
    for m in marginals:
        distribution = m["distribution"].removesuffix("Float").removesuffix("Integer")
        distribution = distribution.replace("LogUniform", "Log-uniform")
        badge_class = (
            "coverage-chip coverage-chip--log" if distribution == "Log-uniform" else "coverage-chip"
        )
        rows.append(
            {
                "parameter": m["name"],
                "type": f'<span class="coverage-chip">{escape(m["type"])}</span>',
                "search_space": (
                    _definition_range(m["lower"], m["upper"])
                    if m["lower"] is not None
                    else ", ".join(map(str, m["choices"]))
                ),
                "observed_range": _definition_range(m["observed_min"], m["observed_max"]),
                "distribution": f'<span class="{badge_class}">{escape(distribution)}</span>',
                "step": m["step"],
                **({"conditional": m["conditional"]} if "conditional" in columns else {}),
                **(
                    {"probabilities": ", ".join(f"{p:.4g}" for p in m["probabilities"] or [])}
                    if "probabilities" in columns
                    else {}
                ),
            }
        )
    font = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    monospace = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    return dash_table.DataTable(
        id=_id("definition-table"),
        data=rows,
        columns=[
            {
                "name": name.replace("_", " ").title(),
                "id": name,
                **({"presentation": "markdown"} if name in ("type", "distribution") else {}),
                **({"type": "numeric"} if name == "step" else {}),
            }
            for name in columns
        ],
        markdown_options={"html": True},
        sort_action="native",
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": font,
            "fontSize": "14px",
            "textAlign": "left",
            "padding": "12px 16px",
            "whiteSpace": "nowrap",
            "height": "auto",
            "border": "none",
        },
        style_data={"borderBottom": "1px solid #eef2f6"},
        style_header={
            "fontFamily": font,
            "fontWeight": "600",
            "backgroundColor": "#f8fafc",
        },
        style_data_conditional=[
            {"if": {"column_id": "parameter"}, "fontFamily": monospace},
            *[
                {
                    "if": {"column_id": name},
                    "fontFamily": monospace,
                    "fontVariantNumeric": "tabular-nums",
                    "textAlign": "right",
                }
                for name in ("search_space", "observed_range", "step", "probabilities")
            ],
            *[
                {"if": {"column_id": name, "row_index": i}, "fontFamily": font, "textAlign": "left"}
                for i, m in enumerate(marginals)
                for name in ("search_space",)
                if m["lower"] is None
            ],
        ],
        style_cell_conditional=[
            {"if": {"column_id": name}, "minWidth": "150px"}
            for name in ("search_space", "observed_range")
        ],
    )


def _overview(data: CoverageData, marginals: List[Dict], sampler: str) -> Any:
    info = data.overview
    metrics = [
        _metric(
            "Trials",
            f"{info['trials']:,}",
            f"{info['complete']:,} complete · {info['failed']:,} failed",
        ),
        _metric("Parameters", info["parameters"]),
        _metric("Sampler", sampler),
    ]
    description = " · ".join(
        [f"{info['parameters']} parameters"]
        + [f"{count} {kind}" for kind, count in info["types"].items() if count]
    )
    return html.Div(
        [
            html.Div(metrics, className="coverage-metric-grid"),
            html.Details(
                [
                    html.Summary(
                        [
                            html.Span(
                                [
                                    "Search-space definition",
                                    html.Small(
                                        [
                                            html.Span(
                                                "Show definition",
                                                className="coverage-search-space-show",
                                            ),
                                            html.Span(
                                                "Hide definition",
                                                className="coverage-search-space-hide",
                                            ),
                                        ],
                                        className="coverage-search-space-action",
                                    ),
                                ],
                                className="coverage-search-space-title",
                            ),
                            html.Small(description, className="coverage-search-space-description"),
                        ]
                    ),
                    (
                        html.P("No conditional parameters", className="coverage-search-space-note")
                        if not info["conditional"]
                        else None
                    ),
                    _definition_table(marginals),
                    html.P(
                        "All sampled trials contribute, including failures "
                        "and repeated configurations. "
                        "Inactive parameters are excluded from marginals."
                    ),
                    html.P(
                        "Trial statuses: "
                        + ", ".join(f"{name}: {n}" for name, n in info["statuses"].items() if n)
                    ),
                ],
                className="coverage-details coverage-search-space",
            ),
        ]
    )


def _univariate_problematic(row: Dict) -> bool:
    return (
        row["coverage"] < 1
        or (row["ks"] is not None and row["ks"] > 0.1)
        or (row["cardinality"] is None and row["max_gap"] > 0.1)
    )


def _ecdf_figure(row: Dict) -> go.Figure:
    integer = row["type"] == "integer"
    observed_x, observed_y = row["ecdf_x"], row["ecdf_y"]
    expected_x = row["expected_cdf_x"] if integer else [0, 1]
    expected_y = row["expected_cdf_y"] if integer else [0, 1]
    if integer:
        # Duplicate the first support value to show its initial jump from zero.
        if observed_x:
            observed_x, observed_y = [observed_x[0], *observed_x], [0, *observed_y]
        expected_x, expected_y = [expected_x[0], *expected_x], [0, *expected_y]
    return go.Figure(
        [
            go.Scatter(
                x=observed_x,
                y=observed_y,
                name="Observed ECDF",
                mode="lines",
                line=dict(color="#7067ff", shape="hv"),
            ),
            go.Scatter(
                x=expected_x,
                y=expected_y,
                name="Expected CDF",
                mode="lines",
                line=dict(
                    color="#94a3b8",
                    width=1,
                    dash="dash",
                    shape="hv" if integer else "linear",
                ),
            ),
        ]
    )


def _distribution_figure(row: Dict) -> go.Figure:
    """Use the same quiet histogram and reference in previews and details."""
    return go.Figure(
        [
            go.Bar(
                x=row["labels"],
                y=row["frequencies"],
                name="Observed",
                marker=dict(color="#7067ff", line=dict(width=0)),
                **({"width": 0.72 / row["bins"]} if row["cardinality"] is None else {}),
            ),
            go.Scatter(
                x=row["labels"],
                y=row["expected"],
                name="Expected",
                mode="lines",
                line=dict(color="#94a3b8", width=1, dash="dash"),
            ),
        ]
    )


def _add_largest_gap(figure: go.Figure, row: Dict) -> None:
    """Anchor a gap bracket and trial rug to exact, unsampled CDF coordinates."""
    if row["cardinality"] is not None:
        return
    orange = "#f4511e"
    gap = row["largest_gap"]
    coordinates = row["trial_coordinates"]
    boundaries = (gap["lower"], gap["upper"]) if gap else ()
    observed_boundaries = (
        [gap[side] for side in ("lower", "upper") if gap[f"{side}_observed"]] if gap else []
    )
    colors = [orange if x in observed_boundaries else "#a39bc4" for x in coordinates]
    figure.update_layout(
        xaxis=dict(anchor="free", position=0, ticklabelstandoff=2),
        yaxis=dict(domain=[0.18, 1]),
        yaxis2=dict(
            anchor="x",
            domain=[0, 0.12],
            range=[0, 1],
            visible=False,
            fixedrange=True,
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=coordinates,
            y=[0.5] * len(coordinates),
            yaxis="y2",
            name="Observed trials",
            mode="markers",
            showlegend=False,
            marker=dict(
                symbol="line-ns",
                size=[12 if x in observed_boundaries else 8 for x in coordinates],
                color=colors,
                line=dict(
                    color=colors,
                    width=[2.5 if x in observed_boundaries else 1 for x in coordinates],
                ),
            ),
            cliponaxis=False,
            hovertemplate="Observed trial<br>Sampling coordinate: %{x:.6g}<extra></extra>",
        )
    )
    if gap is None:
        return
    lower, upper = boundaries
    lower_kind = "observed trial" if gap["lower_observed"] else "search-space boundary"
    upper_kind = "observed trial" if gap["upper_observed"] else "search-space boundary"
    tooltip = (
        "Largest uncovered interval"
        f"<br>Lower boundary: {gap['lower_value']:.6g} ({lower_kind})"
        f"<br>Upper boundary: {gap['upper_value']:.6g} ({upper_kind})"
        f"<br>Width in original units: {gap['upper_value'] - gap['lower_value']:.6g}"
        f"<br>Width: {100 * gap['width']:.2f}% of transformed search space"
    )
    figure.add_shape(
        type="rect",
        x0=lower,
        x1=upper,
        y0=0.18,
        y1=1,
        yref="paper",
        fillcolor="rgba(244,81,30,0.045)",
        line=dict(width=0),
        layer="below",
    )
    for side, boundary in zip(("lower", "upper"), boundaries):
        figure.add_shape(
            type="line",
            x0=boundary,
            x1=boundary,
            y0=0,
            y1=1.1,
            yref="paper",
            line=dict(
                color=orange,
                width=1,
                dash="dash",
            ),
        )
        if not gap[f"{side}_observed"]:
            figure.add_annotation(
                x=boundary,
                y=1.28,
                yref="paper",
                text="Search-space boundary",
                showarrow=False,
                xanchor="left" if side == "lower" else "right",
                yanchor="bottom",
                font=dict(color=orange, size=9),
                hovertext=tooltip,
            )
    figure.add_shape(
        type="line",
        x0=lower,
        x1=upper,
        y0=1.1,
        y1=1.1,
        yref="paper",
        line=dict(color=orange, width=1),
    )
    figure.add_shape(
        type="path",
        path=(f"M {lower},1.05 L {lower},1.15 " f"M {upper},1.05 L {upper},1.15"),
        yref="paper",
        line=dict(color=orange, width=1),
    )
    figure.add_annotation(
        x=(lower + upper) / 2,
        y=1.17,
        yref="paper",
        text="<b>Largest gap</b>",
        showarrow=False,
        yanchor="bottom",
        font=dict(color=orange, size=10),
        hovertext=tooltip,
    )
    # A transparent filled trace makes the interval itself a hover target.
    ymax = max([1e-12, *figure.data[0].y, *figure.data[1].y]) * 1.05
    figure.update_layout(yaxis=dict(range=[0, ymax]))
    figure.add_trace(
        go.Scatter(
            x=[lower, upper, upper, lower, lower],
            y=[0, 0, ymax, ymax, 0],
            name="Largest gap",
            mode="lines",
            line=dict(width=0),
            fill="toself",
            fillcolor="rgba(244,81,30,0.001)",
            hoveron="fills",
            text=tooltip,
            hoverinfo="text",
            showlegend=False,
            cliponaxis=False,
        )
    )


def _univariate_preview(row: Dict, mode: str) -> Any:
    continuous = row["cardinality"] is None
    ecdf = mode == "ecdf" and (continuous or row["type"] == "integer")
    if ecdf:
        figure = _ecdf_figure(row)
    else:
        figure = _distribution_figure(row)
    figure.update_layout(
        template="plotly_white",
        height=180,
        margin=dict(l=8, r=10, t=42, b=26),
        showlegend=False,
        bargap=0.28,
        font=dict(size=10, color="#64748b"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            fixedrange=True,
            showgrid=False,
            nticks=6,
            title=None,
            ticklabelstandoff=12,
        ),
        yaxis=dict(
            fixedrange=True,
            rangemode="tozero",
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            showline=False,
        ),
    )
    if continuous:
        # Uniform CDF coordinates can be labelled in the original parameter units.
        ticks = np.linspace(0, 1, 5)
        if not ecdf and row["distribution"] in ("UniformFloat", "LogUniformFloat"):
            values = (
                np.exp(np.log(row["lower"]) + ticks * np.log(row["upper"] / row["lower"]))
                if row["distribution"] == "LogUniformFloat"
                else row["lower"] + ticks * (row["upper"] - row["lower"])
            )
            figure.update_xaxes(tickvals=ticks, ticktext=[f"{v:.3g}" for v in values])
        figure.update_xaxes(range=[0, 1])
    elif ecdf:
        figure.update_xaxes(type="linear")
    else:
        figure.update_xaxes(type="category")
    if ecdf:
        figure.update_yaxes(range=[0, 1.05])
    _add_largest_gap(figure, row)
    return dcc.Graph(
        figure=figure,
        responsive=True,
        style={"height": "180px"},
        config={"displayModeBar": False, "scrollZoom": False},
        className="coverage-univariate-preview",
    )


def _univariate_rows(
    rows: List[Dict], mode: str, group_by: str, sort_by: str, only_problematic: bool
) -> Any:
    rows = [row for row in rows if not only_problematic or _univariate_problematic(row)]
    if sort_by == "name":
        rows = sorted(rows, key=lambda row: row["name"])
    else:
        rows = sorted(
            rows,
            key=lambda row: (
                (
                    row["coverage"]
                    if sort_by == "coverage"
                    else -(
                        0
                        if sort_by == "max_gap" and row["type"] == "integer"
                        else (row.get(sort_by) or 0)
                    )
                ),
                row["name"],
            ),
        )
    if not rows:
        return html.P("No parameters match the selected filter.", className="coverage-empty")
    groups = (
        [
            (kind.title(), [row for row in rows if row["type"] == kind])
            for kind in ("continuous", "integer", "categorical", "ordinal", "constant")
        ]
        if group_by == "type"
        else [("All parameters", rows)]
    )
    sections = []
    for title, members in groups:
        if not members:
            continue
        items = []
        for row in members:
            discrete = row["cardinality"] is not None
            distribution = row["distribution"].removesuffix("Float").removesuffix("Integer")
            definition = (
                _definition_range(row["lower"], row["upper"])
                if row["lower"] is not None
                else ", ".join(map(str, row["choices"]))
            )
            metrics = [
                _metric(
                    "Values" if discrete else "Bins",
                    (
                        f"{row['unique']} / {row['cardinality']}"
                        if discrete
                        else f"{row['occupied_bins']} / {row['bins']}"
                    ),
                ),
                *(
                    [_metric("Largest gap", _number(row["max_gap"], True))]
                    if row["type"] != "integer"
                    else []
                ),
                _metric("KS", _number(row["ks"]) if row["ks"] is not None else "—"),
            ]
            items.append(
                html.Details(
                    [
                        html.Summary(
                            [
                                html.Div(
                                    [
                                        html.Strong(row["name"]),
                                        html.Small(row["type"].title()),
                                        html.Small(distribution),
                                        html.Small(definition),
                                    ],
                                    className="coverage-univariate-name",
                                ),
                                _univariate_preview(row, mode),
                                html.Div(metrics, className="coverage-univariate-metrics"),
                                html.Span(
                                    "›",
                                    className="coverage-univariate-chevron",
                                    **{"aria-hidden": "true"},
                                ),
                            ],
                            title=f"Show details for {row['name']}",
                        ),
                        html.Div(_univariate(row, mode), className="coverage-univariate-detail"),
                    ],
                    className="coverage-univariate-row",
                )
            )
        sections.append(
            html.Section(
                [
                    html.H4(f"{title} ({len(members)})"),
                    html.Div(items, className="coverage-univariate-list"),
                ],
                className="coverage-univariate-group",
            )
        )
    return html.Div(sections)


def _univariate(row: Dict, mode: str = "distribution") -> Any:
    discrete = row["cardinality"] is not None
    metrics = [
        _metric(
            "Coverage",
            _number(row["coverage"], True),
            "Visited values" if discrete else "Occupied bins",
        ),
        _metric("Unique values", f"{row['unique']:,} / {row['active_trials']:,}"),
    ]
    if row["type"] in ("continuous", "integer", "ordinal"):
        metrics.extend(
            [
                *(
                    [_metric("Largest gap", _number(row["max_gap"], True), "In sampling space")]
                    if row["type"] != "integer"
                    else []
                ),
                (
                    _metric("Values visited", f"{row['unique']} / {row['cardinality']}")
                    if discrete
                    else _metric("Bins occupied", f"{row['occupied_bins']} / {row['bins']}")
                ),
                *(
                    [
                        _metric(
                            "Edge gaps",
                            f"{_number(row['edge_left'], True)} / {_number(row['edge_right'], True)}",
                            "Left / right",
                        )
                    ]
                    if row["type"] != "integer"
                    else []
                ),
                (
                    _metric("KS statistic", _number(row["ks"]))
                    if not discrete
                    else _metric("Largest gap (adjacent values)", row["largest_missing_run"])
                ),
            ]
        )
    if discrete:
        metrics.append(_metric("Unseen values", row["unseen_count"]))
    if row["type"] == "integer":
        metrics.append(_metric("KS distance", _number(row["ks"]), "Descriptive; no p-value"))
    if mode == "ecdf" and (not discrete or row["type"] == "integer"):
        figure = _ecdf_figure(row)
        plot = _graph(
            figure,
            "Value" if row["type"] == "integer" else "Sampling coordinate",
            "Cumulative probability",
        )
        figure.update_xaxes(type="linear")
        if not discrete:
            figure.update_xaxes(range=[0, 1])
        figure.update_yaxes(range=[0, 1.05])
    else:
        figure = _distribution_figure(row)
        plot = _graph(figure, "Value" if discrete else "Sampling coordinate", "Trials")
    figure.update_layout(bargap=0.28, legend=dict(y=1.35), margin=dict(t=70))
    figure.update_yaxes(showgrid=False, zeroline=False, nticks=3)
    if not discrete:
        figure.update_xaxes(range=[0, 1])
    _add_largest_gap(figure, row)
    details = [
        html.P(
            f"{row['active_trials']:,} active · {row['inactive_trials']:,} inactive trials. "
            f"Transformation: {row['transformation']}. "
            f"Largest raw gap: {_number(row['raw_max_gap'])}."
        ),
        html.P(
            "Expected frequencies describe the defined distribution; "
            "constraints can alter marginals. "
            "KS is descriptive for adaptive or repeated trials."
        ),
    ]
    if row["type"] == "integer":
        details.append(
            html.P(
                "Integer ECDF and expected CDF use the valid discrete support. "
                "KS distance is the maximum absolute CDF difference on that support; "
                "no continuous-distribution KS p-value is computed."
            )
        )
        if not row["ecdf_complete"]:
            details.append(
                html.P(
                    "Large integer support: CDF plots show selected valid support values. "
                    "The KS distance is evaluated exactly."
                )
            )
    if discrete:
        details.append(
            _table(
                [
                    {"value": label, "observed": n, "expected": exp, "relative_deviation": dev}
                    for label, n, exp, dev in zip(
                        row["labels"],
                        row["frequencies"],
                        row["expected"],
                        row["relative_deviation"],
                    )
                ]
            )
        )
        if row["unseen"]:
            details.append(html.P("Unseen: " + ", ".join(row["unseen"][:50])))
        if not row["frequency_table_complete"]:
            details.append(html.P("Large discrete range: frequencies list visited values only."))
    return html.Div(
        [
            html.Div(
                [
                    html.Aside(metrics, className="coverage-metric-stack"),
                    html.Div(plot, className="coverage-plot"),
                ],
                className="coverage-diagnostic",
            ),
            html.Details(
                [html.Summary("Parameter details"), *details], className="coverage-details"
            ),
        ]
    )


def _pair_figure(figure: go.Figure, x: str = "", y: str = "", matrix: bool = False) -> go.Figure:
    """Fix the inner plot's vertical domain independently of labels and colorbars."""
    _figure(figure, x, y)
    figure.update_layout(
        height=PAIR_PLOT_HEIGHT,
        margin=dict(l=135 if matrix else 55, r=15, t=10, b=80, autoexpand=False),
        showlegend=False,
        xaxis=dict(domain=[0, 1], automargin=False),
        yaxis=dict(domain=[0, 1], automargin=False),
    )
    return figure


def _pair_panel(title: str, summary: Any, plot: Any) -> Any:
    return html.Section(
        [
            html.H4(title, title=title, className="coverage-pair-panel-header"),
            plot,
            html.Footer(summary, className="coverage-pair-panel-footer"),
        ],
        className="coverage-pair-panel",
    )


def _pair_graph(figure: go.Figure) -> Any:
    return dcc.Graph(
        figure=figure,
        responsive=True,
        style={"height": f"{PAIR_PLOT_HEIGHT}px"},
        className="coverage-pair-plot-row",
        config={"displaylogo": False},
    )


def _pair_matrix(result: Optional[Dict], metric: str, names: List[str]) -> go.Figure:
    if result is None:
        return _pair_figure(
            _empty_figure(
                "Run pairwise analysis to populate the matrix. "
                "A selected scatter is available immediately."
            ),
            matrix=True,
        )
    values = np.full((len(names), len(names)), np.nan)
    lookup = {name: i for i, name in enumerate(names)}
    for row in result["pairs"]:
        if row["x"] in lookup and row["y"] in lookup and row.get(metric) is not None:
            i, j = lookup[row["x"]], lookup[row["y"]]
            values[i, j] = values[j, i] = row[metric]
    percent = metric in ("occupancy", "empty_cell_rate")
    severity = _pair_severity(values, metric)
    figure = go.Figure(
        go.Heatmap(
            x=names,
            y=names,
            z=severity,
            customdata=values,
            colorscale=PAIR_CONCERN_COLORS,
            hoverongaps=False,
            zmin=0,
            zmax=1,
            showscale=False,
            hoverinfo="none",
        )
    )
    if len(names) <= 12:
        for i, j in np.argwhere(np.isfinite(values)):
            figure.add_annotation(
                x=names[j],
                y=names[i],
                text=f"{values[i, j]:.0%}" if percent else f"{values[i, j]:.3g}",
                showarrow=False,
                font=dict(size=12, color=_pair_text_color(float(severity[i, j]))),
            )
    _pair_figure(figure, matrix=True)
    figure.update_layout(
        yaxis=dict(autorange="reversed", showgrid=False, zeroline=False),
        xaxis=dict(tickangle=-35, showgrid=False, zeroline=False),
    )
    return figure


def _scatter_axis(data: CoverageData, column: int, indices: np.ndarray, count: int) -> tuple:
    """Locate coverage cell edges and observations in the same plotting coordinates."""
    axis = data.axes[column]
    if axis.size is None:
        values = data.points[indices, column].tolist()
        edges = np.linspace(0, 1, count + 1)
        options = dict(type="linear", range=[0, 1])
    else:
        cuts = np.array([(cell * axis.size + count - 1) // count for cell in range(count + 1)])
        if axis.kind in ("categorical", "ordinal", "constant"):
            values = [str(data.values[k][axis.name]) for k in indices]
            edges = cuts - 0.5
            options = dict(
                type="category",
                range=[-0.5, axis.size - 0.5],
                categoryorder="array",
                categoryarray=list(map(str, axis.choices)),
            )
        else:
            values = [data.values[k][axis.name] for k in indices]
            edges = axis.hp.lower + cuts * axis.step - axis.step / 2
            ticks = np.unique(np.linspace(0, axis.size - 1, min(axis.size, 6), dtype=int))
            options = dict(
                type="linear",
                range=[float(edges[0]), float(edges[-1])],
                tickmode="array",
                tickvals=axis.values_at(ticks),
            )
    return values, edges, options


def _scatter_cell_shapes(
    grid: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, highlight_empty: bool
) -> List[Dict]:
    shapes = [
        dict(
            type="rect",
            xref="x",
            yref="y",
            layer="below",
            line=dict(width=0),
            fillcolor="#f2f1fa",
            x0=float(x_edges[i]),
            x1=float(x_edges[i + 1]),
            y0=float(y_edges[j]),
            y1=float(y_edges[j + 1]),
        )
        for i, j in np.argwhere(grid == 0)
        if highlight_empty
    ]
    for value in x_edges:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                layer="below",
                x0=float(value),
                x1=float(value),
                y0=float(y_edges[0]),
                y1=float(y_edges[-1]),
                line=dict(color="#e2e6ef", width=0.5),
            )
        )
    for value in y_edges:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                layer="below",
                x0=float(x_edges[0]),
                x1=float(x_edges[-1]),
                y0=float(value),
                y1=float(value),
                line=dict(color="#e2e6ef", width=0.5),
            )
        )
    return shapes


def _scatter(
    data: CoverageData,
    x: str,
    y: str,
    result: Optional[Dict] = None,
    highlight_empty: bool = True,
    elite_indices: Optional[List[int]] = None,
    elite_label: Optional[str] = None,
) -> Any:
    names = [axis.name for axis in data.axes]
    if x not in names or y not in names:
        return _pair_panel(
            "Parameter pair",
            "Select two parameters to inspect their sampled projection.",
            _pair_graph(_pair_figure(_empty_figure("Select two parameters."))),
        )
    i, j = names.index(x), names.index(y)
    grid, active = _pair_grid(data, [i, j], (result or {}).get("bins", DEFAULT_BINS))
    indices = np.flatnonzero(active)
    x_values, x_edges, x_options = _scatter_axis(data, i, indices, grid.shape[0])
    y_values, y_edges, y_options = _scatter_axis(data, j, indices, grid.shape[1])
    figure = go.Figure(
        go.Scattergl(
            x=x_values,
            y=y_values,
            mode="markers",
            marker=dict(size=6, opacity=0.85, color="#3730a3"),
            customdata=[
                [str(data.values[k][x]), str(data.values[k][y]), int(k) + 1] for k in indices
            ],
            hoverinfo="none",
        )
    )
    if elite_indices:
        elite_set = set(elite_indices)
        selected = [position for position, index in enumerate(indices) if index in elite_set]
        figure.data[0].marker.opacity = 0.35
        figure.add_trace(
            go.Scattergl(
                x=[x_values[position] for position in selected],
                y=[y_values[position] for position in selected],
                mode="markers",
                name="Elite trials",
                marker=dict(
                    size=10,
                    symbol="diamond",
                    color="#f59e0b",
                    line=dict(color="#92400e", width=1),
                ),
                hoverinfo="none",
            )
        )
    _pair_figure(figure, x, y)
    figure.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
    figure.update_layout(shapes=_scatter_cell_shapes(grid, x_edges, y_edges, highlight_empty))
    figure.update_xaxes(**x_options)
    figure.update_yaxes(**y_options)
    for update_axis in (figure.update_xaxes, figure.update_yaxes):
        update_axis(
            showgrid=False,
            gridcolor="#eef2f6",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor="#cbd5e1",
            linewidth=1,
            tickfont=dict(color="#475569"),
            title_font=dict(color="#334155"),
        )
    pair = next((p for p in (result or {}).get("pairs", []) if {p["x"], p["y"]} == {x, y}), None)
    metrics = (
        html.Aside(
            [
                _metric(
                    PAIR_METRICS[key],
                    (
                        _number(pair.get(key), key in ("occupancy", "empty_cell_rate"))
                        if pair.get(key) is not None
                        else "No joint observations" if not len(indices) else "No reference samples"
                    ),
                )
                for key in PAIR_METRICS
            ],
            className="coverage-pair-metrics",
        )
        if pair
        else None
    )
    return _pair_panel(
        f"{x} × {y}" + (f" · {elite_label}" if elite_label else ""),
        None,
        html.Div(
            [_pair_graph(figure), metrics],
            className="coverage-pair-plot-row coverage-pair-scatter"
            + (" coverage-pair-scatter--with-metrics" if pair else ""),
        ),
    )


def _empty_figure(message: str) -> go.Figure:
    figure = _figure(go.Figure(), "", "")
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def _module_summary(analysis: str, result: Optional[Dict]) -> Any:
    if result is None:
        return html.Small("Results appear after an explicit start.")
    if analysis == "discrepancy":
        return html.Div(
            [
                _metric("Centered", _number(result["centered"])),
                _metric("L2-star", _number(result.get("l2_star"))),
            ]
        )
    if analysis == "radius":
        return html.Div(
            [_metric("p95", _number(result["p95"])), _metric("Max", _number(result["max"]))]
        )
    if analysis == "baseline":
        metric = result["metrics"].get("centered_discrepancy", {})
        label = "Discrepancy percentile"
        if metric.get("percentile") is None:
            metric = result["metrics"].get("p95_radius", {})
            label = "p95 radius percentile"
        return _metric(
            label,
            (
                f"{metric['percentile']:.0f}%"
                if metric.get("percentile") is not None
                else "Unavailable"
            ),
        )
    return None


def _discrepancy_details(result: Optional[Dict]) -> Any:
    if not result:
        return html.P("Run the discrepancy module to see its details.")
    return html.Div(
        [
            html.H3("Global Discrepancy"),
            html.Div(
                [
                    _metric(label, _number(result.get(key)))
                    for key, label in (
                        ("centered", "Centered"),
                        ("l2_star", "L2-star"),
                        ("wrap_around", "Wrap-around"),
                    )
                ],
                className="coverage-metric-grid",
            ),
            html.P(
                result.get("reason") or "Continuous projection: " + ", ".join(result["parameters"])
            ),
        ]
    )


def _radius_details(result: Optional[Dict]) -> Any:
    if not result:
        return html.P("Run coverage radius to see its distance distribution and uncovered regions.")
    return html.Div(
        [
            html.H3("Coverage Radius"),
            html.Div(
                [
                    _metric(
                        key.upper() if key.startswith("p") else key.title(), _number(result[key])
                    )
                    for key in ("median", "p90", "p95", "p99", "max")
                ],
                className="coverage-metric-grid coverage-radius-metrics",
            ),
            _graph(
                go.Figure(go.Bar(x=result["histogram_x"], y=result["histogram_y"])),
                "Distance to nearest trial",
                "Reference samples",
            ),
            html.Details(
                [
                    html.Summary("Distance definition"),
                    html.P(result["metric"]),
                    html.P(
                        f"{result['reference_samples']:,} reference samples. "
                        "Regions are sampled candidates."
                    ),
                ]
            ),
        ]
    )


def _region_detail(region: Dict, index: int, names: List[str]) -> Any:
    return html.Div(
        [
            html.H4(f"Region {index + 1}"),
            _metric("Distance", _number(region["distance"])),
            html.Dl(
                [
                    html.Div(
                        [
                            html.Dt(name),
                            html.Dd(_number(region[name]) if name in region else "Inactive"),
                        ]
                    )
                    for name in names
                ],
                className="coverage-definition-list",
            ),
        ]
    )


def _baseline_details(result: Optional[Dict]) -> Any:
    if not result:
        return html.P("Run the random baseline to compare against equivalent random designs.")
    cards = []
    for key, metric in result["metrics"].items():
        title = GROWTH_METRICS.get(key, key.replace("_", " ").title())
        if metric.get("mean") is None:
            cards.append(_card(title, [html.P("Unavailable for this search space.")]))
            continue
        observed, lower, upper = metric["observed"], metric["lower"], metric["upper"]
        figure = go.Figure(
            [
                go.Scatter(
                    x=[lower, upper],
                    y=[0, 0],
                    mode="lines",
                    line=dict(width=14, color="#cbd5e1"),
                    name="Random 5–95%",
                ),
                go.Scatter(
                    x=[metric["mean"]],
                    y=[0],
                    mode="markers",
                    marker=dict(symbol="line-ns", size=20, color="#475569"),
                    name="Random mean",
                ),
                go.Scatter(
                    x=[observed],
                    y=[0],
                    mode="markers",
                    marker=dict(size=12, color="#315cd6"),
                    name="Observed",
                ),
            ]
        )
        _figure(figure, "", "")
        figure.update_layout(height=105, margin=dict(l=25, r=25, t=5, b=35), showlegend=False)
        figure.update_yaxes(visible=False, range=[-1, 1])
        if key == "pairwise_occupancy":
            figure.update_xaxes(tickformat=".0%", autorange="reversed")
        cards.append(
            _card(
                title,
                [
                    html.Div(
                        [html.Span("Better coverage"), html.Span("Worse coverage")],
                        className="coverage-range-labels",
                    ),
                    dcc.Graph(
                        figure=figure,
                        responsive=True,
                        style={"height": "105px"},
                        config={"displayModeBar": False},
                    ),
                    html.P(
                        f"Random 5–95%: {_number(lower)} – {_number(upper)} · "
                        f"Observed: {_number(observed)}"
                    ),
                    html.Strong(f"{metric['percentile']:.0f}th percentile"),
                    html.Small(
                        " · Standardized deviation: "
                        + _number(metric.get("standardized_deviation"))
                    ),
                ],
            )
        )
    return html.Div(
        [
            html.H3("Random Baseline"),
            html.P(
                f"{result['designs']} random designs · seed {result['seed']}. "
                "Percentile measures random values below the observed metric; ties count half."
            ),
            html.Div(cards, className="coverage-baseline-grid"),
            html.Details(
                [
                    html.Summary("Detailed values"),
                    _table([{"metric": key, **value} for key, value in result["metrics"].items()]),
                    html.P(
                        "Discrepancy projection: "
                        + (", ".join(result["discrepancy_parameters"]) or "unavailable")
                    ),
                ]
            ),
        ]
    )


def _compatible_growth(realised: Optional[Dict], random: Optional[Dict]) -> bool:
    if not random or random.get("result") is None:
        return False
    if not realised or realised.get("result") is None:
        return True  # The random job includes its own realised trajectory.
    return (
        realised.get("run_id") == random.get("run_id")
        and realised.get("fingerprint") == random.get("fingerprint")
        and realised.get("version") == random.get("version")
        and all(
            realised.get("options", {}).get(key) == random.get("options", {}).get(key)
            for key in ("reference_samples", "seed")
        )
    )


def _growth_figure(
    realised: Optional[Dict], random: Optional[Dict], metric: str, overlay: bool
) -> go.Figure:
    result = (realised or {}).get("result") or (random or {}).get("result")
    if result is None:
        return _empty_figure("Run coverage growth to see the realised trajectory.")
    rows = result["checkpoints"]
    if not any(row.get(metric) is not None for row in rows):
        return _empty_figure(
            "This metric is unavailable in the stored result. "
            "Recompute if it predates the extended growth analysis."
        )
    figure = go.Figure()
    if overlay and random is not None and _compatible_growth(realised, random):
        bands = random["result"]["bands"]
        if any(b.get(metric, {}).get("mean") is not None for b in bands):
            x = [b["trials"] for b in bands]
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=[b.get(metric, {}).get("upper") for b in bands],
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=[b.get(metric, {}).get("lower") for b in bands],
                    fill="tonexty",
                    line=dict(width=0),
                    name="Random 5–95%",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=[b.get(metric, {}).get("mean") for b in bands],
                    name="Random mean",
                    line=dict(dash="dash"),
                )
            )
    figure.add_trace(
        go.Scatter(
            x=[r["trials"] for r in rows],
            y=[r.get(metric) for r in rows],
            name="Realised",
            mode="lines+markers",
        )
    )
    _figure(figure, "Sampled trials", GROWTH_METRICS[metric])
    if metric in ("pairwise_occupancy", "marginal_max_gap"):
        figure.update_yaxes(tickformat=".0%")
    return figure
