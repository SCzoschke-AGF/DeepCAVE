"""Focused, read-only diagnostic views for the search-space coverage suite."""

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Scheme

from deepcave.evaluators.coverage import CoverageData

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
    "discrepancy": "Discrepancy",
    "hole_score": "Hole score",
}


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
        className="coverage-metric",
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
        *[
            _metric(kind.title(), count)
            for kind, count in info["types"].items()
            if kind in ("continuous", "integer") or count
        ],
        _metric("Conditional", info["conditional"]),
    ]
    rows = [
        {
            "parameter": m["name"],
            "type": m["type"],
            "conditional": m["conditional"],
            "search_space": (
                f"{m['lower']} – {m['upper']}"
                if m["lower"] is not None
                else ", ".join(map(str, m["choices"]))
            ),
            "distribution": m["distribution"],
            "probabilities": str(m["probabilities"] or ""),
            "step": m["step"],
            "observed_min": m["observed_min"],
            "observed_max": m["observed_max"],
            "range_fraction": m["range_fraction"],
            "unique_values": m["unique"],
            "active_trials": m["active_trials"],
            "transformation": m["transformation"],
        }
        for m in marginals
    ]
    return html.Div(
        [
            html.Div(metrics, className="coverage-metric-grid"),
            html.Details(
                [
                    html.Summary("Search-space definition"),
                    _table(rows),
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
                className="coverage-details",
            ),
        ]
    )


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
                _metric("Largest gap", _number(row["max_gap"], True), "In sampling space"),
                _metric("Bins occupied", f"{row['occupied_bins']} / {row['bins']}"),
                _metric(
                    "Edge gaps",
                    f"{_number(row['edge_left'], True)} / {_number(row['edge_right'], True)}",
                    "Left / right",
                ),
                (
                    _metric("KS statistic", _number(row["ks"]))
                    if not discrete
                    else _metric("Largest missing run", row["largest_missing_run"])
                ),
            ]
        )
    if discrete:
        metrics.append(_metric("Unseen values", row["unseen_count"]))
    if mode == "ecdf" and not discrete:
        figure = go.Figure(
            [
                go.Scatter(x=row["ecdf_x"], y=row["ecdf_y"], name="Observed ECDF", line_shape="hv"),
                go.Scatter(x=[0, 1], y=[0, 1], name="Uniform expectation", line=dict(dash="dash")),
            ]
        )
        plot = _graph(figure, "Sampling coordinate", "Cumulative frequency")
    else:
        figure = go.Figure(
            [
                go.Bar(x=row["labels"], y=row["frequencies"], name="Observed"),
                go.Scatter(
                    x=row["labels"], y=row["expected"], name="Expected", mode="lines+markers"
                ),
            ]
        )
        plot = _graph(figure, "Value" if discrete else "Sampling coordinate", "Trials")
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


def _pair_matrix(result: Optional[Dict], metric: str, names: List[str]) -> go.Figure:
    if result is None:
        return _empty_figure(
            "Run pairwise analysis to populate the matrix. "
            "A selected scatter is available immediately."
        )
    values = np.full((len(names), len(names)), np.nan)
    lookup = {name: i for i, name in enumerate(names)}
    for row in result["pairs"]:
        if row["x"] in lookup and row["y"] in lookup and row.get(metric) is not None:
            i, j = lookup[row["x"]], lookup[row["y"]]
            values[i, j] = values[j, i] = row[metric]
    percent = metric in ("occupancy", "empty_cell_rate")
    figure = go.Figure(
        go.Heatmap(
            x=names,
            y=names,
            z=values,
            colorscale="Blues",
            reversescale=metric != "occupancy",
            hoverongaps=False,
            zmin=0,
            zmax=1 if percent else None,
            colorbar=dict(title=PAIR_METRICS[metric], tickformat=".0%" if percent else ".3g"),
            texttemplate=("%{z:.0%}" if percent else "%{z:.2g}") if len(names) <= 12 else None,
            hovertemplate="%{y} × %{x}<br>"
            + PAIR_METRICS[metric]
            + (": %{z:.1%}" if percent else ": %{z:.4g}")
            + "<extra></extra>",
        )
    )
    _figure(figure, "", "")
    figure.update_layout(
        height=360,
        margin=dict(l=110, r=35, t=20, b=100),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(tickangle=-35),
    )
    return figure


def _scatter(data: CoverageData, x: str, y: str, result: Optional[Dict] = None) -> Any:
    names = [axis.name for axis in data.axes]
    if x not in names or y not in names:
        return html.P("Select two parameters to inspect their sampled projection.")
    i, j = names.index(x), names.index(y)
    active = np.isfinite(data.points[:, [i, j]]).all(axis=1)
    indices = np.flatnonzero(active)
    values = [
        (
            [str(data.values[k][name]) for k in indices]
            if data.axes[column].kind == "categorical"
            else data.points[active, column]
        )
        for name, column in ((x, i), (y, j))
    ]
    figure = go.Figure(
        go.Scattergl(
            x=values[0],
            y=values[1],
            mode="markers",
            marker=dict(size=5, opacity=0.6),
            customdata=[
                [str(data.values[k][x]), str(data.values[k][y]), int(k) + 1] for k in indices
            ],
            hovertemplate=f"{x}: %{{customdata[0]}}<br>{y}: %{{customdata[1]}}"
            "<br>Trial: %{customdata[2]}<extra></extra>",
        )
    )
    pair = next((p for p in (result or {}).get("pairs", []) if {p["x"], p["y"]} == {x, y}), None)
    metrics = (
        html.Div(
            [
                _metric(
                    PAIR_METRICS[key], _number(pair[key], key in ("occupancy", "empty_cell_rate"))
                )
                for key in PAIR_METRICS
            ],
            className="coverage-metric-grid coverage-pair-metrics",
        )
        if pair
        else None
    )
    return html.Div(
        [
            html.H4(f"{x} × {y}"),
            metrics,
            html.Small(f"{len(indices):,} co-active trials"),
            _graph(figure, x, y),
        ]
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
