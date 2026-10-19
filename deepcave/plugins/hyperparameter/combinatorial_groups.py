"""Compact group diagnostics using saved CCM counts and discretization only."""

from typing import Any, Dict, List, Optional

from collections import Counter
from itertools import combinations, islice, product
from math import ceil, prod

import plotly.graph_objects as go
from dash import dash_table, dcc, html

from deepcave.plugins.hyperparameter.coverage_views import _id


def diagnostic_row(result: Optional[Dict], resolution: Any) -> Optional[Dict]:
    """Find a saved resolution without preparing or measuring trial data."""
    return next(
        (row for row in (result or {}).get("rows", []) if row["resolution"] == resolution), None
    )


def group_records(result: Dict, row: Dict, parameters: Optional[List[str]] = None) -> List[Dict]:
    """Reconstruct every group's diagnostics from saved counts, applying an AND filter."""
    cell = next(
        (cell for cell in row["cells"] if cell["strength"] == row["observed_strength"] + 1), {}
    )
    counts = cell.get("group_covered")
    if counts is None:
        return []
    names = result["parameters"]
    selected = set(parameters or [])
    records = []
    for index, (group, covered) in enumerate(zip(combinations(names, cell["strength"]), counts)):
        if not selected.issubset(group):
            continue
        total = prod(row["levels"][name] for name in group)
        records.append(
            {
                "index": index,
                "parameters": list(group),
                "total": total,
                "covered": covered,
                "missing": total - covered,
                "coverage": covered / total,
            }
        )
    # Names break ties independently of filter, display limit and process history.
    return sorted(records, key=lambda group: (-group["missing"], tuple(group["parameters"])))


def _tags(parameters: List[str]) -> Any:
    return html.Span(
        [html.Span(name, className="coverage-group-tag") for name in parameters],
        className="coverage-group-tags",
    )


def _kpis(groups: List[Dict], strength: int) -> Any:
    worst = groups[0] if groups else None
    values = [
        (f"{strength}-parameter combinations analyzed", f"{len(groups):,}"),
        ("Fully covered", f"{sum(group['missing'] == 0 for group in groups):,}"),
        ("Most missing", f"{worst['missing']:,} / {worst['total']:,}" if worst else "—"),
    ]
    return html.Div(
        [html.Div([html.Small(label), html.Strong(value)]) for label, value in values],
        className="coverage-group-kpis",
        title="Counts reflect the selected parameter filter; all strengths use saved measurements.",
    )


def _distribution(groups: List[Dict], strength: int) -> Any:
    counts = Counter(group["missing"] for group in groups)
    values = sorted(counts)
    figure = go.Figure(
        go.Bar(
            x=values,
            y=[counts[value] for value in values],
            width=0.8,
            marker_color="#8661bd",
            hovertemplate=f"%{{x}} missing bin combinations<br>%{{y}} {strength}-parameter combinations<extra></extra>",
        )
    )
    figure.update_layout(
        height=155,
        margin=dict(l=48, r=8, t=4, b=42),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10, color="#64748b"),
        bargap=0.25,
        xaxis=dict(
            title=f"Missing bin combinations per {strength}-parameter combination",
            tickmode="linear",
            dtick=1,
            range=[-0.5, max(values, default=0) + 0.5],
            fixedrange=True,
        ),
        yaxis=dict(
            title=f"{strength}-parameter combinations",
            rangemode="tozero",
            tickformat=",d",
            dtick=max(1, ceil(max(counts.values(), default=0) / 4)),
            fixedrange=True,
        ),
    )
    # Keep integer bins but let Plotly choose readable ticks for wide supports.
    if values and values[-1] > 20:
        figure.update_xaxes(tickmode="auto", dtick=None)
    return dcc.Graph(
        figure=figure,
        config={"displayModeBar": False},
        responsive=True,
        style={"width": "100%", "height": "155px", "minWidth": 0},
        className="coverage-group-distribution",
    )


def group_overview(result: Dict, row: Dict, parameters: Any = None, limit: Any = 5) -> Any:
    """Render filtered KPIs, all-group distribution and a limited, sorted table."""
    groups = group_records(result, row, parameters)
    strength = row["observed_strength"] + 1
    shown = groups if limit == "all" else groups[: int(limit or 5)]
    rows = []
    for group in shown:
        rows.append(
            html.Tr(
                [
                    html.Td(
                        html.Button(
                            _tags(group["parameters"]),
                            type="button",
                            className="coverage-group-open",
                            title=f"Show the missing bin combinations for this {strength}-parameter combination",
                        )
                    ),
                    html.Td(
                        [
                            html.Strong(f"{group['missing']:,} / {group['total']:,}"),
                            f" — {group['coverage']:.1%} covered",
                        ]
                    ),
                ],
                id={
                    "type": _id("group-open"),
                    "resolution": row["resolution"],
                    "group": group["index"],
                },
                n_clicks=0,
                className="coverage-group-row",
            )
        )
    table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th(f"{strength}-parameter combinations"),
                        html.Th("Missing bin combinations"),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        className="coverage-group-table",
    )
    return html.Div(
        [
            _kpis(groups, strength),
            _distribution(groups, strength),
            html.Small(
                (
                    f"Observed {strength}-parameter combinations · filtered"
                    if parameters
                    else f"Observed {strength}-parameter combinations · all"
                ),
                className="coverage-group-note",
            ),
            (
                table
                if groups
                else html.P(
                    f"No {strength}-parameter combinations contain all selected parameters."
                )
            ),
            html.Small(
                f"Showing {len(shown):,} of {len(groups):,} {strength}-parameter combinations · most missing first",
                className="coverage-group-note",
            ),
        ]
    )


def group_diagnostics(result: Dict, row: Dict, cell: Dict) -> Any:
    """Build the interactive shell; missing tuples are loaded only on a group click."""
    if "group_covered" not in cell or "discretization" not in row:
        return html.Div(
            [
                html.H5(
                    f"{cell['strength']}-parameter combinations preventing strength {cell['strength']}"
                ),
                html.P(
                    f"Refresh coverage to load all {cell['strength']}-parameter combination counts and saved level definitions. "
                    "The existing coverage results remain available."
                ),
            ],
            className="coverage-group-diagnostics",
        )
    return html.Div(
        [
            html.H5(
                f"{cell['strength']}-parameter combinations preventing strength {cell['strength']}"
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Filter parameters (AND)", htmlFor=_id("group-filter")),
                            dcc.Dropdown(
                                id=_id("group-filter"),
                                options=[
                                    {"label": name, "value": name} for name in result["parameters"]
                                ],
                                multi=True,
                                value=[],
                                placeholder="All parameters",
                            ),
                        ],
                        className="coverage-group-filter",
                    ),
                    html.Div(
                        [
                            html.Label("Show", htmlFor=_id("group-limit")),
                            dcc.Dropdown(
                                id=_id("group-limit"),
                                options=[
                                    {"label": "Show 5", "value": 5},
                                    {"label": "Show 10", "value": 10},
                                    {"label": "Show 20", "value": 20},
                                    {"label": "Show all", "value": "all"},
                                ],
                                value=5,
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                        className="coverage-group-limit",
                    ),
                ],
                className="coverage-group-controls",
            ),
            html.Div(group_overview(result, row), id=_id("group-overview")),
            dcc.Store(id=_id("group-selection")),
            html.Div(id=_id("group-drilldown"), **{"aria-live": "polite"}),
        ],
        className="coverage-group-diagnostics",
    )


def missing_tuple_page(result: Dict, row: Dict, group: Dict, page: int = 0) -> tuple:
    """Enumerate a page of genuinely unobserved tuples in the saved encoding."""
    if not group["missing"]:
        return [], []
    indices = [result["parameters"].index(name) for name in group["parameters"]]
    observed = {
        tuple(point[index] for index in indices) for point in row["discretization"]["points"]
    }
    possible = product(*(range(row["levels"][name]) for name in group["parameters"]))
    missing = (values for values in possible if values not in observed)
    page = max(0, min(int(page or 0), max(0, ceil(group["missing"] / 20) - 1)))
    values = list(islice(missing, page * 20, (page + 1) * 20))
    definitions = row["discretization"]["definitions"]
    data, tooltips = [], []
    for values_row in values:
        data.append(
            {
                name: definitions[name][level]["range"]
                for name, level in zip(group["parameters"], values_row)
            }
        )
        tooltips.append(
            {
                name: {"value": definitions[name][level]["range"], "type": "text"}
                for name, level in zip(group["parameters"], values_row)
            }
        )
    return data, tooltips


def group_drilldown(result: Dict, row: Dict, group: Dict) -> Any:
    """Open one group's lazy missing-tuple table and stored level definitions."""
    data, tooltips = missing_tuple_page(result, row, group)
    definitions = row["discretization"]["definitions"]
    return html.Section(
        [
            html.H6("Missing bin combinations"),
            _tags(group["parameters"]),
            html.P(
                f"{group['total']:,} possible · {group['covered']:,} observed · "
                f"{group['missing']:,} missing",
                className="coverage-group-counts",
            ),
            (
                dash_table.DataTable(
                    id=_id("group-tuples"),
                    data=data,
                    columns=[{"id": name, "name": name} for name in group["parameters"]],
                    tooltip_data=tooltips,
                    tooltip_duration=None,
                    page_action="custom",
                    page_current=0,
                    page_size=20,
                    page_count=max(1, ceil(group["missing"] / 20)),
                    style_table={"overflowX": "auto", "maxHeight": "300px", "overflowY": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "padding": "5px 8px",
                        "whiteSpace": "normal",
                        "minWidth": "75px",
                        "maxWidth": "180px",
                        "fontSize": "12px",
                    },
                )
                if group["missing"]
                else html.P(
                    f"This {len(group['parameters'])}-parameter combination covers every level combination."
                )
            ),
            html.Details(
                [
                    html.Summary("Level definitions"),
                    html.Dl(
                        [
                            html.Div(
                                [
                                    html.Dt(name),
                                    html.Dd(
                                        " · ".join(
                                            f"{level['label']}: {level['range']}"
                                            for level in definitions[name]
                                        )
                                    ),
                                ]
                            )
                            for name in group["parameters"]
                        ]
                    ),
                ],
                className="coverage-group-definitions",
            ),
        ],
        className="coverage-group-drilldown",
    )
