"""Compact coverage-gap diagnostics using saved counts and bin assignments."""

from typing import Any, Dict, List, Optional

from itertools import combinations, islice, product
from math import ceil, comb, prod

import numpy as np
from dash import dash_table, dcc, html

from deepcave.evaluators.combinatorial import MAX_GROUPS, MAX_PROJECTIONS, _group_counts
from deepcave.plugins.hyperparameter.coverage_views import _id


def diagnostic_row(result: Optional[Dict], resolution: Any) -> Optional[Dict]:
    """Find a saved resolution without preparing or measuring trial data."""
    return next(
        (row for row in (result or {}).get("rows", []) if row["resolution"] == resolution), None
    )


def diagnostic_cell(result: Dict, row: Dict) -> Optional[Dict]:
    """Use the next strength, or the full-strength cell when everything is covered."""
    target = min(row["observed_strength"] + 1, len(result["parameters"]))
    return next((cell for cell in row["cells"] if cell["strength"] == target), None)


def group_records(result: Dict, row: Dict, parameters: Optional[List[str]] = None) -> List[Dict]:
    """Read exact group counts and apply an AND filter for any interaction strength."""
    if row.get("strength_is_lower_bound"):
        return []
    cell = diagnostic_cell(result, row)
    if not cell:
        return []
    names = result["parameters"]
    strength = cell["strength"]
    counts = cell.get("group_covered")
    if counts is None:
        # Trial-count-limited cells still have exact saved bin assignments.
        # Reconstruct group counts on demand without enumerating missing tuples.
        points = row.get("discretization", {}).get("points")
        group_count = comb(len(names), strength)
        if (
            cell["status"] != "trial_limit"
            or points is None
            or group_count > MAX_GROUPS
            or group_count * len(points) > MAX_PROJECTIONS
        ):
            return []
        sizes = [row["levels"][name] for name in names]
        counts = _group_counts(
            np.asarray(points, dtype=np.int64),
            sizes,
            list(combinations(range(len(names)), strength)),
        ).tolist()
    if len(counts) != comb(len(names), strength):
        return []
    selected = set(parameters or [])
    records = []
    for index, (group, covered) in enumerate(zip(combinations(names, strength), counts)):
        if not selected.issubset(group):
            continue
        total = prod(row["levels"][name] for name in group)
        records.append(
            dict(
                index=index,
                parameters=list(group),
                total=total,
                covered=covered,
                missing=total - covered,
                coverage=covered / total,
            )
        )
    return sorted(records, key=lambda group: (-group["missing"], tuple(group["parameters"])))


def _tags(parameters: List[str]) -> Any:
    return html.Span(
        [html.Span(name, className="coverage-group-tag") for name in parameters],
        className="coverage-group-tags",
    )


def _group_id(kind: str, row: Dict, group: Dict) -> Dict:
    return {"type": _id(kind), "resolution": row["resolution"], "group": group["index"]}


def group_overview(
    result: Dict,
    row: Dict,
    parameters: Any = None,
    limit: Any = 5,
    include_complete: bool = False,
) -> Any:
    """List incomplete groups first, with independent, lazily loaded accordions."""
    groups = group_records(result, row, parameters)
    visible = [group for group in groups if include_complete or group["missing"] > 0]
    shown = visible if limit == "all" else visible[: int(limit or 5)]
    entries = []
    for group in shown:
        line = [
            _tags(group["parameters"]),
            html.Span(f"{group['missing']:,} / {group['total']:,}"),
            html.Span(f"{group['coverage']:.1%}"),
        ]
        if group["missing"]:
            entries.append(
                html.Details(
                    [
                        html.Summary(
                            line,
                            id=_group_id("group-open", row, group),
                            n_clicks=0,
                            className="coverage-group-line",
                            title="Show missing bin combinations",
                        ),
                        html.Div(id=_group_id("group-content", row, group)),
                    ],
                    id=_group_id("group-details", row, group),
                    open=False,
                    className="coverage-group-row",
                )
            )
        else:
            entries.append(html.Div(line, className="coverage-group-line coverage-group-covered"))
    empty = (
        "No parameter groups contain all selected parameters."
        if not groups
        else "No incomplete parameter groups match this filter."
    )
    return html.Div(
        [
            html.H6("Parameter groups" if include_complete else "Incomplete parameter groups"),
            (
                html.Div(
                    [html.Span("Parameter group"), html.Span("Missing"), html.Span("Coverage")],
                    className="coverage-group-line coverage-group-head",
                )
                if entries
                else None
            ),
            html.Div(entries, className="coverage-group-list") if entries else html.P(empty),
        ]
    )


def group_diagnostics(result: Dict, row: Dict, cell: Dict) -> Any:
    """Build a compact shell; tuple enumeration happens only when a group is opened."""
    heading = html.H5(f"Strength {cell['strength']} · Coverage gaps")
    groups = group_records(result, row)
    if not groups:
        message = (
            "Current strength is not fully evaluated; the next strength is unknown."
            if row.get("strength_is_lower_bound")
            else "Refresh coverage to load exact parameter-group counts and bin definitions."
        )
        if cell["status"] == "work_limit" and not row.get("strength_is_lower_bound"):
            message = "Parameter-group coverage is unavailable at the current computation limit."
        return html.Div([heading, html.P(message)], className="coverage-group-diagnostics")
    total = len(groups)
    incomplete = sum(group["missing"] > 0 for group in groups)
    noun = "parameter group" if total == 1 else "parameter groups"
    summary = (
        f"{incomplete:,} of {total:,} {noun} incomplete"
        if incomplete
        else f"All {total:,} {noun} fully covered"
    )
    if "discretization" not in row:
        return html.Div(
            [
                heading,
                html.P(summary),
                html.P("Refresh coverage to load the saved bin definitions."),
            ],
            className="coverage-group-diagnostics",
        )
    return html.Div(
        [
            heading,
            html.P(summary, className="coverage-group-summary"),
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
                                    {"label": str(value), "value": value} for value in (5, 10, 20)
                                ]
                                + [{"label": "All", "value": "all"}],
                                value=5,
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                        className="coverage-group-limit",
                    ),
                    dcc.Checklist(
                        id=_id("group-complete"),
                        options=[{"label": "Include fully covered", "value": "complete"}],
                        value=[],
                        className="coverage-group-complete",
                    ),
                ],
                className="coverage-group-controls",
            ),
            html.Div(group_overview(result, row), id=_id("group-overview")),
        ],
        className="coverage-group-diagnostics",
    )


def missing_tuple_page(result: Dict, row: Dict, group: Dict, page: int = 0) -> tuple:
    """Enumerate at most twenty actual gaps from the supported independent search space."""
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
    definitions = row["discretization"].get("definitions", {})

    def description(name: str, level: int) -> str:
        levels = definitions.get(name, [])
        if level < len(levels):
            return str(
                levels[level].get("range") or levels[level].get("label") or f"Bin {level + 1}"
            )
        return f"Bin {level + 1}"

    data, tooltips = [], []
    for values_row in values:
        record = {
            name: description(name, level) for name, level in zip(group["parameters"], values_row)
        }
        data.append(record)
        tooltips.append({name: {"value": value, "type": "text"} for name, value in record.items()})
    return data, tooltips


def group_drilldown(result: Dict, row: Dict, group: Dict) -> Any:
    """Load only the opened group's first page; each group has independent pagination."""
    data, tooltips = missing_tuple_page(result, row, group)
    return html.Section(
        [
            html.H6("Missing bin combinations"),
            dash_table.DataTable(
                id=_group_id("group-tuples", row, group),
                data=data,
                columns=[{"id": name, "name": name} for name in group["parameters"]],
                tooltip_data=tooltips,
                tooltip_duration=None,
                page_action="custom",
                page_current=0,
                page_size=20,
                page_count=max(1, ceil(group["missing"] / 20)),
                style_table={"overflowX": "auto", "maxHeight": "280px", "overflowY": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "5px 8px",
                    "whiteSpace": "normal",
                    "minWidth": "75px",
                    "maxWidth": "180px",
                    "fontSize": "12px",
                },
            ),
        ],
        className="coverage-group-drilldown",
    )
