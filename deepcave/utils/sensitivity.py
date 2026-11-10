"""Plotly views and table projections for the seed sensitivity evaluator."""

from typing import Any, Dict, Optional, Sequence

from html import escape

import numpy as np
import plotly.graph_objects as go
from dash.dash_table.Format import Format, Scheme
from plotly.subplots import make_subplots

from deepcave.evaluators.sensitivity import (
    canonical,
    fold_diagnostics,
    pareto_ids,
    randomness_ranges,
)
from deepcave.utils.analysis_layout import style_analysis
from deepcave.utils.candidates import CANDIDATE_COLORS
from deepcave.utils.sensitivity_references import reference_key, reference_objective


def styled(figure: go.Figure, x: str = "", y: str = "", height: int = 390) -> go.Figure:
    """Apply the analysis suite's shared Plotly conventions."""
    style_analysis(figure, height=height, show_legend=False)
    figure.update_layout(margin={"t": 25, "b": 65, "l": 70, "r": 35}, hovermode="closest")
    figure.update_xaxes(title_text=x)
    figure.update_yaxes(title_text=y)
    return figure


def table_projection(groups: Sequence[Dict]) -> tuple:
    """Expose sort/filter values without rounding configuration identity or statistics."""
    fields = [
        ("Configuration ID", "label"),
        ("Evaluations", "evaluations"),
        ("Distinct seeds", "seeds"),
        ("Mean", "mean"),
        ("Median", "median"),
        ("SD", "sd"),
        ("IQR", "iqr"),
        ("Minimum", "minimum"),
        ("Maximum", "maximum"),
        ("Best-seed gap", "best_gap"),
        ("Repeated seed pairs", "duplicates"),
        ("Context", "context_label"),
    ]
    columns = []
    for label, name in fields:
        column = {
            "name": label,
            "id": name,
            "type": "text" if name in ("label", "context_label") else "numeric",
        }
        if column["type"] == "numeric":
            column["format"] = (
                {"specifier": "d"}
                if name in ("evaluations", "seeds", "duplicates")
                else Format(precision=6, scheme=Scheme.decimal_or_exponent)
            )
        columns.append(column)
    parameters = sorted({name for group in groups for name in group["parameters"]})
    for name in parameters:
        values = [g["parameters"][name] for g in groups if name in g["parameters"]]
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        columns.append(
            {"name": name, "id": "param:" + name, "type": "numeric" if numeric else "text"}
        )
    rows = []
    for group in groups:
        row = {name: group.get(name) for _, name in fields}
        row["id"] = group["id"]
        row.update({"param:" + name: value for name, value in group["parameters"].items()})
        rows.append(row)
    return rows, columns


def explorer_figure(
    groups: Sequence[Dict],
    maximize: bool,
    frontier: bool = True,
    active: Any = None,
    selected: Sequence[str] = (),
) -> go.Figure:
    """Plot conditional seed SDs against means with stable configuration colors."""
    figure = go.Figure()
    valid = [g for g in groups if g["sd"] is not None]
    hover = [
        f"<b>{escape(g['label'])}</b><br>Mean: {g['mean']:.6g}<br>SD: {g['sd']:.6g}"
        f"<br>Distinct seeds: {g['seeds']}<br>Min / max: {g['minimum']:.6g} / {g['maximum']:.6g}"
        f"<br>Studies: {escape(', '.join(sorted({r['study_name'] for r in g['rows']})))}"
        + ("<br>Few seeds: SD is uncertain" if g["seeds"] < 4 else "")
        for g in valid
    ]
    figure.add_trace(
        go.Scatter(
            x=[g["sd"] for g in valid],
            y=[g["mean"] for g in valid],
            mode="markers",
            customdata=[[g["id"]] for g in valid],
            hovertext=hover,
            hoverinfo="text",
            marker={
                "color": [g["color"] for g in valid],
                "opacity": 0.8,
                "size": [min(23, 6 + 2 * np.sqrt(g["seeds"])) for g in valid],
                "symbol": ["diamond-open" if g["seeds"] < 4 else "circle" for g in valid],
                "line": {
                    "color": ["#0f172a" if g["id"] == active else g["color"] for g in valid],
                    "width": [
                        3 if g["id"] == active else 2 if g["id"] in selected else 0 for g in valid
                    ],
                },
            },
            name="Configurations",
        )
    )
    if frontier:
        ids = set(pareto_ids(valid, maximize))
        points = sorted([g for g in valid if g["id"] in ids], key=lambda g: g["sd"])
        figure.add_trace(
            go.Scatter(
                x=[g["sd"] for g in points],
                y=[g["mean"] for g in points],
                mode="lines",
                line={"color": "#94a3b8", "width": 1, "dash": "dot"},
                hoverinfo="skip",
                name="Pareto frontier",
            )
        )
    styled(figure, "Seed standard deviation", "Mean objective", 440)
    figure.update_layout(
        dragmode="lasso", uirevision="sensitivity-explorer", meta={"sensitivity": True}
    )
    return figure


def trial_label(row: Dict) -> str:
    """Identify raw evaluations even when trial numbers repeat in different studies."""
    return (
        f"{row['study_name']} · Trial {row['trial_number']} · "
        f"M={row['model_seed']} D={row['data_seed']}"
    )


def distribution_reference_trace(groups: Sequence[Dict], references: Optional[Dict]) -> Any:
    """Build only the reference markers, independently of the seed plots."""
    x, y, hover = [], [], []
    for i, group in enumerate(groups):
        trial = references.get(reference_key(group)) if references else None
        value = reference_objective(trial, group) if trial else None
        if value is not None:
            x.append(i)
            y.append(value)
            hover.append(
                f"<b>Reference trial · {escape(group['label'])}</b><br>"
                f"{escape(trial['study_name'])} · Trial {trial['trial_number']}<br>"
                f"Objective: {value:.7g}"
            )
    if not x:
        return None
    return go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker={
            "symbol": "diamond",
            "size": 14,
            "color": "#dc2626",
            "line": {"width": 1, "color": "white"},
        },
        hovertext=hover,
        hoverinfo="text",
        name="Reference trial objective",
        showlegend=True,
    )


def distribution_figure(
    groups: Sequence[Dict], selected: Sequence[str] = (), references: Optional[Dict] = None
) -> go.Figure:
    """Overlay raw evaluations on each configuration's seed-pair box summary."""
    figure = go.Figure()
    for i, group in enumerate(groups):
        rows = group["rows"]
        figure.add_trace(
            go.Box(
                x=[i] * group["seeds"],
                y=[r["objective"] for r in group["replicates"]],
                boxpoints=False,
                width=0.6,
                line={"color": group["color"]},
                fillcolor="rgba(148,163,184,.12)",
                name=group["label"],
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[
                    i + (0.36 * (j / (len(rows) - 1) - 0.5) if len(rows) > 1 else 0)
                    for j in range(len(rows))
                ],
                y=[r["objective"] for r in rows],
                customdata=[[r["id"], group["id"]] for r in rows],
                mode="markers",
                marker={
                    "color": group["color"],
                    "opacity": 0.8,
                    "size": [11 if r["id"] in selected else 7 for r in rows],
                    "line": {
                        "width": [2 if r["id"] in selected else 0 for r in rows],
                        "color": "#0f172a",
                    },
                },
                hovertext=[
                    escape(trial_label(r))
                    + f"<br>Objective: {r['objective']:.7g}<br>Trial ID: {r['trial_id']}"
                    for r in rows
                ],
                hoverinfo="text",
                name=group["label"],
                showlegend=False,
            )
        )
    reference = distribution_reference_trace(groups, references)
    if reference is not None:
        figure.add_trace(reference)
    styled(figure, "Configuration", "Objective")
    figure.update_layout(boxmode="overlay", uirevision=canonical([g["id"] for g in groups]))
    if reference is not None:
        figure.update_layout(
            showlegend=True, legend={"orientation": "h", "x": 0, "y": 1.12}, margin={"t": 50}
        )
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(groups))),
        ticktext=[f"<b>{escape(g['label'])}</b>" for g in groups],
        tickangle=0,
    )
    return figure


def seed_means(group: Dict, axis: str) -> list:
    """Average equally weighted observed pair means within each known seed level."""
    levels: Dict[str, Dict] = {}
    for rep in group["replicates"]:
        seed = rep[axis]
        if seed is not None:
            level = levels.setdefault(canonical(seed), {"seed": seed, "values": []})
            level["values"].append(rep["objective"])
    return [
        {"seed": level["seed"], "objective": float(np.mean(level["values"]))}
        for _, level in sorted(
            levels.items(), key=lambda item: (isinstance(item[1]["seed"], str), item[1]["seed"])
        )
    ]


def align_seed_distributions(figure: go.Figure, groups: Sequence[Dict]) -> None:
    """Keep category centers and plot widths identical across the two seed distributions."""
    figure.update_layout(
        boxmode="group", boxgap=0.3, boxgroupgap=0.3, showlegend=True, margin={"l": 70, "r": 35}
    )
    figure.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=[g["id"] for g in groups],
        tickmode="array",
        tickvals=[g["id"] for g in groups],
        ticktext=[g["label"] for g in groups],
        tickangle=-30,
        range=[-0.5, max(len(groups) - 0.5, 0.5)],
        automargin=False,
    )
    figure.update_yaxes(automargin=False)


def seed_distribution_figure(groups: Sequence[Dict]) -> go.Figure:
    """Compare distributions of data-seed and model-seed means per configuration."""
    figure = go.Figure()
    for axis, label, color in (
        ("data_seed", "Data seed means", "#2563eb"),
        ("model_seed", "Model seed means", "#d97706"),
    ):
        x, y, seeds = [], [], []
        for group in groups:
            for level in seed_means(group, axis):
                x.append(group["id"])
                y.append(level["objective"])
                seeds.append(str(level["seed"]))
        figure.add_trace(
            go.Box(
                x=x,
                y=y,
                customdata=seeds,
                name=label,
                offsetgroup=axis,
                boxpoints="all",
                pointpos=0,
                jitter=0.25,
                marker_color=color,
                hovertemplate="Seed: %{customdata}<br>Mean objective: %{y:.7g}<extra>%{fullData.name}</extra>",
            )
        )
    styled(figure, "Configuration", "Objective")
    align_seed_distributions(figure, groups)
    figure.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, title_text="")
    figure.update_layout(
        height=440,
        margin={"b": 165},
        legend={"orientation": "h", "x": 0, "y": -0.52, "groupclick": "togglegroup"},
    )
    for index, group in enumerate(groups):
        for label, color, offset in (
            ("Data seeds", "#2563eb", -0.175),
            ("Model seeds", "#d97706", 0.175),
        ):
            figure.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=index + offset - 0.1225,
                x1=index + offset + 0.1225,
                y0=-0.11,
                y1=-0.11,
                line={"color": color, "width": 1.5},
            )
            figure.add_annotation(
                x=index + offset,
                y=-0.15,
                xref="x",
                yref="paper",
                text=label,
                showarrow=False,
                yanchor="top",
                font={"size": 12, "color": color},
            )
        figure.add_annotation(
            x=index,
            y=-0.32,
            xref="x",
            yref="paper",
            text=f"<b>{escape(group['label'])}</b>",
            showarrow=False,
            yanchor="top",
            font={"size": 12},
        )
    for index in range(1, len(groups)):
        figure.add_vline(x=index - 0.5, line_color="#cbd5e1", line_width=1)
    return figure


def trials_by_seed_figure(groups: Sequence[Dict]) -> go.Figure:
    """Show raw trials at each fixed seed, alongside equally weighted seed-pair means."""
    figure = go.Figure()
    ticks, labels = [], []
    for axis, prefix, label, color, offset in (
        ("model_seed", "M", "Data seed variability", "#2563eb", -0.25),
        ("data_seed", "D", "Model seed variability", "#d97706", 0.25),
    ):
        x, y, hover, identities = [], [], [], []
        mean_x, mean_y, mean_hover = [], [], []
        minimum_spans, maximum_spans = [], []
        for index, group in enumerate(groups):
            levels = seed_means(group, axis)
            if levels:
                figure.add_shape(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=index + offset - 0.2,
                    x1=index + offset + 0.2,
                    y0=1.01,
                    y1=1.01,
                    line={"color": color, "width": 1.5},
                )
                figure.add_annotation(
                    x=index + offset,
                    y=1.04,
                    xref="x",
                    yref="paper",
                    text=label.replace(" variability", "<br>variability"),
                    showarrow=False,
                    yanchor="bottom",
                    font={"size": 12, "color": color},
                )
                figure.add_shape(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=index + offset - 0.2,
                    x1=index + offset + 0.2,
                    y0=-0.11,
                    y1=-0.11,
                    line={"color": "#d97706" if axis == "model_seed" else "#2563eb", "width": 1.5},
                )
                figure.add_annotation(
                    x=index + offset,
                    y=-0.15,
                    xref="x",
                    yref="paper",
                    text="Model seeds" if axis == "model_seed" else "Data seeds",
                    showarrow=False,
                    yanchor="top",
                    font={"size": 12, "color": "#d97706" if axis == "model_seed" else "#2563eb"},
                )
            for position, level in enumerate(levels):
                center = index + offset + 0.4 * ((position + 0.5) / len(levels) - 0.5)
                seed = level["seed"]
                rows = [row for row in group["rows"] if canonical(row[axis]) == canonical(seed)]
                ticks.append(center)
                labels.append(str(seed))
                mean_x.append(center)
                mean_y.append(level["objective"])
                minimum_spans.append(level["objective"] - min(row["objective"] for row in rows))
                maximum_spans.append(max(row["objective"] for row in rows) - level["objective"])
                mean_hover.append(
                    f"<b>{escape(group['label'])} · {prefix}={escape(str(seed))}</b>"
                    f"<br>Seed-pair mean: {level['objective']:.7g}"
                )
                for point, row in enumerate(rows):
                    jitter = (
                        0.02 / len(levels) * (point / (len(rows) - 1) - 0.5) if len(rows) > 1 else 0
                    )
                    x.append(center + jitter)
                    y.append(row["objective"])
                    identities.append([row["id"], group["id"]])
                    hover.append(
                        f"<b>{escape(group['label'])}</b><br>{escape(trial_label(row))}"
                        f"<br>Objective: {row['objective']:.7g}"
                    )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=label,
                legendgroup=axis,
                marker={"color": color, "size": 7, "opacity": 0.2},
                customdata=identities,
                hovertext=hover,
                hoverinfo="text",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=mean_x,
                y=mean_y,
                mode="markers",
                name="Seed-pair mean",
                legendgroup=axis,
                showlegend=False,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": maximum_spans,
                    "arrayminus": minimum_spans,
                    "width": 0,
                    "thickness": 1.3,
                    "color": color,
                },
                marker={
                    "color": "#d97706" if axis == "model_seed" else "#2563eb",
                    "symbol": "line-ew",
                    "size": 10,
                    "line": {"width": 2, "color": "#d97706" if axis == "model_seed" else "#2563eb"},
                },
                hovertext=mean_hover,
                hoverinfo="text",
            )
        )
    styled(figure, "", "Objective", 490)
    figure.update_layout(
        showlegend=False,
        margin={"t": 75, "b": 165},
        legend={"orientation": "h", "x": 0, "y": -0.52, "groupclick": "togglegroup"},
        uirevision=canonical([group["id"] for group in groups]),
    )
    figure.update_xaxes(
        tickmode="array",
        tickvals=ticks,
        ticktext=labels,
        tickangle=0,
        range=[-0.5, max(len(groups) - 0.5, 0.5)],
        showgrid=False,
        zeroline=False,
    )
    for index, group in enumerate(groups):
        figure.add_annotation(
            x=index,
            y=-0.32,
            xref="x",
            yref="paper",
            text=f"<b>{escape(group['label'])}</b>",
            showarrow=False,
            yanchor="top",
            font={"size": 12},
        )
        figure.add_vline(x=index - 0.5, line_color="#cbd5e1", line_width=1)
    return figure


def randomness_range_figure(groups: Sequence[Dict]) -> go.Figure:
    """Compare the total range with ranges holding the other seed constant."""
    figure = go.Figure()
    for source_index, (axis, label, fixed_label, color, fill) in enumerate(
        (
            (None, "Total range", "Seed combinations", "#64748b", "rgba(100,116,139,0.14)"),
            ("data_seed", "Data randomness", "Model seed", "#2563eb", "rgba(37,99,235,0.14)"),
            ("model_seed", "Model randomness", "Data seed", "#d97706", "rgba(217,119,6,0.14)"),
        )
    ):
        x, y, customdata, sizes, symbols = [], [], [], [], []
        centers, maxima, stem_x, stem_y = [], [], [], []
        half_width = 0.7 / 3 * 0.7 / 2
        for index, group in enumerate(groups):
            if axis is None:
                values = [pair["objective"] for pair in group["replicates"]]
                observations = (
                    [
                        {
                            "fixed_seed": "all",
                            "minimum": min(values),
                            "maximum": max(values),
                            "range": max(values) - min(values),
                        }
                    ]
                    if len(values) >= 2
                    else []
                )
            else:
                observations = randomness_ranges(group, axis)
            if not observations:
                continue
            center = index + (source_index - 1) * 0.7 / 3
            values = [item["range"] for item in observations]
            maximum = max(values)
            centers.append(center)
            maxima.append(maximum)
            stem_x.extend([center, center, None])
            stem_y.extend([0, maximum, None])
            for point_index, observation in enumerate(observations):
                jitter = (
                    0.9 * half_width * (point_index / (len(observations) - 1) - 0.5)
                    if len(observations) > 1
                    else 0
                )
                is_maximum = observation["range"] == maximum
                x.append(center if is_maximum else center + jitter)
                y.append(observation["range"])
                sizes.append(15 if is_maximum else 9)
                symbols.append("line-ew" if is_maximum else "circle")
                customdata.append(
                    [
                        group["label"],
                        fixed_label,
                        str(observation["fixed_seed"]),
                        observation["minimum"],
                        observation["maximum"],
                    ]
                )
        figure.add_trace(
            go.Scatter(
                x=stem_x,
                y=stem_y,
                mode="lines",
                line={"color": fill, "width": 4},
                connectgaps=False,
                name=label,
                legendgroup=axis or "total",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=centers,
                y=maxima,
                text=[
                    f"<b>{label.split()[0]}<br>{maximum:.3g}</b>" for maximum in maxima
                ],
                mode="markers+text",
                marker={"size": 28, "color": "rgba(0,0,0,0)", "line": {"width": 0}},
                textfont={"color": color, "size": 15},
                textposition="top center",
                name=label,
                legendgroup=axis or "total",
                showlegend=False,
                hoverinfo="skip",
                cliponaxis=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                customdata=customdata,
                name=label,
                legendgroup=axis or "total",
                mode="markers",
                marker={
                    "color": color,
                    "size": sizes,
                    "symbol": symbols,
                    "opacity": 0.95,
                    "line": {
                        "color": [color if symbol == "line-ew" else "white" for symbol in symbols],
                        "width": [2 if symbol == "line-ew" else 1.5 for symbol in symbols],
                    },
                },
                cliponaxis=False,
                hovertemplate=(
                    "Configuration ID: %{customdata[0]}"
                    "<br>Randomness source: %{fullData.name}"
                    + (
                        "<br>All observed seed combinations"
                        if axis is None
                        else "<br>Fixed %{customdata[1]}: %{customdata[2]}"
                    )
                    + "<br>Minimum objective: %{customdata[3]:.7g}"
                    "<br>Maximum objective: %{customdata[4]:.7g}"
                    "<br>Objective range: %{y:.7g}<extra></extra>"
                ),
            )
        )
    styled(figure, "Configuration", "Objective range (max − min)", height=450)
    figure.update_layout(margin={"b": 140}, barmode="overlay")
    align_seed_distributions(figure, groups)
    figure.update_xaxes(
        type="linear",
        tickvals=list(range(len(groups))),
        ticktext=[f"<b>{escape(group['label'])}</b>" for group in groups],
        tickangle=0,
        tickfont_size=12,
        title_text="",
        showgrid=False,
    )
    for index in range(1, len(groups)):
        figure.add_vline(x=index - 0.5, line_color="#cbd5e1", line_width=1)
    extent = max(
        (value for trace in figure.data for value in trace.y if value is not None), default=0
    )
    figure.update_yaxes(range=[0, extent * 1.22 or 1.0])
    return figure


def seed_mean_figure(groups: Sequence[Dict], axis: str = "model_seed") -> go.Figure:
    """Plot one configuration line over the observed seed-level means."""
    figure = go.Figure()
    seeds = sorted(
        {r[axis] for g in groups for r in g["replicates"] if r[axis] is not None},
        key=lambda seed: (isinstance(seed, str), seed),
    )
    for group in groups:
        levels = {canonical(level["seed"]): level["objective"] for level in seed_means(group, axis)}
        figure.add_trace(
            go.Scatter(
                x=[str(seed) for seed in seeds],
                y=[levels.get(canonical(seed)) for seed in seeds],
                mode="lines+markers",
                name=group["label"],
                line={"color": group["color"]},
                connectgaps=False,
                hovertemplate="Seed: %{x}<br>Mean objective: %{y:.7g}<extra>%{fullData.name}</extra>",
            )
        )
    styled(figure, "Model seed" if axis == "model_seed" else "Data seed", "Mean objective")
    figure.update_layout(showlegend=True)
    figure.update_xaxes(
        type="category", categoryorder="array", categoryarray=[str(seed) for seed in seeds]
    )
    return figure


def seed_profile_figure(groups: Sequence[Dict], x_axis: str = "data_seed") -> go.Figure:
    """Facet raw evaluations with interchangeable seed axes and line groups."""
    columns = min(4, len(groups))
    rows = (len(groups) + columns - 1) // columns
    height = 100 + 200 * rows
    evaluations = [row for group in groups for row in group["rows"]]
    line_axis = "model_seed" if x_axis == "data_seed" else "data_seed"
    x_seeds = {canonical(row[x_axis]): row[x_axis] for row in evaluations}
    line_seeds = {canonical(row[line_axis]): row[line_axis] for row in evaluations}
    x_keys = sorted(x_seeds, key=lambda key: (isinstance(x_seeds[key], str), x_seeds[key] or 0))
    line_keys = sorted(
        line_seeds, key=lambda key: (isinstance(line_seeds[key], str), line_seeds[key] or 0)
    )
    colors = {key: CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)] for i, key in enumerate(line_keys)}
    figure = make_subplots(
        rows=rows,
        cols=columns,
        shared_yaxes="all",
        horizontal_spacing=0.06,
        vertical_spacing=min(0.15, 55 / height),
        subplot_titles=[escape(g["label"]).replace(" · ", "<br>") for g in groups],
    )
    shown = set()
    for i, group in enumerate(groups):
        facet_row, facet_col = i // columns + 1, i % columns + 1
        for key in line_keys:
            raw = [row for row in group["rows"] if canonical(row[line_axis]) == key]
            if not raw:
                continue
            name = ("Model seed " if line_axis == "model_seed" else "Data seed ") + str(
                line_seeds[key]
            )
            pairs = {
                canonical(rep[x_axis]): rep["objective"]
                for rep in group["replicates"]
                if canonical(rep[line_axis]) == key
            }
            figure.add_trace(
                go.Scatter(
                    x=list(range(len(x_keys))),
                    y=[pairs.get(seed) for seed in x_keys],
                    mode="lines",
                    line={"color": colors[key], "width": 1.5},
                    legendgroup=key,
                    name=name,
                    showlegend=False,
                    connectgaps=False,
                    hoverinfo="skip",
                ),
                row=facet_row,
                col=facet_col,
            )
            figure.add_trace(
                go.Scatter(
                    x=[x_keys.index(canonical(row[x_axis])) for row in raw],
                    y=[row["objective"] for row in raw],
                    mode="markers",
                    marker={"color": colors[key], "size": 5, "opacity": 0.85},
                    name=name,
                    legendgroup=key,
                    showlegend=key not in shown,
                    customdata=[[row["id"]] for row in raw],
                    hovertext=[
                        escape(trial_label(row)) + f"<br>Objective: {row['objective']:.7g}"
                        for row in raw
                    ],
                    hoverinfo="text",
                ),
                row=facet_row,
                col=facet_col,
            )
            shown.add(key)
    values = [row["objective"] for row in evaluations]
    low, high = min(values), max(values)
    padding = (high - low) * 0.08 or max(abs(low) * 0.02, 0.01)
    styled(figure, height=height)
    figure.update_layout(
        margin={"t": 45, "b": 110, "l": 65, "r": 25},
        showlegend=True,
        # Keep the legend below the axis titles at a fixed pixel distance for any row count.
        legend={"orientation": "h", "x": 0, "y": -65 / (height - 155), "yanchor": "top"},
    )
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(x_keys))),
        ticktext=[str(x_seeds[key]) for key in x_keys],
        range=[-0.25, len(x_keys) - 0.75],
        tickfont={"size": 9},
        showgrid=False,
        zeroline=False,
    )
    figure.update_yaxes(
        range=[low - padding, high + padding],
        showticklabels=True,
        nticks=3,
        tickfont={"size": 9},
        zeroline=False,
    )
    figure.update_xaxes(title_text="Data seed" if x_axis == "data_seed" else "Model seed", row=rows)
    figure.update_yaxes(title_text="Objective", col=1)
    for i in range(len(groups), rows * columns):
        figure.update_xaxes(visible=False, row=i // columns + 1, col=i % columns + 1)
        figure.update_yaxes(visible=False, row=i // columns + 1, col=i % columns + 1)
    for i, annotation in enumerate(figure.layout.annotations):
        axis = "xaxis" + (str(i + 1) if i else "")
        annotation.update(x=figure.layout[axis].domain[0], xanchor="left", font={"size": 11})
    return figure


def seed_matrix_figure(group: Dict) -> go.Figure:
    """Retain missing cells and expose every cell's evaluation count in hover."""
    reps = group["replicates"]
    models = sorted({canonical(r["model_seed"]): r["model_seed"] for r in reps}.items())
    data = sorted({canonical(r["data_seed"]): r["data_seed"] for r in reps}.items())
    lookup = {(canonical(r["model_seed"]), canonical(r["data_seed"])): r for r in reps}
    z, counts = [], []
    for dkey, _ in data:
        cells = [lookup.get((mkey, dkey)) for mkey, _ in models]
        z.append([cell["objective"] if cell else None for cell in cells])
        counts.append([cell["n"] if cell else 0 for cell in cells])
    figure = go.Figure(
        go.Heatmap(
            x=[str(v) for _, v in models],
            y=[str(v) for _, v in data],
            z=z,
            customdata=counts,
            colorscale="Cividis",
            reversescale=group["direction"] == "lower",
            hoverongaps=False,
            text=[[f"n={n}" if n > 1 else "" for n in line] for line in counts],
            texttemplate="%{text}",
            hovertemplate=(
                "Model seed: %{x}<br>Data seed: %{y}<br>Seed-pair mean: %{z:.7g}"
                "<br>Evaluations: %{customdata}<extra></extra>"
            ),
            colorbar={"title": "Objective", "thickness": 12},
        )
    )
    styled(figure, "Model seed", "Data seed")
    figure.update_xaxes(type="category")
    figure.update_yaxes(type="category")
    figure.update_layout(plot_bgcolor="#edf0f4")
    return figure


def folds_figure(group: Dict, attribute: str) -> tuple:
    """Show raw trial × fold results without pairing unrelated fold partitions."""
    data = fold_diagnostics(group, attribute)
    figure = go.Figure(
        go.Heatmap(
            x=data["labels"],
            y=[trial_label(r["trial"]) for r in data["rows"]],
            z=[[r["folds"].get(label) for label in data["labels"]] for r in data["rows"]],
            colorscale="Cividis",
            reversescale=group["direction"] == "lower",
            hoverongaps=False,
            hovertemplate="%{y}<br>Fold %{x}: %{z:.7g}<extra></extra>",
            colorbar={"title": "Objective", "thickness": 12},
        )
    )
    styled(figure, "CV fold", "Seed evaluation", max(390, min(750, 160 + len(data["rows"]) * 22)))
    figure.update_xaxes(type="category")
    figure.update_yaxes(type="category", automargin=True)
    figure.update_layout(plot_bgcolor="#edf0f4")
    return figure, data


def global_sensitivity_figure(groups: Sequence[Dict]) -> go.Figure:
    """Compare all observed hyperparameters with shared SD, mean and seed-count scales."""
    groups = [g for g in groups if g["sd"] is not None and g["parameters"]]
    names = sorted({name for g in groups for name in g["parameters"]})
    if not names:
        return styled(go.Figure())
    columns = min(3, len(names))
    rows = (len(names) + columns - 1) // columns
    height = 270 * rows + 35
    plot_height = height - 110
    figure = make_subplots(
        rows=rows,
        cols=columns,
        shared_yaxes="all",
        horizontal_spacing=0.045,
        vertical_spacing=85 / plot_height if rows > 1 else 0,
    )
    styled(figure, height=height)
    counts = sorted({g["seeds"] for g in groups})
    # Anchor the standard diameter to the visible configurations' median seed count.
    size_scale = 8 / np.sqrt(np.median([g["seeds"] for g in groups]))
    for index, name in enumerate(names):
        row, col = divmod(index, columns)
        panel = hyperparameter_figure(groups, name)
        trace = panel.data[0]
        observed = [g for g in groups if name in g["parameters"]]
        trace.marker.update(
            coloraxis="coloraxis",
            showscale=False,
            size=[size_scale * np.sqrt(g["seeds"]) for g in observed],
            opacity=0.9,
            line={"width": 0.5, "color": "#ffffff"},
        )
        trace.showlegend = False
        figure.add_trace(trace, row=row + 1, col=col + 1)
        figure.update_xaxes(
            title_text=name,
            type=panel.layout.xaxis.type,
            tickformat=panel.layout.xaxis.tickformat,
            row=row + 1,
            col=col + 1,
        )
        if panel.layout.xaxis.type != "category":
            values = sorted(set(trace.x))
            if len(values) <= 10:
                figure.update_xaxes(
                    tickmode="array",
                    tickvals=values,
                    ticktext=[
                        np.format_float_scientific(value, precision=4, trim="-", exp_digits=1)
                        if panel.layout.xaxis.type == "log" and value < 0.1
                        else f"{value:g}"
                        for value in values
                    ],
                    row=row + 1,
                    col=col + 1,
                )
        figure.update_yaxes(
            title_text="Seed standard deviation" if col == 0 else None,
            showticklabels=col == 0,
            row=row + 1,
            col=col + 1,
        )
        x_domain = figure.layout["xaxis" + (str(index + 1) if index else "")].domain
        y_domain = figure.layout["yaxis" + (str(index + 1) if index else "")].domain
        header_top = y_domain[1] + 28 / plot_height
        figure.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=x_domain[0],
            x1=x_domain[1],
            y0=y_domain[1],
            y1=header_top,
            fillcolor="#f1f3f6",
            line={"width": 0},
            layer="below",
        )
        figure.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=x_domain[0],
            x1=x_domain[1],
            y0=y_domain[0],
            y1=header_top,
            fillcolor="rgba(0,0,0,0)",
            line={"color": "#d5dce7", "width": 1},
            layer="below",
        )
        figure.add_annotation(
            xref="paper",
            yref="paper",
            x=x_domain[0],
            y=(y_domain[1] + header_top) / 2,
            text=f"<b>{escape(name)}</b>",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            xshift=9,
            font={"size": 13, "color": "#202b4b"},
        )
    legend_counts = sorted({counts[0], counts[len(counts) // 2], counts[-1]})
    for count in legend_counts:
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=str(count),
                marker={"size": size_scale * np.sqrt(count), "color": "#8c9095"},
                showlegend=True,
                hoverinfo="skip",
            )
        )
    figure.update_layout(
        margin={"t": 45, "b": 65, "l": 75, "r": 120},
        plot_bgcolor="#fbfcfe",
        paper_bgcolor="#ffffff",
        coloraxis={
            "colorscale": "Cividis",
            "reversescale": groups[0]["direction"] == "lower",
            "cmin": min(g["mean"] for g in groups),
            "cmax": max(g["mean"] for g in groups),
            "colorbar": {
                "title": {"text": "Mean", "side": "top"},
                "thickness": 15,
                "len": 0.53,
                "y": 1.06,
                "yanchor": "top",
                "x": 1.02,
                "outlinecolor": "#a8afbc",
                "outlinewidth": 1,
                "tickfont": {"size": 11},
            },
        },
        showlegend=True,
        legend={
            "title": {"text": "Seed count"},
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 0.43,
            "yanchor": "top",
            "itemsizing": "trace",
            "font": {"size": 11},
            "itemclick": False,
            "itemdoubleclick": False,
        },
    )
    figure.update_xaxes(tickfont_size=11, title_font_size=12, title_standoff=8)
    figure.update_yaxes(tickfont_size=11, title_font_size=12)
    return figure


def hyperparameter_figure(groups: Sequence[Dict], name: str) -> go.Figure:
    """Respect Optuna's original distribution kind and logarithmic scaling."""
    groups = [g for g in groups if name in g["parameters"] and g["sd"] is not None]
    distributions = [r["distributions"].get(name, {}) for g in groups for r in g["rows"]]
    kinds = {d.get("kind") for d in distributions}
    logs = {d.get("log", False) for d in distributions}
    categorical = "categorical" in kinds or any(
        isinstance(g["parameters"][name], (str, bool)) or g["parameters"][name] is None
        for g in groups
    )
    # Changing distribution semantics are shown categorically instead of inventing a
    # common numerical scale. The exact typed values remain available in hover.
    mixed = len(kinds) > 1 or len(logs) > 1
    x = [
        str(g["parameters"][name]) if categorical or mixed else g["parameters"][name]
        for g in groups
    ]
    figure = go.Figure(
        go.Scatter(
            x=x,
            y=[g["sd"] for g in groups],
            mode="markers",
            customdata=[[g["id"]] for g in groups],
            marker={
                "color": [g["mean"] for g in groups],
                "colorscale": "Cividis",
                "reversescale": bool(groups and groups[0]["direction"] == "lower"),
                "showscale": True,
                "colorbar": {"title": "Mean", "thickness": 12},
                "size": [min(21, 6 + 2 * np.sqrt(g["seeds"])) for g in groups],
            },
            hovertext=[
                f"{escape(g['label'])}<br>{escape(name)}: {escape(repr(g['parameters'][name]))}"
                f"<br>SD: {g['sd']:.6g}<br>Mean: {g['mean']:.6g}<br>Seeds: {g['seeds']}"
                for g in groups
            ],
            hoverinfo="text",
        )
    )
    styled(figure, name, "Seed standard deviation")
    figure.update_xaxes(
        type="category" if categorical or mixed else "log" if logs == {True} else "linear"
    )
    if kinds == {"integer"} and not categorical and not mixed and logs != {True}:
        figure.update_xaxes(tickformat="d")
    return figure
