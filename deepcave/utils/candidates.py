"""Trial identities, exports and plots for the candidate selection workspace."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import hashlib
import json
import math

import plotly.graph_objects as go

from deepcave.runs import AbstractRun
from deepcave.utils.analysis_layout import style_analysis
from deepcave.utils.global_objective import main_objective_options, resolve_main_objective_id
from deepcave.utils.performance_cost import _finite, pareto_indices
from deepcave.utils.styled_plotty import get_color

CANDIDATE_COLORS = [get_color(index) for index in range(36)]


def candidate_data(run: AbstractRun) -> Dict[str, Any]:
    """Prepare all trial states once, preserving native IDs and parameter types."""
    meta = run.get_meta()
    study = meta["optuna_study_name"]
    storage = meta["optuna_storage_identity"]
    objectives = {}
    keymap = {}
    for option in main_objective_options(run):
        token = option["value"]
        index = resolve_main_objective_id(run, token)
        objective = run.get_objective(index)
        key = token if token.startswith("user_attr:") else "metric:" + objective.name
        objectives[key] = {"label": objective.name, "optimize": objective.optimize, "index": index}
        keymap[token] = key
    rows = []
    for trial in meta["optuna_candidates"]:
        # Legacy snapshots can still be explored, but cannot export unknown source IDs.
        identity = [storage, study, trial["trial_id"]]
        if trial["trial_id"] is None:
            identity.append(trial["trial_number"])
        key = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
        metrics = {}
        for token, metric_key in keymap.items():
            if token.startswith("user_attr:"):
                value = trial["user_attrs"].get(token[len("user_attr:") :])
            else:
                index = objectives[metric_key]["index"]
                value = trial["values"][index] if index < len(trial["values"]) else None
            metrics[metric_key] = value if _finite(value) else None
        rows.append(
            {
                **trial,
                "id": key,
                "storage": storage,
                "study_name": study,
                "study_id": meta.get("optuna_study_id"),
                "metrics": metrics,
                "metric_info": objectives,
                "color": CANDIDATE_COLORS[int(key[:8], 16) % len(CANDIDATE_COLORS)],
            }
        )
    return {"rows": rows, "objectives": objectives, "keymap": keymap}


def central_objective(data: Dict[str, Any], state: Any) -> Optional[str]:
    """Resolve the sidebar objective by semantic name across different study schemas."""
    keymap = data.get("keymap", {})
    token = (state or {}).get("value", "objective:0")
    return keymap.get(token) if token in keymap else next(iter(keymap.values()), None)


def selected_trials_toml(rows: Sequence[Dict[str, Any]], objective: Optional[str] = None) -> str:
    """Use the existing export's TOML encoding and metadata conventions for references."""

    def encode(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")

    blocks = []
    if objective is not None:
        blocks.append("[selection]\nobjective = " + encode(objective))
    seen = set()
    for row in rows:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        if row.get("study_id") is None or row.get("trial_id") is None:
            raise ValueError(
                "Original Optuna IDs are missing. Refresh this study's database snapshot."
            )
        lines = ["[[selected_trials]]"]
        for key in ("study_name", "study_id", "trial_id", "trial_number", "storage"):
            lines.append(f"{key} = {encode(row[key])}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def scatter_figure(
    rows: Sequence[Dict[str, Any]],
    objective: Optional[str],
    show_pareto: bool = False,
    direction: Optional[str] = None,
) -> go.Figure:
    """Plot only finite duration/objective pairs; selection presentation stays in the browser."""
    valid = [
        row
        for row in rows
        if _finite(row.get("duration"))
        and row["duration"] >= 0
        and _finite(row["metrics"].get(objective))
    ]
    figure = go.Figure()
    if valid:
        figure.add_trace(
            go.Scattergl(
                x=[row["duration"] for row in valid],
                y=[row["metrics"][objective] for row in valid],
                customdata=[
                    [row["id"], row["study_name"], row["trial_number"], row["color"]]
                    for row in valid
                ],
                mode="markers",
                name="Trials",
                marker={"color": "#a6adbb", "size": 6, "opacity": 0.65},
                hovertemplate=(
                    "%{customdata[1]} · Trial %{customdata[2]}<br>"
                    "Duration: %{x:.6g} s<br>Objective: %{y:.6g}<extra></extra>"
                ),
            )
        )
        directions = {direction or row["metric_info"][objective]["optimize"] for row in valid}
        if show_pareto and len(directions) == 1:
            indices = pareto_indices(
                [(row["duration"], row["metrics"][objective]) for row in valid],
                maximize=next(iter(directions)) == "upper",
            )
            figure.add_trace(
                go.Scatter(
                    x=[valid[i]["duration"] for i in indices],
                    y=[valid[i]["metrics"][objective] for i in indices],
                    mode="lines",
                    line={"color": "#64748b", "width": 1, "dash": "dot"},
                    hoverinfo="skip",
                    name="Pareto frontier",
                )
            )
        elif show_pareto:
            figure.add_annotation(
                text="Pareto frontier unavailable: studies use different directions.",
                x=0.5,
                y=1.06,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
    else:
        figure.add_annotation(
            text="No visible trials with a valid objective and duration.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    style_analysis(figure, height=430, show_legend=False)
    figure.update_layout(
        margin={"t": 40, "l": 62, "r": 18, "b": 55},
        dragmode="select",
        clickmode="event",
        uirevision=objective,
        meta={"candidate_scatter": True},
    )
    figure.update_xaxes(title="Trial duration (s)")
    label = next(
        (row["metric_info"][objective]["label"] for row in valid), objective or "Objective"
    )
    figure.update_yaxes(title=label)
    return figure


def _category(value: Any) -> str:
    """Keep strings, booleans and numeric categories distinguishable."""
    return json.dumps(value, ensure_ascii=False)


def _dimension(
    label: str,
    values: List[Any],
    *,
    categorical: bool = False,
    logarithmic: bool = False,
    integer: bool = False,
) -> Dict[str, Any]:
    """Encode missing values on a labelled lane outside the real numeric range."""
    missing = [value is None or (not categorical and not _finite(value)) for value in values]
    if categorical:
        labels = sorted({_category(value) for value, absent in zip(values, missing) if not absent})
        ticks = list(range(len(labels)))
        encoded = [
            None if absent else labels.index(_category(value))
            for value, absent in zip(values, missing)
        ]
    else:
        encoded = [
            None if absent else math.log10(value) if logarithmic else value
            for value, absent in zip(values, missing)
        ]
        present = [value for value in encoded if value is not None]
        low, high = (min(present), max(present)) if present else (0, 1)
        if low == high:
            ticks = [low]
        elif integer and not logarithmic:
            step = max(1, math.ceil((high - low) / 5))
            ticks = list(range(math.ceil(low), math.floor(high) + 1, step))
        else:
            ticks = [low + (high - low) * i / 4 for i in range(5)]
        labels = [f"{10 ** tick if logarithmic else tick:.5g}" for tick in ticks]
    present = [value for value in encoded if value is not None]
    low, high = (min(present), max(present)) if present else (0, 1)
    padding = (high - low) * 0.15 or max(abs(low) * 0.1, 1)
    if any(missing):
        sentinel = low - padding
        ticks = [sentinel] + ticks
        labels = ["Missing"] + labels
        encoded = [sentinel if value is None else value for value in encoded]
        low = sentinel
    if low == high:
        low, high = low - padding, high + padding
    return {
        "label": label,
        "values": encoded,
        "range": [low, high],
        "tickvals": ticks,
        "ticktext": labels,
    }


def parallel_figure(
    rows: Sequence[Dict[str, Any]],
    objective: Optional[str],
    axes: Optional[List[str]] = None,
) -> Tuple[go.Figure, str]:
    """Compare the selection on typed axes, making heterogeneous/missing data explicit."""
    figure = go.Figure()
    style_analysis(figure, height=440, show_legend=False)
    figure.update_layout(margin={"t": 65, "b": 35, "l": 65, "r": 65})
    if not rows:
        figure.add_annotation(
            text="Select trials from the table or scatterplot to compare their configurations.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        return figure, ""
    parameters = sorted({name for row in rows for name in row["parameters"]})
    dimensions = []
    notes = []
    for name in parameters if axes is None else axes:
        specs = [row["distributions"][name] for row in rows if name in row["distributions"]]
        if not specs:
            continue
        kinds = {(spec["kind"], spec["log"]) for spec in specs}
        mixed = len(kinds) > 1
        categorical = mixed or specs[0]["kind"] == "categorical"
        logarithmic = not categorical and specs[0]["log"]
        if mixed:
            notes.append(f"{name}: different parameter types/scales; shown as categories.")
        values = [row["parameters"].get(name) for row in rows]
        # A real categorical None is a value, distinct from a missing parameter.
        if categorical:
            values = [
                _category(row["parameters"][name]) if name in row["parameters"] else None
                for row in rows
            ]
        dimension = _dimension(
            name + (" (log₁₀)" if logarithmic else " (category)" if categorical else ""),
            values,
            categorical=categorical,
            logarithmic=logarithmic,
            integer=specs[0]["kind"] == "integer",
        )
        if categorical:
            dimension["ticktext"] = [
                json.loads(label) if label != "Missing" else label
                for label in dimension["ticktext"]
            ]
        dimensions.append(dimension)
    info = next(
        (row["metric_info"][objective] for row in rows if objective in row["metric_info"]), {}
    )
    dimensions.extend(
        [
            _dimension(
                "Objective · " + info.get("label", objective or "unavailable"),
                [row["metrics"].get(objective) for row in rows],
            ),
            _dimension("Duration (s)", [row["duration"] for row in rows]),
        ]
    )
    colorscale = []
    count = len(rows)
    for index, row in enumerate(rows):
        colorscale.extend([[index / count, row["color"]], [(index + 1) / count, row["color"]]])
    figure.add_trace(
        go.Parcoords(
            dimensions=dimensions,
            line={
                "color": list(range(count)),
                "colorscale": colorscale,
                "cmin": -0.5,
                "cmax": count - 0.5,
                "showscale": False,
            },
            labelfont={"size": 12},
            tickfont={"size": 11},
            meta={"candidate_ids": [row["id"] for row in rows]},
        )
    )
    # A separate, initially empty overlay makes hover visible without changing selection.
    figure.add_trace(
        go.Parcoords(
            dimensions=[{**dimension, "values": []} for dimension in dimensions],
            line={"color": "#172554"},
            labelfont={"color": "rgba(0,0,0,0)"},
            tickfont={"color": "rgba(0,0,0,0)"},
        )
    )
    figure.update_layout(
        meta={
            "candidate_parallel": True,
            "rows": [
                {
                    "id": row["id"],
                    "study_name": row["study_name"],
                    "trial_number": row["trial_number"],
                }
                for row in rows
            ],
        }
    )
    if any("Missing" in dimension["ticktext"] for dimension in dimensions):
        notes.append("Missing values use a separate labelled lane; they are not numeric zero.")
    return figure, " ".join(notes)
