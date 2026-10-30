# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# noqa: D400
"""One-dimensional trial and elite-distribution analysis for Optuna studies."""

from typing import Any, Callable, Dict, List, Tuple

from collections import OrderedDict
from threading import RLock

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import Hyperparameter, NumericalHyperparameter
from dash import ctx, dcc, html
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import config
from deepcave.plugins.cost_efficiency import importance_bar_figure
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.plugins.hyperparameter.analysis_utils import (
    _add_trial_statistics,
    _collect_trials,
    _get_objectives_without_time,
    _group_parameter_values,
    _is_optuna_run_or_group,
)
from deepcave.runs import AbstractRun
from deepcave.runs.group import Group
from deepcave.utils.analysis_layout import (
    add_colorbar_label,
    analysis_graph,
    analysis_toolbar_item,
    bin_label,
    bin_scale_exponent,
    horizontal_colorbar,
    style_analysis,
    superscript_exponent,
)
from deepcave.utils.cost_efficiency import calculate_importances
from deepcave.utils.layout import get_select_options
from deepcave.utils.performance_cost import extract_trial_data
from deepcave.utils.styled_plotty import get_color, save_image

ELITE_PROPORTION_DEFAULT = 0.10
TAIL_COLOR = "#F58518"
MEDIAN_COLOR = "#54A24B"
BEST_COLOR = "#B279A2"
PERFORMANCE_SERIES = (
    ("tail", "objective_tail", "q90", TAIL_COLOR),
    ("median", "objective_median", "Median", MEDIAN_COLOR),
    ("best", "objective_max", "Best", BEST_COLOR),
)
DEPLETED_COLOR = "#C9CFD8"
BASELINE_COLOR = "#D62728"
TOP_TRIAL_COLOR = "#27364A"

# Positions in paper coordinates inside the enrichment panel's existing bottom margin. The
# line runs through the marker centers and visually defines the compact rug above the colorbar.
TOP_TRIAL_RUG_Y = -0.22
TOP_TRIAL_RUG_LINE_Y = TOP_TRIAL_RUG_Y
ENRICHMENT_X_TITLE_Y = -0.415

ENRICHMENT_HEIGHT = 314
ENRICHMENT_BOTTOM_MARGIN = 128
PANEL_LEFT_MARGIN = 75
TOP_MARGIN = 44
TRIAL_FIGURE_CACHE_SIZE = 8

_TRIAL_FIGURE_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
_TRIAL_FIGURE_CACHE_LOCK = RLock()
_IMPORTANCE_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()


def clear_one_dimensional_figure_cache() -> None:
    """Clear reusable in-process raw-trial figures."""
    with _TRIAL_FIGURE_CACHE_LOCK:
        _TRIAL_FIGURE_CACHE.clear()
        _IMPORTANCE_CACHE.clear()


def _objective_importances(run: AbstractRun, objective_id: int) -> Dict[str, Any]:
    """Reuse computational-cost importance samples across parameter and elite selections."""
    key = _trial_figure_cache_key(run, objective_id, "")
    with _TRIAL_FIGURE_CACHE_LOCK:
        if key is not None and key in _IMPORTANCE_CACHE:
            _IMPORTANCE_CACHE.move_to_end(key)
            return _IMPORTANCE_CACHE[key]
        rows = extract_trial_data(run, objective_id)
        sources = run.runs if isinstance(run, Group) else [run]
        for row in rows:
            row["configuration"] = dict(sources[row["source_index"]].get_config(row["config_id"]))
        result = calculate_importances(run, rows, objective_id=objective_id)
        if key is not None:
            _IMPORTANCE_CACHE[key] = result
            while len(_IMPORTANCE_CACHE) > TRIAL_FIGURE_CACHE_SIZE:
                _IMPORTANCE_CACHE.popitem(last=False)
        return result


def _trial_figure_cache_key(
    run: AbstractRun,
    objective_id: Any,
    hyperparameter_name: str,
) -> Any:
    """Return a data-aware key, or ``None`` for incomplete test doubles."""
    try:
        return (
            run.id,
            id(run),
            run.latest_change,
            len(run.history),
            objective_id,
            hyperparameter_name,
        )
    except (AttributeError, RuntimeError):
        return None


def _cached_trials_figure(
    run: AbstractRun,
    objective_id: Any,
    hyperparameter_name: str,
    build: Callable[[], go.Figure],
) -> go.Figure:
    """Reuse immutable serialized scatter bases across elite-only updates."""
    key = _trial_figure_cache_key(run, objective_id, hyperparameter_name)
    if key is None:
        return build()
    with _TRIAL_FIGURE_CACHE_LOCK:
        cached = _TRIAL_FIGURE_CACHE.get(key)
        if cached is None:
            figure = build()
            figure.update_layout(
                uirevision=f"one-dimensional:{run.id}:{objective_id}:{hyperparameter_name}"
            )
            cached = figure.to_dict()
            _TRIAL_FIGURE_CACHE[key] = cached
            stale = [
                candidate
                for candidate in _TRIAL_FIGURE_CACHE
                if candidate[0] == run.id and candidate[1:4] != key[1:4]
            ]
            for candidate in stale:
                _TRIAL_FIGURE_CACHE.pop(candidate, None)
            while len(_TRIAL_FIGURE_CACHE) > TRIAL_FIGURE_CACHE_SIZE:
                _TRIAL_FIGURE_CACHE.popitem(last=False)
        else:
            _TRIAL_FIGURE_CACHE.move_to_end(key)
        return go.Figure(cached)


def _summarize_trials(
    trials: pd.DataFrame,
    hyperparameter: Hyperparameter,
    optimize: str,
    elite_proportion: float,
) -> Tuple[List[Dict[str, Any]], float, float, int]:
    """Summarize objective location and elite enrichment for each observed bin."""
    tail_quantile = 0.90 if optimize == "upper" else 0.10
    trials, elite_threshold, elite_count = _add_trial_statistics(
        trials,
        optimize,
        elite_proportion,
    )
    grouped_trials = trials.assign(
        _group=_group_parameter_values(trials["hyperparameter_value"], hyperparameter),
    )

    # The realized elite share can deviate from the requested proportion (ties, few trials),
    # so enrichment is measured against what the study actually contains.
    elite_share = elite_count / len(trials) if len(trials) else 0.0

    summary = []
    for group, group_trials in grouped_trials.groupby("_group", observed=True, sort=True):
        group_elite_count = int(group_trials["is_elite"].sum())
        group_elite_rate = group_elite_count / len(group_trials)
        elite_ranks = group_trials["elite_rank"].dropna()
        median_rank = float(elite_ranks.median()) if not elite_ranks.empty else None
        summary.append(
            {
                "range": str(group),
                "n": int(len(group_trials)),
                "elite": group_elite_count,
                "elite_rate_pct": float(100 * group_elite_rate),
                "elite_enrichment": float(group_elite_rate / elite_share) if elite_share else 0.0,
                "elite_median_rank": median_rank,
                "objective_median": float(group_trials["objective"].median()),
                "objective_tail": float(group_trials["objective"].quantile(tail_quantile)),
                "objective_max": float(group_trials["objective"].max()),
            }
        )

    return summary, tail_quantile, elite_threshold, elite_count


def _style_panel(
    figure: go.Figure,
    title: str,
    height: int,
    bottom: int,
    show_legend: bool = False,
) -> None:
    """Apply the established 1-D card styling and fixed plot alignment."""
    style_analysis(figure, height=height, show_legend=show_legend)
    figure.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.02,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 15},
        },
        margin={"t": TOP_MARGIN, "b": bottom, "l": PANEL_LEFT_MARGIN, "r": 22},
        legend={
            "orientation": "h",
            "x": 1,
            "xanchor": "right",
            "y": 1.01,
            "yanchor": "bottom",
            "font": {"size": 12},
        },
    )
    figure.update_yaxes(automargin=False)
    figure.update_xaxes(automargin=False)


def _range_axis(figure: go.Figure, ranges: List[str], hyperparameter_name: str) -> str:
    """Give a range panel the shared bins, order and tick labels."""
    scale_exponent = bin_scale_exponent(ranges)
    axis_title = hyperparameter_name
    if scale_exponent is not None:
        axis_title += f" (×10{superscript_exponent(scale_exponent)})"

    figure.update_xaxes(
        title_text=axis_title,
        tickmode="array",
        tickvals=ranges,
        ticktext=[
            bin_label(value, multiline=False, scale_exponent=scale_exponent) for value in ranges
        ],
        tickangle=0,
        categoryorder="array",
        categoryarray=ranges,
    )
    return axis_title


def _trials_figure(
    trials: pd.DataFrame,
    hyperparameter: Hyperparameter,
    hyperparameter_name: str,
    objective_name: str,
    color: str,
) -> go.Figure:
    """Plot every observed trial as a coverage and raw-data check."""
    figure = go.Figure(
        go.Scatter(
            x=(
                trials["hyperparameter_value"]
                if isinstance(hyperparameter, NumericalHyperparameter)
                else trials["hyperparameter_value"].map(str)
            ),
            y=trials["objective"],
            mode="markers",
            name="Trials",
            showlegend=False,
            marker={
                "color": color,
                "size": 7,
                "opacity": 0.6,
            },
            customdata=trials[["trial_number"]],
            hovertemplate=(
                f"{hyperparameter_name}: %{{x}}<br>"
                f"{objective_name}: %{{y}}<br>Trial: %{{customdata[0]}}<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title_text=hyperparameter_name)
    if getattr(hyperparameter, "log", False):
        figure.update_xaxes(type="log")
    figure.update_yaxes(title_text=objective_name)
    style_analysis(figure, show_legend=False)
    return figure


def _add_performance_summary(
    figure: go.Figure,
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    hyperparameter: Hyperparameter,
    tail_percent: int,
) -> None:
    """Connect bin summaries at their centers on the original parameter axis."""
    groups = _group_parameter_values(trials["hyperparameter_value"], hyperparameter)
    categories = groups.cat.categories
    intervals = isinstance(categories, pd.IntervalIndex)
    numeric = isinstance(hyperparameter, NumericalHyperparameter)
    if intervals:
        spacing = np.geomspace if hyperparameter.log else np.linspace
        edges = spacing(hyperparameter.lower, hyperparameter.upper, len(categories) + 1)
    elif not numeric:
        figure.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=[str(value) for value in categories],
        )
    coordinates = {str(value): index for index, value in enumerate(categories)}
    for key, field, label, color in PERFORMANCE_SERIES:
        if key == "tail":
            label = f"q{tail_percent}"
        x, y, customdata = [], [], []
        for row in summary.to_dict(orient="records"):
            index = coordinates[row["range"]]
            details = [
                row["range"],
                row["n"],
                row["objective_median"],
                row["objective_tail"],
                row["objective_max"],
            ]
            if intervals:
                left, right = edges[index], edges[index + 1]
                center = np.sqrt(left * right) if hyperparameter.log else (left + right) / 2
                x.append(float(center))
            else:
                value = categories[index]
                x.append(float(value) if numeric else str(value))
            y.append(row[field])
            customdata.append(details)
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=label,
                legendgroup=key,
                showlegend=True,
                visible="legendonly",
                line={"color": color, "width": 3},
                marker={"color": color, "size": 10, "symbol": "circle"},
                connectgaps=False,
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]}<br>Trials: %{customdata[1]}"
                    "<br>Median: %{customdata[2]:.4g}"
                    f"<br>q{tail_percent}: %{{customdata[3]:.4g}}"
                    f"<br>Best (max): %{{customdata[4]:.4g}}<extra>{label}</extra>"
                ),
            )
        )
    figure.update_layout(
        showlegend=True,
        title_text=None,
        height=None,
        autosize=True,
        margin_b=96,
        legend={
            "title": {"text": "Show lines:", "side": "left"},
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 0.02,
            "yref": "container",
            "yanchor": "bottom",
            "groupclick": "togglegroup",
            "itemclick": "toggle",
            "itemdoubleclick": False,
            "tracegroupgap": 0,
        },
    )


def _add_bin_means(figure: go.Figure, summary: pd.DataFrame) -> None:
    """Group each series with its equally weighted bin mean across the full plot width."""
    figure.update_layout(
        xaxis2={"overlaying": "x", "range": [0, 1], "visible": False, "fixedrange": True}
    )
    for index, (key, field, _label, color) in enumerate(PERFORMANCE_SERIES, start=1):
        trace = figure.data[index]
        values = summary[field].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            mean = float(values.mean())
            figure.add_trace(
                go.Scatter(
                    x=[0, 1],
                    xaxis="x2",
                    y=[mean, mean],
                    mode="lines",
                    line={"color": color, "width": 1.5, "dash": "dash"},
                    opacity=0.6,
                    name=f"{trace.name} bin mean",
                    legendgroup=key,
                    showlegend=False,
                    visible="legendonly",
                    hovertemplate=f"{trace.name} bin mean: %{{y:.4g}}<extra></extra>",
                )
            )


def _enrichment_figure(
    summary: pd.DataFrame,
    trials: pd.DataFrame,
    hyperparameter: Hyperparameter,
    hyperparameter_name: str,
    objective_name: str,
    optimize: str,
    elite_percent: float,
    rank_maximum: int,
) -> go.Figure:
    """Plot where elite trials accumulate, relative to the study-wide elite share."""
    ranges = summary["range"].tolist()
    title = f"Elite enrichment (best {elite_percent:.0f}%)"
    figure = go.Figure()

    hover = (
        "%{customdata[0]}<br>Trials: %{customdata[1]}<br>"
        "Elite trials: %{customdata[2]} (%{customdata[3]:.1f}%)<br>"
        "Enrichment: %{customdata[4]:.2f}×<br>"
        "Median elite rank: %{customdata[5]}<extra></extra>"
    )
    with_elite = summary[summary["elite_median_rank"].notna()]
    without_elite = summary[summary["elite_median_rank"].isna()]
    figure.add_trace(
        go.Bar(
            x=with_elite["range"],
            y=with_elite["elite_enrichment"],
            marker={
                "color": with_elite["elite_median_rank"],
                "colorscale": "Blues",
                "reversescale": True,
                "cmin": 1,
                "cmax": rank_maximum,
                "showscale": True,
                "colorbar": horizontal_colorbar(),
            },
            customdata=_enrichment_customdata(with_elite),
            hovertemplate=hover,
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Bar(
            x=without_elite["range"],
            y=without_elite["elite_enrichment"],
            marker={"color": DEPLETED_COLOR},
            customdata=_enrichment_customdata(without_elite),
            hovertemplate=hover,
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ranges,
            y=[1.0] * len(ranges),
            mode="lines",
            name="1.0× baseline",
            line={"color": BASELINE_COLOR, "dash": "dash", "width": 1.5},
            hoverinfo="skip",
        )
    )
    axis_title = _range_axis(figure, ranges, hyperparameter_name)
    figure.update_yaxes(
        title_text="Enrichment",
        ticksuffix="×",
        range=[0, max(1.35, 1.15 * float(summary["elite_enrichment"].max()))],
    )
    figure.update_layout(barmode="overlay", bargap=0.2)
    _style_panel(
        figure,
        title,
        ENRICHMENT_HEIGHT,
        bottom=ENRICHMENT_BOTTOM_MARGIN,
        show_legend=True,
    )
    # Plotly may pull a native axis title back into the rug when horizontal space is tight.
    # A paper-anchored title keeps its distance from the markers at every responsive width.
    figure.update_xaxes(title_text=None)
    figure.add_annotation(
        name="enrichment_x_title",
        text=axis_title,
        xref="paper",
        yref="paper",
        x=0.5,
        y=ENRICHMENT_X_TITLE_Y,
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font={"size": 13},
    )
    _add_top_trials_rug(
        figure,
        trials,
        hyperparameter,
        summary["range"].tolist(),
        hyperparameter_name,
        objective_name,
        optimize,
    )
    add_colorbar_label(
        figure,
        "Median elite rank (1 = best)",
        ENRICHMENT_HEIGHT,
        ENRICHMENT_BOTTOM_MARGIN,
        top=TOP_MARGIN,
    )
    return figure


def _add_top_trials_rug(
    figure: go.Figure,
    trials: pd.DataFrame,
    hyperparameter: Hyperparameter,
    ranges: List[str],
    hyperparameter_name: str,
    objective_name: str,
    optimize: str,
) -> None:
    """Add the five best trials below the x-axis without changing the binned plot."""
    top_trials = (
        trials.sort_values(
            ["objective", "trial_number"],
            ascending=[optimize == "lower", True],
            kind="mergesort",
        )
        .head(5)
        .copy()
    )
    if top_trials.empty:
        return

    top_trials["top_rank"] = range(1, len(top_trials) + 1)
    groups = _group_parameter_values(trials["hyperparameter_value"], hyperparameter)
    is_binned = bool(len(groups.cat.categories)) and all(
        isinstance(category, pd.Interval) for category in groups.cat.categories
    )

    if is_binned:
        displayed_indices = {value: index for index, value in enumerate(ranges)}
        bin_indices = [displayed_indices[str(groups.loc[index])] for index in top_trials.index]
    else:
        values = top_trials["hyperparameter_value"].map(str).tolist()
        category_positions = {value: index for index, value in enumerate(ranges)}
        bin_indices = [category_positions[value] for value in values]

    normalized_x = _center_rug_positions_in_bins(bin_indices, len(ranges))

    for x_value, (_, trial) in zip(normalized_x, top_trials.iterrows()):
        rank = int(trial["top_rank"])
        hovertext = (
            f"Rank {rank}<br>Trial: {int(trial['trial_number'])}<br>"
            f"{hyperparameter_name}: {trial['hyperparameter_value']}<br>"
            f"{objective_name}: {trial['objective']:.4g}"
        )
        annotation = {
            "x": x_value,
            "xref": "paper",
            "y": TOP_TRIAL_RUG_Y,
            "yref": "paper",
            "xanchor": "center",
            "yanchor": "middle",
            "showarrow": False,
            "hovertext": hovertext,
            "captureevents": True,
        }
        figure.add_annotation(
            **annotation,
            text="◆",
            font={"color": TOP_TRIAL_COLOR, "size": 17},
            name=f"top_trial_marker_{rank}",
        )
        figure.add_annotation(
            **annotation,
            text=str(rank),
            font={"color": "#FFFFFF", "size": 8},
            name=f"top_trial_rank_{rank}",
        )
    figure.add_annotation(
        text="Top 5 trials",
        xref="paper",
        yref="paper",
        x=0,
        xshift=-5,
        xanchor="right",
        y=TOP_TRIAL_RUG_Y,
        yanchor="middle",
        showarrow=False,
        font={"size": 9, "color": TOP_TRIAL_COLOR},
    )
    figure.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        x1=1,
        y0=TOP_TRIAL_RUG_LINE_Y,
        y1=TOP_TRIAL_RUG_LINE_Y,
        line={"color": "#D9DEE7", "width": 1},
    )


def _center_rug_positions_in_bins(bin_indices: List[int], bin_count: int) -> List[float]:
    """Center each bin's markers under its bar and distribute them side by side."""
    positions = [0.0] * len(bin_indices)
    bin_width = 1 / bin_count
    minimum_gap = min(0.025, 0.2 * bin_width)

    for bin_index in set(bin_indices):
        members = [index for index, value in enumerate(bin_indices) if value == bin_index]
        center = (bin_index + 0.5) * bin_width
        start = center - minimum_gap * (len(members) - 1) / 2
        for offset, member in enumerate(members):
            positions[member] = start + offset * minimum_gap

    return positions


def _enrichment_customdata(rows: pd.DataFrame) -> List[List[Any]]:
    """Bundle the bin facts for the hover, including the optional median elite rank."""
    return [
        [
            row["range"],
            row["n"],
            row["elite"],
            row["elite_rate_pct"],
            row["elite_enrichment"],
            "n/a" if pd.isna(row["elite_median_rank"]) else f"{row['elite_median_rank']:.1f}",
        ]
        for _, row in rows.iterrows()
    ]


class OneDimensionalAnalysis(DynamicPlugin):
    """Visualize one-dimensional objective and elite behavior for an Optuna study."""

    id = "one_dimensional_analysis"
    processing_version = 2
    name = "1-D Analyses"
    icon = "fas fa-chart-bar"
    help = "plugins/one_dimensional_analysis.html"
    activate_run_selection = True
    main_objective_input = "objective_id"
    input_update_inputs = (
        ("importance_click", "one_dimensional_analysis-importance_graph-output", "clickData"),
        (
            "importance_label_click",
            "one_dimensional_analysis-importance_graph-output",
            "clickAnnotationData",
        ),
    )

    def load_input_updates(self, events: Dict[str, Any]) -> Dict[str, Any]:
        """Select the parameter clicked in an importance bar or its label."""
        for name, component_id, property_name in self.input_update_inputs:
            if f"{component_id}.{property_name}" not in ctx.triggered_prop_ids:
                continue
            event = events.get(name) or {}
            if name == "importance_label_click":
                parameter = event.get("annotation", {}).get("name")
            else:
                points = event.get("points") or []
                parameter = points[0].get("customdata") if points else None
            return (
                {"hyperparameter_name": {"value": str(parameter)}} if parameter is not None else {}
            )
        return {}

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """Accept non-empty Optuna studies and Optuna-only groups."""
        return _is_optuna_run_or_group(run) and not run.empty() and len(run.configspace) > 0

    @staticmethod
    def get_input_layout(register: Callable) -> List[Component]:
        """
        Return the complete control toolbar as a single row.

        The globally selected run is shown in the sidebar, so only analysis-specific controls
        are rendered here.
        """
        return [
            html.Div(
                [
                    analysis_toolbar_item(
                        "Hyperparameter",
                        dbc.Select(
                            id=register("hyperparameter_name", ["value", "options"]),
                            placeholder="Select hyperparameter ...",
                        ),
                        width="18rem",
                    ),
                    analysis_toolbar_item(
                        "Elite",
                        dbc.Input(
                            id=register("elite_proportion", "value", type=float),
                            type="number",
                            min=0.01,
                            max=0.99,
                            step=0.01,
                        ),
                        width="9rem",
                        help_text=(
                            "Fraction of the best trials considered elite. For example, "
                            "0.10 selects the best 10%."
                        ),
                    ),
                ],
                className="analysis-toolbar d-flex flex-wrap align-items-center gap-2",
            )
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Any]:
        """Return no additional display filters."""
        return []

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Return default values before a study is selected."""
        return {
            "objective_id": {"options": [], "value": None},
            "hyperparameter_name": {"options": [], "value": None},
            "elite_proportion": {"value": ELITE_PROPORTION_DEFAULT},
        }

    def load_dependency_inputs(
        self,
        run: AbstractRun,
        _previous_inputs: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Populate study-dependent choices."""
        objectives = _get_objectives_without_time(run)
        objective_ids = [run.get_objective_id(objective) for objective in objectives]
        objective_value = inputs["objective_id"]["value"]
        if objective_value not in objective_ids:
            objective_value = objective_ids[0]

        hyperparameter_names = list(run.configspace.keys())
        hyperparameter_value = inputs["hyperparameter_name"]["value"]
        if hyperparameter_value not in hyperparameter_names:
            hyperparameter_value = hyperparameter_names[0]

        return {
            "objective_id": {
                "options": get_select_options(
                    [objective.name for objective in objectives],
                    objective_ids,
                ),
                "value": objective_value,
            },
            "hyperparameter_name": {
                "options": get_select_options(hyperparameter_names),
                "value": hyperparameter_value,
            },
            "elite_proportion": {"value": inputs["elite_proportion"]["value"]},
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Collect trials and calculate bin-level summaries."""
        objective = run.get_objective(inputs["objective_id"])
        if objective is None:
            raise PreventUpdate

        hyperparameter_name = inputs["hyperparameter_name"]
        hyperparameter = run.configspace[hyperparameter_name]
        elite_proportion = float(inputs["elite_proportion"])
        if not 0.01 <= elite_proportion <= 0.99:
            raise ValueError("Elite trial proportion must be between 0.01 and 0.99.")

        trials = _collect_trials(run, inputs["objective_id"], [hyperparameter_name])
        if trials.empty:
            raise ValueError("No successful trials are available for this analysis.")

        trials = trials.rename(columns={hyperparameter_name: "hyperparameter_value"})
        trials = trials.sort_values("trial_number").reset_index(drop=True)
        summary, tail_quantile, elite_threshold, elite_count = _summarize_trials(
            trials,
            hyperparameter,
            objective.optimize,
            elite_proportion,
        )
        return {
            "importances": _objective_importances(run, inputs["objective_id"]),
            "trials": trials.to_dict(orient="records"),
            "summary": summary,
            "tail_quantile": tail_quantile,
            "elite_threshold": elite_threshold,
            "elite_count": elite_count,
        }

    @staticmethod
    def get_output_layout(register: Callable) -> List[Any]:
        """Put importance and combined performance above elite enrichment."""
        return [
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            id=register("importance_graph", "figure"),
                            responsive=True,
                            className="one-dimensional-primary-graph",
                            style={"width": "100%"},
                            config={
                                "toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}
                            },
                        ),
                        className="analysis-card",
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id=register("trials_graph", "figure"),
                                responsive=True,
                                className="one-dimensional-primary-graph",
                                style={"width": "100%"},
                                config={
                                    "toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}
                                },
                            ),
                        ],
                        className="analysis-card",
                        style={"alignSelf": "start"},
                    ),
                    html.Div(
                        analysis_graph(
                            register("enrichment_graph", "figure"),
                            ENRICHMENT_HEIGHT,
                            config.FIGURE_DOWNLOAD_SCALE,
                        ),
                        className="analysis-card",
                        style={"gridColumn": "1 / -1"},
                    ),
                ],
                className="analysis-grid analysis-grid--one-dimensional",
            )
        ]

    @staticmethod
    def load_outputs(
        run: AbstractRun,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> List[Any]:
        """Build selectable importance, raw trials and coupled range analyses."""
        objective = run.get_objective(inputs["objective_id"])
        if objective is None:
            raise PreventUpdate

        hyperparameter_name = inputs["hyperparameter_name"]
        hyperparameter = run.configspace[hyperparameter_name]
        trials = pd.DataFrame(outputs["trials"])
        summary = pd.DataFrame(outputs["summary"])
        tail_percent = int(round(100 * outputs["tail_quantile"]))
        elite_percent = 100 * float(inputs["elite_proportion"])

        trials_figure = _cached_trials_figure(
            run,
            inputs["objective_id"],
            hyperparameter_name,
            lambda: _trials_figure(
                trials,
                hyperparameter,
                hyperparameter_name,
                objective.name,
                get_color(list(run.configspace.keys()).index(hyperparameter_name)),
            ),
        )
        _add_performance_summary(
            trials_figure,
            trials,
            summary,
            hyperparameter,
            tail_percent,
        )
        _add_bin_means(trials_figure, summary)
        enrichment_figure = _enrichment_figure(
            summary,
            trials,
            hyperparameter,
            hyperparameter_name,
            objective.name,
            objective.optimize,
            elite_percent,
            max(2, int(outputs["elite_count"])),
        )

        save_image(trials_figure, "one_dimensional_trials.pdf")
        save_image(enrichment_figure, "one_dimensional_enrichment.pdf")

        importance = outputs["importances"]
        importance_figure = importance_bar_figure(
            run,
            importance["objective"],
            "Objective Importance",
            importance["errors"].get("objective"),
        )
        importance_figure.update_layout(height=None, margin_b=trials_figure.layout.margin.b)
        return [importance_figure, trials_figure, enrichment_figure]
