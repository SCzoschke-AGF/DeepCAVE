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
"""Two-dimensional empirical trial and elite analysis for Optuna studies."""

from typing import Any, Callable, Dict, List, Sequence, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import (
    FloatHyperparameter,
    Hyperparameter,
    IntegerHyperparameter,
)
from dash import ClientsideFunction, Input, Output, State, ctx, dcc, html, no_update
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import config, interactive
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.plugins.hyperparameter.analysis_utils import (
    _add_trial_statistics,
    _collect_trials,
    _get_objectives_without_time,
    _group_parameter_values,
    _is_optuna_run_or_group,
)
from deepcave.plugins.hyperparameter.search_space_state import (
    PLANNED_SEARCH_SPACE_CONTEXT_TYPE,
    PLANNED_SEARCH_SPACE_STORE_ID,
    as_python_value,
    configspace_metadata,
    objective_persistence_key,
    planning_context_fingerprint,
    planning_metadata_fingerprint,
    position_percent,
    selection_from_state,
    selection_metrics,
    slider_domain,
    transform_value,
)
from deepcave.runs import AbstractRun
from deepcave.utils.analysis_layout import (
    analysis_graph,
    bin_label,
    horizontal_colorbar,
    style_analysis_panel,
)
from deepcave.utils.layout import get_select_options, help_button
from deepcave.utils.styled_plotty import save_image

N_BINS = 4
MIN_BINS = 2
MAX_BINS = 20
ELITE_PROPORTION_DEFAULT = 0.10
COLOR_SCALE = "Viridis"
PANEL_HEIGHT = 240
TRIALS_PANEL_HEIGHT = 210
PANEL_BOTTOM_MARGIN = 94
ENRICHMENT_TOP_MARGIN = 16
ENRICHMENT_BOTTOM_MARGIN = 60
MAX_ANNOTATED_X_BINS = 15
MAX_ANNOTATED_Y_BINS = 6
PLANNING_CONTROLS_ID = "two-dimensional-planning-controls"
PLANNING_CONTROL_1_ID = "two-dimensional-planning-control-1"
PLANNING_CONTROL_2_ID = "two-dimensional-planning-control-2"
PLANNING_ACTIONS_ID = "two-dimensional-planning-actions"
PLANNING_SOURCE = "two-dimensional-analysis"
PLANNING_RANGE_TYPE = "two-dimensional-planning-range"
PLANNING_CATEGORIES_TYPE = "two-dimensional-planning-categories"
PLANNING_RESET_ID = "two-dimensional-planning-reset"
MASK_COLOR = "rgba(100, 116, 139, 0.42)"
CACHE_SCHEMA_VERSION = 2


def _observed_categories(groups: pd.Series) -> List[Any]:
    """Trim unused edge groups while retaining gaps inside the observed range."""
    categories = list(groups.cat.categories)
    used_categories = set(groups.dropna())
    used_indices = [
        index for index, category in enumerate(categories) if category in used_categories
    ]
    if not used_indices:
        return []
    return categories[min(used_indices) : max(used_indices) + 1]


def _group_coordinate(group: Any, hyperparameter: Hyperparameter) -> Any:
    """Return an axis coordinate for an empirical group."""
    if isinstance(group, pd.Interval):
        if getattr(hyperparameter, "log", False):
            # ``pd.cut(..., include_lowest=True)`` moves the displayed left edge
            # slightly below the configured lower bound. For small positive log
            # ranges that edge can become negative. Zero is not a valid Plotly
            # coordinate on a logarithmic contour axis, so clip the display-only
            # interval back to the actual hyperparameter domain.
            lower = max(float(group.left), float(hyperparameter.lower))
            upper = min(float(group.right), float(hyperparameter.upper))
            return float(np.sqrt(lower * upper))
        return float(group.mid)
    if isinstance(hyperparameter, (FloatHyperparameter, IntegerHyperparameter)):
        return float(group)
    return str(group)


def _axis_descriptor(group: Any, hyperparameter: Hyperparameter) -> Dict[str, Any]:
    """Return serializable display geometry for one empirical axis group."""
    descriptor = {"label": str(group), "coordinate": _group_coordinate(group, hyperparameter)}
    if isinstance(group, pd.Interval):
        descriptor.update(
            kind="interval",
            lower=max(float(group.left), float(hyperparameter.lower)),
            upper=min(float(group.right), float(hyperparameter.upper)),
        )
    elif isinstance(hyperparameter, (FloatHyperparameter, IntegerHyperparameter)):
        descriptor.update(kind="discrete", value=as_python_value(group))
    else:
        descriptor.update(kind="category", value=as_python_value(group))
    return descriptor


def _supports_contour(hyperparameter: Hyperparameter) -> bool:
    """Return whether an axis has meaningful numeric interpolation geometry."""
    return isinstance(hyperparameter, (FloatHyperparameter, IntegerHyperparameter))


def _contour_coordinates(
    axes: List[Dict[str, Any]],
    hyperparameter: Hyperparameter,
) -> List[float]:
    """Return valid numeric coordinates, including for caches created before the log fix."""
    coordinates = [float(axis["coordinate"]) for axis in axes]
    if getattr(hyperparameter, "log", False):
        lower = float(hyperparameter.lower)
        coordinates = [max(coordinate, lower) for coordinate in coordinates]
    return coordinates


def _pad_contour_axis(
    coordinates: List[float],
    axes: List[Dict[str, Any]],
    matrix: List[List[Any]],
    customdata: List[List[Any]],
    horizontal: bool,
) -> Tuple[List[float], List[List[Any]], List[List[Any]]]:
    """Extend interval-bin contour values from their centers to their outer edges."""
    if not axes or not all(axis.get("kind") == "interval" for axis in axes):
        return coordinates, matrix, customdata

    padded_coordinates = [float(axes[0]["lower"]), *coordinates, float(axes[-1]["upper"])]
    if horizontal:
        padded_matrix = [[row[0], *row, row[-1]] for row in matrix]
        padded_customdata = [[row[0], *row, row[-1]] for row in customdata]
    else:
        padded_matrix = [list(matrix[0]), *matrix, list(matrix[-1])]
        padded_customdata = [list(customdata[0]), *customdata, list(customdata[-1])]
    return padded_coordinates, padded_matrix, padded_customdata


def _summarize_pair(
    trials: pd.DataFrame,
    hyperparameter_1: Hyperparameter,
    hyperparameter_2: Hyperparameter,
    optimize: str,
    elite_proportion: float,
    n_bins_1: int = N_BINS,
    n_bins_2: int = N_BINS,
) -> Tuple[pd.DataFrame, List[Any], List[Any], float, int, List[Dict[str, Any]]]:
    """Calculate direction-aware trial and joint-bin statistics."""
    trials, elite_threshold, elite_count = _add_trial_statistics(
        trials,
        optimize,
        elite_proportion,
    )
    groups_1 = _group_parameter_values(
        trials["hyperparameter_1"],
        hyperparameter_1,
        n_bins=n_bins_1,
        discrete_numeric=True,
    )
    groups_2 = _group_parameter_values(
        trials["hyperparameter_2"],
        hyperparameter_2,
        n_bins=n_bins_2,
        discrete_numeric=True,
    )
    categories_1 = _observed_categories(groups_1)
    categories_2 = _observed_categories(groups_2)
    grouped_trials = trials.assign(_group_1=groups_1, _group_2=groups_2)

    rows = []
    grouped = grouped_trials.groupby(
        ["_group_1", "_group_2"],
        observed=True,
        sort=True,
    )
    for (group_1, group_2), cell_trials in grouped:
        cell_elite_count = int(cell_trials["is_elite"].sum())
        elite_ranks = cell_trials["elite_rank"].dropna()
        rows.append(
            {
                "group_1": str(group_1),
                "group_2": str(group_2),
                "n": int(len(cell_trials)),
                "elite": cell_elite_count,
                "elite_rate_pct": float(100 * cell_elite_count / len(cell_trials)),
                "elite_enrichment": float(cell_elite_count / len(cell_trials) / elite_proportion),
                "elite_median_rank": (
                    float(elite_ranks.median()) if not elite_ranks.empty else None
                ),
                "objective_median": float(cell_trials["objective"].median()),
            }
        )

    return trials, categories_1, categories_2, elite_threshold, elite_count, rows


def _cell_matrix(
    cells: pd.DataFrame,
    x_labels: List[str],
    y_labels: List[str],
    value: str,
) -> List[List[Any]]:
    """Build a rectangular matrix without filling unobserved joint bins."""
    lookup = {(row["group_1"], row["group_2"]): row[value] for _, row in cells.iterrows()}
    return [[lookup.get((x_label, y_label)) for x_label in x_labels] for y_label in y_labels]


def _style_two_dimensional_panel(figure: go.Figure) -> None:
    """Apply the same compact in-card figure styling to every 2-D panel."""
    style_analysis_panel(
        figure,
        height=PANEL_HEIGHT,
        bottom=PANEL_BOTTOM_MARGIN,
        show_legend=False,
        top=16,
    )


def _observed_trials_figure(
    trials: pd.DataFrame,
    hyperparameter_1: Hyperparameter,
    hyperparameter_2: Hyperparameter,
    hyperparameter_name_1: str,
    hyperparameter_name_2: str,
    objective_name: str,
    optimize: str,
    color_by_rank: bool = False,
) -> go.Figure:
    """Build the unbinned trial scatter colored by objective value or trial rank."""
    marker_color = trials["objective_rank"] if color_by_rank else trials["objective"]
    colorbar_title = "Trial rank" if color_by_rank else objective_name
    figure = go.Figure(
        go.Scattergl(
            x=trials["hyperparameter_1"],
            y=trials["hyperparameter_2"],
            mode="markers",
            marker={
                "color": marker_color,
                "colorscale": COLOR_SCALE,
                "reversescale": color_by_rank or optimize == "lower",
                "showscale": True,
                "colorbar": {
                    "orientation": "v",
                    "title": {"text": colorbar_title, "side": "top", "font": {"size": 11}},
                    "thickness": 10,
                    "tickfont": {"size": 10},
                    "nticks": 6,
                    "lenmode": "pixels",
                    "len": TRIALS_PANEL_HEIGHT - 64,
                    "x": 1.02,
                },
                "size": 6,
                "opacity": 0.7,
            },
            customdata=trials[["objective", "trial_number", "objective_rank", "rank_quantile"]],
            hovertemplate=(
                f"{hyperparameter_name_1}: %{{x}}<br>"
                f"{hyperparameter_name_2}: %{{y}}<br>"
                f"{objective_name}: %{{customdata[0]}}<br>"
                "Trial: %{customdata[1]}<br>Global rank: %{customdata[2]:.0f}<br>"
                "Rank quantile: %{customdata[3]:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    _style_two_dimensional_panel(figure)
    figure.update_layout(height=TRIALS_PANEL_HEIGHT, margin={"b": 48, "r": 100})
    figure.update_xaxes(
        title_text=hyperparameter_name_1,
        type="log" if getattr(hyperparameter_1, "log", False) else None,
    )
    figure.update_yaxes(
        title_text=hyperparameter_name_2,
        type="log" if getattr(hyperparameter_2, "log", False) else None,
    )
    return figure


def _format_enrichment(value: float) -> str:
    """Format an enrichment value compactly while making a true zero unambiguous."""
    if np.isclose(value, 0.0):
        return "0×"
    return f"{value:.2f}×"


def _elite_enrichment_figure(
    cells: pd.DataFrame,
    x_labels: List[str],
    y_labels: List[str],
    hyperparameter_name_1: str,
    hyperparameter_name_2: str,
    elite_threshold: float,
    elite_proportion: float,
) -> go.Figure:
    """Build the joint-bin elite-enrichment heatmap using the existing bins."""
    show_cell_text = len(x_labels) <= MAX_ANNOTATED_X_BINS and len(y_labels) <= MAX_ANNOTATED_Y_BINS
    observed_enrichments = [
        float(cell["elite_rate_pct"]) / 100 / elite_proportion
        for _, cell in cells.iterrows()
        if int(cell["n"]) > 0
    ]
    max_enrichment = max(observed_enrichments, default=1.0)
    color_upper_bound = max(1.0, max_enrichment)
    enrichments: List[List[Any]] = []
    elite_customdata: List[List[List[Any]]] = []
    annotations: List[Dict[str, Any]] = []
    cell_lookup = {(row["group_1"], row["group_2"]): row for _, row in cells.iterrows()}
    for y_index, y_label in enumerate(y_labels):
        enrichment_row: List[Any] = []
        customdata_row: List[List[Any]] = []
        for x_index, x_label in enumerate(x_labels):
            cell = cell_lookup.get((x_label, y_label))
            if cell is None or int(cell["n"]) == 0:
                enrichment_row.append(None)
                customdata_row.append(
                    [x_label, y_label, None, None, 0, elite_proportion * 100, "n/a"]
                )
                if show_cell_text:
                    annotations.extend(
                        [
                            {
                                "x": x_index,
                                "y": y_index,
                                "text": "No data",
                                "showarrow": False,
                                "yshift": 6,
                                "font": {"size": 11, "color": "#64748b", "weight": 600},
                            },
                            {
                                "x": x_index,
                                "y": y_index,
                                "text": "n = 0",
                                "showarrow": False,
                                "yshift": -8,
                                "font": {"size": 9, "color": "#64748b"},
                            },
                        ]
                    )
                continue

            elite_rate_pct = float(cell["elite_rate_pct"])
            enrichment = elite_rate_pct / 100 / elite_proportion
            text_color = "#f8fafc" if enrichment / color_upper_bound < 0.62 else "#172033"
            enrichment_row.append(enrichment)
            customdata_row.append(
                [
                    x_label,
                    y_label,
                    elite_rate_pct,
                    int(cell["elite"]),
                    int(cell["n"]),
                    elite_proportion * 100,
                    (
                        f"{cell['elite_median_rank']:.2f}"
                        if pd.notna(cell["elite_median_rank"])
                        else "n/a"
                    ),
                ]
            )
            if show_cell_text:
                annotations.extend(
                    [
                        {
                            "x": x_index,
                            "y": y_index,
                            "text": _format_enrichment(enrichment),
                            "showarrow": False,
                            "yshift": 6,
                            "font": {"size": 13, "color": text_color, "weight": 600},
                        },
                        {
                            "x": x_index,
                            "y": y_index,
                            "text": f"n = {int(cell['n'])}",
                            "showarrow": False,
                            "yshift": -8,
                            "font": {"size": 9, "color": text_color},
                        },
                    ]
                )
        enrichments.append(enrichment_row)
        elite_customdata.append(customdata_row)

    figure = go.Figure(
        go.Heatmap(
            x=x_labels,
            y=y_labels,
            z=enrichments,
            colorscale=COLOR_SCALE,
            zmin=0,
            zmax=color_upper_bound,
            customdata=elite_customdata,
            hoverongaps=False,
            hovertemplate=(
                f"{hyperparameter_name_1}: %{{customdata[0]}}<br>"
                f"{hyperparameter_name_2}: %{{customdata[1]}}<br>"
                "Elite enrichment: %{z:.2f}×<br>"
                "Elite rate: %{customdata[2]:.1f}%<br>"
                "Elite trials: %{customdata[3]}<br>"
                "Trials in bin: %{customdata[4]}<br>"
                "Expected elite rate: %{customdata[5]:.1f}%<br>"
                "Median elite rank: %{customdata[6]}<br>"
                f"Elite threshold: {elite_threshold:.6g}<extra></extra>"
            ),
            showscale=False,
            xgap=2,
            ygap=2,
        )
    )
    style_analysis_panel(
        figure,
        height=PANEL_HEIGHT,
        bottom=ENRICHMENT_BOTTOM_MARGIN,
        show_legend=False,
        top=ENRICHMENT_TOP_MARGIN,
        left=70,
        right=28,
    )
    figure.update_layout(annotations=annotations, plot_bgcolor="#eef1f5")
    figure.update_xaxes(
        title_text=hyperparameter_name_1,
        type="category",
        tickmode="array",
        tickvals=x_labels,
        ticktext=[bin_label(label, multiline=False) for label in x_labels],
        categoryorder="array",
        categoryarray=x_labels,
        domain=[0.06, 0.94],
        title_standoff=4,
    )
    figure.update_yaxes(
        title_text=hyperparameter_name_2,
        type="category",
        tickmode="array",
        tickvals=y_labels,
        ticktext=[bin_label(label) for label in y_labels],
        categoryorder="array",
        categoryarray=y_labels,
    )
    return figure


def _objective_median_figure(
    cells: pd.DataFrame,
    axes_1: List[Dict[str, Any]],
    axes_2: List[Dict[str, Any]],
    hyperparameter_1: Hyperparameter,
    hyperparameter_2: Hyperparameter,
    hyperparameter_name_1: str,
    hyperparameter_name_2: str,
    objective_name: str,
    optimize: str,
) -> go.Figure:
    """Build the empirical median contour, retaining the categorical heatmap fallback."""
    x_labels = [axis["label"] for axis in axes_1]
    y_labels = [axis["label"] for axis in axes_2]
    objective_medians = _cell_matrix(cells, x_labels, y_labels, "objective_median")
    objective_counts = _cell_matrix(cells, x_labels, y_labels, "n")
    objective_customdata = [
        [
            [x_label, y_label, objective_counts[y_index][x_index]]
            for x_index, x_label in enumerate(x_labels)
        ]
        for y_index, y_label in enumerate(y_labels)
    ]
    trace_arguments = {
        "z": objective_medians,
        "colorscale": COLOR_SCALE,
        "reversescale": optimize == "lower",
        "colorbar": horizontal_colorbar(objective_name),
        "customdata": objective_customdata,
        "hovertemplate": (
            f"{hyperparameter_name_1} bin: %{{customdata[0]}}<br>"
            f"{hyperparameter_name_2} bin: %{{customdata[1]}}<br>"
            f"Median {objective_name}: %{{z}}<br>"
            "Trials: %{customdata[2]}<extra></extra>"
        ),
    }
    use_contour = (
        _supports_contour(hyperparameter_1)
        and _supports_contour(hyperparameter_2)
        and len(x_labels) > 1
        and len(y_labels) > 1
    )
    figure = go.Figure()
    if use_contour:
        x_coordinates = _contour_coordinates(axes_1, hyperparameter_1)
        y_coordinates = _contour_coordinates(axes_2, hyperparameter_2)
        x_tick_coordinates = list(x_coordinates)
        y_tick_coordinates = list(y_coordinates)
        x_coordinates, contour_medians, contour_customdata = _pad_contour_axis(
            x_coordinates,
            axes_1,
            objective_medians,
            objective_customdata,
            horizontal=True,
        )
        y_coordinates, contour_medians, contour_customdata = _pad_contour_axis(
            y_coordinates,
            axes_2,
            contour_medians,
            contour_customdata,
            horizontal=False,
        )
        figure.add_trace(
            go.Contour(
                x=x_coordinates,
                y=y_coordinates,
                connectgaps=False,
                z=contour_medians,
                customdata=contour_customdata,
                **{
                    key: value
                    for key, value in trace_arguments.items()
                    if key not in ("z", "customdata")
                },
            )
        )
        figure.update_xaxes(
            tickmode="array",
            tickvals=x_tick_coordinates,
            ticktext=[bin_label(label, multiline=False) for label in x_labels],
            type="log" if getattr(hyperparameter_1, "log", False) else "linear",
        )
        figure.update_yaxes(
            tickmode="array",
            tickvals=y_tick_coordinates,
            ticktext=[bin_label(label) for label in y_labels],
            type="log" if getattr(hyperparameter_2, "log", False) else "linear",
        )
    else:
        figure.add_trace(
            go.Heatmap(
                x=x_labels,
                y=y_labels,
                hoverongaps=False,
                **trace_arguments,
            )
        )
        figure.update_xaxes(
            type="category",
            tickmode="array",
            tickvals=x_labels,
            ticktext=[bin_label(label, multiline=False) for label in x_labels],
            categoryorder="array",
            categoryarray=x_labels,
        )
        figure.update_yaxes(
            type="category",
            tickmode="array",
            tickvals=y_labels,
            ticktext=[bin_label(label) for label in y_labels],
            categoryorder="array",
            categoryarray=y_labels,
        )

    _style_two_dimensional_panel(figure)
    figure.update_xaxes(title_text=hyperparameter_name_1)
    figure.update_yaxes(title_text=hyperparameter_name_2)
    return figure


def _merge_segments(segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge touching one-dimensional mask segments."""
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(segments):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _complement_segments(
    included: List[Tuple[float, float]], domain: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """Return the complement of merged included segments inside a domain."""
    excluded: List[Tuple[float, float]] = []
    cursor = domain[0]
    for start, end in _merge_segments(included):
        start = max(domain[0], start)
        end = min(domain[1], end)
        if start > cursor:
            excluded.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < domain[1]:
        excluded.append((cursor, domain[1]))
    return excluded


def _continuous_axis_segments(
    parameter: Dict[str, Any], selection: Dict[str, Any]
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """Return the selected portion of a continuous original ConfigSpace axis."""
    domain = (float(parameter["current_min"]), float(parameter["current_max"]))
    start = max(domain[0], float(selection["min"]))
    end = min(domain[1], float(selection["max"]))
    return ([(start, end)] if end > start else []), domain


def _binned_axis_segments(
    axes: List[Dict[str, Any]],
    parameter: Dict[str, Any],
    selection: Dict[str, Any],
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """Map exact selected values to categorical heatmap-cell coordinates."""
    domain = (-0.5, len(axes) - 0.5)
    included: List[Tuple[float, float]] = []
    for index, axis in enumerate(axes):
        if parameter["parameter_type"] in ("integer", "float"):
            if axis.get("kind", "discrete") == "interval":
                lower = float(axis["lower"])
                upper = float(axis["upper"])
                selected_lower = max(lower, float(selection["min"]))
                selected_upper = min(upper, float(selection["max"]))
                if selected_upper <= selected_lower:
                    continue
                transformed_lower = transform_value(lower, bool(parameter["log"]))
                transformed_upper = transform_value(upper, bool(parameter["log"]))
                width = transformed_upper - transformed_lower
                if np.isclose(width, 0):
                    continue
                start_fraction = (
                    transform_value(selected_lower, bool(parameter["log"])) - transformed_lower
                ) / width
                end_fraction = (
                    transform_value(selected_upper, bool(parameter["log"])) - transformed_lower
                ) / width
                included.append((index - 0.5 + start_fraction, index - 0.5 + end_fraction))
            else:
                value = float(axis.get("value", axis["coordinate"]))
                if float(selection["min"]) <= value <= float(selection["max"]):
                    included.append((index - 0.5, index + 0.5))
        elif axis.get("value", axis.get("coordinate")) in selection.get("choices", []):
            included.append((index - 0.5, index + 0.5))
    return _merge_segments(included), domain


def _apply_rectangular_mask(
    figure: go.Figure,
    x_included: List[Tuple[float, float]],
    x_domain: Tuple[float, float],
    y_included: List[Tuple[float, float]],
    y_domain: Tuple[float, float],
) -> None:
    """Shade the union of excluded X/Y regions once, without darker intersections."""
    shapes: List[Dict[str, Any]] = []
    base = {
        "type": "rect",
        "xref": "x",
        "yref": "y",
        "fillcolor": MASK_COLOR,
        "line": {"width": 0},
        "layer": "above",
    }
    for x0, x1 in _complement_segments(x_included, x_domain):
        shapes.append({**base, "x0": x0, "x1": x1, "y0": y_domain[0], "y1": y_domain[1]})
    for y0, y1 in _complement_segments(y_included, y_domain):
        for x0, x1 in x_included:
            shapes.append({**base, "x0": x0, "x1": x1, "y0": y0, "y1": y1})
    figure.update_layout(shapes=shapes)


def _apply_continuous_mask(
    figure: go.Figure,
    metadata: List[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
) -> None:
    """Apply an exact data-coordinate mask to a scatter or numeric contour."""
    x_included, x_domain = _continuous_axis_segments(metadata[0], selection[metadata[0]["name"]])
    y_included, y_domain = _continuous_axis_segments(metadata[1], selection[metadata[1]["name"]])
    _apply_rectangular_mask(figure, x_included, x_domain, y_included, y_domain)


def _apply_scatter_mask(
    figure: go.Figure,
    axes_1: List[Dict[str, Any]],
    axes_2: List[Dict[str, Any]],
    metadata: List[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
) -> None:
    """Mask numeric scatter axes exactly and categorical axes by category."""
    axis_geometry = []
    for axes, parameter in zip((axes_1, axes_2), metadata):
        selected = selection[parameter["name"]]
        if parameter["parameter_type"] in ("integer", "float"):
            axis_geometry.append(_continuous_axis_segments(parameter, selected))
        else:
            axis_geometry.append(_binned_axis_segments(axes, parameter, selected))
    _apply_rectangular_mask(
        figure,
        axis_geometry[0][0],
        axis_geometry[0][1],
        axis_geometry[1][0],
        axis_geometry[1][1],
    )


def _apply_binned_mask(
    figure: go.Figure,
    axes_1: List[Dict[str, Any]],
    axes_2: List[Dict[str, Any]],
    metadata: List[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
) -> None:
    """Apply exact fractional-bin or discrete category masking to a heatmap."""
    x_included, x_domain = _binned_axis_segments(
        axes_1, metadata[0], selection[metadata[0]["name"]]
    )
    y_included, y_domain = _binned_axis_segments(
        axes_2, metadata[1], selection[metadata[1]["name"]]
    )
    _apply_rectangular_mask(figure, x_included, x_domain, y_included, y_domain)


def _range_overlay_components(
    metrics: Dict[str, Any], domain: Tuple[float, float]
) -> Tuple[List[Component], List[Component]]:
    """Return orange extension and gray unselected slider overlays."""
    extensions: List[Component] = []
    for start, end in (
        (metrics["min_normalized"], min(metrics["max_normalized"], 0.0)),
        (max(metrics["min_normalized"], 1.0), metrics["max_normalized"]),
    ):
        if end > start:
            left = position_percent(start, domain)
            right = position_percent(end, domain)
            extensions.append(
                html.Span(
                    className="planner-extension-segment",
                    style={"left": f"{left:.6f}%", "width": f"{right - left:.6f}%"},
                )
            )
    minimum = min(100.0, max(0.0, position_percent(metrics["min_normalized"], domain)))
    maximum = min(100.0, max(0.0, position_percent(metrics["max_normalized"], domain)))
    unselected = [
        html.Span(
            className="planner-unselected-segment",
            style={"left": "0", "width": f"{minimum:.6f}%"},
        ),
        html.Span(
            className="planner-unselected-segment",
            style={"left": f"{maximum:.6f}%", "width": f"{100 - maximum:.6f}%"},
        ),
    ]
    return extensions, unselected


def _planning_control(
    parameter: Dict[str, Any], selection: Dict[str, Any], axis_index: int
) -> Component:
    """Build the reduced planner control for one selected 2-D feature."""
    label = "Search range"
    planning_context = parameter.get("planning_context", "")
    if parameter["parameter_type"] not in ("integer", "float"):
        return html.Div(
            [
                dbc.Label(label, className="two-dimensional-range-label"),
                dcc.Dropdown(
                    id={
                        "type": PLANNING_CATEGORIES_TYPE,
                        "index": parameter["index"],
                        "context": planning_context,
                    },
                    options=[
                        {"label": str(value), "value": value} for value in parameter["choices"]
                    ],
                    value=selection["choices"],
                    multi=True,
                    clearable=True,
                ),
            ],
            className="two-dimensional-planning-control",
        )

    metrics = selection_metrics(parameter, selection)
    domain = slider_domain(metrics)
    extensions, unselected = _range_overlay_components(metrics, domain)
    return html.Div(
        [
            html.Div(
                [
                    dbc.Label(label, className="two-dimensional-range-label"),
                    html.Span(
                        f"{selection['min']:.6g} – {selection['max']:.6g}",
                        className="two-dimensional-range-value",
                    ),
                ],
                className="two-dimensional-range-heading",
            ),
            html.Div(
                [
                    dcc.RangeSlider(
                        id={
                            "type": PLANNING_RANGE_TYPE,
                            "index": parameter["index"],
                            "context": planning_context,
                        },
                        min=domain[0],
                        max=domain[1],
                        step=0.001,
                        value=[metrics["min_normalized"], metrics["max_normalized"]],
                        marks=None,
                        allowCross=False,
                        updatemode="mouseup",
                        tooltip={
                            "placement": "bottom",
                            "always_visible": False,
                            "transform": f"hpoPlanning.parameter{axis_index}",
                        },
                        className="planner-range-slider",
                    ),
                    html.Div(extensions, className="planner-extension-layer"),
                    html.Div(unselected, className="planner-unselected-layer"),
                ],
                className="planner-range-control two-dimensional-planning-range-control",
                **{
                    "data-current-min": str(parameter["current_min"]),
                    "data-current-max": str(parameter["current_max"]),
                    "data-log": str(parameter["log"]).lower(),
                    "data-step": "" if parameter.get("step") is None else str(parameter["step"]),
                    "data-parameter-type": parameter["parameter_type"],
                },
            ),
        ],
        className="two-dimensional-planning-control",
    )


def _selected_planning_metadata(
    configspace: Any, names: Sequence[str], objective: Any
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Return selected parameters with run-stable identities and context."""
    complete_metadata = configspace_metadata(configspace, list(configspace.keys()))
    metadata_version = planning_metadata_fingerprint(complete_metadata)
    planning_context = planning_context_fingerprint(metadata_version, objective)
    by_name = {parameter["name"]: parameter for parameter in complete_metadata}
    selected = [dict(by_name[name]) for name in names]
    for parameter in selected:
        parameter["planning_context"] = planning_context
    return selected, metadata_version, planning_context


def _planning_control_is_mounted(children: Any) -> bool:
    """Return whether a planning mount already contains rendered children."""
    return children not in (None, [])


class TwoDimensionalAnalysis(DynamicPlugin):
    """Visualize empirical two-hyperparameter behavior for an Optuna study."""

    id = "two_dimensional_analysis"
    name = "2-D Analyses"
    icon = "fas fa-th"
    help = "plugins/two_dimensional_analysis.html"
    activate_run_selection = True
    main_objective_input = "objective_id"
    # Planning masks are updated by a small browser callback. The shared store is only sampled
    # when a new analytical base is produced, so a range edit cannot call ``process`` or rebuild
    # the three figures on the server.
    output_view_states = (("planned_search_space_state", PLANNED_SEARCH_SPACE_STORE_ID, "data"),)

    def _dict_as_key(self, data: Dict[str, Any], remove_filters: bool = False) -> str:
        """Version cached outputs because axis descriptors now contain mask geometry."""
        key = super()._dict_as_key(data, remove_filters=remove_filters)
        return f"{key}-v{CACHE_SCHEMA_VERSION}"

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """Accept suitable Optuna studies and Optuna-only groups."""
        return _is_optuna_run_or_group(run) and not run.empty() and len(run.configspace) >= 2

    @staticmethod
    def get_input_layout(register: Callable) -> List[Component]:
        """Group feature selection, search range and binning by axis."""
        # The rank switch lives alongside the observed-trials plot.
        register("color_by_rank", "value", filter=True, type=bool)
        feature_cards = []
        for index, (axis, mount) in enumerate(
            (("X", PLANNING_CONTROL_1_ID), ("Y", PLANNING_CONTROL_2_ID)), start=1
        ):
            feature_id = register(f"hyperparameter_name_{index}", ["value", "options"])
            bin_id = register(f"n_bins_{index}", "value", type=int)
            feature_cards.append(
                html.Div(
                    [
                        dbc.Label(
                            f"Hyperparameter {axis}",
                            html_for=feature_id,
                            className="two-dimensional-control-title",
                        ),
                        dbc.Select(
                            id=feature_id,
                            placeholder="Select hyperparameter ...",
                            size="sm",
                            className="two-dimensional-feature-select",
                        ),
                        html.Div(
                            [
                                dbc.Label(
                                    "Bins",
                                    html_for=bin_id,
                                    className="two-dimensional-control-title",
                                ),
                                html.Div(
                                    [
                                        html.Button(
                                            "−",
                                            id=f"{bin_id}-decrement",
                                            type="button",
                                            className="two-dimensional-bin-button",
                                            **{"aria-label": f"Decrease bins for {axis}"},
                                        ),
                                        dbc.Input(
                                            id=bin_id,
                                            type="number",
                                            min=MIN_BINS,
                                            max=MAX_BINS,
                                            step=1,
                                            value=N_BINS,
                                            className="two-dimensional-bin-input",
                                        ),
                                        html.Button(
                                            "+",
                                            id=f"{bin_id}-increment",
                                            type="button",
                                            className="two-dimensional-bin-button",
                                            **{"aria-label": f"Increase bins for {axis}"},
                                        ),
                                    ],
                                    className="two-dimensional-bin-stepper",
                                ),
                            ],
                            className="two-dimensional-binning-control",
                        ),
                        html.Div(id=mount, className="two-dimensional-range-mount"),
                    ],
                    className="two-dimensional-feature-card",
                )
            )
        elite_id = register("elite_proportion", "value", type=float)
        return [
            html.Div(
                [
                    *feature_cards,
                    html.Div(
                        [
                            html.Div(
                                [
                                    dbc.Label(
                                        "Elite",
                                        html_for=elite_id,
                                        className="two-dimensional-control-title",
                                    ),
                                    help_button(
                                        "Fraction of the best trials considered elite. "
                                        "For example, 0.10 selects the best 10%."
                                    ),
                                ],
                                className="two-dimensional-elite-heading",
                            ),
                            dbc.Input(
                                id=elite_id,
                                type="number",
                                min=0.01,
                                max=0.99,
                                step=0.01,
                                size="sm",
                            ),
                            html.Div(id=PLANNING_ACTIONS_ID),
                        ],
                        className="two-dimensional-elite-card",
                    ),
                    dcc.Store(
                        id={"type": PLANNED_SEARCH_SPACE_CONTEXT_TYPE, "source": PLANNING_SOURCE},
                        data=None,
                    ),
                ],
                id=PLANNING_CONTROLS_ID,
                className="two-dimensional-controls",
            )
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Any]:
        """Return no additional display filters."""
        return []

    @interactive
    def register_callbacks(self) -> None:
        """Synchronize the reduced controls with the shared planning state."""
        super().register_callbacks()
        from deepcave import app, run_handler

        for index in (1, 2):
            bin_id = self.get_internal_input_id(f"n_bins_{index}")
            app.clientside_callback(  # type: ignore
                """function(decrease, increase, value, minimum, maximum) {
                    const id = dash_clientside.callback_context.triggered_id;
                    if (!id) return dash_clientside.no_update;
                    const step = id.endsWith("-increment") ? 1 : -1;
                    const current = Number.isFinite(value) ? value : minimum;
                    return Math.max(minimum, Math.min(maximum, current + step));
                }""",
                Output(bin_id, "value"),
                [
                    Input(f"{bin_id}-decrement", "n_clicks"),
                    Input(f"{bin_id}-increment", "n_clicks"),
                ],
                [State(bin_id, "value"), State(bin_id, "min"), State(bin_id, "max")],
                prevent_initial_call=True,
            )

        @app.callback(  # type: ignore
            [
                Output(PLANNING_CONTROL_1_ID, "children"),
                Output(PLANNING_CONTROL_2_ID, "children"),
                Output(PLANNING_ACTIONS_ID, "children"),
                Output(
                    {"type": PLANNED_SEARCH_SPACE_CONTEXT_TYPE, "source": PLANNING_SOURCE},
                    "data",
                ),
            ],
            [
                Input(self.get_internal_input_id("hyperparameter_name_1"), "value"),
                Input(self.get_internal_input_id("hyperparameter_name_2"), "value"),
                Input(self.get_internal_input_id("objective_id"), "value"),
                Input("global-run-selector", "value"),
            ],
            [
                State(PLANNED_SEARCH_SPACE_STORE_ID, "data"),
                State(PLANNING_CONTROL_1_ID, "children"),
                State(PLANNING_CONTROL_2_ID, "children"),
                State(PLANNING_ACTIONS_ID, "children"),
            ],
        )
        def render_planning_controls(
            hyperparameter_name_1: str,
            hyperparameter_name_2: str,
            objective: Any,
            run_id: str,
            planning_state: Any,
            current_control_1: Any,
            current_control_2: Any,
            current_actions: Any,
        ) -> Any:
            if (
                not run_id
                or not hyperparameter_name_1
                or not hyperparameter_name_2
                or objective is None
            ):
                return [], [], [], None
            try:
                run = run_handler.get_run(run_id)
                names = [hyperparameter_name_1, hyperparameter_name_2]
                metadata, metadata_version, planning_context = _selected_planning_metadata(
                    run.configspace, names, objective
                )
            except (KeyError, RuntimeError):
                return [], [], [], None
            selection = selection_from_state(planning_state, run_id, objective, metadata)
            controls = [
                _planning_control(parameter, selection[parameter["name"]], index)
                for index, parameter in enumerate(metadata)
            ]
            actions = html.Div(
                dbc.Button(
                    [html.I(className="fas fa-undo me-2"), "Reset ranges"],
                    id={"type": PLANNING_RESET_ID, "context": planning_context},
                    color="primary",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                ),
                className="two-dimensional-planning-actions",
            )

            control_1: Any = controls[0]
            control_2: Any = controls[1]
            actions_output: Any = actions
            if ctx.triggered_id == self.get_internal_input_id(
                "hyperparameter_name_2"
            ) and _planning_control_is_mounted(current_control_1):
                control_1 = no_update
            elif ctx.triggered_id == self.get_internal_input_id(
                "hyperparameter_name_1"
            ) and _planning_control_is_mounted(current_control_2):
                control_2 = no_update
            if _planning_control_is_mounted(current_actions) and ctx.triggered_id in {
                self.get_internal_input_id("hyperparameter_name_1"),
                self.get_internal_input_id("hyperparameter_name_2"),
            }:
                actions_output = no_update

            return (
                control_1,
                control_2,
                actions_output,
                {
                    "source": PLANNING_SOURCE,
                    "run_id": run_id,
                    "objective_key": objective_persistence_key(objective),
                    "planning_context": planning_context,
                    "metadata_version": metadata_version,
                    "metadata": metadata,
                },
            )

        app.clientside_callback(  # type: ignore
            ClientsideFunction(
                namespace="twoDimensionalPlanning",
                function_name="applyMountedMasks",
            ),
            Output("two-dimensional-planning-mask-revision", "data"),
            Input(PLANNED_SEARCH_SPACE_STORE_ID, "data"),
            State("two-dimensional-planning-mask-revision", "data"),
            prevent_initial_call=True,
        )

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Return defaults before an Optuna study is selected."""
        return {
            "objective_id": {"options": [], "value": None},
            "hyperparameter_name_1": {"options": [], "value": None},
            "hyperparameter_name_2": {"options": [], "value": None},
            "elite_proportion": {"value": ELITE_PROPORTION_DEFAULT},
            "n_bins_1": {"value": N_BINS},
            "n_bins_2": {"value": N_BINS},
            "color_by_rank": {"value": False},
        }

    def load_dependency_inputs(
        self,
        run: AbstractRun,
        previous_inputs: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Populate study-dependent choices and enforce distinct parameters."""
        objectives = _get_objectives_without_time(run)
        objective_ids = [run.get_objective_id(objective) for objective in objectives]
        objective_value = inputs["objective_id"]["value"]
        if objective_value not in objective_ids:
            objective_value = objective_ids[0]

        hyperparameter_names = list(run.configspace.keys())
        hyperparameter_1 = inputs["hyperparameter_name_1"]["value"]
        if hyperparameter_1 not in hyperparameter_names:
            hyperparameter_1 = hyperparameter_names[0]

        hyperparameter_2_names = [name for name in hyperparameter_names if name != hyperparameter_1]
        hyperparameter_2 = inputs["hyperparameter_name_2"]["value"]
        if hyperparameter_2 not in hyperparameter_2_names:
            hyperparameter_2 = hyperparameter_2_names[0]

        return {
            "objective_id": {
                "options": get_select_options(
                    [objective.name for objective in objectives],
                    objective_ids,
                ),
                "value": objective_value,
            },
            "hyperparameter_name_1": {
                "options": get_select_options(hyperparameter_names),
                "value": hyperparameter_1,
            },
            "hyperparameter_name_2": {
                "options": get_select_options(hyperparameter_2_names),
                "value": hyperparameter_2,
            },
            "elite_proportion": {"value": inputs["elite_proportion"]["value"]},
            "n_bins_1": {
                "value": inputs.get("n_bins_1", {}).get("value") or N_BINS,
            },
            "n_bins_2": {
                "value": inputs.get("n_bins_2", {}).get("value") or N_BINS,
            },
            "color_by_rank": {
                "value": bool(inputs.get("color_by_rank", {}).get("value", False)),
            },
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Collect trials and calculate empirical joint-bin summaries."""
        objective = run.get_objective(inputs["objective_id"])
        if objective is None:
            raise PreventUpdate

        hyperparameter_name_1 = inputs["hyperparameter_name_1"]
        hyperparameter_name_2 = inputs["hyperparameter_name_2"]
        if hyperparameter_name_1 == hyperparameter_name_2:
            raise ValueError("Select two different hyperparameters.")

        elite_proportion = float(inputs["elite_proportion"])
        if not 0.01 <= elite_proportion <= 0.99:
            raise ValueError("Elite trial proportion must be between 0.01 and 0.99.")

        n_bins_1 = int(inputs.get("n_bins_1", N_BINS))
        n_bins_2 = int(inputs.get("n_bins_2", N_BINS))
        if not MIN_BINS <= n_bins_1 <= MAX_BINS:
            raise ValueError(f"Feature #1 bin count must be between {MIN_BINS} and {MAX_BINS}.")
        if not MIN_BINS <= n_bins_2 <= MAX_BINS:
            raise ValueError(f"Feature #2 bin count must be between {MIN_BINS} and {MAX_BINS}.")

        trials = _collect_trials(
            run,
            inputs["objective_id"],
            [hyperparameter_name_1, hyperparameter_name_2],
        )
        if trials.empty:
            raise ValueError("No successful trials are available for this analysis.")
        trials = trials.rename(
            columns={
                hyperparameter_name_1: "hyperparameter_1",
                hyperparameter_name_2: "hyperparameter_2",
            }
        ).sort_values("trial_number")

        (
            trials,
            categories_1,
            categories_2,
            elite_threshold,
            elite_count,
            cells,
        ) = _summarize_pair(
            trials,
            run.configspace[hyperparameter_name_1],
            run.configspace[hyperparameter_name_2],
            objective.optimize,
            elite_proportion,
            n_bins_1,
            n_bins_2,
        )
        trial_columns = [
            "trial_number",
            "hyperparameter_1",
            "hyperparameter_2",
            "objective",
            "objective_rank",
            "rank_quantile",
        ]

        return {
            "trials": trials[trial_columns].to_dict(orient="records"),
            "cells": cells,
            "axis_1": [
                _axis_descriptor(group, run.configspace[hyperparameter_name_1])
                for group in categories_1
            ],
            "axis_2": [
                _axis_descriptor(group, run.configspace[hyperparameter_name_2])
                for group in categories_2
            ],
            "elite_threshold": elite_threshold,
            "elite_count": elite_count,
        }

    @staticmethod
    def get_output_layout(register: Callable) -> List[Any]:
        """Show trials across the full width, followed by the two binned views."""
        trials_graph = analysis_graph(
            register("trials_graph", "figure"),
            TRIALS_PANEL_HEIGHT,
            config.FIGURE_DOWNLOAD_SCALE,
        )
        elite_rate_graph = analysis_graph(
            register("elite_rate_graph", "figure"),
            PANEL_HEIGHT,
            config.FIGURE_DOWNLOAD_SCALE,
        )
        objective_graph = analysis_graph(
            register("objective_graph", "figure"),
            PANEL_HEIGHT,
            config.FIGURE_DOWNLOAD_SCALE,
        )

        def card(title: str, graph: Component, trials: bool = False) -> Component:
            header = [
                html.Div(
                    [
                        html.H2(title, className="two-dimensional-card-title"),
                    ],
                    className="two-dimensional-card-heading",
                )
            ]
            if trials:
                header.append(
                    dbc.Switch(
                        id=f"{TwoDimensionalAnalysis.id}-color_by_rank-input",
                        label="Color by trial rank",
                        value=False,
                        className="two-dimensional-color-control",
                    )
                )
            return html.Div(
                [html.Div(header, className="two-dimensional-card-header"), graph],
                className="analysis-card" + (" two-dimensional-trials-card" if trials else ""),
            )

        return [
            html.Div(
                [
                    card(
                        "Observed trials",
                        trials_graph,
                        trials=True,
                    ),
                    card(
                        "Elite enrichment (1x = Baseline)",
                        elite_rate_graph,
                    ),
                    card(
                        "Empirical objective median contour",
                        objective_graph,
                    ),
                ],
                className="analysis-grid analysis-grid--two-dimensional",
            )
        ]

    @staticmethod
    def load_outputs(
        run: AbstractRun,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> List[go.Figure]:
        """Build separate empirical scatter, elite-enrichment, and objective figures."""
        objective = run.get_objective(inputs["objective_id"])
        if objective is None:
            raise PreventUpdate

        hyperparameter_name_1 = inputs["hyperparameter_name_1"]
        hyperparameter_name_2 = inputs["hyperparameter_name_2"]
        hyperparameter_1 = run.configspace[hyperparameter_name_1]
        hyperparameter_2 = run.configspace[hyperparameter_name_2]
        trials = pd.DataFrame(outputs["trials"]).sort_values("rank_quantile")
        cells = pd.DataFrame(outputs["cells"])
        x_labels = [axis["label"] for axis in outputs["axis_1"]]
        y_labels = [axis["label"] for axis in outputs["axis_2"]]

        trials_figure = _observed_trials_figure(
            trials,
            hyperparameter_1,
            hyperparameter_2,
            hyperparameter_name_1,
            hyperparameter_name_2,
            objective.name,
            objective.optimize,
            bool(inputs.get("color_by_rank", False)),
        )
        elite_enrichment_figure = _elite_enrichment_figure(
            cells,
            x_labels,
            y_labels,
            hyperparameter_name_1,
            hyperparameter_name_2,
            outputs["elite_threshold"],
            float(inputs["elite_proportion"]),
        )
        objective_figure = _objective_median_figure(
            cells,
            outputs["axis_1"],
            outputs["axis_2"],
            hyperparameter_1,
            hyperparameter_2,
            hyperparameter_name_1,
            hyperparameter_name_2,
            objective.name,
            objective.optimize,
        )

        metadata = configspace_metadata(
            run.configspace,
            [hyperparameter_name_1, hyperparameter_name_2],
        )
        metadata_version = planning_metadata_fingerprint(metadata)
        mask_context = {
            "run_id": getattr(run, "id", ""),
            "objective_key": objective_persistence_key(inputs["objective_id"]),
            "metadata_version": metadata_version,
            "metadata": metadata,
            "axis_1": outputs["axis_1"],
            "axis_2": outputs["axis_2"],
        }
        objective_mask_kind = (
            "continuous"
            if _supports_contour(hyperparameter_1)
            and _supports_contour(hyperparameter_2)
            and len(outputs["axis_1"]) > 1
            and len(outputs["axis_2"]) > 1
            else "binned"
        )
        analysis_revision = (
            f"{getattr(run, 'id', '')}:{inputs['objective_id']}:"
            f"{hyperparameter_name_1}:{hyperparameter_name_2}"
        )
        for figure, mask_kind, panel in (
            (trials_figure, "scatter", "trials"),
            (elite_enrichment_figure, "binned", "enrichment"),
            (objective_figure, objective_mask_kind, "objective"),
        ):
            figure.update_layout(
                meta={"deepcave_planning_mask": {**mask_context, "kind": mask_kind}},
                uirevision=f"{analysis_revision}:{panel}",
            )
        selection = selection_from_state(
            inputs.get("planned_search_space_state"),
            getattr(run, "id", ""),
            inputs["objective_id"],
            metadata,
        )
        _apply_scatter_mask(
            trials_figure,
            outputs["axis_1"],
            outputs["axis_2"],
            metadata,
            selection,
        )
        _apply_binned_mask(
            elite_enrichment_figure,
            outputs["axis_1"],
            outputs["axis_2"],
            metadata,
            selection,
        )
        if objective_mask_kind == "continuous":
            _apply_continuous_mask(objective_figure, metadata, selection)
        else:
            _apply_binned_mask(
                objective_figure,
                outputs["axis_1"],
                outputs["axis_2"],
                metadata,
                selection,
            )

        save_image(trials_figure, "two_dimensional_trials.pdf")
        save_image(elite_enrichment_figure, "two_dimensional_elite_enrichment.pdf")
        save_image(objective_figure, "two_dimensional_objective.pdf")
        return [trials_figure, elite_enrichment_figure, objective_figure]
