# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Observed near-optimal configuration and Rashomon-set analysis for Optuna studies."""

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import copy
import json
from collections import Counter, OrderedDict
from threading import RLock

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant
from dash import ALL, Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_samples

from deepcave import config, interactive
from deepcave.evaluators.fanova import fANOVA
from deepcave.evaluators.rashomon import (
    TOP_PERCENT_PRESETS,
    cached_pairwise_configuration_distances,
    cluster_configurations,
    diversity_summary,
    parameter_spreads,
    random_subset_baseline,
    select_near_optimal_indices,
    subset_condensed_distances,
    subset_distance_matrix,
)
from deepcave.plugins.hyperparameter.analysis_utils import (
    _get_trial_table,
    get_normalized_trials,
)
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import AbstractRun
from deepcave.runs.status import Status
from deepcave.utils.analysis_layout import analysis_graph, style_analysis_panel
from deepcave.utils.configspace import normalized_configuration_matrix
from deepcave.utils.hash import string_to_hash
from deepcave.utils.layout import get_select_options, help_button

ANALYSIS_DATA_ID = "near-optimal-analysis-data"
HEATMAP_ID = "near-optimal-heatmap"
DIVERSITY_ID = "near-optimal-diversity"
HEATMAP_ORDER_ID = "near-optimal-heatmap-order"
IMPORTANCE_ID = "near-optimal-importance"
CONSTRAINTS_ID = "near-optimal-constraints"
CLUSTER_DETAIL_ID = "near-optimal-cluster-detail"
CLUSTER_COORDINATES_ID = "near-optimal-cluster-coordinates"
CLUSTER_COORDINATES_MODE_ID = "near-optimal-cluster-coordinates-mode"
PARAMETER_STORE_ID = "near-optimal-parameter"
TRIAL_TABLE_ID = "near-optimal-trials"
TRIAL_SEARCH_ID = "near-optimal-trial-search"
TOP_CRITERION_PANEL_ID = "near-optimal-top-criterion-panel"
ABSOLUTE_CRITERION_PANEL_ID = "near-optimal-absolute-criterion-panel"
RELATIVE_CRITERION_PANEL_ID = "near-optimal-relative-criterion-panel"
SELECTION_COUNT_ID = "near-optimal-selection-count"
SELECTION_SHARE_ID = "near-optimal-selection-share"
DIVERSITY_PERCENT_PRESETS = (*TOP_PERCENT_PRESETS, 50, 75, 100)
SELECTION_RANGE_ID = "near-optimal-selection-range"

HEATMAP_LIMIT = 100
PRIMARY_GRAPH_HEIGHT = 396
DIVERSITY_GRAPH_HEIGHT = 280
CLUSTER_COLORS = ("#4169FF", "#439653", "#F2AA16", "#CC3450")
FANOVA_CACHE_SIZE = 8
ANALYSIS_CACHE_SIZE = 32
HEATMAP_DISPLAY_DECIMALS = 6

_FANOVA_CACHE: "OrderedDict[Tuple[str, str, int], Dict[str, Any]]" = OrderedDict()
_FANOVA_CACHE_LOCK = RLock()
_ANALYSIS_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_ANALYSIS_CACHE_LOCK = RLock()


def _run_version(run: AbstractRun) -> str:
    """Return a content-aware run version suitable for bounded in-process caches."""
    try:
        latest_change = run.latest_change
    except (AttributeError, RuntimeError):
        latest_change = 0
    return f"{id(run)}:{latest_change}:{len(run.history)}"


def clear_near_optimal_importance_cache() -> None:
    """Clear cached fANOVA main effects."""
    with _FANOVA_CACHE_LOCK:
        _FANOVA_CACHE.clear()


def clear_near_optimal_analysis_cache() -> None:
    """Clear browser-token-addressable analysis results."""
    with _ANALYSIS_CACHE_LOCK:
        _ANALYSIS_CACHE.clear()


def _cache_analysis_data(data: Dict[str, Any]) -> str:
    """Cache analysis data server-side and return a compact stable browser token."""
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    token = string_to_hash(serialized)
    with _ANALYSIS_CACHE_LOCK:
        _ANALYSIS_CACHE[token] = data
        _ANALYSIS_CACHE.move_to_end(token)
        while len(_ANALYSIS_CACHE) > ANALYSIS_CACHE_SIZE:
            _ANALYSIS_CACHE.popitem(last=False)
    return token


def _get_analysis_data(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Resolve a browser token to its server-side analysis data."""
    if not token:
        return None
    with _ANALYSIS_CACHE_LOCK:
        data = _ANALYSIS_CACHE.get(token)
        if data is not None:
            _ANALYSIS_CACHE.move_to_end(token)
        return data


def _fanova_importances(run: AbstractRun, objective_id: int) -> Dict[str, Any]:
    """Calculate or retrieve global fANOVA main effects without failing the core analysis."""
    key = (run.id, _run_version(run), objective_id)
    with _FANOVA_CACHE_LOCK:
        cached = _FANOVA_CACHE.get(key)
        if cached is not None:
            _FANOVA_CACHE.move_to_end(key)
            return copy.deepcopy(cached)

    objective = run.get_objective(objective_id)
    names = [name for name, hp in run.configspace.items() if not isinstance(hp, Constant)]
    result: Dict[str, Any]
    try:
        evaluator = fANOVA(run)
        evaluator.calculate(
            objective,
            run.get_highest_budget(),
            n_trees=16,
            seed=0,
            statuses=Status.SUCCESS,
        )
        raw = evaluator.get_importances(names)
        importances = {
            name: max(0.0, float(raw[name][0]))
            for name in names
            if name in raw and np.isfinite(raw[name][0])
        }
        if not importances or sum(importances.values()) <= 0:
            raise RuntimeError("fANOVA returned no positive finite main effects.")
        result = {"values": importances, "error": None}
    except Exception as error:
        result = {"values": {}, "error": str(error)}

    with _FANOVA_CACHE_LOCK:
        _FANOVA_CACHE[key] = result
        while len(_FANOVA_CACHE) > FANOVA_CACHE_SIZE:
            _FANOVA_CACHE.popitem(last=False)
    return copy.deepcopy(result)


def _objective_records(run: AbstractRun, objective_id: int) -> Dict[str, Any]:
    """Join successful finite objective values to the shared normalized trial cache."""
    normalized = get_normalized_trials(run, normalized_configuration_matrix)
    table = _get_trial_table(run, objective_id)
    objective_rows = {int(row["trial_number"]): row for _, row in table.iterrows()}
    records = []
    matrix = []
    for record, vector in zip(normalized["records"], normalized["matrix"]):
        objective_row = objective_rows.get(int(record["trial_id"]))
        if objective_row is None:
            continue
        joined = copy.deepcopy(record)
        joined["objective"] = float(objective_row["objective"])
        joined["objective_rank"] = float(objective_row["objective_rank"])
        records.append(joined)
        matrix.append(vector)
    order = np.argsort([record["trial_id"] for record in records])
    return {
        "records": [records[index] for index in order],
        "matrix": np.asarray([matrix[index] for index in order], dtype=float),
        "hyperparameters": normalized["hyperparameters"],
    }


def _criterion_label(criterion: str, value: float) -> str:
    if criterion == "top_percent":
        return f"Top {value:g}%"
    if criterion == "absolute":
        return f"Within absolute Δ {value:g}"
    return f"Within relative Δ {value:g}%"


def _selection_preview(
    run: AbstractRun,
    objective_id: int,
    criterion: str,
    criterion_value: float,
) -> Dict[str, Any]:
    """Build the inexpensive selection summary shown above the full analysis."""
    objective = run.get_objective(objective_id)
    if objective is None:
        raise ValueError("The selected objective is not available.")

    values = []
    trial_ids = []
    for history_index, trial in enumerate(run.history):
        value = trial.costs[objective_id]
        if (
            trial.status != Status.SUCCESS
            or value is None
            or not np.isfinite(value)
        ):
            continue
        values.append(float(value))
        trial_ids.append(int(trial.additional.get("optuna_trial_number", history_index)))

    if not values:
        return {"selected": 0, "eligible": 0, "share": 0.0, "minimum": None, "maximum": None}

    selected_indices = select_near_optimal_indices(
        values,
        trial_ids,
        objective.optimize,
        criterion,
        criterion_value,
    )
    selected_values = np.asarray([values[index] for index in selected_indices], dtype=float)
    return {
        "selected": len(selected_indices),
        "eligible": len(values),
        "share": len(selected_indices) / len(values),
        "minimum": float(np.min(selected_values)),
        "maximum": float(np.max(selected_values)),
    }


def _profile_value(values: Sequence[Any], categorical: bool) -> str:
    finite = [value for value in values if value is not None]
    if not finite:
        return "inactive"
    if categorical:
        mode, count = Counter(finite).most_common(1)[0]
        return f"{mode} ({count / len(finite):.0%})"
    numbers = np.asarray(finite, dtype=float)
    q25, median, q75 = np.quantile(numbers, [0.25, 0.5, 0.75])
    return f"{median:.4g} [{q25:.4g} – {q75:.4g}]"


def _cluster_profiles(
    records: List[Dict[str, Any]],
    labels: Sequence[int],
    configspace: Any,
    parameters: Sequence[str],
) -> List[Dict[str, Any]]:
    profiles = []
    label_values = np.asarray(labels, dtype=int)
    for cluster_id in sorted(set(label_values.tolist())):
        indices = np.flatnonzero(label_values == cluster_id)
        cluster_records = [records[index] for index in indices]
        objectives = np.asarray([record["objective"] for record in cluster_records])
        parameter_rows = []
        for name in parameters:
            values = [record["configuration"].get(name) for record in cluster_records]
            active = [value for value in values if value is not None]
            hp = configspace[name]
            parameter_rows.append(
                {
                    "parameter": name,
                    "summary": _profile_value(
                        values,
                        isinstance(hp, CategoricalHyperparameter),
                    ),
                    "active": len(active) / len(values) if values else 0.0,
                }
            )
        profiles.append(
            {
                "cluster": int(cluster_id),
                "size": len(cluster_records),
                "objective_median": float(np.median(objectives)),
                "objective_min": float(np.min(objectives)),
                "objective_max": float(np.max(objectives)),
                "parameters": parameter_rows,
            }
        )
    return profiles


def _profile_parameter_order(
    records: List[Dict[str, Any]],
    labels: Sequence[int],
    configspace: Any,
    parameters: Sequence[str],
    importance: Mapping[str, float],
) -> Tuple[List[str], Dict[str, float]]:
    """Order parameters by robust separation between their cluster profiles."""
    label_values = np.asarray(labels, dtype=int)
    cluster_ids = sorted(set(label_values.tolist()))
    scores = {}
    for parameter in parameters:
        categorical = isinstance(configspace[parameter], CategoricalHyperparameter)
        summaries = []
        for cluster_id in cluster_ids:
            indices = np.flatnonzero(label_values == cluster_id)
            values = [
                records[index]["configuration"].get(parameter) for index in indices
            ]
            active = [value for value in values if value is not None]
            if not active:
                summaries.append(None)
            elif categorical:
                summaries.append(Counter(active).most_common(1)[0][0])
            else:
                q25, median, q75 = np.quantile(
                    np.asarray(active, dtype=float), [0.25, 0.5, 0.75]
                )
                summaries.append((float(median), float(q75 - q25)))

        pair_scores = []
        for first_index, first in enumerate(summaries[:-1]):
            for second in summaries[first_index + 1 :]:
                if first is None or second is None:
                    pair_scores.append(float(first is not second))
                elif categorical:
                    pair_scores.append(float(first != second))
                else:
                    median_gap = abs(first[0] - second[0])
                    mean_iqr = (first[1] + second[1]) / 2
                    denominator = median_gap + mean_iqr
                    pair_scores.append(0.0 if denominator == 0 else median_gap / denominator)
        scores[parameter] = float(np.mean(pair_scores)) if pair_scores else 0.0

    ordered = sorted(
        parameters,
        key=lambda name: (scores[name], float(importance.get(name, 0.0)), name),
    )
    return ordered, scores


def _distribution_payload(
    eligible: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    labels: Sequence[int],
    parameters: Sequence[str],
) -> Dict[str, Any]:
    payload = {}
    for name in parameters:
        payload[name] = {
            "overall": [record["configuration"].get(name) for record in eligible],
            "near": [record["configuration"].get(name) for record in selected],
            "clusters": {
                str(cluster): [
                    record["configuration"].get(name)
                    for record, label in zip(selected, labels)
                    if label == cluster
                ]
                for cluster in sorted(set(labels))
            },
        }
    return payload


def build_near_optimal_analysis(
    run: AbstractRun,
    objective_id: int,
    criterion: str,
    criterion_value: float,
    distance_metric: str,
    cluster_count: Optional[int],
    parameter_names: Sequence[str],
) -> Dict[str, Any]:
    """Build the complete serializable analysis payload for one Optuna study."""
    objective = run.get_objective(objective_id)
    if objective is None:
        raise ValueError("The selected objective is not available.")
    trial_data = _objective_records(run, objective_id)
    all_records = trial_data["records"]
    all_matrix = trial_data["matrix"]
    if not all_records:
        raise ValueError("No successful finite trials are available for this objective.")

    records = all_records
    matrix = all_matrix
    objectives = [record["objective"] for record in records]
    trial_ids = [record["trial_id"] for record in records]

    selected_indices = select_near_optimal_indices(
        objectives, trial_ids, objective.optimize, criterion, criterion_value
    )
    selected_records = [records[index] for index in selected_indices]
    importance = _fanova_importances(run, objective_id)
    selected_parameters = [name for name in parameter_names if name in run.configspace]
    if not selected_parameters:
        selected_parameters = list(run.configspace.keys())
    weights: Optional[Mapping[str, float]] = None
    metric_used = distance_metric
    metric_warning = None
    if distance_metric == "importance_weighted":
        selected_weight = sum(
            float(importance["values"].get(name, 0.0)) for name in selected_parameters
        )
        if importance["values"] and selected_weight > 0:
            weights = importance["values"]
        else:
            metric_used = "normalized"
            metric_warning = (
                "Importance-weighted distance was unavailable; normalized distance was used. "
                + str(importance["error"] or "Selected parameters have zero total importance.")
            )

    weight_signature = tuple(
        sorted((name, round(float(value), 12)) for name, value in (weights or {}).items())
    )
    cache_key = (
        run.id,
        _run_version(run),
        tuple(trial_ids),
        metric_used,
        tuple(selected_parameters),
        weight_signature,
    )
    all_distances = cached_pairwise_configuration_distances(
        cache_key,
        matrix,
        run.configspace,
        selected_parameters,
        weights,
    )
    selected_distances = subset_condensed_distances(all_distances, len(records), selected_indices)
    diversity = diversity_summary(selected_distances)
    baseline = random_subset_baseline(
        all_distances,
        len(records),
        len(selected_indices),
        repeats=200,
        max_pairs=100_000,
        seed=0,
    )

    diversity_curve = []
    for percent in DIVERSITY_PERCENT_PRESETS:
        indices = select_near_optimal_indices(
            objectives, trial_ids, objective.optimize, "top_percent", percent
        )
        curve_summary = diversity_summary(
            subset_condensed_distances(all_distances, len(records), indices)
        )
        diversity_curve.append({"percent": percent, "count": len(indices), **curve_summary})

    clusters = cluster_configurations(
        selected_distances,
        len(selected_indices),
        requested_clusters=cluster_count,
    )
    labels = clusters["labels"]
    for record, label in zip(selected_records, labels):
        record["cluster"] = int(label)

    order = np.asarray(clusters["order"], dtype=int)
    if order.size > HEATMAP_LIMIT:
        positions = np.linspace(0, order.size - 1, HEATMAP_LIMIT, dtype=int)
        heatmap_indices = order[positions]
    else:
        heatmap_indices = order
    heatmap_matrix = subset_distance_matrix(
        selected_distances, len(selected_indices), heatmap_indices
    )
    heatmap_trials = [selected_records[index]["trial_id"] for index in heatmap_indices]
    heatmap_clusters = [int(labels[index]) for index in heatmap_indices]
    if len(heatmap_indices) > 1 and 1 < len(set(heatmap_clusters)) < len(heatmap_clusters):
        heatmap_silhouette = silhouette_samples(
            heatmap_matrix, heatmap_clusters, metric="precomputed"
        )
    else:
        heatmap_silhouette = np.zeros(len(heatmap_indices), dtype=float)
    heatmap_linkage: List[List[float]] = []
    if len(heatmap_indices) > 1:
        heatmap_condensed = subset_condensed_distances(
            selected_distances, len(selected_indices), heatmap_indices
        )
        heatmap_linkage = np.round(
            linkage(heatmap_condensed, method="average"), HEATMAP_DISPLAY_DECIMALS
        ).tolist()

    spreads = parameter_spreads(matrix, selected_indices, run.configspace, selected_parameters)
    profile_parameters, parameter_distinctness = _profile_parameter_order(
        selected_records,
        labels,
        run.configspace,
        selected_parameters,
        importance["values"],
    )
    profiles = _cluster_profiles(selected_records, labels, run.configspace, profile_parameters)
    distributions = _distribution_payload(records, selected_records, labels, selected_parameters)

    selected_objectives = np.asarray([record["objective"] for record in selected_records])
    best = min(objectives) if objective.optimize == "lower" else max(objectives)
    sampler = str(getattr(run, "meta", {}).get("optuna_sampler", "Unknown sampler"))
    if sampler == "TPESampler":
        sampler_note = (
            "TPESampler adaptively concentrates sampling. The observed near-optimal diversity "
            "therefore reflects both the objective landscape and the explored region."
        )
    elif sampler == "RandomSampler":
        sampler_note = (
            "RandomSampler provides broader direct coverage, but this analysis still describes "
            "only observed trials."
        )
    else:
        sampler_note = (
            f"Sampler: {sampler}. Results describe observed trials only and may reflect the "
            "optimizer's sampling strategy."
        )

    table_rows = []
    for record in selected_records:
        row = {
            "id": int(record["trial_id"]),
            "trial_id": int(record["trial_id"]),
            "objective": float(record["objective"]),
            "cluster": int(record["cluster"]),
        }
        row.update(
            {f"hp::{name}": record["configuration"].get(name) for name in selected_parameters}
        )
        table_rows.append(row)

    return {
        "run_id": run.id,
        "objective_name": objective.name,
        "optimize": objective.optimize,
        "criterion": criterion,
        "criterion_value": float(criterion_value),
        "criterion_label": _criterion_label(criterion, criterion_value),
        "distance_metric": metric_used,
        "metric_warning": metric_warning,
        "importance_error": importance["error"],
        "importance": importance["values"],
        "parameters": selected_parameters,
        "profile_parameters": profile_parameters,
        "sampler": sampler,
        "sampler_note": sampler_note,
        "summary": {
            "best": float(best),
            "eligible": len(records),
            "total_completed": len(all_records),
            "selected": len(selected_records),
            "median_objective": float(np.median(selected_objectives)),
            "objective_min": float(np.min(selected_objectives)),
            "objective_max": float(np.max(selected_objectives)),
            **diversity,
        },
        "baseline": baseline,
        "diversity_curve": diversity_curve,
        "spreads": spreads,
        "clusters": {
            "count": clusters["n_clusters"],
            "profiles": profiles,
            "parameter_distinctness": parameter_distinctness,
        },
        "heatmap": {
            "matrix": np.round(heatmap_matrix, HEATMAP_DISPLAY_DECIMALS).tolist(),
            "trial_ids": heatmap_trials,
            "clusters": heatmap_clusters,
            "silhouette": np.round(heatmap_silhouette, HEATMAP_DISPLAY_DECIMALS).tolist(),
            "linkage": heatmap_linkage,
            "shown": len(heatmap_trials),
            "total": len(selected_records),
        },
        "distributions": distributions,
        "trials": table_rows,
    }


def _format_number(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4g}"


def _summary_card(label: str, value: str, detail: Optional[str] = None) -> Component:
    children: List[Component] = [html.Div(label, className="near-optimal-stat__label")]
    children.append(html.Div(value, className="near-optimal-stat__value"))
    if detail:
        children.append(html.Div(detail, className="near-optimal-stat__detail"))
    return html.Div(children, className="near-optimal-stat")


def diversity_figure(data: Dict[str, Any]) -> go.Figure:
    """Build the right-facing tolerance-versus-diversity needle plot."""
    curve = data["diversity_curve"]
    labels = [f"Top {row['percent']:g}%" for row in curve]
    median = [row["median"] for row in curve]
    q25 = [row["q25"] for row in curve]
    q75 = [row["q75"] for row in curve]
    custom = [[row["count"], row["q25"], row["q75"]] for row in curve]
    stem_x = [value for median_value in median for value in (0, median_value, None)]
    stem_y = [value for label in labels for value in (label, label, None)]
    iqr_x = [value for low, high in zip(q25, q75) for value in (low, high, None)]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=stem_x,
            y=stem_y,
            mode="lines",
            line={"color": "#9AA4B6", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=iqr_x,
            y=stem_y,
            mode="lines",
            line={"color": "rgba(65, 105, 255, 0.3)", "width": 8},
            hoverinfo="skip",
            name="Interquartile range (IQR)",
            legendrank=2,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=median,
            y=labels,
            customdata=custom,
            text=labels,
            mode="markers",
            name="Median pairwise distance",
            legendrank=1,
            marker={"color": "#4169FF", "size": 10},
            hovertemplate=(
                "%{text}<br>Trials: %{customdata[0]}<br>Median: %{x:.4g}"
                "<br>IQR: %{customdata[1]:.4g} – %{customdata[2]:.4g}<extra></extra>"
            ),
        )
    )
    style_analysis_panel(
        figure,
        height=DIVERSITY_GRAPH_HEIGHT,
        bottom=112,
        top=16,
        left=60,
        right=18,
        show_legend=True,
    )
    figure.update_xaxes(
        title_text="Median pairwise distance",
        range=[0, 1.05],
        dtick=0.2,
        tickformat=".1f",
        showline=True,
        linecolor="#9AA4B6",
        tickfont_size=11,
    )
    figure.update_yaxes(
        title_text="Performance tolerance (Top X% of trials)",
        categoryorder="array",
        categoryarray=labels[::-1],
        showline=True,
        linecolor="#9AA4B6",
        tickfont_size=11,
    )
    figure.update_layout(
        font_color="#303A64",
        legend={
            "x": 0.5,
            "xanchor": "center",
            "y": -0.30,
            "yanchor": "top",
            "font": {"size": 10},
            "traceorder": "normal",
        },
    )
    return figure


def _cluster_color(cluster: int) -> str:
    """Use the same cluster color in the matrix strips and legend."""
    palette = CLUSTER_COLORS + tuple(px.colors.qualitative.Alphabet)
    return palette[(cluster - 1) % len(palette)]


def heatmap_figure(data: Dict[str, Any], ordering: str = "hierarchical") -> go.Figure:
    """Build a distance matrix with aligned dendrograms, cluster strips, and silhouettes."""
    heatmap = data["heatmap"]
    matrix = np.asarray(heatmap["matrix"], dtype=float)
    trial_ids = heatmap["trial_ids"]
    clusters = heatmap["clusters"]
    silhouettes = heatmap.get("silhouette")
    if silhouettes is None:
        # Earlier cached analyses predate the silhouette strip. Recover it from the stored
        # distances and cluster assignments instead of failing to render the entire analysis.
        if len(trial_ids) > 1 and 1 < len(set(clusters)) < len(trial_ids):
            silhouettes = silhouette_samples(matrix, clusters, metric="precomputed").tolist()
        else:
            silhouettes = [0.0] * len(trial_ids)
    if matrix.size == 0:
        figure = go.Figure()
        figure.add_annotation(text="No pairwise distances available.", showarrow=False)
        style_analysis_panel(figure, height=PRIMARY_GRAPH_HEIGHT, bottom=55, top=10)
        return figure

    order = list(range(len(trial_ids)))
    dendro = None
    if ordering == "trial":
        order.sort(key=lambda index: trial_ids[index])
    elif ordering == "cluster":
        order.sort(key=lambda index: (clusters[index], trial_ids[index]))
    elif heatmap["linkage"]:
        dendro = dendrogram(np.asarray(heatmap["linkage"]), no_plot=True)
        order = list(map(int, dendro["leaves"]))
    matrix = matrix[np.ix_(order, order)]
    trial_ids = [trial_ids[index] for index in order]
    clusters = [clusters[index] for index in order]
    silhouettes = [silhouettes[index] for index in order]
    positions = [5 + 10 * index for index in range(len(trial_ids))]
    figure = go.Figure()
    if dendro is not None:
        dendrogram_x: List[Optional[float]] = []
        dendrogram_y: List[Optional[float]] = []
        for x, y in zip(dendro["icoord"], dendro["dcoord"]):
            dendrogram_x.extend([*x, None])
            dendrogram_y.extend([*y, None])
        figure.add_trace(
            go.Scatter(
                x=dendrogram_x,
                y=dendrogram_y,
                mode="lines",
                line={"color": "#777D8F", "width": 1},
                xaxis="x",
                yaxis="y2",
                connectgaps=False,
                hoverinfo="skip",
                showlegend=False,
            ),
        )
        figure.add_trace(
            go.Scatter(
                x=dendrogram_y,
                y=dendrogram_x,
                mode="lines",
                line={"color": "#777D8F", "width": 1},
                xaxis="x2",
                yaxis="y",
                connectgaps=False,
                hoverinfo="skip",
                showlegend=False,
            ),
        )
    customdata = [
        [
            [trial_ids[row], trial_ids[column]]
            for column in range(len(trial_ids))
        ]
        for row in range(len(trial_ids))
    ]
    figure.add_trace(
        go.Heatmap(
            z=matrix,
            x=positions,
            y=positions,
            customdata=customdata,
            zmin=0,
            zmax=1,
            colorscale=[[0, "#243775"], [0.25, "#4D8FC5"], [0.6, "#B3D3EC"], [1, "#F3F8FC"]],
            showscale=False,
            xgap=0.4,
            ygap=0.4,
            hovertemplate=(
                "Trial %{customdata[0]} ↔ %{customdata[1]}"
                "<br>Distance: %{z:.4g}<extra></extra>"
            ),
        ),
    )
    cluster_ids = sorted(set(clusters))
    color_indices = {cluster: index for index, cluster in enumerate(cluster_ids)}
    colorscale = [
        (bound, _cluster_color(cluster))
        for index, cluster in enumerate(cluster_ids)
        for bound in (index / len(cluster_ids), (index + 1) / len(cluster_ids))
    ]
    strip_values = [color_indices[cluster] for cluster in clusters]
    strip_custom = [[cluster, trial] for cluster, trial in zip(clusters, trial_ids)]
    for vertical in (False, True):
        figure.add_trace(
            go.Heatmap(
                z=[[value] for value in strip_values] if vertical else [strip_values],
                x=[0] if vertical else positions,
                y=positions if vertical else [0],
                xaxis="x3" if vertical else "x",
                yaxis="y" if vertical else "y3",
                customdata=[[value] for value in strip_custom] if vertical else [strip_custom],
                zmin=-0.5,
                zmax=len(cluster_ids) - 0.5,
                colorscale=colorscale,
                showscale=False,
                xgap=1,
                ygap=1,
                hovertemplate="Trial %{customdata[1]}<br>Cluster %{customdata[0]}<extra></extra>",
            )
        )
    figure.add_trace(
        go.Bar(
            x=silhouettes,
            y=positions,
            width=8,
            orientation="h",
            marker={"color": [_cluster_color(cluster) for cluster in clusters]},
            customdata=[[trial, cluster] for trial, cluster in zip(trial_ids, clusters)],
            xaxis="x4",
            yaxis="y",
            hovertemplate=(
                "Trial %{customdata[0]}<br>Cluster %{customdata[1]}"
                "<br>Silhouette: %{x:.4g}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    tick_step = max(1, int(np.ceil(len(trial_ids) / 24)))
    tick_indices = list(range(0, len(trial_ids), tick_step))
    ticks = {
        "tickmode": "array",
        "tickvals": [positions[index] for index in tick_indices],
        "ticktext": [f"trial_{trial_ids[index]:03d}" for index in tick_indices],
        "tickfont": {"size": 9},
        "showgrid": False,
        "zeroline": False,
        "showline": False,
        "ticks": "",
        "fixedrange": True,
    }
    extent = 10 * len(trial_ids)
    style_analysis_panel(
        figure,
        height=PRIMARY_GRAPH_HEIGHT,
        bottom=64,
        top=4,
        left=8,
        right=8,
    )
    figure.update_layout(
        font_color="#303A64",
        xaxis={
            **ticks,
            "domain": [0.13, 0.73],
            "range": [0, extent],
            "anchor": "y",
            "tickangle": -90,
        },
        yaxis={
            **ticks,
            "domain": [0, 0.84],
            "range": [extent, 0],
            "anchor": "free",
            "position": 0.108,
            "ticklabelposition": "outside left",
        },
        yaxis2={
            "domain": [0.885, 1],
            "anchor": "x",
            "visible": False,
            "rangemode": "tozero",
            "fixedrange": True,
        },
        xaxis2={
            "domain": [0, 0.065],
            "anchor": "y",
            "visible": False,
            "autorange": "reversed",
            "fixedrange": True,
        },
        xaxis3={
            "domain": [0.108, 0.122],
            "anchor": "y",
            "visible": False,
            "range": [-0.5, 0.5],
            "fixedrange": True,
        },
        xaxis4={
            "domain": [0.75, 0.94],
            "anchor": "y",
            "range": [-1, 1],
            "tickmode": "array",
            "tickvals": [-1, 0, 1],
            "ticktext": ["−1", "0", "+1"],
            "tickfont": {"size": 9},
            "title": {"text": "Silhouette coefficient", "font": {"size": 10}},
            "showgrid": False,
            "showline": False,
            "zeroline": True,
            "zerolinecolor": "#cbd5e1",
            "zerolinewidth": 1,
            "fixedrange": True,
        },
        yaxis3={
            "domain": [0.853, 0.88],
            "anchor": "x",
            "visible": False,
            "range": [-0.5, 0.5],
            "fixedrange": True,
        },
    )
    return figure


def importance_figure(data: Dict[str, Any], selected_parameter: Optional[str] = None) -> go.Figure:
    """Plot global parameter importance against near-optimal search-space coverage."""
    spread_by_name = {row["parameter"]: row for row in data["spreads"]}
    names = [
        name
        for name in data["parameters"]
        if name in data["importance"]
        and spread_by_name.get(name, {}).get("near_optimal_coverage") is not None
    ]
    figure = go.Figure()
    if names:
        coverages = [float(spread_by_name[name]["near_optimal_coverage"]) for name in names]
        importances = [float(data["importance"][name]) for name in names]
        positive_importances = [value for value in importances if value > 0]
        importance_floor = min(positive_importances) / 3 if positive_importances else 1e-4
        plot_importances = [max(value, importance_floor) for value in importances]
        importance_threshold = max(float(np.median(importances)), importance_floor)
        coverage_threshold = 0.5

        color_sequence = px.colors.qualitative.Plotly + px.colors.qualitative.Alphabet
        colors_by_parameter = {
            name: color_sequence[index % len(color_sequence)]
            for index, name in enumerate(data["parameters"])
        }
        hover_data = []
        for name in names:
            row = spread_by_name[name]
            hover_data.append(
                [
                    name,
                    float(data["importance"][name]),
                    row.get("near_optimal_spread"),
                    row.get("overall_spread"),
                    row.get("relative_spread"),
                    row.get("near_optimal_active"),
                    row.get("kind", "unknown"),
                    row.get("near_optimal_coverage"),
                ]
            )
        figure.add_trace(
            go.Scatter(
                x=coverages,
                y=plot_importances,
                text=[name if name == selected_parameter else "" for name in names],
                customdata=hover_data,
                mode="markers+text" if selected_parameter in names else "markers",
                textposition=[
                    "top left" if coverage >= 0.5 else "top right" for coverage in coverages
                ],
                textfont={"size": 11, "color": "#334155"},
                marker={
                    "size": [14 if name == selected_parameter else 12 for name in names],
                    "color": [colors_by_parameter[name] for name in names],
                    "line": {"color": "white", "width": 0.8},
                },
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br><br>"
                    "Hyperparameter importance: %{customdata[1]:.4g}<br>"
                    "Near-optimal coverage: %{customdata[7]:.2f}<br>"
                    "Near-optimal spread: %{customdata[2]:.2f}<br>"
                    "Overall spread: %{customdata[3]:.2f}<br>"
                    "Relative spread: %{customdata[4]:.2f}<br>"
                    "Active: %{customdata[5]:.1%}<br>"
                    "Parameter type: %{customdata[6]}<extra></extra>"
                ),
            )
        )
        log_importances = [np.log10(value) for value in plot_importances]
        log_importance_center = np.log10(importance_threshold)
        log_importance_radius = (
            max(
                0.35,
                max(abs(value - log_importance_center) for value in log_importances),
            )
            * 1.35
        )
        figure.update_xaxes(
            title={"text": "Search-space coverage", "standoff": 8},
            type="linear",
            range=[-0.03, 1.03],
            tickmode="array",
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
            showgrid=True,
            gridcolor="#E2E8F0",
        )
        figure.update_yaxes(
            title={"text": "Hyperparameter importance", "standoff": 8},
            type="log",
            dtick=1,
            range=[
                log_importance_center - log_importance_radius,
                log_importance_center + log_importance_radius,
            ],
            showgrid=True,
            gridcolor="#E2E8F0",
        )
        style_analysis_panel(
            figure,
            height=_constraints_panel_height(data),
            bottom=54,
            top=44,
            left=58,
            right=16,
        )
        figure.add_vline(
            x=coverage_threshold, line_dash="dash", line_color="#64748B", line_width=1.5
        )
        figure.add_hline(
            y=importance_threshold, line_dash="dash", line_color="#64748B", line_width=1.5
        )
        quadrant_labels = [
            (
                0.015,
                0.985,
                "left",
                "top",
                "Higher than median importance<br>Lower coverage (< 50%)<br>(critical setting)",
            ),
            (
                0.985,
                0.985,
                "right",
                "top",
                "Higher than median importance<br>Higher coverage (≥ 50%)"
                "<br>(alternative regimes?)",
            ),
            (
                0.015,
                0.015,
                "left",
                "bottom",
                "Lower than median importance<br>Lower coverage (< 50%)",
            ),
            (
                0.985,
                0.015,
                "right",
                "bottom",
                "Lower than median importance<br>Higher coverage (≥ 50%)<br>(flexible)",
            ),
        ]
        for x, y, xanchor, yanchor, label in quadrant_labels:
            figure.add_annotation(
                x=x,
                y=y,
                xref="paper",
                yref="paper",
                xanchor=xanchor,
                yanchor=yanchor,
                align=xanchor,
                text=label,
                showarrow=False,
                font={"size": 9, "color": "#475569"},
            )
    else:
        figure.add_annotation(
            text="Global fANOVA importance is unavailable for this study.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        figure.update_xaxes(title_text="Search-space coverage", range=[0.0, 1.0])
        figure.update_yaxes(title_text="Hyperparameter importance")
        style_analysis_panel(figure, height=_constraints_panel_height(data), bottom=65)
    return figure


def distribution_figure(data: Dict[str, Any], parameter: str, cluster: int) -> go.Figure:
    """Compare one parameter over all, near-optimal, and active-cluster trials."""
    payload = data["distributions"].get(parameter, {})
    overall = [value for value in payload.get("overall", []) if value is not None]
    near = [value for value in payload.get("near", []) if value is not None]
    cluster_values = [
        value for value in payload.get("clusters", {}).get(str(cluster), []) if value is not None
    ]
    figure = go.Figure()
    spread = next(
        (row for row in data["spreads"] if row["parameter"] == parameter),
        {},
    )
    categorical = spread.get("kind") in ("categorical", "ordinal", "constant")
    traces = [
        ("All observed", overall, "#CBD5E1"),
        ("Near-optimal", near, "#60A5FA"),
        (f"Cluster {cluster}", cluster_values, "#2563EB"),
    ]
    for name, values, color in traces:
        if categorical:
            counts = Counter(values)
            figure.add_trace(
                go.Bar(name=name, x=list(counts), y=list(counts.values()), marker_color=color)
            )
        else:
            figure.add_trace(
                go.Histogram(
                    name=name, x=values, histnorm="probability", opacity=0.55, marker_color=color
                )
            )
    figure.update_layout(barmode="overlay" if not categorical else "group")
    figure.update_xaxes(title_text=parameter)
    figure.update_yaxes(title_text="Share" if not categorical else "Trials")
    style_analysis_panel(figure, height=280, bottom=55, show_legend=True)
    return figure


def _cluster_coordinate_axes(
    data: Dict[str, Any], values_by_parameter: Mapping[str, Sequence[Any]]
) -> Dict[str, Dict[str, Any]]:
    """Build independent normalized axes for the cluster profile plot."""
    kind_by_parameter = {row["parameter"]: row["kind"] for row in data["spreads"]}
    axes = {}
    for parameter in data["profile_parameters"]:
        values = list(values_by_parameter[parameter])
        active = [value for value in values if value is not None]
        categorical = kind_by_parameter.get(parameter) == "categorical"
        if not categorical:
            try:
                [float(value) for value in active]
            except (TypeError, ValueError):
                categorical = True

        if categorical:
            categories = list(dict.fromkeys(active))
            if len(active) < len(values):
                categories.append(None)
            denominator = max(1, len(categories) - 1)
            encoding = {value: index / denominator for index, value in enumerate(categories)}
            axes[parameter] = {
                "categorical": True,
                "encoding": encoding,
                "ticks": [
                    (position, "inactive" if value is None else str(value))
                    for value, position in encoding.items()
                ],
            }
        else:
            numbers = np.asarray(active, dtype=float)
            minimum = float(np.min(numbers)) if numbers.size else 0.0
            maximum = float(np.max(numbers)) if numbers.size else 1.0
            if minimum == maximum:
                ticks = [(0.5, _format_number(minimum))]
            else:
                raw_ticks = np.linspace(minimum, maximum, 5)
                ticks = [
                    (
                        float((value - minimum) / (maximum - minimum)),
                        _format_number(float(value)),
                    )
                    for value in raw_ticks
                ]
            if len(active) < len(values):
                ticks.insert(0, (-0.1, "inactive"))
            axes[parameter] = {
                "categorical": False,
                "minimum": minimum,
                "maximum": maximum,
                "ticks": ticks,
            }
    return axes


def _encode_cluster_coordinate(axis: Dict[str, Any], value: Any) -> float:
    """Map one raw value onto its parameter's normalized visual axis."""
    if axis["categorical"]:
        return float(axis["encoding"].get(value, -0.1))
    if value is None:
        return -0.1
    minimum = axis["minimum"]
    maximum = axis["maximum"]
    if minimum == maximum:
        return 0.5
    return float((float(value) - minimum) / (maximum - minimum))


def _cluster_fill_color(color: str, opacity: float) -> str:
    """Convert a hexadecimal cluster color to an rgba fill color."""
    return (
        f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, "
        f"{int(color[5:7], 16)}, {opacity})"
    )


def _cluster_parameter_label(parameter: str) -> str:
    """Wrap long underscored parameter names without rotating them."""
    parts = parameter.split("_")
    if len(parameter) <= 14 or len(parts) == 1:
        return parameter
    split = (len(parts) + 1) // 2
    return f"{'_'.join(parts[:split])}<br>{'_'.join(parts[split:])}"


def cluster_coordinates_figure(data: Dict[str, Any], mode: str = "summary") -> go.Figure:
    """Compare cluster trials or cluster medians and interquartile ranges."""
    profiles = data["clusters"]["profiles"]
    parameters = data["profile_parameters"]
    cluster_ids = [profile["cluster"] for profile in profiles]
    distributions = data["distributions"]
    values_by_parameter = {
        parameter: [
            value
            for cluster in cluster_ids
            for value in distributions[parameter]["clusters"].get(str(cluster), [])
        ]
        for parameter in parameters
    }
    axes = _cluster_coordinate_axes(data, values_by_parameter)
    x_values = list(range(len(parameters)))
    figure = go.Figure()

    if mode == "trials":
        for cluster in cluster_ids:
            cluster_values = {
                parameter: distributions[parameter]["clusters"].get(str(cluster), [])
                for parameter in parameters
            }
            trial_count = max((len(values) for values in cluster_values.values()), default=0)
            trace_x: List[Optional[int]] = []
            trace_y: List[Optional[float]] = []
            hover_values: List[Optional[List[str]]] = []
            for trial_index in range(trial_count):
                for x, parameter in zip(x_values, parameters):
                    values = cluster_values[parameter]
                    value = values[trial_index] if trial_index < len(values) else None
                    trace_x.append(x)
                    trace_y.append(_encode_cluster_coordinate(axes[parameter], value))
                    hover_values.append(
                        [
                            parameter,
                            (
                                "inactive"
                                if value is None
                                else str(value)
                                if axes[parameter]["categorical"]
                                else _format_number(float(value))
                            ),
                        ]
                    )
                trace_x.append(None)
                trace_y.append(None)
                hover_values.append(None)
            figure.add_trace(
                go.Scatter(
                    x=trace_x,
                    y=trace_y,
                    customdata=hover_values,
                    mode="lines",
                    name=f"Cluster {cluster}",
                    line={"color": _cluster_color(cluster), "width": 1},
                    opacity=0.28,
                    hovertemplate=(
                        "%{customdata[0]}<br>Value: %{customdata[1]}"
                        f"<extra>Cluster {cluster}</extra>"
                    ),
                )
            )
    else:
        for cluster in cluster_ids:
            lower_values = []
            median_values = []
            upper_values = []
            hover_values = []
            for parameter in parameters:
                values = distributions[parameter]["clusters"].get(str(cluster), [])
                active = [value for value in values if value is not None]
                axis = axes[parameter]
                if axis["categorical"]:
                    center = Counter(active).most_common(1)[0][0] if active else None
                    lower = upper = center
                elif active:
                    lower, center, upper = np.quantile(
                        np.asarray(active, dtype=float), [0.25, 0.5, 0.75]
                    )
                else:
                    lower = center = upper = None
                lower_values.append(_encode_cluster_coordinate(axis, lower))
                median_values.append(_encode_cluster_coordinate(axis, center))
                upper_values.append(_encode_cluster_coordinate(axis, upper))
                if axis["categorical"]:
                    summary = "inactive" if center is None else str(center)
                else:
                    summary = (
                        "inactive"
                        if center is None
                        else f"{_format_number(center)} "
                        f"[{_format_number(lower)} – {_format_number(upper)}]"
                    )
                hover_values.append([parameter, summary])

            color = _cluster_color(cluster)
            figure.add_trace(
                go.Scatter(
                    x=x_values + x_values[::-1],
                    y=upper_values + lower_values[::-1],
                    mode="lines",
                    line={"color": "rgba(0, 0, 0, 0)", "width": 0},
                    fill="toself",
                    fillcolor=_cluster_fill_color(color, 0.13),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=str(cluster),
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=median_values,
                    customdata=hover_values,
                    mode="lines+markers",
                    name=f"Cluster {cluster}",
                    legendgroup=str(cluster),
                    line={"color": color, "width": 3},
                    marker={
                        "color": color,
                        "size": 7,
                        "line": {"color": "white", "width": 1},
                    },
                    hovertemplate=(
                        "%{customdata[0]}<br>Median / IQR: %{customdata[1]}"
                        f"<extra>Cluster {cluster}</extra>"
                    ),
                )
            )

    shapes = []
    annotations = []
    for x, parameter in zip(x_values, parameters):
        ticks = axes[parameter]["ticks"]
        shapes.append(
            {
                "type": "line",
                "x0": x,
                "x1": x,
                "y0": min(position for position, _label in ticks),
                "y1": max(position for position, _label in ticks),
                "line": {"color": "#cbd5e1", "width": 1},
                "layer": "below",
            }
        )
        for position, label in ticks:
            annotations.append(
                {
                    "x": x,
                    "y": position,
                    "text": label,
                    "showarrow": False,
                    "xshift": -8,
                    "xanchor": "right",
                    "font": {"size": 10, "color": "#64748b"},
                }
            )

    figure.update_layout(
        shapes=shapes,
        annotations=annotations,
        height=390,
        margin={"t": 70, "b": 35, "l": 70, "r": 115},
        font={"size": config.FIGURE_FONT_SIZE},
        hovermode="closest",
        legend={
            "title": {"text": "Clusters"},
            "x": 1.02,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top",
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.update_xaxes(
        range=[-0.08, max(0, len(parameters) - 1) + 0.08],
        tickmode="array",
        tickvals=x_values,
        ticktext=[_cluster_parameter_label(parameter) for parameter in parameters],
        side="top",
        ticks="",
        showgrid=False,
        showline=False,
        zeroline=False,
        fixedrange=True,
    )
    figure.update_yaxes(range=[-0.15, 1.05], visible=False, fixedrange=True)
    return figure


def _constraints_panel_height(data: Dict[str, Any]) -> int:
    """Keep the constraints summary and importance plot at comparable heights."""
    visible_rows = min(len(data["spreads"]), 15)
    return max(380, 96 + visible_rows * 41)


def _constraint_interval(row: Dict[str, Any]) -> Component:
    """Render the selected trials' min/max interval in normalized search-space coordinates."""
    lower = row.get("near_optimal_min")
    median = row.get("near_optimal_median")
    upper = row.get("near_optimal_max")
    if lower is None or median is None or upper is None:
        return html.Div("N/A", className="near-optimal-interval near-optimal-interval--empty")

    lower = min(1.0, max(0.0, float(lower)))
    median = min(1.0, max(0.0, float(median)))
    upper = min(1.0, max(0.0, float(upper)))
    range_class = "near-optimal-interval__range"
    if abs(upper - lower) < 1e-12:
        range_class += " near-optimal-interval__range--point"
    return html.Div(
        [
            html.Div(
                [
                    html.Div(className=range_class),
                    html.Div(className="near-optimal-interval__median"),
                ],
                className="near-optimal-interval__track",
                style={
                    "--interval-start": f"{lower * 100.0:.2f}%",
                    "--interval-width": f"{(upper - lower) * 100.0:.2f}%",
                    "--interval-median": f"{median * 100.0:.2f}%",
                },
            ),
            html.Div(
                [html.Span(f"{lower:.2f}"), html.Span(f"{upper:.2f}")],
                className="near-optimal-interval__labels",
            ),
        ],
        className="near-optimal-interval",
    )


def _constraints_table(data: Dict[str, Any]) -> Component:
    """Render selected-trial intervals within each parameter's normalized search space."""
    spreads = data["spreads"]
    header_cells = [
        html.Div("Parameter", className="near-optimal-constraints__heading"),
        html.Div(
            [
                html.Span("Near-optimal region (min – max)"),
                html.Div(
                    [html.Span("0"), html.Span("1")],
                    className="near-optimal-constraints__axis",
                ),
            ],
            className="near-optimal-constraints__heading",
        ),
        html.Div("Coverage", className="near-optimal-constraints__heading text-end"),
    ]

    rows = []
    for row in spreads:
        lower = row.get("near_optimal_min")
        median = row.get("near_optimal_median")
        upper = row.get("near_optimal_max")
        coverage = row.get("near_optimal_coverage")
        tooltip = (
            f"{row['parameter']}\n"
            f"Minimum: {_format_number(lower)}\n"
            f"Median: {_format_number(median)}\n"
            f"Maximum: {_format_number(upper)}\n"
            f"Coverage: {_format_number(coverage)}\n"
            f"Selected trials: {data['summary']['selected']}\n"
            f"Parameter type: {row.get('kind', 'unknown')}"
        )
        cells = [
            html.Span(row["parameter"], className="near-optimal-constraints__parameter"),
            _constraint_interval(row),
            html.Span(
                "N/A" if coverage is None else f"{float(coverage):.2f}",
                className="near-optimal-constraints__coverage",
            ),
        ]
        rows.append(
            html.Button(
                cells,
                id={"type": "near-optimal-constraint-row", "parameter": row["parameter"]},
                n_clicks=0,
                type="button",
                title=tooltip,
                className="near-optimal-constraints__row",
            )
        )

    scroll_modifier = " near-optimal-constraints__body--scrollable" if len(rows) > 15 else ""
    return html.Div(
        [
            html.Div(
                header_cells,
                className="near-optimal-constraints__header",
            ),
            html.Div(rows, className=f"near-optimal-constraints__body{scroll_modifier}"),
        ],
        id=CONSTRAINTS_ID,
        className="near-optimal-constraints",
    )


def _trial_table(data: Dict[str, Any]) -> Component:
    columns = [
        {"name": "Trial", "id": "trial_id", "type": "numeric"},
        {"name": data["objective_name"], "id": "objective", "type": "numeric"},
        {"name": "Cluster", "id": "cluster", "type": "numeric"},
    ]
    columns.extend({"name": name, "id": f"hp::{name}"} for name in data["parameters"])
    return dash_table.DataTable(
        id=TRIAL_TABLE_ID,
        data=data["trials"],
        columns=columns,
        page_action="native",
        page_size=10,
        sort_action="native",
        sort_mode="multi",
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": "600", "backgroundColor": "#F8FAFC"},
        style_cell={
            "padding": "0.4rem",
            "fontSize": 12,
            "minWidth": "85px",
            "maxWidth": "190px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
    )


def _cluster_detail(data: Dict[str, Any]) -> Component:
    """Compare all cluster profiles side by side in one table."""
    profiles = data["clusters"]["profiles"]
    summaries = {
        profile["cluster"]: {
            row["parameter"]: row["summary"] for row in profile["parameters"]
        }
        for profile in profiles
    }
    rows = [
        html.Tr(
            [html.Th("Median objective", scope="row")]
            + [html.Td(_format_number(profile["objective_median"])) for profile in profiles]
        ),
        html.Tr(
            [html.Th("Objective range", scope="row")]
            + [
                html.Td(f"{profile['objective_min']:.4g} – {profile['objective_max']:.4g}")
                for profile in profiles
            ]
        ),
    ]
    rows.extend(
        html.Tr(
            [html.Th(parameter, scope="row")]
            + [html.Td(summaries[profile["cluster"]][parameter]) for profile in profiles]
        )
        for parameter in data["profile_parameters"]
    )
    return html.Div(
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [html.Th("Parameter / metric")]
                        + [
                            html.Th(f"Cluster {profile['cluster']} (n = {profile['size']})")
                            for profile in profiles
                        ]
                    )
                ),
                html.Tbody(rows),
            ],
            className="near-optimal-profile-table",
        ),
        id=CLUSTER_DETAIL_ID,
        className="near-optimal-profile-table-wrap",
    )


def _heatmap_sidebar(data: Dict[str, Any]) -> Component:
    """Show cluster membership counts and the matrix ordering control."""
    children = [
        html.Div("Cluster assignment", className="near-optimal-heatmap-sidebar__title"),
        html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            className="near-optimal-cluster-dot",
                            style={"backgroundColor": _cluster_color(profile["cluster"])},
                        ),
                        html.Span(f"Cluster {profile['cluster']}"),
                        html.Span(
                            f"{profile['size']} trials",
                            className="near-optimal-cluster-count",
                        ),
                    ],
                    className="near-optimal-cluster-legend-row",
                )
                for profile in data["clusters"]["profiles"]
            ],
            className="near-optimal-cluster-legend",
        ),
        html.Div(
            [
                html.Label("Reorder by", htmlFor=HEATMAP_ORDER_ID),
                dbc.Select(
                    id=HEATMAP_ORDER_ID,
                    value="hierarchical",
                    options=[
                        {"label": "Hierarchical clustering", "value": "hierarchical"},
                        {"label": "Cluster assignment", "value": "cluster"},
                        {"label": "Trial number", "value": "trial"},
                    ],
                    size="sm",
                ),
            ],
            className="near-optimal-heatmap-order",
        ),
    ]
    if data["heatmap"]["shown"] < data["heatmap"]["total"]:
        children.append(
            html.P(
                f"Showing {data['heatmap']['shown']} of {data['heatmap']['total']} selected "
                "trials. Summaries and clusters use the complete set.",
                className="near-optimal-heatmap-note",
            )
        )
    return html.Aside(children, className="near-optimal-heatmap-sidebar")


def _page_layout(data: Dict[str, Any]) -> Component:
    summary = data["summary"]
    selected_detail = (
        f"{summary['selected'] / summary['eligible']:.1%} of eligible completed trials"
    )
    warnings = []
    if data["metric_warning"]:
        warnings.append(dbc.Alert(data["metric_warning"], color="warning", className="py-2"))
    constraints_height = _constraints_panel_height(data)
    analysis_token = _cache_analysis_data(data)
    return html.Div(
        [
            dcc.Store(id=ANALYSIS_DATA_ID, data=analysis_token),
            dcc.Store(
                id=PARAMETER_STORE_ID,
                data=(
                    data["profile_parameters"][0]
                    if data["profile_parameters"]
                    else data["parameters"][0]
                ),
            ),
            html.Div(
                [
                    html.Span("Rashomon Set Analysis", className="near-optimal-subtitle__name"),
                    html.Span(
                        "Explore how many different hyperparameter configurations achieve "
                        "similar observed performance."
                    ),
                ],
                className="near-optimal-subtitle",
            ),
            html.Div(warnings),
            html.Div(
                [
                    html.Div(
                        [
                            _summary_card("Best objective", _format_number(summary["best"])),
                            _summary_card("Selection criterion", data["criterion_label"]),
                            _summary_card(
                                "Selected trials",
                                f"{summary['selected']} / {summary['eligible']}",
                                selected_detail,
                            ),
                            _summary_card(
                                "Objective range",
                                f"{summary['objective_min']:.4g} – {summary['objective_max']:.4g}",
                            ),
                            _summary_card(
                                "Median objective", _format_number(summary["median_objective"])
                            ),
                            _summary_card(
                                "Median pairwise distance",
                                _format_number(summary["median"]),
                                (
                                    None
                                    if summary["q25"] is None
                                    else f"IQR: {summary['q25']:.4g} – {summary['q75']:.4g}"
                                ),
                            ),
                        ],
                        className="analysis-card near-optimal-summary",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-chart-line",
                                        **{"aria-hidden": "true"},
                                    ),
                                    html.Span(
                                        "Configuration Diversity (median pairwise distance)"
                                    ),
                                ],
                                className="analysis-card__header",
                            ),
                            analysis_graph(
                                DIVERSITY_ID,
                                DIVERSITY_GRAPH_HEIGHT,
                                config.FIGURE_DOWNLOAD_SCALE,
                            ),
                        ],
                        className="analysis-card near-optimal-primary-card",
                    ),
                ],
                className="near-optimal-overview-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.I(className="fas fa-th-large", **{"aria-hidden": "true"}),
                            html.Span(
                                "Pairwise Configuration Distance (Near-Optimal Set) "
                                "with Silhouette Plot"
                            ),
                        ],
                        className="analysis-card__header",
                    ),
                    html.Div(
                        [
                            analysis_graph(
                                HEATMAP_ID,
                                PRIMARY_GRAPH_HEIGHT,
                                config.FIGURE_DOWNLOAD_SCALE,
                            ),
                            _heatmap_sidebar(data),
                        ],
                        className="near-optimal-heatmap-body",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Parallel coordinates", className="fw-semibold"),
                                    dbc.RadioItems(
                                        id=CLUSTER_COORDINATES_MODE_ID,
                                        options=[
                                            {"label": "Median + IQR", "value": "summary"},
                                            {"label": "All trials", "value": "trials"},
                                        ],
                                        value="summary",
                                        inline=True,
                                        className="near-optimal-cluster-plot-mode",
                                    ),
                                ],
                                className="near-optimal-cluster-plot-toolbar",
                            ),
                            analysis_graph(
                                CLUSTER_COORDINATES_ID,
                                360,
                                config.FIGURE_DOWNLOAD_SCALE,
                            ),
                        ],
                        className="near-optimal-cluster-plot",
                    ),
                ],
                className="analysis-card near-optimal-primary-card near-optimal-distance-card",
            ),
            html.Div(
                [
                    html.Div(
                        "Distinct Good Configuration Regimes", className="analysis-card__header"
                    ),
                    _cluster_detail(data),
                ],
                className="analysis-card near-optimal-regimes",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "Hyperparameter Constraints", className="analysis-card__header"
                            ),
                            _constraints_table(data),
                        ],
                        className="analysis-card near-optimal-secondary-card",
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Parameter Importance vs. Near-Optimal Coverage",
                                className="analysis-card__header",
                            ),
                            analysis_graph(
                                IMPORTANCE_ID, constraints_height, config.FIGURE_DOWNLOAD_SCALE
                            ),
                        ],
                        className=(
                            "analysis-card near-optimal-secondary-card "
                            "near-optimal-importance-card"
                        ),
                    ),
                ],
                className="near-optimal-secondary-grid",
            ),
            html.Div(
                [
                    html.Div(
                        "Trials in Selected Near-Optimal Set", className="analysis-card__header"
                    ),
                    dbc.Input(
                        id=TRIAL_SEARCH_ID,
                        placeholder="Search trials...",
                        size="sm",
                        className="mb-2",
                    ),
                    _trial_table(data),
                ],
                className="analysis-card near-optimal-trials",
            ),
        ]
    )


class NearOptimalConfigurations(StaticPlugin):
    """Analyze diversity and regimes among observed near-optimal Optuna trials."""

    id = "near_optimal_configurations"
    name = "Rashomon Set Analysis"
    description = "Rashomon Set Analysis"
    icon = "fas fa-layer-group"
    help = "plugins/near_optimal_configurations.html"
    button_caption = "Run analysis"
    activate_run_selection = True
    main_objective_input = "objective_id"

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        return run.prefix == "Optuna" and not run.empty() and len(run.configspace) > 0

    @staticmethod
    def get_input_layout(register: Callable) -> List[Component]:
        return [
            html.Section(
                [
                    html.Div(
                        [
                            html.Div(
                                html.I(className="fas fa-bullseye"),
                                className="near-optimal-control-icon",
                            ),
                            html.Div(
                                [
                                    html.H2("Define near-optimal set"),
                                    html.P(
                                        "Select how to define the set of near-optimal "
                                        "configurations based on objective values."
                                    ),
                                ]
                            ),
                        ],
                        className="near-optimal-control-heading",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dbc.RadioItems(
                                        id=register("criterion", "value"),
                                        options=[
                                            {"label": "Top %", "value": "top_percent"},
                                            {"label": "Absolute Δ", "value": "absolute"},
                                            {"label": "Relative Δ", "value": "relative"},
                                        ],
                                        className="near-optimal-segmented",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            dbc.Label("Top percentage"),
                                                            help_button(
                                                                "Select exactly ceil(p × n) of the "
                                                                "best eligible trials."
                                                            ),
                                                        ],
                                                        className="near-optimal-field-label",
                                                    ),
                                                    dbc.RadioItems(
                                                        id=register(
                                                            "top_percent",
                                                            ["value", "options", "disabled"],
                                                            type=float,
                                                        ),
                                                        className="near-optimal-presets",
                                                    ),
                                                ],
                                                id=TOP_CRITERION_PANEL_ID,
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Maximum absolute distance from best"
                                                    ),
                                                    dbc.Input(
                                                        id=register(
                                                            "absolute_delta",
                                                            ["value", "disabled"],
                                                            type=float,
                                                        ),
                                                        type="number",
                                                        min=0,
                                                        step="any",
                                                    ),
                                                ],
                                                id=ABSOLUTE_CRITERION_PANEL_ID,
                                                style={"display": "none"},
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Maximum relative distance from best (%)"
                                                    ),
                                                    dbc.Input(
                                                        id=register(
                                                            "relative_delta",
                                                            ["value", "disabled"],
                                                            type=float,
                                                        ),
                                                        type="number",
                                                        min=0,
                                                        step="any",
                                                    ),
                                                ],
                                                id=RELATIVE_CRITERION_PANEL_ID,
                                                style={"display": "none"},
                                            ),
                                        ],
                                        className="near-optimal-criterion-value",
                                    ),
                                ],
                                className="near-optimal-definition-controls",
                            ),
                            html.Aside(
                                [
                                    html.Div(
                                        [
                                            html.I(className="fas fa-chart-bar"),
                                            html.Strong("Selected near-optimal set"),
                                        ],
                                        className="near-optimal-preview-title",
                                    ),
                                    html.Div(
                                        [
                                            html.I(className="fas fa-database"),
                                            html.Span("Select a study to preview the trial set."),
                                        ],
                                        id=SELECTION_COUNT_ID,
                                        className="near-optimal-preview-row",
                                    ),
                                    html.Div(
                                        [
                                            html.I(className="fas fa-chart-pie"),
                                            html.Span("— of completed trials"),
                                        ],
                                        id=SELECTION_SHARE_ID,
                                        className="near-optimal-preview-row",
                                    ),
                                    html.Div(
                                        [
                                            html.I(className="fas fa-chart-line"),
                                            html.Span("Objective range: —"),
                                        ],
                                        id=SELECTION_RANGE_ID,
                                        className="near-optimal-preview-row",
                                    ),
                                ],
                                className="near-optimal-selection-preview",
                            ),
                        ],
                        className="near-optimal-definition-grid",
                    ),
                ],
                className="near-optimal-control-card near-optimal-definition-card",
            ),
            html.Details(
                [
                    html.Summary(
                        html.Div(
                            [
                                html.Div(
                                    html.I(className="fas fa-cog"),
                                    className="near-optimal-control-icon",
                                ),
                                html.Div(
                                    [
                                        html.H2("Analysis settings"),
                                    ]
                                ),
                            ],
                            className="near-optimal-control-heading",
                        )
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dbc.Label("Distance metric"),
                                            help_button(
                                                "Importance weighting uses cached global fANOVA "
                                                "main effects."
                                            ),
                                        ],
                                        className="near-optimal-field-label",
                                    ),
                                    dbc.Select(
                                        id=register("distance_metric", ["value", "options"])
                                    ),
                                ],
                                className="near-optimal-setting-field",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dbc.Label("Clusters"),
                                            help_button(
                                                "Choose automatic hierarchical clustering or "
                                                "1–8 clusters."
                                            ),
                                        ],
                                        className="near-optimal-field-label",
                                    ),
                                    dbc.Select(id=register("cluster_count", ["value", "options"])),
                                ],
                                className="near-optimal-setting-field",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dbc.Label("Parameters"),
                                            help_button(
                                                "Choose the hyperparameters included in distance "
                                                "and spread calculations."
                                            ),
                                        ],
                                        className="near-optimal-field-label",
                                    ),
                                    dcc.Dropdown(
                                        id=register("parameter_names", ["value", "options"]),
                                        multi=True,
                                        placeholder="All parameters",
                                    ),
                                ],
                                className="near-optimal-setting-field",
                            ),
                        ],
                        className="near-optimal-settings-grid",
                    ),
                ],
                open=True,
                className="near-optimal-control-card near-optimal-settings-card",
            ),
        ]

    @staticmethod
    def get_filter_layout(_register: Callable) -> List[Component]:
        return []

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "objective_id": {"options": [], "value": None},
            "criterion": {"value": "top_percent"},
            "top_percent": {
                "options": [
                    {"label": f"{percent}%", "value": percent} for percent in TOP_PERCENT_PRESETS
                ],
                "value": 5,
                "disabled": False,
            },
            "absolute_delta": {"value": 0.0, "disabled": True},
            "relative_delta": {"value": 2.0, "disabled": True},
            "distance_metric": {
                "options": get_select_options(
                    ["Normalized configuration distance", "Importance-weighted distance"],
                    ["normalized", "importance_weighted"],
                ),
                "value": "importance_weighted",
            },
            "cluster_count": {
                "options": get_select_options(
                    ["Automatic (hierarchical)", *range(1, 9)], ["auto", *range(1, 9)]
                ),
                "value": "auto",
            },
            "parameter_names": {"options": [], "value": []},
        }

    def load_dependency_inputs(
        self,
        run: AbstractRun,
        previous_inputs: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        names = list(run.configspace.keys())
        previous_run_id = previous_inputs.get("run", {}).get("value")
        run_changed = previous_run_id is not None and previous_run_id != run.id

        criterion = inputs.get("criterion", {}).get("value")
        if criterion not in ("top_percent", "absolute", "relative"):
            criterion = "top_percent"

        top_percent = inputs.get("top_percent", {}).get("value")
        if top_percent not in TOP_PERCENT_PRESETS:
            top_percent = 5
        absolute_delta = inputs.get("absolute_delta", {}).get("value")
        if absolute_delta is None or not np.isfinite(absolute_delta) or absolute_delta < 0:
            absolute_delta = 0.0
        relative_delta = inputs.get("relative_delta", {}).get("value")
        if relative_delta is None or not np.isfinite(relative_delta) or relative_delta < 0:
            relative_delta = 2.0

        distance_metric = inputs.get("distance_metric", {}).get("value")
        if distance_metric not in ("normalized", "importance_weighted"):
            distance_metric = "importance_weighted"
        cluster_count = inputs.get("cluster_count", {}).get("value")
        valid_cluster_counts = {"auto", *range(1, 9), *(str(value) for value in range(1, 9))}
        if cluster_count not in valid_cluster_counts:
            cluster_count = "auto"

        selected = inputs.get("parameter_names", {}).get("value") or names
        if run_changed:
            selected = names
        selected = [name for name in selected if name in names]
        if not selected:
            selected = names
        return {
            "criterion": {"value": criterion},
            "top_percent": {
                "options": [
                    {"label": f"{percent}%", "value": percent} for percent in TOP_PERCENT_PRESETS
                ],
                "value": top_percent,
                "disabled": criterion != "top_percent",
            },
            "absolute_delta": {
                "value": absolute_delta,
                "disabled": criterion != "absolute",
            },
            "relative_delta": {
                "value": relative_delta,
                "disabled": criterion != "relative",
            },
            "distance_metric": {
                "options": get_select_options(
                    ["Normalized configuration distance", "Importance-weighted distance"],
                    ["normalized", "importance_weighted"],
                ),
                "value": distance_metric,
            },
            "cluster_count": {
                "options": get_select_options(
                    ["Automatic (hierarchical)", *range(1, 9)], ["auto", *range(1, 9)]
                ),
                "value": cluster_count,
            },
            "parameter_names": {"options": get_select_options(names), "value": selected},
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        criterion = inputs.get("criterion")
        if criterion not in ("top_percent", "absolute", "relative"):
            criterion = "top_percent"
        criterion_value = {
            "top_percent": inputs.get("top_percent"),
            "absolute": inputs.get("absolute_delta"),
            "relative": inputs.get("relative_delta"),
        }[criterion]
        if criterion_value is None:
            criterion_value = {"top_percent": 5.0, "absolute": 0.0, "relative": 2.0}[criterion]
        cluster_value = inputs.get("cluster_count") or "auto"
        cluster_count = None if cluster_value == "auto" else int(cluster_value)
        objective_id = inputs.get("objective_id")
        if objective_id is None:
            objective_id = 0
        parameter_names = inputs.get("parameter_names") or list(run.configspace.keys())
        return build_near_optimal_analysis(
            run,
            int(objective_id),
            criterion,
            float(criterion_value),
            inputs.get("distance_metric") or "importance_weighted",
            cluster_count,
            parameter_names,
        )

    @staticmethod
    def get_output_layout(register: Callable) -> Component:
        return html.Div(id=register("content", "children"))

    @staticmethod
    def load_outputs(
        _run: AbstractRun, _inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Component:
        page = _page_layout(outputs)
        for component in _walk_components(page):
            component_id = getattr(component, "id", None)
            if component_id == DIVERSITY_ID:
                component.figure = diversity_figure(outputs)
            elif component_id == HEATMAP_ID:
                component.figure = heatmap_figure(outputs)
            elif component_id == IMPORTANCE_ID:
                parameter = (
                    outputs["profile_parameters"][0]
                    if outputs["profile_parameters"]
                    else outputs["parameters"][0]
                )
                component.figure = importance_figure(outputs, parameter)
            elif component_id == CLUSTER_COORDINATES_ID:
                component.figure = cluster_coordinates_figure(outputs)
        return page

    @interactive
    def register_callbacks(self) -> None:
        super().register_callbacks()
        from deepcave import app, run_handler

        @app.callback(
            [
                Output(TOP_CRITERION_PANEL_ID, "style"),
                Output(ABSOLUTE_CRITERION_PANEL_ID, "style"),
                Output(RELATIVE_CRITERION_PANEL_ID, "style"),
                Output(SELECTION_COUNT_ID, "children"),
                Output(SELECTION_SHARE_ID, "children"),
                Output(SELECTION_RANGE_ID, "children"),
            ],
            [
                Input(self.get_internal_input_id("criterion"), "value"),
                Input(self.get_internal_input_id("top_percent"), "value"),
                Input(self.get_internal_input_id("absolute_delta"), "value"),
                Input(self.get_internal_input_id("relative_delta"), "value"),
                Input(self.get_internal_input_id("objective_id"), "value"),
                Input("global-run-selector", "value"),
            ],
        )
        def update_selection_preview(
            criterion: str,
            top_percent: Any,
            absolute_delta: Any,
            relative_delta: Any,
            objective_id: Any,
            run_id: Any,
        ) -> Any:
            criterion = criterion or "top_percent"
            panel_styles = [
                {} if criterion == "top_percent" else {"display": "none"},
                {} if criterion == "absolute" else {"display": "none"},
                {} if criterion == "relative" else {"display": "none"},
            ]
            placeholder = [
                html.I(className="fas fa-database"),
                html.Span("Select a study to preview the trial set."),
            ]
            if run_id is None or objective_id is None:
                return [*panel_styles, placeholder, placeholder, placeholder]

            criterion_value = {
                "top_percent": top_percent,
                "absolute": absolute_delta,
                "relative": relative_delta,
            }.get(criterion)
            if criterion_value is None:
                return [*panel_styles, placeholder, placeholder, placeholder]

            try:
                preview = _selection_preview(
                    run_handler.get_run(run_id),
                    int(objective_id),
                    criterion,
                    float(criterion_value),
                )
            except (IndexError, KeyError, RuntimeError, TypeError, ValueError):
                return [*panel_styles, placeholder, placeholder, placeholder]

            selected = preview["selected"]
            eligible = preview["eligible"]
            count = [
                html.I(className="fas fa-database"),
                html.Span([html.Strong(f"{selected:,}"), f" of {eligible:,} completed trials"]),
            ]
            share = [
                html.I(className="fas fa-chart-pie"),
                html.Span([html.Strong(f"{preview['share']:.1%}"), " of completed trials"]),
            ]
            if preview["minimum"] is None:
                objective_range = "—"
            else:
                objective_range = f"{preview['minimum']:.4g} – {preview['maximum']:.4g}"
            value_range = [
                html.I(className="fas fa-chart-line"),
                html.Span(["Objective range: ", html.Strong(objective_range)]),
            ]
            return [*panel_styles, count, share, value_range]

        @app.callback(
            Output(HEATMAP_ID, "figure"),
            Input(HEATMAP_ORDER_ID, "value"),
            State(ANALYSIS_DATA_ID, "data"),
            prevent_initial_call=True,
        )
        def reorder_heatmap(ordering: str, analysis_token: str) -> Any:
            data = _get_analysis_data(analysis_token)
            if not data:
                raise PreventUpdate
            return heatmap_figure(data, ordering or "hierarchical")

        @app.callback(
            Output(CLUSTER_COORDINATES_ID, "figure"),
            Input(CLUSTER_COORDINATES_MODE_ID, "value"),
            State(ANALYSIS_DATA_ID, "data"),
            prevent_initial_call=True,
        )
        def update_cluster_coordinates(mode: str, analysis_token: str) -> Any:
            data = _get_analysis_data(analysis_token)
            if not data:
                raise PreventUpdate
            return cluster_coordinates_figure(data, mode or "summary")

        @app.callback(
            Output(PARAMETER_STORE_ID, "data"),
            [
                Input({"type": "near-optimal-constraint-row", "parameter": ALL}, "n_clicks"),
                Input(IMPORTANCE_ID, "clickData"),
            ],
            State(PARAMETER_STORE_ID, "data"),
            prevent_initial_call=True,
        )
        def select_parameter(_row_clicks: Any, click_data: Any, current: str) -> Any:
            if ctx.triggered_id == IMPORTANCE_ID and click_data:
                customdata = click_data["points"][0]["customdata"]
                return str(customdata[0] if isinstance(customdata, (list, tuple)) else customdata)
            if isinstance(ctx.triggered_id, dict):
                return str(ctx.triggered_id["parameter"])
            return current or no_update

        @app.callback(
            Output(IMPORTANCE_ID, "figure"),
            Input(PARAMETER_STORE_ID, "data"),
            State(ANALYSIS_DATA_ID, "data"),
            prevent_initial_call=True,
        )
        def highlight_selected_parameter(parameter: str, analysis_token: str) -> Any:
            data = _get_analysis_data(analysis_token)
            if not data:
                raise PreventUpdate
            return importance_figure(data, parameter)

        @app.callback(
            Output(TRIAL_TABLE_ID, "data"),
            Input(TRIAL_SEARCH_ID, "value"),
            State(ANALYSIS_DATA_ID, "data"),
            prevent_initial_call=True,
        )
        def filter_trials(search: Any, analysis_token: str) -> Any:
            data = _get_analysis_data(analysis_token)
            if not data:
                raise PreventUpdate
            rows = data["trials"]
            query = str(search or "").strip().lower()
            if not query:
                return rows
            return [
                row
                for row in rows
                if query in " ".join(str(value).lower() for value in row.values())
            ]


def _walk_components(component: Component) -> Any:
    """Yield one Dash component and all descendants."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "to_plotly_json"):
            yield from _walk_components(child)
