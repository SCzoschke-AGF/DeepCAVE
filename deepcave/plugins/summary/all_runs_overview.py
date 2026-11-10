"""Compare runs using descriptive statistics over successful trials."""

from typing import Any, Dict, List, Optional, Tuple, Union

import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime
from numbers import Real
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, pstdev

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, Patch, State, ctx, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate

from deepcave import interactive
from deepcave.layouts.optuna_metadata import study_metadata_url
from deepcave.plugins import Plugin
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status
from deepcave.runs.trial import Trial
from deepcave.utils.optuna import (
    get_optuna_snapshot_path,
    resolve_optuna_database,
)

STATISTICS = {
    "min": min,
    "max": max,
    "avg": mean,
    "median": median,
    "std": pstdev,
    "range": lambda values: max(values) - min(values),
}
BASE_COLUMNS = [
    {"id": "run", "name": "Run", "type": "text"},
    {"id": "path", "name": "Path", "type": "text"},
    {"id": "optimizer", "name": "Optimizer", "type": "text"},
    {"id": "successful", "name": "Successful Trials", "type": "numeric"},
    {"id": "duration", "name": "Total Trial Duration (s)", "type": "numeric"},
]
DEFAULT_COLUMNS = ["successful", json.dumps([json.dumps(["objective", "Objective0"]), "max"])]
CELL_SELECTION_STYLES = [
    {
        "if": {"state": state},
        "backgroundColor": "#edf4ff",
        "border": "1px solid #b9d3f7",
    }
    for state in ("active", "selected")
]


@dataclass
class OptunaStudyOverview:
    """Study metadata and scalar totals, without materializing Optuna trials."""

    path: Path
    name: str
    objectives: List[Objective]
    history: List[Trial] = field(default_factory=list)
    metadata_url: Optional[str] = None
    study_attributes: Dict[str, Any] = field(default_factory=dict)
    database: Optional[Path] = None
    study_name: Optional[str] = None
    study_id: Optional[int] = None
    successful_count: int = 0
    total_duration: Optional[float] = None
    start_datetime: Optional[str] = None
    user_attribute_names: List[str] = field(default_factory=list)
    prefix = "Optuna"

    def get_meta(self) -> Dict[str, Any]:
        """Expose study metadata while keeping trial attributes in their original data."""
        return {"optuna_user_attrs": self.study_attributes}

    def get_objectives(self) -> List[Objective]:
        """Match the importer's objective names and duration column."""
        return self.objectives

    def metric_samples(self, metrics: List[str]) -> Dict[str, List[float]]:
        """Read only requested values, without parameters, distributions, or trial objects."""
        values: Dict[str, List[float]] = {}
        if not metrics:
            return values
        with closing(sqlite3.connect(self.database)) as connection:
            for metric in metrics:
                source, name = json.loads(metric)
                if source == "user_attr":
                    samples = (
                        json.loads(value)
                        for (value,) in connection.execute(
                            "SELECT a.value_json FROM trial_user_attributes a "
                            "JOIN trials t ON t.trial_id = a.trial_id "
                            "WHERE t.study_id = ? AND t.state = 'COMPLETE' AND a.key = ?",
                            (self.study_id, name),
                        )
                    )
                elif name == "Time" and name == self.objectives[-1].name:
                    samples = (
                        _trial_duration(start, end)
                        for start, end in connection.execute(
                            "SELECT datetime_start, datetime_complete FROM trials "
                            "WHERE study_id = ? AND state = 'COMPLETE'",
                            (self.study_id,),
                        )
                    )
                else:
                    index = next(
                        i for i, objective in enumerate(self.objectives) if objective.name == name
                    )
                    samples = (
                        value
                        for (value,) in connection.execute(
                            "SELECT v.value FROM trial_values v "
                            "JOIN trials t ON t.trial_id = v.trial_id "
                            "WHERE t.study_id = ? AND t.state = 'COMPLETE' "
                            "AND v.objective = ? AND v.value_type = 'FINITE'",
                            (self.study_id, index),
                        )
                    )
                values[metric] = [float(value) for value in samples if _numeric(value)]
        return values


def _trial_duration(start: Optional[str], end: Optional[str]) -> Optional[float]:
    if start is None or end is None:
        return None
    duration = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    return duration if duration >= 0 else None


def _load_optuna_overviews(database: Path, working_directory: Path) -> List[OptunaStudyOverview]:
    """Read metadata and grouped totals with a fixed number of SQLite queries."""
    with closing(sqlite3.connect(database)) as connection:
        connection.create_function("trial_duration", 2, _trial_duration)
        studies = {
            study_id: OptunaStudyOverview(
                path=get_optuna_snapshot_path(database, name, working_directory),
                name=name,
                objectives=[],
                database=database,
                study_name=name,
                study_id=study_id,
                metadata_url=study_metadata_url(database, name),
            )
            for study_id, name in connection.execute(
                "SELECT study_id, study_name FROM studies ORDER BY study_name"
            )
        }
        names = {
            study_id: json.loads(value)
            for study_id, value in connection.execute(
                "SELECT study_id, value_json FROM study_system_attributes WHERE key = 'study:metric_names'"
            )
        }
        for study_id, index, direction in connection.execute(
            "SELECT study_id, objective, direction FROM study_directions ORDER BY study_id, objective"
        ):
            studies[study_id].objectives.append(
                Objective(
                    names[study_id][index] if study_id in names else f"Objective{index}",
                    optimize="lower" if direction == "MINIMIZE" else "upper",
                )
            )
        for study in studies.values():
            study.objectives.append(Objective("Time"))
        for study_id, key, value in connection.execute(
            "SELECT study_id, key, value_json FROM study_user_attributes"
        ):
            studies[study_id].study_attributes[key] = json.loads(value)
        for study_id, count, duration, start in connection.execute(
            "SELECT study_id, SUM(CASE WHEN state = 'COMPLETE' THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN state = 'COMPLETE' THEN trial_duration(datetime_start, datetime_complete) END), "
            "MIN(datetime_start) FROM trials GROUP BY study_id"
        ):
            studies[study_id].successful_count = count
            studies[study_id].total_duration = duration
            studies[study_id].start_datetime = start
        for study_id, key in connection.execute(
            "SELECT DISTINCT t.study_id, a.key FROM trial_user_attributes a "
            "JOIN trials t ON t.trial_id = a.trial_id WHERE t.state = 'COMPLETE' "
            "AND CASE WHEN json_valid(a.value_json) THEN json_type(a.value_json) END IN ('integer', 'real') "
            "AND CAST(a.value_json AS REAL) BETWEEN -1.7976931348623157e308 AND 1.7976931348623157e308"
        ):
            studies[study_id].user_attribute_names.append(key)
    return list(studies.values())


def _numeric(value: Any) -> bool:
    """Accept finite scalar numbers, excluding booleans and structured values."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def summarize_runs(
    runs: List[Union[AbstractRun, OptunaStudyOverview]], selected: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Discover columns, calculating only requested statistics over successful trials."""
    rows = []
    metrics: Dict[str, str] = {}
    study_attributes: Dict[str, str] = {}
    sources = {}
    selected_metrics = {json.loads(key)[0] for key in selected or [] if key.startswith("[")}
    for run in runs:
        successful = [trial for trial in run.history if trial.status == Status.SUCCESS]
        row = {
            "id": str(run.path),
            "run": run.name,
            "path": str(run.path or ""),
            "optimizer": run.prefix,
            "successful": len(successful),
            "metadata_url": getattr(run, "metadata_url", None),
            "database": (
                str(run.database) if isinstance(run, OptunaStudyOverview) and run.database else None
            ),
            "study_name": getattr(run, "study_name", None),
            "start_datetime": getattr(run, "start_datetime", None),
        }
        durations = [
            trial.end_time - trial.start_time
            for trial in successful
            if _numeric(trial.start_time)
            and _numeric(trial.end_time)
            and trial.end_time >= trial.start_time
        ]
        row["duration"] = sum(durations) if durations else None
        if isinstance(run, OptunaStudyOverview):
            row["successful"] = run.successful_count
            row["duration"] = run.total_duration
            sources[row["id"]] = {
                "database": str(run.database),
                "study_id": run.study_id,
                "objectives": [objective.name for objective in run.objectives],
                "user_attributes": run.user_attribute_names,
            }
        for name, value in run.get_meta().get("optuna_user_attrs", {}).items():
            key = json.dumps(["study_attr", name])
            study_attributes[key] = name
            row[key] = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list, bool))
                else value
            )
        values: Dict[str, List[float]] = {}
        promoted = run.get_meta().get("user_attr_objectives", {})
        for index, objective in enumerate(run.get_objectives()):
            source = "user_attr" if objective.name in promoted else "objective"
            metric = json.dumps([source, objective.name])
            metrics[metric] = (
                f"{objective.name} ({'user attribute' if source == 'user_attr' else 'objective'})"
            )
            values[metric] = (
                [
                    float(trial.costs[index])
                    for trial in successful
                    if index < len(trial.costs) and _numeric(trial.costs[index])
                ]
                if selected is None or metric in selected_metrics
                else []
            )
        if isinstance(run, OptunaStudyOverview):
            for name in run.user_attribute_names:
                metric = json.dumps(["user_attr", name])
                metrics[metric] = f"{name} (user attribute)"
                values[metric] = []
        for trial in successful:
            for name, value in trial.additional.get("user_attrs", {}).items():
                if name in promoted or not _numeric(value):
                    continue
                metric = json.dumps(["user_attr", name])
                metrics[metric] = f"{name} (user attribute)"
                samples = values.setdefault(metric, [])
                if selected is None or metric in selected_metrics:
                    samples.append(float(value))
        if isinstance(run, OptunaStudyOverview):
            requested = [
                metric for metric in values if selected is None or metric in selected_metrics
            ]
            if requested:
                values.update(run.metric_samples(requested))
        for metric, samples in values.items():
            for statistic, aggregate in STATISTICS.items():
                key = json.dumps([metric, statistic])
                if selected is None or key in selected:
                    row[key] = aggregate(samples) if samples else None
        rows.append(row)

    columns = [dict(column) for column in BASE_COLUMNS]
    study_options = [
        {"label": name, "value": key}
        for key, name in sorted(study_attributes.items(), key=lambda item: item[1].casefold())
    ]
    columns.extend(
        {"id": item["value"], "name": item["label"], "type": "any"} for item in study_options
    )
    for metric, name in sorted(metrics.items(), key=lambda item: item[1].casefold()):
        columns.extend(
            {
                "id": json.dumps([metric, statistic]),
                "name": f"{name} · {statistic}",
                "type": "numeric",
            }
            for statistic in STATISTICS
        )
    rows.sort(key=lambda row: row["start_datetime"] or "", reverse=True)
    return {
        "rows": rows,
        "columns": columns,
        "study_attributes": study_options,
        "sources": sources,
        "computed_columns": [
            column["id"]
            for column in columns
            if column["id"] not in {item["id"] for item in BASE_COLUMNS}
            and column["type"] == "numeric"
            and (selected is None or column["id"] in selected)
        ],
        "metrics": [{"label": name, "value": metric} for metric, name in metrics.items()],
    }


def load_overview_runs(
    handler: Any, database_value: Optional[str] = None
) -> Tuple[List[Union[AbstractRun, OptunaStudyOverview]], List[str]]:
    """Load imported runs, available directories, and all studies in the selected database."""
    runs: Dict[str, Union[AbstractRun, OptunaStudyOverview]] = {
        str(run.path): run for run in handler.get_runs(apply_filter=False)
    }
    errors = []
    if database_value:
        try:
            database = resolve_optuna_database(database_value, handler.get_working_directory())
            for study in _load_optuna_overviews(database, handler.get_working_directory()):
                path = str(study.path)
                if path in runs:
                    study.name = runs[path].name
                runs[path] = study
        except Exception as exception:
            errors.append(f"Could not read the Optuna database: {exception}")
    for path in handler.get_available_run_paths():
        if path in runs or not handler.is_run(path):
            continue
        run = handler.update_run(path)
        if run is None:
            errors.append(f"Could not load {path}.")
        else:
            runs[path] = run
    return list(runs.values()), errors


def load_selected_statistics(
    handler: Any, data: Dict[str, Any], selected: List[str]
) -> Dict[str, Any]:
    """Fill missing metric columns while retaining previously computed values."""
    available = {column["id"] for column in data["columns"] if column["type"] == "numeric"}
    needed = [
        key
        for key in selected
        if key in available
        and key not in data.get("computed_columns", [])
        and key not in {column["id"] for column in BASE_COLUMNS}
    ]
    if not needed:
        return data
    updated_rows = []
    for row in data["rows"]:
        source = data.get("sources", {}).get(row["id"])
        if source:
            run = OptunaStudyOverview(
                path=Path(row["id"]),
                name=row["run"],
                objectives=[Objective(name) for name in source["objectives"]],
                database=Path(source["database"]),
                study_id=source["study_id"],
                user_attribute_names=source["user_attributes"],
            )
        else:
            run = handler.update_run(row["id"])
        additions = summarize_runs([run], selected=needed)["rows"][0] if run is not None else {}
        updated_rows.append({**row, **{key: additions.get(key) for key in needed}})
    return {
        **data,
        "rows": updated_rows,
        "computed_columns": data.get("computed_columns", []) + needed,
    }


def _database_revision(database_value: Optional[str], handler: Any) -> Optional[List[Any]]:
    """Reuse a summary until SQLite or its write-ahead log changes."""
    if not database_value:
        return None
    database = resolve_optuna_database(database_value, handler.get_working_directory())
    revision: List[Any] = [str(database), str(handler.get_working_directory())]
    for path in (database, Path(str(database) + "-wal")):
        if path.exists():
            stat = path.stat()
            revision.append([stat.st_mtime_ns, stat.st_size])
        else:
            revision.append(None)
    return revision


def table_view(data: Optional[Dict[str, Any]], selected: Optional[List[str]]) -> Tuple[List, List]:
    """Project cached summary rows onto the selected columns without loading runs again."""
    data = data or {"rows": [], "columns": BASE_COLUMNS}
    available = {column["id"]: column for column in data["columns"]}
    columns = [dict(available[key]) for key in ["run"] + (selected or []) if key in available]
    for column in columns:
        column["deletable"] = column["id"] != "run"
        if column["type"] == "numeric":
            column["format"] = (
                Format(precision=0, scheme=Scheme.fixed)
                if column["id"] == "successful"
                else Format(precision=6, scheme=Scheme.decimal_or_exponent)
            )
    return columns, data["rows"]


def group_filter_rows(
    data: Optional[Dict[str, Any]],
    enabled: bool,
    group_by: Optional[List[str]],
    target: Optional[str],
    statistic: str,
) -> List[Dict]:
    """Keep one original row per group; ties retain the first row in default date order."""
    data = data or {"rows": [], "columns": []}
    rows = data["rows"]
    if not enabled or not group_by or not target or statistic not in {"min", "max"}:
        return rows
    numeric = {column["id"] for column in data["columns"] if column["type"] == "numeric"}
    if target not in numeric:
        return rows
    base = {column["id"] for column in BASE_COLUMNS}
    if any(
        key in numeric - base and key not in data.get("computed_columns", [])
        for key in [*group_by, target]
    ):
        return rows  # Keep the current rows visible while a required statistic is loaded.
    winners = {}
    for row in rows:
        group, value = tuple(row.get(key) for key in group_by), row.get(target)
        if any(
            value is None or (isinstance(value, float) and not math.isfinite(value))
            for value in group
        ) or not _numeric(value):
            continue
        previous = winners.get(group)
        if previous is None or (
            value > previous[target] if statistic == "max" else value < previous[target]
        ):
            winners[group] = row
    selected_ids = {row["id"] for row in winners.values()}
    return [row for row in rows if row["id"] in selected_ids]


def column_choices(data: Optional[Dict[str, Any]], query: str, selected: List[str]) -> List[Any]:
    """Group searchable attributes and reveal statistics inside each attribute."""
    query = (query or "").strip().casefold()
    groups = []
    base = [
        {"label": column["name"], "value": column["id"]}
        for column in BASE_COLUMNS
        if column["id"] != "run" and query in column["name"].casefold()
    ]
    if base:
        groups.append(
            html.Details(
                [
                    html.Summary("Run information"),
                    dcc.Checklist(
                        id={"type": "all-runs-column-choice", "metric": "base"},
                        options=base,
                        value=[item["value"] for item in base if item["value"] in selected],
                        labelStyle={"display": "flex"},
                        className="all-runs-column-checklist",
                    ),
                ],
                open=True,
            )
        )
    study_options = [
        item
        for item in (data or {}).get("study_attributes", [])
        if query in item["label"].casefold()
    ]
    if study_options:
        groups.append(
            html.Details(
                [
                    html.Summary(f"Study attributes ({len(study_options)})"),
                    dcc.Checklist(
                        id={"type": "all-runs-column-choice", "metric": "study_attr"},
                        options=study_options,
                        value=[
                            item["value"] for item in study_options if item["value"] in selected
                        ],
                        labelStyle={"display": "flex"},
                        className="all-runs-column-checklist",
                    ),
                ],
                open=True,
            )
        )
    for source, title in [("objective", "Objectives"), ("user_attr", "User attributes")]:
        attributes = []
        for metric in sorted(
            (data or {}).get("metrics", []), key=lambda item: item["label"].casefold()
        ):
            kind, name = json.loads(metric["value"])
            if kind != source or query not in name.casefold():
                continue
            options = [
                {"label": statistic, "value": json.dumps([metric["value"], statistic])}
                for statistic in STATISTICS
            ]
            chosen = [item["value"] for item in options if item["value"] in selected]
            attributes.append(
                html.Details(
                    [
                        html.Summary(
                            [
                                html.Span(name, title=name),
                            ]
                        ),
                        dcc.Checklist(
                            id={"type": "all-runs-column-choice", "metric": metric["value"]},
                            options=options,
                            value=chosen,
                            labelStyle={"display": "flex"},
                            className="all-runs-column-checklist",
                        ),
                    ],
                    className="all-runs-metric-choice",
                )
            )
        if attributes:
            groups.append(
                html.Details(
                    [
                        html.Summary(f"{title} ({len(attributes)})"),
                        *attributes,
                    ],
                    open=True,
                )
            )
    return groups or [html.Div("No matching columns.", className="text-muted p-3")]


class AllRunsOverview(Plugin):
    """Study comparison and selection independent of the active analytical run."""

    id = "all_runs_overview"
    name = "All Runs Overview"
    icon = "fas fa-table"
    requires_run = False

    @interactive
    def register_callbacks(self) -> None:
        """Load on demand and update visible columns using the browser's summary store."""
        from deepcave import app, run_handler

        @app.callback(
            Output("all-runs-all-studies", "style"),
            Output("all-runs-selected-studies", "style"),
            Input("all-runs-tabs", "active_tab"),
        )
        def show_studies(tab: str) -> Any:
            return (
                {"display": "none" if tab == "selected-studies" else "block"},
                {"display": "block" if tab == "selected-studies" else "none"},
            )

        @app.callback(
            Output("all-runs-group-by", "options"),
            Output("all-runs-group-target", "options"),
            Input("all-runs-data", "data"),
        )
        def group_filter_options(data: Optional[Dict]) -> Any:
            if not data:
                raise PreventUpdate
            columns = (data or {}).get("columns", [])
            groups = [{"label": column["name"], "value": column["id"]} for column in columns]
            targets = [
                {"label": column["name"], "value": column["id"]}
                for column in columns
                if column["type"] == "numeric"
            ]
            return groups, targets

        @app.callback(
            Output("all-runs-group-by", "disabled"),
            Output("all-runs-group-target", "disabled"),
            Output("all-runs-group-statistic", "disabled"),
            Input("all-runs-group-enabled", "value"),
        )
        def toggle_group_filter(enabled: bool) -> Any:
            return not enabled, not enabled, not enabled

        @app.callback(
            Output("all-runs-view-count", "children"),
            Input("all-runs-table", "derived_virtual_data"),
        )
        def count_visible_runs(rows: Optional[List[Dict]]) -> str:
            count = len(rows or [])
            return f"{count} {'run' if count == 1 else 'runs'}"

        @app.callback(
            Output("all-runs-active-group", "style"),
            Output("all-runs-group-chips", "children"),
            Input("all-runs-data", "data"),
            Input("all-runs-group-enabled", "value"),
            Input("all-runs-group-by", "value"),
            Input("all-runs-group-target", "value"),
            Input("all-runs-group-statistic", "value"),
        )
        def show_group_chips(
            data: Optional[Dict],
            enabled: bool,
            group_by: Optional[List[str]],
            target: Optional[str],
            statistic: str,
        ) -> Any:
            if not enabled or not group_by or not target:
                return {"display": "none"}, []
            labels = {column["id"]: column["name"] for column in (data or {}).get("columns", [])}
            if any(key not in labels for key in [*group_by, target]):
                return {"display": "none"}, []
            target_name = (
                json.loads(json.loads(target)[0])[1] if target.startswith("[") else labels[target]
            )
            chips = [html.Span(labels[key], className="all-runs-group-chip") for key in group_by]
            chips.append(
                html.Span(
                    f"{target_name} · {statistic}",
                    title=labels[target],
                    className="all-runs-group-chip",
                )
            )
            return {"display": "flex"}, chips

        @app.callback(
            Output("all-runs-group-by", "value"),
            Input("all-runs-group-reset", "n_clicks"),
            prevent_initial_call=True,
        )
        def reset_grouping(clicks: Optional[int]) -> List[str]:
            if not clicks:
                raise PreventUpdate
            return []

        @app.callback(
            Output("all-runs-data", "data"),
            Output("all-runs-status", "children"),
            Input("all-runs-load", "n_clicks"),
            Input("general-optuna-discovery", "data"),
            State("all-runs-columns", "data"),
            State("all-runs-data", "data"),
            State("all-runs-group-enabled", "value"),
            State("all-runs-group-by", "value"),
            State("all-runs-group-target", "value"),
            prevent_initial_call=True,
        )
        def load_runs(
            n_clicks: Optional[int],
            discovery: Optional[Dict],
            selected: Optional[List[str]] = None,
            current: Optional[Dict] = None,
            group_enabled: bool = False,
            group_by: Optional[List[str]] = None,
            target: Optional[str] = None,
        ) -> Any:
            if not n_clicks and discovery is None:
                raise PreventUpdate
            if discovery and discovery.get("error"):
                return (
                    summarize_runs([], selected=[]),
                    "Choose a valid Optuna database to load studies.",
                )
            database_value = (discovery or {}).get("database")
            required = list(selected or [])
            if group_enabled:
                required.extend(
                    key for key in [*(group_by or []), target] if key and key not in required
                )
            revision = _database_revision(database_value, run_handler)
            if revision is not None and current and current.get("database_revision") == revision:
                summary = load_selected_statistics(run_handler, current, required)
                errors = current.get("load_errors", [])
            else:
                runs, errors = load_overview_runs(run_handler, database_value)
                summary = summarize_runs(runs, selected=required)
                summary["database_revision"] = revision
                summary["load_errors"] = errors
            message = f"{len(summary['rows'])} runs loaded"
            if not summary["rows"]:
                message += " No studies found. Choose an Optuna database above."
            if errors:
                message += " " + " ".join(errors)
            return no_update if summary is current else summary, message

        @app.callback(
            Output("all-runs-data", "data"),
            Input("all-runs-columns", "data"),
            Input("all-runs-group-enabled", "value"),
            Input("all-runs-group-by", "value"),
            Input("all-runs-group-target", "value"),
            State("all-runs-data", "data"),
            prevent_initial_call=True,
        )
        def load_statistics(
            selected: Optional[List[str]],
            enabled: bool = False,
            group_by: Optional[List[str]] = None,
            target: Optional[str] = None,
            data: Optional[Dict] = None,
        ) -> Any:
            if not data:
                raise PreventUpdate
            required = list(selected or [])
            if enabled:
                required.extend(
                    key for key in [*(group_by or []), target] if key and key not in required
                )
            updated = load_selected_statistics(run_handler, data, required)
            return no_update if updated is data else updated

        @app.callback(
            Output("all-runs-table", "columns"),
            Output("all-runs-table", "data"),
            Output("all-runs-table", "tooltip_data"),
            Input("all-runs-data", "data"),
            Input("all-runs-columns", "data"),
            Input("all-runs-group-enabled", "value"),
            Input("all-runs-group-by", "value"),
            Input("all-runs-group-target", "value"),
            Input("all-runs-group-statistic", "value"),
            State("all-runs-table", "data"),
        )
        def select_columns(
            data: Optional[Dict[str, Any]],
            selected: Optional[List[str]],
            enabled: bool = False,
            group_by: Optional[List[str]] = None,
            target: Optional[str] = None,
            statistic: str = "max",
            current_rows: Optional[List[Dict]] = None,
        ) -> Any:
            columns, rows = table_view(data, selected)
            rows = group_filter_rows(data, enabled, group_by, target, statistic)
            tooltips = [{"run": {"value": row["run"], "type": "text"}} for row in rows]
            if current_rows is not None and [row["id"] for row in rows] == [
                row["id"] for row in current_rows
            ]:
                changes = Patch()
                changed = False
                for index, row in enumerate(rows):
                    for key, value in row.items():
                        if key not in current_rows[index] or current_rows[index][key] != value:
                            changes[index][key] = value
                            changed = True
                    for key in current_rows[index].keys() - row.keys():
                        del changes[index][key]
                        changed = True
                if all(row["run"] == current["run"] for row, current in zip(rows, current_rows)):
                    tooltips = no_update
                return columns, changes if changed else no_update, tooltips
            return columns, rows, tooltips

        @app.callback(
            Output("all-runs-metadata-navigation", "href"),
            Input("all-runs-table", "active_cell"),
            State("all-runs-table", "derived_viewport_data"),
            prevent_initial_call=True,
        )
        def open_run_metadata(active_cell: Optional[Dict], rows: Optional[List[Dict]]) -> str:
            if not active_cell or active_cell.get("column_id") != "run":
                raise PreventUpdate
            index = active_cell.get("row")
            if not isinstance(index, int) or not rows or not 0 <= index < len(rows):
                raise PreventUpdate
            url = rows[index].get("metadata_url")
            if not url:
                raise PreventUpdate
            return url

        @app.callback(
            Output("all-runs-table", "style_data_conditional"),
            Input("all-runs-table", "sort_by"),
        )
        def highlight_sorted_columns(sort_by: Optional[List[Dict[str, str]]]) -> List[Dict]:
            return [
                {"if": {"column_id": column["column_id"]}, "backgroundColor": "#f0f6ff"}
                for column in sort_by or []
            ] + CELL_SELECTION_STYLES

        @app.callback(
            Output("all-runs-column-tree", "children"),
            Input("all-runs-data", "data"),
            Input("all-runs-column-search", "value"),
            Input("all-runs-columns-toggle", "n_clicks"),
            State("all-runs-table", "columns"),
        )
        def render_column_choices(
            data: Optional[Dict[str, Any]],
            query: str,
            clicks: int,
            columns: List[Dict],
        ) -> List[Any]:
            return column_choices(data, query, [column["id"] for column in columns or []])

        @app.callback(
            Output("all-runs-columns", "data"),
            Input({"type": "all-runs-column-choice", "metric": ALL}, "value"),
            State({"type": "all-runs-column-choice", "metric": ALL}, "id"),
            State({"type": "all-runs-column-choice", "metric": ALL}, "options"),
            State("all-runs-table", "columns"),
            prevent_initial_call=True,
        )
        def update_column_choices(
            values: List[List[str]],
            ids: List[Dict],
            options: List[List[Dict]],
            columns: List[Dict],
        ) -> Any:
            trigger = ctx.triggered_id
            if not isinstance(trigger, dict):
                raise PreventUpdate
            selected = [column["id"] for column in columns or [] if column["id"] != "run"]
            index = next((i for i, item in enumerate(ids) if item == trigger), None)
            if index is None:
                raise PreventUpdate
            visible = {option["value"] for option in options[index]}
            chosen = values[index] or []
            updated = [key for key in selected if key not in visible or key in chosen]
            updated.extend(key for key in chosen if key not in updated)
            return no_update if updated == selected else updated

    def __call__(
        self,
        render_button: bool = False,
        database_controls: Optional[List[Any]] = None,
        selected_studies: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Render the database picker, overview, and selected-study management tabs."""
        columns, rows = table_view(None, DEFAULT_COLUMNS)
        return [
            html.H1(self.name),
            *(database_controls or []),
            dcc.Location(id="all-runs-metadata-navigation", refresh=True),
            dbc.Tabs(
                [
                    dbc.Tab(label="All Studies", tab_id="all-studies"),
                    dbc.Tab(label="Selected Studies", tab_id="selected-studies"),
                ],
                id="all-runs-tabs",
                active_tab="all-studies",
                className="mb-3",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Runs", className="all-runs-table-title"),
                                    html.Span(
                                        "0 runs",
                                        id="all-runs-view-count",
                                        className="all-runs-view-count",
                                    ),
                                ],
                                className="all-runs-toolbar-title",
                            ),
                            html.Details(
                                [
                                    html.Summary(
                                        [
                                            html.I(className="fas fa-columns me-2"),
                                            "Columns",
                                            html.I(className="fas fa-chevron-down ms-2"),
                                        ],
                                        id="all-runs-columns-toggle",
                                        n_clicks=0,
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "Search columns",
                                                htmlFor="all-runs-column-search",
                                                className="visually-hidden",
                                            ),
                                            dbc.Input(
                                                id="all-runs-column-search",
                                                type="search",
                                                placeholder="Search columns",
                                                value="",
                                            ),
                                            html.Div(
                                                id="all-runs-column-tree",
                                                className="all-runs-column-tree",
                                            ),
                                            html.Div(
                                                "Select statistics to show as columns.",
                                                className="all-runs-column-footer",
                                            ),
                                        ],
                                        className="all-runs-column-panel",
                                    ),
                                ],
                                className="all-runs-column-menu",
                            ),
                            html.Details(
                                [
                                    html.Summary(
                                        [
                                            html.I(className="fas fa-object-group me-2"),
                                            "Group by",
                                            html.I(className="fas fa-chevron-down ms-2"),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "Groupby Col",
                                                                html_for="all-runs-group-by",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="all-runs-group-by",
                                                                options=[],
                                                                value=[],
                                                                multi=True,
                                                                placeholder="Select columns",
                                                                disabled=True,
                                                            ),
                                                        ],
                                                        md=12,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "Target Col",
                                                                html_for="all-runs-group-target",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="all-runs-group-target",
                                                                options=[
                                                                    {
                                                                        "label": "Objective0 (objective) · max",
                                                                        "value": DEFAULT_COLUMNS[
                                                                            -1
                                                                        ],
                                                                    }
                                                                ],
                                                                value=DEFAULT_COLUMNS[-1],
                                                                placeholder="Select a numeric column",
                                                                disabled=True,
                                                            ),
                                                        ],
                                                        md=12,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "Statistic",
                                                                html_for="all-runs-group-statistic",
                                                            ),
                                                            dbc.Select(
                                                                id="all-runs-group-statistic",
                                                                options=[
                                                                    {
                                                                        "label": value,
                                                                        "value": value,
                                                                    }
                                                                    for value in ("max", "min")
                                                                ],
                                                                value="max",
                                                                disabled=True,
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Switch(
                                                            id="all-runs-group-enabled",
                                                            label="Group filter",
                                                            value=True,
                                                            className="mb-0 text-nowrap",
                                                        ),
                                                        md=6,
                                                        className="pb-2",
                                                    ),
                                                ],
                                                className="g-3 align-items-end",
                                            ),
                                        ],
                                        className="all-runs-column-panel all-runs-group-panel",
                                    ),
                                ],
                                id="all-runs-group-menu",
                                className="all-runs-column-menu",
                            ),
                            dcc.Store(id="all-runs-columns", data=DEFAULT_COLUMNS),
                        ],
                        className="all-runs-toolbar",
                    ),
                    html.Div(
                        [
                            html.Div(id="all-runs-group-chips", className="all-runs-group-chips"),
                            html.Button(
                                [html.I(className="fas fa-times me-1"), "Reset"],
                                id="all-runs-group-reset",
                                n_clicks=0,
                                type="button",
                                className="all-runs-group-reset",
                                **{"aria-label": "Reset grouping"},
                            ),
                        ],
                        id="all-runs-active-group",
                        className="all-runs-active-group",
                        style={"display": "none"},
                    ),
                    html.Div(
                        [
                            dcc.Store(id="all-runs-data"),
                            dash_table.DataTable(
                                id="all-runs-table",
                                columns=columns,
                                data=rows,
                                row_selectable="multi",
                                selected_rows=[],
                                sort_action="native",
                                sort_mode="multi",
                                filter_action="native",
                                page_action="native",
                                page_size=25,
                                fill_width=False,
                                merge_duplicate_headers=True,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "fontFamily": "Arial, sans-serif",
                                    "fontSize": "13px",
                                    "textAlign": "left",
                                    "padding": "7px 12px",
                                    "height": "34px",
                                    "minWidth": "100px",
                                    "width": "140px",
                                    "maxWidth": "180px",
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                    "whiteSpace": "nowrap",
                                    "color": "#182230",
                                    "backgroundColor": "#fff",
                                    "border": "none",
                                    "borderBottom": "1px solid #e9eaec",
                                },
                                style_cell_conditional=[
                                    {
                                        "if": {"column_id": "run"},
                                        "width": "260px",
                                        "minWidth": "260px",
                                        "maxWidth": "260px",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                        "whiteSpace": "nowrap",
                                        "color": "#2563eb",
                                    },
                                    {
                                        "if": {"column_type": "numeric"},
                                        "width": "auto",
                                        "minWidth": "0px",
                                        "whiteSpace": "nowrap",
                                    },
                                    {
                                        "if": {"column_id": "successful"},
                                        "width": "150px",
                                        "minWidth": "150px",
                                        "maxWidth": "150px",
                                    },
                                ],
                                tooltip_duration=None,
                                style_header={
                                    "fontWeight": "400",
                                    "color": "#586474",
                                    "backgroundColor": "#f7f7f8",
                                    "borderBottom": "1px solid #e5e7eb",
                                },
                                style_filter={
                                    "backgroundColor": "#fafafa",
                                    "color": "#7c8795",
                                    "fontSize": "12px",
                                    "height": "30px",
                                },
                                style_data_conditional=CELL_SELECTION_STYLES,
                            ),
                        ],
                    ),
                ],
                id="all-runs-all-studies",
                className="all-runs-grid",
            ),
            html.Div(
                selected_studies or [],
                id="all-runs-selected-studies",
                style={"display": "none"},
            ),
        ]
