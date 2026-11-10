"""Display original Optuna study and trial metadata without search-space conversion."""

from typing import Any, Dict, List, Optional

import json
from collections import Counter
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html

from deepcave.utils.hash import string_to_hash
from deepcave.utils.optuna import list_optuna_studies, resolve_optuna_database


def study_metadata_url(database: Path, study_name: str) -> str:
    """Identify a study within its source database without exposing file paths in URLs."""
    identifier = string_to_hash(f"{database.resolve()}\0{study_name}")
    return f"/runs/optuna/{identifier}"


def read_study_metadata(database: Path, study_name: str) -> Dict[str, Any]:
    """Read study attributes and every trial state directly from the source database."""
    import optuna
    from optuna.distributions import distribution_to_json

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{database.as_posix()}", skip_table_creation=True
    )
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        study_id = storage.get_study_id_from_name(study_name)
        trials = []
        for trial in study.get_trials(deepcopy=False):
            trials.append(
                {
                    "number": trial.number,
                    "trial_id": trial._trial_id,
                    "state": trial.state.name,
                    "values": trial.values,
                    "datetime_start": trial.datetime_start,
                    "datetime_complete": trial.datetime_complete,
                    "duration_seconds": (
                        trial.duration.total_seconds() if trial.duration is not None else None
                    ),
                    "params": trial.params,
                    "distributions": {
                        name: json.loads(distribution_to_json(distribution))
                        for name, distribution in trial.distributions.items()
                    },
                    "user_attrs": trial.user_attrs,
                    "system_attrs": trial.system_attrs,
                    "intermediate_values": trial.intermediate_values,
                }
            )
        return {
            "study": {
                "name": study.study_name,
                "study_id": study_id,
                "database": str(database),
                "directions": [direction.name for direction in study.directions],
                "metric_names": study.metric_names,
                "trial_count": len(trials),
                "trial_states": dict(Counter(trial["state"] for trial in trials)),
            },
            "user_attrs": study.user_attrs,
            "system_attrs": storage.get_study_system_attrs(study_id),
            "trials": trials,
        }
    finally:
        storage.remove_session()
        storage.engine.dispose()


def _metadata_value(value: Any) -> Any:
    """Preserve complete values and render structured attributes as nested tables."""
    if isinstance(value, dict):
        return _metadata_table(value, nested=True) if value else "—"
    if isinstance(value, (list, tuple)):
        if any(isinstance(item, (dict, list, tuple)) for item in value):
            return html.Ol([html.Li(_metadata_value(item)) for item in value], className="mb-0")
        return ", ".join(str(_metadata_value(item)) for item in value) if value else "—"
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _metadata_table(values: Dict[str, Any], nested: bool = False) -> dbc.Table:
    """Display metadata as attribute/value rows, with readable keys and wrapping values."""
    rows = [
        html.Tr(
            [
                html.Th(str(key), scope="row"),
                html.Td(_metadata_value(value)),
            ]
        )
        for key, value in values.items()
    ]
    children = (
        []
        if nested
        else [
            html.Thead(html.Tr([html.Th("Attribute", scope="col"), html.Th("Value", scope="col")]))
        ]
    )
    children.append(html.Tbody(rows or [html.Tr(html.Td("—", colSpan=2))]))
    return dbc.Table(
        children,
        bordered=False,
        className="optuna-metadata-table" + (" optuna-metadata-table--nested" if nested else ""),
    )


def optuna_metadata_page(identifier: str, handler: Any, database_value: Optional[str]) -> List[Any]:
    """Resolve only studies in the configured database and render their metadata."""
    back = dcc.Link(
        "← All Runs Overview", href="/plugins/all_runs_overview", className="mb-3 d-inline-block"
    )
    try:
        if not database_value:
            raise ValueError("Select an Optuna database in General first.")
        database = resolve_optuna_database(database_value, handler.get_working_directory())
        name = next(
            (
                name
                for name in list_optuna_studies(database)
                if study_metadata_url(database, name).rsplit("/", 1)[-1] == identifier
            ),
            None,
        )
        if name is None:
            raise ValueError("This study is not available in the selected Optuna database.")
        metadata = read_study_metadata(database, name)
    except Exception as exception:
        return [back, html.H1("Run Metadata"), dbc.Alert(str(exception), color="warning")]

    return [
        back,
        html.H1(name, className="h3 text-break"),
        html.H3("Study", className="mt-4"),
        _metadata_table(metadata["study"]),
        html.H3("Study user attributes", className="mt-4"),
        _metadata_table(metadata["user_attrs"]),
        html.H3("Study system attributes", className="mt-4"),
        _metadata_table(metadata["system_attrs"]),
        html.H3(f"Trials ({len(metadata['trials'])})", className="mt-4"),
        html.Div(
            [
                html.Details(
                    [
                        html.Summary(
                            f"Trial {trial['number']} · {trial['state']}", className="p-2"
                        ),
                        _metadata_table(trial),
                    ],
                    className="border rounded mb-2",
                )
                for trial in metadata["trials"]
            ]
        ),
    ]
