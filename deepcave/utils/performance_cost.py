"""Trial-level data and Pareto efficiency for performance versus duration analysis."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
from itertools import groupby

from deepcave.runs import AbstractRun, Status
from deepcave.runs.group import Group
from deepcave.runs.trial import Trial


def _finite(value: Any) -> bool:
    """Return whether a value is a finite scalar number."""
    try:
        return value is not None and not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def extract_trial_duration(trial: Trial, time_id: Optional[int] = None) -> Optional[float]:
    """Return valid elapsed seconds, reusing a converter's precise Time objective if present."""
    if trial.status != Status.SUCCESS:
        return None
    if not _finite(trial.start_time) or not _finite(trial.end_time):
        return None
    if trial.end_time < trial.start_time:
        return None
    duration: Any = trial.end_time - trial.start_time
    if time_id is not None:
        duration = trial.costs[time_id] if time_id < len(trial.costs) else None
    if not _finite(duration) or float(duration) < 0:
        return None
    return float(duration)


def extract_trial_data(run: AbstractRun, objective_id: int) -> List[Dict[str, Any]]:
    """Extract successful trials with finite performance and valid elapsed time.

    Reuse the conventional Time objective when available: Optuna stores the precise timestamp
    difference there, while Run rounds start/end times. A missing Time value indicates missing
    timing information and must not fall back to the converter's placeholder timestamps.
    Otherwise derive duration from the run's start/end times. Zero durations remain valid on
    a linear axis; negative durations are invalid. Read group members directly to retain repeated
    evaluations that the group's configuration-level history can collapse.
    """
    sources = run.runs if isinstance(run, Group) else [run]
    rows = []
    for source_index, source in enumerate(sources):
        time_id = next(
            (
                i
                for i, objective in enumerate(source.get_objectives())
                if objective.name.lower() == "time"
            ),
            None,
        )
        for history_id, trial in enumerate(source.history):
            if objective_id < 0 or objective_id >= len(trial.costs):
                continue
            value = trial.costs[objective_id]
            duration = extract_trial_duration(trial, time_id)
            if not _finite(value) or duration is None:
                continue
            rows.append(
                {
                    "trial_id": trial.additional.get("optuna_trial_number", history_id),
                    "history_id": history_id,
                    "source_index": source_index,
                    "config_id": trial.config_id,
                    "budget": trial.budget,
                    "duration": duration,
                    "objective": float(value),
                }
            )
    return rows


def filter_trial_data(
    rows: Sequence[Dict[str, Any]], budget: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Select all eligible trials or only trials evaluated at the selected budget."""
    return [row for row in rows if budget is None or row["budget"] == budget]


def pareto_indices(points: Sequence[Tuple[float, float]], maximize: bool = False) -> List[int]:
    """Return non-dominated indices ordered by increasing computational cost.

    Cost is always minimized; performance is minimized unless maximize is true. Inputs must be
    finite. Equal pairs remain distinct efficient trials, while equal cost or equal performance
    alone can still imply strict dominance. Sorting followed by a scan takes O(n log n) time.
    """
    if any(not _finite(cost) or not _finite(value) for cost, value in points):
        raise ValueError("Pareto inputs must be finite numbers.")
    ordered = sorted(
        range(len(points)),
        key=lambda index: (points[index][0], -points[index][1] if maximize else points[index][1]),
    )
    efficient: List[int] = []
    best = math.inf
    for _cost, indices in groupby(ordered, key=lambda index: points[index][0]):
        group = list(indices)
        group_best = -points[group[0]][1] if maximize else points[group[0]][1]
        if group_best < best:
            efficient.extend(
                index
                for index in group
                if (-points[index][1] if maximize else points[index][1]) == group_best
            )
            best = group_best
    return efficient


def prepare_plot_data(rows: Sequence[Dict[str, Any]], maximize: bool = False) -> Dict[str, Any]:
    """Prepare one visible trial subset and its frontier for serialization and plotting.

    Use logarithmic duration only for strictly positive data spanning at least two orders of
    magnitude. Keeping zero-duration trials on a linear axis ensures both layers use the same set.
    """
    durations = [row["duration"] for row in rows]
    log_scale = bool(durations) and min(durations) > 0 and max(durations) / min(durations) >= 100
    return {
        "trials": list(rows),
        "pareto_indices": pareto_indices(
            [(row["duration"], row["objective"]) for row in rows], maximize=maximize
        ),
        "xaxis_type": "log" if log_scale else "linear",
    }
