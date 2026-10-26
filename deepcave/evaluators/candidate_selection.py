"""Deterministic global KDE ranking and equal-width hybrid recommendation selection."""

from typing import Dict, List

import json
from collections import deque

import numpy as np

from deepcave.evaluators.candidate_geometry import BIN_COUNT, MIN_BIN_CANDIDATES

SELECTION_VERSION = 1


def configuration_key(config: Dict) -> str:
    """Identify exact native configurations independently of candidate origin."""
    return json.dumps(config, sort_keys=True)


def rank_candidates(candidates: List[Dict]) -> List[Dict]:
    """Rank a shared pool with a stable candidate-ID tie break, without changing scores."""
    if any(not np.isfinite(c["kde_score"]) for c in candidates):
        raise ValueError("Ranking requires finite KDE scores.")
    unique = {}
    for row in sorted(candidates, key=lambda c: (-c["kde_score"], c["id"])):
        unique.setdefault(configuration_key(row["configuration"]), row)
    return [dict(c, kde_rank=i + 1) for i, c in enumerate(unique.values())]


def select_hybrid_candidates(candidates: List[Dict], budget: int) -> List[Dict]:
    """Take ceil(N/2) global winners, then score-ordered bin 1..5 round robin.

    Supply is mandatory even for small budgets. A short or unsupplied pool is an
    explicit error, never a partial recommendation or a fallback strategy.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("Number of candidates must be a positive integer.")
    unique = {}
    for row in rank_candidates(candidates):
        if row.get("bin") not in range(1, BIN_COUNT + 1):
            raise ValueError("Hybrid selection requires distance bins 1..5.")
        unique.setdefault(configuration_key(row["configuration"]), row)
    ordered = list(unique.values())
    counts = [sum(c["bin"] == b for c in ordered) for b in range(1, BIN_COUNT + 1)]
    deficits = [
        f"Bin {b}: {n}/{MIN_BIN_CANDIDATES}"
        for b, n in enumerate(counts, 1)
        if n < MIN_BIN_CANDIDATES
    ]
    if deficits:
        raise ValueError("Insufficient candidate supply. " + "; ".join(deficits))
    if len(ordered) < budget:
        raise ValueError(
            f"Requested {budget} recommendations, but only {len(ordered)} unique candidates exist."
        )
    return select_ranked_hybrid(ordered, ordered, (budget + 1) // 2, budget)


def select_ranked_hybrid(
    global_candidates: List[Dict],
    exploration_candidates: List[Dict],
    global_count: int,
    budget: int,
) -> List[Dict]:
    """Combine supplied ranked arms; callers prepare geometry and enforce bin supply first.

    This selection-only primitive also supports comparison against archived proposal
    arms and different splits, without regenerating candidates or learning any model.
    """
    if not 0 <= global_count <= budget or budget < 1:
        raise ValueError("Global count must be between zero and the positive budget.")

    def ordered_unique(rows: List[Dict]) -> List[Dict]:
        if any(not np.isfinite(c["kde_score"]) for c in rows):
            raise ValueError("Selection requires finite KDE scores.")
        unique = {}
        for row in sorted(rows, key=lambda c: (-c["kde_score"], c.get("kde_rank", 0), c["id"])):
            unique.setdefault(configuration_key(row["configuration"]), row)
        return list(unique.values())

    global_rows = ordered_unique(global_candidates)
    exploration_rows = ordered_unique(exploration_candidates)
    if any(c.get("bin") not in range(1, BIN_COUNT + 1) for c in exploration_rows):
        raise ValueError("Exploration requires distance bins 1..5.")
    if len(global_rows) < global_count:
        raise ValueError("Insufficient global proposals for the requested budget.")
    selected = [dict(c, selection="Global KDE") for c in global_rows[:global_count]]
    used = {configuration_key(c["configuration"]) for c in selected}
    queues = [
        deque(
            c
            for c in exploration_rows
            if c["bin"] == b and configuration_key(c["configuration"]) not in used
        )
        for b in range(1, BIN_COUNT + 1)
    ]
    while len(selected) < budget:
        progress = False
        for queue in queues:
            if queue:
                selected.append(dict(queue.popleft(), selection="Exploration"))
                progress = True
                if len(selected) == budget:
                    break
        if not progress:
            raise ValueError("Exploration pool exhausted before the requested budget was filled.")
    return [dict(c, selection_rank=i + 1) for i, c in enumerate(selected)]
