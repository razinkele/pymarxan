"""Portfolio analysis for collections of conservation planning solutions."""
from __future__ import annotations

import numpy as np

from pymarxan.solvers.base import Solution


def _validate_non_empty(solutions: list[Solution]) -> None:
    """Raise ValueError when the portfolio is empty."""
    if not solutions:
        raise ValueError("Portfolio must contain at least one solution.")


def selection_frequency(solutions: list[Solution]) -> np.ndarray:
    """Compute how often each PU is selected across solutions.

    Returns (n_pu,) float array with values in [0, 1].
    """
    _validate_non_empty(solutions)
    n_pu = len(solutions[0].selected)
    counts = np.zeros(n_pu, dtype=np.float64)
    for sol in solutions:
        counts += sol.selected.astype(np.float64)
    return counts / len(solutions)


def best_solution(solutions: list[Solution]) -> Solution:
    """Return the solution with the lowest objective value."""
    _validate_non_empty(solutions)
    return min(solutions, key=lambda s: s.objective)


def gap_filter(
    solutions: list[Solution],
    gap_tolerance: float = 0.01,
) -> list[Solution]:
    """Keep solutions within *gap_tolerance* of the best objective.

    Cutoff: ``objective <= best + gap_tolerance * max(1, abs(best))``
    This is sign-safe for negative objectives.
    """
    _validate_non_empty(solutions)
    best_obj = min(s.objective for s in solutions)
    cutoff = best_obj + gap_tolerance * max(1.0, abs(best_obj))
    return [s for s in solutions if s.objective <= cutoff]


def solution_diversity(solutions: list[Solution]) -> float:
    """Compute Jaccard-based diversity across solution portfolio.

    Returns the mean pairwise Jaccard distance.  If fewer than two
    solutions are provided the diversity is 0.0.
    """
    _validate_non_empty(solutions)
    n = len(solutions)
    if n < 2:
        return 0.0

    total = 0.0
    pairs = 0
    for i in range(n):
        a = solutions[i].selected
        for j in range(i + 1, n):
            b = solutions[j].selected
            intersection = np.sum(a & b)
            union = np.sum(a | b)
            if union == 0:
                dist = 0.0
            else:
                dist = 1.0 - float(intersection) / float(union)
            total += dist
            pairs += 1
    return total / pairs


def summary_statistics(solutions: list[Solution]) -> dict[str, float]:
    """Return summary stats for cost, boundary, objective, shortfall.

    Also includes ``n_solutions`` and ``mean_n_selected``.
    """
    _validate_non_empty(solutions)
    costs = np.array([s.cost for s in solutions])
    boundaries = np.array([s.boundary for s in solutions])
    objectives = np.array([s.objective for s in solutions])
    shortfalls = np.array([s.shortfall for s in solutions])
    n_selected = np.array([s.n_selected for s in solutions])

    stats: dict[str, float] = {"n_solutions": float(len(solutions))}
    for name, arr in [
        ("cost", costs),
        ("boundary", boundaries),
        ("objective", objectives),
        ("shortfall", shortfalls),
    ]:
        stats[f"{name}_mean"] = float(np.mean(arr))
        stats[f"{name}_std"] = float(np.std(arr))
        stats[f"{name}_min"] = float(np.min(arr))
        stats[f"{name}_max"] = float(np.max(arr))
    stats["mean_n_selected"] = float(np.mean(n_selected))
    return stats
