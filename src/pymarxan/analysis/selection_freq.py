"""Selection frequency analysis across multiple solver runs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymarxan.solvers.base import Solution


@dataclass
class SelectionFrequency:
    """Selection frequency results across multiple solutions."""
    frequencies: np.ndarray
    counts: np.ndarray
    n_solutions: int
    best_solution: Solution | None


def compute_selection_frequency(solutions: list[Solution]) -> SelectionFrequency:
    """Compute how often each planning unit is selected across solutions."""
    if not solutions:
        return SelectionFrequency(
            frequencies=np.array([]),
            counts=np.array([]),
            n_solutions=0,
            best_solution=None,
        )

    n_pu = len(solutions[0].selected)
    counts = np.zeros(n_pu, dtype=int)

    for sol in solutions:
        counts += sol.selected.astype(int)

    n = len(solutions)
    frequencies = counts / n
    best = min(solutions, key=lambda s: s.objective)

    return SelectionFrequency(
        frequencies=frequencies,
        counts=counts,
        n_solutions=n,
        best_solution=best,
    )
