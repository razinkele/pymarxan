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
    """Compute how often each planning unit is selected across solutions.

    With proportion decisions ``counts`` is Σ x (float) and ``frequencies`` the mean x.
    """
    if not solutions:
        return SelectionFrequency(
            frequencies=np.array([]),
            counts=np.array([]),
            n_solutions=0,
            best_solution=None,
        )

    n_pu = len(solutions[0].selected)
    dtype = np.float64 if any(s.is_fractional for s in solutions) else int
    counts = np.zeros(n_pu, dtype=dtype)

    for sol in solutions:
        counts += sol.selected.astype(dtype)

    n = len(solutions)
    frequencies = counts / n
    best = min(solutions, key=lambda s: s.objective)

    return SelectionFrequency(
        frequencies=frequencies,
        counts=counts,
        n_solutions=n,
        best_solution=best,
    )
