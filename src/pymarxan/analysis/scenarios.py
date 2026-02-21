"""Scenario comparison for conservation planning.

Store named scenarios (label + Solution + parameters) and compare them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pymarxan.solvers.base import Solution


@dataclass
class Scenario:
    """A named solver result with its configuration."""

    name: str
    solution: Solution
    parameters: dict = field(default_factory=dict)


class ScenarioSet:
    """Collection of named scenarios for comparison."""

    def __init__(self) -> None:
        self._scenarios: list[Scenario] = []

    def __len__(self) -> int:
        return len(self._scenarios)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._scenarios]

    def add(
        self, name: str, solution: Solution, parameters: dict | None = None
    ) -> None:
        self._scenarios.append(
            Scenario(name=name, solution=solution, parameters=parameters or {})
        )

    def get(self, name: str) -> Scenario:
        for s in self._scenarios:
            if s.name == name:
                return s
        raise KeyError(f"Scenario '{name}' not found")

    def remove(self, name: str) -> None:
        self._scenarios = [s for s in self._scenarios if s.name != name]

    def compare(self) -> pd.DataFrame:
        """Return a DataFrame comparing all scenarios."""
        rows = []
        for s in self._scenarios:
            sol = s.solution
            rows.append({
                "name": s.name,
                "cost": sol.cost,
                "boundary": sol.boundary,
                "objective": sol.objective,
                "n_selected": sol.n_selected,
                "all_targets_met": sol.all_targets_met,
                **s.parameters,
            })
        return pd.DataFrame(rows)

    def overlap_matrix(self) -> np.ndarray:
        """Compute pairwise selection overlap (Jaccard index)."""
        n = len(self._scenarios)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                si = self._scenarios[i].solution.selected
                sj = self._scenarios[j].solution.selected
                intersection = np.sum(si & sj)
                union = np.sum(si | sj)
                matrix[i, j] = intersection / union if union > 0 else 0.0
        return matrix
