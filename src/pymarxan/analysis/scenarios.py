"""Scenario comparison for conservation planning.

Store named scenarios (label + Solution + parameters) and compare them.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pymarxan.solvers.base import Solution

if TYPE_CHECKING:
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.solvers.base import Solver, SolverConfig


@dataclass
class Scenario:
    """A named solver result with its configuration."""

    name: str
    solution: Solution
    parameters: dict = field(default_factory=dict)
    feature_overrides: dict[int, dict[str, float]] | None = None


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
        self,
        name: str,
        solution: Solution,
        parameters: dict | None = None,
        feature_overrides: dict[int, dict[str, float]] | None = None,
    ) -> None:
        self._scenarios.append(
            Scenario(
                name=name,
                solution=solution,
                parameters=parameters or {},
                feature_overrides=feature_overrides,
            )
        )

    def get(self, name: str) -> Scenario:
        for s in self._scenarios:
            if s.name == name:
                return s
        raise KeyError(f"Scenario '{name}' not found")

    def remove(self, name: str) -> None:
        self._scenarios = [s for s in self._scenarios if s.name != name]

    def clone_scenario(
        self,
        source_name: str,
        new_name: str,
        parameter_overrides: dict | None = None,
        feature_overrides: dict[int, dict[str, float]] | None = None,
    ) -> Scenario:
        """Clone an existing scenario with optional modifications.

        Parameters
        ----------
        source_name : str
            Name of the scenario to clone.
        new_name : str
            Name for the new scenario.
        parameter_overrides : dict or None
            Parameters to override in the clone.
        feature_overrides : dict or None
            Feature overrides to set on the clone.

        Returns
        -------
        Scenario
            The newly created and added scenario.
        """
        source = self.get(source_name)
        cloned_solution = copy.deepcopy(source.solution)
        cloned_params = copy.deepcopy(source.parameters)

        if parameter_overrides:
            cloned_params.update(parameter_overrides)

        cloned_feat_overrides = copy.deepcopy(source.feature_overrides)
        if feature_overrides:
            if cloned_feat_overrides is None:
                cloned_feat_overrides = {}
            cloned_feat_overrides.update(feature_overrides)

        scenario = Scenario(
            name=new_name,
            solution=cloned_solution,
            parameters=cloned_params,
            feature_overrides=cloned_feat_overrides,
        )
        self._scenarios.append(scenario)
        return scenario

    def run_with_overrides(
        self,
        name: str,
        problem: ConservationProblem,
        solver: Solver,
        overrides: dict[int, dict[str, float]],
        parameter_overrides: dict | None = None,
        config: SolverConfig | None = None,
    ) -> Scenario:
        """Create scenario by solving with feature overrides applied.

        Parameters
        ----------
        name : str
            Scenario name.
        problem : ConservationProblem
            Base problem (not mutated).
        solver : Solver
            Solver to use.
        overrides : dict
            Feature target/SPF overrides.
        parameter_overrides : dict or None
            Marxan parameter overrides.
        config : SolverConfig or None
            Solver configuration.

        Returns
        -------
        Scenario
            The newly created and added scenario.
        """
        from pymarxan.models.problem import apply_feature_overrides

        if overrides:
            modified = apply_feature_overrides(problem, overrides)
        else:
            modified = problem.clone()

        if parameter_overrides:
            for k, v in parameter_overrides.items():
                modified.parameters[k] = v

        solutions = solver.solve(modified, config)
        best = min(solutions, key=lambda s: s.objective)

        params = dict(modified.parameters)
        scenario = Scenario(
            name=name,
            solution=best,
            parameters=params,
            feature_overrides=overrides if overrides else None,
        )
        self._scenarios.append(scenario)
        return scenario

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
                "has_overrides": s.feature_overrides is not None,
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
