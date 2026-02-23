"""Parameter sweep for conservation planning.

Generalises the BLM calibration pattern to sweep any combination of
problem parameters and collect results.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep.

    Provide either ``param_dicts`` (explicit list of parameter dicts)
    or ``param_grid`` (dict of param_name -> list of values, expanded
    into the cartesian product). If both are provided, ``param_dicts``
    takes precedence.
    """

    param_dicts: list[dict] | None = None
    param_grid: dict[str, list] | None = None
    solver_config: SolverConfig | None = None

    def expand(self) -> list[dict]:
        """Return the list of parameter dicts to sweep over."""
        if self.param_dicts is not None:
            return list(self.param_dicts)
        if self.param_grid is not None:
            keys = sorted(self.param_grid.keys())
            values = [self.param_grid[k] for k in keys]
            return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        return [{}]


@dataclass
class SweepResult:
    """Results of a parameter sweep."""

    param_dicts: list[dict]
    solutions: list[Solution]
    costs: list[float] = field(default_factory=list)
    boundaries: list[float] = field(default_factory=list)
    objectives: list[float] = field(default_factory=list)

    @property
    def best(self) -> Solution:
        """Return the solution with the lowest objective."""
        idx = min(range(len(self.objectives)), key=lambda i: self.objectives[i])
        return self.solutions[idx]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame with one row per sweep point."""
        rows = []
        for i, params in enumerate(self.param_dicts):
            row = {**params}
            row["cost"] = self.costs[i]
            row["boundary"] = self.boundaries[i]
            row["objective"] = self.objectives[i]
            row["n_selected"] = self.solutions[i].n_selected
            row["all_targets_met"] = self.solutions[i].all_targets_met
            rows.append(row)
        return pd.DataFrame(rows)


def run_sweep(
    problem: ConservationProblem,
    solver: Solver,
    config: SweepConfig,
) -> SweepResult:
    """Run a parameter sweep over the given problem."""
    solver_config = config.solver_config or SolverConfig(num_solutions=1)
    param_dicts = config.expand()

    solutions: list[Solution] = []
    costs: list[float] = []
    boundaries: list[float] = []
    objectives: list[float] = []
    feasible_param_dicts: list[dict] = []

    for params in param_dicts:
        modified = ConservationProblem(
            planning_units=problem.planning_units,
            features=problem.features,
            pu_vs_features=problem.pu_vs_features,
            boundary=problem.boundary,
            parameters={**problem.parameters, **params},
        )
        sols = solver.solve(modified, solver_config)
        if not sols:
            continue
        best = min(sols, key=lambda s: s.objective)
        feasible_param_dicts.append(params)
        solutions.append(best)
        costs.append(best.cost)
        boundaries.append(best.boundary)
        objectives.append(best.objective)

    return SweepResult(
        param_dicts=feasible_param_dicts,
        solutions=solutions,
        costs=costs,
        boundaries=boundaries,
        objectives=objectives,
    )
