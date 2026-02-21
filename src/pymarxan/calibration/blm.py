"""BLM (Boundary Length Modifier) calibration for Marxan.

Runs the solver at multiple BLM values to find the cost-boundary trade-off
curve. Users look for the "elbow" where increasing BLM yields diminishing
returns in compactness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


@dataclass
class BLMResult:
    """Results of a BLM calibration sweep."""
    blm_values: list[float]
    costs: list[float]
    boundaries: list[float]
    objectives: list[float]
    solutions: list[Solution]


def calibrate_blm(
    problem: ConservationProblem,
    solver: Solver,
    blm_values: list[float] | None = None,
    blm_min: float = 0.0,
    blm_max: float = 100.0,
    blm_steps: int = 10,
    config: SolverConfig | None = None,
) -> BLMResult:
    """Run a BLM calibration sweep.

    Either provide explicit `blm_values` or use `blm_min/blm_max/blm_steps`
    to generate a linear range.
    """
    if config is None:
        config = SolverConfig(num_solutions=1)

    if blm_values is None:
        blm_values = np.linspace(blm_min, blm_max, blm_steps).tolist()

    costs = []
    boundaries = []
    objectives = []
    solutions_list = []

    for blm in blm_values:
        modified = ConservationProblem(
            planning_units=problem.planning_units,
            features=problem.features,
            pu_vs_features=problem.pu_vs_features,
            boundary=problem.boundary,
            parameters={**problem.parameters, "BLM": blm},
        )
        sols = solver.solve(modified, config)
        best = min(sols, key=lambda s: s.objective)
        costs.append(best.cost)
        boundaries.append(best.boundary)
        objectives.append(best.objective)
        solutions_list.append(best)

    return BLMResult(
        blm_values=blm_values,
        costs=costs,
        boundaries=boundaries,
        objectives=objectives,
        solutions=solutions_list,
    )
