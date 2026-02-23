"""Parallel parameter sweep execution.

Uses concurrent.futures to run parameter sweep points in parallel.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

from pymarxan.calibration.sweep import SweepConfig, SweepResult, run_sweep
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


def _solve_single(
    problem_data: dict,
    solver: Solver,
    params: dict,
    solver_config: SolverConfig,
    index: int,
) -> tuple[int, Solution]:
    """Solve a single sweep point. Returns (index, best_solution)."""
    import pandas as pd

    from pymarxan.models.problem import ConservationProblem

    modified = ConservationProblem(
        planning_units=pd.DataFrame(problem_data["planning_units"]),
        features=pd.DataFrame(problem_data["features"]),
        pu_vs_features=pd.DataFrame(problem_data["pu_vs_features"]),
        boundary=(
            pd.DataFrame(problem_data["boundary"])
            if problem_data["boundary"] is not None
            else None
        ),
        parameters={**problem_data["parameters"], **params},
    )
    sols = solver.solve(modified, solver_config)
    if not sols:
        raise ValueError("Solver returned no solutions (infeasible)")
    best = min(sols, key=lambda s: s.objective)
    return (index, best)


def run_sweep_parallel(
    problem: ConservationProblem,
    solver: Solver,
    config: SweepConfig,
    max_workers: int = 4,
) -> SweepResult:
    """Run a parameter sweep with parallel execution.

    Falls back to sequential ``run_sweep`` when ``max_workers=1``.
    """
    if max_workers <= 1:
        return run_sweep(problem, solver, config)

    solver_config = config.solver_config or SolverConfig(num_solutions=1)
    param_dicts = config.expand()

    # Serialise problem data for pickling across processes
    problem_data = {
        "planning_units": problem.planning_units.to_dict(orient="list"),
        "features": problem.features.to_dict(orient="list"),
        "pu_vs_features": problem.pu_vs_features.to_dict(orient="list"),
        "boundary": (
            problem.boundary.to_dict(orient="list")
            if problem.boundary is not None
            else None
        ),
        "parameters": dict(problem.parameters),
    }

    # Submit all jobs
    indexed_results: dict[int, Solution] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _solve_single, problem_data, solver, params, solver_config, i
            ): i
            for i, params in enumerate(param_dicts)
        }
        for future in as_completed(futures):
            idx, sol = future.result()
            indexed_results[idx] = sol

    # Reassemble in order
    solutions = [indexed_results[i] for i in range(len(param_dicts))]
    costs = [s.cost for s in solutions]
    boundaries = [s.boundary for s in solutions]
    objectives = [s.objective for s in solutions]

    return SweepResult(
        param_dicts=param_dicts,
        solutions=solutions,
        costs=costs,
        boundaries=boundaries,
        objectives=objectives,
    )
