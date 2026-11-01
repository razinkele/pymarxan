"""Parallel parameter sweep execution.

Uses concurrent.futures to run parameter sweep points in parallel.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

from pymarxan.calibration.sweep import SweepConfig, SweepResult, run_sweep
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


def _solve_single(
    problem: ConservationProblem,
    solver: Solver,
    params: dict,
    solver_config: SolverConfig,
    index: int,
) -> tuple[int, Solution | None]:
    """Solve a single sweep point. Returns (index, best_solution), or (index, None) when
    the solver returns no solutions (an infeasible point).

    The whole problem is pickled to the worker and derived with ``copy_with``, exactly
    as ``run_sweep`` does, so its subclass and every optional table (probability,
    connectivity, ...) survive the trip. Any error the solve raises reaches the caller,
    as in ``run_sweep``.
    """
    modified = problem.copy_with(parameters={**problem.parameters, **params})
    sols = solver.solve(modified, solver_config)
    if not sols:
        return (index, None)
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

    Returns what ``run_sweep`` returns. If a point raises, the points still queued are
    cancelled (those already running, or already handed to a worker, finish) and the
    lowest-index failing point's error is raised, as from ``run_sweep``.
    """
    if max_workers <= 1:
        return run_sweep(problem, solver, config)

    solver_config = config.solver_config or SolverConfig(num_solutions=1)
    param_dicts = config.expand()

    # Submit all jobs in sweep order. On the first error, cancel the points still queued
    # and wait for the ones already running (or already handed to a worker): without
    # cancel_futures, leaving the with-block would solve every queued point first. Do
    # not iterate as_completed after cancelling: futures cancelled by shutdown are never
    # reported to it, so the loop would hang.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_solve_single, problem, solver, params, solver_config, i)
            for i, params in enumerate(param_dicts)
        ]
        for future in as_completed(futures):
            if future.exception() is not None:
                executor.shutdown(wait=True, cancel_futures=True)
                break

    # Read the results in sweep order, so the error raised is the lowest-index failing
    # point's, as in run_sweep (points are dispatched in order, so a point below a failed
    # one is never cancelled). Infeasible points are dropped, as run_sweep skips them
    # with `if not sols: continue`.
    indexed_results: dict[int, Solution] = {}
    for future in futures:
        if future.cancelled():
            continue
        idx, sol = future.result()
        if sol is not None:
            indexed_results[idx] = sol

    # Reassemble in original order, keeping only the points that succeeded
    ordered_indices = sorted(indexed_results.keys())
    solutions = [indexed_results[i] for i in ordered_indices]
    surviving_param_dicts = [param_dicts[i] for i in ordered_indices]
    costs = [s.cost for s in solutions]
    boundaries = [s.boundary for s in solutions]
    objectives = [s.objective for s in solutions]

    return SweepResult(
        param_dicts=surviving_param_dicts,
        solutions=solutions,
        costs=costs,
        boundaries=boundaries,
        objectives=objectives,
    )
