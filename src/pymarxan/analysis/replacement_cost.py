"""Replacement-cost importance (Phase 22).

For each PU in the baseline MIP optimum, lock it out and re-solve. The
gap ``optimum_locked_out − optimum_baseline`` is the replacement cost:
how much more the conservation problem costs when this PU is unavailable.

The metric is the gold standard for prioritisation because it captures
*both* a PU's biological contribution and the cost of substitution — a
PU with rare features in an expensive area scores high; a PU that's
easily replaced (even one with rare features but plenty of close
substitutes) scores low.

Cost: ``n_selected + 1`` MIP solves (baseline + one per selected PU).
Use HiGHS (``MIPSolver(mip_backend="highs")``) for problems where this
matters; Phase 21 wired that in.

Reference: Ferrier et al. (2000); Cabeza & Moilanen (2006). *Operations
Research* 53(1): 174-191. https://doi.org/10.1287/opre.1040.0167
"""
from __future__ import annotations

import copy
import math

from pymarxan.models.problem import (
    STATUS_LOCKED_OUT,
    ConservationProblem,
)
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver


def compute_replacement_cost(
    problem: ConservationProblem,
    *,
    solver: MIPSolver | None = None,
) -> dict[int, float]:
    """Compute the replacement-cost score for each planning unit.

    For each PU ``i`` in the baseline MIP optimum, returns the objective
    increase when ``i`` is locked out and the problem is re-solved.
    PUs not in the baseline get score 0.

    Parameters
    ----------
    problem
        Conservation problem.
    solver
        Optional pre-configured :class:`MIPSolver`. Default constructs one
        with auto backend (CBC/HiGHS) and ``mip_*_strategy='drop'``.

    Returns
    -------
    dict[int, float]
        ``planning_unit_id -> replacement_cost``. PUs not in the baseline
        score 0. If lock-out makes the problem infeasible, the score is
        ``float('inf')``.
    """
    if solver is None:
        solver = MIPSolver()

    pu_ids = problem.planning_units["id"].astype(int).to_numpy()
    scores: dict[int, float] = {int(pid): 0.0 for pid in pu_ids}

    # Baseline solve.
    config = SolverConfig(num_solutions=1)
    baseline_sols = solver.solve(problem, config)
    if not baseline_sols:
        # Baseline infeasible — return all-zero scores (nothing to score).
        return scores
    baseline = baseline_sols[0]
    baseline_obj = float(baseline.objective)
    selected_indices = [i for i, sel in enumerate(baseline.selected) if sel]

    for i in selected_indices:
        pid = int(pu_ids[i])
        # Build a problem variant with PU `i` locked out.
        variant = problem.copy_with(
            planning_units=problem.planning_units.copy(),
        )
        variant.planning_units.loc[
            variant.planning_units["id"] == pid, "status"
        ] = STATUS_LOCKED_OUT
        # Deepcopy parameters so we don't mutate the caller's dict.
        variant.parameters = copy.deepcopy(problem.parameters)

        locked_sols = solver.solve(variant, config)
        if not locked_sols:
            scores[pid] = math.inf
            continue
        scores[pid] = float(locked_sols[0].objective) - baseline_obj

    return scores
