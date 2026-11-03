"""Solve a built recovery problem in one of two modes (spec §3.5).

``"targets"``: the cheapest plan meeting every target (zone MIP ``min_set``; BLM allowed).
``"budget"``: the best plan within ``budget`` (zone MIP ``min_shortfall`` or
``max_weighted_features`` with ``COSTBUDGET``; every target weighted equally; boundary terms
dropped, recorded in ``metadata["boundary_dropped"]``). ``method="heuristic"`` (targets mode
only) runs ``RunModePipeline`` with RUNMODE 2 (zone SA then iterative improvement) and
ITIMPTYPE 3 and returns the run with the lowest objective; its shortfall, if any, is in the
``Solution`` — never a cheaper plan presented as feasible.
"""
from __future__ import annotations

import warnings
from typing import cast

from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.run_mode import RunModePipeline
from pymarxan.zones.diagnostics import (
    ZoneMIPNoSolutionWarning,
    diagnose_zone_infeasibility,
    min_forced_cost,
)
from pymarxan.zones.mip_solver import ZoneMIPSolver
from pymarxan.zones.model import ZonalProblem

MODES = ("targets", "budget")
METHODS = ("mip", "heuristic")
BUDGET_OBJECTIVES = ("min_shortfall", "max_weighted_features")
# Unit x zone decisions above which the zone MIP may take minutes with CBC (one binary
# per pair, dense contribution rows): 50 000 = 12 500 units with protect + active + passive
# + their assignment rows. Advisory only: the MIP still runs.
RECOVERY_MIP_SIZE_WARNING = 50_000
HEURISTIC_PARAMETERS = {"RUNMODE": 2, "ITIMPTYPE": 3}


class RecoverySizeWarning(UserWarning):
    """The zone MIP is large enough that it may be slow; consider the heuristic."""


def _boundary_dropped(problem: ZonalProblem, objective: str) -> bool:
    """Mirrors ``ZoneMIPSolver._warn_boundary_ignored``."""
    blm = float(problem.parameters.get("BLM", 0.0))
    return objective != "min_set" and problem.boundary is not None and blm != 0


def solve_recovery(
    problem: ZonalProblem,
    *,
    mode: str = "targets",
    budget: float | None = None,
    method: str = "mip",
    budget_objective: str = "min_shortfall",
    config: SolverConfig | None = None,
) -> Solution:
    """One plan for ``problem`` (from ``build_recovery_problem``), annotated with
    ``metadata["recovery_mode"]``, ``["recovery_method"]`` and ``["boundary_dropped"]``.

    Raises ``ValueError`` for an unknown mode/method, budget mode without a budget or with
    the heuristic, and when the MIP finds no plan (the message carries the diagnostics).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if mode == "budget":
        if budget is None:
            raise ValueError("budget mode needs a budget (total cost cap)")
        if method != "mip":
            raise ValueError("budget mode is solved by the zone MIP only")
        if budget_objective not in BUDGET_OBJECTIVES:
            raise ValueError(f"budget_objective must be one of {BUDGET_OBJECTIVES}")
    objective = "min_set" if mode == "targets" else budget_objective

    if method == "heuristic":
        params = {**problem.parameters, **HEURISTIC_PARAMETERS}
        cfg = config or SolverConfig(num_solutions=10, seed=0)
        runs = RunModePipeline().solve(problem.copy_with(parameters=params), cfg)
        best = min(runs, key=lambda s: s.objective)
    else:
        n_decisions = len(problem.planning_units) * problem.n_zones
        if n_decisions > RECOVERY_MIP_SIZE_WARNING:
            warnings.warn(
                f"zone MIP with {n_decisions} unit-zone decisions (> "
                f"{RECOVERY_MIP_SIZE_WARNING}) may take long; set MIP_TIME_LIMIT or use "
                "method='heuristic'",
                RecoverySizeWarning,
                stacklevel=2,
            )
        solve_problem = problem
        if mode == "budget":
            assert budget is not None
            # equal weights: relative shortfall per target, not the scaled penalty factors
            solve_problem = cast(ZonalProblem, problem.copy_with(
                features=problem.features.assign(spf=1.0),
                parameters={**problem.parameters, "COSTBUDGET": float(budget)},
            ))
        cfg = config or SolverConfig(num_solutions=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ZoneMIPNoSolutionWarning)  # re-raised below
            sols = ZoneMIPSolver(objective=objective).solve(solve_problem, cfg)
        if not sols:
            if mode == "budget":
                floor = min_forced_cost(solve_problem)
                reason = (
                    f"COSTBUDGET {budget:g} is below the least cost any plan must pay "
                    f"({floor:g}: locked-in and must-act units)" if floor > float(budget or 0)
                    else "the time limit was reached"
                )
                raise ValueError("no plan within the budget: " + reason)
            reasons = diagnose_zone_infeasibility(solve_problem) or [
                "targets compete or the time limit was reached"
            ]
            raise ValueError("no plan meets every target: " + "; ".join(reasons))
        best = sols[0]
    best.metadata.update({
        "recovery_mode": mode,
        "recovery_method": method,
        "boundary_dropped": _boundary_dropped(problem, objective),
    })
    return best
