"""Solver-agnostic no-good-cut portfolio generation (Phase 25).

Generates K diverse high-quality MIP solutions by iteratively solving
and adding "no-good cut" constraints that exclude previously-found
selections. Unlike Gurobi-specific top-k / gap / extra portfolios, this
works on any PuLP backend (CBC, HiGHS, Gurobi) since the cut is a
plain linear constraint.

Each cut takes the form::

    Σ_{i: s_i=1} (1 - x_i) + Σ_{i: s_i=0} x_i ≥ 1

i.e. at least one decision variable must flip from the previous
solution ``s``. The MIP returns its new optimum subject to all
accumulated cuts; objectives are therefore weakly non-decreasing in
iteration order.

When the MIP becomes infeasible (every distinct feasible selection has
been enumerated), the function returns the partial list rather than
raising — same shape as ``Solver.solve`` returning an empty list on
infeasibility.
"""
from __future__ import annotations

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver


def generate_portfolio_cuts(
    problem: ConservationProblem,
    *,
    solver: MIPSolver | None = None,
    k: int,
    config: SolverConfig | None = None,
) -> list[Solution]:
    """Generate up to ``k`` diverse solutions via no-good cuts.

    Parameters
    ----------
    problem
        Conservation problem.
    solver
        Optional pre-configured :class:`MIPSolver`. Default constructs
        ``MIPSolver()`` with auto backend.
    k
        Target portfolio size. Must be ≥ 1.
    config
        Optional :class:`SolverConfig`. ``config.metadata`` may carry
        existing keys; this function adds / replaces
        ``"forbidden_selections"`` per iteration.

    Returns
    -------
    list[Solution]
        Up to ``k`` distinct solutions, ordered by iteration (best
        first, by objective). Length may be less than ``k`` if the
        problem has fewer than ``k`` distinct feasible solutions.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive int; got {k!r}.")
    if solver is None:
        solver = MIPSolver()
    if config is None:
        config = SolverConfig(num_solutions=1)

    forbidden: list = []
    portfolio: list[Solution] = []

    for iteration in range(k):
        # Each iteration takes a fresh config (so we don't accumulate
        # state on the caller's instance) and overwrites the
        # forbidden_selections key in metadata.
        iter_metadata = dict(config.metadata) if config.metadata else {}
        iter_metadata["forbidden_selections"] = list(forbidden)
        iter_config = SolverConfig(
            num_solutions=1,
            seed=config.seed,
            verbose=config.verbose,
            metadata=iter_metadata,
        )
        sols = solver.solve(problem, iter_config)
        if not sols:
            # Infeasible under the accumulated cuts → all distinct
            # solutions exhausted. Return what we have.
            break
        sol = sols[0]
        sol.metadata = dict(sol.metadata)
        sol.metadata["portfolio_iteration"] = iteration
        portfolio.append(sol)
        forbidden.append(sol.selected.copy())

    return portfolio
