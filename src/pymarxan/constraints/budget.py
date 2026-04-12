"""Budget constraint — limits total cost of selected planning units.

This is a convenience wrapper around LinearConstraint that auto-builds
coefficients from planning unit costs.
"""
from __future__ import annotations

from pymarxan.constraints.linear import LinearConstraint
from pymarxan.models.problem import ConservationProblem


def budget_constraint(
    problem: ConservationProblem,
    max_budget: float,
    *,
    hard: bool = True,
    penalty_weight: float = 1000.0,
) -> LinearConstraint:
    """Create a budget constraint from a problem's planning unit costs.

    Parameters
    ----------
    problem : ConservationProblem
        The conservation problem (costs read from planning_units).
    max_budget : float
        Maximum total cost allowed.
    hard : bool
        If True, SA rejects moves that would exceed the budget.
    penalty_weight : float
        Penalty per unit of budget violation (soft mode only).

    Returns
    -------
    LinearConstraint
        A <= constraint: Σ cost_i * x_i <= max_budget.
    """
    coefficients = {
        int(row["id"]): float(row["cost"])
        for _, row in problem.planning_units.iterrows()
    }
    return LinearConstraint(
        coefficients=coefficients,
        sense="<=",
        rhs=max_budget,
        penalty_weight=penalty_weight,
        hard=hard,
        label="Budget",
    )
