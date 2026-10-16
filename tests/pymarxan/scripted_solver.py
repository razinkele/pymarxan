"""A CBC wrapper whose per-call hooks rewrite the result: the seam for mocking ``sol_status``."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pulp

from pymarxan.solvers._mip_objectives import constraint_map

Hook = Callable[[pulp.LpProblem], None]


class Scripted(pulp.LpSolver):
    """Solve with CBC, record each model's rows, then run ``hooks[call]`` if given.

    ``models[i]`` maps every constraint name to ``(sense, rhs)`` as solved in call i + 1.
    """

    def __init__(self, hooks: dict[int, Hook] | None = None) -> None:
        super().__init__(msg=False)
        self.inner = pulp.PULP_CBC_CMD(msg=False)
        self.hooks = hooks or {}
        self.calls = 0
        self.models: list[dict[str, tuple[int, float]]] = []

    def actualSolve(self, lp: pulp.LpProblem, **kwargs: Any) -> int:  # noqa: N802
        self.calls += 1
        self.inner.actualSolve(lp)
        self.models.append({
            n: (int(c.sense), -float(c.constant)) for n, c in constraint_map(lp).items()
        })
        hook = self.hooks.get(self.calls)
        if hook is not None:
            hook(lp)
        return int(lp.status)


def fail(lp: pulp.LpProblem) -> None:
    """Hook: the solve found no solution, and left junk values (every x at 1, the rest
    None) so that a caller that forgets to restore the previous stage is caught."""
    lp.status, lp.sol_status = pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound
    for v in lp.variables():
        v.varValue = 1.0 if v.name.startswith("x_") else None


def optimal_without_solution(lp: pulp.LpProblem) -> None:
    """Hook: status Optimal but sol_status NoSolutionFound: must count as no solution."""
    lp.status, lp.sol_status = pulp.LpStatusOptimal, pulp.LpSolutionNoSolutionFound


def not_solved_with_incumbent(lp: pulp.LpProblem) -> None:
    """Hook: status NotSolved (CBC on a time limit) but sol_status IntegerFeasible: an
    incumbent, which must be accepted."""
    lp.status, lp.sol_status = pulp.LpStatusNotSolved, pulp.LpSolutionIntegerFeasible
