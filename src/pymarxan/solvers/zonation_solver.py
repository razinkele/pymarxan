"""Zonation rank-removal as a Solver-ABC adapter (Phase B)."""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import (
    STATUS_LOCKED_IN,
    STATUS_LOCKED_OUT,
    ConservationProblem,
)
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.utils import build_solution
from pymarxan.zonation.rank_removal import rank_removal


class ZonationSolver(Solver):
    """Threshold a Zonation priority ranking into a single reserve.

    Runs :func:`pymarxan.zonation.rank_removal` and selects the top
    ``top_fraction`` of the ranking (a fraction of PUs *by count*; for
    unequal-area/cost PUs this is not a 30%-of-area/budget reserve — read
    ``metadata['performance_curves']['prop_cost_remaining']`` for a budget view).
    PU status is then enforced as a hard constraint (locked-out excluded,
    locked-in included), so the reserve honors locks like every other solver.

    Deterministic: one ranking -> one reserve, so ``solve`` returns a length-1
    list regardless of ``config.num_solutions`` (unlike ``MIPSolver``, which pads
    to ``num_solutions`` identical copies — copies would fake cross-run variety
    to selection-frequency analysis). The full ranking and performance curves
    ride in ``Solution.metadata``. Zonation ranks by biological loss and does not
    optimize to meet feature targets, so a low ``top_fraction`` may leave
    ``all_targets_met`` False by design. Like ``HeuristicSolver``, Zonation is
    blind to PROBMODE 3 / TARGET2 / SEPNUM during ranking but ``build_solution``
    reports those gaps post-hoc, so the ``supports_*`` flags keep their ``True``
    defaults.
    """

    def __init__(
        self,
        *,
        rule: str = "caz",
        top_fraction: float = 0.3,
        warp: int = 1,
        weights: dict[int, float] | None = None,
        use_cost: bool = True,
    ) -> None:
        if rule not in ("caz", "abf"):
            raise ValueError(f"rule must be 'caz' or 'abf', got {rule!r}")
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
        self.rule = rule
        self.top_fraction = top_fraction
        self.warp = warp
        self.weights = weights
        self.use_cost = use_cost

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        result = rank_removal(
            problem,
            rule=self.rule,
            weights=self.weights,
            warp=self.warp,
            use_cost=self.use_cost,
        )
        selected_ids = result.top_fraction(self.top_fraction)
        selected = np.array(
            [int(pid) in selected_ids for pid in problem.planning_units["id"]],
            dtype=bool,
        )
        # Enforce PU status as a hard constraint — top_fraction selects by rank
        # only, so a high fraction could otherwise sweep in a locked-out cell.
        status = problem.planning_units["status"].to_numpy()
        selected[status == STATUS_LOCKED_OUT] = False
        selected[status == STATUS_LOCKED_IN] = True
        blm = float(problem.parameters.get("BLM", 0.0))
        meta = {
            "solver": "zonation",
            "rule": self.rule,
            "top_fraction": self.top_fraction,
            "priority_rank": result.priority_rank,
            "performance_curves": result.performance_curves,
        }
        return [build_solution(problem, selected, blm, metadata=meta)]

    def name(self) -> str:
        return "Zonation (rank-removal)"

    def supports_zones(self) -> bool:
        return False

    def available(self) -> bool:
        return True
