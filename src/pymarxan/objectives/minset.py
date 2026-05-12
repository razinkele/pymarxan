"""Minimum-set objective: minimize cost subject to feature targets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pymarxan.objectives.base import ZonalObjective

if TYPE_CHECKING:
    import pulp

    from pymarxan.models.problem import ConservationProblem


class MinSetObjective(ZonalObjective):
    """Minimize total cost of selected planning units.

    This is the classic Marxan objective. Feature targets are enforced
    via SPF-weighted shortfall penalty (SA) or hard constraints (MIP).
    """

    def name(self) -> str:
        return "MinSet"

    def uses_target_penalty(self) -> bool:
        return True

    def compute_base_score(
        self,
        problem: ConservationProblem,
        selected: np.ndarray,
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> float:
        costs = problem.planning_units["cost"].values
        idx_arr = np.array(
            [pu_index[pid] for pid in problem.planning_units["id"]],
        )
        return float(np.sum(costs * selected[idx_arr]))

    def build_mip_objective(
        self,
        problem: ConservationProblem,
        model: pulp.LpProblem,
        x: dict[int, pulp.LpVariable],
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> pulp.LpAffineExpression:
        import pulp as _pulp

        cost_map = dict(
            zip(
                problem.planning_units["id"],
                problem.planning_units["cost"],
                strict=False,
            ),
        )
        return _pulp.lpSum(
            cost_map.get(pid, 0.0) * x[pid] for pid in x
        )

    # --- Zonal interface ---

    def compute_zone_score(
        self,
        problem: Any,
        assignment: np.ndarray,
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> float:
        total = 0.0
        zone_costs = problem.zone_costs
        pu_ids = problem.planning_units["id"].values
        for i, pid in enumerate(pu_ids):
            idx = pu_index.get(int(pid))
            if idx is not None:
                z = int(assignment[idx])
                if z > 0:
                    total += float(zone_costs.get((int(pid), z), 0.0))
        return total

    def build_zone_mip_objective(
        self,
        problem: Any,
        model: pulp.LpProblem,
        x: dict[tuple[int, int], pulp.LpVariable],
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> pulp.LpAffineExpression:
        import pulp as _pulp

        zone_costs = problem.zone_costs
        return _pulp.lpSum(
            zone_costs.get((pid, z), 0.0) * var
            for (pid, z), var in x.items()
            if z > 0
        )
