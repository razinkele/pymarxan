"""MIP solver using PuLP for exact conservation planning optimization."""
from __future__ import annotations

import numpy as np
import pulp

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.utils import check_targets, compute_boundary


class MIPSolver(Solver):
    """Solver that formulates the Marxan minimum-set problem as a MILP.

    Minimizes:
        sum(cost_i * x_i) + BLM * sum(boundary_ij * y_ij)

    Subject to:
        sum(amount_ij * x_i) >= target_j   for all features j
        y_ij >= x_i - x_j                  for boundary pairs (i != j)
        y_ij >= x_j - x_i                  for boundary pairs (i != j)
        x_i = 1                            for locked-in PUs (status=2)
        x_i = 0                            for locked-out PUs (status=3)
        x_i in {0, 1}
    """

    def name(self) -> str:
        return "MIP (PuLP)"

    def supports_zones(self) -> bool:
        return False

    def available(self) -> bool:
        return True  # pulp is a core dependency

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig()

        blm = float(problem.parameters.get("BLM", 0.0))
        pu_ids = problem.planning_units["id"].tolist()
        pu_index = {pid: i for i, pid in enumerate(pu_ids)}

        # Build the LP model
        model = pulp.LpProblem("MarxanMIP", pulp.LpMinimize)

        # Decision variables: x_i = 1 if PU i is selected
        x = {}
        for _, row in problem.planning_units.iterrows():
            pid = int(row["id"])
            status = int(row["status"])
            if status == 2:
                # Locked in
                x[pid] = pulp.LpVariable(f"x_{pid}", cat="Binary")
                model += x[pid] == 1, f"locked_in_{pid}"
            elif status == 3:
                # Locked out
                x[pid] = pulp.LpVariable(f"x_{pid}", cat="Binary")
                model += x[pid] == 0, f"locked_out_{pid}"
            else:
                x[pid] = pulp.LpVariable(f"x_{pid}", cat="Binary")

        # Objective: cost component
        cost_expr = pulp.lpSum(
            float(row["cost"]) * x[int(row["id"])]
            for _, row in problem.planning_units.iterrows()
        )

        # Boundary component with auxiliary variables
        boundary_expr = 0
        y = {}
        if blm > 0 and problem.boundary is not None:
            for _, brow in problem.boundary.iterrows():
                id1 = int(brow["id1"])
                id2 = int(brow["id2"])
                bval = float(brow["boundary"])

                if id1 == id2:
                    # External/diagonal boundary: contributes when PU is NOT selected
                    # In Marxan, this is the perimeter cost when PU is selected but
                    # adjacent PU is not. For self-edges, it represents external boundary.
                    # Actually skip self-edges in the MIP objective since they represent
                    # the external boundary. We handle them in the actual boundary calculation.
                    # For the MIP, we only linearize the off-diagonal pairs.
                    continue
                else:
                    # Off-diagonal: linearize |x_i - x_j|
                    key = (min(id1, id2), max(id1, id2))
                    if key not in y:
                        y_var = pulp.LpVariable(
                            f"y_{key[0]}_{key[1]}", lowBound=0, upBound=1
                        )
                        y[key] = y_var
                        model += (
                            y_var >= x[key[0]] - x[key[1]],
                            f"abs1_{key[0]}_{key[1]}",
                        )
                        model += (
                            y_var >= x[key[1]] - x[key[0]],
                            f"abs2_{key[0]}_{key[1]}",
                        )
                    boundary_expr += bval * y[key]

        model += cost_expr + blm * boundary_expr, "objective"

        # Constraints: feature targets
        for _, feat_row in problem.features.iterrows():
            fid = int(feat_row["id"])
            target = float(feat_row["target"])
            feat_data = problem.pu_vs_features[
                problem.pu_vs_features["species"] == fid
            ]
            amount_expr = pulp.lpSum(
                float(r["amount"]) * x[int(r["pu"])]
                for _, r in feat_data.iterrows()
            )
            model += amount_expr >= target, f"target_{fid}"

        # Solve
        time_limit = int(problem.parameters.get("MIP_TIME_LIMIT", 300))
        gap = float(problem.parameters.get("MIP_GAP", 0.0))
        verbose = config.verbose
        solver = pulp.PULP_CBC_CMD(
            msg=int(verbose), timeLimit=time_limit, gapRel=gap,
        )
        model.solve(solver)

        # Extract solution
        selected = np.array(
            [bool(round(pulp.value(x[pid]))) for pid in pu_ids], dtype=bool
        )

        # Compute actual cost
        costs = np.asarray(problem.planning_units["cost"].values)
        total_cost = float(np.sum(costs[selected]))

        # Compute actual boundary
        total_boundary = compute_boundary(problem, selected, pu_index)

        # Check which targets are met
        targets_met = check_targets(problem, selected, pu_index)

        # Compute objective value
        objective = total_cost + blm * total_boundary

        sol = Solution(
            selected=selected,
            cost=total_cost,
            boundary=total_boundary,
            objective=objective,
            targets_met=targets_met,
            metadata={"solver": self.name(), "status": pulp.LpStatus[model.status]},
        )

        return [sol] * config.num_solutions
