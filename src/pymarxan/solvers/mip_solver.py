"""MIP solver using PuLP for exact conservation planning optimization."""
from __future__ import annotations

import copy

import numpy as np
import pulp

from pymarxan.models.problem import (
    STATUS_LOCKED_IN,
    STATUS_LOCKED_OUT,
    ConservationProblem,
)
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.utils import build_solution


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

        # Build the LP model
        model = pulp.LpProblem("MarxanMIP", pulp.LpMinimize)

        # Decision variables: x_i = 1 if PU i is selected
        x = {}
        pu_id_arr = problem.planning_units["id"].values
        pu_st_arr = problem.planning_units["status"].values.astype(int)
        for k in range(len(pu_id_arr)):
            pid = int(pu_id_arr[k])
            status = int(pu_st_arr[k])
            x[pid] = pulp.LpVariable(f"x_{pid}", cat="Binary")
            if status == STATUS_LOCKED_IN:
                model += x[pid] == 1, f"locked_in_{pid}"
            elif status == STATUS_LOCKED_OUT:
                model += x[pid] == 0, f"locked_out_{pid}"

        # Objective: cost component
        pu_cost_arr = problem.planning_units["cost"].values.astype(float)
        cost_expr = pulp.lpSum(
            float(pu_cost_arr[k]) * x[int(pu_id_arr[k])]
            for k in range(len(pu_id_arr))
        )

        # Boundary component with auxiliary variables
        boundary_expr = 0
        y = {}
        if blm > 0 and problem.boundary is not None:
            b_id1 = problem.boundary["id1"].values
            b_id2 = problem.boundary["id2"].values
            b_val = problem.boundary["boundary"].values.astype(float)
            for bk in range(len(b_id1)):
                id1 = int(b_id1[bk])
                id2 = int(b_id2[bk])
                bval = float(b_val[bk])

                if id1 == id2:
                    # Self-boundary: external boundary cost when PU is selected
                    boundary_expr += bval * x[id1]
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

        # Connectivity component with auxiliary variables
        conn_expr = 0
        conn_weight = float(problem.parameters.get("CONNECTIVITY_WEIGHT", 0.0))
        asymmetric = bool(
            int(problem.parameters.get("ASYMMETRIC_CONNECTIVITY", 0))
        )
        if conn_weight > 0 and problem.connectivity is not None:
            conn = problem.connectivity
            c_id1 = conn["id1"].values
            c_id2 = conn["id2"].values
            c_val = conn["value"].values.astype(float)

            if asymmetric:
                # Penalize selecting source i without sink j:
                #   z_ij >= x_i - x_j, z_ij >= 0
                #   objective: +conn_weight * c_ij * z_ij
                for ck in range(len(c_id1)):
                    i = int(c_id1[ck])
                    j = int(c_id2[ck])
                    cval = float(c_val[ck])
                    z = pulp.LpVariable(
                        f"zc_{i}_{j}", lowBound=0, cat="Continuous",
                    )
                    model += z >= x[i] - x[j], f"conn_asym_{i}_{j}"
                    conn_expr += conn_weight * cval * z
            else:
                # Reward both selected via binary-AND linearization:
                #   z_ij <= x_i, z_ij <= x_j, z_ij >= x_i + x_j - 1
                #   objective: -conn_weight * c_ij * z_ij
                seen: set[tuple[int, int]] = set()
                for ck in range(len(c_id1)):
                    i = int(c_id1[ck])
                    j = int(c_id2[ck])
                    key = (min(i, j), max(i, j))
                    if key in seen:
                        continue
                    seen.add(key)
                    cval = float(c_val[ck])
                    z = pulp.LpVariable(f"zc_{key[0]}_{key[1]}", cat="Binary")
                    model += z <= x[key[0]], f"conn_and1_{key[0]}_{key[1]}"
                    model += z <= x[key[1]], f"conn_and2_{key[0]}_{key[1]}"
                    model += (
                        z >= x[key[0]] + x[key[1]] - 1,
                        f"conn_and3_{key[0]}_{key[1]}",
                    )
                    conn_expr += -conn_weight * cval * z

        model += cost_expr + blm * boundary_expr + conn_expr, "objective"

        # Constraints: feature targets
        misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
        
        # Pre-group feature data for O(1) lookup instead of O(R) filtering
        feat_groups = problem.pu_vs_features.groupby("species")
        
        feat_ids = problem.features["id"].values
        feat_targets = problem.features["target"].values.astype(float)
        for fi in range(len(feat_ids)):
            fid = int(feat_ids[fi])
            target = float(feat_targets[fi]) * misslevel
            
            if fid in feat_groups.groups:
                feat_data = feat_groups.get_group(fid)
                # Optimize iteration using zip instead of iterrows
                pus = feat_data["pu"].values
                amounts = feat_data["amount"].values
                
                amount_expr = pulp.lpSum(
                    float(amt) * x[int(pu)]
                    for pu, amt in zip(pus, amounts)
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

        if model.status != pulp.constants.LpStatusOptimal:
            return []

        # Extract solution
        selected = np.array(
            [bool(round(pulp.value(x[pid]) or 0.0)) for pid in pu_ids],
            dtype=bool,
        )

        sol = build_solution(
            problem,
            selected,
            blm,
            metadata={"solver": self.name(), "status": pulp.LpStatus[model.status]},
        )
        return [copy.deepcopy(sol) for _ in range(config.num_solutions)]
