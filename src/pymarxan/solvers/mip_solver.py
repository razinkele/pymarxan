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

    Under PROBMODE 3 (Z-score chance constraints, Phase 18) the Z-score
    constraint involves ``√Σ σ²·x`` which is SOCP/QCP territory and not
    solvable by the default CBC backend. The ``mip_chance_strategy``
    knob controls behaviour:

    - ``"drop"`` (default): solve the deterministic relaxation; populate
      ``Solution.prob_shortfalls`` and ``Solution.prob_penalty`` from a
      post-hoc Z-score evaluation. User sees the gap.
    - ``"piecewise"``: tangent-line approximation of the chance constraint
      in CBC. Deferred to Phase 18.5.
    - ``"socp"``: exact SOCP formulation via Gurobi/CPLEX. Deferred to
      Phase 21.
    """

    def __init__(
        self,
        *,
        mip_chance_strategy: str = "drop",
        mip_clump_strategy: str = "drop",
    ) -> None:
        if mip_chance_strategy not in ("drop", "piecewise", "socp"):
            raise ValueError(
                f"Unknown mip_chance_strategy {mip_chance_strategy!r}; "
                "use 'drop' (default), 'piecewise', or 'socp'."
            )
        if mip_clump_strategy not in ("drop", "big_m"):
            raise ValueError(
                f"Unknown mip_clump_strategy {mip_clump_strategy!r}; "
                "use 'drop' (default) or 'big_m' (deferred, NotImplementedError)."
            )
        self.mip_chance_strategy = mip_chance_strategy
        self.mip_clump_strategy = mip_clump_strategy

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

        # PROBMODE 3 gating — see class docstring for rationale.
        probmode = int(problem.parameters.get("PROBMODE", 0))
        if probmode == 3:
            if self.mip_chance_strategy == "piecewise":
                raise NotImplementedError(
                    "mip_chance_strategy='piecewise' (piecewise-linear "
                    "approximation of the chance constraint in CBC) is "
                    "planned for Phase 18.5. Use 'drop' (default) or one "
                    "of the SA / heuristic / iterative-improvement solvers."
                )
            if self.mip_chance_strategy == "socp":
                raise NotImplementedError(
                    "mip_chance_strategy='socp' (exact SOCP formulation via "
                    "Gurobi/CPLEX) lands in Phase 21 when those backends "
                    "are wired in. Use 'drop' (default) or SA for now."
                )

        # TARGET2 / clumping gating (Phase 19). Like PROBMODE 3, the chance-
        # constrained-style CLUMPTYPE math is non-trivial in MILP; the
        # default "drop" strategy solves the deterministic relaxation and
        # build_solution reports the clump-shortfall gap post-hoc.
        has_target2 = (
            "target2" in problem.features.columns
            and (problem.features["target2"] > 0).any()
        )
        if has_target2 and self.mip_clump_strategy == "big_m":
            raise NotImplementedError(
                "mip_clump_strategy='big_m' (network-flow + big-M formulation "
                "of TARGET2 in CBC) is deferred to a later phase. Use 'drop' "
                "(default; post-hoc gap reported on Solution.clump_shortfalls) "
                "or the SA / iterative-improvement solvers for chance-"
                "constrained-style clumping optimality."
            )

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

        # Probability risk premium (Mode 1)
        prob_expr = 0
        prob_mode = int(problem.parameters.get("PROBMODE", 1))
        if problem.probability is not None and prob_mode == 1:
            prob_weight = float(
                problem.parameters.get("PROBABILITYWEIGHTING", 1.0)
            )
            if prob_weight > 0:
                prob_pu = problem.probability["pu"].values
                prob_val = problem.probability["probability"].values
                prob_map: dict[int, float] = {}
                for pk in range(len(prob_pu)):
                    prob_map[int(prob_pu[pk])] = float(prob_val[pk])
                prob_expr = pulp.lpSum(
                    prob_weight
                    * prob_map.get(int(pu_id_arr[k]), 0.0)
                    * float(pu_cost_arr[k])
                    * x[int(pu_id_arr[k])]
                    for k in range(len(pu_id_arr))
                )

        model += (
            cost_expr + blm * boundary_expr + conn_expr + prob_expr,
            "objective",
        )

        # Constraints: feature targets
        misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))

        # Build probability map for Mode 2 (persistence-adjusted amounts)
        prob_map_m2: dict[int, float] = {}
        if (
            problem.probability is not None
            and prob_mode == 2
        ):
            p_pu = problem.probability["pu"].values
            p_val = problem.probability["probability"].values
            for pk in range(len(p_pu)):
                prob_map_m2[int(p_pu[pk])] = float(p_val[pk])

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
                
                if prob_map_m2:
                    # Mode 2: effective_amount = amount * (1 - probability)
                    amount_expr = pulp.lpSum(
                        float(amt) * (1.0 - prob_map_m2.get(int(pu), 0.0))
                        * x[int(pu)]
                        for pu, amt in zip(pus, amounts)
                    )
                else:
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

        # Accept any solver outcome where decision variables have integer
        # incumbent values. CBC sets status=LpStatusNotSolved (not Optimal)
        # when MIP_TIME_LIMIT fires but a feasible solution was already found;
        # returning [] in that case silently loses a usable solution.
        infeasible_statuses = {
            pulp.constants.LpStatusInfeasible,
            pulp.constants.LpStatusUnbounded,
            pulp.constants.LpStatusUndefined,
        }
        has_values = all(pulp.value(x[pid]) is not None for pid in pu_ids)
        if model.status in infeasible_statuses or not has_values:
            return []

        # Extract solution
        selected = np.array(
            [bool(round(pulp.value(x[pid]) or 0.0)) for pid in pu_ids],
            dtype=bool,
        )

        # build_solution populates Solution.prob_shortfalls + prob_penalty
        # automatically when PROBMODE==3. Under the 'drop' strategy, the
        # MIP solved the deterministic relaxation; the chance-constraint
        # gap is what build_solution reports post-hoc.
        meta = {"solver": self.name(), "status": pulp.LpStatus[model.status]}
        if probmode == 3:
            meta["mip_chance_strategy"] = self.mip_chance_strategy
        if has_target2:
            meta["mip_clump_strategy"] = self.mip_clump_strategy
        sol = build_solution(problem, selected, blm, metadata=meta)
        return [copy.deepcopy(sol) for _ in range(config.num_solutions)]
