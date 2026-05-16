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


def _validate_mip_strategy(
    name: str,
    value: str,
    allowed: tuple[str, ...],
    rejected_with_reason: dict[str, str] | None = None,
) -> None:
    """Validate a MIP strategy kwarg at __init__ time (round-3 M7).

    ``rejected_with_reason`` is ``{strategy_name: explanation}`` for values
    that ARE recognised but explicitly rejected with a different reason
    than 'not in allowed set' (e.g. ``"socp"`` for separation, which is
    combinatorial rather than conic).
    """
    if rejected_with_reason and value in rejected_with_reason:
        raise ValueError(
            f"{name}={value!r} is not valid — {rejected_with_reason[value]}"
        )
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


# Phase 21 — MIP backend factory.
#
# PuLP supports several solver backends; pymarxan exposes three by name
# (cbc, highs, gurobi) plus "auto". CBC ships with PuLP — always available.
# HiGHS opt-in via pip-installable binary (or system PATH). Gurobi requires
# the user to install gurobipy separately (pymarxan[gurobi] extra).
_MIP_BACKEND_NAMES = ("auto", "cbc", "highs", "gurobi")


def _available_backends() -> dict[str, bool]:
    """Detect which MIP backends can actually run a solve right now.

    Returns ``{backend_name: bool}``. CBC is always True (ships with PuLP);
    HiGHS / Gurobi True iff the respective binary or Python package is
    discoverable. Callers use this to surface 'why-not' status in the UI
    or pick the best backend under ``"auto"``.
    """
    result: dict[str, bool] = {}
    try:
        result["cbc"] = bool(pulp.PULP_CBC_CMD(msg=False).available())
    except Exception:
        result["cbc"] = False
    try:
        result["highs"] = bool(pulp.HiGHS_CMD(msg=False).available())
    except Exception:
        result["highs"] = False
    try:
        result["gurobi"] = bool(pulp.GUROBI_CMD(msg=False).available())
    except Exception:
        result["gurobi"] = False
    return result


def _make_pulp_solver(
    backend: str,
    *,
    time_limit: int,
    gap: float,
    verbose: bool,
):
    """Construct the PuLP solver instance for a given backend name.

    Parameters
    ----------
    backend
        One of ``"auto"`` (pick the fastest available — HiGHS > CBC),
        ``"cbc"``, ``"highs"``, ``"gurobi"``.
    time_limit, gap, verbose
        Forwarded as ``timeLimit`` / ``gapRel`` / ``msg`` kwargs.

    Returns
    -------
    pulp.LpSolver_CMD
        A PuLP solver instance ready to be passed to ``model.solve(...)``.

    Raises
    ------
    ValueError
        For unknown backend names. Use ``_available_backends()`` for the
        runtime availability check.
    """
    if backend not in _MIP_BACKEND_NAMES:
        raise ValueError(
            f"mip_backend must be one of {_MIP_BACKEND_NAMES}, got "
            f"{backend!r}."
        )
    if backend == "auto":
        # Prefer HiGHS when available (5-50× faster than CBC on large MIPs
        # per Phase 21 design rationale). Fall back to CBC, which ships
        # with PuLP and is therefore always present. We skip the CBC
        # availability probe to avoid an extra PULP_CBC_CMD construction
        # under "auto" — keeps call patterns clean for tests that mock the
        # CBC backend.
        try:
            highs_ok = bool(pulp.HiGHS_CMD(msg=False).available())
        except Exception:
            highs_ok = False
        backend = "highs" if highs_ok else "cbc"

    kwargs = {"msg": int(verbose), "timeLimit": time_limit, "gapRel": gap}
    if backend == "cbc":
        return pulp.PULP_CBC_CMD(**kwargs)
    if backend == "highs":
        return pulp.HiGHS_CMD(**kwargs)
    if backend == "gurobi":
        return pulp.GUROBI_CMD(**kwargs)
    # Unreachable: _MIP_BACKEND_NAMES is exhaustive.
    raise AssertionError(f"unreachable backend branch: {backend!r}")


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
        mip_sep_strategy: str = "drop",
        mip_backend: str = "auto",
        objective: str = "min_set",
    ) -> None:
        _validate_mip_strategy(
            "mip_chance_strategy", mip_chance_strategy,
            ("drop", "piecewise", "socp"),
        )
        _validate_mip_strategy(
            "mip_clump_strategy", mip_clump_strategy, ("drop", "big_m"),
        )
        _validate_mip_strategy(
            "mip_sep_strategy", mip_sep_strategy, ("drop", "big_m"),
            rejected_with_reason={
                "socp": (
                    "separation is a combinatorial constraint (greedy "
                    "maximum independent set), not a conic/probabilistic one. "
                    "Use 'drop' (default; gap reported on "
                    "Solution.sep_shortfalls) or 'big_m' (deferred)."
                ),
            },
        )
        if mip_backend not in _MIP_BACKEND_NAMES:
            raise ValueError(
                f"mip_backend must be one of {_MIP_BACKEND_NAMES}, got "
                f"{mip_backend!r}."
            )
        _validate_mip_strategy(
            "objective", objective,
            ("min_set", "max_features", "min_largest_shortfall", "min_penalties"),
        )
        self.mip_chance_strategy = mip_chance_strategy
        self.mip_clump_strategy = mip_clump_strategy
        self.mip_sep_strategy = mip_sep_strategy
        self.mip_backend = mip_backend
        self.objective = objective

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

        # SEPDISTANCE / SEPNUM gating (Phase 20). MIP "drop" runs the
        # deterministic relaxation and build_solution reports the
        # separation gap on Solution.sep_shortfalls post-hoc.
        has_sep_active = (
            "sepnum" in problem.features.columns
            and "sepdistance" in problem.features.columns
            and (
                (problem.features["sepnum"] > 1)
                & (problem.features["sepdistance"] > 0)
            ).any()
        )
        if has_sep_active and self.mip_sep_strategy == "big_m":
            raise NotImplementedError(
                "mip_sep_strategy='big_m' (pairwise-distance + big-M "
                "formulation of separation in CBC) is deferred to v0.3. "
                "Use 'drop' (default; post-hoc gap reported on "
                "Solution.sep_shortfalls) or SA / iterative-improvement."
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

        # Phase 23: select objective formulation. Default "min_set" preserves
        # pre-Phase-23 behaviour.
        feat_met: dict[int, pulp.LpVariable] = {}
        slack: dict[int, pulp.LpVariable] = {}
        t_var: pulp.LpVariable | None = None
        feat_ids_init = problem.features["id"].values
        if self.objective == "max_features":
            # Binary z_j per feature; relaxed target ≥ target · z_j; cost
            # cap from COSTBUDGET. Maximise Σ z_j (LpMinimize → negate).
            cost_budget = problem.parameters.get("COSTBUDGET")
            if cost_budget is None:
                raise ValueError(
                    "objective='max_features' requires a COSTBUDGET parameter "
                    "on the problem (problem.parameters['COSTBUDGET'] = ...). "
                    "Without a budget the formulation is degenerate — every "
                    "feature target is trivially met by selecting all PUs."
                )
            cost_budget = float(cost_budget)
            for fi in range(len(feat_ids_init)):
                fid = int(feat_ids_init[fi])
                feat_met[fid] = pulp.LpVariable(f"feat_met_{fid}", cat="Binary")
            model += (
                -pulp.lpSum(feat_met[fid] for fid in feat_met),
                "objective",
            )
            model += cost_expr <= cost_budget, "cost_budget"
        elif self.objective == "min_largest_shortfall":
            # Auxiliary slack_j ≥ 0, slack_j ≥ target_j - Σ amount·x;
            # t ≥ slack_j for every feature. Minimise t.
            cost_budget = problem.parameters.get("COSTBUDGET")
            if cost_budget is None:
                raise ValueError(
                    "objective='min_largest_shortfall' requires a COSTBUDGET "
                    "parameter (problem.parameters['COSTBUDGET'] = ...). "
                    "Without a budget the objective is degenerate (buying "
                    "every PU drives every shortfall to 0)."
                )
            cost_budget = float(cost_budget)
            t_var = pulp.LpVariable("max_shortfall", lowBound=0)
            for fi in range(len(feat_ids_init)):
                fid = int(feat_ids_init[fi])
                slack[fid] = pulp.LpVariable(f"slack_{fid}", lowBound=0)
                # Coupling slack ≤ t handled below via t ≥ slack.
            model += (t_var, "objective")
            model += cost_expr <= cost_budget, "cost_budget"
        elif self.objective == "min_penalties":
            # Hierarchical: minimise Σ SPF_j · slack_j first, cost second.
            # Implemented via weighted scalarisation
            #     min  M · Σ SPF_j · slack_j  +  Σ cost_i · x_i
            # with M chosen large enough that any non-zero penalty term
            # dominates the worst-case cost (sum of all costs + 1).
            feat_spf_arr = problem.features["spf"].values.astype(float)
            spf_by_id = {
                int(feat_ids_init[fi]): float(feat_spf_arr[fi])
                for fi in range(len(feat_ids_init))
            }
            for fi in range(len(feat_ids_init)):
                fid = int(feat_ids_init[fi])
                slack[fid] = pulp.LpVariable(f"slack_{fid}", lowBound=0)
            penalty_expr = pulp.lpSum(
                spf_by_id[fid] * slack[fid] for fid in slack
            )
            total_cost_upper = sum(float(c) for c in pu_cost_arr) + 1.0
            model += (
                total_cost_upper * penalty_expr + cost_expr,
                "objective",
            )
        else:
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
                if self.objective == "max_features":
                    # Relaxed target: only enforced when feat_met_j = 1.
                    # Maximising Σ feat_met_j therefore picks as many
                    # features as the cost budget will pay for.
                    model += (
                        amount_expr >= target * feat_met[fid],
                        f"target_{fid}",
                    )
                elif self.objective == "min_largest_shortfall":
                    # slack_j ≥ target_j − amount; t ≥ slack_j.
                    model += (
                        slack[fid] >= target - amount_expr,
                        f"slack_{fid}",
                    )
                    assert t_var is not None  # for mypy: set in init block above
                    model += t_var >= slack[fid], f"max_shortfall_{fid}"
                elif self.objective == "min_penalties":
                    # slack_j ≥ target_j − amount; the objective minimises
                    # Σ SPF_j · slack_j (already added above), so slack_j
                    # is pulled to 0 whenever it can be.
                    model += (
                        slack[fid] >= target - amount_expr,
                        f"slack_{fid}",
                    )
                else:
                    model += amount_expr >= target, f"target_{fid}"

        # Phase 25: no-good cuts for portfolio generation. Each forbidden
        # selection vector contributes one constraint forcing at least
        # one binary variable to differ from it. Passed through
        # config.metadata["forbidden_selections"] as a list of bool arrays
        # indexed by pu_id_arr order.
        forbidden_selections = (
            config.metadata.get("forbidden_selections", [])
            if config.metadata else []
        )
        for forbid_idx, forbidden in enumerate(forbidden_selections):
            # diff_expr = Σ_{i: s_i=1} (1 - x_i) + Σ_{i: s_i=0} x_i
            # Must be ≥ 1 to ensure at least one variable flipped.
            diff_terms = []
            for k in range(len(pu_id_arr)):
                pid = int(pu_id_arr[k])
                if bool(forbidden[k]):
                    diff_terms.append(1 - x[pid])
                else:
                    diff_terms.append(x[pid])
            model += pulp.lpSum(diff_terms) >= 1, f"nogood_cut_{forbid_idx}"

        # Solve
        time_limit = int(problem.parameters.get("MIP_TIME_LIMIT", 300))
        gap = float(problem.parameters.get("MIP_GAP", 0.0))
        verbose = config.verbose
        # Phase 21: dispatch through the backend factory. CBC remains default
        # when mip_backend="auto" but HiGHS wins if available.
        solver = _make_pulp_solver(
            self.mip_backend,
            time_limit=time_limit, gap=gap, verbose=verbose,
        )
        # Record the resolved backend in metadata so users can confirm
        # which solver actually ran. "auto" gets resolved here, so capture
        # the concrete name post-factory rather than self.mip_backend.
        resolved_backend = (
            "highs" if isinstance(solver, pulp.HiGHS_CMD)
            else "gurobi" if isinstance(solver, pulp.GUROBI_CMD)
            else "cbc"
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
        meta = {
            "solver": self.name(),
            "status": pulp.LpStatus[model.status],
            "mip_backend": resolved_backend,
            "objective": self.objective,
        }
        if probmode == 3:
            meta["mip_chance_strategy"] = self.mip_chance_strategy
        if has_target2:
            meta["mip_clump_strategy"] = self.mip_clump_strategy
        if has_sep_active:
            meta["mip_sep_strategy"] = self.mip_sep_strategy
        sol = build_solution(problem, selected, blm, metadata=meta)
        return [copy.deepcopy(sol) for _ in range(config.num_solutions)]
