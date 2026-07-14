"""Tests for Phase 23 extended MIP objectives.

Phase 23 adds three objective formulations beyond the default ``"min_set"``:

- ``"max_features"`` — maximise the count of features whose target is met
  under a cost budget.
- ``"min_largest_shortfall"`` — minimax over per-feature shortfalls.
- ``"min_penalties"`` — hierarchical: minimise SPF-weighted shortfall
  first, cost second.

All three are MIP-only (currently). Each is selected via the
``objective`` kwarg on :class:`MIPSolver`.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver


def _two_feature_problem(cost_budget: float | None = None):
    """2 features, 4 PUs. Each PU contributes 1 unit of one feature. Pure
    min_set requires both targets met → cost 2 (one PU per feature).
    With budget = 1, only one feature target can be met."""
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [1.0, 1.0, 1.0, 1.0],
        "status": [0, 0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"],
        "target": [1.0, 1.0], "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu":      [1, 2, 3, 4],
        "amount":  [1.0, 1.0, 1.0, 1.0],
    })
    params = {}
    if cost_budget is not None:
        params["COSTBUDGET"] = cost_budget
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        parameters=params,
    )


# --- MIPSolver objective kwarg surface ---------------------------------


def test_mip_solver_accepts_objective_kwarg():
    """``MIPSolver(objective="max_features")`` stores the choice."""
    solver = MIPSolver(objective="max_features")
    assert solver.objective == "max_features"


def test_mip_solver_default_objective_is_min_set():
    """Default ``objective`` preserves pre-Phase-23 behaviour."""
    solver = MIPSolver()
    assert solver.objective == "min_set"


def test_mip_solver_rejects_unknown_objective_at_init():
    """Fail-fast: invalid objective name raises ValueError at __init__."""
    with pytest.raises(ValueError, match="objective"):
        MIPSolver(objective="bogus")


# --- MaxFeaturesObjective ----------------------------------------------


def test_max_features_meets_all_targets_when_budget_unconstrained():
    """With a generous budget, ``max_features`` matches ``min_set`` —
    both achieve all targets met."""
    p = _two_feature_problem(cost_budget=10.0)
    sols = MIPSolver(objective="max_features").solve(
        p, SolverConfig(num_solutions=1),
    )
    assert len(sols) == 1
    assert all(sols[0].targets_met.values())


def test_max_features_picks_one_feature_under_tight_budget():
    """COSTBUDGET=1 forces a trade-off — only one of the two targets can
    be met. ``max_features`` should still report exactly one met target
    (the count, maximised)."""
    p = _two_feature_problem(cost_budget=1.0)
    sols = MIPSolver(objective="max_features").solve(
        p, SolverConfig(num_solutions=1),
    )
    assert len(sols) == 1
    n_met = sum(sols[0].targets_met.values())
    assert n_met == 1
    # Cost must respect the budget.
    assert sols[0].cost <= 1.0 + 1e-6


def test_max_features_rejects_when_no_cost_budget():
    """``max_features`` requires a cost budget; missing → ValueError at solve."""
    p = _two_feature_problem(cost_budget=None)
    with pytest.raises(ValueError, match="COSTBUDGET"):
        MIPSolver(objective="max_features").solve(p, SolverConfig(num_solutions=1))


def test_max_features_metadata_records_objective():
    """Solution metadata should record which objective was used so
    downstream analyses know what they're looking at."""
    p = _two_feature_problem(cost_budget=10.0)
    sols = MIPSolver(objective="max_features").solve(
        p, SolverConfig(num_solutions=1),
    )
    assert sols[0].metadata.get("objective") == "max_features"


def test_min_set_metadata_records_objective():
    """Even the default objective should be recorded in metadata."""
    p = _two_feature_problem(cost_budget=10.0)
    sols = MIPSolver().solve(p, SolverConfig(num_solutions=1))
    assert sols[0].metadata.get("objective") == "min_set"


# --- MinLargestShortfallObjective --------------------------------------


def _infeasible_min_set_problem(cost_budget: float):
    """Targets that can't all be met under the budget — the min-largest-
    shortfall objective should distribute the deficit as evenly as
    possible rather than abandon one feature entirely."""
    pu = pd.DataFrame({
        "id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0],
    })
    features = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"],
        "target": [10.0, 10.0], "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu":      [1, 2, 1, 2],
        "amount":  [5.0, 5.0, 5.0, 5.0],
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        parameters={"COSTBUDGET": cost_budget},
    )


def test_min_largest_shortfall_runs_end_to_end():
    """Smoke: ``min_largest_shortfall`` solves and returns a Solution."""
    p = _infeasible_min_set_problem(cost_budget=1.0)
    sols = MIPSolver(objective="min_largest_shortfall").solve(
        p, SolverConfig(num_solutions=1),
    )
    assert len(sols) == 1
    assert sols[0].metadata.get("objective") == "min_largest_shortfall"


def test_min_largest_shortfall_balances_deficit_across_features():
    """If two features have target=10 each but only one PU can be bought
    (each PU supplies 5 of each feature), the minimax objective should
    accept the PU → shortfall = 5 on BOTH features (balanced) rather than
    pick a PU that meets one and abandons the other."""
    p = _infeasible_min_set_problem(cost_budget=1.0)
    sols = MIPSolver(objective="min_largest_shortfall").solve(
        p, SolverConfig(num_solutions=1),
    )
    # Both features should have the same (or near-same) shortfall under
    # a minimax objective — neither is sacrificed.
    sol = sols[0]
    # With one PU selected (cost 1), each feature receives 5, target 10:
    # shortfall is 5 for both. Minimax t = 5.
    assert sol.cost == pytest.approx(1.0, abs=1e-6)
    # Exactly one PU is selected.
    assert sol.selected.sum() == 1


def test_min_largest_shortfall_requires_cost_budget():
    """Without a budget, the objective is degenerate — buy everything,
    shortfall = 0 always. Require an explicit COSTBUDGET."""
    p = _two_feature_problem(cost_budget=None)
    with pytest.raises(ValueError, match="COSTBUDGET"):
        MIPSolver(objective="min_largest_shortfall").solve(
            p, SolverConfig(num_solutions=1),
        )


# --- MinPenaltiesObjective ---------------------------------------------


def test_min_penalties_runs_end_to_end():
    """Smoke: ``min_penalties`` solves and returns a Solution."""
    p = _two_feature_problem(cost_budget=10.0)
    sols = MIPSolver(objective="min_penalties").solve(
        p, SolverConfig(num_solutions=1),
    )
    assert len(sols) == 1
    assert sols[0].metadata.get("objective") == "min_penalties"


def test_min_penalties_meets_all_targets_when_feasible():
    """When the problem is feasible under the budget, ``min_penalties``
    should bring penalty to 0 — same as ``min_set`` would."""
    p = _two_feature_problem(cost_budget=10.0)
    sols = MIPSolver(objective="min_penalties").solve(
        p, SolverConfig(num_solutions=1),
    )
    assert all(sols[0].targets_met.values())


def test_min_penalties_prioritizes_penalty_over_cost():
    """When meeting all targets is feasible but expensive, ``min_penalties``
    still meets the targets — even if it costs more than ``max_features``
    would spend (which would happily abandon a feature to save cost)."""
    p = _two_feature_problem(cost_budget=10.0)
    sols_min_pen = MIPSolver(objective="min_penalties").solve(
        p, SolverConfig(num_solutions=1),
    )
    sols_max_feat = MIPSolver(objective="max_features").solve(
        p, SolverConfig(num_solutions=1),
    )
    # Both meet all targets when budget allows it.
    assert all(sols_min_pen[0].targets_met.values())
    assert all(sols_max_feat[0].targets_met.values())
    # min_penalties is hierarchical: penalty first, cost second. It should
    # not spend MORE than max_features when both achieve zero penalty.
    assert sols_min_pen[0].cost <= sols_max_feat[0].cost + 1e-6


# --- max_weighted_features (spf-weighted; additive, max_features untouched) ---
def test_max_weighted_features_stores_objective():
    solver = MIPSolver(objective="max_weighted_features")
    assert solver.objective == "max_weighted_features"


def test_max_weighted_features_rejects_when_no_cost_budget():
    p = _two_feature_problem()
    with pytest.raises(ValueError, match="COSTBUDGET"):
        MIPSolver(objective="max_weighted_features").solve(
            p, SolverConfig(num_solutions=1)
        )


def test_max_weighted_features_uniform_spf_matches_max_features_count():
    # With every spf == 1, weighted == unweighted: same number of targets met.
    wf = MIPSolver(objective="max_weighted_features").solve(
        _two_feature_problem(cost_budget=1.0), SolverConfig(num_solutions=1)
    )
    mf = MIPSolver(objective="max_features").solve(
        _two_feature_problem(cost_budget=1.0), SolverConfig(num_solutions=1)
    )
    n_wf = sum(bool(v) for v in wf[0].targets_met.values())
    n_mf = sum(bool(v) for v in mf[0].targets_met.values())
    assert n_wf == n_mf


def test_max_weighted_features_prefers_high_spf_under_tight_budget():
    # Two features, budget pays for only one. Feature 2 has spf 10 vs 1;
    # the weighted objective must pick the target that includes feature 2.
    planning_units = pd.DataFrame(
        {"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]}
    )
    features = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 10.0]}
    )
    # feature 1 only in PU1; feature 2 only in PU2.
    puvspr = pd.DataFrame(
        {"species": [1, 2], "pu": [1, 2], "amount": [1.0, 1.0]}
    )
    p = ConservationProblem(
        planning_units, features, puvspr, parameters={"COSTBUDGET": 1.0}
    )
    sol = MIPSolver(objective="max_weighted_features").solve(
        p, SolverConfig(num_solutions=1)
    )[0]
    # PU2 (carrying the spf-10 feature) is chosen.
    assert bool(sol.selected[1]) is True
    assert bool(sol.selected[0]) is False
