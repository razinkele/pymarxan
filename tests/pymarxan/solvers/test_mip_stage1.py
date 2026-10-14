"""Stage 1 changes to the single-zone MIPSolver (spec §1 MIPSolver, S2, S9)."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

ONE = SolverConfig(num_solutions=1)


def _two_pu(cost2: float, *, spf: bool = True) -> ConservationProblem:
    """S2/S9 fixture: PU1 cost 100 holds 0.01 of feature 1 (target 0.01); PU2 holds nothing."""
    pu = pd.DataFrame({"id": [1, 2], "cost": [100.0, cost2], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [0.01]})
    if spf:
        feat["spf"] = [1.0]
    pvf = pd.DataFrame({"species": [1], "pu": [1], "amount": [0.01]})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf, parameters={"BLM": 0.0},
    )


def _with_ghost(budget: float) -> ConservationProblem:
    """Feature 1 in PU1; feature 2 (target 5) has no pu_vs_features rows."""
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["a", "ghost"], "target": [1.0, 5.0], "spf": [3.0, 7.0],
    })
    pvf = pd.DataFrame({"species": [1], "pu": [1], "amount": [1.0]})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf,
        parameters={"COSTBUDGET": budget, "BLM": 0.0},
    )


def test_zero_cost_featureless_pu_still_solves() -> None:
    # S9: v0.38 returned [] because x_2 never entered the model (varValue None).
    sols = MIPSolver().solve(_two_pu(0.0), ONE)
    assert len(sols) == 1
    assert sols[0].selected.tolist() == [True, False]


@pytest.mark.parametrize("objective", ["max_features", "max_weighted_features"])
def test_ghost_feature_not_counted_as_met(objective: str) -> None:
    sol = MIPSolver(objective=objective).solve(_with_ghost(1.0), ONE)[0]
    expected = 3.0 if objective == "max_weighted_features" else 1.0
    # feature 1 (spf 3) only; v0.38 gave 2 / 10 (ghost met for free)
    assert sol.metadata["mip_objective_value"] == pytest.approx(expected)


def test_max_weighted_features_without_spf_column() -> None:
    p = _with_ghost(1.0)
    p = p.copy_with(features=p.features.drop(columns=["spf"]))
    sol = MIPSolver(objective="max_weighted_features").solve(p, ONE)[0]
    assert sol.metadata["mip_objective_value"] == pytest.approx(1.0)
