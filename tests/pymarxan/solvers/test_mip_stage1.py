"""Stage 1 changes to the single-zone MIPSolver (spec §1 MIPSolver, S2, S9)."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pulp
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from tests.pymarxan.lp_canon import canonical_model, captured_models

ONE = SolverConfig(num_solutions=1)
SIMPLE = Path(__file__).parent.parent.parent / "data" / "simple"


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


@pytest.mark.parametrize("target", [1.0, 0.0])
def test_all_zero_cost_min_set_solves(target: float) -> None:
    pu = pd.DataFrame({"id": [1, 2], "cost": [0.0, 0.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [target], "spf": [1.0]})
    pvf = pd.DataFrame({"species": [1], "pu": [1], "amount": [1.0]})
    p = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf, parameters={"BLM": 0.0},
    )
    sols = MIPSolver().solve(p, ONE)
    assert len(sols) == 1
    assert sols[0].selected.tolist() == ([True, False] if target > 0 else [False, False])
    assert sols[0].metadata["mip_objective_value"] == 0.0


def test_build_model_equals_solved_model() -> None:
    p = load_project(SIMPLE)
    built = MIPSolver().build_model(p)
    with captured_models() as models:
        MIPSolver().solve(p, ONE)
    assert isinstance(built, pulp.LpProblem)
    assert canonical_model(built) == canonical_model(models[0])


def test_build_model_refuses_what_solve_refuses() -> None:
    p = load_project(SIMPLE)
    p = p.copy_with(parameters={**p.parameters, "PROBMODE": 3})
    with pytest.raises(NotImplementedError):
        MIPSolver(mip_chance_strategy="socp").build_model(p)


def test_large_mip_warning_once_with_caller_frame() -> None:
    p = load_project(SIMPLE)
    for call in (MIPSolver(warn_above_pu=1).solve, MIPSolver(warn_above_pu=1).build_model):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            call(p)
        large = [x for x in w if "warn_above_pu" in str(x.message)]
        assert len(large) == 1
        assert large[0].filename == __file__
