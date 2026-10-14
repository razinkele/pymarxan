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
from tests.pymarxan.solvers.test_stage1_golden_v038 import make_relabs_problem

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


def test_relative_is_default_and_differs_from_absolute() -> None:
    rel = MIPSolver(objective="min_largest_shortfall").solve(make_relabs_problem(), ONE)[0]
    ab = MIPSolver(objective="min_largest_shortfall", shortfall="absolute").solve(
        make_relabs_problem(), ONE,
    )[0]
    assert rel.selected.tolist() == [False, False, True]
    assert rel.metadata["mip_objective_value"] == pytest.approx(0.7)
    assert rel.metadata["shortfall"] == "relative"
    assert ab.selected.tolist() == [True, False, False]
    assert ab.metadata["mip_objective_value"] == pytest.approx(40.0)


def test_min_shortfall_values() -> None:
    p = make_relabs_problem()
    rel = MIPSolver(objective="min_shortfall").solve(p, ONE)[0]
    ab = MIPSolver(objective="min_shortfall", shortfall="absolute").solve(p, ONE)[0]
    assert rel.metadata["mip_objective_value"] == pytest.approx(1.2)  # PU2/PU3 tie
    assert ab.selected.tolist() == [True, False, False]
    assert ab.metadata["mip_objective_value"] == pytest.approx(50.0)


def test_absolute_min_shortfall_is_spf_weighted() -> None:
    # spf b = 3: PU1 -> 40 + 3 * 10 = 70 (unweighted would report 50).
    w = make_relabs_problem()
    w = w.copy_with(features=w.features.assign(spf=[1.0, 3.0]))
    sol = MIPSolver(objective="min_shortfall", shortfall="absolute").solve(w, ONE)[0]
    assert sol.selected.tolist() == [True, False, False]
    assert sol.metadata["mip_objective_value"] == pytest.approx(70.0)


def test_core_reused_instance_no_state_leak() -> None:
    # Review Focus 5 (core): a _build that caches terms on self would return p1's 0.7 for p2.
    solver = MIPSolver(objective="min_largest_shortfall")
    p1 = make_relabs_problem()
    p2 = p1.copy_with(features=p1.features.assign(target=[40.0, 10.0]))  # PU3 -> 0.5
    solver.build_model(p1)
    a = solver.solve(p2, ONE)[0].metadata["mip_objective_value"]
    b = solver.solve(p1, ONE)[0].metadata["mip_objective_value"]
    assert a == pytest.approx(0.5)
    assert b == pytest.approx(0.7)


def test_min_shortfall_requires_budget() -> None:
    p = make_relabs_problem()
    p = p.copy_with(parameters={"BLM": 0.0})
    with pytest.raises(ValueError, match="min_shortfall.*COSTBUDGET"):
        MIPSolver(objective="min_shortfall").solve(p, ONE)


def test_relative_counts_ghost_feature() -> None:
    sol = MIPSolver(objective="min_largest_shortfall").solve(_with_ghost(1.0), ONE)[0]
    assert sol.metadata["mip_objective_value"] == pytest.approx(1.0)


@pytest.mark.parametrize("misslevel", [0.0, 0.5])
def test_relative_misslevel_divisor(misslevel: float) -> None:
    p = make_relabs_problem()
    p = p.copy_with(parameters={**p.parameters, "MISSLEVEL": misslevel})
    sol = MIPSolver(objective="min_largest_shortfall").solve(p, ONE)[0]
    if misslevel == 0.0:
        assert sol.metadata["shortfall"] == "relative"  # written by Step 3 only: RED now
        assert sol.metadata["mip_objective_value"] == pytest.approx(0.0)  # every R <= 0 skipped
    else:
        # R_a = 50, R_b = 5: PU3 -> a 20/50 = 0.4, b 0 -> 0.4; PU1 -> b 5/5 = 1.0
        assert sol.selected.tolist() == [False, False, True]
        assert sol.metadata["mip_objective_value"] == pytest.approx(0.4)


def test_alias_and_bad_names() -> None:
    assert MIPSolver(objective="max_n_targets_met").objective == "max_features"
    with pytest.raises(ValueError, match="shortfall"):
        MIPSolver(shortfall="percent")
    with pytest.raises(ValueError, match="objective"):
        MIPSolver(objective="max_utility")


def test_min_penalties_lexicographic_on_fractional_data() -> None:
    # v0.38: M = Σcost + 1 = 102; 0.01·102 < 100 so it left PU1 out (target unmet).
    sol = MIPSolver(objective="min_penalties").solve(_two_pu(1.0), ONE)[0]
    assert sol.selected.tolist() == [True, False]
    assert sol.cost == pytest.approx(100.0)
    assert sol.metadata["mip_objective_value"] == pytest.approx(0.0)  # SPF·shortfall


def test_min_penalties_cost_is_secondary() -> None:
    # Both PUs meet the target; pass 2 must take the cheaper one.
    pu = pd.DataFrame({"id": [1, 2], "cost": [5.0, 3.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [0.5]})
    pvf = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]})
    p = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pvf)
    sol = MIPSolver(objective="min_penalties").solve(p, ONE)[0]
    assert sol.selected.tolist() == [False, True]


def test_min_penalties_build_model_is_pass1() -> None:
    m = MIPSolver(objective="min_penalties").build_model(_two_pu(1.0))
    assert "lex_primary_cap" not in m.constraints
    assert {v.name for v, _ in m.objective.items()} == {"s_f1"}


def test_min_penalties_objective_is_spf_weighted_and_cap_is_tight() -> None:
    # The ghost (target 5, spf 7) is unmeetable: pass 1 = 7 * 5 = 35, not 5.
    sol = MIPSolver(objective="min_penalties").solve(_with_ghost(1.0), ONE)[0]
    assert sol.metadata["mip_objective_value"] == pytest.approx(35.0)
    assert sol.selected.tolist() == [True, False]
    # Cap tightness: with spf 1e-6 on feature 1, dropping PU1 (cost 1) adds only 1e-6 of
    # weighted shortfall, so any pass-2 cap looser than 35 * (1 + 2.9e-8) (or abs_tol
    # >= 1e-6) lets cost win and returns [False, False]; the spec's rel_tol = abs_tol = 1e-9
    # keeps PU1.
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["a", "ghost"], "target": [1.0, 5.0], "spf": [1e-6, 7.0],
    })
    pvf = pd.DataFrame({"species": [1], "pu": [1], "amount": [1.0]})
    tight = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf,
        parameters={"COSTBUDGET": 1.0, "BLM": 0.0},
    )
    sol = MIPSolver(objective="min_penalties").solve(tight, ONE)[0]
    assert sol.selected.tolist() == [True, False]
    assert sol.metadata["mip_objective_value"] == pytest.approx(35.0, abs=1e-9)


@pytest.mark.parametrize("status", [[0, 0], [2, 0]])
def test_min_penalties_no_targets_is_min_cost(status: list[int]) -> None:
    # Every target 0: no terms, so the primary is constant and cost alone decides (as v0.38).
    pu = pd.DataFrame({"id": [1, 2], "cost": [5.0, 3.0], "status": status})
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [0.0]})
    pvf = pd.DataFrame({"species": [1], "pu": [1], "amount": [1.0]})
    p = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pvf)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sols = MIPSolver(objective="min_penalties").solve(p, ONE)
    assert len(sols) == 1
    assert sols[0].selected.tolist() == [status[0] == 2, False]
    assert sols[0].metadata["mip_objective_value"] == 0.0


def test_min_penalties_pass2_fallback_warns_at_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    orig = pulp.LpProblem.solve
    calls = {"n": 0}

    def flaky(self: pulp.LpProblem, *a: object, **k: object) -> int:
        calls["n"] += 1
        result = orig(self, *a, **k)
        if calls["n"] == 2:
            self.status = pulp.LpStatusInfeasible
        return result

    monkeypatch.setattr(pulp.LpProblem, "solve", flaky)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sol = MIPSolver(objective="min_penalties").solve(_two_pu(1.0), ONE)[0]
    lex = [w for w in caught if "pass 2" in str(w.message)]
    assert len(lex) == 1 and lex[0].filename == __file__
    assert sol.selected.tolist() == [True, False]  # pass-1 values
    assert sol.metadata["status"] == "Optimal"
