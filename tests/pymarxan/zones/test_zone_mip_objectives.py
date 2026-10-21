"""ZoneMIPSolver budgeted objectives (spec §1, Testing 1-4, 6, 7; Review Focus 1, 2, 5)."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.zones import ZonalProblem, ZoneMIPSolver
from pymarxan.zones.diagnostics import ZoneMIPNoSolutionWarning, min_forced_cost
from tests.pymarxan.zones.marzone_anchor import (
    AMOUNT,
    CONTRIB,
    OPTIMUM,
    OVERALL_TARGET,
    ZONE_COSTS,
    ZONE_TARGET,
    all_assignments,
    make_anchor_problem,
    oracle,
)

ONE = SolverConfig(num_solutions=1)
BUDGETED = ("max_features", "max_weighted_features", "min_shortfall", "min_largest_shortfall")
BUDGETS = (0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 13)


def _with_budget(p: ZonalProblem, b: float) -> ZonalProblem:
    return p.copy_with(parameters={**p.parameters, "COSTBUDGET": b})


def _anchor_terms(a: tuple[int, ...]) -> list[tuple[float, float]]:
    """(achieved, required) for the anchor's two terms: overall, then zone (2, 1)."""
    overall = sum(AMOUNT * CONTRIB[z] for z in a if z)
    zone2 = sum(AMOUNT for z in a if z == 2)
    return [(overall, OVERALL_TARGET), (zone2, ZONE_TARGET)]


def _oracle(a: tuple[int, ...], objective: str, spf: float) -> float:
    terms = _anchor_terms(a)
    short = [max(0.0, r - got) / r for got, r in terms]
    met = [got >= r - 1e-9 for got, r in terms]
    if objective == "max_features":
        return float(sum(met))
    if objective == "max_weighted_features":
        return spf * sum(met)
    if objective == "min_shortfall":
        return spf * sum(short)
    return max(short)


def _cost(a: tuple[int, ...]) -> float:
    return sum(ZONE_COSTS[z][i] for i, z in enumerate(a) if z)


@pytest.mark.parametrize("objective", BUDGETED)
@pytest.mark.parametrize("spf", [1.0, 3.0])
@pytest.mark.parametrize("budget", BUDGETS)
def test_anchor_matches_enumeration(objective: str, spf: float, budget: float) -> None:
    feasible = [a for a in all_assignments() if _cost(a) <= budget + 1e-9]
    values = [_oracle(a, objective, spf) for a in feasible]
    best = max(values) if objective.startswith("max") else min(values)
    p = _with_budget(make_anchor_problem(spf=spf), budget)
    sol = ZoneMIPSolver(objective=objective).solve(p, ONE)[0]
    assert sol.metadata["mip_objective_value"] == pytest.approx(best, abs=1e-6)
    assert sol.metadata["objective"] == objective
    assert _cost(tuple(int(z) for z in sol.zone_assignment)) <= budget + 1e-9


def test_anchor_unique_argmins() -> None:
    hard = [a for a in all_assignments() if oracle(a)["feasible"]]
    assert min(hard, key=lambda a: oracle(a)["cost"]) == OPTIMUM  # (1, 2, 2), cost 7

    def run(obj: str, b: float, spf: float = 1.0) -> tuple[tuple[int, ...], float]:
        s = ZoneMIPSolver(objective=obj).solve(
            _with_budget(make_anchor_problem(spf=spf), b), ONE,
        )[0]
        return tuple(int(z) for z in s.zone_assignment), s.metadata["mip_objective_value"]

    assert run("min_shortfall", 3) == ((2, 2, 2), pytest.approx(0.2))
    assert run("min_shortfall", 3, spf=3.0) == ((2, 2, 2), pytest.approx(0.6))
    assert run("min_largest_shortfall", 3) == ((2, 2, 2), pytest.approx(0.2))
    for obj in BUDGETED:
        assert run(obj, 7)[0] == (1, 2, 2)


def test_each_target_is_its_own_term() -> None:
    # D1: with B in [1, 6] the zone target (2,1) is met but the overall target is not.
    for b in range(1, 7):
        s = ZoneMIPSolver(objective="max_features").solve(
            _with_budget(make_anchor_problem(), b), ONE,
        )[0]
        assert s.metadata["mip_objective_value"] == pytest.approx(1.0)


def _two_feature_anchor() -> ZonalProblem:
    """Testing 3: add feature 2 (target 10, spf 5), amount 10 in PU3 only, contrib z1=1, z2=0."""
    p = make_anchor_problem()
    feats = pd.concat([
        p.features,
        pd.DataFrame({"id": [2], "name": ["f2"], "target": [10.0], "spf": [5.0]}),
    ], ignore_index=True)
    pvf = pd.concat([
        p.pu_vs_features, pd.DataFrame({"species": [2], "pu": [3], "amount": [10.0]}),
    ], ignore_index=True)
    contrib = pd.concat([
        p.zone_contributions,
        pd.DataFrame({"feature": [2, 2], "zone": [1, 2], "contribution": [1.0, 0.0]}),
    ], ignore_index=True)
    return _with_budget(
        p.copy_with(features=feats, pu_vs_features=pvf, zone_contributions=contrib), 7,
    )


def test_weighted_differs_from_count() -> None:
    p = _two_feature_anchor()
    cnt = ZoneMIPSolver(objective="max_features").solve(p, ONE)[0]
    wtd = ZoneMIPSolver(objective="max_weighted_features").solve(p, ONE)[0]
    assert tuple(cnt.zone_assignment) == (1, 2, 2)
    assert cnt.metadata["mip_objective_value"] == pytest.approx(2.0)
    assert tuple(wtd.zone_assignment) == (0, 0, 1)
    assert wtd.metadata["mip_objective_value"] == pytest.approx(5.0)


@pytest.mark.parametrize("objective", BUDGETED)
def test_budget_required(objective: str) -> None:
    with pytest.raises(ValueError, match=f"{objective}.*COSTBUDGET"):
        ZoneMIPSolver(objective=objective).solve(make_anchor_problem(), ONE)


def test_boundary_ignored_warning_once() -> None:
    p = _with_budget(make_anchor_problem(blm=2.0), 7)
    p = p.copy_with(boundary=pd.DataFrame({"id1": [1], "id2": [2], "boundary": [1.0]}))
    for call in (ZoneMIPSolver(objective="min_shortfall").solve,
                 ZoneMIPSolver(objective="min_shortfall").build_model):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            call(p)
        hits = [x for x in w if "ignores boundary penalties" in str(x.message)]
        assert len(hits) == 1
        assert hits[0].filename == __file__


def _boundary_hits(solver: ZoneMIPSolver, p: ZonalProblem) -> int:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solver.solve(p, ONE)
        solver.build_model(p)
    return sum("ignores boundary penalties" in str(x.message) for x in w)


def test_boundary_ignored_warning_negative_cases() -> None:
    bnd = pd.DataFrame({"id1": [1], "id2": [2], "boundary": [1.0]})
    quiet = _with_budget(make_anchor_problem(blm=0.0), 7).copy_with(boundary=bnd)
    # Budgeted, BLM 0, no zone boundary costs: nothing is ignored -> no warning.
    assert _boundary_hits(ZoneMIPSolver(objective="min_shortfall"), quiet) == 0
    zero_zbc = quiet.copy_with(zone_boundary_costs=pd.DataFrame(
        {"zone1": [1], "zone2": [2], "cost": [0.0]}))
    assert _boundary_hits(ZoneMIPSolver(objective="max_features"), zero_zbc) == 0
    # min_set applies the boundary terms: never warns, even with BLM > 0.
    loud = _with_budget(make_anchor_problem(blm=2.0), 7).copy_with(boundary=bnd)
    assert _boundary_hits(ZoneMIPSolver(), loud) == 0


def test_boundary_ignored_warning_zone_boundary_cost_only() -> None:
    # Zone boundary costs are scaled by BLM (MarZone, #9): at BLM 0 nothing is ignored.
    bnd = pd.DataFrame({"id1": [1], "id2": [2], "boundary": [1.0]})
    zbc = pd.DataFrame({"zone1": [1], "zone2": [2], "cost": [3.0]})
    quiet = _with_budget(make_anchor_problem(blm=0.0), 7).copy_with(
        boundary=bnd, zone_boundary_costs=zbc)
    assert _boundary_hits(ZoneMIPSolver(objective="min_shortfall"), quiet) == 0
    loud = _with_budget(make_anchor_problem(blm=0.5), 7).copy_with(
        boundary=bnd, zone_boundary_costs=zbc)
    assert _boundary_hits(ZoneMIPSolver(objective="min_shortfall"), loud) == 2  # solve + build


def test_budget_floor_message() -> None:
    p = make_anchor_problem()
    p = p.copy_with(planning_units=p.planning_units.assign(status=[2, 0, 0]))  # PU1 -> zone 1
    p = _with_budget(p, 4.0)  # forced cost 5 > 4
    assert min_forced_cost(p) == pytest.approx(5.0)
    with pytest.warns(ZoneMIPNoSolutionWarning, match=r"at least 5 > COSTBUDGET 4"):
        assert ZoneMIPSolver(objective="max_features").solve(p, ONE) == []


def _core_problem() -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [2.0, 3.0, 4.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2, 3], "name": ["a", "b", "g"], "target": [10.0, 6.0, 4.0],
        "spf": [1.0, 2.0, 1.0],
    })
    pvf = pd.DataFrame({
        "species": [1, 1, 2, 2], "pu": [1, 2, 2, 3], "amount": [6.0, 5.0, 4.0, 3.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf,
        parameters={"COSTBUDGET": 5.0, "BLM": 0.0},
    )


def _as_one_zone(p: ConservationProblem) -> ZonalProblem:
    pu = p.planning_units
    return ZonalProblem(
        planning_units=pu, features=p.features, pu_vs_features=p.pu_vs_features,
        boundary=None, parameters=dict(p.parameters),
        zones=pd.DataFrame({"id": [1], "name": ["z"]}),
        zone_costs=pd.DataFrame({"pu": pu["id"], "zone": 1, "cost": pu["cost"]}),
        zone_contributions=pd.DataFrame({
            "feature": p.features["id"], "zone": 1, "contribution": 1.0,
        }),
        zone_targets=None, zone_boundary_costs=None,
    )


def _with_target_zero_feature(p: ConservationProblem) -> ConservationProblem:
    """Add feature 4 (target 0, held by PU 1): always met, so it is no term (spec §1)."""
    feat = pd.concat([p.features, pd.DataFrame({
        "id": [4], "name": ["z"], "target": [0.0], "spf": [5.0],
    })], ignore_index=True)
    pvf = pd.concat([p.pu_vs_features, pd.DataFrame({
        "species": [4], "pu": [1], "amount": [2.0],
    })], ignore_index=True)
    return p.copy_with(features=feat, pu_vs_features=pvf)


@pytest.mark.parametrize("objective", BUDGETED)
@pytest.mark.parametrize("ghost", [False, True])
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("misslevel", [1.0, 0.9, 0.0])
def test_cross_solver_oracle(objective: str, ghost: bool, zero: bool, misslevel: float) -> None:
    p = _core_problem()
    if not ghost:
        p = p.copy_with(features=p.features[p.features["id"] != 3].reset_index(drop=True))
    if zero:
        p = _with_target_zero_feature(p)
    p = p.copy_with(parameters={**p.parameters, "MISSLEVEL": misslevel})
    core = MIPSolver(objective=objective).solve(p, ONE)[0]
    zone = ZoneMIPSolver(objective=objective).solve(_as_one_zone(p), ONE)[0]
    assert zone.metadata["mip_objective_value"] == pytest.approx(
        core.metadata["mip_objective_value"], abs=1e-6,
    )


def test_budgeted_missing_zone_cost_is_free() -> None:
    p = make_anchor_problem()
    zc = p.zone_costs
    p = _with_budget(p.copy_with(zone_costs=zc[~((zc["pu"] == 1) & (zc["zone"] == 1))]), 2)
    s = ZoneMIPSolver(objective="max_features").solve(p, ONE)[0]
    # PU1 in zone 1 is free: (1, 2, 2) costs 2 and meets both targets.
    assert tuple(s.zone_assignment) == (1, 2, 2)
    assert s.metadata["mip_objective_value"] == pytest.approx(2.0)


def test_min_forced_cost_missing_pair_is_free() -> None:
    # PU1 and PU2 forced into zone 1; the (PU1, zone 1) cost row is missing -> 0, PU2 costs 6.
    p = make_anchor_problem()
    zc = p.zone_costs
    p = p.copy_with(
        zone_costs=zc[~((zc["pu"] == 1) & (zc["zone"] == 1))],
        planning_units=p.planning_units.assign(status=[2, 2, 0]),
    )
    assert min_forced_cost(p) == pytest.approx(6.0)  # a strict lookup raises KeyError


def test_budgeted_no_terms() -> None:
    p = make_anchor_problem()
    p = _with_budget(
        p.copy_with(features=p.features.assign(target=0.0), zone_targets=None), 3,
    )
    for obj in BUDGETED:
        s = ZoneMIPSolver(objective=obj).solve(p, ONE)
        assert len(s) == 1
        assert s[0].metadata["mip_objective_value"] == pytest.approx(0.0)


def test_reused_instance_no_state_leak() -> None:
    solver = ZoneMIPSolver(objective="min_shortfall")
    p1 = _with_budget(make_anchor_problem(), 3)
    p2 = _with_budget(make_anchor_problem(spf=3.0), 7)
    solver.build_model(p1)
    a = solver.solve(p2, ONE)[0].metadata["mip_objective_value"]
    b = solver.solve(p1, ONE)[0].metadata["mip_objective_value"]
    assert a == pytest.approx(
        ZoneMIPSolver(objective="min_shortfall").solve(p2, ONE)[0].metadata["mip_objective_value"]
    )
    assert b == pytest.approx(0.2)


def test_min_set_metadata_and_alias() -> None:
    s = ZoneMIPSolver().solve(make_anchor_problem(), ONE)[0]
    assert s.metadata["objective"] == "min_set"
    assert s.metadata["mip_objective_value"] == pytest.approx(7.0)
    assert ZoneMIPSolver(objective="max_n_targets_met").objective == "max_features"
    with pytest.raises(ValueError, match="objective"):
        ZoneMIPSolver(objective="min_penalties")


def test_zone_target_term_uses_misslevel() -> None:
    # Zone target 15 at MISSLEVEL 0.5 -> R = 7.5: one PU in zone 2 (amount 10, cost 1)
    # meets it. Without the MISSLEVEL divisor R = 15 and nothing is met under budget 1.
    p = make_anchor_problem(misslevel=0.5)
    p = _with_budget(p.copy_with(zone_targets=p.zone_targets.assign(target=15.0)), 1)
    s = ZoneMIPSolver(objective="max_features").solve(p, ONE)[0]
    assert s.metadata["mip_objective_value"] == pytest.approx(1.0)


def test_zone_shortfall_absolute_differs_and_is_recorded() -> None:
    # Budget 1: one PU in zone 2 meets the zone target; overall falls short by 11 of 15.
    p = _with_budget(make_anchor_problem(), 1)
    rel = ZoneMIPSolver(objective="min_largest_shortfall").solve(p, ONE)[0]
    ab = ZoneMIPSolver(objective="min_largest_shortfall", shortfall="absolute").solve(
        p, ONE,
    )[0]
    assert rel.metadata["shortfall"] == "relative"
    assert rel.metadata["mip_objective_value"] == pytest.approx(11 / 15)
    assert ab.metadata["shortfall"] == "absolute"
    assert ab.metadata["mip_objective_value"] == pytest.approx(11.0)
