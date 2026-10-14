"""MANDATORYALLOCATION (spec §2, Testing 8; Review Focus 3)."""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import pandas as pd
import pytest

from pymarxan.io.readers import read_input_dat
from pymarxan.io.writers import write_input_dat
from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import (
    ZonalProblem,
    ZoneHeuristicSolver,
    ZoneIISolver,
    ZoneLockWarning,
    ZoneMIPSolver,
    ZoneSASolver,
    load_zone_project,
)
from pymarxan.zones.diagnostics import min_forced_cost
from pymarxan.zones.model import rule_violations
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

ONE = SolverConfig(num_solutions=1)
ZONES_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


def _on(p: ZonalProblem, value: object = 1) -> ZonalProblem:
    return p.copy_with(parameters={**p.parameters, "MANDATORYALLOCATION": value})


def _six_pu() -> ZonalProblem:
    """Anchor + PU4 (status 0), PU5 (status 3), PU6 (status 1); zone costs z1 = 8, z2 = 1.5.

    Mandatory optimum: 10.0 at (1, 2, 2, 2, 0, 2) = anchor 7 + PU4 and PU6 in zone 2.
    """
    p = make_anchor_problem()
    pu = pd.concat([
        p.planning_units,
        pd.DataFrame({"id": [4, 5, 6], "cost": [0.0] * 3, "status": [0, 3, 1]}),
    ], ignore_index=True)
    zc = pd.concat([
        p.zone_costs,
        pd.DataFrame({
            "pu": [4, 5, 6, 4, 5, 6], "zone": [1, 1, 1, 2, 2, 2],
            "cost": [8.0, 8.0, 8.0, 1.5, 1.5, 1.5],
        }),
    ], ignore_index=True)
    return p.copy_with(planning_units=pu, zone_costs=zc)


@pytest.mark.parametrize("value", [1, 1.0])
def test_accessor_accepts(value: object) -> None:
    p = _on(make_anchor_problem(), value)
    assert p.mandatory_allocation() is True
    assert p.validate() == []


@pytest.mark.parametrize("value", ["1", True, 2, None])
def test_accessor_rejects(value: object) -> None:
    p = _on(make_anchor_problem(), value)
    assert p.mandatory_allocation() is False
    assert any("MANDATORYALLOCATION" in e for e in p.validate())


def test_rules_free_pus_must_be_assigned() -> None:
    rules = _on(_six_pu()).pu_zone_rules()
    assert [r.may_be_unassigned for r in rules] == [False, False, False, False, True, False]
    assert [r.start for r in rules] == [None, None, None, None, 0, 1]  # starts unchanged


def test_mip_optimum_matches_enumeration() -> None:
    p = _on(_six_pu())
    sol = ZoneMIPSolver().solve(p, ONE)[0]
    assert tuple(int(z) for z in sol.zone_assignment) == (1, 2, 2, 2, 0, 2)
    assert sol.cost == pytest.approx(10.0)  # 7 (anchor) + 1.5 + 1.5 (PU4, PU6 in zone 2)


def test_every_solver_respects_rules() -> None:
    p = _on(_six_pu())
    rules = p.pu_zone_rules()
    solvers = [
        ZoneMIPSolver(),
        ZoneSASolver(num_iterations=20_000, num_temp_steps=100),
        ZoneHeuristicSolver(),
        ZoneIISolver(),  # ITIMPTYPE is read from the parameters, set per case below
    ]
    for solver in solvers:
        # ITIMPTYPE 1 is the removal pass: it must not unassign a must-assign PU (Testing 8)
        cases = [1, 2, 3] if isinstance(solver, ZoneIISolver) else [None]
        for itimp in cases:
            q = p
            if itimp is not None:
                q = p.copy_with(parameters={**p.parameters, "ITIMPTYPE": itimp})
            for s in solver.solve(q, SolverConfig(num_solutions=1, seed=42)):
                assert rule_violations(rules, s.zone_assignment) == []
                zero = [k for k, z in enumerate(s.zone_assignment) if z == 0]
                assert zero == [4]  # only the status-3 PU


def test_heuristic_seed_changes_on_zones_data() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = load_zone_project(ZONES_DIR)
    p = p.copy_with(planning_units=p.planning_units.assign(status=0))
    off = ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1, seed=1))[0]
    on = ZoneHeuristicSolver().solve(_on(p), SolverConfig(num_solutions=1, seed=1))[0]
    assert (off.zone_assignment == 0).any()  # v0.38 seed leaves free PUs unassigned
    assert (on.zone_assignment != 0).all()


def test_mip_zero_target_assigns_cheapest_zone() -> None:
    p = make_anchor_problem()
    p = p.copy_with(features=p.features.assign(target=0.0), zone_targets=None)
    off = ZoneMIPSolver().solve(p, ONE)[0]
    on = ZoneMIPSolver().solve(_on(p), ONE)[0]
    assert tuple(off.zone_assignment) == (0, 0, 0)
    assert tuple(on.zone_assignment) == (2, 2, 2)  # zone 2 costs 1 each


def test_status3_warning_without_lock_tables_once() -> None:
    p = _on(_six_pu())
    assert p.pu_locks is None and p.pu_zones is None
    for call in (ZoneMIPSolver().solve, ZoneMIPSolver().build_model):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            call(p)
        hits = [x for x in w if issubclass(x.category, ZoneLockWarning)]
        assert len(hits) == 1 and hits[0].filename == __file__
        assert "locked out (status 3) while MANDATORYALLOCATION is 1" in str(hits[0].message)
    assert p.validate() == []  # a warning, not an error


def test_budget_floor_with_mandatory() -> None:
    p = _on(make_anchor_problem())
    assert min_forced_cost(p) == pytest.approx(3.0)  # three PUs at zone-2 cost 1
    p = p.copy_with(parameters={**p.parameters, "COSTBUDGET": 2.0})
    with pytest.warns(UserWarning, match=r"at least 3 > COSTBUDGET 2"):
        assert ZoneMIPSolver(objective="max_features").solve(p, ONE) == []


def test_single_zone_forces_every_eligible_pu() -> None:
    p = make_anchor_problem()
    p = p.copy_with(
        zones=p.zones.iloc[:1],
        zone_costs=p.zone_costs[p.zone_costs["zone"] == 1],
        zone_contributions=p.zone_contributions[p.zone_contributions["zone"] == 1],
        zone_targets=None,
    )
    sol = ZoneMIPSolver().solve(_on(p), ONE)[0]
    assert tuple(sol.zone_assignment) == (1, 1, 1)


def test_mandatory_non_contiguous_zone_ids() -> None:
    p = make_anchor_problem()
    remap = {1: 5, 2: 9}
    p = p.copy_with(
        zones=p.zones.assign(id=p.zones["id"].map(remap)),
        zone_costs=p.zone_costs.assign(zone=p.zone_costs["zone"].map(remap)),
        zone_contributions=p.zone_contributions.assign(
            zone=p.zone_contributions["zone"].map(remap)),
        zone_targets=p.zone_targets.assign(zone=p.zone_targets["zone"].map(remap)),
        parameters={**p.parameters, "MANDATORYALLOCATION": 1, "COSTBUDGET": 7.0},
    )
    sol = ZoneMIPSolver(objective="max_features").solve(p, ONE)[0]
    assert set(int(z) for z in sol.zone_assignment) <= {5, 9}
    assert 0 not in set(int(z) for z in sol.zone_assignment)
    min_set = ZoneMIPSolver().solve(p, ONE)[0]  # the min_set cost lookup must not re-index
    assert tuple(int(z) for z in min_set.zone_assignment) == (5, 9, 9)
    assert min_set.cost == pytest.approx(7.0)  # and min_set ignores COSTBUDGET
    tight = p.copy_with(parameters={**p.parameters, "COSTBUDGET": 2.0})
    with pytest.warns(UserWarning, match=r"at least 3 > COSTBUDGET 2"):
        ZoneMIPSolver(objective="max_features").solve(tight, ONE)


def test_input_dat_round_trip(tmp_path: Path) -> None:
    params = {"MANDATORYALLOCATION": 1, "BLM": 0.0}
    write_input_dat(params, tmp_path / "input.dat")
    assert read_input_dat(tmp_path / "input.dat")["MANDATORYALLOCATION"] == 1


def test_off_matches_enumerated_unassigned_optimum() -> None:
    # With the parameter off, the anchor still prefers (1, 2, 2) at 7; unassigned allowed.
    assert tuple(ZoneMIPSolver().solve(make_anchor_problem(), ONE)[0].zone_assignment) == (1, 2, 2)
    assert len(list(itertools.product([0, 1, 2], repeat=3))) == 27
