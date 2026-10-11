"""Zone solvers honour pu_locks / pu_zones / status rules (#6 spec §4)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.run_mode import RunModePipeline
from pymarxan.zones import (
    ZoneHeuristicSolver,
    ZoneIISolver,
    ZoneMIPNoSolutionWarning,
    ZoneMIPSolver,
    ZoneSASolver,
)
from pymarxan.zones.model import rule_violations
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

CONFIG = SolverConfig(num_solutions=1, seed=1)


def _table(rows):
    return pd.DataFrame(rows, columns=["pu", "zone"])


def _with_pu4(**kw):
    p = make_anchor_problem(**kw)
    pu = pd.concat(
        [p.planning_units, pd.DataFrame({"id": [4], "cost": [0.0], "status": [0]})],
        ignore_index=True,
    )
    zc = pd.concat(
        [p.zone_costs, pd.DataFrame({"pu": [4, 4], "zone": [1, 2], "cost": [3.0, 2.0]})],
        ignore_index=True,
    )
    return p.copy_with(planning_units=pu, zone_costs=zc)


def _strict_subset():
    """3 zones; zone 3 is free and fully contributing, but every PU is restricted to {1, 2}.
    Feasible under the restriction ((1, 2, 2) at 7.0); any solver ignoring it drifts to 3."""
    p = make_anchor_problem(spf=10.0)
    zones = pd.DataFrame({"id": [1, 2, 3], "name": ["protect", "restore", "use"]})
    zc = pd.concat(
        [p.zone_costs, pd.DataFrame({"pu": [1, 2, 3], "zone": [3, 3, 3], "cost": [0.0] * 3})],
        ignore_index=True,
    )
    contrib = pd.concat(
        [p.zone_contributions,
         pd.DataFrame({"feature": [1], "zone": [3], "contribution": [1.0]})],
        ignore_index=True,
    )
    return p.copy_with(
        zones=zones, zone_costs=zc, zone_contributions=contrib,
        pu_zones=_table([(i, z) for i in (1, 2, 3) for z in (1, 2)]),
    )


def _excludes_first():
    """_strict_subset + PU 4 (no features; zone costs 0 / 4 / 5) restricted to {2, 3}:
    its set excludes min(zone_ids) = 1, where it would be cheapest, so a solver that seeds
    it at zone 1 and only takes improving moves leaves it out of its rule."""
    p = _strict_subset()
    pu = pd.concat(
        [p.planning_units, pd.DataFrame({"id": [4], "cost": [0.0], "status": [0]})],
        ignore_index=True,
    )
    zc = pd.concat(
        [p.zone_costs,
         pd.DataFrame({"pu": [4, 4, 4], "zone": [1, 2, 3], "cost": [0.0, 4.0, 5.0]})],
        ignore_index=True,
    )
    return p.copy_with(
        planning_units=pu, zone_costs=zc,
        pu_zones=pd.concat([p.pu_zones, _table([(4, 2), (4, 3)])], ignore_index=True),
    )


def _mip(p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ZoneMIPSolver().solve(p, CONFIG)


def _assign(sols):
    return [tuple(int(z) for z in s.zone_assignment) for s in sols]


class TestMIP:
    def test_lock_moves_anchor_optimum(self):
        sols = _mip(make_anchor_problem().copy_with(pu_locks=_table([(1, 2)])))
        assert _assign(sols) == [(2, 1, 2)] and sols[0].objective == pytest.approx(8.0)

    def test_unrestricted_pu4_variant(self):
        sols = _mip(_with_pu4())
        assert _assign(sols) == [(1, 2, 2, 0)] and sols[0].objective == pytest.approx(7.0)

    def test_puzone_forbids_unassigned(self):
        sols = _mip(_with_pu4().copy_with(pu_zones=_table([(4, 1), (4, 2)])))
        assert _assign(sols) == [(1, 2, 2, 2)] and sols[0].objective == pytest.approx(9.0)

    def test_status_semantics_unchanged(self):
        p = make_anchor_problem().copy_with(
            planning_units=make_anchor_problem().planning_units.assign(status=[2, 0, 0]),
        )
        sols = _mip(p)
        assert sols[0].zone_assignment[0] == 1

    def test_strict_subset(self):
        p = _strict_subset()
        sols = _mip(p)
        assert _assign(sols) == [(1, 2, 2)]
        assert rule_violations(p.pu_zone_rules(), sols[0].zone_assignment) == []

    def test_rules_hold_on_output(self):
        p = _with_pu4().copy_with(pu_locks=_table([(1, 2)]), pu_zones=_table([(4, 1), (4, 2)]))
        sols = _mip(p)
        assert sols and rule_violations(p.pu_zone_rules(), sols[0].zone_assignment) == []

    def test_lock_induced_infeasibility_message(self):
        p = make_anchor_problem().copy_with(pu_locks=_table([(1, 2), (2, 2), (3, 2)]))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert ZoneMIPSolver().solve(p, CONFIG) == []
        msgs = [str(w.message) for w in rec if issubclass(w.category, ZoneMIPNoSolutionWarning)]
        assert len(msgs) == 1
        assert "feature 1 is 15 but at most 12" in msgs[0]
        assert "given the PU zone restrictions" in msgs[0]

    def _no_solution_message(self, p):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert ZoneMIPSolver().solve(p, CONFIG) == []
        msgs = [str(w.message) for w in rec if issubclass(w.category, ZoneMIPNoSolutionWarning)]
        assert len(msgs) == 1
        return msgs[0]

    def test_compete_message_carries_restriction_note(self):
        # the "targets compete" branch gets the note too, not only "Unreachable targets"
        p = make_anchor_problem(zone_target_contrib=1).copy_with(pu_zones=_table([(1, 1), (1, 2)]))
        msg = self._no_solution_message(p)
        assert "targets compete" in msg and "given the PU zone restrictions" in msg

    def test_no_restriction_note_without_rules(self):
        msg = self._no_solution_message(make_anchor_problem(zone_target_contrib=1))
        assert "targets compete" in msg and "given the PU zone restrictions" not in msg


def _fast(p, n=3000):
    return p.copy_with(parameters={**p.parameters, "NUMITNS": n, "NUMTEMP": 30})


def _solve(cls, p, n_sol=3, seed=3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls().solve(p, SolverConfig(num_solutions=n_sol, seed=seed))


RESTRICTED = [
    ("lock", lambda: make_anchor_problem(spf=10.0).copy_with(pu_locks=_table([(1, 2)]))),
    ("must_assign", lambda: _with_pu4(spf=10.0).copy_with(pu_zones=_table([(4, 1), (4, 2)]))),
    ("strict_subset", _strict_subset),
    ("excludes_first", _excludes_first),
]


class TestSA:
    @pytest.mark.parametrize("name,make", RESTRICTED, ids=[r[0] for r in RESTRICTED])
    def test_rules_hold(self, name, make):
        p = _fast(make())
        for s in _solve(ZoneSASolver, p):
            assert rule_violations(p.pu_zone_rules(), s.zone_assignment) == []


def _ii(p, itimptype):
    return p.copy_with(parameters={**p.parameters, "ITIMPTYPE": itimptype})


def _tie_pu4():
    """_strict_subset plus PU 4 (no features) restricted to {1, 2, 3} with zone costs
    5 / 1 / 1: zones 2 and 3 tie, both better than the min(allowed) seed 1."""
    p = _strict_subset()
    pu = pd.concat(
        [p.planning_units, pd.DataFrame({"id": [4], "cost": [0.0], "status": [0]})],
        ignore_index=True,
    )
    zc = pd.concat(
        [p.zone_costs,
         pd.DataFrame({"pu": [4, 4, 4], "zone": [1, 2, 3], "cost": [5.0, 1.0, 1.0]})],
        ignore_index=True,
    )
    pz = pd.concat([p.pu_zones, _table([(4, 1), (4, 2), (4, 3)])], ignore_index=True)
    return p.copy_with(planning_units=pu, zone_costs=zc, pu_zones=pz)


class TestHeuristicAndII:
    def test_heuristic_must_assign_seed_is_min_allowed(self):
        # seed min(allowed) = 1; greedy moves to the first strictly better zone (2), and the
        # 2/3 tie keeps it there. Seeding at max(allowed) = 3 would stay at 3.
        (s,) = _solve(ZoneHeuristicSolver, _tie_pu4(), n_sol=1)
        assert int(s.zone_assignment[3]) == 2

    @pytest.mark.parametrize("itimptype", [2, 3])
    def test_ii_must_assign_move_takes_lowest_tied_zone(self, itimptype):
        # addition pass (2) / swap pass (3) try other allowed zones in ascending order
        (s,) = _solve(ZoneIISolver, _ii(_tie_pu4(), itimptype), n_sol=1)
        assert int(s.zone_assignment[3]) == 2

    @pytest.mark.parametrize("name,make", RESTRICTED, ids=[r[0] for r in RESTRICTED])
    def test_heuristic_rules_hold(self, name, make):
        p = make()
        for s in _solve(ZoneHeuristicSolver, p, n_sol=1):
            assert rule_violations(p.pu_zone_rules(), s.zone_assignment) == []

    @pytest.mark.parametrize("itimptype", [1, 2, 3])
    @pytest.mark.parametrize("name,make", RESTRICTED, ids=[r[0] for r in RESTRICTED])
    def test_ii_rules_hold(self, name, make, itimptype):
        p = _ii(make(), itimptype)
        for s in _solve(ZoneIISolver, p, n_sol=1):
            assert rule_violations(p.pu_zone_rules(), s.zone_assignment) == []

    def test_ii_addition_moves_must_assign_pu_to_better_zone(self):
        p = _ii(_with_pu4(spf=10.0).copy_with(pu_zones=_table([(4, 1), (4, 2)])), 2)
        (s,) = _solve(ZoneIISolver, p, n_sol=1)
        assert int(s.zone_assignment[3]) == 2  # seeded at 1 (cost 3), zone 2 costs 2

    def test_improve_repairs_out_of_rule_input(self):
        p = _ii(_with_pu4(spf=10.0).copy_with(
            pu_locks=_table([(1, 2)]), pu_zones=_table([(4, 1), (4, 2)]),
        ), 3)
        (seed_sol,) = _solve(ZoneHeuristicSolver, make_anchor_problem(spf=10.0), n_sol=1)
        bad = seed_sol
        bad.zone_assignment = np.array([1, 2, 2, 0])  # PU1 not in its lock, PU4 unassigned
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = ZoneIISolver().improve(p, bad)
        assert rule_violations(p.pu_zone_rules(), out.zone_assignment) == []

    def test_improve_repairs_out_of_set_zone_of_unlocked_pu(self):
        # Pins the `zone != 0 and zone not in rule.allowed` half of the repair: a
        # zero-only repair (or relying on the II passes) leaves these PUs in zone 3.
        p = _ii(_strict_subset(), 3)
        (seed_sol,) = _solve(ZoneHeuristicSolver, make_anchor_problem(spf=10.0), n_sol=1)
        seed_sol.zone_assignment = np.array([3, 3, 3])  # zone 3 is outside every PU's {1, 2}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = ZoneIISolver().improve(p, seed_sol)
        assert rule_violations(p.pu_zone_rules(), out.zone_assignment) == []

    def test_improve_repairs_into_allowed_set_not_first_zone(self):
        # PU 4 may take only {2, 3}; zone 1 (cost 0) is cheapest, so a repair to
        # min(zone_ids) or to zone 1 is never undone by the II passes.
        p = _ii(_excludes_first(), 3)
        (seed_sol,) = _solve(ZoneHeuristicSolver, make_anchor_problem(spf=10.0), n_sol=1)
        seed_sol.zone_assignment = np.array([1, 2, 2, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = ZoneIISolver().improve(p, seed_sol)
        assert rule_violations(p.pu_zone_rules(), out.zone_assignment) == []

    def test_improve_repair_target_is_min_allowed(self):
        # ITIMPTYPE 1 = removal pass only, which skips must-assign PUs, so PU 4 keeps the
        # zone the repair gave it: min({2, 3}) = 2 (spec), not 3 or min(zone_ids) = 1.
        p = _ii(_excludes_first(), 1)
        (seed_sol,) = _solve(ZoneHeuristicSolver, make_anchor_problem(spf=10.0), n_sol=1)
        seed_sol.zone_assignment = np.array([1, 2, 2, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = ZoneIISolver().improve(p, seed_sol)
        assert tuple(int(z) for z in out.zone_assignment) == (1, 2, 2, 2)

    def test_improve_repairs_status_3_pu_in_a_zone(self):
        base = _with_pu4(spf=10.0)
        p = _ii(base.copy_with(
            planning_units=base.planning_units.assign(status=[0, 0, 3, 0]),
            pu_locks=_table([(1, 2)]), pu_zones=_table([(4, 1), (4, 2)]),
        ), 3)
        (seed_sol,) = _solve(ZoneHeuristicSolver, make_anchor_problem(spf=10.0), n_sol=1)
        seed_sol.zone_assignment = np.array([1, 2, 2, 0])  # PU 3 has status 3 but sits in zone 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = ZoneIISolver().improve(p, seed_sol)
        assert rule_violations(p.pu_zone_rules(), out.zone_assignment) == []

    @pytest.mark.parametrize("runmode", [0, 1, 2, 3, 4, 5, 6])
    def test_pipeline_respects_rules(self, runmode):
        p = _ii(_fast(_with_pu4(spf=10.0).copy_with(
            pu_locks=_table([(1, 2)]), pu_zones=_table([(4, 1), (4, 2)]),
        )), 3)  # ITIMPTYPE 3: the II stage of RUNMODEs 2/3/5/6 actually runs its passes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sols = RunModePipeline(runmode=runmode).solve(p, SolverConfig(num_solutions=2, seed=5))
        for s in sols:
            assert rule_violations(p.pu_zone_rules(), s.zone_assignment) == []

    @pytest.mark.parametrize(
        "cls", [ZoneMIPSolver, ZoneSASolver, ZoneHeuristicSolver, ZoneIISolver],
        ids=lambda c: c.__name__,
    )
    def test_non_contiguous_zone_ids_solvers(self, cls):
        p = make_anchor_problem(spf=10.0)
        relabel = {1: 5, 2: 9}
        p = _ii(_fast(p.copy_with(
            zones=p.zones.assign(id=p.zones["id"].map(relabel)),
            zone_costs=p.zone_costs.assign(zone=p.zone_costs["zone"].map(relabel)),
            zone_contributions=p.zone_contributions.assign(
                zone=p.zone_contributions["zone"].map(relabel),
            ),
            zone_targets=p.zone_targets.assign(zone=p.zone_targets["zone"].map(relabel)),
            planning_units=p.planning_units.assign(status=[2, 1, 0]),
            pu_zones=_table([(3, 9)]),
        )), 3)
        sols = _solve(cls, p, n_sol=1)
        assert sols
        for s in sols:
            assert set(int(z) for z in s.zone_assignment) <= {0, 5, 9}
            assert rule_violations(p.pu_zone_rules(), s.zone_assignment) == []

    @pytest.mark.parametrize(
        "cls", [ZoneMIPSolver, ZoneSASolver, ZoneHeuristicSolver, ZoneIISolver],
        ids=lambda c: c.__name__,
    )
    def test_all_forced_every_solver(self, cls):
        base = _with_pu4(spf=10.0)
        p = _fast(base.copy_with(
            planning_units=base.planning_units.assign(status=[0, 3, 0, 2]),
            pu_locks=_table([(1, 2), (3, 2)]),
            features=base.features.assign(target=0.0),  # forced plan must be feasible
            zone_targets=None,
        ))
        p = _ii(p, 3)
        sols = _solve(cls, p, n_sol=1)
        assert len(sols) == 1  # never vacuous (the MIP must not return [])
        assert tuple(int(z) for z in sols[0].zone_assignment) == (2, 0, 2, 1)
