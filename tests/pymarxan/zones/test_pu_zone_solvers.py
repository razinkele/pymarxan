"""Zone solvers honour pu_locks / pu_zones / status rules (#6 spec §4)."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZoneMIPNoSolutionWarning, ZoneMIPSolver, ZoneSASolver
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
