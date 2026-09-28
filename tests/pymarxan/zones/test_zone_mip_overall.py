"""ZoneMIPSolver on the spec §5 anchor: overall targets are hard constraints (Watts eq. 6)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.mip_solver import ZoneMIPSolver
from pymarxan.zones.objective import check_overall_targets, check_zone_targets
from pymarxan.zones.readers import load_zone_project
from tests.pymarxan.zones.marzone_anchor import OPTIMUM, OPTIMUM_COST, make_anchor_problem

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"
CONFIG = SolverConfig(num_solutions=1)


class TestAnchorBehaviours:
    def test_new_semantics_find_the_hard_optimum(self):
        sol = ZoneMIPSolver().solve(make_anchor_problem(spf=1.0), CONFIG)[0]
        assert tuple(int(z) for z in sol.zone_assignment) == OPTIMUM
        assert sol.cost == pytest.approx(OPTIMUM_COST)
        assert sol.objective == pytest.approx(OPTIMUM_COST)
        assert sol.targets_met == {1: True}
        assert sol.metadata["zone_targets_met"] == {"z2_f1": True}
        assert sol.penalty == 0.0
        assert sol.metadata["overall_penalty"] == 0.0

    def test_old_answer_is_no_longer_returned(self):
        """v0.35 returned (2, 2, 2) at cost 3 with targets_met {1: True} (spec §1)."""
        sol = ZoneMIPSolver().solve(make_anchor_problem(), CONFIG)[0]
        assert tuple(int(z) for z in sol.zone_assignment) != (2, 2, 2)
        assert sol.cost > 3.0

    def test_contribution_weighted_zone_targets_are_infeasible(self):
        assert ZoneMIPSolver().solve(make_anchor_problem(zone_target_contrib=1), CONFIG) == []

    def test_misslevel_relaxes_the_overall_target(self):
        # MISSLEVEL 0.8: overall target 12, zone target 8 -> (2, 2, 2) at cost 3 is feasible.
        sol = ZoneMIPSolver().solve(make_anchor_problem(misslevel=0.8), CONFIG)[0]
        assert tuple(int(z) for z in sol.zone_assignment) == (2, 2, 2)
        assert sol.cost == pytest.approx(3.0)
        assert sol.targets_met == {1: True}

    def test_spf_does_not_change_the_hard_optimum(self):
        sol = ZoneMIPSolver().solve(make_anchor_problem(spf=10.0), CONFIG)[0]
        assert tuple(int(z) for z in sol.zone_assignment) == OPTIMUM

    def test_feature_with_target_but_no_amounts_is_infeasible(self):
        p = make_anchor_problem()
        features = pd.concat([
            p.features,
            pd.DataFrame({"id": [2], "name": ["ghost"], "target": [5.0], "spf": [1.0]}),
        ], ignore_index=True)
        p = p.copy_with(features=features)
        assert ZoneMIPSolver().solve(p, CONFIG) == []

    def test_inert_overall_target_adds_no_constraint(self):
        p = make_anchor_problem()
        p.features.loc[0, "target"] = 0.0
        sol = ZoneMIPSolver().solve(p, CONFIG)[0]
        # Only the raw zone target (zone 2 >= 10) binds: one PU in zone 2 at cost 1.
        assert sol.cost == pytest.approx(1.0)
        assert sol.targets_met == {1: True}


class TestFixtureUnderNewSemantics:
    def test_fixture_is_feasible_and_both_tiers_met(self):
        p = load_zone_project(DATA_DIR)
        sols = ZoneMIPSolver().solve(p, CONFIG)
        assert len(sols) == 1
        sol = sols[0]
        assert sol.all_targets_met
        assert all(sol.metadata["zone_targets_met"].values())
        assert check_overall_targets(p, sol.zone_assignment) == {1: True, 2: True}
        assert all(check_zone_targets(p, sol.zone_assignment).values())

    def test_witness_assignment_is_feasible_by_hand(self):
        """(1, 1, 2, 1): overall f1 = 23 + 3 = 26 >= 20, f2 = 16 + 2.7 = 18.7 >= 15;
        zone 1 raw (23, 16) >= (10, 8); zone 2 raw (6, 9) >= (5, 3)."""
        p = load_zone_project(DATA_DIR)
        a = np.array([1, 1, 2, 1])
        assert check_overall_targets(p, a) == {1: True, 2: True}
        assert all(check_zone_targets(p, a).values())


# Enumerated tests/data/zones (4 PUs x {unassigned, zone1, zone2} = 81 assignments) in a
# throwaway script (not committed), with problem.parameters["BLM"] = 0.0. Kept only the
# assignments meeting every zone target (check_zone_targets) AND every overall target
# (check_overall_targets) and took the minimum compute_zone_objective(p, a, 0.0), separately
# for ZONETARGETCONTRIB=0 (raw) and =1 (weighted). Both modes turned out to share the same
# unique argmin/objective on this fixture: raw had 10 feasible assignments (best
# (1, 1, 2, 2) at 460.0, next 480.0), weighted had 4 feasible assignments -- a strict subset
# of raw's -- with the same best (1, 1, 2, 2) at 460.0 (next 500.0). Both argmins are unique.
RAW_OPTIMUM = (1, 1, 2, 2)
RAW_OBJECTIVE = 460.0
WEIGHTED_OPTIMUM = (1, 1, 2, 2)
WEIGHTED_OBJECTIVE = 460.0


class TestFixtureModeSwitch:
    """Raw vs. weighted zone-target modes on the fixture, now that overall targets bind.

    Empirically the two modes land on the *same* unique optimum here: enforcing the
    overall targets already excludes the Task-3-era raw optimum (1, 1, 0, 2) @ 310 (its
    overall f2 = 12 + 4 x 0.3 = 13.2 < 15), which is exactly what pushed raw up to 460 and
    made it collide with weighted's (already-460) optimum. Raw's feasible set (10
    assignments) still strictly contains weighted's (4); the two just happen to share an
    argmin. Kept as two solves so a future fixture change that does separate them is
    caught by whichever constant drifts.
    """

    def _solve(self, flag: int):
        p = load_zone_project(DATA_DIR)
        p.parameters["BLM"] = 0.0
        p.parameters["ZONETARGETCONTRIB"] = flag
        return ZoneMIPSolver().solve(p, CONFIG)[0]

    def test_raw_mode_optimum(self):
        sol = self._solve(0)
        assert tuple(int(z) for z in sol.zone_assignment) == RAW_OPTIMUM
        assert sol.objective == pytest.approx(RAW_OBJECTIVE)

    def test_weighted_mode_optimum(self):
        sol = self._solve(1)
        assert tuple(int(z) for z in sol.zone_assignment) == WEIGHTED_OPTIMUM
        assert sol.objective == pytest.approx(WEIGHTED_OBJECTIVE)
