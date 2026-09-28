"""Zone SA / heuristic / II on the spec §5 anchor, and cross-solver agreement on the fixture."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.heuristic import ZoneHeuristicSolver
from pymarxan.zones.iterative_improvement import ZoneIISolver
from pymarxan.zones.mip_solver import ZoneMIPSolver
from pymarxan.zones.objective import (
    build_zone_solution,
    check_overall_targets,
    check_zone_targets,
    compute_zone_objective,
)
from pymarxan.zones.readers import load_zone_project
from pymarxan.zones.solver import ZoneSASolver
from tests.pymarxan.zones.marzone_anchor import HEURISTIC_SPF, OPTIMUM_COST, make_anchor_problem

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


def _fixture():
    p = load_zone_project(DATA_DIR)      # BLM 1.0, bound.dat and zoneboundcost.dat present
    p.parameters["NUMITNS"] = 500
    p.parameters["NUMTEMP"] = 20
    p.parameters["ITIMPTYPE"] = 3
    assert float(p.parameters["BLM"]) > 0 and p.boundary is not None
    return p


SOLVERS = [ZoneMIPSolver(), ZoneSASolver(), ZoneHeuristicSolver(), ZoneIISolver()]


class TestCrossSolverAgreement:
    @pytest.mark.parametrize("solver", SOLVERS, ids=lambda s: s.name())
    def test_objective_equals_reference_with_blm_and_boundary(self, solver):
        p = _fixture()
        sol = solver.solve(p, SolverConfig(num_solutions=1, seed=42))[0]
        blm = float(p.parameters["BLM"])
        assert sol.objective == pytest.approx(
            compute_zone_objective(p, sol.zone_assignment, blm), abs=1e-9,
        )

    @pytest.mark.parametrize("solver", SOLVERS, ids=lambda s: s.name())
    def test_targets_met_is_feature_keyed_and_zone_keys_are_strings(self, solver):
        p = _fixture()
        sol = solver.solve(p, SolverConfig(num_solutions=1, seed=42))[0]
        assert set(sol.targets_met) == {1, 2}
        assert sol.targets_met == check_overall_targets(p, sol.zone_assignment)
        assert set(sol.metadata["zone_targets_met"]) == {"z1_f1", "z1_f2", "z2_f1", "z2_f2"}
        expected = {
            f"z{z}_f{f}": v for (z, f), v in check_zone_targets(p, sol.zone_assignment).items()
        }
        assert sol.metadata["zone_targets_met"] == expected
        assert sol.metadata["solver"] == solver.name()
        assert sol.penalty == pytest.approx(
            sol.metadata["overall_penalty"] + sol.metadata["zone_penalty"],
        )

    def test_ii_improve_uses_shared_builder(self):
        p = _fixture()
        start = ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1))[0]
        improved = ZoneIISolver().improve(p, start)
        assert set(improved.targets_met) == {1, 2}
        assert "overall_penalty" in improved.metadata
        assert improved.objective == pytest.approx(
            compute_zone_objective(p, improved.zone_assignment, float(p.parameters["BLM"])),
        )


class TestHeuristicOnAnchor:
    def test_greedy_meets_both_tiers_and_lands_at_or_above_optimum(self):
        """Hand trace at spf = 10 from (0, 0, 0): PU1->z2 (obj 111), PU2->z1 (17),
        PU3->z2 (8); no further improving move. v0.35 stopped at (2, 0, 0) cost 1 because
        the zone target was already met."""
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        sol = ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1))[0]
        assert sol.targets_met == {1: True}
        assert sol.metadata["zone_targets_met"] == {"z2_f1": True}
        assert sol.cost >= OPTIMUM_COST
        assert tuple(int(z) for z in sol.zone_assignment) == (2, 1, 2)
        assert sol.cost == pytest.approx(8.0)
        assert sol.penalty == 0.0

    def test_greedy_no_longer_stops_when_only_zone_targets_are_met(self):
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        sol = ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1))[0]
        assert tuple(int(z) for z in sol.zone_assignment) != (2, 0, 0)

    def test_feature_without_amounts_is_penalised_not_fatal(self):
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        features = pd.concat([
            p.features,
            pd.DataFrame({"id": [2], "name": ["ghost"], "target": [5.0], "spf": [1.0]}),
        ], ignore_index=True)
        p = p.copy_with(features=features)
        sol = ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1))[0]
        assert sol.targets_met == {1: True, 2: False}
        assert sol.penalty == pytest.approx(5.0)
        assert sol.all_targets_met is False


class TestIIOnAnchor:
    def test_swap_pass_meets_both_tiers(self):
        """ITIMPTYPE 3 from the all-first-zone start (1, 1, 1): PU1->z2 (obj 14),
        PU2->z2 (9); second sweep finds nothing. Ends at (2, 2, 1), cost 9."""
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        p.parameters["ITIMPTYPE"] = 3
        sol = ZoneIISolver().solve(p, SolverConfig(num_solutions=1))[0]
        assert sol.targets_met == {1: True}
        assert sol.metadata["zone_targets_met"] == {"z2_f1": True}
        assert sol.cost >= OPTIMUM_COST
        assert tuple(int(z) for z in sol.zone_assignment) == (2, 2, 1)
        assert sol.cost == pytest.approx(9.0)

    def test_itimptype_zero_returns_the_start_unchanged(self):
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        p.parameters["ITIMPTYPE"] = 0
        sol = ZoneIISolver().solve(p, SolverConfig(num_solutions=1))[0]
        assert tuple(int(z) for z in sol.zone_assignment) == (1, 1, 1)
        assert sol.targets_met == {1: True}                       # 30 >= 15
        assert sol.metadata["zone_targets_met"] == {"z2_f1": False}
        assert sol.penalty == pytest.approx(HEURISTIC_SPF * 10.0)

    def test_improve_from_infeasible_corner(self):
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        p.parameters["ITIMPTYPE"] = 3
        start = build_zone_solution(p, np.array([2, 2, 2]), 0.0, solver_name="seed")
        improved = ZoneIISolver().improve(p, start)
        assert improved.targets_met == {1: True}
        assert improved.objective < start.objective


class TestSAOnAnchor:
    def test_sa_reaches_the_optimum_with_both_tiers_met(self):
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        p.parameters["NUMITNS"] = 3000
        p.parameters["NUMTEMP"] = 100
        sols = ZoneSASolver().solve(p, SolverConfig(num_solutions=3, seed=42))
        for sol in sols:
            assert sol.targets_met == {1: True}
            assert sol.metadata["zone_targets_met"] == {"z2_f1": True}
            assert sol.cost >= OPTIMUM_COST
            assert sol.objective == pytest.approx(
                compute_zone_objective(p, sol.zone_assignment, 0.0),
            )
        assert min(s.cost for s in sols) == pytest.approx(OPTIMUM_COST)

    def test_sa_all_locked_in_uses_shared_builder(self):
        p = make_anchor_problem(spf=HEURISTIC_SPF)
        p.planning_units["status"] = 2                     # all locked into zone 1
        p.parameters["NUMITNS"] = 100
        sol = ZoneSASolver().solve(p, SolverConfig(num_solutions=1, seed=1))[0]
        assert tuple(int(z) for z in sol.zone_assignment) == (1, 1, 1)
        assert sol.targets_met == {1: True}
        assert "overall_penalty" in sol.metadata
        assert sol.objective == pytest.approx(compute_zone_objective(p, sol.zone_assignment, 0.0))
