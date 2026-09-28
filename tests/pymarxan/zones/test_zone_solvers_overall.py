"""Zone SA / heuristic / II on the spec §5 anchor, and cross-solver agreement on the fixture."""
from __future__ import annotations

from pathlib import Path

import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.heuristic import ZoneHeuristicSolver
from pymarxan.zones.iterative_improvement import ZoneIISolver
from pymarxan.zones.mip_solver import ZoneMIPSolver
from pymarxan.zones.objective import (
    check_overall_targets,
    check_zone_targets,
    compute_zone_objective,
)
from pymarxan.zones.readers import load_zone_project
from pymarxan.zones.solver import ZoneSASolver

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
