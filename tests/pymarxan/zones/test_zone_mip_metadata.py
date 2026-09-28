"""Issue #1: ZoneMIPSolver.solve must not discard the zone_targets_met metadata."""
from __future__ import annotations

from pathlib import Path

import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.mip_solver import ZoneMIPSolver
from pymarxan.zones.readers import load_zone_project
from pymarxan.zones.solver import ZoneSASolver

ZONES_DIR = Path(__file__).resolve().parents[2] / "data" / "zones"


@pytest.fixture
def zone_problem():
    return load_zone_project(ZONES_DIR)


def _expected_keys(problem) -> set[str]:
    zt = problem.zone_targets
    return {f"z{int(z)}_f{int(f)}" for z, f in zip(zt["zone"], zt["feature"])}


def test_mip_metadata_keeps_zone_targets_met_and_status(zone_problem):
    sol = ZoneMIPSolver().solve(zone_problem, SolverConfig(num_solutions=1))[0]
    assert sol.metadata["status"] == "Optimal"
    assert "solver" in sol.metadata and "mip_backend" in sol.metadata
    met = sol.metadata["zone_targets_met"]
    assert set(met) == _expected_keys(zone_problem)
    assert all(isinstance(v, bool) for v in met.values())
    assert all(met.values())  # zone targets are hard constraints in the MIP


def test_mip_and_sa_agree_on_zone_targets_met_key_format(zone_problem):
    mip = ZoneMIPSolver().solve(zone_problem, SolverConfig(num_solutions=1))[0]
    zone_problem.parameters["NUMITNS"] = 2000
    zone_problem.parameters["NUMTEMP"] = 20
    sa = ZoneSASolver().solve(zone_problem, SolverConfig(num_solutions=1, seed=1))[0]
    assert set(mip.metadata["zone_targets_met"]) == set(sa.metadata["zone_targets_met"])
