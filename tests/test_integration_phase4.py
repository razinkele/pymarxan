"""Integration tests for Phase 4 features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.calibration.sweep import SweepConfig, run_sweep
from pymarxan.calibration.parallel import run_sweep_parallel
from pymarxan.analysis.scenarios import ScenarioSet
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.registry import get_default_registry


def _small_problem() -> ConservationProblem:
    pu = pd.DataFrame({
        "id": [1, 2, 3], "cost": [10.0, 20.0, 15.0], "status": [0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1], "name": ["f1"], "target": [5.0], "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1], "pu": [1, 2, 3], "amount": [3.0, 4.0, 2.0],
    })
    bnd = pd.DataFrame({
        "id1": [1, 2], "id2": [2, 3], "boundary": [1.0, 1.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=bnd,
        parameters={"BLM": 1.0},
    )


def test_sweep_with_mip():
    """End-to-end: sweep BLM with MIP solver."""
    p = _small_problem()
    config = SweepConfig(param_grid={"BLM": [0.0, 1.0, 5.0]})
    result = run_sweep(p, MIPSolver(), config)
    assert len(result.solutions) == 3
    assert all(s.all_targets_met for s in result.solutions)


def test_parallel_sweep_with_mip():
    """End-to-end: parallel sweep with MIP solver."""
    p = _small_problem()
    config = SweepConfig(param_grid={"BLM": [0.0, 1.0]})
    result = run_sweep_parallel(p, MIPSolver(), config, max_workers=2)
    assert len(result.solutions) == 2


def test_scenario_workflow():
    """End-to-end: save scenarios and compare."""
    p = _small_problem()
    solver = MIPSolver()
    ss = ScenarioSet()

    for blm in [0.0, 1.0, 5.0]:
        p.parameters["BLM"] = blm
        sols = solver.solve(p, SolverConfig(num_solutions=1))
        ss.add(f"blm-{blm}", sols[0], {"BLM": blm})

    df = ss.compare()
    assert len(df) == 3
    assert "BLM" in df.columns


def test_registry_create_and_solve():
    """Registry creates working solver instances."""
    reg = get_default_registry()
    solver = reg.create("mip")
    p = _small_problem()
    sols = solver.solve(p, SolverConfig(num_solutions=1))
    assert len(sols) == 1


def test_app_imports():
    """App module imports successfully with new modules."""
    import pymarxan_app.app  # noqa: F401
