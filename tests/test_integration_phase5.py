"""Integration tests for Phase 5 features."""
from __future__ import annotations

import pandas as pd

from pymarxan.analysis.gap_analysis import compute_gap_analysis
from pymarxan.calibration.sensitivity import SensitivityConfig, run_sensitivity
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.registry import get_default_registry


def _problem() -> ConservationProblem:
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [10.0, 20.0, 15.0, 25.0],
        "status": [2, 0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1, 2],
        "name": ["bird", "mammal"],
        "target": [5.0, 4.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1, 2, 2],
        "pu": [1, 2, 3, 3, 4],
        "amount": [3.0, 4.0, 2.0, 5.0, 3.0],
    })
    bnd = pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "boundary": [1.0, 1.0]})
    return ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=bnd,
        parameters={"BLM": 1.0},
    )


def test_heuristic_end_to_end():
    solver = HeuristicSolver()
    sols = solver.solve(_problem(), SolverConfig(num_solutions=1))
    assert len(sols) == 1
    assert sols[0].all_targets_met


def test_gap_analysis_end_to_end():
    result = compute_gap_analysis(_problem())
    df = result.to_dataframe()
    assert len(df) == 2
    assert "percent_protected" in df.columns


def test_sensitivity_end_to_end():
    config = SensitivityConfig(multipliers=[0.8, 1.0, 1.2])
    result = run_sensitivity(_problem(), MIPSolver(), config)
    assert len(result.runs) == 6


def test_registry_includes_greedy():
    reg = get_default_registry()
    assert "greedy" in reg.list_solvers()
    solver = reg.create("greedy")
    sols = solver.solve(_problem(), SolverConfig(num_solutions=1))
    assert len(sols) == 1


def test_app_imports():
    import pymarxan_app.app  # noqa: F401
