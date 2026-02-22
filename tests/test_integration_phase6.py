"""Integration tests for Phase 6 features."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.registry import get_default_registry
from pymarxan.solvers.run_mode import RunModePipeline


def _problem() -> ConservationProblem:
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [10.0, 20.0, 15.0, 25.0],
        "status": [0, 0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1, 2],
        "name": ["bird", "mammal"],
        "target": [5.0, 4.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu": [1, 2, 2, 3],
        "amount": [3.0, 4.0, 5.0, 3.0],
    })
    bnd = pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "boundary": [1.0, 1.0]})
    return ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=bnd,
        parameters={"BLM": 1.0, "NUMITNS": 1000, "NUMTEMP": 100},
    )


@pytest.mark.integration
def test_all_heurtypes_produce_solutions():
    p = _problem()
    for ht in range(8):
        solver = HeuristicSolver(heurtype=ht)
        sols = solver.solve(p, SolverConfig(num_solutions=1, seed=42))
        assert len(sols) == 1
        assert sols[0].n_selected > 0


@pytest.mark.integration
def test_iterative_improvement_improves():
    p = _problem()
    heur = HeuristicSolver(heurtype=0)
    initial = heur.solve(p, SolverConfig(num_solutions=1, seed=42))[0]
    ii = IterativeImprovementSolver(itimptype=2)
    improved = ii.improve(p, initial)
    assert improved.objective <= initial.objective + 1e-10


@pytest.mark.integration
@pytest.mark.slow
def test_runmode_5_end_to_end():
    p = _problem()
    pipeline = RunModePipeline(runmode=5)
    sols = pipeline.solve(p, SolverConfig(num_solutions=1, seed=42))
    assert len(sols) == 1
    assert sols[0].objective >= 0


@pytest.mark.integration
def test_output_roundtrip(tmp_path):
    from pymarxan.io.readers import read_mvbest, read_ssoln, read_sum
    from pymarxan.io.writers import write_mvbest, write_ssoln, write_sum

    p = _problem()
    solver = HeuristicSolver()
    sols = solver.solve(p, SolverConfig(num_solutions=3, seed=42))

    write_mvbest(p, sols[0], tmp_path / "mvbest.csv")
    write_ssoln(p, sols, tmp_path / "ssoln.csv")
    write_sum(sols, tmp_path / "sum.csv")

    mv = read_mvbest(tmp_path / "mvbest.csv")
    ss = read_ssoln(tmp_path / "ssoln.csv")
    sm = read_sum(tmp_path / "sum.csv")

    assert len(mv) == 2
    assert len(ss) == 4
    assert len(sm) == 3


@pytest.mark.integration
def test_registry_includes_new_solvers():
    reg = get_default_registry()
    names = reg.list_solvers()
    assert "iterative_improvement" in names
    assert "pipeline" in names


@pytest.mark.integration
def test_app_imports():
    import pymarxan_app.app  # noqa: F401
