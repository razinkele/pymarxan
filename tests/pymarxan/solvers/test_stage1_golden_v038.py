"""v0.38 MIP models and selections that Stage 1 must not change (spec Testing 4, 5, 12).

GOLDEN was recorded on unmodified v0.38 (e5a066c). Regenerate only with a written reason:
    PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python \
        tests/pymarxan/solvers/test_stage1_golden_v038.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.zones import ZoneMIPSolver, load_zone_project
from tests.pymarxan.lp_canon import canonical_model, captured_models
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

DATA = Path(__file__).parent.parent.parent / "data"
GOLDEN_PATH = DATA / "golden" / "stage1_v038.json"
ONE = SolverConfig(num_solutions=1)


def make_relabs_problem(budget: float = 1.0) -> ConservationProblem:
    """Spec Testing 5: 3 PUs at cost 1. Feature a (target 100): PU1 60, PU3 30.
    Feature b (target 10): PU2 8, PU3 5. Absolute minimax -> PU1 (40), relative -> PU3 (0.7)."""
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "target": [100.0, 10.0], "spf": [1.0, 1.0],
    })
    pvf = pd.DataFrame({
        "species": [1, 1, 2, 2], "pu": [1, 3, 2, 3], "amount": [60.0, 30.0, 8.0, 5.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf,
        parameters={"COSTBUDGET": budget, "BLM": 0.0},
    )


def _infeasible_problem() -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "target": [10.0, 10.0], "spf": [1.0, 1.0],
    })
    pvf = pd.DataFrame({
        "species": [1, 1, 2, 2], "pu": [1, 2, 1, 2], "amount": [5.0, 5.0, 5.0, 5.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf,
        parameters={"COSTBUDGET": 1.0},
    )


def _zones_problem():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_zone_project(DATA / "zones")


def _model_and_selection(solver: Any, problem: Any) -> tuple[dict[str, Any], list[int]]:
    with captured_models() as models:
        sols = solver.solve(problem, ONE)
    sel = [int(v) for v in sols[0].selected] if sols else []
    return canonical_model(models[0]), sel


def record() -> dict[str, Any]:
    simple = load_project(DATA / "simple")
    out: dict[str, Any] = {}
    out["zone_min_set_anchor"], _ = _model_and_selection(ZoneMIPSolver(), make_anchor_problem())
    out["zone_min_set_zones"], _ = _model_and_selection(ZoneMIPSolver(), _zones_problem())
    out["core_min_set_simple"], _ = _model_and_selection(MIPSolver(), simple)
    mls = MIPSolver(objective="min_largest_shortfall")
    out["core_mls_abs_relabs"], out["core_mls_abs_relabs_selected"] = _model_and_selection(
        mls, make_relabs_problem(),
    )
    out["core_mls_abs_infeasible"], _ = _model_and_selection(mls, _infeasible_problem())
    _, out["core_min_penalties_simple_selected"] = _model_and_selection(
        MIPSolver(objective="min_penalties"), simple,
    )
    return out


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(GOLDEN_PATH.read_text())
    return data


def test_zone_min_set_models_unchanged(golden: dict[str, Any]) -> None:
    got, _ = _model_and_selection(ZoneMIPSolver(), make_anchor_problem())
    assert got == golden["zone_min_set_anchor"]
    got, _ = _model_and_selection(ZoneMIPSolver(), _zones_problem())
    assert got == golden["zone_min_set_zones"]


def test_core_min_set_model_unchanged(golden: dict[str, Any]) -> None:
    got, _ = _model_and_selection(MIPSolver(), load_project(DATA / "simple"))
    assert got == golden["core_min_set_simple"]


def test_core_absolute_min_largest_shortfall_unchanged(golden: dict[str, Any]) -> None:
    solver = MIPSolver(objective="min_largest_shortfall", shortfall="absolute")
    got, sel = _model_and_selection(solver, make_relabs_problem())
    assert got == golden["core_mls_abs_relabs"]
    assert sel == golden["core_mls_abs_relabs_selected"] == [1, 0, 0]
    got, _ = _model_and_selection(solver, _infeasible_problem())
    assert got == golden["core_mls_abs_infeasible"]


def test_core_min_penalties_integral_selection_unchanged(golden: dict[str, Any]) -> None:
    _, sel = _model_and_selection(
        MIPSolver(objective="min_penalties"), load_project(DATA / "simple"),
    )
    assert sel == golden["core_min_penalties_simple_selected"]


if __name__ == "__main__":
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(record(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH}")
