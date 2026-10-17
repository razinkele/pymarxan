"""v0.41.1 binary results that Stage 4 (proportion decisions) must not change (spec §3, §8).

GOLDEN was recorded on unmodified v0.41.1 (8b133f4, Stage 4 spec commits only) before any
source change: the binary ``MIPSolver.build_model`` models (canonical form), the models a
binary ``MultiObjectiveMIP`` hierarchical solve hands to the backend, binary
``presolve_check`` messages, and ``build_solution`` outputs for fixed binary selections
(connectivity included). Every entry must reproduce exactly.

v0.43 (written reason): ``compute_connectivity_penalty`` now mirrors the MIP's
connectivity term, so the ``objective`` (and stored ``connectivity``) of the ``conn_*``
entries were re-recorded in place; every other field is unchanged.

Regenerate only with a written reason:
    PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python \
        tests/pymarxan/solvers/test_stage4_golden_v041.py
"""
from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pulp
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.multiobjective import Hierarchical, MaxFeatures, MinCost, MultiObjectiveMIP
from pymarxan.solvers.mip_solver import MIP_OBJECTIVES, MIPSolver
from pymarxan.solvers.presolve import presolve_check
from pymarxan.solvers.utils import build_solution, compute_connectivity_penalty
from tests.pymarxan.lp_canon import canonical_model

DATA = Path(__file__).parent.parent.parent / "data"
GOLDEN_PATH = DATA / "golden" / "stage4_v041.json"
BUDGET = 33.0


def _simple(**params: Any) -> ConservationProblem:
    p = load_project(DATA / "simple")
    return p.copy_with(parameters={**p.parameters, **params})


def _conn(asymmetric: bool) -> ConservationProblem:
    p = _simple(CONNECTIVITY_WEIGHT=2.0, ASYMMETRIC_CONNECTIVITY=int(asymmetric))
    conn = pd.DataFrame({
        "id1": [1, 2, 3, 2, 5], "id2": [2, 1, 4, 5, 6], "value": [4.0, 3.0, 2.5, 6.0, 1.5],
    })
    return p.copy_with(connectivity=conn)


def _locks(**params: Any) -> ConservationProblem:
    p = _simple(**params)
    pu = p.planning_units.copy()
    pu.loc[pu["id"] == 3, "status"] = 2  # locked in
    pu.loc[pu["id"] == 6, "status"] = 3  # locked out
    return p.copy_with(planning_units=pu)


def _prob(probmode: int) -> ConservationProblem:
    p = _simple(PROBMODE=probmode, PROBABILITYWEIGHTING=2.0)
    prob = pd.DataFrame({
        "pu": [1, 2, 3, 4, 5, 6], "probability": [0.1, 0.4, 0.0, 0.25, 0.05, 0.3],
    })
    return p.copy_with(probability=prob)


MODEL_VARIANTS: dict[str, Callable[[], ConservationProblem]] = {
    "simple": _simple,
    "conn_sym": lambda: _conn(False),
    "conn_asym": lambda: _conn(True),
    "locks": _locks,
    "probmode1": lambda: _prob(1),
    "probmode2": lambda: _prob(2),
}
MODEL_CASES = [f"{v}|min_set" for v in MODEL_VARIANTS] + ["simple|min_penalties"] + [
    f"budget|{o}" for o in MIP_OBJECTIVES if o not in ("min_set", "min_penalties")
]

PRESOLVE_CASES: dict[str, Callable[[], ConservationProblem]] = {
    "locks_budget_low": lambda: _locks(COSTBUDGET=10.0),  # P9a: below locked-in cost 20
    "locks_budget_high": lambda: _locks(COSTBUDGET=100.0),  # P11
    "budget_cheapest": lambda: _simple(COSTBUDGET=5.0),  # P9b: below min free cost 8
    "p6_target": lambda: _locks(MISSLEVEL=1.0).copy_with(
        features=_simple().features.assign(target=[30.0, 20.0, 40.0]),
    ),
}

SELECTIONS = {"235": [0, 1, 1, 0, 1, 0], "246": [0, 1, 0, 1, 0, 1], "none": [0] * 6,
              "all": [1] * 6}
SOLUTION_VARIANTS: dict[str, Callable[[], ConservationProblem]] = {
    "simple": _simple,
    "conn_sym": lambda: _conn(False),
    "conn_asym": lambda: _conn(True),
    "probmode1": lambda: _prob(1),
    "misslevel": lambda: _simple(MISSLEVEL=0.8),
}


def _build(case: str) -> dict[str, Any]:
    variant, objective = case.split("|")
    if variant == "budget":
        problem = _simple(COSTBUDGET=BUDGET)
    else:
        problem = MODEL_VARIANTS[variant]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        model = MIPSolver(objective=objective).build_model(problem)
    return canonical_model(model)


def _hierarchical() -> dict[str, Any]:
    """Canonical model at every backend solve of a binary [MaxFeatures, MinCost] run."""
    problem = _simple(COSTBUDGET=BUDGET)
    snaps: list[dict[str, Any]] = []
    orig = pulp.LpProblem.solve

    def spy(self: pulp.LpProblem, *args: Any, **kwargs: Any) -> Any:
        snaps.append(canonical_model(self))
        return orig(self, *args, **kwargs)

    mo = MultiObjectiveMIP([MaxFeatures(), MinCost()], Hierarchical(0.1), mip_backend="cbc")
    pulp.LpProblem.solve = spy  # type: ignore[method-assign]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sols = mo.solve(problem)
    finally:
        pulp.LpProblem.solve = orig  # type: ignore[method-assign]
    s = sols[0]
    return {
        "models": snaps,
        "selected": [int(v) for v in s.selected],
        "objective_values": {k: float(v) for k, v in s.metadata["objective_values"].items()},
        "cost": float(s.cost),
    }


def _solution(case: str) -> dict[str, Any]:
    variant, sel = case.split("|")
    problem = SOLUTION_VARIANTS[variant]()
    selected = np.array(SELECTIONS[sel], dtype=bool)
    blm = float(problem.parameters.get("BLM", 0.0))
    s = build_solution(problem, selected, blm)
    pu_index = {int(p): i for i, p in enumerate(problem.planning_units["id"])}
    return {
        "cost": float(s.cost),
        "boundary": float(s.boundary),
        "objective": float(s.objective),
        "penalty": float(s.penalty),
        "shortfall": float(s.shortfall),
        "targets_met": {str(k): bool(v) for k, v in sorted(s.targets_met.items())},
        "n_selected": int(s.n_selected),
        "connectivity": float(compute_connectivity_penalty(problem, selected, pu_index)),
    }


SOLUTION_CASES = [f"{v}|{s}" for v in SOLUTION_VARIANTS for s in SELECTIONS]


def record() -> dict[str, Any]:
    return {
        "models": {case: _build(case) for case in MODEL_CASES},
        "hierarchical": _hierarchical(),
        "presolve": {k: presolve_check(f()) for k, f in PRESOLVE_CASES.items()},
        "solutions": {case: _solution(case) for case in SOLUTION_CASES},
    }


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(GOLDEN_PATH.read_text())
    return data


@pytest.mark.parametrize("case", MODEL_CASES)
def test_binary_model_unchanged(golden: dict[str, Any], case: str) -> None:
    assert _build(case) == golden["models"][case]


def test_binary_hierarchical_models_unchanged(golden: dict[str, Any]) -> None:
    got = _hierarchical()
    want = golden["hierarchical"]
    assert got["selected"] == want["selected"]
    assert got["cost"] == pytest.approx(want["cost"], abs=1e-9)
    assert got["objective_values"] == pytest.approx(want["objective_values"], abs=1e-9)
    assert len(got["models"]) == len(want["models"])
    for g, w in zip(got["models"], want["models"], strict=True):
        assert g == w


@pytest.mark.parametrize("case", sorted(PRESOLVE_CASES))
def test_binary_presolve_messages_unchanged(golden: dict[str, Any], case: str) -> None:
    assert presolve_check(PRESOLVE_CASES[case]()) == golden["presolve"][case]


def test_presolve_fixtures_fire() -> None:
    """The golden is only useful if the budget / P6 codes actually occur."""
    codes = {k: [m.split(":")[0] for m in presolve_check(f())] for k, f in PRESOLVE_CASES.items()}
    assert "P9a" in codes["locks_budget_low"]
    assert "P11" in codes["locks_budget_high"]
    assert "P9b" in codes["budget_cheapest"]
    assert "P6" in codes["p6_target"]


@pytest.mark.parametrize("case", SOLUTION_CASES)
def test_binary_build_solution_unchanged(golden: dict[str, Any], case: str) -> None:
    got = _solution(case)
    want = golden["solutions"][case]
    for key, value in want.items():
        if isinstance(value, float):
            assert got[key] == pytest.approx(value, rel=1e-12, abs=1e-12), key
        else:
            assert got[key] == value, key


if __name__ == "__main__":
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(record(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH}")
