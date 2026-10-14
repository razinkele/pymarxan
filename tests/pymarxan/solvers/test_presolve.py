"""presolve_check (spec §4, Testing 10, 14; Review Focus 4)."""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers import MIPSolver, PresolveWarning, presolve_check
from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZoneMIPSolver, load_zone_project
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

DATA = Path(__file__).parent.parent.parent / "data"
ONE = SolverConfig(num_solutions=1)


def _core(**over: object) -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [2.0], "spf": [1.0]})
    pvf = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [1.0, 2.0]})
    bnd = pd.DataFrame({"id1": [1], "id2": [2], "boundary": [1.0]})
    params = {"BLM": 0.0}
    kw = {"planning_units": pu, "features": feat, "pu_vs_features": pvf, "boundary": bnd,
          "parameters": params}
    kw.update(over)
    return ConservationProblem(**kw)  # type: ignore[arg-type]


def _codes(msgs: list[str]) -> set[str]:
    return {m.split(":", 1)[0] for m in msgs}


def test_fixtures_clean() -> None:
    assert presolve_check(load_project(DATA / "simple")) == []
    assert presolve_check(make_anchor_problem()) == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert presolve_check(load_zone_project(DATA / "zones")) == []


def test_validate_short_circuit() -> None:
    p = _core(features=pd.DataFrame({"id": [1], "name": ["a"]}))  # no target column
    msgs = presolve_check(p)
    assert len(msgs) == 1 and msgs[0].startswith("P0: problem fails validate()")


@pytest.mark.parametrize("code, problem", [
    ("P1", lambda: _core(planning_units=pd.DataFrame(
        {"id": [1, 2, 3], "cost": [1.0, -3.0, 3.0], "status": [0, 0, 0]}))),
    ("P2", lambda: _core(planning_units=pd.DataFrame(
        {"id": [1, 2, 3], "cost": [0.01, 2.0, 1.2e5], "status": [0, 0, 0]}))),
    ("P3", lambda: _core(pu_vs_features=pd.DataFrame(
        {"species": [1, 1], "pu": [1, 2], "amount": [1e-4, 1e4]}))),
    ("P4", lambda: _core(boundary=pd.DataFrame(
        {"id1": [1, 2], "id2": [2, 3], "boundary": [1e-4, 1e4]}))),
    ("P5", lambda: _core(features=pd.DataFrame(
        {"id": [1, 2], "name": ["a", "g"], "target": [2.0, 10.0], "spf": [1.0, 1.0]}))),
    ("P6", lambda: _core(features=pd.DataFrame(
        {"id": [1], "name": ["a"], "target": [50.0], "spf": [1.0]}))),
    ("P7a", lambda: _core(planning_units=pd.DataFrame(
        {"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [2, 2, 2]}))),
    ("P7b", lambda: _core(planning_units=pd.DataFrame(
        {"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [3, 3, 3]}))),
    ("P8", lambda: _core(parameters={"BLM": 1e8})),
    ("P9a", lambda: _core(
        planning_units=pd.DataFrame(
            {"id": [1, 2, 3], "cost": [1.0, 20.0, 3.0], "status": [0, 2, 0]}),
        parameters={"BLM": 0.0, "COSTBUDGET": 5.0})),
    ("P9b", lambda: _core(parameters={"BLM": 0.0, "COSTBUDGET": 0.5})),
    ("P11", lambda: _core(parameters={"BLM": 0.0, "COSTBUDGET": 100.0})),
])
def test_each_check_triggers(code: str, problem: object) -> None:
    msgs = presolve_check(problem())  # type: ignore[operator]
    assert code in _codes(msgs), msgs


def test_p6_ignores_locked_out_amounts() -> None:
    p = _core(
        planning_units=pd.DataFrame(
            {"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 3, 0]}),
    )  # target 2; PU2's amount 2 is locked out, only 1 is reachable
    assert "P6" in _codes(presolve_check(p))


def test_zone_checks_use_rules() -> None:
    p = make_anchor_problem()
    p = p.copy_with(parameters={**p.parameters, "MANDATORYALLOCATION": 1, "COSTBUDGET": 2.0})
    assert "P9a" in _codes(presolve_check(p))  # floor 3 > 2
    q = make_anchor_problem().copy_with(
        planning_units=make_anchor_problem().planning_units.assign(status=3))
    assert "P7b" in _codes(presolve_check(q))
    # P11 ceiling = Σ dearest allowed option per PU (zone 1: 5 + 6 + 7 = 18), not the
    # cheapest (zone 2: 1 + 1 + 1 = 3).
    r = make_anchor_problem()
    at = r.copy_with(parameters={**r.parameters, "COSTBUDGET": 18.0})
    below = r.copy_with(parameters={**r.parameters, "COSTBUDGET": 17.0})
    assert "P11" in _codes(presolve_check(at))
    assert "P11" not in _codes(presolve_check(below))


def test_presolve_nan_amount_never_raises() -> None:
    p = _core(pu_vs_features=pd.DataFrame(
        {"species": [1, 1], "pu": [1, 2], "amount": [np.nan, 2.0]}))
    msgs = presolve_check(p)
    assert _codes(msgs) & {"P0", "P1"}
    q = _core(planning_units=pd.DataFrame(
        {"id": [1, 2, 3], "cost": [1.0, np.inf, 3.0], "status": [0, 0, 0]}))
    assert _codes(presolve_check(q)) & {"P0", "P1"}


def test_flag_values() -> None:
    for bad in (0, 1, "yes", None):
        with pytest.raises(ValueError, match="presolve_check"):
            MIPSolver(presolve_check=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="presolve_check"):
            ZoneMIPSolver(presolve_check=bad)  # type: ignore[arg-type]


def test_flag_true_warns_once_at_caller() -> None:
    p = _core(parameters={"BLM": 1e8})
    for call in (MIPSolver(presolve_check=True).solve,
                 MIPSolver(presolve_check=True).build_model):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            call(p)
        hits = [x for x in w if issubclass(x.category, PresolveWarning)]
        assert len(hits) == 1 and hits[0].filename == __file__
        assert "P8" in str(hits[0].message)


def test_flag_strict_raises_and_off_silent() -> None:
    p = _core(parameters={"BLM": 1e8})
    with pytest.raises(ValueError, match="P8"):
        MIPSolver(presolve_check="strict").solve(p, ONE)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PresolveWarning)
        MIPSolver().solve(p, ONE)


def test_zone_flag_warns_once_at_caller_and_strict_raises() -> None:
    p = make_anchor_problem().copy_with(
        parameters={"BLM": 0.0, "MANDATORYALLOCATION": 1, "COSTBUDGET": 2.0},  # floor 3 > 2: P9a
    )
    for call in (ZoneMIPSolver(objective="max_features", presolve_check=True).solve,
                 ZoneMIPSolver(objective="max_features", presolve_check=True).build_model):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            call(p)
        hits = [x for x in w if issubclass(x.category, PresolveWarning)]
        assert len(hits) == 1 and hits[0].filename == __file__
        assert "P9a" in str(hits[0].message)
    with pytest.raises(ValueError, match="P9a"):
        ZoneMIPSolver(objective="max_features", presolve_check="strict").solve(p, ONE)


def test_model_scan_p10_tiny_relative_target() -> None:
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [1e-8], "spf": [1.0]})
    p = _core(features=feat, parameters={"BLM": 0.0, "COSTBUDGET": 2.0})
    with pytest.warns(PresolveWarning, match="P10"):
        MIPSolver(objective="min_shortfall", presolve_check=True).solve(p, ONE)
    assert "P10" not in _codes(presolve_check(p))  # data-only function has no model


def test_model_scan_p10_tiny_rhs_without_objective_coefficient() -> None:
    # min_set: no objective coefficient is extreme; the 1e-8 target is only an rhs. Dropping
    # the rhs visit or the small-value branch in scan_model survives the test above.
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [1e-8], "spf": [1.0]})
    with pytest.warns(PresolveWarning, match=r"P10.*rhs of target_1"):
        MIPSolver(presolve_check=True).solve(_core(features=feat), ONE)


def test_presolve_check_guards_internal_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(problem: ConservationProblem) -> list[str]:
        raise RuntimeError("x")

    monkeypatch.setattr("pymarxan.solvers.presolve._core_checks", boom)
    msgs = presolve_check(_core())
    assert len(msgs) == 1 and msgs[0].startswith("P0: presolve check failed")


@pytest.mark.parametrize("module", [
    "pymarxan", "pymarxan.solvers", "pymarxan.zones", "pymarxan.solvers.presolve",
])
def test_import_order_no_cycle(module: str) -> None:
    code = f"import {module}; import pymarxan; pymarxan.presolve_check; pymarxan.PresolveWarning"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[3],
                   env={"PYTHONPATH": "src", "PATH": ""})
