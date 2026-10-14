"""write_problem (spec §3, Testing 9)."""
from __future__ import annotations

import re
import subprocess
import warnings
from pathlib import Path

import pulp
import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers import HeuristicSolver, MIPSolver, write_problem
from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZoneMIPSolver
from tests.pymarxan.solvers.test_stage1_golden_v038 import make_relabs_problem
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

SIMPLE = Path(__file__).parent.parent.parent / "data" / "simple"


def _simple(blm: float):
    p = load_project(SIMPLE)
    return p.copy_with(parameters={**p.parameters, "BLM": blm})


def _solve_mps(path: Path) -> float:
    _, model = pulp.LpProblem.fromMPS(str(path))
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return float(pulp.value(model.objective))


def _solve_lp_with_cbc(path: Path) -> float:
    out = subprocess.run(
        [pulp.PULP_CBC_CMD().path, str(path), "solve"], capture_output=True, text=True,
        check=True,
    ).stdout
    match = re.search(r"Objective value:\s+([-\d.eE+]+)", out)
    assert match, out
    return float(match.group(1))


@pytest.mark.parametrize("suffix", [".mps", ".lp"])
def test_simple_round_trip_blm0(tmp_path: Path, suffix: str) -> None:
    path = write_problem(_simple(0.0), tmp_path / f"m{suffix}")
    value = _solve_mps(path) if suffix == ".mps" else _solve_lp_with_cbc(path)
    assert value == pytest.approx(35.0)


def test_simple_round_trip_blm1_matches_in_process(tmp_path: Path) -> None:
    p = _simple(1.0)
    sol = MIPSolver().solve(p, SolverConfig(num_solutions=1))[0]
    path = write_problem(p, tmp_path / "m.mps")
    assert _solve_mps(path) == pytest.approx(sol.metadata["mip_objective_value"])
    assert _solve_mps(path) == pytest.approx(43.0)


def test_zone_round_trip_and_bound_survives(tmp_path: Path) -> None:
    import pandas as pd
    # PU1 forced into zone 2 moves the optimum from 7 (1,2,2) to 8; x_1_1 is fixed at 0.
    p = make_anchor_problem().copy_with(pu_zones=pd.DataFrame({"pu": [1], "zone": [2]}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # single-zone puzone ZoneLockWarning
        path = write_problem(p, tmp_path / "z.mps")
    assert _solve_mps(path) == pytest.approx(8.0)
    _, model = pulp.LpProblem.fromMPS(str(path))
    assert {v.name: v for v in model.variables()}["x_1_1"].upBound == 0


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".lp"):
        write_problem(_simple(0.0), tmp_path / "m.txt")
    with pytest.raises(TypeError, match="build_model"):
        write_problem(_simple(0.0), tmp_path / "m.lp", solver=HeuristicSolver())
    with pytest.raises(TypeError, match="ZonalProblem"):
        write_problem(make_anchor_problem(), tmp_path / "m.lp", solver=MIPSolver())
    with pytest.raises(TypeError, match="ZonalProblem"):
        write_problem(_simple(0.0), tmp_path / "m.lp", solver=ZoneMIPSolver())


def test_default_solver_picks_by_problem_type(tmp_path: Path) -> None:
    zpath = write_problem(make_anchor_problem(), tmp_path / "z.mps")
    assert _solve_mps(zpath) == pytest.approx(7.0)
    cpath = write_problem(_simple(0.0), tmp_path / "c.mps")
    assert _solve_mps(cpath) == pytest.approx(35.0)
    assert write_problem(make_anchor_problem(), tmp_path / "z.lp").exists()
    assert write_problem(_simple(0.0), tmp_path / "c.lp").exists()


def test_write_problem_exports_the_given_solver_objective(tmp_path: Path) -> None:
    p = make_relabs_problem()  # COSTBUDGET 1: no target is meetable, so max_features = 0
    path = write_problem(p, tmp_path / "w.mps", solver=MIPSolver(objective="max_features"))
    assert _solve_mps(path) == pytest.approx(0.0)
    _, model = pulp.LpProblem.fromMPS(str(path))
    assert "cost_budget" in model.constraints


def test_config_no_good_cut_is_exported(tmp_path: Path) -> None:
    forbid = [0, 1, 0, 1, 0, 1]  # the 35.0 optimum {2, 4, 6}
    cfg = SolverConfig(num_solutions=1, metadata={"forbidden_selections": [forbid]})
    path = write_problem(_simple(0.0), tmp_path / "c.mps", config=cfg)
    _, model = pulp.LpProblem.fromMPS(str(path))
    assert "nogood_cut_0" in model.constraints
    assert _solve_mps(path) == pytest.approx(38.0)  # next best: {4, 5, 6}
    plain = write_problem(_simple(0.0), tmp_path / "d.mps")
    assert "nogood_cut_0" not in pulp.LpProblem.fromMPS(str(plain))[1].constraints
