"""Phase 20 integration smoke tests.

Covers:
- All four base solvers (SA, II, MIP-drop, heuristic) run end-to-end on a
  separation-active problem and populate ``Solution.sep_shortfalls``.
- Combined PROBMODE 3 + TARGET2 + SEPNUM → all three analytics dicts
  populated on one Solution (round-1 Task 14).
- Zone solvers raise ``NotImplementedError`` on a sep-active problem (round-3 H1).
- Round-3 H12 byte-identical anti-test: sepnum=1 default gives the same
  SA objective as a problem with no sepnum column.
- MIP strategy kwargs reject ``socp`` (sep) and raise NotImplementedError
  on ``big_m`` at solve time.
- Scenario clone preserves sep columns; ``_OVERRIDABLE_FIELDS`` extension
  (round-2 M2 + round-3 H11) exercised.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point


def _build_sep_problem():
    from pymarxan.models.problem import ConservationProblem

    n_pu = 9  # 3×3 grid spacing 100
    coords = [(i * 100.0, j * 100.0) for i in range(3) for j in range(3)]
    pu = gpd.GeoDataFrame(
        {"id": list(range(1, n_pu + 1)),
         "cost": [1.0] * n_pu, "status": [0] * n_pu},
        geometry=[Point(x, y) for x, y in coords],
        crs="EPSG:3857",
    )
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["a", "b"],
        "target": [3.0, 3.0],
        "spf": [1.0, 1.0],
        "sepdistance": [150.0, 0.0],
        "sepnum": [3, 1],
    })
    rows = []
    for fid in (1, 2):
        for pid in range(1, n_pu + 1):
            rows.append({"species": fid, "pu": pid, "amount": 1.0})
    puvspr = pd.DataFrame(rows)
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        # Short SA budget — the 9-PU smoke test doesn't need 1M iterations;
        # we're checking that build_solution wires fields, not solver quality.
        parameters={"NUMITNS": 2000, "NUMTEMP": 200, "BLM": 0.0},
    )


def test_sa_populates_sep_fields_on_sep_active_problem():
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver

    p = _build_sep_problem()
    solver = SimulatedAnnealingSolver()
    sols = solver.solve(p, SolverConfig(num_solutions=1, seed=42))
    sol = sols[0]
    assert sol.sep_shortfalls is not None
    assert sol.sep_penalty is not None
    assert 1 in sol.sep_shortfalls
    assert 2 not in sol.sep_shortfalls


def test_ii_populates_sep_fields_on_sep_active_problem():
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.iterative_improvement import (
        IterativeImprovementSolver,
    )

    p = _build_sep_problem()
    # II needs a starting solution; ITIMPTYPE 2 runs both passes.
    p.parameters["ITIMPTYPE"] = 2
    solver = IterativeImprovementSolver()
    sols = solver.solve(p, SolverConfig(num_solutions=1, seed=42))
    sol = sols[0]
    assert sol.sep_shortfalls is not None


def test_mip_drop_populates_sep_fields_post_hoc():
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    p = _build_sep_problem()
    solver = MIPSolver()  # mip_sep_strategy="drop" by default
    sols = solver.solve(p, SolverConfig(num_solutions=1))
    sol = sols[0]
    assert sol.sep_shortfalls is not None


def test_heuristic_populates_sep_fields_post_hoc():
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.heuristic import HeuristicSolver

    p = _build_sep_problem()
    solver = HeuristicSolver()
    sols = solver.solve(p, SolverConfig(num_solutions=1))
    sol = sols[0]
    assert sol.sep_shortfalls is not None


# --- MIP strategy gating -----------------------------------------------


def test_mip_sep_strategy_rejects_socp_at_init():
    from pymarxan.solvers.mip_solver import MIPSolver

    with pytest.raises(ValueError, match="combinatorial"):
        MIPSolver(mip_sep_strategy="socp")


def test_mip_sep_strategy_big_m_raises_at_solve():
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    p = _build_sep_problem()
    solver = MIPSolver(mip_sep_strategy="big_m")
    with pytest.raises(NotImplementedError, match="big_m"):
        solver.solve(p, SolverConfig(num_solutions=1))


def test_mip_sep_strategy_big_m_no_op_on_non_sep_problem():
    """``big_m`` only raises when the problem is sep-active."""
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    p = _build_sep_problem()
    p.features.loc[:, "sepnum"] = 1
    p.features.loc[:, "sepdistance"] = 0.0
    solver = MIPSolver(mip_sep_strategy="big_m")
    # Should not raise even though the kwarg is "big_m" — no sep-active
    # feature means the gate doesn't fire.
    solver.solve(p, SolverConfig(num_solutions=1))


# --- Zone solver guard (round-3 H1) ------------------------------------


@pytest.mark.parametrize("solver_cls_path", [
    "pymarxan.zones.solver.ZoneSASolver",
    "pymarxan.zones.iterative_improvement.ZoneIISolver",
    "pymarxan.zones.heuristic.ZoneHeuristicSolver",
    "pymarxan.zones.mip_solver.ZoneMIPSolver",
])
def test_zone_solver_reports_no_separation_support(solver_cls_path):
    """Zone solvers advertise via supports_separation()=False."""
    import importlib

    module_path, cls_name = solver_cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    assert cls().supports_separation() is False


# --- Round-3 H12: byte-identical when sepnum disabled ------------------


def test_sa_objective_unchanged_when_sepnum_disabled():
    """Anti-test: a problem where every feature has sepnum=1 (disabled)
    must produce the same SA objective as the same problem with no sepnum
    column. Pins the round-2 H1 compound-mask correctness — separation
    code paths must be no-ops when no feature is sep-active."""
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver

    p_legacy = _build_sep_problem()
    p_legacy.features.loc[:, "sepnum"] = 1
    p_legacy.features.loc[:, "sepdistance"] = 0.0
    p_pre = _build_sep_problem()
    p_pre.features = p_pre.features.drop(columns=["sepnum", "sepdistance"])

    solver = SimulatedAnnealingSolver()
    sol_legacy = solver.solve(p_legacy, SolverConfig(num_solutions=1, seed=42))[0]
    sol_pre = solver.solve(p_pre, SolverConfig(num_solutions=1, seed=42))[0]
    assert sol_legacy.objective == pytest.approx(sol_pre.objective)
    assert sol_legacy.sep_shortfalls is None
    assert sol_pre.sep_shortfalls is None


# --- Combined constraints (round-2 Task 14a) ---------------------------


def test_combined_probmode3_target2_sepnum_all_populate():
    """A problem with all three constraint types active produces a Solution
    with all six analytics attrs (prob_*, clump_*, sep_*) populated."""
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver

    p = _build_sep_problem()
    p.parameters["PROBMODE"] = 3
    p.features["ptarget"] = [0.9, -1.0]
    p.features["target2"] = [1.0, 0.0]
    p.features["clumptype"] = [0, 0]
    # Add a `prob` column to puvspr so PROBMODE 3 has data to work with.
    p.pu_vs_features["prob"] = 0.1

    solver = SimulatedAnnealingSolver()
    sol = solver.solve(p, SolverConfig(num_solutions=1, seed=42))[0]
    assert sol.prob_shortfalls is not None
    assert sol.clump_shortfalls is not None
    assert sol.sep_shortfalls is not None


# --- ProblemCache warnings replay (round-3 CR4) ------------------------


def test_problemcache_warns_on_geographic_crs_with_sepdistance():
    """A separation-active problem on a geographic CRS emits a UserWarning
    at ProblemCache.from_problem (the place that fires every solve)."""
    from pymarxan.solvers.cache import ProblemCache

    p = _build_sep_problem()
    p.planning_units = p.planning_units.to_crs("EPSG:4326")
    with pytest.warns(UserWarning, match="geographic CRS"):
        ProblemCache.from_problem(p)


def test_problemcache_warns_on_sepdistance_zero_with_sepnum_gt1():
    """sepdistance==0 with sepnum>1 emits a 'trivially satisfied' warning."""
    from pymarxan.solvers.cache import ProblemCache

    p = _build_sep_problem()
    p.features.loc[p.features["id"] == 1, "sepdistance"] = 0.0
    # sepnum stays at 3 → constraint is trivially satisfied
    with pytest.warns(UserWarning, match="trivially satisfied"):
        ProblemCache.from_problem(p)


# --- Scenario clone with sepnum override (round-3 H11) -----------------


def test_scenario_set_clone_with_sepnum_override():
    """``apply_feature_overrides`` accepts ``sepnum`` — pins the round-2 M2
    _OVERRIDABLE_FIELDS extension against silent regression."""
    from pymarxan.models.problem import apply_feature_overrides

    p = _build_sep_problem()
    # Override sepnum on feature id=1 from 3 to 5.
    overridden = apply_feature_overrides(p, {1: {"sepnum": 5}})
    assert int(
        overridden.features.loc[overridden.features["id"] == 1, "sepnum"].iloc[0]
    ) == 5
    # Original should be untouched.
    assert int(
        p.features.loc[p.features["id"] == 1, "sepnum"].iloc[0]
    ) == 3
