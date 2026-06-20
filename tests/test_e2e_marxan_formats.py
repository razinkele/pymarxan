"""End-to-end load→solve over real-world Marxan file-format variations.

These fixtures (``tests/data/formats/*``) are **synthetic** — small,
hand-generated projects — but they reproduce the format quirks seen in real
public Marxan datasets (prioritizr, MarxanConnect, AdrienBrunel) that the
loader has to tolerate, without bundling any license-encumbered third-party
data. See ``tests/data/formats/README.md`` for the format provenance and
``tests/data/formats/_generate.py`` for the deterministic generator.

Format coverage:
- ``double_tab`` — columns separated by repeated tabs (``id\\t\\tcost``) with
  the boundary column named ``bound`` (MarOpt / AdrienBrunel style).
- ``quoted_csv`` — quoted ``puvspr`` header (``"species","pu","amount"``)
  with comma-delimited rows (MarxanConnect style).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver

FORMATS_DIR = Path(__file__).parent / "data" / "formats"
FIXTURES = ["double_tab", "quoted_csv"]


@pytest.mark.parametrize("name", FIXTURES)
def test_load_real_world_format(name):
    """Each format variant loads into a valid 25-PU / 4-feature problem."""
    p = load_project(FORMATS_DIR / name)
    assert p.validate() == []
    assert len(p.planning_units) == 25
    assert len(p.features) == 4
    assert len(p.pu_vs_features) > 0
    assert p.boundary is not None and len(p.boundary) > 0
    # Real-world columns survive the parse (no spurious "Unnamed" columns).
    for df in (p.planning_units, p.features, p.pu_vs_features, p.boundary):
        assert not any(str(c).startswith("Unnamed") for c in df.columns)
    # `prop` resolved to concrete per-feature targets.
    assert (p.features["target"] > 0).all()


@pytest.mark.parametrize("name", FIXTURES)
def test_solve_real_world_format(name):
    """Both engines find a feasible reserve on each format variant."""
    p = load_project(FORMATS_DIR / name)
    p.parameters["NUMITNS"] = 20000

    sa = SimulatedAnnealingSolver().solve(p, SolverConfig(num_solutions=1, seed=0))[0]
    assert sa.all_targets_met, f"SA missed targets on {name}"

    mip = MIPSolver().solve(p, SolverConfig(num_solutions=1))[0]
    assert mip.all_targets_met, f"MIP missed targets on {name}"
    # Exact optimum is no costlier than the heuristic.
    assert mip.cost <= sa.cost + 1e-6
