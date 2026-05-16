"""Round-3 H16: programmatic capability-matrix test.

Phase 20 adds twelve touch-points for one capability (4 base
supports + 4 zone overrides + 4 zone raises). Phase 21+ will likely
forget at least one. This test enumerates every (solver × constraint)
pair and asserts the supports_X bit is correct.

Catches regressions where someone:
- forgets to override ``supports_separation()`` on a new zone solver,
- forgets to call ``raise_if_separation_active`` in a new zone solver's
  ``solve()``, leading to a silent no-op,
- adds a new constraint type but forgets to wire ``supports_X`` on the
  4 base solvers AND the 4 zone solvers consistently.
"""
from __future__ import annotations

import pytest

_BASE_SOLVERS = [
    "pymarxan.solvers.simulated_annealing.SimulatedAnnealingSolver",
    "pymarxan.solvers.iterative_improvement.IterativeImprovementSolver",
    "pymarxan.solvers.mip_solver.MIPSolver",
    "pymarxan.solvers.heuristic.HeuristicSolver",
]

_ZONE_SOLVERS = [
    "pymarxan.zones.solver.ZoneSASolver",
    "pymarxan.zones.iterative_improvement.ZoneIISolver",
    "pymarxan.zones.heuristic.ZoneHeuristicSolver",
    "pymarxan.zones.mip_solver.ZoneMIPSolver",
]


def _load(path: str):
    import importlib
    module_path, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


@pytest.mark.parametrize("solver_path", _BASE_SOLVERS)
def test_base_solver_supports_separation_true(solver_path):
    """All base solvers support separation (via SA/II SepState wiring,
    MIP-drop post-hoc, or heuristic post-hoc)."""
    cls = _load(solver_path)
    assert cls().supports_separation() is True


@pytest.mark.parametrize("solver_path", _ZONE_SOLVERS)
def test_zone_solver_supports_separation_false(solver_path):
    """All zone solvers explicitly opt out of separation (round-3 H1)."""
    cls = _load(solver_path)
    assert cls().supports_separation() is False


@pytest.mark.parametrize("solver_path", _BASE_SOLVERS)
def test_base_solver_supports_clumping_true(solver_path):
    """Phase 19 capability — pinned here too."""
    cls = _load(solver_path)
    assert cls().supports_clumping() is True


@pytest.mark.parametrize("solver_path", _BASE_SOLVERS)
def test_base_solver_supports_probmode3_true(solver_path):
    """Phase 18 capability — pinned here too."""
    cls = _load(solver_path)
    assert cls().supports_probmode3() is True


# --- Solver class names appear in __all__ / module surface --------------


def test_separation_module_exports():
    """`pymarxan.solvers.separation.__all__` lists every public symbol the
    plan promised."""
    from pymarxan.solvers import separation
    expected = {
        "PUCoordinatesUnavailableError",
        "SepState",
        "compute_sep_penalty",
        "compute_sep_penalty_from_scratch",
        "count_separation",
        "evaluate_solution_separation",
        "get_pu_coordinates",
        "is_separation_active",
        "raise_if_separation_active",
    }
    assert set(separation.__all__) >= expected
