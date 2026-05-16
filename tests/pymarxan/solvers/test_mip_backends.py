"""Tests for Phase 21 MIP backend selection (CBC / HiGHS / Gurobi).

Phase 21 refactors ``MIPSolver`` / ``ZoneMIPSolver`` to dispatch through
a shared backend factory. CBC remains the default; HiGHS is opt-in via
``mip_backend="highs"`` or auto-selected when available; Gurobi requires
``pip install pymarxan[gurobi]``.

The factory uses PuLP's ``solver.available()`` to gate selection. Tests
exercise the resolution logic without requiring the binary to actually
run a solve (since CI machines may not have HiGHS on PATH).
"""
from __future__ import annotations

import pytest

from pymarxan.solvers.mip_solver import (
    _available_backends,
    _make_pulp_solver,
)

# --- Factory: well-known backend names ---------------------------------


def test_make_solver_cbc_default():
    """Default ``"auto"`` resolves to a usable backend (CBC at minimum)."""
    solver = _make_pulp_solver("auto", time_limit=10, gap=0.01, verbose=False)
    assert solver is not None


def test_make_solver_cbc_explicit():
    """``"cbc"`` explicit selection returns a PULP_CBC_CMD instance."""
    import pulp

    solver = _make_pulp_solver("cbc", time_limit=10, gap=0.01, verbose=False)
    assert isinstance(solver, pulp.PULP_CBC_CMD)


def test_make_solver_highs_returns_highs_class():
    """``"highs"`` returns a HiGHS_CMD instance even when the binary isn't
    installed (the error surfaces at solve-time, not factory-time, so users
    can opt in and discover the missing dep when they try to run)."""
    import pulp

    solver = _make_pulp_solver("highs", time_limit=10, gap=0.01, verbose=False)
    assert isinstance(solver, pulp.HiGHS_CMD)


def test_make_solver_gurobi_returns_gurobi_class():
    """``"gurobi"`` returns the GUROBI_CMD instance (the user must have
    installed ``gurobipy`` separately — Gurobi is gated behind an optional
    extra)."""
    import pulp

    solver = _make_pulp_solver("gurobi", time_limit=10, gap=0.01, verbose=False)
    assert isinstance(solver, pulp.GUROBI_CMD)


def test_make_solver_unknown_backend_rejected():
    """Unknown backend names raise ValueError at factory time."""
    with pytest.raises(ValueError, match="mip_backend"):
        _make_pulp_solver(
            "matlab_optim_toolbox", time_limit=10, gap=0.01, verbose=False,
        )


def test_make_solver_passes_kwargs():
    """The factory forwards time_limit / gap / verbose to the chosen backend."""
    solver = _make_pulp_solver("cbc", time_limit=120, gap=0.05, verbose=True)
    # PuLP stores these on the solver instance. Names vary across PuLP
    # versions; check at least one common attribute set was applied.
    assert getattr(solver, "timeLimit", None) == 120 or solver.options


# --- Auto-resolution logic ---------------------------------------------


def test_available_backends_always_includes_cbc():
    """CBC ships with PuLP; auto-detection must always find it."""
    available = _available_backends()
    assert "cbc" in available


def test_available_backends_returns_dict():
    """``_available_backends`` returns ``{name: bool}`` so callers can
    surface 'why-not' status to users."""
    available = _available_backends()
    assert isinstance(available, dict)
    assert all(isinstance(v, bool) for v in available.values())
    assert {"cbc", "highs", "gurobi"} <= set(available.keys())


# --- MIPSolver __init__ surface -----------------------------------------


def test_mip_solver_accepts_mip_backend_kwarg():
    """``MIPSolver(mip_backend="cbc")`` stores the backend choice."""
    from pymarxan.solvers.mip_solver import MIPSolver

    solver = MIPSolver(mip_backend="cbc")
    assert solver.mip_backend == "cbc"


def test_mip_solver_rejects_unknown_backend_at_init():
    """Unknown backend rejected at ``__init__``, not at solve time —
    fail-fast matches the existing strategy-kwarg validation pattern."""
    from pymarxan.solvers.mip_solver import MIPSolver

    with pytest.raises(ValueError, match="mip_backend"):
        MIPSolver(mip_backend="matlab")


def test_mip_solver_default_backend_is_auto():
    """Default ``mip_backend="auto"`` matches the ``"auto"`` factory rule."""
    from pymarxan.solvers.mip_solver import MIPSolver

    solver = MIPSolver()
    assert solver.mip_backend == "auto"


def test_zone_mip_solver_accepts_mip_backend_kwarg():
    """ZoneMIPSolver gets the same kwarg for API symmetry."""
    from pymarxan.zones.mip_solver import ZoneMIPSolver

    solver = ZoneMIPSolver(mip_backend="cbc")
    assert solver.mip_backend == "cbc"


# --- End-to-end solve via CBC ------------------------------------------


def test_mip_solver_solves_via_explicit_cbc():
    """Solving with explicit CBC produces a valid Solution — sanity check
    that the factory routing doesn't break the standard path."""
    import pandas as pd

    from pymarxan.models.problem import ConservationProblem
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    pu = pd.DataFrame({
        "id": [1, 2, 3], "cost": [1.0, 1.5, 1.0], "status": [0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [2.0], "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1], "pu": [1, 2, 3], "amount": [1.0, 1.0, 1.0],
    })
    p = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )

    solver = MIPSolver(mip_backend="cbc")
    sols = solver.solve(p, SolverConfig(num_solutions=1))
    assert len(sols) == 1
    assert sols[0].selected.sum() >= 2
    # Metadata should record which backend was used.
    assert sols[0].metadata.get("mip_backend") == "cbc"
