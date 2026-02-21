"""Tests for solver plugin registry."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.registry import SolverRegistry


class _CustomSolver(Solver):
    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        n = problem.n_planning_units
        return [
            Solution(
                selected=np.ones(n, dtype=bool),
                cost=0.0, boundary=0.0, objective=0.0,
                targets_met={},
            )
        ]

    def name(self) -> str:
        return "custom"

    def supports_zones(self) -> bool:
        return False


def test_register_and_get():
    reg = SolverRegistry()
    reg.register("custom", _CustomSolver)
    solver = reg.create("custom")
    assert solver.name() == "custom"


def test_list_registered():
    reg = SolverRegistry()
    reg.register("custom", _CustomSolver)
    assert "custom" in reg.list_solvers()


def test_get_unknown_raises():
    reg = SolverRegistry()
    with pytest.raises(KeyError):
        reg.create("nonexistent")


def test_register_duplicate_raises():
    reg = SolverRegistry()
    reg.register("custom", _CustomSolver)
    with pytest.raises(ValueError):
        reg.register("custom", _CustomSolver)


def test_register_override():
    reg = SolverRegistry()
    reg.register("custom", _CustomSolver)
    reg.register("custom", _CustomSolver, override=True)
    assert "custom" in reg.list_solvers()


def test_default_registry_has_builtins():
    """The default registry includes built-in solvers."""
    from pymarxan.solvers.registry import get_default_registry

    reg = get_default_registry()
    names = reg.list_solvers()
    assert "mip" in names
    assert "sa" in names
    assert "zone_sa" in names


def test_available_solvers():
    """available_solvers filters by solver.available()."""
    reg = SolverRegistry()
    reg.register("custom", _CustomSolver)
    available = reg.available_solvers()
    assert "custom" in available
