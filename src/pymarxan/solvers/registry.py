"""Solver plugin registry for conservation planning.

Allows registration and discovery of solver implementations by name.
"""
from __future__ import annotations

from pymarxan.solvers.base import Solver


class SolverRegistry:
    """Registry for solver plugins."""

    def __init__(self) -> None:
        self._solvers: dict[str, type[Solver]] = {}

    def register(
        self,
        name: str,
        solver_class: type[Solver],
        override: bool = False,
    ) -> None:
        """Register a solver class under a given name."""
        if name in self._solvers and not override:
            raise ValueError(f"Solver '{name}' is already registered")
        self._solvers[name] = solver_class

    def create(self, name: str) -> Solver:
        """Create a solver instance by name."""
        if name not in self._solvers:
            raise KeyError(f"Unknown solver: '{name}'")
        return self._solvers[name]()

    def list_solvers(self) -> list[str]:
        """Return names of all registered solvers."""
        return sorted(self._solvers.keys())

    def available_solvers(self) -> list[str]:
        """Return names of solvers that are currently available."""
        import logging

        logger = logging.getLogger(__name__)
        result = []
        for name, cls in sorted(self._solvers.items()):
            try:
                instance = cls()
                if instance.available():
                    result.append(name)
            except (ImportError, FileNotFoundError, OSError):
                pass  # Expected: optional dependency missing
            except Exception as exc:
                logger.warning("Solver '%s' failed availability check: %s", name, exc)
        return result


def get_default_registry() -> SolverRegistry:
    """Return a registry pre-loaded with built-in solvers."""
    from pymarxan.solvers.heuristic import HeuristicSolver
    from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
    from pymarxan.solvers.marxan_binary import MarxanBinarySolver
    from pymarxan.solvers.mip_solver import MIPSolver
    from pymarxan.solvers.run_mode import RunModePipeline
    from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
    from pymarxan.zones.solver import ZoneSASolver

    reg = SolverRegistry()
    reg.register("greedy", HeuristicSolver)
    reg.register("iterative_improvement", IterativeImprovementSolver)
    reg.register("mip", MIPSolver)
    reg.register("pipeline", RunModePipeline)
    reg.register("sa", SimulatedAnnealingSolver)
    reg.register("binary", MarxanBinarySolver)
    reg.register("zone_sa", ZoneSASolver)
    return reg
