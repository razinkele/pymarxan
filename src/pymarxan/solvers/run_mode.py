"""RunModePipeline: chains heuristic, SA, and iterative improvement per RUNMODE.

RUNMODE values (matching Marxan):
- 0: SA only (default)
- 1: Heuristic only
- 2: SA then iterative improvement
- 3: Heuristic then iterative improvement
- 4: Heuristic then SA (pick best)
- 5: Heuristic then SA then iterative improvement (pick best of heur+SA, then improve)
- 6: Iterative improvement only (from all-selected)
"""
from __future__ import annotations

from typing import Any

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig

_VALID_RUNMODES = frozenset(range(7))


class RunModePipeline(Solver):
    """Pipeline solver that chains sub-solvers according to RUNMODE 0-6.

    Parameters
    ----------
    runmode : int
        Pipeline mode (0-6). Can be overridden by
        ``problem.parameters["RUNMODE"]`` at solve time.
    """

    def __init__(self, runmode: int = 0) -> None:
        runmode = int(runmode)
        if runmode not in _VALID_RUNMODES:
            msg = f"runmode must be 0-6, got {runmode}"
            raise ValueError(msg)
        self._runmode = runmode

    def name(self) -> str:
        return "run_mode_pipeline"

    def supports_zones(self) -> bool:
        return False

    def solve(
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        # Lazy imports to avoid circular imports
        from pymarxan.solvers.heuristic import HeuristicSolver
        from pymarxan.solvers.iterative_improvement import (
            IterativeImprovementSolver,
        )
        from pymarxan.solvers.simulated_annealing import (
            SimulatedAnnealingSolver,
        )

        if config is None:
            config = SolverConfig(num_solutions=1)

        # Resolve effective runmode: problem.parameters overrides constructor
        effective_runmode = int(
            problem.parameters.get("RUNMODE", self._runmode)
        )
        if effective_runmode not in _VALID_RUNMODES:
            msg = f"runmode must be 0-6, got {effective_runmode}"
            raise ValueError(msg)

        solutions: list[Solution] = []

        for run_idx in range(config.num_solutions):
            # Create per-run config with seed offset for diversity
            run_seed = (
                (config.seed + run_idx) if config.seed is not None else None
            )
            single_config = SolverConfig(
                num_solutions=1,
                seed=run_seed,
                verbose=config.verbose,
                metadata=config.metadata,
            )

            sol = self._run_pipeline(
                problem=problem,
                runmode=effective_runmode,
                config=single_config,
                heuristic_cls=HeuristicSolver,
                sa_cls=SimulatedAnnealingSolver,
                ii_cls=IterativeImprovementSolver,
            )
            solutions.append(sol)

        return solutions

    @staticmethod
    def _run_pipeline(
        problem: ConservationProblem,
        runmode: int,
        config: SolverConfig,
        heuristic_cls: type[Solver],
        sa_cls: type[Solver],
        ii_cls: Any,
    ) -> Solution:
        """Execute the appropriate pipeline for the given RUNMODE."""

        if runmode == 0:
            # SA only
            return sa_cls().solve(problem, config)[0]

        if runmode == 1:
            # Heuristic only
            return heuristic_cls().solve(problem, config)[0]

        if runmode == 2:
            # SA then iterative improvement
            sa_sol = sa_cls().solve(problem, config)[0]
            improved: Solution = ii_cls().improve(problem, sa_sol)
            return improved

        if runmode == 3:
            # Heuristic then iterative improvement
            heur_sol = heuristic_cls().solve(problem, config)[0]
            improved = ii_cls().improve(problem, heur_sol)
            return improved

        if runmode == 4:
            # Heuristic then SA -- pick best (minimum objective)
            heur_sol = heuristic_cls().solve(problem, config)[0]
            sa_sol = sa_cls().solve(problem, config)[0]
            return heur_sol if heur_sol.objective <= sa_sol.objective else sa_sol

        if runmode == 5:
            # Heuristic then SA then iterative improvement
            # Pick best of heuristic and SA, then apply improvement
            heur_sol = heuristic_cls().solve(problem, config)[0]
            sa_sol = sa_cls().solve(problem, config)[0]
            best = (
                heur_sol
                if heur_sol.objective <= sa_sol.objective
                else sa_sol
            )
            improved = ii_cls().improve(problem, best)
            return improved

        # runmode == 6
        # Iterative improvement only (from all-selected)
        ii_sols: list[Solution] = ii_cls().solve(problem, config)
        return ii_sols[0]
