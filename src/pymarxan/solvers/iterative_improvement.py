"""Iterative improvement solver implementing ITIMPTYPE modes 0-3.

Post-processing step that refines an existing solution by systematically
trying removals, additions, or pairwise swaps of planning units.

ITIMPTYPE modes
---------------
- 0: No improvement (return input unchanged)
- 1: Removal pass -- try removing each selected PU, accept if objective
     decreases. Repeat until no improvement found in a full pass.
- 2: Two-step -- removal pass then addition pass (try adding each
     unselected PU, accept if objective decreases).
- 3: Swap -- try pairwise swaps (remove one selected, add one unselected),
     accept if objective decreases.
"""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.utils import build_solution, compute_objective

_VALID_ITIMPTYPES = frozenset(range(4))


class IterativeImprovementSolver(Solver):
    """Iterative improvement solver for conservation planning.

    Parameters
    ----------
    itimptype : int
        Improvement mode (0-3). Can be overridden by
        ``problem.parameters["ITIMPTYPE"]``.
    """

    def __init__(self, itimptype: int = 0) -> None:
        if itimptype not in _VALID_ITIMPTYPES:
            raise ValueError(
                f"Invalid itimptype {itimptype!r}; "
                f"must be one of {sorted(_VALID_ITIMPTYPES)}"
            )
        self._itimptype = itimptype

    # ------------------------------------------------------------------
    # Solver ABC
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "iterative_improvement"

    def supports_zones(self) -> bool:
        return False

    def solve(
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        """Solve by starting from all-selected, then improving."""
        if config is None:
            config = SolverConfig(num_solutions=1)

        blm = float(problem.parameters.get("BLM", 0.0))
        n = problem.n_planning_units
        statuses = problem.planning_units["status"].values.astype(int)

        solutions: list[Solution] = []
        for _ in range(config.num_solutions):
            # Start with all PUs selected, respecting locked-out
            selected = np.ones(n, dtype=bool)
            locked_out = statuses == 3
            selected[locked_out] = False

            initial = build_solution(
                problem, selected, blm, metadata={"solver": self.name()}
            )
            improved = self.improve(problem, initial)
            solutions.append(improved)

        return solutions

    # ------------------------------------------------------------------
    # Public improvement API
    # ------------------------------------------------------------------

    def improve(
        self,
        problem: ConservationProblem,
        solution: Solution,
    ) -> Solution:
        """Improve an existing solution using the configured ITIMPTYPE.

        Parameters
        ----------
        problem : ConservationProblem
            The conservation planning problem.
        solution : Solution
            The starting solution to improve.

        Returns
        -------
        Solution
            The improved solution (or the original if ITIMPTYPE=0).
        """
        itimptype = int(
            problem.parameters.get("ITIMPTYPE", self._itimptype)
        )

        if itimptype == 0:
            return solution
        if itimptype == 1:
            return self._removal_pass_loop(problem, solution)
        if itimptype == 2:
            return self._two_step(problem, solution)
        if itimptype == 3:
            return self._swap_pass_loop(problem, solution)

        raise ValueError(  # pragma: no cover
            f"Invalid itimptype {itimptype!r}"
        )

    # ------------------------------------------------------------------
    # Locked PU helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _locked_sets(
        problem: ConservationProblem,
    ) -> tuple[set[int], set[int], set[int]]:
        """Return (locked_in_indices, locked_out_indices, initial_include_indices)."""
        locked_in: set[int] = set()
        locked_out: set[int] = set()
        initial_include: set[int] = set()
        statuses = problem.planning_units["status"].values.astype(int)
        for i, s in enumerate(statuses):
            if s == 2:
                locked_in.add(i)
            elif s == 3:
                locked_out.add(i)
            elif s == 1:
                initial_include.add(i)
        return locked_in, locked_out, initial_include

    # ------------------------------------------------------------------
    # ITIMPTYPE 1 -- removal pass (repeat until stable)
    # ------------------------------------------------------------------

    def _removal_pass_loop(
        self,
        problem: ConservationProblem,
        solution: Solution,
    ) -> Solution:
        """Repeatedly run removal passes until no improvement found."""
        current = solution
        while True:
            improved = self._removal_pass(problem, current)
            if improved.objective >= current.objective:
                break
            current = improved
        return current

    def _removal_pass(
        self,
        problem: ConservationProblem,
        solution: Solution,
    ) -> Solution:
        """Single removal pass: try removing each selected PU."""
        blm = float(problem.parameters.get("BLM", 0.0))
        pu_ids = problem.planning_units["id"].tolist()
        pu_index = {pid: i for i, pid in enumerate(pu_ids)}
        locked_in, _, _ = self._locked_sets(problem)

        selected = solution.selected.copy()
        current_obj = compute_objective(problem, selected, pu_index, blm)

        improved = True
        while improved:
            improved = False
            for i in range(len(selected)):
                if not selected[i]:
                    continue
                if i in locked_in:
                    continue

                # Try removing this PU
                selected[i] = False
                new_obj = compute_objective(
                    problem, selected, pu_index, blm
                )
                if new_obj < current_obj:
                    current_obj = new_obj
                    improved = True
                else:
                    selected[i] = True  # revert

        return build_solution(
            problem, selected, blm, metadata={"solver": self.name()}
        )

    # ------------------------------------------------------------------
    # ITIMPTYPE 2 -- removal then addition
    # ------------------------------------------------------------------

    def _two_step(
        self,
        problem: ConservationProblem,
        solution: Solution,
    ) -> Solution:
        """Removal pass followed by addition pass."""
        after_removal = self._removal_pass(problem, solution)
        after_addition = self._addition_pass(problem, after_removal)
        return after_addition

    def _addition_pass(
        self,
        problem: ConservationProblem,
        solution: Solution,
    ) -> Solution:
        """Try adding each unselected PU; accept if objective decreases."""
        blm = float(problem.parameters.get("BLM", 0.0))
        pu_ids = problem.planning_units["id"].tolist()
        pu_index = {pid: i for i, pid in enumerate(pu_ids)}
        _, locked_out, _ = self._locked_sets(problem)

        selected = solution.selected.copy()
        current_obj = compute_objective(problem, selected, pu_index, blm)

        improved = True
        while improved:
            improved = False
            for i in range(len(selected)):
                if selected[i]:
                    continue
                if i in locked_out:
                    continue

                # Try adding this PU
                selected[i] = True
                new_obj = compute_objective(
                    problem, selected, pu_index, blm
                )
                if new_obj < current_obj:
                    current_obj = new_obj
                    improved = True
                else:
                    selected[i] = False  # revert

        return build_solution(
            problem, selected, blm, metadata={"solver": self.name()}
        )

    # ------------------------------------------------------------------
    # ITIMPTYPE 3 -- pairwise swap
    # ------------------------------------------------------------------

    def _swap_pass_loop(
        self,
        problem: ConservationProblem,
        solution: Solution,
    ) -> Solution:
        """Try pairwise swaps until no improvement found."""
        blm = float(problem.parameters.get("BLM", 0.0))
        pu_ids = problem.planning_units["id"].tolist()
        pu_index = {pid: i for i, pid in enumerate(pu_ids)}
        locked_in, locked_out, _ = self._locked_sets(problem)

        selected = solution.selected.copy()
        current_obj = compute_objective(problem, selected, pu_index, blm)

        improved = True
        while improved:
            improved = False
            # Indices of selected (removable) and unselected (addable) PUs
            removable = [
                i for i in range(len(selected))
                if selected[i] and i not in locked_in
            ]
            addable = [
                i for i in range(len(selected))
                if not selected[i] and i not in locked_out
            ]

            for r in removable:
                for a in addable:
                    # Try swapping: remove r, add a
                    selected[r] = False
                    selected[a] = True
                    new_obj = compute_objective(
                        problem, selected, pu_index, blm
                    )
                    if new_obj < current_obj:
                        current_obj = new_obj
                        improved = True
                        break  # restart scan with new selection
                    else:
                        # Revert
                        selected[r] = True
                        selected[a] = False

                if improved:
                    break

        return build_solution(
            problem, selected, blm, metadata={"solver": self.name()}
        )
