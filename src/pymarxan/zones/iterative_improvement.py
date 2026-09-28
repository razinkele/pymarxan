"""Iterative improvement solver for multi-zone conservation planning."""

from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.objective import build_zone_solution, compute_zone_objective


class ZoneIISolver(Solver):
    """Zone-aware iterative improvement solver.

    Supports four ITIMPTYPE modes:
      0: No improvement (return input unchanged)
      1: Removal pass (try reassigning each PU to zone 0)
      2: Two-step (removal → addition → repeat until no improvement)
      3: Swap (for each PU, try all alternative zone assignments)

    Both target tiers are part of the objective; mode 0 returns the start assignment
    unchanged (anchor tests set ITIMPTYPE 3).
    """

    def name(self) -> str:
        return "Zone II (Python)"

    def supports_zones(self) -> bool:
        return True

    def supports_separation(self) -> bool:
        # Per-zone SEPDISTANCE / SEPNUM deferred to v0.3 (round-3 H1).
        return False

    def solve(
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        if not isinstance(problem, ZonalProblem):
            raise TypeError("ZoneIISolver requires a ZonalProblem")
        from pymarxan.solvers.separation import raise_if_separation_active
        raise_if_separation_active(problem, "ZoneIISolver")
        if config is None:
            config = SolverConfig()

        blm = float(problem.parameters.get("BLM", 0.0))
        itimptype = int(problem.parameters.get("ITIMPTYPE", 0))
        pu_ids = problem.planning_units["id"].tolist()
        n_pu = len(pu_ids)
        pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
        zone_ids_list = sorted(problem.zone_ids)

        locked = self._parse_locked(problem, pu_id_to_idx, zone_ids_list)
        swappable = [i for i in range(n_pu) if i not in locked]

        solutions: list[Solution] = []
        for run_idx in range(config.num_solutions):
            # Start from all-in-first-zone for non-locked PUs
            assignment = np.zeros(n_pu, dtype=int)
            for idx, zid in locked.items():
                assignment[idx] = zid
            for idx in swappable:
                assignment[idx] = zone_ids_list[0]

            assignment = self._improve(
                problem, assignment, swappable, zone_ids_list, blm, itimptype,
            )
            solutions.append(
                build_zone_solution(
                    problem, assignment, blm, solver_name=self.name(), run=run_idx + 1,
                ),
            )
        return solutions

    def improve(
        self,
        problem: ZonalProblem,
        solution: Solution,
    ) -> Solution:
        """Improve an existing solution (for RUNMODE pipeline)."""
        blm = float(problem.parameters.get("BLM", 0.0))
        itimptype = int(problem.parameters.get("ITIMPTYPE", 0))
        pu_ids = problem.planning_units["id"].tolist()
        n_pu = len(pu_ids)
        pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
        zone_ids_list = sorted(problem.zone_ids)

        locked = self._parse_locked(problem, pu_id_to_idx, zone_ids_list)
        swappable = [i for i in range(n_pu) if i not in locked]

        assignment = (
            solution.zone_assignment.copy()
            if solution.zone_assignment is not None
            else np.zeros(n_pu, dtype=int)
        )
        assignment = self._improve(
            problem, assignment, swappable, zone_ids_list, blm, itimptype,
        )
        return build_zone_solution(problem, assignment, blm, solver_name=self.name(), run=1)

    def _improve(
        self,
        problem: ZonalProblem,
        assignment: np.ndarray,
        swappable: list[int],
        zone_ids: list[int],
        blm: float,
        itimptype: int,
    ) -> np.ndarray:
        if itimptype == 0:
            return assignment
        if itimptype == 1:
            return self._removal_pass(problem, assignment, swappable, blm)
        if itimptype == 2:
            return self._two_step(problem, assignment, swappable, zone_ids, blm)
        if itimptype == 3:
            return self._swap_pass(problem, assignment, swappable, zone_ids, blm)
        return assignment

    def _removal_pass(
        self,
        problem: ZonalProblem,
        assignment: np.ndarray,
        swappable: list[int],
        blm: float,
    ) -> np.ndarray:
        """Try reassigning each PU to zone 0 (unassigned)."""
        current_obj = compute_zone_objective(problem, assignment, blm)
        for idx in swappable:
            old_zone = int(assignment[idx])
            if old_zone == 0:
                continue
            assignment[idx] = 0
            new_obj = compute_zone_objective(problem, assignment, blm)
            if new_obj < current_obj:
                current_obj = new_obj
            else:
                assignment[idx] = old_zone
        return assignment

    def _addition_pass(
        self,
        problem: ZonalProblem,
        assignment: np.ndarray,
        swappable: list[int],
        zone_ids: list[int],
        blm: float,
    ) -> np.ndarray:
        """Try assigning unassigned PUs to their best zone."""
        current_obj = compute_zone_objective(problem, assignment, blm)
        for idx in swappable:
            if int(assignment[idx]) != 0:
                continue
            best_delta = 0.0
            best_zone = 0
            for zid in zone_ids:
                assignment[idx] = zid
                new_obj = compute_zone_objective(problem, assignment, blm)
                delta = new_obj - current_obj
                if delta < best_delta:
                    best_delta = delta
                    best_zone = zid
                assignment[idx] = 0
            if best_zone > 0:
                assignment[idx] = best_zone
                current_obj += best_delta
        return assignment

    def _two_step(
        self,
        problem: ZonalProblem,
        assignment: np.ndarray,
        swappable: list[int],
        zone_ids: list[int],
        blm: float,
    ) -> np.ndarray:
        """Removal → addition → repeat until no improvement."""
        max_rounds = 100
        for _ in range(max_rounds):
            old_obj = compute_zone_objective(problem, assignment, blm)
            assignment = self._removal_pass(problem, assignment, swappable, blm)
            assignment = self._addition_pass(
                problem, assignment, swappable, zone_ids, blm,
            )
            new_obj = compute_zone_objective(problem, assignment, blm)
            if new_obj >= old_obj:
                break
        return assignment

    def _swap_pass(
        self,
        problem: ZonalProblem,
        assignment: np.ndarray,
        swappable: list[int],
        zone_ids: list[int],
        blm: float,
    ) -> np.ndarray:
        """For each PU, try all alternative zone assignments."""
        all_options = [0, *zone_ids]
        current_obj = compute_zone_objective(problem, assignment, blm)
        improved = True
        while improved:
            improved = False
            for idx in swappable:
                old_zone = int(assignment[idx])
                best_delta = 0.0
                best_zone = old_zone
                for zid in all_options:
                    if zid == old_zone:
                        continue
                    assignment[idx] = zid
                    new_obj = compute_zone_objective(problem, assignment, blm)
                    delta = new_obj - current_obj
                    if delta < best_delta:
                        best_delta = delta
                        best_zone = zid
                    assignment[idx] = old_zone
                if best_zone != old_zone:
                    assignment[idx] = best_zone
                    current_obj += best_delta
                    improved = True
        return assignment

    @staticmethod
    def _parse_locked(
        problem: ZonalProblem,
        pu_id_to_idx: dict[int, int],
        zone_ids_list: list[int],
    ) -> dict[int, int]:
        locked: dict[int, int] = {}
        if "status" in problem.planning_units.columns:
            pu_ids = problem.planning_units["id"].values
            statuses = problem.planning_units["status"].values.astype(int)
            for k in range(len(pu_ids)):
                s = int(statuses[k])
                idx = pu_id_to_idx[int(pu_ids[k])]
                if s == 2:
                    locked[idx] = zone_ids_list[0]
                elif s == 3:
                    locked[idx] = 0
        return locked
