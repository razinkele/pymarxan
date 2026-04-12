"""Iterative improvement solver for multi-zone conservation planning."""

from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.objective import (
    check_zone_targets,
    compute_standard_boundary,
    compute_zone_boundary,
    compute_zone_cost,
    compute_zone_objective,
    compute_zone_penalty,
    compute_zone_shortfall,
)


class ZoneIISolver(Solver):
    """Zone-aware iterative improvement solver.

    Supports four ITIMPTYPE modes:
      0: No improvement (return input unchanged)
      1: Removal pass (try reassigning each PU to zone 0)
      2: Two-step (removal → addition → repeat until no improvement)
      3: Swap (for each PU, try all alternative zone assignments)
    """

    def name(self) -> str:
        return "Zone II (Python)"

    def supports_zones(self) -> bool:
        return True

    def solve(
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        if not isinstance(problem, ZonalProblem):
            raise TypeError("ZoneIISolver requires a ZonalProblem")
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
                self._build_zone_solution(problem, assignment, blm, run_idx),
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
        return self._build_zone_solution(problem, assignment, blm, 0)

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

    @staticmethod
    def _build_zone_solution(
        problem: ZonalProblem,
        assignment: np.ndarray,
        blm: float,
        run_idx: int,
    ) -> Solution:
        selected = assignment > 0
        cost = compute_zone_cost(problem, assignment)
        std_boundary = compute_standard_boundary(problem, assignment)
        zone_boundary = compute_zone_boundary(problem, assignment)
        zone_targets = check_zone_targets(problem, assignment)
        zone_penalty = compute_zone_penalty(problem, assignment)
        zone_shortfall = compute_zone_shortfall(problem, assignment)
        obj = cost + blm * std_boundary + zone_boundary + zone_penalty
        return Solution(
            selected=selected,
            cost=cost,
            boundary=std_boundary,
            objective=obj,
            targets_met=zone_targets,
            penalty=zone_penalty,
            shortfall=zone_shortfall,
            zone_assignment=assignment.copy(),
            metadata={
                "solver": "Zone II (Python)",
                "run": run_idx + 1,
                "zone_boundary_cost": round(zone_boundary, 4),
            },
        )
