"""Greedy heuristic solver for multi-zone conservation planning."""

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


class ZoneHeuristicSolver(Solver):
    """Zone-aware greedy heuristic solver.

    Iteratively assigns PUs to zones, selecting the (PU, zone) pair
    that most improves the objective at each step. Stops when all zone
    targets are met or no candidate improves the objective.
    """

    def name(self) -> str:
        return "Zone Heuristic (Python)"

    def supports_zones(self) -> bool:
        return True

    def solve(
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        if not isinstance(problem, ZonalProblem):
            raise TypeError("ZoneHeuristicSolver requires a ZonalProblem")
        if config is None:
            config = SolverConfig()

        blm = float(problem.parameters.get("BLM", 0.0))
        pu_ids = problem.planning_units["id"].tolist()
        n_pu = len(pu_ids)
        pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
        zone_ids_list = sorted(problem.zone_ids)

        locked, initial_include = self._parse_status(
            problem, pu_id_to_idx, zone_ids_list,
        )

        solutions: list[Solution] = []
        for run_idx in range(config.num_solutions):
            assignment = np.zeros(n_pu, dtype=int)
            for idx, zid in locked.items():
                assignment[idx] = zid

            assignment = self._greedy_assign(
                problem, assignment, locked, zone_ids_list, blm,
            )

            solutions.append(
                self._build_zone_solution(problem, assignment, blm, run_idx),
            )
        return solutions

    def _greedy_assign(
        self,
        problem: ZonalProblem,
        assignment: np.ndarray,
        locked: dict[int, int],
        zone_ids: list[int],
        blm: float,
    ) -> np.ndarray:
        """Greedily assign PUs to zones to minimize objective."""
        n_pu = len(assignment)
        swappable = [i for i in range(n_pu) if i not in locked]

        current_obj = compute_zone_objective(problem, assignment, blm)
        improved = True

        while improved:
            improved = False
            best_delta = 0.0
            best_idx = -1
            best_zone = -1

            for idx in swappable:
                old_zone = int(assignment[idx])
                for zid in zone_ids:
                    if zid == old_zone:
                        continue
                    assignment[idx] = zid
                    new_obj = compute_zone_objective(problem, assignment, blm)
                    delta = new_obj - current_obj
                    if delta < best_delta:
                        best_delta = delta
                        best_idx = idx
                        best_zone = zid
                    assignment[idx] = old_zone

            if best_idx >= 0:
                assignment[best_idx] = best_zone
                current_obj += best_delta
                improved = True

            # Check if all targets met
            targets = check_zone_targets(problem, assignment)
            if targets and all(targets.values()):
                break

        return assignment

    @staticmethod
    def _parse_status(
        problem: ZonalProblem,
        pu_id_to_idx: dict[int, int],
        zone_ids_list: list[int],
    ) -> tuple[dict[int, int], set[int]]:
        locked: dict[int, int] = {}
        initial_include: set[int] = set()
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
                elif s == 1:
                    initial_include.add(idx)
        return locked, initial_include

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
                "solver": "Zone Heuristic (Python)",
                "run": run_idx + 1,
                "zone_boundary_cost": round(zone_boundary, 4),
            },
        )
