"""Greedy heuristic solver for multi-zone conservation planning."""

from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.zones.model import (
    PUZoneRule,
    ZonalProblem,
    fixed_zones,
    warn_zone_input_issues,
)
from pymarxan.zones.objective import build_zone_solution, compute_zone_objective


class ZoneHeuristicSolver(Solver):
    """Zone-aware greedy heuristic solver.

    Iteratively assigns PUs to zones, selecting the (PU, zone) pair
    that most improves the objective at each step. Stops when no candidate move
    improves the objective (both target tiers are part of the objective, so meeting
    the zone targets alone no longer ends the search).
    """

    def name(self) -> str:
        return "Zone Heuristic (Python)"

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
            raise TypeError("ZoneHeuristicSolver requires a ZonalProblem")
        from pymarxan.solvers.separation import raise_if_separation_active
        raise_if_separation_active(problem, "ZoneHeuristicSolver")
        warn_zone_input_issues(problem, "ZoneHeuristicSolver")
        if config is None:
            config = SolverConfig()

        blm = float(problem.parameters.get("BLM", 0.0))
        pu_ids = problem.planning_units["id"].tolist()
        n_pu = len(pu_ids)
        rules = problem.pu_zone_rules()
        locked = fixed_zones(rules)

        solutions: list[Solution] = []
        for run_idx in range(config.num_solutions):
            assignment = np.zeros(n_pu, dtype=int)
            for idx, zid in locked.items():
                assignment[idx] = zid
            for idx, rule in enumerate(rules):
                if idx in locked or rule.may_be_unassigned:
                    continue  # free PUs start at 0 as in v0.37 (status 1 has no effect here)
                assignment[idx] = rule.start if rule.start is not None else min(rule.allowed)

            assignment = self._greedy_assign(problem, assignment, locked, rules, blm)

            solutions.append(
                build_zone_solution(
                    problem, assignment, blm, solver_name=self.name(), run=run_idx + 1,
                ),
            )
        return solutions

    def _greedy_assign(
        self,
        problem: ZonalProblem,
        assignment: np.ndarray,
        locked: dict[int, int],
        rules: list[PUZoneRule],
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
                for zid in sorted(rules[idx].allowed):
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

        return assignment
