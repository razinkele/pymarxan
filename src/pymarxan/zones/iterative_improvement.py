"""Iterative improvement solver for multi-zone conservation planning."""

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


def _seed(rule: PUZoneRule) -> int:
    return rule.start if rule.start is not None else min(rule.allowed)


class ZoneIISolver(Solver):
    """Zone-aware iterative improvement solver.

    Supports four ITIMPTYPE modes:
      0: No improvement (solve() returns the start assignment unchanged)
      1: Removal pass (try unassigning each PU that may be unassigned)
      2: Two-step (removal → addition → repeat until no improvement)
      3: Swap (for each PU, try every zone its rule allows, incl. 0 when it may be unassigned)

    Both target tiers are part of the objective; mode 0 returns the start assignment
    unchanged (improve() returns it after the rule repair; anchor tests set ITIMPTYPE 3).
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
        warn_zone_input_issues(problem, "ZoneIISolver")
        if config is None:
            config = SolverConfig()

        blm = float(problem.parameters.get("BLM", 0.0))
        itimptype = int(problem.parameters.get("ITIMPTYPE", 0))
        pu_ids = problem.planning_units["id"].tolist()
        n_pu = len(pu_ids)
        rules = problem.pu_zone_rules()
        locked = fixed_zones(rules)
        swappable = [i for i in range(n_pu) if i not in locked]

        solutions: list[Solution] = []
        for run_idx in range(config.num_solutions):
            # Seed each swappable PU at its rule's start zone, else its smallest allowed zone
            assignment = np.zeros(n_pu, dtype=int)
            for idx, zid in locked.items():
                assignment[idx] = zid
            for idx in swappable:
                assignment[idx] = _seed(rules[idx])

            assignment = self._improve(
                problem, assignment, swappable, rules, blm, itimptype,
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
        """Improve an existing solution (RUNMODE pipeline).

        First repairs it to the PU zone rules: forced / locked-out PUs reset, a zone outside
        a PU's rule (or 0 for a must-assign PU) → its smallest allowed zone."""
        warn_zone_input_issues(problem, "ZoneIISolver")
        blm = float(problem.parameters.get("BLM", 0.0))
        itimptype = int(problem.parameters.get("ITIMPTYPE", 0))
        pu_ids = problem.planning_units["id"].tolist()
        n_pu = len(pu_ids)
        rules = problem.pu_zone_rules()
        locked = fixed_zones(rules)
        swappable = [i for i in range(n_pu) if i not in locked]

        assignment = (
            solution.zone_assignment.copy()
            if solution.zone_assignment is not None
            else np.zeros(n_pu, dtype=int)
        )
        for idx, rule in enumerate(rules):
            zone = int(assignment[idx])
            if idx in locked:
                assignment[idx] = locked[idx]
            elif (zone == 0 and not rule.may_be_unassigned) or (
                zone != 0 and zone not in rule.allowed
            ):
                assignment[idx] = min(rule.allowed)
        assignment = self._improve(
            problem, assignment, swappable, rules, blm, itimptype,
        )
        return build_zone_solution(problem, assignment, blm, solver_name=self.name(), run=1)

    def _improve(
        self,
        problem: ZonalProblem,
        assignment: np.ndarray,
        swappable: list[int],
        rules: list[PUZoneRule],
        blm: float,
        itimptype: int,
    ) -> np.ndarray:
        if itimptype == 0:
            return assignment
        if itimptype == 1:
            return self._removal_pass(problem, assignment, swappable, rules, blm)
        if itimptype == 2:
            return self._two_step(problem, assignment, swappable, rules, blm)
        if itimptype == 3:
            return self._swap_pass(problem, assignment, swappable, rules, blm)
        return assignment

    def _removal_pass(
        self,
        problem: ZonalProblem,
        assignment: np.ndarray,
        swappable: list[int],
        rules: list[PUZoneRule],
        blm: float,
    ) -> np.ndarray:
        """Try unassigning each PU that may be unassigned."""
        current_obj = compute_zone_objective(problem, assignment, blm)
        for idx in swappable:
            if not rules[idx].may_be_unassigned:
                continue
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
        rules: list[PUZoneRule],
        blm: float,
    ) -> np.ndarray:
        """Give each unassigned PU its best allowed zone.
        Move a must-assign PU with several allowed zones to a better one."""
        current_obj = compute_zone_objective(problem, assignment, blm)
        for idx in swappable:
            rule = rules[idx]
            old_zone = int(assignment[idx])
            if old_zone == 0:
                candidates = sorted(rule.allowed)
            elif not rule.may_be_unassigned and len(rule.allowed) > 1:
                candidates = [z for z in sorted(rule.allowed) if z != old_zone]
            else:
                continue
            best_delta = 0.0
            best_zone = old_zone
            for zid in candidates:
                assignment[idx] = zid
                delta = compute_zone_objective(problem, assignment, blm) - current_obj
                if delta < best_delta:
                    best_delta = delta
                    best_zone = zid
                assignment[idx] = old_zone
            if best_zone != old_zone:
                assignment[idx] = best_zone
                current_obj += best_delta
        return assignment

    def _two_step(
        self,
        problem: ZonalProblem,
        assignment: np.ndarray,
        swappable: list[int],
        rules: list[PUZoneRule],
        blm: float,
    ) -> np.ndarray:
        """Removal → addition → repeat until no improvement."""
        max_rounds = 100
        for _ in range(max_rounds):
            old_obj = compute_zone_objective(problem, assignment, blm)
            assignment = self._removal_pass(problem, assignment, swappable, rules, blm)
            assignment = self._addition_pass(
                problem, assignment, swappable, rules, blm,
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
        rules: list[PUZoneRule],
        blm: float,
    ) -> np.ndarray:
        """For each PU, try every zone its rule allows (incl. 0 when it may be unassigned)."""
        current_obj = compute_zone_objective(problem, assignment, blm)
        improved = True
        while improved:
            improved = False
            for idx in swappable:
                old_zone = int(assignment[idx])
                best_delta = 0.0
                best_zone = old_zone
                rule = rules[idx]
                options = ([0] if rule.may_be_unassigned else []) + sorted(rule.allowed)
                for zid in options:
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
