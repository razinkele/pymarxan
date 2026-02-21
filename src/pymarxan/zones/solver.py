"""Simulated annealing solver for multi-zone conservation planning."""
from __future__ import annotations

import math

import numpy as np

from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.objective import (
    compute_zone_cost,
    compute_zone_objective,
    check_zone_targets,
    compute_standard_boundary,
    compute_zone_boundary,
)


class ZoneSASolver(Solver):
    def __init__(
        self,
        num_iterations: int = 1_000_000,
        num_temp_steps: int = 10_000,
    ):
        self._num_iterations = num_iterations
        self._num_temp_steps = num_temp_steps

    def name(self) -> str:
        return "Zone SA (Python)"

    def supports_zones(self) -> bool:
        return True

    def solve(
        self, problem: ZonalProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig()

        blm = float(problem.parameters.get("BLM", 0.0))
        num_iterations = int(
            problem.parameters.get("NUMITNS", self._num_iterations)
        )
        num_temp_steps = int(
            problem.parameters.get("NUMTEMP", self._num_temp_steps)
        )

        pu_ids = problem.planning_units["id"].tolist()
        n_pu = len(pu_ids)
        zone_ids_list = sorted(problem.zone_ids)
        zone_options = [0] + zone_ids_list

        locked: dict[int, int] = {}
        if "status" in problem.planning_units.columns:
            for _, row in problem.planning_units.iterrows():
                s = int(row["status"])
                idx = pu_ids.index(int(row["id"]))
                if s == 2:
                    locked[idx] = zone_ids_list[0]
                elif s == 3:
                    locked[idx] = 0

        swappable = [i for i in range(n_pu) if i not in locked]

        solutions = []
        for run_idx in range(config.num_solutions):
            if config.seed is not None:
                rng = np.random.default_rng(config.seed + run_idx)
            else:
                rng = np.random.default_rng()

            assignment = np.zeros(n_pu, dtype=int)
            for idx, zid in locked.items():
                assignment[idx] = zid
            for idx in swappable:
                assignment[idx] = zone_options[rng.integers(len(zone_options))]

            current_obj = compute_zone_objective(problem, assignment, blm)

            # Estimate initial temperature
            deltas = []
            for _ in range(min(1000, num_iterations // 10)):
                idx = swappable[rng.integers(len(swappable))]
                old_zone = assignment[idx]
                new_zone = zone_options[rng.integers(len(zone_options))]
                if new_zone == old_zone:
                    continue
                assignment[idx] = new_zone
                new_obj = compute_zone_objective(problem, assignment, blm)
                delta = new_obj - current_obj
                if delta > 0:
                    deltas.append(delta)
                assignment[idx] = old_zone

            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                initial_temp = -avg_delta / math.log(0.5)
            else:
                initial_temp = 1.0

            iters_per_step = max(1, num_iterations // num_temp_steps)
            if initial_temp > 0:
                alpha = (0.001 / initial_temp) ** (
                    1.0 / max(1, num_temp_steps)
                )
            else:
                alpha = 0.99

            temp = initial_temp
            best_assignment = assignment.copy()
            best_obj = current_obj
            step_count = 0

            for _ in range(num_iterations):
                idx = swappable[rng.integers(len(swappable))]
                old_zone = assignment[idx]
                new_zone = zone_options[rng.integers(len(zone_options))]
                if new_zone == old_zone:
                    continue

                assignment[idx] = new_zone
                new_obj = compute_zone_objective(problem, assignment, blm)
                delta = new_obj - current_obj

                if delta <= 0:
                    current_obj = new_obj
                elif temp > 0 and rng.random() < math.exp(-delta / temp):
                    current_obj = new_obj
                else:
                    assignment[idx] = old_zone

                if current_obj < best_obj:
                    best_assignment = assignment.copy()
                    best_obj = current_obj

                step_count += 1
                if step_count >= iters_per_step:
                    temp *= alpha
                    step_count = 0

            selected = best_assignment > 0
            cost = compute_zone_cost(problem, best_assignment)
            std_boundary = compute_standard_boundary(problem, best_assignment)
            zone_boundary = compute_zone_boundary(problem, best_assignment)
            zone_targets = check_zone_targets(problem, best_assignment)

            sol = Solution(
                selected=selected,
                cost=cost,
                boundary=std_boundary,
                objective=best_obj,
                targets_met={},
                zone_assignment=best_assignment.copy(),
                metadata={
                    "solver": self.name(),
                    "run": run_idx + 1,
                    "zone_boundary_cost": round(zone_boundary, 4),
                    "zone_targets_met": {
                        f"z{z}_f{f}": v
                        for (z, f), v in zone_targets.items()
                    },
                },
            )
            solutions.append(sol)

        return solutions
