"""Simulated annealing solver for multi-zone conservation planning."""
from __future__ import annotations

import math

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.objective import (
    check_zone_targets,
    compute_standard_boundary,
    compute_zone_boundary,
    compute_zone_cost,
    compute_zone_penalty,
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
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        if not isinstance(problem, ZonalProblem):
            raise TypeError(
                "ZoneSASolver requires a ZonalProblem instance"
            )
        if config is None:
            config = SolverConfig()

        progress = config.metadata.get("progress") if config.metadata else None

        blm = float(problem.parameters.get("BLM", 0.0))
        num_iterations = int(
            problem.parameters.get("NUMITNS", self._num_iterations)
        )
        num_temp_steps = int(
            problem.parameters.get("NUMTEMP", self._num_temp_steps)
        )

        # Build precomputed cache once
        cache = ZoneProblemCache.from_zone_problem(problem)

        pu_ids = problem.planning_units["id"].tolist()
        n_pu = len(pu_ids)
        pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
        zone_ids_list = sorted(problem.zone_ids)
        zone_options = np.array(zone_ids_list, dtype=int)
        n_zone_options = len(zone_options)

        locked: dict[int, int] = {}
        if "status" in problem.planning_units.columns:
            for _, row in problem.planning_units.iterrows():
                s = int(row["status"])
                idx = pu_id_to_idx[int(row["id"])]
                if s == 2:
                    locked[idx] = zone_ids_list[0]
                elif s == 3:
                    locked[idx] = 0

        swappable = np.array(
            [i for i in range(n_pu) if i not in locked], dtype=int
        )
        n_swappable = len(swappable)

        if progress is not None:
            progress.status = "running"
            progress.total_runs = config.num_solutions
            progress.total_iterations = num_iterations

        solutions = []
        for run_idx in range(config.num_solutions):
            if progress is not None:
                progress.current_run = run_idx + 1
                progress.iteration = 0

            if config.seed is not None:
                rng = np.random.default_rng(config.seed + run_idx)
            else:
                rng = np.random.default_rng()

            assignment = np.zeros(n_pu, dtype=int)
            for idx, zid in locked.items():
                assignment[idx] = zid
            for idx in swappable:
                assignment[idx] = zone_options[rng.integers(n_zone_options)]

            # Compute held_per_zone and initial objective from cache
            held_per_zone = cache.compute_held_per_zone(assignment)
            current_obj = cache.compute_full_zone_objective(
                assignment, held_per_zone, blm
            )

            # Estimate initial temperature using delta approach
            deltas = []
            for _ in range(min(1000, num_iterations // 10)):
                idx = int(swappable[rng.integers(n_swappable)])
                old_zone = int(assignment[idx])
                new_zone = int(zone_options[rng.integers(n_zone_options)])
                if new_zone == old_zone:
                    continue
                delta = cache.compute_delta_zone_objective(
                    idx, old_zone, new_zone, assignment, held_per_zone, blm
                )
                if delta > 0:
                    deltas.append(delta)

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
            iter_count = 0

            for _ in range(num_iterations):
                idx = int(swappable[rng.integers(n_swappable)])
                old_zone = int(assignment[idx])
                new_zone = int(zone_options[rng.integers(n_zone_options)])
                if new_zone == old_zone:
                    continue

                delta = cache.compute_delta_zone_objective(
                    idx, old_zone, new_zone, assignment, held_per_zone, blm
                )

                if delta <= 0 or (
                    temp > 0 and rng.random() < math.exp(-delta / temp)
                ):
                    assignment[idx] = new_zone
                    cache.update_held_per_zone(
                        held_per_zone, idx, old_zone, new_zone
                    )
                    current_obj += delta

                    if current_obj < best_obj:
                        best_assignment = assignment.copy()
                        best_obj = current_obj

                step_count += 1
                if step_count >= iters_per_step:
                    temp *= alpha
                    step_count = 0

                iter_count += 1
                if progress is not None and iter_count % 1000 == 0:
                    progress.iteration = iter_count
                    progress.best_objective = best_obj

            selected = best_assignment > 0
            cost = compute_zone_cost(problem, best_assignment)
            std_boundary = compute_standard_boundary(problem, best_assignment)
            zone_boundary = compute_zone_boundary(problem, best_assignment)
            zone_targets = check_zone_targets(problem, best_assignment)
            zone_penalty = compute_zone_penalty(problem, best_assignment)

            # Compute raw shortfall (unweighted by SPF)
            zone_shortfall = 0.0
            if problem.zone_targets is not None:
                misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
                pu_index = {
                    int(pid): i
                    for i, pid in enumerate(
                        problem.planning_units["id"].tolist()
                    )
                }
                for _, trow in problem.zone_targets.iterrows():
                    zid = int(trow["zone"])
                    fid = int(trow["feature"])
                    target = float(trow["target"])
                    contribution = problem.get_contribution(fid, zid)
                    feat_data = problem.pu_vs_features[
                        problem.pu_vs_features["species"] == fid
                    ]
                    achieved = 0.0
                    for _, r in feat_data.iterrows():
                        pid = int(r["pu"])
                        idx = pu_index.get(pid)
                        if (
                            idx is not None
                            and int(best_assignment[idx]) == zid
                        ):
                            achieved += float(r["amount"]) * contribution
                    zone_shortfall += max(
                        0.0, target * misslevel - achieved
                    )

            sol = Solution(
                selected=selected,
                cost=cost,
                boundary=std_boundary,
                objective=best_obj,
                targets_met={},
                penalty=zone_penalty,
                shortfall=zone_shortfall,
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

        if progress is not None:
            progress.status = "done"

        return solutions
