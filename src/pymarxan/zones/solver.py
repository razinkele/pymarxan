"""Simulated annealing solver for multi-zone conservation planning."""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.cooling import CoolingSchedule
from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.model import ZonalProblem, fixed_zones, warn_zone_input_issues
from pymarxan.zones.objective import build_zone_solution


def _propose(
    rng: np.random.Generator, swappable: np.ndarray, opts: list[np.ndarray],
) -> tuple[int, int]:
    """A random swappable PU and a random zone from its options. Same two RNG draws as
    v0.37 for unrestricted PUs (whose options are the shared ``zone_options`` array)."""
    idx = int(swappable[rng.integers(len(swappable))])
    options = opts[idx]
    return idx, int(options[rng.integers(len(options))])


_COOLING_FACTORIES: dict[str, Callable[..., CoolingSchedule]] = {
    "geometric": CoolingSchedule.geometric,
    "exponential": CoolingSchedule.exponential,
    "linear": CoolingSchedule.linear,
    "lundy_mees": CoolingSchedule.lundy_mees,
}


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

    def supports_separation(self) -> bool:
        # Per-zone SEPDISTANCE / SEPNUM deferred to v0.3 (round-3 H1).
        return False

    def solve(
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        if not isinstance(problem, ZonalProblem):
            raise TypeError(
                "ZoneSASolver requires a ZonalProblem instance"
            )
        from pymarxan.solvers.separation import raise_if_separation_active
        raise_if_separation_active(problem, "ZoneSASolver")
        warn_zone_input_issues(problem, "ZoneSASolver")
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
        zone_ids_list = sorted(problem.zone_ids)
        zone_options = np.array(zone_ids_list, dtype=int)

        rules = problem.pu_zone_rules()
        locked = fixed_zones(rules)
        every = frozenset(zone_ids_list)
        opts: list[np.ndarray] = [
            zone_options if rule.allowed == every
            else np.array(sorted(rule.allowed), dtype=int)
            for rule in rules
        ]
        swappable = np.array(
            [i for i in range(n_pu) if i not in locked], dtype=int
        )
        n_swappable = len(swappable)

        if n_swappable == 0:
            # All PUs locked — build forced-assignment solution
            assignment = np.zeros(n_pu, dtype=int)
            for idx, zid in locked.items():
                assignment[idx] = zid
            solutions = []
            for run_idx in range(config.num_solutions):
                solutions.append(build_zone_solution(
                    problem, assignment, blm, solver_name=self.name(), run=run_idx + 1,
                ))
            if progress is not None:
                progress.status = "done"
            return solutions

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
                start = rules[idx].start
                if start is not None:
                    assignment[idx] = start
                else:
                    options = opts[idx]
                    assignment[idx] = options[rng.integers(len(options))]

            # Both accumulators (raw per zone, contribution-weighted overall) from the cache
            held = cache.compute_held(assignment)
            current_obj = cache.compute_full_zone_objective(assignment, held=held, blm=blm)

            # Estimate initial temperature using delta approach
            deltas = []
            for _ in range(min(1000, num_iterations // 10)):
                idx, new_zone = _propose(rng, swappable, opts)
                old_zone = int(assignment[idx])
                if new_zone == old_zone:
                    continue
                delta = cache.compute_delta_zone_objective(
                    idx, old_zone, new_zone, assignment, held=held, blm=blm
                )
                if delta > 0:
                    deltas.append(delta)

            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                initial_temp = -avg_delta / math.log(0.5)
            else:
                initial_temp = 1.0

            initial_temp = max(initial_temp, 0.001)

            # Honour COOLING parameter — was previously hardcoded geometric,
            # silently ignoring user choice of linear/exponential/lundy_mees.
            cooling_name = problem.parameters.get("COOLING", "geometric")
            factory = _COOLING_FACTORIES.get(cooling_name)
            if factory is None:
                msg = f"Unknown COOLING schedule: {cooling_name}"
                raise ValueError(msg)
            schedule = factory(
                initial_temp=initial_temp,
                final_temp=0.001,
                num_steps=num_temp_steps,
            )

            iters_per_step = max(1, num_iterations // num_temp_steps)
            temp = initial_temp
            best_assignment = assignment.copy()
            best_obj = current_obj
            step_count = 0
            iter_count = 0
            temp_step = 0

            for _ in range(num_iterations):
                # Cool and count BEFORE any early-continue
                step_count += 1
                if step_count >= iters_per_step:
                    temp_step = min(temp_step + 1, num_temp_steps)
                    temp = schedule.temperature(temp_step)
                    step_count = 0

                iter_count += 1
                if progress is not None and iter_count % 1000 == 0:
                    progress.iteration = iter_count
                    progress.best_objective = best_obj

                idx, new_zone = _propose(rng, swappable, opts)
                old_zone = int(assignment[idx])
                if new_zone == old_zone:
                    continue

                delta = cache.compute_delta_zone_objective(
                    idx, old_zone, new_zone, assignment, held=held, blm=blm
                )

                if delta <= 0 or (
                    temp > 0 and rng.random() < math.exp(-delta / temp)
                ):
                    assignment[idx] = new_zone
                    cache.update_held(held, idx, old_zone, new_zone)
                    current_obj += delta

                    if current_obj < best_obj:
                        best_assignment = assignment.copy()
                        best_obj = current_obj

            solutions.append(build_zone_solution(
                problem, best_assignment, blm, solver_name=self.name(), run=run_idx + 1,
            ))

        if progress is not None:
            progress.status = "done"

        return solutions
