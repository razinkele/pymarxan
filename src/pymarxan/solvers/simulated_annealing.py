"""Native Python simulated annealing solver for Marxan conservation planning."""
from __future__ import annotations

import math

import numpy as np

from pymarxan.models.problem import (
    STATUS_INITIAL_INCLUDE,
    STATUS_LOCKED_IN,
    STATUS_LOCKED_OUT,
    ConservationProblem,
)
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.utils import build_solution

HISTORY_SAMPLE_INTERVAL = 1000


class SimulatedAnnealingSolver(Solver):
    """Simulated annealing solver implemented natively in Python/NumPy.

    Uses ProblemCache for O(degree + features_per_pu) delta computation
    on each iteration instead of recomputing the full objective.
    """

    def __init__(
        self,
        num_iterations: int = 1_000_000,
        num_temp_steps: int = 10_000,
        initial_prop: float = 0.5,
    ):
        self._num_iterations = num_iterations
        self._num_temp_steps = num_temp_steps
        self._initial_prop = initial_prop

    def name(self) -> str:
        return "Simulated Annealing (Python)"

    def supports_zones(self) -> bool:
        return False

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
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
        initial_prop = float(
            problem.parameters.get("PROP", self._initial_prop)
        )

        progress = config.metadata.get("progress") if config.metadata else None

        # Build cache once — all DataFrame iteration happens here
        cache = ProblemCache.from_problem(problem)
        n_pu = cache.n_pu

        # Identify locked and swappable PUs from cached statuses
        locked_in: list[int] = []
        locked_out: list[int] = []
        initial_include: list[int] = []
        swappable: list[int] = []
        for i in range(n_pu):
            s = int(cache.statuses[i])
            if s == STATUS_LOCKED_IN:
                locked_in.append(i)
            elif s == STATUS_LOCKED_OUT:
                locked_out.append(i)
            elif s == STATUS_INITIAL_INCLUDE:
                initial_include.append(i)
                swappable.append(i)  # swappable!
            else:
                swappable.append(i)

        n_swappable = len(swappable)

        if n_swappable == 0:
            # Everything is locked — just build the forced solution
            selected = np.zeros(n_pu, dtype=bool)
            for idx in locked_in:
                selected[idx] = True
            sol = build_solution(
                problem, selected, blm,
                metadata={"solver": self.name()},
            )
            return [sol] * config.num_solutions

        # Convert swappable to numpy array for fast indexing
        swappable_arr = np.array(swappable, dtype=np.intp)

        if progress is not None:
            progress.status = "running"
            progress.total_runs = config.num_solutions
            progress.total_iterations = num_iterations

        solutions: list[Solution] = []
        for run_idx in range(config.num_solutions):
            if progress is not None:
                progress.current_run = run_idx + 1
                progress.iteration = 0

            # Determine seed for this run
            if config.seed is not None:
                rng = np.random.default_rng(config.seed + run_idx)
            else:
                rng = np.random.default_rng()

            # Initialize selection
            selected = np.zeros(n_pu, dtype=bool)
            for idx in locked_in:
                selected[idx] = True
            for idx in initial_include:
                selected[idx] = True
            # Randomly select ~initial_prop of swappable PUs
            for idx in swappable:
                if idx not in initial_include and rng.random() < initial_prop:
                    selected[idx] = True

            # Initialize incremental state
            held = cache.compute_held(selected)
            total_cost = float(np.sum(cache.costs[selected]))
            current_obj = cache.compute_full_objective(selected, held, blm)

            # Estimate initial temperature via sampling (using delta approach)
            deltas: list[float] = []
            n_samples = min(1000, num_iterations // 10)
            for _ in range(n_samples):
                idx = int(swappable_arr[rng.integers(n_swappable)])
                delta = cache.compute_delta_objective(
                    idx, selected, held, total_cost, blm
                )
                if delta > 0:
                    deltas.append(delta)

            if deltas:
                # Set T so ~50% of worsening moves are accepted
                avg_delta = sum(deltas) / len(deltas)
                initial_temp = -avg_delta / math.log(0.5)
            else:
                initial_temp = 1.0

            # Compute cooling factor
            iters_per_step = max(1, num_iterations // num_temp_steps)
            if initial_temp > 0:
                alpha = (0.001 / initial_temp) ** (
                    1.0 / max(1, num_temp_steps)
                )
            else:
                alpha = 0.99

            # Main SA loop
            temp = initial_temp
            best_selected = selected.copy()
            best_obj = current_obj
            step_count = 0

            # History recording for convergence plot
            history: dict[str, list] = {
                "iteration": [0],
                "objective": [current_obj],
                "best_objective": [current_obj],
                "temperature": [temp],
            }
            iter_count = 0

            for _ in range(num_iterations):
                # Pick random swappable PU
                idx = int(swappable_arr[rng.integers(n_swappable)])

                # Compute delta without flipping
                delta = cache.compute_delta_objective(
                    idx, selected, held, total_cost, blm
                )

                # Acceptance criterion
                if delta <= 0 or (
                    temp > 0 and rng.random() < math.exp(-delta / temp)
                ):
                    # Accept the move — update incremental state
                    sign = -1.0 if selected[idx] else 1.0
                    selected[idx] = not selected[idx]
                    held += sign * cache.pu_feat_matrix[idx]
                    total_cost += sign * cache.costs[idx]
                    current_obj += delta

                    # Track best
                    if current_obj < best_obj:
                        best_selected = selected.copy()
                        best_obj = current_obj

                # Cool
                step_count += 1
                if step_count >= iters_per_step:
                    temp *= alpha
                    step_count = 0

                # Sample history
                iter_count += 1
                if iter_count % HISTORY_SAMPLE_INTERVAL == 0:
                    history["iteration"].append(iter_count)
                    history["objective"].append(current_obj)
                    history["best_objective"].append(best_obj)
                    history["temperature"].append(temp)
                    if progress is not None:
                        progress.iteration = iter_count
                        progress.best_objective = best_obj

            # Record final state if not already sampled
            if history["iteration"][-1] != num_iterations:
                history["iteration"].append(num_iterations)
                history["objective"].append(current_obj)
                history["best_objective"].append(best_obj)
                history["temperature"].append(temp)

            sol = build_solution(
                problem, best_selected, blm,
                metadata={
                    "solver": self.name(),
                    "run": run_idx + 1,
                    "initial_temp": round(initial_temp, 4),
                    "final_temp": round(temp, 6),
                    "best_objective": round(best_obj, 4),
                    "history": history,
                },
            )
            solutions.append(sol)

        if progress is not None:
            progress.status = "done"

        return solutions
