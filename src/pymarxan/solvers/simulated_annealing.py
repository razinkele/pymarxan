"""Native Python simulated annealing solver for Marxan conservation planning."""
from __future__ import annotations

import math

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.utils import build_solution, compute_objective


class SimulatedAnnealingSolver(Solver):
    """Simulated annealing solver implemented natively in Python/NumPy."""

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

        pu_ids = problem.planning_units["id"].tolist()
        n_pu = len(pu_ids)
        pu_index = {pid: i for i, pid in enumerate(pu_ids)}

        # Identify locked PUs
        locked_in = set()
        locked_out = set()
        if "status" in problem.planning_units.columns:
            for _, row in problem.planning_units.iterrows():
                s = int(row["status"])
                idx = pu_index[int(row["id"])]
                if s == 2:
                    locked_in.add(idx)
                elif s == 3:
                    locked_out.add(idx)

        # Swappable indices (not locked)
        swappable = [
            i for i in range(n_pu)
            if i not in locked_in and i not in locked_out
        ]

        if not swappable:
            # Everything is locked — just build the forced solution
            selected = np.zeros(n_pu, dtype=bool)
            for idx in locked_in:
                selected[idx] = True
            sol = build_solution(
                problem, selected, blm,
                metadata={"solver": self.name()},
            )
            return [sol] * config.num_solutions

        solutions = []
        for run_idx in range(config.num_solutions):
            # Determine seed for this run
            if config.seed is not None:
                rng = np.random.default_rng(config.seed + run_idx)
            else:
                rng = np.random.default_rng()

            # Initialize selection
            selected = np.zeros(n_pu, dtype=bool)
            for idx in locked_in:
                selected[idx] = True
            # Randomly select ~initial_prop of swappable PUs
            for idx in swappable:
                if rng.random() < initial_prop:
                    selected[idx] = True

            current_obj = compute_objective(
                problem, selected, pu_index, blm
            )

            # Estimate initial temperature via sampling
            deltas = []
            for _ in range(min(1000, num_iterations // 10)):
                idx = swappable[rng.integers(len(swappable))]
                selected[idx] = not selected[idx]
                new_obj = compute_objective(
                    problem, selected, pu_index, blm
                )
                delta = new_obj - current_obj
                if delta > 0:
                    deltas.append(delta)
                selected[idx] = not selected[idx]  # revert

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

            for iteration in range(num_iterations):
                # Pick random swappable PU and flip
                idx = swappable[rng.integers(len(swappable))]
                selected[idx] = not selected[idx]

                new_obj = compute_objective(
                    problem, selected, pu_index, blm
                )
                delta = new_obj - current_obj

                # Acceptance criterion
                if delta <= 0:
                    current_obj = new_obj
                elif temp > 0 and rng.random() < math.exp(
                    -delta / temp
                ):
                    current_obj = new_obj
                else:
                    selected[idx] = not selected[idx]  # reject

                # Track best
                if current_obj < best_obj:
                    best_selected = selected.copy()
                    best_obj = current_obj

                # Cool
                step_count += 1
                if step_count >= iters_per_step:
                    temp *= alpha
                    step_count = 0

            sol = build_solution(
                problem, best_selected, blm,
                metadata={
                    "solver": self.name(),
                    "run": run_idx + 1,
                    "initial_temp": round(initial_temp, 4),
                    "final_temp": round(temp, 6),
                    "best_objective": round(best_obj, 4),
                },
            )
            solutions.append(sol)

        return solutions
