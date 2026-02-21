"""Greedy heuristic solver for conservation planning.

Selects planning units one-by-one based on cost-effectiveness
(marginal contribution per unit cost). Fast baseline for comparison
with SA and MIP solvers.
"""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


class HeuristicSolver(Solver):
    """Greedy cost-effectiveness heuristic solver."""

    def solve(
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig(num_solutions=1)

        rng = np.random.default_rng(config.seed)
        solutions = []

        for _ in range(config.num_solutions):
            sol = self._solve_once(problem, rng)
            solutions.append(sol)

        return solutions

    def _solve_once(
        self,
        problem: ConservationProblem,
        rng: np.random.Generator,
    ) -> Solution:
        n = problem.n_planning_units
        pu_ids = problem.planning_units["id"].values
        costs = problem.planning_units["cost"].values.astype(float)
        statuses = problem.planning_units["status"].values.astype(int)

        selected = np.zeros(n, dtype=bool)

        # Lock-in (status 2) and lock-out (status 3)
        locked_in = statuses == 2
        locked_out = statuses == 3
        selected[locked_in] = True

        # Build feature contribution lookup: pu_index -> {fid: amount}
        pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
        contributions: dict[int, dict[int, float]] = {}
        for _, row in problem.pu_vs_features.iterrows():
            pid = int(row["pu"])
            fid = int(row["species"])
            amount = float(row["amount"])
            idx = pu_id_to_idx.get(pid)
            if idx is not None:
                contributions.setdefault(idx, {})[fid] = amount

        # Track remaining need per feature
        remaining: dict[int, float] = {}
        for _, row in problem.features.iterrows():
            fid = int(row["id"])
            target = float(row["target"])
            remaining[fid] = target

        # Subtract locked-in contributions
        for idx in np.where(selected)[0]:
            for fid, amount in contributions.get(int(idx), {}).items():
                if fid in remaining:
                    remaining[fid] -= amount

        # Greedy loop: select most cost-effective PU until all targets met
        available = np.where(~selected & ~locked_out)[0]
        # Add small noise for diversity across runs
        noise = rng.uniform(0.0, 0.01, size=n)

        while any(r > 0 for r in remaining.values()) and len(available) > 0:
            best_idx = -1
            best_score = -1.0

            for idx in available:
                marginal = 0.0
                for fid, amount in contributions.get(int(idx), {}).items():
                    if remaining.get(fid, 0.0) > 0:
                        marginal += min(amount, remaining[fid])
                cost = max(costs[idx], 1e-10)
                score = marginal / cost + noise[idx]
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx < 0 or best_score <= 0:
                break

            selected[best_idx] = True
            for fid, amount in contributions.get(int(best_idx), {}).items():
                if fid in remaining:
                    remaining[fid] -= amount
            available = available[available != best_idx]

        # Compute objective
        blm = float(problem.parameters.get("BLM", 0.0))
        total_cost = float(costs[selected].sum())

        boundary_val = 0.0
        if problem.boundary is not None and blm > 0:
            for _, row in problem.boundary.iterrows():
                i = pu_id_to_idx.get(int(row["id1"]))
                j = pu_id_to_idx.get(int(row["id2"]))
                if i is not None and j is not None:
                    if selected[i] != selected[j]:
                        boundary_val += float(row["boundary"])

        # Check targets
        targets_met: dict[int, bool] = {}
        for _, row in problem.features.iterrows():
            fid = int(row["id"])
            targets_met[fid] = remaining.get(fid, 0.0) <= 0

        # Penalty
        penalty = 0.0
        for _, row in problem.features.iterrows():
            fid = int(row["id"])
            if not targets_met[fid]:
                spf = float(row.get("spf", 1.0))
                shortfall = max(remaining.get(fid, 0.0), 0.0)
                penalty += spf * shortfall

        objective = total_cost + blm * boundary_val + penalty

        return Solution(
            selected=selected,
            cost=total_cost,
            boundary=boundary_val,
            objective=objective,
            targets_met=targets_met,
            metadata={"solver": "greedy"},
        )

    def name(self) -> str:
        return "greedy"

    def supports_zones(self) -> bool:
        return False
