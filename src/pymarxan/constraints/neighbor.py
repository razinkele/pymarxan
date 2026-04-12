"""Neighbor constraints — require selected PUs to have minimum neighbors.

MinNeighborConstraint: Each selected PU must have at least `min_neighbors`
selected neighbors. This is an IncrementalConstraint with O(degree) delta.

The penalty is proportional to the total deficit:
  penalty = penalty_weight * Σ max(0, min_neighbors - count_selected_neighbors[i])
  for all selected PUs i.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pymarxan.constraints.base import (
    ConstraintResult,
    IncrementalConstraint,
)

if TYPE_CHECKING:
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.solvers.cache import ProblemCache


@dataclass
class MinNeighborConstraint(IncrementalConstraint):
    """Each selected PU must have >= min_neighbors selected neighbors.

    Parameters
    ----------
    min_neighbors : int
        Minimum number of selected neighbors required.
    penalty_weight : float
        Penalty per unit of deficit.
    """

    min_neighbors: int = 1
    penalty_weight: float = 100.0

    def name(self) -> str:
        return f"MinNeighbor({self.min_neighbors})"

    # ------------------------------------------------------------------
    # Constraint interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        problem: ConservationProblem,
        selected: np.ndarray,
    ) -> ConstraintResult:
        """Count neighbor violations using cache adjacency."""
        from pymarxan.solvers.cache import ProblemCache

        cache = ProblemCache.from_problem(problem)
        total_deficit = 0
        n_violating = 0
        for i in range(len(selected)):
            if not selected[i]:
                continue
            start = cache.adj_start[i]
            end = cache.adj_start[i + 1]
            if start < end:
                nbs = cache.adj_indices[start:end]
                n_sel = int(np.sum(selected[nbs]))
            else:
                n_sel = 0
            deficit = max(0, self.min_neighbors - n_sel)
            total_deficit += deficit
            if deficit > 0:
                n_violating += 1
        return ConstraintResult(
            satisfied=total_deficit == 0,
            violation=float(total_deficit),
            description=(
                f"{n_violating} PUs with < {self.min_neighbors} neighbors"
            ),
        )

    def penalty(
        self,
        problem: ConservationProblem,
        selected: np.ndarray,
    ) -> float:
        result = self.evaluate(problem, selected)
        return self.penalty_weight * result.violation

    # ------------------------------------------------------------------
    # Incremental interface
    # ------------------------------------------------------------------

    def init_state(
        self,
        problem: ConservationProblem,
        selected: np.ndarray,
        cache: ProblemCache,
    ) -> dict[str, Any]:
        """State: neighbor_count[i] = number of selected neighbors of i."""
        n_pu = len(selected)
        neighbor_count = np.zeros(n_pu, dtype=np.int32)
        for i in range(n_pu):
            if not selected[i]:
                continue
            start = cache.adj_start[i]
            end = cache.adj_start[i + 1]
            if start < end:
                nbs = cache.adj_indices[start:end]
                for nb in nbs:
                    neighbor_count[nb] += 1
        return {"neighbor_count": neighbor_count}

    def compute_delta(
        self,
        idx: int,
        selected: np.ndarray,
        state: dict[str, Any],
        cache: ProblemCache,
    ) -> float:
        """O(degree) delta computation for neighbor constraint."""
        adding = not selected[idx]
        nc = state["neighbor_count"]

        start = cache.adj_start[idx]
        end = cache.adj_start[idx + 1]
        nbs = (
            cache.adj_indices[start:end]
            if start < end
            else np.array([], dtype=np.int32)
        )

        delta = 0.0

        if adding:
            # 1. New PU deficit
            new_deficit_self = max(0, self.min_neighbors - int(nc[idx]))
            delta += self.penalty_weight * new_deficit_self

            # 2. Each selected neighbor gains +1 neighbor count
            for nb in nbs:
                if selected[nb]:
                    old_def = max(0, self.min_neighbors - int(nc[nb]))
                    new_def = max(
                        0, self.min_neighbors - int(nc[nb]) - 1
                    )
                    delta += self.penalty_weight * (new_def - old_def)
        else:
            # 1. Remove PU's own deficit
            old_deficit_self = max(0, self.min_neighbors - int(nc[idx]))
            delta -= self.penalty_weight * old_deficit_self

            # 2. Each selected neighbor loses 1 neighbor count
            for nb in nbs:
                if selected[nb]:
                    old_def = max(0, self.min_neighbors - int(nc[nb]))
                    new_def = max(
                        0, self.min_neighbors - int(nc[nb]) + 1
                    )
                    delta += self.penalty_weight * (new_def - old_def)

        return delta

    def update_state(
        self,
        idx: int,
        selected: np.ndarray,
        state: dict[str, Any],
        cache: ProblemCache,
    ) -> None:
        """Update neighbor counts after flip (called AFTER selected mutated)."""
        nc = state["neighbor_count"]
        start = cache.adj_start[idx]
        end = cache.adj_start[idx + 1]
        if start < end:
            nbs = cache.adj_indices[start:end]
            sign = 1 if selected[idx] else -1
            for nb in nbs:
                nc[nb] += sign
