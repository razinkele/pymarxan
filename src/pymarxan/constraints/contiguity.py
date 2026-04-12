"""Contiguity constraint — ensures selected PUs form a single connected component.

MIP-only constraint. For SA, use post-hoc evaluation (too expensive for inner loop).
"""
from __future__ import annotations

import warnings
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np

from pymarxan.constraints.base import Constraint, ConstraintResult

if TYPE_CHECKING:
    import pulp

    from pymarxan.models.problem import ConservationProblem


def _build_adjacency(boundary) -> dict[int, set[int]]:
    """Build adjacency dict from a boundary DataFrame (id1, id2, boundary)."""
    adj: dict[int, set[int]] = defaultdict(set)
    for _, row in boundary.iterrows():
        a, b = int(row["id1"]), int(row["id2"])
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    return dict(adj)


def count_connected_components(
    selected_ids: set[int],
    adjacency: dict[int, set[int]],
) -> int:
    """Count connected components among *selected_ids* using BFS.

    Parameters
    ----------
    selected_ids : set[int]
        IDs of selected planning units.
    adjacency : dict[int, set[int]]
        Adjacency list keyed by PU ID.

    Returns
    -------
    int
        Number of connected components (0 if no PUs selected).
    """
    if not selected_ids:
        return 0

    visited: set[int] = set()
    components = 0
    for start in selected_ids:
        if start in visited:
            continue
        components += 1
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, ()):
                if neighbor in selected_ids and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return components


class ContiguityConstraint(Constraint):
    """Ensures selected planning units form a single connected component.

    MIP-only constraint. For SA, use post-hoc repair (best-effort).

    Parameters
    ----------
    penalty_weight : float
        Penalty per extra connected component beyond 1.
    max_pu_warning : int
        Emit a warning when the number of PUs exceeds this threshold
        (network-flow MIP formulation scales poorly).
    """

    def __init__(
        self, penalty_weight: float = 1000.0, max_pu_warning: int = 5000
    ):
        self._penalty_weight = penalty_weight
        self._max_pu_warning = max_pu_warning

    def name(self) -> str:
        return "ContiguityConstraint"

    def evaluate(
        self, problem: ConservationProblem, selected: np.ndarray
    ) -> ConstraintResult:
        """BFS to count connected components among selected PUs."""
        pu_ids = problem.planning_units["id"].values
        selected_ids = {int(pu_ids[i]) for i in range(len(selected)) if selected[i]}

        if not selected_ids:
            return ConstraintResult(
                satisfied=True, violation=0.0, description="No PUs selected (vacuously contiguous)"
            )

        adj = _build_adjacency(problem.boundary) if problem.boundary is not None else {}
        n_comp = count_connected_components(selected_ids, adj)
        violation = max(0.0, n_comp - 1)
        return ConstraintResult(
            satisfied=n_comp <= 1,
            violation=violation,
            description=f"{n_comp} connected component(s) among {len(selected_ids)} selected PUs",
        )

    def penalty(
        self, problem: ConservationProblem, selected: np.ndarray
    ) -> float:
        result = self.evaluate(problem, selected)
        return self._penalty_weight * result.violation

    # ------------------------------------------------------------------
    # MIP formulation: single-commodity network flow
    # ------------------------------------------------------------------

    def apply_to_mip(
        self,
        problem: ConservationProblem,
        model: pulp.LpProblem,
        x: dict[int, pulp.LpVariable],
    ) -> None:
        """Single-commodity network flow formulation with root-selection binaries.

        Variables
        ---------
        r_i ∈ {0,1}  — root indicator for PU i
        f_ij >= 0     — flow on edge (i,j)

        Constraints
        -----------
        r_i <= x_i                                (root must be selected)
        Σ r_i = 1                                 (exactly one root)
        Σ_j f_ji - Σ_j f_ij >= x_i - (n-1)*r_i  (flow conservation)
        f_ij <= (n-1) * x_i                       (flow capacity — sender)
        f_ij <= (n-1) * x_j                       (flow capacity — receiver)

        Gate: skip if all feature targets are zero or there are no features.
        Warn if n_pu > max_pu_warning.
        """
        import pulp as _pulp

        # Gate: skip if no meaningful targets
        if problem.features is not None and len(problem.features) > 0:
            targets = problem.features["target"].values
            if np.all(targets <= 0):
                return
        else:
            return

        n = len(x)
        if n > self._max_pu_warning:
            warnings.warn(
                f"ContiguityConstraint: {n} planning units — "
                f"network-flow MIP may be slow (threshold={self._max_pu_warning})",
                stacklevel=2,
            )

        pu_ids = sorted(x.keys())
        pu_set = set(pu_ids)

        # Build adjacency (directed edges in both directions)
        adj: dict[int, set[int]] = defaultdict(set)
        if problem.boundary is not None:
            for _, row in problem.boundary.iterrows():
                a, b = int(row["id1"]), int(row["id2"])
                if a != b and a in pu_set and b in pu_set:
                    adj[a].add(b)
                    adj[b].add(a)

        n_cap = max(n - 1, 1)  # max flow on any edge

        # Root-selection binaries
        r = {
            pid: _pulp.LpVariable(f"contiguity_root_{pid}", cat="Binary")
            for pid in pu_ids
        }

        # Root must be selected
        for pid in pu_ids:
            model += r[pid] <= x[pid], f"contiguity_root_sel_{pid}"

        # Exactly one root
        model += (
            _pulp.lpSum(r[pid] for pid in pu_ids) == 1,
            "contiguity_one_root",
        )

        # Flow variables for each directed edge
        edge_set: set[tuple[int, int]] = set()
        for a in pu_ids:
            for b in adj.get(a, ()):
                if (a, b) not in edge_set:
                    edge_set.add((a, b))
        edges = sorted(edge_set)

        f = {
            (a, b): _pulp.LpVariable(f"contiguity_flow_{a}_{b}", lowBound=0)
            for (a, b) in edges
        }

        # Flow conservation: net inflow >= x_i - n*r_i
        # Root can produce up to n-1 units; non-root selected must consume 1.
        for pid in pu_ids:
            inflow = _pulp.lpSum(
                f[(a, b)] for (a, b) in edges if b == pid
            )
            outflow = _pulp.lpSum(
                f[(a, b)] for (a, b) in edges if a == pid
            )
            model += (
                inflow - outflow >= x[pid] - n * r[pid],
                f"contiguity_flow_cons_{pid}",
            )

        # Flow capacity
        for a, b in edges:
            model += (
                f[(a, b)] <= n_cap * x[a],
                f"contiguity_cap_send_{a}_{b}",
            )
            model += (
                f[(a, b)] <= n_cap * x[b],
                f"contiguity_cap_recv_{a}_{b}",
            )
