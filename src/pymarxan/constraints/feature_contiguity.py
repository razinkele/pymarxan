"""Feature contiguity constraint — per-feature connected component enforcement.

MIP-only constraint. Ensures that for each specified feature, the selected PUs
contributing to that feature form a single connected component.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np

from pymarxan.constraints.base import Constraint, ConstraintResult

if TYPE_CHECKING:
    import pulp

    from pymarxan.models.problem import ConservationProblem


class FeatureContiguityConstraint(Constraint):
    """Ensures PUs contributing to specified features form connected subgraphs.

    For each specified feature, checks that PUs with amount > 0 for that feature
    AND that are selected form a single connected component.

    MIP-only. Uses per-feature flow variables (increases model size significantly).

    Parameters
    ----------
    feature_ids : list[int] | None
        Features to enforce contiguity for. ``None`` means all features.
    penalty_weight : float
        Penalty per disconnected component per feature.
    """

    def __init__(
        self,
        feature_ids: list[int] | None = None,
        penalty_weight: float = 1000.0,
    ):
        self._feature_ids = feature_ids
        self._penalty_weight = penalty_weight

    def name(self) -> str:
        return "FeatureContiguityConstraint"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, problem: ConservationProblem, selected: np.ndarray
    ) -> ConstraintResult:
        """For each target feature, BFS among selected PUs with amount > 0."""
        pu_ids = problem.planning_units["id"].values
        selected_ids = {
            int(pu_ids[i]) for i in range(len(selected)) if selected[i]
        }

        feature_ids = self._resolve_feature_ids(problem)
        if not feature_ids:
            return ConstraintResult(
                satisfied=True,
                violation=0.0,
                description="No features to check",
            )

        adj = self._build_adjacency(problem)
        pvf = problem.pu_vs_features

        total_violation = 0.0
        details: list[str] = []
        for fid in feature_ids:
            # PUs with amount > 0 for this feature
            feat_rows = pvf[pvf["species"] == fid]
            contributing = {
                int(r) for r in feat_rows.loc[feat_rows["amount"] > 0, "pu"]
            }
            subset = contributing & selected_ids

            if len(subset) <= 1:
                continue

            n_comp = self._count_components(subset, adj)
            extra = n_comp - 1
            if extra > 0:
                total_violation += extra
                details.append(f"feature {fid}: {n_comp} components")

        satisfied = total_violation == 0.0
        desc = (
            "All feature subgraphs contiguous"
            if satisfied
            else "; ".join(details)
        )
        return ConstraintResult(
            satisfied=satisfied,
            violation=total_violation,
            description=desc,
        )

    def penalty(
        self, problem: ConservationProblem, selected: np.ndarray
    ) -> float:
        result = self.evaluate(problem, selected)
        return self._penalty_weight * result.violation

    # ------------------------------------------------------------------
    # MIP formulation: per-feature network flow
    # ------------------------------------------------------------------

    def apply_to_mip(
        self,
        problem: ConservationProblem,
        model: pulp.LpProblem,
        x: dict[int, pulp.LpVariable],
    ) -> None:
        """Per-feature network flow formulation.

        For each feature *f* with target > 0, adds a single-commodity flow
        restricted to PUs where ``amount[i, f] > 0``.
        """
        import pulp as _pulp

        feature_ids = self._resolve_feature_ids(problem)
        if not feature_ids:
            return

        # Only constrain features with target > 0
        if problem.features is not None:
            targets = dict(
                zip(problem.features["id"], problem.features["target"])
            )
        else:
            return

        pu_set = set(x.keys())
        adj_full = self._build_adjacency(problem)
        pvf = problem.pu_vs_features

        for fid in feature_ids:
            if targets.get(fid, 0) <= 0:
                continue

            feat_rows = pvf[pvf["species"] == fid]
            contributing = {
                int(r)
                for r in feat_rows.loc[feat_rows["amount"] > 0, "pu"]
            } & pu_set

            if len(contributing) <= 1:
                continue

            feat_pus = sorted(contributing)
            n = len(feat_pus)
            n_cap = max(n - 1, 1)  # max flow on any edge
            tag = f"fc{fid}"

            # Adjacency restricted to contributing PUs
            feat_adj: dict[int, set[int]] = defaultdict(set)
            for pid in feat_pus:
                for nb in adj_full.get(pid, ()):
                    if nb in contributing:
                        feat_adj[pid].add(nb)

            # Root-selection binaries
            r = {
                pid: _pulp.LpVariable(f"{tag}_root_{pid}", cat="Binary")
                for pid in feat_pus
            }

            for pid in feat_pus:
                model += r[pid] <= x[pid], f"{tag}_root_sel_{pid}"

            model += (
                _pulp.lpSum(r[pid] for pid in feat_pus) == 1,
                f"{tag}_one_root",
            )

            # Directed edges
            edges: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for a in feat_pus:
                for b in feat_adj.get(a, ()):
                    if (a, b) not in seen:
                        edges.append((a, b))
                        seen.add((a, b))

            f = {
                (a, b): _pulp.LpVariable(
                    f"{tag}_flow_{a}_{b}", lowBound=0
                )
                for a, b in edges
            }

            # Flow conservation
            for pid in feat_pus:
                inflow = _pulp.lpSum(
                    f[(a, b)] for a, b in edges if b == pid
                )
                outflow = _pulp.lpSum(
                    f[(a, b)] for a, b in edges if a == pid
                )
                model += (
                    inflow - outflow >= x[pid] - n * r[pid],
                    f"{tag}_flow_cons_{pid}",
                )

            # Flow capacity
            for a, b in edges:
                model += (
                    f[(a, b)] <= n_cap * x[a],
                    f"{tag}_cap_send_{a}_{b}",
                )
                model += (
                    f[(a, b)] <= n_cap * x[b],
                    f"{tag}_cap_recv_{a}_{b}",
                )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_feature_ids(
        self, problem: ConservationProblem
    ) -> list[int]:
        """Return feature IDs to check."""
        if self._feature_ids is not None:
            return list(self._feature_ids)
        if problem.features is not None and len(problem.features) > 0:
            return list(problem.features["id"].values)
        return []

    @staticmethod
    def _build_adjacency(
        problem: ConservationProblem,
    ) -> dict[int, set[int]]:
        adj: dict[int, set[int]] = defaultdict(set)
        if problem.boundary is not None:
            for _, row in problem.boundary.iterrows():
                a, b = int(row["id1"]), int(row["id2"])
                if a != b:
                    adj[a].add(b)
                    adj[b].add(a)
        return dict(adj)

    @staticmethod
    def _count_components(
        node_set: set[int], adjacency: dict[int, set[int]]
    ) -> int:
        if not node_set:
            return 0
        visited: set[int] = set()
        components = 0
        for start in node_set:
            if start in visited:
                continue
            components += 1
            queue = deque([start])
            visited.add(start)
            while queue:
                node = queue.popleft()
                for nb in adjacency.get(node, ()):
                    if nb in node_set and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
        return components
