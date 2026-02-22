"""ProblemCache — precomputed NumPy arrays for fast solver iteration.

Converts ConservationProblem DataFrames into dense NumPy arrays once,
then provides O(degree + features_per_pu) delta computation for
single-PU flips.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.utils import compute_cost_threshold_penalty


@dataclass(frozen=True)
class ProblemCache:
    """Precomputed arrays derived from a ConservationProblem.

    All DataFrame iteration happens once during ``from_problem()``.
    Subsequent calls to ``compute_held``, ``compute_full_objective``,
    and ``compute_delta_objective`` use only NumPy operations.

    Attributes
    ----------
    costs : np.ndarray
        (n_pu,) float64 — cost of each planning unit.
    statuses : np.ndarray
        (n_pu,) int32 — status of each planning unit.
    pu_id_to_idx : dict[int, int]
        Mapping from planning unit ID to array index.
    pu_feat_matrix : np.ndarray
        (n_pu, n_feat) float64 — amount of each feature in each PU.
    feat_targets : np.ndarray
        (n_feat,) float64 — target for each feature.
    feat_spf : np.ndarray
        (n_feat,) float64 — species penalty factor for each feature.
    feat_id_to_col : dict[int, int]
        Mapping from feature ID to column index in pu_feat_matrix.
    neighbors : list[list[tuple[int, float]]]
        Adjacency list — neighbors[i] is list of (j, boundary_weight).
    self_boundary : np.ndarray
        (n_pu,) float64 — diagonal boundary value for each PU.
    misslevel : float
        MISSLEVEL parameter (default 1.0).
    cost_thresh : float
        COSTTHRESH parameter (default 0.0).
    thresh_pen1 : float
        THRESHPEN1 parameter (default 0.0).
    thresh_pen2 : float
        THRESHPEN2 parameter (default 0.0).
    """

    n_pu: int
    n_feat: int
    costs: np.ndarray
    statuses: np.ndarray
    pu_id_to_idx: dict[int, int]
    pu_feat_matrix: np.ndarray
    feat_targets: np.ndarray
    feat_spf: np.ndarray
    feat_id_to_col: dict[int, int]
    neighbors: list[list[tuple[int, float]]]
    self_boundary: np.ndarray
    misslevel: float
    cost_thresh: float
    thresh_pen1: float
    thresh_pen2: float

    @classmethod
    def from_problem(cls, problem: ConservationProblem) -> ProblemCache:
        """Build a ProblemCache from a ConservationProblem.

        All DataFrame iteration happens here, once.

        Parameters
        ----------
        problem : ConservationProblem
            The conservation problem to cache.

        Returns
        -------
        ProblemCache
            Precomputed arrays ready for fast solver iteration.
        """
        pu_df = problem.planning_units
        feat_df = problem.features
        puvspr_df = problem.pu_vs_features

        n_pu = len(pu_df)
        n_feat = len(feat_df)

        # Planning unit arrays
        pu_ids = pu_df["id"].values
        pu_id_to_idx: dict[int, int] = {}
        for i, pid in enumerate(pu_ids):
            pu_id_to_idx[int(pid)] = i

        costs = pu_df["cost"].values.astype(np.float64)
        statuses = pu_df["status"].values.astype(np.int32)

        # Feature arrays
        feat_ids = feat_df["id"].values
        feat_id_to_col: dict[int, int] = {}
        for j, fid in enumerate(feat_ids):
            feat_id_to_col[int(fid)] = j

        feat_targets = feat_df["target"].values.astype(np.float64)
        feat_spf = feat_df["spf"].values.astype(np.float64)

        # PU-feature matrix (dense)
        pu_feat_matrix = np.zeros((n_pu, n_feat), dtype=np.float64)
        for _, row in puvspr_df.iterrows():
            pu_id = int(row["pu"])
            sp_id = int(row["species"])
            amount = float(row["amount"])
            i = pu_id_to_idx.get(pu_id)
            j = feat_id_to_col.get(sp_id)
            if i is not None and j is not None:
                pu_feat_matrix[i, j] = amount

        # Boundary: adjacency list + self-boundary
        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n_pu)]
        self_boundary = np.zeros(n_pu, dtype=np.float64)

        if problem.boundary is not None:
            for _, row in problem.boundary.iterrows():
                id1 = int(row["id1"])
                id2 = int(row["id2"])
                bval = float(row["boundary"])

                if id1 == id2:
                    idx = pu_id_to_idx.get(id1)
                    if idx is not None:
                        self_boundary[idx] = bval
                else:
                    idx1 = pu_id_to_idx.get(id1)
                    idx2 = pu_id_to_idx.get(id2)
                    if idx1 is not None and idx2 is not None:
                        neighbors[idx1].append((idx2, bval))
                        neighbors[idx2].append((idx1, bval))

        # Cached scalars
        params = problem.parameters
        misslevel = float(params.get("MISSLEVEL", 1.0))
        cost_thresh = float(params.get("COSTTHRESH", 0.0))
        thresh_pen1 = float(params.get("THRESHPEN1", 0.0))
        thresh_pen2 = float(params.get("THRESHPEN2", 0.0))

        return cls(
            n_pu=n_pu,
            n_feat=n_feat,
            costs=costs,
            statuses=statuses,
            pu_id_to_idx=pu_id_to_idx,
            pu_feat_matrix=pu_feat_matrix,
            feat_targets=feat_targets,
            feat_spf=feat_spf,
            feat_id_to_col=feat_id_to_col,
            neighbors=neighbors,
            self_boundary=self_boundary,
            misslevel=misslevel,
            cost_thresh=cost_thresh,
            thresh_pen1=thresh_pen1,
            thresh_pen2=thresh_pen2,
        )

    # ------------------------------------------------------------------
    # Compute methods
    # ------------------------------------------------------------------

    def compute_held(self, selected: np.ndarray) -> np.ndarray:
        """Compute the amount held per feature for the given selection.

        Parameters
        ----------
        selected : np.ndarray
            (n_pu,) boolean array indicating selected planning units.

        Returns
        -------
        np.ndarray
            (n_feat,) float64 — total amount held for each feature.
        """
        return self.pu_feat_matrix[selected].sum(axis=0)

    def compute_full_objective(
        self,
        selected: np.ndarray,
        held: np.ndarray,
        blm: float,
    ) -> float:
        """Compute the full Marxan objective value.

        objective = cost + BLM * boundary + penalty [+ cost_threshold_penalty]

        Parameters
        ----------
        selected : np.ndarray
            (n_pu,) boolean selection array.
        held : np.ndarray
            (n_feat,) float64 — amount held per feature (from compute_held).
        blm : float
            Boundary length modifier.

        Returns
        -------
        float
            The full objective value.
        """
        # Cost
        total_cost = float(np.sum(self.costs[selected]))

        # Boundary
        boundary = self._compute_boundary(selected)

        # Penalty: SPF * max(0, target - held) for each feature
        shortfalls = np.maximum(0.0, self.feat_targets - held)
        penalty = float(np.dot(self.feat_spf, shortfalls))

        obj = total_cost + blm * boundary + penalty

        # Cost threshold penalty
        if self.cost_thresh > 0:
            obj += compute_cost_threshold_penalty(
                total_cost, self.cost_thresh, self.thresh_pen1, self.thresh_pen2
            )

        return obj

    def compute_delta_objective(
        self,
        idx: int,
        selected: np.ndarray,
        held: np.ndarray,
        total_cost: float,
        blm: float,
    ) -> float:
        """Compute the change in objective from flipping PU at index idx.

        This is O(degree + features_per_pu) instead of O(B + F*P).

        Parameters
        ----------
        idx : int
            Index of the planning unit to flip.
        selected : np.ndarray
            (n_pu,) boolean selection array (current state).
        held : np.ndarray
            (n_feat,) float64 — current amount held per feature.
        total_cost : float
            Current total cost of the selection.
        blm : float
            Boundary length modifier.

        Returns
        -------
        float
            The change in objective (positive means objective increases).
        """
        adding = not selected[idx]
        sign = 1.0 if adding else -1.0

        # --- Cost delta ---
        cost_delta = sign * self.costs[idx]

        # --- Boundary delta ---
        boundary_delta = 0.0
        # Self-boundary: only contributes when PU is selected
        boundary_delta += sign * self.self_boundary[idx]
        # Neighbor boundary: shared edges
        for j, w in self.neighbors[idx]:
            if adding:
                # Adding idx: if j is selected, shared boundary goes away
                # (was +w because mismatch, now both selected => no mismatch)
                # If j is not selected, shared boundary appears
                # (was no mismatch, now idx selected but j not => +w)
                if selected[j]:
                    boundary_delta -= w  # Mismatch resolved
                else:
                    boundary_delta += w  # New mismatch
            else:
                # Removing idx: if j is selected, new mismatch
                # If j is not selected, mismatch resolved
                if selected[j]:
                    boundary_delta += w  # New mismatch
                else:
                    boundary_delta -= w  # Mismatch resolved

        # --- Penalty delta ---
        # Only features present in this PU can change their shortfall
        pu_amounts = self.pu_feat_matrix[idx]
        old_shortfalls = np.maximum(0.0, self.feat_targets - held)
        new_held = held + sign * pu_amounts
        new_shortfalls = np.maximum(0.0, self.feat_targets - new_held)
        penalty_delta = float(
            np.dot(self.feat_spf, new_shortfalls - old_shortfalls)
        )

        delta = cost_delta + blm * boundary_delta + penalty_delta

        # --- Cost threshold delta ---
        if self.cost_thresh > 0:
            new_cost = total_cost + cost_delta
            old_ct_penalty = compute_cost_threshold_penalty(
                total_cost,
                self.cost_thresh,
                self.thresh_pen1,
                self.thresh_pen2,
            )
            new_ct_penalty = compute_cost_threshold_penalty(
                new_cost,
                self.cost_thresh,
                self.thresh_pen1,
                self.thresh_pen2,
            )
            delta += new_ct_penalty - old_ct_penalty

        return delta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_boundary(self, selected: np.ndarray) -> float:
        """Compute total boundary for a selection using cached arrays.

        Parameters
        ----------
        selected : np.ndarray
            (n_pu,) boolean selection array.

        Returns
        -------
        float
            Total boundary length.
        """
        # Self-boundary of selected PUs
        total = float(np.sum(self.self_boundary[selected]))

        # Shared boundary for mismatched neighbors
        for i in range(len(selected)):
            if not selected[i]:
                continue
            for j, w in self.neighbors[i]:
                # Only count when exactly one is selected; avoid double-count
                # by only counting (i, j) when i < j or j is not selected
                if not selected[j]:
                    total += w
        return total
