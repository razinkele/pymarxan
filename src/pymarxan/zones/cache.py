"""ZoneProblemCache — precomputed NumPy arrays for fast multi-zone solver iteration.

Converts ZonalProblem DataFrames into dense NumPy arrays once,
then provides O(degree + features) delta computation for single-PU
zone reassignments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymarxan.zones.model import ZonalProblem


@dataclass(frozen=True)
class ZoneProblemCache:
    """Precomputed arrays derived from a ZonalProblem.

    All DataFrame iteration happens once during ``from_zone_problem()``.
    Subsequent calls use only NumPy operations.

    Attributes
    ----------
    n_pu : int
        Number of planning units.
    n_feat : int
        Number of conservation features.
    n_zones : int
        Number of zones (excluding unassigned=0).
    pu_id_to_idx : dict[int, int]
        Mapping from planning unit ID to array index.
    feat_id_to_col : dict[int, int]
        Mapping from feature ID to column index.
    zone_id_to_col : dict[int, int]
        Mapping from zone ID to column index (1..n_zones).
        Column 0 is reserved for "unassigned".
    pu_feat_matrix : np.ndarray
        (n_pu, n_feat) float64 — amount of each feature in each PU.
    feat_spf : np.ndarray
        (n_feat,) float64 — species penalty factor for each feature.
    zone_cost_matrix : np.ndarray
        (n_pu, n_zones+1) float64 — cost of placing each PU in each zone.
        Column 0 (unassigned) is always 0.
    contribution_matrix : np.ndarray
        (n_zones+1, n_feat) float64 — contribution multiplier per (zone, feature).
        Row 0 (unassigned) is always 0.
    zone_target_matrix : np.ndarray
        (n_zones+1, n_feat) float64 — target per (zone, feature).
        Row 0 (unassigned) is always 0.
    zone_boundary_costs : dict[tuple[int, int], float]
        Cross-zone boundary cost keyed by (zone_col1, zone_col2).
    neighbors : list[list[tuple[int, float]]]
        Adjacency list — neighbors[i] is list of (j, boundary_weight).
    self_boundary : np.ndarray
        (n_pu,) float64 — diagonal boundary value for each PU.
    """

    n_pu: int
    n_feat: int
    n_zones: int
    pu_id_to_idx: dict[int, int]
    feat_id_to_col: dict[int, int]
    zone_id_to_col: dict[int, int]
    pu_feat_matrix: np.ndarray
    feat_spf: np.ndarray
    zone_cost_matrix: np.ndarray
    contribution_matrix: np.ndarray
    zone_target_matrix: np.ndarray
    zone_boundary_costs: dict[tuple[int, int], float]
    neighbors: list[list[tuple[int, float]]]
    self_boundary: np.ndarray
    conn_neighbors: list[list[tuple[int, float]]]
    conn_weight: float

    @classmethod
    def from_zone_problem(cls, problem: ZonalProblem) -> ZoneProblemCache:
        """Build a ZoneProblemCache from a ZonalProblem.

        All DataFrame iteration happens here, once.

        Parameters
        ----------
        problem : ZonalProblem
            The zonal conservation problem to cache.

        Returns
        -------
        ZoneProblemCache
            Precomputed arrays ready for fast zone solver iteration.
        """
        pu_df = problem.planning_units
        feat_df = problem.features

        n_pu = len(pu_df)
        n_feat = len(feat_df)
        n_zones = problem.n_zones

        # --- PU index mapping ---
        pu_ids = pu_df["id"].values
        pu_id_to_idx: dict[int, int] = {}
        for i, pid in enumerate(pu_ids):
            pu_id_to_idx[int(pid)] = i

        # --- Feature index mapping ---
        feat_ids = feat_df["id"].values
        feat_id_to_col: dict[int, int] = {}
        for j, fid in enumerate(feat_ids):
            feat_id_to_col[int(fid)] = j

        feat_spf = np.asarray(feat_df["spf"].values, dtype=np.float64)

        # --- Zone index mapping (col 0 = unassigned) ---
        zone_ids = sorted(problem.zone_ids)
        zone_id_to_col: dict[int, int] = {}
        for k, zid in enumerate(zone_ids):
            zone_id_to_col[zid] = k + 1  # 1-indexed; 0 = unassigned

        # --- PU-feature matrix (dense) — use shared builder ---
        pu_feat_matrix = problem.build_pu_feature_matrix()

        # --- Zone cost matrix: (n_pu, n_zones+1) ---
        zone_cost_matrix = np.zeros((n_pu, n_zones + 1), dtype=np.float64)
        zc_pu = problem.zone_costs["pu"].values
        zc_zn = problem.zone_costs["zone"].values
        zc_ct = problem.zone_costs["cost"].values
        for k in range(len(zc_pu)):
            ri = pu_id_to_idx.get(int(zc_pu[k]))
            zcol = zone_id_to_col.get(int(zc_zn[k]))
            if ri is not None and zcol is not None:
                zone_cost_matrix[ri, zcol] = float(zc_ct[k])

        # --- Contribution matrix: (n_zones+1, n_feat) ---
        # Default: 1.0 for all zone-feature pairs (standard Marxan behavior)
        contribution_matrix = np.zeros((n_zones + 1, n_feat), dtype=np.float64)
        # Row 0 stays zero (unassigned contributes nothing)
        # Fill defaults for actual zones
        for zcol in range(1, n_zones + 1):
            contribution_matrix[zcol, :] = 1.0
        # Override with explicit contributions if available
        if problem.zone_contributions is not None:
            zc_feat = problem.zone_contributions["feature"].values
            zc_zone = problem.zone_contributions["zone"].values
            zc_val = problem.zone_contributions["contribution"].values
            for k in range(len(zc_feat)):
                fcol = feat_id_to_col.get(int(zc_feat[k]))
                zcol = zone_id_to_col.get(int(zc_zone[k]))
                if fcol is not None and zcol is not None:
                    contribution_matrix[zcol, fcol] = float(zc_val[k])

        # --- Zone target matrix: (n_zones+1, n_feat) ---
        zone_target_matrix = np.zeros((n_zones + 1, n_feat), dtype=np.float64)
        if problem.zone_targets is not None:
            zt_zn = problem.zone_targets["zone"].values
            zt_ft = problem.zone_targets["feature"].values
            zt_tg = problem.zone_targets["target"].values
            for k in range(len(zt_zn)):
                zcol = zone_id_to_col.get(int(zt_zn[k]))
                fcol = feat_id_to_col.get(int(zt_ft[k]))
                if zcol is not None and fcol is not None:
                    zone_target_matrix[zcol, fcol] = float(zt_tg[k])

        # Apply MISSLEVEL to zone targets (match objective.py behavior)
        misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
        zone_target_matrix *= misslevel

        # --- Zone boundary costs: dict[(zone_col, zone_col)] -> cost ---
        zbc: dict[tuple[int, int], float] = {}
        if problem.zone_boundary_costs is not None:
            zb_z1 = problem.zone_boundary_costs["zone1"].values
            zb_z2 = problem.zone_boundary_costs["zone2"].values
            zb_ct = problem.zone_boundary_costs["cost"].values
            for k in range(len(zb_z1)):
                zcol1 = zone_id_to_col.get(int(zb_z1[k]))
                zcol2 = zone_id_to_col.get(int(zb_z2[k]))
                if zcol1 is not None and zcol2 is not None:
                    zbc[(zcol1, zcol2)] = float(zb_ct[k])
                    zbc[(zcol2, zcol1)] = float(zb_ct[k])

        # --- Connectivity: adjacency list ---
        conn_neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n_pu)]
        conn_weight = float(problem.parameters.get("CONNECTIVITY_WEIGHT", 0.0))

        if problem.connectivity is not None and conn_weight != 0.0:
            c_id1 = problem.connectivity["id1"].values
            c_id2 = problem.connectivity["id2"].values
            c_val = problem.connectivity["value"].values
            for k in range(len(c_id1)):
                idx1 = pu_id_to_idx.get(int(c_id1[k]))
                idx2 = pu_id_to_idx.get(int(c_id2[k]))
                if idx1 is not None and idx2 is not None:
                    conn_neighbors[idx1].append((idx2, float(c_val[k])))
                    conn_neighbors[idx2].append((idx1, float(c_val[k])))

        # --- Boundary: adjacency list + self-boundary ---
        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n_pu)]
        self_boundary = np.zeros(n_pu, dtype=np.float64)

        if problem.boundary is not None:
            b_id1 = problem.boundary["id1"].values
            b_id2 = problem.boundary["id2"].values
            b_val = problem.boundary["boundary"].values
            for k in range(len(b_id1)):
                i1 = int(b_id1[k])
                i2 = int(b_id2[k])
                bval = float(b_val[k])

                if i1 == i2:
                    idx = pu_id_to_idx.get(i1)
                    if idx is not None:
                        self_boundary[idx] = bval
                else:
                    idx1 = pu_id_to_idx.get(i1)
                    idx2 = pu_id_to_idx.get(i2)
                    if idx1 is not None and idx2 is not None:
                        neighbors[idx1].append((idx2, bval))
                        neighbors[idx2].append((idx1, bval))

        return cls(
            n_pu=n_pu,
            n_feat=n_feat,
            n_zones=n_zones,
            pu_id_to_idx=pu_id_to_idx,
            feat_id_to_col=feat_id_to_col,
            zone_id_to_col=zone_id_to_col,
            pu_feat_matrix=pu_feat_matrix,
            feat_spf=feat_spf,
            zone_cost_matrix=zone_cost_matrix,
            contribution_matrix=contribution_matrix,
            zone_target_matrix=zone_target_matrix,
            zone_boundary_costs=zbc,
            neighbors=neighbors,
            self_boundary=self_boundary,
            conn_neighbors=conn_neighbors,
            conn_weight=conn_weight,
        )

    # ------------------------------------------------------------------
    # Held-per-zone computation
    # ------------------------------------------------------------------

    def compute_held_per_zone(self, assignment: np.ndarray) -> np.ndarray:
        """Compute held amounts per (zone, feature).

        For each zone z and feature f:
            held[z_col, f] = sum over PUs i where assignment[i] maps to z_col
                             of pu_feat_matrix[i, f] * contribution_matrix[z_col, f]

        Parameters
        ----------
        assignment : np.ndarray
            (n_pu,) int — zone ID for each PU (0 = unassigned).

        Returns
        -------
        np.ndarray
            (n_zones+1, n_feat) float64 — held amount per (zone_col, feature).
        """
        held = np.zeros((self.n_zones + 1, self.n_feat), dtype=np.float64)
        # Map zone IDs in assignment to column indices
        for i in range(self.n_pu):
            zid = int(assignment[i])
            if zid == 0:
                zcol = 0
            else:
                zcol = self.zone_id_to_col.get(zid, 0)
            if zcol == 0:
                continue  # unassigned contributes nothing
            # Add contribution: amount * contribution factor
            held[zcol] += self.pu_feat_matrix[i] * self.contribution_matrix[zcol]
        return held

    def update_held_per_zone(
        self,
        held: np.ndarray,
        idx: int,
        old_zone: int,
        new_zone: int,
    ) -> None:
        """In-place incremental update of held_per_zone after a zone change.

        Parameters
        ----------
        held : np.ndarray
            (n_zones+1, n_feat) float64 — current held, modified in place.
        idx : int
            Index of the planning unit being changed.
        old_zone : int
            Previous zone ID (0 = unassigned).
        new_zone : int
            New zone ID (0 = unassigned).
        """
        amounts = self.pu_feat_matrix[idx]

        # Remove from old zone
        if old_zone != 0:
            old_col = self.zone_id_to_col.get(old_zone, 0)
            if old_col != 0:
                held[old_col] -= amounts * self.contribution_matrix[old_col]

        # Add to new zone
        if new_zone != 0:
            new_col = self.zone_id_to_col.get(new_zone, 0)
            if new_col != 0:
                held[new_col] += amounts * self.contribution_matrix[new_col]

    # ------------------------------------------------------------------
    # Full objective computation
    # ------------------------------------------------------------------

    def compute_full_zone_objective(
        self,
        assignment: np.ndarray,
        held_per_zone: np.ndarray,
        blm: float,
    ) -> float:
        """Compute the full MarZone objective using precomputed arrays.

        objective = zone_cost + BLM * standard_boundary + zone_boundary + zone_penalty

        Parameters
        ----------
        assignment : np.ndarray
            (n_pu,) int — zone ID for each PU (0 = unassigned).
        held_per_zone : np.ndarray
            (n_zones+1, n_feat) float64 — from compute_held_per_zone.
        blm : float
            Boundary length modifier.

        Returns
        -------
        float
            The full zone objective value.
        """
        # --- Zone cost ---
        zone_cost = 0.0
        for i in range(self.n_pu):
            zid = int(assignment[i])
            if zid == 0:
                continue
            zcol = self.zone_id_to_col.get(zid, 0)
            zone_cost += self.zone_cost_matrix[i, zcol]

        # --- Standard boundary (selected = zone > 0) ---
        selected = assignment > 0
        std_boundary = self._compute_standard_boundary(selected)

        # --- Zone boundary ---
        zone_boundary = self._compute_zone_boundary(assignment)

        # --- Zone penalty ---
        zone_penalty = self._compute_zone_penalty(held_per_zone)

        # --- Connectivity ---
        connectivity = self._compute_zone_connectivity(assignment)

        return zone_cost + blm * std_boundary + zone_boundary + zone_penalty + connectivity

    # ------------------------------------------------------------------
    # Delta objective computation
    # ------------------------------------------------------------------

    def compute_delta_zone_objective(
        self,
        idx: int,
        old_zone: int,
        new_zone: int,
        assignment: np.ndarray,
        held_per_zone: np.ndarray,
        blm: float,
    ) -> float:
        """Compute the change in zone objective from reassigning PU idx.

        This is O(degree + features) instead of O(PU * features).

        Parameters
        ----------
        idx : int
            Index of the planning unit to reassign.
        old_zone : int
            Current zone ID of PU idx (0 = unassigned).
        new_zone : int
            New zone ID for PU idx (0 = unassigned).
        assignment : np.ndarray
            (n_pu,) int — current zone assignment (before the change).
        held_per_zone : np.ndarray
            (n_zones+1, n_feat) float64 — current held amounts.
        blm : float
            Boundary length modifier.

        Returns
        -------
        float
            Change in objective (positive = objective increases).
        """
        if old_zone == new_zone:
            return 0.0

        old_col = self.zone_id_to_col.get(old_zone, 0) if old_zone != 0 else 0
        new_col = self.zone_id_to_col.get(new_zone, 0) if new_zone != 0 else 0

        # --- Cost delta ---
        cost_delta = (
            self.zone_cost_matrix[idx, new_col]
            - self.zone_cost_matrix[idx, old_col]
        )

        # --- Standard boundary delta ---
        was_selected = old_zone > 0
        will_be_selected = new_zone > 0
        std_boundary_delta = 0.0
        if was_selected != will_be_selected:
            # Selection status is changing
            if will_be_selected:
                # Adding to selection
                std_boundary_delta = self._boundary_delta_add(idx, assignment > 0)
            else:
                # Removing from selection
                std_boundary_delta = self._boundary_delta_remove(idx, assignment > 0)

        # --- Zone boundary delta ---
        zone_boundary_delta = self._zone_boundary_delta(
            idx, old_zone, new_zone, assignment
        )

        # --- Penalty delta ---
        penalty_delta = self._penalty_delta(
            idx, old_col, new_col, held_per_zone
        )

        # --- Connectivity delta ---
        connectivity_delta = self._connectivity_delta(idx, old_zone, new_zone, assignment)

        return float(cost_delta + blm * std_boundary_delta + zone_boundary_delta + penalty_delta + connectivity_delta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_standard_boundary(self, selected: np.ndarray) -> float:
        """Compute standard boundary for selected PUs (zone > 0).

        Same logic as ProblemCache._compute_boundary: self-boundary of
        selected PUs + shared boundary for mismatched neighbors.
        """
        total = float(np.sum(self.self_boundary[selected]))
        for i in range(self.n_pu):
            if not selected[i]:
                continue
            for j, w in self.neighbors[i]:
                if not selected[j]:
                    total += w
        return total

    def _compute_zone_boundary(self, assignment: np.ndarray) -> float:
        """Compute zone boundary cost between adjacent PUs in different zones.

        For each edge (i, j) where both are in different non-zero zones,
        add the zone_boundary_costs for that pair of zones.
        """
        total = 0.0
        for i in range(self.n_pu):
            zid_i = int(assignment[i])
            if zid_i == 0:
                continue
            zcol_i = self.zone_id_to_col.get(zid_i, 0)
            for j, _w in self.neighbors[i]:
                if j <= i:
                    continue  # count each edge once
                zid_j = int(assignment[j])
                if zid_j == 0:
                    continue
                if zid_i == zid_j:
                    continue
                zcol_j = self.zone_id_to_col.get(zid_j, 0)
                total += self.zone_boundary_costs.get((zcol_i, zcol_j), 0.0)
        return total

    def _compute_zone_penalty(self, held_per_zone: np.ndarray) -> float:
        """Compute penalty for unmet zone-specific targets.

        penalty = sum over (zone, feature) of spf[f] * max(0, target[z,f] - held[z,f])
        """
        shortfalls = np.maximum(0.0, self.zone_target_matrix - held_per_zone)
        # shortfalls shape: (n_zones+1, n_feat)
        # Multiply each feature's shortfall by its SPF and sum
        total = float(np.sum(shortfalls * self.feat_spf[np.newaxis, :]))
        return total

    def _boundary_delta_add(self, idx: int, selected: np.ndarray) -> float:
        """Boundary delta when adding PU idx to the selection."""
        delta = self.self_boundary[idx]
        for j, w in self.neighbors[idx]:
            if selected[j]:
                delta -= w  # mismatch resolved
            else:
                delta += w  # new mismatch
        return float(delta)

    def _boundary_delta_remove(self, idx: int, selected: np.ndarray) -> float:
        """Boundary delta when removing PU idx from the selection."""
        delta = -self.self_boundary[idx]
        for j, w in self.neighbors[idx]:
            if selected[j]:
                delta += w  # new mismatch
            else:
                delta -= w  # mismatch resolved
        return float(delta)

    def _zone_boundary_delta(
        self,
        idx: int,
        old_zone: int,
        new_zone: int,
        assignment: np.ndarray,
    ) -> float:
        """Compute zone boundary cost delta for changing PU idx's zone.

        For each neighbor j of idx:
        - Remove any cross-zone cost from (old_zone, zone_j) if applicable
        - Add any cross-zone cost from (new_zone, zone_j) if applicable
        """
        delta = 0.0
        for j, _w in self.neighbors[idx]:
            zid_j = int(assignment[j])
            if zid_j == 0:
                continue
            zcol_j = self.zone_id_to_col.get(zid_j, 0)

            # Remove old cross-zone cost
            if old_zone != 0 and old_zone != zid_j:
                old_col = self.zone_id_to_col.get(old_zone, 0)
                delta -= self.zone_boundary_costs.get((old_col, zcol_j), 0.0)

            # Add new cross-zone cost
            if new_zone != 0 and new_zone != zid_j:
                new_col = self.zone_id_to_col.get(new_zone, 0)
                delta += self.zone_boundary_costs.get((new_col, zcol_j), 0.0)
        return delta

    def _penalty_delta(
        self,
        idx: int,
        old_col: int,
        new_col: int,
        held_per_zone: np.ndarray,
    ) -> float:
        """Compute penalty delta for changing PU idx from old_col to new_col.

        Only the two affected zone rows (old_col, new_col) can have
        their shortfalls change.
        """
        amounts = self.pu_feat_matrix[idx]
        delta = 0.0

        # Old zone: decrease held => possibly increase shortfall
        if old_col != 0:
            old_contrib = amounts * self.contribution_matrix[old_col]
            old_held = held_per_zone[old_col]
            old_targets = self.zone_target_matrix[old_col]

            old_shortfall = np.maximum(0.0, old_targets - old_held)
            new_shortfall = np.maximum(0.0, old_targets - (old_held - old_contrib))
            delta += float(np.dot(self.feat_spf, new_shortfall - old_shortfall))

        # New zone: increase held => possibly decrease shortfall
        if new_col != 0:
            new_contrib = amounts * self.contribution_matrix[new_col]
            new_held = held_per_zone[new_col]
            new_targets = self.zone_target_matrix[new_col]

            old_shortfall = np.maximum(0.0, new_targets - new_held)
            new_shortfall = np.maximum(0.0, new_targets - (new_held + new_contrib))
            delta += float(np.dot(self.feat_spf, new_shortfall - old_shortfall))

        return delta

    def _compute_zone_connectivity(self, assignment: np.ndarray) -> float:
        """Compute connectivity bonus for PUs in the same non-zero zone.

        For each edge (i, j) with value v where both are in the same
        non-zero zone, subtract v (bonus). Weighted by conn_weight.
        """
        if self.conn_weight == 0.0:
            return 0.0

        total = 0.0
        for i in range(self.n_pu):
            zid_i = int(assignment[i])
            if zid_i == 0:
                continue
            for j, v in self.conn_neighbors[i]:
                if j <= i:
                    continue  # count each edge once
                zid_j = int(assignment[j])
                if zid_i == zid_j:
                    total -= v
        return self.conn_weight * total

    def _connectivity_delta(
        self,
        idx: int,
        old_zone: int,
        new_zone: int,
        assignment: np.ndarray,
    ) -> float:
        """Compute connectivity delta for changing PU idx's zone.

        For each connectivity neighbor j of idx:
        - If j was in old_zone (same zone), we lose a bonus: +v
        - If j is in new_zone (will be same zone), we gain a bonus: -v
        """
        if self.conn_weight == 0.0:
            return 0.0

        delta = 0.0
        for j, v in self.conn_neighbors[idx]:
            zid_j = int(assignment[j])
            if zid_j == 0:
                continue
            # Lose bonus from old same-zone connections
            if old_zone != 0 and zid_j == old_zone:
                delta += v
            # Gain bonus from new same-zone connections
            if new_zone != 0 and zid_j == new_zone:
                delta -= v
        return self.conn_weight * delta
