"""ProblemCache — precomputed NumPy arrays for fast solver iteration.

Converts ConservationProblem DataFrames into dense NumPy arrays once,
then provides O(degree + features_per_pu) delta computation for
single-PU flips.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
    adj_indices : np.ndarray
        (n_edges,) int32 — flattened array of neighbor indices.
    adj_weights : np.ndarray
        (n_edges,) float64 — flattened array of boundary weights.
    adj_start : np.ndarray
        (n_pu + 1,) int32 — start index in adj arrays for each PU.
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
    adj_indices: np.ndarray
    adj_weights: np.ndarray
    adj_start: np.ndarray
    self_boundary: np.ndarray
    misslevel: float
    cost_thresh: float
    thresh_pen1: float
    thresh_pen2: float
    # PROBMODE 3 (Z-score chance constraints) — populated for all problems
    # but only used when probmode == 3. See solvers/probability.py for the math.
    probmode: int = 0
    prob_weight: float = 1.0
    # expected_matrix[i, j] = amount_ij · (1 − p_ij); collapses to amount when
    # there is no `prob` column or when prob == 0.
    expected_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # var_matrix[i, j] = amount_ij² · p_ij · (1 − p_ij); all-zero when no `prob`.
    var_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # feat_ptarget[j] = per-feature probability target; ≤0 means disabled.
    feat_ptarget: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Phase 19 clumping (TARGET2 / CLUMPTYPE / type-4 species). All-zero /
    # inactive when no feature has target2 > 0, so non-clumping problems
    # pay zero per-flip cost.
    feat_target2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    feat_clumptype: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    feat_baseline_penalty: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # feat_uses_pu[j] = sorted PU indices where amount_ij > 0; precomputed
    # so ClumpState can iterate only the participating PUs per feature.
    feat_uses_pu: list[np.ndarray] = field(default_factory=list)
    clumping_active: bool = False

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

        n_pu = len(pu_df)
        n_feat = len(feat_df)

        # Planning unit arrays
        pu_ids = pu_df["id"].values
        pu_id_to_idx: dict[int, int] = {}
        for i, pid in enumerate(pu_ids):
            pu_id_to_idx[int(pid)] = i

        costs = np.asarray(pu_df["cost"].values, dtype=np.float64)
        statuses = np.asarray(pu_df["status"].values, dtype=np.int32)

        # Feature arrays
        feat_ids = feat_df["id"].values
        feat_id_to_col: dict[int, int] = {}
        for j, fid in enumerate(feat_ids):
            feat_id_to_col[int(fid)] = j

        feat_targets = np.asarray(feat_df["target"].values, dtype=np.float64)
        feat_spf = np.asarray(feat_df["spf"].values, dtype=np.float64)

        # PU-feature matrix (dense) — use shared builder
        pu_feat_matrix = problem.build_pu_feature_matrix()

        # Boundary: adjacency list + self-boundary
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
                        # Store both directions
                        neighbors[idx1].append((idx2, bval))
                        neighbors[idx2].append((idx1, bval))

        # Flatten neighbors to CSR-like arrays for fast indexing
        adj_indices_list = []
        adj_weights_list = []
        adj_start = np.zeros(n_pu + 1, dtype=np.int32)
        
        current_idx = 0
        for i in range(n_pu):
            adj_start[i] = current_idx
            # Sort neighbors by index for better memory access pattern
            # (though strictly not required for logic)
            nbs = sorted(neighbors[i], key=lambda x: x[0])
            for neighbor_idx, weight in nbs:
                adj_indices_list.append(neighbor_idx)
                adj_weights_list.append(weight)
                current_idx += 1
        adj_start[n_pu] = current_idx
        
        adj_indices = np.array(adj_indices_list, dtype=np.int32)
        adj_weights = np.array(adj_weights_list, dtype=np.float64)

        # Cached scalars
        params = problem.parameters
        misslevel = float(params.get("MISSLEVEL", 1.0))
        cost_thresh = float(params.get("COSTTHRESH", 0.0))
        thresh_pen1 = float(params.get("THRESHPEN1", 0.0))
        thresh_pen2 = float(params.get("THRESHPEN2", 0.0))

        # PROBMODE 3 state (zero-cost when probmode != 3)
        probmode = int(params.get("PROBMODE", 0))
        prob_weight = float(params.get("PROBABILITYWEIGHTING", 1.0))

        # Per-cell Bernoulli probability — read once into a dense matrix.
        # The expected and variance matrices are derived from amount + prob
        # so they can be summed via boolean indexing exactly like
        # pu_feat_matrix during full / delta objective evaluation.
        prob_matrix = np.zeros_like(pu_feat_matrix)
        if (
            problem.pu_vs_features is not None
            and "prob" in problem.pu_vs_features.columns
        ):
            pv_pu = problem.pu_vs_features["pu"].values
            pv_sp = problem.pu_vs_features["species"].values
            pv_pr = problem.pu_vs_features["prob"].values
            for k in range(len(pv_pu)):
                pi = pu_id_to_idx.get(int(pv_pu[k]))
                fj = feat_id_to_col.get(int(pv_sp[k]))
                if pi is not None and fj is not None:
                    prob_matrix[pi, fj] = float(pv_pr[k])

        # Marxan PROB2D math:
        #   E[T_j contribution from i] = amount_ij * (1 - p_ij)
        #   Var[T_j contribution from i] = amount_ij^2 * p_ij * (1 - p_ij)
        expected_matrix = pu_feat_matrix * (1.0 - prob_matrix)
        var_matrix = (pu_feat_matrix ** 2) * prob_matrix * (1.0 - prob_matrix)

        # Per-feature probability target; default -1 (disabled sentinel) for
        # legacy specs without a ptarget column (read_spec already fills this).
        if "ptarget" in feat_df.columns:
            feat_ptarget = np.asarray(feat_df["ptarget"].values, dtype=np.float64)
        else:
            feat_ptarget = np.full(n_feat, -1.0, dtype=np.float64)

        # Phase 19 clumping precompute (target2, clumptype, baseline_penalty,
        # feat_uses_pu). Defaults match read_spec's legacy fill: target2=0
        # disables clumping for the feature.
        if "target2" in feat_df.columns:
            feat_target2 = np.asarray(
                feat_df["target2"].values, dtype=np.float64,
            )
        else:
            feat_target2 = np.zeros(n_feat, dtype=np.float64)
        if "clumptype" in feat_df.columns:
            feat_clumptype = np.asarray(
                feat_df["clumptype"].values, dtype=np.int8,
            )
        else:
            feat_clumptype = np.zeros(n_feat, dtype=np.int8)
        clumping_active = bool(np.any(feat_target2 > 0))

        # Greedy "cost to meet target" baseline penalty per feature.
        # Computed lazily — only when clumping is active — to keep the
        # cold path zero-cost.
        if clumping_active:
            from pymarxan.solvers.clumping import compute_baseline_penalty
            feat_baseline_penalty = compute_baseline_penalty(problem)
        else:
            feat_baseline_penalty = np.zeros(n_feat, dtype=np.float64)

        # Sparse participation index per feature: PUs where amount > 0.
        # Built unconditionally because it's cheap and ClumpState relies
        # on it for both add and remove paths.
        feat_uses_pu: list[np.ndarray] = []
        for j in range(n_feat):
            participating = np.where(pu_feat_matrix[:, j] > 0)[0].astype(np.int32)
            feat_uses_pu.append(participating)

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
            adj_indices=adj_indices,
            adj_weights=adj_weights,
            adj_start=adj_start,
            self_boundary=self_boundary,
            misslevel=misslevel,
            cost_thresh=cost_thresh,
            thresh_pen1=thresh_pen1,
            thresh_pen2=thresh_pen2,
            probmode=probmode,
            prob_weight=prob_weight,
            expected_matrix=expected_matrix,
            var_matrix=var_matrix,
            feat_ptarget=feat_ptarget,
            feat_target2=feat_target2,
            feat_clumptype=feat_clumptype,
            feat_baseline_penalty=feat_baseline_penalty,
            feat_uses_pu=feat_uses_pu,
            clumping_active=clumping_active,
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
        result: np.ndarray = self.pu_feat_matrix[selected].sum(axis=0)
        return result

    def _compute_zscore_penalty(
        self, held_expected: np.ndarray, held_variance: np.ndarray,
    ) -> float:
        """Marxan-faithful Z-score chance-constraint penalty.

        Active only when ``self.probmode == 3``; returns 0 otherwise (the
        caller's hot path can skip this branch entirely via the probmode
        check).  Features with ptarget ≤ 0 are skipped.  Features with
        zero variance contribute 0 (Marxan's "degenerate → P=1").
        """
        if self.probmode != 3 or self.n_feat == 0:
            return 0.0
        from scipy.stats import norm  # local import keeps cold path cheap
        total = 0.0
        for j in range(self.n_feat):
            ptarget = float(self.feat_ptarget[j])
            if ptarget <= 0:
                continue
            var = float(held_variance[j])
            if var <= 0:
                # Degenerate / no uncertainty — deterministic path handles miss
                continue
            mean = float(held_expected[j])
            target = float(self.feat_targets[j])
            z = (target - mean) / np.sqrt(var)
            prob_met = float(norm.sf(z))
            shortfall = max(0.0, (ptarget - prob_met) / ptarget)
            total += float(self.feat_spf[j]) * shortfall
        return self.prob_weight * total

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

        # Penalty: SPF * max(0, target*misslevel - held) for each feature.
        # Type-4 features (target2 > 0) are EXCLUDED from this path because
        # their penalty uses the Marxan-faithful `baseline · SPF · fractional`
        # form on `held_eff` instead, computed externally via ClumpState.
        effective_targets = self.feat_targets * self.misslevel
        shortfalls = np.maximum(0.0, effective_targets - held)
        if self.clumping_active:
            det_spf = self.feat_spf * (self.feat_target2 <= 0)
        else:
            det_spf = self.feat_spf
        penalty = float(np.dot(det_spf, shortfalls))

        obj = total_cost + blm * boundary + penalty

        # Cost threshold penalty
        if self.cost_thresh > 0:
            obj += compute_cost_threshold_penalty(
                total_cost, self.cost_thresh, self.thresh_pen1, self.thresh_pen2
            )

        # PROBMODE 3 chance-constraint penalty (additive; no-op when probmode!=3)
        if self.probmode == 3:
            held_expected = self.expected_matrix[selected].sum(axis=0)
            held_variance = self.var_matrix[selected].sum(axis=0)
            obj += self._compute_zscore_penalty(held_expected, held_variance)

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
        # Get neighbors from CSR arrays
        start = self.adj_start[idx]
        end = self.adj_start[idx + 1]
        
        # Self-boundary: only contributes when PU is selected
        # (sign is +1 if adding, -1 if removing)
        boundary_delta = sign * self.self_boundary[idx]
        
        if start < end:
            nbs = self.adj_indices[start:end]
            weights = self.adj_weights[start:end]
            
            # Vectorized boundary calculation
            # If adding (sign=1):
            #   neighbor selected => -weight (connected, boundary removed)
            #   neighbor not selected => +weight (not connected, boundary added)
            #   multiplier = (1 if not sel else -1) = 1 - 2*sel
            # If removing (sign=-1):
            #   neighbor selected => +weight (was connected, now broken) 
            #   neighbor not selected => -weight (was isolated, now gone)
            #   multiplier = (1 if sel else -1) = 2*sel - 1
            
            # Combined multiplier logic:
            # multiplier = sign * (1 - 2 * selected[nbs])
            # But selected is bool, so need conversion
            
            neighbor_selected = selected[nbs]
            # Convert bool to float: True -> 1.0, False -> 0.0
            # multiplier = sign * (1.0 - 2.0 * neighbor_selected)
            
            # Using loop for clarity vs small array overhead:
            # Vectorized numpy op is faster for degree > ~5-10
            
            multipliers = sign * (1.0 - 2.0 * neighbor_selected.astype(np.float64))
            boundary_delta += np.dot(weights, multipliers)

        # --- Penalty delta ---
        # Only features present in this PU can change their shortfall.
        # Type-4 features (target2 > 0) are excluded — their penalty delta
        # is supplied externally by ClumpState.delta_penalty.
        pu_amounts = self.pu_feat_matrix[idx]
        effective_targets = self.feat_targets * self.misslevel
        old_shortfalls = np.maximum(0.0, effective_targets - held)
        new_held = held + sign * pu_amounts
        new_shortfalls = np.maximum(0.0, effective_targets - new_held)
        if self.clumping_active:
            det_spf = self.feat_spf * (self.feat_target2 <= 0)
        else:
            det_spf = self.feat_spf
        penalty_delta = float(
            np.dot(det_spf, new_shortfalls - old_shortfalls)
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

        # --- PROBMODE 3 chance-constraint delta ---
        # Recompute the Z-score penalty before and after the flip from the
        # full selection mask. This is O(n_feat) per flip — heavier than
        # the deterministic delta but cheap relative to the SA outer loop.
        # The future optimisation (sparse per-PU feature index) lands in a
        # later patch if benchmarks demand it.
        if self.probmode == 3:
            held_exp_before = self.expected_matrix[selected].sum(axis=0)
            held_var_before = self.var_matrix[selected].sum(axis=0)
            held_exp_after = held_exp_before + sign * self.expected_matrix[idx]
            held_var_after = held_var_before + sign * self.var_matrix[idx]
            # Underflow guard — held_var must be non-negative
            np.clip(held_var_after, 0.0, None, out=held_var_after)
            prob_before = self._compute_zscore_penalty(
                held_exp_before, held_var_before,
            )
            prob_after = self._compute_zscore_penalty(
                held_exp_after, held_var_after,
            )
            delta += prob_after - prob_before

        return float(delta)

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
        # Iterate only selected nodes
        selected_indices = np.where(selected)[0]
        
        for i in selected_indices:
            start = self.adj_start[i]
            end = self.adj_start[i + 1]
            if start == end:
                continue
                
            nbs = self.adj_indices[start:end]
            weights = self.adj_weights[start:end]
            
            # For each neighbor j of selected node i:
            # If j is NOT selected, we add weight (boundary exposed)
            # If j IS selected, we do nothing (internal edge) 
            # Note: The "double counting" issue in original code handled (i<j) check logic
            # implicitly by looping all edges?
            # Original code:
            # for j, w in self.neighbors[i]:
            #    if not selected[j]: total += w
            
            # Vectorized check for neighbors not selected
            # logic: sum(weights where not selected[nbs])
            
            mask_not_selected = ~selected[nbs]
            total += np.sum(weights[mask_not_selected])
            
        # WAIT: The above counts each boundary twice?
        # If i is selected and j is not: i adds boundary w.
        # If j is not selected, we don't iterate j. So only counted once.
        # Correct.
        
        return total
