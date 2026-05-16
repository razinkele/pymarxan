"""Marxan-faithful clumping math (Phase 19 — TARGET2 / CLUMPTYPE).

Pure-functional helpers (`partial_pen4`, `compute_feature_components`,
`compute_baseline_penalty`, `compute_clump_penalty_from_scratch`,
`evaluate_solution_clumping`) plus the mutable `ClumpState` companion to
`ProblemCache` for the SA inner loop.

The math mirrors Marxan v4's ``clumping.cpp::PartialPen4`` and
``score_change.cpp::computeChangePenalty`` line-for-line, with the formulas
validated against the C++ source by the multi-agent design review.

Marxan's "type-4 species" formulation says a feature only counts toward its
target via *contiguous patches* of selected planning units whose summed amount
reaches ``target2``.  Sub-target patches contribute according to ``CLUMPTYPE``:

    if occ >= target2:    return occ                 # full credit
    CLUMPTYPE 0:          return 0                   # binary
    CLUMPTYPE 1:          return occ / 2             # "nicer step"
    CLUMPTYPE 2:          return occ² / target2      # graduated / quadratic

The penalty for feature ``j`` is then ``baseline_penalty_j · SPF_j ·
fractional_shortfall_j`` where ``fractional_shortfall_j = max(0,
(target_j · MISSLEVEL − held_eff_j) / target_j)``.  ``baseline_penalty_j`` is
the greedy "cost to meet target_j from cheapest PUs" — pymarxan's analogue of
Marxan's pre-computed ``spec.penalty``.

References
----------
- Marxan v4 source: ``clumping.cpp::PartialPen4``, ``score_change.cpp``.
- Ball, I. R., Possingham, H. P., & Watts, M. (2009). *Spatial Conservation
  Prioritization* (pp. 185–195). Oxford University Press.
  https://doi.org/10.1093/oso/9780199547760.003.0014
- Metcalfe, K. et al. (2015). *Conservation Biology* 29(6): 1615–1625.
  https://doi.org/10.1111/cobi.12571
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pymarxan.models.problem import ConservationProblem


def partial_pen4(occ: float, target2: float, clumptype: int) -> float:
    """Per-clump contribution to a feature's effective held amount.

    Direct port of ``clumping.cpp::PartialPen4`` — see module docstring for
    the source formulas. ``target2 <= 0`` returns 0 defensively (a feature
    with target2 disabled shouldn't reach this function in the first place,
    but the guard avoids a division-by-zero on CLUMPTYPE 2).
    """
    if target2 <= 0:
        return 0.0
    if occ >= target2:
        return float(occ)
    if clumptype == 0:
        return 0.0
    if clumptype == 1:
        return float(occ) / 2.0
    if clumptype == 2:
        return float(occ) * float(occ) / float(target2)
    return 0.0


def compute_feature_components(
    selected: np.ndarray,
    feat_amounts: np.ndarray,
    adj_indices: np.ndarray,
    adj_start: np.ndarray,
) -> list[np.ndarray]:
    """Connected components of (selected ∩ has_feature) under the adjacency.

    PUs are "participants" in this feature iff they are selected AND have
    a positive amount of the feature. PUs that lack the feature do NOT
    bridge components — matches Marxan's ``rtnClumpSpecAtPu`` convention.

    Parameters
    ----------
    selected
        (n_pu,) boolean selection mask.
    feat_amounts
        (n_pu,) feature amount per PU (the relevant column of
        ``pu_feat_matrix``).
    adj_indices, adj_start
        CSR-format adjacency from ``ProblemCache``. Edge weights are
        irrelevant for component detection.

    Returns
    -------
    list[np.ndarray]
        One ndarray of participant PU indices per component. Order
        within each array and order across the list are unspecified.
    """
    n_pu = len(selected)
    participants = selected & (feat_amounts > 0)
    if not participants.any():
        return []

    # Build a sub-graph CSR over participating PU indices only.
    # connected_components on the full graph works too if we use the full
    # CSR plus a mask, but scipy doesn't accept a node mask — easier to
    # construct an n_pu × n_pu binary CSR and mask its rows/cols.
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n_pu):
        if not participants[i]:
            continue
        start = adj_start[i]
        end = adj_start[i + 1]
        for k in range(start, end):
            j = int(adj_indices[k])
            if participants[j]:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=np.int8)
    graph = csr_matrix(
        (data, (rows, cols)), shape=(n_pu, n_pu),
    )
    n_components, labels = connected_components(
        csgraph=graph, directed=False, return_labels=True,
    )

    # Group participant PU indices by component label.
    comps: list[list[int]] = [[] for _ in range(n_components)]
    for i in range(n_pu):
        if participants[i]:
            comps[labels[i]].append(i)
    return [np.array(c, dtype=np.int32) for c in comps if c]


def compute_baseline_penalty(problem: ConservationProblem) -> np.ndarray:
    """Greedy "cost to meet target" per feature — pymarxan's analogue of
    Marxan's pre-computed ``spec.penalty`` constant.

    For each feature ``j`` with ``target_j > 0``: sort PUs by
    ``cost_i / amount_ij`` ascending (cheapest per-unit-amount first);
    accumulate amount until ``target_j`` is met; record the accumulated
    cost. Features with no positive target get a baseline of 0.

    Returns
    -------
    np.ndarray
        (n_feat,) float64 — baseline penalty per feature.
    """
    feat_ids = problem.features["id"].astype(int).values
    feat_target = problem.features["target"].astype(float).values
    n_feat = len(feat_ids)
    baselines = np.zeros(n_feat, dtype=np.float64)

    costs = problem.planning_units["cost"].astype(float).values
    pu_ids = problem.planning_units["id"].astype(int).values
    pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}

    # Group puvspr by species for easy iteration.
    puvspr = problem.pu_vs_features
    if len(puvspr) == 0:
        return baselines
    groups = puvspr.groupby("species")

    for j, fid in enumerate(feat_ids):
        target = float(feat_target[j])
        if target <= 0:
            continue
        if int(fid) not in groups.groups:
            # Feature has no PUs offering it — baseline equals target
            # itself (no way to make it cheap; serves as a "stiff" penalty).
            baselines[j] = target
            continue
        group = groups.get_group(int(fid))
        # cost-per-unit-amount, ascending. np.asarray narrows the pandas
        # `.values` union (ExtensionArray | ndarray) to a concrete ndarray
        # so downstream arithmetic + masking type-check.
        amounts = np.asarray(group["amount"].astype(float).values, dtype=np.float64)
        pus = np.asarray(group["pu"].astype(int).values, dtype=np.int64)
        # Lookup PU costs by id
        cost_per_pu = np.array([costs[pu_id_to_idx[int(p)]] for p in pus])
        # Filter out PUs that supply 0 of the feature (defensive)
        valid = amounts > 0
        if not valid.any():
            baselines[j] = target
            continue
        ratios = cost_per_pu[valid] / amounts[valid]
        order = np.argsort(ratios)
        amounts_sorted = amounts[valid][order]
        costs_sorted = cost_per_pu[valid][order]

        cum_amount = 0.0
        cum_cost = 0.0
        for amt, c in zip(amounts_sorted, costs_sorted):
            need = target - cum_amount
            if amt >= need:
                # Partial cost proportional to the fraction we actually need
                cum_cost += c * (need / amt)
                cum_amount = target
                break
            cum_cost += c
            cum_amount += amt
        baselines[j] = cum_cost

    return baselines


def _compute_held_eff_per_feature(
    problem: ConservationProblem,
    selected: np.ndarray,
    adj_indices: np.ndarray,
    adj_start: np.ndarray,
    pu_feat_matrix: np.ndarray,
) -> np.ndarray:
    """Per-feature effective held amount under clumping rules. Helper used
    by ``compute_clump_penalty_from_scratch`` and ``evaluate_solution_clumping``.
    """
    feat_ids = problem.features["id"].astype(int).values
    feat_target2 = problem.features["target2"].astype(float).values
    feat_clumptype = problem.features["clumptype"].astype(int).values
    n_feat = len(feat_ids)
    held_eff = np.zeros(n_feat, dtype=np.float64)

    for j in range(n_feat):
        target2 = float(feat_target2[j])
        if target2 <= 0:
            # No clumping: held_eff = raw sum
            held_eff[j] = float(pu_feat_matrix[selected, j].sum())
            continue
        clumptype = int(feat_clumptype[j])
        comps = compute_feature_components(
            selected, pu_feat_matrix[:, j], adj_indices, adj_start,
        )
        total = 0.0
        for comp in comps:
            occ = float(pu_feat_matrix[comp, j].sum())
            total += partial_pen4(occ, target2, clumptype)
        held_eff[j] = total

    return held_eff


def compute_clump_penalty_from_scratch(
    problem: ConservationProblem,
    selected: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Reference impl of the type-4 penalty path.

    Used in tests and as the verification baseline for
    ``ClumpState.delta_penalty`` (Batch 3). NOT used in the SA inner
    loop. Cost: O(edges + n_features × n_components).

    Returns
    -------
    (held_eff, penalty)
        ``held_eff`` is (n_feat,) effective held under CLUMPTYPE rules;
        ``penalty`` is ``Σ_j baseline_penalty_j · SPF_j · fractional_shortfall_j``
        with ``fractional_shortfall_j = max(0, (T·MISSLEVEL − held_eff)/T)``.
    """
    if problem.boundary is None or len(problem.boundary) == 0:
        # No adjacency → every selected PU is its own clump. The component
        # detection still works (yields singletons) but we can shortcut
        # the matrix build cost. For simplicity, just call the general
        # path with an empty adjacency.
        adj_indices = np.zeros(0, dtype=np.int32)
        adj_start = np.zeros(len(problem.planning_units) + 1, dtype=np.int32)
    else:
        # Build CSR adjacency from boundary DataFrame. (ProblemCache does
        # this already; we duplicate it here so the from-scratch reference
        # impl works without the cache.)
        n_pu = len(problem.planning_units)
        pu_ids = problem.planning_units["id"].astype(int).values
        pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
        b = problem.boundary
        edges: list[tuple[int, int]] = []
        for k in range(len(b)):
            i1 = int(b["id1"].iloc[k])
            i2 = int(b["id2"].iloc[k])
            if i1 == i2:
                continue
            idx1 = pu_id_to_idx.get(i1)
            idx2 = pu_id_to_idx.get(i2)
            if idx1 is not None and idx2 is not None:
                edges.append((idx1, idx2))
                edges.append((idx2, idx1))
        edges.sort()
        adj_start = np.zeros(n_pu + 1, dtype=np.int32)
        adj_list: list[int] = []
        cur = 0
        for i in range(n_pu):
            adj_start[i] = cur
            for src, dst in edges:
                if src == i:
                    adj_list.append(dst)
                    cur += 1
        adj_start[n_pu] = cur
        adj_indices = np.array(adj_list, dtype=np.int32)

    pu_feat_matrix = problem.build_pu_feature_matrix()
    held_eff = _compute_held_eff_per_feature(
        problem, selected, adj_indices, adj_start, pu_feat_matrix,
    )

    # Marxan-faithful fractional shortfall × baseline_penalty × SPF
    feat_target = problem.features["target"].astype(float).values
    feat_spf = problem.features["spf"].astype(float).values
    feat_target2 = problem.features["target2"].astype(float).values
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    baselines = compute_baseline_penalty(problem)

    penalty = 0.0
    n_feat = len(feat_target)
    for j in range(n_feat):
        if feat_target2[j] <= 0:
            # Non-type-4 features are handled by the existing deterministic
            # path; this reference impl returns 0 contribution for them.
            continue
        target = float(feat_target[j])
        if target <= 0:
            continue
        fractional = max(0.0, (target * misslevel - held_eff[j]) / target)
        penalty += float(baselines[j]) * float(feat_spf[j]) * fractional

    return held_eff, float(penalty)


def evaluate_solution_clumping(
    problem: ConservationProblem,
    selected: np.ndarray,
) -> tuple[dict[int, float], float]:
    """Post-hoc clumping evaluator for the MIP "drop" strategy and for
    populating ``Solution.clump_shortfalls`` / ``clump_penalty``.

    The shortfall reported is the **raw** shortfall ``(T·MISSLEVEL − held_eff)``
    (clamped at 0), in the same units as the amounts — easier for users to
    interpret in the UI. The ``penalty`` value is the same Marxan-faithful
    ``baseline · SPF · fractional`` that the in-objective path uses.

    Returns
    -------
    (shortfalls, penalty)
        ``shortfalls`` is a dict keyed by feature_id, only including
        features with ``target2 > 0`` (others have no clumping shortfall
        by definition). ``penalty`` is the weighted total.
    """
    feat_ids = problem.features["id"].astype(int).values
    feat_target = problem.features["target"].astype(float).values
    feat_target2 = np.asarray(
        problem.features["target2"].astype(float).values, dtype=np.float64,
    )
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))

    # Quick-exit when no feature is type-4
    if not (feat_target2 > 0).any():
        return {}, 0.0

    held_eff, penalty = compute_clump_penalty_from_scratch(problem, selected)

    shortfalls: dict[int, float] = {}
    for j, fid in enumerate(feat_ids):
        if feat_target2[j] <= 0:
            continue
        target = float(feat_target[j])
        raw_shortfall = max(0.0, target * misslevel - float(held_eff[j]))
        shortfalls[int(fid)] = raw_shortfall

    return shortfalls, penalty


# ----------------------------------------------------------------------
# ClumpState — mutable companion to ProblemCache for the SA inner loop
# ----------------------------------------------------------------------


def _build_components_for_feature(
    cache, selected: np.ndarray, j: int,
) -> tuple[np.ndarray, dict[int, float]]:
    """Build (comp_id_per_pu, comp_occ) for feature j under `selected`.

    Uses the cache's CSR adjacency, restricted to PUs that both have the
    feature and are selected. PUs that lack the feature do NOT bridge
    components (Marxan ``rtnClumpSpecAtPu`` convention).

    Returns
    -------
    comp_id_per_pu
        (n_pu,) int32 — component id for each PU, or -1 if the PU is not
        a participant.
    comp_occ
        dict mapping component id → total occupancy of the feature in that
        component (Σ amount_ij over participating PUs in the component).
    """
    n_pu = cache.n_pu
    feat_amounts = cache.pu_feat_matrix[:, j]
    components = compute_feature_components(
        selected, feat_amounts, cache.adj_indices, cache.adj_start,
    )
    comp_id_per_pu = np.full(n_pu, -1, dtype=np.int32)
    comp_occ: dict[int, float] = {}
    for cid, comp in enumerate(components):
        comp_id_per_pu[comp] = cid
        comp_occ[cid] = float(feat_amounts[comp].sum())
    return comp_id_per_pu, comp_occ


def _held_eff_for_feature(
    cache, j: int, comp_occ: dict[int, float],
) -> float:
    """Σ partial_pen4(occ, target2_j, clumptype_j) over components of feature j."""
    target2 = float(cache.feat_target2[j])
    clumptype = int(cache.feat_clumptype[j])
    if target2 <= 0:
        return 0.0  # non-clumping path uses raw amount sum elsewhere
    return sum(
        partial_pen4(occ, target2, clumptype) for occ in comp_occ.values()
    )


class ClumpState:
    """Mutable companion to ``ProblemCache`` for the SA / iterative-improvement
    inner loop. Maintains per-feature connected-component bookkeeping and the
    effective held amount under clumping rules.

    The v1 implementation does a per-affected-feature full recompute on each
    flip (using ``scipy.sparse.csgraph.connected_components`` on the
    participant subgraph). Cost per flip: ``O(n_features_with_target2 ·
    (edges_in_participant_subgraph + n_components))``. The fully-incremental
    union-find / bounded-local-BFS optimisation from the architect review is
    an opt-in future patch — adopted only if benchmarks show the v1
    full-recompute is too slow on a realistic 1k+-PU problem.
    """

    def __init__(
        self,
        selected: np.ndarray,
        comp_id_per_pu: list[np.ndarray],
        comp_occ: list[dict[int, float]],
        held_eff: np.ndarray,
    ) -> None:
        self.selected = selected
        self.comp_id_per_pu = comp_id_per_pu
        self.comp_occ = comp_occ
        self.held_eff = held_eff

    @classmethod
    def from_selection(cls, cache, selected: np.ndarray) -> ClumpState:
        """One-time full build. Called once before the SA loop starts."""
        n_feat = cache.n_feat
        comp_id_per_pu: list[np.ndarray] = []
        comp_occ: list[dict[int, float]] = []
        held_eff = np.zeros(n_feat, dtype=np.float64)
        for j in range(n_feat):
            if float(cache.feat_target2[j]) > 0:
                ids, occs = _build_components_for_feature(cache, selected, j)
                comp_id_per_pu.append(ids)
                comp_occ.append(occs)
                held_eff[j] = _held_eff_for_feature(cache, j, occs)
            else:
                # Non-clumping feature: held_eff is raw amount sum so callers
                # can treat held_effective() as a single source of truth.
                comp_id_per_pu.append(np.full(cache.n_pu, -1, dtype=np.int32))
                comp_occ.append({})
                held_eff[j] = float(
                    cache.pu_feat_matrix[selected, j].sum()
                )
        return cls(
            selected=selected.copy(),
            comp_id_per_pu=comp_id_per_pu,
            comp_occ=comp_occ,
            held_eff=held_eff,
        )

    def held_effective(self) -> np.ndarray:
        """Return the current per-feature effective held amount."""
        return self.held_eff

    def delta_penalty(self, cache, idx: int, adding: bool) -> float:
        """Δ(clumping penalty) for flipping PU ``idx``. Does NOT mutate state.

        Only features with ``target2_j > 0`` AND ``amount_ij > 0`` are
        recomputed — others can't be affected by the flip.
        """
        n_feat = cache.n_feat
        # Synthesise the post-flip selection without mutating self.selected
        sel_after = self.selected.copy()
        sel_after[idx] = adding
        misslevel = float(cache.misslevel)
        total_delta = 0.0
        for j in range(n_feat):
            target2 = float(cache.feat_target2[j])
            if target2 <= 0:
                continue
            target = float(cache.feat_targets[j])
            if target <= 0:
                continue
            # Cheap path: if PU idx doesn't carry this feature, the flip
            # can't change the feature's clumps — skip.
            if cache.pu_feat_matrix[idx, j] == 0:
                continue
            _, new_occs = _build_components_for_feature(cache, sel_after, j)
            new_held_j = _held_eff_for_feature(cache, j, new_occs)
            old_held_j = float(self.held_eff[j])
            old_frac = max(
                0.0, (target * misslevel - old_held_j) / target,
            )
            new_frac = max(
                0.0, (target * misslevel - new_held_j) / target,
            )
            total_delta += (
                float(cache.feat_baseline_penalty[j])
                * float(cache.feat_spf[j])
                * (new_frac - old_frac)
            )
        return total_delta

    def apply_flip(self, cache, idx: int, adding: bool) -> None:
        """Commit the flip: update selected mask, component bookkeeping, and
        held_eff for all affected features.

        Non-clumping features only need their raw-sum cached value updated
        when ``amount_ij != 0``; clumping features go through the full
        component rebuild for now.
        """
        sign = 1.0 if adding else -1.0
        self.selected[idx] = adding
        for j in range(cache.n_feat):
            amt = float(cache.pu_feat_matrix[idx, j])
            target2 = float(cache.feat_target2[j])
            if target2 <= 0:
                # Non-clumping: keep held_eff as the raw selected-amount sum
                # so callers can compare against compute_held().
                if amt != 0:
                    self.held_eff[j] += sign * amt
                continue
            if amt == 0:
                continue  # flip doesn't touch feature j's clumps
            ids, occs = _build_components_for_feature(cache, self.selected, j)
            self.comp_id_per_pu[j] = ids
            self.comp_occ[j] = occs
            self.held_eff[j] = _held_eff_for_feature(cache, j, occs)
