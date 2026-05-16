"""Shared utility functions for conservation planning solvers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


def _compute_achieved(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
) -> dict[int, float]:
    """Vectorized computation of achieved amount per feature for selected PUs.

    Returns a dict mapping feature id -> total achieved amount.
    """
    puvspr = problem.pu_vs_features
    pu_col = puvspr["pu"].values
    species_col = puvspr["species"].values
    amount_col = puvspr["amount"].values

    # Build boolean mask: True for rows whose PU is selected
    pu_indices = np.array(
        [pu_index.get(int(pid), -1) for pid in pu_col], dtype=np.intp
    )
    valid = pu_indices >= 0
    # Safe indexing: use 0 for invalid indices, then mask out
    safe_indices = np.where(valid, pu_indices, 0)
    sel_mask = valid & selected[safe_indices]

    # Use only selected rows and groupby species
    selected_amounts = pd.DataFrame({
        "species": species_col[sel_mask],
        "amount": amount_col[sel_mask],
    })
    if selected_amounts.empty:
        return {int(fid): 0.0 for fid in problem.features["id"]}

    totals = selected_amounts.groupby("species")["amount"].sum()
    result = {int(fid): 0.0 for fid in problem.features["id"]}
    for fid, amount in totals.items():
        # groupby keys come back typed as Hashable; the species column is int
        result[int(fid)] = float(amount)  # type: ignore[call-overload]
    return result


def compute_boundary(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
) -> float:
    """Compute total boundary length for a given selection.

    Diagonal entries (id1==id2) represent external boundary, added when PU is selected.
    Off-diagonal entries represent shared boundary, added when exactly one PU is selected.
    """
    if problem.boundary is None:
        return 0.0

    bnd = problem.boundary
    id1_col = bnd["id1"].values
    id2_col = bnd["id2"].values
    bval_col = bnd["boundary"].values.astype(np.float64)

    total = 0.0
    for i in range(len(id1_col)):
        id1 = int(id1_col[i])
        id2 = int(id2_col[i])
        bval = float(bval_col[i])

        if id1 == id2:
            idx = pu_index.get(id1)
            if idx is not None and selected[idx]:
                total += bval
        else:
            idx1 = pu_index.get(id1)
            idx2 = pu_index.get(id2)
            if idx1 is not None and idx2 is not None:
                if selected[idx1] != selected[idx2]:
                    total += bval
    return total


def check_targets(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
    *,
    _achieved: dict[int, float] | None = None,
) -> dict[int, bool]:
    """Check which feature targets are met by the selection.

    Uses the MISSLEVEL parameter (default 1.0) to allow fractional
    target achievement.  A target is considered met when the achieved
    amount >= target * misslevel.
    """
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    achieved = (
        _achieved if _achieved is not None
        else _compute_achieved(problem, selected, pu_index)
    )

    targets_met: dict[int, bool] = {}
    feat_ids = problem.features["id"].values
    feat_targets = problem.features["target"].values
    for i in range(len(feat_ids)):
        fid = int(feat_ids[i])
        target = float(feat_targets[i])
        targets_met[fid] = achieved.get(fid, 0.0) >= target * misslevel
    return targets_met


def compute_feature_shortfalls(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
    *,
    _achieved: dict[int, float] | None = None,
) -> dict[int, float]:
    """Compute the shortfall for each feature (target*misslevel - achieved, min 0)."""
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    achieved = (
        _achieved if _achieved is not None
        else _compute_achieved(problem, selected, pu_index)
    )

    shortfalls: dict[int, float] = {}
    feat_ids = problem.features["id"].values
    feat_targets = problem.features["target"].values
    for i in range(len(feat_ids)):
        fid = int(feat_ids[i])
        target = float(feat_targets[i]) * misslevel
        shortfalls[fid] = max(0.0, target - achieved.get(fid, 0.0))
    return shortfalls


def compute_penalty(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
    *,
    _achieved: dict[int, float] | None = None,
) -> float:
    """Compute the total feature penalty (SPF * shortfall for each feature)."""
    shortfalls = compute_feature_shortfalls(problem, selected, pu_index, _achieved=_achieved)

    total = 0.0
    feat_ids = problem.features["id"].values
    feat_spf = (
        problem.features["spf"].values
        if "spf" in problem.features.columns
        else np.ones(len(feat_ids))
    )
    for i in range(len(feat_ids)):
        fid = int(feat_ids[i])
        spf = float(feat_spf[i])
        total += spf * shortfalls.get(fid, 0.0)
    return total


def compute_cost_threshold_penalty(
    total_cost: float,
    cost_thresh: float,
    thresh_pen1: float,
    thresh_pen2: float,
) -> float:
    """Compute penalty for exceeding a cost threshold.

    Parameters
    ----------
    total_cost : float
        Total cost of the current selection.
    cost_thresh : float
        Cost threshold above which a penalty applies.
    thresh_pen1 : float
        Fixed penalty applied when cost exceeds threshold.
    thresh_pen2 : float
        Multiplier for the amount by which cost exceeds threshold.

    Returns
    -------
    float
        ``0.0`` if *total_cost* <= *cost_thresh*, otherwise
        ``thresh_pen1 + thresh_pen2 * (total_cost - cost_thresh)``.
    """
    if total_cost <= cost_thresh:
        return 0.0
    return thresh_pen1 + thresh_pen2 * (total_cost - cost_thresh)


def compute_connectivity_penalty(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
) -> float:
    """Compute connectivity penalty for selected planning units.

    Connectivity penalizes *loss* of connections between selected PUs.
    For each edge (i,j) with value v where both i and j are selected,
    v contributes negatively (bonus). Where only one is selected,
    v contributes positively (penalty for broken connection).

    Total = CONNECTIVITY_WEIGHT * Σ edges where exactly one PU selected * value
          - CONNECTIVITY_WEIGHT * Σ edges where both PUs selected * value

    If ASYMMETRIC_CONNECTIVITY is set, directionality is preserved.

    Returns 0.0 if no connectivity data.
    """
    if problem.connectivity is None:
        return 0.0

    conn_weight = float(
        problem.parameters.get("CONNECTIVITY_WEIGHT", 0.0)
    )
    if conn_weight == 0.0:
        return 0.0

    conn = problem.connectivity
    id1_col = conn["id1"].values
    id2_col = conn["id2"].values
    val_col = conn["value"].values.astype(np.float64)

    total = 0.0
    for k in range(len(id1_col)):
        idx1 = pu_index.get(int(id1_col[k]))
        idx2 = pu_index.get(int(id2_col[k]))
        if idx1 is None or idx2 is None:
            continue
        val = float(val_col[k])
        s1 = selected[idx1]
        s2 = selected[idx2]
        if s1 and s2:
            total -= val  # bonus for connected pair
        elif s1 or s2:
            total += val  # penalty for broken connection
    return conn_weight * total


def compute_probability_penalty(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
) -> float:
    """Dispatch on ``PROBMODE`` to the appropriate probability penalty.

    - ``PROBMODE == 0``: no probability term, returns 0.
    - ``PROBMODE == 1``: Marxan risk premium
      ``γ · Σ prob_i · cost_i · x_i`` (per-PU PROB1D).
    - ``PROBMODE == 2``: persistence-adjusted amounts (handled upstream
      via amount substitution); returns 0 here.
    - ``PROBMODE == 3``: Marxan-faithful Z-score chance-constraint
      penalty using per-cell ``prob`` and per-feature ``ptarget``.
    """
    prob_mode = int(problem.parameters.get("PROBMODE", 1))
    if prob_mode == 1:
        return _compute_probmode1_risk_premium(problem, selected)
    if prob_mode == 3:
        return _compute_probmode3_zscore_penalty(problem, selected)
    return 0.0


def _compute_probmode1_risk_premium(
    problem: ConservationProblem,
    selected: np.ndarray,
) -> float:
    """PROBMODE 1: γ · Σ_i x_i · prob_i · cost_i (per-PU risk premium)."""
    if problem.probability is None:
        return 0.0

    prob_weight = float(problem.parameters.get("PROBABILITYWEIGHTING", 1.0))
    if prob_weight == 0.0:
        return 0.0

    prob_pu = problem.probability["pu"].values
    prob_val = problem.probability["probability"].values
    prob_map: dict[int, float] = {}
    for k in range(len(prob_pu)):
        prob_map[int(prob_pu[k])] = float(prob_val[k])

    costs = problem.planning_units["cost"].values
    pu_ids = problem.planning_units["id"].values

    total = 0.0
    for i in range(len(pu_ids)):
        if selected[i]:
            pid = int(pu_ids[i])
            prob = prob_map.get(pid, 0.0)
            total += prob * float(costs[i])
    return prob_weight * total


def _compute_probmode3_zscore_penalty(
    problem: ConservationProblem,
    selected: np.ndarray,
) -> float:
    """PROBMODE 3: Marxan-faithful Z-score chance-constraint penalty.

    Computes E[T_j] and Var[T_j] from the per-cell ``prob`` column on
    ``pu_vs_features`` (Marxan PROB2D wire format):

        E[T_j]   = Σ_i x_i · amount_ij · (1 − p_ij)
        Var[T_j] = Σ_i x_i · amount_ij² · p_ij · (1 − p_ij)

    Then routes to ``compute_zscore_per_feature`` and
    ``compute_zscore_penalty`` which implement the Marxan sign convention
    ``Z = (T − E)/sqrt(V)`` and ptarget-normalised shortfall.
    """
    # Lazy import: keep solvers.utils free of scipy unless mode 3 is used.
    from pymarxan.solvers.probability import (
        compute_zscore_penalty,
        compute_zscore_per_feature,
    )

    prob_weight = float(problem.parameters.get("PROBABILITYWEIGHTING", 1.0))
    if prob_weight == 0.0:
        return 0.0

    # Build per-feature expected and variance from selected PUs.
    pu_ids = problem.planning_units["id"].values
    selected_pu_ids = {int(pu_ids[i]) for i in range(len(pu_ids)) if selected[i]}
    if not selected_pu_ids:
        # Empty reserve: E=0, Var=0 -> sentinel Z; penalty zero anyway
        # for features whose ptarget≤0, otherwise norm.sf(4) ≈ 0 < ptarget
        # which produces a maximal shortfall. That's the right behaviour
        # for an empty reserve under chance constraints.
        pass

    puvspr = problem.pu_vs_features
    has_prob = "prob" in puvspr.columns

    achieved_mean: dict[int, float] = {}
    achieved_variance: dict[int, float] = {}
    if len(puvspr) > 0:
        pv_pu = puvspr["pu"].values
        pv_sp = puvspr["species"].values
        pv_am = puvspr["amount"].values
        pv_pr = puvspr["prob"].values if has_prob else None
        for k in range(len(pv_pu)):
            pid = int(pv_pu[k])
            if pid not in selected_pu_ids:
                continue
            fid = int(pv_sp[k])
            amount = float(pv_am[k])
            p = float(pv_pr[k]) if pv_pr is not None else 0.0
            achieved_mean[fid] = achieved_mean.get(fid, 0.0) + amount * (1.0 - p)
            achieved_variance[fid] = (
                achieved_variance.get(fid, 0.0) + (amount ** 2) * p * (1.0 - p)
            )

    # Feature-side data
    feat_ids = problem.features["id"].astype(int).values
    feat_target = problem.features["target"].astype(float).values
    feat_spf = problem.features["spf"].astype(float).values
    if "ptarget" in problem.features.columns:
        feat_ptarget = problem.features["ptarget"].astype(float).values
    else:
        # Legacy spec without ptarget -> all disabled
        return 0.0

    # Skip work entirely if every feature is disabled
    if not any(pt > 0 for pt in feat_ptarget):
        return 0.0

    targets = {int(feat_ids[j]): float(feat_target[j]) for j in range(len(feat_ids))}
    spf = {int(feat_ids[j]): float(feat_spf[j]) for j in range(len(feat_ids))}
    prob_targets = {
        int(feat_ids[j]): float(feat_ptarget[j]) for j in range(len(feat_ids))
    }

    # Features absent from achieved_* (no selected PU contributed) get
    # E=0, Var=0 -> Marxan sentinel Z=4 -> P≈0 -> max shortfall.
    for fid in targets:
        achieved_mean.setdefault(fid, 0.0)
        achieved_variance.setdefault(fid, 0.0)

    z_per_feat = compute_zscore_per_feature(achieved_mean, achieved_variance, targets)
    return compute_zscore_penalty(z_per_feat, prob_targets, spf, weight=prob_weight)


def compute_objective_terms(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
    blm: float,
    *,
    _achieved: dict[int, float] | None = None,
) -> dict[str, float]:
    """Compute all objective terms and return as a dict.

    Terms included:
      - ``"base"``: total cost (MinSet semantics)
      - ``"boundary"``: BLM * total boundary length
      - ``"penalty"``: SPF * shortfall per feature
      - ``"cost_threshold"``: penalty for exceeding cost threshold
      - ``"probability"``: Mode 1 risk premium (PROBABILITYWEIGHTING)
      - ``"connectivity"``: connectivity penalty/bonus (CONNECTIVITY_WEIGHT)

    All terms use lower-is-better convention.  The ``"objective"`` key
    holds the sum of all terms.

    Returns
    -------
    dict[str, float]
        Keys include ``"base"``, ``"boundary"``, ``"penalty"``,
        ``"cost_threshold"``, ``"probability"``, and ``"objective"``.
    """
    costs = np.asarray(problem.planning_units["cost"].values)
    total_cost = float(np.sum(costs[selected]))
    total_boundary = compute_boundary(problem, selected, pu_index)
    achieved = _achieved if _achieved is not None else _compute_achieved(
        problem, selected, pu_index,
    )
    penalty = compute_penalty(problem, selected, pu_index, _achieved=achieved)

    terms: dict[str, float] = {
        "base": total_cost,
        "boundary": blm * total_boundary,
        "penalty": penalty,
    }

    cost_thresh = float(problem.parameters.get("COSTTHRESH", 0.0))
    if cost_thresh > 0:
        thresh_pen1 = float(problem.parameters.get("THRESHPEN1", 0.0))
        thresh_pen2 = float(problem.parameters.get("THRESHPEN2", 0.0))
        terms["cost_threshold"] = compute_cost_threshold_penalty(
            total_cost, cost_thresh, thresh_pen1, thresh_pen2,
        )
    else:
        terms["cost_threshold"] = 0.0

    # Probability risk premium (Mode 1 only)
    terms["probability"] = compute_probability_penalty(
        problem, selected, pu_index,
    )

    # Connectivity penalty/bonus
    terms["connectivity"] = compute_connectivity_penalty(
        problem, selected, pu_index,
    )

    terms["objective"] = sum(terms.values())
    return terms


def compute_objective(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
    blm: float,
) -> float:
    """Compute the full Marxan objective: cost + BLM*boundary + penalty.

    When COSTTHRESH > 0, an additional cost-threshold penalty is added.
    """
    return compute_objective_terms(problem, selected, pu_index, blm)["objective"]


def build_solution(
    problem: ConservationProblem,
    selected: np.ndarray,
    blm: float,
    metadata: dict | None = None,
) -> Solution:
    """Build a complete Solution from a selection array."""
    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}

    costs = np.asarray(problem.planning_units["cost"].values)
    total_cost = float(np.sum(costs[selected]))
    total_boundary = compute_boundary(problem, selected, pu_index)

    # Compute achieved amounts once, pass to all downstream functions
    achieved = _compute_achieved(problem, selected, pu_index)
    targets_met = check_targets(problem, selected, pu_index, _achieved=achieved)
    shortfalls = compute_feature_shortfalls(problem, selected, pu_index, _achieved=achieved)
    total_shortfall = sum(shortfalls.values())

    terms = compute_objective_terms(
        problem, selected, pu_index, blm, _achieved=achieved,
    )

    # PROBMODE 3: populate per-feature Z-score gap on the Solution so the
    # UI can display it. The penalty itself is already part of terms["objective"]
    # via compute_probability_penalty.
    prob_shortfalls: dict[int, float] | None = None
    prob_penalty_val: float | None = None
    if int(problem.parameters.get("PROBMODE", 0)) == 3:
        from pymarxan.solvers.probability import evaluate_solution_chance
        prob_shortfalls, prob_penalty_val = evaluate_solution_chance(
            problem, selected,
        )

    # Phase 19 TARGET2 / clumping: populate per-feature clump shortfall gap
    # and total penalty when any feature has target2 > 0. Same shape as
    # PROBMODE 3 — single source of truth, every solver path inherits.
    clump_shortfalls: dict[int, float] | None = None
    clump_penalty_val: float | None = None
    if (
        "target2" in problem.features.columns
        and (problem.features["target2"] > 0).any()
    ):
        from pymarxan.solvers.clumping import evaluate_solution_clumping
        clump_shortfalls, clump_penalty_val = evaluate_solution_clumping(
            problem, selected,
        )

    return Solution(
        selected=selected.copy(),
        cost=total_cost,
        boundary=total_boundary,
        objective=terms["objective"],
        targets_met=targets_met,
        penalty=terms["penalty"],
        shortfall=total_shortfall,
        metadata=metadata or {},
        prob_shortfalls=prob_shortfalls,
        prob_penalty=prob_penalty_val,
        clump_shortfalls=clump_shortfalls,
        clump_penalty=clump_penalty_val,
    )
