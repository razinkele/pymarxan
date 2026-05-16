"""Ferrier importance score (Phase 22).

Per Ferrier et al. (2000) `Biological Conservation` 93(3): 303-325 the
importance of planning unit ``i`` is the SPF-weighted sum of its
contribution to meeting each feature's target. Higher = more important.

Formula (pymarxan implementation):

    ferrier_i = Σ_j SPF_j · min(amount_ij, target_j) / target_j

Properties:
- Closed form — no MIP re-solves required (cf. replacement_cost).
- Skips features with ``target_j <= 0`` (denominator would blow up).
- Clamps the per-feature contribution at ``1.0 · SPF_j`` so a single
  very-rich PU can't dominate the score.
- Locked-out PUs (status==3) score 0 — they can never be in a reserve.

Reference: Ferrier, Pressey & Barrett (2000). A new predictor of the
irreplaceability of areas for achieving a conservation goal, its
application to real-world planning, and a research agenda for further
refinement. *Biological Conservation* 93(3): 303-325.
https://doi.org/10.1016/S0006-3207(99)00149-4
"""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem


def compute_ferrier_importance(
    problem: ConservationProblem,
) -> dict[int, float]:
    """Compute the Ferrier-style importance score for each planning unit.

    Parameters
    ----------
    problem
        Conservation problem with ``features['target']`` and
        ``features['spf']`` columns. Locked-out PUs (status == 3) score 0.

    Returns
    -------
    dict[int, float]
        Mapping ``planning_unit_id -> score``. Scores are non-negative,
        unbounded above (proportional to SPF magnitudes).
    """
    pu_ids = problem.planning_units["id"].astype(int).to_numpy()
    statuses = np.asarray(
        problem.planning_units["status"].values, dtype=int,
    )
    feat_targets = np.asarray(
        problem.features["target"].values, dtype=np.float64,
    )
    feat_spf = np.asarray(
        problem.features["spf"].values, dtype=np.float64,
    )

    pu_feat_matrix = problem.build_pu_feature_matrix()
    locked_out = statuses == 3

    # Per-feature mask: only features with target > 0 contribute. Avoids
    # the divide-by-zero and matches the irreplaceability convention.
    target_positive = feat_targets > 0
    if not target_positive.any() or len(pu_ids) == 0:
        return {int(pid): 0.0 for pid in pu_ids}

    # Clamp amount at the target so a rich PU can't absorb all importance.
    # Broadcasting: (n_pu, n_feat) ≤ (n_feat,)
    safe_targets = np.where(target_positive, feat_targets, 1.0)
    clamped = np.minimum(pu_feat_matrix, feat_targets[np.newaxis, :])
    # Per-PU per-feature contribution: amount / target * SPF
    contrib = (clamped / safe_targets[np.newaxis, :]) * feat_spf[np.newaxis, :]
    # Zero out features that have zero target.
    contrib[:, ~target_positive] = 0.0

    per_pu_score = contrib.sum(axis=1)
    per_pu_score[locked_out] = 0.0

    return {int(pid): float(per_pu_score[i]) for i, pid in enumerate(pu_ids)}
