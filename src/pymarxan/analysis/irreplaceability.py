"""Irreplaceability analysis for conservation planning.

Computes how irreplaceable each planning unit is based on its contribution
to meeting conservation targets. A PU is fully irreplaceable (1.0) if
removing it makes a target unachievable.
"""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem


def compute_irreplaceability(
    problem: ConservationProblem,
) -> dict[int, float]:
    """Compute irreplaceability score for each planning unit.

    Score is the fraction of features for which this PU is critical
    (i.e., removing it would make the target unachievable from remaining PUs).

    Vectorized implementation using a PU-feature matrix.
    """
    pu_ids = problem.planning_units["id"].values
    feat_ids = problem.features["id"].values
    feat_targets = problem.features["target"].values.astype(np.float64)

    n_pu = len(pu_ids)
    n_feat = len(feat_ids)

    pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
    feat_id_to_col = {int(fid): j for j, fid in enumerate(feat_ids)}

    # Build PU-feature amount matrix using shared utility
    pu_feat_matrix = problem.build_pu_feature_matrix()

    # Total amount per feature: shape (n_feat,)
    total_per_feat = pu_feat_matrix.sum(axis=0)

    # Only features with positive targets
    positive_mask = feat_targets > 0
    n_positive_target = int(positive_mask.sum())

    if n_positive_target == 0:
        return {int(pid): 0.0 for pid in pu_ids}

    # For each PU, compute remaining = total - pu_contribution per feature
    # A PU is critical for feature j if remaining[j] < target[j]
    # remaining[i, j] = total[j] - pu_feat_matrix[i, j]
    # critical[i, j] = (remaining[i, j] < target[j]) AND (target[j] > 0)
    remaining = total_per_feat[np.newaxis, :] - pu_feat_matrix  # (n_pu, n_feat)
    critical = (remaining < feat_targets[np.newaxis, :]) & positive_mask[np.newaxis, :]
    critical_counts = critical.sum(axis=1)  # (n_pu,)

    scores: dict[int, float] = {}
    for i, pid in enumerate(pu_ids):
        scores[int(pid)] = float(critical_counts[i]) / n_positive_target

    return scores
