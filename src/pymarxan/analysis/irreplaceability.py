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
    feat_targets = problem.features["target"].values.astype(np.float64)
    statuses = problem.planning_units["status"].values.astype(int)

    # Build PU-feature amount matrix using shared utility
    pu_feat_matrix = problem.build_pu_feature_matrix()

    # Locked-out PUs can never be selected, so they don't contribute to
    # available amount, and they themselves are never irreplaceable.
    locked_out = statuses == 3
    available_matrix = pu_feat_matrix.copy()
    available_matrix[locked_out, :] = 0.0

    # Total available amount per feature (excluding locked-out PUs)
    total_per_feat = available_matrix.sum(axis=0)

    # Apply MISSLEVEL — match solver/build_solution/export_summary behaviour
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    effective_targets = feat_targets * misslevel

    # Only features with positive targets count toward the score denominator
    positive_mask = feat_targets > 0
    n_positive_target = int(positive_mask.sum())

    if n_positive_target == 0:
        return {int(pid): 0.0 for pid in pu_ids}

    # For each selectable PU, compute remaining = total - pu_contribution per feature
    # critical[i, j] = (remaining[i, j] < effective_target[j]) AND target[j] > 0
    remaining = total_per_feat[np.newaxis, :] - available_matrix
    critical = (remaining < effective_targets[np.newaxis, :]) & positive_mask[np.newaxis, :]
    critical_counts = critical.sum(axis=1)
    # Locked-out PUs cannot be critical
    critical_counts[locked_out] = 0

    scores: dict[int, float] = {}
    for i, pid in enumerate(pu_ids):
        scores[int(pid)] = float(critical_counts[i]) / n_positive_target

    return scores
