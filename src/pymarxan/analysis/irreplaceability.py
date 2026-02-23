"""Irreplaceability analysis for conservation planning.

Computes how irreplaceable each planning unit is based on its contribution
to meeting conservation targets. A PU is fully irreplaceable (1.0) if
removing it makes a target unachievable.
"""
from __future__ import annotations

from pymarxan.models.problem import ConservationProblem


def compute_irreplaceability(
    problem: ConservationProblem,
) -> dict[int, float]:
    """Compute irreplaceability score for each planning unit.

    Score is the fraction of features for which this PU is critical
    (i.e., removing it would make the target unachievable from remaining PUs).
    """
    pu_ids = problem.planning_units["id"].tolist()
    feature_totals = problem.feature_amounts()

    pu_contributions: dict[int, dict[int, float]] = {pid: {} for pid in pu_ids}
    for _, row in problem.pu_vs_features.iterrows():
        pid = int(row["pu"])
        fid = int(row["species"])
        amount = float(row["amount"])
        pu_contributions[pid][fid] = amount

    n_positive_target = sum(
        1 for _, r in problem.features.iterrows() if float(r.get("target", 0.0)) > 0
    )
    scores: dict[int, float] = {}

    for pid in pu_ids:
        critical_count = 0
        contributions = pu_contributions.get(pid, {})

        for _, feat_row in problem.features.iterrows():
            fid = int(feat_row["id"])
            target = float(feat_row.get("target", 0.0))
            if target <= 0:
                continue

            total = feature_totals.get(fid, 0.0)
            pu_amount = contributions.get(fid, 0.0)

            remaining = total - pu_amount
            if remaining < target:
                critical_count += 1

        scores[pid] = critical_count / n_positive_target if n_positive_target > 0 else 0.0

    return scores
