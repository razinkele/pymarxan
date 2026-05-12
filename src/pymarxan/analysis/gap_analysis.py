"""Gap analysis for conservation planning.

Compares current protection levels (PUs with status=2) against
feature targets to identify protection gaps.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pymarxan.models.problem import ConservationProblem


@dataclass
class GapResult:
    """Results of a gap analysis."""

    feature_ids: list[int]
    feature_names: list[str]
    targets: dict[int, float]
    total_amount: dict[int, float]
    protected_amount: dict[int, float]
    gap: dict[int, float]
    target_met: dict[int, bool]

    def to_dataframe(self) -> pd.DataFrame:
        fid_to_name = dict(zip(self.feature_ids, self.feature_names))
        rows = []
        for fid in self.feature_ids:
            total = self.total_amount[fid]
            protected = self.protected_amount[fid]
            rows.append({
                "feature_id": fid,
                "feature_name": fid_to_name[fid],
                "target": self.targets[fid],
                "total_amount": total,
                "protected_amount": protected,
                "gap": self.gap[fid],
                "target_met": self.target_met[fid],
                "percent_protected": (
                    protected / total * 100 if total > 0 else 0.0
                ),
            })
        return pd.DataFrame(rows)


def compute_gap_analysis(problem: ConservationProblem) -> GapResult:
    """Compute protection gap for each feature."""
    protected_pu_ids = set(
        problem.planning_units.loc[
            problem.planning_units["status"] == 2, "id"
        ]
    )

    feature_ids = problem.features["id"].tolist()
    feature_names = problem.features["name"].tolist()

    targets: dict[int, float] = dict(
        zip(problem.features["id"].astype(int), problem.features["target"].astype(float))
    )

    total_amount = problem.feature_amounts()

    # Vectorized: filter pu_vs_features to protected PUs and groupby species
    puvspr = problem.pu_vs_features
    mask = puvspr["pu"].isin(protected_pu_ids)
    protected_amount: dict[int, float]
    if mask.any():
        prot_totals = puvspr.loc[mask].groupby("species")["amount"].sum()
        protected_amount = {
            fid: float(prot_totals.get(fid, 0.0)) for fid in feature_ids
        }
    else:
        protected_amount = {fid: 0.0 for fid in feature_ids}

    # Apply MISSLEVEL so gap analysis agrees with solver / build_solution /
    # export_summary — previously gap was measured against the raw target.
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))

    gap: dict[int, float] = {}
    target_met: dict[int, bool] = {}
    for fid in feature_ids:
        effective_target = targets[fid] * misslevel
        shortfall = max(effective_target - protected_amount[fid], 0.0)
        gap[fid] = shortfall
        target_met[fid] = shortfall <= 0

    return GapResult(
        feature_ids=feature_ids,
        feature_names=feature_names,
        targets=targets,
        total_amount=total_amount,
        protected_amount=protected_amount,
        gap=gap,
        target_met=target_met,
    )
