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
        rows = []
        for fid in self.feature_ids:
            total = self.total_amount[fid]
            protected = self.protected_amount[fid]
            rows.append({
                "feature_id": fid,
                "feature_name": self.feature_names[self.feature_ids.index(fid)],
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

    targets: dict[int, float] = {}
    for _, row in problem.features.iterrows():
        targets[int(row["id"])] = float(row["target"])

    total_amount = problem.feature_amounts()
    protected_amount: dict[int, float] = {fid: 0.0 for fid in feature_ids}

    for _, row in problem.pu_vs_features.iterrows():
        pid = int(row["pu"])
        fid = int(row["species"])
        if pid in protected_pu_ids and fid in protected_amount:
            protected_amount[fid] += float(row["amount"])

    gap: dict[int, float] = {}
    target_met: dict[int, bool] = {}
    for fid in feature_ids:
        shortfall = max(targets[fid] - protected_amount[fid], 0.0)
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
