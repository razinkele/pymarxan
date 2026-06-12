"""Area-based representation reporting against a policy threshold.

This answers the Kunming-Montreal Global Biodiversity Framework "Target 3"
question — *did my plan protect at least 30% of each feature?* — for a
given solution. It differs from :mod:`pymarxan.analysis.gap_analysis`,
which scores existing protection (``status == 2``) against each feature's
optimisation target; here we score an arbitrary solution against a single
uniform threshold.

See Robinson et al. (2024), *PLoS Biology*,
https://doi.org/10.1371/journal.pbio.3002613.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


@dataclass
class RepresentationResult:
    """Per-feature representation of a solution against a threshold."""

    feature_ids: list[int]
    feature_names: list[str]
    total_amount: dict[int, float]
    represented_amount: dict[int, float]
    pct_represented: dict[int, float]
    meets_threshold: dict[int, bool]
    threshold: float
    n_features_meeting: int
    fraction_features_meeting: float

    def to_dataframe(self) -> pd.DataFrame:
        name_of = dict(zip(self.feature_ids, self.feature_names))
        return pd.DataFrame(
            {
                "feature_id": self.feature_ids,
                "feature_name": [name_of[f] for f in self.feature_ids],
                "total_amount": [self.total_amount[f] for f in self.feature_ids],
                "represented_amount": [
                    self.represented_amount[f] for f in self.feature_ids
                ],
                "pct_represented": [
                    self.pct_represented[f] for f in self.feature_ids
                ],
                "meets_threshold": [
                    self.meets_threshold[f] for f in self.feature_ids
                ],
            }
        )


def compute_representation(
    problem: ConservationProblem,
    solution: Solution,
    *,
    threshold: float = 0.30,
) -> RepresentationResult:
    """Report how much of each feature a solution represents.

    Args:
        problem: The conservation problem.
        solution: The solution (its selected units define the reserve).
        threshold: Fraction of each feature's total amount that must be
            represented to "meet" it (default 0.30 for the 30x30 goal).

    Returns:
        A :class:`RepresentationResult` with per-feature totals,
        represented amounts, percentages, threshold pass/fail, and a
        summary of how many features clear the threshold.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    feature_ids = [int(f) for f in problem.features["id"]]
    feature_names = [str(n) for n in problem.features["name"]]
    total_amount = {int(f): float(a) for f, a in problem.feature_amounts().items()}

    selected_ids = {
        int(pid)
        for pid, sel in zip(problem.planning_units["id"], solution.selected)
        if sel
    }
    puvspr = problem.pu_vs_features
    mask = puvspr["pu"].isin(selected_ids)
    if mask.any():
        rep_totals = puvspr.loc[mask].groupby("species")["amount"].sum()
        represented_amount = {
            fid: float(rep_totals.get(fid, 0.0)) for fid in feature_ids
        }
    else:
        represented_amount = {fid: 0.0 for fid in feature_ids}

    pct_represented: dict[int, float] = {}
    meets_threshold: dict[int, bool] = {}
    for fid in feature_ids:
        total = total_amount.get(fid, 0.0)
        rep = represented_amount[fid]
        frac = rep / total if total > 0 else 0.0
        pct_represented[fid] = frac * 100.0
        meets_threshold[fid] = frac >= threshold

    n_meeting = sum(meets_threshold.values())
    return RepresentationResult(
        feature_ids=feature_ids,
        feature_names=feature_names,
        total_amount=total_amount,
        represented_amount=represented_amount,
        pct_represented=pct_represented,
        meets_threshold=meets_threshold,
        threshold=threshold,
        n_features_meeting=n_meeting,
        fraction_features_meeting=(
            n_meeting / len(feature_ids) if feature_ids else 0.0
        ),
    )
