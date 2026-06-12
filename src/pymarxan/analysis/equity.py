"""Distributional-equity analysis of a conservation solution.

Equity asks whether the benefits (or burdens) of a reserve are shared
evenly across social or spatial groups, or concentrated in a few. The
core metric is the Gini coefficient of the value captured per group: 0 is
perfectly even, approaching ``(n_groups - 1) / n_groups`` when one group
captures everything.

See Gopalakrishna et al. (2024), *PNAS*,
https://doi.org/10.1073/pnas.2402970121.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


@dataclass
class EquityResult:
    """How a solution's value distributes across groups."""

    group_values: dict[str, float]
    group_shares: dict[str, float]
    gini: float
    min_share: float
    max_share: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "group": list(self.group_values.keys()),
                "value": list(self.group_values.values()),
                "share": [self.group_shares[g] for g in self.group_values],
            }
        )


def _gini(values: list[float]) -> float:
    """Gini coefficient of a list of non-negative group totals."""
    n = len(values)
    total = sum(values)
    if n == 0 or total <= 0:
        return 0.0
    diffs = sum(abs(a - b) for a in values for b in values)
    return diffs / (2 * n * total)


def compute_equity(
    problem: ConservationProblem,
    solution: Solution,
    groups: Mapping[int, str],
    *,
    value: str | Mapping[int, float] = "cost",
) -> EquityResult:
    """Compute the distributional equity of a solution across groups.

    Args:
        problem: The conservation problem (supplies PU ids and costs).
        solution: The solution whose selected units are evaluated.
        groups: Mapping from planning-unit id to a group label.
        value: What to distribute across groups for each *selected* unit —
            ``"cost"`` (the PU cost, i.e. who bears the burden),
            ``"count"`` (one per PU), or an explicit ``{pu_id: value}``
            mapping (e.g. a per-PU benefit).

    Returns:
        An :class:`EquityResult` with per-group totals, shares, and the
        Gini coefficient of the group totals.
    """
    pu_ids = problem.planning_units["id"].to_numpy()

    if isinstance(value, Mapping):
        values_map = {int(k): float(v) for k, v in value.items()}
    elif value == "cost":
        values_map = {
            int(p): float(c)
            for p, c in zip(
                problem.planning_units["id"], problem.planning_units["cost"]
            )
        }
    elif value == "count":
        values_map = {int(p): 1.0 for p in pu_ids}
    else:
        raise ValueError(
            f"unknown value {value!r}; use 'cost', 'count', or a {{pu_id: value}} mapping"
        )

    # Initialize every group that appears, in first-seen order.
    group_values: dict[str, float] = {g: 0.0 for g in dict.fromkeys(groups.values())}
    for pid, sel in zip(pu_ids, solution.selected):
        if sel:
            g = groups.get(int(pid))
            if g is not None:
                group_values[g] += values_map.get(int(pid), 0.0)

    total = sum(group_values.values())
    if total > 0:
        group_shares = {g: v / total for g, v in group_values.items()}
    else:
        group_shares = {g: 0.0 for g in group_values}

    shares = list(group_shares.values())
    return EquityResult(
        group_values=group_values,
        group_shares=group_shares,
        gini=_gini(list(group_values.values())),
        min_share=min(shares) if shares else 0.0,
        max_share=max(shares) if shares else 0.0,
    )
