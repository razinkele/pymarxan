"""Dynamic (multi-period) reserve scheduling — Costello–Polasky informed-myopic.

Static reserve selection assumes you protect everything at once. In reality you
protect sites over time on a per-period budget while unprotected sites are lost
to development. Costello & Polasky (2004, doi:10.1016/j.reseneeco.2003.11.005)
showed that the exact dynamic program is intractable but a simple **informed-
myopic** heuristic captures most of its value: each period, protect the sites
with the highest *expected value at risk* — value weighted by the probability
the site would otherwise be lost over the remaining horizon — not the highest
value (naive) nor the highest risk alone.

``dynamic_reserve_greedy`` returns the protection **schedule** and the expected
retained conservation value under it (a site protected at period ``t`` is
retained with probability ``(1-p)^t`` having survived ``t`` prior exposure
rounds; a never-protected site with ``(1-p)^T``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DynamicScheduleResult:
    """Outcome of a dynamic reserve schedule."""

    schedule: list[set[int]]      # protected site indices, one set per period
    protected: set[int]
    expected_value: float


def _expected_value(
    values: np.ndarray,
    surv: np.ndarray,            # per-site per-period survival prob (1 - p)
    protect_period: dict[int, int],
    horizon: int,
) -> float:
    total = 0.0
    for i in range(len(values)):
        t = protect_period.get(i, horizon)  # never protected → full horizon exposure
        total += values[i] * surv[i] ** t
    return float(total)


def dynamic_reserve_greedy(
    values,
    loss_prob,
    budgets,
    costs=None,
    *,
    prioritize: str = "value_at_risk",
) -> DynamicScheduleResult:
    """Schedule site protection over periods under per-period budgets.

    Args:
        values: Per-site conservation value (length N).
        loss_prob: Per-period probability an unprotected site is lost; scalar
            or length-N array, each in [0, 1].
        budgets: Per-period cost cap (length T = number of periods).
        costs: Per-site protection cost (length N); defaults to 1 each.
        prioritize: ``"value_at_risk"`` (Costello–Polasky informed-myopic:
            value × P(lost over remaining horizon) / cost) or ``"value"``
            (naive value / cost baseline).

    Returns:
        :class:`DynamicScheduleResult` with the per-period schedule, the set of
        protected sites, and the expected retained value.
    """
    if prioritize not in ("value_at_risk", "value"):
        raise ValueError(
            f"unknown prioritize {prioritize!r}; use 'value_at_risk' or 'value'"
        )
    values = np.asarray(values, dtype=float)
    n = len(values)
    p = np.broadcast_to(np.asarray(loss_prob, dtype=float), (n,)).astype(float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("loss_prob values must lie in [0, 1]")
    cost = np.ones(n) if costs is None else np.asarray(costs, dtype=float)
    budgets = list(budgets)
    horizon = len(budgets)
    surv = 1.0 - p

    protect_period: dict[int, int] = {}
    schedule: list[set[int]] = []
    for t, budget in enumerate(budgets):
        remaining = horizon - t  # exposure rounds a still-unprotected site faces
        # priority per unprotected site
        unprotected = [i for i in range(n) if i not in protect_period]
        if prioritize == "value_at_risk":
            risk = 1.0 - surv ** remaining  # P(lost over the remaining horizon)
            score = {i: values[i] * risk[i] / cost[i] for i in unprotected}
        else:
            score = {i: values[i] / cost[i] for i in unprotected}

        chosen: set[int] = set()
        spent = 0.0
        for i in sorted(unprotected, key=lambda j: score[j], reverse=True):
            if score[i] <= 0:
                continue
            if spent + cost[i] <= budget + 1e-9:
                chosen.add(i)
                spent += cost[i]
                protect_period[i] = t
        schedule.append(chosen)

    return DynamicScheduleResult(
        schedule=schedule,
        protected=set(protect_period),
        expected_value=_expected_value(values, surv, protect_period, horizon),
    )
