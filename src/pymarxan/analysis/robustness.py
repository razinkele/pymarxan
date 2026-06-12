"""Multi-scenario robustness: minimax-regret plan selection.

Given a cost matrix of candidate plans (rows) evaluated under scenarios
(columns), choose the plan that is most robust to which scenario turns out
to be true — either by minimax regret (smallest worst-case regret) or by
minimax cost (smallest worst-case cost).

The field's lighter-weight "no-regrets overlap" is already available via
:func:`pymarxan.analysis.selection_freq.compute_selection_frequency`.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution
from pymarxan.solvers.utils import compute_objective


@dataclass
class RegretResult:
    """Robustness analysis of plans across scenarios (costs, lower better)."""

    cost_matrix: np.ndarray
    regret_matrix: np.ndarray
    max_regret: np.ndarray
    plan_labels: list[Any]
    scenario_labels: list[Any]
    minimax_regret_plan: Any
    minimax_cost_plan: Any

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "plan": self.plan_labels,
                "max_regret": self.max_regret.tolist(),
                "worst_case_cost": self.cost_matrix.max(axis=1).tolist(),
            }
        )


def minimax_regret(
    cost_matrix: np.ndarray,
    plan_labels: list[Any] | None = None,
    scenario_labels: list[Any] | None = None,
) -> RegretResult:
    """Select the most robust plan from a plans-by-scenarios cost matrix.

    Args:
        cost_matrix: ``(n_plans, n_scenarios)`` array of costs (lower is
            better) — plan ``i`` evaluated under scenario ``j``.
        plan_labels: Optional labels for the rows (default: indices).
        scenario_labels: Optional labels for the columns (default: indices).

    Returns:
        A :class:`RegretResult` with the regret matrix, per-plan worst-case
        regret, the minimax-regret plan, and the minimax-cost plan.
    """
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be 2-D (plans x scenarios)")
    n_plans, n_scen = cost.shape
    plans = list(plan_labels) if plan_labels is not None else list(range(n_plans))
    scens = (
        list(scenario_labels)
        if scenario_labels is not None
        else list(range(n_scen))
    )

    best_per_scenario = cost.min(axis=0)  # best (lowest) cost in each column
    regret = cost - best_per_scenario  # broadcast over rows
    max_regret = regret.max(axis=1)
    worst_case_cost = cost.max(axis=1)

    return RegretResult(
        cost_matrix=cost,
        regret_matrix=regret,
        max_regret=max_regret,
        plan_labels=plans,
        scenario_labels=scens,
        minimax_regret_plan=plans[int(np.argmin(max_regret))],
        minimax_cost_plan=plans[int(np.argmin(worst_case_cost))],
    )


def evaluate_plans_across_scenarios(
    problems: Mapping[str, ConservationProblem],
    solutions: Mapping[str, Solution],
    *,
    blm: float = 0.0,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Evaluate each plan's objective under every scenario problem.

    Args:
        problems: Mapping of scenario label to its :class:`ConservationProblem`.
        solutions: Mapping of plan label to a :class:`Solution` whose
            ``selected`` array is re-evaluated under each scenario.
        blm: Boundary-length modifier used for the objective.

    Returns:
        ``(cost_matrix, plan_labels, scenario_labels)`` where
        ``cost_matrix[i, j]`` is plan ``i`` evaluated under scenario ``j``.
        Feed it straight into :func:`minimax_regret`.
    """
    plan_labels = list(solutions.keys())
    scenario_labels = list(problems.keys())
    matrix = np.zeros((len(plan_labels), len(scenario_labels)))

    for j, slabel in enumerate(scenario_labels):
        problem = problems[slabel]
        pu_index = {int(pid): k for k, pid in enumerate(problem.planning_units["id"])}
        for i, plabel in enumerate(plan_labels):
            selected = np.asarray(solutions[plabel].selected, dtype=bool)
            matrix[i, j] = compute_objective(problem, selected, pu_index, blm)
    return matrix, plan_labels, scenario_labels
