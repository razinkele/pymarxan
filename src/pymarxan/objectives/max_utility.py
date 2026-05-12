"""Maximum utility objective: maximize weighted proportional coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pymarxan.objectives.base import Objective

if TYPE_CHECKING:
    import pulp

    from pymarxan.models.problem import ConservationProblem


class MaxUtilityObjective(Objective):
    """Maximize weighted proportional feature coverage subject to budget.

    Utility for each feature is ``min(achieved_f / target_f, 1.0)``,
    weighted by feature importance. The score is negated for the
    lower-is-better convention:
    ``base = -Σ weight_f * min(achieved_f / target_f, 1.0)``.

    Requires ``BUDGET`` parameter and all feature targets > 0.
    Binary-only (does not implement ZonalObjective).
    """

    def name(self) -> str:
        return "MaxUtility"

    def uses_target_penalty(self) -> bool:
        return False

    def compute_base_score(
        self,
        problem: ConservationProblem,
        selected: np.ndarray,
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> float:
        feat_ids = list(problem.features["id"])
        targets = dict(
            zip(problem.features["id"], problem.features["target"], strict=False),
        )
        weights = {}
        if "spf" in problem.features.columns:
            weights = dict(
                zip(problem.features["id"], problem.features["spf"], strict=False),
            )

        achieved = self._compute_achieved(
            problem, selected, effective_amounts, pu_index,
        )

        total_utility = 0.0
        for fid in feat_ids:
            fid_int = int(fid)
            t = float(targets.get(fid_int, 0.0))
            if t <= 0:
                msg = (
                    f"MaxUtilityObjective requires all feature targets > 0, "
                    f"but feature {fid_int} has target {t}"
                )
                raise ValueError(msg)
            a = achieved.get(fid_int, 0.0)
            w = float(weights.get(fid_int, 1.0))
            total_utility += w * min(a / t, 1.0)
        return -total_utility

    def build_mip_objective(
        self,
        problem: ConservationProblem,
        model: pulp.LpProblem,
        x: dict[int, pulp.LpVariable],
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> pulp.LpAffineExpression:
        import pulp as _pulp

        feat_ids = problem.features["id"].values
        u = {}
        weights = {}
        if "spf" in problem.features.columns:
            weights = dict(
                zip(problem.features["id"], problem.features["spf"], strict=False),
            )

        for fid in feat_ids:
            fid_int = int(fid)
            t = float(problem.features.loc[
                problem.features["id"] == fid, "target"
            ].iloc[0])
            if t <= 0:
                msg = (
                    f"MaxUtilityObjective requires all feature targets > 0, "
                    f"but feature {fid_int} has target {t}"
                )
                raise ValueError(msg)
            u[fid_int] = _pulp.LpVariable(
                f"u_util_{fid_int}", lowBound=0, upBound=1.0,
            )

        return _pulp.lpSum(
            -float(weights.get(fid, 1.0)) * u[fid] for fid in u
        )

    def build_mip_constraints(
        self,
        problem: ConservationProblem,
        model: pulp.LpProblem,
        x: dict[int, pulp.LpVariable],
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> None:
        import pulp as _pulp

        feat_ids = problem.features["id"].values
        puvspr = problem.pu_vs_features
        feat_id_list = list(problem.features["id"])

        for fid in feat_ids:
            fid_int = int(fid)
            t = float(problem.features.loc[
                problem.features["id"] == fid, "target"
            ].iloc[0])
            if t <= 0:
                continue

            rows = puvspr[puvspr["species"] == fid_int]
            fidx = feat_id_list.index(fid_int)
            prop_expr = _pulp.lpSum(
                float(effective_amounts[pu_index[int(row["pu"])], fidx])
                / t * x[int(row["pu"])]
                for _, row in rows.iterrows()
                if int(row["pu"]) in x and int(row["pu"]) in pu_index
            )

            u_var = model.variablesDict().get(f"u_util_{fid_int}")
            if u_var is not None:
                model += u_var <= prop_expr, f"util_cap_prop_{fid_int}"
                model += u_var <= 1.0, f"util_cap_one_{fid_int}"

        # Budget constraint
        budget = float(problem.parameters.get("BUDGET", float("inf")))
        if budget < float("inf"):
            cost_map = dict(
                zip(
                    problem.planning_units["id"],
                    problem.planning_units["cost"],
                    strict=False,
                ),
            )
            model += (
                _pulp.lpSum(
                    cost_map.get(pid, 0.0) * x[pid] for pid in x
                )
                <= budget
            ), "budget_constraint"

    @staticmethod
    def _compute_achieved(
        problem: ConservationProblem,
        selected: np.ndarray,
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> dict[int, float]:
        feat_ids = list(problem.features["id"])
        achieved: dict[int, float] = {int(f): 0.0 for f in feat_ids}
        puvspr = problem.pu_vs_features
        for _, row in puvspr.iterrows():
            pid = int(row["pu"])
            fid = int(row["species"])
            idx = pu_index.get(pid)
            if idx is not None and selected[idx]:
                fidx = feat_ids.index(fid)
                achieved[fid] += float(effective_amounts[idx, fidx])
        return achieved
