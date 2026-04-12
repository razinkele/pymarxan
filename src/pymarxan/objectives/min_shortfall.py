"""Minimum shortfall objective: minimize total target shortfall."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pymarxan.objectives.base import ZonalObjective

if TYPE_CHECKING:
    import pulp

    from pymarxan.models.problem import ConservationProblem


class MinShortfallObjective(ZonalObjective):
    """Minimize total feature target shortfall subject to budget.

    Shortfall for each feature is ``max(0, target_f - achieved_f)``.
    Score: ``base = Σ max(0, target_f - achieved_f)``.

    Requires ``BUDGET`` parameter (without budget, optimal is to select
    everything).
    """

    def name(self) -> str:
        return "MinShortfall"

    def uses_target_penalty(self) -> bool:
        return False

    def compute_base_score(
        self,
        problem: Any,
        selected: np.ndarray,
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> float:
        feat_ids = list(problem.features["id"])
        targets = dict(
            zip(problem.features["id"], problem.features["target"], strict=False),
        )
        achieved = self._compute_achieved(
            problem, selected, effective_amounts, pu_index,
        )
        total_shortfall = 0.0
        for fid in feat_ids:
            t = float(targets.get(int(fid), 0.0))
            a = achieved.get(int(fid), 0.0)
            total_shortfall += max(0.0, t - a)
        return total_shortfall

    def build_mip_objective(
        self,
        problem: Any,
        model: pulp.LpProblem,
        x: dict[int, pulp.LpVariable],
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> pulp.LpAffineExpression:
        import pulp as _pulp

        feat_ids = problem.features["id"].values
        s = {}
        for fid in feat_ids:
            fid_int = int(fid)
            t = float(problem.features.loc[
                problem.features["id"] == fid, "target"
            ].iloc[0])
            if t <= 0:
                continue
            s[fid_int] = _pulp.LpVariable(
                f"s_short_{fid_int}", lowBound=0,
            )
        return _pulp.lpSum(s[fid] for fid in s)

    def build_mip_constraints(
        self,
        problem: Any,
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
            achieved_expr = _pulp.lpSum(
                float(effective_amounts[pu_index[int(row["pu"])], fidx])
                * x[int(row["pu"])]
                for _, row in rows.iterrows()
                if int(row["pu"]) in x and int(row["pu"]) in pu_index
            )

            s_var = model.variablesDict().get(f"s_short_{fid_int}")
            if s_var is not None:
                model += (
                    s_var >= t - achieved_expr
                ), f"shortfall_def_{fid_int}"

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

    # --- Zonal interface ---

    def compute_zone_score(
        self,
        problem: Any,
        assignment: np.ndarray,
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> float:
        feat_ids = list(problem.features["id"])
        targets = dict(
            zip(problem.features["id"], problem.features["target"], strict=False),
        )
        achieved = self._compute_zone_achieved(
            problem, assignment, effective_amounts, pu_index,
        )
        total_shortfall = 0.0
        for fid in feat_ids:
            t = float(targets.get(int(fid), 0.0))
            a = achieved.get(int(fid), 0.0)
            total_shortfall += max(0.0, t - a)
        return total_shortfall

    def build_zone_mip_objective(
        self,
        problem: Any,
        model: pulp.LpProblem,
        x: dict[tuple[int, int], pulp.LpVariable],
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> pulp.LpAffineExpression:
        import pulp as _pulp

        feat_ids = problem.features["id"].values
        s = {}
        for fid in feat_ids:
            fid_int = int(fid)
            t = float(problem.features.loc[
                problem.features["id"] == fid, "target"
            ].iloc[0])
            if t <= 0:
                continue
            s[fid_int] = _pulp.LpVariable(
                f"zs_short_{fid_int}", lowBound=0,
            )
        return _pulp.lpSum(s[fid] for fid in s)

    # --- Helpers ---

    @staticmethod
    def _compute_achieved(
        problem: Any,
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

    @staticmethod
    def _compute_zone_achieved(
        problem: Any,
        assignment: np.ndarray,
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> dict[int, float]:
        feat_ids = list(problem.features["id"])
        achieved: dict[int, float] = {int(f): 0.0 for f in feat_ids}
        puvspr = problem.pu_vs_features
        zone_contributions = getattr(problem, "zone_contributions", {})
        for _, row in puvspr.iterrows():
            pid = int(row["pu"])
            fid = int(row["species"])
            idx = pu_index.get(pid)
            if idx is not None:
                z = int(assignment[idx])
                if z > 0:
                    fidx = feat_ids.index(fid)
                    contrib = float(
                        zone_contributions.get((z, fid), 1.0),
                    )
                    achieved[fid] += (
                        float(effective_amounts[idx, fidx]) * contrib
                    )
        return achieved
